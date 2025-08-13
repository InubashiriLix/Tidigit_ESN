# -*- coding: utf-8 -*-
"""
train_sdk_esn.py — 用 SDK 等价仿真在 CUDA 上训练 ESN
- 小模型(≤576x128) 直接用 CudaSDKArraySim.calculate（严格 SDK 接口）
- 大模型自动用等价的 batched GEMV 快仿真（语义一致，速度更快）
- 架构保持：int4 权重 × 整数输入 → int16 饱和 → 行尺度/增益 → tanh → 线性 W_out → CTC
"""

import argparse, random, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm.auto import tqdm

# === 你的数据集/配置（保持不变） ===
from dataset import TidigitDataset, collate_fn
from config import RAW_TEST_FOLDER_ABS_PATH, RAW_TRAIN_FOLDER_ABS_PATH

# === SDK 等价 CUDA 仿真（严格接口） ===
from he100_cuda_sdk import CudaSDKArraySim, DEVICE

# === 等价快仿真（batched GEMV，数值语义同 SDK） ===
from he100_cuda import torch_batched_mv_int4_fast


# -------------------------- 实用函数 --------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pbar(it, desc="", total=None, enable=True, leave=False):
    return (
        tqdm(it, desc=desc, total=total, dynamic_ncols=True, leave=leave)
        if enable
        else it
    )


def ctc_collapse(seq, blank=0):
    out, prev = [], None
    for t in seq:
        if t == blank:
            prev = t
            continue
        if t != prev:
            out.append(t)
        prev = t
    return out


def power_iteration_numpy(W: np.ndarray, iters: int = 100) -> float:
    n = W.shape[0]
    x = np.random.randn(n).astype(np.float64)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(iters):
        x = W @ x
        x /= np.linalg.norm(x) + 1e-12
    lam = float(x @ (W @ x))
    return abs(lam)


def quantize_rowwise_int4(W: np.ndarray, mode="percentile", perc=99.0, eps=1e-8):
    Wf = W.astype(np.float32, copy=False)
    H, C = Wf.shape
    q = np.empty_like(Wf, dtype=np.int8)
    s = np.empty((H,), dtype=np.float32)
    for i in range(H):
        row = Wf[i]
        a = np.abs(row)
        if mode == "percentile":
            thr = np.percentile(a, perc)
            scale = max(thr, eps) / 7.0
        else:
            scale = max(a.max(), eps) / 7.0
        qi = np.round(row / scale).astype(np.int32)
        q[i] = np.clip(qi, -8, 7).astype(np.int8)
        s[i] = scale
    return q, s


# -------------------------- SDK/快仿真调度 --------------------------
MAX_R, MAX_C = 576, 128


class SDKOrFastMatvec:
    """
    统一入口：
      - 若 (rows<=576 且 cols<=128)，用严格 SDK 仿真 (CudaSDKArraySim)
      - 否则，用等价 batched GEMV（torch_batched_mv_int4_fast）
    接口：
      y = mv_res(h_int)  # (B,H) -> (B,H)
      y = mv_in(u_int)   # (B,F) -> (B,H)
    """

    def __init__(self, W_res_i4: np.ndarray, W_in_i4: np.ndarray):
        self.W_res_i4_np = W_res_i4
        self.W_in_i4_np = W_in_i4
        H, H2 = W_res_i4.shape
        H3, F = W_in_i4.shape
        assert H == H2 == H3
        self.H, self.F = H, F

        self.use_sdk_res = H <= MAX_R and H <= MAX_C  # res: (H,H)
        self.use_sdk_in = H <= MAX_R and self.F <= MAX_C  # in : (H,F)

        # 常驻 CUDA 权重（用于 fast 路径）
        self.W_res_i8 = torch.from_numpy(W_res_i4).to(DEVICE, non_blocking=True)
        self.W_in_i8 = torch.from_numpy(W_in_i4).to(DEVICE, non_blocking=True)

        # SDK 仿真实例（小模型时使用），并把权重写入
        self.sim_res = None
        self.sim_in = None
        if self.use_sdk_res:
            self.sim_res = CudaSDKArraySim(MAX_R, MAX_C)
            self.sim_res.set_weight_int4(W_res_i4, [0, 0, H, H])
        if self.use_sdk_in:
            self.sim_in = CudaSDKArraySim(MAX_R, MAX_C)

    @torch.inference_mode()
    def mv_res(self, H_int32_BH: torch.Tensor) -> torch.Tensor:
        """
        输入： (B,H) int32
        输出： (B,H) int16 语义（这里直接用 int32 保存，后续再饱和/缩放）
        """
        assert H_int32_BH.dtype in (torch.int32, torch.int16)
        B, H = H_int32_BH.shape
        if self.use_sdk_res:
            # 严格 SDK：每个 batch 调一次 calculate（ret_mode=0，expand_mode=1/0 都等价）
            out_list = []
            H_cpu = H_int32_BH.detach().cpu().numpy().astype(np.int32)
            for b in range(B):
                y = self.sim_res.calculate(
                    H_cpu[b : b + 1, :],
                    addr=[0, 0, self.H, self.H],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )  # (1,H)
                out_list.append(torch.from_numpy(y))
            Y = torch.cat(out_list, dim=0).to(DEVICE)
            return Y.to(torch.int32)
        else:
            # 快仿真（等价语义）：(B,H) × (H,H)^T -> (B,H)
            return torch_batched_mv_int4_fast(self.W_res_i8, H_int32_BH, it_time=1).to(
                torch.int32
            )

    @torch.inference_mode()
    def mv_in(self, U_int32_BF: torch.Tensor) -> torch.Tensor:
        """
        输入： (B,F) int32
        输出： (B,H) int32
        严格 SDK 情况：对 W_in^T(F,H) 按列分块 (cw≤MAX_C=128) 写入 (h=cw?, NO)
        正确做法：每块写成 (h=F, w=cw)，输入长度=F 与 h 对齐；输出就是 (B,cw)，直接拼接到 (B,H)。
        """
        assert U_int32_BF.dtype in (torch.int32, torch.int16)
        B, F = U_int32_BF.shape
        if self.use_sdk_in:
            # 整个 batch 一次搬到 CPU，避免循环中反复 .cpu()
            U_cpu = U_int32_BF.detach().cpu().numpy().astype(np.int32)
            Y_chunks = []
            # 逐列块切 W_in^T：原始 W_in_i4 形状 (H,F)，取 [c0:c0+cw,:] 再转置 → (F,cw)
            for c0 in range(0, self.H, MAX_C):
                cw = min(MAX_C, self.H - c0)
                blkT = self.W_in_i4_np[c0 : c0 + cw, :].T  # (F, cw)
                # 把该列块写入到阵列的 (y=0,x=0,h=F,w=cw)
                self.sim_in.set_weight_int4(blkT, [0, 0, self.F, cw])
                # 对整个 batch 计算，输入长度=F 对齐 h
                y_blk = self.sim_in.calculate(
                    U_cpu,
                    addr=[0, 0, self.F, cw],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )  # (B, cw)
                Y_chunks.append(torch.from_numpy(y_blk))
            Y = torch.cat(Y_chunks, dim=1).to(DEVICE)  # (B,H)，按列拼接
            return Y.to(torch.int32)
        else:
            # 快仿真： (B,F) × (F,H) = (B,H)
            return torch_batched_mv_int4_fast(self.W_in_i8, U_int32_BF, it_time=1).to(
                torch.int32
            )


# -------------------------- 模型 --------------------------
class SDK_ESN_Model(nn.Module):
    """
    仍然：int4 权重 × 整数输入 → int16 饱和 → 行尺度/增益 → tanh → W_out
    - 权重行量化（-8..7）+ 行尺度 s_row
    - 可学增益 g_res/g_in（若开启）
    """

    def __init__(
        self,
        W_res_float: np.ndarray,
        W_in_float: np.ndarray,
        vocab_with_blank: int,
        *,
        input_int_scale: float = 3.0,
        row_quant_perc: float = 99.0,
        match_spectral: bool = True,
        learn_gains: bool = True,
        init_hidden_std_target: float = 0.35,
        seed: int = 1337,
    ):
        super().__init__()
        set_seed(seed)

        # 行量化到 int4
        self.W_res_i4, self.S_res_row_np = quantize_rowwise_int4(
            W_res_float, mode="percentile", perc=row_quant_perc
        )
        self.W_in_i4, self.S_in_row_np = quantize_rowwise_int4(
            W_in_float, mode="percentile", perc=row_quant_perc
        )

        self.H = int(self.W_res_i4.shape[0])
        self.F = int(self.W_in_i4.shape[1])

        # 谱半径匹配（与你之前一致）
        if match_spectral:
            rho_orig = power_iteration_numpy(W_res_float.astype(np.float64), iters=120)
            approx = (self.S_res_row_np[:, None] / 7.0).astype(
                np.float64
            ) * self.W_res_i4.astype(np.float64)
            rho_quant = power_iteration_numpy(approx, iters=120)
            gamma = float(rho_orig / max(rho_quant, 1e-12))
            self.S_res_row_np = (self.S_res_row_np * gamma).astype(np.float32)
            tqdm.write(
                f"[Spectral] rho_orig={rho_orig:.6f} rho_quant={rho_quant:.6f} gamma={gamma:.6f}"
            )

        # CUDA 常量
        self.S_res_row = torch.from_numpy(self.S_res_row_np).to(DEVICE)
        self.S_in_row = torch.from_numpy(self.S_in_row_np).to(DEVICE)

        # SDK or fast dispatcher
        self.mv = SDKOrFastMatvec(self.W_res_i4, self.W_in_i4)

        # 可学增益
        self.g_res = (
            nn.Parameter(torch.tensor(1.0, device=DEVICE)) if learn_gains else None
        )
        self.g_in = (
            nn.Parameter(torch.tensor(1.0, device=DEVICE)) if learn_gains else None
        )

        # 输出层
        self.W_out = nn.Linear(self.H, vocab_with_blank, device=DEVICE)

        self.input_int_scale = float(input_int_scale)
        self.init_hidden_std_target = float(init_hidden_std_target)

    def _row_scales(self):
        s_res = self.S_res_row / 49.0
        s_in = self.S_in_row / (7.0 * max(1.0, self.input_int_scale))
        if self.g_res is not None:
            s_res = s_res * self.g_res
        if self.g_in is not None:
            s_in = s_in * self.g_in
        return s_res.view(1, -1), s_in.view(1, -1)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,F) float32 (CUDA)
        return: (B,T,H) float32
        """
        assert x.is_cuda
        B, T, Fdim = x.shape
        assert Fdim == self.F
        s_res, s_in = self._row_scales()  # (1,H)

        h = torch.zeros(B, self.H, device=DEVICE, dtype=torch.float32)
        outs = []
        for t in range(T):
            u_int = torch.round(x[:, t, :].detach() * self.input_int_scale).to(
                torch.int32
            )  # (B,F)
            h_int = torch.round(h.detach().clamp(-1.0, 1.0) * 7.0).to(
                torch.int32
            )  # (B,H)

            # --- 核心：用 SDK 或快仿真执行“整数点积 → it_time(=1) → int16 饱和（这里以 int32 存）”
            out_res_i32 = self.mv.mv_res(h_int)  # (B,H)
            out_in_i32 = self.mv.mv_in(u_int)  # (B,H)

            # 饱和在行尺度之前（硬件等价）
            out_res_i16 = (
                torch.clamp(out_res_i32, -32768, 32767)
                .to(torch.int16)
                .to(torch.float32)
            )
            out_in_i16 = (
                torch.clamp(out_in_i32, -32768, 32767).to(torch.int16).to(torch.float32)
            )

            a = out_res_i16 * s_res + out_in_i16 * s_in  # (B,H)
            h = torch.tanh(a)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hs = self.forward_hidden(x)  # (B,T,H)
        B, T, H = Hs.shape
        return self.W_out(Hs.reshape(B * T, H)).reshape(B, T, -1)

    @torch.no_grad()
    def auto_gain_tune(
        self, calib_loader, steps=6, factor=1.25, target_std=None, pbar_on=True
    ):
        if self.g_res is None or self.g_in is None:
            tqdm.write("[AutoGain] learn_gains=False，跳过校准")
            return
        tgt = float(
            target_std if target_std is not None else self.init_hidden_std_target
        )

        def measure_std() -> float:
            for batch in calib_loader:
                mfccs, *_ = batch
                Hs = self.forward_hidden(mfccs.to(DEVICE, non_blocking=True).float())
                return float(Hs.std().item())
            return 0.0

        std0 = measure_std()
        tqdm.write(f"[AutoGain] start std={std0:.4f} → target={tgt:.4f}")
        g = float(self.g_res.item())
        for it in pbar(
            range(steps), desc="auto-gain", total=steps, enable=pbar_on, leave=False
        ):
            if std0 < tgt * 0.98:
                g *= factor
            elif std0 > tgt * 1.02:
                g /= factor
            else:
                break
            self.g_res.copy_(torch.tensor(g, device=DEVICE))
            self.g_in.copy_(torch.tensor(g, device=DEVICE))
            std0 = measure_std()
            tqdm.write(f"[AutoGain] iter{it + 1}: std={std0:.4f}, g={g:.4f}")
        tqdm.write(f"[AutoGain] done: std={std0:.4f}, g={g:.4f}")


# -------------------------- 训练/评估 --------------------------
def labels_to_targets(labels, device, vocab_with_blank: int):
    idx_list = []
    if isinstance(labels, list):
        for L in labels:
            if L.dim() == 2:
                idx = torch.argmax(L, dim=1) + 1
            else:
                idx = L[L != -1] + 1
            idx_list.append(idx.to(device))
    else:
        raise ValueError("labels expect list of Tensors")

    target_lengths = torch.tensor(
        [len(x) for x in idx_list], dtype=torch.long, device=device
    )
    targets = torch.cat(idx_list).to(device)
    assert int(targets.max().item()) < vocab_with_blank, "targets exceed vocab(+blank)"
    return targets, target_lengths


@torch.no_grad()
def evaluate(
    model: SDK_ESN_Model,
    dataloader,
    idx_to_char: dict[int, str],
    *,
    pbar_on=True,
    live_acc=False,
):
    model.eval()
    preds = []
    for batch in pbar(dataloader, desc="Eval", total=len(dataloader), enable=pbar_on):
        mfccs, labels, mfcc_lengths, paths, captions = batch
        x = mfccs.to(DEVICE, non_blocking=True).float()

        logits = model(x)  # (B,T,C)
        y = logits.permute(1, 0, 2)  # (T,B,C)
        ids = torch.argmax(y, dim=2).cpu().numpy()  # (T,B)

        B = ids.shape[1]
        inner = pbar(
            range(B), desc="Decode", total=B, enable=pbar_on and live_acc, leave=False
        )

        for i in inner:
            raw = ids[:, i].tolist()
            seq = ctc_collapse(raw, blank=0)
            chars = []
            for t in seq:
                if t == 0:
                    continue
                r = t - 1
                if r in idx_to_char:
                    chars.append(idx_to_char[r])
            pred = "".join(chars)
            truth = captions[i]
            preds.append((paths[i], truth, pred))
        break
    return preds


def build_random_reservoir(H: int, F: int, spectral_radius: float, seed: int = 1337):
    rng = np.random.default_rng(seed)
    W_res = rng.uniform(-0.5, 0.5, size=(H, H)).astype(np.float32)
    eigvals = np.linalg.eigvals(W_res.astype(np.float64))
    rho = np.max(np.abs(eigvals))
    W_res *= spectral_radius / max(float(rho), 1e-9)
    W_in = rng.uniform(-0.5, 0.5, size=(H, F)).astype(np.float32)
    return W_res, W_in


# -------------------------- CLI --------------------------
def get_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=2000)
    ap.add_argument("--input-size", type=int, default=13)
    ap.add_argument("--spectral-radius", type=float, default=0.9)
    ap.add_argument("--input-int-scale", type=float, default=3.0)
    ap.add_argument("--row-quant-perc", type=float, default=99.0)
    ap.add_argument("--no-spectral-match", action="store_true")
    ap.add_argument("--no-learn-gains", action="store_true")
    ap.add_argument("--auto-gain-steps", type=int, default=6)
    ap.add_argument("--auto-gain-target-std", type=float, default=0.35)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--pbar", action="store_true")
    ap.add_argument("--live-acc", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default="sdk_train_ckpt.pth")
    return ap.parse_args()


# -------------------------- 主程序 --------------------------
def main():
    args = get_cli()
    set_seed(args.seed)

    # 数据
    train_ds = TidigitDataset(use_onehot=True)
    test_ds = TidigitDataset(data_dir=RAW_TRAIN_FOLDER_ABS_PATH, use_onehot=True)

    def take_subset(ds, frac: float):
        if frac >= 0.999 or frac <= 0.0:
            return ds
        all_paths = list(ds.dataset_path_list)
        n_all = len(all_paths)
        k = max(1, int(n_all * frac))
        idx = np.random.permutation(n_all)[:k]
        keep_paths = [all_paths[i] for i in idx]
        ds.data_cap_mfcc_dict = {p: ds.data_cap_mfcc_dict[p] for p in keep_paths}
        ds.dataset_path_list = keep_paths
        return ds

    train_ds = take_subset(train_ds, args.subset)
    test_ds = take_subset(test_ds, args.subset)

    train_ld = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_ld = data.DataLoader(
        test_ds,
        batch_size=max(1, len(test_ds)),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    vocab_with_blank = len(train_ds.caption_encoder.char_vocab) + 1  # 36 + blank

    # 库
    W_res_f, W_in_f = build_random_reservoir(
        args.hidden, args.input_size, args.spectral_radius, seed=args.seed
    )

    model = SDK_ESN_Model(
        W_res_f,
        W_in_f,
        vocab_with_blank=vocab_with_blank,
        input_int_scale=args.input_int_scale,
        row_quant_perc=args.row_quant_perc,
        match_spectral=(not args.no_spectral_match),
        learn_gains=(not args.no_learn_gains),
        init_hidden_std_target=args.auto_gain_target_std,
        seed=args.seed,
    ).to(DEVICE)

    # Auto-gain（小校准集）
    calib_ds = take_subset(TidigitDataset(use_onehot=True), min(0.02, args.subset))
    calib_ld = data.DataLoader(
        calib_ds,
        batch_size=min(8, len(calib_ds)),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model.auto_gain_tune(
        calib_ld,
        steps=args.auto_gain_steps,
        target_std=args.auto_gain_target_std,
        pbar_on=args.pbar,
    )

    # 只训 W_out 和 g_res/g_in（若开启）
    params = list(model.W_out.parameters())
    if model.g_res is not None:
        params += [model.g_res]
    if model.g_in is not None:
        params += [model.g_in]
    optim = torch.optim.Adam(params, lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    best_char = -1.0
    for ep in pbar(
        range(1, args.epochs + 1), desc="Epochs", total=args.epochs, enable=args.pbar
    ):
        model.train()
        total_loss, nb = 0.0, 0
        for batch in pbar(
            train_ld, desc=f"Train e{ep}", total=len(train_ld), enable=args.pbar
        ):
            mfccs, labels, mfcc_len, *_ = batch
            x = mfccs.to(DEVICE, non_blocking=True).float()

            logits = model(x)  # (B,T,C)
            Tlen = logits.shape[1]
            logp = F.log_softmax(logits, dim=2).permute(1, 0, 2).contiguous()  # (T,B,C)
            in_len = mfcc_len.to(DEVICE, non_blocking=True).clamp_max(Tlen)
            targets, tgt_len = labels_to_targets(labels, DEVICE, vocab_with_blank)

            loss = crit(logp, targets, in_len, tgt_len)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.W_out.parameters(), 1.0)
            optim.step()

            total_loss += float(loss.item())
            nb += 1

        avg_loss = total_loss / max(1, nb)
        tqdm.write(f"[Train] ep{ep} loss={avg_loss:.4f}")

        # 评估
        preds = evaluate(
            model,
            test_ld,
            test_ds.caption_encoder.idx_to_char,
            pbar_on=args.pbar,
            live_acc=args.live_acc,
        )
        char_ok, char_tot = 0, 0
        for _, truth, pred in preds:
            if truth == pred:
                char_ok += 1
            for k, ch in enumerate(truth):
                if k < len(pred) and pred[k] == ch:
                    char_ok += 1
            char_tot += len(truth)
        char_acc = char_ok / max(1, char_tot)
        tqdm.write(f"[Eval] char-accuracy: {char_ok}/{char_tot} = {char_acc:.4f}")

        if char_acc > best_char:
            best_char = char_acc
            torch.save(
                {
                    "W_res_int": model.W_res_i4,
                    "S_res_row": model.S_res_row_np,
                    "W_in_int": model.W_in_i4,
                    "S_in_row": model.S_in_row_np,
                    "input_int_scale": model.input_int_scale,
                    "W_out_state": model.W_out.state_dict(),
                    "g_res": float(model.g_res.item())
                    if model.g_res is not None
                    else 1.0,
                    "g_in": float(model.g_in.item()) if model.g_in is not None else 1.0,
                    "best_char_acc": best_char,
                },
                args.save,
            )
            tqdm.write(f"[Save] best={best_char:.4f} -> {args.save}")

    tqdm.write("训练完成。")


if __name__ == "__main__":
    main()

# python '/home/inubash#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sdk_esn.py — 用 SDK 等价仿真在 CUDA 上训练 ESN
- 小模型(≤576x128) 直接用 CudaSDKArraySim.calculate（严格 SDK 接口）
- 大模型自动用等价的 batched GEMV 快仿真（语义一致，速度更快）
- 架构保持：int4 权重 × 整数输入 → int16 饱和 → 行尺度/增益 → tanh → 线性 W_out → CTC
"""

import argparse, random, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm.auto import tqdm

# === 你的数据集/配置（保持不变） ===
from dataset import TidigitDataset, collate_fn
from config import RAW_TEST_FOLDER_ABS_PATH, RAW_TRAIN_FOLDER_ABS_PATH

# === SDK 等价 CUDA 仿真（严格接口） ===
from he100_cuda_sdk import CudaSDKArraySim, DEVICE

# === 等价快仿真（batched GEMV，数值语义同 SDK） ===
from he100_cuda import torch_batched_mv_int4_fast


# -------------------------- 实用函数 --------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pbar(it, desc="", total=None, enable=True, leave=False):
    return (
        tqdm(it, desc=desc, total=total, dynamic_ncols=True, leave=leave)
        if enable
        else it
    )


def ctc_collapse(seq, blank=0):
    out, prev = [], None
    for t in seq:
        if t == blank:
            prev = t
            continue
        if t != prev:
            out.append(t)
        prev = t
    return out


def power_iteration_numpy(W: np.ndarray, iters: int = 100) -> float:
    n = W.shape[0]
    x = np.random.randn(n).astype(np.float64)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(iters):
        x = W @ x
        x /= np.linalg.norm(x) + 1e-12
    lam = float(x @ (W @ x))
    return abs(lam)


def quantize_rowwise_int4(W: np.ndarray, mode="percentile", perc=99.0, eps=1e-8):
    Wf = W.astype(np.float32, copy=False)
    H, C = Wf.shape
    q = np.empty_like(Wf, dtype=np.int8)
    s = np.empty((H,), dtype=np.float32)
    for i in range(H):
        row = Wf[i]
        a = np.abs(row)
        if mode == "percentile":
            thr = np.percentile(a, perc)
            scale = max(thr, eps) / 7.0
        else:
            scale = max(a.max(), eps) / 7.0
        qi = np.round(row / scale).astype(np.int32)
        q[i] = np.clip(qi, -8, 7).astype(np.int8)
        s[i] = scale
    return q, s


# -------------------------- SDK/快仿真调度 --------------------------
MAX_R, MAX_C = 576, 128


class SDKOrFastMatvec:
    """
    统一入口：
      - 若 (rows<=576 且 cols<=128)，用严格 SDK 仿真 (CudaSDKArraySim)
      - 否则，用等价 batched GEMV（torch_batched_mv_int4_fast）
    接口：
      y = mv_res(h_int)  # (B,H) -> (B,H)
      y = mv_in(u_int)   # (B,F) -> (B,H)
    """

    def __init__(self, W_res_i4: np.ndarray, W_in_i4: np.ndarray):
        self.W_res_i4_np = W_res_i4
        self.W_in_i4_np = W_in_i4
        H, H2 = W_res_i4.shape
        H3, F = W_in_i4.shape
        assert H == H2 == H3
        self.H, self.F = H, F

        self.use_sdk_res = H <= MAX_R and H <= MAX_C  # res: (H,H)
        self.use_sdk_in = H <= MAX_R and self.F <= MAX_C  # in : (H,F)

        # 常驻 CUDA 权重（用于 fast 路径）
        self.W_res_i8 = torch.from_numpy(W_res_i4).to(DEVICE, non_blocking=True)
        self.W_in_i8 = torch.from_numpy(W_in_i4).to(DEVICE, non_blocking=True)

        # SDK 仿真实例（小模型时使用），并把权重写入
        self.sim_res = None
        self.sim_in = None
        if self.use_sdk_res:
            self.sim_res = CudaSDKArraySim(MAX_R, MAX_C)
            self.sim_res.set_weight_int4(W_res_i4, [0, 0, H, H])
        if self.use_sdk_in:
            self.sim_in = CudaSDKArraySim(MAX_R, MAX_C)

    @torch.inference_mode()
    def mv_res(self, H_int32_BH: torch.Tensor) -> torch.Tensor:
        """
        输入： (B,H) int32
        输出： (B,H) int16 语义（这里直接用 int32 保存，后续再饱和/缩放）
        """
        assert H_int32_BH.dtype in (torch.int32, torch.int16)
        B, H = H_int32_BH.shape
        if self.use_sdk_res:
            # 严格 SDK：每个 batch 调一次 calculate（ret_mode=0，expand_mode=1/0 都等价）
            out_list = []
            H_cpu = H_int32_BH.detach().cpu().numpy().astype(np.int32)
            for b in range(B):
                y = self.sim_res.calculate(
                    H_cpu[b : b + 1, :],
                    addr=[0, 0, self.H, self.H],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )  # (1,H)
                out_list.append(torch.from_numpy(y))
            Y = torch.cat(out_list, dim=0).to(DEVICE)
            return Y.to(torch.int32)
        else:
            # 快仿真（等价语义）：(B,H) × (H,H)^T -> (B,H)
            return torch_batched_mv_int4_fast(self.W_res_i8, H_int32_BH, it_time=1).to(
                torch.int32
            )

    @torch.inference_mode()
    def mv_in(self, U_int32_BF: torch.Tensor) -> torch.Tensor:
        """
        输入： (B,F) int32
        输出： (B,H) int32
        """
        assert U_int32_BF.dtype in (torch.int32, torch.int16)
        B, F = U_int32_BF.shape
        if self.use_sdk_in:
            out_list = []
            U_cpu = U_int32_BF.detach().cpu().numpy().astype(np.int32)
            for b in range(B):
                y = self.sim_in.calculate(
                    U_cpu[b : b + 1, :],
                    addr=[0, 0, self.H, self.F],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )  # (1,H)  注意 W_in 的形状是 (H,F)，SDK 的输出按 (num,w) = (1,H)
                out_list.append(torch.from_numpy(y))
            Y = torch.cat(out_list, dim=0).to(DEVICE)
            return Y.to(torch.int32)
        else:
            # 快仿真： (B,F) × (F,H) = (B,H)
            return torch_batched_mv_int4_fast(self.W_in_i8, U_int32_BF, it_time=1).to(
                torch.int32
            )


# -------------------------- 模型 --------------------------
class SDK_ESN_Model(nn.Module):
    """
    仍然：int4 权重 × 整数输入 → int16 饱和 → 行尺度/增益 → tanh → W_out
    - 权重行量化（-8..7）+ 行尺度 s_row
    - 可学增益 g_res/g_in（若开启）
    """

    def __init__(
        self,
        W_res_float: np.ndarray,
        W_in_float: np.ndarray,
        vocab_with_blank: int,
        *,
        input_int_scale: float = 3.0,
        row_quant_perc: float = 99.0,
        match_spectral: bool = True,
        learn_gains: bool = True,
        init_hidden_std_target: float = 0.35,
        seed: int = 1337,
    ):
        super().__init__()
        set_seed(seed)

        # 行量化到 int4
        self.W_res_i4, self.S_res_row_np = quantize_rowwise_int4(
            W_res_float, mode="percentile", perc=row_quant_perc
        )
        self.W_in_i4, self.S_in_row_np = quantize_rowwise_int4(
            W_in_float, mode="percentile", perc=row_quant_perc
        )

        self.H = int(self.W_res_i4.shape[0])
        self.F = int(self.W_in_i4.shape[1])

        # 谱半径匹配（与你之前一致）
        if match_spectral:
            rho_orig = power_iteration_numpy(W_res_float.astype(np.float64), iters=120)
            approx = (self.S_res_row_np[:, None] / 7.0).astype(
                np.float64
            ) * self.W_res_i4.astype(np.float64)
            rho_quant = power_iteration_numpy(approx, iters=120)
            gamma = float(rho_orig / max(rho_quant, 1e-12))
            self.S_res_row_np = (self.S_res_row_np * gamma).astype(np.float32)
            tqdm.write(
                f"[Spectral] rho_orig={rho_orig:.6f} rho_quant={rho_quant:.6f} gamma={gamma:.6f}"
            )

        # CUDA 常量
        self.S_res_row = torch.from_numpy(self.S_res_row_np).to(DEVICE)
        self.S_in_row = torch.from_numpy(self.S_in_row_np).to(DEVICE)

        # SDK or fast dispatcher
        self.mv = SDKOrFastMatvec(self.W_res_i4, self.W_in_i4)

        # 可学增益
        self.g_res = (
            nn.Parameter(torch.tensor(1.0, device=DEVICE)) if learn_gains else None
        )
        self.g_in = (
            nn.Parameter(torch.tensor(1.0, device=DEVICE)) if learn_gains else None
        )

        # 输出层
        self.W_out = nn.Linear(self.H, vocab_with_blank, device=DEVICE)

        self.input_int_scale = float(input_int_scale)
        self.init_hidden_std_target = float(init_hidden_std_target)

    def _row_scales(self):
        s_res = self.S_res_row / 49.0
        s_in = self.S_in_row / (7.0 * max(1.0, self.input_int_scale))
        if self.g_res is not None:
            s_res = s_res * self.g_res
        if self.g_in is not None:
            s_in = s_in * self.g_in
        return s_res.view(1, -1), s_in.view(1, -1)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,F) float32 (CUDA)
        return: (B,T,H) float32
        """
        assert x.is_cuda
        B, T, Fdim = x.shape
        assert Fdim == self.F
        s_res, s_in = self._row_scales()  # (1,H)

        h = torch.zeros(B, self.H, device=DEVICE, dtype=torch.float32)
        outs = []
        for t in range(T):
            u_int = torch.round(x[:, t, :].detach() * self.input_int_scale).to(
                torch.int32
            )  # (B,F)
            h_int = torch.round(h.detach().clamp(-1.0, 1.0) * 7.0).to(
                torch.int32
            )  # (B,H)

            # --- 核心：用 SDK 或快仿真执行“整数点积 → it_time(=1) → int16 饱和（这里以 int32 存）”
            out_res_i32 = self.mv.mv_res(h_int)  # (B,H)
            out_in_i32 = self.mv.mv_in(u_int)  # (B,H)

            # 饱和在行尺度之前（硬件等价）
            out_res_i16 = (
                torch.clamp(out_res_i32, -32768, 32767)
                .to(torch.int16)
                .to(torch.float32)
            )
            out_in_i16 = (
                torch.clamp(out_in_i32, -32768, 32767).to(torch.int16).to(torch.float32)
            )

            a = out_res_i16 * s_res + out_in_i16 * s_in  # (B,H)
            h = torch.tanh(a)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hs = self.forward_hidden(x)  # (B,T,H)
        B, T, H = Hs.shape
        return self.W_out(Hs.reshape(B * T, H)).reshape(B, T, -1)

    @torch.no_grad()
    def auto_gain_tune(
        self, calib_loader, steps=6, factor=1.25, target_std=None, pbar_on=True
    ):
        if self.g_res is None or self.g_in is None:
            tqdm.write("[AutoGain] learn_gains=False，跳过校准")
            return
        tgt = float(
            target_std if target_std is not None else self.init_hidden_std_target
        )

        def measure_std() -> float:
            for batch in calib_loader:
                mfccs, *_ = batch
                Hs = self.forward_hidden(mfccs.to(DEVICE, non_blocking=True).float())
                return float(Hs.std().item())
            return 0.0

        std0 = measure_std()
        tqdm.write(f"[AutoGain] start std={std0:.4f} → target={tgt:.4f}")
        g = float(self.g_res.item())
        for it in pbar(
            range(steps), desc="auto-gain", total=steps, enable=pbar_on, leave=False
        ):
            if std0 < tgt * 0.98:
                g *= factor
            elif std0 > tgt * 1.02:
                g /= factor
            else:
                break
            self.g_res.copy_(torch.tensor(g, device=DEVICE))
            self.g_in.copy_(torch.tensor(g, device=DEVICE))
            std0 = measure_std()
            tqdm.write(f"[AutoGain] iter{it + 1}: std={std0:.4f}, g={g:.4f}")
        tqdm.write(f"[AutoGain] done: std={std0:.4f}, g={g:.4f}")


# -------------------------- 训练/评估 --------------------------
def labels_to_targets(labels, device, vocab_with_blank: int):
    idx_list = []
    if isinstance(labels, list):
        for L in labels:
            if L.dim() == 2:
                idx = torch.argmax(L, dim=1) + 1
            else:
                idx = L[L != -1] + 1
            idx_list.append(idx.to(device))
    else:
        raise ValueError("labels expect list of Tensors")

    target_lengths = torch.tensor(
        [len(x) for x in idx_list], dtype=torch.long, device=device
    )
    targets = torch.cat(idx_list).to(device)
    assert int(targets.max().item()) < vocab_with_blank, "targets exceed vocab(+blank)"
    return targets, target_lengths


@torch.no_grad()
def evaluate(
    model: SDK_ESN_Model,
    dataloader,
    idx_to_char: dict[int, str],
    *,
    pbar_on=True,
    live_acc=False,
):
    model.eval()
    preds = []
    for batch in pbar(dataloader, desc="Eval", total=len(dataloader), enable=pbar_on):
        mfccs, labels, mfcc_lengths, paths, captions = batch
        x = mfccs.to(DEVICE, non_blocking=True).float()

        logits = model(x)  # (B,T,C)
        y = logits.permute(1, 0, 2)  # (T,B,C)
        ids = torch.argmax(y, dim=2).cpu().numpy()  # (T,B)

        B = ids.shape[1]
        inner = pbar(
            range(B), desc="Decode", total=B, enable=pbar_on and live_acc, leave=False
        )

        for i in inner:
            raw = ids[:, i].tolist()
            seq = ctc_collapse(raw, blank=0)
            chars = []
            for t in seq:
                if t == 0:
                    continue
                r = t - 1
                if r in idx_to_char:
                    chars.append(idx_to_char[r])
            pred = "".join(chars)
            truth = captions[i]
            preds.append((paths[i], truth, pred))
        break
    return preds


def build_random_reservoir(H: int, F: int, spectral_radius: float, seed: int = 1337):
    rng = np.random.default_rng(seed)
    W_res = rng.uniform(-0.5, 0.5, size=(H, H)).astype(np.float32)
    eigvals = np.linalg.eigvals(W_res.astype(np.float64))
    rho = np.max(np.abs(eigvals))
    W_res *= spectral_radius / max(float(rho), 1e-9)
    W_in = rng.uniform(-0.5, 0.5, size=(H, F)).astype(np.float32)
    return W_res, W_in


# -------------------------- CLI --------------------------
def get_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=2000)
    ap.add_argument("--input-size", type=int, default=13)
    ap.add_argument("--spectral-radius", type=float, default=0.9)
    ap.add_argument("--input-int-scale", type=float, default=3.0)
    ap.add_argument("--row-quant-perc", type=float, default=99.0)
    ap.add_argument("--no-spectral-match", action="store_true")
    ap.add_argument("--no-learn-gains", action="store_true")
    ap.add_argument("--auto-gain-steps", type=int, default=6)
    ap.add_argument("--auto-gain-target-std", type=float, default=0.35)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--pbar", action="store_true")
    ap.add_argument("--live-acc", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default="sdk_train_ckpt.pth")
    return ap.parse_args()


# -------------------------- 主程序 --------------------------
def main():
    args = get_cli()
    set_seed(args.seed)

    # 数据
    train_ds = TidigitDataset(use_onehot=True)
    test_ds = TidigitDataset(data_dir=RAW_TRAIN_FOLDER_ABS_PATH, use_onehot=True)

    def take_subset(ds, frac: float):
        if frac >= 0.999 or frac <= 0.0:
            return ds
        all_paths = list(ds.dataset_path_list)
        n_all = len(all_paths)
        k = max(1, int(n_all * frac))
        idx = np.random.permutation(n_all)[:k]
        keep_paths = [all_paths[i] for i in idx]
        ds.data_cap_mfcc_dict = {p: ds.data_cap_mfcc_dict[p] for p in keep_paths}
        ds.dataset_path_list = keep_paths
        return ds

    train_ds = take_subset(train_ds, args.subset)
    test_ds = take_subset(test_ds, args.subset)

    train_ld = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_ld = data.DataLoader(
        test_ds,
        batch_size=max(1, len(test_ds)),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    vocab_with_blank = len(train_ds.caption_encoder.char_vocab) + 1  # 36 + blank

    # 库
    W_res_f, W_in_f = build_random_reservoir(
        args.hidden, args.input_size, args.spectral_radius, seed=args.seed
    )

    model = SDK_ESN_Model(
        W_res_f,
        W_in_f,
        vocab_with_blank=vocab_with_blank,
        input_int_scale=args.input_int_scale,
        row_quant_perc=args.row_quant_perc,
        match_spectral=(not args.no_spectral_match),
        learn_gains=(not args.no_learn_gains),
        init_hidden_std_target=args.auto_gain_target_std,
        seed=args.seed,
    ).to(DEVICE)

    # Auto-gain（小校准集）
    calib_ds = take_subset(TidigitDataset(use_onehot=True), min(0.02, args.subset))
    calib_ld = data.DataLoader(
        calib_ds,
        batch_size=min(8, len(calib_ds)),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model.auto_gain_tune(
        calib_ld,
        steps=args.auto_gain_steps,
        target_std=args.auto_gain_target_std,
        pbar_on=args.pbar,
    )

    # 只训 W_out 和 g_res/g_in（若开启）
    params = list(model.W_out.parameters())
    if model.g_res is not None:
        params += [model.g_res]
    if model.g_in is not None:
        params += [model.g_in]
    optim = torch.optim.Adam(params, lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    best_char = -1.0
    for ep in pbar(
        range(1, args.epochs + 1), desc="Epochs", total=args.epochs, enable=args.pbar
    ):
        model.train()
        total_loss, nb = 0.0, 0
        for batch in pbar(
            train_ld, desc=f"Train e{ep}", total=len(train_ld), enable=args.pbar
        ):
            mfccs, labels, mfcc_len, *_ = batch
            x = mfccs.to(DEVICE, non_blocking=True).float()

            logits = model(x)  # (B,T,C)
            Tlen = logits.shape[1]
            logp = F.log_softmax(logits, dim=2).permute(1, 0, 2).contiguous()  # (T,B,C)
            in_len = mfcc_len.to(DEVICE, non_blocking=True).clamp_max(Tlen)
            targets, tgt_len = labels_to_targets(labels, DEVICE, vocab_with_blank)

            loss = crit(logp, targets, in_len, tgt_len)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.W_out.parameters(), 1.0)
            optim.step()

            total_loss += float(loss.item())
            nb += 1

        avg_loss = total_loss / max(1, nb)
        tqdm.write(f"[Train] ep{ep} loss={avg_loss:.4f}")

        # 评估
        preds = evaluate(
            model,
            test_ld,
            test_ds.caption_encoder.idx_to_char,
            pbar_on=args.pbar,
            live_acc=args.live_acc,
        )
        char_ok, char_tot = 0, 0
        for _, truth, pred in preds:
            if truth == pred:
                char_ok += 1
            for k, ch in enumerate(truth):
                if k < len(pred) and pred[k] == ch:
                    char_ok += 1
            char_tot += len(truth)
        char_acc = char_ok / max(1, char_tot)
        tqdm.write(f"[Eval] char-accuracy: {char_ok}/{char_tot} = {char_acc:.4f}")

        if char_acc > best_char:
            best_char = char_acc
            torch.save(
                {
                    "W_res_int": model.W_res_i4,
                    "S_res_row": model.S_res_row_np,
                    "W_in_int": model.W_in_i4,
                    "S_in_row": model.S_in_row_np,
                    "input_int_scale": model.input_int_scale,
                    "W_out_state": model.W_out.state_dict(),
                    "g_res": float(model.g_res.item())
                    if model.g_res is not None
                    else 1.0,
                    "g_in": float(model.g_in.item()) if model.g_in is not None else 1.0,
                    "best_char_acc": best_char,
                },
                args.save,
            )
            tqdm.write(f"[Save] best={best_char:.4f} -> {args.save}")

    tqdm.write("训练完成。")


if __name__ == "__main__":
    main()

# python '/home/inubashiri/01_ESN/TidigitClassification/experiments/train_sdk_esn.py' --hidden 512 --input-size 13 --epochs 200 --pbar --live-acc

# 我希望你使用这个做一个推理脚本出来iri/01_ESN/TidigitClassification/experiments/train_sdk_esn.py' --hidden 512 --input-size 13 --epochs 200 --pbar --live-acc
