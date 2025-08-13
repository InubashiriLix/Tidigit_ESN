#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_sdk_esn.py — 从训练 ckpt 载入并在 TIDIGITS 上推理/评估
- 严格仿真语义：int4 权重 × 整数输入 → (×it_time=1) → int16 饱和 → 行尺度/增益 → tanh → W_out
- 默认按阵列上限(576x128)决定后端：小模型走严格 SDK 仿真；大模型用等价 CUDA 快仿真
- 若加 --strict-sdk，则强制只用严格 SDK（超限直接报错）
"""

import argparse, random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.utils.data as data

# 你的数据与配置
from dataset import TidigitDataset, collate_fn
from config import RAW_TEST_FOLDER_ABS_PATH

# 严格仿真（CUDA）
from he100_cuda_sdk import CudaSDKArraySim, DEVICE

# 等价快仿真（batched GEMV）
from he100_cuda import torch_batched_mv_int4_fast


# ----------------- 工具 -----------------
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


# ----------------- SDK/快仿真调度 -----------------
MAX_R, MAX_C = 576, 128


class SDKMatvec:
    """
    统一入口（推理用）：
      - strict=False: 小模型走严格 SDK；大模型走等价 CUDA 快仿真
      - strict=True : 只走严格 SDK，超限直接报错
    提供：
      y_res = mv_res(h_int)  # (B,H) -> (B,H)
      y_in  = mv_in(u_int)   # (B,F) -> (B,H)
    """

    def __init__(self, W_res_i4: np.ndarray, W_in_i4: np.ndarray, *, strict: bool):
        self.W_res_i4_np = W_res_i4
        self.W_in_i4_np = W_in_i4
        H, H2 = W_res_i4.shape
        H3, F = W_in_i4.shape
        assert H == H2 == H3
        self.H, self.F = H, F

        if strict:
            if H > MAX_R or H > MAX_C or F > MAX_C:
                raise ValueError(
                    f"[strict-sdk] exceed array size: W_res={H}x{H}, W_in={H}x{F}, need H<=576 and H,F<=128"
                )
            self.use_sdk_res = True
            self.use_sdk_in = True
        else:
            self.use_sdk_res = H <= MAX_R and H <= MAX_C
            self.use_sdk_in = H <= MAX_R and F <= MAX_C

        # fast 路径的常驻权重
        self.W_res_i8 = torch.from_numpy(W_res_i4).to(DEVICE, non_blocking=True)
        self.W_in_i8 = torch.from_numpy(W_in_i4).to(DEVICE, non_blocking=True)

        # SDK 仿真实例（小模型/strict 时使用），写入一次权重
        self.sim_res = None
        self.sim_in = None
        if self.use_sdk_res:
            self.sim_res = CudaSDKArraySim(MAX_R, MAX_C)
            self.sim_res.set_weight_int4(W_res_i4, [0, 0, H, H])
        if self.use_sdk_in:
            self.sim_in = CudaSDKArraySim(MAX_R, MAX_C)
            self.sim_in.set_weight_int4(W_in_i4, [0, 0, H, F])

    @torch.inference_mode()
    def mv_res(self, H_int32_BH: torch.Tensor) -> torch.Tensor:
        assert H_int32_BH.dtype in (torch.int32, torch.int16)
        B, H = H_int32_BH.shape
        if self.use_sdk_res:
            out = []
            X = H_int32_BH.detach().cpu().numpy().astype(np.int32)
            for b in range(B):
                y = self.sim_res.calculate(
                    X[b : b + 1, :],
                    [0, 0, self.H, self.H],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )
                out.append(torch.from_numpy(y))
            return torch.cat(out, dim=0).to(DEVICE).to(torch.int32)
        else:
            return torch_batched_mv_int4_fast(self.W_res_i8, H_int32_BH, it_time=1).to(
                torch.int32
            )

    @torch.inference_mode()
    def mv_in(self, U_int32_BF: torch.Tensor) -> torch.Tensor:
        assert U_int32_BF.dtype in (torch.int32, torch.int16)
        B, F = U_int32_BF.shape
        if self.use_sdk_in:
            out = []
            X = U_int32_BF.detach().cpu().numpy().astype(np.int32)
            for b in range(B):
                y = self.sim_in.calculate(
                    X[b : b + 1, :],
                    [0, 0, self.H, self.F],
                    it_time=1,
                    data_type=-1,
                    expand_mode=1,
                    ret_mode=0,
                )
                out.append(torch.from_numpy(y))  # (1,H)
            return torch.cat(out, dim=0).to(DEVICE).to(torch.int32)
        else:
            return torch_batched_mv_int4_fast(self.W_in_i8, U_int32_BF, it_time=1).to(
                torch.int32
            )


# ----------------- 推理模型（从 ckpt 直接载入量化权重） -----------------
class SDK_ESN_Infer(nn.Module):
    """
    - 直接使用 ckpt 中的 int4 权重与行尺度（不再做行量化/谱匹配）
    - 仅做前向推理；g_res/g_in 为常量（来自 ckpt）
    """

    def __init__(
        self,
        W_res_i4: np.ndarray,
        S_res_row: np.ndarray,
        W_in_i4: np.ndarray,
        S_in_row: np.ndarray,
        vocab_with_blank: int,
        *,
        input_int_scale: float,
        g_res: float,
        g_in: float,
        strict_sdk: bool,
    ):
        super().__init__()
        self.H = int(W_res_i4.shape[0])
        self.F = int(W_in_i4.shape[1])

        # 行尺度 & 增益（常量）
        self.S_res_row = torch.from_numpy(S_res_row.astype(np.float32)).to(DEVICE)
        self.S_in_row = torch.from_numpy(S_in_row.astype(np.float32)).to(DEVICE)
        self.g_res = torch.tensor(float(g_res), device=DEVICE)
        self.g_in = torch.tensor(float(g_in), device=DEVICE)

        # SDK/fast 调度器
        self.mv = SDKMatvec(W_res_i4, W_in_i4, strict=strict_sdk)

        # 输出层
        self.W_out = nn.Linear(self.H, vocab_with_blank, device=DEVICE)

        self.input_int_scale = float(input_int_scale)

    def _row_scales(self):
        s_res = (self.S_res_row / 49.0) * self.g_res
        s_in = (self.S_in_row / (7.0 * max(1.0, self.input_int_scale))) * self.g_in
        return s_res.view(1, -1), s_in.view(1, -1)

    @torch.inference_mode()
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

            out_res_i32 = self.mv.mv_res(h_int)  # (B,H)
            out_in_i32 = self.mv.mv_in(u_int)  # (B,H)

            out_res_i16 = (
                torch.clamp(out_res_i32, -32768, 32767)
                .to(torch.int16)
                .to(torch.float32)
            )
            out_in_i16 = (
                torch.clamp(out_in_i32, -32768, 32767).to(torch.int16).to(torch.float32)
            )

            a = out_res_i16 * s_res + out_in_i16 * s_in
            h = torch.tanh(a)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hs = self.forward_hidden(x)  # (B,T,H)
        B, T, H = Hs.shape
        return self.W_out(Hs.reshape(B * T, H)).reshape(B, T, -1)


# ----------------- CLI / 主逻辑 -----------------
def get_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="sdk_train_ckpt.pth")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pbar", action="store_true")
    ap.add_argument(
        "--strict-sdk",
        action="store_true",
        help="强制仅用严格 SDK 仿真，超阵列上限将报错",
    )
    ap.add_argument(
        "--save-preds",
        type=str,
        default="",
        help="可选：保存 (path, truth, pred) 到 TSV",
    )
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = get_cli()
    set_seed(args.seed)

    # ---- 数据（测试集） ----
    test_ds = TidigitDataset(data_dir=RAW_TEST_FOLDER_ABS_PATH, use_onehot=True)

    # 子集
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

    test_ds = take_subset(test_ds, args.subset)
    test_ld = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    vocab_with_blank = len(test_ds.caption_encoder.char_vocab) + 1
    idx_to_char = test_ds.caption_encoder.idx_to_char

    # ---- 载入 ckpt ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    W_res_i4 = ckpt["W_res_int"]  # np.ndarray int8 [H,H]
    S_res_row = ckpt["S_res_row"]  # np.ndarray float32 [H]
    W_in_i4 = ckpt["W_in_int"]  # np.ndarray int8 [H,F]
    S_in_row = ckpt["S_in_row"]  # np.ndarray float32 [H]
    input_int_scale = float(ckpt.get("input_int_scale", 3.0))
    g_res = float(ckpt.get("g_res", 1.0))
    g_in = float(ckpt.get("g_in", 1.0))
    W_out_state = ckpt["W_out_state"]

    # ---- 构建推理模型并加载线性层 ----
    model = SDK_ESN_Infer(
        W_res_i4=W_res_i4,
        S_res_row=S_res_row,
        W_in_i4=W_in_i4,
        S_in_row=S_in_row,
        vocab_with_blank=vocab_with_blank,
        input_int_scale=input_int_scale,
        g_res=g_res,
        g_in=g_in,
        strict_sdk=args.strict_sdk,
    ).to(DEVICE)
    model.W_out.load_state_dict(W_out_state)
    model.eval()

    # ---- 推理/解码 ----
    all_preds = []
    seq_ok = seq_tot = 0
    char_ok = char_tot = 0

    for batch in pbar(test_ld, desc="Infer", total=len(test_ld), enable=args.pbar):
        mfccs, labels, mfcc_lengths, paths, captions = batch
        x = mfccs.to(DEVICE, non_blocking=True).float()

        logits = model(x)  # (B,T,C)
        y = logits.permute(1, 0, 2)  # (T,B,C)
        ids = torch.argmax(y, dim=2).cpu().numpy()  # (T,B)

        B = ids.shape[1]
        for i in range(B):
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
            all_preds.append((paths[i], truth, pred))

            # 统计
            seq_tot += 1
            if pred == truth:
                seq_ok += 1
            L = len(truth)
            char_tot += L
            for k, ch in enumerate(truth):
                if k < len(pred) and pred[k] == ch:
                    char_ok += 1

    # ---- 打印指标 ----
    seq_acc = seq_ok / max(1, seq_tot)
    char_acc = char_ok / max(1, char_tot)
    print(
        f"[Eval] seq-acc={seq_ok}/{seq_tot}={seq_acc:.4f} | char-acc={char_ok}/{char_tot}={char_acc:.4f}"
    )

    # ---- 可选保存结果 ----
    if args.save_preds:
        with open(args.save_preds, "w", encoding="utf-8") as f:
            for p, t, q in all_preds:
                f.write(f"{p}\t{t}\t{q}\n")
        print(f"[Save] preds -> {args.save_preds}")


if __name__ == "__main__":
    main()

#  python inference.py \
#   --ckpt sdk_train_ckpt.pth \
#   --batch-size 64 \
#   --subset 1.0 \
#   --pbar \
#   --strict-sdk
#
