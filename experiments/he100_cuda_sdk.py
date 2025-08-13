# he100_cuda_sdk.py
# -*- coding: utf-8 -*-
"""
CUDA 等价快仿真：与 SDKArray 的主要接口严格对齐（权重 IO + calculate）
- 内部存 int4 权重（-8..7），对外 get_weight 返回 0..15、get_weight_int4 返回 -8..7
- calculate:
    * 输入形状 (num, h)，addr=[y,x,h,w]，输出 (num, w)
    * data_type: -1 自动 / 0..7 固定区间
    * expand_mode: 0 全展开（full），1 位展开（bitwise）
    * ret_mode: 0 返回累加后结果 (int32)；1 返回展开前“原始”结果：
        - full: (num, L, w)  每个脉冲一帧
        - bitwise: (num, B, w) 每个幅度位一帧（不乘 2^b）
- it_time: 乘到**最终累加后**（与真实阵列积分一致），结果保持 int32
"""

from __future__ import annotations
import math
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["CudaSDKArraySim"]


def _autodetect_data_type(x_abs_max: int) -> int:
    # 对齐 SDK 注释区间：[-1,1],[ -2,1],[-4,3],...,[-128,127]
    if x_abs_max > 127:
        return 7
    if x_abs_max > 63:
        return 6
    if x_abs_max > 31:
        return 5
    if x_abs_max > 15:
        return 4
    if x_abs_max > 7:
        return 3
    if x_abs_max > 3:
        return 2
    if x_abs_max > 1:
        return 1
    return 0


def _amplitude_bits_from_dtype(dt: int) -> int:
    # int4([-8..7]) => 幅度位=3；int3([-4..3])=>2；int2([-2..1])=>1；int1.5([-1..1])=>1
    if dt <= 0:
        return 1
    return dt


def _sanitize_it_time(it: int) -> int:
    # SDK 接口本身不限制，但真实板卡 1..63 常见；这里夹紧，避免溢出
    it = int(it)
    return 1 if it < 1 else (63 if it > 63 else it)


def _check_addr(addr, max_rows=576, max_cols=128):
    y, x, h, w = map(int, addr)
    if not (0 <= y < y + h <= max_rows and 0 <= x < x + w <= max_cols):
        raise ValueError(f"addr {addr} 超出范围 ([0,{max_rows}],[0,{max_cols}])")
    return y, x, h, w


class CudaSDKArraySim:
    MAX_ROWS = 576
    MAX_COLS = 128

    def __init__(self, rows: int = MAX_ROWS, cols: int = MAX_COLS):
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA 设备来运行 CudaSDKArraySim")
        if not (1 <= rows <= self.MAX_ROWS and 1 <= cols <= self.MAX_COLS):
            raise ValueError("rows/cols exceed device size")
        self.rows = rows
        self.cols = cols
        # 内部以 int4（-8..7）保存
        self._W_i4 = torch.zeros((rows, cols), dtype=torch.int8, device=DEVICE)

    # ---------------------- 权重读写 ----------------------
    def get_weight(self, addr):
        y, x, h, w = _check_addr(addr, self.MAX_ROWS, self.MAX_COLS)
        sub = self._W_i4[y : y + h, x : x + w].to(torch.int16) + 8
        return sub.to(torch.uint8).cpu().numpy()  # [0..15]

    def get_weight_int4(self, addr):
        y, x, h, w = _check_addr(addr, self.MAX_ROWS, self.MAX_COLS)
        return self._W_i4[y : y + h, x : x + w].clone().cpu().numpy()  # [-8..7]

    def set_weight(self, weight: np.ndarray, addr, *, prog_cycle: int = 20):
        y, x, h, w = _check_addr(addr, self.MAX_ROWS, self.MAX_COLS)
        w_np = np.asarray(weight, dtype=np.int16)
        if w_np.shape != (h, w):
            raise ValueError("block shape mismatch vs addr h/w")
        if w_np.min() < 0 or w_np.max() > 15:
            raise ValueError("block values must be in [0,15]")
        w_i4 = (w_np.astype(np.int16) - 8).astype(np.int8)  # 转 int4
        self._W_i4[y : y + h, x : x + w] = torch.from_numpy(w_i4).to(DEVICE)

    def set_weight_int4(self, weight: np.ndarray, addr, *, prog_cycle: int = 20):
        y, x, h, w = _check_addr(addr, self.MAX_ROWS, self.MAX_COLS)
        w_np = np.asarray(weight, dtype=np.int8)
        if w_np.shape != (h, w):
            raise ValueError("block shape mismatch vs addr h/w")
        if w_np.min() < -8 or w_np.max() > 7:
            raise ValueError("block values must be in [-8,7]")
        self._W_i4[y : y + h, x : x + w] = torch.from_numpy(w_np).to(DEVICE)

    # ---------------------- 展开实现（仅 ret_mode=1 需要显式构造） ----------------------
    @staticmethod
    def _full_expand_cpu(x_i: np.ndarray) -> np.ndarray:
        # 逐样本：返回 (L, H) in {-1,0,1}
        v = x_i.astype(np.int32, copy=False)
        max_abs = int(np.max(np.abs(v)))
        thresholds = [3, 7, 15, 31, 63, 127]
        counts = [3, 7, 15, 31, 63, 127]
        for th, cnt in zip(thresholds, counts):
            if max_abs <= th:
                expand_len = cnt
                break
        else:
            raise ValueError("value too large for full-expand table")
        out = np.zeros((expand_len, v.size), dtype=np.int8)
        for k in range(expand_len):
            mask = np.abs(v) > k
            out[k, mask] = np.sign(v[mask]).astype(np.int8)
        return out  # (L,H)

    @staticmethod
    def _bitwise_expand_numeric_cpu(x_i: np.ndarray, amp_bits: int) -> np.ndarray:
        # 逐样本：返回 (B, H) in {-1,0,1}
        v = x_i.astype(np.int32, copy=False)
        signs = np.sign(v).astype(np.int8)
        absv = np.abs(v).astype(np.int32)
        out = np.zeros((amp_bits, v.size), dtype=np.int8)
        for b in range(amp_bits):
            bit = ((absv >> b) & 1).astype(np.int8)
            out[b] = signs * bit
        return out  # (B,H)

    # ---------------------- 计算 ----------------------
    def calculate(
        self,
        input: np.ndarray,
        addr,
        runner=None,
        it_time: int = 5,
        data_type: int = -1,
        expand_mode: int = 1,
        ret_mode: int = 0,
    ):
        """
        等价 SDKArray.calculate（支持两种朝向）：
        - 若 input 长度 = h：输出形状 (num, w)   ← V @ W
        - 若 input 长度 = w：输出形状 (num, h)   ← V @ Wᵀ
        ret_mode=1 时返回展开前“原始帧”（full: (num,L,⋅), bitwise: (num,B,⋅)）
        """
        y, x, h, w = _check_addr(addr, self.MAX_ROWS, self.MAX_COLS)

        x_np = np.asarray(input)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        if x_np.ndim != 2:
            raise ValueError(f"输入数据维度 {x_np.shape} 错误，应为二维 (num, len)")

        num, L = x_np.shape
        if L != h and L != w:
            raise ValueError(
                f"输入长度={L} 与 addr(h={h}, w={w}) 不匹配，应为 h 或 w 之一"
            )

        along_w = (
            L == w
        )  # True: 输入长度等于 w，输出维度为 h；False: 等于 h，输出维度为 w

        # 数据类型 & 幅度位
        x_abs_max = int(np.max(np.abs(x_np))) if x_np.size else 0
        dt = _autodetect_data_type(x_abs_max) if data_type == -1 else int(data_type)
        amp_bits = _amplitude_bits_from_dtype(dt)
        it = _sanitize_it_time(it_time)

        # 取权重块
        W = self._W_i4[y : y + h, x : x + w].to(torch.int32)  # (h,w) on CUDA

        if ret_mode == 0:
            # 直接一次性等价计算（整数和在 FP32 中无损）
            V = torch.from_numpy(x_np.astype(np.int32)).to(DEVICE)
            Vf, Wf = V.float(), W.float()
            if along_w:
                Y = torch.matmul(Vf, Wf.t())  # (num,h)
            else:
                Y = torch.matmul(Vf, Wf)  # (num,w)
            if it != 1:
                Y = Y * float(it)
            return Y.to(torch.int32).cpu().numpy()

        # ret_mode == 1: 返回展开前帧
        outs = []
        if expand_mode == 0:  # full
            for i in range(num):
                pulses = self._full_expand_cpu(x_np[i])  # (Lfull, L)
                P = torch.from_numpy(pulses.astype(np.int32)).to(DEVICE).float()
                Wf = W.float()
                if along_w:
                    Yi = torch.matmul(P, Wf.t())  # (Lfull, h)
                else:
                    Yi = torch.matmul(P, Wf)  # (Lfull, w)
                if it != 1:
                    Yi = Yi * float(it)
                outs.append(Yi.to(torch.int32).unsqueeze(0))  # (1, Lfull, outdim)
            Ystk = torch.cat(outs, dim=0)  # (num, Lfull, outdim)
            return Ystk.cpu().numpy()

        elif expand_mode == 1:  # bitwise
            for i in range(num):
                pulses = self._bitwise_expand_numeric_cpu(x_np[i], amp_bits)  # (B, L)
                P = torch.from_numpy(pulses.astype(np.int32)).to(DEVICE).float()
                Wf = W.float()
                if along_w:
                    Yi = torch.matmul(P, Wf.t())  # (B, h)
                else:
                    Yi = torch.matmul(P, Wf)  # (B, w)
                if it != 1:
                    Yi = Yi * float(it)
                outs.append(Yi.to(torch.int32).unsqueeze(0))  # (1, B, outdim)
            Ystk = torch.cat(outs, dim=0)  # (num, B, outdim)
            return Ystk.cpu().numpy()
        else:
            raise ValueError("expand_mode must be 0(full) or 1(bitwise)")
