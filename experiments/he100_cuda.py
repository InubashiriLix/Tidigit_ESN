# he100_cuda.py
import torch

__all__ = [
    "torch_mv_int4_fast",
    "torch_batched_mv_int4_fast",
    "torch_sdk_mv_int4_fast",
]


def _sanitize_it_time(it_time: int) -> int:
    it = int(it_time)
    if it < 1:
        it = 1
    if it > 63:
        it = 63
    return it


@torch.inference_mode()
def torch_mv_int4_fast(
    W_int8: torch.Tensor, v_int32: torch.Tensor, it_time: int = 1
) -> torch.Tensor:
    """
    单向量 matvec (CUDA)：
      W_int8: (R,C) torch.int8  cuda
      v_int32: (C,) torch.int32 cuda
    返回 (R,) torch.int16，语义：int32 求和 → *it_time → int16 饱和。
    实现以 FP32 GEMV 代替整数求和（在本范围内无损），再 clamp 到 int16。
    """
    assert W_int8.is_cuda and v_int32.is_cuda, "Please move inputs to CUDA"
    it = _sanitize_it_time(it_time)
    acc = torch.matmul(W_int8.float(), v_int32.float()) * float(it)  # (R,)
    acc = torch.clamp(acc, -32768.0, 32767.0)
    return acc.to(torch.int16)


@torch.inference_mode()
def torch_batched_mv_int4_fast(
    W_int8: torch.Tensor, V_int32: torch.Tensor, it_time: int = 1
) -> torch.Tensor:
    """
    批量 matvec (CUDA)：
      W_int8: (R,C) int8  cuda
      V_int32: (B,C) int32 cuda
    返回 (B,R) int16
    """
    assert W_int8.is_cuda and V_int32.is_cuda, "Please move inputs to CUDA"
    it = _sanitize_it_time(it_time)
    acc = torch.matmul(V_int32.float(), W_int8.float().t()) * float(it)  # (B,R)
    acc = torch.clamp(acc, -32768.0, 32767.0)
    return acc.to(torch.int16)


@torch.inference_mode()
def torch_sdk_mv_int4_fast(
    W_int8: torch.Tensor,
    v_or_V_int32: torch.Tensor,
    *,
    rows_step=576,
    cols_step=128,
    it_time=1,
) -> torch.Tensor:
    """
    兼容“分块接口”的直通实现。CUDA 下通常不需要分块。
    - 1D 输入：返回 (R,)
    - 2D 输入：返回 (B,R)
    """
    if v_or_V_int32.dim() == 1:
        return torch_mv_int4_fast(W_int8, v_or_V_int32, it_time)
    elif v_or_V_int32.dim() == 2:
        return torch_batched_mv_int4_fast(W_int8, v_or_V_int32, it_time)
    else:
        raise ValueError("v_or_V_int32 must be rank-1 or rank-2 tensor")
