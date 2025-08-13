# tests/test_cuda_sdkarray_weight_io.py
import numpy as np
import torch
import pytest

from he100_cuda_sdk import CudaSDKArraySim

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@cuda
def test_weight_set_get_roundtrip_and_ranges():
    sim = CudaSDKArraySim(576, 128)
    addr = [100, 20, 10, 12]

    W_i4 = np.random.randint(-8, 8, (10, 12), dtype=np.int8)
    sim.set_weight_int4(W_i4, addr)
    out_i4 = sim.get_weight_int4(addr)
    np.testing.assert_array_equal(out_i4, W_i4)

    W_u4 = (W_i4.astype(np.int16) + 8).astype(np.uint8)
    out_u4 = sim.get_weight(addr)
    np.testing.assert_array_equal(out_u4, W_u4)


@cuda
def test_it_time_and_shapes():
    sim = CudaSDKArraySim(576, 128)
    h, w = 16, 32
    addr = [0, 0, h, w]
    W = np.full((h, w), 7, dtype=np.int8)
    sim.set_weight_int4(W, addr)

    V = np.full((2, h), 7, dtype=np.int32)
    # it_time 被夹紧到 [1,63]，我们测 63
    out = sim.calculate(
        V, addr=addr, it_time=1000, data_type=3, expand_mode=1, ret_mode=0
    )
    # 参考
    ref = (V.astype(np.int64) @ W.astype(np.int64)) * 63
    np.testing.assert_array_equal(out, ref.astype(np.int32))
