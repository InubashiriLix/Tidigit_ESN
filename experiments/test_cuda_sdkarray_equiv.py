# tests/test_cuda_sdkarray_equiv.py
import numpy as np
import torch
import pytest

from he100_cuda_sdk import CudaSDKArraySim

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@cuda
@pytest.mark.parametrize("expand_mode", [0, 1])
@pytest.mark.parametrize("data_type", [-1, 0, 1, 2, 3, 4, 5, 6, 7])
def test_calculate_matches_direct_matmul(expand_mode, data_type):
    sim = CudaSDKArraySim(576, 128)

    # 随机子块
    h, w = 64, 80
    y, x = 10, 20
    addr = [y, x, h, w]

    # 填权重：int4 [-8..7]
    W_i4 = np.random.randint(-8, 8, size=(h, w), dtype=np.int8)
    sim.set_weight_int4(W_i4, addr)

    # 输入范围按 data_type
    ranges = {
        -1: (-8, 7),
        0: (-1, 1),
        1: (-2, 1),
        2: (-4, 3),
        3: (-8, 7),
        4: (-16, 15),
        5: (-32, 31),
        6: (-64, 63),
        7: (-128, 127),
    }
    lo, hi = ranges[data_type]
    num = 5
    V = np.random.randint(lo, hi + 1, size=(num, h), dtype=np.int32)

    # 仿真计算
    out = sim.calculate(
        V,
        addr=addr,
        it_time=5,
        data_type=data_type,
        expand_mode=expand_mode,
        ret_mode=0,
    )
    # 参考：直接整形乘法累加后乘 it_time（线性等价于“逐脉冲/逐位展开再求和”）
    ref = (V.astype(np.int64) @ W_i4.astype(np.int64)) * 5
    ref = ref.astype(np.int32)

    np.testing.assert_array_equal(out, ref)


@cuda
def test_ret_mode_full_and_bitwise_shapes():
    sim = CudaSDKArraySim(576, 128)
    h, w = 8, 9
    addr = [0, 0, h, w]
    sim.set_weight_int4(np.random.randint(-8, 8, (h, w), dtype=np.int8), addr)

    V = np.array([[1, -2, 0, 3, -1, 2, -3, 0]], dtype=np.int32)  # (1,h)

    # full: L 取决于 max|x|=3 -> 7 脉冲
    Y_full = sim.calculate(
        V, addr=addr, it_time=1, data_type=-1, expand_mode=0, ret_mode=1
    )
    assert Y_full.shape[0] == 1 and Y_full.shape[2] == w
    assert Y_full.shape[1] in (3, 7, 15, 31, 63, 127)

    # bitwise: amp_bits=2（因为 max|x|=3 属于 int3）
    Y_bit = sim.calculate(
        V, addr=addr, it_time=1, data_type=-1, expand_mode=1, ret_mode=1
    )
    assert Y_bit.shape == (1, 2, w)  # (num,B,w)
