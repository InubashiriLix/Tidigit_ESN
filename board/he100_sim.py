#!/usr/bin/env python3
"""he100_sim.py — local simulator for the HE‑100 in‑memory‑computing accelerator.
This module mimics the public SDK described in the datasheet so you can develop
and debug Echo‑State Networks (ESN) or other models *without* direct access to
the real hardware.  All numerical behaviour (quantisation, block writes, MAC
with integration time, bit‑ or full‑expansion) is reproduced as closely as the
documentation allows.

Exposed API (mirrors SDK section 3.5):
    ▸ get_weight(addr)
    ▸ get_weight_int4(addr)
    ▸ set_weight(weight, addr, prog_cycle=20)
    ▸ set_weight_int4(weight, addr, prog_cycle=20)
    ▸ calculate(input, addr, it_time, data_type, expand_mode=1)

A comprehensive unittest suite (run this file directly) verifies that
  • write ↔ read round‑trips are perfect,
  • calculate() matches a NumPy reference dot‑product for random blocks,
  • both bit‑wise & full expansion helpers behave as specified.

Usage example
-------------
>>> from he100_sim import HE100Sim, bitwise_expand
>>> sim = HE100Sim()
>>> w = np.random.randint(-8, 8, (1, 128), dtype=np.int8)
>>> sim.set_weight_int4(w, [0, 0, 1, 128])
>>> x = np.random.randint(-1, 2, 128, dtype=np.int8)  # +1/0/‑1
>>> y = sim.calculate(x, [0, 0, 1, 128], it_time=10, data_type=3)
>>> np.allclose(y, (w @ x) * 10)
True
"""

from __future__ import annotations

import itertools
import math
import unittest
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "HE100Sim",
    "bitwise_expand",
    "full_expand",
    "quantize_to_int4",
    "quantize_to_uint8",
]

###############################################################################
# Simulator core
###############################################################################


class HE100Sim:
    """Pure‑Python simulator for the HE‑100 resistive‑RAM MAC array."""

    # Board limits (fixed for current silicon revision)
    MAX_ROWS = 576
    MAX_COLS = 128

    def __init__(self, rows: int = MAX_ROWS, cols: int = MAX_COLS):
        if not (1 <= rows <= self.MAX_ROWS and 1 <= cols <= self.MAX_COLS):
            raise ValueError("rows/cols exceed device size")
        self.rows = rows
        self.cols = cols
        # Signed 8‑bit storage: we store int4 (‑8..7) directly, but also accept
        # uint8 (0..15) by offsetting later if needed.
        self._array = np.zeros((rows, cols), dtype=np.int8)

    # ---------------------------------------------------------------------
    # Weight read / write helpers (SDK 3.5.1 ‑ 3.5.4)
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_addr(addr: Sequence[int]) -> Tuple[int, int, int, int]:
        """Return (y, x, h, w) verifying the address is inside the array."""
        if len(addr) != 4:
            raise ValueError("addr must have four elements [y, x, h, w]")
        y, x, h, w = map(int, addr)
        if not (0 <= y < HE100Sim.MAX_ROWS and 0 <= x < HE100Sim.MAX_COLS):
            raise ValueError("addr start out of range")
        if not (1 <= h <= HE100Sim.MAX_ROWS - y and 1 <= w <= HE100Sim.MAX_COLS - x):
            raise ValueError("h/w exceed array bounds")
        return y, x, h, w

    # ---------- read ----------

    def get_weight(self, addr: Sequence[int]) -> np.ndarray:
        """Return uint8 [0..15] block as stored."""
        y, x, h, w = self._validate_addr(addr)
        return self._array[y : y + h, x : x + w].astype(np.uint8)

    def get_weight_int4(self, addr: Sequence[int]) -> np.ndarray:
        """Return signed int4 [‑8..7] block."""
        y, x, h, w = self._validate_addr(addr)
        return self._array[y : y + h, x : x + w].copy()

    # ---------- write ----------

    def set_weight(
        self, weight: np.ndarray, addr: Sequence[int], prog_cycle: int = 20
    ) -> None:
        """Write uint8 weight block (0‑15)."""
        self._write_block(weight, addr, allowed=(0, 15))

    def set_weight_int4(
        self, weight: np.ndarray, addr: Sequence[int], prog_cycle: int = 20
    ) -> None:
        """Write signed int4 weight block (‑8..7)."""
        self._write_block(weight, addr, allowed=(-8, 7))

    # Internal common path
    def _write_block(self, block: np.ndarray, addr: Sequence[int], *, allowed):
        y, x, h, w = self._validate_addr(addr)
        block = np.asarray(block)
        if block.shape != (h, w):
            raise ValueError("block shape mismatch vs addr h/w")
        lo, hi = allowed
        if not (block.min() >= lo and block.max() <= hi):
            raise ValueError(f"block values must be in [{lo},{hi}]")
        self._array[y : y + h, x : x + w] = block.astype(np.int8)

    # ---------------------------------------------------------------------
    # MAC operation (SDK 3.5.5)
    # ---------------------------------------------------------------------

    def calculate(
        self,
        input: np.ndarray,
        addr: Sequence[int],
        it_time: int,
        data_type: int,
        expand_mode: int = 1,
    ) -> np.ndarray:
        """Simulate one multiply‑accumulate on a given block.

        *Accepts* both 1‑D and 2‑D *input* to emulate vector‑times‑matrix or
        matrix‑times‑vector.  In practice most ESN cases send a single vector.
        """
        if not (1 <= it_time <= 63):
            raise ValueError("it_time must be 1‑63")
        # data_type merely records the nominal bit‑width; simulation ignores it.
        y, x, h, w = self._validate_addr(addr)
        sub = self._array[y : y + h, x : x + w].astype(np.int32)  # (h,w)
        vec = np.asarray(input, dtype=np.int32)

        # Basic broadcasting rules like hardware: if h==1 treat sub as a row
        # vector; else if w==1 treat as column. Otherwise expect matching dims.
        if h == 1:
            if vec.shape != (w,):
                raise ValueError("input length must match block width")
            acc = int(sub[0] @ vec)
        elif w == 1:
            if vec.shape != (h,):
                raise ValueError("input length must match block height")
            acc = int(vec @ sub[:, 0])
        else:
            # General case: treat input as 2‑D so shapes broadcast for dot.
            if vec.ndim == 1 and vec.shape[0] == w:
                # vec (w,)  → output (h,)
                acc = (sub @ vec).astype(np.int32)
            elif vec.ndim == 1 and vec.shape[0] == h:
                # vec (h,)  → output (w,)
                acc = (vec @ sub).astype(np.int32)
            elif vec.shape == (h, w):
                acc = int(np.sum(sub * vec))
            else:
                raise ValueError("input shape incompatible with block dimensions")

        # Integrate (scale) and clamp to int16 range
        acc = np.asarray(acc) * it_time
        return np.clip(acc, -32768, 32767).astype(np.int16)


###############################################################################
# Helper utilities
###############################################################################


def bitwise_expand(vec: np.ndarray, *, bit_width: int) -> np.ndarray:
    """按位展开算法（SDK 5.1.1）。

    Parameters
    ----------
    vec : 1‑D array of *signed* integers (two's complement) with magnitude < 2^{bit_width‑1}
    bit_width : total bits *including* sign bit. For Int4 supply 4, Int8 supply 8.

    Returns
    -------
    out : (bit_width‑1, len(vec)) int8 matrix of +1/0/‑1
    """
    vec = np.asarray(vec, dtype=np.int32)
    max_val = 2 ** (bit_width - 1)
    if not (np.abs(vec) < max_val).all():
        raise ValueError("values exceed given bit_width range")
    n = vec.size
    out = np.empty((bit_width - 1, n), dtype=np.int8)
    for b in range(bit_width - 1):
        bits = ((vec >> b) & 1) * 2 - 1  # 0→‑1, 1→+1
        out[b] = bits.astype(np.int8)
    return out


def full_expand(vec: np.ndarray) -> List[np.ndarray]:
    """全展开算法（SDK 5.1.2）—返回 +1/0/‑1 *序列* 列表。"""
    vec = np.asarray(vec, dtype=np.int32)
    max_abs = int(np.max(np.abs(vec)))
    # Determine expansion length from table
    thresholds = [3, 7, 15, 31, 63, 127]
    counts = [3, 7, 15, 31, 63, 127]
    for th, cnt in zip(thresholds, counts):
        if max_abs <= th:
            expand_len = cnt
            break
    else:
        raise ValueError("value too large for full‑expand table")

    seqs: List[np.ndarray] = []
    start = 0
    for _ in range(expand_len):
        layer = np.zeros_like(vec, dtype=np.int8)
        mask = np.abs(vec) > start
        layer[mask] = np.sign(vec[mask]).astype(np.int8)
        seqs.append(layer)
        start += 1
    return seqs


def quantize_to_int4(weights: np.ndarray) -> np.ndarray:
    """Symmetric ‑1..1 → int4 (‑8..7) quantiser."""
    w = np.clip(weights, -1.0, 1.0)
    return np.round(w * 7).astype(np.int8)


def quantize_to_uint8(weights: np.ndarray) -> np.ndarray:
    """Asymmetric quantiser → 0..15 (uint8)."""
    w_min, w_max = float(weights.min()), float(weights.max())
    if math.isclose(w_min, w_max):
        return np.zeros_like(weights, dtype=np.uint8)
    w_norm = (weights - w_min) / (w_max - w_min)
    return np.round(w_norm * 15).astype(np.uint8)


###############################################################################
# Test‑suite (python -m unittest he100_sim)
###############################################################################


class TestHE100Sim(unittest.TestCase):
    def setUp(self):
        self.sim = HE100Sim()

    # -------------------- weight IO --------------------

    def test_write_read_int4_roundtrip(self):
        blk = np.random.randint(-8, 8, (10, 12), dtype=np.int8)
        addr = [100, 20, 10, 12]
        self.sim.set_weight_int4(blk, addr)
        out = self.sim.get_weight_int4(addr)
        np.testing.assert_array_equal(blk, out)

    def test_write_read_uint8_roundtrip(self):
        blk = np.random.randint(0, 16, (5, 7), dtype=np.uint8)
        addr = [10, 10, 5, 7]
        self.sim.set_weight(blk, addr)
        out = self.sim.get_weight(addr)
        np.testing.assert_array_equal(blk, out)

    # -------------------- calculate --------------------

    def test_calculate_row_vector(self):
        w = np.random.randint(-8, 8, (1, 50), dtype=np.int8)
        x = np.random.randint(-1, 2, 50, dtype=np.int8)
        self.sim.set_weight_int4(w, [0, 0, 1, 50])
        y = self.sim.calculate(x, [0, 0, 1, 50], it_time=5, data_type=3)
        ref = int((w[0] @ x) * 5)
        self.assertEqual(y, ref)

    def test_calculate_col_vector(self):
        w = np.random.randint(-8, 8, (30, 1), dtype=np.int8)
        x = np.random.randint(-1, 2, 30, dtype=np.int8)
        self.sim.set_weight_int4(w, [0, 0, 30, 1])
        y = self.sim.calculate(x, [0, 0, 30, 1], it_time=2, data_type=3)
        ref = int((x @ w[:, 0]) * 2)
        self.assertEqual(y, ref)

    # -------------------- expansion helpers --------------------

    def test_bitwise_expand(self):
        vec = np.array([3, -2, 0], dtype=np.int32)  # Int4 range
        exp = bitwise_expand(vec, bit_width=4)
        self.assertEqual(exp.shape, (3, 3))  # 4‑1 bits
        # Check manual first column (value 3 = 011)
        self.assertListEqual(exp[:, 0].tolist(), [+1, +1, -1])

    def test_full_expand(self):
        vec = np.array([5, -3], dtype=np.int32)
        seqs = full_expand(vec)
        self.assertEqual(len(seqs), 7)  # max |value| is 5 → 7 steps per table
        self.assertEqual(seqs[0][0], +1)  # first layer for 5 is +1
        self.assertEqual(seqs[0][1], -1)  # first layer for -3 is -1


if __name__ == "__main__":
    unittest.main()

