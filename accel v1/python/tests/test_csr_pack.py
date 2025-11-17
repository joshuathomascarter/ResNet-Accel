# test_csr_pack.py - Byte-exact parity test for csr_map.py
# Verifies field packing/unpacking matches RTL implementation
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_uart.csr_map import *


def test_CTRL():
    v = pack_CTRL(1, 0, 1)
    assert unpack_CTRL(v) == (1, 0, 1)


def test_DIMS():
    v = pack_DIMS(7, 11, 22)
    assert unpack_DIMS(v) == (7, 11, 22)


def test_TILES():
    v = pack_TILES(1, 2, 3)
    assert unpack_TILES(v) == (1, 2, 3)


def test_INDEX():
    v = pack_INDEX(5, 9)
    assert unpack_INDEX(v) == (5, 9)


def test_BUFF():
    v = pack_BUFF(1, 0, 1, 1)
    assert unpack_BUFF(v) == (1, 0, 1, 1)


def test_SCALE():
    v = pack_SCALE(55, 77)
    assert unpack_SCALE(v) == (55, 77)


def test_UART():
    v = pack_UART(1234, 1)
    assert unpack_UART(v) == (1234, 1)


def test_STATUS():
    v = pack_STATUS(0, 1, 1)
    assert unpack_STATUS(v) == (0, 1, 1)


def test_Config_to_bytes_from_bytes():
    from host_uart.csr_map import (
        Config,
        DIMS_M,
        DIMS_N,
        DIMS_K,
        TILES_Tm,
        TILES_Tn,
        TILES_Tk,
        INDEX_m,
        INDEX_n,
        INDEX_k,
        BUFF,
        SCALE_Sa,
        SCALE_Sw,
        pack_u32,
        pack_f32,
    )

    cfg = Config(M=7, N=11, K=22, Tm=1, Tn=2, Tk=3, m_idx=5, n_idx=9, k_idx=13, Sa=0.5, Sw=2.0, wrA=1, wrB=0)
    reg_img = cfg.to_bytes()
    # Check a few fields in the register image
    assert reg_img[DIMS_M : DIMS_M + 4] == pack_u32(7)
    assert reg_img[DIMS_N : DIMS_N + 4] == pack_u32(11)
    assert reg_img[DIMS_K : DIMS_K + 4] == pack_u32(22)
    assert reg_img[TILES_Tm : TILES_Tm + 4] == pack_u32(1)
    assert reg_img[TILES_Tn : TILES_Tn + 4] == pack_u32(2)
    assert reg_img[TILES_Tk : TILES_Tk + 4] == pack_u32(3)
    assert reg_img[INDEX_m : INDEX_m + 4] == pack_u32(5)
    assert reg_img[INDEX_n : INDEX_n + 4] == pack_u32(9)
    assert reg_img[INDEX_k : INDEX_k + 4] == pack_u32(13)
    assert reg_img[SCALE_Sa : SCALE_Sa + 4] == pack_f32(0.5)
    assert reg_img[SCALE_Sw : SCALE_Sw + 4] == pack_f32(2.0)
    # Round-trip
    cfg2 = Config.from_bytes(reg_img)
    assert cfg2.M == 7 and cfg2.N == 11 and cfg2.K == 22
    assert cfg2.Tm == 1 and cfg2.Tn == 2 and cfg2.Tk == 3
    assert cfg2.m_idx == 5 and cfg2.n_idx == 9 and cfg2.k_idx == 13
    assert abs(cfg2.Sa - 0.5) < 1e-6 and abs(cfg2.Sw - 2.0) < 1e-6
    assert cfg2.wrA == 1 and cfg2.wrB == 0
