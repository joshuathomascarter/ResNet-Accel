"""
csr_map.py - ACCEL-v1 CSR register map

Little-endian, 32-bit aligned register map with explicit offsets,
bitfields, and helpers for packing/unpacking and config serialization.
"""

from dataclasses import dataclass, fields
import struct
from typing import Any, Dict, List, Tuple

LE = "<"  # little-endian

# Register byte offsets (32-bit aligned)
CTRL = 0x00  #: Control register
DIMS_M = 0x04  #: Matrix M dimension
DIMS_N = 0x08  #: Matrix N dimension
DIMS_K = 0x0C  #: Matrix K dimension
TILES_Tm = 0x10  #: Tile size Tm
TILES_Tn = 0x14  #: Tile size Tn
TILES_Tk = 0x18  #: Tile size Tk
INDEX_m = 0x1C  #: m index
INDEX_n = 0x20  #: n index
INDEX_k = 0x24  #: k index
BUFF = 0x28  #: Buffer control
SCALE_Sa = 0x2C  #: Activation scale (float32)
SCALE_Sw = 0x30  #: Weight scale (float32)
# UART registers removed (0x34, 0x38)
STATUS = 0x3C  #: Status register
# DMA CSR registers (BSR DMA)
DMA_LAYER = 0x50  #: DMA: layer selection (0-7)
DMA_CTRL = 0x51  #: DMA: control (start/reset)
DMA_COUNT = 0x52  #: DMA: blocks written
DMA_STATUS = 0x53  #: DMA: status

# Performance monitor registers (Read-Only)
PERF_TOTAL = 0x40  #: Total cycles from start to done
PERF_ACTIVE = 0x44  #: Cycles where busy was high
PERF_IDLE = 0x48  #: Cycles where busy was low

# ---------------------------------------------------------------------------
# BSR / HYBRID SCHEDULER REGISTERS (0xC0 - 0xDF)
# ---------------------------------------------------------------------------
# These registers control the BSR sparse format and scheduler mode selection.
# The accelerator has two schedulers sharing the same 14×14 systolic array:
#   - BSR Scheduler (bsr_scheduler.sv): For sparse layers with BSR weights
#   - Dense Scheduler (scheduler.sv): For fully-connected layers (100% dense)
#
# BSR_CONFIG[0] selects which scheduler drives the compute:
#   0 = BSR sparse scheduler
#   1 = Dense GEMM scheduler

BSR_CONFIG = 0xC0       #: BSR configuration/scheduler mode select
BSR_NUM_BLOCKS = 0xC4   #: Number of non-zero BSR blocks
BSR_BLOCK_ROWS = 0xC8   #: Block grid rows
BSR_BLOCK_COLS = 0xCC   #: Block grid columns
BSR_STATUS = 0xD0       #: BSR engine status
BSR_ERROR_CODE = 0xD4   #: BSR error code (RO)
BSR_PTR_ADDR = 0xD8     #: row_ptr array DRAM address
BSR_IDX_ADDR = 0xDC     #: col_idx array DRAM address

# BSR_CONFIG bits
BSR_CONFIG_SCHED_MODE = 1 << 0   #: Scheduler mode: 0=BSR, 1=Dense
BSR_CONFIG_MODE_BSR = 0         #: Use BSR sparse scheduler
BSR_CONFIG_MODE_DENSE = 1 << 0  #: Use Dense GEMM scheduler
BSR_CONFIG_VERIFY = 1 << 1      #: Enable verification (legacy)
BSR_CONFIG_ZERO_SKIP = 1 << 2   #: Enable zero-skip (legacy)

# BSR_STATUS bits
BSR_STATUS_READY = 1 << 0   #: BSR engine ready
BSR_STATUS_BUSY = 1 << 1    #: BSR engine busy
BSR_STATUS_DONE = 1 << 2    #: BSR operation complete
BSR_STATUS_ERROR = 1 << 3   #: BSR error flag

# CTRL bits
CTRL_START = 1 << 0  #: Start pulse (W1P)
CTRL_ABORT = 1 << 1  #: Abort pulse (W1P)
CTRL_IRQEN = 1 << 2  #: Interrupt enable (RW)

# STATUS bits
STS_BUSY = 1 << 0  #: Accelerator busy (RO)
STS_DONE_TILE = 1 << 1  #: Tile done (R/W1C)
# UART error bits removed

# BUFF bits
WR_A = 1 << 0  #: Write bank A
WR_B = 1 << 1  #: Write bank B
RD_A = 1 << 8  #: Read bank A
RD_B = 1 << 9  #: Read bank B

# Field/type metadata for auto-docs and validation
CSR_LAYOUT = [
    (CTRL, "CTRL", "u32", "Control register"),
    (DIMS_M, "DIMS_M", "u32", "Matrix M dimension"),
    (DIMS_N, "DIMS_N", "u32", "Matrix N dimension"),
    (DIMS_K, "DIMS_K", "u32", "Matrix K dimension"),
    (TILES_Tm, "TILES_Tm", "u32", "Tile size Tm"),
    (TILES_Tn, "TILES_Tn", "u32", "Tile size Tn"),
    (TILES_Tk, "TILES_Tk", "u32", "Tile size Tk"),
    (INDEX_m, "INDEX_m", "u32", "m index"),
    (INDEX_n, "INDEX_n", "u32", "n index"),
    (INDEX_k, "INDEX_k", "u32", "k index"),
    (BUFF, "BUFF", "u32", "Buffer control bits"),
    (SCALE_Sa, "SCALE_Sa", "f32", "Activation scale"),
    (SCALE_Sw, "SCALE_Sw", "f32", "Weight scale"),
    (STATUS, "STATUS", "u32", "Status register"),
    (PERF_TOTAL, "PERF_TOTAL", "u32", "Total cycles from start to done (RO)"),
    (PERF_ACTIVE, "PERF_ACTIVE", "u32", "Cycles where busy was high (RO)"),
    (PERF_IDLE, "PERF_IDLE", "u32", "Cycles where busy was low (RO)"),
    (DMA_LAYER, "DMA_LAYER", "u32", "BSR DMA: Active Layer"),
    (DMA_CTRL, "DMA_CTRL", "u32", "BSR DMA: Control (start, reset)"),
    (DMA_COUNT, "DMA_COUNT", "u32", "BSR DMA: Blocks loaded"),
    (DMA_STATUS, "DMA_STATUS", "u32", "BSR DMA status"),
    (BSR_CONFIG, "BSR_CONFIG", "u32", "BSR config / scheduler mode select"),
    (BSR_NUM_BLOCKS, "BSR_NUM_BLOCKS", "u32", "Number of non-zero BSR blocks"),
    (BSR_BLOCK_ROWS, "BSR_BLOCK_ROWS", "u32", "Block grid rows"),
    (BSR_BLOCK_COLS, "BSR_BLOCK_COLS", "u32", "Block grid columns"),
    (BSR_STATUS, "BSR_STATUS", "u32", "BSR engine status"),
]

FIELD_TYPES = {
    "u32": (4, lambda x: struct.pack(LE + "I", x), lambda b: struct.unpack(LE + "I", b)[0]),
    "f32": (4, lambda x: struct.pack(LE + "f", x), lambda b: struct.unpack(LE + "f", b)[0]),
}


def pack_u32(x: int) -> bytes:
    """Pack unsigned 32-bit integer (little-endian)"""
    return struct.pack(LE + "I", x)


def pack_f32(x: float) -> bytes:
    """Pack 32-bit float (little-endian)"""
    return struct.pack(LE + "f", x)


def unpack_u32(b: bytes) -> int:
    """Unpack unsigned 32-bit integer (little-endian)"""
    return struct.unpack(LE + "I", b)[0]


def unpack_f32(b: bytes) -> float:
    """Unpack 32-bit float (little-endian)"""
    return struct.unpack(LE + "f", b)[0]


# Test helper functions (for backwards compatibility with test_csr_pack.py)
def pack_CTRL(start: int, abort: int, irq_en: int) -> bytes:
    """Pack CTRL register fields"""
    word = (start & 1) | ((abort & 1) << 1) | ((irq_en & 1) << 2)
    return pack_u32(word)


def unpack_CTRL(b: bytes) -> tuple:
    """Unpack CTRL register fields"""
    word = unpack_u32(b)
    return (word & 1, (word >> 1) & 1, (word >> 2) & 1)


def pack_DIMS(m: int, n: int, k: int) -> bytes:
    """Pack DIMS registers (M, N, K as separate 32-bit words)"""
    return pack_u32(m) + pack_u32(n) + pack_u32(k)


def unpack_DIMS(b: bytes) -> tuple:
    """Unpack DIMS registers"""
    return (unpack_u32(b[0:4]), unpack_u32(b[4:8]), unpack_u32(b[8:12]))


def pack_TILES(tm: int, tn: int, tk: int) -> bytes:
    """Pack TILES registers (Tm, Tn, Tk as separate 32-bit words)"""
    return pack_u32(tm) + pack_u32(tn) + pack_u32(tk)


def unpack_TILES(b: bytes) -> tuple:
    """Unpack TILES registers"""
    return (unpack_u32(b[0:4]), unpack_u32(b[4:8]), unpack_u32(b[8:12]))


def pack_INDEX(m_idx: int, n_idx: int) -> bytes:
    """Pack INDEX registers (m_idx, n_idx as separate 32-bit words)"""
    return pack_u32(m_idx) + pack_u32(n_idx)


def unpack_INDEX(b: bytes) -> tuple:
    """Unpack INDEX registers"""
    return (unpack_u32(b[0:4]), unpack_u32(b[4:8]))


def pack_BUFF(wr_a: int, wr_b: int, rd_a: int, rd_b: int) -> bytes:
    """Pack BUFF register fields"""
    word = ((wr_a & 1) << 0) | ((wr_b & 1) << 1) | ((rd_a & 1) << 8) | ((rd_b & 1) << 9)
    return pack_u32(word)


def unpack_BUFF(b: bytes) -> tuple:
    """Unpack BUFF register fields"""
    word = unpack_u32(b)
    return ((word >> 0) & 1, (word >> 1) & 1, (word >> 8) & 1, (word >> 9) & 1)


def pack_SCALE(sa: int, sw: int) -> bytes:
    """Pack SCALE registers (Sa, Sw as separate floats, but this test uses ints)"""
    return pack_f32(float(sa)) + pack_f32(float(sw))


def unpack_SCALE(b: bytes) -> tuple:
    """Unpack SCALE registers (returns as ints for test compatibility)"""
    return (int(unpack_f32(b[0:4])), int(unpack_f32(b[4:8])))


def pack_UART(len_max: int, crc_en: int) -> bytes:
    """Pack UART configuration registers"""
    return pack_u32(len_max) + pack_u32(crc_en)


def unpack_UART(b: bytes) -> tuple:
    """Unpack UART configuration registers"""
    return (unpack_u32(b[0:4]), unpack_u32(b[4:8]))


def pack_STATUS(busy: int, done: int, error: int) -> bytes:
    """Pack STATUS register fields"""
    word = (busy & 1) | ((done & 1) << 1) | ((error & 1) << 8)
    return pack_u32(word)


def unpack_STATUS(b: bytes) -> tuple:
    """Unpack STATUS register fields"""
    word = unpack_u32(b)
    return (word & 1, (word >> 1) & 1, (word >> 8) & 1)


@dataclass
class Config:
    """Accelerator configuration (matches CSR layout)"""

    M: int
    N: int
    K: int
    Tm: int
    Tn: int
    Tk: int
    m_idx: int = 0
    n_idx: int = 0
    k_idx: int = 0
    Sa: float = 1.0
    Sw: float = 1.0
    wrA: int = 0
    wrB: int = 0

    def to_bytes(self) -> bytes:
        """Serialize config to register image (for dumps or programming)"""
        reg_img = bytearray(STATUS + 4)
        for addr, name, typ, _ in CSR_LAYOUT:
            val = 0  # Default value

            # Map field names to Config attributes
            if name == "DIMS_M":
                val = self.M
            elif name == "DIMS_N":
                val = self.N
            elif name == "DIMS_K":
                val = self.K
            elif name == "TILES_Tm":
                val = self.Tm
            elif name == "TILES_Tn":
                val = self.Tn
            elif name == "TILES_Tk":
                val = self.Tk
            elif name == "INDEX_m":
                val = self.m_idx
            elif name == "INDEX_n":
                val = self.n_idx
            elif name == "INDEX_k":
                val = self.k_idx
            elif name == "SCALE_Sa":
                val = self.Sa
            elif name == "SCALE_Sw":
                val = self.Sw
            elif name == "BUFF":
                val = (self.wrA & 1) * WR_A | (self.wrB & 1) * WR_B
            # CTRL and STATUS are not set here (runtime registers)

            sz, pack, _ = FIELD_TYPES[typ]
            reg_img[addr : addr + sz] = pack(val)
        return bytes(reg_img)

    @classmethod
    def from_bytes(cls, b: bytes) -> "Config":
        """Deserialize config from register image"""
        kwargs = {}
        for addr, name, typ, _ in CSR_LAYOUT:
            sz, _, unpack = FIELD_TYPES[typ]
            if addr + sz > len(b):
                continue
            val = unpack(b[addr : addr + sz])

            # Map register names to Config attributes
            if name == "DIMS_M":
                kwargs["M"] = val
            elif name == "DIMS_N":
                kwargs["N"] = val
            elif name == "DIMS_K":
                kwargs["K"] = val
            elif name == "TILES_Tm":
                kwargs["Tm"] = val
            elif name == "TILES_Tn":
                kwargs["Tn"] = val
            elif name == "TILES_Tk":
                kwargs["Tk"] = val
            elif name == "INDEX_m":
                kwargs["m_idx"] = val
            elif name == "INDEX_n":
                kwargs["n_idx"] = val
            elif name == "INDEX_k":
                kwargs["k_idx"] = val
            elif name == "SCALE_Sa":
                kwargs["Sa"] = val
            elif name == "SCALE_Sw":
                kwargs["Sw"] = val
            elif name == "BUFF":
                kwargs["wrA"] = 1 if (val & WR_A) else 0
                kwargs["wrB"] = 1 if (val & WR_B) else 0

        # Set defaults for any missing fields
        for field in ["M", "N", "K", "Tm", "Tn", "Tk"]:
            if field not in kwargs:
                kwargs[field] = 0

        return cls(**kwargs)


def to_writes(cfg: Config) -> List[Tuple[int, bytes]]:
    """List of (addr, payload_bytes) tuples to program CSRs."""
    return [
        (DIMS_M, pack_u32(cfg.M)),
        (DIMS_N, pack_u32(cfg.N)),
        (DIMS_K, pack_u32(cfg.K)),
        (TILES_Tm, pack_u32(cfg.Tm)),
        (TILES_Tn, pack_u32(cfg.Tn)),
        (TILES_Tk, pack_u32(cfg.Tk)),
        (INDEX_m, pack_u32(cfg.m_idx)),
        (INDEX_n, pack_u32(cfg.n_idx)),
        (INDEX_k, pack_u32(cfg.k_idx)),
        (BUFF, pack_u32((cfg.wrA & 1) * WR_A | (cfg.wrB & 1) * WR_B)),
        (SCALE_Sa, pack_f32(cfg.Sa)),
        (SCALE_Sw, pack_f32(cfg.Sw)),
    ]


def make_ctrl_start(irq_en: bool) -> bytes:
    """Pack CTRL register for start pulse (optionally enable IRQ)"""
    word = (CTRL_IRQEN if irq_en else 0) | CTRL_START
    return pack_u32(word)


def make_ctrl_abort() -> bytes:
    """Pack CTRL register for abort pulse"""
    return pack_u32(CTRL_ABORT)

