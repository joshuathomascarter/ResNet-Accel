"""
uart_driver.py - Host-side UART driver with packet framing

Frame format: SYNC0(0xA5) SYNC1(0x5A) | LEN | CMD | PAYLOAD[LEN] | CRC8
- CRC covers: LEN, CMD, PAYLOAD
- Provides: make_packet(), parse_stream_incremental(), send_packet(), recv_packet()
- Includes: LoopbackSerial for testing

For real hardware, replace LoopbackSerial with pyserial.Serial:
    ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0)
    driver = UARTDriver(ser)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

SYNC0 = 0xA5
SYNC1 = 0x5A

# Commands (Accel v1)
LOAD_A_TILE = 0x01
LOAD_B_TILE = 0x02
START_TILE = 0x03
READ_C_TILE = 0x04


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def make_packet(cmd: int, payload: bytes) -> bytes:
    if not (0 <= cmd <= 255):
        raise ValueError("cmd must be 0..255")
    if len(payload) > 255:
        raise ValueError("payload too long (max 255)")
    hdr = bytes([SYNC0, SYNC1, len(payload) & 0xFF, cmd & 0xFF])
    c = crc8(hdr[2:] + payload)  # CRC over LEN|CMD|PAYLOAD
    return hdr + payload + bytes([c])


@dataclass
class Packet:
    cmd: int
    payload: bytes


class StreamParser:
    """Incremental parser for framed packets."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> None:
        self.buf += data

    def try_parse(self) -> Tuple[Optional[Packet], int]:
        """
        Try to parse at most one packet.
        Returns (Packet or None, bytes_consumed).
        If CRC fails, returns (None, n) where n>0 means bytes were dropped to resync.
        """
        i = 0
        b = self.buf
        while i + 4 <= len(b):
            if b[i] != SYNC0 or b[i + 1] != SYNC1:
                i += 1
                continue
            ln = b[i + 2]
            total = 4 + ln + 1
            if i + total > len(b):
                break  # incomplete
            chunk = b[i : i + total]
            crc_calc = crc8(chunk[2:-1])
            if crc_calc != chunk[-1]:
                # CRC fail: drop the leading SYNC0 and rescan from next byte
                return (None, i + 1)
            pkt = Packet(cmd=chunk[3], payload=bytes(chunk[4:-1]))
            return (pkt, i + total)
        return (None, i)


class LoopbackSerial:
    """Minimal in-process serial-like transport for tests."""

    def __init__(self):
        self.rx = bytearray()

    def write(self, data: bytes) -> int:
        self.rx += data  # loopback
        return len(data)

    def read(self, n: int) -> bytes:
        out = bytes(self.rx[:n])
        self.rx = self.rx[n:]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self.rx)


class UARTDriver:
    """Convenience wrapper to send/receive framed packets over a serial-like stream."""

    def __init__(self, ser_like):
        self.ser = ser_like  # must have write(), read(n), in_waiting
        self.parser = StreamParser()

    def send_packet(self, cmd: int, payload: bytes) -> None:
        pkt = make_packet(cmd, payload)
        self.ser.write(pkt)

    def poll(self, max_read: int = 1024) -> Optional[Packet]:
        if getattr(self.ser, "in_waiting", 0):
            chunk = self.ser.read(min(self.ser.in_waiting, max_read))
            if chunk:
                self.parser.feed(chunk)
        pkt, consumed = self.parser.try_parse()
        if consumed:
            # drop consumed bytes
            self.parser.buf = self.parser.buf[consumed:]
        return pkt

    def recv_packet(self, timeout_s: float = 0.5) -> Optional[Packet]:
        """Poll until a packet arrives or timeout (busy-wait; adapt for your runtime)."""
        import time

        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            pkt = self.poll()
            if pkt is not None:
                return pkt
            time.sleep(0.001)
        return None


# --- Self-test
def self_test_loopback():
    ser = LoopbackSerial()
    drv = UARTDriver(ser)
    payload = b"hello"
    drv.send_packet(LOAD_A_TILE, payload)
    got = drv.recv_packet(timeout_s=0.1)
    assert got is not None, "no packet received"
    assert got.cmd == LOAD_A_TILE and got.payload == payload
    print("Loopback self-test PASS")


if __name__ == "__main__":
    self_test_loopback()
