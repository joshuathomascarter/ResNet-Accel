#!/usr/bin/env python3
"""
spi_driver.py — Minimal SPI Host Driver for ACCEL-v1 (Optional Phase 4b)
===========================================================================

Purpose:
  Provides simple SPI-based control of ACCEL-v1 for embedded hosts without AXI support.
  
Features:
  - SPI Mode 0 transactions (CSR read/write, DMA bursts)
  - Chunked data transfer
  - Optional CRC-32 for robustness
  - Example usage (sparse matrix loading)

Author: ACCEL-v1 Team
"""

import struct
import time
from typing import List, Tuple


class SPIDriver:
    """SPI-based driver for ACCEL-v1 control."""
    
    # SPI Command Codes
    CMD_CSR_WRITE = 0x00
    CMD_CSR_READ  = 0x10
    CMD_DMA_BURST = 0x20
    
    # CSR Addresses
    CSR_DMA_LAYER  = 0x50
    CSR_DMA_CTRL   = 0x51
    CSR_DMA_COUNT  = 0x52
    CSR_DMA_STATUS = 0x53
    CSR_DMA_BURST  = 0x54
    
    def __init__(self, spi_port: str = "/dev/spidev0.0", speed_hz: int = 1_000_000):
        """
        Initialize SPI driver.
        
        Args:
            spi_port: Linux SPI device path (e.g., /dev/spidev0.0)
            speed_hz: SPI clock frequency (default 1 MHz)
        """
        self.spi_port = spi_port
        self.speed_hz = speed_hz
        self.bytes_sent = 0
        self.spi_errors = 0
        
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = speed_hz
            self.spi.mode = 0  # Mode 0: CPOL=0, CPHA=0
        except ImportError:
            print("[WARNING] spidev not installed; SPI operations will be simulated")
            self.spi = None
    
    def spi_transaction(self, tx_data: bytes, rx_len: int = None) -> bytes:
        """
        Perform SPI transaction (full-duplex).
        
        Args:
            tx_data: Bytes to transmit
            rx_len: Number of bytes to receive (default = len(tx_data))
            
        Returns:
            Received bytes
        """
        if rx_len is None:
            rx_len = len(tx_data)
        
        if self.spi:
            # Real SPI hardware
            rx_data = self.spi.xfer2(list(tx_data))[:rx_len]
            self.bytes_sent += len(tx_data)
            return bytes(rx_data)
        else:
            # Simulated SPI (for testing)
            print(f"[SPI] TX: {tx_data.hex()}")
            self.bytes_sent += len(tx_data)
            return bytes(rx_len)  # Echo zeros
    
    def csr_write(self, addr: int, value: int) -> bool:
        """
        Write to CSR register.
        
        Args:
            addr: CSR address (0x50-0x54)
            value: 32-bit value
            
        Returns:
            True if successful
        """
        # Packet: [CMD(8)][ADDR(16)][DATA(32)]
        tx_data = struct.pack(">BHI", self.CMD_CSR_WRITE | (addr & 0x0F), 0, value)
        
        try:
            self.spi_transaction(tx_data)
            print(f"[SPI] CSR Write addr=0x{addr:02x} value=0x{value:08x}")
            return True
        except Exception as e:
            print(f"[ERROR] CSR write failed: {e}")
            self.spi_errors += 1
            return False
    
    def csr_read(self, addr: int) -> int:
        """
        Read from CSR register.
        
        Args:
            addr: CSR address (0x50-0x54)
            
        Returns:
            32-bit register value
        """
        # Packet: [CMD(8)][ADDR(16)][DUMMY(32)]
        tx_data = struct.pack(">BHI", self.CMD_CSR_READ | (addr & 0x0F), 0, 0)
        
        try:
            rx_data = self.spi_transaction(tx_data, rx_len=7)
            value = struct.unpack(">I", rx_data[3:7])[0]
            print(f"[SPI] CSR Read addr=0x{addr:02x} value=0x{value:08x}")
            return value
        except Exception as e:
            print(f"[ERROR] CSR read failed: {e}")
            self.spi_errors += 1
            return 0
    
    def set_layer_config(self, layer_id: int, num_rows: int, 
                        num_cols: int, total_blocks: int) -> bool:
        """
        Configure DMA for sparse layer.
        
        Args:
            layer_id: Layer index (0-3)
            num_rows: Block rows in layer
            num_cols: Block columns
            total_blocks: Total blocks
            
        Returns:
            True if successful
        """
        config = ((layer_id & 0xF) << 24) | \
                 ((num_rows & 0xFF) << 16) | \
                 ((num_cols & 0xFF) << 8) | \
                 (total_blocks & 0xFF)
        return self.csr_write(self.CSR_DMA_LAYER, config)
    
    def enable_crc(self, enable: bool = True) -> bool:
        """Enable/disable CRC verification."""
        return self.csr_write(self.CSR_DMA_CTRL, 1 if enable else 0)
    
    def dma_burst_write(self, words: List[int], chunk_size: int = 16) -> Tuple[bool, int]:
        """
        Write 32-bit words via SPI DMA bursts.
        
        Args:
            words: List of 32-bit words
            chunk_size: Words per SPI transaction (limited by bandwidth)
            
        Returns:
            (success, words_sent)
        """
        words_sent = 0
        
        for word in words:
            if words_sent % chunk_size == 0:
                time.sleep(0.001)  # Throttle
            
            # Packet: [CMD(8)][ADDR(16)][DATA(32)]
            tx_data = struct.pack(">BHI", self.CMD_DMA_BURST, 0, word)
            
            try:
                self.spi_transaction(tx_data)
                words_sent += 1
            except Exception as e:
                print(f"[ERROR] DMA burst failed at word {words_sent}: {e}")
                self.spi_errors += 1
                return (False, words_sent)
        
        print(f"[SPI] DMA Burst: {words_sent} words sent")
        return (True, words_sent)
    
    def poll_status(self, timeout_ms: int = 1000) -> bool:
        """
        Poll DMA status until DONE or timeout.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if DMA completes
        """
        start_time = time.time()
        
        while (time.time() - start_time) * 1000 < timeout_ms:
            status = self.csr_read(self.CSR_DMA_STATUS)
            if status & 0x1:  # DONE bit
                print(f"[SPI] Transfer complete (status=0x{status:08x})")
                return True
            time.sleep(0.01)
        
        print(f"[SPI] Timeout waiting for transfer")
        return False
    
    def get_word_count(self) -> int:
        """Get DMA word count."""
        return self.csr_read(self.CSR_DMA_COUNT)
    
    def close(self):
        """Close SPI device."""
        if self.spi:
            self.spi.close()


def example_usage():
    """Example: Load sparse MNIST layer via SPI."""
    
    # Initialize SPI driver (1 MHz clock)
    driver = SPIDriver(speed_hz=1_000_000)
    
    try:
        # Enable CRC
        driver.enable_crc(True)
        
        # Configure layer
        driver.set_layer_config(
            layer_id=0,
            num_rows=32,
            num_cols=32,
            total_blocks=512
        )
        
        # Load sparse weights (simulated)
        weights = [0x12345678 + i for i in range(256)]
        
        # Send via DMA bursts (16 words per transaction)
        success, sent = driver.dma_burst_write(weights, chunk_size=16)
        
        if success:
            # Poll for completion
            if driver.poll_status(timeout_ms=5000):
                print(f"[SUCCESS] Transfer complete")
                count = driver.get_word_count()
                print(f"  Words written: {count}")
            else:
                print(f"[ERROR] Transfer timeout")
        else:
            print(f"[ERROR] Burst transfer failed")
    
    finally:
        driver.close()
        print(f"\n[STATS] Total bytes sent: {driver.bytes_sent}")
        print(f"[STATS] SPI errors: {driver.spi_errors}")


if __name__ == '__main__':
    example_usage()
