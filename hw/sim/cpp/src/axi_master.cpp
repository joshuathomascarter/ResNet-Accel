/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                          AXI_MASTER.CPP                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  IMPLEMENTS: axi_master.hpp                                               ║
 * ║  REPLACES: sw/host/axi_driver.py, sw/host_axi/axi_master.py              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  DevMemBackend - For real FPGA hardware                                   ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  Constructor(phys_addr, size):                                            ║
 * ║    1. fd_ = open("/dev/mem", O_RDWR | O_SYNC)                             ║
 * ║       - O_SYNC ensures writes go directly to hardware                     ║
 * ║       - Requires root privileges (sudo)                                   ║
 * ║    2. mapped_base_ = mmap(NULL, size, PROT_READ|PROT_WRITE,               ║
 * ║                           MAP_SHARED, fd_, phys_addr)                     ║
 * ║       - Maps physical memory to virtual address space                     ║
 * ║    3. Check for MAP_FAILED and throw on error                             ║
 * ║                                                                           ║
 * ║  Destructor:                                                              ║
 * ║    1. munmap(mapped_base_, size_)                                         ║
 * ║    2. close(fd_)                                                          ║
 * ║                                                                           ║
 * ║  write32(addr, data):                                                     ║
 * ║    1. Calculate offset: offset = (addr - phys_base_) / 4                  ║
 * ║    2. Write: mapped_base_[offset] = data                                  ║
 * ║    - Use volatile to prevent compiler optimization                        ║
 * ║                                                                           ║
 * ║  read32(addr):                                                            ║
 * ║    1. Calculate offset: offset = (addr - phys_base_) / 4                  ║
 * ║    2. Return: mapped_base_[offset]                                        ║
 * ║                                                                           ║
 * ║  barrier():                                                               ║
 * ║    ARM-specific memory barriers to ensure coherency:                      ║
 * ║    __sync_synchronize();           // GCC built-in                        ║
 * ║    asm volatile("dsb sy":::"memory");  // Data Synchronization Barrier   ║
 * ║    asm volatile("dmb sy":::"memory");  // Data Memory Barrier             ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  VerilatorBackend - For simulation (TEMPLATE - in header)                 ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  tick():                                                                  ║
 * ║    model_->clk = 0; model_->eval();                                       ║
 * ║    model_->clk = 1; model_->eval();                                       ║
 * ║    cycle_count_++;                                                        ║
 * ║                                                                           ║
 * ║  write32(addr, data) - AXI4-Lite Write:                                   ║
 * ║    // Address phase                                                       ║
 * ║    model_->s_axi_awvalid = 1;                                             ║
 * ║    model_->s_axi_awaddr = addr;                                           ║
 * ║    while (!model_->s_axi_awready) tick();                                 ║
 * ║    tick();                                                                ║
 * ║    model_->s_axi_awvalid = 0;                                             ║
 * ║                                                                           ║
 * ║    // Data phase                                                          ║
 * ║    model_->s_axi_wvalid = 1;                                              ║
 * ║    model_->s_axi_wdata = data;                                            ║
 * ║    model_->s_axi_wstrb = 0xF;  // All 4 bytes valid                       ║
 * ║    while (!model_->s_axi_wready) tick();                                  ║
 * ║    tick();                                                                ║
 * ║    model_->s_axi_wvalid = 0;                                              ║
 * ║                                                                           ║
 * ║    // Response phase                                                      ║
 * ║    model_->s_axi_bready = 1;                                              ║
 * ║    while (!model_->s_axi_bvalid) tick();                                  ║
 * ║    tick();                                                                ║
 * ║    model_->s_axi_bready = 0;                                              ║
 * ║                                                                           ║
 * ║  read32(addr) - AXI4-Lite Read:                                           ║
 * ║    // Address phase                                                       ║
 * ║    model_->s_axi_arvalid = 1;                                             ║
 * ║    model_->s_axi_araddr = addr;                                           ║
 * ║    while (!model_->s_axi_arready) tick();                                 ║
 * ║    tick();                                                                ║
 * ║    model_->s_axi_arvalid = 0;                                             ║
 * ║                                                                           ║
 * ║    // Data phase                                                          ║
 * ║    model_->s_axi_rready = 1;                                              ║
 * ║    while (!model_->s_axi_rvalid) tick();                                  ║
 * ║    uint32_t data = model_->s_axi_rdata;                                   ║
 * ║    tick();                                                                ║
 * ║    model_->s_axi_rready = 0;                                              ║
 * ║    return data;                                                           ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  SoftwareModelBackend - For unit testing                                  ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  Just use a std::vector<uint32_t> as memory.                              ║
 * ║  Simulate status register behavior:                                       ║
 * ║    - When CTRL.START is written, set STATUS.BUSY                          ║
 * ║    - On next STATUS read, clear BUSY, set DONE                            ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  AXIMaster - High-level wrapper                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  Simple delegation to backend:                                            ║
 * ║    write_reg(offset, value) -> backend_->write32(base_ + offset, value)   ║
 * ║    read_reg(offset) -> backend_->read32(base_ + offset)                   ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "axi_master.hpp"

#include <stdexcept>
#include <cstring>

// Only compile DevMemBackend on Linux
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// -----------------------------------------------------------------------------
// DevMemBackend Implementation
// -----------------------------------------------------------------------------

DevMemBackend::DevMemBackend(uint64_t phys_addr, size_t size)
    : fd_(-1), mapped_base_(nullptr), phys_base_(phys_addr), map_size_(size) {
    
    // TODO: Implement
    //
    // fd_ = open("/dev/mem", O_RDWR | O_SYNC);
    // if (fd_ < 0) {
    //     throw std::runtime_error("Failed to open /dev/mem. Run with sudo.");
    // }
    //
    // void* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE,
    //                     MAP_SHARED, fd_, phys_addr);
    // if (mapped == MAP_FAILED) {
    //     close(fd_);
    //     throw std::runtime_error("mmap failed for accelerator registers");
    // }
    //
    // mapped_base_ = static_cast<volatile uint32_t*>(mapped);
}

DevMemBackend::~DevMemBackend() {
    // TODO: Implement
    //
    // if (mapped_base_) {
    //     munmap(const_cast<uint32_t*>(mapped_base_), map_size_);
    // }
    // if (fd_ >= 0) {
    //     close(fd_);
    // }
}

void DevMemBackend::write32(uint64_t addr, uint32_t data) {
    // TODO: Implement
    //
    // size_t offset = (addr - phys_base_) / 4;
    // mapped_base_[offset] = data;
}

uint32_t DevMemBackend::read32(uint64_t addr) {
    // TODO: Implement
    //
    // size_t offset = (addr - phys_base_) / 4;
    // return mapped_base_[offset];
    return 0;
}

void DevMemBackend::write_burst(uint64_t addr, const uint32_t* data, size_t count) {
    // TODO: Implement
    //
    // size_t offset = (addr - phys_base_) / 4;
    // for (size_t i = 0; i < count; i++) {
    //     mapped_base_[offset + i] = data[i];
    // }
}

void DevMemBackend::read_burst(uint64_t addr, uint32_t* data, size_t count) {
    // TODO: Implement
    //
    // size_t offset = (addr - phys_base_) / 4;
    // for (size_t i = 0; i < count; i++) {
    //     data[i] = mapped_base_[offset + i];
    // }
}

void DevMemBackend::flush() {
    // TODO: Implement
    // __sync_synchronize();
}

void DevMemBackend::barrier() {
    // TODO: Implement - ARM memory barriers
    //
    // __sync_synchronize();
    // #ifdef __aarch64__
    // asm volatile("dsb sy" ::: "memory");
    // asm volatile("dmb sy" ::: "memory");
    // #elif defined(__arm__)
    // asm volatile("dsb" ::: "memory");
    // asm volatile("dmb" ::: "memory");
    // #endif
}

#endif // __linux__

// -----------------------------------------------------------------------------
// SoftwareModelBackend Implementation
// -----------------------------------------------------------------------------

SoftwareModelBackend::SoftwareModelBackend(uint64_t base_addr, size_t size)
    : base_addr_(base_addr) {
    memory_.resize(size / 4, 0);
}

void SoftwareModelBackend::write32(uint64_t addr, uint32_t data) {
    // TODO: Implement with simulated register behavior
    //
    // size_t idx = (addr - base_addr_) / 4;
    // if (idx < memory_.size()) {
    //     memory_[idx] = data;
    //     
    //     // Simulate CTRL register behavior
    //     if (addr == base_addr_ + 0x00) {  // CTRL register
    //         if (data & 0x01) {  // START bit
    //             memory_[1] |= 0x01;  // Set STATUS.BUSY
    //         }
    //         if (data & 0x02) {  // RESET bit
    //             memory_[1] = 0;  // Clear STATUS
    //         }
    //     }
    // }
}

uint32_t SoftwareModelBackend::read32(uint64_t addr) {
    // TODO: Implement with simulated status behavior
    //
    // size_t idx = (addr - base_addr_) / 4;
    //
    // // Simulate STATUS register - auto-complete
    // if (addr == base_addr_ + 0x04) {  // STATUS register
    //     if (memory_[1] & 0x01) {  // If BUSY
    //         memory_[1] = 0x02;  // Clear BUSY, set DONE
    //     }
    // }
    //
    // return (idx < memory_.size()) ? memory_[idx] : 0;
    return 0;
}

void SoftwareModelBackend::write_burst(uint64_t addr, const uint32_t* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        write32(addr + i * 4, data[i]);
    }
}

void SoftwareModelBackend::read_burst(uint64_t addr, uint32_t* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = read32(addr + i * 4);
    }
}

void SoftwareModelBackend::flush() {
    // No-op for software model
}

void SoftwareModelBackend::barrier() {
    // No-op for software model
}

// -----------------------------------------------------------------------------
// AXIMaster Implementation
// -----------------------------------------------------------------------------

AXIMaster::AXIMaster(std::unique_ptr<AXIBackend> backend, uint64_t base_addr)
    : backend_(std::move(backend)), base_addr_(base_addr) {
}

void AXIMaster::write_reg(uint32_t offset, uint32_t value) {
    backend_->write32(base_addr_ + offset, value);
}

uint32_t AXIMaster::read_reg(uint32_t offset) {
    return backend_->read32(base_addr_ + offset);
}

void AXIMaster::set_bits(uint32_t offset, uint32_t mask) {
    uint32_t val = read_reg(offset);
    write_reg(offset, val | mask);
}

void AXIMaster::clear_bits(uint32_t offset, uint32_t mask) {
    uint32_t val = read_reg(offset);
    write_reg(offset, val & ~mask);
}

bool AXIMaster::test_bits(uint32_t offset, uint32_t mask) {
    return (read_reg(offset) & mask) == mask;
}

void AXIMaster::write_memory(uint64_t addr, const void* data, size_t bytes) {
    const uint32_t* words = static_cast<const uint32_t*>(data);
    size_t word_count = (bytes + 3) / 4;
    backend_->write_burst(addr, words, word_count);
}

void AXIMaster::read_memory(uint64_t addr, void* data, size_t bytes) {
    uint32_t* words = static_cast<uint32_t*>(data);
    size_t word_count = (bytes + 3) / 4;
    backend_->read_burst(addr, words, word_count);
}

AXIBackend* AXIMaster::get_backend() {
    return backend_.get();
}
