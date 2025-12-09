/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                          AXI_MASTER.HPP                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/host/axi_driver.py, sw/host_axi/axi_master.py              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Low-level AXI4 and AXI4-Lite bus master implementation.               ║
 * ║    Provides cycle-accurate register access for both Verilator            ║
 * ║    simulation and real FPGA hardware.                                    ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                               ║
 * ║    • Verilator is C++ - no FFI overhead for signal toggling              ║
 * ║    • volatile pointers for correct hardware register semantics           ║
 * ║    • Burst transfers need precise timing impossible in Python            ║
 * ║    • Memory barriers (dsb, dmb) require inline assembly                  ║
 * ║    • mmap() syscall overhead is amortized in C++ but not Python          ║
 * ║                                                                           ║
 * ║  WHAT THIS FILE DOES:                                                     ║
 * ║    1. Abstract the AXI protocol into simple read/write calls             ║
 * ║    2. Provide backend implementations for:                               ║
 * ║       - VerilatorBackend: Toggle signals, advance simulation clock       ║
 * ║       - DevMemBackend: /dev/mem mmap for real FPGA                       ║
 * ║       - SoftwareModelBackend: Pure software for unit testing             ║
 * ║    3. Handle AXI handshaking (valid/ready protocol)                      ║
 * ║    4. Support burst transfers for DMA efficiency                         ║
 * ║    5. Provide memory barriers for cache coherency                        ║
 * ║                                                                           ║
 * ║  AXI4-LITE TRANSACTION FLOW (what you must implement):                   ║
 * ║                                                                           ║
 * ║    WRITE:                                                                 ║
 * ║    ┌────┐   awvalid=1, awaddr=X    ┌────┐                                ║
 * ║    │ M  │ ─────────────────────────▶│ S  │                                ║
 * ║    │ A  │◀───────────────────────── │ L  │  awready=1                    ║
 * ║    │ S  │   wvalid=1, wdata=Y      │ A  │                                ║
 * ║    │ T  │ ─────────────────────────▶│ V  │                                ║
 * ║    │ E  │◀───────────────────────── │ E  │  wready=1                     ║
 * ║    │ R  │◀───────────────────────── │    │  bvalid=1, bresp=OK           ║
 * ║    └────┘   bready=1               └────┘                                ║
 * ║                                                                           ║
 * ║    READ:                                                                  ║
 * ║    ┌────┐   arvalid=1, araddr=X    ┌────┐                                ║
 * ║    │ M  │ ─────────────────────────▶│ S  │                                ║
 * ║    │ A  │◀───────────────────────── │ L  │  arready=1                    ║
 * ║    │ S  │◀───────────────────────── │ A  │  rvalid=1, rdata=Y            ║
 * ║    │ T  │   rready=1               │ V  │                                ║
 * ║    │ E  │ ─────────────────────────▶│ E  │                                ║
 * ║    │ R  │                          │    │                                ║
 * ║    └────┘                          └────┘                                ║
 * ║                                                                           ║
 * ║  REGISTER MAP (must match hw/rtl/csr.sv):                                ║
 * ║    0x00 - CTRL        : Control (start, reset, mode)                     ║
 * ║    0x04 - STATUS      : Status (busy, done, error)                       ║
 * ║    0x10 - ACT_BASE    : Activation buffer address                        ║
 * ║    0x14 - WGT_BASE    : Weight buffer address                            ║
 * ║    0x18 - OUT_BASE    : Output buffer address                            ║
 * ║    0x20-0x3C - Layer config registers                                    ║
 * ║    0x40-0x4C - Quantization registers                                    ║
 * ║    0x50-0x5C - Performance counters                                      ║
 * ║    0x70-0x7C - BSR scheduler registers                                   ║
 * ║                                                                           ║
 * ║  KEY CLASSES TO IMPLEMENT:                                                ║
 * ║    • AXIBackend (abstract base) - Pure virtual interface                 ║
 * ║    • VerilatorBackend - Template class for any Verilator model           ║
 * ║    • DevMemBackend - Linux /dev/mem implementation                       ║
 * ║    • SoftwareModelBackend - For unit testing without hardware            ║
 * ║    • AXIMaster - High-level wrapper using any backend                    ║
 * ║                                                                           ║
 * ║  IMPLEMENTATION NOTES:                                                    ║
 * ║    • VerilatorBackend needs template<typename VModel> for any DUT        ║
 * ║    • DevMemBackend requires root privileges (sudo) for /dev/mem          ║
 * ║    • Use volatile for all hardware register pointers                     ║
 * ║    • ARM barriers: __sync_synchronize() and asm("dsb sy")                ║
 * ║    • Timeout loops to prevent infinite hangs on hardware errors          ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef AXI_MASTER_HPP
#define AXI_MASTER_HPP

#include <cstdint>
#include <memory>
#include <string>

/**
 * Abstract base class for AXI backends
 * Implement this interface for each target (Verilator, FPGA, software)
 */
class AXIBackend {
public:
    virtual ~AXIBackend() = default;
    
    // Single-word operations (AXI4-Lite)
    // TODO: Implement in derived classes
    virtual void write32(uint64_t addr, uint32_t data) = 0;
    virtual uint32_t read32(uint64_t addr) = 0;
    
    // Burst operations (AXI4 full)
    // TODO: Implement in derived classes  
    virtual void write_burst(uint64_t addr, const uint32_t* data, size_t count) = 0;
    virtual void read_burst(uint64_t addr, uint32_t* data, size_t count) = 0;
    
    // Synchronization
    // TODO: Implement in derived classes
    virtual void flush() = 0;      // Ensure writes complete
    virtual void barrier() = 0;    // Memory barrier
    
    // Backend info
    virtual std::string name() const = 0;
    virtual bool is_simulation() const = 0;
};

/**
 * Verilator backend - for simulation
 * Template parameter VModel is the Verilator-generated class (e.g., Vaccel_top)
 * 
 * TODO: Implement the following:
 *   - tick(): Toggle clock, call model->eval()
 *   - write32(): Full AXI4-Lite write handshake
 *   - read32(): Full AXI4-Lite read handshake
 *   - Track cycle count for performance analysis
 */
template<typename VModel>
class VerilatorBackend : public AXIBackend {
public:
    explicit VerilatorBackend(VModel* model, uint64_t base_addr = 0);
    
    void write32(uint64_t addr, uint32_t data) override;
    uint32_t read32(uint64_t addr) override;
    void write_burst(uint64_t addr, const uint32_t* data, size_t count) override;
    void read_burst(uint64_t addr, uint32_t* data, size_t count) override;
    void flush() override;
    void barrier() override;
    std::string name() const override { return "Verilator"; }
    bool is_simulation() const override { return true; }
    
    // Simulation-specific
    void tick();                    // Advance one clock cycle
    uint64_t get_cycle_count() const;
    VModel* get_model();
    
private:
    VModel* model_;
    uint64_t base_addr_;
    uint64_t cycle_count_;
};

/**
 * /dev/mem backend - for real FPGA hardware
 * 
 * TODO: Implement the following:
 *   - Constructor: open("/dev/mem"), mmap() to physical address
 *   - Destructor: munmap(), close()
 *   - write32(): Direct volatile pointer write
 *   - read32(): Direct volatile pointer read
 *   - barrier(): ARM memory barriers for cache coherency
 */
class DevMemBackend : public AXIBackend {
public:
    explicit DevMemBackend(uint64_t phys_addr, size_t size);
    ~DevMemBackend();
    
    void write32(uint64_t addr, uint32_t data) override;
    uint32_t read32(uint64_t addr) override;
    void write_burst(uint64_t addr, const uint32_t* data, size_t count) override;
    void read_burst(uint64_t addr, uint32_t* data, size_t count) override;
    void flush() override;
    void barrier() override;
    std::string name() const override { return "DevMem"; }
    bool is_simulation() const override { return false; }
    
private:
    int fd_;
    volatile uint32_t* mapped_base_;
    uint64_t phys_base_;
    size_t map_size_;
};

/**
 * Software model backend - for unit testing
 * 
 * TODO: Implement the following:
 *   - Simulated register file (std::vector<uint32_t>)
 *   - Basic status register behavior (auto-complete after start)
 */
class SoftwareModelBackend : public AXIBackend {
public:
    explicit SoftwareModelBackend(uint64_t base_addr = 0, size_t size = 0x10000);
    
    void write32(uint64_t addr, uint32_t data) override;
    uint32_t read32(uint64_t addr) override;
    void write_burst(uint64_t addr, const uint32_t* data, size_t count) override;
    void read_burst(uint64_t addr, uint32_t* data, size_t count) override;
    void flush() override;
    void barrier() override;
    std::string name() const override { return "SoftwareModel"; }
    bool is_simulation() const override { return true; }
    
private:
    std::vector<uint32_t> memory_;
    uint64_t base_addr_;
};

/**
 * High-level AXI master class
 * Wraps any backend with convenient methods
 */
class AXIMaster {
public:
    explicit AXIMaster(std::unique_ptr<AXIBackend> backend, uint64_t base_addr = 0);
    
    // Register operations
    // TODO: Implement these
    void write_reg(uint32_t offset, uint32_t value);
    uint32_t read_reg(uint32_t offset);
    void set_bits(uint32_t offset, uint32_t mask);
    void clear_bits(uint32_t offset, uint32_t mask);
    bool test_bits(uint32_t offset, uint32_t mask);
    
    // Memory operations
    // TODO: Implement these
    void write_memory(uint64_t addr, const void* data, size_t bytes);
    void read_memory(uint64_t addr, void* data, size_t bytes);
    
    // Get underlying backend
    AXIBackend* get_backend();
    
private:
    std::unique_ptr<AXIBackend> backend_;
    uint64_t base_addr_;
};

#endif // AXI_MASTER_HPP
