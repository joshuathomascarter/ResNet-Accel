/**
 * @file axi_master.hpp
 * @brief AXI-Lite master interface for register access with multiple backend support
 * @author ResNet-Accel Team
 * @date 2024
 * @copyright MIT License
 * 
 * @details
 * Provides unified AXI-Lite register access with support for:
 * - **DevMemBackend**: Direct FPGA access via /dev/mem mmap 🔥
 * - **VerilatorBackend**: Verilator RTL simulation
 * - **SoftwareModelBackend**: Software behavioral model
 * 
 * All backends implement the same interface, enabling seamless switching
 * between simulation, software model, and hardware deployment.
 */

#ifndef AXI_MASTER_HPP
#define AXI_MASTER_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace resnet_accel {

/**
 * @brief Abstract backend interface for AXI-Lite register operations
 * 
 * Defines the contract all AXI backend implementations must follow.
 */
class AXIBackend {
public:
    virtual ~AXIBackend() = default;

    /**
     * @brief Write 32-bit value to register
     * @param offset Register offset in bytes
     * @param value Value to write
     */
    virtual void write32(std::uint32_t offset, std::uint32_t value) = 0;

    /**
     * @brief Read 32-bit value from register
     * @param offset Register offset in bytes
     * @return Register value
     */
    [[nodiscard]] virtual std::uint32_t read32(std::uint32_t offset) = 0;

    /**
     * @brief Write burst of 32-bit values
     * @param offset Starting offset in bytes
     * @param data Pointer to data array
     * @param count Number of 32-bit words
     */
    virtual void write_burst(std::uint32_t offset, const std::uint32_t* data, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            write32(offset + i * 4, data[i]);
        }
    }

    /**
     * @brief Read burst of 32-bit values
     * @param offset Starting offset in bytes
     * @param data Pointer to destination array
     * @param count Number of 32-bit words
     */
    virtual void read_burst(std::uint32_t offset, std::uint32_t* data, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            data[i] = read32(offset + i * 4);
        }
    }

    /**
     * @brief Memory barrier to ensure ordering
     */
    virtual void barrier() = 0;

    /**
     * @brief Flush write buffers
     */
    virtual void flush() { barrier(); }

    /** @brief Get base physical address */
    [[nodiscard]] virtual std::uint64_t get_base_addr() const = 0;

    /** @brief Get address space size in bytes */
    [[nodiscard]] virtual std::size_t get_size() const = 0;

    /** @brief Get backend name for debugging */
    [[nodiscard]] virtual std::string_view name() const = 0;

    /** @brief Check if this is a simulation backend */
    [[nodiscard]] virtual bool is_simulation() const = 0;
};

//==============================================================================
// DevMem Backend - Direct FPGA Access via /dev/mem
//==============================================================================

/**
 * @brief AXI backend using /dev/mem for direct FPGA register access
 * 
 * @warning Requires root or CAP_SYS_RAWIO capability
 * @warning Memory region must be mapped in device tree
 * 
 * ## Usage
 * ```cpp
 * auto backend = std::make_unique<DevMemBackend>(0x43C00000, 4096);
 * backend->write32(0x00, 0x03);  // Write to CTRL register
 * ```
 */
class DevMemBackend final : public AXIBackend {
public:
    /**
     * @brief Construct backend for physical address region
     * @param base_addr Physical base address (e.g., 0x43C00000)
     * @param size Size of address space in bytes
     * @throws std::runtime_error if /dev/mem unavailable or mmap fails
     */
    DevMemBackend(std::uint64_t base_addr, std::size_t size)
        : base_addr_(base_addr), size_(size), fd_(-1), mapped_base_(nullptr) {
#ifdef __linux__
        fd_ = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0) {
            throw std::runtime_error("DevMemBackend: Failed to open /dev/mem");
        }
        
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, fd_, static_cast<off_t>(base_addr));
        if (ptr == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("DevMemBackend: Failed to mmap");
        }
        
        mapped_base_ = static_cast<volatile std::uint32_t*>(ptr);
#else
        (void)base_addr; (void)size;
        throw std::runtime_error("DevMemBackend only supported on Linux");
#endif
    }

    ~DevMemBackend() override {
#ifdef __linux__
        if (mapped_base_) {
            munmap(const_cast<std::uint32_t*>(mapped_base_), size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
#endif
    }

    DevMemBackend(const DevMemBackend&) = delete;
    DevMemBackend& operator=(const DevMemBackend&) = delete;

    void write32(std::uint32_t offset, std::uint32_t value) override {
#ifdef __linux__
        if (offset >= size_) {
            throw std::out_of_range("DevMemBackend: Offset out of range");
        }
        mapped_base_[offset / 4] = value;
        __sync_synchronize();  // Ensure write completes
#else
        (void)offset; (void)value;
#endif
    }

    [[nodiscard]] std::uint32_t read32(std::uint32_t offset) override {
#ifdef __linux__
        if (offset >= size_) {
            throw std::out_of_range("DevMemBackend: Offset out of range");
        }
        __sync_synchronize();  // Ensure read is fresh
        return mapped_base_[offset / 4];
#else
        (void)offset;
        return 0;
#endif
    }

    void barrier() override {
#ifdef __linux__
        __sync_synchronize();
#endif
    }

    [[nodiscard]] std::uint64_t get_base_addr() const override { return base_addr_; }
    [[nodiscard]] std::size_t get_size() const override { return size_; }
    [[nodiscard]] std::string_view name() const override { return "DevMem"; }
    [[nodiscard]] bool is_simulation() const override { return false; }

private:
    std::uint64_t base_addr_;
    std::size_t size_;
    int fd_;
    volatile std::uint32_t* mapped_base_;
};

//==============================================================================
// Software Model Backend - Pure C++ Behavioral Model
//==============================================================================

/**
 * @brief Software behavioral model for testing without hardware
 * 
 * Simulates register behavior including:
 * - Status register updates on START/RESET
 * - Default array size registers
 * - Write-to-clear interrupt flags
 * 
 * ## Usage
 * ```cpp
 * auto backend = std::make_unique<SoftwareModelBackend>(0x43C00000, 4096);
 * backend->set_trace(true);  // Enable debug output
 * ```
 */
class SoftwareModelBackend final : public AXIBackend {
public:
    /**
     * @brief Construct software model
     * @param base_addr Virtual base address (for address calculations)
     * @param size Size of register space
     */
    SoftwareModelBackend(std::uint64_t base_addr, std::size_t size)
        : base_addr_(base_addr), size_(size), registers_(size / 4, 0), trace_(false) {
        // Initialize default register values
        registers_[0x3C / 4] = 0;      // STATUS = IDLE
        registers_[0x10 / 4] = 16;     // ARRAY_SIZE_ROWS = 16
        registers_[0x14 / 4] = 16;     // ARRAY_SIZE_COLS = 16
        registers_[0x18 / 4] = 16;     // ARRAY_SIZE_DEPTH = 16
    }

    ~SoftwareModelBackend() override = default;

    void write32(std::uint32_t offset, std::uint32_t value) override {
        if (offset >= size_) {
            throw std::out_of_range("SoftwareModelBackend: Offset out of range");
        }
        
        handle_write(offset, value);
        registers_[offset / 4] = value;
        
        if (trace_) {
            std::cout << "[SWModel] W 0x" << std::hex << std::setw(4) << std::setfill('0') 
                     << offset << " = 0x" << std::setw(8) << value << std::dec << "\n";
        }
    }

    [[nodiscard]] std::uint32_t read32(std::uint32_t offset) override {
        if (offset >= size_) {
            throw std::out_of_range("SoftwareModelBackend: Offset out of range");
        }
        
        std::uint32_t val = registers_[offset / 4];
        
        if (trace_) {
            std::cout << "[SWModel] R 0x" << std::hex << std::setw(4) << std::setfill('0')
                     << offset << " = 0x" << std::setw(8) << val << std::dec << "\n";
        }
        
        return val;
    }

    void barrier() override {}

    [[nodiscard]] std::uint64_t get_base_addr() const override { return base_addr_; }
    [[nodiscard]] std::size_t get_size() const override { return size_; }
    [[nodiscard]] std::string_view name() const override { return "SoftwareModel"; }
    [[nodiscard]] bool is_simulation() const override { return true; }

    /** @brief Enable/disable register access tracing */
    void set_trace(bool enable) { trace_ = enable; }

    /** @brief Direct access to register array for testing */
    [[nodiscard]] std::vector<std::uint32_t>& registers() { return registers_; }

private:
    std::uint64_t base_addr_;
    std::size_t size_;
    std::vector<std::uint32_t> registers_;
    bool trace_;

    /**
     * @brief Handle special register write behaviors
     * @param offset Register offset
     * @param value Value being written
     */
    void handle_write(std::uint32_t offset, std::uint32_t value) {
        // CTRL register (0x00)
        if (offset == 0x00) {
            if (value & 1) {  // START bit
                registers_[0x3C / 4] = 2;  // Set BUSY
            } else if (value & 2) {  // RESET bit
                registers_[0x3C / 4] = 0;  // Clear to IDLE
            }
        }
        // STATUS register (0x3C) - write-to-clear interrupt flags
        else if (offset == 0x3C) {
            registers_[0x3C / 4] &= ~value;
        }
    }
};

//==============================================================================
// Verilator Backend - RTL Simulation
//==============================================================================

/**
 * @brief AXI backend for Verilator RTL simulation
 * 
 * @tparam VModel Verilator model class (e.g., Vtop)
 * 
 * Implements full AXI-Lite handshaking protocol with:
 * - AWVALID/AWREADY for write address
 * - WVALID/WREADY for write data
 * - BVALID/BREADY for write response
 * - ARVALID/ARREADY for read address
 * - RVALID/RREADY for read data
 * 
 * ## Usage
 * ```cpp
 * Vtop* model = new Vtop;
 * auto backend = std::make_unique<VerilatorBackend<Vtop>>(model, 0x43C00000, 4096);
 * backend->write32(0x00, 0x01);  // Performs full AXI transaction
 * ```
 */
template<typename VModel>
class VerilatorBackend final : public AXIBackend {
public:
    /**
     * @brief Construct Verilator backend
     * @param model Pointer to Verilator model instance
     * @param base_addr Virtual base address
     * @param size Address space size
     */
    VerilatorBackend(VModel* model, std::uint64_t base_addr, std::size_t size)
        : model_(model), base_addr_(base_addr), size_(size), cycles_(0) {}

    ~VerilatorBackend() override = default;

    /**
     * @brief AXI-Lite write transaction
     * 
     * Performs full handshaking:
     * 1. Assert AWVALID, WVALID
     * 2. Wait for AWREADY, WREADY
     * 3. Wait for BVALID
     * 4. Deassert all signals
     */
    void write32(std::uint32_t offset, std::uint32_t value) override {
        // Address write channel
        model_->s_axi_awaddr = offset;
        model_->s_axi_awvalid = 1;
        
        // Data write channel
        model_->s_axi_wdata = value;
        model_->s_axi_wstrb = 0xF;  // All bytes valid
        model_->s_axi_wvalid = 1;
        
        // Write response channel
        model_->s_axi_bready = 1;
        
        // Wait for address and data acceptance
        while (!model_->s_axi_awready || !model_->s_axi_wready) {
            tick();
        }
        tick();
        
        // Deassert address and data
        model_->s_axi_awvalid = 0;
        model_->s_axi_wvalid = 0;
        
        // Wait for write response
        while (!model_->s_axi_bvalid) {
            tick();
        }
        tick();
        
        model_->s_axi_bready = 0;
    }

    /**
     * @brief AXI-Lite read transaction
     * 
     * Performs full handshaking:
     * 1. Assert ARVALID
     * 2. Wait for ARREADY
     * 3. Wait for RVALID
     * 4. Capture RDATA, deassert signals
     */
    [[nodiscard]] std::uint32_t read32(std::uint32_t offset) override {
        // Address read channel
        model_->s_axi_araddr = offset;
        model_->s_axi_arvalid = 1;
        model_->s_axi_rready = 1;
        
        // Wait for address acceptance
        while (!model_->s_axi_arready) {
            tick();
        }
        tick();
        
        model_->s_axi_arvalid = 0;
        
        // Wait for read data
        while (!model_->s_axi_rvalid) {
            tick();
        }
        
        std::uint32_t data = model_->s_axi_rdata;
        tick();
        
        model_->s_axi_rready = 0;
        return data;
    }

    void barrier() override { tick(); }

    [[nodiscard]] std::uint64_t get_base_addr() const override { return base_addr_; }
    [[nodiscard]] std::size_t get_size() const override { return size_; }
    [[nodiscard]] std::string_view name() const override { return "Verilator"; }
    [[nodiscard]] bool is_simulation() const override { return true; }

    /** @brief Advance simulation by one clock cycle */
    void tick() {
        model_->clk = 0;
        model_->eval();
        model_->clk = 1;
        model_->eval();
        ++cycles_;
    }

    /** @brief Get total simulation cycles */
    [[nodiscard]] std::uint64_t get_cycles() const { return cycles_; }

    /** @brief Get direct access to Verilator model */
    [[nodiscard]] VModel* get_model() { return model_; }

private:
    VModel* model_;
    std::uint64_t base_addr_;
    std::size_t size_;
    std::uint64_t cycles_;
};

//==============================================================================
// AXI Master - High-Level Register Access API
//==============================================================================

/**
 * @brief High-level AXI-Lite master for convenient register access
 * 
 * Wraps an AXI backend with convenient methods for:
 * - Individual register read/write
 * - Bit manipulation (set/clear/test)
 * - Burst transfers
 * - Memory-mapped buffer access
 * 
 * ## Example
 * ```cpp
 * auto backend = std::make_unique<DevMemBackend>(0x43C00000, 4096);
 * AXIMaster axi(std::move(backend), 0x43C00000);
 * 
 * axi.write_reg(CSR::CTRL, 0x01);     // Start
 * axi.set_bits(CSR::CTRL, 0x04);      // Set interrupt enable
 * bool done = axi.test_bits(CSR::STATUS, 0x01);  // Check DONE bit
 * ```
 */
class AXIMaster {
public:
    /**
     * @brief Construct AXI master with backend
     * @param backend Backend implementation (takes ownership)
     * @param base_addr Physical base address
     */
    AXIMaster(std::unique_ptr<AXIBackend> backend, std::uint64_t base_addr)
        : backend_(std::move(backend)), base_addr_(base_addr) {}

    ~AXIMaster() = default;

    // Non-copyable, movable
    AXIMaster(const AXIMaster&) = delete;
    AXIMaster& operator=(const AXIMaster&) = delete;
    AXIMaster(AXIMaster&&) noexcept = default;
    AXIMaster& operator=(AXIMaster&&) noexcept = default;

    //==========================================================================
    // Register Access
    //==========================================================================

    /**
     * @brief Write to register
     * @param offset Register offset in bytes
     * @param value Value to write
     */
    void write_reg(std::uint32_t offset, std::uint32_t value) {
        backend_->write32(offset, value);
    }

    /**
     * @brief Read from register
     * @param offset Register offset in bytes
     * @return Register value
     */
    [[nodiscard]] std::uint32_t read_reg(std::uint32_t offset) {
        return backend_->read32(offset);
    }

    /**
     * @brief Set specific bits in register (read-modify-write)
     * @param offset Register offset
     * @param mask Bit mask to set
     */
    void set_bits(std::uint32_t offset, std::uint32_t mask) {
        write_reg(offset, read_reg(offset) | mask);
    }

    /**
     * @brief Clear specific bits in register (read-modify-write)
     * @param offset Register offset
     * @param mask Bit mask to clear
     */
    void clear_bits(std::uint32_t offset, std::uint32_t mask) {
        write_reg(offset, read_reg(offset) & ~mask);
    }

    /**
     * @brief Test if specific bits are set
     * @param offset Register offset
     * @param mask Bit mask to test
     * @return true if all bits in mask are set
     */
    [[nodiscard]] bool test_bits(std::uint32_t offset, std::uint32_t mask) {
        return (read_reg(offset) & mask) == mask;
    }

    //==========================================================================
    // Burst Transfers
    //==========================================================================

    /**
     * @brief Write burst of values
     * @param offset Starting offset
     * @param vals Vector of values to write
     */
    void write_burst(std::uint32_t offset, const std::vector<std::uint32_t>& vals) {
        backend_->write_burst(offset, vals.data(), vals.size());
        backend_->barrier();
    }

    /**
     * @brief Read burst of values
     * @param offset Starting offset
     * @param count Number of words to read
     * @return Vector of read values
     */
    [[nodiscard]] std::vector<std::uint32_t> read_burst(std::uint32_t offset, std::size_t count) {
        std::vector<std::uint32_t> vals(count);
        backend_->read_burst(offset, vals.data(), count);
        return vals;
    }

    //==========================================================================
    // Memory Access
    //==========================================================================

    /**
     * @brief Write to memory-mapped buffer
     * @param addr Absolute physical address
     * @param data Pointer to source data
     * @param bytes Number of bytes to write
     */
    void write_memory(std::uint64_t addr, const void* data, std::size_t bytes) {
        backend_->write_burst(static_cast<std::uint32_t>(addr - base_addr_),
                             static_cast<const std::uint32_t*>(data), 
                             (bytes + 3) / 4);
    }

    /**
     * @brief Read from memory-mapped buffer
     * @param addr Absolute physical address
     * @param data Pointer to destination buffer
     * @param bytes Number of bytes to read
     */
    void read_memory(std::uint64_t addr, void* data, std::size_t bytes) {
        backend_->read_burst(static_cast<std::uint32_t>(addr - base_addr_),
                            static_cast<std::uint32_t*>(data), 
                            (bytes + 3) / 4);
    }

    //==========================================================================
    // Backend Access
    //==========================================================================

    /** @brief Ensure all writes complete */
    void barrier() { backend_->barrier(); }

    /** @brief Get underlying backend (for advanced use) */
    [[nodiscard]] AXIBackend* get_backend() { return backend_.get(); }

    /** @brief Get base physical address */
    [[nodiscard]] std::uint64_t get_base_addr() const { return base_addr_; }

private:
    std::unique_ptr<AXIBackend> backend_;
    std::uint64_t base_addr_;
};

} // namespace resnet_accel

#endif // AXI_MASTER_HPP
