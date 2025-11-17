#!/bin/bash
# =============================================================================
# quick_test.sh — Quick test script for AXI Master Integration
# =============================================================================
# Usage:
#   cd /workspaces/ACCEL-v1/accel\ v1
#   bash quick_test.sh [test_type]
#
# Test types: cocotb, verilog, python, all (default)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/workspaces/ACCEL-v1/accel v1"
TEST_TYPE="${1:-all}"
VERBOSE="${VERBOSE:-0}"

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# =============================================================================
# Pre-flight checks
# =============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check Cocotb (optional, only for cocotb tests)
    if [[ "$TEST_TYPE" == "cocotb" ]] || [[ "$TEST_TYPE" == "all" ]]; then
        if ! python3 -c "import cocotb" 2>/dev/null; then
            print_error "Cocotb not installed"
            echo "Install with: pip install cocotb"
            exit 1
        fi
        print_success "Cocotb installed"
    fi
    
    # Check iverilog (optional, only for verilog tests)
    if [[ "$TEST_TYPE" == "verilog" ]] || [[ "$TEST_TYPE" == "all" ]]; then
        if ! command -v iverilog &> /dev/null; then
            print_error "iverilog not found"
            echo "Install with: apt-get install iverilog"
            exit 1
        fi
        print_success "iverilog found: $(iverilog -V 2>&1 | head -1)"
    fi
}

# =============================================================================
# Test: Python AXI Master Simulator
# =============================================================================

test_python_axi() {
    print_header "Test 1: Python AXI Master Simulator"
    
    cd "$PROJECT_ROOT"
    
    python3 << 'PYTHON_TEST'
import sys
sys.path.insert(0, 'python/host')

from axi_master_sim import AXIMasterSim, AXIResponse

# Create simulator
axi = AXIMasterSim(name="QuickTest", debug=True)

print("\n[Test 1.1] Single Write")
success, resp = axi.write_single(0x50, 0xDEADBEEF)
assert success and resp == AXIResponse.OKAY, "Write failed"
print("✓ Write successful")

print("\n[Test 1.2] Single Read")
data, resp = axi.read_single(0x50)
assert data == 0xDEADBEEF, f"Data mismatch: {data:08x} != 0xdeadbeef"
print(f"✓ Read successful: 0x{data:08x}")

print("\n[Test 1.3] Burst Write (2 beats - valid CSR range)")
success, beats, responses = axi.write_burst(0x50, [0x11, 0x22])
assert success and beats == 2, "Burst failed"
print(f"✓ Burst write successful: {beats} beats")

print("\n[Test 1.4] DMA FIFO Operations")
axi.dma_write(0xFFFFFFFF)
axi.dma_write(0x12345678)
data = axi.dma_read()
assert data == 0xFFFFFFFF, "FIFO read failed"
print("✓ FIFO operations successful")

print("\n[Test 1.5] Metrics")
axi.print_metrics()

print("\n✓ All Python tests passed!")
PYTHON_TEST
    
    if [ $? -eq 0 ]; then
        print_success "Python AXI simulator tests passed"
    else
        print_error "Python AXI simulator tests failed"
        exit 1
    fi
}

# =============================================================================
# Test: Verilog Testbench
# =============================================================================

test_verilog_tb() {
    print_header "Test 2: Verilog Enhanced Testbench"
    
    cd "$PROJECT_ROOT"
    
    print_info "Compiling Verilog..."
    iverilog -g2009 -Wall -Winfclip \
        -o tb/tb_axi.vvp \
        verilog/host_iface/axi_lite_slave_v2.sv \
        verilog/host_iface/tb_axi_lite_slave_enhanced.sv
    
    print_info "Running simulation..."
    vvp tb/tb_axi.vvp -v 2>&1 | tail -20
    
    if grep -q "ALL PASSED" tb/tb_axi.log 2>/dev/null || \
       vvp tb/tb_axi.vvp -v 2>&1 | grep -q "ALL PASSED"; then
        print_success "Verilog testbench passed"
    else
        print_error "Verilog testbench failed"
        exit 1
    fi
}

# =============================================================================
# Test: Cocotb Integration
# =============================================================================

test_cocotb() {
    print_header "Test 3: Cocotb Python ↔ Verilog Integration"
    
    cd "$PROJECT_ROOT"
    
    print_info "Running Cocotb tests..."
    make -f tb/Makefile.cocotb SIM=iverilog 2>&1 | tail -30
    
    if [ $? -eq 0 ]; then
        print_success "Cocotb integration tests passed"
    else
        print_error "Cocotb integration tests failed"
        exit 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "=========================================="
    echo "AXI Master Integration Quick Test"
    echo "=========================================="
    echo "Project: ACCEL-v1"
    echo "Test Type: $TEST_TYPE"
    echo ""
    
    check_prerequisites
    
    case "$TEST_TYPE" in
        python)
            test_python_axi
            ;;
        verilog)
            test_verilog_tb
            ;;
        cocotb)
            test_cocotb
            ;;
        all)
            test_python_axi
            test_verilog_tb
            test_cocotb
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            echo "Valid types: python, verilog, cocotb, all"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ALL TESTS PASSED!                                         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Run main
main
