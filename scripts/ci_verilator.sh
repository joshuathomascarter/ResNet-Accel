#!/bin/bash
# =============================================================================
# ci_verilator.sh — Continuous Integration with Verilator
# =============================================================================
# Purpose:
#   Automated regression testing for ACCEL-v1 RTL modules.
#   Uses Makefile.verilator to build and run tests.
#
# Usage:
#   ./ci_verilator.sh [test_name]
#   ./ci_verilator.sh all
#
# =============================================================================

set -e

WORKSPACE="/workspaces/ACCEL-v1"
BUILD_DIR="${WORKSPACE}/build/verilator"
LOG_DIR="${WORKSPACE}/accel/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}ACCEL-v1 CI: Verilator Regression Suite${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# ============================================================================
# Helper Functions
# ============================================================================

run_make_target() {
    local target=$1
    local description=$2
    
    echo -ne "${YELLOW}[TEST]${NC} $description ... "
    
    if make -f Makefile.verilator $target > "$LOG_DIR/$target.log" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$target")
        return 1
    fi
}

run_executable() {
    local exe=$1
    local description=$2
    
    echo -ne "${YELLOW}[RUN]${NC} $description ... "
    
    if $exe > "$LOG_DIR/run_$description.log" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$description")
        return 1
    fi
}

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$LOG_DIR"

# Check for Verilator
if ! command -v verilator &> /dev/null; then
    echo -e "${RED}ERROR: Verilator not installed${NC}"
    exit 1
fi

VERILATOR_VERSION=$(verilator --version 2>&1 | head -1)
echo -e "Verilator: ${GREEN}${VERILATOR_VERSION}${NC}\n"

cd "$WORKSPACE"

# ============================================================================
# Test Suite
# ============================================================================

TEST_SUITE=$1
if [ -z "$TEST_SUITE" ]; then
    TEST_SUITE="all"
fi

# Test 1: Linting
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "lint" ]]; then
    run_make_target "lint" "RTL Linting" || true
fi

# Test 2: Stress Test Build & Run
if [[ "$TEST_SUITE" == "all" || "$TEST_SUITE" == "stress" ]]; then
    run_make_target "stress" "Build Stress Test"
    if [ $? -eq 0 ]; then
        run_executable "./build/verilator/Vaccel_top_stress" "Run Stress Test" || true
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Test Summary${NC}"
echo -e "${YELLOW}========================================${NC}\n"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo -e "Total:  ${TOTAL_TESTS}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  • $test"
        echo -e "    Logs: $LOG_DIR/$test.log"
    done
    exit 1
else
    echo -e "\n${GREEN}✓ All tests passed!${NC}\n"
    exit 0
fi
