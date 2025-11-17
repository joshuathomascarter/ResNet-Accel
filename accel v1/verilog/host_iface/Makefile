# =============================================================================
# Makefile for tb_axi_lite_slave.sv
# =============================================================================
# Compile and run Verilog testbench with iverilog/Verilator
#
# Usage:
#   make compile     - Compile only
#   make sim         - Run simulation
#   make wave        - Generate waveform (VCD)
#   make clean       - Remove generated files
#   make help        - Show this help
# =============================================================================

# Paths
VERILOG_DIR := .
VERILOG_FILES := $(VERILOG_DIR)/axi_lite_slave_v2.sv $(VERILOG_DIR)/tb_axi_lite_slave.sv

# Compiler options
IVERILOG := iverilog
VVPTOOL := vvp
GTKWAVE := gtkwave

# Output files
COMPILED := test.vvp
WAVEFORM := test.vcd

# Default target
all: sim

# Compile Verilog
compile: $(COMPILED)

$(COMPILED): $(VERILOG_FILES)
	@echo "[COMPILE] Compiling Verilog files..."
	$(IVERILOG) -g2009 -o $(COMPILED) $(VERILOG_FILES)
	@echo "[COMPILE] ✓ Compilation successful"

# Run simulation
sim: $(COMPILED)
	@echo "[SIM] Running simulation..."
	$(VVPTOOL) $(COMPILED)

# Run simulation with waveform dump
wave: $(COMPILED)
	@echo "[SIM] Running simulation with waveform dump..."
	$(VVPTOOL) -v $(WAVEFORM) $(COMPILED)
	@echo "[WAVE] Waveform saved to $(WAVEFORM)"
	@if command -v $(GTKWAVE) > /dev/null; then \
		$(GTKWAVE) $(WAVEFORM) &; \
	else \
		echo "[WARN] GTKWave not installed, skipping waveform viewer"; \
	fi

# Clean generated files
clean:
	@echo "[CLEAN] Removing generated files..."
	rm -f $(COMPILED) $(WAVEFORM) *.o *.so
	@echo "[CLEAN] ✓ Clean complete"

# Help
help:
	@echo "Makefile targets:"
	@echo "  make compile  - Compile Verilog files"
	@echo "  make sim      - Run simulation"
	@echo "  make wave     - Run simulation with waveform dump"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help"

.PHONY: all compile sim wave clean help
