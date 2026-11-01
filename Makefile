# Für den Cortex-A72 (AArch64), nativ auf dem Zielsystem:
#   make
# Cross-Compile:
#   make CXX=aarch64-linux-gnu-g++
CXX      ?= g++
ARCH     ?= $(shell uname -m)
ifeq ($(ARCH),aarch64)
  CPUFLAGS ?= -mcpu=cortex-a72
else ifneq (,$(findstring arm,$(ARCH)))
  CPUFLAGS ?= -mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard
endif
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra $(CPUFLAGS)

test_neon_sum: test_neon_sum.cpp neon_sum.cpp neon_sum.h
	$(CXX) $(CXXFLAGS) -o $@ test_neon_sum.cpp neon_sum.cpp

clean:
	rm -f test_neon_sum
.PHONY: clean
