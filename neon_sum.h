// neon_sum.h - Summe eines beliebig großen uint16-Arrays,
// optimiert für ARM Cortex-A72 (NEON / ASIMD)
#pragma once

#include <cstddef>
#include <cstdint>

namespace neon {

// Summe von count uint16-Werten. Das Ergebnis ist 64 Bit breit und kann
// daher für jede praktisch mögliche Arraygröße nicht überlaufen.
std::uint64_t sum(const std::uint16_t* values, std::size_t count) noexcept;

} // namespace neon
