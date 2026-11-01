// neon_sum.h - Summe beliebig großer Arrays, optimiert für ARM Cortex-A72 (NEON / ASIMD)
#pragma once

#include <cstddef>
#include <cstdint>

namespace neon {

// Summe von n float-Werten. Hinweis: Die Additionsreihenfolge weicht von einer
// sequentiellen Schleife ab, daher kann das Ergebnis im letzten Bit abweichen.
float sum(const float* data, std::size_t n) noexcept;

// Summe von n double-Werten (NEON nur unter AArch64, sonst Skalar-Fallback).
double sum(const double* data, std::size_t n) noexcept;

// Summe von n int32-Werten. Akkumuliert intern in 64 Bit -> kein Überlauf.
std::int64_t sum(const std::int32_t* data, std::size_t n) noexcept;

// Summe von n uint8-Werten (z.B. Bilddaten). Akkumuliert in 64 Bit.
std::uint64_t sum(const std::uint8_t* data, std::size_t n) noexcept;

} // namespace neon
