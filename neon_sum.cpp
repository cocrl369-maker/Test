// neon_sum.cpp - NEON-optimierte uint16-Array-Summe für ARM Cortex-A72
//
// Optimierungen für den Cortex-A72:
//  * UADALP (vpadalq_u16): addiert je zwei benachbarte u16 und akkumuliert sie
//    in u32-Lanes - Erweiterung + Addition in einer Instruktion.
//  * 4 unabhängige Akkumulatoren: UADALP hat auf dem A72 mehrere Zyklen
//    Latenz. Mit 4 Ketten muss keine Instruktion auf die vorige warten und die
//    Lade-Einheit (128 Bit pro Zyklus) wird der limitierende Faktor.
//  * vld1q_u16_x4 lädt 64 Byte (= eine Cache-Line des A72) pro Iteration.
//  * Software-Prefetch (PRFM PLDL1STRM) 256 Byte voraus.
//  * Die u32-Zwischensummen werden rechtzeitig vor einem Überlauf in u64
//    übertragen; horizontale Reduktion nur einmal am Ende.
//
// Kompilieren (AArch64):  g++ -O3 -mcpu=cortex-a72 -c neon_sum.cpp
// Kompilieren (ARMv7/32): g++ -O3 -mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard -c neon_sum.cpp

#include "neon_sum.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define NEON_SUM_HAVE_NEON 1
#else
#  define NEON_SUM_HAVE_NEON 0
#endif

namespace neon {

std::uint64_t sum(const std::uint16_t* values, std::size_t count) noexcept {
    std::size_t index = 0;
    std::uint64_t result = 0;

#if NEON_SUM_HAVE_NEON
    constexpr std::size_t kBlock = 32;          // 4 Vektoren x 8 u16 = 64 Byte
    constexpr std::size_t kPrefetchBytes = 256; // 4 Cache-Lines voraus

    // Jede u32-Lane erhält pro Block max. 2 * 65535 = 131070.
    // 32768 Blöcke -> max. 4294901760 < 2^32 - 1, also kein Überlauf.
    constexpr std::size_t kInnerBlocks = 32768;
    static_assert(2ull * 65535ull * kInnerBlocks <= 0xFFFFFFFFull,
                  "u32-Akkumulatoren könnten überlaufen");

    uint64x2_t total = vdupq_n_u64(0);

    while (index + kBlock <= count) {
        uint32x4_t accumulator0 = vdupq_n_u32(0);
        uint32x4_t accumulator1 = vdupq_n_u32(0);
        uint32x4_t accumulator2 = vdupq_n_u32(0);
        uint32x4_t accumulator3 = vdupq_n_u32(0);

        std::size_t block_count = (count - index) / kBlock;
        if (block_count > kInnerBlocks) block_count = kInnerBlocks;

        const std::uint16_t* block_start = values + index;
        for (std::size_t block = 0; block < block_count; ++block, block_start += kBlock) {
            __builtin_prefetch(reinterpret_cast<const char*>(block_start) + kPrefetchBytes, 0, 0);
            const uint16x8x4_t loaded = vld1q_u16_x4(block_start);
            accumulator0 = vpadalq_u16(accumulator0, loaded.val[0]);
            accumulator1 = vpadalq_u16(accumulator1, loaded.val[1]);
            accumulator2 = vpadalq_u16(accumulator2, loaded.val[2]);
            accumulator3 = vpadalq_u16(accumulator3, loaded.val[3]);
        }
        index += block_count * kBlock;

        // u32 -> u64 (jeweils paarweise, damit auch hier nichts überläuft)
        total = vpadalq_u32(total, accumulator0);
        total = vpadalq_u32(total, accumulator1);
        total = vpadalq_u32(total, accumulator2);
        total = vpadalq_u32(total, accumulator3);
    }

    // Rest in 8er-Schritten (max. 3 Vektoren -> kein Überlauf möglich)
    if (index + 8 <= count) {
        uint32x4_t remainder_accumulator = vdupq_n_u32(0);
        for (; index + 8 <= count; index += 8)
            remainder_accumulator = vpadalq_u16(remainder_accumulator, vld1q_u16(values + index));
        total = vpadalq_u32(total, remainder_accumulator);
    }

#  if defined(__aarch64__)
    result = vaddvq_u64(total);
#  else
    result = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);
#  endif
#endif

    for (; index < count; ++index) result += values[index];
    return result;
}

} // namespace neon
