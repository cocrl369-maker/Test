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

std::uint64_t sum(const std::uint16_t* data, std::size_t n) noexcept {
    std::size_t i = 0;
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

    while (i + kBlock <= n) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        std::size_t blocks = (n - i) / kBlock;
        if (blocks > kInnerBlocks) blocks = kInnerBlocks;

        const std::uint16_t* p = data + i;
        for (std::size_t b = 0; b < blocks; ++b, p += kBlock) {
            __builtin_prefetch(reinterpret_cast<const char*>(p) + kPrefetchBytes, 0, 0);
            const uint16x8x4_t v = vld1q_u16_x4(p);
            acc0 = vpadalq_u16(acc0, v.val[0]);
            acc1 = vpadalq_u16(acc1, v.val[1]);
            acc2 = vpadalq_u16(acc2, v.val[2]);
            acc3 = vpadalq_u16(acc3, v.val[3]);
        }
        i += blocks * kBlock;

        // u32 -> u64 (jeweils paarweise, damit auch hier nichts überläuft)
        total = vpadalq_u32(total, acc0);
        total = vpadalq_u32(total, acc1);
        total = vpadalq_u32(total, acc2);
        total = vpadalq_u32(total, acc3);
    }

    // Rest in 8er-Schritten (max. 3 Vektoren -> kein Überlauf möglich)
    if (i + 8 <= n) {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i + 8 <= n; i += 8)
            acc = vpadalq_u16(acc, vld1q_u16(data + i));
        total = vpadalq_u32(total, acc);
    }

#  if defined(__aarch64__)
    result = vaddvq_u64(total);
#  else
    result = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);
#  endif
#endif

    for (; i < n; ++i) result += data[i];
    return result;
}

} // namespace neon
