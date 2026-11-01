// neon_sum.cpp - NEON-optimierte Array-Summen für ARM Cortex-A72
//
// Optimierungen für den Cortex-A72:
//  * 4 unabhängige 128-Bit-Akkumulatoren: ASIMD-FADD hat auf dem A72 eine Latenz
//    von 4 Zyklen. Mit nur einem Akkumulator würde jede Addition auf die vorige
//    warten. 4 Ketten verstecken die Latenz, sodass die Lade-Einheit
//    (1x 128 Bit pro Zyklus) der limitierende Faktor wird.
//  * vld1q_*_x4 lädt 64 Byte (= eine Cache-Line des A72) pro Iteration.
//  * Software-Prefetch einige Cache-Lines voraus für große Arrays.
//  * Integer: Pairwise-Add-Accumulate-Long (SADALP/UADALP) erweitert direkt in
//    breitere Lanes -> überlauffrei ohne extra Instruktionen.
//  * Horizontale Reduktion erst ganz am Ende, danach skalarer Rest (< 1 Block).
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

namespace {

// Prefetch-Distanz in Bytes (ca. 4 Cache-Lines à 64 Byte voraus).
constexpr std::size_t kPrefetchBytes = 256;

inline void prefetch(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0 /*read*/, 0 /*streaming, kein Cache-Halten*/);
#else
    (void)p;
#endif
}

#if NEON_SUM_HAVE_NEON
inline float hsum(float32x4_t v) noexcept {
#  if defined(__aarch64__)
    return vaddvq_f32(v);
#  else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#  endif
}

inline std::int64_t hsum(int64x2_t v) noexcept {
#  if defined(__aarch64__)
    return vaddvq_s64(v);
#  else
    return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
#  endif
}

inline std::uint64_t hsum(uint64x2_t v) noexcept {
#  if defined(__aarch64__)
    return vaddvq_u64(v);
#  else
    return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
#  endif
}
#endif

} // namespace

// ---------------------------------------------------------------- float
float sum(const float* data, std::size_t n) noexcept {
    std::size_t i = 0;
    float result = 0.0f;

#if NEON_SUM_HAVE_NEON
    constexpr std::size_t kBlock = 16; // 4 Vektoren x 4 floats = 64 Byte
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (; i + kBlock <= n; i += kBlock) {
        prefetch(reinterpret_cast<const char*>(data + i) + kPrefetchBytes);
        const float32x4x4_t v = vld1q_f32_x4(data + i);
        acc0 = vaddq_f32(acc0, v.val[0]);
        acc1 = vaddq_f32(acc1, v.val[1]);
        acc2 = vaddq_f32(acc2, v.val[2]);
        acc3 = vaddq_f32(acc3, v.val[3]);
    }
    for (; i + 4 <= n; i += 4)
        acc0 = vaddq_f32(acc0, vld1q_f32(data + i));

    result = hsum(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#endif

    for (; i < n; ++i) result += data[i];
    return result;
}

// ---------------------------------------------------------------- double
double sum(const double* data, std::size_t n) noexcept {
    std::size_t i = 0;
    double result = 0.0;

#if NEON_SUM_HAVE_NEON && defined(__aarch64__)
    constexpr std::size_t kBlock = 8; // 4 Vektoren x 2 doubles = 64 Byte
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    for (; i + kBlock <= n; i += kBlock) {
        prefetch(reinterpret_cast<const char*>(data + i) + kPrefetchBytes);
        const float64x2x4_t v = vld1q_f64_x4(data + i);
        acc0 = vaddq_f64(acc0, v.val[0]);
        acc1 = vaddq_f64(acc1, v.val[1]);
        acc2 = vaddq_f64(acc2, v.val[2]);
        acc3 = vaddq_f64(acc3, v.val[3]);
    }
    for (; i + 2 <= n; i += 2)
        acc0 = vaddq_f64(acc0, vld1q_f64(data + i));

    result = vaddvq_f64(vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)));
#else
    // Skalar mit 4 Akkumulatoren (auch ohne NEON besser als 1 Kette).
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (; i + 4 <= n; i += 4) {
        a0 += data[i]; a1 += data[i + 1]; a2 += data[i + 2]; a3 += data[i + 3];
    }
    result = (a0 + a1) + (a2 + a3);
#endif

    for (; i < n; ++i) result += data[i];
    return result;
}

// ---------------------------------------------------------------- int32 -> int64
std::int64_t sum(const std::int32_t* data, std::size_t n) noexcept {
    std::size_t i = 0;
    std::int64_t result = 0;

#if NEON_SUM_HAVE_NEON
    constexpr std::size_t kBlock = 16; // 4 Vektoren x 4 int32 = 64 Byte
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int64x2_t acc2 = vdupq_n_s64(0);
    int64x2_t acc3 = vdupq_n_s64(0);

    for (; i + kBlock <= n; i += kBlock) {
        prefetch(reinterpret_cast<const char*>(data + i) + kPrefetchBytes);
        const int32x4x4_t v = vld1q_s32_x4(data + i);
        // SADALP: acc[k] += v[2k] + v[2k+1] (in 64 Bit)
        acc0 = vpadalq_s32(acc0, v.val[0]);
        acc1 = vpadalq_s32(acc1, v.val[1]);
        acc2 = vpadalq_s32(acc2, v.val[2]);
        acc3 = vpadalq_s32(acc3, v.val[3]);
    }
    for (; i + 4 <= n; i += 4)
        acc0 = vpadalq_s32(acc0, vld1q_s32(data + i));

    result = hsum(vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3)));
#endif

    for (; i < n; ++i) result += data[i];
    return result;
}

// ---------------------------------------------------------------- uint8 -> uint64
std::uint64_t sum(const std::uint8_t* data, std::size_t n) noexcept {
    std::size_t i = 0;
    std::uint64_t result = 0;

#if NEON_SUM_HAVE_NEON
    // u8 -> u16 per UADALP. Jede u16-Lane erhält pro Block max. 2*2*255 = 1020
    // (zwei Vektoren pro Akkumulator). 64 Blöcke -> max. 65280 < 65535.
    // Danach wird der u16-Zwischenstand in u64 gekippt.
    constexpr std::size_t kBlock = 64;       // 4 Vektoren x 16 Byte
    constexpr std::size_t kInnerBlocks = 64; // sicher gegen u16-Überlauf
    uint64x2_t total = vdupq_n_u64(0);

    while (i + kBlock <= n) {
        uint16x8_t acc0 = vdupq_n_u16(0);
        uint16x8_t acc1 = vdupq_n_u16(0);
        std::size_t blocks = (n - i) / kBlock;
        if (blocks > kInnerBlocks) blocks = kInnerBlocks;

        for (std::size_t b = 0; b < blocks; ++b, i += kBlock) {
            prefetch(data + i + kPrefetchBytes);
            const uint8x16x4_t v = vld1q_u8_x4(data + i);
            acc0 = vpadalq_u8(acc0, v.val[0]);
            acc1 = vpadalq_u8(acc1, v.val[1]);
            acc0 = vpadalq_u8(acc0, v.val[2]);
            acc1 = vpadalq_u8(acc1, v.val[3]);
        }
        // u16 -> u32 -> u64
        const uint32x4_t s32 = vpaddlq_u16(acc0);
        total = vpadalq_u32(total, vpadalq_u16(s32, acc1));
    }
    result = hsum(total);
#endif

    for (; i < n; ++i) result += data[i];
    return result;
}

} // namespace neon
