// Test & Benchmark für neon::sum
#include "neon_sum.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::printf("FAIL: " __VA_ARGS__);            \
            std::printf("\n");                            \
            ++failures;                                   \
        }                                                 \
    } while (0)

int main() {
    std::mt19937 rng(42);

    // Viele Längen testen, inkl. Rest-Elemente und ungerader Startadressen.
    for (std::size_t n = 0; n < 300; ++n) {
        for (std::size_t off = 0; off < 3; ++off) {
            std::vector<std::int32_t> vi(n + off);
            std::vector<std::uint8_t> vu(n + off);
            std::vector<float> vf(n + off);
            std::vector<double> vd(n + off);
            std::uniform_int_distribution<std::int32_t> di(INT32_MIN, INT32_MAX);
            std::uniform_int_distribution<int> du(0, 255);
            std::uniform_real_distribution<float> df(-1.0f, 1.0f);
            std::int64_t ri = 0; std::uint64_t ru = 0; double rf = 0, rd = 0;
            for (std::size_t k = 0; k < n + off; ++k) {
                vi[k] = di(rng); vu[k] = (std::uint8_t)du(rng);
                vf[k] = df(rng); vd[k] = vf[k];
                if (k >= off) { ri += vi[k]; ru += vu[k]; rf += vf[k]; rd += vd[k]; }
            }
            CHECK(neon::sum(vi.data() + off, n) == ri, "int32 n=%zu", n);
            CHECK(neon::sum(vu.data() + off, n) == ru, "uint8 n=%zu", n);
            CHECK(std::fabs(neon::sum(vf.data() + off, n) - rf) < 1e-4 * (1 + n), "float n=%zu", n);
            CHECK(std::fabs(neon::sum(vd.data() + off, n) - rd) < 1e-9 * (1 + n), "double n=%zu", n);
        }
    }

    // Überlauf-Test: große Werte / lange uint8-Arrays.
    {
        std::vector<std::int32_t> big(100003, INT32_MAX);
        CHECK(neon::sum(big.data(), big.size()) == (std::int64_t)INT32_MAX * 100003, "int32 overflow");
        std::vector<std::uint8_t> u(1000003, 255);
        CHECK(neon::sum(u.data(), u.size()) == 255ull * 1000003, "uint8 overflow");
    }

    // Kleiner Benchmark
    {
        const std::size_t n = 1 << 24;
        std::vector<float> v(n, 1.0f);
        auto t0 = std::chrono::steady_clock::now();
        float s = 0;
        for (int r = 0; r < 10; ++r) s += neon::sum(v.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        std::printf("float sum of %zu elements: %.2f ms/run (%.2f GB/s), result=%g\n",
                    n, ms, n * sizeof(float) / ms / 1e6, s / 10);
    }

    std::printf(failures ? "%d FAILURES\n" : "All tests passed\n", failures);
    return failures ? 1 : 0;
}
