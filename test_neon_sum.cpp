// Test & Benchmark für neon::sum (uint16)
#include "neon_sum.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

static int failures = 0;

static std::uint64_t reference(const std::uint16_t* values, std::size_t count) {
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < count; ++index) total += values[index];
    return total;
}

static void check(const std::uint16_t* d, std::size_t n, const char* what) {
    const std::uint64_t got = neon::sum(d, n), exp = reference(d, n);
    if (got != exp) {
        std::printf("FAIL %s n=%zu: got %llu, expected %llu\n", what, n,
                    (unsigned long long)got, (unsigned long long)exp);
        ++failures;
    }
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 65535);

    // Alle Längen 0..299, auch mit nicht ausgerichteten Startadressen.
    for (std::size_t n = 0; n < 300; ++n) {
        for (std::size_t off = 0; off < 3; ++off) {
            std::vector<std::uint16_t> v(n + off);
            for (auto& x : v) x = (std::uint16_t)dist(rng);
            check(v.data() + off, n, "random");
        }
    }

    // Überlauf-Test: nur Maximalwerte, mehr als 32768 Blöcke (> 1M Elemente),
    // damit der Übertrag u32 -> u64 mehrfach greift.
    {
        std::vector<std::uint16_t> v(3 * 32768 * 32 + 77, 65535);
        check(v.data(), v.size(), "max values");
        check(v.data() + 1, v.size() - 1, "max values, unaligned");
    }

    // Kleiner Benchmark
    {
        const std::size_t n = 1 << 25; // 64 MiB
        std::vector<std::uint16_t> v(n);
        for (auto& x : v) x = (std::uint16_t)dist(rng);
        const int runs = 10;
        std::uint64_t s = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < runs; ++r) s += neon::sum(v.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
        std::printf("uint16 sum of %zu elements: %.2f ms/run (%.2f GB/s), result=%llu\n",
                    n, ms, n * sizeof(std::uint16_t) / ms / 1e6,
                    (unsigned long long)(s / runs));
    }

    std::printf(failures ? "%d FAILURES\n" : "All tests passed\n", failures);
    return failures ? 1 : 0;
}
