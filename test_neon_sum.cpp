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

static void check(const std::uint16_t* values, std::size_t count, const char* label) {
    const std::uint64_t actual = neon::sum(values, count);
    const std::uint64_t expected = reference(values, count);
    if (actual != expected) {
        std::printf("FAIL %s count=%zu: got %llu, expected %llu\n", label, count,
                    (unsigned long long)actual, (unsigned long long)expected);
        ++failures;
    }
}

int main() {
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> distribution(0, 65535);

    // Alle Längen 0..299, auch mit nicht ausgerichteten Startadressen.
    for (std::size_t count = 0; count < 300; ++count) {
        for (std::size_t offset = 0; offset < 3; ++offset) {
            std::vector<std::uint16_t> values(count + offset);
            for (auto& value : values) value = (std::uint16_t)distribution(generator);
            check(values.data() + offset, count, "random");
        }
    }

    // Überlauf-Test: nur Maximalwerte, mehr als 32768 Blöcke (> 1M Elemente),
    // damit der Übertrag u32 -> u64 mehrfach greift.
    {
        std::vector<std::uint16_t> values(3 * 32768 * 32 + 77, 65535);
        check(values.data(), values.size(), "max values");
        check(values.data() + 1, values.size() - 1, "max values, unaligned");
    }

    // Kleiner Benchmark
    {
        const std::size_t count = 1 << 25; // 64 MiB
        std::vector<std::uint16_t> values(count);
        for (auto& value : values) value = (std::uint16_t)distribution(generator);
        const int runs = 10;
        std::uint64_t accumulated = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int run = 0; run < runs; ++run) accumulated += neon::sum(values.data(), count);
        const auto stop = std::chrono::steady_clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count() / runs;
        std::printf("uint16 sum of %zu elements: %.2f ms/run (%.2f GB/s), result=%llu\n",
                    count, milliseconds, count * sizeof(std::uint16_t) / milliseconds / 1e6,
                    (unsigned long long)(accumulated / runs));
    }

    std::printf(failures ? "%d FAILURES\n" : "All tests passed\n", failures);
    return failures ? 1 : 0;
}
