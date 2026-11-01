# neon::sum – uint16-Array-Summe für ARM Cortex-A72 (NEON)

```cpp
#include "neon_sum.h"
std::vector<uint16_t> v = ...;
uint64_t s = neon::sum(v.data(), v.size());
```

`std::uint64_t neon::sum(const std::uint16_t* data, std::size_t n)` summiert ein
uint16-Array beliebiger Größe. Das Ergebnis ist 64 Bit breit und läuft nicht über.

## Optimierungen für den A72
- **UADALP** (`vpadalq_u16`): Erweiterung u16 → u32 und Addition in einer Instruktion.
- **4 unabhängige Akkumulatoren**: verstecken die Latenz, dann begrenzt nur noch die
  Load-Pipe (128 Bit/Zyklus) bzw. die Speicherbandbreite.
- **64 Byte pro Iteration** (`vld1q_u16_x4`) = eine Cache-Line des A72.
- **Software-Prefetch** (`prfm pldl1strm`) 256 Byte voraus.
- **Überlaufschutz**: Nach spätestens 32768 Blöcken (1 Mi Elemente) werden die
  u32-Zwischensummen per `vpadalq_u32` in u64 übertragen (per `static_assert` geprüft).
- Ohne NEON (z.B. x86) wird automatisch ein skalarer Fallback kompiliert.

## Bauen & Testen
```sh
make && ./test_neon_sum                      # nativ auf dem A72 (z.B. Raspberry Pi 4)
make CXX=aarch64-linux-gnu-g++               # Cross-Compile
```
Compiler-Flags: `-O3 -mcpu=cortex-a72` (AArch64) bzw. zusätzlich
`-mfpu=neon-fp-armv8 -mfloat-abi=hard` im 32-Bit-Modus.
