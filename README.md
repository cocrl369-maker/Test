# neon::sum – Array-Summe für ARM Cortex-A72 (NEON)

`neon_sum.h` / `neon_sum.cpp` stellen Summenfunktionen für Arrays beliebiger Größe bereit:

| Funktion | Eingabe | Rückgabe | NEON-Kern |
|---|---|---|---|
| `neon::sum(const float*, size_t)` | float | float | `ld1 {4 Reg}` + 4× `fadd` |
| `neon::sum(const double*, size_t)` | double | double | `fadd .2d` (nur AArch64) |
| `neon::sum(const int32_t*, size_t)` | int32 | int64 (überlauffrei) | `sadalp` |
| `neon::sum(const uint8_t*, size_t)` | uint8 | uint64 (überlauffrei) | `uadalp` |

```cpp
#include "neon_sum.h"
std::vector<float> v = ...;
float s = neon::sum(v.data(), v.size());
```

## Optimierungen für den A72
- **4 unabhängige Akkumulatoren**: ASIMD-FADD hat 4 Zyklen Latenz. Mehrere Ketten
  verstecken diese Latenz; dann begrenzt nur noch die Load-Pipe (128 Bit/Zyklus) bzw.
  die Speicherbandbreite.
- **64 Byte pro Iteration** (`vld1q_*_x4`) = eine Cache-Line des A72.
- **Software-Prefetch** (`prfm pldl1strm`) 256 Byte voraus.
- **Widening Accumulate** (`SADALP`/`UADALP`) für Integer: Erweiterung und Addition in
  einer Instruktion, kein Überlauf.
- Horizontale Reduktion nur einmal am Ende, Rest-Elemente skalar.
- Ohne NEON (z.B. x86) wird automatisch ein skalarer Fallback kompiliert.

## Bauen & Testen
```sh
make && ./test_neon_sum                      # nativ auf dem A72 (z.B. Raspberry Pi 4)
make CXX=aarch64-linux-gnu-g++               # Cross-Compile
```
Compiler-Flags: `-O3 -mcpu=cortex-a72` (AArch64) bzw. zusätzlich
`-mfpu=neon-fp-armv8 -mfloat-abi=hard` im 32-Bit-Modus.

Hinweis: Bei `float`/`double` unterscheidet sich die Additionsreihenfolge von einer
sequentiellen Schleife, das Ergebnis kann daher minimal (Rundung) abweichen.
