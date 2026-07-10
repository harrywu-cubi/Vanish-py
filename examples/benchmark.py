# examples/benchmark.py
"""Benchmark naive per-pixel loops vs the vectorized NumPy pipeline."""
import time
import numpy as np
from vanish import core, _naive


def _bench(fn, *args, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    return (time.perf_counter() - start) / repeat


def main():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    energy = core.energy_map(img)

    t_naive_e = _bench(_naive.energy_map_naive, img)
    t_vec_e = _bench(core.energy_map, img, repeat=10)
    t_naive_dp = _bench(_naive.cumulative_energy_naive, energy)
    t_vec_dp = _bench(lambda e: core.cumulative_energy(e), energy, repeat=10)

    print(f"energy map:  naive {t_naive_e:8.4f}s  vec {t_vec_e:8.4f}s  "
          f"speedup {t_naive_e / t_vec_e:6.1f}x")
    print(f"DP table:    naive {t_naive_dp:8.4f}s  vec {t_vec_dp:8.4f}s  "
          f"speedup {t_naive_dp / t_vec_dp:6.1f}x")


if __name__ == "__main__":
    main()
