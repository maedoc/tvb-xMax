"""Measure warmed NumPy versus JAX surrogate-forward throughput.

This intentionally compares only equivalent surrogate forwards.  It does not
claim simulator speedup; use it to decide the portability/performance trade.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from tvb_xmax.compiler import codegen, numpy_codegen


def _median(call, repeats=20):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); call(); samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main():
    rng = np.random.default_rng(42)
    nlat, dparam, dfeat, hidden = 16, 4, 76, 128
    for batch in (1, 64, 4096):
        u = rng.normal(size=(batch, nlat)).astype("f")
        theta = rng.random(size=(batch, dparam)).astype("f")
        xf = np.tanh(u[:, :dparam] @ rng.normal(size=(dparam, dfeat))).astype("f")
        np_trunk, np_head, _ = numpy_codegen.train_surrogate((u, theta, xf), nlat, dfeat,
                                                               hidden=hidden, niter=10)
        np_apply, np_trunk_apply, np_head_apply = numpy_codegen.rebuild_apply_fns(np_trunk, np_head)
        def np_batch():
            np_head_apply(np_trunk_apply(np.concatenate([u, theta], axis=1)))
        np_batch(); np_time = _median(np_batch)

        jax_apply, _ = codegen.make_surrogate_apply(nlat, dparam, dfeat, hidden=hidden)
        jax_batch = jax.jit(jax.vmap(jax_apply))
        ju, jt = jnp.asarray(u), jnp.asarray(theta)
        jax_batch(ju, jt).block_until_ready()
        jax_time = _median(lambda: jax_batch(ju, jt).block_until_ready())
        print(f"batch={batch:4d}  numpy={np_time * 1e3:8.3f} ms  "
              f"jax={jax_time * 1e3:8.3f} ms  numpy/jax={np_time / jax_time:6.2f}x")


if __name__ == "__main__":
    main()
