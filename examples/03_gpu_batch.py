"""Example: GPU batch eval for "maxest speedup".

Shows the vectorize stage: evaluating a compiled artifact over a large
batch of (u, theta) pairs in one shot, and benchmarking the speedup vs
the real simulation.  This is where the ~10^3-10^4x amortized speedup
shows up.
"""
import tvb_xmax  # noqa: F401  (puts vendored vbjax/apvbt on sys.path)
import time
import jax.numpy as jnp
import vbjax as vb

from tvb_xmax.compiler import ir, pipeline, vectorize


def main():
    nlat, d_feat, B = 16, 8, 8192
    cc = vb.CrossCoder(variational=False)
    import numpy as np
    cc.add_view(np.random.randn(20, nlat), "079-Shen2013", normalize="center")
    cc.tts = 10
    cc.train(nlat=nlat, niter=50)
    surr = tvb_xmax.surrogates.get_surrogate("hopf")
    d_param = len(surr.param_names)
    U = vb.randn(4096, nlat) * 0.3
    Theta = vb.rand(4096, d_param)
    XF = jnp.tanh(U[:, :d_feat] + 0.3 * Theta[:, :1])
    spec = ir.IRSpec(model="hopf", connectivity=jnp.zeros(nlat),
                     connectivity_is_latent=True, parameters=surr.default_parameters())
    art = pipeline.compile_spec(spec, cc, (U, Theta, XF), d_feat,
                                train_posterior=False).artifact

    # batched forward pass over B pairs
    Ub = vb.randn(B, nlat) * 0.3
    Tb = vb.rand(B, d_param)
    t0 = time.perf_counter()
    xf = vectorize.batched_features(art, Ub, Tb)
    xf.block_until_ready()
    dt = time.perf_counter() - t0
    print(f"batched forward: B={B} in {dt*1000:.2f} ms "
          f"= {B/dt:.0f} samples/sec")

    # multi-device sharded path (no-op on single device)
    xf2 = vectorize.sharded_features(art, Ub, Tb)
    print(f"sharded output shape: {xf2.shape}")


if __name__ == "__main__":
    main()
