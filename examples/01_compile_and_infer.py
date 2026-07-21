"""Example: compile a Hopf surrogate and run the fast path.

This is the "hello world" of tvb-max.  It shows the full compiler pipeline:
frontend -> lower -> optimize -> codegen -> posterior, then a fast inference
run that replaces the SDE simulation with a single forward pass.

Run: python examples/01_compile_and_infer.py
"""
import jax.numpy as jnp
import vbjax as vb

import tvb_max
from tvb_max.compiler import ir, pipeline


def make_fake_sim_budget(nlat, d_param, d_feat, n=4096, key=vb.key):
    """Stand-in for apvbt's sample_model output.

    In real use, you'd call ``apvbt.sample_model(xc, model, mvn, parc, ...)``
    to get (theta, xf) from actual vbjax simulations, then encode the
    connectomes to latents via the cross-coder.  Here we synthesize a
    toy mapping so the example runs without the 6GB HCP/1KB dataset.
    """
    U = vb.randn(n, nlat, key=key) * 0.3
    Theta = vb.rand(n, d_param, key=key)
    # toy "simulation": features = nonlinear function of (u, theta)
    XF = jnp.concatenate([U[:, :d_feat], Theta[:, :d_feat]], axis=1)
    return U, Theta, XF


def main():
    nlat, d_feat = 16, 8
    # 1. build a cross-coder (here a trivial single-view one for the example)
    cc = vb.CrossCoder(variational=False)
    import numpy as np
    cc.add_view(np.random.randn(20, nlat), "079-Shen2013", normalize="center")
    cc.tts = 10
    cc.train(nlat=nlat, niter=50, show_progress=False) if hasattr(cc, "show_progress") else cc.train(nlat=nlat, niter=50)

    # 2. one-time simulation budget (the only place real sims happen)
    surr = tvb_max.surrogates.get_surrogate("hopf")
    d_param = len(surr.param_names)
    sim_pairs = make_fake_sim_budget(nlat, d_param, d_feat)

    # 3. COMPILE: frontend -> lower -> optimize -> codegen -> posterior
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(nlat),       # latent passed directly
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
        target="posterior",
        n_posterior=1000,
    )
    report = pipeline.compile_spec(spec, cc, sim_pairs, d_feat,
                                   train_posterior=False)  # skip sbi for speed
    art = report.artifact
    print(f"compiled {art.model}/{art.feature}: mse={art.surrogate_mse:.3e} "
          f"speedup~{report.speedup_vs_sim:.0f}x  stages={list(report.stages)}")

    # 4. RUN (fast path): no simulation, just a forward pass
    run_spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(nlat),
        connectivity_is_latent=True,
        parameters={"k": 0.2, "D": 0.3},
        target="features",
    )
    out = pipeline.run(art, run_spec, cc)
    print(f"features = {out['features']}")


if __name__ == "__main__":
    main()
