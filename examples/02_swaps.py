"""Example: the four free swaps.

Shows that once an artifact is compiled, you can swap parcellation,
parameters, model, and features without re-simulating.  Parcellation +
parameter swaps reuse the exact same artifact; model + feature swaps
select a different pre-compiled artifact from the registry.

Run: python examples/02_swaps.py
"""
import tvb_xmax  # noqa: F401  (puts vendored vbjax/apvbt on sys.path)
import jax.numpy as jnp
import vbjax as vb

from tvb_xmax.compiler import ir, pipeline, swap


def compile_toy(model, nlat=16, d_feat=8):
    """Compile a tiny artifact from synthetic data (see example 01)."""
    cc = vb.CrossCoder(variational=False)
    import numpy as np
    cc.add_view(np.random.randn(20, nlat), "079-Shen2013", normalize="center")
    cc.tts = 10
    cc.train(nlat=nlat, niter=50)
    surr = tvb_xmax.surrogates.get_surrogate(model)
    d_param = len(surr.param_names)
    U = vb.randn(1024, nlat) * 0.3
    Theta = vb.rand(1024, d_param)
    XF = jnp.tanh(U[:, :d_feat] + 0.3 * Theta[:, :1])
    spec = ir.IRSpec(model=model, connectivity=jnp.zeros(nlat),
                     connectivity_is_latent=True, parameters=surr.default_parameters())
    return pipeline.compile_spec(spec, cc, (U, Theta, XF), d_feat,
                                 train_posterior=False).artifact, cc


def main():
    art_hopf, cc = compile_toy("hopf")

    base = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                     connectivity_is_latent=True,
                     parameters={"k": 0.15, "D": 0.4}, target="features")

    # --- swap 1: parameters (free, same artifact) ---
    s_params = swap.swap_parameters(base, k=0.25, D=0.1)
    out = pipeline.run(art_hopf, s_params, cc)
    print(f"swap params:  features={out['features']}")

    # --- swap 2: parcellation (free, same artifact) ---
    # in real use you'd pass a real (n,n) matrix in a different parcellation;
    # the cross-coder encodes it to the same latent space.
    s_parc = swap.swap_parcellation(base, connectivity=jnp.zeros(16),
                                   parcellation="150-Destrieux")
    s_parc = ir.IRSpec(**{**s_parc.__dict__, "connectivity_is_latent": True})
    out = pipeline.run(art_hopf, s_parc, cc)
    print(f"swap parc:    features={out['features']}")

    # --- swap 3: model (needs a different compiled artifact) ---
    art_mpr, _ = compile_toy("mpr")
    s_model = swap.swap_model(base, model="mpr")
    out = pipeline.run(art_mpr, s_model, cc)
    print(f"swap model:   features={out['features']}")

    # --- swap 4: features (needs an artifact compiled for that feature) ---
    s_feat = swap.swap_features(base, feature="fc")
    # art_hopf was compiled for 'var'; a real 'fc' artifact would be separate
    print(f"swap features: (would use a var->fc artifact, spec={s_feat.feature})")


if __name__ == "__main__":
    main()
