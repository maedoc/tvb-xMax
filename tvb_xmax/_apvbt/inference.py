# Extracted from apvbt (https://github.com/ins-amu/apvbt) — see vendor/README.md
import pickle
import numpy as np

from ..cde import MDNEstimator


def uniform_var(a, b):
    return (b - a) ** 2 / 12.0


def to_torch(x, device="cpu"):
    """Compatibility shim retained for old apvbt callers.

    tvb-xMax no longer uses Torch; callers receive a float32 NumPy array.
    """
    return np.asarray(x, dtype=np.float32)


def run_sbi(theta, features, fname=None, prog=True, **algo_args):
    """Compatibility wrapper using tvb-xMax's in-tree conditional MDN."""
    algo = algo_args.pop("algo", "mdn")
    if algo != "mdn":
        raise ValueError("only the in-tree 'mdn' posterior is available")
    theta, features = np.asarray(theta), np.asarray(features)
    posterior = MDNEstimator(theta.shape[-1], features.shape[-1], **algo_args)
    posterior.train(theta, features, prog=prog)

    if fname:
        with open(fname, "wb") as fd:
            pickle.dump(posterior, fd)

    return posterior


def posterior_diags(p_us, po_us, true_us):
    po_us = np.asarray(po_us)
    po_u = np.mean(po_us, axis=0)
    po_sd = np.std(po_us, axis=0)
    po_z = np.abs((po_u - true_us) / po_sd)
    p_var = np.var(p_us, axis=0) if hasattr(p_us, "size") else uniform_var(*p_us)
    po_shrink = np.array(1 - po_sd**2 / p_var)
    # check true in 90% ci
    q5, q95 = np.quantile(po_us, [0.05, 0.95], axis=0)
    ci90 = np.array((q5 < true_us) * (true_us < q95))
    return po_shrink, po_z, ci90
