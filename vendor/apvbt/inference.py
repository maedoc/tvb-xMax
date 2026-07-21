import io
import contextlib
import pickle
import numpy as np
import torch
from sbi.inference import NPE_C, NPE_A


def uniform_var(a, b):
    return (b - a) ** 2 / 12.0


def to_torch(x, device="cpu"):
    import torch

    return torch.from_numpy(np.array(x)).float().to(device=device)


def _run_sbi(theta, features, prog, algo_args):
    "actually run SBI"
    device = algo_args.get("device", "cpu")
    NPE = {
        "maf": NPE_C,
        "mdn": NPE_A,
    }[algo_args.pop("algo", "maf")]
    mu = to_torch(np.mean(theta, axis=0), device=device)
    cov = to_torch(np.cov(theta.T), device=device)
    prior = torch.distributions.MultivariateNormal(mu, cov)
    inference = NPE(prior=prior, show_progress_bars=prog, **algo_args)
    inference.append_simulations(
        to_torch(theta, device=device), to_torch(features, device=device)
    )
    inference.train()
    posterior = inference.build_posterior()
    return posterior


def run_sbi(theta, features, fname=None, prog=True, **algo_args):
    "convenience wrapper for SBI."

    if prog:
        posterior = _run_sbi(theta, features, prog, algo_args)
    else:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            posterior = _run_sbi(theta, features, prog, algo_args)

    if fname:
        with open(fname, "wb") as fd:
            pickle.dump(posterior, fd)

    return posterior


def posterior_diags(p_us, po_us, true_us):
    po_u = np.mean(po_us.numpy(), axis=0)
    po_sd = np.std(po_us.numpy(), axis=0)
    po_z = np.abs((po_u - true_us) / po_sd)
    p_var = np.var(p_us, axis=0) if hasattr(p_us, "size") else uniform_var(*p_us)
    po_shrink = np.array(1 - po_sd**2 / p_var)
    # check true in 90% ci
    q5, q95 = np.quantile(po_us, [0.05, 0.95], axis=0)
    ci90 = np.array((q5 < true_us) * (true_us < q95))
    return po_shrink, po_z, ci90
