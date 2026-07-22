# Extracted from apvbt (https://github.com/ins-amu/apvbt) — see vendor/README.md
"""Simulation sampling functions for SBI training.

This module provides functions to sample parameters and run brain dynamics simulations
for training simulation-based inference (SBI) models.
"""

import numpy as np
import tqdm
from collections.abc import Callable
from .inference import to_torch, run_sbi, posterior_diags


def sample_subj_model(w, model, num_batch, batch_size, use_pmap=True, prog=True):
    """Sample simulations for a single subject (fixed connectome).

    Args:
        w: Connectome matrix (n_roi, n_roi)
        model: Dynamics model function
        num_batch: Number of batches to sample
        batch_size: Batch size
        use_pmap: Whether to use pmap for parallelization
        prog: Whether to show progress bar

    Returns:
        Tuple of (thetas, xfs) where:
        - thetas: Parameter samples (n_total, 2) with [k, D]
        - xfs: Feature samples (n_total, n_features)
    """
    import jax, jax.numpy as jp, vbjax as vb
    w = w + jp.zeros((batch_size, 1, 1))
    assert w.ndim == 3
    parm_keys = jax.random.split(vb.key, (2, num_batch))
    thetas, xfs = [], []
    iters = (tqdm.trange if prog else range)(num_batch)
    for i in iters:
        # TODO generalize priors
        k = 0.1 + vb.rand(batch_size, key=parm_keys[0, i])*0.2
        D = 0.2 + vb.rand(batch_size, key=parm_keys[1, i])*0.2
        xf = model(w, k, D, use_pmap=use_pmap)
        thetas.append(jp.c_[k, D])
        xfs.append(xf)
    thetas = jp.array(thetas).reshape(-1, thetas[0].shape[1])
    xfs = jp.array(xfs).reshape(-1, xfs[0].shape[1])
    return thetas, xfs


def sample_model(xc, model, mvn, parc, num_batch, batch_size, prog=True, use_pmap=True):
    """Sample simulations from cohort distribution (sampled connectomes).

    Args:
        xc: XCode instance
        model: Dynamics model function
        mvn: MvNorm distribution in latent space
        parc: Parcellation name
        num_batch: Number of batches to sample
        batch_size: Batch size per batch
        prog: Whether to show progress bar
        use_pmap: Whether to use pmap for parallelization

    Returns:
        Tuple of (thetas, xfs) where:
        - thetas: Parameter samples (n_total, n_latent+2) with [latent_codes, k, D]
        - xfs: Feature samples (n_total, n_features)
    """
    import jax, jax.numpy as jp, vbjax as vb

    parm_keys = jax.random.split(vb.key, (2, num_batch))
    thetas, xfs = [], []
    iters = (tqdm.trange if prog else range)(num_batch)
    for i in iters:
        # TODO generalize priors
        u = mvn.sample(batch_size)
        k = 0.1 + vb.rand(batch_size, key=parm_keys[0, i])*0.2
        D = 0.2 + vb.rand(batch_size, key=parm_keys[1, i])*0.2
        w = xc.decode_conn(parc, u)
        xf = model(w, k, D, use_pmap=use_pmap)
        thetas.append(jp.concat([u, jp.vstack([k, D]).T], axis=1))
        xfs.append(xf)
    thetas = jp.array(thetas).reshape(-1, thetas[0].shape[1])
    xfs = jp.array(xfs).reshape(-1, xfs[0].shape[1])
    return thetas, xfs


def bench_cohort_model(
    xc,
    model: Callable,
    parc: str = '079-Shen2013',
    arch: int = 8,
    num_batch: int = 32,
    batch_size: int = 128,
    use_pmap=True
):
    """Benchmark cohort-level SBI (in-sample).

    Args:
        xc: XCode instance
        model: Dynamics model function
        parc: Parcellation name
        arch: Latent dimension
        num_batch: Number of batches for training
        batch_size: Batch size
        use_pmap: Whether to use pmap

    Returns:
        Tuple of (mean_shrinkage, mean_zscore)
    """
    mvn = xc.calc_mvn(arch)
    thetas, xfs = sample_model(
        xc, model, mvn, parc, num_batch, batch_size,
        use_pmap=use_pmap)
    posterior = run_sbi(thetas, xfs)
    thetas_hat = posterior.sample_batched((200,), x=to_torch(xfs[:batch_size]))
    ps, pz, ci = posterior_diags(thetas, thetas_hat, thetas[:batch_size])
    return ps.mean().item(), pz.mean()


def bench_model(xc, model, parc='079-Shen2013',
                arch=8, num_batch=32, batch_size=128,
                do_subjets=False, num_postcd=None, inflate=1,
                use_pmap=True, return_everything=False,
                prog=True
                ):
    """Comprehensive benchmark comparing cohort vs subject-level SBI.

    Args:
        xc: XCode instance
        model: Dynamics model function
        parc: Parcellation name
        arch: Latent dimension
        num_batch: Number of batches for training
        batch_size: Batch size
        do_subjets: Whether to also run subject-level SBI
        num_postcd: Number of posterior samples (default: same as training)
        inflate: Inflation factor for covariance
        use_pmap: Whether to use pmap
        return_everything: If True, return locals() dict instead
        prog: Whether to show progress bars

    Returns:
        Tuple of (diags_cd, diags_subj) or locals() dict if return_everything=True
        - diags_cd: Cohort-level diagnostics
        - diags_subj: Subject-level diagnostics (or None if do_subjets=False)
    """
    import jax, jax.numpy as jp, vbjax as vb

    # setup ground truth
    w = xc.get_conn(parc)  # (n_test, n_parc, n_parc)
    u = xc.encode_conn(arch, parc)
    k = 0.1 + vb.rand(len(w))*0.2
    D = 0.2 + vb.rand(len(w))*0.2
    theta = jp.concat([u, jp.c_[k, D]], axis=1)
    xf = model(w, k, D, use_pmap=False)  # (n_test, n_parc)

    # build & apply cohort sbi
    mvn = xc.calc_mvn(arch)
    mvn.u_cov = mvn.u_cov * inflate  # inflate a bit
    theta_cdhat, xf_cdhat = sample_model(xc, model, mvn, parc, num_batch, batch_size,
                                         prog=prog, use_pmap=use_pmap)
    posterior_cd = run_sbi(theta_cdhat, xf_cdhat, prog=prog)
    po_theta_cdhat = posterior_cd.sample_batched(
        (num_postcd or theta_cdhat.shape[0],), x=np.array(xf), show_progress_bars=prog)
    # (nsamp, xf.shape[0], arch+2)
    diags_cd = posterior_diags(
        theta_cdhat[:, arch:], po_theta_cdhat[..., arch:], theta[:, arch:])

    if do_subjets:
        # sbi subject
        diags_subj = []
        for it in tqdm.trange(len(xf)):  # 74 subjects
            theta_hat, xf_hat = sample_subj_model(
                w[it], model, num_batch, batch_size, prog=prog, use_pmap=use_pmap)
            posterior = run_sbi(theta_hat, xf_hat, prog=prog)
            po_theta_hat = posterior.sample(
                (theta_hat.shape[0],), x=np.array(xf[it]), show_progress_bars=prog)
            diags_it = posterior_diags(
                theta_hat, po_theta_hat, theta[it, arch:])
            diags_subj.append(diags_it)
    else:
        diags_subj = None

    if return_everything:
        return locals()
    # TODO compare stats of diags from sbi-subject to sbi-cohort
    return diags_cd, diags_subj
