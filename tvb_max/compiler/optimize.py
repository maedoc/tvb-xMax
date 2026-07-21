"""Optimize: IR transforms on the lowered program.

These are cheap, pure reparameterizations that make the surrogate trunk
better conditioned and enable the free swaps.  None of them touch the
(expensive) simulation; they only reshape the IR tensors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ir import IRProgram


def condition_latent(prog: IRProgram, mvn=None) -> IRProgram:
    """Whiten the latent ``u`` against the cohort MVN.

    If ``mvn`` (a ``vbjax.MvNorm`` / ``apvbt.MvNorm``) is given, subtract the
    cohort mean and divide by the per-dim std so the surrogate sees a
    roughly N(0,I) input.  This is what makes a *new* subject's latent
    comparable to the training distribution without re-simulating.
    """
    if mvn is None:
        return prog
    mu = jnp.asarray(mvn.u_mean if hasattr(mvn, "u_mean") else mvn.mean)
    cov = jnp.asarray(mvn.u_cov if hasattr(mvn, "u_cov") else mvn.cov)
    std = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-6, None))
    u = (prog.u - mu) / std
    return IRProgram(**{**prog.__dict__, "u": u})


def reparam_heterogeneous(prog: IRProgram) -> IRProgram:
    """Model-specific reparameterization hook.

    For Hopf, heterogeneous ``eta``/``omega`` arrays are summarized by their
    mean+std so the surrogate input stays fixed-dim regardless of parcellation
    size.  This is the key transform that lets a *single* compiled artifact
    serve 079-Shen2013 and 294-Julich-Brain alike.
    """
    # placeholder: real impl inspects prog.model and folds any vector params
    # into summary stats appended to theta.  For now a no-op pass-through.
    return prog


def optimize(prog: IRProgram, mvn=None) -> IRProgram:
    """Run the full IR optimization pass pipeline."""
    prog = condition_latent(prog, mvn)
    prog = reparam_heterogeneous(prog)
    return prog
