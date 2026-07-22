"""Tests for the swap module — the "four free swaps" feature.

Covers parcellation, parameters, model, and feature swaps, including
dispatch, field preservation, validation, and integration with the
compile pipeline.
"""

import jax.numpy as jnp
import pytest

from tvb_xmax import ir
from tvb_xmax.compiler import swap, frontend, pipeline
from tests.conftest import NLAT, D_PARAM, D_FEAT, N_REGIONS


def test_swap_parcellation_new_spec():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
    )
    new_spec = swap.swap_parcellation(
        base, connectivity=jnp.eye(10), parcellation="new_parc"
    )
    assert new_spec.connectivity_is_latent is False
    assert new_spec.parcellation == "new_parc"
    assert new_spec.model == base.model
    assert new_spec.parameters == base.parameters
    assert new_spec.feature == base.feature


def test_swap_parameters_validates_bounds():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        target="features",
    )
    new_spec = swap.swap_parameters(base, k=0.9, D=0.1)
    assert new_spec.parameters["k"] == 0.9
    assert new_spec.parameters["D"] == 0.1
    parsed = frontend.parse(new_spec)
    assert parsed is new_spec


def test_swap_parameters_partial_update():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
    )
    new_spec = swap.swap_parameters(base, k=0.9)
    assert new_spec.parameters["k"] == 0.9
    assert new_spec.parameters["D"] == 0.4


def test_swap_model_resets_parameters():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
    )
    new_spec = swap.swap_model(base, model="mpr")
    assert new_spec.model == "mpr"
    assert new_spec.parameters == {}
    assert new_spec.connectivity is base.connectivity


def test_swap_features_new_feature():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
    )
    new_spec = swap.swap_features(base, feature="fc")
    assert new_spec.feature == "fc"
    assert new_spec.model == base.model
    assert new_spec.parameters == base.parameters


def test_apply_swap_all_kinds():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
    )
    s1 = swap.apply_swap(base, ir.SwapKind.PARCELLATION,
                         connectivity=jnp.eye(10), parcellation="new_parc")
    assert isinstance(s1, ir.IRSpec)
    assert s1.parcellation == "new_parc"
    assert s1.connectivity_is_latent is False

    s2 = swap.apply_swap(base, ir.SwapKind.PARAMETERS, k=0.99)
    assert isinstance(s2, ir.IRSpec)
    assert s2.parameters["k"] == 0.99

    s3 = swap.apply_swap(base, ir.SwapKind.MODEL, model="mpr")
    assert isinstance(s3, ir.IRSpec)
    assert s3.model == "mpr"
    assert s3.parameters == {}

    s4 = swap.apply_swap(base, ir.SwapKind.FEATURES, feature="bold")
    assert isinstance(s4, ir.IRSpec)
    assert s4.feature == "bold"

    with pytest.raises(KeyError):
        swap.apply_swap(base, "bogus")


def test_swap_preserves_unrelated_fields():
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parcellation=None,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
        target="posterior",
        n_posterior=2000,
        seed=7,
    )
    new = swap.swap_parameters(base, k=0.8)
    assert new.model == base.model
    assert new.connectivity is base.connectivity
    assert new.connectivity_is_latent == base.connectivity_is_latent
    assert new.parcellation == base.parcellation
    assert new.feature == base.feature
    assert new.target == base.target
    assert new.n_posterior == base.n_posterior
    assert new.seed == base.seed


def test_swap_parcellation_to_matrix_then_lower(
    toy_crosscoder, toy_sim_budget, toy_mvn
):
    base = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.15, "D": 0.4},
        feature="var",
        target="features",
    )
    report = pipeline.compile_spec(
        base, toy_crosscoder, toy_sim_budget, D_FEAT,
        mvn=toy_mvn, train_posterior=False, niter=0,
    )
    artifact = report.artifact

    swapped = swap.swap_parcellation(
        base, connectivity=jnp.eye(N_REGIONS), parcellation="parc_a"
    )
    result = pipeline.run(artifact, swapped, toy_crosscoder, mvn=toy_mvn)
    assert "features" in result
    assert result["features"].shape[-1] == D_FEAT
