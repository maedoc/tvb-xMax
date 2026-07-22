"""Tests for the compiler frontend parser."""
import jax.numpy as jnp
import pytest

from tvb_xmax import ir
from tvb_xmax.compiler import frontend
from tvb_xmax.surrogates import list_surrogates


def test_parse_valid_spec(toy_nlat):
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        parameters={"k": 0.5, "D": 0.3},
        target="features",
    )
    result = frontend.parse(spec)
    assert result is spec


def test_parse_unknown_model(toy_nlat):
    spec = ir.IRSpec(
        model="nonexistent_model",
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        parameters={},
    )
    with pytest.raises(KeyError, match=r"no surrogate compiled for model"):
        frontend.parse(spec)


def test_parse_out_of_bounds_param(toy_nlat):
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        parameters={"k": 5.0},
    )
    with pytest.raises(ValueError, match=r"k"):
        frontend.parse(spec)


def test_parse_missing_parcellation():
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(5),
        connectivity_is_latent=False,
        parcellation=None,
    )
    with pytest.raises(ValueError, match=r"parcellation"):
        frontend.parse(spec)


@pytest.mark.parametrize("model", list_surrogates())
def test_parse_all_models_accept_defaults(model, toy_nlat):
    spec = ir.IRSpec(
        model=model,
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        parameters={},
    )
    result = frontend.parse(spec)
    assert result is spec


def test_parse_invalid_target(toy_nlat):
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        target="invalid",
    )
    with pytest.raises(ValueError, match=r"target"):
        frontend.parse(spec)


def test_parse_preserves_fields(toy_nlat):
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(toy_nlat),
        connectivity_is_latent=True,
        parameters={"k": 0.5, "D": 0.3},
        feature="fc",
        target="features",
        n_posterior=500,
        seed=99,
    )
    result = frontend.parse(spec)
    assert result.model == "hopf"
    assert result.feature == "fc"
    assert result.target == "features"
    assert result.n_posterior == 500
    assert result.seed == 99
