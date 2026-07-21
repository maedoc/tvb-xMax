"""Smoke tests for the tvb-max IR + compiler surface."""
import jax.numpy as jnp
import pytest

import tvb_max
from tvb_max import ir
from tvb_max.compiler import swap
from tvb_max.surrogates import get_surrogate, list_surrogates


def test_imports():
    assert tvb_max.__version__ == "0.1.0"
    assert hasattr(tvb_max, "IRSpec")
    assert hasattr(tvb_max, "CompiledArtifact")


def test_six_models_registered():
    names = list_surrogates()
    assert set(names) == {"hopf", "mpr", "wilson-cowan",
                          "wong-wang", "kuramoto", "fitzhugh-nagumo"}


def test_hopf_param_space():
    s = get_surrogate("hopf")
    ps = s.get_parameter_space()
    assert set(ps.parameters) == {"k", "D", "eta", "omega"}
    assert ps.parameters["k"].bounds == (0.0, 1.0)
    assert s.param_names == ("k", "D", "eta", "omega")


def test_validate_params_in_bounds():
    s = get_surrogate("hopf")
    assert s.validate_parameters({"k": 0.5, "D": 0.3}) == []


def test_validate_params_out_of_bounds():
    s = get_surrogate("hopf")
    errs = s.validate_parameters({"k": 5.0})
    assert len(errs) == 1
    assert "k" in errs[0]


def test_irspec_validate_requires_parc_for_matrix():
    spec = ir.IRSpec(model="hopf", connectivity=jnp.zeros((5, 5)),
                     connectivity_is_latent=False)
    with pytest.raises(ValueError):
        spec.validate()


def test_swap_parameters_reuses_artifact_identity():
    base = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                     connectivity_is_latent=True,
                     parameters={"k": 0.15, "D": 0.4}, target="features")
    new = swap.swap_parameters(base, k=0.25)
    assert new.parameters["k"] == 0.25
    assert new.model == base.model  # same artifact serves


def test_swap_model_changes_target():
    base = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                     connectivity_is_latent=True, parameters={})
    new = swap.swap_model(base, model="mpr")
    assert new.model == "mpr"


def test_swap_table_dispatch():
    base = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                     connectivity_is_latent=True, parameters={"k": 0.1})
    new = swap.apply_swap(base, "parameters", k=0.9)
    assert new.parameters["k"] == 0.9
    with pytest.raises(KeyError):
        swap.apply_swap(base, "bogus")
