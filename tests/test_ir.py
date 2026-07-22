"""Smoke tests for the tvb-xMax IR + compiler surface."""
import jax
import jax.numpy as jnp
import pytest

import tvb_xmax
from tvb_xmax import ir
from tvb_xmax.compiler import swap
from tvb_xmax.surrogates import get_surrogate, list_surrogates
from conftest import NLAT, D_PARAM, D_FEAT


def test_imports():
    assert tvb_xmax.__version__ == "0.1.0"
    assert hasattr(tvb_xmax, "IRSpec")
    assert hasattr(tvb_xmax, "CompiledArtifact")


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


# ---------------------------------------------------------------------------
# Edge-case coverage for ir.py (T2.10)
# ---------------------------------------------------------------------------


def test_irspec_validate_connectivity_none():
    """Coverage: IRSpec with None connectivity raises ValueError."""
    spec = ir.IRSpec(model="hopf", connectivity=None)
    with pytest.raises(ValueError, match="connectivity is required"):
        spec.validate()


def test_irspec_validate_unknown_target():
    """Coverage: IRSpec with unknown target raises ValueError."""
    spec = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                     connectivity_is_latent=True, target="bogus")
    with pytest.raises(ValueError, match="unknown target"):
        spec.validate()


def test_irprogram_d_param_nlat(toy_nlat, toy_d_param):
    """Coverage: IRProgram.d_param and nlat properties."""
    prog = ir.IRProgram(
        model="hopf",
        u=jnp.zeros(toy_nlat),
        theta=jnp.zeros(toy_d_param),
        param_names=("k", "D", "eta", "omega"),
        feature="var",
        target="features",
        n_posterior=100,
        seed=42,
    )
    assert prog.d_param == toy_d_param
    assert prog.nlat == toy_nlat
    assert prog.theta.shape[0] == prog.d_param


def test_compiled_artifact_call(toy_sim_budget):
    """Coverage: CompiledArtifact.__call__."""
    from tvb_xmax.compiler import codegen
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=50)
    u = jnp.zeros(NLAT)
    theta = jnp.zeros(D_PARAM)
    features = artifact(u, theta)
    assert features.shape == (D_FEAT,)


def test_compiled_artifact_getstate(toy_sim_budget):
    """Coverage: __getstate__ drops callables."""
    from tvb_xmax.compiler import codegen
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=50)
    state = artifact.__getstate__()
    assert state["surrogate_apply"] is None
    assert state["trunk_apply"] is None
    assert state["head_apply"] is None
    assert state["posterior_sample"] is None
    assert state["trunk_params"] is not None
    assert state["head_params"] is not None


def test_compiled_artifact_setstate_restores(toy_sim_budget):
    """Coverage: __setstate__ rebuilds callables from params."""
    from tvb_xmax.compiler import codegen
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=50)
    state = artifact.__getstate__()
    restored = ir.CompiledArtifact.__new__(ir.CompiledArtifact)
    restored.__setstate__(state)
    assert restored.surrogate_apply is not None
    assert restored.trunk_apply is not None
    assert restored.head_apply is not None
    # Verify it produces the same output
    u = jnp.zeros(NLAT)
    theta = jnp.zeros(D_PARAM)
    original = artifact(u, theta)
    restored_out = restored(u, theta)
    assert jnp.allclose(original, restored_out, atol=1e-5)
