"""Tests for all 6 surrogate targets (parameter spaces, validation, edge cases)."""

import jax.numpy as jnp
import pytest

from tvb_xmax.surrogates import get_surrogate, list_surrogates


# Expected parameter counts per model {name: count}
EXPECTED_COUNTS = {
    "hopf": 4,
    "mpr": 4,
    "wilson-cowan": 4,
    "wong-wang": 4,
    "kuramoto": 3,
    "fitzhugh-nagumo": 4,
}

# A float param known to be universally present and out-of-bounds at 5.0
OUT_OF_BOUNDS_PARAM = "k"
OUT_OF_BOUNDS_VAL = 5.0


def test_all_models_are_known():
    expected = set(EXPECTED_COUNTS)
    assert set(list_surrogates()) == expected


@pytest.mark.parametrize("model", list_surrogates())
def test_all_models_have_required_params(model):
    s = get_surrogate(model)
    ps = s.get_parameter_space()
    for required in ("k", "D"):
        assert required in ps.parameters, f"{model} missing {required}"


@pytest.mark.parametrize(
    "model,expected_count",
    [(m, EXPECTED_COUNTS[m]) for m in list_surrogates()],
)
def test_model_param_count(model, expected_count):
    s = get_surrogate(model)
    ps = s.get_parameter_space()
    assert len(ps.parameters) == expected_count


@pytest.mark.parametrize("model", list_surrogates())
def test_validate_defaults_pass(model):
    s = get_surrogate(model)
    assert s.validate_parameters({}) == []


@pytest.mark.parametrize("model", list_surrogates())
def test_validate_out_of_bounds(model):
    s = get_surrogate(model)
    errs = s.validate_parameters({OUT_OF_BOUNDS_PARAM: OUT_OF_BOUNDS_VAL})
    assert len(errs) >= 1
    assert OUT_OF_BOUNDS_PARAM in errs[0]


@pytest.mark.parametrize("model", list_surrogates())
def test_nlat_all_same(model):
    s = get_surrogate(model)
    assert s.nlat == 16


@pytest.mark.parametrize("model", list_surrogates())
def test_default_parameters_are_in_bounds(model):
    s = get_surrogate(model)
    ps = s.get_parameter_space()
    defaults = s.default_parameters()
    for name, pdef in ps.parameters.items():
        assert name in defaults, f"{model}: {name} missing from defaults"
        val = defaults[name]
        lo, hi = pdef.bounds
        assert lo <= val <= hi, (
            f"{model}.{name}={val} not in [{lo}, {hi}]"
        )


@pytest.mark.parametrize("model", list_surrogates())
def test_model_citations(model):
    s = get_surrogate(model)
    assert isinstance(s.citation, str) and len(s.citation) > 0
