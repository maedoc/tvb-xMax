"""Frontend: parse + validate an :class:`IRSpec` into a resolved program.

Resolves the model name against :mod:`tvb_max.surrogates`, resolves the
feature extractor, and validates the parameter dict against the model's
declared :class:`ParameterSpace`.  No tensors are touched yet.
"""

from __future__ import annotations

from ..ir import IRSpec
from ..surrogates import get_surrogate, list_surrogates


def parse(spec: IRSpec) -> IRSpec:
    """Validate and normalize a source spec.

    Raises:
        KeyError: if ``spec.model`` is not a registered surrogate target.
        ValueError: if parameters are out of bounds or required ones missing.
    """
    spec.validate()
    if spec.model not in list_surrogates():
        raise KeyError(
            f"no surrogate compiled for model {spec.model!r}; "
            f"available: {list_surrogates()}"
        )
    surr = get_surrogate(spec.model)
    # validate the user-supplied parameters against the model's parameter space
    errors = surr.validate_parameters(spec.parameters)
    if errors:
        raise ValueError("parameter validation failed: " + "; ".join(errors))
    return spec
