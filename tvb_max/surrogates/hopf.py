"""Hopf oscillator surrogate target.

Parameter space mirrors apvbt's ``HopfModel``: coupling ``k``, noise ``D``,
bifurcation ``eta`` (heterogeneous), frequency ``omega`` (heterogeneous).
The surrogate summarizes the heterogeneous arrays by mean+std so the
input dim is parcellation-invariant.
"""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("hopf")
class HopfSurrogate(SurrogateTarget):
    nlat = 16
    citation = "Deco et al. (2017) Dynamical Brain Biomarkers"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Coupling strength", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise intensity", 0.4),
                "eta": ParameterDefinition("eta", "array", (-2.0, 2.0),
                                           DistributionType.NORMAL,
                                           {"mean": 1.0, "std": 0.1},
                                           "Bifurcation parameter", 1.0, hetero=True),
                "omega": ParameterDefinition("omega", "array",
                                             (0.9 * 3.14159, 1.1 * 3.14159),
                                             DistributionType.UNIFORM, {},
                                             "Natural frequency", 3.14159, hetero=True),
            },
            state_dim=2,
        )
