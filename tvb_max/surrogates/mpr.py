"""Multi-Population Rate (MPR) surrogate target.

Parameter space mirrors apvbt's ``MPRModel`` and vbjax's ``mpr_dfun``.
"""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("mpr")
class MPRSurrogate(SurrogateTarget):
    nlat = 16
    citation = "Hansen et al. (2015) Functional connectivity dynamics modeling"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Global coupling", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise intensity", 0.4),
                "J": ParameterDefinition("J", "array", (0.5, 2.0),
                                          DistributionType.NORMAL,
                                          {"mean": 1.0, "std": 0.1},
                                          "Local gain", 1.0, hetero=True),
                "w": ParameterDefinition("w", "array", (0.0, 1.0),
                                          DistributionType.UNIFORM, {},
                                          "Excitatory weight", 0.5, hetero=True),
            },
            state_dim=2,
        )
