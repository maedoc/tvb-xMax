"""FitzHugh-Nagumo surrogate target."""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("fitzhugh-nagumo")
class FitzHughNagumoSurrogate(SurrogateTarget):
    nlat = 16
    citation = "FitzHugh (1961) Impulses in nerve membranes"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Coupling", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise", 0.4),
                "a": ParameterDefinition("a", "float", (-1.0, 1.0),
                                          DistributionType.UNIFORM, {},
                                          "Recovery parameter", 0.7),
                "b": ParameterDefinition("b", "float", (0.0, 2.0),
                                         DistributionType.UNIFORM, {},
                                         "Recovery rate", 0.8),
            },
            state_dim=2,
        )
