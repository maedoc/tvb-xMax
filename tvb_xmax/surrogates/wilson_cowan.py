"""Wilson-Cowan surrogate target."""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("wilson-cowan")
class WilsonCowanSurrogate(SurrogateTarget):
    nlat = 16
    citation = "Wilson & Cowan (1972) Excitatory/inhibitory interactions"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Coupling", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise", 0.4),
                "tau_e": ParameterDefinition("tau_e", "float", (5.0, 20.0),
                                              DistributionType.UNIFORM, {},
                                              "Excitatory time constant", 10.0),
                "tau_i": ParameterDefinition("tau_i", "float", (5.0, 30.0),
                                              DistributionType.UNIFORM, {},
                                              "Inhibitory time constant", 15.0),
            },
            state_dim=2,
        )
