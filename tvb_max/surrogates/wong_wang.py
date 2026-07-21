"""Wong-Wang surrogate target."""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("wong-wang")
class WongWangSurrogate(SurrogateTarget):
    nlat = 16
    citation = "Wong & Wang (2006) Recurrent excitatory/inhibitory network"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Coupling", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise", 0.4),
                "w": ParameterDefinition("w", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Local recurrence", 0.9),
                "I0": ParameterDefinition("I0", "float", (0.0, 0.5),
                                          DistributionType.UNIFORM, {},
                                          "External input", 0.3),
            },
            state_dim=2,
        )
