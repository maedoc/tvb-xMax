"""Kuramoto surrogate target."""

from .base import (SurrogateTarget, ParameterSpace, ParameterDefinition,
                   DistributionType, register)


@register("kuramoto")
class KuramotoSurrogate(SurrogateTarget):
    nlat = 16
    citation = "Kuramoto (1984) Chemical oscillations, waves, turbulence"

    def get_parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition("k", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Coupling", 0.15),
                "D": ParameterDefinition("D", "float", (0.0, 1.0),
                                         DistributionType.UNIFORM, {},
                                         "Noise", 0.4),
                "omega": ParameterDefinition("omega", "array",
                                             (0.9 * 3.14159, 1.1 * 3.14159),
                                             DistributionType.UNIFORM, {},
                                             "Natural frequency", 3.14159, hetero=True),
            },
            state_dim=1,
        )
