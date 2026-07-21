"""Agent registry: model name -> agent class."""
from .hopf_agent import HopfAgent
from .mpr_agent import MPRAgent

AGENTS = {
    "hopf": HopfAgent,
    "mpr": MPRAgent,
}
