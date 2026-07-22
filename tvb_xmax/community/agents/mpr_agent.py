"""MPR openclaw agent - owns the 'mpr' surrogate target."""
from .base import OpenClawAgentBase


class MPRAgent(OpenClawAgentBase):
    def __init__(self, llm_endpoint="http://localhost:8081/v1"):
        super().__init__("mpr", llm_endpoint)
