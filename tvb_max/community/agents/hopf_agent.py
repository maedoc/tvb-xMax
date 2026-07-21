"""Hopf openclaw agent - owns the 'hopf' surrogate target."""
from .base import OpenClawAgentBase


class HopfAgent(OpenClawAgentBase):
    def __init__(self, llm_endpoint="http://localhost:8081/v1"):
        super().__init__("hopf", llm_endpoint)
