"""Base openclaw agent: LLM-driven hyperparameter proposal + compile loop."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTask:
    model: str
    goal: str = "improve_calibration"
    current_rank: int = 999


@dataclass
class AgentResult:
    model: str
    artifact_id: str = ""
    speedup: float = 0.0
    sbc: float = 0.0
    c2st: float = 0.5
    mse: float = float("inf")
    notes: str = ""

    def summary(self) -> str:
        return (f"[{self.model}] artifact={self.artifact_id} "
                f"speedup={self.speedup:.1f}x sbc={self.sbc:.3f} "
                f"c2st={self.c2st:.3f} mse={self.mse:.2e} - {self.notes}")


class OpenClawAgentBase:
    """Base class: talks to an LLM endpoint to propose hyperparameters."""

    def __init__(self, model: str, llm_endpoint: str = "http://localhost:8081/v1"):
        self.model = model
        self.llm_endpoint = llm_endpoint

    async def propose_hyperparameters(self, task: AgentTask) -> dict:
        """Ask the LLM for the next hyperparameter set.

        The prompt encodes the current rank and goal; the LLM returns JSON.
        Parody note: a production version would use structured output +
        retrieval over the literature for the model's known good ranges.
        """
        prompt = (f"You are the openclaw agent for the {self.model} brain "
                  f"dynamics model. Current leaderboard rank: {task.current_rank}. "
                  f"Goal: {task.goal}. Propose nlat, hidden, niter, lr, "
                  f"sim_budget as JSON.")
        try:
            req = urllib.request.Request(
                f"{self.llm_endpoint}/chat/completions",
                data=json.dumps({
                    "model": "gemma4-12b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                }).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            # extract JSON from the response
            start, end = txt.find("{"), txt.rfind("}")
            return json.loads(txt[start:end + 1])
        except Exception:
            # fallback defaults
            return {"nlat": 16, "hidden": 128, "niter": 2000,
                    "lr": 3e-4, "sim_budget": 4096}

    async def compile_and_submit(self, params: dict) -> AgentResult:
        """Run the compile pipeline with the proposed params and submit.

        Real impl calls the tvb-max API /compile endpoint.  Here we return
        a placeholder result so the loop is runnable without a live API.
        """
        return AgentResult(model=self.model, notes=f"compiled with {params}")
