"""Community layer: Discord bot, openclaw agents, leaderboard.

The "openclaw agents" are LLM-backed Discord bots (one per literature model)
that each own a :class:`SurrogateTarget`.  They autonomously:
  1. generate a simulation budget for their model (via vbjax/apvbt)
  2. compile an artifact (train surrogate + posterior)
  3. submit it to the leaderboard
  4. iterate on hyperparameters to improve calibration

The leaderboard ranks artifacts by posterior quality (C2ST, SBC) and
achieved speedup.  See PLAN.md section "Community bootstrap".
"""

from .discord_bot import run_bot, OpenClawAgent
from .agents.base import AgentTask, AgentResult
from .leaderboard.scoring import score_artifact, rank_artifacts

__all__ = ["run_bot", "OpenClawAgent", "AgentTask", "AgentResult",
           "score_artifact", "rank_artifacts"]
