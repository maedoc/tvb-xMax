"""Discord bot hosting the openclaw agents.

Each agent is a Discord bot account backed by an LLM (default: the local
Gemma 4 12B server via the gemma4-server-integration skill, or any
OpenAI-compatible endpoint).  The agent listens for commands in its
model's channel and runs compile/iterate loops, posting results to the
shared #leaderboard channel.

Requires: ``pip install discord.py`` and an LLM endpoint.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

from .agents.base import AgentTask, AgentResult
from .agents.hopf_agent import HopfAgent
from .agents.mpr_agent import MPRAgent


@dataclass
class OpenClawAgent:
    """One openclaw agent bound to a model + Discord channel."""

    name: str                 # e.g. 'openclaw-hopf'
    model: str                # 'hopf' | 'mpr' | ...
    channel_id: int
    llm_endpoint: str = "http://localhost:8081/v1"  # gemma4 server
    discord_token: Optional[str] = None
    _bot: object = field(default=None, repr=False)

    async def step(self) -> AgentResult:
        """One autonomous step: generate budget -> compile -> submit.

        This is the agent's main loop body.  The LLM picks hyperparameters
        (nlat, hidden, niter, lr, sim_budget) based on the current leaderboard
        rank, then runs the compile pipeline and reports the result.
        """
        from ..agents import AGENTS
        agent_cls = AGENTS[self.model]
        agent = agent_cls(llm_endpoint=self.llm_endpoint)
        task = AgentTask(model=self.model, goal="improve_calibration")
        # ask the LLM for hyperparameters (parody: a real prompt would go here)
        params = await agent.propose_hyperparameters(task)
        result = await agent.compile_and_submit(params)
        return result

    async def run_forever(self):
        while True:
            try:
                res = await self.step()
                await self._post_to_discord(res)
            except Exception as e:
                print(f"[{self.name}] step failed: {e}")
            await asyncio.sleep(int(os.environ.get("TVBXMAX_AGENT_INTERVAL", "300")))

    async def _post_to_discord(self, result: AgentResult):
        if self._bot is None:
            print(f"[{self.name}] {result.summary()}")
            return
        # real impl: channel = self._bot.get_channel(self.channel_id)
        #           await channel.send(result.summary())


def run_bot(token: str, agents: list[OpenClawAgent]):
    """Start the Discord bot hosting the given agents.

    Each agent gets its own background task running :meth:`run_forever`.
    """
    try:
        import discord
        from discord.ext import commands
    except ImportError:
        print("discord.py not installed; running agents headless")
        asyncio.run(_run_headless(agents))
        return

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!tvb ", intents=intents)

    @bot.event
    async def on_ready():
        for a in agents:
            a._bot = bot
            asyncio.create_task(a.run_forever())
        print(f"tvb-xMax bot online with {len(agents)} agents")

    @bot.command()
    async def leaderboard(ctx):
        from .leaderboard.scoring import rank_artifacts
        # in real impl, pull from the API; here just echo
        await ctx.send("see /api/v1/leaderboard")

    bot.run(token)


async def _run_headless(agents):
    await asyncio.gather(*[a.run_forever() for a in agents])
