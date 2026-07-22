# tvb-xMax Agent Guide

This document provides guidance for AI agents (human or LLM) working on tvb-xMax. It mirrors the structure of `apvbt/AGENTS.md` so anyone familiar with the parent project can navigate this one.

## Overview

tvb-xMax is an "advanced AI math compiler" for virtual brain simulation — a parody-flavored but real system that wraps `apvbt` + `vbjax` and replaces the SDE simulation with a trained neural surrogate at inference time. The cross-coder latent is the IR; swapping parcellation/parameters/model/features is free.

## Quick Reference

### Key Documents
- **PLAN.md** — complete design (compiler architecture + community bootstrap)
- **README.md** — user-facing overview + quick start
- **AGENTS.md** — this file

### Agent Workflow (session iteration)
1. **Awareness**: read `PLAN.md` (Part I then Part II), `git log -n3`, `git status`.
2. **Action**: pick a phase from PLAN.md §8 (compiler) or §15 (bootstrap), implement it, run tests, commit.
3. **Reflection**: update PLAN.md open-questions table if something changed.

## Project Context

### What tvb-xMax does
Compiles a one-time simulation budget (from `apvbt.sample_model`) into a neural surrogate + amortized posterior. Every subsequent inference is a GPU forward pass instead of an SDE integration.

### Architecture
```
tvb_xmax/
├── ir.py                  # IR dataclasses (IRSpec, IRProgram, CompiledArtifact)
├── compiler/              # 8-stage pipeline (frontend→lower→optimize→codegen→vectorize→posterior→pipeline→swap)
├── surrogates/            # one SurrogateTarget per literature model (6 scaffolded)
├── api/                   # FastAPI + auth + ratelimit
└── community/             # Discord bot + openclaw agents + leaderboard
```

### Extension Points
1. **New literature model** — add one file in `surrogates/` with `@register("name")` declaring the `ParameterSpace`. Used by the frontend; no core changes.
2. **New feature head** — compile an artifact for a new `feature` string; the trunk is reusable.
3. **New agent** — add one file in `community/agents/` subclassing `OpenClawAgentBase`.

## Agent Skills

### Running tests
```bash
pip install pytest jax jaxlib numpy vbjax
pytest                            # all tests
pytest -m "not slow"              # skip slow
pytest tests/test_ir.py -v        # specific
```

### Git workflow
Same conventions as apvbt: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`. Single tag line + paragraph body.

### Adding a surrogate target (new literature model)
1. Create `tvb_xmax/surrogates/<model>.py`
2. Subclass `SurrogateTarget`, decorate with `@register("<model>")`
3. Implement `get_parameter_space()` mirroring the apvbt `DynamicsModel` parameter space
4. Add to `surrogates/__init__.py` exports
5. Add a test in `tests/test_surrogates.py`
6. (Optional) add an openclaw agent in `community/agents/<model>_agent.py`
7. Commit with `feat: add <model> surrogate target`

### Adding an openclaw agent
1. Create `tvb_xmax/community/agents/<model>_agent.py`
2. Subclass `OpenClawAgentBase`, set `self.model = "<model>"`
3. Register in `community/agents/__init__.py` `AGENTS` dict
4. Add a `OpenClawAgent(...)` entry in `community/discord_bot.run_bot` callers
5. Commit with `feat: add openclaw-<model> agent`

### Working with the cross-coder IR
- The IR latent `u` is produced by `vbjax.CrossCoder.encode(nlat, parc)` or `apvbt.XCode.encode_conn(arch, parc)`.
- `compiler/lower.py` handles both raw-matrix and pre-encoded-latent inputs.
- `optimize.condition_latent` whitens `u` against `CrossCoder.calc_mvn(arch)`.

### Debugging the compiler
Each stage is pure and returns a new IR object, so you can introspect:
```python
from tvb_xmax.compiler import frontend, lower, optimize
spec = frontend.parse(my_spec)
prog = lower.lower(spec, crosscoder)
prog = optimize.optimize(prog, mvn)
print(prog.u, prog.theta, prog.param_names)
```

## Project Conventions
- PEP 8, 4-space indent, type hints, Google-style docstrings
- JAX arrays everywhere downstream of `lower` (so the whole IR is traceable)
- Decorator-based registries (same as apvbt)
- Parody tone in prose, rigorous substance in code

## Troubleshooting
- **GPU not found** — `vectorize.sharded_features` falls back to `vmap` on a single device.
- **latent dim mismatch** — the surrogate's `nlat` must match the cross-coder architecture used in `lower`.
- **"lower returns same u for all subjects"** — the cross-coder's `encode` method wasn't receiving the subject's connectivity matrix. Fixed by `_encode_subject()` in `lower.py` which extracts and normalizes the subject's triu vector before applying the trained encoder weights. If you see this symptom, check that `_encode_subject` is being called (not the old `crosscoder.encode(...)` which only returned the cohort mean).
- **"no surrogate compiled for model X"** — register it in `surrogates/` or check `list_surrogates()`.
- **sbi not installed** — `posterior.attach_posterior` needs `sbi` + `torch`; pass `train_posterior=False` to skip.
- **"train_surrogate raises TypeError or doesn't converge"** — the hand-rolled `_adam` optimizer in `codegen.py` had a structural mismatch between `init` (flattened 2N elements) and `update` (N-element grads from `jax.grad`). Replaced with `optax.adam`. If training fails, check that `import optax` is present and the training loop uses `optimizer.update(g, opt_state, params)` + `optax.apply_updates(params, updates)`.

*Last updated: 2026-07-21*
