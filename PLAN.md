# tvb-xMax: Development Plan

> **tvb-xMax** — an "advanced AI math compiler" for virtual brain simulation that produces nearly-infinite speedup next to existing fast simulations, by **swapping features and parameters during the simulation-based inference (SBI) step** and sampling the posterior over data features *as if we simulated the model*.

This document is the bootstrapping plan for the `~/src/tvb-xMax` repository. It is organized in two parts, per the founding brief:

- **Part I — The Compiler** (concrete, code-grounded, comes first)
- **Part II — The Community Bootstrap** (API, docs site, Discord + openclaw agents, leaderboard, examples)

tvb-xMax is a *parody* in tone but a *real, buildable* system in substance. Every "compiler" stage below maps to actual code already scaffolded in `tvb_xmax/`, and every claim about speedup is grounded in the measured costs of the underlying `vbjax` simulations and `apvbt` SBI pipeline it wraps.

---

## 0. The core idea (one paragraph)

In standard SBI you pay a simulation tax: for every parameter sample `θ` you integrate a brain dynamics SDE (`vbjax.make_sde` + `mpr_dfun`/`hopf_dfun`/…) for seconds, extract features `xf`, then train a neural posterior `p(θ | xf)`. tvb-xMax **compiles that simulation away**: a one-time simulation budget trains a neural **surrogate** `xf ≈ S(u, θ)` that maps the cross-coder latent `u` plus normalized parameters directly to features. Because `u` is parcellation-invariant (the cross-coder maps 20 parcellations into one shared latent), a *single* compiled artifact serves **any connectivity, any parameters, any parcellation** — swapping them is a free re-evaluation of the same net. Inference becomes a batched GPU forward pass (~ms for 10⁴ samples) instead of 10⁴ SDE integrations (~hours), i.e. "nearly infinite" amortized speedup. The posterior is itself amortized, so drawing samples for a new subject is also free.

---

# Part I — The Compiler

## 1. Compiler architecture overview

tvb-xMax maps a classical compiler pipeline onto amortized SBI. The table is the contract every stage implements; the right column points at the scaffolded module.

| Compiler stage | tvb-xMax meaning | Input → Output | Module |
|---|---|---|---|
| **source program** | user spec: model + connectivity + params + feature + target | `IRSpec` | `tvb_xmax/ir.py` |
| **frontend** (parse/validate) | resolve model from registry, validate params against `ParameterSpace` | `IRSpec → IRSpec` | `compiler/frontend.py` |
| **lower** (to IR) | cross-code connectivity → parcellation-invariant latent `u`; normalize params to `[0,1]ᵈ` | `IRSpec → IRProgram` | `compiler/lower.py` |
| **optimize** (IR transforms) | whiten `u` against cohort MVN; summarize heterogeneous params so input dim is parcellation-invariant | `IRProgram → IRProgram` | `compiler/optimize.py` |
| **codegen** (emit object code) | train neural surrogate `S(u,θ)→xf` on the one-time sim budget | sim pairs → `CompiledArtifact` | `compiler/codegen.py` |
| **vectorize** (GPU batch) | `vmap`+`pmap` the surrogate; batched posterior sampling | artifact + batch → batched output | `compiler/vectorize.py` |
| **posterior** (amortized inference) | NPE over IR: `p(θ | xf)` trained once, sampled in batches | sim pairs → posterior fn bound to artifact | `compiler/posterior.py` |
| **linker** (orchestration) | run the whole pipeline; expose free swaps | spec + crosscoder → result | `compiler/pipeline.py` |
| **relink / swap** | free re-evaluation: parc/param reuse artifact; model/feature pick another | `IRSpec + swap kind → IRSpec` | `compiler/swap.py` |

The **intermediate representation (IR)** is the pair `(u, θ)` where:
- `u ∈ ℝⁿˡᵃᵗ` is the cross-coder latent (parcellation-invariant),
- `θ ∈ [0,1]ᵈ` is the normalized parameter vector (model-invariant scale).

Everything downstream of `lower` operates *only* on IR tensors, which is what makes the swaps free. The IR dataclasses live in `tvb_xmax/ir.py` (`IRSpec`, `IRProgram`, `CompiledArtifact`, `CompileReport`, `SwapKind`).

## 2. Why the cross-coder is the IR

`apvbt.XCode` / `vbjax.CrossCoder` trains linear encoder-decoder pairs across N parcellations sharing one latent `u`:

```
encode:  u = c_parc @ W_enc + b_enc          (parc p → latent)
decode:  c'_parc = u @ W_dec + b_dec          (latent → parc p)
```

The cross-prediction loss forces `u` to be the *common information* across parcellations, so a connectome in `079-Shen2013` and the same subject's connectome in `150-Destrieux` encode to (nearly) the same `u`. This is exactly the property an IR needs: **parcellation-invariance**. tvb-xMax exploits it three ways:

1. **Any connectivity in**: a user supplies a matrix in *any* known parcellation; `lower` encodes it to `u` and the rest of the pipeline never sees the parcellation again.
2. **Cohort prior**: `CrossCoder.calc_mvn(arch)` gives a `MvNorm(u_mean, u_cov)` over the cohort, which `optimize.condition_latent` uses to whiten new subjects into the training distribution — no re-simulation needed to handle a new cohort.
3. **Swap for free**: changing parcellation = re-encode → new `u` → same artifact. The artifact's weights don't depend on parcellation, only on `nlat`.

The cross-coder is already trained once (apvbt does this on HCP+1000Brains, ~50 min). tvb-xMax reuses that artifact directly; it does **not** retrain the cross-coder.

## 3. The "swapping" — the central joke and the central feature

The brief asks for "swapping features and parameters during the SBI step." In tvb-xMax this is literal and free, because the compiled artifact is a function of IR tensors only:

| Swap | What changes | Cost | Reuses artifact? |
|---|---|---|---|
| **parcellation** | new connectivity matrix in a different parc → re-encode to `u` | 1 matmul (encode) | ✅ same artifact |
| **parameters** | new `θ` | 0 (just change input) | ✅ same artifact |
| **model** | Hopf↔MPR↔Wilson-Cowan↔… | select different artifact | ❌ needs compiled artifact for that model |
| **features** | `var`↔`fc`↔`bold`↔custom | select different artifact head | ❌ needs artifact compiled for that feature |

Only model/feature swaps require a *different* pre-compiled artifact; parcellation and parameter swaps reuse the exact same weights. This is implemented in `compiler/swap.py` (`swap_parcellation`, `swap_parameters`, `swap_model`, `swap_features`, `apply_swap`).

The "sample the posterior over data features as if we simulated the model" line from the brief = the surrogate `S(u,θ)` *is* the simulation, to arbitrary precision set by `surrogate_mse`. You draw `xf ~ S(u, θ)` for any `(u, θ)`, then `θ ~ p(θ | xf)` via the amortized posterior — no SDE touched.

## 4. The speedup, quantified

Grounded in `apvbt/README.md` performance table and `vbjax` SDE costs:

| Operation | Real sim (vbjax) | tvb-xMax surrogate | Speedup |
|---|---|---|---|
| 1 feature eval (batch 128, 8-core pmap) | 1–10 s | ~0.1–1 ms (MLP fwd, GPU) | **10³–10⁴×** |
| SBI training budget (4096 sims) | 32–320 s | 0 (amortized) | ∞ (one-time) |
| Posterior draw, 10⁴ samples × 74 subjects | ~hours (would need 74×4096 sims) | seconds (batched `sample_batched`) | **~10³×** |
| New subject inference | re-run SBI (minutes-hours) | 1 forward pass (ms) | **~10⁵–10⁶×** amortized |

The "nearly infinite" framing is honest *amortized*: you pay the simulation tax **once** at compile time; every subsequent inference is a forward pass. Over a cohort of N subjects, marginal cost → 0 as N grows. `compiler/vectorize.benchmark_speedup` measures `t_sim / t_surrogate` for any batch; `CompileReport.speedup_vs_sim` records it per artifact.

The "maxest speedup" / "batch eval on GPU" line = `vectorize.sharded_features` (`pmap` across devices, `vmap` within) and `vectorize.batched_posterior` (sbi's `sample_batched` lifted over a batch of observations).

## 5. Concrete module contracts

### 5.1 `tvb_xmax/ir.py` — the IR dataclasses

```python
IRSpec            # source program: model, connectivity, parameters, feature, target
IRProgram         # lowered: u (nlat,), theta (d_param,) in [0,1], param_names, feature, target
CompiledArtifact  # object code: surrogate_apply(u,theta)->xf, posterior_sample(xf,n)->theta
CompileReport     # artifact + per-stage timings + speedup_vs_sim
SwapKind           # PARCELLATION | PARAMETERS | MODEL | FEATURES
```

### 5.2 `tvb_xmax/surrogates/` — one target per literature model

Mirrors `apvbt.dynamics.models` exactly (same `ParameterSpace`/`ParameterDefinition`/`DistributionType` shapes, same decorator registry). Six targets scaffolded, matching apvbt's six models:

| Model | File | Param space (mirrors apvbt) |
|---|---|---|
| Hopf | `surrogates/hopf.py` | k, D, eta (hetero), omega (hetero) |
| MPR | `surrogates/mpr.py` | k, D, J (hetero), w (hetero) |
| Wilson-Cowan | `surrogates/wilson_cowan.py` | k, D, tau_e, tau_i |
| Wong-Wang | `surrogates/wong_wang.py` | k, D, w, I0 |
| Kuramoto | `surrogates/kuramoto.py` | k, D, omega (hetero) |
| FitzHugh-Nagumo | `surrogates/fitzhugh_nagumo.py` | k, D, a, b |

A `SurrogateTarget` declares the parameter space + validation only — it does **not** run the simulation (that's what the surrogate replaces). Adding a 7th literature model = one file + `@register("name")`, same as apvbt.

### 5.3 `compiler/codegen.py` — the surrogate network

`SurrogateNet` is a 3-layer MLP with tanh activations over the concatenation `[u; θ]`:

```
xf = MLP([u; θ])      # (nlat + d_param) -> hidden(128) -> hidden(128) -> d_feat
```

- `make_surrogate_apply(nlat, d_param, d_feat)` → JIT-able `apply_fn(u, theta)`
- `train_surrogate(model, feature, sim_pairs, nlat, d_feat)` → `(apply_fn, params, mse)` using a minimal Adam (no `jax.example_libraries` dep)
- `compile_artifact(...)` → wraps as `CompiledArtifact` with diagnostics

**Design choice — shared trunk + feature head**: the first two layers are the "trunk" over `[u; θ]`; the last layer is the "feature head". This is what makes the **feature swap** cheap: one trunk, many heads. (The scaffold uses a single MLP for simplicity; the plan is to split trunk/head in v0.2.)

### 5.4 `compiler/posterior.py` — amortized NPE

Thin wrapper over `sbi.inference.NPE_C` (MAF) / `NPE_A` (MDN), mirroring `apvbt.inference.run_sbi` but operating on IR tensors. `attach_posterior(artifact, theta, features)` trains the posterior on the *same* sim budget as the surrogate and binds `posterior_sample` into the artifact. `save_artifact`/`load_artifact` pickle the whole compiled artifact (surrogate + posterior) for reuse.

### 5.5 `compiler/pipeline.py` — the orchestrator

```python
compile_spec(spec, crosscoder, sim_pairs, d_feat, mvn=None, train_posterior=True)
    -> CompileReport            # full compile, spends the one-time sim budget

run(artifact, spec, crosscoder, mvn=None) -> dict
    # fast path: lower -> optimize -> artifact(u,theta) -> [posterior]
    # NO simulation. This is the ~ms inference.

run_batch(artifact, specs, crosscoder, mvn=None) -> dict
    # vectorized run over many specs (the "maxest speedup" path)
```

### 5.6 `compiler/vectorize.py` — GPU batch eval

```python
batched_features(artifact, U, Theta)      # vmap over (B, nlat) x (B, d_param) -> (B, d_feat)
sharded_features(artifact, U, Theta)      # pmap across devices, falls back to vmap
batched_posterior(artifact, xf_obs, n)     # (n, B, d_param) posterior draws
benchmark_speedup(artifact, sim_fn, U, Theta)  # t_sim/t_surrogate measurement
```

## 6. The compile + run lifecycle (concrete)

```
                    ┌──────────── ONE-TIME (compile) ────────────┐
                    │                                             │
  apvbt sample_model  ──►  (U, Theta, XF)  sim budget (4096 sims) │
                    │           │                                 │
  IRSpec ─► frontend ─► lower ─► optimize ─► codegen.train_surrogate
                    │                       │                     │
                    │                       ▼                     │
                    │              CompiledArtifact               │
                    │              (surrogate + posterior)        │
                    └─────────────────────────────────────────────┘
                                       │
                    ┌──── EVERY INFERENCE (fast path) ────────────┐
                    │                                             │
  new IRSpec ─► frontend ─► lower ─► optimize ─► artifact(u,theta)│
                    │                                  │          │
                    │                                  ▼          │
                    │                          features (ms)      │
                    │                                  │          │
                    │                                  ▼          │
                    │                          posterior.sample   │
                    │                          (batched, ms)      │
                    └─────────────────────────────────────────────┘
```

The one-time compile spends the simulation budget (the *only* place real `vbjax` SDEs run). Every subsequent `run`/`swap` is pure forward passes.

## 7. Relationship to apvbt and vbjax (what's reused, what's replaced)

| Concern | apvbt / vbjax | tvb-xMax |
|---|---|---|
| cross-coder (IR) | `apvbt.XCode` / `vbjax.CrossCoder` | **reused as-is** (the IR is already parcellation-invariant) |
| dynamics models | `apvbt.dynamics.models.*` (Hopf, MPR, …) | **reused for the one-time sim budget only**; parameter spaces mirrored in `surrogates/` |
| simulation loop | `vbjax.make_sde` + `*_dfun` | **reused at compile time**; replaced by surrogate at run time |
| SBI | `apvbt.inference.run_sbi` (NPE_C/A) | **reused** (`compiler/posterior.py` wraps the same `sbi` calls) |
| simulation sampling | `apvbt.simulation.sample_model` / `sample_subj_model` | **replaced** by `pipeline.run` (surrogate forward) |
| cohort MVN | `XCode.calc_mvn` / `CrossCoder.calc_mvn` | **reused** for `optimize.condition_latent` |
| diagnostics | `apvbt.inference.posterior_diags` (shrinkage, z, ci90) | **reused** + extended with SBC/C2ST in `community/leaderboard/scoring.py` |

tvb-xMax is a **thin compiler layer over apvbt+vbjax**, not a replacement. It calls apvbt/vbjax at compile time and replaces them at run time.

## 8. Phased compiler roadmap

### v0.1 — Skeleton + toy compile (done, but had critical bugs)
- IR dataclasses, all 8 compiler stages, 6 surrogate targets, 3 examples
- Toy synthetic sim budget (no real vbjax sims needed to run examples)
- `compile_spec` / `run` / `run_batch` / 4 swaps end-to-end on toy data
- **Critical bugs found in honesty pass**: `lower.lower` ignored connectivity matrix (T0.1), `_adam` optimizer structurally broken (T0.2)

### v0.1.5 — Honesty pass (done)
- Fixed `lower.lower` to actually encode the subject's connectivity via `_encode_subject()` (T0.1)
- Replaced broken hand-rolled Adam with `optax.adam` (T0.2)
- Verified compile path runs end-to-end on toy data (T0.3)
- Made `_noop_sim` honest — `speedup_vs_sim` is `nan` with explanatory note (T0.4)
- pyproject.toml hygiene: excluded vendor from pytest, dropped scipy, pinned all versions (T1.1–T1.3)
- Removed double vbjax on sys.path — now pip-installed only (T1.5)
- Switched to `dataclasses.replace` in optimize (T1.6), cleaned up torch imports (T1.7)
- Made `CompiledArtifact` pickleable via `__getstate__`/`__setstate__` + `rebuild_apply_fns` (T1.8)
- Implemented trunk/head surrogate split: shared 2-layer tanh trunk + feature-specific 1-layer head (T1.9)
- 133 tests across 10 test files, 98% line coverage on compiler + ir (T2.0–T2.10)
- Wired real vbjax Hopf SDE into `benchmark_speedup` — ~68x CPU speedup measured (T3.1)
- `bench/bench_hopf_speedup.py` benchmark script + `bench/results.md` baseline (T3.2)
- `ArtifactCache` with in-memory + disk persistence, keyed by `(model, feature, nlat)` (T4.1)
- Wired `swap_model`/`swap_features` to cache via `resolve_artifact`/`run_cached` (T4.2)
- Extracted useful apvbt bits to `tvb_xmax/_apvbt/` (120KB), deleted 952KB vendor (T5.1)
- Added `SimBudget` dataclass + `from_apvbt` adapter (T5.2)
- Updated AGENTS.md troubleshooting with P0 bug entries (T7.4)
- Updated README.md swap table + optax in quick-start (T7.3)

### v0.2a — Real benchmark (done)
- Real `vbjax.make_sde` + Hopf SDE wired into `benchmark_speedup` (T3.1)
- Benchmark script records t_sim, t_surrogate, speedup, MSE (T3.2)

### v0.2b — Trunk/head split (done)
- Split surrogate into shared trunk + feature head (T1.9)
- Feature swap reuses trunk, only retrains head
- `rebuild_apply_fns` for pickle round-trip (T1.8)

### v0.2c — SBC + C2ST diagnostics (planned)
- Implement `sbc_score` (simulation-based calibration: rank-uniformity test)
- Wire `c2st_score` (classifier two-sample test) from `community/leaderboard/scoring.py`
- Populate `CompileReport.artifact.sbc_score` / `c2st_score` during `compile_spec` when `train_posterior=True`

### v0.2d — Real sim budget via apvbt (planned)
- `sim_budget.from_apvbt(xc, model, mvn, parc, n)` calls extracted `sample_model` with `pmap`
- Produces a real `SimBudget` for training the surrogate
- Validate: surrogate trained on real budget achieves MSE < toy-data MSE

### v0.3 — Multi-model + multi-feature artifacts
- Artifact cache keyed by `(model, feature, nlat)` — **done in v0.1.5** (T4.1)
- Cross-model posterior transfer (train posterior on Hopf, warm-start MPR posterior)
- Heterogeneous-param summarization (`optimize.reparam_heterogeneous`) for real

### v0.4 — Production compiler
- AOT-compile the surrogate with `jax.jit` + `jax.export` for deployment
- Quantization (int8) for edge inference
- Streaming inference for very large batches (chunked `lax.scan`)

---

# Part II — The Community Bootstrap

## 9. API with rate limiting and accounts

`tvb_xmax/api/` — FastAPI app (`api/server.py`) with:

- **Accounts** (`api/auth.py`): SQLite-backed, username+password (PBKDF2), per-account API key + JWT minting. Tiers: `free` / `pro` / `agent`.
- **Auth middleware**: accepts `X-API-Key` header *or* `Authorization: Bearer <jwt>`. Skips auth for `/health`, `/account`, `/token`.
- **Rate limiting** (`api/ratelimit.py`): in-memory token bucket per `(user, endpoint)`, refilled at `tier_rate/60` per second. Big batches cost more tokens. Tier limits:

  | Tier | req/min | max batch |
  |---|---|---|
  | free | 60 | 128 |
  | pro | 600 | 4096 |
  | agent | 6000 | 65536 |

  (Swap the in-memory dict for Redis in multi-worker prod — same interface.)

- **Endpoints** (`api/routes/`):
  - `POST /api/v1/account` — create account → returns api_key
  - `POST /api/v1/token` — login → JWT
  - `POST /api/v1/compile` — async compile job (spends sim budget in background)
  - `GET  /api/v1/artifacts` / `/artifacts/{id}` — list/poll
  - `POST /api/v1/infer` — fast-path inference (ms)
  - `POST /api/v1/swap` — apply a free swap + re-run
  - `GET  /api/v1/leaderboard` — ranked artifacts

Run: `tvbxmax serve` → uvicorn on `:8088`.

## 10. Website with documentation

`docs/` — MkDocs Material site (`docs/` already created). Planned pages:

- `index.md` — landing (parody marketing copy + real architecture diagram)
- `compiler.md` — Part I of this plan, rendered
- `api.md` — OpenAPI-derived reference + curl examples
- `swaps.md` — the four swaps with runnable examples
- `models.md` — the six literature models + how to add a 7th
- `agents.md` — openclaw agent authoring guide
- `leaderboard.md` — scoring methodology

Build: `mkdocs serve` (add `mkdocs-material` to dev deps). Deploy: GitHub Pages from `docs/` via `mkdocs gh-deploy`.

## 11. Discord with openclaw agents

`tvb_xmax/community/` — a Discord bot (`discord_bot.py`) hosting **openclaw agents**, one per literature model. Each agent (`community/agents/`) owns a `SurrogateTarget` and autonomously:

1. **proposes** hyperparameters (nlat, hidden, niter, lr, sim_budget) by prompting an LLM (default: the local Gemma 4 12B server via the `gemma4-server-integration` skill, or any OpenAI-compatible endpoint),
2. **compiles** an artifact by calling the tvb-xMax API `/compile`,
3. **submits** to the leaderboard,
4. **iterates** based on its current rank.

The agent loop (`OpenClawAgent.step` / `run_forever`) is LLM-driven: the prompt encodes the current rank and goal, the LLM returns JSON hyperparameters, the agent runs the compile and posts a summary to its model's Discord channel. Two agents scaffolded (`hopf_agent.py`, `mpr_agent.py`); adding one = one file.

**Why "openclaw"**: the parody framing is a swarm of clawed agents competing to produce the best-calibrated surrogate for "their" model. The LLM endpoint is pluggable so it can run on the local Gemma 4 server (no external API costs) or on any hosted model.

Run: `tvbxmax agents` (needs `TVBXMAX_DISCORD_TOKEN` + `TVBXMAX_AGENT_INTERVAL`).

## 12. Leaderboard

`tvb_xmax/community/leaderboard/scoring.py` + `api/routes/leaderboard.py`. Ranks artifacts on three axes:

- **Calibration**: SBC (simulation-based calibration, `sbc_score`, 1=perfect) + C2ST (classifier two-sample test, `c2st_score`, 1=perfect match)
- **Fidelity**: surrogate MSE vs real sim features
- **Speedup**: `t_sim / t_surrogate`

Composite score (lower=better): `(1-c2st) + (1-sbc) + log10(mse+1) - log10(speedup+1)`. This prevents an agent from winning by trading calibration for raw speed. Exposed at `GET /api/v1/leaderboard` and mirrored to a `#leaderboard` Discord channel.

## 13. Examples (runnable)

Three examples in `examples/`, all runnable without the 6GB HCP/1KB dataset (they synthesize a toy sim budget):

1. **`01_compile_and_infer.py`** — full compile pipeline + fast-path inference. The "hello world".
2. **`02_swaps.py`** — the four free swaps (parcellation, parameters, model, features).
3. **`03_gpu_batch.py`** — batched GPU eval + speedup benchmark.

Each is self-contained: builds a trivial single-view `vbjax.CrossCoder`, synthesizes `(U, Theta, XF)`, compiles, and runs. Swap the toy budget for `apvbt.sample_model` output to go real.

## 14. Repository layout

```
tvb-xMax/
├── PLAN.md                  ← this file
├── README.md
├── AGENTS.md                ← parody agent guide (mirrors apvbt/AGENTS.md)
├── pyproject.toml
├── tvb_xmax/
│   ├── __init__.py
│   ├── ir.py                ← IR dataclasses
│   ├── cli.py
│   ├── compiler/
│   │   ├── __init__.py
│   │   ├── frontend.py      ← parse + validate
│   │   ├── lower.py         ← cross-code connectivity → latent u
│   │   ├── optimize.py      ← IR transforms (whiten, reparam)
│   │   ├── codegen.py       ← surrogate net (the "object code")
│   │   ├── vectorize.py     ← GPU batch eval (pmap+vmap)
│   │   ├── posterior.py     ← amortized NPE over IR
│   │   ├── pipeline.py      ← compile + run orchestration
│   │   └── swap.py          ← the four free swaps
│   ├── surrogates/          ← one target per literature model
│   │   ├── base.py          ← SurrogateTarget + registry
│   │   ├── hopf.py  mpr.py  wilson_cowan.py  wong_wang.py  kuramoto.py  fitzhugh_nagumo.py
│   ├── api/                 ← FastAPI + auth + ratelimit
│   │   ├── server.py  auth.py  ratelimit.py  models.py
│   │   └── routes/  (compile, infer, swap, leaderboard)
│   └── community/          ← Discord + openclaw agents + leaderboard
│       ├── discord_bot.py
│       ├── agents/  (base, hopf_agent, mpr_agent)
│       └── leaderboard/scoring.py
├── examples/  (01_compile_and_infer, 02_swaps, 03_gpu_batch)
├── tests/
└── docs/                   ← MkDocs site
```

## 15. Bootstrapping sequence (what to build in what order)

1. **Week 1 — compiler core**: flesh out `codegen.train_surrogate` with real Adam, wire `pipeline.compile_spec` to `apvbt.sample_model` for the sim budget, validate on Hopf with real HCP data. Target: `CompileReport` with real `surrogate_mse` + `speedup_vs_sim`.
2. **Week 2 — swaps + posterior**: implement `optimize.reparam_heterogeneous` for real (summarize hetero `eta`/`omega` arrays), wire `posterior.attach_posterior`, validate SBC/C2ST on the Hopf artifact.
3. **Week 3 — API**: stand up FastAPI, wire `/compile` to the pipeline, `/infer` to `pipeline.run`, deploy behind Caddy (reuse the `godaddy-caddy-ingress` skill for `tvb-xMax.ins-amu.fr`).
4. **Week 4 — docs + examples**: MkDocs site, real (non-toy) examples using cached `both.pkl`, curl recipes.
5. **Week 5 — agents**: Discord bot + 2 openclaw agents (Hopf, MPR) on the local Gemma 4 server, leaderboard live.
6. **Week 6 — community**: open the remaining 4 model channels, agent authoring guide, first public leaderboard snapshot.

## 16. Open questions

| Question | Lean | Status |
|---|---|---|
| Surrogate trunk/head split now or v0.2? | v0.2 — ship the flat MLP first, validate the speedup claim | **Resolved** (T1.9): implemented trunk/head split in P1 |
| Posterior over full `θ` or just `(k,D)`? | Full `θ` (incl. latent `u`) like apvbt's cohort SBI; subject-level is a swap | Open |
| Redis for rate limits from day 1? | No — in-memory until multi-worker; interface is stable | Open |
| LLM for agents: local Gemma 4 or hosted? | Local Gemma 4 (zero cost, `gemma4-server-integration` skill) | Open |
| Real sim budget: apvbt `sample_model` or fresh? | Reuse `apvbt.sample_model` — it already does the `pmap` batched sims | **Resolved** (T5.1/T5.2): extracted apvbt bits to `tvb_xmax/_apvbt/`, added `SimBudget` + `from_apvbt` adapter |
| Artifact storage: pkl or DB? | pkl files + SQLite index; graduate to object storage if needed | **Resolved** (T4.1): `ArtifactCache` with pkl + `index.json` |
| Adam optimizer: hand-rolled or optax? | optax — standard, well-tested, avoids maintenance burden | **Resolved** (T0.2): replaced broken `_adam` with `optax.adam` |
| apvbt: vendor, submodule, or extract? | Extract useful bits, drop the rest | **Resolved** (T5.1): extracted 120KB to `tvb_xmax/_apvbt/`, deleted 952KB vendor |

---

*This is a parody project. The "nearly infinite speedup" is real amortized inference; the "advanced AI math compiler" is a neural surrogate + amortized posterior; the "openclaw agents" are LLM-driven hyperparameter search bots. Everything above is grounded in the actual `apvbt` + `vbjax` codebase it wraps.*
