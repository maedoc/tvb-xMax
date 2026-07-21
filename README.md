# tvb-max

> An **advanced AI math compiler** for virtual brain simulation that produces nearly-infinite speedup next to existing fast simulations, by swapping features and parameters during the simulation-based inference step and sampling the posterior over data features *as if we simulated the model*.

tvb-max treats amortized simulation-based inference as a compiler: the **cross-coder is the IR**, a trained **neural surrogate is the object code** that replaces the SDE simulation, and **GPU batch eval** gives the speedup. Because the IR is parcellation-invariant, swapping connectivity / parameters / models / features is free.

It is a parody in tone and a real, buildable system in substance — a thin compiler layer over [`apvbt`](https://github.com/ins-amu/apvbt) and [`vbjax`](https://github.com/ins-amu/vbjax).

## The pipeline (compiler stages)

```
source program ─► frontend ─► lower ─► optimize ─► codegen ─► vectorize ─► posterior
   IRSpec          parse       cross-code    whiten      surrogate    GPU batch    NPE
                  validate     to latent u   reparam     (object code) pmap+vmap    amortized
```

| Stage | Does | Module |
|---|---|---|
| frontend | resolve model, validate params | `compiler/frontend.py` |
| lower | cross-code connectivity → latent `u` | `compiler/lower.py` |
| optimize | whiten `u`, summarize hetero params | `compiler/optimize.py` |
| codegen | train surrogate `S(u,θ)→xf` | `compiler/codegen.py` |
| vectorize | `pmap`+`vmap` GPU batch | `compiler/vectorize.py` |
| posterior | amortized NPE over IR | `compiler/posterior.py` |

## The four free swaps

| Swap | Cost | Reuses artifact? |
|---|---|---|
| parcellation | 1 encode matmul | yes |
| parameters | 0 | yes |
| model | select other artifact | no (needs compile) |
| features | select other head | no (needs compile) |

## Quick start

```bash
pip install jax jaxlib numpy vbjax sbi torch fastapi uvicorn pydantic
python examples/01_compile_and_infer.py   # compile + fast-path infer
python examples/02_swaps.py               # the four swaps
python examples/03_gpu_batch.py           # GPU batch + speedup
```

## Library use

```python
import vbjax as vb
from tvb_max.compiler import ir, pipeline, swap

# 1. one-time simulation budget (use apvbt.sample_model in real use)
U, Theta, XF = ...  # from apvbt.sample_model(xc, model, mvn, parc, ...)

# 2. COMPILE (spends the sim budget once)
spec = ir.IRSpec(model="hopf", connectivity=jnp.zeros(16),
                 connectivity_is_latent=True,
                 parameters={"k": 0.15, "D": 0.4}, target="posterior")
report = pipeline.compile_spec(spec, crosscoder, (U, Theta, XF), d_feat=8)
artifact = report.artifact   # the "object code"

# 3. RUN (fast path, no simulation)
out = pipeline.run(artifact, spec, crosscoder)   # ms

# 4. SWAP for free
new_spec = swap.swap_parameters(spec, k=0.25, D=0.1)
out = pipeline.run(artifact, new_spec, crosscoder)  # same artifact
```

## API + community

- **API**: `tvbmax serve` → FastAPI on `:8088` with accounts, JWT, per-tier rate limiting. `POST /compile`, `/infer`, `/swap`, `GET /leaderboard`.
- **Discord + openclaw agents**: `tvbmax agents` → one LLM-backed bot per literature model, each compiling artifacts and competing on the leaderboard.
- **Leaderboard**: ranks artifacts by SBC + C2ST calibration, surrogate MSE, and `t_sim/t_surrogate` speedup.

See **[PLAN.md](PLAN.md)** for the full design (compiler first, then community bootstrap).

## What's reused from apvbt / vbjax

- **cross-coder** (`vbjax.CrossCoder` / `apvbt.XCode`) — reused as the IR
- **dynamics models** (`apvbt.dynamics.models`) — reused at compile time for the sim budget; parameter spaces mirrored in `surrogates/`
- **SBI** (`apvbt.inference.run_sbi` / `sbi`) — reused in `compiler/posterior.py`
- **cohort MVN** (`CrossCoder.calc_mvn`) — reused for latent whitening

tvb-max calls apvbt/vbjax at **compile time** and replaces them at **run time**.

## License

MIT (parody; underlying libs retain their own licenses).
