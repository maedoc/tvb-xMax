# Vendored dependencies

This directory **no longer contains vendored package copies**.

## What changed (T5.1)

The vendored ``apvbt`` snapshot that previously lived under ``vendor/apvbt/``
has been **extracted** and moved to ``tvb_max/_apvbt/``.  Only the modules
that tvb-max actually needs at compile time were kept:

| Module                                    | Purpose                         |
|-------------------------------------------|---------------------------------|
| ``tvb_max/_apvbt/data.py``               | ``XCode`` cross-coder data class |
| ``tvb_max/_apvbt/crosscoder.py``         | Training / encode / decode fns  |
| ``tvb_max/_apvbt/simulation.py``         | ``sample_model`` / ``sample_subj_model`` |
| ``tvb_max/_apvbt/inference.py``          | ``run_sbi`` helper (reference)  |
| ``tvb_max/_apvbt/utils.py``              | ``MvNorm``, ``triu_to_mat``, …  |

Everything else (datasets, benchmarks, API server, dynamics model wrappers,
regimes, reports, visualisation, tests, ``__pycache__``) was **dropped**.
The original source lives at https://github.com/ins-amu/apvbt.

``vbjax`` remains a normal pip dependency (``vbjax>=0.0.19,<0.1``) declared
in ``pyproject.toml``.

## What is NOT vendored (kept as pip deps)

- ``vbjax`` — on PyPI (0.0.19); declared in ``pyproject.toml``
- ``jax`` / ``jaxlib`` — large, GPU-specific builds
- ``sbi`` — large, actively maintained (mackelab/sbi)
- ``torch`` — required by ``sbi``
- ``numpy``, ``scipy``, ``fastapi``, ``uvicorn``, ``pydantic``, ``discord.py``
