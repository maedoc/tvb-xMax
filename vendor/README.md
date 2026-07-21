# Vendored dependencies

This directory contains **pinned snapshots** of two ins-amu projects that
tvb-max builds on. They are vendored (rather than pip-installed) so the
repo is self-contained and the "compiler" runs against known-good versions
of the simulation substrate and the cross-coder/SBI patterns.

`tvb_max/__init__.py` and `conftest.py` prepend this directory to
`sys.path`, so `import vbjax` and `import apvbt` resolve here, not to
anything in site-packages.

## What's vendored

| Package | Version | Source | License |
|---|---|---|---|
| `vbjax` | v0.0.19 | https://github.com/ins-amu/vbjax | (see upstream) |
| `apvbt` | snapshot of `~/src/apvbt` | https://github.com/ins-amu/apvbt | (see upstream) |

Only the Python source is vendored. Large data files (`*.pkl`, `*.npz`,
`*.zip`), `__pycache__/`, and the `env/` virtualenv were excluded.

## What is NOT vendored (kept as pip deps)

- `jax` / `jaxlib` — large, actively maintained, GPU-specific builds
- `sbi` — large, actively maintained (mackelab/sbi)
- `torch` — required by `sbi`
- `numpy`, `scipy`, `fastapi`, `uvicorn`, `pydantic`, `discord.py`

These are declared in `pyproject.toml` and installed normally.

## Updating the vendored copies

```bash
# vbjax (from a local install or clone)
rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
  /path/to/vbjax/ vendor/vbjax/

# apvbt (from the sibling repo)
rsync -a --exclude='env/' --exclude='__pycache__/' \
  --exclude='*.pkl' --exclude='*.npz' --exclude='*.zip' \
  --exclude='.git/' --exclude='htmlcov/' --exclude='.coverage' \
  /home/duke/src/apvbt/apvbt/ vendor/apvbt/
```

## Why vendor instead of pip / submodule?

- **vbjax** is small (~340K, 20 modules) and slow-moving; vendoring pins a
  known-good version so the surrogate training is reproducible.
- **apvbt** is the sibling project, not on PyPI; vendoring makes tvb-max
  standalone. Note: apvbt is actively developed, so the vendored snapshot
  will drift — re-sync with the command above when needed. A git submodule
  is a reasonable alternative if you prefer to track apvbt's main branch.
