"""Extracted from apvbt (https://github.com/ins-amu/apvbt) — see vendor/README.md

This sub-package contains the subset of apvbt that tvb-xMax needs at
compile time (cross-coder data class, simulation-budget samplers, and
small utilities).  Everything else (datasets, benchmarks, API server,
dynamics model wrappers, regimes, reports, visualisation) was dropped.
"""

from . import utils
from . import data
from . import crosscoder
from . import inference
from . import simulation

# Re-export the symbols the rest of tvb-xMax references.
from .utils import MvNorm, triu_to_mat, small, all_conf_rates
from .data import XCode
from .crosscoder import encode_conn, decode_conn, calc_mvn, train
from .simulation import sample_model, sample_subj_model
from .inference import run_sbi, posterior_diags, to_torch
