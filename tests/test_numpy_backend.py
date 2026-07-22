"""Parity checks for the portable NumPy compiler backend."""

import numpy as np

from tvb_xmax import IRSpec
from tvb_xmax.cde import MAFEstimator
from tvb_xmax.compiler import numpy_pipeline
from tvb_xmax.compiler import numpy_sim


def _spec(nlat, target="both"):
    return IRSpec(model="hopf", connectivity=np.zeros(nlat), connectivity_is_latent=True,
                  parameters={"k": .5, "D": .3}, feature="var", target=target,
                  n_posterior=4)


def test_numpy_compile_run_and_batch(toy_crosscoder, toy_sim_budget, toy_nlat, toy_d_feat):
    report = numpy_pipeline.compile_spec(_spec(toy_nlat), toy_crosscoder, toy_sim_budget,
                                         toy_d_feat, niter=20, train_posterior=False)
    artifact = report.artifact
    assert artifact.backend == "numpy"
    out = numpy_pipeline.run(artifact, _spec(toy_nlat, "features"), toy_crosscoder)
    assert out["features"].shape == (toy_d_feat,)
    batch = numpy_pipeline.run_batch(artifact, [_spec(toy_nlat, "features")] * 3, toy_crosscoder)
    assert batch["features"].shape == (3, toy_d_feat)


def test_numpy_maf_samples(toy_sim_budget):
    _, theta, features = toy_sim_budget
    estimator = MAFEstimator(theta.shape[-1], features.shape[-1], n_flows=2, hidden_units=12)
    estimator.train(theta, features, n_iter=5)
    assert estimator.sample_batched((3,), features[:2]).shape == (3, 2, theta.shape[-1])


def test_numpy_hopf_simulation_features():
    connectivity = np.array([[0, .1, .2], [.1, 0, .15], [.2, .15, 0]], dtype=np.float32)
    features = numpy_sim.hopf_features(connectivity, [.2, .01, .1, 1.], n_steps=20)
    assert features.shape == (3,)
    assert np.all(np.isfinite(features))
