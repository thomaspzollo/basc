"""Fast sanity tests that need no model outputs: pytest tests/"""
import numpy as np
import pandas as pd

from basc.answers import compute_correctness, extract_last_boxed, probe_agree_matrix, qa_correct
from basc.metrics import get_ece2, selective_acc_at_abstention
from basc.sampling import draw_probe_indices, three_way_split
from basc.stage1 import fit_predict_binomial_logistic
from basc.stage2 import fit_binary_isotonic


def test_extract_last_boxed():
    assert extract_last_boxed("so \\boxed{3} then \\boxed{ 42 }") == "42"
    assert extract_last_boxed("no answer") == ""


def test_correctness_and_agreement_rules():
    assert qa_correct("\\text{Paris", ["paris, france"]) == 1   # extracted answers keep no closing brace
    assert qa_correct("", ["anything"]) == 0
    df = pd.DataFrame({"answer": ["12", "13"], "ground_truth": ["12", "12"]})
    assert compute_correctness(df, "gsm8k").tolist() == [1, 0]
    agree = probe_agree_matrix(np.array(["Paris", ""], dtype=object),
                               np.array([["paris, france", "Rome"], ["", "x"]], dtype=object), "trivia_qa")
    assert agree.tolist() == [[True, False], [True, False]]


def test_splits_and_draws_are_deterministic():
    a, b = three_way_split(1000, seed=3), three_way_split(1000, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert [len(x) for x in a] == [200, 200, 600]
    assert len(np.unique(np.concatenate(a))) == 1000
    p = draw_probe_indices(50, 100, 4, seed=7)
    assert p.shape == (50, 4) and np.array_equal(p, draw_probe_indices(50, 100, 4, seed=7))
    assert all(len(set(row)) == 4 for row in p)


def test_stage2_nearest_knot_prediction():
    iso = fit_binary_isotonic(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert np.allclose(iso.predict(np.array([0.0, 0.14, 0.86, 1.0])), [0, 0, 1, 1])
    assert set(np.unique(iso.predict(np.linspace(0, 1, 101)))) <= set(iso.knots_y)


def test_basc_recalibrates_synthetic_data():
    """Two-stage BASC on synthetic data where the latent self-consistency is a
    logistic function of the features: recalibrated confidence tracks agreement."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=8)

    def sample(n, n_probes):
        X = rng.normal(size=(n, 8))
        s = 1 / (1 + np.exp(-(X @ w)))
        return X, s, rng.binomial(1, s[:, None], size=(n, n_probes))

    X1, _, B1 = sample(2000, 4)
    X2, _, B2 = sample(2000, 1)
    Xt, st, _ = sample(5000, 1)
    U = fit_predict_binomial_logistic(X1, np.full(len(X1), 4), B1.sum(1), np.concatenate([X2, Xt]))
    conf = fit_binary_isotonic(U[:len(X2)], B2[:, 0]).predict(U[len(X2):])
    outcome = rng.binomial(1, st)
    assert get_ece2(conf, outcome) < 0.05
    assert np.corrcoef(conf, st)[0, 1] > 0.9


def test_selective_accuracy():
    df = selective_acc_at_abstention(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]), [0.0, 0.5])
    assert df["sel_acc"].tolist() == [0.5, 1.0]
    assert df.loc[1, "rej_acc"] == 0.0
