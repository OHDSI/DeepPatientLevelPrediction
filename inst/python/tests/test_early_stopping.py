"""Regression tests for EarlyStopping `min_epochs` (issue #36).

"Add minimum number of epochs to EarlyStopper - To stop the training
stopping too soon."

The trainer calls the stopper once per epoch with the validation metric.
`min_epochs` guarantees at least that many epochs are trained before
early stopping is allowed to fire, even when patience is exhausted sooner.

Note: the EarlyStopping class itself is pure Python, but importing
Estimator pulls in torch and the full data pipeline. To keep this a
lightweight unit test, the class is loaded from the real Estimator.py
source and executed in isolation.
"""
import ast
import os


def load_early_stopping():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Estimator.py"
    )
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EarlyStopping":
            namespace = {}
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), path, "exec"),
                namespace,
            )
            return namespace["EarlyStopping"]
    raise AssertionError("EarlyStopping class not found in Estimator.py")


EarlyStopping = load_early_stopping()


def test_min_epochs_defers_early_stop():
    stopper = EarlyStopping(patience=1, min_epochs=3, verbose=False, mode="max")
    stopper(0.8)  # epoch 1 sets the best score
    assert stopper.early_stop is False
    stopper(0.7)  # epoch 2: patience exhausted...
    assert stopper.counter == 1
    assert stopper.early_stop is False  # ...but min_epochs not reached yet
    stopper(0.6)  # epoch 3: grace period over -> stop fires
    assert stopper.early_stop is True


def test_min_epochs_still_trains_through_grace_period():
    stopper = EarlyStopping(patience=1, min_epochs=4, verbose=False, mode="max")
    for metric in (0.8, 0.7, 0.6):
        stopper(metric)
        assert stopper.early_stop is False
    stopper(0.5)
    assert stopper.early_stop is True


def test_default_behavior_unchanged():
    stopper = EarlyStopping(patience=2, verbose=False, mode="max")
    stopper(0.8)
    assert stopper.early_stop is False
    stopper(0.7)
    assert stopper.early_stop is False
    stopper(0.6)
    assert stopper.early_stop is True  # stops exactly when patience is exhausted


def test_explicit_min_epochs_zero_matches_default():
    stopper = EarlyStopping(patience=2, min_epochs=0, verbose=False, mode="max")
    stopper(0.8)
    stopper(0.7)
    assert stopper.early_stop is False
    stopper(0.6)
    assert stopper.early_stop is True


def test_recovery_during_grace_period():
    stopper = EarlyStopping(patience=1, min_epochs=5, verbose=False, mode="max")
    stopper(0.8)
    stopper(0.7)  # patience exhausted but min_epochs guards
    assert stopper.early_stop is False
    stopper(0.9)  # improvement resets the counter
    assert stopper.improved is True
    assert stopper.counter == 0
    assert stopper.early_stop is False


def test_min_epochs_with_min_mode():
    # mode="min": lower metric is better
    stopper = EarlyStopping(patience=1, min_epochs=3, verbose=False, mode="min")
    stopper(0.2)
    assert stopper.early_stop is False
    stopper(0.3)  # worse -> counter hits patience, but grace period holds
    assert stopper.early_stop is False
    stopper(0.4)  # grace period over -> stop fires
    assert stopper.early_stop is True
