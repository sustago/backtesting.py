# Optuna Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Optuna as an alternative optimizer in `backtesting.py`, with optional DB persistence, resume support, and a progress callback.

**Architecture:** New inner function `_optimize_optuna()` inside `Backtest.optimize()`, following the same pattern as `_optimize_openbox()`. New parameters added to the `optimize()` signature. Optuna is a lazy-imported optional dependency.

**Tech Stack:** Optuna (lazy import), SQLAlchemy (via Optuna's storage), PostgreSQL/SQLite (optional)

**Spec:** `docs/superpowers/specs/2026-04-03-optuna-integration-design.md`

---

### Task 1: Add Optuna parameters to `optimize()` signature

**Files:**
- Modify: `backtesting/backtesting.py:1254-1272` (signature)
- Modify: `backtesting/backtesting.py:1273-1338` (docstring)
- Modify: `backtesting/backtesting.py:1369-1370` (return_optimization guard)
- Modify: `backtesting/backtesting.py:1542-1545` (method dispatch)

- [ ] **Step 1: Add new parameters to the `optimize()` signature**

Change the signature at line 1254 from:

```python
def optimize(self, *,
             maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
             method: str = 'openbox',
             max_tries: Optional[Union[int, float]] = None,
             constraint: Optional[Callable[[dict], bool]] = None,
             outcome_constraints: Optional[list[Callable[[pd.Series], bool]]] = None,
             return_heatmap: bool = False,
             return_optimization: bool = False,
             random_state: Optional[int] = None,
             n_initial_points: Optional[int] = None,
             init_strategy: Literal['random', 'random_explore_first', 'latin_hypercube', 'default', 'sobol'] = 'random_explore_first',
             n_workers: int = 1,
             advisor_type: Literal['bo', 'tpe', 'ea', 'random', 'mcadvisor'] = "tpe",
             acq_type: str = 'auto',
             use_constrained_model: bool = False,
             return_configs: int = 100,
             init_configs: list[dict] | None = None,
             fixed_params: dict | None = None,
             **kwargs) -> Tuple[pd.Series, List]:
```

to:

```python
def optimize(self, *,
             maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
             method: str = 'openbox',
             max_tries: Optional[Union[int, float]] = None,
             constraint: Optional[Callable[[dict], bool]] = None,
             outcome_constraints: Optional[list[Callable[[pd.Series], bool]]] = None,
             return_heatmap: bool = False,
             return_optimization: bool = False,
             random_state: Optional[int] = None,
             n_initial_points: Optional[int] = None,
             init_strategy: Literal['random', 'random_explore_first', 'latin_hypercube', 'default', 'sobol'] = 'random_explore_first',
             n_workers: int = 1,
             advisor_type: Literal['bo', 'tpe', 'ea', 'random', 'mcadvisor'] = "tpe",
             acq_type: str = 'auto',
             use_constrained_model: bool = False,
             return_configs: int = 100,
             init_configs: list[dict] | None = None,
             fixed_params: dict | None = None,
             optuna_storage: Optional[str] = None,
             optuna_study_name: Optional[str] = None,
             optuna_sampler: Literal['tpe', 'cmaes', 'random'] = 'tpe',
             progress_callback: Optional[Callable[[int, int, float], None]] = None,
             **kwargs) -> Tuple[pd.Series, List]:
```

- [ ] **Step 2: Update the `return_optimization` guard**

Change line 1369 from:

```python
if return_optimization and method != 'skopt':
    raise ValueError("return_optimization=True only valid if method='skopt'")
```

to:

```python
if return_optimization and method not in ('skopt', 'optuna'):
    raise ValueError("return_optimization=True only valid if method='skopt' or method='optuna'")
```

- [ ] **Step 3: Update the method dispatch**

Change lines 1542-1545 from:

```python
if method == 'openbox':
    output = _optimize_openbox()
else:
    raise ValueError(f"Method should be 'grid', 'openbox' or 'skopt', not {method!r}")
return output
```

to:

```python
if method == 'openbox':
    output = _optimize_openbox()
elif method == 'optuna':
    output = _optimize_optuna()
else:
    raise ValueError(f"Method should be 'openbox' or 'optuna', not {method!r}")
return output
```

- [ ] **Step 4: Update the docstring**

Add to the docstring (after the existing `method` description around line 1286):

```python
* `"optuna"` which uses [Optuna](https://optuna.org/) for optimization.
  Supports optional persistence via `optuna_storage` (SQLAlchemy URL)
  and resume of interrupted studies via `optuna_study_name`.

`optuna_storage` is an optional SQLAlchemy connection string
(e.g. ``"postgresql://user:pass@host/db"`` or ``"sqlite:///study.db"``).
If not set, optimization runs in-memory without persistence or resume.

`optuna_study_name` is an optional name for the Optuna study. If not set,
a name is auto-generated from the strategy class name and current timestamp.
When resuming, pass the same study name and storage to continue.

`optuna_sampler` selects the Optuna sampler algorithm.
Options: ``"tpe"`` (default), ``"cmaes"``, ``"random"``.

`progress_callback` is an optional function called after each Optuna trial
with signature ``(current_trial: int, max_trials: int, current_value: float)``.
``current_value`` is ``float('nan')`` for pruned trials.
```

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `python -m backtesting.test 2>&1 | tail -5`
Expected: All existing tests pass (no changes to behavior yet).

- [ ] **Step 6: Commit**

```bash
git add backtesting/backtesting.py
git commit -m "✨: Add Optuna parameters to optimize() signature"
```

---

### Task 2: Write tests for Optuna integration

**Files:**
- Modify: `backtesting/test/_test.py`

- [ ] **Step 1: Write test for basic Optuna optimization**

Add after the `test_method_openbox_with_given_initial_configs` method (line ~607):

```python
def test_method_optuna(self):
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    res, best_configs = bt.optimize(
        fast=range(2, 20), slow=np.arange(2, 20, dtype=object), blubb=[-2.0],
        constraint=lambda p: p.fast < p.slow,
        max_tries=15,
        method='optuna',
        random_state=42,
    )
    self.assertIsInstance(res, pd.Series)
    self.assertIsInstance(best_configs, list)
    self.assertGreater(len(best_configs), 0)
    self.assertLessEqual(len(best_configs), 15)
```

- [ ] **Step 2: Write test for Optuna with return_optimization**

```python
def test_method_optuna_return_study(self):
    import optuna
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    res, best_configs, study = bt.optimize(
        fast=range(2, 20), slow=np.arange(2, 20, dtype=object),
        constraint=lambda p: p.fast < p.slow,
        max_tries=10,
        method='optuna',
        return_optimization=True,
        random_state=42,
    )
    self.assertIsInstance(res, pd.Series)
    self.assertIsInstance(study, optuna.study.Study)
    self.assertEqual(study.direction, optuna.study.StudyDirection.MAXIMIZE)
```

- [ ] **Step 3: Write test for Optuna resume via SQLite**

```python
def test_method_optuna_resume(self):
    import optuna
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_study.db")
        storage = f"sqlite:///{db_path}"
        study_name = "test_resume_study"

        bt = Backtest(GOOG.iloc[:100], SmaCross)

        # First run: 5 trials
        res1, configs1 = bt.optimize(
            fast=range(2, 20), slow=range(2, 20),
            constraint=lambda p: p.fast < p.slow,
            max_tries=5,
            method='optuna',
            optuna_storage=storage,
            optuna_study_name=study_name,
            random_state=42,
        )

        # Verify 5 trials exist
        study = optuna.load_study(study_name=study_name, storage=storage)
        first_run_trials = len(study.trials)
        self.assertGreater(first_run_trials, 0)
        self.assertLessEqual(first_run_trials, 5)

        # Resume: max_tries=10 total, should run ~5 more
        res2, configs2 = bt.optimize(
            fast=range(2, 20), slow=range(2, 20),
            constraint=lambda p: p.fast < p.slow,
            max_tries=10,
            method='optuna',
            optuna_storage=storage,
            optuna_study_name=study_name,
            random_state=42,
        )

        study = optuna.load_study(study_name=study_name, storage=storage)
        self.assertGreater(len(study.trials), first_run_trials)
        self.assertLessEqual(len(study.trials), 10)
```

- [ ] **Step 4: Write test for progress callback**

```python
def test_method_optuna_progress_callback(self):
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    callback_calls = []

    def on_progress(current, total, value):
        callback_calls.append((current, total, value))

    res, _ = bt.optimize(
        fast=range(2, 20), slow=range(2, 20),
        constraint=lambda p: p.fast < p.slow,
        max_tries=10,
        method='optuna',
        progress_callback=on_progress,
        random_state=42,
    )
    self.assertGreater(len(callback_calls), 0)
    # All calls should have total=10
    for current, total, value in callback_calls:
        self.assertEqual(total, 10)
        self.assertIsInstance(current, int)
        self.assertIsInstance(value, float)
```

- [ ] **Step 5: Write test for auto-generated study name**

```python
def test_method_optuna_auto_study_name(self):
    import optuna
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    _, _, study = bt.optimize(
        fast=range(2, 20), slow=range(2, 20),
        constraint=lambda p: p.fast < p.slow,
        max_tries=5,
        method='optuna',
        return_optimization=True,
        random_state=42,
    )
    self.assertTrue(study.study_name.startswith("bt_SmaCross_"))
```

- [ ] **Step 6: Write test for sampler selection**

```python
def test_method_optuna_samplers(self):
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    for sampler in ('tpe', 'cmaes', 'random'):
        with self.subTest(sampler=sampler):
            res, _ = bt.optimize(
                fast=range(2, 20), slow=range(2, 20),
                constraint=lambda p: p.fast < p.slow,
                max_tries=5,
                method='optuna',
                optuna_sampler=sampler,
                random_state=42,
            )
            self.assertIsInstance(res, pd.Series)
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `python -m unittest backtesting.test._test.TestBacktest.test_method_optuna -v 2>&1 | tail -5`
Expected: FAIL or ERROR (since `_optimize_optuna` doesn't exist yet).

- [ ] **Step 8: Commit**

```bash
git add backtesting/test/_test.py
git commit -m "✅: Add tests for Optuna optimizer integration"
```

---

### Task 3: Implement `_optimize_optuna()`

**Files:**
- Modify: `backtesting/backtesting.py` (insert new function before the dispatch block at line ~1542)

- [ ] **Step 1: Add the `_optimize_optuna()` inner function**

Insert before the dispatch block (before `if method == 'openbox':` at line 1542):

```python
def _optimize_optuna() -> Tuple[pd.Series, List]:
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Need package 'optuna'. Install it with: pip install optuna"
        ) from None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Sampler
    sampler_map = {
        'tpe': optuna.samplers.TPESampler,
        'cmaes': optuna.samplers.CmaEsSampler,
        'random': optuna.samplers.RandomSampler,
    }
    sampler = sampler_map[optuna_sampler](seed=random_state)

    # Study name
    study_name = optuna_study_name or f"bt_{self._strategy.__name__}_{hex(int(time()))[2:8]}"

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Resume logic: max_tries is total count
    nonlocal max_tries
    max_tries = (200 if max_tries is None else
                 max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                 int(max_tries))

    existing = len([
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE,
                       optuna.trial.TrialState.PRUNED)
    ])
    remaining = max(0, max_tries - existing)

    # Objective
    def objective(trial: optuna.trial.Trial):
        params = {}
        for key, values in kwargs.items():
            values = np.asarray(values)
            if values.dtype.kind in 'mM':
                values = values.astype(int)

            if len(values) == 1:
                params[key] = values.item()
            elif values.dtype.kind in 'iumM':
                params[key] = trial.suggest_int(key, int(values.min()), int(values.max()))
            elif values.dtype.kind == 'f' or (
                values.dtype.kind == 'O'
                and all(isinstance(v, Decimal) for v in values)
            ):
                params[key] = trial.suggest_float(key, float(values.min()), float(values.max()))
            else:
                params[key] = trial.suggest_categorical(key, values.tolist())

        if not constraint(AttrDict(params)):
            raise optuna.TrialPruned()

        res = self.run(**params, fixed_params=fixed_params)
        value = maximize(res)

        if np.isnan(value):
            raise optuna.TrialPruned()

        return value

    # Progress callback wrapper
    def _optuna_callback(study, trial):
        if progress_callback is None:
            return
        value = trial.value if trial.value is not None else float('nan')
        progress_callback(len(study.trials), max_tries, value)

    # Run optimization
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[_optuna_callback])

    # Extract results
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise ValueError('No admissible parameter combinations to test')

    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    best_configs = [t.params for t in sorted_trials[:return_configs]]

    best_params = study.best_trial.params
    stats = self.run(**best_params, fixed_params=fixed_params)

    if return_optimization:
        return (stats, best_configs, study)
    return (stats, best_configs)
```

- [ ] **Step 2: Add `time` import**

The `time` function is not currently imported in `backtesting.py`. Add to the imports at the top of the file:

```python
from time import time
```

- [ ] **Step 3: Run the basic Optuna test**

Run: `python -m unittest backtesting.test._test.TestBacktest.test_method_optuna -v 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 4: Run all new Optuna tests**

Run: `python -m unittest backtesting.test._test.TestBacktest.test_method_optuna backtesting.test._test.TestBacktest.test_method_optuna_return_study backtesting.test._test.TestBacktest.test_method_optuna_resume backtesting.test._test.TestBacktest.test_method_optuna_progress_callback backtesting.test._test.TestBacktest.test_method_optuna_auto_study_name backtesting.test._test.TestBacktest.test_method_optuna_samplers -v 2>&1 | tail -20`
Expected: All 6 tests PASS.

- [ ] **Step 5: Run all existing tests for regression check**

Run: `python -m backtesting.test 2>&1 | tail -5`
Expected: All tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add backtesting/backtesting.py
git commit -m "✨: Implement Optuna optimizer integration"
```

---

### Task 4: Lint and final verification

**Files:**
- Modify: `backtesting/backtesting.py` (if lint issues found)
- Modify: `backtesting/test/_test.py` (if lint issues found)

- [ ] **Step 1: Run ruff**

Run: `ruff check backtesting/backtesting.py backtesting/test/_test.py 2>&1`
Expected: No errors. If there are errors, fix them.

- [ ] **Step 2: Run mypy**

Run: `mypy backtesting/backtesting.py 2>&1 | tail -10`
Expected: No new errors introduced. Pre-existing errors are acceptable.

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m backtesting.test 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add backtesting/backtesting.py backtesting/test/_test.py
git commit -m "♻️: Fix lint issues in Optuna integration"
```

(Skip this commit if no lint fixes were needed.)
