# Optuna Integration in backtesting.py

## Motivation

OpenBox als einziger Optimizer ist langsam, ineffizient und kann abgebrochene Optimierungen nicht wieder aufnehmen. Optuna bietet effizientere Sampler (TPE), native Persistenz via PostgreSQL/SQLite und nahtloses Resume abgebrochener Läufe.

Hauptnutzer ist ggi-cara, das Optuna bereits für CatBoost HPO verwendet und dieselbe PostgreSQL-Datenbank für beide Anwendungsfälle nutzen will.

## Scope

**In scope:**
- Optuna als neuer Optimizer via `method='optuna'`
- Optionale Persistenz (in-memory oder DB-backed)
- Resume abgebrochener Optimierungen
- Progress-Callback
- Parameter-Constraints (bestehender `constraint`-Parameter)

**Out of scope:**
- Parallelisierung (`n_workers`) für Optuna
- Outcome-Constraints für Optuna
- Optuna-Dashboard-Integration
- `init_configs` / `init_strategy` für Optuna (Optuna hat eigene Sampler-Logik)

## Design

### Neue Parameter in `optimize()`

```python
def optimize(self, *,
             # ... bestehende Parameter ...
             optuna_storage: Optional[str] = None,
             optuna_study_name: Optional[str] = None,
             optuna_sampler: Literal['tpe', 'cmaes', 'random'] = 'tpe',
             progress_callback: Optional[Callable[[int, int, float], None]] = None,
             **kwargs):
```

| Parameter | Typ | Default | Beschreibung |
|---|---|---|---|
| `optuna_storage` | `Optional[str]` | `None` | SQLAlchemy Connection-String (z.B. `postgresql://user:pass@host/db`). Ohne: in-memory, kein Resume. |
| `optuna_study_name` | `Optional[str]` | `None` | Expliziter Study-Name. Ohne: auto-generiert als `bt_{StrategyClass}_{hex(int(time.time()))[:6]}`. |
| `optuna_sampler` | `Literal['tpe', 'cmaes', 'random']` | `'tpe'` | Optuna Sampler-Algorithmus. |
| `progress_callback` | `Optional[Callable[[int, int, float], None]]` | `None` | Callback nach jedem Trial: `(current_trial, max_trials, current_value)`. `current_value` ist `float('nan')` bei geprunten Trials. |

### Methode `method='optuna'`

Erweitert den Dispatch am Ende von `optimize()`:

```python
if method == 'openbox':
    output = _optimize_openbox()
elif method == 'optuna':
    output = _optimize_optuna()
else:
    raise ValueError(f"Method should be 'openbox' or 'optuna', not {method!r}")
```

### Innere Funktion `_optimize_optuna()`

Analog zum bestehenden Pattern von `_optimize_openbox()` als innere Funktion in `optimize()`.

#### 1. Lazy Import

```python
def _optimize_optuna():
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Need package 'optuna'. Install it with: pip install optuna"
        ) from None
```

Optuna wird nicht als Dependency in `setup.py` aufgenommen. Der Aufrufer muss es selbst installieren.

#### 2. Sampler-Instanziierung

```python
sampler_map = {
    'tpe': optuna.samplers.TPESampler,
    'cmaes': optuna.samplers.CmaEsSampler,
    'random': optuna.samplers.RandomSampler,
}
sampler = sampler_map[optuna_sampler](seed=random_state)
```

#### 3. Study erstellen / laden

```python
study_name = optuna_study_name or f"bt_{self._strategy.__name__}_{hex(int(time.time()))[2:8]}"

study = optuna.create_study(
    study_name=study_name,
    storage=optuna_storage,
    direction="maximize",
    sampler=sampler,
    load_if_exists=True,
)
```

`load_if_exists=True` ermoeglicht Resume: Existiert die Study bereits in der Storage, wird sie geladen statt neu erstellt.

#### 4. Objective Function

```python
def objective(trial: optuna.trial.Trial):
    params = {}
    for key, values in kwargs.items():
        values = np.asarray(values)

        if values.dtype.kind in 'mM':  # datetime/timedelta
            values = values.astype(int)

        if len(values) == 1:
            params[key] = values[0]
        elif values.dtype.kind in 'iumM':
            params[key] = trial.suggest_int(key, int(values.min()), int(values.max()))
        elif values.dtype.kind == 'f' or (values.dtype.kind == 'O'
                                          and all(isinstance(v, Decimal) for v in values)):
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
```

Da `direction="maximize"` gesetzt ist, wird der Wert nicht negiert.

#### 5. Resume-Logik (max_tries als Gesamtzahl)

```python
existing_completed = len([
    t for t in study.trials
    if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
])
remaining = max(0, max_tries - existing_completed)
```

Falls `remaining == 0`: Kein neuer Optimize-Lauf, direkt zu Ergebnis-Extraktion.

#### 6. Progress Callback Wrapper

```python
def _optuna_callback(study, trial):
    if progress_callback is None:
        return
    value = trial.value if trial.value is not None else float('nan')
    progress_callback(len(study.trials), max_tries, value)
```

#### 7. Optimize-Aufruf

```python
study.optimize(objective, n_trials=remaining, callbacks=[_optuna_callback])
```

#### 8. Ergebnis-Extraktion

```python
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not completed_trials:
    raise ValueError('No admissible parameter combinations to test')

sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
best_configs = [t.params for t in sorted_trials[:return_configs]]

best_params = study.best_trial.params
stats = self.run(**best_params, fixed_params=fixed_params)

return (stats, best_configs)
```

#### 9. return_optimization

Wenn `return_optimization=True`, wird das `study`-Objekt als drittes Element zurueckgegeben:

```python
if return_optimization:
    return (stats, best_configs, study)
return (stats, best_configs)
```

Die bestehende Pruefung `if return_optimization and method != 'skopt'` wird angepasst auf `method not in ('skopt', 'optuna')`.

### Nutzung aus ggi-cara

```python
from apps.strategies.predictive_models import get_optuna_storage_url

bt = Backtest(data, MyStrategy, cash=10_000, commission=0.002)

def on_progress(current, total, value):
    logger.info(f"Trial {current}/{total}, value={value:.4f}")

stats, configs = bt.optimize(
    method='optuna',
    maximize='SQN',
    max_tries=200,
    optuna_storage=get_optuna_storage_url(),
    optuna_study_name=f"bt_{strategy}_{asset}_{horizon}",
    optuna_sampler='tpe',
    progress_callback=on_progress,
    fast=range(5, 30),
    slow=range(20, 60),
)
```

Resume nach Abbruch — exakt derselbe Aufruf. Optuna laedt die bestehende Study, zaehlt erledigte Trials, und fuehrt nur die verbleibenden aus.

### DB-Isolation gegenueber ggi-cara

Optuna isoliert Studies ueber den `study_name` innerhalb derselben Datenbank. Solange backtesting.py-Studies ein eigenes Prefix nutzen (`bt_...`) und ggi-cara's CatBoost-Studies ihr bestehendes Prefix behalten (`CatBoost_hyperparams_...`), gibt es keine Interferenz. Kein Schema-Trennung noetig.

### Edge Cases

| Szenario | Verhalten |
|---|---|
| `optuna_storage` ohne installiertes `psycopg2` | SQLAlchemy/Optuna wirft eigenen Fehler |
| `max_tries` bereits erreicht (Resume) | Kein neuer Optimize-Lauf, best result aus bestehenden Trials |
| Alle Trials pruned (Constraint) | `ValueError('No admissible parameter combinations to test')` |
| `optuna_study_name` ohne `optuna_storage` | Study-Name wird gesetzt aber nach Prozessende verloren (in-memory) |
| `progress_callback` mit `method='openbox'` | Wird ignoriert (nur Optuna) |
