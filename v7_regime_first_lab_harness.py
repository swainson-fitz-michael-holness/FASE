from __future__ import annotations

"""
V7 lab harness: regime-first cortical law discovery with explicit shield.

Core idea
---------
1. Freeze two teacher modes:
   - v6.2-style turbo / overclock teacher
   - v6.3-style stock / production teacher
2. Keep the autonomic shield explicit and unchanged.
3. Generate teacher-executed trajectories from a digital-twin server plant.
4. Label each trajectory state with a symbolic control regime.
5. Distill a symbolic regime law with FASE v21 (one-vs-rest regime heads).
6. Use per-regime action templates as the first white-box cortical policy.
7. Run a student-vs-teacher disagreement miner.
8. Replay hard states with higher weight and refit.
9. Gate the discovered regime law by agreement + compactness.

This file is a research harness, not a production service.
It assumes a local single-file FASE_v21_kfold.py that exposes run_fase_kfold(...).
"""

import importlib.util
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1. DIGITAL TWIN / PLANT
# ============================================================
@dataclass
class PlantConfig:
    ambient_temp: float = 22.0
    active_cooling_coeff: float = 0.12
    passive_exchange_coeff: float = 0.04
    capacity_scale: float = 30.0
    heat_clock_coeff: float = 1.0
    heat_work_coeff: float = 0.08
    data_complexity: float = 0.5
    failure_temp: float = 100.0
    failure_drop: float = 1500.0
    temp_sensor_noise: float = 0.0
    load_sensor_noise: float = 0.0
    cooling_sensor_noise: float = 0.0
    clock_sensor_noise: float = 0.0
    hidden_thermal_lag: float = 0.0


class ServerEnvironment:
    def __init__(self, cfg: PlantConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.npr = np.random.default_rng(seed)

        self.temp = 40.0
        self.clock_speed = 2.0
        self.cooling_valve = 0.5
        self.data_load = 50.0
        self.packets_dropped = 0.0
        self.total_processed = 0.0
        self.last_processed = 0.0

        self.is_alive = True
        self.ticks_survived = 0
        self.max_ticks = 100

        self.min_temp_seen = self.temp
        self.max_temp_seen = self.temp
        self.prev_temp = self.temp
        self.hidden_heat_state = 0.0

    def observe(self) -> Dict[str, float]:
        return {
            "temp": float(self.temp + self.npr.normal(0.0, self.cfg.temp_sensor_noise)),
            "clock_speed": float(self.clock_speed + self.npr.normal(0.0, self.cfg.clock_sensor_noise)),
            "cooling_valve": float(self.cooling_valve + self.npr.normal(0.0, self.cfg.cooling_sensor_noise)),
            "data_load": float(self.data_load + self.npr.normal(0.0, self.cfg.load_sensor_noise)),
            "packets_dropped": float(self.packets_dropped),
            "last_processed": float(self.last_processed),
            "ticks_survived": float(self.ticks_survived),
        }

    def step(self, throttle_data: bool, boost_cooling: bool, reduce_clock: bool) -> None:
        if not self.is_alive:
            return

        self.prev_temp = self.temp
        self.temp = float(np.clip(self.temp, 0.0, 150.0))

        if throttle_data:
            self.data_load = max(10.0, self.data_load - 20.0)
        else:
            self.data_load += self.rng.uniform(5.0, 25.0)

        if boost_cooling:
            self.cooling_valve = min(1.0, self.cooling_valve + 0.2)
        else:
            self.cooling_valve = max(0.1, self.cooling_valve - 0.1)

        if reduce_clock:
            self.clock_speed = max(1.0, self.clock_speed - 0.5)
        else:
            self.clock_speed = min(5.0, self.clock_speed + 0.2)

        processing_capacity = self.clock_speed * self.cfg.capacity_scale
        if self.data_load > processing_capacity:
            self.last_processed = processing_capacity
            self.packets_dropped += self.data_load - processing_capacity
        else:
            self.last_processed = self.data_load
        self.total_processed += self.last_processed

        instant_heat = (
            self.cfg.heat_clock_coeff * self.clock_speed
            + self.cfg.heat_work_coeff * self.last_processed * self.cfg.data_complexity
        )

        if self.cfg.hidden_thermal_lag > 0.0:
            self.hidden_heat_state = (
                (1.0 - self.cfg.hidden_thermal_lag) * instant_heat
                + self.cfg.hidden_thermal_lag * self.hidden_heat_state
            )
            heat_generation = self.hidden_heat_state
        else:
            heat_generation = instant_heat

        gap = max(0.0, self.temp - self.cfg.ambient_temp)
        active_cooling = self.cfg.active_cooling_coeff * self.cooling_valve * gap
        passive_exchange = self.cfg.passive_exchange_coeff * (self.cfg.ambient_temp - self.temp)

        self.temp += heat_generation - active_cooling + passive_exchange
        self.temp = float(np.clip(self.temp, 0.0, 150.0))

        self.min_temp_seen = min(self.min_temp_seen, self.temp)
        self.max_temp_seen = max(self.max_temp_seen, self.temp)

        self.ticks_survived += 1
        if self.temp >= self.cfg.failure_temp or self.packets_dropped > self.cfg.failure_drop:
            self.is_alive = False


# ============================================================
# 2. EXPLICIT AUTONOMIC SHIELD
# ============================================================
class AutonomicShield:
    def __init__(self):
        self.reset_episode()

    def reset_episode(self) -> None:
        self.level_counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}

    def assess(self, obs: Dict[str, float], predicted_temp: float, temp_slope: float) -> Tuple[str, float, float]:
        capacity = max(1.0, obs["clock_speed"] * 30.0)
        queue_ratio = obs["data_load"] / capacity
        drop_ratio = obs["packets_dropped"] / 1500.0

        risk_score = max(
            (obs["temp"] - 60.0) / 35.0,
            (predicted_temp - 62.0) / 28.0,
            temp_slope / 4.0,
            (queue_ratio - 0.95) / 0.50,
            drop_ratio / 0.75,
        )

        if obs["temp"] >= 95.0 or predicted_temp >= 92.0 or obs["packets_dropped"] >= 1200.0:
            level = "RED"
        elif obs["temp"] >= 84.0 or predicted_temp >= 82.0 or temp_slope >= 3.0 or obs["packets_dropped"] >= 800.0:
            level = "ORANGE"
        elif obs["temp"] >= 72.0 or predicted_temp >= 72.0 or temp_slope >= 1.75 or queue_ratio >= 1.10:
            level = "YELLOW"
        else:
            level = "GREEN"

        self.level_counts[level] += 1
        return level, float(risk_score), float(queue_ratio)

    def apply(
        self,
        obs: Dict[str, float],
        proposed_action: Tuple[bool, bool, bool],
        predicted_temp: float,
        temp_slope: float,
    ) -> Tuple[Tuple[bool, bool, bool], str, int]:
        throttle, boost, reduce_clock = proposed_action
        level, _, queue_ratio = self.assess(obs, predicted_temp, temp_slope)

        if level == "RED":
            return (True, True, True), level, 3

        if level == "ORANGE":
            final_action = (
                throttle or (queue_ratio > 0.95) or (predicted_temp > 86.0),
                True,
                True,
            )
            return final_action, level, 2 if final_action != proposed_action else 1

        if level == "YELLOW":
            final_action = (
                throttle or (queue_ratio > 1.05) or (predicted_temp > 78.0),
                True,
                reduce_clock or (predicted_temp > 80.0) or (temp_slope > 2.2),
            )
            return final_action, level, 1 if final_action != proposed_action else 0

        return proposed_action, level, 0


# ============================================================
# 3. LIGHTWEIGHT PREDICTOR FOR TEACHER CONTEXT
# ============================================================
class RollingTempPredictor:
    def __init__(self, window: int = 5):
        self.window = window
        self.temp_hist: List[float] = []

    def reset(self) -> None:
        self.temp_hist = []

    def predict(self, obs: Dict[str, float]) -> float:
        self.temp_hist.append(float(obs["temp"]))
        self.temp_hist = self.temp_hist[-self.window :]
        if len(self.temp_hist) < 2:
            return float(obs["temp"])
        slope = self.temp_hist[-1] - self.temp_hist[-2]
        return float(obs["temp"] + 1.5 * slope)


# ============================================================
# 4. FROZEN TEACHER POLICIES
# ============================================================
class TeacherTurbo62:
    """High-throughput teacher inspired by the v6.2 posture."""

    def propose(self, features: Dict[str, float]) -> Tuple[bool, bool, bool]:
        queue_ratio = features["queue_ratio"]
        pred_temp = features["pred_temp"]
        temp = features["temp"]
        slope = features["temp_slope"]

        throttle = queue_ratio > 1.18
        boost = (pred_temp > 71.0) or (temp > 70.0 and slope > 0.8)
        reduce_clock = (pred_temp > 83.5 and slope > 0.5) or (temp > 86.0)
        return bool(throttle), bool(boost), bool(reduce_clock)


class TeacherStock63:
    """Production / longevity teacher inspired by the v6.3 posture."""

    def propose(self, features: Dict[str, float]) -> Tuple[bool, bool, bool]:
        queue_ratio = features["queue_ratio"]
        pred_temp = features["pred_temp"]
        temp = features["temp"]
        slope = features["temp_slope"]
        d_amb = features["temp_minus_ambient"]

        throttle = (queue_ratio > 1.05) or (pred_temp > 78.0)
        boost = (pred_temp > 68.0) or (temp > 66.0) or (d_amb > 42.0)
        reduce_clock = (pred_temp > 76.0) or (temp > 74.0 and slope > 0.6)
        return bool(throttle), bool(boost), bool(reduce_clock)


# ============================================================
# 5. REGIME FACTORIZATION
# ============================================================
REGIMES: Tuple[str, ...] = (
    "relaxed",
    "efficiency",
    "thermal_caution",
    "congestion_caution",
    "brake",
)
REGIME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(REGIMES)}


def label_regime(
    feature_map: Dict[str, float],
    executed_action: Tuple[int, int, int],
    shield_level: str,
    override_code: int,
) -> str:
    throttle, boost, reduce_clock = executed_action
    temp = feature_map.get("temp", 0.0)
    pred_temp = feature_map.get("pred_temp", temp)
    queue_ratio = feature_map.get("queue_ratio", 0.0)

    if shield_level in ("ORANGE", "RED") or override_code >= 2 or (throttle and boost and reduce_clock):
        return "brake"
    if reduce_clock and boost:
        return "thermal_caution"
    if throttle and not reduce_clock:
        return "congestion_caution"
    if (
        (not throttle)
        and (not boost)
        and (not reduce_clock)
        and temp < 60.0
        and pred_temp < 68.0
        and queue_ratio < 0.95
    ):
        return "relaxed"
    return "efficiency"


# ============================================================
# 6. FEATURE ENGINEERING FOR DISTILLATION
# ============================================================
BASE_FEATURES: List[str] = [
    "temp",
    "clock_speed",
    "cooling_valve",
    "data_load",
    "packets_dropped",
    "last_processed",
    "pred_temp",
    "temp_slope",
    "queue_ratio",
    "drop_ratio",
    "temp_minus_ambient",
    "pred_minus_temp",
    "dist_to_yellow",
    "dist_to_orange",
    "dist_to_red",
]


@dataclass
class SensorScenario:
    name: str
    add_features: Tuple[str, ...] = ()
    remove_features: Tuple[str, ...] = ()


@dataclass
class TopologyScenario:
    name: str
    plant_cfg: PlantConfig


@dataclass
class DistillRecord:
    mode: str
    sensor_scenario: str
    topology_scenario: str
    episode: int
    tick: int
    feature_map: Dict[str, float]
    proposed_action: Tuple[int, int, int]
    executed_action: Tuple[int, int, int]
    shield_level: str
    override_code: int
    reward_like: float
    regime_label: str


def build_feature_map(obs: Dict[str, float], env: ServerEnvironment, pred_temp: float) -> Dict[str, float]:
    capacity = max(1.0, obs["clock_speed"] * env.cfg.capacity_scale)
    queue_ratio = obs["data_load"] / capacity
    temp_slope = obs["temp"] - env.prev_temp
    drop_ratio = obs["packets_dropped"] / env.cfg.failure_drop
    return {
        "temp": float(obs["temp"]),
        "clock_speed": float(obs["clock_speed"]),
        "cooling_valve": float(obs["cooling_valve"]),
        "data_load": float(obs["data_load"]),
        "packets_dropped": float(obs["packets_dropped"]),
        "last_processed": float(obs["last_processed"]),
        "pred_temp": float(pred_temp),
        "temp_slope": float(temp_slope),
        "queue_ratio": float(queue_ratio),
        "drop_ratio": float(drop_ratio),
        "temp_minus_ambient": float(obs["temp"] - env.cfg.ambient_temp),
        "pred_minus_temp": float(pred_temp - obs["temp"]),
        "dist_to_yellow": float(72.0 - pred_temp),
        "dist_to_orange": float(82.0 - pred_temp),
        "dist_to_red": float(92.0 - pred_temp),
        # optional synthetic / added sensors
        "temp_x_load": float(obs["temp"] * obs["data_load"]),
        "thermal_pressure": float((pred_temp - env.cfg.ambient_temp) * max(queue_ratio, 0.0)),
        "cooling_margin": float(1.0 - obs["cooling_valve"]),
        "throughput_margin": float(capacity - obs["data_load"]),
    }


def select_features(feature_map: Dict[str, float], scenario: SensorScenario) -> Dict[str, float]:
    out = dict(feature_map)
    for key in scenario.remove_features:
        out.pop(key, None)
    for key in scenario.add_features:
        if key not in out:
            raise KeyError(f"Requested added feature '{key}' is not computed.")
    return out


# ============================================================
# 7. DATA GENERATION
# ============================================================
def reward_like(env: ServerEnvironment) -> float:
    return float((env.last_processed / 150.0) - 0.08 * min(1.0, env.packets_dropped / 1500.0))


def generate_teacher_dataset(
    teacher_name: str,
    teacher_policy: Any,
    shield: AutonomicShield,
    sensor_scenarios: Sequence[SensorScenario],
    topology_scenarios: Sequence[TopologyScenario],
    episodes_per_combo: int = 12,
    horizon: int = 100,
    seed: int = 42,
) -> List[DistillRecord]:
    records: List[DistillRecord] = []
    seed_ctr = seed

    for topo in topology_scenarios:
        for sensor_sc in sensor_scenarios:
            for ep in range(episodes_per_combo):
                env = ServerEnvironment(cfg=topo.plant_cfg, seed=seed_ctr)
                env.max_ticks = horizon
                predictor = RollingTempPredictor(window=5)
                predictor.reset()
                shield.reset_episode()
                seed_ctr += 1

                while env.is_alive and env.ticks_survived < env.max_ticks:
                    obs = env.observe()
                    pred_temp = predictor.predict(obs)
                    fmap = build_feature_map(obs, env, pred_temp)
                    fmap = select_features(fmap, sensor_sc)

                    proposed = teacher_policy.propose(fmap)
                    executed, shield_level, override_code = shield.apply(
                        obs,
                        proposed,
                        predicted_temp=fmap["pred_temp"],
                        temp_slope=fmap["temp_slope"],
                    )
                    env.step(*executed)
                    regime_label = label_regime(fmap, tuple(int(x) for x in executed), shield_level, int(override_code))

                    records.append(
                        DistillRecord(
                            mode=teacher_name,
                            sensor_scenario=sensor_sc.name,
                            topology_scenario=topo.name,
                            episode=ep,
                            tick=env.ticks_survived,
                            feature_map=fmap,
                            proposed_action=tuple(int(x) for x in proposed),
                            executed_action=tuple(int(x) for x in executed),
                            shield_level=shield_level,
                            override_code=int(override_code),
                            reward_like=reward_like(env),
                            regime_label=regime_label,
                        )
                    )
    return records


# ============================================================
# 8. FASE v21 ADAPTER
# ============================================================
class FASEV21Adapter:
    def __init__(self, module_path: str):
        self.module_path = Path(module_path)
        if not self.module_path.exists():
            raise FileNotFoundError(f"FASE v21 module not found: {self.module_path}")
        module_name = f"fase_v21_user_module_{self.module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(self.module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import module from {self.module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.module = module
        if not hasattr(module, "run_fase_kfold"):
            raise AttributeError("FASE v21 module must expose run_fase_kfold")

    def fit_binary(self, X: np.ndarray, y: np.ndarray, config_patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = dict(getattr(self.module, "CONFIG"))
        if config_patch:
            for k, v in config_patch.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    merged = dict(cfg[k])
                    merged.update(v)
                    cfg[k] = merged
                else:
                    cfg[k] = v
        return self.module.run_fase_kfold(X, y, Sigma=None, K=cfg.get("K_FOLDS", 5), seed=cfg.get("SEEDS", [42])[0], config=cfg)

    @staticmethod
    def raw_score(report: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        model = report["model"]
        return model.predict(X).ravel()


# ============================================================
# 9. MATRIX HELPERS
# ============================================================
def records_to_design_matrix(records: Sequence[DistillRecord], feature_names: List[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for r in records:
        row: List[float] = []
        for nm in feature_names:
            if nm.endswith("__present"):
                base = nm[:-10]
                row.append(1.0 if base in r.feature_map else 0.0)
            else:
                row.append(float(r.feature_map.get(nm, 0.0)))
        rows.append(row)
    return np.array(rows, dtype=float)


def collect_feature_names(records: Sequence[DistillRecord]) -> List[str]:
    raw_feature_names = sorted({k for r in records for k in r.feature_map.keys()})
    return raw_feature_names + [f"{nm}__present" for nm in raw_feature_names]


def hamming_bits(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return int(a[0] != b[0]) + int(a[1] != b[1]) + int(a[2] != b[2])


# ============================================================
# 10. REGIME-FIRST STUDENT
# ============================================================
@dataclass
class CompactnessGate:
    min_regime_accuracy: float = 0.90
    min_exact_agreement: float = 0.90
    max_avg_hamming: float = 0.15
    max_mean_num_ops: float = 12.0
    max_mean_median_bits: float = 128.0
    min_mean_template_purity: float = 0.90


class RegimeFirstStudent:
    def __init__(
        self,
        feature_names: List[str],
        regime_reports: Dict[str, Dict[str, Any]],
        action_templates: Dict[str, Tuple[int, int, int]],
        template_purity: Dict[str, float],
    ):
        self.feature_names = feature_names
        self.regime_reports = regime_reports
        self.action_templates = action_templates
        self.template_purity = template_purity

    def predict_regime_scores(self, adapter: FASEV21Adapter, records: Sequence[DistillRecord]) -> np.ndarray:
        X = records_to_design_matrix(records, self.feature_names)
        score_cols = []
        for regime in REGIMES:
            score_cols.append(adapter.raw_score(self.regime_reports[regime], X))
        return np.stack(score_cols, axis=1)

    def predict_regimes(self, adapter: FASEV21Adapter, records: Sequence[DistillRecord]) -> List[str]:
        scores = self.predict_regime_scores(adapter, records)
        ids = np.argmax(scores, axis=1)
        return [REGIMES[int(i)] for i in ids]

    def predict_actions(self, adapter: FASEV21Adapter, records: Sequence[DistillRecord]) -> np.ndarray:
        pred_regimes = self.predict_regimes(adapter, records)
        actions = [self.action_templates[r] for r in pred_regimes]
        return np.array(actions, dtype=int)


def infer_action_templates(records: Sequence[DistillRecord]) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, float]]:
    by_regime: Dict[str, Counter] = defaultdict(Counter)
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        by_regime[r.regime_label][tuple(r.executed_action)] += 1
        counts[r.regime_label] += 1

    templates: Dict[str, Tuple[int, int, int]] = {}
    purity: Dict[str, float] = {}
    global_majority = Counter(tuple(r.executed_action) for r in records).most_common(1)[0][0]

    for regime in REGIMES:
        if counts.get(regime, 0) == 0:
            templates[regime] = global_majority
            purity[regime] = 0.0
            continue
        majority_action, freq = by_regime[regime].most_common(1)[0]
        templates[regime] = majority_action
        purity[regime] = float(freq) / float(counts[regime])
    return templates, purity


def compactness_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    og_stability = report.get("og_stability", {})
    og_min_bits = report.get("og_min_bits", {})
    num_ops = float(len(og_stability))
    median_bits = float(np.median(list(og_min_bits.values()))) if og_min_bits else 0.0
    return {
        "num_consensus_ops": num_ops,
        "median_op_bits": median_bits,
        "R2_OOF": float(report.get("R2_oof", np.nan)),
        "MSE_OOF": float(report.get("MSE_oof", np.nan)),
    }


def fit_regime_student(
    adapter: FASEV21Adapter,
    train_records: Sequence[DistillRecord],
    config_patch: Optional[Dict[str, Any]] = None,
) -> RegimeFirstStudent:
    feature_names = collect_feature_names(train_records)
    X = records_to_design_matrix(train_records, feature_names)
    regime_reports: Dict[str, Dict[str, Any]] = {}

    for regime in REGIMES:
        y = np.array([1.0 if r.regime_label == regime else 0.0 for r in train_records], dtype=float)
        regime_reports[regime] = adapter.fit_binary(X, y, config_patch=config_patch)

    action_templates, template_purity = infer_action_templates(train_records)
    return RegimeFirstStudent(feature_names, regime_reports, action_templates, template_purity)


# ============================================================
# 11. DISAGREEMENT MINER + WEIGHTED REPLAY
# ============================================================
@dataclass
class HardExample:
    index: int
    score: float
    pred_regime: str
    true_regime: str
    pred_action: Tuple[int, int, int]
    true_action: Tuple[int, int, int]
    mismatch_bits: int


def regime_accuracy(true_regimes: Sequence[str], pred_regimes: Sequence[str]) -> float:
    return float(np.mean([int(t == p) for t, p in zip(true_regimes, pred_regimes)])) if true_regimes else 0.0


def evaluate_regime_student(
    adapter: FASEV21Adapter,
    student: RegimeFirstStudent,
    records: Sequence[DistillRecord],
    gate: Optional[CompactnessGate] = None,
) -> Dict[str, Any]:
    pred_regimes = student.predict_regimes(adapter, records)
    pred_actions = student.predict_actions(adapter, records)
    true_regimes = [r.regime_label for r in records]
    true_actions = np.array([r.executed_action for r in records], dtype=int)

    exact = np.all(pred_actions == true_actions, axis=1)
    hammings = np.sum(pred_actions != true_actions, axis=1)

    regime_reports = {regime: compactness_from_report(student.regime_reports[regime]) for regime in REGIMES}
    mean_num_ops = float(np.mean([v["num_consensus_ops"] for v in regime_reports.values()]))
    mean_median_bits = float(np.mean([v["median_op_bits"] for v in regime_reports.values()]))
    mean_template_purity = float(np.mean(list(student.template_purity.values())))

    metrics = {
        "regime_accuracy": regime_accuracy(true_regimes, pred_regimes),
        "exact_agreement": float(np.mean(exact)),
        "avg_hamming": float(np.mean(hammings)),
        "n_eval": float(len(records)),
        "mean_template_purity": mean_template_purity,
        "mean_num_ops": mean_num_ops,
        "mean_median_bits": mean_median_bits,
    }

    compactness_gate = None
    if gate is not None:
        compactness_gate = {
            "regime_accuracy_pass": metrics["regime_accuracy"] >= gate.min_regime_accuracy,
            "exact_agreement_pass": metrics["exact_agreement"] >= gate.min_exact_agreement,
            "avg_hamming_pass": metrics["avg_hamming"] <= gate.max_avg_hamming,
            "mean_num_ops_pass": metrics["mean_num_ops"] <= gate.max_mean_num_ops,
            "mean_median_bits_pass": metrics["mean_median_bits"] <= gate.max_mean_median_bits,
            "template_purity_pass": metrics["mean_template_purity"] >= gate.min_mean_template_purity,
        }
        compactness_gate["all_pass"] = all(compactness_gate.values())

    return {
        "metrics": metrics,
        "regime_reports": regime_reports,
        "action_templates": {k: list(v) for k, v in student.action_templates.items()},
        "template_purity": student.template_purity,
        "compactness_gate": compactness_gate,
    }


def mine_disagreement(
    adapter: FASEV21Adapter,
    student: RegimeFirstStudent,
    records: Sequence[DistillRecord],
    top_frac: float = 0.20,
    replay_strength: int = 4,
) -> Tuple[List[DistillRecord], Dict[str, Any]]:
    pred_regimes = student.predict_regimes(adapter, records)
    pred_actions = student.predict_actions(adapter, records)

    hard_examples: List[HardExample] = []
    for i, (rec, preg, pact) in enumerate(zip(records, pred_regimes, pred_actions)):
        true_action = tuple(int(x) for x in rec.executed_action)
        pred_action = tuple(int(x) for x in pact)
        mism = hamming_bits(pred_action, true_action)
        regime_mism = int(preg != rec.regime_label)

        boundary_bonus = 0.0
        if rec.shield_level == "YELLOW":
            boundary_bonus += 0.5
        elif rec.shield_level in ("ORANGE", "RED"):
            boundary_bonus += 1.0

        score = (
            1.0 * mism
            + 0.75 * regime_mism
            + 0.35 * rec.override_code
            + 0.15 * max(rec.reward_like, 0.0)
            + boundary_bonus
        )

        if score > 0.0:
            hard_examples.append(
                HardExample(
                    index=i,
                    score=float(score),
                    pred_regime=preg,
                    true_regime=rec.regime_label,
                    pred_action=pred_action,
                    true_action=true_action,
                    mismatch_bits=mism,
                )
            )

    if not hard_examples:
        return list(records), {
            "num_hard_examples": 0,
            "mean_hard_score": 0.0,
            "replay_size": len(records),
        }

    hard_examples.sort(key=lambda h: h.score, reverse=True)
    k = max(1, int(len(records) * top_frac))
    chosen = hard_examples[:k]

    replay_records: List[DistillRecord] = list(records)
    for hx in chosen:
        rec = records[hx.index]
        copies = 1 + min(replay_strength, int(np.ceil(hx.score)))
        replay_records.extend([rec] * copies)

    stats = {
        "num_hard_examples": len(chosen),
        "mean_hard_score": float(np.mean([h.score for h in chosen])),
        "max_hard_score": float(max(h.score for h in chosen)),
        "replay_size": len(replay_records),
        "sample_hard_examples": [
            {
                "score": h.score,
                "pred_regime": h.pred_regime,
                "true_regime": h.true_regime,
                "pred_action": list(h.pred_action),
                "true_action": list(h.true_action),
                "mismatch_bits": h.mismatch_bits,
            }
            for h in chosen[:10]
        ],
    }
    return replay_records, stats


# ============================================================
# 12. FULL V7 LAB HARNESS
# ============================================================
def split_records(records: Sequence[DistillRecord], frac: float = 0.7) -> Tuple[List[DistillRecord], List[DistillRecord]]:
    n = len(records)
    k = int(frac * n)
    return list(records[:k]), list(records[k:])


def save_records_jsonl(records: Sequence[DistillRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")


def save_pickle_checkpoint(obj: Any, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_pickle_checkpoint(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def try_save_pickle_checkpoint(obj: Any, path: str, label: str = "checkpoint") -> bool:
    try:
        save_pickle_checkpoint(obj, path)
        return True
    except Exception as exc:
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Warning: could not save {label} to {path}: {exc}")
        return False


def resolve_fase_module_path(preferred_path: str) -> str:
    preferred = Path(preferred_path)
    if preferred.exists():
        return str(preferred)

    fallback = Path("FASE_v21.py")
    if fallback.exists():
        print(
            f"Warning: requested {preferred_path} not found. "
            f"Falling back to {fallback}."
        )
        return str(fallback)

    raise FileNotFoundError(
        f"Could not find requested FASE module '{preferred_path}' "
        "or fallback 'FASE_v21.py'."
    )


def run_v7_lab_harness(
    fase_v21_path: str,
    output_dir: str = "v7_lab_outputs",
    episodes_per_combo: int = 10,
    horizon: int = 100,
    replay_rounds: int = 2,
    reuse_checkpoints: bool = True,
    config_patch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    sensor_scenarios = [
        SensorScenario(name="base"),
        SensorScenario(name="extra_thermal_features", add_features=("temp_x_load", "thermal_pressure", "cooling_margin")),
        SensorScenario(name="reduced_sensor_set", remove_features=("packets_dropped", "last_processed")),
    ]

    topology_scenarios = [
        TopologyScenario(name="base_topology", plant_cfg=PlantConfig()),
        TopologyScenario(name="denser_compute", plant_cfg=PlantConfig(heat_work_coeff=0.095, capacity_scale=28.0)),
        TopologyScenario(name="better_cooling", plant_cfg=PlantConfig(active_cooling_coeff=0.14)),
        TopologyScenario(name="thermal_lag", plant_cfg=PlantConfig(hidden_thermal_lag=0.40)),
        TopologyScenario(name="warmer_room", plant_cfg=PlantConfig(ambient_temp=28.0)),
    ]

    gate = CompactnessGate()
    shield = AutonomicShield()

    turbo_jsonl = os.path.join(output_dir, "teacher_v62_turbo.jsonl")
    stock_jsonl = os.path.join(output_dir, "teacher_v63_stock.jsonl")
    turbo_pkl = os.path.join(output_dir, "teacher_v62_turbo.pkl")
    stock_pkl = os.path.join(output_dir, "teacher_v63_stock.pkl")

    if reuse_checkpoints and os.path.exists(turbo_pkl) and os.path.exists(stock_pkl):
        turbo_records = load_pickle_checkpoint(turbo_pkl)
        stock_records = load_pickle_checkpoint(stock_pkl)
        print("Loaded teacher trajectory checkpoints.")
    else:
        turbo_records = generate_teacher_dataset(
            teacher_name="v6_2_turbo",
            teacher_policy=TeacherTurbo62(),
            shield=shield,
            sensor_scenarios=sensor_scenarios,
            topology_scenarios=topology_scenarios,
            episodes_per_combo=episodes_per_combo,
            horizon=horizon,
            seed=101,
        )
        stock_records = generate_teacher_dataset(
            teacher_name="v6_3_stock",
            teacher_policy=TeacherStock63(),
            shield=shield,
            sensor_scenarios=sensor_scenarios,
            topology_scenarios=topology_scenarios,
            episodes_per_combo=episodes_per_combo,
            horizon=horizon,
            seed=202,
        )
        save_records_jsonl(turbo_records, turbo_jsonl)
        save_records_jsonl(stock_records, stock_jsonl)
        save_pickle_checkpoint(turbo_records, turbo_pkl)
        save_pickle_checkpoint(stock_records, stock_pkl)

    adapter = FASEV21Adapter(resolve_fase_module_path(fase_v21_path))
    train_config_patch = {
        "K_FOLDS": 5,
        "COMPARE_WITH_PYSR": False,
        "OGSET": {
            "bag_boots": 8,
            "final_min_freq": 0.60,
            "final_min_sign_stab": 0.90,
            "final_min_bits": 6.0,
        },
    }
    if config_patch:
        for key, value in config_patch.items():
            if isinstance(value, dict) and isinstance(train_config_patch.get(key), dict):
                merged = dict(train_config_patch[key])
                merged.update(value)
                train_config_patch[key] = merged
            else:
                train_config_patch[key] = value

    final_results: Dict[str, Any] = {}
    for mode_name, records in [("v6_2_turbo", turbo_records), ("v6_3_stock", stock_records)]:
        mode_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        train_records, eval_records = split_records(records, frac=0.7)

        current_train = list(train_records)
        mining_history: List[Dict[str, Any]] = []
        student: Optional[RegimeFirstStudent] = None

        for rnd in range(replay_rounds + 1):
            ckpt_path = os.path.join(mode_dir, f"student_round_{rnd}.pkl")
            if reuse_checkpoints and os.path.exists(ckpt_path):
                try:
                    student = load_pickle_checkpoint(ckpt_path)
                except Exception as exc:
                    print(f"Warning: could not load student checkpoint {ckpt_path}: {exc}")
                    student = fit_regime_student(adapter, current_train, config_patch=train_config_patch)
                    try_save_pickle_checkpoint(student, ckpt_path, label=f"{mode_name} student round {rnd}")
            else:
                student = fit_regime_student(adapter, current_train, config_patch=train_config_patch)
                try_save_pickle_checkpoint(student, ckpt_path, label=f"{mode_name} student round {rnd}")

            train_eval = evaluate_regime_student(adapter, student, train_records, gate=gate)
            with open(os.path.join(mode_dir, f"round_{rnd}_train_eval.json"), "w", encoding="utf-8") as f:
                json.dump(train_eval, f, indent=2)

            if rnd < replay_rounds:
                replay_train, mine_stats = mine_disagreement(adapter, student, train_records, top_frac=0.20, replay_strength=4)
                current_train = replay_train
                mining_history.append(mine_stats)
                with open(os.path.join(mode_dir, f"round_{rnd}_mine_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(mine_stats, f, indent=2)

        assert student is not None
        eval_report = evaluate_regime_student(adapter, student, eval_records, gate=gate)
        eval_report["mining_history"] = mining_history
        final_results[mode_name] = eval_report

        with open(os.path.join(mode_dir, "final_eval_report.json"), "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

    summary_path = os.path.join(output_dir, "v7_lab_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n================ V7 LAB SUMMARY ================")
    for mode_name, report in final_results.items():
        m = report["metrics"]
        g = report.get("compactness_gate") or {}
        print(
            f"{mode_name}: "
            f"regime_acc={m['regime_accuracy']:.4f} | "
            f"exact_agreement={m['exact_agreement']:.4f} | "
            f"avg_hamming={m['avg_hamming']:.4f} | "
            f"template_purity={m['mean_template_purity']:.4f} | "
            f"mean_ops={m['mean_num_ops']:.2f} | "
            f"mean_bits={m['mean_median_bits']:.2f} | "
            f"gate_all_pass={g.get('all_pass', False)}"
        )

    return final_results


if __name__ == "__main__":
    fase_v21_path = "FASE_v21_kfold.py"
    run_v7_lab_harness(
        fase_v21_path=fase_v21_path,
        output_dir="v7_lab_outputs",
        episodes_per_combo=8,
        horizon=100,
        replay_rounds=2,
        reuse_checkpoints=True,
    )
