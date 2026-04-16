from __future__ import annotations

"""
Pre-v7 pipeline: distill explicit cortical laws from frozen teacher trajectories
while keeping the autonomic shield explicit.

Design goals
------------
1. Freeze two teacher modes:
   - v6.2-style turbo/performance teacher
   - v6.3-style stock/production teacher
2. Generate trajectory datasets from a digital-twin server environment.
3. Apply sensor additions and topology changes to test transfer and recovery.
4. Distill one explicit white-box cortical law per control bit using FASE v21.
5. Keep the autonomic shield explicit at runtime.
6. Report not only reward-like metrics, but agreement, compactness, and transfer.

This file is intentionally a research harness, not a production service.
It assumes you have the uploaded single-file FASE_v21_kfold.py available locally.
"""

import importlib.util
import sys
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

        # workload dynamics
        if throttle_data:
            self.data_load = max(10.0, self.data_load - 20.0)
        else:
            self.data_load += self.rng.uniform(5.0, 25.0)

        # cooling actuator dynamics
        if boost_cooling:
            self.cooling_valve = min(1.0, self.cooling_valve + 0.2)
        else:
            self.cooling_valve = max(0.1, self.cooling_valve - 0.1)

        # clock actuator dynamics
        if reduce_clock:
            self.clock_speed = max(1.0, self.clock_speed - 0.5)
        else:
            self.clock_speed = min(5.0, self.clock_speed + 0.2)

        # processing and drops
        processing_capacity = self.clock_speed * self.cfg.capacity_scale
        if self.data_load > processing_capacity:
            self.last_processed = processing_capacity
            self.packets_dropped += (self.data_load - processing_capacity)
        else:
            self.last_processed = self.data_load
        self.total_processed += self.last_processed

        # heat generation
        instant_heat = (
            self.cfg.heat_clock_coeff * self.clock_speed
            + self.cfg.heat_work_coeff * self.last_processed * self.cfg.data_complexity
        )

        # optional hidden lag state for topology-shift scenarios
        if self.cfg.hidden_thermal_lag > 0.0:
            self.hidden_heat_state = (
                (1.0 - self.cfg.hidden_thermal_lag) * instant_heat
                + self.cfg.hidden_thermal_lag * self.hidden_heat_state
            )
            heat_generation = self.hidden_heat_state
        else:
            heat_generation = instant_heat

        # ambient-coupled cooling
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
# 2. EXPLICIT AUTONOMIC SHIELD (kept explicit)
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
    """
    Lightweight stand-in for the temporal predictor when generating distillation traces.
    We keep this simple because the point of pre-v7 is law distillation, not predictor SOTA.
    """

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
    """
    High-throughput teacher inspired by the v6.2 operating posture.
    Aggressive near capacity; relies more on shield at the boundary.
    """

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
    """
    Production / longevity teacher inspired by the v6.3 posture.
    More conservative and agreement-oriented.
    """

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
# 5. FEATURE ENGINEERING FOR DISTILLATION
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
        # optional synthetic/additional sensors
        "temp_x_load": float(obs["temp"] * obs["data_load"]),
        "thermal_pressure": float((pred_temp - env.cfg.ambient_temp) * max(queue_ratio, 0.0)),
        "cooling_margin": float(1.0 - obs["cooling_valve"]),
        "throughput_margin": float(capacity - obs["data_load"]),
    }


# ============================================================
# 6. DATA GENERATION
# ============================================================
def reward_like(env: ServerEnvironment) -> float:
    return float((env.last_processed / 150.0) - 0.08 * min(1.0, env.packets_dropped / 1500.0))


def select_features(feature_map: Dict[str, float], scenario: SensorScenario) -> Dict[str, float]:
    out = dict(feature_map)
    for key in scenario.remove_features:
        out.pop(key, None)
    for key in scenario.add_features:
        if key not in out:
            raise KeyError(f"Requested added feature '{key}' is not computed.")
    return out


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
                        )
                    )
    return records


# ============================================================
# 7. FASE v21 ADAPTER
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

    def fit_bit(self, X: np.ndarray, y: np.ndarray, config_patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = dict(getattr(self.module, "CONFIG"))
        if config_patch:
            for k, v in config_patch.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    merged = dict(cfg[k])
                    merged.update(v)
                    cfg[k] = merged
                else:
                    cfg[k] = v
        report = self.module.run_fase_kfold(X, y, Sigma=None, K=cfg.get("K_FOLDS", 5), seed=cfg.get("SEEDS", [42])[0], config=cfg)
        return report

    @staticmethod
    def predict_bit(report: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        model = report["model"]
        yhat = model.predict(X).ravel()
        return (yhat >= 0.5).astype(int)


# ============================================================
# 8. DISTILLATION EXPERIMENTS
# ============================================================
def records_to_matrix(records: Sequence[DistillRecord]) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No records provided.")

    # Union of all feature names across heterogeneous sensor scenarios
    raw_feature_names = sorted({k for r in records for k in r.feature_map.keys()})

    rows = []
    for r in records:
        row = []

        # Value channels
        for nm in raw_feature_names:
            row.append(float(r.feature_map.get(nm, 0.0)))

        # Presence-mask channels
        for nm in raw_feature_names:
            row.append(1.0 if nm in r.feature_map else 0.0)

        rows.append(row)

    feature_names = raw_feature_names + [f"{nm}__present" for nm in raw_feature_names]

    X = np.array(rows, dtype=float)
    y_throttle = np.array([r.executed_action[0] for r in records], dtype=float)
    y_boost = np.array([r.executed_action[1] for r in records], dtype=float)
    y_reduce = np.array([r.executed_action[2] for r in records], dtype=float)

    return X, feature_names, y_throttle, y_boost, y_reduce


def action_agreement(y_true_bits: np.ndarray, y_pred_bits: np.ndarray) -> Dict[str, float]:
    exact = np.all(y_true_bits == y_pred_bits, axis=1)
    hamming = np.sum(y_true_bits != y_pred_bits, axis=1)
    return {
        "exact_agreement": float(np.mean(exact)),
        "avg_hamming": float(np.mean(hamming)),
    }


def compactness_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    og_stability = report.get("og_stability", {})
    og_min_bits = report.get("og_min_bits", {})
    num_ops = float(len(og_stability))
    median_bits = float(np.median(list(og_min_bits.values()))) if og_min_bits else 0.0
    return {
        "num_consensus_ops": num_ops,
        "median_op_bits": median_bits,
    }


def fit_distilled_laws(
    adapter: FASEV21Adapter,
    train_records: Sequence[DistillRecord],
    config_patch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    X, feature_names, y_t, y_b, y_r = records_to_matrix(train_records)
    reports = {
        "throttle": adapter.fit_bit(X, y_t, config_patch=config_patch),
        "boost": adapter.fit_bit(X, y_b, config_patch=config_patch),
        "reduce_clock": adapter.fit_bit(X, y_r, config_patch=config_patch),
    }
    return {"reports": reports, "feature_names": feature_names}


def predict_distilled_actions(adapter: FASEV21Adapter, distilled: Dict[str, Any], records: Sequence[DistillRecord]) -> np.ndarray:
    feature_names = distilled["feature_names"]
    X = np.array([[r.feature_map[nm] for nm in feature_names] for r in records], dtype=float)
    yhat_t = adapter.predict_bit(distilled["reports"]["throttle"], X)
    yhat_b = adapter.predict_bit(distilled["reports"]["boost"], X)
    yhat_r = adapter.predict_bit(distilled["reports"]["reduce_clock"], X)
    return np.stack([yhat_t, yhat_b, yhat_r], axis=1)


def evaluate_distillation(
    adapter: FASEV21Adapter,
    distilled: Dict[str, Any],
    eval_records: Sequence[DistillRecord],
) -> Dict[str, Any]:
    y_true = np.array([r.executed_action for r in eval_records], dtype=int)
    y_pred = predict_distilled_actions(adapter, distilled, eval_records)

    metrics = action_agreement(y_true, y_pred)
    metrics["n_eval"] = float(len(eval_records))

    bit_reports = {}
    for bit_name in ("throttle", "boost", "reduce_clock"):
        bit_reports[bit_name] = compactness_from_report(distilled["reports"][bit_name])
        bit_reports[bit_name]["R2_OOF"] = float(distilled["reports"][bit_name].get("R2_oof", np.nan))
        bit_reports[bit_name]["MSE_OOF"] = float(distilled["reports"][bit_name].get("MSE_oof", np.nan))

    return {"metrics": metrics, "bit_reports": bit_reports}


# ============================================================
# 9. FULL PRE-v7 HARNESS
# ============================================================
def split_records(records: Sequence[DistillRecord], frac: float = 0.7) -> Tuple[List[DistillRecord], List[DistillRecord]]:
    n = len(records)
    k = int(frac * n)
    return list(records[:k]), list(records[k:])


def save_records_jsonl(records: Sequence[DistillRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            row = asdict(rec)
            f.write(json.dumps(row) + "\n")


def run_pre_v7_pipeline(
    fase_v21_path: str,
    output_dir: str = "pre_v7_outputs",
    episodes_per_combo: int = 10,
    horizon: int = 100,
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

    shield = AutonomicShield()

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

    save_records_jsonl(turbo_records, os.path.join(output_dir, "teacher_v62_turbo.jsonl"))
    save_records_jsonl(stock_records, os.path.join(output_dir, "teacher_v63_stock.jsonl"))

    adapter = FASEV21Adapter(fase_v21_path)

    results: Dict[str, Any] = {}
    for mode_name, records in [("v6_2_turbo", turbo_records), ("v6_3_stock", stock_records)]:
        train_records, eval_records = split_records(records, frac=0.7)

        distilled = fit_distilled_laws(
            adapter,
            train_records,
            config_patch={
                "K_FOLDS": 5,
                "COMPARE_WITH_PYSR": False,
                "OGSET": {
                    "bag_boots": 8,
                    "final_min_freq": 0.60,
                    "final_min_sign_stab": 0.90,
                    "final_min_bits": 6.0,
                },
            },
        )
        eval_report = evaluate_distillation(adapter, distilled, eval_records)
        results[mode_name] = eval_report

    with open(os.path.join(output_dir, "pre_v7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ PRE-v7 SUMMARY ================")
    for mode_name, report in results.items():
        m = report["metrics"]
        print(
            f"{mode_name}: "
            f"exact_agreement={m['exact_agreement']:.4f} | "
            f"avg_hamming={m['avg_hamming']:.4f} | "
            f"n_eval={int(m['n_eval'])}"
        )
        for bit_name, bref in report["bit_reports"].items():
            print(
                f"  {bit_name:>12s}: "
                f"OOF_R2={bref['R2_OOF']:.4f} | "
                f"OOF_MSE={bref['MSE_OOF']:.6f} | "
                f"num_ops={int(bref['num_consensus_ops'])} | "
                f"median_bits={bref['median_op_bits']:.2f}"
            )

    return results


if __name__ == "__main__":
    # Edit this path to wherever you place the uploaded FASE_v21_kfold.py file.
    fase_v21_path = "FASE_v21.py"
    run_pre_v7_pipeline(
        fase_v21_path=fase_v21_path,
        output_dir="pre_v7_outputs",
        episodes_per_combo=8,
        horizon=100,
    )
