#!/usr/bin/env python3
"""
FASE-v24 Gate 1: Operator Genesis from Repeated Anomalous Transformations
=========================================================================

Goal
----
Infer a new typed rewrite operator omega directly from repeated concrete
transformations, without selecting the operator from a supplied meta-alphabet.

Pipeline
--------
concrete anomalous transformations
    -> typed least-general generalization (anti-unification)
    -> linked variable abstraction across LHS/RHS
    -> executable rewrite schema omega
    -> held-out structural validation
    -> active perturbation / counterfactual queries
    -> stability + compression + false-promotion gate

The implementation is intentionally domain-general at the representation level:
all domains are encoded as typed term trees. The included benchmark tasks cover
algebra, lambda calculus, chemistry, molecular biology, generic structures, and
null/nonstationary controls.

Dependencies: Python 3.10+, numpy. pandas is optional (CSV uses stdlib).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ======================================================================================
# Typed term language
# ======================================================================================

@dataclass(frozen=True)
class Term:
    op: str
    typ: str
    children: Tuple["Term", ...] = field(default_factory=tuple)
    value: Optional[Any] = None

    def is_meta(self) -> bool:
        return self.op == "$"

    def is_atom(self) -> bool:
        return self.op == "Atom"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def canonical(self) -> str:
        if self.is_meta():
            return f"?{self.value}:{self.typ}"
        if self.is_atom():
            return f"{self.typ}[{json.dumps(self.value, sort_keys=True)}]"
        if not self.children:
            return f"{self.op}:{self.typ}"
        return f"{self.op}:{self.typ}(" + ",".join(c.canonical() for c in self.children) + ")"

    def pretty(self) -> str:
        if self.is_meta():
            return f"?{self.value}:{self.typ}"
        if self.is_atom():
            return str(self.value)
        if not self.children:
            return self.op
        return f"{self.op}(" + ", ".join(c.pretty() for c in self.children) + ")"


def Atom(value: Any, typ: str) -> Term:
    return Term("Atom", typ, (), value)


def Node(op: str, typ: str, *children: Term) -> Term:
    return Term(op, typ, tuple(children), None)


def Meta(name: str, typ: str) -> Term:
    return Term("$", typ, (), name)


@dataclass(frozen=True)
class Example:
    lhs: Term
    rhs: Term


@dataclass(frozen=True)
class RewriteSchema:
    lhs: Term
    rhs: Term

    def alpha_normalized(self) -> "RewriteSchema":
        mapping: Dict[str, str] = {}

        def ren(t: Term) -> Term:
            if t.is_meta():
                old = str(t.value)
                if old not in mapping:
                    mapping[old] = f"v{len(mapping)}"
                return Meta(mapping[old], t.typ)
            return Term(t.op, t.typ, tuple(ren(c) for c in t.children), t.value)

        return RewriteSchema(ren(self.lhs), ren(self.rhs))

    def canonical(self) -> str:
        s = self.alpha_normalized()
        return f"{s.lhs.canonical()} -> {s.rhs.canonical()}"

    def pretty(self) -> str:
        s = self.alpha_normalized()
        return f"{s.lhs.pretty()}  ->  {s.rhs.pretty()}"

    def variables_lhs(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        _collect_vars(self.lhs, out)
        return out

    def variables_rhs(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        _collect_vars(self.rhs, out)
        return out

    def grounded(self) -> bool:
        lv = self.variables_lhs()
        rv = self.variables_rhs()
        return all(k in lv and lv[k] == typ for k, typ in rv.items())

    def complexity(self) -> float:
        n_vars = len(self.variables_lhs())
        return float(self.lhs.size() + self.rhs.size() + 0.5 * n_vars)

    def apply(self, term: Term) -> Optional[Term]:
        bindings: Dict[str, Term] = {}
        if not match(self.lhs, term, bindings):
            return None
        return instantiate(self.rhs, bindings)


def _collect_vars(t: Term, out: Dict[str, str]) -> None:
    if t.is_meta():
        out[str(t.value)] = t.typ
        return
    for c in t.children:
        _collect_vars(c, out)


def match(pattern: Term, concrete: Term, bindings: Dict[str, Term]) -> bool:
    if pattern.typ != concrete.typ:
        return False
    if pattern.is_meta():
        name = str(pattern.value)
        if name in bindings:
            return bindings[name] == concrete
        bindings[name] = concrete
        return True
    if pattern.op != concrete.op or pattern.value != concrete.value:
        return False
    if len(pattern.children) != len(concrete.children):
        return False
    return all(match(p, c, bindings) for p, c in zip(pattern.children, concrete.children))


def instantiate(template: Term, bindings: Mapping[str, Term]) -> Term:
    if template.is_meta():
        return bindings[str(template.value)]
    return Term(template.op, template.typ, tuple(instantiate(c, bindings) for c in template.children), template.value)


# ======================================================================================
# Typed anti-unification with cross-side variable linking
# ======================================================================================

class AntiUnifier:
    def __init__(self) -> None:
        self.memo: Dict[Tuple[str, Tuple[str, ...]], Term] = {}
        self.counter = 0

    @staticmethod
    def _key(terms: Sequence[Term]) -> Tuple[str, Tuple[str, ...]]:
        typ = terms[0].typ if terms else "Unknown"
        return typ, tuple(t.canonical() for t in terms)

    def _new_meta(self, typ: str) -> Term:
        m = Meta(f"u{self.counter}", typ)
        self.counter += 1
        return m

    def lgg(self, terms: Sequence[Term], reuse_memo: bool = True) -> Term:
        if not terms:
            raise ValueError("anti-unification requires at least one term")
        types = {t.typ for t in terms}
        if len(types) != 1:
            # Type disagreement cannot become a sound typed variable.
            return self._new_meta("Any")

        key = self._key(terms)
        if reuse_memo and key in self.memo:
            return self.memo[key]

        if all(t == terms[0] for t in terms):
            result = terms[0]
            self.memo[key] = result
            return result

        same_shape = (
            len({t.op for t in terms}) == 1
            and len({len(t.children) for t in terms}) == 1
            and len({json.dumps(t.value, sort_keys=True) for t in terms}) == 1
            and not terms[0].is_atom()
            and not terms[0].is_meta()
        )
        if same_shape:
            arity = len(terms[0].children)
            children = tuple(self.lgg([t.children[i] for t in terms], reuse_memo=True) for i in range(arity))
            result = Term(terms[0].op, terms[0].typ, children, terms[0].value)
        else:
            result = self._new_meta(terms[0].typ)

        self.memo[key] = result
        return result


def infer_schema(examples: Sequence[Example]) -> Optional[RewriteSchema]:
    if len(examples) < 2:
        return None
    au = AntiUnifier()
    lhs = au.lgg([e.lhs for e in examples], reuse_memo=True)
    rhs = au.lgg([e.rhs for e in examples], reuse_memo=True)
    schema = RewriteSchema(lhs, rhs).alpha_normalized()
    if not schema.grounded():
        return None
    # Reject an unconstrained one-variable identity or a fully generic root rule.
    if schema.lhs.is_meta():
        return None
    if schema.lhs == schema.rhs:
        return None
    return schema


# ======================================================================================
# Schema scoring, compression, stability, active perturbation
# ======================================================================================

@dataclass
class SchemaScore:
    canonical: str
    pretty: str
    train_exact_accuracy: float
    train_coverage: float
    heldout_exact_accuracy: float
    perturbation_exact_accuracy: float
    equivariance_accuracy: float
    bootstrap_stability: float
    compression_ratio: float
    complexity: float
    grounded: bool
    support: int
    pass_gate: bool


@dataclass
class GateConfig:
    min_train_accuracy: float = 0.75
    min_heldout_accuracy: float = 0.95
    min_perturbation_accuracy: float = 0.95
    min_equivariance_accuracy: float = 0.95
    min_bootstrap_stability: float = 0.60
    min_compression_ratio: float = 1.20
    min_support: int = 3
    ransac_trials: int = 128
    subset_min: int = 3
    subset_max: int = 8
    active_pool: int = 256
    top_candidates: int = 8


def exact_accuracy(schema: RewriteSchema, examples: Sequence[Example]) -> Tuple[float, float, int]:
    if not examples:
        return 0.0, 0.0, 0
    exact = 0
    covered = 0
    for e in examples:
        pred = schema.apply(e.lhs)
        if pred is not None:
            covered += 1
            if pred == e.rhs:
                exact += 1
    return exact / len(examples), covered / len(examples), exact


def schema_compression(schema: RewriteSchema, examples: Sequence[Example]) -> float:
    inliers: List[Example] = []
    for e in examples:
        if schema.apply(e.lhs) == e.rhs:
            inliers.append(e)
    if not inliers:
        return 0.0
    raw = sum(e.lhs.size() + e.rhs.size() for e in inliers)
    schema_cost = schema.lhs.size() + schema.rhs.size() + len(schema.variables_lhs())
    binding_cost = 0
    for e in inliers:
        b: Dict[str, Term] = {}
        if match(schema.lhs, e.lhs, b):
            binding_cost += sum(max(1, t.size()) for t in b.values())
    encoded = schema_cost + binding_cost
    return float(raw / max(encoded, 1))


def candidate_schemas(
    examples: Sequence[Example],
    rng: np.random.Generator,
    cfg: GateConfig,
) -> Tuple[List[RewriteSchema], Counter]:
    if len(examples) < cfg.min_support:
        return [], Counter()
    schemas: Dict[str, RewriteSchema] = {}
    counts: Counter = Counter()

    def add(subset: Sequence[Example]) -> None:
        s = infer_schema(subset)
        if s is None:
            return
        c = s.canonical()
        schemas[c] = s
        counts[c] += 1

    add(examples)
    n = len(examples)
    lo = min(cfg.subset_min, n)
    hi = min(cfg.subset_max, n)
    for _ in range(cfg.ransac_trials):
        k = int(rng.integers(lo, hi + 1))
        idx = rng.choice(n, size=k, replace=False)
        add([examples[int(i)] for i in idx])
    return list(schemas.values()), counts


def rank_schemas(schemas: Sequence[RewriteSchema], examples: Sequence[Example], counts: Counter) -> List[RewriteSchema]:
    scored = []
    total = max(sum(counts.values()), 1)
    for s in schemas:
        acc, cov, support = exact_accuracy(s, examples)
        comp = schema_compression(s, examples)
        stability = counts[s.canonical()] / total
        scored.append((-
            acc, -cov, -support, -stability, -comp, s.complexity(), s.canonical(), s
        ))
    scored.sort()
    return [x[-1] for x in scored]


def random_term_for_type(typ: str, rng: np.random.Generator, depth: int = 0) -> Term:
    token = int(rng.integers(0, 1_000_000))
    if typ == "Expr":
        if depth < 1 and rng.random() < 0.35:
            a = Atom(f"e{token}", "Expr")
            b = Atom(f"e{int(rng.integers(0,1_000_000))}", "Expr")
            return Node("Add", "Expr", a, b)
        return Atom(f"e{token}", "Expr")
    if typ in {"Group", "Sequence", "Argument", "Name", "Promoter"}:
        return Atom(f"{typ.lower()}_{token}", typ)
    if typ == "Scalar":
        return Atom(float(rng.normal()), "Scalar")
    if typ == "Molecule":
        return Node("Fragment", "Molecule", Atom(f"R{token}", "Group"))
    return Atom(f"{typ.lower()}_{token}", typ)


def instantiate_random_lhs(schema: RewriteSchema, rng: np.random.Generator) -> Term:
    bindings = {name: random_term_for_type(typ, rng) for name, typ in schema.variables_lhs().items()}
    return instantiate(schema.lhs, bindings)


def typed_perturb(term: Term, salt: str = "g", scalar_shift: float = 0.75) -> Term:
    if term.is_atom():
        if term.typ == "Scalar" and isinstance(term.value, (int, float)):
            return Atom(float(term.value) + scalar_shift, term.typ)
        if term.typ in {"Expr", "Group", "Sequence", "Argument", "Name", "Promoter"}:
            return Atom(f"{salt}_{term.value}", term.typ)
        return term
    return Term(term.op, term.typ, tuple(typed_perturb(c, salt, scalar_shift) for c in term.children), term.value)


def equivariance_accuracy(schema: RewriteSchema, probes: Sequence[Term]) -> float:
    if not probes:
        return 0.0
    ok = 0
    valid = 0
    for x in probes:
        y = schema.apply(x)
        if y is None:
            continue
        gx = typed_perturb(x)
        gy_pred = schema.apply(gx)
        if gy_pred is None:
            continue
        valid += 1
        if gy_pred == typed_perturb(y):
            ok += 1
    return ok / valid if valid else 0.0


def select_active_queries(
    top: Sequence[RewriteSchema],
    seed_schema: RewriteSchema,
    oracle,
    budget: int,
    pool: int,
    rng: np.random.Generator,
) -> Tuple[List[Example], List[Dict[str, Any]]]:
    if budget <= 0:
        return [], []
    candidates: List[Tuple[float, str, Term]] = []
    for _ in range(pool):
        x = instantiate_random_lhs(seed_schema, rng)
        preds = []
        for s in top:
            y = s.apply(x)
            preds.append("<no-match>" if y is None else y.canonical())
        disagreement = len(set(preds)) / max(len(preds), 1)
        candidates.append((disagreement, x.canonical(), x))
    candidates.sort(key=lambda z: (-z[0], z[1]))
    selected = candidates[:budget]
    examples = []
    audit = []
    for disagreement, _, x in selected:
        y = oracle(x)
        if y is None:
            continue
        examples.append(Example(x, y))
        audit.append({"disagreement": disagreement, "lhs": x.pretty(), "rhs": y.pretty()})
    return examples, audit


# ======================================================================================
# Synthetic cross-domain task suite
# ======================================================================================

class Task:
    def __init__(self, name: str, expected_status: str):
        self.name = name
        self.expected_status = expected_status

    def sample_lhs(self, rng: np.random.Generator, index: int) -> Term:
        raise NotImplementedError

    def oracle(self, lhs: Term, index: Optional[int] = None) -> Optional[Term]:
        raise NotImplementedError

    def corrupt_rhs(self, lhs: Term, rng: np.random.Generator) -> Term:
        return random_term_for_type(self.oracle(lhs, 0).typ if self.oracle(lhs, 0) else lhs.typ, rng)

    def examples(self, n: int, rng: np.random.Generator, corruption: float = 0.0) -> List[Example]:
        out = []
        for i in range(n):
            lhs = self.sample_lhs(rng, i)
            rhs = self.oracle(lhs, i)
            if rhs is None:
                continue
            if corruption > 0 and rng.random() < corruption:
                rhs = self.corrupt_rhs(lhs, rng)
            out.append(Example(lhs, rhs))
        return out


class DistributivityTask(Task):
    def __init__(self): super().__init__("algebra_distributivity", "promoted")
    def sample_lhs(self, rng, index):
        k = random_term_for_type("Expr", rng)
        x = random_term_for_type("Expr", rng)
        y = random_term_for_type("Expr", rng)
        return Node("Mul", "Expr", k, Node("Add", "Expr", x, y))
    def oracle(self, lhs, index=None):
        if lhs.op != "Mul" or len(lhs.children) != 2 or lhs.children[1].op != "Add": return None
        k, add = lhs.children
        x, y = add.children
        return Node("Add", "Expr", Node("Mul", "Expr", k, x), Node("Mul", "Expr", k, y))


class BetaIdentityTask(Task):
    def __init__(self): super().__init__("lambda_beta_identity", "promoted")
    def sample_lhs(self, rng, index):
        name = Atom(f"x{int(rng.integers(0,1_000_000))}", "Name")
        arg = random_term_for_type("Expr", rng)
        lam = Node("Lam", "Function", name, Node("Var", "Expr", name))
        return Node("App", "Expr", lam, arg)
    def oracle(self, lhs, index=None):
        if lhs.op != "App" or len(lhs.children) != 2: return None
        lam, arg = lhs.children
        if lam.op != "Lam" or len(lam.children) != 2: return None
        binder, body = lam.children
        if body.op == "Var" and body.children and body.children[0] == binder:
            return arg
        return None


class HydrogenationTask(Task):
    def __init__(self): super().__init__("chemistry_hydrogenation", "promoted")
    def sample_lhs(self, rng, index):
        r1 = random_term_for_type("Group", rng)
        r2 = random_term_for_type("Group", rng)
        alkene = Node("Alkene", "Molecule", r1, r2)
        return Node("React", "Reaction", alkene, Node("H2", "Reagent"))
    def oracle(self, lhs, index=None):
        if lhs.op != "React" or len(lhs.children) != 2: return None
        alkene, h2 = lhs.children
        if alkene.op != "Alkene" or h2.op != "H2": return None
        return Node("Alkane", "Molecule", *alkene.children)


class TranscriptionTask(Task):
    def __init__(self): super().__init__("biology_transcription", "promoted")
    def sample_lhs(self, rng, index):
        promoter = random_term_for_type("Promoter", rng)
        seq = random_term_for_type("Sequence", rng)
        gene = Node("Gene", "Gene", promoter, Node("Coding", "CodingSequence", seq))
        return Node("Express", "Process", gene)
    def oracle(self, lhs, index=None):
        if lhs.op != "Express" or not lhs.children: return None
        gene = lhs.children[0]
        if gene.op != "Gene" or len(gene.children) != 2: return None
        coding = gene.children[1]
        if coding.op != "Coding" or not coding.children: return None
        return Node("RNA", "RNA", coding.children[0])


class SwapTask(Task):
    def __init__(self): super().__init__("structural_swap", "promoted")
    def sample_lhs(self, rng, index):
        a = random_term_for_type("Expr", rng)
        b = random_term_for_type("Expr", rng)
        return Node("Pair", "Pair", a, b)
    def oracle(self, lhs, index=None):
        if lhs.op != "Pair" or len(lhs.children) != 2: return None
        return Node("Pair", "Pair", lhs.children[1], lhs.children[0])


class HiddenSwitchTask(SwapTask):
    def __init__(self): Task.__init__(self, "null_hidden_switching", "abstain")
    def oracle(self, lhs, index=None):
        if lhs.op != "Pair" or len(lhs.children) != 2: return None
        # Hidden parity is deliberately absent from the observed term.
        if (0 if index is None else index) % 2 == 0:
            return Node("Pair", "Pair", lhs.children[1], lhs.children[0])
        return lhs


class RandomRewriteTask(Task):
    def __init__(self): super().__init__("null_random_rewrite", "abstain")
    def sample_lhs(self, rng, index):
        return Node("Wrap", "Expr", random_term_for_type("Expr", rng))
    def oracle(self, lhs, index=None):
        # Deterministic-looking interface, but output is not a function of visible input.
        i = 0 if index is None else index
        return Node("Noise", "Expr", Atom(f"noise_{i * 7919 + 17}", "Expr"))


class IdentityTask(Task):
    def __init__(self): super().__init__("null_identity_no_anomaly", "abstain")
    def sample_lhs(self, rng, index): return random_term_for_type("Expr", rng)
    def oracle(self, lhs, index=None): return lhs


def make_task(name: str) -> Task:
    table = {
        "algebra_distributivity": DistributivityTask,
        "lambda_beta_identity": BetaIdentityTask,
        "chemistry_hydrogenation": HydrogenationTask,
        "biology_transcription": TranscriptionTask,
        "structural_swap": SwapTask,
        "null_hidden_switching": HiddenSwitchTask,
        "null_random_rewrite": RandomRewriteTask,
        "null_identity_no_anomaly": IdentityTask,
    }
    return table[name]()


TASK_NAMES = [
    "algebra_distributivity",
    "lambda_beta_identity",
    "chemistry_hydrogenation",
    "biology_transcription",
    "structural_swap",
    "null_hidden_switching",
    "null_random_rewrite",
    "null_identity_no_anomaly",
]


# ======================================================================================
# One condition
# ======================================================================================

def run_condition(
    task_name: str,
    seed: int,
    train_size: int,
    corruption: float,
    active_budget: int,
    heldout_size: int,
    cfg_dict: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.time()
    cfg = GateConfig(**cfg_dict)
    task = make_task(task_name)
    rng = np.random.default_rng(seed)
    train = task.examples(train_size, rng, corruption)
    heldout = task.examples(heldout_size, np.random.default_rng(seed + 1_000_003), 0.0)

    schemas0, counts0 = candidate_schemas(train, rng, cfg)
    ranked0 = rank_schemas(schemas0, train, counts0)
    selected0 = ranked0[0] if ranked0 else None

    active_examples: List[Example] = []
    active_audit: List[Dict[str, Any]] = []
    if selected0 is not None and active_budget > 0:
        top = ranked0[: cfg.top_candidates]
        active_examples, active_audit = select_active_queries(
            top, selected0, lambda x: task.oracle(x, 10_000_000), active_budget, cfg.active_pool, rng
        )

    augmented = train + active_examples
    schemas, counts = candidate_schemas(augmented, rng, cfg)
    ranked = rank_schemas(schemas, augmented, counts)
    selected = ranked[0] if ranked else None

    result: Dict[str, Any] = {
        "task": task_name,
        "expected_status": task.expected_status,
        "seed": seed,
        "train_size": train_size,
        "corruption": corruption,
        "active_budget": active_budget,
        "heldout_size": heldout_size,
        "train_examples_actual": len(train),
        "active_examples_added": len(active_examples),
        "candidate_count_before": len(ranked0),
        "candidate_count_after": len(ranked),
        "active_queries": active_audit,
    }

    if selected is None:
        observed = "abstain"
        score = None
    else:
        tr_acc, tr_cov, support = exact_accuracy(selected, augmented)
        ho_acc, _, _ = exact_accuracy(selected, heldout)

        # Independent counterfactual probes from the learned LHS schema.
        probe_rng = np.random.default_rng(seed + 2_000_003)
        probes = [instantiate_random_lhs(selected, probe_rng) for _ in range(max(32, active_budget))]
        perturb_examples = []
        for x in probes:
            y = task.oracle(x, 20_000_000)
            if y is not None:
                perturb_examples.append(Example(x, y))
        pert_acc, _, _ = exact_accuracy(selected, perturb_examples)
        equiv = equivariance_accuracy(selected, probes)
        total_counts = max(sum(counts.values()), 1)
        stability = counts[selected.canonical()] / total_counts
        compression = schema_compression(selected, augmented)
        passed = (
            selected.grounded()
            and support >= cfg.min_support
            and tr_acc >= cfg.min_train_accuracy
            and ho_acc >= cfg.min_heldout_accuracy
            and pert_acc >= cfg.min_perturbation_accuracy
            and equiv >= cfg.min_equivariance_accuracy
            and stability >= cfg.min_bootstrap_stability
            and compression >= cfg.min_compression_ratio
        )
        observed = "promoted" if passed else "abstain"
        score = SchemaScore(
            canonical=selected.canonical(),
            pretty=selected.pretty(),
            train_exact_accuracy=tr_acc,
            train_coverage=tr_cov,
            heldout_exact_accuracy=ho_acc,
            perturbation_exact_accuracy=pert_acc,
            equivariance_accuracy=equiv,
            bootstrap_stability=stability,
            compression_ratio=compression,
            complexity=selected.complexity(),
            grounded=selected.grounded(),
            support=support,
            pass_gate=passed,
        )

    result["observed_status"] = observed
    result["status_correct"] = observed == task.expected_status
    result["false_promotion"] = task.expected_status == "abstain" and observed == "promoted"
    result["false_abstention"] = task.expected_status == "promoted" and observed == "abstain"
    result["selected_schema"] = None if score is None else asdict(score)
    result["elapsed_sec"] = time.time() - started
    return result


# ======================================================================================
# Matrix runner with checkpoint/resume and Apple Silicon aware process parallelism
# ======================================================================================

def detect_performance_cores() -> Optional[int]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return None
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except Exception:
            pass
    return None


def resolve_workers(value: str) -> int:
    if value != "auto":
        return max(1, int(value))
    perf = detect_performance_cores()
    if perf:
        return perf
    return max(1, min(os.cpu_count() or 1, 8))


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=False, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def condition_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row[k]) for k in ("task", "seed", "train_size", "corruption", "active_budget"))


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {condition_key(r): r for r in data.get("rows", [])}


def matrix_spec(name: str):
    if name == "smoke":
        return [42], [12], [0.0], [8], 32
    if name == "quick":
        return [42, 1337], [6, 12, 24], [0.0, 0.1], [0, 16], 48
    if name == "full":
        return [42, 1337, 2025, 9001], [4, 8, 16, 32], [0.0, 0.05, 0.1, 0.2], [0, 8, 32], 64
    raise ValueError(name)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    by_task: Dict[str, Dict[str, Any]] = {}
    for task in TASK_NAMES:
        rr = [r for r in rows if r["task"] == task]
        if not rr:
            continue
        by_task[task] = {
            "runs": len(rr),
            "status_accuracy": sum(bool(r["status_correct"]) for r in rr) / len(rr),
            "promotion_rate": sum(r["observed_status"] == "promoted" for r in rr) / len(rr),
            "false_promotions": sum(bool(r["false_promotion"]) for r in rr),
            "false_abstentions": sum(bool(r["false_abstention"]) for r in rr),
            "median_elapsed_sec": float(np.median([r["elapsed_sec"] for r in rr])),
        }
    positives = [r for r in rows if r["expected_status"] == "promoted"]
    nulls = [r for r in rows if r["expected_status"] == "abstain"]
    return {
        "runs": total,
        "status_accuracy": sum(bool(r["status_correct"]) for r in rows) / total if total else 0.0,
        "positive_promotion_rate": sum(r["observed_status"] == "promoted" for r in positives) / len(positives) if positives else 0.0,
        "null_false_promotion_rate": sum(r["observed_status"] == "promoted" for r in nulls) / len(nulls) if nulls else 0.0,
        "false_promotions": sum(bool(r["false_promotion"]) for r in rows),
        "false_abstentions": sum(bool(r["false_abstention"]) for r in rows),
        "by_task": by_task,
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    flat = []
    for r in rows:
        s = r.get("selected_schema") or {}
        flat.append({
            "task": r["task"], "expected_status": r["expected_status"], "observed_status": r["observed_status"],
            "status_correct": r["status_correct"], "seed": r["seed"], "train_size": r["train_size"],
            "corruption": r["corruption"], "active_budget": r["active_budget"],
            "schema": s.get("pretty"), "train_accuracy": s.get("train_exact_accuracy"),
            "heldout_accuracy": s.get("heldout_exact_accuracy"), "perturbation_accuracy": s.get("perturbation_exact_accuracy"),
            "equivariance_accuracy": s.get("equivariance_accuracy"), "stability": s.get("bootstrap_stability"),
            "compression_ratio": s.get("compression_ratio"), "complexity": s.get("complexity"),
            "false_promotion": r["false_promotion"], "false_abstention": r["false_abstention"],
            "elapsed_sec": r["elapsed_sec"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()) if flat else ["task"])
        w.writeheader(); w.writerows(flat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["smoke", "quick", "full"], default="smoke")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--out-dir", default="runs/v24_gate1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=TASK_NAMES)
    ap.add_argument("--ransac-trials", type=int, default=None)
    args = ap.parse_args()

    # Keep each process single-threaded; the workload is process-parallel.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    seeds, train_sizes, corruptions, active_budgets, heldout_size = matrix_spec(args.matrix)
    cfg = GateConfig()
    if args.ransac_trials is not None:
        cfg.ransac_trials = args.ransac_trials
    cfg_dict = asdict(cfg)

    out_dir = Path(args.out_dir)
    checkpoint_path = out_dir / "v24_gate1_checkpoint.json"
    report_path = out_dir / "v24_gate1_report.json"
    csv_path = out_dir / "v24_gate1_rows.csv"

    done = load_checkpoint(checkpoint_path) if args.resume else {}
    jobs = []
    for task in args.tasks:
        for seed in seeds:
            for n in train_sizes:
                for corr in corruptions:
                    for budget in active_budgets:
                        spec = {"task": task, "seed": seed, "train_size": n, "corruption": corr, "active_budget": budget}
                        if condition_key(spec) not in done:
                            jobs.append((task, seed, n, corr, budget, heldout_size, cfg_dict))

    workers = resolve_workers(args.workers)
    started = time.time()
    rows = list(done.values())
    print(f"[v24-g1] matrix={args.matrix} jobs={len(jobs)} resumed={len(done)} workers={workers}")

    if workers == 1:
        iterator = (run_condition(*j) for j in jobs)
        for row in iterator:
            rows.append(row)
            atomic_json_dump({"rows": rows}, checkpoint_path)
            print(f"[{row['observed_status']}] {row['task']} seed={row['seed']} n={row['train_size']} corr={row['corruption']} active={row['active_budget']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_condition, *j) for j in jobs]
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                atomic_json_dump({"rows": rows}, checkpoint_path)
                print(f"[{row['observed_status']}] {row['task']} seed={row['seed']} n={row['train_size']} corr={row['corruption']} active={row['active_budget']}")

    rows.sort(key=lambda r: (r["task"], r["seed"], r["train_size"], r["corruption"], r["active_budget"]))
    report = {
        "protocol": "FASE-v24 Gate-1 repeated anomaly -> typed abstraction -> generated rewrite schema omega",
        "theorem_target": "Promote a new executable typed rewrite schema from recurrent anomalous transformations only when held-out structure, active perturbations, equivariance, stability, and compression jointly support it; otherwise abstain.",
        "matrix": args.matrix,
        "tasks": args.tasks,
        "config": cfg_dict,
        "parallel": {
            "requested": args.workers,
            "resolved_workers": workers,
            "apple_silicon": platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"},
            "detected_performance_cores": detect_performance_cores(),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "rows": rows,
        "summary": aggregate(rows),
        "elapsed_sec": time.time() - started,
    }
    atomic_json_dump(report, report_path)
    atomic_json_dump({"rows": rows}, checkpoint_path)
    write_csv(rows, csv_path)
    print(json.dumps(report["summary"], indent=2))
    print(f"WROTE {report_path}\nWROTE {csv_path}\nWROTE {checkpoint_path}")


if __name__ == "__main__":
    main()
