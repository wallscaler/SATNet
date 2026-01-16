"""
SATNet validator (single-file pattern).

what_measuring:
  We measure the optimality of SAT-encoded optimization solutions produced by
  miner-submitted solver containers on benchmark instances.

miner_endpoints:
  Miners commit Docker image URLs on-chain via `subtensor.set_commitment(...)`.
  Validators read commitments and execute miner code via Basilica containers.

request_format:
  JSON object passed to Actor.evaluate(task=...):
    {
      "task_id": "tsp_4_001:123456",
      "instance": {
        "id": "tsp_4_001",
        "kind": "tsp" | "sat",
        "payload": {...},
        "best_known": 21.0,
        "difficulty_tier": "tiny"
      },
      "time_budget_s": 5
    }

response_format:
  JSON object returned from Actor.evaluate:
    {
      "solution": {...},
      "objective_value": 21.0,  # optional; validator recomputes
      "runtime_ms": 1234,       # optional; validator measures wall time
      "metadata": {...}         # optional
    }

scoring_criteria:
  quality_ratio = min(1.0, best_known / miner_value)
  speed_factor = min(1.0, time_budget_s / elapsed_s)
  difficulty_multiplier = per-tier constant
  raw_score = quality_ratio * speed_factor * difficulty_multiplier
  final_score = EMA(time-decayed raw_score) * credibility_ema^2.5

failure_handling:
  timeout -> score 0, credibility penalty
  invalid solution -> score 0, credibility penalty * 2
  crash -> score 0, no extra penalty
  malformed response -> treat as invalid solution

instance_selection:
  seed = sha256(block_hash || instance_family || round_index)
  sampling = published ordered list; take next N after seed-index

constraints:
  - validator.py only (no miner code or reference solvers)
  - no secret eval sets (public benchmarks only)
  - container execution with network-isolated, resource-limited environments
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import click
import bittensor as bt
from bittensor_wallet import Wallet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DIFFICULTY_TIERS = {
    "tiny": {"time_budget_s": 5, "difficulty_multiplier": 0.5},
    "small": {"time_budget_s": 15, "difficulty_multiplier": 1.0},
    "medium": {"time_budget_s": 45, "difficulty_multiplier": 1.5},
    "large": {"time_budget_s": 120, "difficulty_multiplier": 2.0},
}

SCORE_EMA_ALPHA = 0.2
CREDIBILITY_EMA_ALPHA = 0.1
CREDIBILITY_POWER = 2.5
SCORE_HALF_LIFE_S = 3600
SOFTMAX_TEMPERATURE = 0.5

MAX_MINERS_PER_ROUND = int(os.getenv("MAX_MINERS_PER_ROUND", "64"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "4"))
INSTANCE_FAMILY = os.getenv("INSTANCE_FAMILY", "tsp")
BASILICA_MODE = os.getenv("BASILICA_MODE", "basilica")  # basilica|docker|url

CPU_LIMIT = os.getenv("BASILICA_CPU_LIMIT", "2000m")
MEM_LIMIT = os.getenv("BASILICA_MEM_LIMIT", "8Gi")


@dataclasses.dataclass
class InstanceSpec:
    id: str
    family: str
    kind: str  # "tsp" | "sat"
    payload: Dict[str, Any]
    best_known: float
    difficulty_tier: str


@dataclasses.dataclass
class ScoreState:
    score_ema: float = 0.0
    credibility_ema: float = 1.0
    last_update_ts: float = 0.0


DEFAULT_CATALOG: List[InstanceSpec] = [
    InstanceSpec(
        id="tsp_4_001",
        family="tsp",
        kind="tsp",
        payload={
            "distance_matrix": [
                [0, 2, 9, 10],
                [1, 0, 6, 4],
                [15, 7, 0, 8],
                [6, 3, 12, 0],
            ]
        },
        best_known=21.0,
        difficulty_tier="tiny",
    ),
    InstanceSpec(
        id="sat_3_001",
        family="sat",
        kind="sat",
        payload={"cnf": [[1, 2], [-1, 3], [-2, -3]]},
        best_known=1.0,
        difficulty_tier="tiny",
    ),
]


def load_catalog(path: Optional[str]) -> List[InstanceSpec]:
    if not path:
        return DEFAULT_CATALOG
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    catalog: List[InstanceSpec] = []
    for item in raw:
        catalog.append(
            InstanceSpec(
                id=item["id"],
                family=item["family"],
                kind=item["kind"],
                payload=item["payload"],
                best_known=float(item["best_known"]),
                difficulty_tier=item["difficulty_tier"],
            )
        )
    return catalog


def get_block_hash(subtensor: bt.Subtensor, block_number: int) -> str:
    if hasattr(subtensor, "get_block_hash"):
        return subtensor.get_block_hash(block_number)
    if hasattr(subtensor, "substrate") and hasattr(subtensor.substrate, "get_block_hash"):
        return subtensor.substrate.get_block_hash(block_number)
    raise RuntimeError("Subtensor does not expose block hash API")


def derive_seed(block_hash: str, instance_family: str, round_index: int) -> int:
    message = f"{block_hash}|{instance_family}|{round_index}".encode()
    return int(hashlib.sha256(message).hexdigest(), 16)


def sample_instances(
    catalog: List[InstanceSpec],
    family: str,
    seed: int,
    sample_size: int,
) -> List[InstanceSpec]:
    ordered = sorted([inst for inst in catalog if inst.family == family], key=lambda i: i.id)
    if not ordered:
        raise RuntimeError(f"No instances for family={family}")
    start = seed % len(ordered)
    return [ordered[(start + i) % len(ordered)] for i in range(sample_size)]


def build_task(instance: InstanceSpec, round_index: int) -> Dict[str, Any]:
    tier = DIFFICULTY_TIERS[instance.difficulty_tier]
    return {
        "task_id": f"{instance.id}:{round_index}",
        "instance": {
            "id": instance.id,
            "kind": instance.kind,
            "payload": instance.payload,
            "best_known": instance.best_known,
            "difficulty_tier": instance.difficulty_tier,
        },
        "time_budget_s": tier["time_budget_s"],
    }


def time_decay(score: float, age_s: float) -> float:
    if score <= 0 or age_s <= 0:
        return score
    decay = math.exp(-age_s * math.log(2) / SCORE_HALF_LIFE_S)
    return score * decay


def ema_update(prev: float, new: float, alpha: float) -> float:
    return (alpha * new) + ((1 - alpha) * prev)


def get_commitment(subtensor: bt.Subtensor, netuid: int, uid: int) -> Optional[str]:
    try:
        commitment = subtensor.get_commitment(netuid, uid)
    except Exception:
        return None
    if not commitment:
        return None
    return commitment


def verify_tsp_solution(payload: Dict[str, Any], solution: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    matrix = payload.get("distance_matrix")
    tour = solution.get("tour")
    if not isinstance(matrix, list) or not isinstance(tour, list):
        return False, None
    n = len(matrix)
    if n == 0 or len(tour) != n or len(set(tour)) != n:
        return False, None
    if any(not isinstance(i, int) or i < 0 or i >= n for i in tour):
        return False, None
    length = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        length += float(matrix[a][b])
    return True, length


def verify_sat_solution(payload: Dict[str, Any], solution: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    cnf = payload.get("cnf")
    assignment = solution.get("assignment")
    if not isinstance(cnf, list) or not isinstance(assignment, dict):
        return False, None
    for clause in cnf:
        satisfied = False
        for literal in clause:
            var = abs(int(literal))
            value = assignment.get(str(var), assignment.get(var))
            if value is None:
                continue
            value = bool(value)
            if (literal > 0 and value) or (literal < 0 and not value):
                satisfied = True
                break
        if not satisfied:
            return False, None
    return True, 1.0


def compute_raw_score(
    best_known: float,
    miner_value: float,
    elapsed_s: float,
    difficulty_tier: str,
) -> float:
    if miner_value <= 0 or best_known <= 0 or elapsed_s <= 0:
        return 0.0
    quality_ratio = min(1.0, best_known / miner_value)
    budget = DIFFICULTY_TIERS[difficulty_tier]["time_budget_s"]
    speed_factor = min(1.0, budget / elapsed_s)
    difficulty_multiplier = DIFFICULTY_TIERS[difficulty_tier]["difficulty_multiplier"]
    return quality_ratio * speed_factor * difficulty_multiplier


async def call_miner_container(
    image: str,
    task: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        from affinetes import env as af_env
    except Exception as exc:
        raise RuntimeError("Affinetes is required for Basilica execution") from exc

    env_vars = {}
    if os.getenv("CHUTES_API_KEY"):
        env_vars["CHUTES_API_KEY"] = os.getenv("CHUTES_API_KEY")
    if os.getenv("BASILICA_API_TOKEN"):
        env_vars["BASILICA_API_TOKEN"] = os.getenv("BASILICA_API_TOKEN")

    load_kwargs = {
        "mode": BASILICA_MODE,
        "image": image,
        "cpu_limit": CPU_LIMIT,
        "mem_limit": MEM_LIMIT,
        "env_vars": env_vars,
    }
    # Prefer explicit network isolation when supported by the runtime.
    network_access = os.getenv("BASILICA_NETWORK_ACCESS")
    if network_access is not None:
        load_kwargs["network_access"] = network_access.lower() == "true"
    try:
        env = af_env.load_env(**load_kwargs)
    except TypeError:
        load_kwargs.pop("network_access", None)
        env = af_env.load_env(**load_kwargs)
    try:
        return await asyncio.wait_for(env.evaluate(task=task), timeout=timeout_s)
    finally:
        await env.cleanup()


async def evaluate_miner(
    uid: int,
    image: str,
    instance: InstanceSpec,
    round_index: int,
) -> Tuple[int, float, bool, bool, bool]:
    """Returns (uid, score, success, crashed, invalid)."""
    task = build_task(instance, round_index)
    start = time.time()
    try:
        response = await call_miner_container(
            image=image,
            task=task,
            timeout_s=task["time_budget_s"],
        )
    except asyncio.TimeoutError:
        return uid, 0.0, False, False, False  # timeout: normal penalty
    except Exception as exc:
        logger.warning("UID %s evaluation failed: %s", uid, exc)
        return uid, 0.0, False, True, False  # crash: no extra penalty
    elapsed = max(time.time() - start, 1e-3)

    if not isinstance(response, dict) or "solution" not in response:
        return uid, 0.0, False, False, True  # malformed -> invalid: double penalty

    solution = response.get("solution", {})
    if instance.kind == "tsp":
        valid, miner_value = verify_tsp_solution(instance.payload, solution)
    elif instance.kind == "sat":
        valid, miner_value = verify_sat_solution(instance.payload, solution)
    else:
        return uid, 0.0, False, False, True  # unknown kind -> invalid

    if not valid or miner_value is None:
        return uid, 0.0, False, False, True  # invalid solution: double penalty

    raw = compute_raw_score(
        best_known=instance.best_known,
        miner_value=miner_value,
        elapsed_s=elapsed,
        difficulty_tier=instance.difficulty_tier,
    )
    return uid, raw, True, False, False


async def evaluate_miners(
    miner_jobs: List[Tuple[int, str, InstanceSpec]],
    round_index: int,
    concurrency: int,
) -> Dict[int, Tuple[float, bool, bool, bool]]:
    """Returns {uid: (score, success, crashed, invalid)}."""
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[int, Tuple[float, bool, bool, bool]] = {}

    async def run_job(uid: int, image: str, instance: InstanceSpec) -> None:
        async with semaphore:
            uid_out, score, success, crashed, invalid = await evaluate_miner(uid, image, instance, round_index)
            results[uid_out] = (score, success, crashed, invalid)

    tasks = [run_job(uid, image, instance) for uid, image, instance in miner_jobs]
    await asyncio.gather(*tasks, return_exceptions=False)
    return results


def apply_coldkey_dedup(
    uids: List[int],
    scores: List[float],
    coldkeys: List[str],
) -> List[float]:
    best_by_coldkey: Dict[str, Tuple[int, float]] = {}
    for uid, score in zip(uids, scores):
        ck = coldkeys[uid]
        if ck not in best_by_coldkey or score > best_by_coldkey[ck][1]:
            best_by_coldkey[ck] = (uid, score)
    deduped = []
    for uid, score in zip(uids, scores):
        if best_by_coldkey[coldkeys[uid]][0] == uid:
            deduped.append(score)
        else:
            deduped.append(0.0)
    return deduped


def softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    if all(s <= 0 for s in scores):
        return [1.0 / len(scores)] * len(scores)
    scaled = [s / max(temperature, 1e-6) for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def select_miners(
    metagraph: bt.Metagraph,
    seed: int,
    max_miners: int,
) -> List[int]:
    uids = list(range(metagraph.n))
    rng = random_from_seed(seed)
    rng.shuffle(uids)
    return uids[: min(max_miners, len(uids))]


def random_from_seed(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng


def update_score_state(
    state: ScoreState,
    raw_score: float,
    success: bool,
    crashed: bool,
    invalid: bool,
    now: float,
) -> None:
    """Update EMA scores with appropriate credibility penalties.
    
    Penalty logic:
      - success: credibility moves toward 1.0
      - timeout (not success, not crashed, not invalid): single penalty toward 0.0
      - invalid/malformed: double penalty (two EMA updates toward 0.0)
      - crash: no credibility change
    """
    age = now - state.last_update_ts if state.last_update_ts else 0.0
    decayed = time_decay(state.score_ema, age)
    state.score_ema = ema_update(decayed, raw_score, SCORE_EMA_ALPHA)
    if crashed:
        # crash: no extra penalty
        pass
    elif invalid:
        # invalid/malformed: apply double penalty (two updates toward 0)
        state.credibility_ema = ema_update(state.credibility_ema, 0.0, CREDIBILITY_EMA_ALPHA)
        state.credibility_ema = ema_update(state.credibility_ema, 0.0, CREDIBILITY_EMA_ALPHA)
    else:
        target = 1.0 if success else 0.0
        state.credibility_ema = ema_update(state.credibility_ema, target, CREDIBILITY_EMA_ALPHA)
    state.last_update_ts = now


def decay_only(state: ScoreState, now: float) -> None:
    age = now - state.last_update_ts if state.last_update_ts else 0.0
    state.score_ema = time_decay(state.score_ema, age)
    state.last_update_ts = now


@click.command()
@click.option("--network", default=lambda: os.getenv("NETWORK", "finney"))
@click.option("--netuid", type=int, default=lambda: int(os.getenv("NETUID", "1")))
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"))
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"))
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=lambda: os.getenv("LOG_LEVEL", "INFO"))
@click.option("--catalog-path", default=lambda: os.getenv("BENCHMARK_CATALOG_PATH"))
@click.option("--family", default=lambda: os.getenv("INSTANCE_FAMILY", INSTANCE_FAMILY))
@click.option("--sample-size", type=int, default=lambda: int(os.getenv("SAMPLE_SIZE", SAMPLE_SIZE)))
@click.option("--max-miners", type=int, default=lambda: int(os.getenv("MAX_MINERS_PER_ROUND", MAX_MINERS_PER_ROUND)))
@click.option("--concurrency", type=int, default=lambda: int(os.getenv("EVAL_CONCURRENCY", "8")))
@click.option("--dry-run", is_flag=True, default=False)
def main(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    log_level: str,
    catalog_path: Optional[str],
    family: str,
    sample_size: int,
    max_miners: int,
    concurrency: int,
    dry_run: bool,
) -> None:
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    wallet = Wallet(name=coldkey, hotkey=hotkey)
    subtensor = bt.Subtensor(network=network)
    metagraph = bt.Metagraph(netuid=netuid, network=network)

    if not subtensor.is_hotkey_registered(netuid, wallet.hotkey.ss58_address):
        raise RuntimeError("Validator hotkey not registered on subnet")

    catalog = load_catalog(catalog_path)
    state: Dict[int, ScoreState] = {}
    last_set_block = 0

    while True:
        try:
            metagraph.sync(subtensor=subtensor, lite=True)
            current_block = subtensor.get_current_block()
            params = subtensor.get_subnet_hyperparameters(netuid)
            weights_rate_limit = max(int(params.weights_rate_limit), 1)

            if current_block - last_set_block < weights_rate_limit:
                time.sleep(12)
                continue

            round_index = current_block // weights_rate_limit
            block_hash = get_block_hash(subtensor, current_block)
            seed = derive_seed(block_hash, family, round_index)
            instances = sample_instances(catalog, family, seed, sample_size)

            miner_uids = select_miners(metagraph, seed, max_miners)
            miner_jobs: List[Tuple[int, str, InstanceSpec]] = []
            for idx, uid in enumerate(miner_uids):
                image = get_commitment(subtensor, netuid, uid)
                if not image:
                    continue
                instance = instances[idx % len(instances)]
                miner_jobs.append((uid, image, instance))

            logger.info("Evaluating %s miners on %s instances", len(miner_jobs), len(instances))
            results = asyncio.run(evaluate_miners(miner_jobs, round_index, concurrency))

            now = time.time()
            for uid in range(metagraph.n):
                if uid not in state:
                    state[uid] = ScoreState()
                if uid in results:
                    raw_score, success, crashed, invalid = results[uid]
                    update_score_state(state[uid], raw_score, success, crashed, invalid, now)
                else:
                    decay_only(state[uid], now)

            raw_scores = [
                state[uid].score_ema * (state[uid].credibility_ema ** CREDIBILITY_POWER)
                for uid in range(metagraph.n)
            ]
            deduped_scores = apply_coldkey_dedup(
                uids=list(range(metagraph.n)),
                scores=raw_scores,
                coldkeys=metagraph.coldkeys,
            )
            weights = softmax(deduped_scores, SOFTMAX_TEMPERATURE)

            if dry_run:
                logger.info("Dry run: skipping set_weights")
            else:
                success = subtensor.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    uids=list(range(metagraph.n)),
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                if success:
                    last_set_block = current_block
                    logger.info("Weights set at block %s", current_block)
                else:
                    logger.warning("set_weights failed at block %s", current_block)

            time.sleep(12)
        except KeyboardInterrupt:
            logger.info("Validator stopped")
            break
        except Exception as exc:
            logger.error("Validator loop error: %s", exc)
            time.sleep(12)


if __name__ == "__main__":
    main()
