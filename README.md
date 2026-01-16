<p align="center">
  <img src="voibes.jpeg" alt="voibes">
</p>

# SATNet — Decentralized Combinatorial Optimization Subnet

SATNet is a Bittensor subnet where miners submit **solver code** as Docker
images and validators execute that code in **sandboxed containers** to solve
SAT‑encoded optimization problems. The flagship track is **TSP**, with other
SAT families supported via public benchmarks.

### What We Measure (One Sentence)
We measure the optimality of SAT‑encoded optimization solutions produced by
miner‑submitted solver containers on benchmark instances.

### Why SAT + TSP
- **SAT is universal**: any NP problem can be reduced to SAT with deterministic
  verification.
- **TSP is high‑value**: logistics, routing, manufacturing, and sequencing.
- **Benchmarks are mature**: public SAT/TSP suites enable objective scoring.

### How It Works
- **Miners** publish Docker images (solver code) and commit image URLs on‑chain.
- **Validators** pull images, run them in containers, and score outputs.
- **Scoring** rewards better solutions, faster runtimes, and harder instances.

### Scoring (High‑Level)
- `quality_ratio = min(1.0, best_known / miner_value)`
- `speed_factor = min(1.0, time_budget / elapsed_time)`
- `score = quality_ratio * speed_factor * difficulty_multiplier`
- Scores are smoothed with EMA and multiplied by **credibility^2.5**.

### Failure Policy
- Timeout → score 0, single credibility penalty
- Invalid or malformed → score 0, double credibility penalty
- Crash → score 0, no extra penalty

### Instance Selection
- **Seed:** `sha256(block_hash || instance_family || round_index)`
- **Sampling:** public ordered list; take the next `N` after seed‑index

### Constraints (Non‑Negotiable)
- Validator‑only development (no miner code in repo)
- No secret eval sets (assume all secrets leak)
- Compute costs on miners, not validators
- Containers for software competition (Basilica)

## Quick Start
1. Open this repo in Cursor.
2. Review `@knowledge/` for invariants and mechanism patterns.
3. Configure and run `validator.py` with your benchmark catalog.

## What’s Inside
- `@knowledge/` — design rules, invariants, and mechanism patterns
- `validator.py` — validator implementation (Basilica containers, EMA scoring)

Made by Const <3
