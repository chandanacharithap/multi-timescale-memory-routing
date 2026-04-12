# Baseline MVP Snapshot

This repository now contains a frozen dependency-light baseline for the multi-timescale memory controller MVP.

What this baseline is:
- a proof-of-execution scaffold for the six-action controller
- a synthetic pilot used to validate routing, trace logging, reward accounting, guards, baselines, and reversible patch behavior
- the implementation that future real-stack replacements should be compared against structurally

What this baseline is not:
- a paper result
- a meaningful accuracy claim
- a substitute for real datasets, retrievers, models, or adaptation mechanisms

Current benchmark fixtures:
- `demo`: synthetic pilot stream for pipeline debugging and baseline smoke tests
- `coverage`: synthetic action-coverage stream whose purpose is only to guarantee that all six actions are exercised and logged

Current routed commands:
- `python3 -m multitimescale_memory --mode router --benchmark demo`

Coverage audit command:
- `python3 -m multitimescale_memory --mode coverage_probe --benchmark coverage`

Interpretation:
- `demo` is the routed synthetic pilot.
- `coverage` is an action-probe fixture that forces each action once so reachability and logging can be validated independently of router preference.

Replacement order from here:
1. Replace the synthetic benchmark stream with one real routing benchmark and one real freshness/update benchmark.
2. Replace the synthetic retrieval source with a real retrieval corpus.
3. Replace the frozen toy model with a real open model.
4. Replace the in-memory stores with the intended experiment stores.
5. Replace synthetic temporary adaptation with a real temporary adapter mechanism.
6. Replace synthetic durable patch promotion with a real durable patch mechanism.

Evaluation order from here:
1. `param_only`
2. `always_retrieve`
3. `retrieve_gate`
4. `router`

Guardrail:
- Do not treat synthetic `accuracy` as a research result. In this baseline, synthetic metrics only mean the pipeline is behaving as designed.
