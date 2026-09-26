# Validation plan

MNEMOS is not validated by remembering more. The active thesis is that an
agent can retrieve useful historical context without silently converting old,
derived, or weakly sourced memory into present truth or execution authority.

## Primary safety metric

**Silent-trust rate:** percentage of retrieved memories that are stale or
outside the requested provenance policy but still enter agent prompt context
without an explicit opt-in.

Target: **0**.

## Benchmark corpus v0.1

- fresh episodic trace memories
- LLM-derived semantic memories
- procedural memories
- externally sourced evidence with expiry
- expired evidence
- unknown provenance
- contradictory memories
- superseded memories
- graph-neighbor expansion containing stale nodes
- mixed provenance queries
- explicit stale-memory opt-in
- source trace missing/present cases

## Metrics

- silent-trust rate
- stale-memory rejection recall
- provenance-filter precision
- contradiction/supersession leakage
- relevant-memory recall after trust filters
- graph expansion trust-boundary preservation
- prompt provenance completeness
- retrieval latency
- memory retention/decay stability

## Exit gate

MNEMOS can graduate from memory research to a KAVI memory subsystem when stale
or disallowed memories cannot silently enter default prompt context, provenance
is preserved through storage and retrieval, and trust filtering does not destroy
acceptable recall on the labeled corpus.

Memory remains advisory. No memory value, including a remembered instruction,
can grant runtime authority; KCC remains the authority plane.
