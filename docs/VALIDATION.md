# Validation plan

MNEMOS is not validated by remembering more. The active thesis is that an
agent can retrieve useful historical context without silently converting old,
derived, or weakly sourced memory into present truth or execution authority.

## Primary safety metric

**Silent-trust rate:** percentage of retrieved memories that are stale or
outside the requested provenance policy but still enter agent prompt context
without an explicit opt-in.

Target: **0**.

## Admission benchmark corpus v1

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


## Admission audit

Every query now returns a machine-readable `admission_report` with:

- evaluated candidate count
- policy-eligible candidate count
- returned memory count
- rejection counts grouped by deterministic reason
- rejected memory IDs grouped by reason
- `policy_leak_count`

The admission reasons currently include:

- `inactive_status`
- `stale_evidence`
- `provenance_not_allowed`
- `missing_source_trace`

The same policy function is applied to direct vector matches and graph-expanded
neighbors. This creates an inspectable invariant: any memory entering final prompt
context must produce no rejection reason under the active query policy.

For evidence-sensitive tasks, `require_source_trace=true` rejects memories that
cannot be tied back to an originating AgentTrace.

The original six-case smoke benchmark remains available:

```bash
python benchmark/admission.py
```

The expanded executable matrix is:

```bash
python -m benchmark.admission_matrix
```

Current CI checkpoint:

- direct admission-policy cases: **20/20 expected outcomes**
- graph-expansion trust-boundary cases: **4/4 expected outcomes**
- total contract checks: **24/24**
- observed policy leaks: **0**
- direct rejection distribution: 6 provenance, 3 stale, 2 missing-source-trace, 3 inactive-status
- stale-evidence explicit opt-in is exercised
- rejection precedence is regression-locked
- stale/disallowed/missing-trace graph neighbors are rejected before final prompt context
- a policy-eligible graph neighbor remains admissible

This benchmark is a policy-boundary test, not a claim about factual-memory accuracy
on LoCoMo, LongMemEval, BEAM, or production workloads.
