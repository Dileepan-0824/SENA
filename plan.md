Plan: Trajekt Multiplex Brokerage Drift
Build an asymmetric temporal multiplex pipeline from sampled Enron email and SocioPatterns proximity data to quantify how department homophily relates to brokerage erosion across layers, then train a custom model to predict next-window brokerage drift risk.

Steps

Phase 1 - Scope Lock and Reproducibility (blocks all later phases)
1.1 Lock objective: cross-layer brokerage drift prediction (not attrition or promotion forecasting).
1.2 Lock deliverables: analytics and modeling outputs; dashboard or UI optional, not mandatory.
1.3 Lock hardware profile for RTX 2060 Mobile (memory-safe defaults and mixed precision policy).
1.4 Lock experiment protocol (seed policy, temporal split IDs, run naming).

Phase 2 - Sampled Data Validation and Canonicalization (depends on 1; email and proximity checks can run in parallel)
2.1 Validate sampled files already present: email_edges_sampled.csv, proximity_edges.csv, node_departments.csv, email_edges_aggregated.csv.
2.2 Parse and normalize timestamps to a common timeline (ISO email timestamps vs integer proximity ticks).
2.3 Canonicalize node IDs with a harmonization table (email-address node-space vs numeric node-space).
2.4 Add email identity cleaning rules for malformed handles and prefixes in aggregated email edges.
2.5 Register a data quality report (coverage, missingness, overlap rates, duplicate rates).

Phase 3 - Multiplex Alignment Strategy (depends on 2)
3.1 Use department labels primarily from proximity nodes (high-confidence source).
3.2 Treat email layer as department-agnostic when department is unknown; retain confident labels when available.
3.3 Define asymmetric multiplex assumptions explicitly (different populations and ID systems).
3.4 Build node-overlap scenarios (strict overlap vs relaxed union) and track downstream impact.

Phase 4 - Temporal Graph Construction (depends on 2 and 3; weekly and monthly builds can run in parallel)
4.1 Build weekly and monthly email snapshots (directed, weighted).
4.2 Build weekly and monthly proximity snapshots (undirected or symmetrized, weighted by contact duration or frequency).
4.3 Build multiplex snapshots with edge-type channels and synchronized time bins.
4.4 Compute base temporal features: degree dynamics, reciprocity, persistence, burstiness, inter-event intervals.
4.5 Run snapshot QA checks: monotonic bins, component behavior, node count drift, timestamp continuity.

Phase 5 - Core SENA Diagnostics (depends on 4)
5.1 Triadic closure: within-layer and cross-layer motif statistics over time.
5.2 Homophily: department assortativity and same-department interaction ratios (primarily proximity-supported).
5.3 Structural holes: Burt constraint, effective size, brokerage role indicators.
5.4 Temporal betweenness: rolling influence and broker-strata transitions.
5.5 Cross-layer correspondence: broker-rank association, including Spearman trend by window.

Phase 6 - Custom Objective Definition (depends on 5)
6.1 Primary target: next-window brokerage drift risk.
6.2 Target formulation A: significant drop in temporal betweenness rank.
6.3 Target formulation B: significant increase in structural constraint.
6.4 Publish thresholding and class-balance diagnostics for both formulations.

Phase 7 - Custom Modeling Pipeline (depends on 6)
7.1 Build feature matrix from multiplex diagnostics and temporal deltas.
7.2 Train the project-specific model for brokerage-drift risk scoring.
7.3 Train lightweight sanity baselines for comparison.
7.4 Evaluate with strict forward temporal splits and leakage controls.
7.5 Report performance and interpretation (calibration, key feature effects, subgroup behavior).

Phase 8 - Robustness and Reporting (depends on 5, 6, and 7)
8.1 Sensitivity to overlap strategy (strict vs relaxed harmonization).
8.2 Sensitivity to target definition (A vs B) and threshold choices.
8.3 Ablations: remove homophily, remove structural-hole metrics, remove one layer.
8.4 Final narrative: where homophily appears to weaken brokerage opportunities across layers.

Relevant files

email_edges_sampled.csv - temporal email edges for dynamic digital layer.
proximity_edges.csv - temporal proximity contacts for physical layer.
node_departments.csv - department mapping for proximity node-space.
email_edges_aggregated.csv - static weighted email edges for global structure and baseline strength features.
plan.md - project execution plan.
Planned new modules:

src/ingest/ - parsing and timestamp or node canonicalization.
src/alignment/ - node-space harmonization and overlap strategy.
src/graphs/ - weekly or monthly layer and multiplex builders.
src/analysis/ - closure, homophily, structural holes, temporal betweenness, cross-layer correlation.
src/modeling/target_builder.py - brokerage-drift targets A and B.
src/modeling/custom_model.py - custom risk model.
src/modeling/baselines.py - lightweight sanity baselines.
src/modeling/train.py - temporal training and evaluation loop.
src/modeling/evaluate.py - metrics, calibration, ablation outputs.
Verification

Data validation passes for all sampled files; overlap and missingness reports are generated.
Graph validation confirms aligned snapshot bins and correct layer semantics (directedness and weights).
Metric validation confirms closure, homophily, constraint, and betweenness with toy-graph and bounds checks.
Objective validation confirms target A and B class distributions and temporal stability.
Modeling validation confirms strict forward splits, no leakage, and consistent uplift vs sanity baselines.
Scientific validation confirms directional stability of homophily to brokerage weakening under sensitivity runs.
Decisions

Primary objective: brokerage drift prediction in a multiplex organizational setting.
Dataset mode: sampled data already available and treated as source of truth.
Layer asymmetry: proximity has stronger department support; email can be partially department-agnostic.
UI or dashboard: optional and not mandatory.
Hardware: RTX 2060 Mobile compatible with memory-aware settings.
Further Considerations

If node overlap is low, report both strict-overlap and relaxed-union results.
Keep causal claims conservative since datasets come from different organizations and periods.
Prefer interpretable outputs to strengthen thesis and report quality.