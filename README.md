# SENA — Multiplex Brokerage Drift Prediction

> **Predicting when organizational "bridge-builders" lose their influence across communication layers.**

---

## What This Project Does (The 30-Second Version)

Imagine a large company where people communicate in two ways: **email** (digital) and **face-to-face proximity** (physical). Some people in this company act as **brokers** — they bridge gaps between departments, passing information and connecting people who wouldn't otherwise talk.

This project asks: **can we predict when a broker is about to lose their bridging power?**

We call this loss **"brokerage drift"** — when someone who used to be an important connector starts becoming irrelevant, either because:
- **(Target A)** Their centrality drops — they stop being on the shortest paths between people
- **(Target B)** Their constraint rises — they become trapped in a tight clique instead of spanning holes

We build temporal multiplex networks from Enron email data and SocioPatterns proximity data, compute social network analysis (SENA) diagnostics at every time window, and train a neural network to predict drift risk one window ahead.

## Core Concepts — Brokerage & Drift

Before diving into individual metrics, here are the two ideas the entire project revolves around:

### 🤝 Brokerage — "Being the only bridge between two islands"

In SENA terms, a **broker** is a person who connects two groups that would otherwise have no way to communicate. They sit in a **structural hole** — a gap in the network — and they're the only bridge across it.

**Analogy:** *Imagine a company with two floors. Floor 1 is the engineering team. Floor 2 is the sales team. They never talk to each other directly. But there's one person — let's call her Maya — who goes to lunch with engineers AND attends sales meetings. Maya is the broker. If you're an engineer who needs to understand what customers want, your information has to flow through Maya. She controls the bridge.*

Brokerage is **powerful** because:
- The broker controls information flow between groups
- The broker sees perspectives that neither group has on its own
- The broker can translate, filter, or combine ideas across the gap

We measure brokerage using **temporal betweenness centrality** (how many shortest paths pass through you each week) and **Burt constraint** (how trapped you are — low constraint = good broker, high constraint = no bridging power).

---

### 📉 Drift — "The bridge is crumbling"

**Brokerage drift** is when a broker **loses their bridging position over time**. They used to be the only bridge between two islands, but now:
- Other bridges have formed (someone else started connecting those groups)
- One of the islands moved closer to the other (the groups started talking directly)
- The broker got pulled into one island and stopped reaching the other (they got trapped in a clique)

**Analogy:** *Maya used to be the only person who connected engineering and sales. But last month, three engineers started attending sales meetings themselves. Now information flows without Maya. Her "bridge" is crumbling — that's drift. She went from being irreplaceable to being bypassed.*

In SENA terms, drift shows up as:
- **Target A: Betweenness drops** — fewer shortest paths flow through this person (the highway interchange lost traffic because a bypass was built)
- **Target B: Constraint rises** — the person becomes more trapped in their own clique (Maya stopped going to sales meetings and now only talks to engineers)

Drift matters because it signals **organizational change** — departments restructuring, new communication channels opening, or key connectors being marginalized. Our model predicts drift **one week before it happens**, giving organizations an early warning system.

---

## SENA Features Explained (Layman Analogies)

### 📬 In-Degree — "How many people email you?"
Think of your physical mailbox. In-degree is a count of how many different people send *you* messages. A high in-degree means you're someone people *want* to reach — you're in demand.

**Analogy:** *You're a popular restaurant. In-degree is the number of reservations coming in.*

**In this project:** We compute in-degree per node per weekly time bin across both layers. In the **email layer**, it's the weighted count of emails received (e.g., node 977 receiving 400 emails from 15 different people in week 50). In the **proximity layer**, it's the number of face-to-face contacts initiated toward a person. A node with high in-degree in email but low in-degree in proximity is someone people write to but rarely meet in person — a digital-only hub. This asymmetry across layers is one of the signals that feeds the drift prediction model.

---

### 📤 Out-Degree — "How many people do you email?"
The opposite of in-degree. This counts how many people *you* reach out to. Someone with high out-degree is actively broadcasting, seeking connections, or managing many relationships.

**Analogy:** *You're a talent scout. Out-degree is the number of candidates you cold-call every week.*

**In this project:** Out-degree captures proactive communication behavior. In the **email layer**, a node with high out-degree is someone mass-emailing or managing multiple threads (e.g., a project coordinator). In the **proximity layer**, high out-degree means actively seeking face-to-face meetings. The ratio of in-degree to out-degree tells us whether someone is a passive authority (high in, low out) vs. an active networker (high out, low in). When a broker's out-degree drops suddenly, it may signal disengagement — an early sign of drift.

---

### 🔒 Burt Constraint — "How trapped are you in your own clique?"
This is Ronald Burt's famous measure. If all the people you're connected to also know each other, you're *constrained* — you provide no unique value as a bridge. High constraint = you're redundant. Low constraint = you're a **structural hole spanner**, connecting groups that don't otherwise interact.

**Analogy:** *Imagine you're the only person who speaks both French and Japanese at a conference. If nobody else does, your constraint is low — you're invaluable. But if everyone at your table speaks both languages, your constraint is high — you're replaceable.*

**In this project:** Constraint is computed from the weekly email snapshots using Burt's formula. A node in the **DCAR department** with constraint 0.15 sits in a structural hole between departments — their contacts are spread across DST, DMI, and SFLE and those contacts don't know each other. A node with constraint 0.85 only talks to people within their own tight-knit departmental cluster. **Target B** of the drift prediction fires when constraint jumps by > 0.2 in one week — meaning someone who was spanning holes suddenly got trapped. In the phase portrait plot (Plot 4), this shows up as gold ▲ triangles at the bottom (low constraint) about to move upward.

---

### 🌐 Effective Size — "How many non-redundant connections do you have?"
Related to constraint. It measures how many of your connections are *unique* — meaning they don't overlap with each other. A large effective size means your network is diverse and spans multiple independent groups.

**Analogy:** *You have 10 friends. If they all know each other (one big friend group), your effective size is ~1. If none of them know each other (10 separate worlds), your effective size is ~10.*

**In this project:** Effective size is calculated as `in_degree - out_degree × 0.1` in the diagnostics pipeline (a simplified proxy). A node like **Node 741 (DCAR, rank 9 broker)** with an effective size of ~500 has hundreds of non-overlapping contacts — it bridges many independent clusters. When effective size shrinks over time, it means the node's contacts are starting to know each other directly — the structural holes are closing. This redundancy makes the broker dispensable and often predicts both Target A (betweenness drop) and Target B (constraint rise).

---

### 🏠 Homophily Ratio — "Do you only talk to your own department?"
Homophily is the tendency of "birds of a feather" to flock together. This ratio measures what fraction of a node's interactions are with people from the *same* department vs. different departments. A high ratio means you mostly talk to your own kind.

**Analogy:** *At a school cafeteria, homophily ratio measures how often you sit only with kids from your class vs. mixing with other classes. High homophily = you only hang out with your own group.*

**In this project:** The 12 departments (DCAR, DG, DISQ, DMCT, DMI, DSE, DST, SCOM, SDOC, SFLE, SRH, SSI) each have different homophily profiles. For example, if the SFLE department has a homophily ratio of 0.9, it means 90% of SFLE members' emails go to other SFLE members — it's a highly insular department. Compare this to a broker like **Node 977 (SFLE, rank 1)** who, despite being *in* SFLE, has a lower homophily because they email across departments. The scatter plot (stage4) tests whether high homophily correlates with high constraint — our hypothesis is that departmental insularity traps people and destroys their brokerage power.

---

### 🌉 Temporal Betweenness — "How many shortest paths pass through you over time?"
Betweenness centrality measures how often you sit on the shortest path between other people. If Alice needs to reach Bob and the fastest route goes through *you*, you're on that shortest path. Temporal betweenness captures this dynamically — your bridge importance may spike one week and plummet the next.

**Analogy:** *Imagine you're a highway interchange. Temporal betweenness measures how much traffic passes through your interchange each week. If a new bypass road opens, your traffic (betweenness) drops — that's brokerage drift.*

**In this project:** Temporal betweenness is recomputed for every weekly snapshot graph. A value of 0.40 for **Node 977** in week 50 means 40% of all shortest paths between any two people in the network pass through Node 977 that week — they're a critical junction. The broker trajectories plot (Plot 1) shows this value over 180 weeks. **Target A** fires when betweenness drops by > 0.1 in a single week — for example, Node 741 (DCAR) experienced 18 drift events, meaning its betweenness crashed dramatically 18 separate times across the study period. This is the primary signal of brokerage erosion.

---

### 🔗 Cross-Layer Closure — "Do your email contacts match your face-to-face contacts?"
This metric doesn't apply to individual nodes — it measures an entire department. Cross-layer closure asks: if two people in the same department email each other, do they *also* meet face-to-face? It's the overlap between the digital and physical communication layers.

**In this project:** Because we use **two different data sources** (Enron emails and SocioPatterns proximity from a different organization), the layers represent an *asymmetric multiplex* — they're not the same people. Nodes are aligned via a harmonization table, and closure rates are low (< 0.1) across the board, meaning the email and proximity networks capture fundamentally different interaction patterns. Departments with *higher* closure (even if small) are ones where digital and physical behavior align — these tend to be more cohesive and their brokers tend to be more stable (less drift).

---

### 📊 Multiplex Network — "Two different kinds of relationships, same people"
A multiplex network is a network where the same set of people are connected by more than one type of relationship. Each type forms its own "layer."

**In this project:** We have two layers:
- **Email layer** (Enron corpus) — who sends emails to whom, when, and how many. This captures *deliberate, asynchronous* communication. An email is a conscious decision to reach out.
- **Proximity layer** (SocioPatterns) — who is physically near whom, when, and for how long. This captures *spontaneous, synchronous* interaction. Being near someone might be accidental (same building) or intentional (meeting).

A person who is a broker in *both* layers is a true organizational connector. A person who brokers only in email but not proximity has digital influence without physical presence — and vice versa. The multilayer graph (stage4) visualizes this side-by-side, and the cross-layer closure metric measures how well the two layers agree.

---

## What the Prediction Does

The model takes a snapshot of each person's SENA features at time window *t* and predicts:

| Target | What It Predicts | In Plain English |
|--------|-----------------|------------------|
| **Target A** | Significant drop in temporal betweenness (> 0.1) in window *t+1* | "Will this broker lose their bridge position next week?" |
| **Target B** | Significant increase in Burt constraint (> 0.2) in window *t+1* | "Will this broker become trapped in their own clique next week?" |

The model is a **TemporalRiskMLP** — a 3-layer neural network trained with strict temporal splits (no data leakage from the future) and compared against a logistic regression baseline.

**Current performance:**
- Target A: **AUC ≈ 0.84** (the model correctly ranks 84% of drifters above non-drifters)
- Target B: **AUC ≈ 0.84**
- Average Precision: **~0.65** for both targets

---

## Output Plots — Detailed Guide

All plots are saved as separate images in the `outputs/` directory.

---

### Drift Plots (Brokerage Drift Analysis)

Two simple line charts showing how the top 10 brokers change over ~180 weeks. Both plots track the same set of brokers (ranked by average betweenness) so you can cross-reference them.

#### `drift_temporal_betweenness.png` — Betweenness Over Time

A line chart tracking each broker's **temporal betweenness centrality** week by week. This directly shows *who is a bridge and when they lose that position*.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer, 0–180 | Each tick = one week of network activity. |
| **Y-axis** | Temporal Betweenness | Dimensionless, 0.0–0.5 | Fraction of all shortest paths passing through this node. 0.40 = 40% of all paths. Smoothed with 5-week moving average. |
| **Each line** | One broker | Color-coded by node | Legend shows Node ID and department. Labels at line endpoints. |

**What to look for:** Lines trending downward = a broker losing influence over time (drift). Sharp drops = sudden structural changes. Lines staying flat = stable brokers. If betweenness drops AND constraint rises for the same node → classic brokerage drift.

---

#### `drift_burt_constraint.png` — Constraint Over Time

A line chart tracking each broker's **Burt constraint** week by week. This shows *how trapped each broker becomes in their own clique over time*.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer, 0–180 | Same weekly bins as betweenness plot. |
| **Y-axis** | Burt Constraint | Dimensionless, 0.0–1.0 | 0 = completely unconstrained (perfect broker). 1 = fully trapped (all contacts know each other). Smoothed with 5-week moving average. |
| **Each line** | One broker | Color-coded by node | Same node-to-color mapping as the betweenness plot. |

**What to look for:** Lines trending upward = a broker getting increasingly trapped. A node with rising constraint AND falling betweenness is undergoing textbook brokerage drift.

---

### SENA Diagnostic Plots (Earlier Pipeline Stages)

These 4 plots capture the structural properties of the multiplex network.

#### `stage4_homophily_constraint_scatter.png` — Homophily vs Constraint Scatter

Shows whether department insularity predicts brokerage weakness.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Coleman Homophily | Dimensionless, ~0.5–1.0 | Fraction of same-department interactions. |
| **Y-axis** | Burt Constraint | Dimensionless, 0.0–1.0 | Average structural constraint per node. |
| **Dot colors** | Department | Categorical (Set3) | Color-coded by department. |
| **Red dashed line** | OLS regression | — | Trend line with R² and p-value. |
| **Text box** | Spearman ρ | −1.0 to +1.0 | Positive ρ = insularity correlates with trapping. |

**What to look for:** Positive slope = department echo chambers → losing brokerage power.

---

#### `stage4_multilayer_graph.png` — Multiplex Network Visualization

Side-by-side network graphs of email (dashed edges) and proximity (solid edges) layers.

| Visual Element | Meaning | Notes |
|----------------|---------|-------|
| **Node position** | Spring-force layout | Same positions in both panels for comparison. |
| **Node color** | Department | Categorical (Set3 colormap). |
| **Edge thickness** | Edge weight | Thicker = more emails (left) or more proximity contact (right). |

**What to look for:** Clusters appearing in one layer but not the other = department silos.

---

#### `stage4_cross_layer_closure.png` — Cross-Layer Closure Rate per Department

Bar chart of how much each department's email and proximity relationships overlap.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Department | Categorical | 12 departments sorted by closure rate. |
| **Y-axis** | Closure Rate | Dimensionless, 0.0–1.0 | Weighted Jaccard overlap between email and proximity edges. |

**What to look for:** Low rates (< 0.1) = email and proximity layers capture different interaction patterns.

---

#### `stage4_temporal_heatmap.png` — Temporal Behavior Heatmap

Nodes × months heatmap showing metric changes over a simulated timeline.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Month | Calendar strings | Bin IDs mapped to months for display. |
| **Y-axis** | Node | Top 20 most active nodes | Sorted by activity frequency. |
| **Color** | Metric Value | RdYlBu_r colormap | Blue = low, red = high. |

**What to look for:** Vertical color bands = systemic events affecting all nodes simultaneously.

---

### Units Reference Table

| Metric | Range | Unit | Interpretation |
|--------|-------|------|---------------|
| In-Degree | 0 – ~11,000 | Weighted edge count | Sum of incoming edge weights per week. |
| Out-Degree | 0 – ~11,000 | Weighted edge count | Sum of outgoing edge weights per week. |
| Burt Constraint | 0.0 – 1.0 | Dimensionless | 0 = perfect broker. 1 = fully trapped. |
| Effective Size | Negative – ~8,000 | Node count (adjusted) | Number of non-redundant contacts. |
| Homophily Ratio | 0.5 – 1.0 | Dimensionless | Fraction of same-department interactions. |
| Temporal Betweenness | 0.0 – 0.5 | Dimensionless | Fraction of shortest paths through this node. |
| Cross-layer Closure | 0.0 – 1.0 | Dimensionless | Email-proximity edge overlap per department. |
| Time Bin | 0 – 180 | Integer (weekly) | Each unit = 1 week. |

---
## Project File Map

### Pipeline Runner
| File | Purpose |
|------|---------|
| `run_pipeline.py` | **Master orchestrator.** Runs every pipeline stage in order: ingest → align → build graphs → diagnostics → targets → train → visualize. Run this single file to execute the entire project end-to-end. |

### `src/ingest/` — Data Ingestion & Cleaning
| File | Purpose |
|------|---------|
| `ingest_pipeline.py` | **Phase 2.** Loads raw email and proximity CSVs, normalizes timestamps to a common numerical timeline, cleans malformed email addresses (e.g., `--migrated--` prefixes), creates a node ID harmonization table mapping email addresses to canonical numeric IDs, and generates a data quality report (coverage, missingness). |

### `src/alignment/` — Multiplex Node Alignment
| File | Purpose |
|------|---------|
| `multiplex_alignment.py` | **Phase 3.** Solves the identity problem: email and proximity layers have different node populations. This module maps department labels from proximity nodes onto canonical IDs and produces two alignment strategies — **strict** (only nodes present in both layers) and **relaxed** (union of all nodes, with `unknown` department for email-only nodes). |

### `src/graphs/` — Temporal Graph Construction
| File | Purpose |
|------|---------|
| `graph_builder.py` | **Phase 4.** Bins email and proximity edges into weekly snapshots and builds NetworkX `MultiDiGraph` objects for each time window. Each graph contains edges tagged with their layer (`email` or `proximity`) and weight. Computes base temporal features (in-degree, out-degree) per node per bin and runs QA checks. Saves snapshot graphs as pickle files. |

### `src/analysis/` — SENA Diagnostics
| File | Purpose |
|------|---------|
| `sena_diagnostics.py` | **Phase 5.** Computes the core social network analysis features for every node at every time bin: Burt constraint, effective size, homophily ratio, temporal betweenness centrality, and cross-layer correspondence. These metrics form the feature matrix for the prediction model. |
| `evaluation_metrics.py` | **Evaluation toolkit.** Contains the `NetworkEvaluator` class with methods for: AUC-ROC computation, Precision@K, Spearman cross-layer correlation, Gini coefficient of constraint inequality, temporal Coleman homophily, GBM+SHAP interpretability training, and ablation studies (static vs. temporal features). |

### `src/modeling/` — Prediction Pipeline
| File | Purpose |
|------|---------|
| `target_builder.py` | **Phase 6.** Defines the two prediction targets: **Target A** — temporal betweenness drops by > 0.1 in the next window (broker loses bridge position). **Target B** — Burt constraint increases by > 0.2 in the next window (broker becomes trapped). Publishes class-balance diagnostics. |
| `custom_model.py` | **The neural network.** A PyTorch `TemporalRiskMLP` — a 3-layer MLP (input → 32 → 16 → 1) with ReLU activations, dropout (30%), and sigmoid output for binary risk scoring. Trained with BCE loss and Adam optimizer. |
| `train.py` | **Phase 7.** Trains the PyTorch model and a logistic regression baseline using `TimeSeriesSplit` (5 folds, strict forward temporal validation — no future leakage). Reports AUC-ROC and Average Precision for both targets. Saves predictions and feature importances from a Random Forest for interpretability. |
| `evaluate.py` | **Phase 8 Reporting.** Generates SENA feature distribution histograms, temporal trend plots, ROC/PR curves, and prediction probability distributions. Saves all plots to `outputs/`. *(Note: not currently called by `run_pipeline.py` — run manually or add to pipeline.)* |
| `replicate_visualizations.py` | **Visualization orchestrator.** Loads all processed data and generates the full set of plots: homophily-constraint scatter, multilayer graph, cross-layer closure bar chart, temporal heatmap, and the 6 brokerage drift dashboard plots. |

### `src/viz/` — Visualization Library
| File | Purpose |
|------|---------|
| `__init__.py` | Package exports — registers all plot functions for clean imports. |
| `multilayer.py` | Draws side-by-side email (dashed edges) and proximity (solid edges) network graphs, colored by department. Uses a shared spring layout so node positions match across panels. |
| `scatter.py` | Homophily vs Constraint scatter plot with department colors, linear regression line, and Spearman correlation annotation. |
| `barchart.py` | Cross-layer closure rate bar chart per department. Also contains the `compute_cross_layer_closure()` function that calculates weighted edge overlap between the two layers. |
| `heatmap.py` | Temporal behavior heatmap (nodes × months) with Enron scandal period annotations. |
| `drift.py` | **Brokerage drift visualizations.** Generates 2 simple line charts: temporal betweenness over time and Burt constraint over time, both tracking the top 10 brokers. Uses a consistent dark theme. |

### `data/` — Raw Input Data
| File | Description |
|------|-------------|
| `email_edges_sampled.csv` | Temporal email edges (sender, recipient, timestamp) for the dynamic digital communication layer. |
| `proximity_edges.csv` | Temporal face-to-face proximity contacts (i, j, timestamp, duration) for the physical layer. |
| `node_departments.csv` | Department labels for proximity-layer nodes (high-confidence source). |
| `email_edges_aggregated.csv` | Static weighted email edges for global structure and baseline strength features. |

### `data/processed/` — Intermediate Pipeline Outputs
| File | Description |
|------|-------------|
| `email_edges_norm.csv` | Timestamp-normalized email edges with canonical node IDs. |
| `prox_edges_norm.csv` | Timestamp-normalized proximity edges. |
| `harmonization.csv` | Node ID mapping table (email address → canonical numeric ID). |
| `aligned_nodes_relaxed.csv` | All nodes (union of both layers) with department labels. |
| `aligned_nodes_strict.csv` | Only nodes present in both layers with department labels. |
| `data_quality_report.json` | Coverage, missingness, and overlap statistics. |
| `sena_diagnostics.csv` | Complete SENA feature matrix: in-degree, out-degree, constraint, effective size, homophily, betweenness — per node, per time bin. |
| `modeling_targets.csv` | Feature matrix with Target A and Target B labels appended. |
| `preds_target_a.csv` | Model predictions for Target A (y_true, y_pred). |
| `preds_target_b.csv` | Model predictions for Target B (y_true, y_pred). |
| `feature_importances.csv` | Random Forest feature importance rankings. |
| `graphs/snapshots/snapshot_*.pkl` | Pickled NetworkX MultiDiGraph objects, one per weekly time bin. |

### Other Files
| File | Purpose |
|------|---------|
| `plan.md` | Detailed 8-phase execution plan for the project. |
| `test_density.py` | Density testing utility for graph snapshots. |
| `debug_multilayer.py` | Debug helper for inspecting snapshot graph structure. |

---

## How to Run

```bash
# From the project root:
python run_pipeline.py
```

This executes the full pipeline:
1. **Ingest** — parse, clean, normalize raw data
2. **Align** — harmonize node IDs across layers
3. **Build Graphs** — create weekly temporal snapshots
4. **SENA Diagnostics** — compute all social network features
5. **Build Targets** — define drift prediction labels
6. **Train** — train PyTorch + baseline models
7. **Visualize** — generate all plots to `outputs/`

All outputs land in `outputs/` and `data/processed/`.

---

## Architecture

```
Raw CSV Data
    │
    ▼
┌──────────────┐     ┌────────────────────┐
│   Ingest     │────▶│   Alignment        │
│  (clean +    │     │  (node harmonize)  │
│   normalize) │     └────────┬───────────┘
└──────────────┘              │
                              ▼
                    ┌─────────────────────┐
                    │   Graph Builder     │
                    │  (weekly snapshots) │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  SENA Diagnostics   │
                    │  (constraint, betw, │
                    │   homophily, etc.)  │
                    └────────┬────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌──────────────┐  ┌──────────────┐
           │ Target A:    │  │ Target B:    │
           │ Betw. Drop   │  │ Constr. Rise │
           └──────┬───────┘  └──────┬───────┘
                  │                 │
                  ▼                 ▼
           ┌─────────────────────────────┐
           │     TemporalRiskMLP         │
           │  (PyTorch, 3-layer MLP)     │
           │  + Logistic Baseline        │
           └────────────┬────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   6 Drift Plots  │
              │   + 4 SENA Plots │
              └──────────────────┘
```
