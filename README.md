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

All plots are saved as separate images in the `outputs/` directory. Below is a detailed breakdown of every plot, including what each axis measures, the units used, and what patterns to look for.

---

### Drift Plots (Brokerage Drift Analysis)

These 6 plots form the core brokerage drift analysis. They all use a dark theme for readability.

#### 1. `drift_01_broker_trajectories.png` — Broker Influence Over Time

Tracks how the top 12 most influential brokers' centrality changes week-by-week. This is the "storyline" of the project — you can see individual brokers rise and fall.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer week index, 0–180 | Each tick = one week. Bin 0 is the earliest week in the dataset. The email layer spans ~3.5 years of Enron data; proximity spans a different period but is aligned to the same bin index. |
| **Y-axis** | Temporal Betweenness | Dimensionless, range 0.0–0.5 | Normalized betweenness centrality. A value of 0.25 means this node sits on ~25% of all shortest paths in the network that week. Higher = more influential as a bridge. The values are smoothed with a 5-point moving average for readability. |
| **Red dots** | Drift events | Binary flag | A dot appears whenever a broker's raw (unsmoothed) betweenness drops by more than **0.1** (i.e., >10 percentage points of shortest paths) from one week to the next. |
| **Line colors** | Department | Categorical | Each broker's line is colored by their department (legend at bottom). |

**What to look for:** Sudden downward spikes with red dots = a broker losing their bridge position. Flat lines = stable brokers. Lines that trend downward over many weeks = gradual marginalization.

---

#### 2. `drift_02_event_waterfall.png` — Drift Event Timeline

Every single drift event across all nodes, laid out as a timeline. This is the "big picture" view — it answers *when* does drift happen across the organization?

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer week index, 0–180 | Same weekly bins as Plot 1. |
| **Y-axis** | Drifting Node (ordinal) | Arbitrary ordering, 0–N | Each unique node that drifted at least once gets a row. Nodes are sorted by **first drift time** — nodes at the bottom drifted first, nodes at the top drifted later. There are no y-axis labels because there are too many nodes. |
| **Dot colors** | Department | Categorical | Colored by department from the `node_departments.csv` mapping. |

**What to look for:** Dense **vertical columns** of dots = many nodes drifted in the same week (systemic shock). Sparse dots = isolated individual drift. Diagonal patterns = cascading drift spreading through the network over consecutive weeks.

---

#### 3. `drift_03_population_drift_rate.png` — Population Drift Rate

Shows the organization-wide drift prevalence over time. This answers: *is drift getting worse?*

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer week index, 0–180 | Same weekly bins. |
| **Y-axis** | Drift Fraction | Ratio, range 0.0–1.0 | The fraction of all active nodes in that week that experienced drift. A value of 0.3 means 30% of nodes drifted that week. Smoothed with a 5-point moving average. |
| **Blue line + fill** | Target A: Betweenness Drop | Fraction of nodes where betweenness fell > 0.1 | Week-over-week fraction of nodes losing bridge position. |
| **Gold line + fill** | Target B: Constraint Rise | Fraction of nodes where constraint rose > 0.2 | Week-over-week fraction of nodes becoming more trapped. |

**What to look for:** Sustained high rates (~30%+) = chronic organizational instability. Sudden spikes = shock events. The two lines tracking each other = betweenness loss and constraint gain are correlated (expected — losing bridges often means getting trapped).

---

#### 4. `drift_04_phase_portrait.png` — Betweenness vs Constraint Phase Space

A 2D scatter that shows where each node-week observation sits in "influence space" and which ones are about to drift. The arrows show *where drifters are headed*.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Temporal Betweenness | Dimensionless, 0.0–0.5 | Same normalized betweenness as Plot 1. Right side = high influence (broker). Left side = low influence (peripheral). |
| **Y-axis** | Burt Constraint | Dimensionless, 0.0–1.0 | Burt's structural constraint. Top = heavily constrained (trapped in clique). Bottom = unconstrained (spanning structural holes). |
| **Grey dots** | Stable nodes | — | Node-week observations where neither Target A nor Target B triggered the next week. |
| **Red ▼ triangles** | Drift A nodes | — | Nodes about to lose betweenness (> 0.1 drop next week). Note they cluster at higher betweenness — you need to *have* influence to *lose* it. |
| **Gold ▲ triangles** | Drift B nodes | — | Nodes about to gain constraint (> 0.2 increase next week). These cluster at lower constraint — they're about to get trapped. |
| **Red arrows** | Drift direction vectors | Δbetweenness, Δconstraint | Drawn for the 30 most dramatic Target A drifters. Arrow tail = current position, arrow head = next-week position. Arrows pointing **left** = losing betweenness. Arrows pointing **up** = gaining constraint. |

**What to look for:** Arrows consistently pointing left-and-up = brokers are simultaneously losing centrality and getting trapped. Clustering of red triangles at the right edge = high-influence brokers are the ones most at risk.

---

#### 5. `drift_05_risk_heatmap.png` — Predicted Drift Risk Heatmap

The model's output: predicted probability of drift for the 25 most at-risk nodes over time. This is the **actionable** plot — it tells you *who* is about to drift and *when*.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Time Bin (Week) | Integer week index | Only shows the time windows covered by the evaluation folds (roughly the last 80% of the dataset, since TimeSeriesSplit uses expanding training windows). |
| **Y-axis** | Node ID | Numeric node identifiers | The 25 nodes with the highest mean predicted drift risk. Sorted top-to-bottom by decreasing average risk. |
| **Color scale** | Predicted Drift Probability | Continuous, 0.0–1.0 | Dark blue/black = low risk (< 0.2). Red = moderate risk (0.4–0.7). Bright red = high risk (> 0.7). The scale is a custom gradient: `#181c25 → #1a3a5c → #ff6b6b → #ff2222`. |

**What to look for:** Persistent hot-red horizontal bands = chronically at-risk nodes. Intermittent red patches = episodic risk. Cool (dark) rows = stable nodes that rarely trigger the model.

---

#### 6. `drift_06_model_evaluation.png` — ROC & Precision-Recall Curves

Standard ML evaluation showing the model's discriminative ability for both prediction targets.

| Panel | Metric | Unit / Scale | Notes |
|-------|--------|-------------|-------|
| **Left panel: ROC Curve** | | | |
| X-axis | False Positive Rate (FPR) | Ratio, 0.0–1.0 | Fraction of non-drifters incorrectly flagged as drifters. |
| Y-axis | True Positive Rate (TPR) | Ratio, 0.0–1.0 | Fraction of actual drifters correctly identified. |
| Dashed diagonal | Random classifier baseline | — | A model no better than coin-flipping follows this line (AUC = 0.5). |
| **Right panel: PR Curve** | | | |
| X-axis | Recall | Ratio, 0.0–1.0 | Of all actual drifters, what fraction did the model catch? |
| Y-axis | Precision | Ratio, 0.0–1.0 | Of all nodes the model flagged as drifters, what fraction actually drifted? |
| **AUC** | Area Under the ROC Curve | Dimensionless, 0.5–1.0 | 0.84 = the model ranks a random drifter above a random non-drifter 84% of the time. |
| **AP** | Average Precision | Dimensionless, 0.0–1.0 | 0.65 = weighted mean precision across all recall thresholds. Accounts for class imbalance better than AUC. |

**What to look for:** Curves hugging the top-left corner = strong model. AUC > 0.8 = good discriminative power. AP > 0.5 with ~32% class prevalence = meaningful lift over random.

---

### SENA Diagnostic Plots (Earlier Pipeline Stages)

These 4 plots are generated by `replicate_visualizations.py` and capture the structural properties of the multiplex network.

#### `stage4_homophily_constraint_scatter.png` — Homophily vs Constraint Scatter

Shows the relationship between how much a node talks to their own department (homophily) and how structurally trapped they are (constraint). This tests the project's core hypothesis: *does department insularity predict brokerage weakness?*

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Coleman Homophily | Dimensionless ratio, ~0.5–1.0 | Fraction of a node's interactions that are with same-department peers. Averaged across all time windows per node. Values near 0.5 = roughly equal cross-department and same-department interaction. Values near 1.0 = almost exclusively same-department. |
| **Y-axis** | Burt Constraint | Dimensionless, 0.0–1.0 | Average structural constraint across all time windows per node. |
| **Dot colors** | Department | Categorical (Set3 colormap) | Each node is colored by its department label. Helps reveal whether departments cluster in specific regions of this space. |
| **Red dashed line** | OLS regression line | — | Linear fit showing the overall trend. Annotated with R² and p-value. |
| **Text box** | Spearman ρ | Dimensionless, −1.0 to +1.0 | Non-parametric rank correlation between homophily and constraint. A positive ρ means higher homophily is associated with higher constraint (supporting the thesis that department insularity → structural trapping). |

**What to look for:** Positive slope = homophily and constraint are correlated (department echo chambers → losing brokerage power). Department clusters in different quadrants = some departments are structurally advantaged vs. disadvantaged.

---

#### `stage4_multilayer_graph.png` — Multiplex Network Visualization

Side-by-side visualization of the two communication layers as network graphs. Shows *how* the organization looks structurally in email vs. face-to-face.

| Panel | Layer | Edge Style | Notes |
|-------|-------|-----------|-------|
| **Left** | Email Layer | Dashed lines | Shows who emails whom. Edge thickness is proportional to email volume (weight, in number of emails). |
| **Right** | Proximity Layer | Solid lines | Shows who is physically near whom. Edge thickness is proportional to contact weight (duration or frequency). |

| Visual Element | Meaning | Unit / Scale |
|----------------|---------|-------------|
| **Node position** | Spring-force layout | Arbitrary 2D coordinates (dimensionless) | Both panels use the **exact same node positions** (computed from the combined graph with `seed=42`), so you can visually track the same person across panels. |
| **Node color** | Department | Categorical (Set3 colormap) | Same department coloring as the scatter plot. |
| **Node size** | Fixed | 100pt (all nodes equal size) | Size does not encode any metric. |
| **Edge thickness** | Edge weight | Proportional, 1.0–4.0 pt linewidth | Thicker = more emails (left) or more proximity contact (right). Scaled as `1.0 + (weight / max_weight) × 3.0`. |
| **Edge opacity** | Fixed | 0.6 alpha | Uniform transparency for readability. |

**What to look for:** Clusters that appear in one layer but not the other = department silos that only exist in email or only in proximity. Nodes that are central in both panels = true organizational brokers. Nodes central in one panel but peripheral in the other = layer-specific influencers.

**Note on node subset:** The plot shows ~35 nodes from the largest connected component of nodes that have edges in *both* layers. This is a deterministic subset (not random) — it selects the first 35 nodes from the overlap component, sorted by ID.

---

#### `stage4_cross_layer_closure.png` — Cross-Layer Closure Rate per Department

A bar chart measuring how much each department's email relationships are *also* present as proximity relationships (and vice versa). This captures the "coherence" between digital and physical interaction.

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Department | Categorical | Department names from `node_departments.csv` (e.g., DCAR, DG, DISQ, DMCT, DMI, DSE, DST, SCOM, SDOC, SFLE, SRH, SSI). Sorted by closure rate descending. |
| **Y-axis** | Cross-layer Closure Rate | Dimensionless ratio, 0.0–1.0 | Weighted Jaccard-like overlap: `Σ min(w_email, w_prox) / Σ max(w_email, w_prox)` for all edges within a department. A value of 0.05 means only 5% of within-department interaction weight is shared across both layers. |
| **Bar colors** | Rank-based gradient | Viridis colormap | Purely cosmetic — darker bars = higher rank. |
| **Labels above bars** | Exact closure rate | 3 decimal places | Printed value for precise reading. |

**What to look for:** Low closure rates across the board (< 0.1) = the email and proximity layers capture very different interaction patterns (expected for an asymmetric multiplex from different organizations). Departments with relatively higher closure = teams where digital and physical communication align.

---

#### `stage4_temporal_heatmap.png` — Temporal Behavior Heatmap

A nodes × months heatmap showing how a network metric changes over a simulated timeline (mapped to 2001–2016 months for visualization convenience).

| Axis | Metric | Unit / Scale | Notes |
|------|--------|-------------|-------|
| **X-axis** | Month | Calendar strings (e.g., Jan-01, Feb-01 …) | Weekly bin IDs are mapped to month strings via `year = 2001 + (bin_id // 12)`, `month = (bin_id % 12) + 1`. This is a display convention, not real calendar dates. |
| **Y-axis** | Node | Top 20 most active nodes | Sorted by frequency of appearance. No y-axis labels (too many rows). |
| **Color scale** | Metric Value | RdYlBu_r colormap, centered at 0 | In the current pipeline, this shows random-uniform predicted risk values (0.0–1.0). Blue = low, yellow = moderate, red = high. |
| **Red dashed lines** | Scandal period annotations | — | Vertical lines marking "Early Crisis", "Peak Scandal", and "Bankruptcy" periods from the Enron timeline, if the data spans those months. |

**What to look for:** Horizontal color bands = a node's behavior is consistent over time. Vertical color bands = a systemic event affecting all nodes simultaneously. Transitions from blue to red = increasing risk or metric spike.

---

### Units Reference Table

A quick reference for every metric's scale used across all plots:

| Metric | Range | Unit | Interpretation |
|--------|-------|------|---------------|
| In-Degree | 0 – ~11,000 | Weighted edge count | Sum of incoming edge weights per week. Large because edges are weighted by volume. |
| Out-Degree | 0 – ~11,000 | Weighted edge count | Sum of outgoing edge weights per week. |
| Burt Constraint | 0.0 – 1.0 | Dimensionless (proportion) | 0 = maximally unconstrained (perfect broker). 1 = maximally constrained (completely redundant). |
| Effective Size | Negative – ~8,000 | Node count (adjusted) | Conceptually the number of non-redundant contacts. Can be negative when redundancy exceeds raw degree. |
| Homophily Ratio | 0.5 – 1.0 | Dimensionless (fraction) | Fraction of interactions with same-department peers. 0.5 = baseline chance. 1.0 = exclusively same-department. |
| Temporal Betweenness | 0.0 – 0.5 | Dimensionless (normalized) | Fraction of all shortest paths passing through this node. 0.0 = peripheral. 0.5 = on half of all shortest paths. |
| Cross-layer Closure | 0.0 – 1.0 | Dimensionless (weighted Jaccard) | Overlap between email and proximity edges for a department. |
| Drift Fraction | 0.0 – 1.0 | Proportion of population | Fraction of all active nodes experiencing drift. |
| Predicted Drift Probability | 0.0 – 1.0 | Probability (sigmoid output) | Model's estimated probability that a node will drift next week. |
| AUC-ROC | 0.5 – 1.0 | Dimensionless | 0.5 = random. 1.0 = perfect ranking. |
| Average Precision | 0.0 – 1.0 | Dimensionless | Weighted mean precision at each recall threshold. |
| Time Bin | 0 – 180 | Integer (weekly index) | Each unit = 1 week. Bin 0 = the first week of the earliest data source. |

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
| `drift.py` | **Brokerage drift visualizations.** Generates 6 separate publication-quality plots: broker trajectories, drift event waterfall, population drift rate, phase portrait, risk heatmap, and model evaluation curves. All plots use a consistent dark theme. |

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
