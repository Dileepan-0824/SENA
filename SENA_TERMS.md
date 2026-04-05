# SENA Glossary: Social Evolutionary Network Analysis

This document translates standard network analysis terminology into the specific context of **SENA (Social Evolutionary Network Analysis)** as used in our brokerage drift pipeline.

In a SENA framework, a corporate communication network is not treated as a static map of "who knows who." Instead, it is treated as a **living, evolving ecosystem** where physical and digital interactions form a *multiplex* environment. People (nodes) compete for structural advantages (brokerage), and the network constantly shifts, causing those advantages to erode over time (drift).

---

## 1. Core Framework Concepts

### The Multiplex Hierarchy
* **Multiplex Network:** An ecosystem where the exact same population interacts across multiple parallel layers. In SENA, we use two fundamental layers:
  * **Layer 1: The Digital Layer (Email).** Asynchronous, deliberate, and recorded. It reflects conscious information routing.
  * **Layer 2: The Physical Layer (Face-to-Face Proximity).** Synchronous, spontaneous, and ephemeral. It reflects actual physical alignment and unspoken social cohesion.

### Social Evolution
* **Evolutionary Topography:** Networks naturally attempt to close open gaps. If Person A introduces Person B and Person C, eventually B and C will start talking directly without needing A. In SENA, this natural closure of the network is treated as an "evolutionary force" that erodes a broker's power.

---

## 2. Brokerage Elements in SENA

If the network is an ecosystem, **Brokerage** is the highest evolutionary fitness state. 

* **The Bridge (Temporal Betweenness):** The percentage of the organization's information traffic that *must* physically or digitally pass through you. High betweenness means you are an infrastructural bottleneck.
* **The Structural Hole:** A gap between two isolated parts of the organization. A broker "spans" this hole, controlling what crosses it.
* **Burt Constraint (The Trap):** The evolutionary pressure that kills brokerage. Constraint measures how redundant your network is. High constraint means you are trapped in an "echo chamber" where all your connections already talk directly to each other. You no longer control any unique bridges.
* **Effective Size (Non-Redundancy):** Instead of counting how many people you talk to, SENA counts how many *independent worlds* you touch. If you communicate with 50 people, but they all sit at the same desk, your Effective Size is 1.

---

## 3. The "Drift" Phenomenon

The core thesis of this project is that **brokerage is temporary**.

* **Brokerage Drift:** The inevitable decay of a broker's structural influence over time. As the organization evolves, new bridges are built by others, and old silos break down. The broker slowly drifts from being the "center of the universe" to being just another peripheral node.
* **Target A (The Bridge Collapse):** A sudden, systemic drop in a person's Temporal Betweenness (>10%). The broker is suddenly bypassed.
* **Target B (The Silo Trap):** A sudden, systemic spike in a person's Burt Constraint (>20%). The broker gets dragged into a clique and loses their cross-departmental reach.

---

## 4. SENA Diagnostic Metrics (The Signals)

Our prediction algorithm uses the following SENA diagnostics to forecast an impending drift event:

* **Homophily Ratio (Department Insularity):** The tendency of nodes to retreat into their "own kind" (their own department). A rising homophily ratio is a massive red flag in SENA that a broker is adopting siloed behavior and about to experience Target B drift.
* **In/Out-Degree Asymmetry:** 
  * If a broker has high out-degree but low in-degree, they are desperately broadcasting but nobody is listening. 
  * If they have high in-degree but low out-degree, they are a passive authority. SENA tracks sudden shifts in this asymmetry as an early warning sign of disengagement.
* **Cross-Layer Closure:** The ultimate multiplex check. If a department's digital (email) network perfectly matches its physical (proximity) network, that department is highly formalized and rigid. Brokers in these departments face massive constraint pressure.

---

## Summary of the Prediction
In SENA terms: The algorithm monitors the **multiplex evolutionary topography** at Time $t$ to predict exactly when the **evolutionary forces of constraint** will inevitably swallow an active structural broker at Time $t+1$.
