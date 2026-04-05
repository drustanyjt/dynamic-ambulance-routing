# Dynamic Ambulance Routing

A dynamic ambulance relocation model using Mixed-Integer Linear Programming (MILP) to optimise ambulance positioning across dispatch posts in response to time-varying demand.

## Problem

Emergency Medical Services must position ambulances to minimise response times. Demand patterns shift throughout the day, so static placement is suboptimal. This project solves a MILP at regular intervals to dynamically reassign ambulances, balancing coverage of high-demand areas against repositioning costs.

## Model

The core optimisation minimises:

```
min  Σ_ij c_ij * x_ij  -  Σ_k b_k * y_k
```

Where:
- `x_ij ∈ {0,1}` — ambulance `i` assigned to post `j`
- `y_k ∈ {0,1}` — call hotspot `k` is covered
- `c_ij` — cost (Euclidean distance) of moving ambulance `i` to post `j`
- `b_k` — benefit weight (call intensity) of covering hotspot `k`

Subject to each ambulance being assigned to exactly one post, and coverage requiring at least one ambulance at a nearby post.

## Structure

- `workbook.ipynb` — Main notebook: MILP solver, Gaussian demand model, simulation loop, and visualisations
- `simulation.py` — Simulation class (in progress)
- `requirements.txt` — Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12+. Dependencies: SciPy, NumPy, Matplotlib, Jupyter.

## Context

Part of an ongoing research collaboration on improving EMS operations in Penang, Malaysia. See the [project report](https://www.overleaf.com/read/69d14ff20e7e3ce53ad1ecbd) for full details.
