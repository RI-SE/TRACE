# TRACE Architecture

---

# System Overview

TRACE transforms vehicle trajectories into:

- Semantic driving scenarios
- Structured context models
- Event timelines
- Behavioral rules
- Human-understandable explanations
- Scenario descriptors
- Similarity-search signatures
- OpenSCENARIO simulations

Core philosophy:

```text
Traffic Data
    -> Context
    -> Signals
    -> Events
    -> Rules
    -> Descriptor    
    -> Explanations
    -> Simulation
```

---

# High-Level Architecture

```text
MainApp
│
├── LaneMapViewer
│     ├── RoadInfrastructure
│     ├── OpenLABEL Import
│     ├── Manual Trajectories
│     └── Context Generation
│
└── ScenarioRuleGUI
      ├── Scenario Engine
      ├── GoRules Engine
      ├── Explanation Engine
      └── OpenSCENARIO Export
```

Data flow:

```text
Mask/Layout/OpenDRIVE
          ↓
RoadInfrastructure
          ↓
Trajectories
          ↓
ContextBuilder
          ↓
Signal Engine
          ↓
Scenario Engine
          ↓
Event Detection
          ↓
GoRules DSL
          ↓
Rule Generation
          ↓
Scenario Descriptor
          ↓
Explanation + Export
```

---

# Primary Modules

## MainApp

Responsibilities:

- Connects LaneMapViewer and ScenarioRuleGUI.
- Routes contexts.
- Synchronizes duration.
- Synchronizes masks and metadata.
- Synchronizes OpenDRIVE paths.

Design rule:

- Viewer owns raw data.
- Scenario GUI owns semantic reasoning.
- MainApp owns coordination only.

---

## LaneMapViewer

Viewer and data acquisition layer.

### Supported Inputs

1. Manual waypoint trajectories
2. OpenLABEL trajectories
3. Road segmentation masks
4. Layout metadata
5. Homography matrices
6. OpenDRIVE maps

### Major Responsibilities

- Display lane maps
- Display trajectories
- Build vehicle contexts
- Build pairwise contexts
- Perform similarity search
- Build semantic databases

### Context Ownership

The viewer creates the authoritative context used by the entire system.

---

## ScenarioRuleGUI

Semantic reasoning layer.

Responsibilities:

- Context ingestion
- Rule generation
- Insight generation
- Rule explanation
- Scenario export
- OpenSCENARIO export

Pipeline:

```text
Context
  → Engine Enrichment
  → Event Detection
  → Rule Evaluation
  → Explanation
  → Export
```

---

# Canonical Context Model

Everything in TRACE revolves around a context dictionary.

## Required Structure

```python
{
  "ready": True,
  "context_version": "timeline_v2",
  "duration_seconds": float,

  "vehicles": {
      "V1": {...},
      "V2": {...}
  },

  "pair": {
      "timeline": {...}
  }
}
```

## Vehicle Timeline Frame

```python
{
    "region_id": int,
    "world": {
        "x": float,
        "y": float
    },
    "sd": {
        "s": float,
        "d": float
    }
}
```

## Pair Timeline Frame

```python
{
    "distance": float,
    "delta_s": float,
    "delta_d": float
}
```

The context schema is treated as the system contract.

---

# Coordinate Systems

TRACE uses three coordinate spaces.

## Pixel Space

Used for:

- Rendering
- Annotation
- Mask lookup

## World Space

Obtained through homography.

```text
Pixel -> World
```

## Frenet Space

```text
s = longitudinal position

d = lateral offset
```

Computed against lane centerlines.

---

# RoadInfrastructure Layer

RoadInfrastructure is the geometric foundation.

## Region Types

Supported:

- lane
- connecting_lane
- overlap

## Topology Knowledge

Stores:

- successors
- predecessors
- adjacency
- merge relationships
- shared junctions

## Geometry Services

Provides:

### Projection

```text
Pixel -> World
World -> Pixel
World -> Frenet
```

### Centerlines

- compute centerlines
- cache centerlines
- project to centerlines

### Lane Relationships

Detects:

- same lane
- adjacent lane
- merge relationship
- successor relationship
- shared junction

---

# Signal Engine

Purpose:

Convert raw trajectories into motion signals.

## ContextBuilder

Generates:

### Vehicle Context

Contains:

- world position
- region id
- frenet coordinates
- global s coordinate

### Pairwise Context

Contains:

- distance
- TTC
- lane relation
- interaction relevance

---

## Kinematics

Computes:

- vx
- vy
- speed
- frenet velocity

---

## SignalComputer

Computes:

- lane relations
- TTC
- distance metrics
- interaction significance

---

# Scenario Engine

Purpose:

Transform context into semantic information.

## Major Stages

```text
Context
   ↓
Signal Enrichment
   ↓
Topology Enrichment
   ↓
Behavior Features
   ↓
Event Detection
   ↓
Rule Generation
```

---

# Path Semantics

The engine classifies route relationships.

Possible outputs:

```text
same_lane
shared_path
adjacent_flow
shared_junction
merge
crossing
independent
unknown
```

These labels become major inputs for event and rule reasoning.

---

# Event Detection

TRACE is event-first.

Rules are derived from events.

## Event Types

### Lane Change

Generated using:

- lane transitions
- topology analysis

### Proximity

Generated using:

- distance thresholds
- convergence signals
- hysteresis logic

### Waiting

Generated using:

- low speed
- sustained duration

---

# Event Timeline

Canonical event format:

```python
{
    "type": "proximity",
    "t_start": float,
    "t_end": float,
    "participants": [...],
    "metrics": {...},
    "features": {...}
}
```

---

# Temporal Graph System

Events are converted into a graph.

## Edge Types

### before

```text
A -> B
```

### overlaps

```text
A overlaps B
```

### supports

```text
Event supports Rule
```

Used for higher-order behavior extraction.

---

# GoRules Rule Engine

TRACE uses an external DSL.

Rule definitions live outside Python.

Advantage:

- analysts can modify behavior logic
- code changes are minimized

## Inputs

Examples:

```text
path relationship
minimum distance
TTC
lane changes
waiting duration
convergence
sign flips
```

## Rule Categories

### Lane Behavior

- lane_change
- aggressive_lane_change
- smooth_lane_change
- cut_in

### Interaction Behavior

- merge
- yield
- overtake
- avoidance

### Safety Behavior

- near_miss
- dangerous_following
- critical_proximity

### Waiting Behavior

- waiting_for_gap
- waiting_for_merge_gap
- gap_acceptance

### Route Behavior

- connector_navigation
- complex_route_navigation
- multi_stage_lane_positioning

---

# Behavioral Features

The scenario engine extracts:

## Longitudinal Features

- closing rate
- TTC
- distance trend

## Lateral Features

- lateral crossings
- lateral rates

## Interaction Features

- convergence
- sign flips
- ordering changes

---

# Scenario Descriptor

One of TRACE's most important outputs.

Purpose:

- retrieval
- similarity search
- indexing
- explainability

## Components

### Event Graph

Contains:

- event nodes
- rule nodes
- temporal links

### Search Index

Contains:

- event types
- rule types
- transitions
- motifs

### Numerical Fingerprint

Contains:

- speed statistics
- distance statistics
- TTC
- lane-change counts
- event counts

### Topology Fingerprint

Contains:

- path relationship
- merge characteristics
- crossing characteristics
- junction participation

---

# Similarity Search

Similarity scoring combines:

## Event Similarity

Sequence-based comparison.

## Rule Similarity

Behavior overlap.

## Semantic Similarity

Interaction pattern matching.

## Infrastructure Similarity

Road usage similarity.

Scenarios are compared using signatures generated from descriptors and event sequences.

---

# Explanation Engine

Purpose:

Translate machine reasoning into human-readable descriptions.

Outputs:

- scenario summary
- event summary
- timeline explanation
- maneuver explanation
- risk observations

Example:

```text
Vehicle 1 changes lane.
Vehicles enter close proximity.
Near-miss likelihood is high.
```

---

# LSTM Behavioral Modeling

Optional ML subsystem.

Model:

```text
LSTM-VAE
```

Architecture:

- path embedding
- trajectory encoder
- latent space
- decoder

## Produced Signature

```python
{
  z_mu,
  z_std,
  z_norm,
  aggressiveness
}
```

Potential uses:

- clustering
- anomaly detection
- retrieval
- future hybrid reasoning

---

# OpenSCENARIO Export

Purpose:

Convert inferred scenarios into executable simulations.

Generated artifacts:

### Road Network

OpenDRIVE reference.

### Entities

- vehicle1
- vehicle2

### Spawn Logic

Primary:

```text
LanePosition
```

Fallback:

```text
WorldPosition
```

### Maneuvers

- speed actions
- lane keeping
- lane changes
- stops

### Storyboard

```text
Init
Act
Triggers
Maneuver Groups
Stop Trigger
```

---

# Design Principles

## 1. Context-Driven Architecture

Every subsystem consumes the same context model.

## 2. Topology First

Map semantics dominate behavioral reasoning.

## 3. Event First

Rules are built from events.

## 4. Explainability First

Every rule should be explainable.

## 5. Simulation Ready

Outputs are designed for OpenSCENARIO export.

## 6. Extensible DSL

Behavior logic lives in external rule definitions.

## 7. Hybrid AI

Supports both:

- symbolic reasoning (GoRules)
- learned embeddings (LSTM-VAE)

---

