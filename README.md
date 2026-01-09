<p float="left">
    <img src="icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.12.8-blue?logo=python)](https://www.python.org/downloads/release/python-3918/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)

<h1 align="center"># Decoupled Energy System Optimization</h1>

**⚠️ Note**: *Last update on 09.01.2026*

<div align="left"> This repository is the official code of the paper <strong>"The Model-Build-Manifest: A Dependency Injection pattern for Structural Coupling in Sector-Coupled Energy Systems"</strong></div> 

# Decoupled Energy System Optimization

Implementation of the **Model-Build-Manifest** pattern for dependency injection in sector-coupled energy systems optimization.

## Overview

This project demonstrates a novel decoupling mechanism that separates:
- **Simulator Domain**: Physical model understanding (manifests + constraint formulas)
- **Controller Domain**: Generic optimization logic (reads manifests, assembles models)

The key innovation is that the controller can build and solve optimization problems **without any knowledge of power system physics**. All physics knowledge is encoded in:
1. The manifest (declarative interface)
2. The constraint library (mathematical formula implementations)

## Project Structure

```
Project_Root/
│
├── simulator/                    # [Simulator Domain] Physical model logic
│   ├── __init__.py
│   ├── model_data.py            # Pandapower network loading & preprocessing
│   ├── manifest_factory.py      # Core: Generates JSON manifest
│   └── constraint_library.py    # Core: Mathematical formula implementations
│
├── controller/                   # [Controller Domain] Generic optimizer
│   ├── __init__.py
│   ├── optimizer.py             # Model Assembler: manifest -> Pyomo model
│   └── solver_config.py         # Solver configuration
│
├── data/                         # Configuration files and input data
│   ├── elec_simbench_config.json
│   └── profiles/
│
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_workflow.py         # Complete integration tests
│   └── test_manifest.py         # Manifest generation tests
│
├── main.py                       # Entry point demonstrating the workflow
└── README.md
```

## Key Concepts

### Model-Build-Manifest Pattern

According to the paper, this mechanism achieves decoupling through:

1. **Manifest (manifest.json)**: A standardized JSON file that declares:
   - Sets: Index sets for the optimization model
   - Parameters: Static parameters (line admittances, voltage bounds)
   - Variables: Decision variables with domains and bounds
   - Constraints: Abstract constraint-building instructions

2. **Constraint Library**: Python functions implementing mathematical formulas:
   ```python
   def vm_pu_lb_constr(model, i, t):
       """Voltage magnitude lower bound constraint"""
       if i in model.slack_nodes:
           return pe.Constraint.Skip
       lhs = model.U[i, t]**2 + model.W[i, t]**2
       rhs = model.vm_pu_lb[i]**2
       return lhs >= rhs
   ```

3. **Model Assembler**: Generic controller code that:
   - Reads manifests from simulators
   - Iterates through constraint instructions
   - Calls corresponding library functions
   - Builds complete Pyomo models

### Separation of Concerns

| Aspect | Simulator | Controller |
|--------|-----------|------------|
| **Knows** | Power system physics | Control strategies |
| **Provides** | Manifest + Constraint library | Assembly logic |
| **Doesn't know** | MPC algorithms | Power flow equations |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandapower pyomo numpy pytest

# Install IPOPT solver (optional but recommended)
# On Ubuntu/Debian:
sudo apt-get install coinor-libipopt-dev
pip install ipopt

# Or use conda:
conda install -c conda-forge ipopt
```

## Usage

### Basic Demo

```bash
# Run the complete workflow demonstration
python main.py

# Generate only the manifest (without solving)
python main.py --manifest-only

# Specify output directory
python main.py --output-dir ./my_output
```

### Programmatic Usage

```python
from simulator import ManifestFactory, ModelData, CONSTRAINT_LIBRARY
from controller import ModelAssembler, MPCController, SolverConfig

# === SIMULATOR SIDE ===
# Load network and generate manifest
import pandapower as pp
net = pp.create_empty_network(sn_mva=1.0)
# ... configure network ...

config = {"horizon": 1, "v_min": 0.9, "v_max": 1.1, ...}
model_data = ModelData(net, config)
manifest = ManifestFactory(model_data).create_manifest()

# Save manifest for controller
with open("manifest.json", "w") as f:
    json.dump(manifest, f)

# === CONTROLLER SIDE ===
# Load manifest and build model (no physics knowledge needed!)
assembler = ModelAssembler(CONSTRAINT_LIBRARY)
assembler.load_manifest_from_file("manifest.json")

mpc = MPCController(assembler, config)
model = mpc.build_model()
mpc.set_objective("quadratic_exchange")
mpc.fix_slack_nodes()

results = mpc.solve()
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_workflow.py -v -s

# Run with coverage
pytest tests/ --cov=simulator --cov=controller
```

## License
This code is licensed under the **[MIT License](LICENSE)**.
For any issues or any intention of cooperation, please feel free to contact me at **[xuanhao.mu@kit.edu](xuanhao.mu@kit.edu)**.