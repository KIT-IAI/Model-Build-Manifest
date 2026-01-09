# -*- coding: utf-8 -*-
"""
Optimizer Module (Model Assembler)
==================================

Core component of the Model-Build-Manifest pattern from the Controller's perspective.
This module is a GENERIC Model Assembler that:
1. Reads manifest.json files from simulators
2. Iterates through constraint-building instructions
3. Calls corresponding functions from the constraint library
4. Dynamically builds optimization problems without domain-specific knowledge

The key advantage: The controller does not need to understand physics!
If a new physical model is added, only the manifest needs to change.
"""

import json
import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
from typing import Dict, Any, Callable, Optional, List

# Pyomo domain mapping
PYOMO_DOMAINS = {
    "Reals": pe.Reals,
    "PositiveReals": pe.PositiveReals,
    "NonNegativeReals": pe.NonNegativeReals,
    "NegativeReals": pe.NegativeReals,
    "NonPositiveReals": pe.NonPositiveReals,
    "Integers": pe.Integers,
    "PositiveIntegers": pe.PositiveIntegers,
    "NonNegativeIntegers": pe.NonNegativeIntegers,
    "NegativeIntegers": pe.NegativeIntegers,
    "NonPositiveIntegers": pe.NonPositiveIntegers,
    "Binary": pe.Binary,
    "Boolean": pe.Boolean
}


class ModelAssembler:
    """
    Generic Model Assembler that builds Pyomo models from manifests.

    This class is the core of the decoupling mechanism:
    - It reads manifests (declarative interface from simulators)
    - It uses a constraint library (injected dependency)
    - It builds complete optimization models without physics knowledge

    According to the paper, at each simulation step:
    1. Create an empty optimization model
    2. Iterate through all stored manifests
    3. For each instruction, call the corresponding library function
    4. Add dynamic information (states, forecasts, objectives)
    5. Solve and return results
    """

    def __init__(self, constraint_library: Dict[str, Callable]):
        """
        Initialize the ModelAssembler.

        Args:
            constraint_library: Dictionary mapping rule names to Python functions
        """
        self.constraint_library = constraint_library
        self.manifests: List[Dict] = []

    def register_manifest(self, manifest: Dict) -> None:
        """
        Register a manifest from a simulator.

        During initialization phase, each simulator sends its manifest
        to the controller, which stores this information.

        Args:
            manifest: Manifest dictionary from a simulator
        """
        self.manifests.append(manifest)
        print(f"Registered manifest with {len(manifest.get('Constraints', {}))} constraints")

    def load_manifest_from_file(self, filepath: str) -> None:
        """
        Load and register a manifest from a JSON file.

        Args:
            filepath: Path to the manifest JSON file
        """
        with open(filepath, 'r') as f:
            manifest = json.load(f)
        self.register_manifest(manifest)

    def build_model(self, config: Dict) -> pe.ConcreteModel:
        """
        Build a complete Pyomo model from all registered manifests.

        This is the core assembly process described in the paper.

        Args:
            config: Configuration dictionary (including horizon, etc.)

        Returns:
            Assembled Pyomo ConcreteModel
        """
        # Step 1: Create empty optimization model
        print("Step Build: Creating empty Pyomo model...")
        model = pe.ConcreteModel()

        # Step 2: Iterate through all manifests
        for manifest in self.manifests:
            # Store Enet keys for reference
            if "Enet_keys" in manifest:
                model.Enet_keys = manifest["Enet_keys"]

            # Step 3a: Initialize sets
            self._initialize_sets(model, manifest, config)

            # Step 3b: Initialize parameters
            self._initialize_parameters(model, manifest)

            # Step 3c: Define variables
            self._define_variables(model, manifest)

            # Step 3d: Define constraints (using library functions)
            self._define_constraints(model, manifest)

        print("Step Build: Model assembly complete!")
        return model

    # ==================== Private Assembly Methods ====================

    def _initialize_sets(self, model: pe.ConcreteModel, manifest: Dict, config: Dict) -> None:
        """Initialize all sets from the manifest."""
        H = config.get("horizon", 1)
        model.T = pe.RangeSet(0, H - 1)
        # T_plus_one is needed for storage energy state variables (E[i, t] and E[i, t+1])
        model.T_plus_one = pe.RangeSet(0, H)
        # T_minus_one is needed for ramp constraints that access t+1 (so t can only go to H-2)
        model.T_minus_one = pe.RangeSet(0, H - 2) if H > 1 else pe.RangeSet(0, 0)

        # Also store dt_min for storage constraints
        model.dt_min = config.get("dt_min", 15)

        if "Sets" not in manifest:
            return

        for set_name, set_info in manifest["Sets"].items():
            set_type = set_info.get("type", "simple")

            if set_type == "simple":
                set_data = set_info.get("data", [])
                setattr(model, set_name, pe.Set(initialize=set_data))
                print(f"  Created SimpleSet 'model.{set_name}'")

            elif set_type == "indexed":
                index_name = set_info.get("index")
                if not hasattr(model, index_name):
                    raise AttributeError(
                        f"Set '{set_name}' depends on 'model.{index_name}', which doesn't exist.")
                index_set = getattr(model, index_name)

                raw_data = set_info.get("data", {})
                # Convert string keys to int keys for Pyomo compatibility
                processed_data = {int(k): v for k, v in raw_data.items()}

                # Use dict directly instead of lambda to avoid closure issues
                setattr(model, set_name, pe.Set(
                    index_set,
                    initialize=processed_data
                ))
                print(f"  Created IndexedSet 'model.{set_name}'")

    def _initialize_parameters(self, model: pe.ConcreteModel, manifest: Dict) -> None:
        """Initialize all parameters from the manifest."""
        if "Parameters" not in manifest:
            return

        for param_name, param_info in manifest["Parameters"].items():
            param_type = param_info.get("type", "simple")

            if param_type == "simple":
                setattr(model, param_name, pe.Param(initialize=param_info.get("data")))
                print(f"  Created SimpleParam 'model.{param_name}'")

            elif param_type == "indexed":
                index_name = param_info.get("index")
                if not hasattr(model, index_name):
                    raise AttributeError(
                        f"Parameter '{param_name}' depends on 'model.{index_name}', which doesn't exist.")
                index_set = getattr(model, index_name)

                raw_data = param_info.get("data", {})
                default_val_raw = param_info.get("default")
                is_tuple_indexed = index_name in ["edges", "lines", "trafo"]
                is_set_valued = param_info.get("domain") == "Any"

                processed_data = {}
                for k_str, v in raw_data.items():
                    if is_tuple_indexed:
                        key = tuple(map(int, k_str.split(',')))
                    else:
                        key = int(k_str)

                    if is_set_valued:
                        processed_data[key] = set(v)
                    else:
                        processed_data[key] = v

                if is_set_valued:
                    default_val = set() if default_val_raw is None else set(default_val_raw)
                    param_domain = pe.Any
                else:
                    default_val = 0.0 if default_val_raw is None else default_val_raw
                    param_domain = pe.Reals

                setattr(model, param_name, pe.Param(
                    index_set,
                    initialize=processed_data,
                    default=default_val,
                    within=param_domain
                ))
                print(f"  Created IndexedParam 'model.{param_name}'")

    def _define_variables(self, model: pe.ConcreteModel, manifest: Dict) -> None:
        """Define all variables from the manifest."""
        if "Variables" not in manifest:
            return

        for var_name, var_info in manifest["Variables"].items():
            index_sets = []
            for idx_name in var_info.get("indices", []):
                if not hasattr(model, idx_name):
                    raise AttributeError(
                        f"Variable '{var_name}' depends on 'model.{idx_name}', which doesn't exist.")
                index_sets.append(getattr(model, idx_name))

            kwargs = {}

            # Domain
            domain_str = var_info.get("domain", "Reals")
            if domain_str not in PYOMO_DOMAINS:
                raise ValueError(f"Variable '{var_name}' has unknown domain '{domain_str}'.")
            kwargs["domain"] = PYOMO_DOMAINS[domain_str]

            # Initialize
            initialize = var_info.get("initialize")
            if initialize is not None:
                kwargs["initialize"] = initialize

            # Bounds
            bounds_list = var_info.get("bounds")
            if bounds_list is not None:
                kwargs["bounds"] = tuple(bounds_list)

            var = pe.Var(*index_sets, **kwargs)
            setattr(model, var_name, var)
            print(f"  Created Var 'model.{var_name}'")

    def _define_constraints(self, model: pe.ConcreteModel, manifest: Dict) -> None:
        """
        Define all constraints using the constraint library.

        This is the key dependency injection mechanism:
        - The manifest declares WHAT constraints to create (rule_name)
        - The library provides HOW to implement them (functions)
        """
        if "Constraints" not in manifest:
            return

        for constr_name, info in manifest["Constraints"].items():
            rule_name = info.get("rule_name")

            if rule_name not in self.constraint_library:
                print(f"  Warning: Rule '{rule_name}' not in library. Skipping '{constr_name}'.")
                continue

            rule_func = self.constraint_library[rule_name]

            index_sets = []
            for idx_name in info.get("indices", []):
                if not hasattr(model, idx_name):
                    raise AttributeError(
                        f"Constraint '{constr_name}' depends on 'model.{idx_name}', which doesn't exist.")
                index_sets.append(getattr(model, idx_name))

            setattr(model, constr_name, pe.Constraint(*index_sets, rule=rule_func))
            print(f"  Created Constraint 'model.{constr_name}'")


class MPCController:
    """
    MPC Controller that uses the ModelAssembler for optimization.

    This class provides higher-level MPC functionality:
    - Time series handling
    - Value fixing for boundary conditions
    - Objective function definition
    - Solver management
    - Result extraction
    """

    def __init__(self,
                 assembler: ModelAssembler,
                 config: Dict,
                 solver_config: Optional['SolverConfig'] = None):
        """
        Initialize the MPC Controller.

        Args:
            assembler: ModelAssembler instance
            config: Configuration dictionary
            solver_config: Solver configuration (optional)
        """
        self.assembler = assembler
        self.config = config
        self.solver_config = solver_config or SolverConfig()
        self.model: Optional[pe.ConcreteModel] = None

    def build_model(self) -> pe.ConcreteModel:
        """Build the optimization model from manifests."""
        self.model = self.assembler.build_model(self.config)
        return self.model

    def set_objective(self, objective_type: str = "quadratic_exchange",
                      Enet: Any = None) -> None:
        """
        Set the objective function for the optimization.

        Args:
            objective_type: Type of objective ("quadratic_exchange" or "min_cost")
            Enet: Pandapower network (needed for min_cost objective)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        if objective_type == "quadratic_exchange":
            self.model.obj_func = sum(
                self.model.P[i, t] ** 2 + self.model.Q[i, t] ** 2
                for i in self.model.slack_nodes
                for t in self.model.T
            )
        elif objective_type == "min_cost" and Enet is not None:
            self.model.obj_func = 0
            for i in Enet.poly_cost.index.values:
                et = Enet.poly_cost.et.loc[i]
                element = Enet.poly_cost.element.loc[i]
                bus = Enet[et].loc[element].bus
                cp0 = Enet.poly_cost.cp0_eur.loc[i]
                cp1 = Enet.poly_cost.cp1_eur_per_mw.loc[i]
                cp2 = Enet.poly_cost.cp2_eur_per_mw2.loc[i]
                cq0 = Enet.poly_cost.cq0_eur.loc[i]
                cq1 = Enet.poly_cost.cq1_eur_per_mvar.loc[i]
                cq2 = Enet.poly_cost.cq2_eur_per_mvar2.loc[i]
                self.model.obj_func += sum(
                    cp0 + cp1 * abs(self.model.P[bus, t]) + cp2 * abs(self.model.P[bus, t]) ** 2 +
                    cq0 + cq1 * abs(self.model.Q[bus, t]) + cq2 * abs(self.model.Q[bus, t]) ** 2
                    for t in self.model.T
                )
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")

        self.model.obj = pe.Objective(expr=self.model.obj_func, sense=pe.minimize)

    def fix_slack_nodes(self) -> None:
        """Fix voltage at slack nodes (U=1, W=0)."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        for i in self.model.slack_nodes:
            self.model.U[i, :].fix(1)  # Real part = 1
            self.model.W[i, :].fix(0)  # Imaginary part = 0

    def fix_loads_static(self, Enet: Any) -> None:
        """Fix load values from static Pandapower network data."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        if not Enet.load.empty:
            for i in self.model.pq_nodes:
                self.model.Pload[i, :].fix(
                    Enet.load['p_mw'].loc[Enet.load.index == i].values[0])
                self.model.Qload[i, :].fix(
                    Enet.load['q_mvar'].loc[Enet.load.index == i].values[0])

    def fix_generators_static(self, Enet: Any) -> None:
        """Fix generator values from static Pandapower network data."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        if not Enet.gen.empty:
            for i in self.model.generator_nodes:
                if 'p_mw' in Enet.gen.columns:
                    self.model.Pgen[i, :].fix(
                        Enet.gen['p_mw'].loc[Enet.gen.index == i].values[0])
                if 'q_mvar' in Enet.gen.columns:
                    self.model.Qgen[i, :].fix(
                        Enet.gen['q_mvar'].loc[Enet.gen.index == i].values[0])

        if not Enet.sgen.empty:
            for i in self.model.sgen_nodes:
                self.model.Psgen[i, :].fix(
                    Enet.sgen['p_mw'].loc[Enet.sgen.index == i].values[0])
                self.model.Qsgen[i, :].fix(
                    Enet.sgen['q_mvar'].loc[Enet.sgen.index == i].values[0])

    def solve(self) -> Any:
        """
        Solve the optimization problem.

        Returns:
            Solver results object
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        solver = po.SolverFactory(self.solver_config.solver_name)
        for opt_name, opt_value in self.solver_config.options.items():
            solver.options[opt_name] = opt_value

        results = solver.solve(self.model, tee=self.solver_config.tee)

        if self.solver_config.verbose:
            print(f"Solver status: {results.solver.termination_condition.value}")

        return results

    def extract_results(self, Enet: Any) -> Dict[str, np.ndarray]:
        """
        Extract optimization results.

        Args:
            Enet: Pandapower network for topology information

        Returns:
            Dictionary containing result arrays
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        H = self.config['horizon']
        vn = {i: k for i, k in enumerate(Enet.bus['vn_kv'].values)}

        V, Theta, P, Q = [], [], [], []
        pf, pt, qf, qt = [], [], [], []

        for t in self.model.T:
            for i in self.model.nodes:
                V.append(np.sqrt(self.model.U[i, t].value ** 2 + self.model.W[i, t].value ** 2))
                Theta.append(np.arctan2(self.model.W[i, t].value, self.model.U[i, t].value))
                P.append(self.model.P[i, t].value)
                Q.append(self.model.Q[i, t].value)

            for i, j in self.model.edges:
                pf.append(self.model.p_f[i, j, t].value)
                pt.append(self.model.p_t[i, j, t].value)
                qf.append(self.model.q_f[i, j, t].value)
                qt.append(self.model.q_t[i, j, t].value)

        n_buses = Enet.bus.shape[0]
        n_edges = len(list(self.model.edges))

        results = {
            'V_pu': np.array(V).reshape(H, n_buses).T,
            'Theta_rad': np.array(Theta).reshape(H, n_buses).T,
            'P_mw': np.array(P).reshape(H, n_buses).T,
            'Q_mvar': np.array(Q).reshape(H, n_buses).T,
            'pf_mw': np.array(pf).reshape(H, n_edges).T,
            'pt_mw': np.array(pt).reshape(H, n_edges).T,
            'qf_mvar': np.array(qf).reshape(H, n_edges).T,
            'qt_mvar': np.array(qt).reshape(H, n_edges).T,
            'objective': pe.value(self.model.obj)
        }

        return results


# Import SolverConfig to avoid circular imports
from .solver_config import SolverConfig