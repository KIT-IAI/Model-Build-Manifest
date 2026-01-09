#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Entry Point
================

Demonstrates the complete Model-Build-Manifest workflow for decoupled
energy system optimization.

This script shows:
1. How the SIMULATOR generates a manifest (without knowing control logic)
2. How the CONTROLLER builds an optimization problem (without knowing physics)
3. How they communicate through the standardized manifest interface

Usage:
    python main.py [--manifest-only] [--solve] [--output-dir PATH]
"""

import argparse
import json
import os
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from simulator import ManifestFactory, ModelData, CONSTRAINT_LIBRARY
from controller import ModelAssembler, MPCController, SolverConfig


def create_test_network():
    """
    Create a test distribution network.
    
    This simulates the 14-node low-voltage grid from the paper,
    but simplified for demonstration.
    """
    print("=" * 60)
    print("Creating test network...")
    print("=" * 60)
    
    net = pp.create_empty_network(sn_mva=1.0)
    
    # Create buses (20 kV distribution grid)
    buses = {}
    buses['slack'] = pp.create_bus(net, vn_kv=20.0, name="Slack (Grid Connection)")
    buses['b1'] = pp.create_bus(net, vn_kv=20.0, name="Distribution Node 1")
    buses['b2'] = pp.create_bus(net, vn_kv=20.0, name="Distribution Node 2")
    buses['b3'] = pp.create_bus(net, vn_kv=20.0, name="Distribution Node 3")
    buses['pv'] = pp.create_bus(net, vn_kv=20.0, name="PV Node")
    buses['load'] = pp.create_bus(net, vn_kv=20.0, name="Load Node")
    
    # External grid (slack bus)
    pp.create_ext_grid(net, bus=buses['slack'], vm_pu=1.0)
    
    # Create lines (radial topology)
    pp.create_line(net, from_bus=buses['slack'], to_bus=buses['b1'], 
                   length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=buses['b1'], to_bus=buses['b2'], 
                   length_km=0.5, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=buses['b2'], to_bus=buses['b3'], 
                   length_km=0.3, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=buses['b1'], to_bus=buses['pv'], 
                   length_km=0.4, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=buses['b2'], to_bus=buses['load'], 
                   length_km=0.6, std_type="NAYY 4x50 SE")
    
    # Create loads
    pp.create_load(net, bus=buses['b2'], p_mw=0.05, q_mvar=0.02, name="Load 1")
    pp.create_load(net, bus=buses['b3'], p_mw=0.03, q_mvar=0.01, name="Load 2")
    pp.create_load(net, bus=buses['load'], p_mw=0.08, q_mvar=0.04, name="Main Load")
    
    # Create PV generator
    pp.create_sgen(net, bus=buses['pv'], p_mw=0.12, q_mvar=0.0, name="PV Plant")
    
    print(f"Network created with {len(net.bus)} buses and {len(net.line)} lines")
    return net


def run_simulator(network, config, output_dir):
    """
    SIMULATOR DOMAIN
    ================
    
    The simulator maintainer understands the physical model and is
    responsible for:
    1. Loading and validating the Pandapower network
    2. Generating the manifest.json (declarative interface)
    3. Providing the constraint library (mathematical formulas)
    
    The simulator does NOT know anything about control strategies!
    """
    print("\n" + "=" * 60)
    print("SIMULATOR: Generating manifest...")
    print("=" * 60)
    
    # Step 1: Create ModelData (validates and prepares network)
    model_data = ModelData(network, config)
    
    # Step 2: Create manifest factory
    manifest_factory = ManifestFactory(model_data)
    
    # Step 3: Generate and save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_factory.save_manifest(manifest_path)
    
    manifest = manifest_factory.create_manifest()
    
    print(f"\nManifest generated with:")
    print(f"  - {len(manifest['Sets'])} sets")
    print(f"  - {len(manifest['Parameters'])} parameters")
    print(f"  - {len(manifest['Variables'])} variables")
    print(f"  - {len(manifest['Constraints'])} constraints")
    
    return manifest_path, model_data.Enet


def run_controller(manifest_path, config, Enet):
    """
    CONTROLLER DOMAIN
    =================
    
    The controller maintainer understands control strategies but NOT physics.
    They are responsible for:
    1. Reading manifests from simulators
    2. Building optimization problems using the constraint library
    3. Adding control-specific objectives and dynamic information
    4. Solving and returning results
    
    The controller does NOT know anything about power systems equations!
    """
    print("\n" + "=" * 60)
    print("CONTROLLER: Building and solving optimization problem...")
    print("=" * 60)
    
    # Step 1: Create model assembler with constraint library
    # Note: CONSTRAINT_LIBRARY is injected - the controller doesn't define formulas!
    assembler = ModelAssembler(CONSTRAINT_LIBRARY)
    
    # Step 2: Load manifest from simulator
    print(f"\nLoading manifest from: {manifest_path}")
    assembler.load_manifest_from_file(manifest_path)
    
    # Step 3: Create MPC controller
    solver_config = SolverConfig(
        solver_name="ipopt",
        options={"tol": 1e-8, "max_iter": 1000},
        verbose=True,
        tee=False
    )
    mpc = MPCController(assembler, config, solver_config)
    
    # Step 4: Build the Pyomo model
    print("\nBuilding Pyomo model from manifest...")
    model = mpc.build_model()
    
    # Step 5: Set objective function (control strategy)
    print("\nSetting objective function: minimize P² + Q² at slack")
    mpc.set_objective("quadratic_exchange")
    
    # Step 6: Fix boundary conditions
    print("Fixing boundary conditions...")
    mpc.fix_slack_nodes()
    mpc.fix_loads_static(Enet)
    mpc.fix_generators_static(Enet)
    
    # Step 7: Solve
    print("\nSolving optimization problem...")
    try:
        results = mpc.solve()
        
        # Step 8: Extract results
        res = mpc.extract_results(Enet)
        return res
        
    except Exception as e:
        print(f"\nSolver error: {e}")
        print("Note: Make sure IPOPT is installed (pip install ipopt)")
        return None


def print_results(results):
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if results is None:
        print("No results available (solver may not be installed)")
        return
    
    print(f"\nOptimal objective value: {results['objective']:.6f}")
    print(f"\nVoltage magnitudes (p.u.):")
    for i, v in enumerate(results['V_pu'][:, 0]):
        print(f"  Bus {i}: {v:.4f}")
    
    print(f"\nVoltage range: {results['V_pu'].min():.4f} - {results['V_pu'].max():.4f} p.u.")
    
    print(f"\nActive power injections (MW):")
    for i, p in enumerate(results['P_mw'][:, 0]):
        print(f"  Bus {i}: {p:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model-Build-Manifest Decoupled Optimization Demo"
    )
    parser.add_argument(
        "--manifest-only", 
        action="store_true",
        help="Only generate manifest, don't solve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for manifest and results"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    config = {
        "horizon": 1,
        "v_min": 0.9,
        "v_max": 1.1,
        "flow_constraint": False,
        "objective": "quadratic_exchange",
        "with_timeseries": False
    }
    
    print("\n" + "=" * 60)
    print("MODEL-BUILD-MANIFEST DEMONSTRATION")
    print("Decoupled Energy System Optimization")
    print("=" * 60)
    
    # Create test network
    # network = create_test_network()
    network = create_simbench_network()  # Alternatively, use a SimBench network
    
    # Run simulator (generates manifest)
    manifest_path, Enet = run_simulator(network, config, args.output_dir)
    
    if args.manifest_only:
        print("\n--manifest-only flag set. Exiting without solving.")
        return
    
    # Run controller (builds and solves optimization)
    results = run_controller(manifest_path, config, Enet)
    
    # Print results
    print_results(results)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in: {args.output_dir}")
    print("  - manifest.json: The declarative interface")
    print("\nKey takeaway: The controller built the optimization problem")
    print("without any knowledge of power system physics!")


if __name__ == "__main__":
    main()
