#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case Study Network Builder
===========================

Builds the 14-node low-voltage distribution network used in the OSMSES 2026
paper and exports all parameters and time-series profiles for reproducibility.

Network: SimBench '1-LV-rural1--1-sw'
Modifications:
  - 4 additional PV generators at buses 12, 14, 6, 9
  - Storage max_p_mw symmetrised (max_p_mw = -min_p_mw)
  - External grid voltage set to 1.0 p.u.
  - Optional: P2G unit at bus 11, line 2 thermal limit adjusted

Two scenarios are generated:
  - Baseline (S0): original SimBench profiles
  - Disturbance (S1): PV output scaled by 1.5x for selected units at t=48..55

Usage:
    python create_case_network.py [--with-ptg] [--output-dir ./]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pandapower as pp
import simbench


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def create_base_network():
    """Load SimBench grid and apply paper-specific modifications."""
    code = "1-LV-rural1--1-sw"
    net = simbench.get_simbench_net(code)

    # Duplicate first 4 sgens as additional PV at buses 12, 14, 6, 9
    new_sgens = net.sgen.iloc[0:4].copy()
    net.sgen = pd.concat([net.sgen, new_sgens], ignore_index=True)

    new_indices = [8, 9, 10, 11]
    target_buses = [12, 14, 6, 9]
    pv_names = [
        "LV1.101 SGen 9",
        "LV1.101 SGen 10",
        "LV1.101 SGen 11",
        "LV1.101 SGen 12",
    ]
    for i, idx in enumerate(new_indices):
        net.sgen.at[idx, "name"] = pv_names[i]
        net.sgen.at[idx, "bus"] = target_buses[i]

    # Symmetrise storage power limits
    net.storage["max_p_mw"] = -net.storage["min_p_mw"]

    # Fix external grid voltage
    net.ext_grid["vm_pu"] = 1.0

    # Round all numeric DataFrames to 4 decimals
    for key, df in net.items():
        if isinstance(df, pd.DataFrame):
            net[key] = df.round(4)

    return net


def add_ptg_unit(net):
    """Add Power-to-Gas unit and adjust line thermal limit."""
    net.line.at[2, "max_i_ka"] = 0.0404

    ptg_data = {
        "max_p_mw": {0: 0.05},
        "bus": {0: 11},
        "junction": {0: 2},
        "heating_value": {0: 11.55},
        "efficiency": {0: 0.9},
    }
    net["PtG"] = pd.DataFrame(data=ptg_data)
    return net


# ---------------------------------------------------------------------------
# Time-series profile generation
# ---------------------------------------------------------------------------

def _is_affected(index, affected_indices):
    if isinstance(affected_indices, int):
        return index == affected_indices
    return index in affected_indices


def generate_profiles(net):
    """
    Generate load and PV profiles for baseline and disturbance scenarios.

    Returns profiles attached to the net object as custom attributes:
      net["profiles_load_p"]  shape (2, N_load, T)
      net["profiles_load_q"]  shape (2, N_load, T)
      net["profiles_pv_p"]    shape (2, N_sgen, T)
    """
    time_slice = slice(96 * 5, 96 * 7)  # 2 days (192 steps of 15 min)

    # Disturbance parameters
    t_start = 48
    duration = 8
    gd_factor = 1.5
    affected_pv = [0, 8, 9, 10, 7]

    profiles_load_p = [[], []]
    profiles_load_q = [[], []]
    profiles_pv_p = [[], []]

    # Load profiles (identical for both scenarios)
    for i, profile_name in enumerate(net.load.profile.values):
        p = (
            net.profiles["load"][profile_name + "_pload"].values[time_slice]
            * net.load.at[i, "p_mw"]
        )
        q = (
            net.profiles["load"][profile_name + "_qload"].values[time_slice]
            * net.load.at[i, "q_mvar"]
        )
        profiles_load_p[0].append(p.copy())
        profiles_load_q[0].append(q.copy())
        profiles_load_p[1].append(p.copy())
        profiles_load_q[1].append(q.copy())

    # PV profiles (disturbance in scenario 1)
    for i, profile_name in enumerate(net.sgen.profile.values):
        p = (
            net.profiles["renewables"][profile_name].values[time_slice]
            * net.sgen.at[i, "p_mw"]
        )
        profiles_pv_p[0].append(p.copy())

        p_disturbed = p.copy()
        if _is_affected(i, affected_pv):
            p_disturbed[t_start : t_start + duration] *= gd_factor
        profiles_pv_p[1].append(p_disturbed)

    net["profiles_load_p"] = np.array(profiles_load_p)
    net["profiles_load_q"] = np.array(profiles_load_q)
    net["profiles_pv_p"] = np.array(profiles_pv_p)

    return net


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    """Convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    return obj


def export_network_params(net, output_dir):
    """Export static network parameters to JSON."""
    params = {
        "simbench_code": "1-LV-rural1--1-sw",
        "sn_mva": float(net.sn_mva),
        "bus": net.bus[["name", "vn_kv", "type"]].to_dict(orient="index"),
        "line": net.line[
            ["name", "from_bus", "to_bus", "length_km", "r_ohm_per_km",
             "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "type"]
        ].to_dict(orient="index"),
        "ext_grid": net.ext_grid[["bus", "vm_pu"]].to_dict(orient="index"),
    }
    if not net.trafo.empty:
        params["trafo"] = net.trafo[
            ["name", "hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv",
             "vk_percent", "vkr_percent"]
        ].to_dict(orient="index")

    path = os.path.join(output_dir, "network_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2, default=_to_serialisable)
    print(f"  Saved {path}")


def export_component_params(net, output_dir):
    """Export component parameters (BESS, PV, Load, P2G) to JSON."""
    components = {}

    # Storage / BESS
    if not net.storage.empty:
        units = []
        for idx in net.storage.index:
            units.append({
                "index": int(idx),
                "bus": int(net.storage.at[idx, "bus"]),
                "name": str(net.storage.at[idx, "name"]),
                "max_p_mw": float(net.storage.at[idx, "max_p_mw"]),
                "min_p_mw": float(net.storage.at[idx, "min_p_mw"]),
                "max_e_mwh": float(net.storage.at[idx, "max_e_mwh"]),
                "min_e_mwh": float(net.storage.at[idx, "min_e_mwh"]),
                "soc_percent": float(net.storage.at[idx, "soc_percent"]),
            })
        components["BESS"] = units

    # Static generators (PV)
    if not net.sgen.empty:
        units = []
        for idx in net.sgen.index:
            units.append({
                "index": int(idx),
                "bus": int(net.sgen.at[idx, "bus"]),
                "name": str(net.sgen.at[idx, "name"]),
                "p_mw": float(net.sgen.at[idx, "p_mw"]),
                "q_mvar": float(net.sgen.at[idx, "q_mvar"]),
            })
        components["PV"] = units

    # Loads
    if not net.load.empty:
        units = []
        for idx in net.load.index:
            units.append({
                "index": int(idx),
                "bus": int(net.load.at[idx, "bus"]),
                "name": str(net.load.at[idx, "name"]),
                "p_mw": float(net.load.at[idx, "p_mw"]),
                "q_mvar": float(net.load.at[idx, "q_mvar"]),
            })
        components["Load"] = units

    # P2G (if present)
    if "PtG" in net.keys() and isinstance(net["PtG"], pd.DataFrame):
        units = []
        for idx in net["PtG"].index:
            units.append({
                "index": int(idx),
                "bus": int(net["PtG"].at[idx, "bus"]),
                "max_p_mw": float(net["PtG"].at[idx, "max_p_mw"]),
                "junction": int(net["PtG"].at[idx, "junction"]),
                "heating_value": float(net["PtG"].at[idx, "heating_value"]),
                "efficiency": float(net["PtG"].at[idx, "efficiency"]),
            })
        components["P2G"] = units

    # MPC configuration
    components["MPC_config"] = {
        "mpc_steps": 96,
        "horizon": 3,
        "dt_min": 15,
        "v_min": 0.9,
        "v_max": 1.1,
        "objective": "quadratic_exchange",
        "solver": "ipopt",
    }

    path = os.path.join(output_dir, "component_params.json")
    with open(path, "w") as f:
        json.dump(components, f, indent=2, default=_to_serialisable)
    print(f"  Saved {path}")


def export_profiles(net, output_dir):
    """Export time-series profiles as CSV files."""
    profile_dir = os.path.join(output_dir, "profiles")
    os.makedirs(profile_dir, exist_ok=True)

    scenario_names = ["baseline", "disturbance"]

    for s, name in enumerate(scenario_names):
        # Load P
        df = pd.DataFrame(
            net["profiles_load_p"][s].T,
            columns=[f"load_{i}" for i in range(net["profiles_load_p"].shape[1])],
        )
        path = os.path.join(profile_dir, f"load_p_mw_{name}.csv")
        df.to_csv(path, index_label="timestep")
        print(f"  Saved {path}")

        # Load Q
        df = pd.DataFrame(
            net["profiles_load_q"][s].T,
            columns=[f"load_{i}" for i in range(net["profiles_load_q"].shape[1])],
        )
        path = os.path.join(profile_dir, f"load_q_mvar_{name}.csv")
        df.to_csv(path, index_label="timestep")
        print(f"  Saved {path}")

        # PV P
        df = pd.DataFrame(
            net["profiles_pv_p"][s].T,
            columns=[f"sgen_{i}" for i in range(net["profiles_pv_p"].shape[1])],
        )
        path = os.path.join(profile_dir, f"pv_p_mw_{name}.csv")
        df.to_csv(path, index_label="timestep")
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the OSMSES 2026 case study network and export data."
    )
    parser.add_argument(
        "--with-ptg",
        action="store_true",
        help="Include the Power-to-Gas unit (P2G scenario).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for exported files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building case study network...")
    net = create_base_network()

    if args.with_ptg:
        print("Adding P2G unit...")
        net = add_ptg_unit(net)

    print("Generating time-series profiles...")
    net = generate_profiles(net)

    print(f"\nExporting to {args.output_dir}:")
    export_network_params(net, args.output_dir)
    export_component_params(net, args.output_dir)
    export_profiles(net, args.output_dir)

    # Save complete pandapower network as JSON
    net_path = os.path.join(args.output_dir, "case_network.json")
    pp.to_json(net, net_path)
    print(f"  Saved {net_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
