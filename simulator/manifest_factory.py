# -*- coding: utf-8 -*-
"""
Manifest Factory Module
=======================

Core component of the Model-Build-Manifest pattern.
This module generates the standardized JSON manifest that declares:
1. Sets: Index sets for the optimization model
2. Parameters: Static parameters required to build the model
3. Variables: Decision variables with their domains and bounds
4. Constraints: Abstract constraint-building instructions

The manifest serves as a declarative interface between the simulator and controller.
"""

import json
import numpy as np
from typing import Dict, Any, Optional
from .model_data import ModelData


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    return obj


class ManifestFactory:
    """
    Factory class for generating Model-Build-Manifest JSON files.

    According to the paper, the simulator maintainer is responsible for:
    1. Understanding the physical model
    2. Providing a standardized manifest.json
    3. Providing a constraint library with formula implementations

    This class handles item (2) - generating the manifest.
    """

    def __init__(self, model_data: ModelData):
        """
        Initialize the ManifestFactory.

        Args:
            model_data: ModelData instance containing the Pandapower network
        """
        self.model_data = model_data
        self.Enet = model_data.Enet
        self.config = model_data.config

    def create_manifest(self) -> Dict[str, Any]:
        """
        Create the complete manifest dictionary.

        Returns:
            Dictionary containing Sets, Parameters, Variables, and Constraints
        """
        manifest = {
            "Sets": {},
            "Parameters": {},
            "Variables": {},
            "Constraints": {},
            "Enet_keys": list(self.Enet.keys())
        }

        self._add_sets(manifest)
        self._add_mappings(manifest)
        self._add_parameters(manifest)
        self._add_variables(manifest)
        self._add_constraints(manifest)

        # Convert all NumPy types to native Python types for JSON serialization
        return _convert_numpy_types(manifest)

    def save_manifest(self, filepath: str) -> None:
        """
        Save the manifest to a JSON file.

        Args:
            filepath: Path to save the manifest JSON file
        """
        manifest = self.create_manifest()
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2, cls=NumpyEncoder)
        print(f"Manifest saved to {filepath}")

    # ==================== Private Methods ====================

    def _add_sets(self, manifest: Dict) -> None:
        """Add basic sets to the manifest."""
        manifest["Sets"]["nodes"] = {
            "type": "simple",
            "data": self.model_data.get_buses().tolist()
        }
        manifest["Sets"]["edges"] = {
            "type": "simple",
            "data": self.model_data.get_edges('all')
        }
        manifest["Sets"]["lines"] = {
            "type": "simple",
            "data": self.model_data.get_edges('line')
        }
        manifest["Sets"]["trafo"] = {
            "type": "simple",
            "data": self.model_data.get_edges('trafo')
        }
        manifest["Sets"]["slack_nodes"] = {
            "type": "simple",
            "data": self.model_data.get_bus_indices('ext_grid').tolist()
        }
        manifest["Sets"]["pq_nodes"] = {
            "type": "simple",
            "data": self.Enet.load.index.astype(int).tolist()
        }
        manifest["Sets"]["sgen_nodes"] = {
            "type": "simple",
            "data": self.Enet.sgen.index.astype(int).tolist()
        }
        manifest["Sets"]["generator_nodes"] = {
            "type": "simple",
            "data": self.Enet.gen.index.astype(int).tolist()
        }

        # Extended component sets (storage, PtG, GtP)
        if "storage" in self.Enet.keys() and not self.Enet.storage.empty:
            manifest["Sets"]["storage_nodes"] = {
                "type": "simple",
                "data": self.Enet.storage.index.astype(int).tolist()
            }

        if "PtG" in self.Enet.keys() and not self.Enet.PtG.empty:
            manifest["Sets"]["PtG_nodes"] = {
                "type": "simple",
                "data": self.Enet.PtG.index.astype(int).tolist()
            }

        if "GtP" in self.Enet.keys() and not self.Enet.GtP.empty:
            manifest["Sets"]["GtP_nodes"] = {
                "type": "simple",
                "data": self.Enet.GtP.index.astype(int).tolist()
            }

    def _add_mappings(self, manifest: Dict) -> None:
        """Add indexed mappings (bus -> components) to the manifest."""
        mappings = self.model_data.get_component_mappings()

        for mapping_name, mapping_data in mappings.items():
            manifest["Sets"][mapping_name] = {
                "type": "indexed",
                "index": "nodes",
                "data": mapping_data
            }

    def _add_parameters(self, manifest: Dict) -> None:
        """Add all parameters to the manifest."""
        self._add_line_parameters(manifest)
        self._add_simple_parameters(manifest)
        self._add_node_voltage_parameters(manifest)
        self._add_inflow_outflow_parameters(manifest)
        self._add_flow_limit_parameters(manifest)
        self._add_extended_component_parameters(manifest)

    def _add_line_parameters(self, manifest: Dict) -> None:
        """Add line admittance parameters."""
        g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf = self.model_data.get_line_parameters()

        param_names = ['g_ff', 'b_ff', 'g_ft', 'b_ft', 'g_tt', 'b_tt', 'g_tf', 'b_tf']
        param_values = [g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf]

        for name, values in zip(param_names, param_values):
            # Convert (f, t) tuple keys to "f,t" string keys for JSON compatibility
            processed_data = {f"{f},{t}": v for (f, t), v in values.items()}
            manifest["Parameters"][name] = {
                "type": "indexed",
                "index": "edges",
                "data": processed_data,
                "default": 0.0
            }

    def _add_simple_parameters(self, manifest: Dict) -> None:
        """Add simple (scalar) parameters."""
        manifest["Parameters"]["sn_mva"] = {
            "type": "simple",
            "data": self.Enet.sn_mva
        }

    def _add_node_voltage_parameters(self, manifest: Dict) -> None:
        """Add node voltage parameters."""
        # Node voltage levels
        vn_data = {str(i): k for i, k in enumerate(self.Enet.bus['vn_kv'].values)}
        manifest["Parameters"]["vn"] = {
            "type": "indexed",
            "index": "nodes",
            "data": vn_data,
            "default": 0.0
        }

        # Voltage bounds
        vm_pu_lb, vm_pu_ub = self.model_data.get_node_voltage_bounds()

        manifest["Parameters"]["vm_pu_lb"] = {
            "type": "indexed",
            "index": "nodes",
            "data": vm_pu_lb,
            "default": self.config["v_min"]
        }
        manifest["Parameters"]["vm_pu_ub"] = {
            "type": "indexed",
            "index": "nodes",
            "data": vm_pu_ub,
            "default": self.config["v_max"]
        }

    def _add_inflow_outflow_parameters(self, manifest: Dict) -> None:
        """Add inflow/outflow set parameters."""
        inflow_set, outflow_set = self.model_data.get_inflow_outflow()

        # Convert to JSON-compatible format
        inflow_data = {str(k): list(v) for k, v in inflow_set.items()}
        outflow_data = {str(k): list(v) for k, v in outflow_set.items()}

        manifest["Parameters"]["inflow_set"] = {
            "type": "indexed",
            "index": "nodes",
            "data": inflow_data,
            "default": [],
            "domain": "Any"
        }
        manifest["Parameters"]["outflow_set"] = {
            "type": "indexed",
            "index": "nodes",
            "data": outflow_data,
            "default": [],
            "domain": "Any"
        }

    def _add_flow_limit_parameters(self, manifest: Dict) -> None:
        """Add flow constraint parameters based on config."""
        flow_constraint = self.config.get("flow_constraint", False)

        if flow_constraint in ['power', 'both']:
            trafo_limits = {
                (int(f), int(t)): s
                for f, t, s in self.Enet.trafo[['hv_bus', 'lv_bus', 'sn_mva']].values
            }
            processed_data = {f"{f},{t}": v for (f, t), v in trafo_limits.items()}
            manifest["Parameters"]["line_limit_s"] = {
                "type": "indexed",
                "index": "trafo",
                "data": processed_data,
                "default": 0.0
            }

        if flow_constraint in ['current', 'both']:
            line_limits = {
                (int(f), int(t)): i
                for f, t, i in self.Enet.line[['from_bus', 'to_bus', 'max_i_ka']].values
            }
            processed_data = {f"{f},{t}": v for (f, t), v in line_limits.items()}
            manifest["Parameters"]["line_limit_i"] = {
                "type": "indexed",
                "index": "lines",
                "data": processed_data,
                "default": 0.0
            }

    def _add_extended_component_parameters(self, manifest: Dict) -> None:
        """Add parameters for extended components (storage, PtG, GtP)."""
        # Storage parameters
        if "storage" in self.Enet.keys() and not self.Enet.storage.empty:
            # Storage maximum power
            storage_pmax = {
                str(int(i)): p for i, p in
                zip(self.Enet.storage.index, self.Enet.storage['max_p_mw'].values)
            }
            manifest["Parameters"]["Storage_Pmax"] = {
                "type": "indexed",
                "index": "storage_nodes",
                "data": storage_pmax,
                "default": 0.0
            }

            # Storage maximum energy
            storage_emax = {
                str(int(i)): e for i, e in
                zip(self.Enet.storage.index, self.Enet.storage['max_e_mwh'].values)
            }
            manifest["Parameters"]["Storage_E_max"] = {
                "type": "indexed",
                "index": "storage_nodes",
                "data": storage_emax,
                "default": 0.0
            }

            # Storage minimum power (if available)
            if 'min_p_mw' in self.Enet.storage.columns:
                storage_pmin = {
                    str(int(i)): p for i, p in
                    zip(self.Enet.storage.index, self.Enet.storage['min_p_mw'].values)
                }
                manifest["Parameters"]["Storage_Pmin"] = {
                    "type": "indexed",
                    "index": "storage_nodes",
                    "data": storage_pmin,
                    "default": 0.0
                }

        # PtG parameters
        if "PtG" in self.Enet.keys() and not self.Enet.PtG.empty:
            ptg_pmax = {
                str(int(i)): p for i, p in
                zip(self.Enet.PtG.index, self.Enet.PtG['max_p_mw'].values)
            }
            manifest["Parameters"]["PtG_Pmax"] = {
                "type": "indexed",
                "index": "PtG_nodes",
                "data": ptg_pmax,
                "default": 0.0
            }

        # GtP parameters
        if "GtP" in self.Enet.keys() and not self.Enet.GtP.empty:
            gtp_pmax = {
                str(int(i)): p for i, p in
                zip(self.Enet.GtP.index, self.Enet.GtP['max_p_mw'].values)
            }
            manifest["Parameters"]["GtP_Pmax"] = {
                "type": "indexed",
                "index": "GtP_nodes",
                "data": gtp_pmax,
                "default": 0.0
            }

    def _add_variables(self, manifest: Dict) -> None:
        """Add variable declarations to the manifest."""
        manifest["Variables"] = {
            # Node variables
            "P": {"indices": ["nodes", "T"], "domain": "Reals",
                  "initialize": None, "bounds": None},
            "Q": {"indices": ["nodes", "T"], "domain": "Reals",
                  "initialize": None, "bounds": None},
            "U": {"indices": ["nodes", "T"], "domain": "Reals",
                  "initialize": 1.0, "bounds": None},
            "W": {"indices": ["nodes", "T"], "domain": "Reals",
                  "initialize": 0.0, "bounds": None},

            # Load variables
            "Pload": {"indices": ["pq_nodes", "T"], "domain": "Reals",
                      "initialize": None, "bounds": None},
            "Qload": {"indices": ["pq_nodes", "T"], "domain": "Reals",
                      "initialize": None, "bounds": None},

            # Static generator variables
            "Psgen": {"indices": ["sgen_nodes", "T"], "domain": "Reals",
                      "initialize": None, "bounds": None},
            "Qsgen": {"indices": ["sgen_nodes", "T"], "domain": "Reals",
                      "initialize": None, "bounds": [0.0, 0.0]},

            # Generator variables
            "Pgen": {"indices": ["generator_nodes", "T"], "domain": "Reals",
                     "initialize": None, "bounds": None},
            "Qgen": {"indices": ["generator_nodes", "T"], "domain": "Reals",
                     "initialize": None, "bounds": None},

            # Branch variables (from bus)
            "p_f": {"indices": ["edges", "T"], "domain": "Reals",
                    "initialize": None, "bounds": None},
            "q_f": {"indices": ["edges", "T"], "domain": "Reals",
                    "initialize": None, "bounds": None},
            "i_f_real": {"indices": ["edges", "T"], "domain": "Reals",
                         "initialize": None, "bounds": None},
            "i_f_imag": {"indices": ["edges", "T"], "domain": "Reals",
                         "initialize": None, "bounds": None},

            # Branch variables (to bus)
            "p_t": {"indices": ["edges", "T"], "domain": "Reals",
                    "initialize": None, "bounds": None},
            "q_t": {"indices": ["edges", "T"], "domain": "Reals",
                    "initialize": None, "bounds": None},
            "i_t_real": {"indices": ["edges", "T"], "domain": "Reals",
                         "initialize": None, "bounds": None},
            "i_t_imag": {"indices": ["edges", "T"], "domain": "Reals",
                         "initialize": None, "bounds": None},
        }

        # Add extended component variables
        self._add_extended_component_variables(manifest)

    def _add_extended_component_variables(self, manifest: Dict) -> None:
        """Add variables for extended components (storage, PtG, GtP)."""
        # Storage variables
        if "storage" in self.Enet.keys() and not self.Enet.storage.empty:
            manifest["Variables"]["Psto"] = {
                "indices": ["storage_nodes", "T"],
                "domain": "Reals",
                "initialize": None,
                "bounds": None
            }
            manifest["Variables"]["Qsto"] = {
                "indices": ["storage_nodes", "T"],
                "domain": "Reals",
                "initialize": None,
                "bounds": [0.0, 0.0]
            }
            # Energy state needs T+1 indices (including initial state)
            manifest["Variables"]["E"] = {
                "indices": ["storage_nodes", "T_plus_one"],
                "domain": "NonNegativeReals",
                "initialize": None,
                "bounds": None
            }

        # PtG variables
        if "PtG" in self.Enet.keys() and not self.Enet.PtG.empty:
            manifest["Variables"]["Pptg"] = {
                "indices": ["PtG_nodes", "T"],
                "domain": "NonNegativeReals",
                "initialize": None,
                "bounds": None
            }
            manifest["Variables"]["Qptg"] = {
                "indices": ["PtG_nodes", "T"],
                "domain": "Reals",
                "initialize": None,
                "bounds": [0.0, 0.0]
            }

        # GtP variables
        if "GtP" in self.Enet.keys() and not self.Enet.GtP.empty:
            manifest["Variables"]["Pgtp"] = {
                "indices": ["GtP_nodes", "T"],
                "domain": "NonNegativeReals",
                "initialize": None,
                "bounds": None
            }
            manifest["Variables"]["Qgtp"] = {
                "indices": ["GtP_nodes", "T"],
                "domain": "Reals",
                "initialize": None,
                "bounds": [0.0, 0.0]
            }

    def _add_constraints(self, manifest: Dict) -> None:
        """Add constraint declarations to the manifest."""
        manifest["Constraints"] = {}

        # Voltage constraints
        manifest["Constraints"]["vm_pu_lb_constr"] = {
            "indices": ["nodes", "T"],
            "rule_name": "vm_pu_lb_constr"
        }
        manifest["Constraints"]["vm_pu_ub_constr"] = {
            "indices": ["nodes", "T"],
            "rule_name": "vm_pu_ub_constr"
        }

        # Power balance constraints
        manifest["Constraints"]["nodal_power_balance_P"] = {
            "indices": ["nodes", "T"],
            "rule_name": "nodal_power_balance_P"
        }
        manifest["Constraints"]["nodal_power_balance_Q"] = {
            "indices": ["nodes", "T"],
            "rule_name": "nodal_power_balance_Q"
        }

        # Power injection constraints: use extended version if storage/PtG/GtP present
        has_storage = "storage" in self.Enet.keys() and not self.Enet.storage.empty
        has_ptg = "PtG" in self.Enet.keys() and not self.Enet.PtG.empty
        has_gtp = "GtP" in self.Enet.keys() and not self.Enet.GtP.empty
        use_extended = has_storage or has_ptg or has_gtp

        if use_extended:
            manifest["Constraints"]["nodal_power_inj_P"] = {
                "indices": ["nodes", "T"],
                "rule_name": "nodal_power_injection_P_extended"
            }
            manifest["Constraints"]["nodal_power_inj_Q"] = {
                "indices": ["nodes", "T"],
                "rule_name": "nodal_power_injection_Q_extended"
            }
        else:
            manifest["Constraints"]["nodal_power_inj_P"] = {
                "indices": ["nodes", "T"],
                "rule_name": "nodal_power_injection_P"
            }
            manifest["Constraints"]["nodal_power_inj_Q"] = {
                "indices": ["nodes", "T"],
                "rule_name": "nodal_power_injection_Q"
            }

        # Edge flow constraints (P/Q)
        manifest["Constraints"]["edge_flow_p_from"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_p_from"
        }
        manifest["Constraints"]["edge_flow_p_to"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_p_to"
        }
        manifest["Constraints"]["edge_flow_q_from"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_q_from"
        }
        manifest["Constraints"]["edge_flow_q_to"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_q_to"
        }

        # Edge flow constraints (I)
        manifest["Constraints"]["edge_flow_i_from_real"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_i_from_real"
        }
        manifest["Constraints"]["edge_flow_i_from_imag"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_i_from_imag"
        }
        manifest["Constraints"]["edge_flow_i_to_real"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_i_to_real"
        }
        manifest["Constraints"]["edge_flow_i_to_imag"] = {
            "indices": ["edges", "T"],
            "rule_name": "edge_flow_i_to_imag"
        }

        # Conditional constraints (flow limits)
        flow_constraint = self.config.get("flow_constraint", False)

        if flow_constraint in ['power', 'both']:
            manifest["Constraints"]["line_limit_s_from"] = {
                "indices": ["trafo", "T"],
                "rule_name": "line_limit_s_from"
            }
            manifest["Constraints"]["line_limit_s_to"] = {
                "indices": ["trafo", "T"],
                "rule_name": "line_limit_s_to"
            }

        if flow_constraint in ['current', 'both']:
            manifest["Constraints"]["line_limit_i_from_con"] = {
                "indices": ["lines", "T"],
                "rule_name": "line_limit_i_from"
            }
            manifest["Constraints"]["line_limit_i_to_con"] = {
                "indices": ["lines", "T"],
                "rule_name": "line_limit_i_to"
            }

        # Storage constraints
        if has_storage:
            manifest["Constraints"]["storage_equation"] = {
                "indices": ["storage_nodes", "T"],
                "rule_name": "storage_equation"
            }

            # Storage ramp constraints (optional)
            # Note: ramp constraints use T_minus_one because they access t+1
            if self.config.get("with_storage_ramp", False):
                manifest["Constraints"]["storage_ramp_up"] = {
                    "indices": ["storage_nodes", "T_minus_one"],
                    "rule_name": "storage_ramp_up"
                }
                manifest["Constraints"]["storage_ramp_down"] = {
                    "indices": ["storage_nodes", "T_minus_one"],
                    "rule_name": "storage_ramp_down"
                }