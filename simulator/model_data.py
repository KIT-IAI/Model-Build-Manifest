# -*- coding: utf-8 -*-
"""
Model Data Module
=================

Handles Pandapower network loading, validation, and preprocessing.
This module is part of the simulator domain and understands the physical model.
"""

import pandapower as pp
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


class ModelData:
    """
    Handles Pandapower network data loading and preprocessing.
    
    This class is responsible for:
    1. Loading and validating the Pandapower network
    2. Running initial power flow
    3. Extracting network topology information
    4. Computing line parameters from admittance matrices
    """
    
    # Configuration constants
    ALLOWED_OBJECTIVES = ['quadratic_exchange', 'transmission_losses', 'min_cost', 
                          "variable_cost", "binary_cost"]
    ALLOWED_FLOWCONSTR = ['power', 'current', False, 'both']
    NECESSARY_TIME_SERIES_PROFILES = ['profiles_load_p', 'profiles_load_q']
    NECESSARY_TIME_SERIES_SGEN_PROFILES = ['profiles_pv_p', 'profiles_pv_q']
    NECESSARY_TIME_SERIES_GEN_PROFILES = ['profiles_gen_p', 'profiles_gen_q']
    NECESSARY_TIME_SERIES_STO_PROFILES = ['profiles_sto_soc']
    
    STANDARD_CONFIG = {
        "solver": "ipopt",
        "solver_options": {"verbose": True, "tee": False},
        "with_timeseries": False,
        "mpc_with_initial_values": True,
        "mpc_steps": 96,
        "horizon": 96,
        "dt_min": 15,
        "flow_constraint": False,
        "v_min": 0.9,
        "v_max": 1.1,
        "with_storage_ramp": False,
        "objective": "min_cost"
    }

    def __init__(self, Enet: pp.pandapowerNet, config: Optional[Dict] = None):
        """
        Initialize the ModelData with a Pandapower network.
        
        Args:
            Enet: Pandapower network object
            config: Configuration dictionary (optional)
        """
        self.config = self._merge_config(config or {})
        self.Enet = self._validate_and_prepare(Enet)
        
    def _merge_config(self, user_config: Dict) -> Dict:
        """Merge user config with standard config."""
        config = self.STANDARD_CONFIG.copy()
        config.update(user_config)
        config["horizon"] = 1 if not config["with_timeseries"] else config["horizon"]
        return config
    
    def _validate_and_prepare(self, Enet: pp.pandapowerNet) -> pp.pandapowerNet:
        """Validate and prepare the Pandapower network."""
        print("Step Prepare: Validating Enet")
        Enet = pp.pandapowerNet(Enet)
        
        self._check_sn_mva(Enet)
        self._check_objective(Enet)
        self._check_flow_constraint()
        self._check_time_series(Enet)
        self._add_small_resistance(Enet)
        self._run_powerflow(Enet)
        
        return pp.pandapowerNet(Enet)
    
    def _check_sn_mva(self, Enet: pp.pandapowerNet):
        """Check if Norm apparent power equals 1."""
        assert Enet.sn_mva == 1, f'Normscheinleistung in MVA: {Enet.sn_mva}, should be 1.0!'
    
    def _check_objective(self, Enet: pp.pandapowerNet):
        """Check if objective is valid."""
        objective = self.config["objective"]
        assert objective in self.ALLOWED_OBJECTIVES, \
            f'Objective "{objective}" not allowed. Must be in {self.ALLOWED_OBJECTIVES}'
        if objective == "min_cost" and Enet.poly_cost.empty:
            raise ValueError('Objective "min_cost" not allowed. No cost data available!')
    
    def _check_flow_constraint(self):
        """Check if flow constraint is valid."""
        flow_constraint = self.config["flow_constraint"]
        assert flow_constraint in self.ALLOWED_FLOWCONSTR, \
            f'"flow_constraint" {flow_constraint} not allowed. Must be in {self.ALLOWED_FLOWCONSTR}'
    
    def _check_time_series(self, Enet: pp.pandapowerNet):
        """Check time series profiles if enabled."""
        if not self.config["with_timeseries"]:
            return
            
        H = self.config["horizon"]
        if not set(self.NECESSARY_TIME_SERIES_PROFILES).issubset(Enet.keys()):
            return
            
        assert len(Enet['profiles_load_p']) == len(Enet['profiles_load_q']) == len(Enet['profiles_pv_p']), \
            'Number of time series scenarios do not match!'
            
        for i in range(len(Enet['profiles_load_p'])):
            t_all = Enet['profiles_load_p'][i].shape[1]
            assert H <= t_all, 'Horizon H exceeds lengths of profiles!'
    
    def _add_small_resistance(self, Enet: pp.pandapowerNet):
        """Add small resistance for switches."""
        if 'switch' in Enet.keys() and Enet.switch.shape[0] != 0:
            print('Switches in net. Adding small resistance for consistency.')
            Enet.switch['z_ohm'] = 0.1
    
    def _run_powerflow(self, Enet: pp.pandapowerNet):
        """Run power flow if no solution exists."""
        if ('_ppc' not in Enet.keys()) or ('switch' in Enet.keys()):
            print('No powerflow solution yet -> running initialization powerflow!')
            try:
                pp.runpp(Enet, numba=False)
            except:
                if '_ppc' not in Enet.keys():
                    raise ValueError('Powerflow did not converge and no _ppc exists!')
    
    # ==================== Data Extraction Methods ====================
    
    def get_buses(self) -> np.ndarray:
        """Get bus indices from the network."""
        return self.Enet._ppc['internal']['bus'][:, 0].astype(int)
    
    def get_edges(self, edge_type: str = "all") -> List[List[int]]:
        """
        Get edge indices from the network.
        
        Args:
            edge_type: One of "all", "line", or "trafo"
        """
        if edge_type == "all":
            return [[f, t] for f, t in np.real(
                self.Enet._ppc['internal']['branch'][:, [0, 1]]).astype(int)]
        elif edge_type == "line":
            return [[f, t] for f, t in self.Enet.line[['from_bus', 'to_bus']].values.astype(int)]
        elif edge_type == "trafo":
            return [[f, t] for f, t in self.Enet.trafo[['hv_bus', 'lv_bus']].values.astype(int)]
        else:
            raise ValueError(f"Invalid edge_type: {edge_type}")
    
    def get_bus_indices(self, component_name: str) -> np.ndarray:
        """Get bus indices for a given component."""
        if component_name in self.Enet.keys():
            return self.Enet[component_name]['bus'].values.astype(int)
        raise ValueError(f"Invalid component_name: {component_name}")
    
    def get_line_parameters(self) -> Tuple[Dict, ...]:
        """
        Extract line admittance parameters from Yf and Yt matrices.
        
        Returns:
            Tuple of 8 dictionaries: g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf
        """
        Yf = self.Enet._ppc['internal']['Yf']
        Yt = self.Enet._ppc['internal']['Yt']
        
        g_ff, b_ff, g_ft, b_ft = {}, {}, {}, {}
        g_tt, b_tt, g_tf, b_tf = {}, {}, {}, {}
        
        edges = self.get_edges("all")
        
        for i, (f, t) in enumerate(edges):
            g_ff[(f, t)] = np.real(Yf[i, f])
            b_ff[(f, t)] = np.imag(Yf[i, f])
            g_ft[(f, t)] = np.real(Yf[i, t])
            b_ft[(f, t)] = np.imag(Yf[i, t])
            g_tt[(f, t)] = np.real(Yt[i, t])
            b_tt[(f, t)] = np.imag(Yt[i, t])
            g_tf[(f, t)] = np.real(Yt[i, f])
            b_tf[(f, t)] = np.imag(Yt[i, f])
        
        return (g_ff, b_ff, g_ft, b_ft, g_tt, b_tt, g_tf, b_tf)
    
    def get_inflow_outflow(self) -> Tuple[Dict[int, set], Dict[int, set]]:
        """Create mapping between input and output of edges."""
        inflow_set = defaultdict(set)
        outflow_set = defaultdict(set)
        
        for i, j in self.get_edges("all"):
            inflow_set[j].add(i)
            outflow_set[i].add(j)
        
        return dict(inflow_set), dict(outflow_set)
    
    def get_node_voltage_bounds(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get voltage lower and upper bounds for all nodes."""
        v_min = self.config["v_min"]
        v_max = self.config["v_max"]
        
        vm_pu_lb = {}
        vm_pu_ub = {}
        
        for i in self.Enet.bus.index.values:
            key = str(i)
            
            if "min_vm_pu" not in self.Enet.bus.columns:
                vm_pu_lb[key] = v_min
            else:
                vm_pu_lb[key] = v_min if np.isnan(self.Enet.bus["min_vm_pu"][i]) else self.Enet.bus["min_vm_pu"][i]
            
            if "max_vm_pu" not in self.Enet.bus.columns:
                vm_pu_ub[key] = v_max
            else:
                vm_pu_ub[key] = v_max if np.isnan(self.Enet.bus["max_vm_pu"][i]) else self.Enet.bus["max_vm_pu"][i]
        
        return vm_pu_lb, vm_pu_ub
    
    def get_component_mappings(self) -> Dict[str, Dict[str, List]]:
        """Create mappings from buses to components."""
        mappings = {
            'mapping_load': {},
            'mapping_sgen': {},
            'mapping_storage': {},
            'mapping_gen': {},
            'mapping_PtG': {},
            'mapping_GtP': {}
        }
        
        for i in self.Enet.bus.index.values:
            key = str(int(i))
            mappings['mapping_load'][key] = list(
                self.Enet.load[self.Enet.load['bus'] == i].index.values)
            mappings['mapping_sgen'][key] = list(
                self.Enet.sgen[self.Enet.sgen['bus'] == i].index.values)
            mappings['mapping_storage'][key] = list(
                self.Enet.storage[self.Enet.storage['bus'] == i].index.values)
            mappings['mapping_gen'][key] = list(
                self.Enet.gen[self.Enet.gen['bus'] == i].index.values)
            
            if "PtG" in self.Enet.keys():
                mappings['mapping_PtG'][key] = list(
                    self.Enet.PtG[self.Enet.PtG['bus'] == i].index.values)
            if "GtP" in self.Enet.keys():
                mappings['mapping_GtP'][key] = list(
                    self.Enet.GtP[self.Enet.GtP['bus'] == i].index.values)
        
        return mappings
