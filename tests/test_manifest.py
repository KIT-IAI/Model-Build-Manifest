# -*- coding: utf-8 -*-
"""
Test Manifest Generation
========================

Tests for the ManifestFactory class to ensure correct manifest generation.
"""

import pytest
import json
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import ManifestFactory, ModelData


class TestManifestFactory:
    """Test suite for ManifestFactory."""
    
    @pytest.fixture
    def simple_network(self):
        """Create a simple 4-bus network for testing."""
        net = pp.create_empty_network(sn_mva=1.0)
        
        # Create buses
        b0 = pp.create_bus(net, vn_kv=20.0, name="Slack Bus")
        b1 = pp.create_bus(net, vn_kv=20.0, name="Load Bus 1")
        b2 = pp.create_bus(net, vn_kv=20.0, name="Load Bus 2")
        b3 = pp.create_bus(net, vn_kv=20.0, name="Gen Bus")
        
        # Create external grid (slack)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        
        # Create lines
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, 
                       std_type="NAYY 4x50 SE")
        pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0,
                       std_type="NAYY 4x50 SE")
        pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1.0,
                       std_type="NAYY 4x50 SE")
        
        # Create loads
        pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.05)
        pp.create_load(net, bus=b2, p_mw=0.15, q_mvar=0.08)
        
        # Create generator
        pp.create_sgen(net, bus=b3, p_mw=0.2, q_mvar=0.0)
        
        return net
    
    @pytest.fixture
    def model_data(self, simple_network):
        """Create ModelData from the simple network."""
        config = {
            "horizon": 1,
            "v_min": 0.9,
            "v_max": 1.1,
            "flow_constraint": False,
            "objective": "quadratic_exchange",
            "with_timeseries": False
        }
        return ModelData(simple_network, config)
    
    @pytest.fixture
    def manifest_factory(self, model_data):
        """Create ManifestFactory instance."""
        return ManifestFactory(model_data)
    
    def test_create_manifest_structure(self, manifest_factory):
        """Test that manifest has required top-level keys."""
        manifest = manifest_factory.create_manifest()
        
        assert "Sets" in manifest
        assert "Parameters" in manifest
        assert "Variables" in manifest
        assert "Constraints" in manifest
        assert "Enet_keys" in manifest
    
    def test_sets_nodes(self, manifest_factory, simple_network):
        """Test that nodes set is correctly created."""
        manifest = manifest_factory.create_manifest()
        
        assert "nodes" in manifest["Sets"]
        assert manifest["Sets"]["nodes"]["type"] == "simple"
        assert len(manifest["Sets"]["nodes"]["data"]) == 4
    
    def test_sets_edges(self, manifest_factory):
        """Test that edges sets are correctly created."""
        manifest = manifest_factory.create_manifest()
        
        assert "edges" in manifest["Sets"]
        assert "lines" in manifest["Sets"]
        assert manifest["Sets"]["edges"]["type"] == "simple"
        assert len(manifest["Sets"]["lines"]["data"]) == 3
    
    def test_sets_mappings(self, manifest_factory):
        """Test that component mappings are created."""
        manifest = manifest_factory.create_manifest()
        
        assert "mapping_load" in manifest["Sets"]
        assert "mapping_sgen" in manifest["Sets"]
        assert manifest["Sets"]["mapping_load"]["type"] == "indexed"
    
    def test_parameters_line_admittance(self, manifest_factory):
        """Test that line admittance parameters are created."""
        manifest = manifest_factory.create_manifest()
        
        for param in ['g_ff', 'b_ff', 'g_ft', 'b_ft', 'g_tt', 'b_tt', 'g_tf', 'b_tf']:
            assert param in manifest["Parameters"]
            assert manifest["Parameters"][param]["type"] == "indexed"
            assert manifest["Parameters"][param]["index"] == "edges"
    
    def test_parameters_voltage_bounds(self, manifest_factory):
        """Test that voltage bound parameters are created."""
        manifest = manifest_factory.create_manifest()
        
        assert "vm_pu_lb" in manifest["Parameters"]
        assert "vm_pu_ub" in manifest["Parameters"]
        
        # Check values
        for key, value in manifest["Parameters"]["vm_pu_lb"]["data"].items():
            assert value == 0.9  # v_min from config
        for key, value in manifest["Parameters"]["vm_pu_ub"]["data"].items():
            assert value == 1.1  # v_max from config
    
    def test_variables_structure(self, manifest_factory):
        """Test that variables are correctly defined."""
        manifest = manifest_factory.create_manifest()
        
        required_vars = ['P', 'Q', 'U', 'W', 'Pload', 'Qload', 'Psgen', 'Qsgen',
                         'p_f', 'q_f', 'p_t', 'q_t', 'i_f_real', 'i_f_imag']
        
        for var_name in required_vars:
            assert var_name in manifest["Variables"]
            assert "indices" in manifest["Variables"][var_name]
            assert "domain" in manifest["Variables"][var_name]
    
    def test_constraints_structure(self, manifest_factory):
        """Test that constraints are correctly defined."""
        manifest = manifest_factory.create_manifest()
        
        required_constraints = [
            'vm_pu_lb_constr', 'vm_pu_ub_constr',
            'nodal_power_balance_P', 'nodal_power_balance_Q',
            'nodal_power_inj_P', 'nodal_power_inj_Q',
            'edge_flow_p_from', 'edge_flow_p_to',
            'edge_flow_q_from', 'edge_flow_q_to'
        ]
        
        for constr_name in required_constraints:
            assert constr_name in manifest["Constraints"]
            assert "indices" in manifest["Constraints"][constr_name]
            assert "rule_name" in manifest["Constraints"][constr_name]
    
    def test_manifest_json_serializable(self, manifest_factory, tmp_path):
        """Test that manifest can be saved to JSON."""
        manifest = manifest_factory.create_manifest()
        
        # Should not raise any errors
        filepath = tmp_path / "test_manifest.json"
        with open(filepath, 'w') as f:
            json.dump(manifest, f)
        
        # Verify it can be loaded back
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == manifest
    
    def test_save_manifest(self, manifest_factory, tmp_path):
        """Test the save_manifest method."""
        filepath = tmp_path / "output_manifest.json"
        manifest_factory.save_manifest(str(filepath))
        
        assert filepath.exists()
        
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert "Sets" in loaded
        assert "Constraints" in loaded


class TestModelData:
    """Test suite for ModelData class."""
    
    @pytest.fixture
    def simple_network(self):
        """Create a simple network for testing."""
        net = pp.create_empty_network(sn_mva=1.0)
        
        b0 = pp.create_bus(net, vn_kv=20.0)
        b1 = pp.create_bus(net, vn_kv=20.0)
        
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, 
                       std_type="NAYY 4x50 SE")
        pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.05)
        
        return net
    
    def test_model_data_creation(self, simple_network):
        """Test ModelData creation."""
        config = {"horizon": 1, "with_timeseries": False, 
                  "objective": "quadratic_exchange", "flow_constraint": False}
        model_data = ModelData(simple_network, config)
        
        assert model_data.Enet is not None
        assert model_data.config is not None
    
    def test_get_buses(self, simple_network):
        """Test get_buses method."""
        model_data = ModelData(simple_network, {
            "horizon": 1, "with_timeseries": False,
            "objective": "quadratic_exchange", "flow_constraint": False
        })
        
        buses = model_data.get_buses()
        assert len(buses) == 2
    
    def test_get_edges(self, simple_network):
        """Test get_edges method."""
        model_data = ModelData(simple_network, {
            "horizon": 1, "with_timeseries": False,
            "objective": "quadratic_exchange", "flow_constraint": False
        })
        
        all_edges = model_data.get_edges("all")
        line_edges = model_data.get_edges("line")
        
        assert len(all_edges) >= 1
        assert len(line_edges) == 1
    
    def test_get_line_parameters(self, simple_network):
        """Test line parameter extraction."""
        model_data = ModelData(simple_network, {
            "horizon": 1, "with_timeseries": False,
            "objective": "quadratic_exchange", "flow_constraint": False
        })
        
        params = model_data.get_line_parameters()
        assert len(params) == 8  # 8 parameter dictionaries
        
        # Check that parameters have values
        for param_dict in params:
            assert len(param_dict) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
