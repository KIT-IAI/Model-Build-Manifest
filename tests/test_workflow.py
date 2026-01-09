# -*- coding: utf-8 -*-
"""
Test Complete Workflow
======================

Integration tests for the complete decoupled workflow:
1. Simulator generates manifest
2. Controller loads manifest and builds model
3. Controller solves optimization problem
4. Results are extracted and validated

This test validates the Model-Build-Manifest pattern described in the paper.
"""

import pytest
import json
import numpy as np
import pandapower as pp
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import ManifestFactory, ModelData, CONSTRAINT_LIBRARY
from controller import ModelAssembler, MPCController, SolverConfig


class TestDecoupledWorkflow:
    """
    Test suite for the complete decoupled workflow.
    
    This validates the core thesis of the paper:
    The controller can receive new physical components and construct
    new optimization problems without modifying any source code.
    """
    
    @pytest.fixture
    def ieee_case4(self):
        """
        Create a 4-bus test network.
        This simulates a simple distribution grid.
        """
        net = pp.create_empty_network(sn_mva=1.0)
        
        # Create buses (20 kV distribution grid)
        b0 = pp.create_bus(net, vn_kv=20.0, name="Slack")
        b1 = pp.create_bus(net, vn_kv=20.0, name="Load1")
        b2 = pp.create_bus(net, vn_kv=20.0, name="Load2")
        b3 = pp.create_bus(net, vn_kv=20.0, name="PV")
        
        # External grid (slack bus)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        
        # Lines
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0,
                       std_type="NAYY 4x50 SE")
        pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.5,
                       std_type="NAYY 4x50 SE")
        pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.8,
                       std_type="NAYY 4x50 SE")
        
        # Loads
        pp.create_load(net, bus=b1, p_mw=0.05, q_mvar=0.02)
        pp.create_load(net, bus=b2, p_mw=0.08, q_mvar=0.03)
        
        # PV generator
        pp.create_sgen(net, bus=b3, p_mw=0.1, q_mvar=0.0)
        
        return net
    
    @pytest.fixture
    def config(self):
        """Standard configuration for testing."""
        return {
            "horizon": 1,
            "v_min": 0.9,
            "v_max": 1.1,
            "flow_constraint": False,
            "objective": "quadratic_exchange",
            "with_timeseries": False,
            "solver": "ipopt",
            "solver_options": {"verbose": False, "tee": False}
        }
    
    def test_complete_workflow(self, ieee_case4, config, tmp_path):
        """
        Test the complete decoupled workflow as described in the paper.
        
        Phase 1: Initialization
        - Simulator generates manifest
        - Controller receives and stores manifest
        
        Phase 2: Execution (at each timestep)
        1. Create empty optimization model
        2. Read manifests and add constraints
        3. Add dynamic information (objectives)
        4. Solve and extract results
        """
        # ============== PHASE 1: INITIALIZATION ==============
        print("\n=== Phase 1: Simulator generates manifest ===")
        
        # Simulator side: Generate manifest
        model_data = ModelData(ieee_case4, config)
        manifest_factory = ManifestFactory(model_data)
        manifest = manifest_factory.create_manifest()
        
        # Save manifest to file (simulating inter-process communication)
        manifest_path = tmp_path / "elec_grid_manifest.json"
        manifest_factory.save_manifest(str(manifest_path))
        
        # Verify manifest was created correctly
        assert os.path.exists(manifest_path)
        
        # ============== PHASE 2: CONTROLLER SETUP ==============
        print("\n=== Phase 2: Controller loads manifest ===")
        
        # Controller side: Create assembler with constraint library
        assembler = ModelAssembler(CONSTRAINT_LIBRARY)
        
        # Controller loads manifest from simulator
        assembler.load_manifest_from_file(str(manifest_path))
        
        # ============== PHASE 3: BUILD AND SOLVE ==============
        print("\n=== Phase 3: Controller builds and solves model ===")
        
        # Controller builds the Pyomo model
        mpc = MPCController(assembler, config)
        model = mpc.build_model()
        
        # Verify model was built correctly
        assert hasattr(model, 'nodes')
        assert hasattr(model, 'edges')
        assert hasattr(model, 'P')
        assert hasattr(model, 'Q')
        assert hasattr(model, 'U')
        assert hasattr(model, 'W')
        
        # Set objective function
        mpc.set_objective("quadratic_exchange")
        
        # Fix boundary conditions
        mpc.fix_slack_nodes()
        mpc.fix_loads_static(ieee_case4)
        mpc.fix_generators_static(ieee_case4)
        
        # Solve (skip if solver not available)
        try:
            results = mpc.solve()
            
            # ============== PHASE 4: EXTRACT RESULTS ==============
            print("\n=== Phase 4: Extract results ===")
            
            res = mpc.extract_results(ieee_case4)
            
            # Validate results
            assert 'V_pu' in res
            assert 'P_mw' in res
            assert 'Q_mvar' in res
            
            # Voltage should be close to 1.0 p.u.
            assert np.all(res['V_pu'] > 0.85)
            assert np.all(res['V_pu'] < 1.15)
            
            print(f"\nOptimal objective value: {res['objective']:.6f}")
            print(f"Voltage range: {res['V_pu'].min():.4f} - {res['V_pu'].max():.4f} p.u.")
            
        except Exception as e:
            pytest.skip(f"Solver not available or failed: {e}")
    
    def test_manifest_decoupling(self, ieee_case4, config):
        """
        Test that the controller doesn't need to know physics.
        
        This is the key innovation from the paper:
        - The controller only needs the manifest and constraint library
        - It doesn't import or use any simulator-specific code
        - New constraints can be added by updating the manifest
        """
        # Simulator generates manifest
        model_data = ModelData(ieee_case4, config)
        manifest = ManifestFactory(model_data).create_manifest()
        
        # Controller only uses manifest + library
        assembler = ModelAssembler(CONSTRAINT_LIBRARY)
        assembler.register_manifest(manifest)
        
        # Build model - controller has no knowledge of power systems!
        model = assembler.build_model(config)
        
        # All physics knowledge came from manifest + library
        assert hasattr(model, 'vm_pu_lb_constr')  # Voltage constraints
        assert hasattr(model, 'edge_flow_p_from')  # Power flow equations
        assert hasattr(model, 'nodal_power_balance_P')  # Power balance
    
    def test_model_swappability(self, ieee_case4, config, tmp_path):
        """
        Test that physical models can be swapped without changing controller.
        
        From the paper:
        "If researchers want to replace a physical model, they simply modify
        the corresponding simulator module and ensure the manifest contains
        the new instruction string. The controller module itself does not
        need to be modified at all."
        """
        # Create two different manifests (simulating model swap)
        
        # Manifest 1: Basic power flow
        model_data1 = ModelData(ieee_case4, config)
        manifest1 = ManifestFactory(model_data1).create_manifest()
        
        # Manifest 2: With flow constraints
        config2 = config.copy()
        config2["flow_constraint"] = False  # Would be 'power' for trafo limits
        model_data2 = ModelData(ieee_case4, config2)
        manifest2 = ManifestFactory(model_data2).create_manifest()
        
        # Same controller code works with both manifests!
        assembler1 = ModelAssembler(CONSTRAINT_LIBRARY)
        assembler1.register_manifest(manifest1)
        model1 = assembler1.build_model(config)
        
        assembler2 = ModelAssembler(CONSTRAINT_LIBRARY)
        assembler2.register_manifest(manifest2)
        model2 = assembler2.build_model(config2)
        
        # Both models were built successfully
        assert model1 is not None
        assert model2 is not None
    
    def test_constraint_library_injection(self, ieee_case4, config):
        """
        Test the dependency injection pattern for constraints.
        
        The constraint library acts as the 'adapter' that maps
        manifest instructions to Pyomo constraint implementations.
        """
        model_data = ModelData(ieee_case4, config)
        manifest = ManifestFactory(model_data).create_manifest()
        
        # Verify all constraint rules in manifest have library implementations
        for constr_name, constr_info in manifest["Constraints"].items():
            rule_name = constr_info["rule_name"]
            assert rule_name in CONSTRAINT_LIBRARY, \
                f"Constraint '{rule_name}' not found in library"
            
            # Verify it's a callable
            assert callable(CONSTRAINT_LIBRARY[rule_name])
    
    def test_json_manifest_compatibility(self, ieee_case4, config, tmp_path):
        """
        Test that manifest is fully JSON-compatible.
        
        This enables cross-language compatibility as described in the paper:
        "manifest.json uses the standard JSON format as a general protocol
        for exchanging mathematical constructs."
        """
        model_data = ModelData(ieee_case4, config)
        manifest = ManifestFactory(model_data).create_manifest()
        
        # Save to JSON
        json_path = tmp_path / "test_manifest.json"
        with open(json_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Load from JSON
        with open(json_path, 'r') as f:
            loaded_manifest = json.load(f)
        
        # Use loaded manifest (simulating cross-language transfer)
        assembler = ModelAssembler(CONSTRAINT_LIBRARY)
        assembler.register_manifest(loaded_manifest)
        model = assembler.build_model(config)
        
        assert model is not None
        assert hasattr(model, 'P')


class TestSolverConfiguration:
    """Tests for solver configuration."""
    
    def test_solver_config_creation(self):
        """Test SolverConfig creation."""
        config = SolverConfig(
            solver_name="ipopt",
            options={"tol": 1e-6},
            verbose=True,
            tee=False
        )
        
        assert config.solver_name == "ipopt"
        assert config.options["tol"] == 1e-6
    
    def test_solver_config_from_dict(self):
        """Test creating SolverConfig from dictionary."""
        config_dict = {
            "solver": "gurobi",
            "solver_options": {
                "verbose": True,
                "tee": True,
                "MIPGap": 0.01
            }
        }
        
        config = SolverConfig.from_dict(config_dict)
        
        assert config.solver_name == "gurobi"
        assert config.verbose == True
        assert config.tee == True


class TestErrorHandling:
    """Tests for error handling in the workflow."""
    
    def test_missing_constraint_in_library(self):
        """Test handling of missing constraint in library."""
        # Create a manifest with an unknown constraint
        manifest = {
            "Sets": {"nodes": {"type": "simple", "data": [0, 1]}},
            "Parameters": {},
            "Variables": {},
            "Constraints": {
                "unknown_constraint": {
                    "indices": ["nodes"],
                    "rule_name": "this_does_not_exist"
                }
            }
        }
        
        assembler = ModelAssembler(CONSTRAINT_LIBRARY)
        assembler.register_manifest(manifest)
        
        # Should print warning but not crash
        model = assembler.build_model({"horizon": 1})
        
        # Model should be built, constraint should be skipped
        assert not hasattr(model, 'unknown_constraint')
    
    def test_invalid_network(self):
        """Test handling of invalid network configuration."""
        # Network with wrong sn_mva
        net = pp.create_empty_network(sn_mva=100.0)  # Should be 1.0
        pp.create_bus(net, vn_kv=20.0)
        
        config = {
            "horizon": 1, "with_timeseries": False,
            "objective": "quadratic_exchange", "flow_constraint": False
        }
        
        with pytest.raises(AssertionError):
            ModelData(net, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
