# -*- coding: utf-8 -*-
"""
Solver Configuration Module
===========================

Configuration management for optimization solvers.
This module is part of the controller domain and handles solver settings.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SolverConfig:
    """
    Configuration class for optimization solvers.
    
    Attributes:
        solver_name: Name of the solver (e.g., "ipopt", "gurobi", "glpk")
        options: Dictionary of solver-specific options
        verbose: Whether to print solver status
        tee: Whether to display solver output
    """
    solver_name: str = "ipopt"
    options: Dict[str, Any] = field(default_factory=lambda: {
        "halt_on_ampl_error": "yes",
        "tol": 1e-15
    })
    verbose: bool = True
    tee: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SolverConfig':
        """
        Create a SolverConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing solver configuration
            
        Returns:
            SolverConfig instance
        """
        return cls(
            solver_name=config_dict.get("solver", "ipopt"),
            options=config_dict.get("solver_options", {}).copy(),
            verbose=config_dict.get("solver_options", {}).get("verbose", True),
            tee=config_dict.get("solver_options", {}).get("tee", False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary containing solver configuration
        """
        return {
            "solver": self.solver_name,
            "solver_options": {
                **self.options,
                "verbose": self.verbose,
                "tee": self.tee
            }
        }


# Preset configurations for common solvers
IPOPT_CONFIG = SolverConfig(
    solver_name="ipopt",
    options={
        "halt_on_ampl_error": "yes",
        "tol": 1e-15,
        "max_iter": 3000
    }
)

GUROBI_CONFIG = SolverConfig(
    solver_name="gurobi",
    options={
        "MIPGap": 1e-4,
        "TimeLimit": 3600
    }
)

GLPK_CONFIG = SolverConfig(
    solver_name="glpk",
    options={}
)
