# -*- coding: utf-8 -*-
"""
Controller Module
=================

This module contains the generic optimization controller logic.
According to the Model-Build-Manifest pattern, controller maintainers:
1. Understand control strategies (NOT physics)
2. Write general assembly logic that reads manifests
3. Build and solve optimization problems dynamically

The controller does NOT contain domain-specific physics knowledge.
"""

from .optimizer import ModelAssembler, MPCController
from .solver_config import SolverConfig

__all__ = ['ModelAssembler', 'MPCController', 'SolverConfig']
