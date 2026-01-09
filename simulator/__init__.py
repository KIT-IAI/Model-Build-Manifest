# -*- coding: utf-8 -*-
"""
Simulator Module
================

This module contains the physical model domain logic.
According to the Model-Build-Manifest pattern, simulator maintainers:
1. Understand the physical model
2. Provide a standardized manifest.json (declarative interface)
3. Provide a constraint library (mathematical formula implementations)

The simulator does NOT contain control logic.
"""

from .manifest_factory import ManifestFactory
from .model_data import ModelData
from .constraint_library import CONSTRAINT_LIBRARY

__all__ = ['ManifestFactory', 'ModelData', 'CONSTRAINT_LIBRARY']
