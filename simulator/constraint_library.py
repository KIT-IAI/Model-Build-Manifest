# -*- coding: utf-8 -*-
"""
Constraint Library Module
=========================

Core component of the Model-Build-Manifest pattern.
This module contains all mathematical formula implementations (Python functions)
that correspond to the constraint instructions declared in the manifest.

According to the paper:
- The simulator maintainer provides this library
- The controller maintainer only needs to call predefined functions
- This is the "dependency injection" pattern applied to mathematical modeling

Each function follows the Pyomo constraint rule signature:
    def constraint_rule(model, *indices) -> Expression or Constraint.Skip
"""

import pyomo.environ as pe
from typing import Callable, Dict


# ============================================================================
# VOLTAGE CONSTRAINTS
# ============================================================================

def vm_pu_lb_constr(model, i, t):
    """
    Voltage magnitude lower bound constraint.
    See MA Equation 2.33
    
    U² + W² >= vm_pu_lb² (skip for slack nodes)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.U[i, t]**2 + model.W[i, t]**2
        rhs = model.vm_pu_lb[i]**2
        return lhs >= rhs


def vm_pu_ub_constr(model, i, t):
    """
    Voltage magnitude upper bound constraint.
    
    U² + W² <= vm_pu_ub² (skip for slack nodes)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.U[i, t]**2 + model.W[i, t]**2
        rhs = model.vm_pu_ub[i]**2
        return lhs <= rhs


# ============================================================================
# POWER BALANCE CONSTRAINTS
# ============================================================================

def nodal_power_balance_P(model, i, t):
    """
    Active power balance at each node.
    See MA Equation 2.29
    
    P[i,t] + Σ p_t[k,i,t] (inflow) + Σ p_f[i,l,t] (outflow) = 0
    """
    expr = (model.P[i, t] + 
            sum(model.p_t[k, i, t] for k in model.inflow_set[i]) + 
            sum(model.p_f[i, l, t] for l in model.outflow_set[i]) == 0)
    return expr


def nodal_power_balance_Q(model, i, t):
    """
    Reactive power balance at each node.
    See MA Equation 2.30
    
    Q[i,t] + Σ q_t[k,i,t] (inflow) + Σ q_f[i,l,t] (outflow) = 0
    """
    expr = (model.Q[i, t] + 
            sum(model.q_t[k, i, t] for k in model.inflow_set[i]) + 
            sum(model.q_f[i, l, t] for l in model.outflow_set[i]) == 0)
    return expr


def nodal_power_injection_P(model, i, t):
    """
    Active power injection constraint.
    See MA Equation 2.31
    
    P[i,t] = -Pgen - Psgen + Pload (skip for slack nodes)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.P[i, t]
        rhs = (-sum(model.Pgen[k, t] for k in model.mapping_gen[i]) 
               - sum(model.Psgen[k, t] for k in model.mapping_sgen[i])
               + sum(model.Pload[k, t] for k in model.mapping_load[i]))
        return rhs == lhs


def nodal_power_injection_Q(model, i, t):
    """
    Reactive power injection constraint.
    See MA Equation 2.32
    
    Q[i,t] = -Qgen - Qsgen + Qload (skip for slack nodes)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.Q[i, t]
        rhs = (-sum(model.Qgen[k, t] for k in model.mapping_gen[i]) 
               - sum(model.Qsgen[k, t] for k in model.mapping_sgen[i])
               + sum(model.Qload[k, t] for k in model.mapping_load[i]))
        return rhs == lhs


# ============================================================================
# EDGE FLOW CONSTRAINTS (POWER)
# ============================================================================

def edge_flow_p_from(model, i, j, t):
    """
    Active power flow from bus i to bus j (at from-bus side).
    See MA Equation 2.25 in rectangular coordinates
    """
    expr = model.p_f[i, j, t] == (
        model.g_ff[i, j] * (model.U[i, t]**2 + model.W[i, t]**2) +
        (model.U[i, t] * model.U[j, t] + model.W[i, t] * model.W[j, t]) * model.g_ft[i, j] +
        (model.W[i, t] * model.U[j, t] - model.U[i, t] * model.W[j, t]) * model.b_ft[i, j]
    )
    return expr


def edge_flow_p_to(model, i, j, t):
    """
    Active power flow from bus i to bus j (at to-bus side).
    See MA Equation 2.27 in rectangular coordinates
    """
    expr = model.p_t[i, j, t] == (
        model.g_tt[i, j] * (model.U[j, t]**2 + model.W[j, t]**2) +
        (model.U[i, t] * model.U[j, t] + model.W[i, t] * model.W[j, t]) * model.g_tf[i, j] +
        (model.W[j, t] * model.U[i, t] - model.U[j, t] * model.W[i, t]) * model.b_tf[i, j]
    )
    return expr


def edge_flow_q_from(model, i, j, t):
    """
    Reactive power flow from bus i to bus j (at from-bus side).
    See MA Equation 2.26 in rectangular coordinates
    """
    expr = model.q_f[i, j, t] == (
        -model.b_ff[i, j] * (model.U[i, t]**2 + model.W[i, t]**2) +
        (model.W[i, t] * model.U[j, t] - model.U[i, t] * model.W[j, t]) * model.g_ft[i, j] -
        (model.U[i, t] * model.U[j, t] + model.W[i, t] * model.W[j, t]) * model.b_ft[i, j]
    )
    return expr


def edge_flow_q_to(model, i, j, t):
    """
    Reactive power flow from bus i to bus j (at to-bus side).
    See MA Equation 2.28 in rectangular coordinates
    """
    expr = model.q_t[i, j, t] == (
        -model.b_tt[i, j] * (model.U[j, t]**2 + model.W[j, t]**2) +
        (model.W[j, t] * model.U[i, t] - model.U[j, t] * model.W[i, t]) * model.g_tf[i, j] -
        (model.U[i, t] * model.U[j, t] + model.W[i, t] * model.W[j, t]) * model.b_tf[i, j]
    )
    return expr


# ============================================================================
# EDGE FLOW CONSTRAINTS (CURRENT)
# ============================================================================

def edge_flow_i_from_real(model, i, j, t):
    """
    Real part of current at from-bus side.
    See MA Equation 2.20 in rectangular coordinates
    """
    expr = model.i_f_real[i, j, t] == (
        model.g_ff[i, j] * model.U[i, t] - model.b_ff[i, j] * model.W[i, t] +
        model.g_ft[i, j] * model.U[j, t] - model.b_ft[i, j] * model.W[j, t]
    )
    return expr


def edge_flow_i_from_imag(model, i, j, t):
    """
    Imaginary part of current at from-bus side.
    See MA Equation 2.21 in rectangular coordinates
    """
    expr = model.i_f_imag[i, j, t] == (
        model.b_ff[i, j] * model.U[i, t] + model.g_ff[i, j] * model.W[i, t] +
        model.b_ft[i, j] * model.U[j, t] + model.g_ft[i, j] * model.W[j, t]
    )
    return expr


def edge_flow_i_to_real(model, i, j, t):
    """
    Real part of current at to-bus side.
    See MA Equation 2.22 in rectangular coordinates
    """
    expr = model.i_t_real[i, j, t] == (
        model.g_tt[i, j] * model.U[j, t] - model.b_tt[i, j] * model.W[j, t] +
        model.g_tf[i, j] * model.U[i, t] - model.b_tf[i, j] * model.W[i, t]
    )
    return expr


def edge_flow_i_to_imag(model, i, j, t):
    """
    Imaginary part of current at to-bus side.
    See MA Equation 2.23 in rectangular coordinates
    """
    expr = model.i_t_imag[i, j, t] == (
        model.b_tt[i, j] * model.U[j, t] + model.g_tt[i, j] * model.W[j, t] +
        model.b_tf[i, j] * model.U[i, t] + model.g_tf[i, j] * model.W[i, t]
    )
    return expr


# ============================================================================
# LINE LIMIT CONSTRAINTS
# ============================================================================

def line_limit_s_from(model, i, j, t):
    """
    Apparent power limit at from-bus side (for transformers).
    
    p_f² + q_f² <= S_limit²
    """
    expr = 0 >= model.p_f[i, j, t]**2 + model.q_f[i, j, t]**2 - model.line_limit_s[i, j]**2
    return expr


def line_limit_s_to(model, i, j, t):
    """
    Apparent power limit at to-bus side (for transformers).
    
    p_t² + q_t² <= S_limit²
    """
    expr = 0 >= model.p_t[i, j, t]**2 + model.q_t[i, j, t]**2 - model.line_limit_s[i, j]**2
    return expr


def line_limit_i_from(model, i, j, t):
    """
    Current magnitude limit at from-bus side (for lines).
    Limits in kA, converted from p.u.
    """
    expr = 0 >= (
        (model.i_f_real[i, j, t]**2 + model.i_f_imag[i, j, t]**2) * 
        (model.sn_mva / (pe.sqrt(3) * model.vn[j]))**2 - 
        model.line_limit_i[i, j]**2
    )
    return expr


def line_limit_i_to(model, i, j, t):
    """
    Current magnitude limit at to-bus side (for lines).
    Limits in kA, converted from p.u.
    """
    expr = 0 >= (
        (model.i_t_real[i, j, t]**2 + model.i_t_imag[i, j, t]**2) * 
        (model.sn_mva / (pe.sqrt(3) * model.vn[j]))**2 - 
        model.line_limit_i[i, j]**2
    )
    return expr


# ============================================================================
# NON-BASIC COMPONENTS (Storage, PtG, GtP)
# ============================================================================

def storage_equation(model, i, t):
    """
    Storage energy balance equation.
    See MA Equation 2.35
    
    E[t+1] - E[t] = P_sto * (dt/60)
    """
    expr = model.E[i, t + 1] - model.E[i, t] - model.Psto[i, t] * (model.dt_min / 60.) == 0.
    return expr


def storage_ramp_up(model, i, t):
    """
    Storage ramp-up limit.
    See MA Equation 4.2
    
    P_sto[t] - P_sto[t+1] <= P_max * 0.2
    """
    expr = model.Psto[i, t] - model.Psto[i, t + 1] <= model.Storage_Pmax[i] * 0.2
    return expr


def storage_ramp_down(model, i, t):
    """
    Storage ramp-down limit.
    See MA Equation 4.3
    
    P_sto[t+1] - P_sto[t] <= P_max * 0.2
    """
    expr = model.Psto[i, t + 1] - model.Psto[i, t] <= model.Storage_Pmax[i] * 0.2
    return expr


def nodal_power_injection_P_extended(model, i, t):
    """
    Extended active power injection with storage, GtP, and PtG.
    See MA Equation 2.31 (extended)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.P[i, t]
        rhs = (-sum(model.Pgen[k, t] for k in model.mapping_gen[i]) 
               - sum(model.Psgen[k, t] for k in model.mapping_sgen[i])
               + sum(model.Pload[k, t] for k in model.mapping_load[i]))
        
        if "storage" in model.Enet_keys:
            rhs += sum(model.Psto[k, t] for k in model.mapping_storage[i])
        if "GtP" in model.Enet_keys:
            rhs += -sum(model.Pgtp[k, t] for k in model.mapping_GtP[i])
        if "PtG" in model.Enet_keys:
            rhs += sum(model.Pptg[k, t] for k in model.mapping_PtG[i])
        
        return rhs == lhs


def nodal_power_injection_Q_extended(model, i, t):
    """
    Extended reactive power injection with storage, GtP, and PtG.
    See MA Equation 2.32 (extended)
    """
    if i in model.slack_nodes:
        return pe.Constraint.Skip
    else:
        lhs = model.Q[i, t]
        rhs = (-sum(model.Qgen[k, t] for k in model.mapping_gen[i]) 
               - sum(model.Qsgen[k, t] for k in model.mapping_sgen[i])
               + sum(model.Qload[k, t] for k in model.mapping_load[i]))
        
        if "storage" in model.Enet_keys:
            rhs += sum(model.Qsto[k, t] for k in model.mapping_storage[i])
        if "GtP" in model.Enet_keys:
            rhs += -sum(model.Qgtp[k, t] for k in model.mapping_GtP[i])
        if "PtG" in model.Enet_keys:
            rhs += sum(model.Qptg[k, t] for k in model.mapping_PtG[i])
        
        return rhs == lhs


# ============================================================================
# CONSTRAINT LIBRARY REGISTRY
# ============================================================================

CONSTRAINT_LIBRARY: Dict[str, Callable] = {
    # Voltage constraints
    "vm_pu_lb_constr": vm_pu_lb_constr,
    "vm_pu_ub_constr": vm_pu_ub_constr,
    
    # Power balance constraints
    "nodal_power_balance_P": nodal_power_balance_P,
    "nodal_power_balance_Q": nodal_power_balance_Q,
    "nodal_power_injection_P": nodal_power_injection_P,
    "nodal_power_injection_Q": nodal_power_injection_Q,
    
    # Edge flow constraints (power)
    "edge_flow_p_from": edge_flow_p_from,
    "edge_flow_p_to": edge_flow_p_to,
    "edge_flow_q_from": edge_flow_q_from,
    "edge_flow_q_to": edge_flow_q_to,
    
    # Edge flow constraints (current)
    "edge_flow_i_from_real": edge_flow_i_from_real,
    "edge_flow_i_from_imag": edge_flow_i_from_imag,
    "edge_flow_i_to_real": edge_flow_i_to_real,
    "edge_flow_i_to_imag": edge_flow_i_to_imag,
    
    # Line limit constraints
    "line_limit_s_from": line_limit_s_from,
    "line_limit_s_to": line_limit_s_to,
    "line_limit_i_from": line_limit_i_from,
    "line_limit_i_to": line_limit_i_to,
    
    # Storage constraints
    "storage_equation": storage_equation,
    "storage_ramp_up": storage_ramp_up,
    "storage_ramp_down": storage_ramp_down,
    
    # Extended power injection (with non-basic components)
    "nodal_power_injection_P_extended": nodal_power_injection_P_extended,
    "nodal_power_injection_Q_extended": nodal_power_injection_Q_extended,
}
