# 1D/2D Steady-State Reservoir Simulator

A lightweight, pure NumPy cell-centered finite-volume (Two-Point Flux Approximation — TPFA) simulator for steady-state single-phase Darcy flow under pure Neumann boundary conditions.

# Core Physical & Numerical Assumptions
Steady-State Conditions: 
The flow regime has reached equilibrium over time. Storage effects, fluid accumulation, and time-dependent transient pressure waves are ignored.

Single-Phase, Incompressible Flow:
Only one fluid (such as water or oil) occupies the pore volume. The fluid has constant density and constant dynamic viscosity. There are no phase changes, saturation changes, relative permeability curves, or capillary pressure effects.

Rigid Rock Matrix:
The rock does not deform, compact, or expand as pressure changes (rock compressibility is zero). Porosity and absolute permeability fields are static.

Axis-Aligned Permeability:
Permeability can vary spatially and differ between directions ($x$ vs. $y$), but principal flow axes must align with the Cartesian grid. Cross-directional permeability terms are zero, meaning flow will not deflect off-axis due to rock fabric alone.

2D Planar / 1D Flow (Unit Thickness):
The reservoir is treated as a flat slab with a uniform thickness of 1 meter. Vertical flow variations and vertical cross-flow are omitted.

Negligible Gravity Effects:
Flow is driven strictly by pressure gradients created by wells and boundary fluxes. Buoyancy, fluid head, and dip angle elevation changes are not considered.

Strict Global Mass Balance (Pure Neumann Boundaries):
Because all reservoir boundaries specify a flow rate (or zero flow) rather than a fixed boundary pressure, the total volume of fluid injected by wells and boundaries must exactly equal the total volume produced. If net flow does not balance to zero, a steady-state solution is physically impossible.

Single Reference Pressure Datum:
Because only flow rates are specified across all boundaries, the absolute pressure level is unconstrained (infinite valid pressure fields with the same gradient). To establish a physical baseline, the solver artificially pins exactly one cell to a fixed reference pressure.

Two-Point Flux Approximation (TPFA):
The flow rate across any grid cell face depends strictly on the pressure difference between the two adjacent cell centers. This assumption is accurate for orthogonal grids aligned with the permeability axes.

## Requirements & Installation

* Python 3.8+
* `numpy`
* `matplotlib`

Install dependencies via `pip`:

```bash
pip install numpy matplotlib
```

The script contains four pre-configured scenarios that can be executed directly from your terminal:
```bash
# Central producer with fluid recharging from all four boundaries
python simulator.py 2d_single

# 4-corner injectors pushing fluid to 2 central producers inside a sealed box
python simulator.py 2d_multi

# 1D convergence test showing why harmonic averaging beats midpoint sampling
python simulator.py 1d_res

# 1D demonstration showing how sub-grid thin faults are captured accurately
python simulator.py 1d_fault
```