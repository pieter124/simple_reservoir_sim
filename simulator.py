#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

DARCY_TO_M2 = 9.869233e-13
MD_TO_M2 = 1e-3 * DARCY_TO_M2


# ==============================================================================
# UNIFIED RESERVOIR SOLVER (1D & 2D, Pure NumPy, Neumann BCs Only)
# Assumptions: Constant density, steady state, constant viscosity, unit thickness
# ==============================================================================
def solve_reservoir(
    nx, ny=1, 
    lx=1000.0, ly=1.0, 
    dx=None, dy=None,
    k_x=None, k_y=None, 
    mu=1e-3, 
    wells=None, 
    boundary_flux=None, 
    p_ref=10e6, 
    ref_cell=(0, 0)
):
    if dx is None:
        dx = np.full(nx, lx / nx)
    if dy is None:
        dy = np.full(ny, ly / ny)

    if k_x is None:
        k_x = np.full((ny, nx), 100.0 * MD_TO_M2)
    elif np.ndim(k_x) == 1:
        k_x = np.tile(k_x, (ny, 1))
    elif np.isscalar(k_x):
        k_x = np.full((ny, nx), k_x)

    if k_y is None:
        k_y = k_x.copy()
    elif np.isscalar(k_y):
        k_y = np.full((ny, nx), k_y)

    boundary_flux = boundary_flux or {}
    q_west = boundary_flux.get('west', 0.0)
    q_east = boundary_flux.get('east', 0.0)
    q_south = boundary_flux.get('south', 0.0)
    q_north = boundary_flux.get('north', 0.0)

    n_cells = nx * ny
    def idx(i, j): return j * nx + i

    A = np.zeros((n_cells, n_cells))
    b = np.zeros(n_cells)

    for j in range(ny):
        for i in range(nx):
            c = idx(i, j)

            # --- X-Direction Connections ---
            if i > 0:
                dx_l = 0.5 * dx[i - 1]
                dx_r = 0.5 * dx[i]
                perm_harm = (dx_l + dx_r) / (dx_l / k_x[j, i - 1] + dx_r / k_x[j, i])
                T_west = (perm_harm / mu) * dy[j] / (dx_l + dx_r)
                A[c, c] += T_west
                A[c, idx(i - 1, j)] -= T_west
            else:
                b[c] += q_west * dy[j]

            if i < nx - 1:
                dx_l = 0.5 * dx[i]
                dx_r = 0.5 * dx[i + 1]
                perm_harm = (dx_l + dx_r) / (dx_l / k_x[j, i] + dx_r / k_x[j, i + 1])
                T_east = (perm_harm / mu) * dy[j] / (dx_l + dx_r)
                A[c, c] += T_east
                A[c, idx(i + 1, j)] -= T_east
            else:
                b[c] += q_east * dy[j]

            # --- Y-Direction Connections ---
            if ny > 1:
                if j > 0:
                    dy_b = 0.5 * dy[j - 1]
                    dy_t = 0.5 * dy[j]
                    perm_harm = (dy_b + dy_t) / (dy_b / k_y[j - 1, i] + dy_t / k_y[j, i])
                    T_south = (perm_harm / mu) * dx[i] / (dy_b + dy_t)
                    A[c, c] += T_south
                    A[c, idx(i, j - 1)] -= T_south
                else:
                    b[c] += q_south * dx[i]

                if j < ny - 1:
                    dy_b = 0.5 * dy[j]
                    dy_t = 0.5 * dy[j + 1]
                    perm_harm = (dy_b + dy_t) / (dy_b / k_y[j, i] + dy_t / k_y[j + 1, i])
                    T_north = (perm_harm / mu) * dx[i] / (dy_b + dy_t)
                    A[c, c] += T_north
                    A[c, idx(i, j + 1)] -= T_north
                else:
                    b[c] += q_north * dx[i]

    if wells:
        for i_w, j_w, q_w in wells:
            b[idx(i_w, j_w)] += q_w

    ref_idx = idx(ref_cell[0], ref_cell[1])
    A[ref_idx, :] = 0.0
    A[ref_idx, ref_idx] = 1.0
    b[ref_idx] = p_ref

    p = np.linalg.solve(A, b)
    
    x_faces = np.concatenate(([0.0], np.cumsum(dx)))
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_faces = np.concatenate(([0.0], np.cumsum(dy)))
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    if ny == 1:
        return x_centers, p
    return x_centers, y_centers, p.reshape((ny, nx))


# ==============================================================================
# FACE-BASED VECTOR VELOCITY CALCULATION
# ==============================================================================
def compute_darcy_velocity_2d(p, dx, dy, k_x, k_y, mu=1e-3, boundary_flux=None):
    ny, nx = p.shape
    u_x = np.zeros_like(p)
    u_y = np.zeros_like(p)

    boundary_flux = boundary_flux or {}
    q_w = boundary_flux.get('west', 0.0)
    q_e = boundary_flux.get('east', 0.0)
    q_s = boundary_flux.get('south', 0.0)
    q_n = boundary_flux.get('north', 0.0)

    for j in range(ny):
        for i in range(nx):
            if i > 0:
                dx_l = 0.5 * dx[i - 1]
                dx_r = 0.5 * dx[i]
                k_harm = (dx_l + dx_r) / (dx_l / k_x[j, i - 1] + dx_r / k_x[j, i])
                f_west = -(k_harm / mu) * (p[j, i] - p[j, i - 1]) / (dx_l + dx_r)
            else:
                f_west = q_w

            if i < nx - 1:
                dx_l = 0.5 * dx[i]
                dx_r = 0.5 * dx[i + 1]
                k_harm = (dx_l + dx_r) / (dx_l / k_x[j, i] + dx_r / k_x[j, i + 1])
                f_east = -(k_harm / mu) * (p[j, i + 1] - p[j, i]) / (dx_l + dx_r)
            else:
                f_east = -q_e

            u_x[j, i] = 0.5 * (f_west + f_east)

    for j in range(ny):
        for i in range(nx):
            if j > 0:
                dy_b = 0.5 * dy[j - 1]
                dy_t = 0.5 * dy[j]
                k_harm = (dy_b + dy_t) / (dy_b / k_y[j - 1, i] + dy_t / k_y[j, i])
                f_south = -(k_harm / mu) * (p[j, i] - p[j - 1, i]) / (dy_b + dy_t)
            else:
                f_south = q_s

            if j < ny - 1:
                dy_b = 0.5 * dy[j]
                dy_t = 0.5 * dy[j + 1]
                k_harm = (dy_b + dy_t) / (dy_b / k_y[j, i] + dy_t / k_y[j + 1, i])
                f_north = -(k_harm / mu) * (p[j + 1, i] - p[j, i]) / (dy_b + dy_t)
            else:
                f_north = -q_n

            u_y[j, i] = 0.5 * (f_south + f_north)

    return u_x, u_y


# ==============================================================================
# SCENARIOS
# ==============================================================================
def run_scenario_2d_single_well():
    nx, ny = 31, 31
    lx, ly = 1000.0, 1000.0
    dx, dy = np.full(nx, lx / nx), np.full(ny, ly / ny)
    mu = 1e-3
    k_val = 100.0 * MD_TO_M2
    p_ref = 20e6

    q_prod = -6.0e-4
    prod_cell = (nx // 2, ny // 2)
    wells = [(prod_cell[0], prod_cell[1], q_prod)]

    q_boundary_total = -q_prod
    perimeter = 2.0 * (lx + ly)
    flux_density = q_boundary_total / perimeter
    boundary_flux = {'west': flux_density, 'east': flux_density, 'south': flux_density, 'north': flux_density}

    x, y, p = solve_reservoir(
        nx=nx, ny=ny, lx=lx, ly=ly, dx=dx, dy=dy,
        k_x=k_val, wells=wells, boundary_flux=boundary_flux, p_ref=p_ref, ref_cell=(0, 0)
    )

    k_x_mat = np.full((ny, nx), k_val)
    k_y_mat = np.full((ny, nx), k_val)
    u_x, u_y = compute_darcy_velocity_2d(p, dx, dy, k_x_mat, k_y_mat, mu=mu, boundary_flux=boundary_flux)

    speed = np.hypot(u_x, u_y)
    u_x_dir = np.divide(u_x, speed, out=np.zeros_like(u_x), where=speed > 0)
    u_y_dir = np.divide(u_y, speed, out=np.zeros_like(u_y), where=speed > 0)

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 8))

    cp = ax.contourf(X, Y, p / 1e6, levels=30, cmap='viridis')
    fig.colorbar(cp, ax=ax, label='Pressure [MPa]')
    lines = ax.contour(X, Y, p / 1e6, levels=12, colors='white', alpha=0.35, linewidths=0.8)
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.2f')

    step = 2
    ax.quiver(
        X[::step, ::step], Y[::step, ::step], u_x_dir[::step, ::step], u_y_dir[::step, ::step], 
        color='white', alpha=0.85, width=0.0035, headwidth=4, headlength=5, label='Darcy Velocity Direction'
    )

    ax.scatter([x[prod_cell[0]]], [y[prod_cell[1]]], color='red', marker='v', s=160, edgecolors='black', linewidth=1.5, zorder=10,
               label=f'Producer (Sink): q = {q_prod:+.2e} m³/s')
    ax.plot([], [], color='white', linestyle='--', linewidth=2,
            label=f'Boundary Influx (Source): Q = {q_boundary_total:+.2e} m³/s')

    ax.set_title(f"Scenario 1: 2D Single Well with Neumann Influx\n(Total Boundary Influx = {q_boundary_total:+.2e} m³/s, sum Q = 0)", fontsize=11, fontweight='bold')
    ax.set_xlabel("X [m]", fontweight='bold')
    ax.set_ylabel("Y [m]", fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


def run_scenario_2d_multi_well():
    nx, ny = 35, 35
    lx, ly = 1000.0, 1000.0
    dx, dy = np.full(nx, lx / nx), np.full(ny, ly / ny)
    mu = 1e-3
    k_val = 150.0 * MD_TO_M2
    p_ref = 15e6

    wells = [
        (7, 7, 3.0e-4),       # Injector 1
        (7, 27, 3.0e-4),      # Injector 2
        (27, 7, 3.0e-4),      # Injector 3
        (27, 27, 3.0e-4),     # Injector 4
        (12, 17, -6.0e-4),    # Producer 1
        (22, 17, -6.0e-4),    # Producer 2
    ]

    assert np.isclose(sum(w[2] for w in wells), 0.0), "Wells must sum to 0 for no-flow boundaries."

    x, y, p = solve_reservoir(
        nx=nx, ny=ny, lx=lx, ly=ly, dx=dx, dy=dy,
        k_x=k_val, wells=wells, boundary_flux=None, p_ref=p_ref, ref_cell=(12, 17)
    )

    k_x_mat = np.full((ny, nx), k_val)
    k_y_mat = np.full((ny, nx), k_val)
    u_x, u_y = compute_darcy_velocity_2d(p, dx, dy, k_x_mat, k_y_mat, mu=mu, boundary_flux=None)

    speed = np.hypot(u_x, u_y)
    u_x_dir = np.divide(u_x, speed, out=np.zeros_like(u_x), where=speed > 0)
    u_y_dir = np.divide(u_y, speed, out=np.zeros_like(u_y), where=speed > 0)

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 8))

    cp = ax.contourf(X, Y, p / 1e6, levels=35, cmap='plasma')
    fig.colorbar(cp, ax=ax, label='Pressure [MPa]')
    lines = ax.contour(X, Y, p / 1e6, levels=14, colors='white', alpha=0.35, linewidths=0.8)
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.2f')

    step = 2
    ax.quiver(
        X[::step, ::step], Y[::step, ::step], u_x_dir[::step, ::step], u_y_dir[::step, ::step], 
        color='white', alpha=0.85, width=0.0035, headwidth=4, headlength=5, label='Darcy Velocity Direction'
    )

    for idx_w, (i_w, j_w, rate) in enumerate(wells, start=1):
        if rate > 0:
            ax.scatter([x[i_w]], [y[j_w]], color='cyan', marker='^', s=140, edgecolors='black', linewidth=1.5, zorder=10,
                       label=f'Inj #{idx_w} ({i_w},{j_w}): q = {rate:+.2e} m³/s')
        else:
            ax.scatter([x[i_w]], [y[j_w]], color='lime', marker='v', s=160, edgecolors='black', linewidth=1.5, zorder=10,
                       label=f'Prod #{idx_w} ({i_w},{j_w}): q = {rate:+.2e} m³/s')

    ax.set_title(r"Scenario 2: 2D Multiple Injectors & Producers (Impermeable Boundary $\partial p/\partial n=0$, $\sum Q=0$)", fontsize=11, fontweight='bold')
    ax.set_xlabel("X [m]", fontweight='bold')
    ax.set_ylabel("Y [m]", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


def run_scenario_1d_variable_res():
    """
    Scenario 3: 1D reservoir with a thin low-permeability fault inside Cell 5.
    Compares Midpoint Sampling vs Cell Harmonic Averaging against the analytical solution.
    """
    l_domain = 600.0
    mu = 1e-3
    q_well = 2.5e-5
    p_ref = 10e6

    # Geology: 500 mD Sandstone with a 1 m fault gouge of 1 mD
    k_sand = 500.0 * MD_TO_M2
    k_fault = 1.0 * MD_TO_M2
    fault_x1 = 310.0
    fault_x2 = 311.0  # 1 m wide fault inside cell [300, 360]
    w_fault = fault_x2 - fault_x1

    def k_exact(x):
        return np.where((x >= fault_x1) & (x <= fault_x2), k_fault, k_sand)

    # Exact continuous integral: int_0^x (1 / k(s)) ds
    def exact_integral(x_arr):
        I = np.zeros_like(x_arr, dtype=float)
        for idx_pt, x in enumerate(x_arr):
            if x <= fault_x1:
                I[idx_pt] = x / k_sand
            elif x <= fault_x2:
                I[idx_pt] = fault_x1 / k_sand + (x - fault_x1) / k_fault
            else:
                I[idx_pt] = fault_x1 / k_sand + w_fault / k_fault + (x - fault_x2) / k_sand
        return I

    # Uniform grid: nx = 10 (each cell is 60 m wide)
    nx = 10
    dx_val = l_domain / nx
    dx_arr = np.full(nx, dx_val)
    x_faces = np.linspace(0.0, l_domain, nx + 1)
    x_cents = 0.5 * (x_faces[:-1] + x_faces[1:])
    wells_1d = [(0, 0, q_well), (nx - 1, 0, -q_well)]

    # --- Method 1: Midpoint Sampling (k_i = k(x_c)) ---
    # Cell 5 center is at 330 m, completely missing the fault at 310-311 m
    k_mid = k_exact(x_cents)
    _, p_mid = solve_reservoir(
        nx=nx, ny=1, lx=l_domain, ly=1.0,
        dx=dx_arr, k_x=k_mid, mu=mu, wells=wells_1d,
        boundary_flux=None, p_ref=p_ref, ref_cell=(nx - 1, 0)
    )

    # --- Method 2: Cell Harmonic Average (dx / int (1/k) dx) ---
    delta_I = exact_integral(x_faces[1:]) - exact_integral(x_faces[:-1])
    k_harm = dx_arr / delta_I  # Cell 5 drops from 500 mD to ~53.7 mD
    _, p_harm = solve_reservoir(
        nx=nx, ny=1, lx=l_domain, ly=1.0,
        dx=dx_arr, k_x=k_harm, mu=mu, wells=wells_1d,
        boundary_flux=None, p_ref=p_ref, ref_cell=(nx - 1, 0)
    )

    # --- Analytical Solution ---
    x_dense = np.linspace(0.0, l_domain, 2000)
    I_dense = exact_integral(x_dense)
    x_ref = x_cents[-1]
    I_ref = exact_integral(np.array([x_ref]))[0]
    p_exact = p_ref + q_well * mu * (I_ref - I_dense)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    # Subplot 1: Pressure
    ax1.plot(x_dense, p_exact / 1e6, 'k-', linewidth=2.5, label='Analytical Exact', zorder=1)
    ax1.plot(x_cents, p_mid / 1e6, 'o--', color='#ff7f0e', linewidth=2.0, markersize=7,
             label='Midpoint Sampling: Misses fault completely (Zero fault ΔP!)')
    ax1.plot(x_cents, p_harm / 1e6, 's--', color='#0099cc', linewidth=2.0, markersize=7,
             label='Harmonic Averaging: Captures the 25 MPa fault pressure drop')

    ax1.scatter([0.0], [p_exact[0] / 1e6], color='blue', marker='^', s=90, edgecolors='black', zorder=10,
                label=f'Injector (Source, cell 0): q = {q_well:+.2e} m³/s')
    ax1.scatter([x_ref], [p_ref / 1e6], color='red', marker='v', s=90, edgecolors='black', zorder=10,
                label=f'Producer (Sink, cell nx-1): q = {-q_well:+.2e} m³/s')

    ax1.axvspan(fault_x1, fault_x2, color='crimson', alpha=0.35, label='Fault Zone (1 m wide, 1 mD)')
    ax1.set_ylabel("Pressure [MPa]", fontweight='bold')
    ax1.set_title("Single Fault in Cell 5: Midpoint Sampling vs Harmonic Averaging", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5)

    # Subplot 2: Permeability
    x_steps = np.repeat(x_faces, 2)[1:-1]
    k_mid_steps = np.repeat(k_mid / MD_TO_M2, 2)
    k_harm_steps = np.repeat(k_harm / MD_TO_M2, 2)

    ax2.plot(x_steps, k_mid_steps, color='#ff7f0e', linestyle='--', linewidth=2.0,
             label='Midpoint Sampled: Flat 500 mD everywhere (Fault ignored)')
    ax2.plot(x_steps, k_harm_steps, color='#0099cc', linestyle='-', linewidth=2.2,
             label=f'Harmonic Averaged: Cell 5 effectively choked to {k_harm[5]/MD_TO_M2:.1f} mD')

    ax2.scatter(x_cents, k_mid / MD_TO_M2, color='#ff7f0e', s=45, zorder=5)
    ax2.scatter(x_cents, k_harm / MD_TO_M2, color='#0099cc', s=45, marker='s', zorder=5)

    ax2.axvspan(fault_x1, fault_x2, color='crimson', alpha=0.35, label='Actual 1 mD Fault')
    ax2.set_xlabel("Distance [m]", fontweight='bold')
    ax2.set_ylabel("Permeability [mD]", fontweight='bold')
    ax2.set_ylim(0, 550)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=8.5)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    scenarios = {
        "2d_single": run_scenario_2d_single_well,
        "2d_multi": run_scenario_2d_multi_well,
        "1d_res": run_scenario_1d_variable_res
    }

    parser = argparse.ArgumentParser(
        description="Unified Reservoir Simulator (Pure NumPy, Neumann BCs Only)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "scenario",
        choices=list(scenarios.keys()),
        help=(
            "Scenario to run:\n"
            "  2d_single : 2D domain with 1 single well and Neumann boundary influx\n"
            "  2d_multi  : 2D multiple injectors and producers (impermeable BC, sum Q = 0)\n"
            "  1d_res    : 1D single-cell fault test (Midpoint vs Harmonic)\n"
        )
    )

    args = parser.parse_args()
    scenarios[args.scenario]()


if __name__ == "__main__":
    main()