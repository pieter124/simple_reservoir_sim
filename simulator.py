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
    """
    Solves steady-state single-phase Darcy flow in 1D or 2D using pure NumPy.
    """
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

            # --- X-Direction Connections (Face Harmonic Mean) ---
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

            # --- Y-Direction Connections (Face Harmonic Mean) ---
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
    """Scenario 1: Single well with 2D contour map and 1D cross-sectional profile."""
    nx, ny = 31, 31
    lx, ly = 1000.0, 1000.0
    dx, dy = np.full(nx, lx / nx), np.full(ny, ly / ny)
    mu = 1e-3
    k_val = 100.0 * MD_TO_M2
    p_ref = 20e6

    q_prod = -6.0e-4
    j_mid = ny // 2
    prod_cell = (nx // 2, j_mid)
    wells = [(prod_cell[0], prod_cell[1], q_prod)]

    q_boundary_total = -q_prod
    assert np.isclose(q_boundary_total + q_prod, 0.0), "Mass balance violated!"

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.25, 1.0]})

    # 2D Map
    cp = ax1.contourf(X, Y, p / 1e6, levels=30, cmap='viridis')
    fig.colorbar(cp, ax=ax1, label='Pressure [MPa]')
    ax1.contour(X, Y, p / 1e6, levels=12, colors='white', alpha=0.35, linewidths=0.8)
    step = 2
    ax1.quiver(X[::step, ::step], Y[::step, ::step], u_x_dir[::step, ::step], u_y_dir[::step, ::step],
               color='white', alpha=0.85, width=0.0035, headwidth=4, headlength=5)
    ax1.scatter([x[prod_cell[0]]], [y[prod_cell[1]]], color='red', marker='v', s=160, edgecolors='black', zorder=10,
                label=f'Producer: q = {q_prod:+.2e} m³/s')
    ax1.axhline(y[j_mid], color='cyan', linestyle=':', linewidth=1.8, label=f'Slice line (Y = {y[j_mid]:.0f} m)')
    ax1.set_title("2D Pressure & Darcy Velocity Vectors", fontweight='bold')
    ax1.set_xlabel("X [m]", fontweight='bold')
    ax1.set_ylabel("Y [m]", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)

    # 1D Cross-Sectional Profile through the Well
    ax2.plot(x, p[j_mid, :] / 1e6, 'o-', color='#ff7f0e', linewidth=2.0, markersize=5,
             label=f'Pressure Slice along Y={y[j_mid]:.0f} m')
    ax2.scatter([x[prod_cell[0]]], [p[j_mid, prod_cell[0]] / 1e6], color='red', marker='v', s=140, edgecolors='black', zorder=10,
                label=f'Producer Sink Node (X={x[prod_cell[0]]:.0f} m)')
    ax2.set_title("1D Cross-Sectional Pressure Profile", fontweight='bold')
    ax2.set_xlabel("X [m]", fontweight='bold')
    ax2.set_ylabel("Pressure [MPa]", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=8.5)

    plt.tight_layout()
    plt.show()


def run_scenario_2d_multi_well():
    """Scenario 2: Multi-well pattern with 2D contour map and 1D cross-sectional profile."""
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.25, 1.0]})

    # 2D Map
    cp = ax1.contourf(X, Y, p / 1e6, levels=35, cmap='plasma')
    fig.colorbar(cp, ax=ax1, label='Pressure [MPa]')
    ax1.contour(X, Y, p / 1e6, levels=14, colors='white', alpha=0.35, linewidths=0.8)
    step = 2
    ax1.quiver(X[::step, ::step], Y[::step, ::step], u_x_dir[::step, ::step], u_y_dir[::step, ::step],
               color='white', alpha=0.85, width=0.0035, headwidth=4, headlength=5)

    j_slice = 17
    for idx_w, (i_w, j_w, rate) in enumerate(wells, start=1):
        if rate > 0:
            ax1.scatter([x[i_w]], [y[j_w]], color='cyan', marker='^', s=140, edgecolors='black', linewidth=1.5, zorder=10)
        else:
            ax1.scatter([x[i_w]], [y[j_w]], color='lime', marker='v', s=160, edgecolors='black', linewidth=1.5, zorder=10)

    ax1.axhline(y[j_slice], color='white', linestyle='--', linewidth=1.8, label=f'Slice line through Producers (Y={y[j_slice]:.0f} m)')
    ax1.set_title("2D Pressure & Darcy Velocities", fontweight='bold')
    ax1.set_xlabel("X [m]", fontweight='bold')
    ax1.set_ylabel("Y [m]", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)

    # 1D Cross-Sectional Profile through both producers
    ax2.plot(x, p[j_slice, :] / 1e6, 's-', color='#2ca02c', linewidth=2.0, markersize=5,
             label=f'Pressure Slice along Y={y[j_slice]:.0f} m')
    ax2.scatter([x[12], x[22]], [p[j_slice, 12] / 1e6, p[j_slice, 22] / 1e6], color='lime', marker='v', s=140,
                edgecolors='black', zorder=10, label='Producers on Slice')
    ax2.set_title("1D Pressure Profile Across Producers", fontweight='bold')
    ax2.set_xlabel("X [m]", fontweight='bold')
    ax2.set_ylabel("Pressure [MPa]", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower center', fontsize=8.5)

    plt.tight_layout()
    plt.show()


def run_scenario_1d_variable_res():
    """
    Scenario 3: 1D grid convergence comparing Midpoint Sampling vs Exact Harmonic Averaging
    across multiple resolutions on a non-uniform gradient-adaptive grid.
    """
    l_domain = 600.0
    mu = 1e-3
    q_well = 2.5e-4
    p_ref = 10e6
    resolutions = [10, 24, 60]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    k_min = 15.0 * MD_TO_M2
    k_max = 500.0 * MD_TO_M2
    mid = l_domain / 2.0
    w = 55.0

    def k_func(x):
        u = (x - mid) / w
        return (k_min + k_max * (u**2)) / (1.0 + u**2)

    def dk_dx_func(x):
        u = (x - mid) / w
        return (2.0 * u / w * (k_max - k_min)) / (1.0 + u**2)**2

    def continuous_integral(x_arr):
        u = (x_arr - mid) / w
        u0 = -mid / w
        scale = w * (k_max - k_min) / (k_max * np.sqrt(k_min * k_max))
        c_ratio = np.sqrt(k_max / k_min)
        term_x = (w * u) / k_max + scale * np.arctan(c_ratio * u)
        term_0 = (w * u0) / k_max + scale * np.arctan(c_ratio * u0)
        return term_x - term_0

    def make_adaptive_grid(nx, alpha=4.0):
        x_dense = np.linspace(0.0, l_domain, 2000)
        grad_k = np.abs(dk_dx_func(x_dense))
        weight = 1.0 + alpha * (grad_k / np.max(grad_k))
        cum_w = np.concatenate(([0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * np.diff(x_dense))))
        cum_w /= cum_w[-1]

        xi = np.linspace(0.0, 1.0, nx + 1)
        x_faces = np.interp(xi, cum_w, x_dense)
        dx_arr = np.diff(x_faces)
        x_cents = 0.5 * (x_faces[:-1] + x_faces[1:])
        return x_faces, dx_arr, x_cents

    x_dense = np.linspace(0, l_domain, 1000)
    I_dense = continuous_integral(x_dense)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    # Solve across resolutions using Exact Cell Harmonic Averaging
    for nx, col in zip(resolutions, colors):
        x_faces, dx_arr, x_cents = make_adaptive_grid(nx)
        wells_1d = [(0, 0, q_well), (nx - 1, 0, -q_well)]

        # Exact Cell Harmonic Average: dx / int_{x_L}^{x_R} (1/k) dx
        delta_I = continuous_integral(x_faces[1:]) - continuous_integral(x_faces[:-1])
        k_harm = dx_arr / delta_I

        x_c, p_harm = solve_reservoir(
            nx=nx, ny=1, lx=l_domain, ly=1.0,
            dx=dx_arr, k_x=k_harm, mu=mu, wells=wells_1d,
            boundary_flux=None, p_ref=p_ref, ref_cell=(nx - 1, 0)
        )

        ax1.plot(x_c, p_harm / 1e6, marker='s', markersize=4, linestyle='-',
                 label=f'Harmonic Averaged ({nx} cells)', color=col)

    # Reference Midpoint Sampling run on coarsest grid (nx=10) to illustrate contrast
    x_faces_c, dx_arr_c, x_cents_c = make_adaptive_grid(resolutions[0])
    k_mid_coarse = k_func(x_cents_c)
    _, p_mid_coarse = solve_reservoir(
        nx=resolutions[0], ny=1, lx=l_domain, ly=1.0,
        dx=dx_arr_c, k_x=k_mid_coarse, mu=mu, wells=[(0, 0, q_well), (resolutions[0] - 1, 0, -q_well)],
        boundary_flux=None, p_ref=p_ref, ref_cell=(resolutions[0] - 1, 0)
    )
    ax1.plot(x_cents_c, p_mid_coarse / 1e6, 'o--', color='darkred', linewidth=1.6, markersize=5,
             label=f'Midpoint Sampled ({resolutions[0]} cells, misses resistance)')

    # Analytical Reference
    x_ref_analytical = x_cents[-1]
    I_ref = continuous_integral(np.array([x_ref_analytical]))[0]
    p_exact = p_ref + q_well * mu * (I_ref - I_dense)

    ax1.plot(x_dense, p_exact / 1e6, 'k-', linewidth=2.5, label='Analytical Exact Continuous', zorder=1)
    ax1.scatter([0.0], [p_exact[0] / 1e6], color='blue', marker='o', s=80, edgecolors='black', zorder=10,
                label=f'Injector: q = {q_well:+.2e} m³/s')
    ax1.scatter([x_ref_analytical], [p_ref / 1e6], color='red', marker='s', s=80, edgecolors='black', zorder=10,
                label=f'Producer: q = {-q_well:+.2e} m³/s')

    ax1.set_ylabel("Pressure [MPa]", fontweight='bold')
    ax1.set_title("Scenario 3: 1D Pressure Convergence Using Cell Harmonic Averaging", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5)

    # Subplot 2: Continuous k(x) vs Harmonic Averaged Blocks vs Midpoint Blocks (nx=10)
    k_profile_md = k_func(x_dense) / MD_TO_M2
    ax2.plot(x_dense, k_profile_md, 'k-', linewidth=2.5, label='Exact Continuous Permeability k(x)')

    # Compute harmonic average values on coarse grid for plotting
    delta_I_c = continuous_integral(x_faces_c[1:]) - continuous_integral(x_faces_c[:-1])
    k_harm_coarse = (dx_arr_c / delta_I_c) / MD_TO_M2

    x_steps = np.repeat(x_faces_c, 2)[1:-1]
    k_mid_steps = np.repeat(k_mid_coarse / MD_TO_M2, 2)
    k_harm_steps = np.repeat(k_harm_coarse, 2)

    ax2.plot(x_steps, k_mid_steps, color='darkred', linestyle='--', linewidth=1.8,
             label=f'Midpoint Sampled Blocks (nx={resolutions[0]})')
    ax2.plot(x_steps, k_harm_steps, color=colors[0], linestyle='-', linewidth=2.0, alpha=0.9,
             label=f'Harmonic Averaged Blocks (nx={resolutions[0]}, properly choked)')
    ax2.scatter(x_cents_c, k_harm_coarse, color=colors[0], s=35, marker='s', zorder=5, label='Harmonic Block Values')

    ax2_twin = ax2.twinx()
    grad_norm = np.abs(dk_dx_func(x_dense))
    grad_norm /= np.max(grad_norm)
    ax2_twin.fill_between(x_dense, grad_norm, color='purple', alpha=0.15, label='Permeability Gradient |dk/dx|')
    ax2_twin.set_ylabel("Normalized |dk/dx|", color='purple', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    ax2_twin.set_ylim(0, 1.3)

    ax2.set_xlabel("Distance [m]", fontweight='bold')
    ax2.set_ylabel("Permeability [mD]", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    plt.show()


def run_scenario_1d_fault():
    """
    Scenario 4: 1D reservoir with a thin low-permeability fault inside Cell 5.
    Compares Midpoint Sampling vs Cell Harmonic Averaging against the analytical solution.
    """
    l_domain = 600.0
    mu = 1e-3
    q_well = 2.5e-5
    p_ref = 10e6

    k_sand = 500.0 * MD_TO_M2
    k_fault = 1.0 * MD_TO_M2
    fault_x1 = 310.0
    fault_x2 = 311.0  # 1 m wide fault inside cell [300, 360]
    w_fault = fault_x2 - fault_x1

    def k_exact(x):
        return np.where((x >= fault_x1) & (x <= fault_x2), k_fault, k_sand)

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

    nx = 10
    dx_val = l_domain / nx
    dx_arr = np.full(nx, dx_val)
    x_faces = np.linspace(0.0, l_domain, nx + 1)
    x_cents = 0.5 * (x_faces[:-1] + x_faces[1:])
    wells_1d = [(0, 0, q_well), (nx - 1, 0, -q_well)]

    # Method 1: Midpoint Sampling
    k_mid = k_exact(x_cents)
    _, p_mid = solve_reservoir(
        nx=nx, ny=1, lx=l_domain, ly=1.0,
        dx=dx_arr, k_x=k_mid, mu=mu, wells=wells_1d,
        boundary_flux=None, p_ref=p_ref, ref_cell=(nx - 1, 0)
    )

    # Method 2: Cell Harmonic Average
    delta_I = exact_integral(x_faces[1:]) - exact_integral(x_faces[:-1])
    k_harm = dx_arr / delta_I
    _, p_harm = solve_reservoir(
        nx=nx, ny=1, lx=l_domain, ly=1.0,
        dx=dx_arr, k_x=k_harm, mu=mu, wells=wells_1d,
        boundary_flux=None, p_ref=p_ref, ref_cell=(nx - 1, 0)
    )

    # Analytical Solution
    x_dense = np.linspace(0.0, l_domain, 2000)
    I_dense = exact_integral(x_dense)
    x_ref = x_cents[-1]
    I_ref = exact_integral(np.array([x_ref]))[0]
    p_exact = p_ref + q_well * mu * (I_ref - I_dense)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    ax1.plot(x_dense, p_exact / 1e6, 'k-', linewidth=2.5, label='Analytical Exact', zorder=1)
    ax1.plot(x_cents, p_mid / 1e6, 'o--', color='#ff7f0e', linewidth=2.0, markersize=7,
             label='Midpoint Sampling: Misses fault completely (Zero fault ΔP!)')
    ax1.plot(x_cents, p_harm / 1e6, 's--', color='#0099cc', linewidth=2.0, markersize=7,
             label='Harmonic Averaging: Captures the fault pressure drop')

    ax1.scatter([0.0], [p_exact[0] / 1e6], color='blue', marker='^', s=90, edgecolors='black', zorder=10,
                label=f'Injector (Source, cell 0): q = {q_well:+.2e} m³/s')
    ax1.scatter([x_ref], [p_ref / 1e6], color='red', marker='v', s=90, edgecolors='black', zorder=10,
                label=f'Producer (Sink, cell nx-1): q = {-q_well:+.2e} m³/s')

    ax1.axvspan(fault_x1, fault_x2, color='crimson', alpha=0.35, label='Fault Zone (1 m wide, 1 mD)')
    ax1.set_ylabel("Pressure [MPa]", fontweight='bold')
    ax1.set_title("Scenario 4: Single Fault in Cell 5 (Midpoint Sampling vs Harmonic Averaging)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5)

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
        "1d_res": run_scenario_1d_variable_res,
        "1d_fault": run_scenario_1d_fault
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
            "  2d_single : 2D domain with 1 well + 1D pressure cross-section\n"
            "  2d_multi  : 2D multi-well pattern + 1D pressure cross-section\n"
            "  1d_res    : 1D grid convergence using exact cell harmonic averaging\n"
            "  1d_fault  : 1D single-cell fault test (Midpoint vs Harmonic)\n"
        )
    )

    args = parser.parse_args()
    scenarios[args.scenario]()


if __name__ == "__main__":
    main()