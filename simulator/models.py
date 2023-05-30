"""
Module to solve viscous flow. This module contains the function viscous_flow
that takes a discretization class and a data class as arguments. The
discretization class is found at discretization.ViscouFlow and the data is found
at cases.BaseData or inherited classes.
"""

import numpy as np
import viscous_porepy as pp
import viscous_porepy.ad as ad
import time

from simulator import adaptivity
from utils import projection
from utils import viz
from utils import utils


def initiate_simulation(problem, disc):
    """
    Obtain discretization operators, projection operators, initial values,
    boundary conditions, etc.
    """
    # Get information from problem
    gb = problem.gb
    flow_kw = problem.flow_keyword
    tran_kw = problem.transport_keyword

    # define shorthand notation for discretizations
    # Flow
    flux = disc.mat[flow_kw]["flux"]
    bound_flux = disc.mat[flow_kw]["bound_flux"]
    trace_p_cell = disc.mat[flow_kw]["trace_cell"]
    trace_p_face = disc.mat[flow_kw]["trace_face"]
    bc_val_p = disc.mat[flow_kw]["bc_values"]
    i_x = disc.mat[flow_kw]["i_x"]  # moving frame velocity
    kn = disc.mat[flow_kw]["kn"]
    mu = problem.viscosity

    # Transport
    diff = disc.mat[tran_kw]["flux"]
    bound_diff = disc.mat[tran_kw]["bound_flux"]
    trace_c_cell = disc.mat[tran_kw]["trace_cell"]
    trace_c_face = disc.mat[tran_kw]["trace_face"]
    bc_val_c = disc.mat[tran_kw]["bc_values"]
    Dn = disc.mat[tran_kw]["dn"]

    # Define projections between grids
    (
        primary2mortar,
        secondary2mortar,
        mortar2primary,
        mortar2secondary,
    ) = projection.mixed_dim_projections(gb)
    # And between cells and faces
    avg = projection.cells2faces_avg(gb)
    div = projection.faces2cells(gb)

    # Assemble geometric values
    mass_weight = disc.mat[tran_kw]["mass_weight"]
    cell_volumes = gb.cell_volumes() * mass_weight
    mortar_volumes = gb.cell_volumes_mortar()
    # mortar_area = mortar_volumes * (primary2mortar * avg * specific_volume)
    return (
        gb,
        flux,
        bound_flux,
        trace_p_cell,
        trace_p_face,
        bc_val_p,
        i_x,
        kn,
        mu,
        diff,
        bound_diff,
        trace_c_cell,
        trace_c_face,
        bc_val_c,
        Dn,
        primary2mortar,
        secondary2mortar,
        mortar2primary,
        mortar2secondary,
        avg,
        div,
        mass_weight,
        cell_volumes,
        mortar_volumes,
    )


def initiate_variables(disc, c0):
    """
    Find the initial solution
    """
    gb = disc.problem.gb
    lam_c0 = np.zeros(gb.num_mortar_cells())
    # Initial guess for the pressure and mortar flux
    p0_init = np.zeros(gb.num_cells())
    lam0_init = np.zeros(gb.num_mortar_cells())
    # Define Ad variables
    p, lam = ad.initAdArrays([p0_init, lam0_init])
    # define dofs indices
    p_ix = slice(gb.num_cells())
    lam_ix = slice(gb.num_cells(), gb.num_cells() + gb.num_mortar_cells())
    c_ix = slice(
        gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + gb.num_mortar_cells(),
    )
    lam_c_ix = slice(
        2 * gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + 2 * gb.num_mortar_cells(),
    )
    # Solve with Newton (should converge in 1 or 2 iterations. Is non-linear due to
    # upstream weights)
    sol = np.hstack((p.val, lam.val))
    q = np.zeros(gb.num_faces())

    err = np.inf
    newton_it = 0
    newton_it = 0
    while err > 1e-9:
        newton_it += 1
        q = darcy(disc, p, c0, lam)
        eq_init = ad.concatenate(
            (
                mass_conservation(disc, lam, q),
                coupling_law_p(disc, p, lam, c0),
            )
        )
        err = np.max(np.abs(eq_init.val))
        sol = sol - disc.linear_solver(eq_init.jac, eq_init.val)
        p.val = sol[p_ix]
        lam.val = sol[lam_ix]

        if newton_it > 20:
            raise RuntimeError('Failed to converge Newton iteration in variable initiation')

    # Now that we have solved for initial condition, initalize full problem
    p, lam, c, lam_c = ad.initAdArrays([p.val, lam.val, c0, lam_c0])
    sol = np.hstack((p.val, lam.val, c.val, lam_c.val))

    q = darcy(disc, p, c, lam)
    return p, lam, q, c, lam_c, sol, p_ix, lam_ix, c_ix, lam_c_ix


def darcy(disc, p, c, lam):
    """
    Discretization of Darcy's law
    """
    kw = disc.problem.flow_keyword
    flux = disc.mat[kw]["flux"]
    bound_flux = disc.mat[kw]["bound_flux"]
    bc_val_p = disc.mat[kw]["bc_values"]
    i_x = disc.mat[kw]["i_x"]  # moving frame velocity
    mu = disc.viscosity()

    mortar2primary = disc.proj["mortar2primary"]
    v = flux * p
    return (
        (flux * p) * (mu(disc.upwind(c, v)) ** -1)
        + bound_flux * bc_val_p
        + bound_flux * mortar2primary * lam
        - i_x
    )


def trace(disc, kw, p, lam):
    "Discretization of the trace operator"
    trace_p_cell = disc.mat[kw]["trace_cell"]
    trace_p_face = disc.mat[kw]["trace_face"]
    bc_val_p = disc.mat[kw]["bc_values"]
    return trace_p_cell * p + trace_p_face * (lam + bc_val_p)


def mass_conservation(disc, lam, q):
    "Discretization of the mass conservation equation"
    div = disc.proj["div"]
    mortar2secondary = disc.proj["mortar2secondary"]
    return div * q - mortar2secondary * lam


def coupling_law_p(disc, p, lam, c):
    "Discretization of the coupling law for fluid flux"
    kw = disc.problem.flow_keyword
   
    kn = disc.mat[kw]["kn"]
    mu = disc.viscosity()
    mortar_volumes = disc.geo["mortar_volumes"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"]
    avg = disc.proj["avg"]
    if isinstance(lam, ad.Ad_array):
        lam_flux = lam.val
    else:
        lam_flux = lam
    primary_flag = (lam_flux > 0).astype(np.int)
    secondary_flag = 1 - primary_flag
    c_upw = (secondary2mortar * c) * secondary_flag + (primary2mortar * avg * c) * primary_flag

    return lam / kn / mortar_volumes + (
        secondary2mortar * p - primary2mortar * trace(disc, kw, p, mortar2primary * lam)
    ) / mu(c_upw)


def upwind(disc, c, lam, q):
    "Discretization of the advective transport with upwind weighting"
    div = disc.proj["div"]
    avg = disc.proj["avg"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"]
    mortar2secondary = disc.proj["mortar2secondary"]
    return (div * (disc.upwind(c, q) * q) + disc.mortar_upwind(
        c, lam, div, avg, primary2mortar, secondary2mortar, mortar2primary, mortar2secondary
    ))


def diffusive(disc, c, lam_c):
    "Discretizatio of the diffusive transport"
    kw = disc.problem.transport_keyword
    diff = disc.mat[kw]["flux"]
    bc_val_c = disc.mat[kw]["bc_values"]
    bound_diff = disc.mat[kw]["bound_flux"]
    mortar2primary = disc.proj["mortar2primary"]
    mortar2secondary = disc.proj["mortar2secondary"]
    div = disc.proj["div"]
    return (
        div * (diff * c + bound_diff * (mortar2primary * lam_c + bc_val_c))
        - mortar2secondary * lam_c
    )


def transport(disc, lam, lam0, c, c0, lam_c, lam_c0, q, q0, dt):
    "Discretization of the transport equation"
    kw = disc.problem.transport_keyword
    mass_weight = disc.mat[kw]["mass_weight"]
    theta = disc.mat[kw]["theta"]
    return (
        (c - c0) * (mass_weight / dt)
        + (upwind(disc, c, lam, q) + diffusive(disc, c, lam_c)) * theta
        + (upwind(disc, c0, lam0, q0) + diffusive(disc, c0, lam_c0)) * (1 - theta)
    )


def coupling_law_c(disc, c, lam_c):
    "Discretization of the transport between domains"
    kw = disc.problem.transport_keyword
    Dn = disc.mat[kw]["dn"]
    mortar_volumes = disc.geo["mortar_volumes"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"]
    return (
        lam_c / Dn / mortar_volumes
        + (secondary2mortar * c
           - primary2mortar * trace(disc, kw, c, mortar2primary * lam_c))
    )


def viscous_flow(disc, problem, verbosity=1):
    """
    Solve the coupled problem of fluid flow and temperature transport, where the
    viscosity is depending on the viscosity.
    Darcy's law and mass conservation is solved for the fluid flow:
    u = -K/mu(c) grad p,   div u = 0,
    where mu(c) is a given concentration depending viscosity.
    The concentration is advective and diffusive:
    \partial phi c /\partial t + div (cu) -div (D grad c) = 0.

    A darcy type coupling is assumed between grids of different dimensions:
    lambda = -kn/mu(c) * (p^lower - p^higher),
    and similar for the diffusivity:
    lambda_c = -D * (c^lower - c^higher).

    Parameters:
    disc (discretization.ViscousFlow): A viscous flow discretization class
    problem (problem.ViscousProblem): a viscous flow problem class

    Returns:
    None

    The solution is exported to vtk and csv.
    """
    simulation_start = time.time()

    # Define ad variables
    if verbosity > 0:
        print("Apply initial grid adaptivity")
    # We solve for inital pressure and mortar flux by fixing the temperature
    # to the initial value.
    c0 = problem.initial_concentration()
    for i in range(problem.max_grid_level):
        adaptivity.adapt_domain(disc, c0)
        disc.update_problem(problem)
        c0 = problem.initial_concentration()
    gb = problem.gb
    if verbosity > 0:
        print("Initiate variables")
    p, lam, q, c, lam_c, sol, p_ix, lam_ix, c_ix, lam_c_ix = initiate_variables(
        disc, c0
    )
    if verbosity > 0:
        print("Prepare time stepping")
    dt = problem.time_step_param["initial_dt"]
    t = 0
    k = 0

    # Export initial condition
    exporter = pp.Exporter(
        gb,
        problem.time_step_param["file_name"],
        problem.time_step_param["vtk_folder_name"],
        fixed_grid=False,
    )
    updated_grid = False
    cmooth = (c.val + c0) / 2
    viz.split_variables(gb, [p.val, c.val, cmooth], ["pressure", "concentration", "smooth"])
    if problem.write_vtk_for_time(t, k):
        exporter.write_vtk(["pressure", "concentration"], time_dependent=True, grid=gb)
        times = [0]
    else:
        times = []

    # Store average concentration
    out_file_name = (
        problem.time_step_param["csv_folder_name"] + "/" + problem.time_step_param["file_name"] + ".csv"
    )
    out_file_smooth_name = (
        problem.time_step_param["csv_folder_name"] +
        "/" + problem.time_step_param["file_name"] +
        "_smooth.csv"
    )
    out_file = utils.initiate_csc_file(out_file_name)
    out_file_smooth = utils.initiate_csc_file(out_file_smooth_name)
    utils.store_time_data(problem, out_file, gb, 0.0)
    utils.store_time_data(problem, out_file_smooth, gb, 0.0, "smooth")

    time_disc_tot = 0
    time_output_tot = 0
    time_vtk_tot = 0
    time_nwtn_tot = 0
    time_adpt_tot = 0
    # Time iteration
    while t <= problem.time_step_param["end_time"] - dt + 1e-8:
        time_step_start = time.time()
        time_disc = 0
        time_nwtn = 0
        t += dt
        k += 1
        if verbosity > 0:
            print("Solving time step: ", k, " dt: ", dt, " Time: ", t)

        time_adaptive_start = time.time()
        if problem.adapt_domain(t, k):
            c0, adapted_domain = adaptivity.adapt_domain(disc, c.val)
            if adapted_domain:
                disc.update_problem(problem)
                gb = problem.gb
                p, lam, q, c, lam_c, sol, p_ix, lam_ix, c_ix, lam_c_ix = initiate_variables(
                    disc, c0
                )
                updated_grid = True

        time_adaptive = time.time() - time_adaptive_start
        time_adpt_tot += time_adaptive

        p0 = p.val
        lam0 = lam.val
        c0 = c.val
        lam_c0 = lam_c.val
        q0 = q.val

        err = np.inf
        newton_it = 0
        sol0 = sol.copy()
        # Newton iteration
        while err > 1e-9:
            newton_it += 1
            # Calculate flux
            q = darcy(disc, p, c, lam)
            tic = time.time()
            equation = ad.concatenate(
                (
                    mass_conservation(disc, lam, q),
                    coupling_law_p(disc, p, lam, c),
                    transport(disc, lam, lam0, c, c0, lam_c, lam_c0, q, q0, dt),
                    coupling_law_c(disc, c, lam_c),
                )
            )
            err = np.max(np.abs(equation.val))
            time_disc += time.time() - tic
            if err < 1e-9:
                break
            tic = time.time()
            if verbosity > 1:
                print("newton iteration number: ", newton_it - 1, ". Error: ", err)
            
            sol = sol - disc.linear_solver(equation.jac, equation.val)
            time_nwtn += time.time() - tic
            # Update variables
            p.val = sol[p_ix]
            lam.val = sol[lam_ix]

            c.val = sol[c_ix]
            lam_c.val = sol[lam_c_ix]

            if err != err or newton_it > 10 or err > 10e10:
                # Reset
                if verbosity > 0:
                    print("failed Netwon, reducing time step")
                t -= dt / 2
                dt = dt / 2
                p.val = p0
                lam.val = lam0

                c.val = c0
                lam_c.val = lam_c0

                sol = sol0
                err = np.inf
                newton_it = 0
            # print(err)
        # Newton solver finished
        # Update auxillary variables
        q.val = darcy(disc, p.val, c.val, lam.val)

        if verbosity > 0:
            print("Converged Newton in : ", newton_it - 1, " iterations. Error: ", err)
        if newton_it < 3:
            dt = dt * 1.2
        elif newton_it < 7:
            dt *= 1.1
        dt = problem.sugguest_time_step(t, dt)
        
        # Store solution
        time_output_start = time.time()
        cmooth = (c.val + c0) / 2
        viz.split_variables(gb, [p.val, c.val, cmooth], ["pressure", "concentration", "smooth"])
        utils.store_time_data(problem, out_file, gb, t)
        utils.store_time_data(problem, out_file_smooth, gb, t, "smooth")
        time_out = time.time() - time_output_start
        time_output_tot += time_out

        time_vtk_start = time.time()
        if problem.write_vtk_for_time(t, k):
            if updated_grid:
                exporter.write_vtk(
                    ["pressure", "concentration"], time_dependent=True, grid=gb
                )
            else:
                exporter.write_vtk(["pressure", "concentration"], time_dependent=True)
            times.append(t)
            exporter.write_pvd(timestep=np.array(times))
        
        # Calculate the solution times
        time_vtk = time.time() - time_vtk_start
        time_vtk_tot += time.time() - time_vtk_start

        time_step_time = time.time() - time_step_start
        time_left = time_step_time * (problem.time_step_param["end_time"] - t) / dt

        time_disc_tot += time_disc
        time_nwtn_tot += time_nwtn
        if verbosity > 0:
            print("Time step took: {0:.3f} s".format(time_step_time))
            print("Adaptivity took: {0:.3f} s".format(time_adaptive))
            print("Discretization took: {0:.3f} s".format(time_disc))
            print("Solving linear system took: {0:.3f} s".format(time_nwtn))
            print("Writing output files took: {0:.3f} s".format(time_out + time_vtk))
            print(
                "Estimated time left: {0:.3f} s ({1:.3f} h)".format(
                    time_left, time_left / 3600
                )
            )
            print("-------------------------------------------------------------------------------\n")
    # Time stepping finished
    time_vtk_start = time.time()
    exporter.write_pvd(timestep=np.array(times))
    out_file.close()
    time_vtk_tot += time.time() - time_output_start

    if verbosity > 0:

        print(
            "\n Finished simulation. It took: {0:.3f} s ({1:.3f} h)".format(
                time.time() - simulation_start, (time.time() - simulation_start) / 3600
            )
        )
        print(
            "Adaptivity took: {0:.3f} s  ({1:.3f} h)".format(
                time_adpt_tot, time_adpt_tot / 3600
            )
        )
        print(
            "Discretization took: {0:.3f} s  ({1:.3f} h)".format(
                time_disc_tot, time_disc_tot / 3600
            )
        )
        print(
            "Solving linear system took: {0:.3f} s ({1:.3f} h)".format(
                time_nwtn_tot, time_nwtn_tot / 3600
            )
        )
        print(
            "Writing output files took: {0:.3f} s ({1:.3f} h)".format(
                time_output_tot, time_output_tot / 3600
            )
        )
        print(
            "Writing VTK files took: {0:.3f} s ({1:.3f} h)".format(
                time_vtk_tot, time_vtk_tot / 3600
            )
        )
