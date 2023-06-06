"""
Module for handling vizulisation


Copyright 2023 Runar Lie Berge

This file is part of ViscFrac.

ViscFrac is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

ViscFrac is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
ViscFrac. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import math

try:
    import viscous_porepy as pp
except ModuleNotFoundError:
    print("could not load PorePy")

def split_variables(gb, variables, names):
    dof_start = 0
    for g, d in gb:
        dof_end = dof_start + g.num_cells
        for i, var in enumerate(variables):
            if isinstance(var, pp.ad.Ad_array):
                var = var.val
            if d.get(pp.STATE) is None:
                d[pp.STATE] = dict()
            d[pp.STATE][names[i]] = var[dof_start:dof_end]
        dof_start = dof_end


def set_unique_file_name(folder, name, file_extension=".pvd"):
    i = 0
    while i < 1000:
        exists = os.path.isfile(folder + "/" + name + "_run_" + str(i) + file_extension)
        if not exists:
            return name + "_run_" + str(i)
        i += 1
    raise ValueError(
        "Could not set unique file name. Reached maximum value of unique names"
    )

def format_data(formater, value):
    # docstring inherited
    e = math.floor(math.log10(abs(value)))
    s = round(value / 10**e, 10)
    exponent = formater._format_maybe_minus_and_locale("%d", e)
    if s%1==0:
        format = "%d"
    elif s % 0.1 <= 1e-14:
        format = "%1.1f"
    else:
        "%1.10f"
    significand = formater._format_maybe_minus_and_locale(format, s)
    if e == 0:
        return significand
    elif formater._useMathText or formater._usetex:
        exponent = "10^{%s}" % exponent
        return (exponent if s == 1  # reformat 1x10^y as 10^y
                else rf"{significand} \times {exponent}")
    else:
        return f"{significand}e{exponent}"