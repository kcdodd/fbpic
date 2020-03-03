# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the GPU using CUDA.
"""
from numba import cuda
import math
from scipy.constants import c
import numpy as np

# -------------------------------
# Particle shape Factor functions
# -------------------------------

# Linear shapes
@cuda.jit(device=True, inline=True)
def z_shape_linear(cell_position, index):
    iz = int(math.ceil(cell_position)) - 1
    s = 0.
    if index == 0:
        s = iz+1.-cell_position
    elif index == 1:
        s = cell_position - iz
    return s

@cuda.jit(device=True, inline=True)
def r_shape_linear(cell_position, index):
    flip_factor = 1.
    ir = int(math.ceil(cell_position)) - 1
    s = 0.
    if index == 0:
        if ir < 0:
            flip_factor = -1.
        s = flip_factor*(ir+1.-cell_position)
    elif index == 1:
        s = flip_factor*(cell_position - ir)
    return s

# Cubic shapes
@cuda.jit(device=True, inline=True)
def z_shape_cubic(cell_position, index):
    iz = int(math.ceil(cell_position)) - 2
    s = 0.
    if index == 0:
        s = (-1./6.)*((cell_position-iz)-2)**3
    elif index == 1:
        s = (1./6.)*(3*((cell_position-(iz+1))**3)-6*((cell_position-(iz+1))**2)+4)
    elif index == 2:
        s = (1./6.)*(3*(((iz+2)-cell_position)**3)-6*(((iz+2)-cell_position)**2)+4)
    elif index == 3:
        s = (-1./6.)*(((iz+3)-cell_position)-2)**3
    return s

@cuda.jit(device=True, inline=True)
def r_shape_cubic(cell_position, index):
    flip_factor = 1.
    ir = int(math.ceil(cell_position)) - 2
    s = 0.
    if index == 0:
        if ir < 0:
            flip_factor = -1.
        s = flip_factor*(-1./6.)*((cell_position-ir)-2)**3
    elif index == 1:
        if ir+1 < 0:
            flip_factor = -1.
        s = flip_factor*(1./6.)*(3*((cell_position-(ir+1))**3)-6*((cell_position-(ir+1))**2)+4)
    elif index == 2:
        if ir+2 < 0:
            flip_factor = -1.
        s = flip_factor*(1./6.)*(3*(((ir+2)-cell_position)**3)-6*(((ir+2)-cell_position)**2)+4)
    elif index == 3:
        if ir+3 < 0:
            flip_factor = -1.
        s = flip_factor*(-1./6.)*(((ir+3)-cell_position)-2)**3
    return s
