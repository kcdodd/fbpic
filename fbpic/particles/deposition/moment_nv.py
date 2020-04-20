# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with threading.
"""
from fbpic.discrete import (
  ArrayOp,
  tmp_ndarray,
  tmp_numba_device_ndarray,
  ndarray_fill )

import math
from scipy.constants import c
import numpy as np

from fbpic.utils.threading import (
  nthreads,
  njit_parallel,
  prange,
  get_chunk_indices )

from .threading_methods import (
  Sz_linear,
  Sr_linear,
  Sz_cubic,
  Sr_cubic )

from fbpic.fields.numba_methods import sum_reduce_2d_array

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model
  from .cuda_methods import (
    r_shape_linear,
    z_shape_linear,
    z_shape_cubic,
    r_shape_cubic )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DepositMomentNV ( ArrayOp ):
  """Deposit velocity

  nv = density * v / c
  nu = density * gamma * v / c

  density = 1/cell

  multiply by cell/m^3 for physical density

  to get current density, multiply by q

  Can also compute spatial component (nu) of 4-velocty by providing gamma-1

  to get momentum density, m0 * nu
  to get center of momentum frame, c * nu / ( nke + n )
  see also DepositMomentN
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    ux, uy, uz,
    gamma_minus_1,
    dz, zmin, dr, rmin,
    ptcl_shape,
    gpu = False ):
    """
    Parameters
    ----------
    grid : array<complex>(nm, 3, nz, nr)
      output grid of density * gamma * v / c

      interpolation grid for azimuthal modes 0, 1, ..., nm-1.

    coeff : float
      coefficient to multiply moment before adding to output grid
    weight : array
      The weights of the particles
    cell_idx : array
      The cell index of the particle
    prefix_sum : array
      Represents the cumulative sum of
      the particles per cell
    x : array
      particle positions
    y : array
    z : array
    ux : array
      particle unitless momenta (gamma * v / c)
    uy : array
    uz : array
    gamma_minus_1 : array, optional
      gamma - 1

      If not specified, computes density * gamma * v / c, instead of density * v / c
    dz : float
      z cell size
    zmin : float
    dr : float
      radial cell size
    rmin : float
    ptcl_shape : str
      shape of particles to use for deposition {'linear', 'cubic'}
    gpu : bool
      Use gpu implementation (if available) even when ndarrays are passed in as arguments
    """

    super().exec(
      grid = grid,
      coeff = coeff,
      weight = weight,
      cell_idx = cell_idx,
      prefix_sum = prefix_sum,
      x = x, y = y, z = z,
      ux = ux, uy = uy, uz = uz,
      gamma_minus_1 = gamma_minus_1,
      dz = dz, zmin = zmin,
      dr = dr, rmin = rmin,
      ptcl_shape = ptcl_shape,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):
    #...........................................................................
    @self.attr
    @cuda.jit
    def _cuda_linear_one_mode(x, y, z, w, q,
                             ux, uy, uz,
                             gamma_minus_1,
                             invdz, zmin, Nz,
                             invdr, rmin, Nr,
                             j_r_m, j_t_m, j_z_m, m,
                             cell_idx, prefix_sum):
        """
        Deposition of the current J using numba on the GPU.
        Iterates over the cells and over the particles per cell.
        Calculates the weighted amount of J that is deposited to the
        4 cells surounding the particle based on its shape (linear).

        The particles are sorted by their cell index (the lower cell
        in r and z that they deposit to) and the deposited field
        is split into 4 variables (one for each possible direction,
        e.g. upper in z, lower in r) to maintain parallelism while
        avoiding any race conditions.
        """
        # Get the 1D CUDA grid
        i = cuda.grid(1)
        # Deposit the field per cell in parallel (for threads < number of cells)
        if i < prefix_sum.shape[0]:
            # Retrieve index of upper grid point (in z and r) from prefix-sum index
            # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
            iz_upper = int( i / (Nr+1) )
            ir_upper = int( i - iz_upper * (Nr+1) )
            # Calculate the inclusive offset for the current cell
            # It represents the number of particles contained in all other cells
            # with an index smaller than i + the total number of particles in the
            # current cell (inclusive).
            incl_offset = np.int32(prefix_sum[i])
            # Calculate the frequency per cell from the offset and the previous
            # offset (prefix_sum[i-1]).
            if i > 0:
                frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
            if i == 0:
                frequency_per_cell = np.int32(incl_offset)

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode for r,t and z.
            J_r_m_00 = 0. + 0.j
            J_t_m_00 = 0. + 0.j
            J_z_m_00 = 0. + 0.j
            J_r_m_01 = 0. + 0.j
            J_t_m_01 = 0. + 0.j
            J_z_m_01 = 0. + 0.j
            J_r_m_10 = 0. + 0.j
            J_t_m_10 = 0. + 0.j
            J_z_m_10 = 0. + 0.j
            J_r_m_11 = 0. + 0.j
            J_t_m_11 = 0. + 0.j
            J_z_m_11 = 0. + 0.j

            # Loop over the number of particles per cell
            for j in range(frequency_per_cell):
                # Get the particle index
                # ----------------------
                # (Since incl_offset is a cumulative sum of particle number,
                # and since python index starts at 0, one has to add -1)
                ptcl_idx = incl_offset-1-j

                # Preliminary arrays for the cylindrical conversion
                # --------------------------------------------
                # Position
                xj = x[ptcl_idx]
                yj = y[ptcl_idx]
                zj = z[ptcl_idx]
                # Velocity
                uxj = ux[ptcl_idx]
                uyj = uy[ptcl_idx]
                uzj = uz[ptcl_idx]
                # Weights
                wj = q * w[ptcl_idx] / ( 1. + gamma_minus_1[ptcl_idx] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                # Calculate azimuthal factor
                exptheta_m = 1. + 0.j
                for _ in range(m):
                    exptheta_m *= (cos + 1.j*sin)

                # Get weights for the deposition
                # --------------------------------------------
                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5

                # Calculate the currents
                # ----------------------
                J_r_m_scal = wj*(cos*uxj + sin*uyj) * exptheta_m
                J_t_m_scal = wj*(cos*uyj - sin*uxj) * exptheta_m
                J_z_m_scal = wj*uzj * exptheta_m

                J_r_m_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m_scal
                J_t_m_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m_scal
                J_z_m_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m_scal
                J_r_m_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m_scal
                J_t_m_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m_scal
                J_z_m_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m_scal

                J_r_m_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m_scal
                J_t_m_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m_scal
                J_z_m_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m_scal
                J_r_m_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m_scal
                J_t_m_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m_scal
                J_z_m_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m_scal

            # Calculate longitudinal indices at which to add charge
            iz0 = iz_upper - 1
            iz1 = iz_upper
            if iz0 < 0:
                iz0 += Nz
            # Calculate radial indices at which to add charge
            ir0 = ir_upper - 1
            ir1 = min( ir_upper, Nr-1 )
            if ir0 < 0:
                # Deposition below the axis: fold index into physical region
                ir0 = -(1 + ir0)

            # Atomically add the registers to global memory
            if frequency_per_cell > 0:
                # jr
                cuda.atomic.add(j_r_m.real, (iz0, ir0), J_r_m_00.real)
                cuda.atomic.add(j_r_m.real, (iz0, ir1), J_r_m_10.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir0), J_r_m_01.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir1), J_r_m_11.real)
                if m > 0:
                    cuda.atomic.add(j_r_m.imag, (iz0, ir0), J_r_m_00.imag)
                    cuda.atomic.add(j_r_m.imag, (iz0, ir1), J_r_m_10.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir0), J_r_m_01.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir1), J_r_m_11.imag)
                # jt
                cuda.atomic.add(j_t_m.real, (iz0, ir0), J_t_m_00.real)
                cuda.atomic.add(j_t_m.real, (iz0, ir1), J_t_m_10.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir0), J_t_m_01.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir1), J_t_m_11.real)
                if m > 0:
                    cuda.atomic.add(j_t_m.imag, (iz0, ir0), J_t_m_00.imag)
                    cuda.atomic.add(j_t_m.imag, (iz0, ir1), J_t_m_10.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir0), J_t_m_01.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir1), J_t_m_11.imag)
                # jz
                cuda.atomic.add(j_z_m.real, (iz0, ir0), J_z_m_00.real)
                cuda.atomic.add(j_z_m.real, (iz0, ir1), J_z_m_10.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir0), J_z_m_01.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir1), J_z_m_11.real)
                if m > 0:
                    cuda.atomic.add(j_z_m.imag, (iz0, ir0), J_z_m_00.imag)
                    cuda.atomic.add(j_z_m.imag, (iz0, ir1), J_z_m_10.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir0), J_z_m_01.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir1), J_z_m_11.imag)


    #...........................................................................
    @self.attr
    @cuda.jit
    def _cuda_cubic_one_mode(
        x, y, z, w, q,
        ux, uy, uz,
        gamma_minus_1,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        j_r_m, j_t_m, j_z_m, m,
        cell_idx, prefix_sum):
        """
        Deposition of the current J using numba on the GPU.
        Iterates over the cells and over the particles per cell.
        Calculates the weighted amount of J that is deposited to the
        16 cells surounding the particle based on its shape (cubic).

        The particles are sorted by their cell index (the lower cell
        in r and z that they deposit to) and the deposited field
        is split into 16 variables (one for each cell) to maintain
        parallelism while avoiding any race conditions.

        """
        # Get the 1D CUDA grid
        i = cuda.grid(1)
        # Deposit the field per cell in parallel (for threads < number of cells)
        if i < prefix_sum.shape[0]:
            # Retrieve index of upper grid point (in z and r) from prefix-sum index
            # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
            iz_upper = int( i / (Nr+1) )
            ir_upper = int( i - iz_upper * (Nr+1) )
            # Calculate the inclusive offset for the current cell
            # It represents the number of particles contained in all other cells
            # with an index smaller than i + the total number of particles in the
            # current cell (inclusive).
            incl_offset = np.int32(prefix_sum[i])
            # Calculate the frequency per cell from the offset and the previous
            # offset (prefix_sum[i-1]).
            if i > 0:
                frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
            if i == 0:
                frequency_per_cell = np.int32(incl_offset)

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode for r,t and z.
            J_r_m_00 = 0. + 0.j
            J_t_m_00 = 0. + 0.j
            J_z_m_00 = 0. + 0.j

            J_r_m_01 = 0. + 0.j
            J_t_m_01 = 0. + 0.j
            J_z_m_01 = 0. + 0.j

            J_r_m_02 = 0. + 0.j
            J_t_m_02 = 0. + 0.j
            J_z_m_02 = 0. + 0.j

            J_r_m_03 = 0. + 0.j
            J_t_m_03 = 0. + 0.j
            J_z_m_03 = 0. + 0.j

            J_r_m_10 = 0. + 0.j
            J_t_m_10 = 0. + 0.j
            J_z_m_10 = 0. + 0.j

            J_r_m_11 = 0. + 0.j
            J_t_m_11 = 0. + 0.j
            J_z_m_11 = 0. + 0.j

            J_r_m_12 = 0. + 0.j
            J_t_m_12 = 0. + 0.j
            J_z_m_12 = 0. + 0.j

            J_r_m_13 = 0. + 0.j
            J_t_m_13 = 0. + 0.j
            J_z_m_13 = 0. + 0.j

            J_r_m_20 = 0. + 0.j
            J_t_m_20 = 0. + 0.j
            J_z_m_20 = 0. + 0.j

            J_r_m_21 = 0. + 0.j
            J_t_m_21 = 0. + 0.j
            J_z_m_21 = 0. + 0.j

            J_r_m_22 = 0. + 0.j
            J_t_m_22 = 0. + 0.j
            J_z_m_22 = 0. + 0.j

            J_r_m_23 = 0. + 0.j
            J_t_m_23 = 0. + 0.j
            J_z_m_23 = 0. + 0.j

            J_r_m_30 = 0. + 0.j
            J_t_m_30 = 0. + 0.j
            J_z_m_30 = 0. + 0.j

            J_r_m_31 = 0. + 0.j
            J_t_m_31 = 0. + 0.j
            J_z_m_31 = 0. + 0.j

            J_r_m_32 = 0. + 0.j
            J_t_m_32 = 0. + 0.j
            J_z_m_32 = 0. + 0.j

            J_r_m_33 = 0. + 0.j
            J_t_m_33 = 0. + 0.j
            J_z_m_33 = 0. + 0.j

            # Loop over the number of particles per cell
            for j in range(frequency_per_cell):
                # Get the particle index
                # ----------------------
                # (Since incl_offset is a cumulative sum of particle number,
                # and since python index starts at 0, one has to add -1)
                ptcl_idx = incl_offset-1-j

                # Preliminary arrays for the cylindrical conversion
                # --------------------------------------------
                # Position
                xj = x[ptcl_idx]
                yj = y[ptcl_idx]
                zj = z[ptcl_idx]
                # Velocity
                uxj = ux[ptcl_idx]
                uyj = uy[ptcl_idx]
                uzj = uz[ptcl_idx]
                # Weights
                wj = q * w[ptcl_idx] / ( 1. + gamma_minus_1[ptcl_idx] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                # Calculate azimuthal factor
                exptheta_m = 1. + 0.j
                for _ in range(m):
                    exptheta_m *= (cos + 1.j*sin)

                # Get weights for the deposition
                # --------------------------------------------
                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5

                # Calculate the currents
                # --------------------------------------------
                # Mode 0
                # Mode 1
                J_r_m_scal = wj*(cos*uxj + sin*uyj) * exptheta_m
                J_t_m_scal = wj*(cos*uyj - sin*uxj) * exptheta_m
                J_z_m_scal = wj*uzj * exptheta_m

                J_r_m_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m_scal
                J_r_m_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m_scal
                J_r_m_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m_scal
                J_r_m_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m_scal

                J_r_m_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m_scal
                J_r_m_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m_scal
                J_r_m_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m_scal
                J_r_m_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m_scal

                J_r_m_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m_scal
                J_r_m_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m_scal
                J_r_m_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m_scal
                J_r_m_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m_scal

                J_r_m_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m_scal
                J_r_m_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m_scal
                J_r_m_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m_scal
                J_r_m_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m_scal

                J_t_m_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m_scal
                J_t_m_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m_scal
                J_t_m_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m_scal
                J_t_m_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m_scal

                J_t_m_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m_scal
                J_t_m_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m_scal
                J_t_m_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m_scal
                J_t_m_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m_scal

                J_t_m_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m_scal
                J_t_m_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m_scal
                J_t_m_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m_scal
                J_t_m_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m_scal

                J_t_m_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m_scal
                J_t_m_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m_scal
                J_t_m_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m_scal
                J_t_m_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m_scal

                J_z_m_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m_scal
                J_z_m_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m_scal
                J_z_m_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m_scal
                J_z_m_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m_scal

                J_z_m_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m_scal
                J_z_m_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m_scal
                J_z_m_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m_scal
                J_z_m_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m_scal

                J_z_m_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m_scal
                J_z_m_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m_scal
                J_z_m_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m_scal
                J_z_m_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m_scal

                J_z_m_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m_scal
                J_z_m_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m_scal
                J_z_m_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m_scal
                J_z_m_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m_scal

            # Calculate longitudinal indices at which to add charge
            iz0 = iz_upper - 2
            iz1 = iz_upper - 1
            iz2 = iz_upper
            iz3 = iz_upper + 1
            if iz0 < 0:
                iz0 += Nz
            if iz1 < 0:
                iz1 += Nz
            if iz3 > Nz-1:
                iz3 -= Nz
            # Calculate radial indices at which to add charge
            ir0 = ir_upper - 2
            ir1 = min( ir_upper - 1, Nr-1 )
            ir2 = min( ir_upper    , Nr-1 )
            ir3 = min( ir_upper + 1, Nr-1 )
            if ir0 < 0:
                # Deposition below the axis: fold index into physical region
                ir0 = -(1 + ir0)
            if ir1 < 0:
                # Deposition below the axis: fold index into physical region
                ir1 = -(1 + ir1)

            # Atomically add the registers to global memory
            if frequency_per_cell > 0:
                # jr
                cuda.atomic.add(j_r_m.real, (iz0, ir0), J_r_m_00.real)
                cuda.atomic.add(j_r_m.real, (iz0, ir1), J_r_m_10.real)
                cuda.atomic.add(j_r_m.real, (iz0, ir2), J_r_m_20.real)
                cuda.atomic.add(j_r_m.real, (iz0, ir3), J_r_m_30.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir0), J_r_m_01.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir1), J_r_m_11.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir2), J_r_m_21.real)
                cuda.atomic.add(j_r_m.real, (iz1, ir3), J_r_m_31.real)
                cuda.atomic.add(j_r_m.real, (iz2, ir0), J_r_m_02.real)
                cuda.atomic.add(j_r_m.real, (iz2, ir1), J_r_m_12.real)
                cuda.atomic.add(j_r_m.real, (iz2, ir2), J_r_m_22.real)
                cuda.atomic.add(j_r_m.real, (iz2, ir3), J_r_m_32.real)
                cuda.atomic.add(j_r_m.real, (iz3, ir0), J_r_m_03.real)
                cuda.atomic.add(j_r_m.real, (iz3, ir1), J_r_m_13.real)
                cuda.atomic.add(j_r_m.real, (iz3, ir2), J_r_m_23.real)
                cuda.atomic.add(j_r_m.real, (iz3, ir3), J_r_m_33.real)
                if m > 0:
                    cuda.atomic.add(j_r_m.imag, (iz0, ir0), J_r_m_00.imag)
                    cuda.atomic.add(j_r_m.imag, (iz0, ir1), J_r_m_10.imag)
                    cuda.atomic.add(j_r_m.imag, (iz0, ir2), J_r_m_20.imag)
                    cuda.atomic.add(j_r_m.imag, (iz0, ir3), J_r_m_30.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir0), J_r_m_01.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir1), J_r_m_11.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir2), J_r_m_21.imag)
                    cuda.atomic.add(j_r_m.imag, (iz1, ir3), J_r_m_31.imag)
                    cuda.atomic.add(j_r_m.imag, (iz2, ir0), J_r_m_02.imag)
                    cuda.atomic.add(j_r_m.imag, (iz2, ir1), J_r_m_12.imag)
                    cuda.atomic.add(j_r_m.imag, (iz2, ir2), J_r_m_22.imag)
                    cuda.atomic.add(j_r_m.imag, (iz2, ir3), J_r_m_32.imag)
                    cuda.atomic.add(j_r_m.imag, (iz3, ir0), J_r_m_03.imag)
                    cuda.atomic.add(j_r_m.imag, (iz3, ir1), J_r_m_13.imag)
                    cuda.atomic.add(j_r_m.imag, (iz3, ir2), J_r_m_23.imag)
                    cuda.atomic.add(j_r_m.imag, (iz3, ir3), J_r_m_33.imag)
                # jt
                cuda.atomic.add(j_t_m.real, (iz0, ir0), J_t_m_00.real)
                cuda.atomic.add(j_t_m.real, (iz0, ir1), J_t_m_10.real)
                cuda.atomic.add(j_t_m.real, (iz0, ir2), J_t_m_20.real)
                cuda.atomic.add(j_t_m.real, (iz0, ir3), J_t_m_30.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir0), J_t_m_01.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir1), J_t_m_11.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir2), J_t_m_21.real)
                cuda.atomic.add(j_t_m.real, (iz1, ir3), J_t_m_31.real)
                cuda.atomic.add(j_t_m.real, (iz2, ir0), J_t_m_02.real)
                cuda.atomic.add(j_t_m.real, (iz2, ir1), J_t_m_12.real)
                cuda.atomic.add(j_t_m.real, (iz2, ir2), J_t_m_22.real)
                cuda.atomic.add(j_t_m.real, (iz2, ir3), J_t_m_32.real)
                cuda.atomic.add(j_t_m.real, (iz3, ir0), J_t_m_03.real)
                cuda.atomic.add(j_t_m.real, (iz3, ir1), J_t_m_13.real)
                cuda.atomic.add(j_t_m.real, (iz3, ir2), J_t_m_23.real)
                cuda.atomic.add(j_t_m.real, (iz3, ir3), J_t_m_33.real)
                if m > 0:
                    cuda.atomic.add(j_t_m.imag, (iz0, ir0), J_t_m_00.imag)
                    cuda.atomic.add(j_t_m.imag, (iz0, ir1), J_t_m_10.imag)
                    cuda.atomic.add(j_t_m.imag, (iz0, ir2), J_t_m_20.imag)
                    cuda.atomic.add(j_t_m.imag, (iz0, ir3), J_t_m_30.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir0), J_t_m_01.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir1), J_t_m_11.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir2), J_t_m_21.imag)
                    cuda.atomic.add(j_t_m.imag, (iz1, ir3), J_t_m_31.imag)
                    cuda.atomic.add(j_t_m.imag, (iz2, ir0), J_t_m_02.imag)
                    cuda.atomic.add(j_t_m.imag, (iz2, ir1), J_t_m_12.imag)
                    cuda.atomic.add(j_t_m.imag, (iz2, ir2), J_t_m_22.imag)
                    cuda.atomic.add(j_t_m.imag, (iz2, ir3), J_t_m_32.imag)
                    cuda.atomic.add(j_t_m.imag, (iz3, ir0), J_t_m_03.imag)
                    cuda.atomic.add(j_t_m.imag, (iz3, ir1), J_t_m_13.imag)
                    cuda.atomic.add(j_t_m.imag, (iz3, ir2), J_t_m_23.imag)
                    cuda.atomic.add(j_t_m.imag, (iz3, ir3), J_t_m_33.imag)
                # jz
                cuda.atomic.add(j_z_m.real, (iz0, ir0), J_z_m_00.real)
                cuda.atomic.add(j_z_m.real, (iz0, ir1), J_z_m_10.real)
                cuda.atomic.add(j_z_m.real, (iz0, ir2), J_z_m_20.real)
                cuda.atomic.add(j_z_m.real, (iz0, ir3), J_z_m_30.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir0), J_z_m_01.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir1), J_z_m_11.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir2), J_z_m_21.real)
                cuda.atomic.add(j_z_m.real, (iz1, ir3), J_z_m_31.real)
                cuda.atomic.add(j_z_m.real, (iz2, ir0), J_z_m_02.real)
                cuda.atomic.add(j_z_m.real, (iz2, ir1), J_z_m_12.real)
                cuda.atomic.add(j_z_m.real, (iz2, ir2), J_z_m_22.real)
                cuda.atomic.add(j_z_m.real, (iz2, ir3), J_z_m_32.real)
                cuda.atomic.add(j_z_m.real, (iz3, ir0), J_z_m_03.real)
                cuda.atomic.add(j_z_m.real, (iz3, ir1), J_z_m_13.real)
                cuda.atomic.add(j_z_m.real, (iz3, ir2), J_z_m_23.real)
                cuda.atomic.add(j_z_m.real, (iz3, ir3), J_z_m_33.real)
                if m > 0:
                    cuda.atomic.add(j_z_m.imag, (iz0, ir0), J_z_m_00.imag)
                    cuda.atomic.add(j_z_m.imag, (iz0, ir1), J_z_m_10.imag)
                    cuda.atomic.add(j_z_m.imag, (iz0, ir2), J_z_m_20.imag)
                    cuda.atomic.add(j_z_m.imag, (iz0, ir3), J_z_m_30.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir0), J_z_m_01.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir1), J_z_m_11.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir2), J_z_m_21.imag)
                    cuda.atomic.add(j_z_m.imag, (iz1, ir3), J_z_m_31.imag)
                    cuda.atomic.add(j_z_m.imag, (iz2, ir0), J_z_m_02.imag)
                    cuda.atomic.add(j_z_m.imag, (iz2, ir1), J_z_m_12.imag)
                    cuda.atomic.add(j_z_m.imag, (iz2, ir2), J_z_m_22.imag)
                    cuda.atomic.add(j_z_m.imag, (iz2, ir3), J_z_m_32.imag)
                    cuda.atomic.add(j_z_m.imag, (iz3, ir0), J_z_m_03.imag)
                    cuda.atomic.add(j_z_m.imag, (iz3, ir1), J_z_m_13.imag)
                    cuda.atomic.add(j_z_m.imag, (iz3, ir2), J_z_m_23.imag)
                    cuda.atomic.add(j_z_m.imag, (iz3, ir3), J_z_m_33.imag)

    #.........................................................................
    @self.attr
    @cuda.jit
    def _cuda_linear(
        x, y, z, w, q,
        ux, uy, uz,
        gamma_minus_1,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        j_r_m0, j_r_m1,
        j_t_m0, j_t_m1,
        j_z_m0, j_z_m1,
        cell_idx, prefix_sum):
        """
        Deposition of the current J using numba on the GPU.
        Iterates over the cells and over the particles per cell.
        Calculates the weighted amount of J that is deposited to the
        4 cells surounding the particle based on its shape (linear).

        The particles are sorted by their cell index (the lower cell
        in r and z that they deposit to) and the deposited field
        is split into 4 variables (one for each possible direction,
        e.g. upper in z, lower in r) to maintain parallelism while
        avoiding any race conditions.

        """
        # Get the 1D CUDA grid
        i = cuda.grid(1)
        # Deposit the field per cell in parallel (for threads < number of cells)
        if i < prefix_sum.shape[0]:
            # Retrieve index of upper grid point (in z and r) from prefix-sum index
            # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
            iz_upper = int( i / (Nr+1) )
            ir_upper = int( i - iz_upper * (Nr+1) )
            # Calculate the inclusive offset for the current cell
            # It represents the number of particles contained in all other cells
            # with an index smaller than i + the total number of particles in the
            # current cell (inclusive).
            incl_offset = np.int32(prefix_sum[i])
            # Calculate the frequency per cell from the offset and the previous
            # offset (prefix_sum[i-1]).
            if i > 0:
                frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
            if i == 0:
                frequency_per_cell = np.int32(incl_offset)

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode for r,t and z.

            J_r_m0_00 = 0.
            J_r_m1_00 = 0. + 0.j
            J_t_m0_00 = 0.
            J_t_m1_00 = 0. + 0.j
            J_z_m0_00 = 0.
            J_z_m1_00 = 0. + 0.j

            J_r_m0_01 = 0.
            J_r_m1_01 = 0. + 0.j
            J_t_m0_01 = 0.
            J_t_m1_01 = 0. + 0.j
            J_z_m0_01 = 0.
            J_z_m1_01 = 0. + 0.j

            J_r_m0_10 = 0.
            J_r_m1_10 = 0. + 0.j
            J_t_m0_10 = 0.
            J_t_m1_10 = 0. + 0.j
            J_z_m0_10 = 0.
            J_z_m1_10 = 0. + 0.j

            J_r_m0_11 = 0.
            J_r_m1_11 = 0. + 0.j
            J_t_m0_11 = 0.
            J_t_m1_11 = 0. + 0.j
            J_z_m0_11 = 0.
            J_z_m1_11 = 0. + 0.j


            # Loop over the number of particles per cell
            for j in range(frequency_per_cell):
                # Get the particle index
                # ----------------------
                # (Since incl_offset is a cumulative sum of particle number,
                # and since python index starts at 0, one has to add -1)
                ptcl_idx = incl_offset-1-j

                # Preliminary arrays for the cylindrical conversion
                # --------------------------------------------
                # Position
                xj = x[ptcl_idx]
                yj = y[ptcl_idx]
                zj = z[ptcl_idx]
                # Velocity
                uxj = ux[ptcl_idx]
                uyj = uy[ptcl_idx]
                uzj = uz[ptcl_idx]
                # Weights
                wj = q * w[ptcl_idx] / ( 1. + gamma_minus_1[ptcl_idx] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                exptheta_m0 = 1.
                exptheta_m1 = cos + 1.j*sin

                # Get weights for the deposition
                # --------------------------------------------
                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5

                # Calculate the currents
                # ----------------------
                # Mode 0
                J_r_m0_scal = wj*(cos*uxj + sin*uyj) * exptheta_m0
                J_t_m0_scal = wj*(cos*uyj - sin*uxj) * exptheta_m0
                J_z_m0_scal = wj*uzj * exptheta_m0
                # Mode 1
                J_r_m1_scal = wj*(cos*uxj + sin*uyj) * exptheta_m1
                J_t_m1_scal = wj*(cos*uyj - sin*uxj) * exptheta_m1
                J_z_m1_scal = wj*uzj * exptheta_m1

                J_r_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m1_scal

                J_r_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m1_scal

            # Calculate longitudinal indices at which to add charge
            iz0 = iz_upper - 1
            iz1 = iz_upper
            if iz0 < 0:
                iz0 += Nz
            # Calculate radial indices at which to add charge
            ir0 = ir_upper - 1
            ir1 = min( ir_upper, Nr-1 )
            if ir0 < 0:
                # Deposition below the axis: fold index into physical region
                ir0 = -(1 + ir0)

            # Atomically add the registers to global memory
            if frequency_per_cell > 0:
                # jr: Mode 0
                cuda.atomic.add(j_r_m0.real, (iz0, ir0), J_r_m0_00.real)
                cuda.atomic.add(j_r_m0.real, (iz0, ir1), J_r_m0_10.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir0), J_r_m0_01.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir1), J_r_m0_11.real)
                # jr: Mode 1
                cuda.atomic.add(j_r_m1.real, (iz0, ir0), J_r_m1_00.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir0), J_r_m1_00.imag)
                cuda.atomic.add(j_r_m1.real, (iz0, ir1), J_r_m1_10.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir1), J_r_m1_10.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir0), J_r_m1_01.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir0), J_r_m1_01.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir1), J_r_m1_11.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir1), J_r_m1_11.imag)
                # jt: Mode 0
                cuda.atomic.add(j_t_m0.real, (iz0, ir0), J_t_m0_00.real)
                cuda.atomic.add(j_t_m0.real, (iz0, ir1), J_t_m0_10.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir0), J_t_m0_01.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir1), J_t_m0_11.real)
                # jt: Mode 1
                cuda.atomic.add(j_t_m1.real, (iz0, ir0), J_t_m1_00.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir0), J_t_m1_00.imag)
                cuda.atomic.add(j_t_m1.real, (iz0, ir1), J_t_m1_10.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir1), J_t_m1_10.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir0), J_t_m1_01.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir0), J_t_m1_01.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir1), J_t_m1_11.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir1), J_t_m1_11.imag)
                # jz: Mode 0
                cuda.atomic.add(j_z_m0.real, (iz0, ir0), J_z_m0_00.real)
                cuda.atomic.add(j_z_m0.real, (iz0, ir1), J_z_m0_10.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir0), J_z_m0_01.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir1), J_z_m0_11.real)
                # jz: Mode 1
                cuda.atomic.add(j_z_m1.real, (iz0, ir0), J_z_m1_00.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir0), J_z_m1_00.imag)
                cuda.atomic.add(j_z_m1.real, (iz0, ir1), J_z_m1_10.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir1), J_z_m1_10.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir0), J_z_m1_01.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir0), J_z_m1_01.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir1), J_z_m1_11.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir1), J_z_m1_11.imag)

    #...........................................................................
    @self.attr
    @cuda.jit
    def _cuda_cubic(
        x, y, z, w, q,
        ux, uy, uz,
        gamma_minus_1,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        j_r_m0, j_r_m1,
        j_t_m0, j_t_m1,
        j_z_m0, j_z_m1,
        cell_idx, prefix_sum):
        """
        Deposition of the current J using numba on the GPU.
        Iterates over the cells and over the particles per cell.
        Calculates the weighted amount of J that is deposited to the
        16 cells surounding the particle based on its shape (cubic).

        The particles are sorted by their cell index (the lower cell
        in r and z that they deposit to) and the deposited field
        is split into 16 variables (one for each cell) to maintain
        parallelism while avoiding any race conditions.

        """
        # Get the 1D CUDA grid
        i = cuda.grid(1)
        # Deposit the field per cell in parallel (for threads < number of cells)
        if i < prefix_sum.shape[0]:
            # Retrieve index of upper grid point (in z and r) from prefix-sum index
            # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
            iz_upper = int( i / (Nr+1) )
            ir_upper = int( i - iz_upper * (Nr+1) )
            # Calculate the inclusive offset for the current cell
            # It represents the number of particles contained in all other cells
            # with an index smaller than i + the total number of particles in the
            # current cell (inclusive).
            incl_offset = np.int32(prefix_sum[i])
            # Calculate the frequency per cell from the offset and the previous
            # offset (prefix_sum[i-1]).
            if i > 0:
                frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
            if i == 0:
                frequency_per_cell = np.int32(incl_offset)

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode for r,t and z.
            J_r_m0_00 = 0.
            J_t_m0_00 = 0.
            J_z_m0_00 = 0.
            J_r_m1_00 = 0. + 0.j
            J_t_m1_00 = 0. + 0.j
            J_z_m1_00 = 0. + 0.j

            J_r_m0_01 = 0.
            J_t_m0_01 = 0.
            J_z_m0_01 = 0.
            J_r_m1_01 = 0. + 0.j
            J_t_m1_01 = 0. + 0.j
            J_z_m1_01 = 0. + 0.j

            J_r_m0_02 = 0.
            J_t_m0_02 = 0.
            J_z_m0_02 = 0.
            J_r_m1_02 = 0. + 0.j
            J_t_m1_02 = 0. + 0.j
            J_z_m1_02 = 0. + 0.j

            J_r_m0_03 = 0.
            J_t_m0_03 = 0.
            J_z_m0_03 = 0.
            J_r_m1_03 = 0. + 0.j
            J_t_m1_03 = 0. + 0.j
            J_z_m1_03 = 0. + 0.j

            J_r_m0_10 = 0.
            J_t_m0_10 = 0.
            J_z_m0_10 = 0.
            J_r_m1_10 = 0. + 0.j
            J_t_m1_10 = 0. + 0.j
            J_z_m1_10 = 0. + 0.j

            J_r_m0_11 = 0.
            J_t_m0_11 = 0.
            J_z_m0_11 = 0.
            J_r_m1_11 = 0. + 0.j
            J_t_m1_11 = 0. + 0.j
            J_z_m1_11 = 0. + 0.j

            J_r_m0_12 = 0.
            J_t_m0_12 = 0.
            J_z_m0_12 = 0.
            J_r_m1_12 = 0. + 0.j
            J_t_m1_12 = 0. + 0.j
            J_z_m1_12 = 0. + 0.j

            J_r_m0_13 = 0.
            J_t_m0_13 = 0.
            J_z_m0_13 = 0.
            J_r_m1_13 = 0. + 0.j
            J_t_m1_13 = 0. + 0.j
            J_z_m1_13 = 0. + 0.j

            J_r_m0_20 = 0.
            J_t_m0_20 = 0.
            J_z_m0_20 = 0.
            J_r_m1_20 = 0. + 0.j
            J_t_m1_20 = 0. + 0.j
            J_z_m1_20 = 0. + 0.j

            J_r_m0_21 = 0.
            J_t_m0_21 = 0.
            J_z_m0_21 = 0.
            J_r_m1_21 = 0. + 0.j
            J_t_m1_21 = 0. + 0.j
            J_z_m1_21 = 0. + 0.j

            J_r_m0_22 = 0.
            J_t_m0_22 = 0.
            J_z_m0_22 = 0.
            J_r_m1_22 = 0. + 0.j
            J_t_m1_22 = 0. + 0.j
            J_z_m1_22 = 0. + 0.j

            J_r_m0_23 = 0.
            J_t_m0_23 = 0.
            J_z_m0_23 = 0.
            J_r_m1_23 = 0. + 0.j
            J_t_m1_23 = 0. + 0.j
            J_z_m1_23 = 0. + 0.j

            J_r_m0_30 = 0.
            J_t_m0_30 = 0.
            J_z_m0_30 = 0.
            J_r_m1_30 = 0. + 0.j
            J_t_m1_30 = 0. + 0.j
            J_z_m1_30 = 0. + 0.j

            J_r_m0_31 = 0.
            J_t_m0_31 = 0.
            J_z_m0_31 = 0.
            J_r_m1_31 = 0. + 0.j
            J_t_m1_31 = 0. + 0.j
            J_z_m1_31 = 0. + 0.j

            J_r_m0_32 = 0.
            J_t_m0_32 = 0.
            J_z_m0_32 = 0.
            J_r_m1_32 = 0. + 0.j
            J_t_m1_32 = 0. + 0.j
            J_z_m1_32 = 0. + 0.j

            J_r_m0_33 = 0.
            J_t_m0_33 = 0.
            J_z_m0_33 = 0.
            J_r_m1_33 = 0. + 0.j
            J_t_m1_33 = 0. + 0.j
            J_z_m1_33 = 0. + 0.j

            # Loop over the number of particles per cell
            for j in range(frequency_per_cell):
                # Get the particle index
                # ----------------------
                # (Since incl_offset is a cumulative sum of particle number,
                # and since python index starts at 0, one has to add -1)
                ptcl_idx = incl_offset-1-j

                # Preliminary arrays for the cylindrical conversion
                # --------------------------------------------
                # Position
                xj = x[ptcl_idx]
                yj = y[ptcl_idx]
                zj = z[ptcl_idx]
                # Velocity
                uxj = ux[ptcl_idx]
                uyj = uy[ptcl_idx]
                uzj = uz[ptcl_idx]
                # Weights
                wj = q * w[ptcl_idx] / ( 1. + gamma_minus_1[ptcl_idx] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                exptheta_m0 = 1.
                exptheta_m1 = cos + 1.j*sin

                # Get weights for the deposition
                # --------------------------------------------
                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5

                # Calculate the currents
                # --------------------------------------------
                # Mode 0
                J_r_m0_scal = wj*(cos*uxj + sin*uyj) * exptheta_m0
                J_t_m0_scal = wj*(cos*uyj - sin*uxj) * exptheta_m0
                J_z_m0_scal = wj*uzj * exptheta_m0
                # Mode 1
                J_r_m1_scal = wj*(cos*uxj + sin*uyj) * exptheta_m1
                J_t_m1_scal = wj*(cos*uyj - sin*uxj) * exptheta_m1
                J_z_m1_scal = wj*uzj * exptheta_m1

                J_r_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_t_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_z_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

            # Calculate longitudinal indices at which to add charge
            iz0 = iz_upper - 2
            iz1 = iz_upper - 1
            iz2 = iz_upper
            iz3 = iz_upper + 1
            if iz0 < 0:
                iz0 += Nz
            if iz1 < 0:
                iz1 += Nz
            if iz3 > Nz-1:
                iz3 -= Nz
            # Calculate radial indices at which to add charge
            ir0 = ir_upper - 2
            ir1 = min( ir_upper - 1, Nr-1 )
            ir2 = min( ir_upper    , Nr-1 )
            ir3 = min( ir_upper + 1, Nr-1 )
            if ir0 < 0:
                # Deposition below the axis: fold index into physical region
                ir0 = -(1 + ir0)
            if ir1 < 0:
                # Deposition below the axis: fold index into physical region
                ir1 = -(1 + ir1)

            # Atomically add the registers to global memory
            if frequency_per_cell > 0:
                # jr: Mode 0
                cuda.atomic.add(j_r_m0.real, (iz0, ir0), J_r_m0_00.real)
                cuda.atomic.add(j_r_m0.real, (iz0, ir1), J_r_m0_10.real)
                cuda.atomic.add(j_r_m0.real, (iz0, ir2), J_r_m0_20.real)
                cuda.atomic.add(j_r_m0.real, (iz0, ir3), J_r_m0_30.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir0), J_r_m0_01.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir1), J_r_m0_11.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir2), J_r_m0_21.real)
                cuda.atomic.add(j_r_m0.real, (iz1, ir3), J_r_m0_31.real)
                cuda.atomic.add(j_r_m0.real, (iz2, ir0), J_r_m0_02.real)
                cuda.atomic.add(j_r_m0.real, (iz2, ir1), J_r_m0_12.real)
                cuda.atomic.add(j_r_m0.real, (iz2, ir2), J_r_m0_22.real)
                cuda.atomic.add(j_r_m0.real, (iz2, ir3), J_r_m0_32.real)
                cuda.atomic.add(j_r_m0.real, (iz3, ir0), J_r_m0_03.real)
                cuda.atomic.add(j_r_m0.real, (iz3, ir1), J_r_m0_13.real)
                cuda.atomic.add(j_r_m0.real, (iz3, ir2), J_r_m0_23.real)
                cuda.atomic.add(j_r_m0.real, (iz3, ir3), J_r_m0_33.real)
                # jr: Mode 1
                cuda.atomic.add(j_r_m1.real, (iz0, ir0), J_r_m1_00.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir0), J_r_m1_00.imag)
                cuda.atomic.add(j_r_m1.real, (iz0, ir1), J_r_m1_10.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir1), J_r_m1_10.imag)
                cuda.atomic.add(j_r_m1.real, (iz0, ir2), J_r_m1_20.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir2), J_r_m1_20.imag)
                cuda.atomic.add(j_r_m1.real, (iz0, ir3), J_r_m1_30.real)
                cuda.atomic.add(j_r_m1.imag, (iz0, ir3), J_r_m1_30.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir0), J_r_m1_01.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir0), J_r_m1_01.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir1), J_r_m1_11.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir1), J_r_m1_11.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir2), J_r_m1_21.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir2), J_r_m1_21.imag)
                cuda.atomic.add(j_r_m1.real, (iz1, ir3), J_r_m1_31.real)
                cuda.atomic.add(j_r_m1.imag, (iz1, ir3), J_r_m1_31.imag)
                cuda.atomic.add(j_r_m1.real, (iz2, ir0), J_r_m1_02.real)
                cuda.atomic.add(j_r_m1.imag, (iz2, ir0), J_r_m1_02.imag)
                cuda.atomic.add(j_r_m1.real, (iz2, ir1), J_r_m1_12.real)
                cuda.atomic.add(j_r_m1.imag, (iz2, ir1), J_r_m1_12.imag)
                cuda.atomic.add(j_r_m1.real, (iz2, ir2), J_r_m1_22.real)
                cuda.atomic.add(j_r_m1.imag, (iz2, ir2), J_r_m1_22.imag)
                cuda.atomic.add(j_r_m1.real, (iz2, ir3), J_r_m1_32.real)
                cuda.atomic.add(j_r_m1.imag, (iz2, ir3), J_r_m1_32.imag)
                cuda.atomic.add(j_r_m1.real, (iz3, ir0), J_r_m1_03.real)
                cuda.atomic.add(j_r_m1.imag, (iz3, ir0), J_r_m1_03.imag)
                cuda.atomic.add(j_r_m1.real, (iz3, ir1), J_r_m1_13.real)
                cuda.atomic.add(j_r_m1.imag, (iz3, ir1), J_r_m1_13.imag)
                cuda.atomic.add(j_r_m1.real, (iz3, ir2), J_r_m1_23.real)
                cuda.atomic.add(j_r_m1.imag, (iz3, ir2), J_r_m1_23.imag)
                cuda.atomic.add(j_r_m1.real, (iz3, ir3), J_r_m1_33.real)
                cuda.atomic.add(j_r_m1.imag, (iz3, ir3), J_r_m1_33.imag)
                # jt: Mode 0
                cuda.atomic.add(j_t_m0.real, (iz0, ir0), J_t_m0_00.real)
                cuda.atomic.add(j_t_m0.real, (iz0, ir1), J_t_m0_10.real)
                cuda.atomic.add(j_t_m0.real, (iz0, ir2), J_t_m0_20.real)
                cuda.atomic.add(j_t_m0.real, (iz0, ir3), J_t_m0_30.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir0), J_t_m0_01.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir1), J_t_m0_11.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir2), J_t_m0_21.real)
                cuda.atomic.add(j_t_m0.real, (iz1, ir3), J_t_m0_31.real)
                cuda.atomic.add(j_t_m0.real, (iz2, ir0), J_t_m0_02.real)
                cuda.atomic.add(j_t_m0.real, (iz2, ir1), J_t_m0_12.real)
                cuda.atomic.add(j_t_m0.real, (iz2, ir2), J_t_m0_22.real)
                cuda.atomic.add(j_t_m0.real, (iz2, ir3), J_t_m0_32.real)
                cuda.atomic.add(j_t_m0.real, (iz3, ir0), J_t_m0_03.real)
                cuda.atomic.add(j_t_m0.real, (iz3, ir1), J_t_m0_13.real)
                cuda.atomic.add(j_t_m0.real, (iz3, ir2), J_t_m0_23.real)
                cuda.atomic.add(j_t_m0.real, (iz3, ir3), J_t_m0_33.real)
                # jt: Mode 1
                cuda.atomic.add(j_t_m1.real, (iz0, ir0), J_t_m1_00.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir0), J_t_m1_00.imag)
                cuda.atomic.add(j_t_m1.real, (iz0, ir1), J_t_m1_10.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir1), J_t_m1_10.imag)
                cuda.atomic.add(j_t_m1.real, (iz0, ir2), J_t_m1_20.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir2), J_t_m1_20.imag)
                cuda.atomic.add(j_t_m1.real, (iz0, ir3), J_t_m1_30.real)
                cuda.atomic.add(j_t_m1.imag, (iz0, ir3), J_t_m1_30.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir0), J_t_m1_01.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir0), J_t_m1_01.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir1), J_t_m1_11.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir1), J_t_m1_11.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir2), J_t_m1_21.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir2), J_t_m1_21.imag)
                cuda.atomic.add(j_t_m1.real, (iz1, ir3), J_t_m1_31.real)
                cuda.atomic.add(j_t_m1.imag, (iz1, ir3), J_t_m1_31.imag)
                cuda.atomic.add(j_t_m1.real, (iz2, ir0), J_t_m1_02.real)
                cuda.atomic.add(j_t_m1.imag, (iz2, ir0), J_t_m1_02.imag)
                cuda.atomic.add(j_t_m1.real, (iz2, ir1), J_t_m1_12.real)
                cuda.atomic.add(j_t_m1.imag, (iz2, ir1), J_t_m1_12.imag)
                cuda.atomic.add(j_t_m1.real, (iz2, ir2), J_t_m1_22.real)
                cuda.atomic.add(j_t_m1.imag, (iz2, ir2), J_t_m1_22.imag)
                cuda.atomic.add(j_t_m1.real, (iz2, ir3), J_t_m1_32.real)
                cuda.atomic.add(j_t_m1.imag, (iz2, ir3), J_t_m1_32.imag)
                cuda.atomic.add(j_t_m1.real, (iz3, ir0), J_t_m1_03.real)
                cuda.atomic.add(j_t_m1.imag, (iz3, ir0), J_t_m1_03.imag)
                cuda.atomic.add(j_t_m1.real, (iz3, ir1), J_t_m1_13.real)
                cuda.atomic.add(j_t_m1.imag, (iz3, ir1), J_t_m1_13.imag)
                cuda.atomic.add(j_t_m1.real, (iz3, ir2), J_t_m1_23.real)
                cuda.atomic.add(j_t_m1.imag, (iz3, ir2), J_t_m1_23.imag)
                cuda.atomic.add(j_t_m1.real, (iz3, ir3), J_t_m1_33.real)
                cuda.atomic.add(j_t_m1.imag, (iz3, ir3), J_t_m1_33.imag)
                # jz: Mode 0
                cuda.atomic.add(j_z_m0.real, (iz0, ir0), J_z_m0_00.real)
                cuda.atomic.add(j_z_m0.real, (iz0, ir1), J_z_m0_10.real)
                cuda.atomic.add(j_z_m0.real, (iz0, ir2), J_z_m0_20.real)
                cuda.atomic.add(j_z_m0.real, (iz0, ir3), J_z_m0_30.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir0), J_z_m0_01.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir1), J_z_m0_11.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir2), J_z_m0_21.real)
                cuda.atomic.add(j_z_m0.real, (iz1, ir3), J_z_m0_31.real)
                cuda.atomic.add(j_z_m0.real, (iz2, ir0), J_z_m0_02.real)
                cuda.atomic.add(j_z_m0.real, (iz2, ir1), J_z_m0_12.real)
                cuda.atomic.add(j_z_m0.real, (iz2, ir2), J_z_m0_22.real)
                cuda.atomic.add(j_z_m0.real, (iz2, ir3), J_z_m0_32.real)
                cuda.atomic.add(j_z_m0.real, (iz3, ir0), J_z_m0_03.real)
                cuda.atomic.add(j_z_m0.real, (iz3, ir1), J_z_m0_13.real)
                cuda.atomic.add(j_z_m0.real, (iz3, ir2), J_z_m0_23.real)
                cuda.atomic.add(j_z_m0.real, (iz3, ir3), J_z_m0_33.real)
                # jz: Mode 1
                cuda.atomic.add(j_z_m1.real, (iz0, ir0), J_z_m1_00.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir0), J_z_m1_00.imag)
                cuda.atomic.add(j_z_m1.real, (iz0, ir1), J_z_m1_10.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir1), J_z_m1_10.imag)
                cuda.atomic.add(j_z_m1.real, (iz0, ir2), J_z_m1_20.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir2), J_z_m1_20.imag)
                cuda.atomic.add(j_z_m1.real, (iz0, ir3), J_z_m1_30.real)
                cuda.atomic.add(j_z_m1.imag, (iz0, ir3), J_z_m1_30.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir0), J_z_m1_01.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir0), J_z_m1_01.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir1), J_z_m1_11.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir1), J_z_m1_11.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir2), J_z_m1_21.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir2), J_z_m1_21.imag)
                cuda.atomic.add(j_z_m1.real, (iz1, ir3), J_z_m1_31.real)
                cuda.atomic.add(j_z_m1.imag, (iz1, ir3), J_z_m1_31.imag)
                cuda.atomic.add(j_z_m1.real, (iz2, ir0), J_z_m1_02.real)
                cuda.atomic.add(j_z_m1.imag, (iz2, ir0), J_z_m1_02.imag)
                cuda.atomic.add(j_z_m1.real, (iz2, ir1), J_z_m1_12.real)
                cuda.atomic.add(j_z_m1.imag, (iz2, ir1), J_z_m1_12.imag)
                cuda.atomic.add(j_z_m1.real, (iz2, ir2), J_z_m1_22.real)
                cuda.atomic.add(j_z_m1.imag, (iz2, ir2), J_z_m1_22.imag)
                cuda.atomic.add(j_z_m1.real, (iz2, ir3), J_z_m1_32.real)
                cuda.atomic.add(j_z_m1.imag, (iz2, ir3), J_z_m1_32.imag)
                cuda.atomic.add(j_z_m1.real, (iz3, ir0), J_z_m1_03.real)
                cuda.atomic.add(j_z_m1.imag, (iz3, ir0), J_z_m1_03.imag)
                cuda.atomic.add(j_z_m1.real, (iz3, ir1), J_z_m1_13.real)
                cuda.atomic.add(j_z_m1.imag, (iz3, ir1), J_z_m1_13.imag)
                cuda.atomic.add(j_z_m1.real, (iz3, ir2), J_z_m1_23.real)
                cuda.atomic.add(j_z_m1.imag, (iz3, ir2), J_z_m1_23.imag)
                cuda.atomic.add(j_z_m1.real, (iz3, ir3), J_z_m1_33.real)
                cuda.atomic.add(j_z_m1.imag, (iz3, ir3), J_z_m1_33.imag)

  #-----------------------------------------------------------------------------
  def init_cpu( self ):
    #...........................................................................
    @self.attr
    @njit_parallel
    def _cpu_linear(
        x, y, z, w, q,
        ux, uy, uz,
        gamma_minus_1,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        grid, Nm,
        nthreads, ptcl_chunk_indices):

        # Deposit the field per cell in parallel (for threads < number of cells)
        for i_thread in prange( nthreads ):

            # Allocate thread-local array
            jr_scal = np.zeros( Nm, dtype=np.complex128 )
            jt_scal = np.zeros( Nm, dtype=np.complex128 )
            jz_scal = np.zeros( Nm, dtype=np.complex128 )

            # Loop over all particles in thread chunk
            for i_ptcl in range( ptcl_chunk_indices[i_thread],
                                 ptcl_chunk_indices[i_thread+1] ):

                # Position
                xj = x[i_ptcl]
                yj = y[i_ptcl]
                zj = z[i_ptcl]
                # Velocity
                uxj = ux[i_ptcl]
                uyj = uy[i_ptcl]
                uzj = uz[i_ptcl]
                # Weights
                wj = q * w[i_ptcl] / ( 1. + gamma_minus_1[i_ptcl] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                # Calculate contribution from this particle to each mode
                jr_scal[0] = wj * (cos*uxj + sin*uyj)
                jt_scal[0] = wj * (cos*uyj - sin*uxj)
                jz_scal[0] = wj * uzj
                for m in range(1,Nm):
                    jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                    jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                    jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5
                # Index of the lowest cell of `global_array` that gets modified
                # by this particle (note: `global_array` has 2 guard cells)
                # (`min` function avoids out-of-bounds access at high r)
                ir_cell = min( int(math.floor(r_cell))+2, Nr+2 )
                iz_cell = int(math.floor( z_cell )) + 2

                # Add contribution of this particle to the global array
                for m in range(Nm):
                    grid[0, i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0) * jr_scal[m]
                    grid[0, i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1) * jr_scal[m]
                    grid[0, i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0) * jr_scal[m]
                    grid[0, i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1) * jr_scal[m]

                    grid[1, i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0) * jt_scal[m]
                    grid[1, i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1) * jt_scal[m]
                    grid[1, i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0) * jt_scal[m]
                    grid[1, i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1) * jt_scal[m]

                    grid[2, i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0) * jz_scal[m]
                    grid[2, i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1) * jz_scal[m]
                    grid[2, i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0) * jz_scal[m]
                    grid[2, i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1) * jz_scal[m]


        return

    #...........................................................................
    @self.attr
    @njit_parallel
    def _cpu_cubic(
        x, y, z, w, q,
        ux, uy, uz,
        gamma_minus_1,
        invdz, zmin, Nz,
        invdr, rmin, Nr,
        grid, Nm,
        nthreads, ptcl_chunk_indices):

        # Deposit the field per cell in parallel (for threads < number of cells)
        for i_thread in prange( nthreads ):

            # Allocate thread-local array
            jr_scal = np.zeros( Nm, dtype=np.complex128 )
            jt_scal = np.zeros( Nm, dtype=np.complex128 )
            jz_scal = np.zeros( Nm, dtype=np.complex128 )

            # Loop over all particles in thread chunk
            for i_ptcl in range( ptcl_chunk_indices[i_thread],
                                 ptcl_chunk_indices[i_thread+1] ):

                # Position
                xj = x[i_ptcl]
                yj = y[i_ptcl]
                zj = z[i_ptcl]
                # Velocity
                uxj = ux[i_ptcl]
                uyj = uy[i_ptcl]
                uzj = uz[i_ptcl]
                # Weights
                wj = q * w[i_ptcl] / ( 1. + gamma_minus_1[i_ptcl] )

                # Cylindrical conversion
                rj = math.sqrt(xj**2 + yj**2)
                # Avoid division by 0.
                if (rj != 0.):
                    invr = 1./rj
                    cos = xj*invr  # Cosine
                    sin = yj*invr  # Sine
                else:
                    cos = 1.
                    sin = 0.
                # Calculate contribution from this particle to each mode
                jr_scal[0] = wj * (cos*uxj + sin*uyj)
                jt_scal[0] = wj * (cos*uyj - sin*uxj)
                jz_scal[0] = wj * uzj
                for m in range(1,Nm):
                    jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                    jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                    jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

                # Positions of the particles, in the cell unit
                r_cell = invdr*(rj - rmin) - 0.5
                z_cell = invdz*(zj - zmin) - 0.5
                # Index of the lowest cell of `global_array` that gets modified
                # by this particle (note: `global_array` has 2 guard cells)
                # (`min` function avoids out-of-bounds access at high r)
                ir_cell = min( int(math.floor(r_cell))+1, Nr )
                iz_cell = int(math.floor( z_cell )) + 1

                # Add contribution of this particle to the global array
                for m in range(Nm):
                    grid[0, i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3)*jr_scal[m]

                    grid[0, i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3)*jr_scal[m]

                    grid[0, i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3)*jr_scal[m]

                    grid[0, i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2)*jr_scal[m]
                    grid[0, i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3)*jr_scal[m]

                    grid[1, i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3)*jt_scal[m]

                    grid[1, i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3)*jt_scal[m]

                    grid[1, i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3)*jt_scal[m]

                    grid[1, i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2)*jt_scal[m]
                    grid[1, i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3)*jt_scal[m]

                    grid[2, i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3)*jz_scal[m]

                    grid[2, i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3)*jz_scal[m]

                    grid[2, i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3)*jz_scal[m]

                    grid[2, i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2)*jz_scal[m]
                    grid[2, i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3)*jz_scal[m]

        return

  #-----------------------------------------------------------------------------
  def exec_cpu( self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    ux, uy, uz,
    gamma_minus_1,
    dz, zmin,
    dr, rmin,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    # (One copy per thread ; 2 guard cells on each side in z and r,
    # in order to store contributions from, at most, cubic shape factors ;
    # these deposition guard cells are folded into the regular box
    # inside `sum_reduce_2d_array`)
    threaded_shape = ( 3, nthreads, len(grid) ) + tuple( x + 4 for x in grid[0][0].shape )
    # allocate temporary array for parallel writes from each thread
    threaded_grid = tmp_ndarray( threaded_shape, dtype = grid[0][0].dtype )
    ndarray_fill.exec( threaded_grid, 0. )

    ptcl_chunk_indices = get_chunk_indices(x.shape[0], nthreads)

    if gamma_minus_1 is None:
      gamma_minus_1 = tmp_ndarray( shape = x.shape, dtype = x.dtype )
      ndarray_fill.exec( gamma_minus_1, 0.0 )

    # Deposit J using CPU threading
    if ptcl_shape == 'linear':
      self._cpu_linear(
        x, y, z, weight, coeff,
        ux, uy, uz,
        gamma_minus_1,
        1./dz, zmin, grid[0][0].shape[0],
        1./dr, rmin, grid[0][0].shape[1],
        threaded_grid, len(grid),
        nthreads, ptcl_chunk_indices )

    elif ptcl_shape == 'cubic':
      self._cpu_cubic(
        x, y, z, weight, coeff,
        ux, uy, uz,
        gamma_minus_1,
        1./dz, zmin, grid[0][0].shape[0],
        1./dr, rmin, grid[0][0].shape[1],
        threaded_grid, len(grid),
        nthreads, ptcl_chunk_indices )

    for m in range(len(grid)):
      sum_reduce_2d_array( threaded_grid[0], grid[m][0], m )
      sum_reduce_2d_array( threaded_grid[1], grid[m][1], m )
      sum_reduce_2d_array( threaded_grid[2], grid[m][2], m )


  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    ux, uy, uz,
    gamma_minus_1,
    dz, zmin,
    dr, rmin,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    # Define optimal number of CUDA threads per block for deposition
    # and gathering kernels (determined empirically)
    if ptcl_shape == "cubic":
      deposit_tpb = 32
    else:
      deposit_tpb = 16 if cuda_gpu_model == "V100" else 8

    dim_grid_2d_flat, dim_block_2d_flat = \
        cuda_tpb_bpg_1d(prefix_sum.shape[0], TPB=deposit_tpb)

    if gamma_minus_1 is None:
      gamma_minus_1 = tmp_numba_device_ndarray( shape = x.shape, dtype = x.dtype )
      ndarray_fill.exec( gamma_minus_1, 0.0 )

    # Deposit J in each of four directions
    if ptcl_shape == 'linear':
      if len(grid) == 2:
        self._cuda_linear[
          dim_grid_2d_flat, dim_block_2d_flat](
            x, y, z, weight, coeff,
            ux, uy, uz,
            gamma_minus_1,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[0][0], grid[1][0],
            grid[0][1], grid[1][1],
            grid[0][2], grid[1][2],
            cell_idx, prefix_sum )
      else:
        for m in range(len(grid)):
          self._cuda_linear_one_mode[
              dim_grid_2d_flat, dim_block_2d_flat](
              x, y, z, weight, coeff,
              ux, uy, uz,
              gamma_minus_1,
              1./dz, zmin, grid[0][0].shape[0],
              1./dr, rmin, grid[0][0].shape[1],
              grid[m][0], grid[m][1], grid[m][2], m,
              cell_idx, prefix_sum )

    elif ptcl_shape == 'cubic':
      if len(grid) == 2:
        self._cuda_cubic[
          dim_grid_2d_flat, dim_block_2d_flat](
            x, y, z, weight, coeff,
            ux, uy, uz,
            gamma_minus_1,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[0][0], grid[1][0],
            grid[0][1], grid[1][1],
            grid[0][2], grid[1][2],
            cell_idx, prefix_sum )
      else:
        for m in range(len(grid)):
          self._cuda_cubic_one_mode[
            dim_grid_2d_flat, dim_block_2d_flat](
              x, y, z, weight, coeff,
              ux, uy, uz,
              gamma_minus_1,
              1./dz, zmin, grid[0][0].shape[0],
              1./dr, rmin, grid[0][0].shape[1],
              grid[m][0], grid[m][1], grid[m][2], m,
              cell_idx, prefix_sum )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
deposit_moment_nv = DepositMomentNV()
