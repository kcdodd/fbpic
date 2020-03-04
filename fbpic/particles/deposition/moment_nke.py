# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with threading.
"""
from fbpic.discete import (
  ArrayOp,
  tmp_ndarray )

import math
from scipy.constants import c
import numpy as np

from fbpic.utils.threading import nthreads, get_chunk_indices
from .threading_methods import (
  Sz_linear,
  Sr_linear,
  Sz_cubic,
  Sr_cubic )


from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model
  from .cuda_methods import (
    r_shape_linear,
    z_shape_linear,
    z_shape_cubic,
    r_shape_cubic )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DepositMomentNKE ( ArrayOp ):
  """Deposit unitless kinetic energy moment

  nke = density * ( gamma - 1 )

  density = 1/cell

  multiply by cell/m^3 for physical density

  to get kenetic energy density, m0 * c^2 * nke (m0 = particle rest mass)
  to get total energy density, m0 * c^2 * ( nke + n )
  to get internal thermal energy, m0 * c^2 * ( sqrt( (nke + n)^2 - nu^2 ) - n )
  see also DepositMomentN, DepositMomentNU
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    gammam1,
    dz, zmin, dr, rmin,
    ptcl_shape,
    gpu = False ):
    """
    Parameters
    ----------
    grid : array<complex>(nm, nz, nr)
      output grid of density * ( gamma * v * c )^2 (aka (pc)^2/m0^2 )

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
    gammam1 : array
      gamma-1, kinetic part of relativistic Lorentz factor
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
      gammam1 = gammam1,
      dz = dz, zmin = zmin, dr = dr, rmin = rmin,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):

    @self.attr
    @cuda.jit
    def _cuda_linear_one_mode(
      x, y, z, w, coeff,
      gammam1,
      invdz, zmin, Nz,
      invdr, rmin, Nr,
      grid, m,
      cell_idx, prefix_sum):

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
        ke_m_00 = 0. + 0.j
        ke_m_01 = 0. + 0.j
        ke_m_10 = 0. + 0.j
        ke_m_11 = 0. + 0.j

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
          # gamma
          gammam1j = gammam1[ptcl_idx]
          # Weights
          wj = coeff * w[ptcl_idx]

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

          # Calculate the trace of the second moment
          # ----------------------

          ke_m_scal = wj * gammam1

          ke_m_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * ke_m_scal
          ke_m_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * ke_m_scal
          ke_m_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * ke_m_scal
          ke_m_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * ke_m_scal

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
          cuda.atomic.add(grid.real, (iz0, ir0), ke_m_00.real)
          cuda.atomic.add(grid.real, (iz0, ir1), ke_m_10.real)
          cuda.atomic.add(grid.real, (iz1, ir0), ke_m_01.real)
          cuda.atomic.add(grid.real, (iz1, ir1), ke_m_11.real)
          if m > 0:
            # For azimuthal modes beyond m=0: add imaginary part
            cuda.atomic.add(grid.imag, (iz0, ir0), ke_m_00.imag)
            cuda.atomic.add(grid.imag, (iz0, ir1), ke_m_10.imag)
            cuda.atomic.add(grid.imag, (iz1, ir0), ke_m_01.imag)
            cuda.atomic.add(grid.imag, (iz1, ir1), ke_m_11.imag)

    @self.attr
    @cuda.jit
    def _cuda_linear(
      x, y, z, w, coeff,
      gammam1,
      invdz, zmin, Nz,
      invdr, rmin, Nr,
      grid_m0, grid_m1,
      cell_idx, prefix_sum):

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

        ke_m0_00 = 0.
        ke_m0_01 = 0.
        ke_m0_10 = 0.
        ke_m0_11 = 0.
        ke_m1_00 = 0. + 0.j
        ke_m1_01 = 0. + 0.j
        ke_m1_10 = 0. + 0.j
        ke_m1_11 = 0. + 0.j

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
          # gamma
          gammam1j = gammam1[ptcl_idx]
          # Weights
          wj = coeff * w[ptcl_idx]

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
          ke = wj * gammam1j
          # Mode 0
          ke_m0_scal = ke * exptheta_m0
          # Mode 1
          ke_m1_scal = ke * exptheta_m1

          ke_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * ke_m0_scal
          ke_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * ke_m0_scal
          ke_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * ke_m1_scal
          ke_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * ke_m1_scal
          ke_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * ke_m0_scal
          ke_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * ke_m0_scal
          ke_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * ke_m1_scal
          ke_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * ke_m1_scal

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

          # Mode 0
          cuda.atomic.add(grid_m0.real, (iz0, ir0), ke_m0_00.real)
          cuda.atomic.add(grid_m0.real, (iz0, ir1), ke_m0_10.real)
          cuda.atomic.add(grid_m0.real, (iz1, ir0), ke_m0_01.real)
          cuda.atomic.add(grid_m0.real, (iz1, ir1), ke_m0_11.real)
          # Mode 1
          cuda.atomic.add(grid_m1.real, (iz0, ir0), ke_m1_00.real)
          cuda.atomic.add(grid_m1.imag, (iz0, ir0), ke_m1_00.imag)
          cuda.atomic.add(grid_m1.real, (iz0, ir1), ke_m1_10.real)
          cuda.atomic.add(grid_m1.imag, (iz0, ir1), ke_m1_10.imag)
          cuda.atomic.add(grid_m1.real, (iz1, ir0), ke_m1_01.real)
          cuda.atomic.add(grid_m1.imag, (iz1, ir0), ke_m1_01.imag)
          cuda.atomic.add(grid_m1.real, (iz1, ir1), ke_m1_11.real)
          cuda.atomic.add(grid_m1.imag, (iz1, ir1), ke_m1_11.imag)


  # #-----------------------------------------------------------------------------
  # def init_cpu( self ):
  #
  #   from fbpic.utils.threading import njit_parallel, prange
  #
  #
  # #-----------------------------------------------------------------------------
  # def exec_cpu( self, x, y, z, weight ):
  #   pass

  #-----------------------------------------------------------------------------
  def exec_numba_cuda( self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    gammam1,
    dz, zmin, dr, rmin,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    # Define optimal number of CUDA threads per block for deposition
    # and gathering kernels (determined empirically)
    if ptcl_shape == "cubic":
      deposit_tpb = 32
    else:
      deposit_tpb = 16 if cuda_gpu_model == "V100" else 8

    # Get the threads per block and the blocks per grid
    dim_grid_2d_flat, dim_block_2d_flat = \
      cuda_tpb_bpg_1d(prefix_sum.shape[0], TPB=deposit_tpb)

    if ptcl_shape == "linear":
      if grid.shape[0] == 2:
        self._cuda_linear[
          dim_grid_2d_flat, dim_block_2d_flat](
            x, y, z, weight, coeff,
            gammam1,
            1./dz, zmin, grid.shape[-2],
            1./dr, rmin, grid.shape[-1],
            grid[0], grid[1],
            cell_idx, prefix_sum )

      else:
        for m in range(grid.shape[0]):
          self._cuda_linear_one_mode[
            dim_grid_2d_flat, dim_block_2d_flat](
              x, y, z, weight, coeff,
              gammam1,
              1./dz, zmin, grid.shape[-2],
              1./dr, rmin, grid.shape[-1],
              grid[m], m,
              cell_idx, prefix_sum)

    elif ptcl_shape == "cubic":
      raise NotImplementedError()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
deposit_moment_nke = DepositMomentNKE()
