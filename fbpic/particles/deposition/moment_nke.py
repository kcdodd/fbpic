
from fbpic.discete import ArrayOp

import math
from scipy.constants import c
import numpy as np

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model
    from .cuda_methods import (
      r_shape_linear,
      z_shape_linear )

class DepositMomentNKE ( ArrayOp ):
  """Deposit trace of 2nd distribution moment

  nke = density * ( gamma * v * c )^2 (aka (pc)^2/m0^2 )

  density = 1/cell

  multiply by cell/m^3 for physical density

  to get kenetic energy density, multiply by m0^2 (m0 = particle rest mass)
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    ux, uy, uz, gamma,
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
    ux : array
      particle unitless momenta (gamma * v / c)
    uy : array
    uz : array
    gamma : array
      relativistic gammae
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
      grid,
      coeff,
      weight,
      cell_idx,
      prefix_sum,
      x, y, z,
      ux, uy, uz, gamma,
      dz, zmin, dr, rmin )

  #-----------------------------------------------------------------------------
  def init_gpu( self ):

    self._deposit_tpb = 16 if cuda_gpu_model == "V100" else 8

    @cuda.jit
    def _gpu_linear_one_mode(
      x, y, z, w, coeff,
      ux, uy, uz, gamma,
      invdz, zmin, Nz,
      invdr, rmin, Nr,
      grid, m,
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

      Parameters
      ----------
      x, y, z : 1darray of floats (in meters)
          The position of the particles

      w : 1d array of floats
          The weights of the particles

      coeff : float

      ux, uy, uz : 1darray of floats
          The unitless momentum of the particles. u is gamma*v/c

      gamma : 1darray of floats
          The inverse of the relativistic gamma factor

      grid: 2darrays of complexs
          output grid of ( gamma * v * c )^2 (aka (pc)^2/m0^2 )

          interpolation grid for mode m.
          (is modified by this function)

      m: int
          The index of the azimuthal mode considered

      invdz, invdr : float (in meters^-1)
          Inverse of the grid step along the considered direction

      zmin, rmin : float (in meters)
          Position of the edge of the simulation box,
          along the direction considered

      Nz, Nr : int
          Number of gridpoints along the considered direction

      cell_idx : 1darray of integers
          The cell index of the particle

      prefix_sum : 1darray of integers
          Represents the cumulative sum of
          the particles per cell
      """
      c4 = c**4

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
        v2_m_00 = 0. + 0.j
        v2_m_01 = 0. + 0.j
        v2_m_10 = 0. + 0.j
        v2_m_11 = 0. + 0.j

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
          # Inverse gamma
          gammaj = gamma[ptcl_idx]
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

          v2_m_scal = wj * c4 * gammaj**2 * ( uxj**2 + uyj**2 + uzj**2 )

          v2_m_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * v2_m_scal
          v2_m_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * v2_m_scal
          v2_m_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * v2_m_scal
          v2_m_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * v2_m_scal

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
          cuda.atomic.add(grid.real, (iz0, ir0), v2_m_00.real)
          cuda.atomic.add(grid.real, (iz0, ir1), v2_m_10.real)
          cuda.atomic.add(grid.real, (iz1, ir0), v2_m_01.real)
          cuda.atomic.add(grid.real, (iz1, ir1), v2_m_11.real)
          if m > 0:
            # For azimuthal modes beyond m=0: add imaginary part
            cuda.atomic.add(grid.imag, (iz0, ir0), v2_m_00.imag)
            cuda.atomic.add(grid.imag, (iz0, ir1), v2_m_10.imag)
            cuda.atomic.add(grid.imag, (iz1, ir0), v2_m_01.imag)
            cuda.atomic.add(grid.imag, (iz1, ir1), v2_m_11.imag)


    @cuda.jit
    def _gpu_linear(
      x, y, z, w, coeff,
      ux, uy, uz, gamma,
      invdz, zmin, Nz,
      invdr, rmin, Nr,
      grid_m0, grid_m1,
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

      Parameters
      ----------
      x, y, z : 1darray of floats (in meters)
          The position of the particles

      w : 1d array of floats
          The weights of the particles
          (For ionizable atoms: weight times the ionization level)

      coeff : float

      ux, uy, uz : 1darray of floats
          The unitless momentum of the particles. u is gamma*v/c

      gamma : 1darray of floats
          The inverse of the relativistic gamma factor

      grid_m0, grid_m1,: 2darrays of complexs
          output grid of ( gamma * v * c )^2 (aka (pc)^2/m0^2 )

          interpolation grid for mode 0 and 1.
          (is modified by this function)

      invdz, invdr : float (in meters^-1)
          Inverse of the grid step along the considered direction

      zmin, rmin : float (in meters)
          Position of the edge of the simulation box,
          along the direction considered

      Nz, Nr : int
          Number of gridpoints along the considered direction

      cell_idx : 1darray of integers
          The cell index of the particle

      prefix_sum : 1darray of integers
          Represents the cumulative sum of
          the particles per cell
      """

      c4 = c**4

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

        v2_m0_00 = 0.
        v2_m0_01 = 0.
        v2_m0_10 = 0.
        v2_m0_11 = 0.
        v2_m1_00 = 0. + 0.j
        v2_m1_01 = 0. + 0.j
        v2_m1_10 = 0. + 0.j
        v2_m1_11 = 0. + 0.j

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
          # Inverse gamma
          gammaj = gamma[ptcl_idx]
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
          v2 = wj * c4 * gammaj**2 * ( uxj**2 + uyj**2 + uzj**2 )
          # Mode 0
          v2_m0_scal = v2 * exptheta_m0
          # Mode 1
          v2_m1_scal = v2 * exptheta_m1

          v2_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * v2_m0_scal
          v2_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * v2_m0_scal
          v2_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * v2_m1_scal
          v2_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * v2_m1_scal
          v2_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * v2_m0_scal
          v2_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * v2_m0_scal
          v2_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * v2_m1_scal
          v2_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * v2_m1_scal

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
          cuda.atomic.add(grid_m0.real, (iz0, ir0), v2_m0_00.real)
          cuda.atomic.add(grid_m0.real, (iz0, ir1), v2_m0_10.real)
          cuda.atomic.add(grid_m0.real, (iz1, ir0), v2_m0_01.real)
          cuda.atomic.add(grid_m0.real, (iz1, ir1), v2_m0_11.real)
          # Mode 1
          cuda.atomic.add(grid_m1.real, (iz0, ir0), v2_m1_00.real)
          cuda.atomic.add(grid_m1.imag, (iz0, ir0), v2_m1_00.imag)
          cuda.atomic.add(grid_m1.real, (iz0, ir1), v2_m1_10.real)
          cuda.atomic.add(grid_m1.imag, (iz0, ir1), v2_m1_10.imag)
          cuda.atomic.add(grid_m1.real, (iz1, ir0), v2_m1_01.real)
          cuda.atomic.add(grid_m1.imag, (iz1, ir0), v2_m1_01.imag)
          cuda.atomic.add(grid_m1.real, (iz1, ir1), v2_m1_11.real)
          cuda.atomic.add(grid_m1.imag, (iz1, ir1), v2_m1_11.imag)


    self._gpu_linear_one_mode = _gpu_linear_one_mode
    self._gpu_linear = _gpu_linear

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
  def exec_gpu( self,
    grid,
    coeff,
    weight,
    cell_idx,
    prefix_sum,
    x, y, z,
    ux, uy, uz, gamma,
    dz, zmin, dr, rmin,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    Nm = grid.shape[0]

    # Get the threads per block and the blocks per grid
    dim_grid_2d_flat, dim_block_2d_flat = \
      cuda_tpb_bpg_1d(prefix_sum.shape[0], TPB=self._deposit_tpb)

    if ptcl_shape == "linear":
      if Nm == 2:
        self._gpu_linear[
          dim_grid_2d_flat, dim_block_2d_flat](
            x, y, z, weight, coeff,
            ux, uy, uz, gamma,
            1./dz, zmin, grid.shape[-2],
            1./dr, rmin, grid.shape[-1],
            grid[0], grid[1],
            cell_idx, prefix_sum )
            
      else:
        for m in range(Nm):
          self._gpu_linear_one_mode[
            dim_grid_2d_flat, dim_block_2d_flat](
              x, y, z, weight, coeff,
              ux, uy, uz, gamma,
              1./dz, zmin, grid.shape[-2],
              1./dr, rmin, grid.shape[-1],
              grid[m], m,
              cell_idx, prefix_sum)

    elif ptcl_shape == "cubic":
      raise NotImplementedError()

deposit_moment_nke = DepositMomentNKE()
