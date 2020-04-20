

from .gathering.threading_methods import gather_field_numba_linear, \
        gather_field_numba_cubic
from .gathering.threading_methods_one_mode import erase_eb_numba, \
    gather_field_numba_linear_one_mode, gather_field_numba_cubic_one_mode

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from .gathering.cuda_methods import (
    gather_field_gpu_linear,
    gather_field_gpu_cubic )

  from .gathering.cuda_methods_one_mode import (
    gather_field_gpu_linear_one_mode,
    gather_field_gpu_cubic_one_mode )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GatherVector ( ArrayOp ):
  """Gather a vector field onto particles
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    vector,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin, rmax,
    ptcl_shape,
    gpu = False ):
    """
    Parameters
    ----------
    vx : array
      particle vector x value ( gathered from interpolation grid )
    vy : array
    vz : array
    x : array
      particle positionsvx, vy, vz,
    y : array
    z : array
    grid : array<complex>(nm, 3, nz, nr)
      input grid of vector value

      interpolation grid for azimuthal modes 0, 1, ..., nm-1.
    dz : float
      z cell size
    zmin : float
    dr : float
      radial cell size
    rmin : float
    rmax : float
    ptcl_shape : str
      shape of particles to use for deposition {'linear', 'cubic'}
    gpu : bool
      Use gpu implementation (if available) even when ndarrays are passed in as arguments
    """

    super().exec(
      vector = vector,
      x = x, y = y, z = z,
      grid = grid,
      dz = dz, zmin = zmin,
      dr = dr, rmin = rmin, rmax = rmax,
      ptcl_shape = ptcl_shape,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def exec_cpu( self,
    vector,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin, rmax,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    ptcl_chunk_indices = get_chunk_indices(x.shape[0], nthreads)

    # Deposit J using CPU threading
    if ptcl_shape == 'linear':
      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_field_numba_linear(
          x, y, z,
          rmax,
          1./dz, zmin, grid[0][0].shape[0],
          1./dr, rmin, grid[0][0].shape[1],
          grid[0][0], grid[0][1], grid[0][2],
          grid[1][0], grid[1][1], grid[1][2],
          vector[0], vector[1], vectpr[2] )
      else:
        # Generic version for arbitrary number of modes
        for m in range(len(grid)):
          gather_field_numba_linear_one_mode(
            x, y, z,
            rmax,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[m][0], grid[m][1], grid[m][2], m,
            vector[0], vector[1], vectpr[2] )

    elif ptcl_shape == 'cubic':

      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_field_numba_cubic(
          x, y, z,
          rmax,
          1./dz, zmin, grid[0][0].shape[0],
          1./dr, rmin, grid[0][0].shape[1],
          grid[0][0], grid[0][1], grid[0][2],
          grid[1][0], grid[1][1], grid[1][2],
          vector[0], vector[1], vectpr[2],
          nthreads, ptcl_chunk_indices )
      else:
        # Generic version for arbitrary number of modes
        for m in range(len(grid)):
          gather_field_numba_cubic_one_mode(
            x, y, z,
            rmax,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[m][0], grid[m][1], grid[m][2], m,
            vector[0], vector[1], vectpr[2],
            nthreads, ptcl_chunk_indices )


  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self,
    vector,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin, rmax,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    # Define optimal number of CUDA threads per block for deposition
    # and gathering kernels (determined empirically)
    if ptcl_shape == "cubic":
      deposit_tpb = 32
    else:
      deposit_tpb = 16 if cuda_gpu_model == "V100" else 8

    dim_grid_2d_flat, dim_block_2d_flat = \
        cuda_tpb_bpg_1d( x.shape[0], TPB=deposit_tpb )

    # Deposit J in each of four directions
    if ptcl_shape == 'linear':
      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_field_gpu_linear[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[0][0], grid[0][1], grid[0][2],
            grid[1][0], grid[1][1], grid[1][2],
            vector[0], vector[1], vectpr[2] )
      else:
        # Generic version for arbitrary number of modes
        for m in range(Nm):
          gather_field_gpu_linear_one_mode[
            dim_grid_1d, dim_block_1d](
              x, y, z,
              rmax,
              1./dz, zmin, grid[0][0].shape[0],
              1./dr, rmin, grid[0][0].shape[1],
              grid[m][0], grid[m][1], grid[m][2], m,
              vector[0], vector[1], vectpr[2] )

    elif ptcl_shape == 'cubic':
      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_field_gpu_cubic[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[0][0], grid[0][1], grid[0][2],
            grid[1][0], grid[1][1], grid[1][2],
            vector[0], vector[1], vectpr[2] )
    else:
      # Generic version for arbitrary number of modes
      for m in range(len(grid)):
        gather_field_gpu_cubic_one_mode[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0][0].shape[0],
            1./dr, rmin, grid[0][0].shape[1],
            grid[m][0], grid[m][1], grid[m][2], m,
            vector[0], vector[1], vectpr[2] )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
gather_vector = GatherVector()
