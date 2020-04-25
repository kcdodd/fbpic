
from fbpic.discrete import (
  ArrayOp )

from .threading_methods import (
  gather_scalar_numba_linear,
  gather_scalar_numba_cubic )

from .threading_methods_one_mode import (
    gather_scalar_numba_linear_one_mode,
    gather_scalar_numba_cubic_one_mode )

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model

  from .cuda_methods import (
    gather_scalar_gpu_linear,
    gather_scalar_gpu_cubic )

  from .cuda_methods_one_mode import (
    gather_scalar_gpu_linear_one_mode,
    gather_scalar_gpu_cubic_one_mode )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GatherScalar ( ArrayOp ):
  """Gather a scalar field onto particles
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    scalar,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin, rmax,
    ptcl_shape,
    gpu = False ):
    """
    Parameters
    ----------
    scalar : array(nptcl)
      gathered scalar value
    x : array
      particle positionsvx, vy, vz,
    y : array
    z : array
    grid : array<complex>(nm nz, nr)
      input grid of scalar value

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
      scalar = scalar,
      x = x, y = y, z = z,
      grid = grid,
      dz = dz, zmin = zmin,
      dr = dr, rmin = rmin, rmax = rmax,
      ptcl_shape = ptcl_shape,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def exec_cpu( self,
    scalar,
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
        gather_scalar_numba_linear(
          x, y, z,
          rmax,
          1./dz, zmin, grid[0].shape[0],
          1./dr, rmin, grid[0].shape[1],
          grid[0],
          grid[1],
          scalar )
      else:
        # Generic version for arbitrary number of modes
        for m in range(len(grid)):
          gather_scalar_numba_linear_one_mode(
            x, y, z,
            rmax,
            1./dz, zmin, grid[0].shape[0],
            1./dr, rmin, grid[0].shape[1],
            grid[m], m,
            scalar )

    elif ptcl_shape == 'cubic':

      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_scalar_numba_cubic(
          x, y, z,
          rmax,
          1./dz, zmin, grid[0].shape[0],
          1./dr, rmin, grid[0].shape[1],
          grid[0],
          grid[1],
          scalar,
          nthreads, ptcl_chunk_indices )
      else:
        # Generic version for arbitrary number of modes
        for m in range(len(grid)):
          gather_scalar_numba_cubic_one_mode(
            x, y, z,
            rmax,
            1./dz, zmin, grid[0].shape[0],
            1./dr, rmin, grid[0].shape[1],
            grid[m], m,
            scalar,
            nthreads, ptcl_chunk_indices )


  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self,
    scalar,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin, rmax,
    ptcl_shape ):

    assert ptcl_shape in ['linear', 'cubic']

    # Define optimal number of CUDA threads per block for deposition
    # and gathering kernels (determined empirically)
    if ptcl_shape == "cubic":
      gather_tpb = 256
    else:
      gather_tpb = 128

    dim_grid_1d, dim_block_1d = \
        cuda_tpb_bpg_1d( x.shape[0], TPB=gather_tpb )

    # Deposit J in each of four directions
    if ptcl_shape == 'linear':
      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_scalar_gpu_linear[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0].shape[0],
            1./dr, rmin, grid[0].shape[1],
            grid[0],
            grid[1],
            scalar )
      else:
        # Generic version for arbitrary number of modes
        for m in range(Nm):
          gather_scalar_gpu_linear_one_mode[
            dim_grid_1d, dim_block_1d](
              x, y, z,
              rmax,
              1./dz, zmin, grid[0].shape[0],
              1./dr, rmin, grid[0].shape[1],
              grid[m], m,
              scalar )

    elif ptcl_shape == 'cubic':
      if len(grid) == 2:
        # Optimized version for 2 modes
        gather_scalar_gpu_cubic[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0].shape[0],
            1./dr, rmin, grid[0].shape[1],
            grid[0],
            grid[1],
            scalar )
    else:
      # Generic version for arbitrary number of modes
      for m in range(len(grid)):
        gather_scalar_gpu_cubic_one_mode[
          dim_grid_1d, dim_block_1d](
            x, y, z,
            rmax,
            1./dz, zmin, grid[0].shape[0],
            1./dr, rmin, grid[0].shape[1],
            grid[m], m,
            scalar )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
gather_scalar = GatherScalar()
