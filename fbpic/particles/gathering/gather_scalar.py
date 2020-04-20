
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GatherScalar ( ArrayOp ):
  """Gather a scalar field onto particles
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    s,
    x, y, z,
    grid,
    dz, zmin,
    dr, rmin,
    ptcl_shape,
    gpu = False ):
    """
    Parameters
    ----------
    s : array
      output particle scalar value ( gathered from interpolation grid )
    x : array
      particle positions
    y : array
    z : array
    grid : array<complex>(nm, nz, nr)
      input grid of scalar value

      interpolation grid for azimuthal modes 0, 1, ..., nm-1.
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
      gamma_minus_1 = gamma_minus_1,
      dz = dz, zmin = zmin,
      dr = dr, rmin = rmin,
      ptcl_shape = ptcl_shape,
      gpu = gpu )
