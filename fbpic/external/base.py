
from fbpic.discrete import ArrayOp

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ExternalFrameField ( ArrayOp ):
  """External fields specified in the solution reference frame
  """

  def exec( self,
    x, y, z, t,
    Ex, Ey, Ez,
    Bx, By, Bz ):
    """

    Parameters
    ----------
    x : array
      cartesian coordinates to evaluate field
    y : array
    z : array
    t : float
      simulation time of the evaluation
    Ex : array
      cartesian electric field to add to at each position
    Ey : array
    Ez : array
    Bx : array
      cartesian magnetic field to add to at each position
    By : array
    Bz : array
    """

    super().exec(
      x = x, y = y, z = z, t = t,
      Ex = Ex, Ey = Ey, Ez = Ez,
      Bx = Bx, By = By, Bz = Bz )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ExternalSymmetricFrameField ( ArrayOp ):
  """External axially symmetric fields specified in the solution reference frame
  """

  def exec( self,
    r, z, t,
    Er, Et, Ez,
    Br, Bt, Bz ):
    """

    Parameters
    ----------
    r : array
      cylindrical coordinates to evaluate field
    z : array
    t : float
      simulation time of the evaluation
    Er : array
      cylindrical electric field to add to at each position
    Et : array
    Ez : array
    Br : array
      cylindrical magnetic field to add to at each position
    Bt : array
    Bz : array
    """

    super().exec(
      r = r, z = z, t = t,
      Er = Er, Et = Et, Ez = Ez,
      Br = Br, Bt = Bt, Bz = Bz )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ExternalFrameCharge ( ArrayOp ):
  """External axially symmetric source specified in the solution reference frame
  """

  def exec( self,
    r, z, t, dt,
    Er, Et, Ez, phi,
    Br, Bt, Bz,
    rho,
    ext_rho ):
    """

    Parameters
    ----------
    r : array
      cylindrical coordinates to evaluate source
    z : array
    t : float
      simulation time of the evaluation
    dt : float
    Er : array
      cylindrical electric field at each position
    Et : array
    Ez : array
    phi : array
      electric potential at each position
    Br : array
      cylindrical magnetic field at each position
    Bt : array
    Bz : array
    rho : array
      charge density to add to at each position
    ext_rho : rho
      external charge density, may be used as working state (zero at begining of sim).
    """

    super().exec(
      r = r, z = z, t = t, dt = dt,
      Er = Er, Et = Et, Ez = Ez, phi = phi,
      Br = Br, Bt = Bt, Bz = Bz,
      rho = rho,
      ext_rho = ext_rho )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ExternalFrameCurrent ( ArrayOp ):
  """External axially symmetric source specified in the solution reference frame
  """

  def exec( self,
    r, z, t,
    Er, Et, Ez, phi,
    Br, Bt, Bz,
    Jr, Jt, Jz ):
    """

    Parameters
    ----------
    r : array
      cylindrical coordinates to evaluate source
    z : array
    t : float
      simulation time of the evaluation
    Er : array
      cylindrical electric field at each position
    Et : array
    Ez : array
    phi : array
      electric potential at each position
    Br : array
      cylindrical magnetic field at each position
    Bt : array
    Bz : array
    Jr : array
      cylindrical current density to add to at each position
    Jt : array
    Jz : array
    """

    super().exec(
      r = r, z = z, t = t,
      Er = Er, Et = Et, Ez = Ez, phi = phi,
      Br = Br, Bt = Bt, Bz = Bz,
      Jr = Jr, Jt = Jt, Jz = Jz )
