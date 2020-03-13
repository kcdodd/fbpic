from fbpic.discrete import (
  ArrayOp,
  tmp_ndarray,
  tmp_numba_device_ndarray,
  ndarray_fill )

import math
import numpy as np

from fbpic.utils.threading import (
  nthreads,
  njit_parallel,
  prange,
  get_chunk_indices )

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SolvePotential ( ArrayOp ):
  """Solves potential defined as div(grad(phi)) = div(E)
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    phi,
    Ep,
    Em,
    Ez,
    inv_k2,
    kr,
    kz,
    gpu = False ):

    super().exec(
      phi = phi,
      Ep = Ep,
      Em = Em,
      Ez = Ez,
      inv_k2 = inv_k2,
      kr = kr,
      kz = kz,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):
    from numba import cuda

    @self.attr
    @cuda.jit
    def _gpu( phi, Ep, Em, Ez, inv_k2, kr, kz ):

      i = cuda.grid(1)

      if i < phi.shape[0]:
        divE = kr[i]*( Ep[i] - Em[i] ) + 1.j*kz[i]*Ez[i]

        if inv_k2[i] == 0.0:
          phi[i] = 0.0
        else:
          phi[i] = divE * inv_k2[i]

  #-----------------------------------------------------------------------------
  def init_cpu( self ):

    @self.attr
    @njit_parallel
    def _cpu( phi, Ez, kz, np, nt, nf ):

      for i in prange( np ):
        offset = i*nt

        for j in range(nt):
          divE = kr[offset + j]*( Ep[offset + j] - Em[offset + j] ) + 1.j*kz[offset + j]*Ez[offset + j]

          if inv_k2[offset + j] == 0.0:
            phi[offset + j] = 0.0
          else:
            phi[offset + j] = divE * inv_k2[offset + j]


      offset = np * nt

      for j in range(nf):
        divE = kr[offset + j]*( Ep[offset + j] - Em[offset + j] ) + 1.j*kz[offset + j]*Ez[offset + j]

        if inv_k2[offset + j] == 0.0:
          phi[offset + j] = 0.0
        else:
          phi[offset + j] = divE * inv_k2[offset + j]

  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self,
    phi,
    Ep,
    Em,
    Ez,
    inv_k2,
    kr,
    kz ):

    if len(phi.shape) > 1:
      phi = phi.ravel()
      Ep = Ep.ravel()
      Em = Em.ravel()
      Ez = Ez.ravel()
      inv_k2 = inv_k2.ravel()
      kr = kr.ravel()
      kz = kz.ravel()

    bpg, tpb = cuda_tpb_bpg_1d( phi.shape[0] )

    self._gpu[bpg, tpb](
      phi,
      Ep,
      Em,
      Ez,
      inv_k2,
      kr,
      kz )

  #-----------------------------------------------------------------------------
  def exec_cpu( self,
    phi,
    Ep,
    Em,
    Ez,
    inv_k2,
    kr,
    kz ):

    if len(phi.shape) > 1:
      phi = phi.ravel()
      Ep = Ep.ravel()
      Em = Em.ravel()
      Ez = Ez.ravel()
      inv_k2 = inv_k2.ravel()
      kr = kr.ravel()
      kz = kz.ravel()

    nt = phi.shape[0] // nthreads
    nf = phi.shape[0] % nthreads

    self._cpu(
        phi,
        Ep,
        Em,
        Ez,
        inv_k2,
        kr,
        kz,
        nthreads,
        nt,
        nf )

solve_potential = SolvePotential()
