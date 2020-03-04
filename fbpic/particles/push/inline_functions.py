# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the functions for the particle pusher
"""
import math

def push_p_vay( ux_i, uy_i, uz_i, gammam1_i,
    Ex, Ey, Ez, Bx, By, Bz, econst, bconst ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # Get the magnetic rotation vector
    taux = bconst*Bx
    tauy = bconst*By
    tauz = bconst*Bz
    tau2 = taux**2 + tauy**2 + tauz**2

    inv_gamma_i = 1. / ( gammam1_i + 1 )

    # Get the momenta at the half timestep
    uxp = ux_i + econst*Ex \
    + inv_gamma_i*( uy_i*tauz - uz_i*tauy )
    uyp = uy_i + econst*Ey \
    + inv_gamma_i*( uz_i*taux - ux_i*tauz )
    uzp = uz_i + econst*Ez \
    + inv_gamma_i*( ux_i*tauy - uy_i*taux )

    up2 = uxp**2 + uyp**2 + uzp**2
    sigma = 1 + up2 - tau2
    utau = uxp*taux + uyp*tauy + uzp*tauz

    # Get the new gamma - 1
    if ux_i**2 + uy_i**2 + uz_i**2 < 1e-6:
      # use series expansion for low velocities (u < 0.001)
      # approximation within roundoff of 'exact' expression, and avoids
      # finite precision errors of subtraction operation to store kinetic part
      # of lorentz factor
      gammam1_f = 0.5 * ( up2 + utau**2 )
    else:
      gammam1_f = math.sqrt( 0.5 * ( sigma + math.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) ) ) - 1

    inv_gamma_f = 1. / ( gammam1_f + 1 )

    # Reuse the tau and utau arrays to save memory
    tx = inv_gamma_f*taux
    ty = inv_gamma_f*tauy
    tz = inv_gamma_f*tauz
    ut = inv_gamma_f*utau
    s = 1./( 1 + tau2*inv_gamma_f**2 )

    # Get the new u
    ux_f = s*( uxp + tx*ut + uyp*tz - uzp*ty )
    uy_f = s*( uyp + ty*ut + uzp*tx - uxp*tz )
    uz_f = s*( uzp + tz*ut + uxp*ty - uyp*tx )

    return( ux_f, uy_f, uz_f, gammam1_f )
