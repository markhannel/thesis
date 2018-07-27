'''
Produce a radial profile of the hologram caused by a polystyrene (n=1.59)
scatterer in water 20 um above the focal plane. Demonstrate the
geometrical phase difference between the incident field and the scattered 
field.
'''

import numpy as np
import lorenzmie.theory.spheredhm as sph
import pylab as pl
import latex_options
import os

def ret_profiles():
    a_p = 1.0  # [um]
    n_p = 1.40 # [1]
    z_p = 200 # [pixel]
    r_p = [-200, 0, z_p]
    n_m = 1.339
    lamb = 0.447 # [um]
    mpp = 0.135  # [um/pix]
    
    scat_field = sph.spheredhm(r_p, a_p, n_p, n_m, lamb=lamb, dim=[400,1]).flatten()

    k_pix = 2*np.pi*n_m*mpp/lamb
    x = np.arange(scat_field.size, dtype=complex)
    geom_field = np.ones(scat_field.shape, dtype=complex)*np.exp(1.j*k_pix*(z_p - np.sqrt(x**2+z_p**2)))/np.sqrt(x**2 + z_p**2)
    
    return scat_field, np.real(geom_field)

def main():
    rad_pro, geom_pro = ret_profiles()

    end = 200
    fig, ax = pl.subplots(figsize=(12,8))
    ax.plot(rad_pro[0:end], label='Radial Profile')
    ax.plot(geom_pro[0:end] + 1, label='Geometric Factor')
    ax.set_xlabel('X [pixel]')
    ax.set_ylabel('Intensity [arbs]')
    #ax.set_ylim(0, 1.5)
    pl.legend()
    pl.show()

if __name__ == '__main__':
    main()
    
    
