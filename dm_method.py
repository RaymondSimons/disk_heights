# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:37:48 2020

@author: KatieCampos
"""

# dm_method.py
# Summary: Mock up a method for creating a general Navarro-Frenk-White dark 
# matter halo model to find the radial surface mass density in the disk.
# Author: Kathleen Hamilton-Campos, intern at STScI - kahamil@umd.edu
# Date: Summer 2020

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from halotools.empirical_models import PrebuiltSubhaloModelFactory, halo_mass_to_halo_radius, NFWProfile, density_threshold
from astropy.cosmology import Planck15
import astropy.units as u
import argparse
from scipy.integrate import quad

# define parameters
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='set parameters for VELA simulation analysis')
    
    parser.add_argument('--scale', metavar='scale', type=float, action='store',
                        help='which scale? default is 0.56')
    parser.set_defaults(scale=0.56)
    
    parser.add_argument('--mdef', metavar='mdef', type=str, action='store',
                        help='what is the halo mass definition? default is vir')
    parser.set_defaults(mdef="vir")
    
    parser.add_argument('--radmin', metavar='radmin', type=int, action='store',
                        help='what is the minimum halo radius? default is 1')
    parser.set_defaults(radmin=1)

    parser.add_argument('--inc', metavar='inc', type=int, action='store',
                        help='what is the radial increment? default is 1')
    parser.set_defaults(inc=1)
    
    parser.add_argument('--z', metavar='z', type=int, action='store',
                        help='what is the disc thickness? default is 20')
    parser.set_defaults(z=20)

    args = parser.parse_args()
    return args

args = parse_args()

scale = args.scale
mdef = args.mdef
radmin = args.radmin
inc = args.inc
z = args.z

# set cosmology
redshift = 1/scale - 1
cosmo = Planck15
h = Planck15.h
conc = 5

# needs to run first to initialize package
# halotools.test_installation()

# determine dark mass and virial radius
model = PrebuiltSubhaloModelFactory('behroozi10', redshift = redshift)
stars_mass = 60850451172.24926
# stars_mass in units Msun from simulation
log_stars_mass = np.log10(stars_mass)
log_dark_mass = model.mean_log_halo_mass(log_stellar_mass=log_stars_mass)
dm_stellar = 10**log_dark_mass
virial = halo_mass_to_halo_radius(dm_stellar, cosmo, redshift, mdef)
virial_kpc = (virial * u.Mpc).to('kpc') / h
virial_rad = np.arange(radmin, virial_kpc.value, inc)
scaled_rad = virial_rad / np.max(virial_rad)

# calculate the density threshold for dimensionless mass density calculation
nfw = NFWProfile()
rho_thresh = density_threshold(cosmo, redshift, mdef)
# rho_thresh in units Msun*h^2/Mpc^3
rho_units = (rho_thresh * u.Msun / u.Mpc**3).to('Msun/kpc3') * h**2
dimless_massdens = nfw.dimensionless_mass_density(scaled_rad, conc)
mass_dens = dimless_massdens * rho_units
# mass_dens is in units Msun/kpc^3

# create an array of heights to take vertical components into account
height_arr = np.arange(-z, z+1)
height_scale = height_arr / z

# prepare arrays
nfw_heights = np.zeros(len(height_scale))
nfw_avg = np.zeros(len(scaled_rad))

# for each radial position, calculate the 3D mass density for the array of 
# heights above and below the disk; save the mean of those for each radius
for r_ind, radial in enumerate(scaled_rad):
    for z_ind, height in enumerate(height_scale):
        threed = np.sqrt(radial**2 + height**2)
        nfw_heights[z_ind] = (nfw.dimensionless_mass_density(threed, conc)) * rho_units.value
    nfw_avg[r_ind] = np.mean(nfw_heights)

# integrate the NFW density profile to check the dark mass calculation
r_s = virial_kpc.value / conc
gc = np.log(1 + conc) - (conc / (1 + conc))
f = lambda r: r**2 * 4 * np.pi * rho_units.value * ((conc**3 / (3 * gc))) / ((r / r_s) * (1 + (r / r_s))**2)
dm_int = quad(f, radmin, virial_kpc.value)

plt.plot(virial_rad, nfw_avg, label='Height')
plt.plot(virial_rad, mass_dens, label='Radial')
plt.xlabel('Radius (kpc)')
plt.xlim([0, 50])
plt.ylabel('Dark Matter Density (Msun/Mpc^3)')
plt.yscale('log')
plt.legend()
plt.show()

print('The dark matter mass according to the stellar mass-halo mass relation is {:.2e}'.format(dm_stellar))
print('The dark matter mass according to the NFW density integration is {:.2e}'.format(dm_int[0]))
