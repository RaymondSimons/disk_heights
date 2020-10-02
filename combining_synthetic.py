# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:14:13 2020

@author: KatieCampos
"""


# combining_synthetic.py
# Summary: Combine scale height (from edge_heights.py) - and surface
# mass density (mass_map.py) to calculate vertical velocity dispersion for 
# VELA.
# Author: Kathleen Hamilton-Campos, intern at STScI - kahamil@umd.edu
# Date: Summer 2020

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.constants import G
from astropy import units as u

comp_dict = np.load('/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13-0.56-Comparisons.npy', allow_pickle=True)[()]

star_radius = comp_dict['stellar radius']
star_stdev = comp_dict['star stdev']

const = 2.
G_conv = G.to('km3/(kg s2)')

h_z_0 = (interp_h_z * u.kpc).to('km')
smd_star_unit = ((interp_smd_star * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
smd_gas_unit = ((interp_smd_gas * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
smd_sg_unit = ((interp_smd_sg * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
sigma_synth_star = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_star_unit)).value
sigma_synth_gas = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_gas_unit)).value
sigma_synth_sg = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_sg_unit)).value

fig= plt.figure(figsize=(20,10))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(kpc_range, sigma_synth_star, ':', label='Synthetic Data: Stars')
axes.plot(kpc_range, sigma_synth_gas, ':', label='Synthetic Data: Gas')
axes.plot(kpc_range, sigma_synth_sg, ':', label='Synthetic Data: Stars and Gas')
axes.plot(star_radius, star_stdev, '-.', label='Simulated Measured: All Stars')
axes.plot(star_radius, new_stdev, '-.', label='Simulation Measured: New Stars')
axes.plot(star_radius, young_stdev, '-.', label='Simulation Measured: Young Stars')
axes.plot(star_radius, intermediate_stdev, '-.', label='Simulation Measured: Intermediate Stars')
axes.plot(star_radius, old_stdev, '-.', label='Simulation Measured: Old Stars')
axes.plot(rad_comp, sigma_mass, '-', label='Simulation Calculated: Stellar Mass')
axes.plot(rad_comp, sigma_mg, '-', label='Simulation Calculated: Stars and Gas')
axes.plot(rad_comp, sigma_all, '-', label='Simulation Calculated: Stars, Gas, and DM')
plt.title(r'$\sigma$ for {} at {}'.format(synth, scale))
#plt.xlim([0, np.max(rad_comp)])
#plt.ylim([0, 200])
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Velocity Dispersion (km/s)')
#plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), size=(5,5))
plt.savefig('/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13 Synthetic/{}SigmaCompScale{}.png'.format(synth, scale))
plt.show()
'''
plt.plot(kpc_range, sigma_synth_star, '-.', color='#f000f0', label='Synthetic Data: Stars')
plt.plot(kpc_range, sigma_synth_sg, '-.', color='#b000b0', label='Synthetic Data: Stars and Gas')
plt.plot(rad_comp, sigma_mass, '-', color='#f000f0', label='Simulation Calculated: Stellar Mass')
plt.plot(rad_comp, sigma_mg, '-', color='#b000b0', label='Simulation Calculated: Stars and Gas')
plt.xlim([0, np.max(rad_comp)])
plt.ylim([0, np.max(sigma_mg)+5])
plt.savefig('/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13 Synthetic/{}SigmaCompScaleTalk{}.png'.format(synth, scale))
plt.show()
'''
'''

plt.plot(kpc_range, sigma_synth_star, '-.', color='tan', label='Synthetic Data: Stars')
#plt.plot(kpc_range, sigma_synth_gas, '-', label='Synthetic Data: Gas')
plt.plot(kpc_range, sigma_synth_sg, '-.', color='grey', label='Synthetic Data: Stars and Gas')
plt.plot(gas_radius, gas_stdev, '-', color='green', alpha=0.4, label='Simulated Measured: Gas')
plt.plot(star_radius, star_stdev, '-', color='black', label='Simulated Measured: All Stars')
plt.plot(star_radius, new_stdev, '-', color='blue', alpha=0.4, label='Simulation Measured: New Stars')
plt.plot(star_radius, young_stdev, '-', color='cyan', alpha=0.4, label='Simulation Measured: Young Stars')
plt.plot(star_radius, intermediate_stdev, '-', color='magenta', alpha=0.4, label='Simulation Measured: Intermediate Stars')
plt.plot(star_radius, old_stdev, '-', color='red', alpha=0.4, label='Simulation Measured: Old Stars')
#plt.plot(rad_comp, sigma_mass, '-', label='Simulation Calculated: Stellar Mass')
#plt.plot(rad_comp, sigma_mg, '-', label='Simulation Calculated: Stars and Gas')
#plt.plot(rad_comp, sigma_all, '-', label='Simulation Calculated: Stars, Gas, and DM')
#plt.title(r'$\sigma$ for {} at {}'.format(synth, scale))
plt.xlim([0, np.max(rad_comp)])
plt.ylim([0, 150])
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Velocity Dispersion (km/s)')
#plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), size=(5,5))
plt.savefig('/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13 Synthetic/PresentationAlpha{}SigmaCompScale{}.png'.format(synth, scale))
plt.show()
'''