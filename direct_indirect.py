# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:41:03 2020

@author: KatieCampos
"""

# direct_indirect.py
# Summary: Analyze VELA edge-on synthetic data to determine scale height and 
# face-on synthetic data to determine surface mass density to calculating 
# vertical velocity dispersion.
# Author: Kathleen Hamilton-Campos, intern at STScI - kahamil@umd.edu
# Date: Summer 2020

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import CircularAnnulus
from photutils import aperture_photometry
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.constants import G
from astropy import units as u
import argparse

# Define a custom model for fitting mass profiles
@custom_model
def sech_sq_model(z, amp = 1., z_prime = 0., z_0 = 1.):
    sech_eq = amp*(4. / (np.exp((z-z_prime)/z_0) + np.exp(-(z-z_prime)/z_0))**2)
    return sech_eq

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='set parameters for VELA simulation analysis')
    
    parser.add_argument('--scale', metavar='scale', type=str, action='store',
                        help='which scale? default is 0.56')
    parser.set_defaults(scale="0.56")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd? default is no')
    parser.set_defaults(pwd=False)
    
    parser.add_argument('--z_0', metavar='z_0', type=float, action='store',
                        help='initial guess for z_0? default is 1.0')
    parser.set_defaults(z_0=1.0)

    parser.add_argument('--inc', metavar='inc', type=float, action='store',
                        help='increment for shells? default width is 10')
    parser.set_defaults(inc=10)
    
    parser.add_argument('--edgepix', metavar='edgepix', type=float, action='store',
                        help='pixel to kpc? Default is 0.452624')
    parser.set_defaults(edgepix=0.452624)
    
    parser.add_argument('--facepix', metavar='facepix', type=float, action='store',
                        help='pixel to kpc? Default is 0.07')
    parser.set_defaults(facepix=0.07)
    
    parser.add_argument('--edgeloc', metavar='edgeloc', type=str, action='store',
                        help='where is the synthetic edge-on file located?')
    parser.set_defaults(edgeloc='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Synthetic/0.56/hlsp_vela_hst_wfc3_vela13-cam01-a0.560_f125w_v6_sim-smc.fits')
    
    parser.add_argument('--faceloc', metavar='faceloc', type=str, action='store',
                        help='where is the synthetic data face-on file located?')
    parser.set_defaults(faceloc='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Synthetic/0.56/hlsp_vela_none_none_vela13-cam00-a0.560_aux_v6_sim.fits')
    
    parser.add_argument('--dictloc', metavar='dictloc', type=str, action='store',
                        help='where is the comparison dictionary located?')
    parser.set_defaults(dictloc='/Users/Kitty/Documents/College/Summer2020/Simulations/comp_56.npy')

    parser.add_argument('--figdir', metavar='figdir', type=str, action='store',
                        help='where should figures be saved?')
    parser.set_defaults(figdir='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Synthetic/0.56')

    args = parser.parse_args()
    return args

args = parse_args()

scale = args.scale
inc = args.inc
edgepix = args.edgepix
facepix = args.facepix
edgeloc = args.edgeloc
faceloc = args.faceloc
dictloc = args.dictloc
figdir = args.figdir
z_0 = args.z_0

comp_dict = np.load(dictloc, allow_pickle="True")[()]

cyl_rad = comp_dict[scale]
rad_comp = np.arange(1, cyl_rad+1)

# Read in fits file
with fits.open(edgeloc) as edge:
    # Read in data from the point-spread function
    edge_pointdata = edge[0].data

# Find the maximum surface brightness in the data
max_SB = edge_pointdata.max()

# Make a noise screen at the max percentage: adding Gaussian noise
percent = max_SB/100.
noise_screen = np.random.normal(0, percent, edge_pointdata.shape)
noise_map = edge_pointdata + noise_screen

# Preparing to plot - performing a sanity check to ensure that rows and columns are aligned with the major and minor axes like we think
len_z = len(noise_map)
range_z = np.arange(len_z)
kpc_range = edgepix * range_z
center_z = np.ceil(len_z/2)

for pix_comp in range_z:
    inc_col = edge_pointdata[:, pix_comp]
    inc_row = edge_pointdata[pix_comp, :]
    plt.plot(inc_col, 'b-')
    plt.plot(inc_row, 'k-')
plt.savefig('{}/col_row_comp.png'.format(figdir))
plt.show()

# Find scale heights via sech^2 fitting
h_z_synth = np.zeros(shape=len_z)

# Creating the mass profiles
for shell in range_z:
    m_init = sech_sq_model(amp = np.max(edge_pointdata[:, shell]), z_prime = kpc_range[np.argmax(edge_pointdata[:, shell])], z_0 = z_0)
    fit = LevMarLSQFitter()
    m = fit(m_init, kpc_range, edge_pointdata[:, shell])
    h_z_synth[shell] = m.z_0.value
    
    # Plotting the mass profile for each shell
    plt.plot(kpc_range, edge_pointdata[:, shell], '*', label='Mass Profile')
    plt.plot(kpc_range, m(kpc_range), label='Fit')
    plt.title('Mass Profile of Cam01-F125 at scale {} and Vertical Distance {}kpc'.format(scale, kpc_range[shell]))
    plt.xlabel('Horizontal Distance (kpc)')
    plt.ylabel(r'Column Surface Brightness ($\mu$J/$arcsec^{2}$)')
    plt.legend()
    plt.savefig('{}/MassProfile-Cam01-F125-atScale{}Loc{}.png'.format(figdir, scale, shell))
    plt.show()

kpc_center = center_z * edgepix
kpc_abs = np.abs(kpc_range - kpc_center)
sort = np.argsort(kpc_abs)
kpc_sort = kpc_abs[sort]
h_z_sort = h_z_synth[sort]

plt.plot(kpc_sort, h_z_sort, '*')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Scale Height (kpc)')
plt.title('Scale Heights for Cam01-F125 at scale {}'.format(scale))
plt.savefig('{}/ScaleHeight-Cam01-F125-atScale{}Line.png'.format(figdir, scale))
plt.show()

with fits.open(faceloc) as face:
    stellar_mass = face[1].data[4]
    gas_mass = face[1].data[0]

len_m = len(stellar_mass)
range_m = np.arange(len_m)
center_m = np.ceil(len_m/2)
kpc_range_m = facepix * range_m
kpc_center = facepix * center_m
kpc_abs = np.abs(kpc_range_m - (kpc_center))

half_len_m = round(len_m/2)
half_range_m = np.arange(0, half_len_m)
half_kpc_range_m = facepix * half_range_m
half_kpc_abs = np.abs(half_kpc_range_m - (kpc_center))

# Find surface mass density via aperture photometry
phot_star = np.zeros(shape=half_len_m)
phot_gas = np.zeros(shape=half_len_m)
phot_sg = np.zeros(shape=half_len_m)
aper = np.zeros(shape=half_len_m)
smd_star = np.zeros(shape=half_len_m)
smd_gas = np.zeros(shape=half_len_m)
smd_sg = np.zeros(shape=half_len_m)

# Define shells, centered on galactic center, for aperture photometry
for shell in half_range_m:
    outer = shell + inc
    inner = shell

    aperture = CircularAnnulus([center_m, center_m], inner, outer)
    aper_area = aperture.area() * facepix**2
    aper[shell] = aper_area

    phot_s = aperture_photometry(stellar_mass, aperture)
    phot_g = aperture_photometry(gas_mass, aperture)
    phot_vals = float(phot_s['aperture_sum'])
    phot_valg = float(phot_g['aperture_sum'])
    phot_add = phot_vals + phot_valg

    phot_star[int(shell)] = phot_vals
    phot_gas[shell]= phot_valg
    phot_sg[shell] = phot_add

    smd_s = phot_vals / aperture.area()
    smd_g = phot_valg / aperture.area()
    smd_add = phot_add / aperture.area()

    smd_star[shell] = smd_s
    smd_gas[shell] = smd_g
    smd_sg[shell] = smd_add

plt.plot(half_kpc_abs, smd_star, '*')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel(r'Surface Mass Density (Msun/$kpc^{2})$')
plt.yscale('log')
plt.title('Surface Mass Density [Stellar Mass] of Cam00-Aux at scale {}'.format(scale))
plt.savefig('{}/SurfMassDensStar-Cam00-Aux-atScale{}.png'.format(figdir, scale))
plt.show()

plt.plot(half_kpc_abs, smd_gas, '*')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel(r'Surface Mass Density (Msun/$kpc^{2})$')
plt.yscale('log')
plt.title('Surface Mass Density of [Gas Mass] Cam00-Aux at scale {}'.format(scale))
plt.savefig('{}/SurfMassDensGas-Cam00-Aux-atScale{}.png'.format(figdir, scale))
plt.show()

plt.plot(half_kpc_abs, smd_sg, '*')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel(r'Surface Mass Density (Msun/$kpc^{2})$')
plt.yscale('log')
plt.title('Surface Mass Density of [Stellar and Gas Mass] Cam00-Aux at scale {}'.format(scale))
plt.savefig('{}/SurfMassDensStarGas-Cam00-Aux-atScale{}.png'.format(figdir, scale))
plt.show()


kpc_interp = np.arange(0, kpc_center)
interp_h_z = np.interp(kpc_interp, kpc_sort, h_z_sort)
interp_smd_star = np.interp(kpc_interp, half_kpc_range_m, smd_star)
interp_smd_gas = np.interp(kpc_interp, half_kpc_range_m, smd_gas)
interp_smd_sg = np.interp(kpc_interp,  half_kpc_range_m, smd_sg)

const = 2.
G_conv = G.to('km3/(kg s2)')

h_z_0 = (interp_h_z * u.kpc).to('km')
smd_star_unit = ((interp_smd_star * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
smd_gas_unit = ((interp_smd_gas * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
smd_sg_unit = ((interp_smd_sg * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
sigma_synth_star = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_star_unit)).value
sigma_synth_gas = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_gas_unit)).value
sigma_synth_sg = (np.sqrt(const * np.pi * G_conv * h_z_0 * smd_sg_unit)).value



star_radius = comp_dict['star radius']
star_stdev = np.array(comp_dict['star stdev'])
sigma_mass = comp_dict['sigma mass']


plt.plot(kpc_interp, sigma_synth_star, '-.', color='tan', label='Indirect (Synthetic) Stars')
plt.plot(kpc_interp, sigma_synth_sg, '-.', color='grey', label='Indirect (Synthetic) Stars and Gas')
plt.plot(star_radius, star_stdev, '-', color='black', label='Direct Stars')
plt.plot(rad_comp, sigma_mass, '*', label='Direct Calc')
plt.title(r'Comparison of $\sigma$ at {}'.format(scale))
plt.xlim([0, np.max(rad_comp)])
plt.ylim([0, 150])
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Velocity Dispersion (km/s)')
plt.legend()
#plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), size=(5,5))
plt.savefig('{}/SigmaCompScale{}.png'.format(figdir, scale))
plt.show()


smd_mass = comp_dict['smd stars']
smd_mg = comp_dict['smd mg']
interp_smd_mass = np.interp(kpc_interp, rad_comp, smd_mass)
interp_smd_mg = np.interp(kpc_interp, rad_comp, smd_mg)
smd_mass_unit = ((interp_smd_mass * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
smd_mg_unit = ((interp_smd_mg * u.Msun / (u.kpc * u.kpc))).to('kg/(km2)')
interp_star_stdev = (np.interp(kpc_interp, star_radius, star_stdev) * u.km / u.s)
h_z_sim = comp_dict['h_z sim']
interp_h_z_sim = np.interp(kpc_interp, rad_comp, h_z_sim)
h_z_sim0 = (interp_h_z_sim * u.kpc).to('km')


c_id = interp_star_stdev**2 / (np.pi * G_conv * h_z_0 * smd_mass_unit)
c_ii = interp_star_stdev**2 / (np.pi * G_conv * h_z_0 * smd_gas_unit)
c_dd = interp_star_stdev**2 / (np.pi * G_conv * h_z_sim0 * smd_mass_unit)
c_di = interp_star_stdev**2 / (np.pi * G_conv * h_z_sim0 * smd_gas_unit)
c_ig = interp_star_stdev**2 / (np.pi * G_conv * h_z_0 * smd_mg_unit)
c_dg = interp_star_stdev**2 / (np.pi * G_conv * h_z_sim0 * smd_mg_unit)

plt.plot(kpc_interp, c_id, '*', label='Indirect h_z, direct smd')
plt.plot(kpc_interp, c_ii, '*', label='Indirect h_z, indirect smd')
plt.plot(kpc_interp, c_dd, '*', label='Direct h_z, direct smd')
plt.plot(kpc_interp, c_di, '*', label='Direct h_z, indirect smd')
plt.plot(kpc_interp, c_ig, '*', label='Indirect h_z, direct gas')
plt.plot(kpc_interp, c_dg, '*', label='Direct h_z, direct gas')
plt.legend()
plt.title('Calibrating c at Scale {}'.format(scale))
plt.xlabel('Radius (kpc)')
plt.ylabel('c (dimensionless)')
plt.yscale('log')
plt.savefig('{}/C_Span-Scale{}.png'.format(figdir, scale))
plt.show()

plt.plot(kpc_interp, c_ii, '*', label='Indirect h_z, indirect smd')
plt.legend()
plt.title('Calibrating c at Scale {}'.format(scale))
plt.xlabel('Radius (kpc)')
plt.ylabel('c (dimensionless)')
plt.xlim([10, 30])
plt.ylim([0,10])
plt.savefig('{}/C_Span-Scale{}-IndirectOnly.png'.format(figdir, scale))
plt.show()
'''
if scale == '0.33':
    c_min = 0.68824665
    c_max = 3.4923533
    sigma_z_min = np.sqrt(np.pi * G_conv * h_z_0 * smd_gas_unit * c_min)
    sigma_z_max = np.sqrt(np.pi * G_conv * h_z_0 * smd_gas_unit * c_max)

if scale == '0.56':
    c_min = 3.1677031
    c_max = 6.5958701
    sigma_z_min = np.sqrt(np.pi * G_conv * h_z_0 * smd_gas_unit * c_min)
    sigma_z_max = np.sqrt(np.pi * G_conv * h_z_0 * smd_gas_unit * c_max)

plt.plot(kpc_interp, sigma_z_min, label='Min')
plt.plot(kpc_interp, sigma_z_max, label='Max')
'''

sigma_synth_star_min = (np.sqrt(1.5 * np.pi * G_conv * h_z_0 * smd_star_unit)).value
sigma_synth_star_max = (np.sqrt(2 * np.pi * G_conv * h_z_0 * smd_star_unit)).value

sigma_synth_sg_min = (np.sqrt(1.5 * np.pi * G_conv * h_z_0 * smd_sg_unit)).value
sigma_synth_sg_max = (np.sqrt(2 * np.pi * G_conv * h_z_0 * smd_sg_unit)).value

scatter_alpha= 0.25

numsize = 20
legsize = 25
wordsize = 30

fig, axes = plt.subplots(1,2, figsize = (24,8))

axes[0].plot(star_radius, std_stars, linewidth=6, color='black', label='All Stars')
axes[0].plot(star_radius, std_new, linewidth=3, color='blue', label='New Stars')
axes[0].plot(star_radius, std_young, linewidth=3, color='cyan', label='Young Stars')
axes[0].plot(star_radius, std_intermediate, linewidth=3, color='magenta',  label='Intermediate Stars')
axes[0].plot(star_radius, std_old, linewidth=3, color='red', label='Old Stars')
axes[0].tick_params(axis = 'both', labelsize=numsize)
axes[0].set_xlim([0, np.max(star_radius)])
axes[0].set_xticks([0., 5., 10., 15., 20.])
axes[0].set_xlabel('Radius (kpc)', fontsize=wordsize)
axes[0].set_ylabel(r'$\sigma_{z}$ (km/s)', fontsize=wordsize)
axes[0].set_ylim([0, 150])
axes[0].legend(fontsize=legsize)

axes[1].fill_between(kpc_interp, sigma_synth_star_min, sigma_synth_star_max, alpha = scatter_alpha, color = 'tan', label = r'Indirect Method (using $\Sigma_{*}$)')
axes[1].fill_between(kpc_interp, sigma_synth_sg_min, sigma_synth_sg_max, alpha = scatter_alpha, color = 'grey', label = r'Indirect Method (using $\Sigma_{bary}$)')
axes[1].plot(star_radius, star_stdev, '-', linewidth=6, color='black', label='Direct Measurement')
axes[1].tick_params(axis = 'both', labelsize=numsize)
axes[1].set_xlim([0, np.max(rad_comp)])
axes[1].set_xlabel('Radius (kpc)', fontsize=wordsize)
axes[1].set_ylabel(r'$\sigma_{z}$ (km/s)', fontsize=wordsize)
axes[1].set_ylim([0, 150])
axes[1].legend(fontsize=legsize)


fig.savefig('{}/PaperPlot.png'.format(figdir))


axes[0].set_ylabel('Log Mass (Msun)')
axes[1].set_xlabel('Redshift (z)')
axes[2].set_ylabel('Log Metallicity (Zsun)')
axes[2].yaxis.tick_right()
axes[2].yaxis.set_label_position("right")

axes[0].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
axes[1].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
axes[2].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False)

fig.savefig('MetalDistributionOrigComp.png')


'''
*calcualte R90 from mass map - how?
Radial profile of accumulated mass, photutils
-measure where the center is
*create functions for each step

*factor of a million difference...
'''
