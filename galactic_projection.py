# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:35:23 2020

@author: Kathleen Hamilton
"""

# galactic_analysis.py
# Summary: Analyze VELA simulations to determine velocity dispersion.
# Author: Kathleen Hamilton-Campos, intern at STScI, summer 2020 - kahamil@umd.edu

# Import necessary libraries
import yt
import numpy as np

# Simulation name
sim = 'VELA13'

# Load in simulation and dictionary
ds = yt.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13/VELA13/10MpcBox_csf512_a0.560.d')
ad = ds.all_data()
gal_dict = np.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13_galprops.npy', allow_pickle='true', encoding='latin1')[()]

# Set centers
cens = gal_dict['true_center']
gal_cen = cens[0]
cen_cen = yt.YTArray(gal_cen, 'kpc')
com = ad.quantities.center_of_mass()
cen_com = yt.YTArray([com], 'kpc')

# Set sphere
sphere_rad = 100
proj_width = 100
ob_sphere = ds.sphere(cen_cen, (sphere_rad, 'kpc'))

if False:
    # Basic projections
    yt.ProjectionPlot(ds, "x", "density").save('{}_Density_Projection_X'.format(sim))
    yt.ProjectionPlot(ds, "y", "density").save('{}_Density_Projection_y'.format(sim))
    yt.ProjectionPlot(ds, "z", "density").save('{}_Density_Projection_z'.format(sim))

# Calculating stellar angular momentum
star_ang_mom_x = ob_sphere.quantities.total_quantity([("stars", "particle_angular_momentum_x")])
star_ang_mom_y = ob_sphere.quantities.total_quantity([("stars", "particle_angular_momentum_y")])
star_ang_mom_z = ob_sphere.quantities.total_quantity([("stars", "particle_angular_momentum_z")])
star_ang_mom = yt.YTArray([star_ang_mom_x, star_ang_mom_y, star_ang_mom_z])

# Setting face-on and edge-on projection directions for the stars
star_ang_mom_tot = np.sqrt(sum(star_ang_mom**2))
star_ang_mom_norm = star_ang_mom / star_ang_mom_tot

edge_on_dir = np.random.randn(3)
edge_on_dir -= edge_on_dir.dot(star_ang_mom_norm) * star_ang_mom_norm / np.linalg.norm(star_ang_mom_norm)**2

# Calculating gas angular momentum
gas_ang_mom_x = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_x")])
gas_ang_mom_y = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_y")])
gas_ang_mom_z = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_z")])
gas_ang_mom = yt.YTArray([gas_ang_mom_x, gas_ang_mom_y, gas_ang_mom_z])

# Setting face-on and edge-on projection directions for the gas
gas_ang_mom_tot = np.sqrt(sum(gas_ang_mom**2))
gas_ang_mom_norm = gas_ang_mom / gas_ang_mom_tot

edge_on_gas = np.random.randn(3)
edge_on_gas -= edge_on_gas.dot(gas_ang_mom_norm) * gas_ang_mom_norm / np.linalg.norm(gas_ang_mom_norm)**2

if True:
    # Making the projections: currently each one renders in a bit over half an hour
    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = star_ang_mom_norm, fields = ('deposit', 'stars_density'), center = cen_cen, width=(proj_width, 'kpc'), data_source = ob_sphere)
        prj.save('StarsFaceOn{}-{}.png'.format(sim, proj_width))

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = ('deposit', 'stars_density'), center = cen_cen, width=(proj_width, 'kpc'))
        prj.save('StarsEdgeOn{}-{}.png'.format(sim, proj_width))

    if False:
        prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'))
        prj.save('GasFaceOn{}-{}.png'.format(sim, proj_width))

    if False:
        prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_gas, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'))
        prj.save('GasEdgeOn{}-{}.png'.format(sim, proj_width))