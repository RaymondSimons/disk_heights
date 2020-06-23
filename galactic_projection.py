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
import matplotlib.pyplot as plt

# Simulation name
sim = 'VELA13'

# Load in simulation and dictionary
ds = yt.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13/VELA13/10MpcBox_csf512_a0.560.d')
ad = ds.all_data()
gal_dict = np.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13_galprops.npy', allow_pickle='true', encoding='latin1')[()]

def _new_stars(pfilter, data):
    """Filter star particles with creation time < 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='new_stars', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 20, age >= 0)
    return filter

def _young_stars(pfilter, data):
    """Filter star particles with creation time < 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='all', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 100, age > 20)
    return filter

def _intermediate_stars(pfilter, data):
    """Filter star particles with creation time > 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_old_stars, filtered_type='all', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Myr') >= 100, age < 1000)
    return filter

def _old_stars(pfilter, data):
    """Filter star particles with creation time > 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_old_stars, filtered_type='all', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_or(age.in_units('Myr') >= 1000, age < 0)
    return filter

yt.add_particle_filter("new_stars", function=_new_stars, filtered_type='new_stars', requires=["particle_creation_time"])
yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='all', requires=["particle_creation_time"])
yt.add_particle_filter("intermediate_stars", function=_intermediate_stars, filtered_type='all', requires=["particle_creation_time"])
yt.add_particle_filter("old_stars", function=_old_stars, filtered_type='all', requires=["particle_creation_time"])

ds.add_particle_filter("new_stars")
ds.add_particle_filter("young_stars")
ds.add_particle_filter("intermediate_stars")
ds.add_particle_filter("old_stars")

# Set centers
cens = gal_dict['true_center']
gal_cen = cens[0]
cen_cen = yt.YTArray(gal_cen, 'kpc')
com = ad.quantities.center_of_mass()
cen_com = yt.YTArray([com], 'kpc')

proj_width = 100

# Set sphere
sphere_rad = 100
sp = ds.sphere(cen_cen, (sphere_rad, 'kpc'))

if False:
    # Basic projections
    yt.ProjectionPlot(ds, "x", "density").save('{}_Density_Projection_X'.format(sim))
    yt.ProjectionPlot(ds, "y", "density").save('{}_Density_Projection_y'.format(sim))
    yt.ProjectionPlot(ds, "z", "density").save('{}_Density_Projection_z'.format(sim))

# Calculating stellar angular momentum
star_ang_mom_x = sp.quantities.total_quantity([("stars", "particle_angular_momentum_x")])
star_ang_mom_y = sp.quantities.total_quantity([("stars", "particle_angular_momentum_y")])
star_ang_mom_z = sp.quantities.total_quantity([("stars", "particle_angular_momentum_z")])
star_ang_mom = yt.YTArray([star_ang_mom_x, star_ang_mom_y, star_ang_mom_z])

# Setting face-on and edge-on projection directions for the stars
star_ang_mom_tot = np.sqrt(sum(star_ang_mom**2))
star_ang_mom_norm = star_ang_mom / star_ang_mom_tot

edge_on_dir = np.random.randn(3)
edge_on_dir -= edge_on_dir.dot(star_ang_mom_norm) * star_ang_mom_norm / np.linalg.norm(star_ang_mom_norm)**2

# Calculating gas angular momentum
gas_ang_mom_x = sp.quantities.total_quantity([("gas", "angular_momentum_x")])
gas_ang_mom_y = sp.quantities.total_quantity([("gas", "angular_momentum_y")])
gas_ang_mom_z = sp.quantities.total_quantity([("gas", "angular_momentum_z")])
gas_ang_mom = yt.YTArray([gas_ang_mom_x, gas_ang_mom_y, gas_ang_mom_z])

# Setting face-on and edge-on projection directions for the gas
gas_ang_mom_tot = np.sqrt(sum(gas_ang_mom**2))
gas_ang_mom_norm = gas_ang_mom / gas_ang_mom_tot

edge_on_gas = np.random.randn(3)
edge_on_gas -= edge_on_gas.dot(gas_ang_mom_norm) * gas_ang_mom_norm / np.linalg.norm(gas_ang_mom_norm)**2

# Set cylinder
cyl_rad = 50
cyl_h = 30
cyl = ds.disk(center = cen_cen, normal = star_ang_mom_norm, radius = (cyl_rad, 'kpc'), height = (cyl_h, 'kpc'))

# Making profile plots
if True:
    prof_gas = yt.create_profile(cyl, 'radius', ('gas', 'velocity_magnitude'), weight_field = ('gas', 'cell_mass'))
    radius = prof_gas.x.to('kpc')
    std_gas = prof_gas.standard_deviation['gas', 'velocity_magnitude']
    
    prof_new =  yt.create_profile(cyl, ('new_stars', 'particle_position_cylindrical_radius'), ('new_stars', 'particle_velocity_cylindrical_radius'), weight_field = ('new_stars', 'particle_mass'))
    std_new = prof_new.standard_deviation['new_stars', 'particle_velocity_cylindrical_radius']
    
    prof_young =  yt.create_profile(cyl, ('young_stars', 'particle_position_cylindrical_radius'), ('young_stars', 'particle_velocity_cylindrical_radius'), weight_field = ('young_stars', 'particle_mass'))
    std_young = prof_new.standard_deviation['young_stars', 'particle_velocity_cylindrical_radius']
    
    prof_intermediate =  yt.create_profile(cyl, ('intermediate_stars', 'particle_position_cylindrical_radius'), ('intermediate_stars', 'particle_velocity_cylindrical_radius'), weight_field = ('intermediate_stars', 'particle_mass'))
    std_intermediate = prof_new.standard_deviation['intermediate_stars', 'particle_velocity_cylindrical_radius']
    
    prof_old =  yt.create_profile(cyl, ('old_stars', 'particle_position_cylindrical_radius'), ('old_stars', 'particle_velocity_cylindrical_radius'), weight_field = ('old_stars', 'particle_mass'))
    std_old = prof_new.standard_deviation['old_stars', 'particle_velocity_cylindrical_radius']
    
    plt.plot(radius, std_gas, label='Gas')
    plt.plot(radius, std_new, label='New Stars')
    plt.plot(radius, std_young, label='Young Stars')
    plt.plot(radius, std_intermediate, label='Intermediate Stars')
    plt.plot(radius, std_old, label='Old Stars')
    plt.title('Velocity Dispersion for {}'.format(sim))
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Velocity [km/s]')
    plt.legend()
    plt.savefig('VelocityDispersion{}.png'.format(sim))

# Making the projections: currently each one renders in a bit over half an hour
if False:
    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = star_ang_mom_norm, fields = ('deposit', 'stars_density'), center = cen_cen, width=(proj_width, 'kpc'), data_source = sp)
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

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = star_ang_mom_norm, fields = ('deposit', 'stars_density'), center = cen_cen, width = (proj_width, 'kpc'), data_source = cyl)
        prj.save('StarsFaceOnCyl{}-{}.png'.format(sim, proj_width))

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = ('deposit', 'stars_density'), center = cen_cen, width = (proj_width, 'kpc'), data_source = cyl)
        prj.save('StarsEdgeOnCyl{}-{}.png'.format(sim, proj_width))
