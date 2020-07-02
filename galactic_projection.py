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
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.constants import G
from astropy import units as u

# Simulation name
sim = 'VELA13'

# Define a custom model for fitting mass profiles
@custom_model
def sech_sq_model(z, amp = 1., z_prime = 0., z_0 = 1.):
    sech_eq = amp*(4. / (np.exp((z-z_prime)/z_0) + np.exp(-(z-z_prime)/z_0))**2)
    return sech_eq

# Load in simulation and dictionary
ds = yt.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13/VELA13/10MpcBox_csf512_a0.560.d')
ad = ds.all_data()
gal_dict = np.load('C:/Users/Kitty/Documents/College/Summer 2020/Simulations/VELA13_galprops.npy', allow_pickle='true', encoding='latin1')[()]

# Define new filters to distinguish stars by age
def _new_stars(pfilter, data):
    """Filter star particles with creation time < 20 Myr ago
    To use: yt.add_particle_filter("new_stars", function=_young_stars, filtered_type='stars', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 20, age.in_units('Myr') >= 0)
    return filter

def _young_stars(pfilter, data):
    """Filter star particles with creation time < 10 Myr ago
    To use: yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='all', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Myr') <= 100, age.in_units('Myr') > 20)
    return filter

def _intermediate_stars(pfilter, data):
    """Filter star particles with creation time 100 Myr - 1 Gyr ago
    To use: yt.add_particle_filter("intermediate_stars", function=_old_stars, filtered_type='stars', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_and(age.in_units('Gyr') <= 1, age.in_units('Myr') > 100)
    return filter

def _old_stars(pfilter, data):
    """Filter star particles with creation time > 1 Gyr ago
    To use: yt.add_particle_filter("old_stars", function=_old_stars, filtered_type='stars', requires=["particle_creation_time"])"""
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = np.logical_or(age.in_units('Gyr') > 1, age.in_units('Myr') < 0)
    return filter

yt.add_particle_filter("new_stars", function=_new_stars, filtered_type='stars', requires=["particle_creation_time"])
yt.add_particle_filter("young_stars", function=_young_stars, filtered_type='stars', requires=["particle_creation_time"])
yt.add_particle_filter("intermediate_stars", function=_intermediate_stars, filtered_type='stars', requires=["particle_creation_time"])
yt.add_particle_filter("old_stars", function=_old_stars, filtered_type='stars', requires=["particle_creation_time"])

ds.add_particle_filter("new_stars")
ds.add_particle_filter("young_stars")
ds.add_particle_filter("intermediate_stars")
ds.add_particle_filter("old_stars")

ad = ds.all_data()

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

# Basic projections
if False:
    yt.ProjectionPlot(ds, "x", "density").save('{}_Density_Projection_X'.format(sim))
    yt.ProjectionPlot(ds, "y", "density").save('{}_Density_Projection_y'.format(sim))
    yt.ProjectionPlot(ds, "z", "density").save('{}_Density_Projection_z'.format(sim))

# Angular momentum orientations
if True:
    
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
cyl_rad = 50.
cyl_h = 30.
cyl = ds.disk(center = cen_cen, normal = star_ang_mom_norm, radius = (cyl_rad, 'kpc'), height = (cyl_h, 'kpc'))

# Making profile plots for velocity dispersion in the cylinder
if True:
    prof_gas = yt.create_profile(cyl, 'radius', ('gas', 'velocity_magnitude'), weight_field = ('gas', 'cell_mass'))
    radius = prof_gas.x.to('kpc')
    std_gas = prof_gas.standard_deviation['gas', 'velocity_magnitude'].to('km/s')

    # The dictionary prevents stars from being log binned
    dic = {}
    dic[('stars', 'particle_position_cylindrical_radius')] = False
    prof_stars = yt.create_profile(cyl, ('stars', 'particle_position_cylindrical_radius'), ('stars', 'particle_velocity_cylindrical_z'), weight_field = ('stars', 'particle_mass'), logs = dic)
    std_stars = prof_stars.standard_deviation['stars', 'particle_velocity_cylindrical_z'].to('km/s')
    
    dic = {}
    dic[('new_stars', 'particle_position_cylindrical_radius')] = False
    prof_new =  yt.create_profile(cyl, ('new_stars', 'particle_position_cylindrical_radius'), ('new_stars', 'particle_velocity_cylindrical_z'), weight_field = ('new_stars', 'particle_mass'), logs = dic)
    std_new = prof_new.standard_deviation['new_stars', 'particle_velocity_cylindrical_z'].to('km/s')

    dic = {}
    dic[('young_stars', 'particle_position_cylindrical_radius')] = False    
    prof_young =  yt.create_profile(cyl, ('young_stars', 'particle_position_cylindrical_radius'), ('young_stars', 'particle_velocity_cylindrical_z'), weight_field = ('young_stars', 'particle_mass'), logs = dic)
    std_young = prof_young.standard_deviation['young_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    
    dic = {}
    dic[('intermediate_stars', 'particle_position_cylindrical_radius')] = False
    prof_intermediate =  yt.create_profile(cyl, ('intermediate_stars', 'particle_position_cylindrical_radius'), ('intermediate_stars', 'particle_velocity_cylindrical_z'), weight_field = ('intermediate_stars', 'particle_mass'), logs = dic)
    std_intermediate = prof_intermediate.standard_deviation['intermediate_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    
    dic = {}
    dic[('old_stars', 'particle_position_cylindrical_radius')] = False
    prof_old =  yt.create_profile(cyl, ('old_stars', 'particle_position_cylindrical_radius'), ('old_stars', 'particle_velocity_cylindrical_z'), weight_field = ('old_stars', 'particle_mass'), logs = dic)
    std_old = prof_old.standard_deviation['old_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    
    # Plot the velocity dispersion for stars distinguished by age
    plt.plot(radius, std_gas, label='Gas')
    plt.plot(radius, std_stars, label='All Stars')
    plt.plot(radius, std_new, label='New Stars')
    plt.plot(radius, std_young, label='Young Stars')
    plt.plot(radius, std_intermediate, label='Intermediate Stars')
    plt.plot(radius, std_old, label='Old Stars')
    plt.title('Velocity Dispersion for {}'.format(sim))
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Velocity [km/s]')
    plt.legend()
    plt.savefig('VelocityDispersion{}.png'.format(sim))
    plt.show()


# Creating vertical mass profiles to determine scale height
if True:
    total_mass = np.zeros(shape=(int(cyl_rad)))
    total_gas = np.zeros(shape=(int(cyl_rad)))
    total_dm = np.zeros(shape=(int(cyl_rad)))
    sigma_shell = np.zeros(shape=(int(cyl_rad)))
    sigma_mass = np.zeros(shape=(int(cyl_rad)))
    sigma_mg = np.zeros(shape=(int(cyl_rad)))
    sigma_all = np.zeros(shape=(int(cyl_rad)))
    
    dic = {}
    dic[('stars', 'particle_position_cylindrical_z')] = False

    inc = 1.
    for shell in np.arange(1., cyl_rad):
        # Setting the boundaries of the toroidal shell
        outer_rad = shell + inc
        inner_rad = shell
        outer = ds.disk(center = cen_cen, normal = star_ang_mom_norm, radius = (outer_rad, 'kpc'), height = (cyl_h, 'kpc'))
        inner = ds.disk(center = cen_cen, normal = star_ang_mom_norm, radius = (inner_rad, 'kpc'), height = (cyl_h, 'kpc'))
        toroid = outer - inner
        shell_cen = shell + (inc / 2)
        toroid.set_field_parameter('normal', inner.get_field_parameter('normal'))
        toroid.set_field_parameter('center', inner.get_field_parameter('center'))
        
        # Saving the total star mass in the toroidal shell
        mass_quant = toroid.quantities.total_quantity(['stars_mass']).to('Msun')
        total_mass[int(shell)] = mass_quant

        # Saving the total gas mass in the toroidal shell        
        gas_quant = toroid.quantities.total_quantity(['cell_mass']).to('Msun')
        total_gas[int(shell)] = gas_quant
        
        # Saving the total dark matter mass in the toroidal shell
        dm_quant = toroid.quantities.total_quantity(['darkmatter_mass']).to('Msun')
        total_dm[int(shell)] = dm_quant
        
        # The "weighted_variance" actually returns the standard deviation, as checked with np.std.
        shell_stars = toroid.quantities.weighted_variance(fields = ('stars', 'particle_velocity_cylindrical_z'), weight = ('stars', 'particle_mass'))
        #prof_stars_vel = yt.create_profile(toroid, ('stars', 'particle_position_cylindrical_z'), ('stars', 'particle_velocity_cylindrical_z'), weight_field = ('stars', 'particle_mass'), logs = dic)
        #v_z = prof_stars_vel.standard_deviation['stars', 'particle_velocity_cylindrical_z'].to('km/s')
        #sigma[int(shell)] = v_z
        sigma_shell[int(shell)] = shell_stars[0].to('km/s')
        
        # Creating a mass profile
        prof_stars_mass = yt.create_profile(toroid, ('stars', 'particle_position_cylindrical_z'), ('stars', 'particle_mass'), weight_field = None, logs = dic)
        height = prof_stars_mass.x.to('kpc')
        accum_mass = prof_stars_mass['stars', 'particle_mass'].to('Msun')
        
        # Making the arrays dimensionless
        npheight = np.array(height)
        npaccum = np.array(accum_mass)
        
        # Fitting the mass profiles
        m_init = sech_sq_model(amp = np.max(npaccum), z_prime = npheight[np.argmax(npaccum)], z_0 = 1.)
        fit = LevMarLSQFitter()
        m = fit(m_init, npheight, npaccum)
        
        # Calculating sigma from eqn 1 in the proposal; differentiating by stars, stars and gas, and complete dynamical mass
        G_conv = G.to('km3/(kg s2)')
        z_0 = (m.z_0 * u.kpc).to('km')
        outer_conv = (outer_rad * u.kpc).to('km')
        inner_conv = (inner_rad * u.kpc).to('km')
        proj_area = np.pi * (outer_conv**2 - inner_conv**2)
        sigma_dyn = ((mass_quant * u.Msun).to('kg') + (gas_quant * u.Msun).to('kg') + (dm_quant * u.Msun).to('kg')) / (proj_area)
        mass_dens = (mass_quant * u.Msun).to('kg') / (proj_area)
        mass_gas = ((mass_quant * u.Msun).to('kg') + (gas_quant * u.Msun).to('kg')) / (proj_area)
        sigma_mass_calc = np.sqrt(2*np.pi*G_conv*z_0*mass_dens)
        sigma_mg_calc = np.sqrt(2*np.pi*G_conv*z_0*mass_gas)
        sigma_dyn_calc = np.sqrt(2*np.pi*G_conv*z_0*sigma_dyn)
        sigma_mass[int(shell)] = sigma_mass_calc.value
        sigma_mg[int(shell)] = sigma_mg_calc.value
        sigma_all[int(shell)] = sigma_dyn_calc.value
        

        # Plotting the mass profile for each shell
        plt.plot(npheight, npaccum, '*', label='Mass Profile')
        plt.plot(npheight, m(npheight), label='Fit')
        plt.title('Mass Profile of {}: Shell Center {}kpc, Width {}kpc'.format(sim, shell_cen, inc*2))
        plt.xlabel('Vertical Height [kpc]')
        plt.ylabel('Mass [Msun]')
        plt.legend()
        plt.savefig('MassProfile{}Center{}Width{}.png'.format(sim, shell_cen, inc*2))
        plt.show()
        

    # Plotting the predicted and measured vertical velocity dispersion
    one_to_one = np.arange(0,80)
    plt.plot(sigma_all, sigma_shell, '*', label=r'$\Sigma_{dyn}$')
    plt.plot(sigma_mg, sigma_shell, '*', label='Stars and Gas')
    plt.plot(sigma_mass, sigma_shell, '*', label='Stars')
    plt.plot(one_to_one, one_to_one, label='One-to-One')
    plt.title('Observed vs. Theoretical Vertical Velocity Dispersion for {}'.format(sim))
    plt.xlabel('Predicted Vertical Velocity Dispersion (km/s)')
    plt.ylabel('Measured Vertical Velocity Dispersion (km/s)')
    plt.legend()
    plt.savefig('ObsVsTheo-{}'.format(sim))
    plt.show()


rad_50 = np.arange(1,51)

# Plotting the comparisons of vertical velocity dispersion via different methods
plt.plot(radius, std_stars, label='Standard Deviation Projection')
plt.plot(rad_50, sigma_shell, label='Shell Sigma')
plt.plot(rad_50, sigma_all, label=r'Theoretical $\sigma_z$ with $Sigma_{dyn}$')
plt.plot(rad_50, sigma_mg, label=r'Theoretical $\sigma_z$ with Stars and Gas')
plt.plot(rad_50, sigma_mass, label=r'Theoretical $\sigma_z$ with Stars')
plt.title('Comparisons')
plt.xlabel('Radius (kpc)')
plt.ylabel('Vertical Velocity Dispersion (km/s)')
plt.legend()
plt.savefig('Comparisons.png')
plt.show()


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
