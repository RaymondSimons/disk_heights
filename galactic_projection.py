# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:35:23 2020

@author: Kathleen Hamilton
"""

# galactic_analysis.py
# Summary: This script will analyze VELA simulations to determine velocity
# dispersion. The code is customizable, allowing the main parameters to be set
# on the command line. Stars are differentiated by age, with the velocity 
# dispersion of each directly measured. Scale height and surface mass density
# are determined for an indirect measure of velocity dispersion as well.
# Author: Kathleen Hamilton-Campos, intern at STScI, summer 2020 - kahamil@umd.edu

# Import necessary libraries
import yt
import numpy as np
import matplotlib.pyplot as plt
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

    parser.add_argument('--sim', metavar='sim', type=str, action='store',
                        help='which sim? default is VELA13')
    parser.set_defaults(sim="VELA13")
    
    parser.add_argument('--scale', metavar='scale', type=str, action='store',
                        help='which scale? default is 0.56')
    parser.set_defaults(scale="0.56")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd? default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--z_0', metavar='z_0', type=float, action='store',
                        help='initial guess for z_0? default is 1.0')
    parser.set_defaults(z_0=1.0)

    parser.add_argument('--cyl_h', metavar='cyl_h', type=float, action='store',
                        help='cylinder height? default is 30.0 (kpc)')
    parser.set_defaults(cyl_h=30.0)

    parser.add_argument('--inc', metavar='inc', type=float, action='store',
                        help='increment for shells? default width is 1.0')
    parser.set_defaults(inc=1.0)
    
    parser.add_argument('--pre_dict', metavar='pre_dict', type=str, action='store',
                        help='where is the dictionary?')
    parser.set_defaults(pre_dict='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Data/VELA13_galprops.npy')
    
    parser.add_argument('--proj_width', metavar='proj_width', type=float, action='store',
                        help='projection width? default is 100.0 (kpc)')
    parser.set_defaults(proj_width=100.0)
    
    parser.add_argument('--sphere_rad', metavar='sphere_rad', type=float, action='store',
                        help='sphere radius? default is 100.0 (kpc)')
    parser.set_defaults(sphere_rad = 100.0)
    
    parser.add_argument('--const', metavar='const', type=float, action='store',
                        help='sigma squared constant? default is 2.')
    parser.set_defaults(const=2.0)
    
    parser.add_argument('--loc', metavar='loc', type=str, action='store',
                        help='where are the simulations located?')
    parser.set_defaults(loc='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Data/0.56/10MpcBox_csf512_a0.560.d')

    parser.add_argument('--figdir', metavar='figdir', type=str, action='store',
                        help='where should figures be saved?')
    parser.set_defaults(figdir='/Users/Kitty/Documents/College/Summer2020/Simulations/VELA13_Simulations/0.56')

    args = parser.parse_args()
    return args

'''
def dir_arg(scale, sim):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='set parameters for VELA simulation analysis')

    parser.add_argument('--loc', metavar='loc', type=str, action='store',
                        help='where are the simulations located?')
    parser.set_defaults(loc='/Users/Kitty/Documents/College/Summer 2020/Simulations/{} Data/{}/10MpcBox_csf512_a{}0.d'.format(sim, scale, scale))

    parser.add_argument('--figdir', metavar='figdir', type=str, action='store',
                        help='where should figures be saved?')
    parser.set_defaults(figdir='/Users/Kitty/Documents/College/Summer 2020/Simulations/{} Simulations/{}'.format(sim, scale))
    
    arg = parser.parse_args()
    return arg
'''

# Set which snapshots the code will analyze and their proper parameters
args = parse_args()

sim = args.sim
scale = args.scale
loc = args.loc
ds = yt.load(loc)
figdir = args.figdir

pre_dict = args.pre_dict
gal_dict = np.load(pre_dict, allow_pickle='true', encoding='latin1')[()]

dict_loc = np.where(gal_dict['scale']==float(scale))
dict_ind = np.array(dict_loc[0])[0]

proj_width = args.proj_width

sim_dict = {}
sim_dict = {scale : {'scale', scale}}

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

redshift = round(float(ds.current_redshift), 2)
sim_dict.update([('redshift', redshift)])

# Set centers
cens = gal_dict['true_center']
gal_cen = cens[dict_ind]
cen_cen = yt.YTArray(gal_cen, 'kpc')
sim_dict.update([('center', cen_cen)])

# Set sphere
sphere_rad = args.sphere_rad
sp = ds.sphere(cen_cen, (sphere_rad, 'kpc'))

# Basic projections
if False:
    prj_x = yt.ProjectionPlot(ds, "x", "density")
    prj_x.save('{}/{}_Density_Projection_x-z={}.png'.format(figdir, sim, redshift))
    prj_y = yt.ProjectionPlot(ds, "y", "density")
    prj_y.save('{}/{}_Density_Projection_y-z={}.png'.format(figdir, sim, redshift))
    prj_z = yt.ProjectionPlot(ds, "z", "density")
    prj_z.save('{}/{}_Density_Projection_z-z={}.png'.format(figdir, sim, redshift))
    
# See where the galaxy ends at a particular snapshot by finding the radius at which 90% of the galaxy's stellar mass is enclosed
if True:
    dic = {}
    dic[('stars', 'particle_position_spherical_radius')] = False

    gal_rad = yt.create_profile(sp, ('stars', 'particle_position_spherical_radius'), ('stars', 'particle_mass'), weight_field = None, accumulation=True, logs = dic)
    gal_ext = gal_rad.x.to('kpc')
    gal_mass = gal_rad['stars', 'particle_mass'].to('Msun')
    max_mass = np.max(gal_mass)

    r90 = max_mass * 0.9
    r90_loc = np.max(np.where(gal_mass < r90))
    gal_r90 = gal_ext[r90_loc]
    mass_r90 = gal_mass[r90_loc]

    plt.plot(gal_ext, gal_mass)
    plt.plot(gal_r90, mass_r90, '*', markersize=12)
    plt.title('Accumulated Mass Profile for {} at z={}'.format(sim, redshift))
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Mass (Msun)')
    plt.savefig('{}/AccumulatedMassProfile{}z={}.png'.format(figdir, sim, redshift))
    plt.show()

    print(gal_r90, mass_r90)

    sim_dict.update([('r90', gal_r90)])
    sim_dict.update([('r90 mass', mass_r90)])
   
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
    sim_dict.update([('star L vector', star_ang_mom_norm)])

    edge_on_dir = np.random.randn(3)
    edge_on_dir -= edge_on_dir.dot(star_ang_mom_norm) * star_ang_mom_norm / np.linalg.norm(star_ang_mom_norm)**2
    sim_dict.update([('star edge-on', edge_on_dir)])
    
    # Calculating gas angular momentum
    gas_ang_mom_x = sp.quantities.total_quantity([("gas", "angular_momentum_x")])
    gas_ang_mom_y = sp.quantities.total_quantity([("gas", "angular_momentum_y")])
    gas_ang_mom_z = sp.quantities.total_quantity([("gas", "angular_momentum_z")])
    gas_ang_mom = yt.YTArray([gas_ang_mom_x, gas_ang_mom_y, gas_ang_mom_z])

    # Setting face-on and edge-on projection directions for the gas
    gas_ang_mom_tot = np.sqrt(sum(gas_ang_mom**2))
    gas_ang_mom_norm = gas_ang_mom / gas_ang_mom_tot
    sim_dict.update([('gas L vector', gas_ang_mom_norm)])

    edge_on_gas = np.random.randn(3)
    edge_on_gas -= edge_on_gas.dot(gas_ang_mom_norm) * gas_ang_mom_norm / np.linalg.norm(gas_ang_mom_norm)**2
    sim_dict.update([('gas edge-on', edge_on_gas)])

# Set cylinder
cyl_rad = np.ceil(gal_r90.value)
cyl_h = args.cyl_h
cyl = ds.disk(center = cen_cen, normal = star_ang_mom_norm, radius = (gal_r90), height = (cyl_h, 'kpc'))

# Making the projections: currently each one renders in a bit over half an hour
if False:
    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = star_ang_mom_norm, fields = ('deposit', 'stars_density'), center = cen_cen, width=(proj_width, 'kpc'), data_source = cyl)
        prj.save('{}/StarsFaceOn{}-z={}.png'.format(figdir, sim, redshift))

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = ('deposit', 'stars_density'), center = cen_cen, width=(proj_width, 'kpc'), data_source = cyl)
        prj.save('{}/StarsEdgeOn{}-z={}.png'.format(figdir, sim, redshift))

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), data_source = cyl)
        prj.save('{}/GasFaceOn{}-z={}.png'.format(figdir, sim, redshift))

    if True:
        prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_gas, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), data_source = cyl)
        prj.save('{}/GasEdgeOn{}-z={}.png'.format(figdir, sim, redshift))

# Making profile plots for velocity dispersion in the cylinder
if True:
    prof_gas = yt.create_profile(cyl, 'radius', ('gas', 'velocity_magnitude'), weight_field = ('gas', 'cell_mass'))
    gas_radius = prof_gas.x.to('kpc')
    std_gas = prof_gas.standard_deviation['gas', 'velocity_magnitude'].to('km/s')
    sim_dict.update([('gas radius', gas_radius)])
    sim_dict.update([('gas stdev', std_gas)])

    # The dictionary prevents stars from being log binned
    dic = {}
    dic[('stars', 'particle_position_cylindrical_radius')] = False
    prof_stars = yt.create_profile(cyl, ('stars', 'particle_position_cylindrical_radius'), ('stars', 'particle_velocity_cylindrical_z'), weight_field = ('stars', 'particle_mass'), logs = dic)
    star_radius = prof_stars.x.to('kpc')
    std_stars = prof_stars.standard_deviation['stars', 'particle_velocity_cylindrical_z'].to('km/s')
    sim_dict.update([('star radius', star_radius)])
    sim_dict.update([('star stdev', std_stars)])

    dic = {}
    dic[('new_stars', 'particle_position_cylindrical_radius')] = False
    prof_new =  yt.create_profile(cyl, ('new_stars', 'particle_position_cylindrical_radius'), ('new_stars', 'particle_velocity_cylindrical_z'), weight_field = ('new_stars', 'particle_mass'), logs = dic)
    std_new = prof_new.standard_deviation['new_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    sim_dict.update([('new star stdev', std_new)])

    dic = {}
    dic[('young_stars', 'particle_position_cylindrical_radius')] = False    
    prof_young =  yt.create_profile(cyl, ('young_stars', 'particle_position_cylindrical_radius'), ('young_stars', 'particle_velocity_cylindrical_z'), weight_field = ('young_stars', 'particle_mass'), logs = dic)
    std_young = prof_young.standard_deviation['young_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    sim_dict.update([('young star stdev', std_young)])

    dic = {}
    dic[('intermediate_stars', 'particle_position_cylindrical_radius')] = False
    prof_intermediate =  yt.create_profile(cyl, ('intermediate_stars', 'particle_position_cylindrical_radius'), ('intermediate_stars', 'particle_velocity_cylindrical_z'), weight_field = ('intermediate_stars', 'particle_mass'), logs = dic)
    std_intermediate = prof_intermediate.standard_deviation['intermediate_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    sim_dict.update([('intermediate star stdev', std_intermediate)])

    dic = {}
    dic[('old_stars', 'particle_position_cylindrical_radius')] = False
    prof_old =  yt.create_profile(cyl, ('old_stars', 'particle_position_cylindrical_radius'), ('old_stars', 'particle_velocity_cylindrical_z'), weight_field = ('old_stars', 'particle_mass'), logs = dic)
    std_old = prof_old.standard_deviation['old_stars', 'particle_velocity_cylindrical_z'].to('km/s')
    sim_dict.update([('old star stdev', std_old)])

    # Plot the velocity dispersion for stars distinguished by age
    plt.plot(gas_radius, std_gas, color='green', label='Gas')
    plt.plot(star_radius, std_stars, linewidth=3, color='black', label='All Stars')
    plt.plot(star_radius, std_new, color='blue', label='New Stars')
    plt.plot(star_radius, std_young, color='cyan', label='Young Stars')
    plt.plot(star_radius, std_intermediate, color='magenta',  label='Intermediate Stars')
    plt.plot(star_radius, std_old, color='red', label='Old Stars')
    plt.title('Velocity Dispersion for {} at z={}'.format(sim, redshift))
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Velocity [km/s]')
    plt.xlim([0, np.max(star_radius)])
    plt.ylim([0, np.max(std_stars)])
    plt.legend()
    plt.savefig('{}/VelocityDispersion{}-z={}.png'.format(figdir, sim, redshift))
    plt.show()

# Creating vertical mass profiles to determine scale height
if False:
    total_mass = np.zeros(shape=(int(cyl_rad)))
    total_gas = np.zeros(shape=(int(cyl_rad)))
    total_dm = np.zeros(shape=(int(cyl_rad)))
    sigma_shell = np.zeros(shape=(int(cyl_rad)))
    sigma_mass = np.zeros(shape=(int(cyl_rad)))
    sigma_mg = np.zeros(shape=(int(cyl_rad)))
    sigma_all = np.zeros(shape=(int(cyl_rad)))
    h_z = np.zeros(shape=(int(cyl_rad)))

    dic = {}
    dic[('stars', 'particle_position_cylindrical_z')] = False

    inc = args.inc
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
        h_z[int(shell)] = m.z_0.value
        
        # Calculating sigma from eqn 1 in the proposal; differentiating by stars, stars and gas, and complete dynamical mass
        const = args.const
        G_conv = G.to('km3/(kg s2)')
        z_0 = (m.z_0 * u.kpc).to('km')
        outer_conv = (outer_rad * u.kpc).to('km')
        inner_conv = (inner_rad * u.kpc).to('km')
        proj_area = np.pi * (outer_conv**2 - inner_conv**2)
        sigma_dyn = ((mass_quant * u.Msun).to('kg') + (gas_quant * u.Msun).to('kg') + (dm_quant * u.Msun).to('kg')) / (proj_area)
        mass_dens = (mass_quant * u.Msun).to('kg') / (proj_area)
        mass_gas = ((mass_quant * u.Msun).to('kg') + (gas_quant * u.Msun).to('kg')) / (proj_area)
        sigma_mass_calc = np.sqrt(const*np.pi*G_conv*z_0*mass_dens)
        sigma_mg_calc = np.sqrt(const*np.pi*G_conv*z_0*mass_gas)
        sigma_dyn_calc = np.sqrt(const*np.pi*G_conv*z_0*sigma_dyn)
        sigma_mass[int(shell)] = sigma_mass_calc.value
        sigma_mg[int(shell)] = sigma_mg_calc.value
        sigma_all[int(shell)] = sigma_dyn_calc.value

        # Plotting the mass profile for each shell
        plt.plot(npheight, npaccum, '*', label='Mass Profile')
        plt.plot(npheight, m(npheight), label='Fit')
        plt.title('Mass Profile of {}-z={}: Shell Center {}kpc, Width {}kpc'.format(sim, redshift, shell_cen, inc))
        plt.xlabel('Vertical Height [kpc]')
        plt.ylabel('Mass [Msun]')
        plt.legend()
        plt.savefig('{}/MassProfile{}-z={}Center{}Width{}.png'.format(figdir, sim, redshift, shell_cen, inc))
        plt.show()
        
    # Plotting the predicted and measured vertical velocity dispersion
    one_to_one = np.arange(0, np.max(sigma_shell))
    plt.plot(sigma_all, sigma_shell, '*', color='purple', label=r'$\Sigma_{dyn}$')
    plt.plot(sigma_mg, sigma_shell, '*', color='orchid', label='Stars and Gas')
    plt.plot(sigma_mass, sigma_shell, '*', color='plum', label='Stars')
    plt.plot(one_to_one, one_to_one, color='teal', label='One-to-One')
    plt.title('Observed vs. Theoretical Vertical Velocity Dispersion for {} at z={}'.format(sim, redshift))
    plt.xlabel('Predicted Vertical Velocity Dispersion (km/s)')
    plt.ylabel('Measured Vertical Velocity Dispersion (km/s)')
    plt.legend()
    plt.savefig('{}/ObsVsTheo-{}-z={}.png'.format(figdir, sim, redshift))
    plt.show()
    
    sim_dict.update([('sigma - stars, gas, dm', sigma_all)])
    sim_dict.update([('sigma - stars, gas', sigma_mg)])
    sim_dict.update([('sigma - stars', sigma_mass)])
    sim_dict.update([('sigma - measured', sigma_shell)])
    sim_dict.update([('scale height', h_z)])

if False:
    rad_comp = np.arange(1, cyl_rad+1)
        
    # Plotting the comparisons of vertical velocity dispersion via different methods
    plt.plot(star_radius, std_stars, '*', color='black', label='Standard Deviation Projection')
    plt.plot(rad_comp, sigma_shell, label='Shell Sigma')
    plt.plot(rad_comp, sigma_all, '*', color='purple', label=r'Theoretical $\sigma_z$ with $Sigma_{dyn}$')
    plt.plot(rad_comp, sigma_mg, '*', color='orchid', label=r'Theoretical $\sigma_z$ with Stars and Gas')
    plt.plot(rad_comp, sigma_mass, '*', color='plum', label=r'Theoretical $\sigma_z$ with Stars')
    plt.title('Comparisons for {} at z = {}'.format(sim, redshift))
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Vertical Velocity Dispersion (km/s)')
    plt.legend()
    plt.savefig('{}/Comparisons-{}-z={}.png'.format(figdir, sim, redshift))
    plt.show()

if False:
    np.save('{}/{}-z={}.npy'.format(figdir, sim, redshift), sim_dict)