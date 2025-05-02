import yt
import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import *

Msun = 1.99e33
seconds_in_yr = 3.15e7

def compute_fstar_high():
    m_star_high = 8.0 * Msun
    m_imf_min = 0.08 * Msun
    m_imf_max = 120.0 * Msun
    m_imf_break = 1.0 * Msun
    imf_disp = 0.55
    imf_mu = log10(0.2 * Msun)
    alpha = 2.35
    log_m_imf_break = log10(m_imf_break)
    log_m_imf_min = log(m_imf_min)
    arg_m_imf_break = (log_m_imf_break - imf_mu) / (sqrt(2.) * imf_disp);
    arg_m_imf_min = (log_m_imf_min - imf_mu) / (sqrt(2.) * imf_disp);
    pow_alpha_m_imf_max = pow(m_imf_max, 2.0 - alpha);
    pow_alpha_m_imf_break = pow(m_imf_break, 2.0 - alpha);
    pow_alpha_m_star_high = pow(m_star_high, 2.0 - alpha);
    norm_ratio = pow(m_imf_break, (1 - alpha)) * imf_disp * sqrt(2.0 * pi) / exp(-(arg_m_imf_break * arg_m_imf_break));
    total_star_mass = ((2. - alpha) * norm_ratio) * exp(imf_mu + imf_disp * imf_disp / 2) * (erf(arg_m_imf_break - imf_disp / sqrt(2.)) - erf(arg_m_imf_min - imf_disp / sqrt(2.))) + pow_alpha_m_imf_max - pow_alpha_m_imf_break
    mass_highmass_stars = pow_alpha_m_imf_max - pow_alpha_m_star_high
    fstar_high = mass_highmass_stars / total_star_mass
    return fstar_high

if __name__ == "__main__":
    ## make histogram for each plotfile
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfiles", nargs='*')
    args = parser.parse_args()

    for i, my_plotfile in enumerate(args.plotfiles):
        # load data
        ds = yt.load(my_plotfile)
        field_prefix, field_name = zip(*ds.field_list)

        if 'StochasticStellarPop_particles' not in field_prefix: 
            continue   

        ad = ds.all_data()
        mass = ad[('StochasticStellarPop_particles', 'particle_real_comp0')]
        vx = ad[('StochasticStellarPop_particles', 'particle_real_comp1')]
        vy = ad[('StochasticStellarPop_particles', 'particle_real_comp2')]
        vz = ad[('StochasticStellarPop_particles', 'particle_real_comp3')]
        birth_time = ad[('StochasticStellarPop_particles', 'particle_real_comp4')].value
        death_time = ad[('StochasticStellarPop_particles', 'particle_real_comp5')].value
        stage = ad[('StochasticStellarPop_particles', 'particle_int_comp0')].value
        
        # filter by age
        print(f"current time: {ds.current_time.value/seconds_in_yr/1.0e6:f} Myr")
        age = np.ones_like(birth_time) * ds.current_time.value - birth_time
        age_cut = age < (3.0e6 * seconds_in_yr)
        mass = mass[age_cut]
        stage = stage[age_cut]
        print(f"number of young stars (< 3 Myr): {len(mass)}")
        
        # sum mass below, above the mass cut
        max_death_time = np.max(death_time)
        print(f"max death time: {max_death_time:e}")
        is_composite = (death_time > 1e300) # this is a hack
        print(f"number of low-mass particles (death_time > 1e300): {np.count_nonzero(is_composite)}")

        total_mass = np.sum(mass)
        total_low_mass = np.sum(mass[is_composite])
        mean_low_mass = np.mean(mass[is_composite])
        median_low_mass = np.median(mass[is_composite])
        total_high_mass = np.sum(mass[~is_composite])
        mean_high_mass = np.mean(mass[~is_composite])
        median_high_mass = np.median(mass[~is_composite])
        print(f"total mass of low mass star particles: {total_low_mass / Msun:e} Msun")
        print(f"mean mass of low mass star particles: {mean_low_mass / Msun:.3f} Msun")
        print(f"median mass of low mass star particles: {median_low_mass / Msun:.3f} Msun")
        print(f"total mass of high mass stars: {total_high_mass / Msun:e} Msun")
        print(f"mean mass of high mass stars: {mean_high_mass / Msun:.3f} Msun")
        print(f"median mass of high mass stars: {median_high_mass / Msun:.3f} Msun")
        print(f"high mass / total mass: {total_high_mass / total_mass:.3f}")
        print(f"(expected) high mass / total mass: {compute_fstar_high():.3f}")

        # filter out low-mass particles
        mass = mass[~is_composite]
        
        # make histogram of stellar masses
        mass_in_Msun = mass / Msun

        min_mass = 0.1 # Msun
        max_mass = 120. # Msun
        nbins = 20
        my_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), nbins)
        counts, bin_edges = np.histogram(mass_in_Msun.value, bins=my_bins, density=False) # returns counts
        bin_cen = 0.5*(bin_edges[:-1] + bin_edges[1:])
        dM = bin_edges[1:] - bin_edges[:-1]
        dN_dM = counts / dM
        dNdM_err = np.sqrt(counts) / dM

        alpha = -2.35 # Salpeter dN/dM power-law slope
        norm = 1e6 # arbitrary
        m = np.logspace(0, np.log10(max_mass), nbins)
        plt.title(f"t = {ds.current_time.value / 1.0e6 / seconds_in_yr:.2f} Myr")
        plt.plot(m, norm * m**alpha, '--', label='Salpeter IMF')
        plt.errorbar(bin_cen, dN_dM, yerr=dNdM_err, label='stellar masses')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$dN/dM$')
        plt.xlabel(r'stellar mass ($M_{\odot}$)')
        plt.tight_layout()
        plt.savefig(f"stellar_mf_{i:05}.png")
        plt.clf()
