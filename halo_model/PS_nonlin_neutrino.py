"""functions for my full massive neutrino halo model"""

#packages needed
import numpy as np
from scipy import integrate
#own pachages
from cosmology.basic_cosmology import *
from halo_bias import *
from halo_mass_function import *
from cold_density_profile import *
from neutrino_density_profile import *
from PS_nonlin_cold import *



#### halo model with massive neutrinos, one function for everything
def func_full_halo_model_nu_sophie(M, k, PS_cold, PS_nu, k_sigma, PS_sigma, cosmo_dic, hmcode_dic, neutrino_dic, 
                            alpha = False, eta_given = False, LCDM=True, nu_one_halo=False, one_halo_damping = False, two_halo_damping = False):
    """ 
    My Full Halo Model with massive neutrinos, see my masterhesis eq TBC to see the full formula
    all modifications form HMcode2020 https://arxiv.org/abs/2009.01858 can be used (if wanted)
    by default use only the damping in the one halo term on large scales (for all parts) 
    to ensure the correct behaviour on large scales
    M in solar_mass/h
    in power_spec_dic and power_spec_dic_sigma the PS and k's are stored 
    and all units are either in (Mpc/h)^3 or h/Mpc
    NOTE: Two PS dicionaries are needed bacause we have to be carefull with the calulation of sigma
    returns total non-linear matter power spectrum in (Mpc/h)^3 at k
    """
    ####Cold matter term#####
    PS_cold_nonlin = func_non_lin_PS_matter(M, k, PS_cold, k_sigma, PS_sigma, cosmo_dic, hmcode_dic, cosmo_dic['Omega_db_0'], cosmo_dic['Omega_db_0'], 
                                            alpha = alpha, eta_given = eta_given, LCDM=LCDM, nu_one_halo=False, 
                                            one_halo_damping = one_halo_damping, two_halo_damping = two_halo_damping)[0]
    
    ##compute all ingridients for the diff halo model parts##
    halo_mass_func_arr = func_halo_mass_function(M, k_sigma, PS_sigma, cosmo_dic, 
                                                 cosmo_dic['Omega_db_0'], cosmo_dic['Omega_db_0'], LCDM = LCDM) # for the integral over M ~[0, inf]
    halo_mass_func_arr_2 = func_halo_mass_function(neutrino_dic['M_int'], k_sigma, PS_sigma, cosmo_dic, 
                                                   cosmo_dic['Omega_db_0'], cosmo_dic['Omega_db_0'], LCDM = LCDM) # for the integral over reduced array [M_cut, inf]

    
    dens_profile_cold_arr = func_dens_profile_kspace(M, k, k_sigma, PS_sigma, cosmo_dic, hmcode_dic, cosmo_dic['Omega_db_0'], 
                                                     cosmo_dic['Omega_db_0'], eta_given = eta_given, LCDM=LCDM) # for the integral over M ~[0, inf]
    dens_profile_cold_arr_2 = func_dens_profile_kspace(neutrino_dic['M_int'], k, k_sigma, PS_sigma, cosmo_dic, hmcode_dic, cosmo_dic['Omega_db_0'], 
                                                       cosmo_dic['Omega_db_0'], eta_given = eta_given, LCDM=LCDM) # for the integral over reduced array [M_cut, inf]    
    dens_profile_nu_arr_2 = func_density_profile_kspace_neutrino(neutrino_dic['M_int'], k, cosmo_dic, neutrino_dic['M_cut'], LCDM = LCDM) # this integral in only in the reduced one

    
    halo_bias_arr = func_halo_bias(M, k_sigma, PS_sigma, cosmo_dic, cosmo_dic['Omega_db_0'], LCDM = LCDM) # for the integral over M ~[0, inf]
    halo_bias_arr_2 = func_halo_bias(neutrino_dic['M_int'], k_sigma, PS_sigma, cosmo_dic, cosmo_dic['Omega_db_0'], LCDM = LCDM) # for the integral over reduced array [M_cut, inf]

    
    ##cross one halo term##
    integrand_arr_one_halo_cross =  neutrino_dic['M_int'][:, None] * neutrino_dic['M_nu'][:, None] * halo_mass_func_arr_2[:, None] \
                                    * dens_profile_cold_arr_2 * dens_profile_nu_arr_2 # integral over reduced array [M_cut, inf]
    one_halo_term_cross = integrate.simps(integrand_arr_one_halo_cross, x = neutrino_dic['M_int'], axis = 0)\
                          / (func_rho_comp_0(cosmo_dic['Omega_db_0']) * func_rho_comp_0(cosmo_dic['Omega_nu_0']) * neutrino_dic['frac_cluster']) 
    if one_halo_damping == True:
        one_halo_term_cross = (k/hmcode_dic['k_star'])**4 / (1+(k/hmcode_dic['k_star'])**4) \
                          * one_halo_term_cross
    else:
        one_halo_term_cross = one_halo_term_cross
    
    ##cross two halo term##
    integrand_arr_two_halo_cross = M[:, None] * halo_mass_func_arr[:, None] * halo_bias_arr[:, None] * dens_profile_cold_arr  # integral over M ~[0, inf]
    integrand_arr_two_halo_2_cross = neutrino_dic['M_nu'][:, None] * halo_mass_func_arr_2[:, None] * halo_bias_arr_2[:, None] * dens_profile_nu_arr_2 # integral over reduced array [M_cut, inf]
    
    #summand to take care of nummericals issues of the integral, see appendix A in https://arxiv.org/abs/2005.00009
    summand2_cross = func_dens_profile_kspace(np.min(M), k,  k_sigma, PS_sigma, cosmo_dic, hmcode_dic,  cosmo_dic['Omega_db_0'], cosmo_dic['Omega_db_0'], eta_given = eta_given, LCDM=LCDM) \
                     * ( 1 - integrate.simps(M[:, None] * halo_mass_func_arr[:, None] * halo_bias_arr[:, None], x = M, axis = 0) \
                     / func_rho_comp_0(cosmo_dic['Omega_db_0'])) 
    factor2_cross = integrate.simps(integrand_arr_two_halo_cross, x = M, axis = 0) / func_rho_m_comp(cosmo_dic, cosmo_dic['Omega_db_0']) + summand2_cross   
    
    two_halo_term_cross = PS_cold \
                          * factor2_cross \
                          * integrate.simps(integrand_arr_two_halo_2_cross, x = neutrino_dic['M_int'], axis = 0) \
                          / (func_rho_comp_0(cosmo_dic['Omega_nu_0']) * neutrino_dic['frac_cluster']) 
    
    #######Cross term #####
    PS_total_cross = neutrino_dic['frac_cluster']*(one_halo_term_cross+two_halo_term_cross)[0, :] + (1-neutrino_dic['frac_cluster'])*np.sqrt(PS_nu * PS_cold_nonlin)
    
    ## neutrino one and two halo term####
    integrand_arr_one_halo_nu =  neutrino_dic['M_nu'][:, None]**2 * halo_mass_func_arr_2[:, None] * np.array(dens_profile_nu_arr_2)**2 #for the integral over reduced array [M_cut, inf] 
    integrand_arr_two_halo_nu = neutrino_dic['M_nu'][:, None] * halo_mass_func_arr_2[:, None] * halo_bias_arr_2[:, None] * dens_profile_nu_arr_2 #for the integral over reduced array [M_cut, inf]
    
    one_halo_term_nu =  integrate.simps(integrand_arr_one_halo_nu, x = neutrino_dic['M_int'], axis = 0) \
                        / (func_rho_comp_0(cosmo_dic['Omega_nu_0']) * neutrino_dic['frac_cluster'])**2 
    if one_halo_damping == True:
        one_halo_term_cross = (k/hmcode_dic['k_star'])**4 / (1+(k/hmcode_dic['k_star'])**4) \
                              * one_halo_term_nu
    else:
        one_halo_term_cross = one_halo_term_nu
    
    two_halo_term_nu =  PS_cold * integrate.simps(integrand_arr_two_halo_nu, x = neutrino_dic['M_int'], axis = 0)**2  \
                        / (func_rho_comp_0(cosmo_dic['Omega_nu_0']) * neutrino_dic['frac_cluster'])**2 
    
    ####### neutrino term ######
    PS_total_nu = neutrino_dic['frac_cluster']**2 * (one_halo_term_nu + two_halo_term_nu) \
                  + 2*(1- neutrino_dic['frac_cluster'])* neutrino_dic['frac_cluster']*np.sqrt((one_halo_term_nu + two_halo_term_nu)*PS_nu) \
                  + (1- neutrino_dic['frac_cluster'])**2 * PS_nu  
    
    
    #####stick all together to the total matter non-lin PS####
    PS_total_matter = (cosmo_dic['Omega_db_0']/cosmo_dic['Omega_m_0'])**2 * PS_cold_nonlin \
                      + 2*cosmo_dic['Omega_db_0']*cosmo_dic['Omega_nu_0']/cosmo_dic['Omega_m_0']**2 * PS_total_cross \
                    + (cosmo_dic['Omega_nu_0']/cosmo_dic['Omega_m_0'])**2 * PS_total_nu
    return PS_total_matter, PS_cold_nonlin, PS_total_cross, PS_total_nu