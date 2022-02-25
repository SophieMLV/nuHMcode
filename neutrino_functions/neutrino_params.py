"""
functions for the fraction of clustered neutrinos and dictionary for neutrino parameters
"""

#packages needed
import numpy as np
from scipy import interpolate, misc, optimize

#own pachages
from cosmology.basic_cosmology import *
from halo_model.cold_density_profile import *
from halo_model.halo_bias import *
from halo_model.halo_mass_function import *
from halo_model.neutrino_density_profile import *




#M_cut for cdm halos for which we found no neutrinos halos around cdm halos with mass < M_cut see https://arxiv.org/abs/1410.6813 eq 4.19
def func_M_cut(cosmo_dic, LCDM = True):
    """
    M_cut for cdm halos for which we found no neutrinos halos around 
    cdm halos with mass < M_cut see https://arxiv.org/abs/1410.6813 eq 4.19
    returns the mass of cdm+b halos for which neutribos no longer cluster
    in units of M_sol/h
    """
    Omega_cb_0 = cosmo_dic['Omega_d_0'] + cosmo_dic['Omega_b_0']
    if LCDM == True:
        Delta_vir = func_Delta_vir_LCDM(cosmo_dic, Omega_cb_0)
    else:
        Delta_vir = func_Delta_vir(cosmo_dic)
    
    def func_find_root(M):
        return func_neutrino_halo_mass(M, cosmo_dic, M_cut = 1, LCDM = LCDM) - \
               0.12 * func_rho_m_comp(cosmo_dic, cosmo_dic['Omega_nu_0']) \
               /func_rho_m_comp(cosmo_dic, Omega_cb_0) * M / Delta_vir
    return float(optimize.root(func_find_root, 1e13).x)



#create a dictionary with all important parameter for neutrinoss


def func_neutrino_param_dic(M_halo, k_sigma, PS_sigma, cosmo_dic, LCDM=True):
    """
    generate dictionary with parameetrs for neutrinos
    """
    M_cut = func_M_cut(cosmo_dic, LCDM = LCDM)
    M_arr_int = np.geomspace(M_cut, np.max(M_halo), num=len(M_halo))
    HMF = func_halo_mass_function(M_arr_int, k_sigma, PS_sigma, cosmo_dic, cosmo_dic['Omega_db_0'], cosmo_dic['Omega_db_0'], LCDM = LCDM) 
    M_nu = func_neutrino_halo_mass(M_arr_int, cosmo_dic, M_cut, LCDM = LCDM)
    bias = func_halo_bias(M_arr_int, k_sigma, PS_sigma, cosmo_dic, cosmo_dic['Omega_db_0'], LCDM = LCDM)
    integrand_arr = HMF * M_nu * bias 
    frac_nu_cluster = integrate.simps(y= integrand_arr, x = M_arr_int)/func_rho_comp_0(cosmo_dic['Omega_nu_0'])
    
    #generate dictionary
    neutrino_param_dic = {}
    neutrino_param_dic['M_cut'] = M_cut
    neutrino_param_dic['M_int'] = M_arr_int
    neutrino_param_dic['M_nu'] = np.array(M_nu) 
    neutrino_param_dic['frac_cluster'] = frac_nu_cluster
    return neutrino_param_dic
    
