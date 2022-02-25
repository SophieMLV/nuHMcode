"""
functions for neutrino density profile
"""


#packages needed
import numpy as np
from scipy import integrate
#own pachages
from cosmology.basic_cosmology import *
from cosmology.overdensities import *
from cold_density_profile import *

#use the halo density profile in https://arxiv.org/abs/1212.4855v2 with the given parameters. 
####-------here we only can have the total mass of neutrinos of 0.6eV and 0.3eV------#####

#alpha paremeter see paper figure 10
def func_alpha_nu_profile(M, cosmoc_dic):
    """
    parameter for the neutrino density profile from
    https://arxiv.org/abs/1212.4855v2 figure 10 and eq3.5
    M in solar_mass/h
    """
    def func_transition(x):
        return 0.5 + 0.5 * np.tanh((x-10**(13.5))/1e13)
    
    if cosmoc_dic['mnu'] == 0.6:
        return func_transition(M) * ((-4.62449381390627) + 0.18610288729325442 * np.log(M)) + (1-func_transition(M)) * (-3.64 + 0.15 * np.log(M))
    elif cosmoc_dic['mnu'] == 0.3:
        return func_transition(M) * ((-6.689848499879118) + 0.23790527192214206 * np.log(M)) + (1-func_transition(M)) * (-2.06 + 0.09 * np.log(M))
    else:
        print('WARNING: total neutrinos mass is either 0.6eV nor 0.3eV')
        return 'total neutrinos mass is either 0.6eV nor 0.3eV'
        

#kappa paremter se paper figure 10 an eq. 3.5  
def func_kappa_nu_profile(M, cosmo_dic):
    """
    parameter for the neutrino density profile from
    https://arxiv.org/abs/1212.4855v2 figure 10 and eq3.5
    M in solar_mass/h
    """
    if cosmo_dic['mnu'] == 0.6:
        return (0.24 + 1.144 * 10**(-20) * M**(1.7))*(1e-3)**func_alpha_nu_profile(M, cosmo_dic) #last factor due to dimensions
    elif cosmo_dic['mnu'] == 0.3:
        return (0.19 + 3.242 * 10**(-19) * M**(1.5))*(1e-3)**func_alpha_nu_profile(M, cosmo_dic )#last factor due to dimensions
    else:
        print('WARNING: total neutrinos mass is either 0.6eV nor 0.3eV')
        return 'total neutrinos mass is either 0.6eV nor 0.3eV'
    
    
#rho_c paremter se paper figure 10 an eq. 3.4  
def func_rhoc_nu_profile(M, cosmoc_dic):
    """
    parameter for the neutrino density profile from
    https://arxiv.org/abs/1212.4855v2 figure 10 and eq3.4
    M in solar_mass/h
    """
    if cosmoc_dic['mnu'] == 0.6:
        return 3.7478 * 10**(-8) * M**(0.64)
    elif cosmoc_dic['mnu'] == 0.3:
        return 6.056 * 10**(-8) * M**(0.58)
    else:
        print('WARNING: total neutrinos mass is either 0.6eV nor 0.3eV')
        return 'total neutrinos mass is either 0.6eV nor 0.3eV'

        
#r_c paremter se paper figure 10 an eq. 3.4  be carefull:in the paper in units of kpc/h
def func_rc_nu_profile(M, cosmoc_dic):
    """
    parameter for the neutrino density profile from
    https://arxiv.org/abs/1212.4855v2 figure 10 and eq3.4
    NOTE:in the reference we habe kpc/h but returned units here
    are Mpc/h
    M in solar_mass/h
    """
    if cosmoc_dic['mnu'] == 0.6:
            return 2.046 * 10**(-4) * M**(0.43) * 10**(-3)
    elif cosmoc_dic['mnu'] == 0.3:
        return 4.029 * 10**(-8) * M**(0.68) * 10**(-3)
    else:
        print('WARNING: total neutrinos mass is either 0.6eV nor 0.3eV')
        return 'total neutrinos mass is either 0.6eV nor 0.3eV'
        
        
    
#neutrino density profile as rho_nu^halo = delta_nu bar(rho_nu)
def func_density_profile_neutrino(M, r, cosmo_dic):
    """
    r in units of Mpc/h and M in solar_mass/h
    returns the neutrino density profile which depends on the mass of the CDM halo
    in units of solar_mass/Mpc^3*h^2 at radius r
    """
    if isinstance(M, (int, float)) == True:
        if M < 10**(13.5):
            delta_neutrino = func_kappa_nu_profile(M, cosmo_dic) / (r**(func_alpha_nu_profile(M, cosmo_dic)) )
            return delta_neutrino * func_rho_comp_0(cosmo_dic['Omega_nu_0'])
        else:
            delta_neutrino = func_rhoc_nu_profile(M, cosmo_dic) / (1 + (r/func_rc_nu_profile(M, cosmo_dic))**(func_alpha_nu_profile(M, cosmo_dic)) ) 
            return delta_neutrino * func_rho_comp_0(cosmo_dic['Omega_nu_0']) #func_rho_m_comp(cosmo_dic, cosmo_dic['Omega_nu_0'])
    else:
        dens_arr = []
        for m in M:
            if m < 10**(13.5):
                dens_arr.append(func_kappa_nu_profile(m, cosmo_dic) / \
                (r**(func_alpha_nu_profile(m, cosmo_dic))) * func_rho_comp_0(cosmo_dic['Omega_nu_0']))
            else:
                dens_arr.append(func_rhoc_nu_profile(m, cosmo_dic) / \
                (1 + (r/func_rc_nu_profile(m, cosmo_dic))**(func_alpha_nu_profile(m, cosmo_dic))) * func_rho_comp_0(cosmo_dic['Omega_nu_0']))
        return dens_arr





#mass neutrino halo:
def func_neutrino_halo_mass(M, cosmo_dic, M_cut, LCDM = True):
    """
    M in units of solar_mass/h
    return mass of neutrino halo around CDM halo
    in units of solar_mass/h
    """
    Omega_cb_0 = cosmo_dic['Omega_db_0']
    Omega_cb_0 = cosmo_dic['Omega_db_0']
    r_v = func_r_vir(M, cosmo_dic, Omega_cb_0, LCDM = LCDM)
    
    if isinstance(M, (int, float)) == True:
        r_arr = np.geomspace(1e-10, r_v, num=1000)    
        integrand = r_arr**2* func_density_profile_neutrino(M, r_arr, cosmo_dic)
        return 4 * np.pi * integrate.simps(y = integrand, x = r_arr)
    else:
        mass_arr = []
        for idx, m in enumerate(M):
            r_arr = np.geomspace(1e-10, r_v[idx], num=1000)
            integrand = r_arr**2 * func_density_profile_neutrino(m, r_arr, cosmo_dic)
            mass_arr.append(4 * np.pi * integrate.simps(y = integrand, x = r_arr) )
        return mass_arr

    

#neuntrino denity profile kspace 
def func_density_profile_kspace_neutrino(M, k, cosmo_dic, M_cut, LCDM = True):
    """
    r in units of Mpc/h and M in solar_mass/h
    return kspace denist profile for the neutrino halo
    the normalised density profile is demensionles
    """
    halo_mass = func_neutrino_halo_mass(M, cosmo_dic, M_cut, LCDM = LCDM)
    Omega_cb_0 = cosmo_dic['Omega_db_0']
    r_v = func_r_vir(M, cosmo_dic, Omega_cb_0, LCDM = LCDM)
    
    if isinstance(M, (int, float)) == True:
        r_arr = np.geomspace(1e-10, r_v, num=1000)
        integrand = r_arr**2 * np.sin(np.outer(k, r_arr)) / (np.outer(k, r_arr)) * func_density_profile_neutrino(M, r, cosmo_dic)
        return 4 * np.pi / halo_mass * integrate.simps(y= integrand, x = r_arr, axis = -1)
    
    else:
        dens_profile_kspace_arr = []
        for idx, m in enumerate(M):
            r_arr = np.geomspace(1e-10, r_v[idx], num=1000)
            dens_prof_ks_nu = func_density_profile_neutrino(m, r_arr, cosmo_dic) 
            integrand = r_arr**2 * np.sin(np.outer(k, r_arr)) / (np.outer(k, r_arr)) * dens_prof_ks_nu
            dens_profile_kspace_arr.append(4*np.pi * integrate.simps(y=integrand, x = r_arr, axis = -1)/ halo_mass[idx])    
        return dens_profile_kspace_arr