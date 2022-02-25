"""
functions for the critical overdensity in spherical/ellipsoidal collaps and halo overdensity
"""

#packages
import numpy as np
from scipy import integrate

#own packages
from basic_cosmology import *


def func_D_z_unnorm(z, cosmo_dic):
    """
    returns unnormalised grwoth function 
    """
    def integrand(x):
        return (1+x)/func_E_z(x, cosmo_dic)**3
    
    factor = 5*cosmo_dic['Omega_m_0']/2
    D = factor * func_E_z(z, cosmo_dic) * integrate.quad(integrand, z, np.inf)[0]
    return D

def func_D_z_norm(z, cosmo_dic):
    """
    returns normlised grwoth function, ie D(0)= 1
    this is used to scale the power spectrum for diffrent z's
    """
    normalisation = func_D_z_unnorm(0., cosmo_dic)
    growth = func_D_z_unnorm(z, cosmo_dic)
    
    return growth/normalisation

def func_G_z(cosmo_dic):
    """
    returns integrated grwoth function
    this is needed for the critical density and overdensity computed as in 
    HMCode2020: https://arxiv.org/abs/2009.01858
    """
    z = cosmo_dic['z']
    def integrand(x):
        return func_D_z_unnorm(x, cosmo_dic)/(1+x)
    
    integral = integrate.quad(integrand, z, np.inf)[0] 
    return integral



def func_quadratics(x, y, i):
    """
    function critical overdensity and virial overdensity from Mead 2020 eq.
    """
    P_matrix = [[ -0.0069, -0.0208, 0.0312, 0.0021], [0.0001, -0.0647, -0.0417, 0.0646], 
                [-0.79, -10.17, 2.51, 6.51], [-1.89, 0.38, 18.8, -15.87]]
    return P_matrix[i][0] + P_matrix[i][1] *(1-x) + P_matrix[i][2] *(1-x)**2 + P_matrix[i][3] * (1-y)


def func_delta_c(cosmo_dic):
    """
    critical density for spherical/ellepsoidal collapse as is
    HMCode2020: https://arxiv.org/abs/2009.01858
    """
    alpha1, alpha2, aplha3, alpha4 = 1., 0., 1., 2.
    
    z = cosmo_dic['z']
    g = func_D_z_unnorm(z, cosmo_dic)
    G = func_G_z(cosmo_dic)
    
    summand1 = func_quadratics(g * (1+z), G * (1+z), 0) * np.log10(func_Omega_comp_z(cosmo_dic, cosmo_dic['Omega_m_0']))**alpha1
    summand2 = func_quadratics(g * (1+z), G * (1+z), 1) * np.log10(func_Omega_comp_z(cosmo_dic, cosmo_dic['Omega_m_0']))**alpha2
    
    factor1 = 1 + summand1 + summand2
    #at the moment neutrinos dont cluster at all --> propably change fragtion  Omega_nu_value/Omega_m_value
    factor2 = 1.686 * (1-0.041 * cosmo_dic['Omega_nu_0']/cosmo_dic['Omega_m_0'])
    
    delta_c = factor1 * factor2
    return delta_c

def func_delta_c_LCDM(cosmo_dic):
    """
    returns critical denity for spherical/ellepsoidal collapse for LCDM cosmos
    """                    
    return 1.686 #* func_Omega_m_z(cosmo_dic)**(0.0055)

def func_Delta_vir(cosmo_dic):
    """
    halo overdensity as in
    HMCode2020: https://arxiv.org/abs/2009.01858
    """                  
    alpha1, alpha2, alpha3, alpha4 = 1., 0., 1., 2.
    
    z = cosmo_dic['z']
    g = func_D_z_unnorm(z, cosmo_dic)
    G = func_G_z(cosmo_dic)
    
    summand1 = func_quadratics(g *(1+z), G * (1+z), 2) * np.log10(func_Omega_comp_z(cosmo_dic, cosmo_dic['Omega_m_0']))**alpha3
    summand2 = func_quadratics(g *(1+z), G * (1+z), 3) * np.log10(func_Omega_comp_z(cosmo_dic, cosmo_dic['Omega_m_0']))**alpha4
    
    factor1 = 1 + summand1 + summand2
    #at the moment neutrinos dont cluster at all --> propably change fraction  Omega_nu_value/Omega_m_value
    factor2 = 177.7 * (1+0.763 * cosmo_dic['Omega_nu_0']/cosmo_dic['Omega_m_0'])
    
    Delta_v = factor1 * factor2
    return Delta_v
    
def func_Delta_vir_LCDM(cosmo_dic, Omega_0):
    """
    halo overdensity for LCDM comos,
    make the change, that only matter of the type 
    Omega_0 is take into accound for the overdensity
    """
    x = func_Omega_comp_z(cosmo_dic, Omega_0) -1
    #x = func_Omega_comp_z(cosmo_dic, cosmo_dic['Omega_m_0']) -1
    return (18*np.pi**2 + 82*x - 39*x**2) / (x+1)

def func_r_vir(M, cosmo_dic, Omega_0, LCDM=True):
    """
    M in solar_mass/h where M is matter of the type Omega_0
    returns comoving virial radius in Mpc/h
    """
    if LCDM == True:
        Delta_vir = func_Delta_vir_LCDM(cosmo_dic, Omega_0)
    else:
        Delta_vir = func_Delta_vir(cosmo_dic)
    #print(Delta_vir)
    #print(M)
    #print(func_rho_comp_0(Omega_0))
    return (3. * M / (4. * np.pi * func_rho_comp_0(Omega_0) * Delta_vir ))**(1./3.)

