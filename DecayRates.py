"""
Module to calculate the decay rates of HNLs coupled to
    muon neutrinos. We are especially interested in
    those that lead to gamma rays
    (Rates from Carenza, Lucente,... 2024)
"""

#Initialization
import numpy as np #Package for array functions

from numpy import random as rand
from numpy import sin, cos, sqrt, pi, log
import matplotlib
from matplotlib import pyplot as plt

GF = 1.16637e-5 #GeV^{-2}
alpha = 1/137

def Gamma_nu_gamma(mN,U):
    '''
    Function to determine the rate
    N -> \gamma + \nu
    ----------
    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    
    Gamma = 9* alpha* GF**2 * mN**5 * U**2/(2048 * pi**4)
    
    return(Gamma)

def Gamma_3nu(mN,U):
    '''
    Function to determine the rate
    N ->  \nu + \nu + \nu (flavor agnostic)
    ----------
    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    prefactor = GF**2 * mN**5 *U**2
    threemu = 1/(384*pi**3) #3 muon neutrinos rate
    onemu = 1/(768 * pi**3) #1 muon neutrino, 2 others
    
    Gamma = prefactor * (threemu + 2*onemu)
    return(Gamma)

def Gamma_nu_ee(mN,U):
    '''
    Function to determine the rate
    N ->   \nu + e+ + e-
    ----------
    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    kin_req = np.heaviside(mN - 1.02e-3,0)
    Sw = sqrt(0.23) #sin(theta_W)
    gL = -1/2 + Sw**2
    gR = Sw**2
    
    Gamma = (gL**2 + gR**2) * GF**2 * mN**5 * U**2 / (192* pi**3)
    return(Gamma * kin_req)

def Gamma_nu_e_mu(mN,U):
    '''
    Function to determine the rate
    N ->   \nu + e+ + \mu-
    ----------
    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    m_mu = 0.1057 #GeV
    kin_req = np.heaviside(mN - .1062,0)
    prefactor = GF**2 * mN**5 * U**2 / (384 * pi**3)
    
    first_factor = 2 * (1 - m_mu**2/mN**2) * (2 + 9* m_mu**2/mN**2)
    second_factor = 2 * m_mu**2/mN**2 * (1 - m_mu**2 / mN**2) \
        *(-6 - 6*m_mu**2/mN**2 + m_mu**4/mN**4 + 6 * log(m_mu**2/mN**2))
        
    Gamma = prefactor*(first_factor + second_factor)
    return(Gamma*kin_req)

def Gamma_nu_pi(mN,U):
    '''
    Function to determine the rate
    N ->   \nu + pion
    ----------
    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    m_pi = 0.135 #GeV
    f_pi = 0.135 #GeV
    
    kin_req = np.heaviside(mN - 0.1396,0)
    
    Gamma = GF**2 * mN**3 * U**2 * f_pi**2 * (1-m_pi**2/mN**2)**2/ (32*pi)
    
    return(Gamma*kin_req)

def Gamma_tot(mN,U,flavor = "mu"):
    '''
    Function to determine total decay rate
    (valid for masses up to 200 MeV)

    mN : Mass of HNL (GeV)
    U : U_{\mu,4} matrix element

    Returns
    -------
    Gamma: Decay Rate (GeV)
    '''
    if flavor == "mu":
        Gamma = Gamma_nu_gamma(mN,U) + Gamma_3nu(mN,U)\
            + Gamma_nu_ee(mN,U) + Gamma_nu_e_mu(mN,U)\
                +Gamma_nu_pi(mN,U)
    
    elif flavor == "tau":
        Gamma = Gamma_nu_gamma(mN,U) + Gamma_3nu(mN,U)\
            + Gamma_nu_ee(mN,U) + Gamma_nu_pi(mN,U)
    
    return(Gamma)

U = 1
mNvals = np.linspace(0.1,0.300,201)

B_nu_gamma = Gamma_nu_gamma(mNvals,U)/Gamma_tot(mNvals,U,"tau")
B_3nu = Gamma_3nu(mNvals,U)/Gamma_tot(mNvals,U,"tau")
B_nu_ee = Gamma_nu_ee(mNvals,U)/Gamma_tot(mNvals,U,"tau")
#B_nu_e_mu = Gamma_nu_e_mu(mNvals,U)/Gamma_tot(mNvals,U)
B_nu_pi = Gamma_nu_pi(mNvals,U)/Gamma_tot(mNvals,U,"tau")
'''
fig = plt.figure()
plt.plot(mNvals,B_nu_gamma, label = "$\gamma \\nu$")
plt.plot(mNvals,B_3nu, label = "$3 \\nu$")
plt.plot(mNvals,B_nu_ee, label = "$\\nu e^{+} e^{-}$")
plt.plot(mNvals,B_nu_e_mu, label = "$\\nu e \mu$")
plt.plot(mNvals,B_nu_pi,label = "$\\nu \pi$")
plt.legend()
plt.yscale('log')
plt.ylim([1e-5,2])
'''

Inv_GeV_to_Sec = 6.58e-25

print("Decay Time",Inv_GeV_to_Sec/Gamma_tot(0.2,10**(-4.5)))