# -*- coding: utf-8 -*-
"""
Module to determine photon flux from HNL flux

Here, we assume coupling to muon neutrinos.
We assume all production happens at the same place
and time
"""

#Initialization
import numpy as np #Package for array functions

from numpy import random as rand
from numpy import sin, cos, sqrt, pi, log,exp
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
import DecayRates as DR

GeVtoInvCm = 5.06e13
'''
ProbENVals = np.array([])
ProbEgammaVals = np.array([])
ProbCosThetaVals = np.array([])
ProbVals = np.array([])

Probfile = open("Mass200MeV_Pion_PEgammaCosTheta.csv")
for line in Probfile:
    line = line.split(',')
    ProbENVals = np.append(ProbENVals, float(line[0]))
    ProbEgammaVals = np.append(ProbEgammaVals,float(line[1]))
    ProbCosThetaVals = np.append(ProbCosThetaVals,float(line[2]))
    ProbVals = np.append(ProbVals,float(line[3]))
    
Probfile.close()
'''

def dNdEesc(dNdEprod,En,U,mN, Renv = 1e14):
    '''
    Function to determine the amount of HNLs which
    escape the supernova envelope
    
    Parameters
    ----------
    dNdEprod : Differential production rate (GeV^{-1})
    En : Energy of HNL (GeV)
    U : U_{\mu 4} matrix element
    mN : HNL mass (GeV)
    Renv : Size of the envelope (cm, default 10^{12} cm)

    Returns
    -------
    dNdE_esc : Differential rate of HNLs escaping (GeV^{-1})
    '''
    DecGamma = DR.Gamma_tot(mN,U) #GeV
    Boost = En/mN
    beta = sqrt(1-1/Boost**2)
    
    DecLength = Boost*beta/(DecGamma * GeVtoInvCm) #cm
    
    dNdE_esc = dNdEprod * exp(-Renv/DecLength) #GeV^{-1}
    return(dNdE_esc)

def PDFdist(En,U,mN,x):
    '''
    Probability density function for decay position x

    Parameters
    ----------
    En : Energy of HNL (GeV)
    U : U_{\mu 4} matrix element
    mN : HNL mass (GeV)
    x: distance (cm)

    Returns
    -------
    Px = probability density (cm^{-1})

    '''
    DecGamma = DR.Gamma_tot(mN,U) #GeV
    Boost = En/mN
    beta = sqrt(1-1/Boost**2)
    
    DecLength = Boost*beta/(DecGamma * GeVtoInvCm) #cm
    
    Px = 1/DecLength *exp(-x/DecLength)
    return(Px)

def CosThetaLabNuGamma(En,mN,Egamma):
    '''
    Cosine of the outgoing photon relative to the HNL
        when the decay is N-> \nu \gamma
    Parameters
    ----------
    En : Energy of HNL (GeV)
    mN : HNL mass (GeV)
    Egamma : Energy of the outgoing photon (GeV)

    Returns
    -------
    CosTheta = Cosine of the angle in the lab frame
    '''
    Boost = En/mN
    beta =  sqrt(1-1/Boost**2)
    
    numerator = Egamma + (mN/2)*(-1+beta**2)*Boost
    denominator = Egamma*beta
    
    CosTheta = numerator/denominator
    return(CosTheta)

def dNdEgammadtNuGamma(Egamma,tdelay,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \gamma
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time delay relative to traveling
        in a straight line at c (sec)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    '''
    c = 3e10 #cm/s
    
    #dNdE_esc = dNdEesc(dNdEprod,EnVals,U,mN) #GeV^{-1}
    
    Boost = EnVals/mN
    beta  = sqrt(1-1/Boost**2)
    v = beta*c #cm/s
    
    Emin = Boost*(mN/2)*(1-beta)
    Emax = Boost*(mN/2)*(1+beta)
    
    CosTheta = CosThetaLabNuGamma(EnVals,mN,Egamma)
    
    xval = tdelay * (1/v -(1/c) *CosTheta)**(-1) #cm
    
    #Need to reject decays which occur within the envelope
    
    Integrand = dNdEprod * PDFdist(EnVals,U,mN,xval)* (xval/tdelay) \
        * (Emax - Emin)**(-1) * np.heaviside(xval - Renv,1) \
            * np.heaviside(Egamma - Emin,0) * np.heaviside(Emax - Egamma,0) #GeV^{-2} s^{-1}
        
    dE = EnVals[1] - EnVals[0]
    
    BranchFrac = DR.Gamma_nu_gamma(mN,U)/DR.Gamma_tot(mN,U)
    
    return(BranchFrac*np.sum(Integrand*dE))
    
    
def PEgammaCosThetaPi0(Egamma,cosTheta,dEgamma,dcosTheta,En,mN):
    '''
    Function to calculate the probability of getting
        Egamma as photon energy and cosTheta as final angle
        from the N -> \pi + \nu decay
        
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    cosTheta: Cosine of final scattering angle
    dEgamma: Small energy spacing (GeV)
    dcosTheta: Small cosTheta spacing
    EN : HNL energy (GeV)
    mN: HNL mass (GeV)
    
    returns:
    -----------
    PEgammaCosTheta
    '''
    mpi0 = 0.135 #GeV
    EpiCOM = (mN**2 + mpi0**2)/(2*mN) #GeV
    gammaPi = EpiCOM/mpi0
    #print(gammaPi)
    betaPi = sqrt(1 - 1/gammaPi**2)
    
    gammaN = En/mN
    betaN = sqrt(1 - 1/gammaN**2)
    
    
    
    LambdaHNLToLab = np.array([[gammaN, gammaN*betaN,0,0],
                               [gammaN*betaN,gammaN,0,0],
                               [0,0,1,0],[0,0,0,1]])
    
    itterations = 100000
    diffelement = 1/itterations
    prob = 0
    #Monte Carlo Integration
    
    for i in range(itterations):
        Cos_Theta_Prime = -1 + 2 * rand.random()
        Sin_Theta_Prime = sqrt(1-Cos_Theta_Prime**2)
        Cos_Alpha = -1 + 2*rand.random()
        Sin_Alpha = sqrt(1-Cos_Alpha**2)
        Beta = 2*pi*rand.random()
        Cos_Beta = cos(Beta)
        Sin_Beta = sin(Beta)
        LambdaPiToHNL = np.array([[gammaPi,gammaPi*betaPi*Cos_Theta_Prime,
                                   gammaPi*betaPi*Sin_Theta_Prime,0],
                                  [gammaPi*betaPi*Cos_Theta_Prime, 1 + (gammaPi -1)*Cos_Theta_Prime**2, 
                                   (gammaPi -1) * Cos_Theta_Prime*Sin_Theta_Prime,0],
                                  [gammaPi*betaPi*Sin_Theta_Prime,(gammaPi -1)*Cos_Theta_Prime*Sin_Theta_Prime,
                                   1 + (gammaPi -1)*Sin_Theta_Prime**2,0],
                                  [0,0,0,1]])
        
        LambdaTot = np.matmul(LambdaHNLToLab,LambdaPiToHNL)
        
        pgammapi = np.transpose(np.array([[mpi0/2, mpi0/2*Sin_Alpha*Cos_Beta,
                                           mpi0/2*Sin_Alpha*Sin_Beta,mpi0/2*Cos_Alpha]]))
        
        pgammaLab = np.matmul(LambdaTot,pgammapi)
        
        Egamma_cal = pgammaLab[0]
        cosThetacal = pgammaLab[1]/pgammaLab[0]
        
        prob += np.heaviside(Egamma_cal - (Egamma-dEgamma/2),0) * np.heaviside(Egamma+dEgamma/2 - Egamma_cal,1)\
            * np.heaviside(cosThetacal - (cosTheta - dcosTheta/2),0) * np.heaviside(cosTheta+dcosTheta/2 - cosThetacal,1)\
                *diffelement /(dEgamma*dcosTheta)
        
    
    #Riemann Integration
    '''
    Cos_Theta_Prime_Vals = np.linspace(-1,1,100)
    Sin_Theta_Prime_Vals = sqrt(1 - Cos_Theta_Prime_Vals)
    dCos_Theta = Cos_Theta_Prime_Vals[1] - Cos_Theta_Prime_Vals[0]
    
    Cos_Alpha_Vals = np.linspace(-1,1,100)
    Sin_Alpha_Vals = sqrt(1 - Cos_Alpha_Vals**2)
    dCos_Alpha = Cos_Alpha_Vals[1] - Cos_Alpha_Vals[0]
    
    Beta_Vals = np.linspace(0,2*pi,150)
    dBeta = Beta_Vals[1] - Beta_Vals[0]
    
    for i in range(len(Cos_Theta_Prime_Vals)):
        print(i)
        Cos_Theta_Prime = Cos_Theta_Prime_Vals[i]
        Sin_Theta_Prime = Sin_Theta_Prime_Vals[i]
        LambdaPiToHNL = np.array([[gammaPi,gammaPi*betaPi*Cos_Theta_Prime,
                                   gammaPi*betaPi*Sin_Theta_Prime,0],
                                  [gammaPi*betaPi*Cos_Theta_Prime, 1 + (gammaPi -1)*Cos_Theta_Prime**2, 
                                   (gammaPi -1) * Cos_Theta_Prime*Sin_Theta_Prime,0],
                                  [gammaPi*betaPi*Sin_Theta_Prime,(gammaPi -1)*Cos_Theta_Prime*Sin_Theta_Prime,
                                   1 + (gammaPi -1)*Sin_Theta_Prime**2,0],
                                  [0,0,0,1]])
        
        LambdaTot = np.matmul(LambdaHNLToLab,LambdaPiToHNL)
        
        
        for j in range(len(Cos_Alpha_Vals)):
            for k in range(len(Beta_Vals)):
                Cos_Alpha = Cos_Alpha_Vals[j]
                Sin_Alpha = Sin_Alpha_Vals[j]
                Cos_Beta = cos(Beta_Vals[k])
                Sin_Beta = sin(Beta_Vals[k])
                
                pgammapi = np.transpose(np.array([[mpi0/2, mpi0/2*Sin_Alpha*Cos_Beta,
                                                   mpi0/2*Sin_Alpha*Sin_Beta,mpi0/2*Cos_Alpha]]))
                pgammaLab = np.matmul(LambdaTot,pgammapi)
                
                Egamma_cal = pgammaLab[0]
                cosThetacal = pgammaLab[1]/pgammaLab[0]
                
                prob += np.heaviside(Egamma_cal - Egamma,0) * np.heaviside(Egamma+dEgamma - Egamma_cal,1)\
                    * np.heaviside(cosThetacal - cosTheta,0) * np.heaviside(cosTheta+dcosTheta - cosThetacal,1)\
                        *dCos_Theta * dCos_Alpha * dBeta /(dEgamma*dcosTheta)
    '''
    
    
    
    return(prob)

def dNdEgammadtNuPi(Egamma,tdelay,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \pi
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time delay relative to traveling
        in a straight line at c (sec)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns
    -------------
    dNdEgammadt : Double differential production rate of photons (GeV^{-1} s^{-1})
    '''
    #Pion-gamma Energy/properties in rest frame of HNL
    Epi = (mN**2 + mpi**2)/(2*mN)
    gammapi = (Epi/mpi)
    betapi = sqrt(1-1/gammapi**2)
    Eminprime = mpi/2 * gammapi * (1 - betapi)
    Emaxprime = mpi/2 * gammapi * (1 + betapi)
    
    dEgamma = 1e-2
    
    c = 3e10 #cm/s
    
    #I account for this by requiring the decay region
    #   to be outside the envelope
    #dNdE_esc = dNdEesc(dNdEprod,EnVals,U,mN) #GeV^{-1}
    
    CosThetaVals = np.linspace(-1,1,50)
    dCosTheta = CosThetaVals[1] - CosThetaVals[0]
    
    dEn = EnVals[1] - EnVals[0] #GeV
    
    Integral = 0
    
    if mN > 0.195 and mN < 0.205:
        EnValsLong = np.array([])
        CosThetaValsLong = np.array([])
        dNdE_Long = np.array([])
        
        
        for k in range(len(EnVals)):
            En = EnVals[k]

            for CosTheta in CosThetaVals:
                EnValsLong = np.append(EnValsLong,En)
                CosThetaValsLong = np.append(CosThetaValsLong,CosTheta)
                dNdE_Long = np.append(dNdE_Long,dNdEprod[k])
                
        Boost = EnValsLong/mN    
        beta  = sqrt(1-1/Boost**2)
        v = beta*c #cm/s
        xval = tdelay * (1/v -(1/c) *CosThetaValsLong)**(-1) #cm
        Px = PDFdist(EnValsLong,U,mN,xval)
        
        PEgCosTheta= sp.interpolate.griddata((ProbENVals,ProbEgammaVals,ProbCosThetaVals),ProbVals,
                                      (EnValsLong,Egamma*np.ones(len(EnValsLong)),CosThetaValsLong),fill_value = 0)
       
        Integral += np.sum(dCosTheta*dEn * dNdE_Long * Px * PEgCosTheta\
            * xval/tdelay * np.heaviside(xval - Renv,1) )#GeV^{-1} s^{-1}
    else:
    
        for i in range(len(CosThetaVals)):
            CosTheta = CosThetaVals[i]
            print('CosTheta',CosTheta)
            
            for j in range(len(EnVals)):
                En = EnVals[j]
                dNdE = dNdEprod[j]
                
                Boost = En/mN
                beta  = sqrt(1-1/Boost**2)
                v = beta*c #cm/s
                xval = tdelay * (1/v -(1/c) *CosTheta)**(-1) #cm
                Px = PDFdist(En,U,mN,xval)
                
                CosThetaMin = (Egamma + Emaxprime * (-1 + beta**2)*Boost)\
                    /(Egamma*beta)
                CosThetaMax = (Egamma + Eminprime * (-1 + beta**2)*Boost)\
                    /(Egamma*beta)
                
                if CosTheta > dCosTheta + CosThetaMax:
                    PEgCosTheta = 0
                elif CosTheta < -dCosTheta + CosThetaMin:
                    PEgCosTheta = 0
                else:
                    PEgCosTheta = PEgammaCosThetaPi0(Egamma,CosTheta,dEgamma,dCosTheta,En,mN) #GeV^{-1}
                    #print("Wrong One")
                
                Integral += dCosTheta*dEn * dNdE * Px * PEgCosTheta\
                    * xval/tdelay * np.heaviside(xval - Renv,1) #GeV^{-1} s^{-1}
    
    BranchFrac = DR.Gamma_nu_pi(mN,U)/DR.Gamma_tot(mN,U)
    
    return(BranchFrac* Integral)

def dNdEgammadtNuGammaRaffelt(Egamma,tdelay,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \gamma
    Follows Raffelt 1993
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time delay relative to traveling
        in a straight line at c (sec)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns:
    d2NdEgammadt : Double differeintal rate of gamma rays [GeV^{-1} s^{-1}]
    '''
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    pnVals = sqrt(EnVals**2 - mN**2) #GeV
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_gamma(mN,U)/DR.Gamma_tot(mN,U)
    
    prefactor = exp(-2*Egamma*tdelay / (mN*tau)) * (2*Egamma*Branch)/(mN * tau) #s^{-1}
    
    x = 2 * Boost * Beta * Egamma * tdelay/mN * c #cm
    Egammamin = mN/2 * Boost * (1 - Beta)
    Egammamax = mN/2 * Boost * (1 + Beta)
    
    req = np.heaviside(Egamma - Egammamin,0) * np.heaviside(Egammamax - Egamma,0)\
        *np.heaviside(x - Renv,0)
    
    Integral = np.sum(dNdEprod/pnVals * req*dEn) #GeV^{-1}
    
    d2NdEgammadt = Integral * prefactor #GeV^{-1} s^{-1}
    
    return(d2NdEgammadt)
    

def dNdEgammadtNuPiRaffelt(Egamma,tdelay,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \pi
    Follows Rafelt 1993
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time delay relative to traveling
        in a straight line at c (sec)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns
    -------------
    d2NdEgammadt : Double differential production rate of photons (GeV^{-1} s^{-1})
    '''   
    mpi = 0.135 #GeV
    EpiRest = (mN**2 + mpi**2)/(2*mN) #GeV
    Boostpi = EpiRest/mpi
    BetaPi = sqrt(1-1/Boostpi**2)
    
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_pi(mN,U)/DR.Gamma_tot(mN,U)
    
    CosThetaRest = np.transpose([np.linspace(-1,1,1000)]) #Angle in the HNL rest frame
    dCosThetaRest = CosThetaRest[1,0] - CosThetaRest[0,0]
    
    CosThetaLab = (Beta + CosThetaRest)/(1 + Beta*CosThetaRest)
    
    x = tdelay * c * (1/Beta - CosThetaLab)**(-1) #cm
    esc_req = np.heaviside(x - Renv,0)
    
    omega = Egamma/(Boost * (1 + Beta*CosThetaRest)) #Photon energy in rest frame GeV
    
    f_omega = 2 * (Boostpi * mpi* BetaPi)**(-1)\
        *np.heaviside(omega - Boostpi * mpi/2 * (1 - BetaPi),0)\
            *np.heaviside(Boostpi*mpi/2 * (1+BetaPi) - omega,0) #GeV^{-1}
            
    Integrand = dNdEprod * exp(-Boost*(1+Beta*CosThetaRest)*tdelay/tau)\
        *(Branch/tau) * f_omega/2 * esc_req #GeV^{-2} s^{-1} 
        
    d2NdEgammadt = np.sum(Integrand*dEn*dCosThetaRest)
    
    return(d2NdEgammadt)


def dNdEgammadtNuGammaCarenzaTrial(Egamma,tdec,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \gamma
    Follows Carenza 2023 (not entirely sure, their stuff
                          is confusing and incorrect)
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdec: time of decay
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns:
    d2NdEgammadt : Double differeintal rate of gamma rays [GeV^{-1} s^{-1}]
    '''
    
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    pnVals = sqrt(EnVals**2 - mN**2) #GeV
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_gamma(mN,U)/DR.Gamma_tot(mN,U)
    
    dec_length = Boost * Beta * tau * c #cm
    dNdEesc = dNdEprod * exp(-Renv/dec_length) #GeV^{-1}
    
    Integrand = (Branch)/(Boost**2 * mN * Beta * tau)\
        *exp(-tdec/(Boost*tau)) * dNdEesc #GeV^{-2} s^{-1}
    
    kin_req = np.heaviside((2*mN)/(Egamma*Boost) - (1-Beta),0) \
        * np.heaviside((1+Beta) - (2*mN)/(Egamma*Boost) ,0)
        
    d2NdEgammadt = np.sum(dEn*Integrand*kin_req) #GeV^{-1} s^{-1}
    return(d2NdEgammadt)
        
    
def dNdEgammadtNuPiCarenzaTrial(Egamma,tdec,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \pi
    Follows Carenza 2023 (not entirely sure, their stuff
                          is confusing and incorrect)
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time of decay (s)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns
    -------------
    d2NdEgammadt : Double differential production rate of photons (GeV^{-1} s^{-1})
    ''' 
    
    mpi = 0.135 #GeV
    EpiRest = (mN**2 - mpi**2)/(2*mN) #GeV
    Boostpi = EpiRest/mpi
    BetaPi = sqrt(1-1/Boostpi**2)
    
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_pi(mN,U)/DR.Gamma_tot(mN,U)
    
    dec_length = Boost * Beta * tau * c #cm
    dNdEesc = dNdEprod * exp(-Renv/dec_length) #GeV^{-1}
    
    CosThetaRest = np.transpose([np.linspace(-1,1,1000)]) #Angle in the HNL rest frame
    dCosThetaRest = CosThetaRest[1,0] - CosThetaRest[0,0]
    
    omega = Egamma/(Boost * (1 + Beta*CosThetaRest)) #Photon energy in rest frame GeV
    
    f_omega = 2 * Branch* (Boostpi * mpi* BetaPi)**(-1)\
        *np.heaviside(omega - Boostpi * mpi/2 * (1 - BetaPi),0)\
            *np.heaviside(Boostpi*mpi/2 * (1+BetaPi) - omega,0) #GeV^{-1}
    
    Integrand = 1/(2*Boost**2 * tau * (1 + Beta * CosThetaRest)) \
        *f_omega * exp(-tdec/(Boost*tau)) * dNdEesc

    d2NdEgammadt = np.sum(Integrand*dEn*dCosThetaRest)
    
    return(d2NdEgammadt)
    
def dNdEgammadtNuGammaCarenza(Egamma,tdec,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \gamma
    Follows Carenza 2023 (not entirely sure, their stuff
                          is confusing and incorrect)
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdec: time of decay
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns:
    d2NdEgammadt : Double differeintal rate of gamma rays [GeV^{-1} s^{-1}]
    '''
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_gamma(mN,U)/DR.Gamma_tot(mN,U)
    
    dec_length = Beta*Boost*tau*c #cm
    
    dNdEesc = dNdEprod * exp(-Renv/dec_length) #GeV^{-1}
    
    Ebar = mN/2
    Emin = mN*(EnVals**2 - Ebar**2)/(2*EnVals*Ebar)
    
    prefactor = mN/(2*Ebar) * Branch
    
    pN = sqrt(EnVals**2 - mN**2)
    
    Integrand = (1/pN) * dNdEesc * (c*Beta*exp(-tdec*Beta/dec_length))/(dec_length)\
        *np.heaviside(EnVals - Emin,0) # GeV^{-2} s^{-1}
    
    d2NdEgammadt = np.sum(prefactor*Integrand*dEn)
    
    return(d2NdEgammadt)

def dNdEgammadtNuPiCarenza(Egamma,tdec,mN,U,EnVals,dNdEprod,Renv = 1e14):
    '''
    Function to determine the double differential rate of
    gamma ray production from N-> \nu \pi
    Follows Carenza 2023 (not entirely sure, their stuff
                          is confusing and incorrect)
    
    Parameters:
    -----------
    Egamma: Energy of outgoing gamma ray (GeV)
    tdelay: time of decay (s)
    mN: HNL mass (GeV)
    U : U_{\mu 4} matrix element
    EnVals: Energy values for the HNL (must be equally spaced) (GeV)
    dNdEprod : Differential production of HNLs (GeV^{-1})
    
    Returns
    -------------
    d2NdEgammadt : Double differential production rate of photons (GeV^{-1} s^{-1})
    '''
    dEn = EnVals[1] - EnVals[0]
    Boost = EnVals/mN
    Beta = sqrt(1-1/Boost**2)
    c = 3e10 #cm/s
    
    Inv_GeV_to_Sec = 6.58e-25
    tau = Inv_GeV_to_Sec / DR.Gamma_tot(mN,U) #s
    Branch = DR.Gamma_nu_pi(mN,U)/DR.Gamma_tot(mN,U)
    
    dec_length = Beta*Boost*tau*c #cm
    
    dNdEesc = dNdEprod * exp(-Renv/dec_length) #GeV^{-1}
    
    mpi = 0.135
    EbarN = (mN**2 - mpi**2)/(2*mN)
    EminN = mN * (EnVals**2 + EbarN**2)/(2*EnVals*EbarN)
    
    prefactorN = mN * Branch / (2*EbarN)
    pN = sqrt(EnVals**2 - mN**2)
    
    IntegrandN = (1/pN) * dNdEesc * (c*Beta*exp(-tdec*Beta/dec_length))/(dec_length)\
        *np.heaviside(EnVals - EminN,0) #GeV^{-2} s^{-1}
    
    dRpidEpi = np.sum(prefactorN * IntegrandN * dEn) #GeV^{-1} s^{-1}
    
    EpiVals = np.linspace(0.14,1,100)
    dEpi = EpiVals[1] - EpiVals[0]
    
    EbarPi = mpi/2
    EminPi = (EpiVals**2 + EbarPi**2)/(2*EbarPi*EpiVals)
    
    prefactorpi = mpi/(2*EbarPi)
    
    ppi = sqrt(EpiVals**2 - mpi**2)
    
    Integrandpi = (1/ppi) * dRpidEpi
    
    d2NdEgammadt = np.sum(prefactorpi*Integrandpi*dEpi)
    
    return(d2NdEgammadt)
    
    
    

            
mpi = 0.135                
mN = 0.2


'''
EnVals = np.linspace(mN,1,30)

file = open('Mass200MeV_Pion_PEgammaCosTheta.csv','w')
for En in EnVals:

    gammaN = En/mN
    betaN = sqrt(1 - 1/gammaN**2)
    Epi = (mN**2 + mpi**2)/(2*mN)
    gammapi = (Epi/mpi)
    betapi = sqrt(1-1/gammapi**2)
    
    Eminprime = mpi/2 * gammapi * (1 - betapi)
    Emaxprime = mpi/2 * gammapi * (1 + betapi)
    
    EgammaVals = np.linspace(0,En,30)
    dE = EgammaVals[1] - EgammaVals[0]
    CosThetaVals = np.linspace(-1,1,60)
    dcosTheta = CosThetaVals[1] - CosThetaVals[0]
    prob_array = np.zeros((len(EgammaVals),len(CosThetaVals)))
    
    for i in range(len(EgammaVals)):
        Egamma = EgammaVals[i]
        CosThetaMin = (Egamma + Emaxprime * (-1 + betaN**2)*gammaN)\
            /(Egamma*betaN)
        CosThetaMax = (Egamma + Eminprime * (-1 + betaN**2)*gammaN)\
            /(Egamma*betaN)
        print('Egamma', Egamma)
        for j in range(len(CosThetaVals)):
            CosTheta = CosThetaVals[j]
    
            if Egamma > dE + ((mN**2)/(2*(En-sqrt(En**2 - mN**2)*(CosTheta+dcosTheta)))):
                prob_array[i,j] = 0
    
            if CosTheta > dcosTheta + CosThetaMax:
                prob_array[i,j] = 0
            elif CosTheta < -dcosTheta + CosThetaMin:
                prob_array[i,j] = 0
            
            else:
                prob_array[i,j] = dE*dcosTheta * PEgammaCosThetaPi0(Egamma,CosTheta,dE,dcosTheta,En,mN)
            #print(PEgammaCosThetaPi0(EgammaVals[14],CosThetaVals[11],dE,dcosTheta,En,mN)) 
            
            file.write(str(En) +',' + str(Egamma) + ',' + str(CosTheta) + ',' + str(prob_array[i,j]/(dE*dcosTheta)) + "\n")

file.close()

file2 = open("fake file","r") 
        
fig = plt.figure()
#plt.plot(CosThetaVals,(mN**2)/(2*(En-sqrt(En**2 - mN**2)*CosThetaVals)))

gammaN = En/mN
betaN = sqrt(1 - 1/gammaN**2)
Epi = (mN**2 + mpi**2)/(2*mN)
gammapi = (Epi/mpi)
betapi = sqrt(1-1/gammapi**2)

Eminprime = mpi/2 * gammapi * (1 - betapi)
Emaxprime = mpi/2 * gammapi * (1 + betapi)

Eprimevals = np.linspace(Eminprime,Emaxprime,200)
CosThetaMin = np.array([])
CosThetaMax = np.array([])


CosThetaMin = (EgammaVals + Emaxprime * (-1 + betaN**2)*gammaN)\
    /(EgammaVals*betaN)
CosThetaMax = (EgammaVals + Eminprime * (-1 + betaN**2)*gammaN)\
    /(EgammaVals*betaN)
    
plt.plot(CosThetaMax + dcosTheta, EgammaVals)
plt.plot(CosThetaMin - dcosTheta, EgammaVals)
        
plt.contourf(CosThetaVals,EgammaVals,prob_array)
'''

'''
for Egamma in EgammaVals:
    index = 0
    for Eprime in Eprimevals:
        #CosTheta = (Egamma/Eprime) * (1/gammaN)\
        #    *(betaN + 1/betaN * ((Egamma/(Eprime*gammaN)) -1))**(-1)
        CosTheta = (Egamma + Eprime * (-1 + betaN**2)*gammaN)\
            /(Egamma*betaN)
        if index == 0:
            CurrentMin = CosTheta
            CurrentMax = CosTheta
        else:
            if CosTheta<CurrentMin and CosTheta > -1:
                CurrentMin = CosTheta
            if CosTheta>CurrentMax and CosTheta < 1:
                CurrentMax = CosTheta
        
        index += 1
    CosThetaMin = np.append(CosThetaMin,CurrentMin)
    CosThetaMax = np.append(CosThetaMax,CurrentMax)
'''