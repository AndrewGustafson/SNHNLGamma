"""
Obtain Constraints on decays into gamma rays
    given production rate
"""

#Initialization
import numpy as np #Package for array functions

from numpy import random as rand
from numpy import sin, cos, sqrt, pi, log,exp
import matplotlib
from matplotlib import pyplot as plt
#plt.style.use(["science","ieee","bright"])
import DecayRates as DR
import DecayGeometries as DG
import scipy as sp

mpi = 0.135 #GeV

#Read In Data
# Solar Maximum Mission Data
obstimes = np.array([]) #s
counts = np.array([])
eff_area = 63 #cm^2
background_counts = 6.3*10.24 #per 10.24s bin

file = open("SMM-Timing-Data.csv","r")
line_index = 0
for line in file:
    #print(line)
    if line_index > 0.5:
        line = line.split(',')
        obstimes = np.append(obstimes,float(line[0])*60 - (35*60 + 41.37))
        counts = np.append(counts,float(line[1]))
    line_index += 1
    
file.close()

Carenza_mN = np.array([])
Carenza_U2 = np.array([])
file = open("Carenza 1987a Gamma Ray Constraints Tau.csv","r")
line_index = 0
for line in file:
    if line_index > 0.5:
        line = line.split(',')
        #print(line)
        Carenza_mN = np.append(Carenza_mN,float(line[0]) * 1e-3)
        Carenza_U2 = np.append(Carenza_U2, float(line[1]))
    line_index += 1
file.close()

# Here for mN = 200 MeV, U = 1

#masses = np.array([0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5])
#mass_names = ["010","030","050","070","100","150","200","300","400","500"]

masses = np.array([0.15,0.2,0.3])
mass_names = ["150","200","300"]

U_upper_lims = np.array([])
U_lower_lims = np.array([])

#char_masses = np.array([0.05,0.07,0.1])
#char_mass_names = ["050","070","100"]
#char_Us = np.array([[1e-5],[1e-5],[1e-5]])


name_index = -1
for mN in masses:
    print("mass",mN)
    name_index += 1

    ENvals = np.array([]) #GeV
    dNdENvalsUnscaled = np.array([]) #GeV^{-1}
    
    MeV_mass_str = mass_names[name_index]
    
    file = open("DiffNumLumE_Mass"+str(MeV_mass_str)+"MeV_18p8MSun_Mixing.txt","r")
    for line in file:
        line = line.replace("{","")
        line = line.replace("}","")
        line = line.replace("\n","")
        line = line.replace("*^","E")
        
        line = line.split(',')
        
        ENvals = np.append(ENvals,float(line[0]))
        dNdENvalsUnscaled = np.append(dNdENvalsUnscaled,float(line[1]))
        
    file.close()
    
    dEN = ENvals[1] - ENvals[0]
    #print('Total Produced', np.sum(dEN*dNdENvalsUnscaled))
    
    '''
    fig = plt.figure()
    plt.plot(ENvals,dNdENvalsUnscaled)
    '''
    
    Env = 2e12 #cm
    fig = plt.figure("U Constraints")

    #Uvals = char_Us[name_index]#np.logspace(-8,0,25)
    Uvals = np.logspace(-8,0,150)
    fluence_vals = np.array([])
    chi_sq_vals = np.array([])
    
    
    for U in Uvals:
        #print("U",U)
        EgammaVals = np.linspace(0.025,0.1,50) #GeV
        tvals = np.linspace(0,223 ,51) #s  np.linspace(800,1500,51) 
        '''
        if U == char_Us[0][1]:
            tvals = np.linspace(0,25,151)
        else:
            continue
        '''
        dEgamma = EgammaVals[1] - EgammaVals[0]
        dt = tvals[1] - tvals[0]
        
        dNdEgammadt = np.zeros((len(EgammaVals),len(tvals)))
        
        dNdENProd = U**2 * dNdENvalsUnscaled
        
        
        for i in range(len(EgammaVals)):
            Egamma = EgammaVals[i] #GeV
            #print("Egamma",Egamma)
            for j in range(len(tvals)):
                tdelay = tvals[j] #s
                #print("tdelay",tdelay)
                dNdEgammadt[i,j] += DG.dNdEgammadtNuGammaRaffelt(Egamma,tdelay,mN,U,ENvals,dNdENProd,Renv=Env)
                #dNdEgammadt[i,j] += DG.dNdEgammadtNuGammaCarenza(Egamma,tdelay,mN,U,ENvals,dNdENProd,Renv=Env)
                if mN > mpi:
                    #print("pi nu",DG.dNdEgammadtNuPiRaffelt(Egamma,tdelay,mN,U,ENvals,dNdENProd,Renv = Env))
                    dNdEgammadt[i,j] += DG.dNdEgammadtNuPiRaffelt(Egamma,tdelay,mN,U,ENvals,dNdENProd,Renv=Env)
                    #dNdEgammadt[i,j] += DG.dNdEgammadtNuPiCarenza(Egamma,tdelay,mN,U,ENvals,dNdENProd,Renv=Env)
                    #nothing = 0
        
        
        distance = 50*3e21 #cm
        '''
        fig = plt.figure()
        plt.contourf(EgammaVals,tvals,np.transpose(dNdEgammadt)/ (4*pi*distance**2))
        plt.xlabel("$E_{\gamma}$ [GeV]")
        plt.ylabel("t [s]")
        plt.colorbar()
        #plt.title("$\\frac{d^2\Phi}{dE_{\gamma}dt}$ [$GeV^{-1} s^{-1} cm^{-2}$] ; $m_{N}$ =" +str(round(mN,3))
        #          +"  GeV ; $log_{10}(U^2)$ ="+str(round(np.log10(U**2),2)),fontsize = 8)
        plt.title("$m_{N}$ =" +str(round(mN,3))
                  +"  GeV ; $log_{10}(U^2)$ ="+str(round(np.log10(U**2),2)))
        '''
        
        total_fluence = np.sum(dNdEgammadt * dEgamma * dt / (4*pi*distance**2)) #cm^{-2}
        
        fluence_vals = np.append(fluence_vals,total_fluence)
        
        #plt.plot([0.1,0.1],[0,25],color = "r")
        #plt.plot([0.025,0.1],[223.2,223.2],color = "r")
        
        #print('total_fluence',total_fluence)
        '''
        ind_fluence_vals = np.array([])
        
        for i in range(len(obstimes)):
            if i == 0:
                time_req = np.heaviside((obstimes[1] + obstimes[0])/2 - tvals,1)
            elif i == (len(obstimes)-1):
                dtobs = obstimes[i] - obstimes[i-1]
                time_req = np.heaviside((obstimes[i] + dtobs/2) - tvals,1)\
                    * np.heaviside(tvals - (obstimes[i] - dtobs/2),0)
                    
                #print(np.sum(time_req))
            else:
                time_req = np.heaviside((obstimes[i+1]+obstimes[i])/2 - tvals,1)\
                    *np.heaviside(tvals - ((obstimes[i] + obstimes[i-1])/2),0)
                #print(np.sum(time_req))
            
            ind_flu = np.sum(dNdEgammadt * dEgamma * dt * time_req \
                             / (4*pi*distance**2)) #cm^{-2}
            
            ind_fluence_vals = np.append(ind_fluence_vals,ind_flu)
        
        
        
        fig = plt.figure()
        plt.plot(obstimes,counts)
        plt.plot(obstimes,ind_fluence_vals*eff_area + background_counts)
        
        theory_counts = ind_fluence_vals *eff_area + background_counts
        
        chi_sq = np.sum((theory_counts - counts)**2/counts)
        chi_sq_vals = np.append(chi_sq_vals,chi_sq)
        '''
        '''   
        fig = plt.figure("U Constraints")
        plt.plot(Uvals**2,fluence_vals/(tvals[-1]-tvals[0]),label = "$R_{env} $= 10^"+str(round(np.log10(Env),2)) + " cm ; 10 - 30 min")
        
        fig = plt.figure("Chi2")
        plt.plot(Uvals**2,chi_sq_vals)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("$|U_{\mu,4}|^2$")
        plt.ylabel("$\\chi^2$ [$cm^{-2} s^{-1}$]")
        '''
    
    '''
    fig = plt.figure("U Constraints")
    plt.plot([Uvals[0]**2,Uvals[-1]**2],np.array([1.38,1.38])/223.2)
    
    plt.xlabel("$|U_{\mu,4}|^2$")
    plt.ylabel("Flux [$cm^{-2} s^{-1}$]")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim([1e-18,1e6])
    
    plt.figure("Chi2")
    plt.plot([Uvals[0]**2,Uvals[-1]**2],np.array([36,36]))
    '''
    
    #For Normal Time
    '''
    try:
        lower_index = np.where(fluence_vals > 1.38)[0][0]
        upper_index = np.where(fluence_vals > 1.38)[0][-1]
        lower_U = Uvals[lower_index]
        upper_U = Uvals[upper_index]
        U_upper_lims = np.append(U_upper_lims,upper_U)
        U_lower_lims = np.append(U_lower_lims,lower_U)
    except:
        U_upper_lims = np.append(U_upper_lims,1e-14)
        U_lower_lims = np.append(U_lower_lims,1)
    '''
    #For LaterTimes
    
    tot_time = tvals[-1] - tvals[0]
    try:
        lower_index = np.where(fluence_vals/tot_time > 1.38/223.2)[0][0]
        upper_index = np.where(fluence_vals/tot_time > 1.38/223.2)[0][-1]
        lower_U = Uvals[lower_index]
        upper_U = Uvals[upper_index]
        U_upper_lims = np.append(U_upper_lims,upper_U)
        U_lower_lims = np.append(U_lower_lims,lower_U)
    except:
        U_upper_lims = np.append(U_upper_lims,1e-14)
        U_lower_lims = np.append(U_lower_lims,1)
    
    
fig = plt.figure("U2 Compare 2")
plt.plot(masses[:-1],U_upper_lims[:-1]**2,color = "red",label = "$R_{env} = 2 \\times 10^{12} cm$ SMM 0-223s")
plt.plot(masses[:-1],U_lower_lims[:-1]**2,color = "red")

#plt.plot(Carenza_mN,Carenza_U2,color = "purple",label = "Carenza Limits")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$m_{N} [GeV]$")
plt.ylabel("$|U_{\\tau 4}|^2$")
plt.legend(fontsize = 5)
