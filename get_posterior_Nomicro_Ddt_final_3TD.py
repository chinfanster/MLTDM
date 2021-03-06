#!/usr/bin/env python
import time
import pyfits
import get_intepo
import numpy as np
import corner
import emcee
from get_text_number import get_number, get_TD
import get_intepo
import sys
import os.path

input_file= sys.argv[1]
#input_file="Ddt_0.5R0_incl00_PA00"


##define the prior of the fermat potential
tau_BA=pyfits.getdata(get_number(input_file,"tau_BA"))
tau_CA=pyfits.getdata(get_number(input_file,"tau_CA"))
tau_DA=pyfits.getdata(get_number(input_file,"tau_DA"))

##obtain the mean of the fermat potential for initial guess
tau_BA_mean=np.mean(tau_BA)
tau_CA_mean=np.mean(tau_CA)
tau_DA_mean=np.mean(tau_DA)

tau_BCD_A=np.vstack((tau_BA,tau_CA,tau_DA)).T

##we don't need this line if samples are more enough.
#outlier=np.array([])
#for i in range(tau_BCD_A.shape[0]):
#    BA_1,CA_1,DA_1=np.percentile(tau_BCD_A, 1, axis=0, keepdims=True)[0]
#    BA_99,CA_99,DA_99=np.percentile(tau_BCD_A, 99, axis=0, keepdims=True)[0]
#    if not BA_1<tau_BCD_A[i][0]<BA_99 and CA_1<tau_BCD_A[i][1]<CA_99 and DA_1<tau_BCD_A[i][2]<DA_99:
#        outlier=np.append(outlier,int(i))
#outlier=outlier.astype(int)
#tau_BCD_A=np.delete(tau_BCD_A, outlier, 0)

##get the inteporation of the Fermat potentail in 3D.
prior_tau_BCD_A,tau_BA_min,tau_BA_max,tau_CA_min,tau_CA_max,tau_DA_min,tau_DA_max=get_intepo.three_dim_intepo(tau_BCD_A,30,30,30)


##the range of Ddt
Ddt_min=float(get_number(input_file,"TD_boundary_min"))
Ddt_max=float(get_number(input_file,"TD_boundary_max")) ##Mpc
Ddt_mean=(Ddt_min+Ddt_max)/2

# Now, let's setup some parameters that define the MCMC
ndim = int(get_number(input_file,"N_parameters")) ## 3 tau + 1 Ddt 
nwalkers = int(get_number(input_file,"N_walkers"))

# Set the range of each parameters
p0_min = np.array([tau_BA_min, tau_CA_min, tau_DA_min, Ddt_min])
p0_max = np.array([tau_BA_max, tau_CA_max, tau_DA_max, Ddt_max])
p0size = p0_max - p0_min

# Initialize the chain
# Choice 2: chain is initialized in a tight ball around the expected values
p0 = [[tau_BA_mean,tau_CA_mean,tau_DA_mean,Ddt_mean] + 0.01*p0size*np.random.rand(ndim) for i in range(nwalkers)] 

#p0=np.asarray(p0)

# Visualize the initialization
#fig = corner.corner(p0)
#fig.set_size_inches(10,10)

# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

def lnprior1(theta):
    tau_BA,tau_CA,tau_DA,Ddt = theta
    if tau_BA_min < tau_BA < tau_BA_max and tau_CA_min < tau_CA < tau_CA_max and tau_DA_min < tau_DA < tau_DA_max and Ddt_min < Ddt < Ddt_max and prior_tau_BCD_A([tau_BA,tau_CA,tau_DA])>0:
        return np.log(prior_tau_BCD_A([tau_BA,tau_CA,tau_DA]))
    else:
        return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, observed_TD, icov):
    tau_BA,tau_CA,tau_DA,Ddt = theta
    TD_BA=tau_BA*Ddt
    TD_CA=tau_CA*Ddt
    TD_DA=tau_DA*Ddt
    model_TD=np.array([TD_BA,TD_CA,TD_DA])
    diff= observed_TD-model_TD
    return -np.dot(diff,np.dot(icov,diff))/2.0

def lnprob(theta, observed_TD, icov):
    lp1 = lnprior1(theta)
    if not np.isfinite(lp1):
        return -np.inf
    return lp1+lnlike(theta, observed_TD, icov)

##set up the measured time delays and their uncertainties
obTD_BA,BA_sigma = get_TD(input_file,"TD_BA")
obTD_CA,CA_sigma = get_TD(input_file,"TD_CA")
obTD_DA,DA_sigma = get_TD(input_file,"TD_DA")
icov=np.linalg.inv(np.diagflat([[BA_sigma**2,CA_sigma**2,DA_sigma**2]]))
print('the covariance matrix is',icov)
observed_TD=np.array([obTD_BA,obTD_CA,obTD_DA])


nsteps = int(get_number(input_file,"N_steps"))
# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(observed_TD, icov))

##this is the burn-in process
import time
time0 = time.time()
# burnin phase
pos, prob, state  = sampler.run_mcmc(p0, int(get_number(input_file,"burn_in")))
time1=time.time()
print("The first %s steps time:" %(int(get_number(input_file,"burn_in"))),time1-time0)
print(sampler.acceptance_fraction)
lnlike(pos[-1],observed_TD,icov)
sampler.reset()

##really working on the long run using the last step from the burn-in.
time1=time.time()
pos, prob, state=sampler.run_mcmc(pos, nsteps)
time2=time.time()
print(time2-time1)
samples=sampler.flatchain
print(samples.shape[0])
print(sampler.acceptance_fraction)
pyfits.writeto(input_file+'.fits',samples,clobber=True)
##output will be the same file with .fits.
##ploting your results using ChainConsumer is a good choice.
