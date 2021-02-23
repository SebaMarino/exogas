import numpy as np
import matplotlib.pyplot as plt
import exogas
import time #import gmtime, strftime
# from exogas.constants import *

### system parameters

Mstar=1.5
Lstar=6.2 # if not mandatory, it will be set according to the luminosity following a simple MS relation.
fco=0.1 # mass fraction of CO in planetesimals
rbelt=120.0 #  au
width=60.  # au, FWHM
tf=1.0e7   # yr final time of the simulation
alpha=1.0e-4 # alpha viscosity parameter
fir=5.e-3    # fractional luminosity at t=tf
fion=0.0    # fraction of carbon that is ionised
fCI=1.-fion
tcoll=1.0e6 # collisional timescale. Set to None or negative to consider a constant fir and CO input rate
dt_skip =10 # the output will contain 1 snapshot per every dt_skip simulated epochs. This helps to save space since the typical simulation timestep is 60 yr. is parameter set the number of simulated epochs to skip 
 
#### INITIALIZE SIMULATION
sim=exogas.simulation.simulation(tf=tf, 
                          Mstar=Mstar,
                          Lstar=Lstar,
                          fco=fco,
                          rbelt=rbelt,
                          width=width,
                          alpha=alpha,
                          fir=fir,
                          fion=fion,
                          tcoll=tcoll,
                          dt_skip=dt_skip,
                          # MdotCO=MdotCO # This line is optional depending on if you want to set MdotCO to a specific value or not  
                         )

#### RUN SIMULATION
print( 'Running simulation...')
T1=time.gmtime()
sim.viscous_evolution()
T2=time.gmtime()
print('Execution time in sec: ',time.mktime(T2)-time.mktime(T1)) 
