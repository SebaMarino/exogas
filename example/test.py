import numpy as np
import matplotlib.pyplot as plt
import gas_simulation as simulation
from matplotlib import rc
import time #import gmtime, strftime
import general_functions as gf 
from constants import *


plt.style.use('style1')
font= {'family':'Times New Roman', 'size': 14}
rc('font', **font)
fs=12 # fontsize for text apart from labels


run=True
save=True

figure_name='./Figure_test_Sigma_evolution_photoncounting'

##### INPUT PARAMETERS
Mstar=1.5
Lstar=6.2 # gas1D.M_to_L(Mstar)
fco=0.1
rbelt=122.0 # au # Moor+2017 + Gaia dr3
width=62.  # au, FWHM, Moor+2017 + Gaia dr3
tf=1.0e7
alpha=1.0e-4
fir=5.e-3
fion=0.1
fCI=1.-fion
tcoll=1.0e6

carbon_capture=True
pcapture=1.

rinobs=30.
routobs=190.



#### INITIALIZE SIMULATION
sim=simulation.simulation(tf=tf, Mstar=Mstar, Lstar=Lstar, rbelt=rbelt, width=width, alpha=alpha, fir=fir, fion=fion, tcoll=tcoll, carbon_capture=carbon_capture, pcapture=pcapture)


#### RUN SIMULATION
S_filename='Sigma_gas_evol_%1.1e_%1.1e_phcount_fixed_mdot'%(alpha, fir)
t_filename='ts_evol_%1.1e_%1.1e__phcount_fixed_mdot'%(alpha, fir,)
print( 'Running simulation...')
T1=time.gmtime()
sim.viscous_evolution()
T2=time.gmtime()
print('Execution time in sec: ',time.mktime(T2)-time.mktime(T1)) 
np.save(S_filename, sim.Sigma_g)
np.save(t_filename, sim.ts)



### CALCULATE TOTAL MASS

rmax_mtot=3.0 * sim.rbelt*(1.0+sim.tf/sim.tvis)
mask_mtot=sim.grid.rs<rmax_mtot
MCOs=np.sum(sim.Sigma_g[0,mask_mtot,:].T*sim.grid.hs[mask_mtot]*sim.grid.rs[mask_mtot]*2.0*np.pi, axis=1)
MC1s=np.sum(sim.Sigma_g[1,mask_mtot,:].T*sim.grid.hs[mask_mtot]*sim.grid.rs[mask_mtot]*2.0*np.pi, axis=1)


###############
## PLOTTING
###############

print('plotting')

# plt.plot(sim.ts_sim, sim.fir)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# factor 2 no longer necessary as now it is done with photon counting
sigma_C1c=(1./sigma_c1)*m_c1/Mearth*au_cm**2.0 # mearth/au2
sigma_COc=(1./sigma_co)*m_co/Mearth*au_cm**2.0 # mearth/au2

### EPOCHS TO PLOT
ts_plot=np.logspace(3, int(np.log10(tf)), int(np.log10(tf))-3+1)
fig=plt.figure(figsize=(13,4))

ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)


#### plottomg surface densities
for i, ti in enumerate(ts_plot):
    it=0
    for k in range(len(sim.ts)):
        if sim.ts[k]>=ti:
            it=k
            break

    color1=gf.fcolor_x(i, len(ts_plot), colormap='viridis')

    ax1.plot(sim.grid.rs, sim.Sigma_g[0,:,it]*fCI, color=color1, label=gf.number_to_text(ts_plot[i]/1.0e6)+' Myr')
    ax2.plot(sim.grid.rs, sim.Sigma_g[1,:,it]*fCI, color=color1, label=gf.number_to_text(ts_plot[i]/1.0e6)+' Myr')

    
ax1.axhline(sigma_COc, color='grey', ls='dashed')
ax2.axhline(sigma_C1c, color='grey', ls='dashed')

ymax=max( np.max(sim.Sigma_g[0,:,:]), np.max(sim.Sigma_g[1,:,:]), sigma_C1c)*2.0
ymin=min( sim.Sigma_g[0,sim.ibelt,it], sigma_COc)/100.0
for axi in [ax1, ax2]:
    axi.set_xlim(1.0, 3.0e3)
    axi.set_ylim(ymin, ymax)
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlabel('Radius [au]')
ax1.legend(frameon=True, loc=3, fontsize=fs)

ytext1=ymax * 10** ( -0.45 *(np.log10(ymax)-np.log10(ymin))  )
ytext2=ymax * 10** ( -0.13 *(np.log10(ymax)-np.log10(ymin))  )
ytext3=ymax * 10** ( -0.07 *(np.log10(ymax)-np.log10(ymin))  )

ax1.axvline(rinobs, color='grey', ls='--')
ax1.axvline(routobs, color='grey', ls='--')

ax1.text(2.0, ytext2, r'$\alpha=10^{%1.0f}$'%np.log10(alpha)+'\n $\dot{M}_\mathrm{CO}=%1.2f$'%(sim.MdotCO[-1]*1.0e6)+r' $M_{\oplus}$/Myr', fontsize=fs*0.9)
# ax1.text(7.0, ytext2, r'$\alpha=10^{%1.0f}$'%np.log10(alpha)+'\n $M_\mathrm{CO}=%1.0f$'%(Mtot*fco)+r' $M_{\oplus}$', fontsize=fs)
# ax1.text(7.0, ytext2, r'$\alpha=10^{%1.0f}$'%np.log10(alpha)+'\n $M_\mathrm{CO}=$'+gf.number_to_text(Mtot*fco)+r' $M_{\oplus}$', fontsize=fs)

# ax1.text(1.5, ytext3, label_plot, fontsize=fs)
ax1.text(1.0e3, ytext3, 'CO', fontsize=fs)
ax2.text(1.0e3, ytext3, 'CI', fontsize=fs)
ax1.set_ylabel(r'Surface density [$M_{\oplus}$ au$^{-2}$]')

#### plotting masses

ax3.plot(sim.ts/1.0e6, MCOs, color='C0', label='CO')
ax3.plot(sim.ts/1.0e6, MC1s*fCI, color='C1', label='C')
# ax3.plot(ts/1.0e6, Mtot_CO, color='black', label='CO in planetesimals')

ax3.set_xscale('log')
ax3.set_yscale('log')

MCO_st=sim.MdotCO*120.0
MC1_st=sim.MdotCO*m_c1/m_co * (2.*sim.rbelt / (3.*sim.nus_au2_yr[0]*(1./sim.grid.rs[0])))*(1.+2.*(rmax_mtot/sim.rbelt)**0.5-1.0) # From integrating Metzeger equations

ax3.plot(sim.ts_sim/1.0e6, MCO_st, color='C0', ls='dashed')
ax3.plot(sim.ts_sim/1.0e6, MC1_st, color='C1', ls='dashed')
ax3.set_ylabel(r'Gas mass [$M_{\oplus}$]')
ax3.set_xlabel(r'Time [Myr]')
ax3.set_xlim(ts_plot[0]/1.0e6, ts_plot[-1]/1.0e6)
ax3.set_ylim(1.0e-8, 1.0e2)

ax3.legend(frameon=True, loc=3, fontsize=fs)

plt.tight_layout()

if save:
    plt.savefig(figure_name+'_%1.1e_%1.1e.pdf'%(alpha, fir), transparent=True)
    plt.show()


# tph_t=simulation.tau_CO3(Sigma_g1[0,sim.ibelt,:], Sigma_g1[1,sim.ibelt,:])
# tph_r1=gas1D.tau_CO3(Sigma_g1[0,:,-1], Sigma_g1[1,:,-1])
# tph_r2=gas1D.tau_CO2(Sigma_g1[0,:,-1], Sigma_g1[1,:,-1])

# plt.subplot(211)
# plt.plot(rs, tph_r1)
# plt.plot(rs, tph_r2)
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(np.min(tph_r1),np.max(tph_r1))

# plt.subplot(212)
# plt.plot(rs, Sigma_g1[0,:,-1])
# plt.plot(rs, Sigma_g1[1,:,-1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
