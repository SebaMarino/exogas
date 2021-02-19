import numpy as np
from scipy import interpolate
from constants import *
import sys

### CO PHOTODISSOCIATION PHOTON COUNTING

try:
    SCO_grid=np.loadtxt('./Sigma_CO_Mearth_au2.txt')
    SC1_grid=np.loadtxt('./Sigma_C1_Mearth_au2.txt')
    tauCO_grid=np.loadtxt('./tau_CO_yr.txt')
    log10tau_interp=interpolate.RectBivariateSpline( np.log10(SC1_grid),np.log10(SCO_grid), np.log10(tauCO_grid)) # x and y must be swaped, i.e. (y,x) https://github.com/scipy/scipy/issues/3164
    
    # log10tau_interp=interpolate.interp2d(np.log10(SCO_grid), np.log10(SC1_grid), np.log10(tauCO_grid))

    
    # N=200
    # NCOs2=np.logspace(1, 30, N) # cm-2
    # NCs2=np.logspace(5, 30, N)  # cm-2

    # Sigma_CO2=NCOs2*m_co/Mearth*au_cm**2.
    # Sigma_C12=NCs2*m_c1/Mearth*au_cm**2.
    # tau2D2=10**(log10tau_interp(np.log10(Sigma_CO2),np.log10(Sigma_C12)))
    # print tau2D2

except:
    print('Interpolaiton of CO photodissociation from photon counting did not work')


class simulation:


    def __init__(self, Mstar=None, Lstar=None, rmin=None, resolution=None, rmax0=None, rbelt=None, width=None, fir=None, fco=None, alpha=None, fion=None, mu0=None, Sigma_floor=None, rc=None, constant_CO_input=None, tf=None, dt0=None, verbose=True, dt_skip=10, diffusion=True, photodissociation=True, carbon_capture=False, pcapture=None, I=None, MdotCO=None, tcoll=None ):

        # input units are Mearth, au, yr, radians
        
        ################################
        ### default parameters
        ################################

        # system
        default_Mstar=2.0 # Msun
        # default_Lstar=M_to_L(default_Mstar) # Lsun
        default_rmin=1.0  # au
        default_resolution=0.1
        default_rmax0=3.0e3 # au
        
        ## belt parameters
        default_rbelt=100.0 # au 
        default_width=default_rbelt*0.5  # au, FWHM, Moor+2017 + Gaia dr3
        default_sig_belt=default_width/(2.0*np.sqrt(2.*np.log(2.)))
        default_fir=1.0e-3 # fractional luminosity
        default_tcoll=-1.
        
        ## gas parameters
        default_fco=0.1
        default_alpha=1.0e-3
        default_fion=0.0
        default_mu0=14.0
        default_Sigma_floor=1.0e-50 # Mearth/au2
        default_rc=50. # au

        ##  simulation parameters
        default_constant_CO_input=True
        default_tf=1.0e7 # yr
        default_dt0=60. # yr (maximum dt)

        default_pcapture=1.
        default_I =0.025 # inclination dispersion
        
        ### system
        self.Mstar=Mstar if Mstar is not None else default_Mstar
        self.Lstar=Lstar if Lstar is not None else M_to_L(self.Mstar)
        self.rmin=rmin if rmin is not None and rmin>0. else default_rmin
        
        if rmax0 is not None and rmax0>self.rmin:
            self.rmax0=rmax0 
        else:
            if default_rmax0> self.rmin:
                self.rmax0=default_rmax0
            else:
                raise ValueError('rmin>rmax')
    
        self.resolution=resolution if resolution is not None else default_resolution

        ## belt
        self.rbelt=rbelt if rbelt is not None else default_rbelt
        self.width=width if width is not None else self.rbelt*0.5
        self.sig_belt=self.width/(2.0*np.sqrt(2.*np.log(2.)))
        self.fir=fir if fir is not None else default_fir
        try: 
            if tcoll>0.:
                self.tcoll=tcoll
            else: self.tcoll=default_tcoll
        except:
            self.tcoll=default_tcoll
        ## gas parameters
        self.fco=fco if fco is not None else default_fco
        self.alpha=alpha if alpha is not None else default_alpha
        self.fion=fion if fion is not None else default_fion
        self.mu0=mu0 if mu0 is not None else default_mu0
        self.Sigma_floor=Sigma_floor if Sigma_floor is not None else default_Sigma_floor
        self.rc=rc if rc is not None else default_rc

        # self.constant_CO_input=constant_CO_input if constant_CO_input is not None else default_constant_CO_input
        self.tf=tf if tf is not None else default_tf
        self.dt0=dt0 if dt0 is not None else default_dt0
        self.dt_skip=dt_skip
        
        ################################
        #### calculate basic properties of the simulation
        ################################

        ## temperature and viscosity
        self.Tb=278.3*(self.Lstar**0.25)*self.rbelt**(-0.5) # K # Temperature at belt
        self.cs_b=np.sqrt(kb*self.Tb/(self.mu0*mp)) # m/s sounds speed at belt
        self.tvis=tau_vis(self.rbelt, self.alpha, self.cs_b, self.Mstar)

        ## spatial grid
        # Rmax_cs=(3.0e4)**4.0*(Mstar*mu0*mp/(2.*kb*278.0*Lstar**0.25))**2.
        rmax=max(self.rmax0, 3.0 * self.rbelt*(1.0+self.tf/self.tvis))# when Mdot becomes very small, the maximum radius is very important as it sets the evolution timescale.
        Nr=N_optim_radial_grid(self.rmin, rmax, self.rbelt, self.resolution)
        self.grid = simulation_grid(rmin=self.rmin, rmax=rmax, Nr=Nr, p=0.5) 
        for ir in range(Nr):
            if self.grid.rs[ir]>self.rbelt:
                self.ibelt=ir
                break
        
        ## temporal grid
        self.Ts=278.3*(self.Lstar**0.25)*self.grid.rs**(-0.5) # K
        self.cs=np.sqrt(kb*self.Ts/(self.mu0*mp))
        self.Omegas=2.0*np.pi*np.sqrt(self.Mstar/(self.grid.rs**3.0)) # 1/yr
        self.Omegas_s=self.Omegas/year_s
        self.mus=np.ones(self.grid.Nr)*self.mu0
        self.nus=self.alpha*kb*self.Ts/(self.mus*mp)/(self.Omegas_s) # m2/s 1.0e13*np.ones(Nr) #
        self.nus_au2_yr=self.nus*year_s/(au_m**2.0)
        self.dt=min(0.02*self.grid.hs[0]**2./self.nus_au2_yr[0], self.dt0) # yr 
        self.Nt=int(self.tf/self.dt)+1
        self.ts_sim=np.linspace(0.0, self.tf, self.Nt)


        #### CO input rate

        if MdotCO==None: # calculate Mdot CO based on fir
            if self.tcoll<0.:
                print('fixed CO input rate based on constant fractional luminosity')
                MdotCO_fixed= self.fco* 1.2e-3 * self.rbelt**1.5 / self.width  * self.fir**2. * self.Lstar * self.Mstar**(-0.5) # Mearth/ yr
                self.MdotCO=MdotCO_fixed*np.ones(self.Nt)
            else:
                print('varying CO input rate based on final fractional luminosity and tcoll given by the user')
                MdotCO_final= self.fco* 1.2e-3 * self.rbelt**1.5 / self.width  * self.fir**2. * self.Lstar * self.Mstar**(-0.5) # Mearth/ yr
                self.MdotCO=MdotCO_final*(1.+self.tf/self.tcoll)**2./(1.+self.ts_sim/self.tcoll)**2.
                
        elif MdotCO>0.:
            if self.tcoll<0.:
                print('fixed CO input rate based on Mdot given by the user')
                self.MdotCO=np.ones(self.Nt)*MdotCO
            else:
                print('varying CO input rate based on final Mdot and tcoll given by the user')
                self.MdotCO=MdotCO*(1.+self.tf/self.tcoll)**2./(1.+self.ts_sim/self.tcoll)**2.
        else:
            raise ValueError('input MdotCO must be a float greater than zero')

        
                
        ## switches

        self.diffusion=diffusion
        self.photodissociation=photodissociation
        self.carbon_capture=carbon_capture

        if self.carbon_capture:
            self.pcapture=pcapture if (pcapture is not None and (pcapture<=1. and pcapture>=0.)) else default_pcapture
            self.I=I if I is not None else default_I
                                       
        if verbose:
            print('Rmax = %1.1f au'%(self.grid.rmax))
            print('Nr = %i'%(self.grid.Nr))
            print('Nt=%i'%self.Nt)
            print('dt = %1.1f yr'%self.dt)
            print('dt vis = %1.1f yr'%(0.02*self.grid.hs[0]**2./self.nus_au2_yr[0]))
            print('tvis = %1.1e yr'%self.tvis)
            print('Mdot CO at t=0 is %1.1e Mearth/yr'%(self.MdotCO[0]))
            print('Mdot CO at t=tf is %1.1e Mearth/yr'%(self.MdotCO[-1]))

        ### produce CO and C surface density grid

        ### initial condition
        self.Sigma0=np.zeros((2, Nr))
        self.Sigma0[:,:]=self.Sigma_floor*(self.grid.rs/self.rc)**(-1.)*np.exp(-(self.grid.rs/self.rc))


    ##############################################
    ################ FUNCTIONS ###################
    ##############################################


    #### function to advance one step
    def Sigma_next(self, Sigma_prev, MdotCO):
    
        ###########################################
        ################ viscous evolution
        ###########################################
        Sdot_vis, Sigma_vr_halfs=self.Sigma_dot_vis(Sigma_prev)
        Snext= Sigma_prev + self.dt*Sdot_vis # viscous evolution

            
        ###########################################
        ############### inner boundary condition
        ###########################################

        #if np.all(Snext[:,2])>0.0:
        #    Snext[:,0]=np.minimum(Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]), Snext[:,1]*(rs[0]/rs[1])**(np.log(Snext[:,2]/Snext[:,1])/np.log(rs[2]/rs[1])))
        #else: 
        Snext[:,0]=Snext[:,1]*(self.nus_au2_yr[1]/self.nus_au2_yr[0]) # constant Mdot
    
        ###########################################
        ############# Outer boundary condition (power law or constant mass)
        ###########################################

        if np.all(Snext[:,-3])>0.0: # minimum between power law and constant Mdot
            Snext[:,-1]=np.minimum(Snext[:,-2]*(self.nus_au2_yr[-2]/self.nus_au2_yr[-1]), Snext[:,-2]*(self.grid.rs[-1]/self.grid.rs[-2])**(np.log(Snext[:,-2]/Snext[:,-3])/np.log(self.grid.rs[-2]/self.grid.rs[-3])))
        else: 
            Snext[:,-1]=Snext[:,-2]*(self.nus_au2_yr[-2]/self.nus_au2_yr[-1])

        ###########################################
        ################ diffusion evolution (this has to come after photodissociation and input rate, otherwise we get weird strong wiggles that diverge resulting in nans)
        ###########################################
        if self.diffusion:
            Snext=Snext+self.dt*self.Diffusion(Snext)

        ###########################################
        ############### CO mas input rate
        ###########################################
        Snext[0,:]=Snext[0,:]+self.dt*self.Sig_dot_p_Gauss(MdotCO)
        Snext[1,:]=Snext[1,:]*1.
            
        ###########################################
        ############## photodissociation
        ###########################################
        if self.photodissociation:
            tphCO=tau_CO_photon_counting(Sigma_prev[0,:], Sigma_prev[1,:], fion=self.fion)
            Sdot_ph=Sigma_prev[0,:]/tphCO
            Snext[0,:]=Snext[0,:]-self.dt*Sdot_ph
            Snext[1,:]=Snext[1,:]+self.dt*Sdot_ph*muc1co
            #Snext2[0,Snext2[0,:]<0.0]=0.0

        ###########################################
        ############## carbon capture
        ###########################################    
        # if carbon_capture and fir!=None and cs!=None:
        #     Snext2[1,:]=Snext2[1,:]-Snext2[1,:]*R_c_capture(rs, r0, width, cs, fir, I, P_capture)
 
    
        Snext[Snext[:,:]<0.0]=0.0
        
        return Snext


    def Sigma_dot_vis(self, Sigmas):
  
        ########## CALCULATE VR*Sigma=F1
        
        Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(1.+4./3.) # CO+C+O
        eps=np.ones((2,self.grid.Nr))*0.5
        mask_m=Sigma_tot>0.0
        eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
        eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]
    

        G1s=Sigma_tot*self.nus_au2_yr*np.sqrt(self.grid.rs) # Nr
        Sigma_vr_halfs=-3.0*(G1s[1:]-G1s[:-1])/(self.grid.rs[1:]-self.grid.rs[:-1])/np.sqrt(self.grid.rhalfs[1:-1]) # Nr-1
        
        ############## CALCULATE dSIGMA/dT
        eps_halfs=np.zeros((2,self.grid.Nr-1))
        eps_halfs[:,:]=np.where(Sigma_vr_halfs[:]>0.0, eps[:,:-1], eps[:,1:])
    
        G2s=self.grid.rhalfs[1:-1]*Sigma_vr_halfs  # Nr-1
        G3s=G2s*eps_halfs    #  2x(Nr-1)
        Sdot=np.zeros((2,self.grid.Nr))
        Sdot[:,1:-1]=-(G3s[:,1:]-G3s[:,:-1])*2./(self.grid.rhalfs[2:-1]**2.-self.grid.rhalfs[1:-2]**2.) # Nr-2

        return Sdot, Sigma_vr_halfs


    def Diffusion(self,Sigmas):

        Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(28./12.) # CO+C+O
        eps=np.ones((2,self.grid.Nr))*0.5
        mask_m=Sigma_tot>0.
        eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
        eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]

        # # geometric average
        eps_half=np.sqrt(eps[:,1:]*eps[:,:-1]) # Nr-1 at cell boundaries
        eps_dot=(eps_half[:,1:]-eps_half[:,:-1])/(self.grid.hs[1:-1]) # Nr-2 at cell centers

        F=self.grid.rs[1:-1]*self.nus_au2_yr[1:-1]*Sigma_tot[1:-1]*eps_dot # Nr-2
        
        Sdot_diff=np.zeros((2,self.grid.Nr))    
        Sdot_diff[:,1:-2]= (F[:,1:]-F[:,:-1])/(self.grid.rs[2:-1]-self.grid.rs[1:-2])/(self.grid.rs[1:-2]) # Nr-3
    
        return Sdot_diff

    def Sig_dot_p_Gauss(self, MdotCO):
        
        mask_belt=((self.grid.rs<self.rbelt+self.width) & (self.grid.rs>self.rbelt-self.width))
        Sdot_comets=np.zeros(self.grid.Nr)
        Sdot_comets[mask_belt]=np.exp( -2* (self.grid.rs[mask_belt]-self.rbelt)**2.0 / (2.*self.sig_belt**2.) ) # factor 2 inside exponential is to make Mdot prop to Sigma**2 
        Sdot_comets[mask_belt]=MdotCO*Sdot_comets[mask_belt]/(2.*np.pi*np.sum(Sdot_comets[mask_belt]*self.grid.rs[mask_belt]*self.grid.hs[mask_belt]))

        return Sdot_comets

    def viscous_evolution(self):
        ### function to evolve the disc until self.tf

        if isinstance(self.dt_skip, int) and self.dt_skip>0:
            if self.dt_skip>1:  #  skips dt_skip to make arrays smaller
                if (self.Nt-1)%self.dt_skip==0:
                    self.Nt2=int((self.Nt-1)/self.dt_skip+1)
                else:
                    self.Nt2=int((self.Nt-1)/self.dt_skip+2)
            elif self.dt_skip==1:
                self.Nt2=self.Nt
        else:
            raise ValueError('not a valid dt_skip')

    
        self.ts=np.zeros(self.Nt2)
        self.Sigma_g=np.zeros((2,self.grid.Nr,self.Nt2))
    
        ## Temperature and angular velocity
        # Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
        # Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
        # Omegas_s=Omegas/year_s # Omega in s-1
        # ## default viscosity
        # mus=np.ones(Nr)*mu0
    
    
        self.Sigma_g[:,:,0]=self.Sigma0
            
        Sigma_temp=self.Sigma_g[:,:,0]*1.0
        j=1
        for i in range(1,self.Nt):
            mask_m=np.sum(Sigma_temp, axis=0)>0.0
            self.mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
    
            Sigma_temp=self.Sigma_next(Sigma_temp, self.MdotCO[i])
            
            if i%self.dt_skip==0.0 or i==self.Nt-1:
                self.Sigma_g[:,:,j]=Sigma_temp*1.
                self.ts[j]=self.ts_sim[i]
                j+=1
        print('simulation finished')
        # return Sigma_g, ts2



            
class simulation_grid:

    default_rmin=1.0   # au
    default_rmax=3.0e3 # au
    default_Nr=100
    default_p=0.5
    
    def __init__(self, rmin=None, rmax=None, Nr=None, p=None):
        self.rmin=rmin if rmin is not None else default_rmin
        self.rmax=rmax if rmax is not None else default_rmax
        self.Nr=Nr if Nr is not None else default_Nr
        self.p=p if p is not None else default_p


        u=np.linspace(self.rmin**p, self.rmax**p, self.Nr+1) # Nr+1
        self.rhalfs=u**(1./p) # Nr+1
        self.hs=self.rhalfs[1:]-self.rhalfs[:-1] # Nr
        self.rs=0.5*(self.rhalfs[1:] + self.rhalfs[:-1])

        

    
### miscelaneous functions

def M_to_L(Mstar): # stellar mass to stellar L MS

    if hasattr(Mstar,"__len__"):
        L=np.zeros(Mstar.shape[0])
        L[Mstar<0.43]=0.23*Mstar[Mstar<0.43]**2.3
        mask2= ((Mstar>=0.43))# & (M<2)).
        L[mask2]=Mstar[mask2]**4.
        mask3= (Mstar>=2.) & (Mstar<20.)
        L[mask3]=1.4*Mstar[mask3]**3.5
        L[Mstar>55.]=3.2e4*Mstar[Mstar>55.]
        
        
    else:
        L=0.0
        if Mstar<0.45:
            L=0.23*Mstar**2.3
        elif Mstar<2.:
            L=Mstar**4.
        elif Mstar<20.:
            L=1.4*Mstar**3.5
        else:
            L=3.2e4*Mstar

    return L

def tau_vis(r, alpha, cs, Mstar):
    # r in au
    # cs in m/s
    # Mstar in Msun
    
    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (r*au_m)**2.0*Omega/(alpha*cs**2.)/year_s #/3.0

def radial_grid_powerlaw(rmin, rmax, Nr, alpha):

    u=np.linspace(rmin**alpha, rmax**alpha, Nr+1) # Nr+1
    rhalfs=u**(1./alpha) # Nr+1
    hs=rhalfs[1:]-rhalfs[:-1] # Nr
    rs=0.5*(rhalfs[1:] + rhalfs[:-1])
    return rs, rhalfs, hs



def N_optim_radial_grid(rmin, rmax, rb, res):

    Nr=10
    f=0
    while True:
        rs, rhalfs, hs = radial_grid_powerlaw(rmin, rmax, Nr, 0.5)  #0.5)
        for i in range(1,Nr):
            if rs[i]>rb:
                dr=rs[i]-rs[i-1]
                break
        if hs[i-1]/rs[i-1]<res:
            break
        else:
            Nr=int(Nr*1.2)
    return Nr

def tau_CO_photon_counting(Sigma_CO, Sigma_C1, fion=0.): # interpolate calculations based on photon counting

    tau=np.ones(Sigma_CO.shape[0])*130. # unshielded
    # to avoid nans we use a floor value for sigmas of 1e-50
    Sigma_COp=Sigma_CO*1. 
    Sigma_C1p=Sigma_C1*1.*(1.-fion)
    Sigma_COp[Sigma_COp<1.0e-50]=1.0e-50
    Sigma_C1p[Sigma_C1p<1.0e-50]=1.0e-50

    # mask=(Sigma_CO>1.0e-100) & (Sigma_C1>1.0e-100) # if not we get error in interpolation function and we get NaNs
    # if Sigma_CO[mask].shape[0]>0:
        # tau[mask]=10**(log10tau_interp(np.log10(Sigma_C1[mask]),np.log10(Sigma_CO[mask]), grid=False)) # yr, it must be called with C1 first because of column and raws definition. Tested with jupyter notebook and also here https://github.com/scipy/scipy/issues/3164

    tau=10**(log10tau_interp(np.log10(Sigma_C1p),np.log10(Sigma_COp), grid=False)) # yr, it must be called with C1 first because of column and raws definition. Tested with jupyter notebook and also here https://github.com/scipy/scipy/issues/3164
    return tau # yr
