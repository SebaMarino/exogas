import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
from exogas.constants import *
import exogas.radial_evolution as revol

class simulation:

    def __init__(self, Mstar=None, Lstar=None, Nz=None, zmax_to_H=None, T=None, rbelt=None, width=None, MdotCO=None,  alphar=None, alphav=None, mu0=None,tf=None, dt0=None, verbose=True, diffusion=True, photodissociation=True, Ntout=None):


        # fco=None, fir=None, fion=None, mu0=None, Sigma_floor=None, rc=None,  tf=None, dt0=None, verbose=True, dt_skip=10, diffusion=True, photodissociation=True, carbon_capture=False, pcapture=None, MdotCO=None, tcoll=None, co_reformation=False,  preform=None, mixed=True):
      
        
        ################################
        ### default parameters
        ################################

        # system
        default_Mstar=2.0 # Msun
        default_Nz=20
        default_zmax_to_H=4. # Zmax/H
        
        ## belt parameters
        default_rbelt=100.0 # au 
        default_width=default_rbelt*0.5  # au, FWHM
        default_sig_belt=default_width/(2.0*np.sqrt(2.*np.log(2.)))
        default_MdotCO=1.0e-7 # Mearth/yr
        
        #default_fir=1.0e-3 # fractional luminosity
        #default_tcoll=-1.
        
        ## gas parameters
        default_alphar=1.0e-3
        default_alphav=1.0e-3
        # default_fco=0.1
        # default_fion=0.0
        default_mu0=14.0
        
        ##  simulation parameters
        default_tf=1.0e6 # yr
        default_dt0=40. # yr (maximum dt)
        default_Ntout=10
        
        # default_pcapture=1.
        # default_preform=1.
        
        ### system
        self.Mstar=Mstar if Mstar is not None else default_Mstar
        self.Lstar=Lstar if Lstar is not None else revol.M_to_L(self.Mstar)
        self.Nz=Nz if Nz is not None and Nz>0 else default_Nz
        self.zmax_to_H=zmax_to_H if zmax_to_H is not None and zmax_to_H>0 else default_zmax_to_H
        
    
        #### belt
        self.rbelt=rbelt if rbelt is not None else default_rbelt
        self.width=width if width is not None else self.rbelt*0.5
        self.sig_belt=self.width/(2.0*np.sqrt(2.*np.log(2.)))
        self.MdotCO=MdotCO if MdotCO is not None and MdotCO>0.0 else default_MdotCO
        # self.fir=fir if fir is not None else default_fir
        # try: 
        #     if tcoll>0.:
        #         self.tcoll=tcoll
        #     else: self.tcoll=default_tcoll
        # except:
        #     self.tcoll=default_tcoll

        #### gas parameters
        # self.fco=fco if fco is not None else default_fco
        self.alphar=alphar if alphar is not None else default_alphar
        self.alphav=alphav if alphav is not None else default_alphav
        # self.fion=fion if fion is not None else default_fion
        self.mu0=mu0 if mu0 is not None else default_mu0
        self.tf=tf if tf is not None else default_tf
        self.dt0=dt0 if dt0 is not None else default_dt0
        
        ################################
        #### calculate basic properties of the simulation
        ################################

        #### temperature and viscosity at the belt center
        if T==None or T<=0.0:
            self.Tb=278.3*(self.Lstar**0.25)*self.rbelt**(-0.5) # K # Temperature at belt
        else: self.Tb=T
        self.cs=np.sqrt(kb*self.Tb/(self.mu0*mp)) # m/s sounds speed at belt
        self.Omega=2.0*np.pi*np.sqrt(self.Mstar/(self.rbelt**3.0)) # 1/yr
        self.Omega_s=self.Omega/year_s
        self.H=self.cs/self.Omega_s/au_m # au
        self.nur=self.alphar*kb*self.Tb/(self.mu0*mp)/(self.Omega_s) # m2/s 1.0e13*np.ones(Nr) #
        self.nur_au2_yr=self.nur*year_s/(au_m**2.0)
        self.nuv=self.alphav*kb*self.Tb/(self.mu0*mp)/(self.Omega_s) # m2/s 1.0e13*np.ones(Nr) #
        self.nuv_au2_yr=self.nuv*year_s/(au_m**2.0)

        ## viscous timescales
        self.tvisr=self.rbelt**2./self.nur_au2_yr
        self.tvisv=self.H**2./self.nuv_au2_yr

        
        #### spatial grid
        self.zmax=self.zmax_to_H*self.H # au
        self.zs=np.linspace(0., self.zmax, self.Nz)
        self.dz=(self.zs[1]-self.zs[0])

        
        #### temporal grid
        self.dt=min(0.1*self.dz**2./self.nuv_au2_yr, self.dt0) # yr 
        self.Nt=int(self.tf/self.dt)+1
        self.ts_sim=np.linspace(0.0, self.tf, self.Nt)
        self.Ntout=Ntout if Ntout is not None and Ntout>0 else default_Ntout
        self.dt_skip=int((self.Nt)/self.Ntout)
        
        
        dir_path = os.path.dirname(os.path.realpath(__file__))+'/photodissociation/'

        table_selfshielding=np.loadtxt(dir_path+'self-shielding.dat')
        self.logfCO = interpolate.interp1d(np.log10(table_selfshielding[0,:]), np.log10(table_selfshielding[1,:]))
        self.minNCO=table_selfshielding[0,0]
        self.maxNCO=table_selfshielding[0,-1]
        
        self.diffusion=diffusion
        self.photodissociation=photodissociation

        self.verbose=verbose                           
        if self.verbose:
            print('Zmax = %1.1f au'%(self.zmax))
            print('Nz = %i'%(self.Nz))
            print('Nt simulation=%i'%self.Nt)
            print('simulation timestep = %1.1f yr'%self.dt)
            # print('vertical diffusion timescale to cross one vertical bin = %1.1e yr'%(self.dz**2./self.nuv_au2_yr))
            print('t_diffusion = %1.1e yr'%self.tvisv)
            print('t_vis = %1.1e yr'%self.tvisr)
            print('MdotCO at t=0 is %1.1e Mearth/yr'%(self.MdotCO))
            print('T = %1.1f K'%(self.Tb))


        ### expected surface density
        self.Sigma_eq=self.MdotCO/(3.*np.pi*self.nur_au2_yr)
        self.rho_eq_midplane=self.Sigma_eq/(np.sqrt(2.*np.pi)*self.H)
        self.Sigma_dot=self.MdotCO/(2.*np.pi*self.rbelt*np.sqrt(2.*np.pi)*self.sig_belt)
        self.rho_eq_unshielded=self.Sigma_dot*130./(np.sqrt(2.*np.pi)*self.H)

        ######## initial condition
        self.rhos0=np.zeros((3, self.Nz)) 
        self.rhotot=np.zeros(self.Nz)
        self.rhotot= self.rho_eq_unshielded * np.exp( - 0.5* (self.zs/self.H)**2 )
        self.rhos0[1,:]=self.rhotot*0.01
        self.rhos0[0,:]=self.rhotot-self.rhos0[1,:]*28./12
        
      

    ##############################################
    ################ METHODS ###################
    ##############################################



    def Diffusion(self, rho_temp):
        # assume 0 index is midplane and -1 index is at Zmax
    
        rhosp1=np.concatenate([rho_temp[:,1][:,None],rho_temp], axis=1)

    
        rho_tot=rhosp1[0,:]+(rhosp1[1,:]+rhosp1[2,:])*(28./12.) # CO+CI+CII+O, N+1
        eps=np.ones((3,self.Nz+1))
        mask_m=rho_tot>0.

        eps[0,mask_m]=rhosp1[0,mask_m]/rho_tot[mask_m]
        eps[1,mask_m]=rhosp1[1,mask_m]/rho_tot[mask_m]
        eps[2,mask_m]=rhosp1[2,mask_m]/rho_tot[mask_m]
    
        eps_dot_half=(eps[:,1:]-eps[:,:-1])/(self.dz) # Nz at cell boundaries
    
        rho_tot_half=(rho_tot[1:]+rho_tot[:-1])/2. # Nz at cell boundaries
    
        F_half=(self.nuv_au2_yr*rho_tot_half*eps_dot_half) # Nz at cell boundaries
        
        rhodot_diff=np.zeros((3,self.Nz))    
        rhodot_diff[:,:-1]= (F_half[:,1:]-F_half[:,:-1])/(self.dz) # Nz-1
    
        return rhodot_diff

    def Viscous_eovlution(self, rho_temp):

        tau= 2./3. * self.rbelt / self.nur_au2_yr * np.sqrt(2.*np.pi)*self.sig_belt # yr
        return rho_temp/tau

    def Gas_input(self):
        
        # Mdot in earth masses / yr
        return self.MdotCO/(2.*np.pi*self.rbelt*self.sig_belt*np.sqrt(2.*np.pi)) * np.exp(-0.5* (self.zs/self.H)**2.)/(np.sqrt(2.*np.pi)*self.H)


    def Photodissociation_CO(self,rho_temp):
    
        NCO_tot=2*np.trapz(rho_temp[0,:], self.zs)*Mearth/au_cm**2./(28.*mp)
        NCI_tot=2*np.trapz(rho_temp[1,:], self.zs)*Mearth/au_cm**2./(12.*mp)


        NCOs_top=np.cumsum(rho_temp[0,::-1])[::-1]*self.dz*Mearth/au_cm**2./(28.*mp)
        NCIs_top=np.cumsum(rho_temp[1,::-1])[::-1]*self.dz*Mearth/au_cm**2./(12.*mp)

        # NCOs_top=np.array([np.trapz(rho_temp[0,i:], self.zs[i:]) for i in range(len(self.zs))])*Mearth/au_cm**2./(28.*mp)
        # NCIs_top=np.array([np.trapz(rho_temp[1,i:], self.zs[i:]) for i in range(len(self.zs))])*Mearth/au_cm**2./(12.*mp)
    
        NCOs_bottom=NCO_tot-NCOs_top
        NCIs_bottom=NCI_tot-NCIs_top

        return (self.shielding_CO(NCOs_top, NCIs_top)  + self.shielding_CO(NCOs_bottom, NCIs_bottom))*rho_temp[0,:]/260. # Mearth/yr

    def shielding_CO(self, NCO, NCI):

        ## CO self-shielding
        kco=np.ones(len(NCO))
        
        mask1=(NCO>=self.minNCO) & (NCO<=self.maxNCO)
        kco[ mask1]=10.0**self.logfCO(np.log10(NCO[mask1]))
        mask2=( NCO<self.minNCO)
        kco[mask2]=1.0
        mask3=(NCO>self.maxNCO)
        kco[mask3]=0.0

        # CI shielding
        kcI=np.exp(-NCI*sigma_c1)

        return kco*kcI 

    def vertical_evolution(self):
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
        if self.verbose:
            print('Nt output=%i'%self.Nt2)
    
        self.ts=np.zeros(self.Nt2)
        self.rhos=np.zeros((3, self.Nz,self.Nt2 ))
        self.rhos[:,:,0]=self.rhos0
            
        rho_temp=self.rhos[:,:,0]*1.0
        j=1
        for i in range(1,self.Nt):
            
            rho_temp=self.Rho_next(rho_temp)
            
            if i%self.dt_skip==0.0 or i==self.Nt-1:
                self.rhos[:,:,j]=rho_temp*1.
                self.ts[j]=self.ts_sim[i]
                j+=1
        print('simulation finished')




    def Rho_next(self, rho_temp):

        ###########################################
        ################ viscous evolution
        ###########################################
        rho_next=rho_temp - self.dt*self.Viscous_eovlution(rho_temp)
        ###########################################
        ############### CO mas input rate
        ###########################################
        rho_next[0,:]=rho_next[0,:] + self.dt*self.Gas_input()
        ###########################################
        ################ diffusion evolution 
        ###########################################
        if self.diffusion:
            rho_next=rho_next+self.dt*self.Diffusion(rho_next)
        ###########################################
        ############## photodissociation
        ###########################################
        if self.photodissociation:
            R_photodissociation=self.Photodissociation_CO(rho_temp)
            rho_next[0,:]=rho_next[0,:]-self.dt*R_photodissociation
            rho_next[1,:]=rho_next[1,:]+12./28.*self.dt*R_photodissociation

        return rho_next
        
    

