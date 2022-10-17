# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:29:22 2021

@author: David Conran (original code), dnc7309@rit.edu
Modified by Austin Bergstrom
"""
import numpy as np
from scipy.integrate import simps
from scipy import interpolate
import matplotlib.pyplot as plt


class ParametricSysModel(object):
    
    def __init__(self, num_samples, f_num, pix_pitch, ff, wl):
        
        # Satellite System Parameters used in creating unique MTF/PRF model.
        self.num_samples = num_samples  # Number of samples
        self.f_num = f_num  # F-number
        self.pix_pitch = pix_pitch * 1e-6  # Pixel pitch [meters] ***Define wl in microns***
        self.ff = ff  # Aperture Fill Factor (1-(Ds/Dp)**2)
        self.wl = wl*1e-6  # Average wavelength [meters] ***Define wl in microns***
        
        # Defining two important parameters for system model. Used in scaling various MTF's
        self.f_opt_cutoff = 1 / (self.wl * self.f_num)  # incoherent optical cutoff freq [cyc/m]
        self.f_det_cutoff = 1 / self.pix_pitch  # detector cutoff freq [cyc/m]

        # Defining important parameters used in the model simulation. Nothing physical. 
        self.secondary_diameter_frac = np.sqrt(1 - self.ff)  # Fractional diameter of secondary to primary
        self.Ro = 2.0 * np.sqrt(self.num_samples / np.pi)  # Radius of aperture scaled with N (ask Dave)
        self.Ri = self.secondary_diameter_frac * self.Ro  # Radius of secondary aperture scaled to Ro (ask Dave)
        self.N_set = np.linspace(-self.num_samples // 2, (self.num_samples // 2) - 1, self.num_samples)  # ap space
        self.x, self.y = np.meshgrid(self.N_set,self.N_set)  # 2D matrices of aperture space
        
        # Defining variables used in saving important steps in the modeling process.
        self.open_apt = None  # Circlar aperture w/o central obscuration.
        self._psf = None  # PSF of circular aperture w/o central obscuration. Used for PSF scaling

        # Normalized frequency (Nyquist at 0.5 cyc/pixel), Frequency (Nyquist at half fdco cyc/m)
        self.fn, self.fx = None, None
        self.xn, self.xi = None, None  # Normalized PSF axis (pixels), Normalized PSF axis interpolated (pixels)

        #  MTFs from various sources
        self.aperture_mtf, self.det_tf, self.wfe, self.smear, self.jitter = None, None, None, None, None
        self.prf, self.mtf_sys = None, None  # Point Response Function and System MTF (related by FT)
        self.intp_prf = None  # Interpolated PRF (System PSF)
        self.ensqared_energy = None
        self.dxi = None
        self.uprf = None  # what does the "u" stand for?
        self.mtf_kernel = None  # blur kernel derived from the system mtf?
        self.sys_aperture = None
        self.cal_scale = None
        self.zero = None
        self.optical_psf = None

        self.apply_scaling()
        self.mtf_aperture()
        
    @staticmethod
    def fft2(img):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

    @staticmethod
    def ifft2(img):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(img)))

    @staticmethod
    def circ(x, y, r0):
        # believe this i basically a hyper-Gaussian
        eta = 50
        r = x**2 + y**2
        apt = np.exp(-(r/(r0**2))**eta)
        return apt

    @staticmethod
    def rect_2d(x, y, r01):
        eta = 50
        rect = np.exp(-((x/r01)**2)**eta)*np.exp(-((y/r01)**2)**eta)
        return rect

    @staticmethod
    def zcr(x, y):
        return x[np.diff(np.sign(y)) != 0]

    @staticmethod
    def mean_wl(wls,data):
        return simps(wls*data,wls)/simps(data,wls)

    @staticmethod
    def intp_1d(data, wl1=400, wl2=1000, wl_sp=1):
        # wavelengths need to be specified in nm
        x, y = data[:, 0], data[:, 1:]
        intp_x = np.arange(wl1, wl2+wl_sp, wl_sp)
        f = interpolate.interp1d(x, y, axis=0, kind='slinear')
        intp_y = f(intp_x)
        return intp_x, intp_y

    @staticmethod
    def intp_2d(fx, fy, img, fxi, fyi):
        intp_psf = interpolate.griddata((fx.ravel(),fy.ravel()), img.ravel(), (fxi,fyi), 
                                        method='cubic', fill_value=0)
        return intp_psf

    @staticmethod
    def center_crop(array, threshold_frac=0.999, step=4):
        h, w = np.shape(array)
        h_ctr = h // 2
        w_ctr = w // 2
        h_odd, w_odd = True, True
        top_edge, bottom_edge = h_ctr
        top_edge, bottom_edge = h_ctr

        if h % 2 != 0:
            h_odd = True
            h_ctr += 1
            top_edge, bottom_edge = h_ctr
        if w % 2 != 0:
            w_odd = True
            w_ctr += 1
            top_edge, bottom_edge = h_ctr

        crop_frac = 0
        while crop_frac < threshold_frac:
            bottom_edge += step
            top_edge -= step

    def get_zero(self):
        zeros = np.int64(self.zcr(self.N_set[self.num_samples // 2:self.num_samples // 2 + 300],
                                  self._psf[self.num_samples // 2, self.num_samples // 2:self.num_samples // 2 + 301])) + 1 # Finding 5th zero index
        x = np.array([zeros[0] - 1, zeros[0]])
        y = np.array([self._psf[self.num_samples // 2, x[0] + self.num_samples // 2], self._psf[self.num_samples // 2, x[1] + self.num_samples // 2]])
        
        f = interpolate.interp1d(y, x, axis=0, kind='slinear')
        intp_x = f(0)
        return intp_x
    
    def create_freq(self, dx):
        df = (1 / (dx * self.num_samples))
        fx = np.arange(-(1/(2*dx)), (1/(2*dx)), df)
        return fx
    
    def create_spat(self,dx):
        df = (1 / (dx * self.num_samples))
        x = np.arange(-(1/(2*df)),(1/(2*df)),dx)
        return x
    
    def aperture2ipsf(self, aperture):
        psf = abs(self.fft2(aperture))**2
        norm_psf = psf/np.sum(psf)
        return norm_psf
    
    def psf2mtf(self,psf):
        n = len(psf)
        otf = abs(self.fft2(psf))
        mtf = otf/otf[n//2,n//2]
        return mtf
    
    def mtf2prf(self, mtf):
        sys_psf = abs(self.ifft2(mtf))
        prf = sys_psf/np.sum(sys_psf)
        return prf

    def mtf_detector(self):
        # rho is a normalized frequency wrt the detector cutoff freq
        rho = self.fx/self.f_det_cutoff
        fx,fy = np.meshgrid(rho,rho)
        mask_x,mask_y = (fx != 0),(fy != 0)
        out_x,out_y = np.ones(fx.shape),np.ones(fy.shape)
        out_x[mask_x] = (np.sin(np.pi*fx[mask_x])/(np.pi*fx[mask_x]))
        out_y[mask_y] = (np.sin(np.pi*fy[mask_y])/(np.pi*fy[mask_y]))
        return out_x*out_y
    
    def mtf_smear(self, P):
        # rho is a normalized frequency wrt the detector cutoff freq
        rho = self.fx/self.f_det_cutoff
        fx, fy = np.meshgrid(rho,rho)
        mask_x, mask_y = (fx != 0), (fy != 0)
        out_x, out_y = np.ones(fx.shape), np.ones(fy.shape)
        out_x[mask_x] = (np.sin(np.pi*fx[mask_x]*P)/(np.pi*fx[mask_x]*P))
        out_y[mask_y] = (np.sin(np.pi*fy[mask_y]*P)/(np.pi*fy[mask_y]*P))
        return out_x * out_y
    
    def mtf_wfe_shannon(self, wl_rms):
        # rho is a normalized frequency wrt the optical cutoff freq
        rho = self.fx / self.f_opt_cutoff
        fx, fy = np.meshgrid(rho, rho)
        return np.nan_to_num(1-((wl_rms/0.18)**2)*np.sqrt(1-4*(np.sqrt(fx**2+fy**2)-(1/2))**2), nan=1)
    
    def mtf_wfe_hufnagel(self, wl_rms, l_corr):
        # rho is a normalized frequency wrt the optical cutoff freq
        rho = self.fx/self.f_opt_cutoff
        fx,fy = np.meshgrid(rho,rho)
        return np.exp(-4*(np.pi**2)*(wl_rms**2)*(1-np.exp(-4*((fx**2 + fy**2)/(l_corr**2)))))
    
    def mtf_jitter(self, sigma_j):
        # rho is a normalized frequency wrt the detector cutoff freq
        rho = self.fx/self.f_det_cutoff
        fx, fy = np.meshgrid(rho,rho)
        return np.exp(-2*(fx**2 + fy**2)*(np.pi**2)*(sigma_j**2))
    
    def apply_scaling(self):
        self.open_apt = self.circ(self.x, self.y, self.Ro)  # unobscured aperture
        self._psf = np.real(self.fft2(self.open_apt))  # Coherent PSF of unobscured aperture
        self.zero = self.get_zero()  # Find 5th zero w/ sub pixel accuracy
        self.cal_scale = (1 / self.zero) * (3.8317 / np.pi) * (self.wl * self.f_num)  # Scale 1st zero w/ sys parameters
        # Defining Frequency Response of the Optical System
        self.fx = self.create_freq(self.cal_scale) # Frequency domain [cyc/m]
        self.fn = self.create_freq(self.cal_scale/self.pix_pitch) # Frequency domain [cyc/pixel]
        self.xn = self.create_spat(self.cal_scale/self.pix_pitch) # PSF axis [pixels]
    
    def mtf_aperture(self):
        if self.ff != 0:  # shouldn't this be if self.ff != 1?
            self.sys_aperture = self.open_apt - self.circ(self.x, self.y, self.Ri)
            self.aperture_mtf = self.psf2mtf(self.aperture2ipsf(self.sys_aperture))
        else:
            self.aperture_mtf = self.psf2mtf(abs(self._psf) ** 2)  # Incoherent PSF requires the squaring
    
    def mtf_system(self, p, sigma_j, wl_rms, l_corr=None):
        if l_corr is None:
            self.wfe = self.mtf_wfe_shannon(wl_rms)
        else:
            self.wfe = self.mtf_wfe_hufnagel(wl_rms, l_corr)
        self.det_tf = self.mtf_detector()
        self.smear = self.mtf_smear(p)
        self.jitter = self.mtf_jitter(sigma_j)
        self.mtf_sys = self.aperture_mtf * self.det_tf * self.wfe * self.smear * self.jitter
        self.mtf_kernel = self.aperture_mtf * self.wfe * self.smear * self.jitter
        self.prf = self.mtf2prf(self.mtf_sys)  ####### needs to be mtf_sys, testing mtf_kernel
        self.optical_psf = self.mtf2prf(self.mtf_kernel)
    
    def interp_prf(self, size):
        #size = 5
        self.dxi = 0.01
        #self.mtf2prf(self.mtf_sys)
        bool_ = (abs(self.xn) <= size+2)
        xigrid = np.ix_(bool_, bool_)
        ensq_prf = self.prf[xigrid]
        ensq_axis = self.xn[bool_]
        self.ensqared_energy = np.sum(ensq_prf)
        xn, yn = np.meshgrid(ensq_axis, ensq_axis)
        self.xi = np.arange(-(size + 1), (size + 1), self.dxi)
        xn_, yn_ = np.meshgrid(self.xi, self.xi)
        self.intp_prf = self.intp_2d(xn, yn, ensq_prf, xn_, yn_)
        # return
    
    def create_uprf(self, sub_shiftx=None, sub_shifty=None, sub_sample=None):

        # If no shifts were given, return an unshifted uprf
        if (sub_shiftx is None) & (sub_shifty is None) & (sub_sample is None):
            sub_shiftx, sub_shifty = 0, 0
        M = len(self.xi)  # length of the spatial axis {-(size+1):dxi:(size+1)-dxi}
        p_sz = round((1/self.dxi))  # Index step size for discrete sampling
        samples = np.arange(p_sz, M, p_sz)  # Sampling array spaced by 1 pixel
        # If a single sub-pixel shift is given, returns shifted uprf

        if (sub_shiftx is not None) & (sub_shifty is not None):
            # Converts sub-pixel shift (-0.5 to 0.5 pixels) to an index shift
            sub_x, sub_y = round(sub_shiftx / self.dxi), -1 * round(sub_shifty / self.dxi)
            # Creates a 2D shifted sample grid to create a phased uprf
            sx, sy = np.meshgrid(samples - sub_x, samples - sub_y)
            uprf = np.array([self.intp_prf[sx, sy]])  # Samples PRF

        else:
            uprf_set = []
            # fact = 1/(sub_sample+1)
            # sub_set = np.linspace(-(0.5-fact),0.5-fact,sub_sample)
            # Generates a 1D array of random shift (-0.5 to 0.5 pixels)
            sub_set = np.random.rand(sub_sample) - 0.5
            subx, suby = np.meshgrid(sub_set, sub_set) # Creates 2D arrary for all x,y pairs of sub-pixel shifts
            # Unravels 2D x,y pairs and converts to index shifts
            subx, suby = np.int64(np.round(subx.ravel() / self.dxi)), -np.int64(np.round(suby.ravel() / self.dxi))
            # Loops through sub-pixel index shifts and appends to a list to be returned
            for i in range(len(subx)):
                sx, sy = np.meshgrid(samples-subx[i],samples-suby[i]) # 2D sample grid
                uprf_set.append(self.intp_prf[sx, sy]) # Samples PRF
            uprf = np.array(uprf_set) # Creates final uprf array
            
        dim, row, cols = uprf.shape
        # Find the total area under a single uprf or multiple uprfs
        total = uprf.reshape(dim, row*cols).sum(axis=-1)[:, None, None] # dimension (dim,1,1) for division
        # Reintroduce missing energy when cropping in interpolation function (interp_prf)
        self.uprf = (uprf / total) * self.ensqared_energy  # save uprf as instance attribute






if __name__ == '__main__':

    model = ParametricSysModel(2 ** 11, 14.55, 32, 0.8631, 0.912)
    # model.scaling()  # added to __init__()
    # model.mtf_aperture()  # added to __init__()
    model.mtf_system(0.1, 0.1, 0.1, 0.1)
    model.interp_prf(size=7)
    model.create_uprf()

    plt.plot(model.fn, model.mtf_sys[model.num_samples // 2, :])
    plt.plot(model.fn, model.det_tf[model.num_samples // 2, :])
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)

    plt.figure(figsize=(12,10))
    plt.plot(model.fn, model.aperture_mtf[model.num_samples // 2, :], label='Aperture')
    plt.plot(model.fn, model.det_tf[model.num_samples // 2, :], label='Detector')
    plt.plot(model.fn, model.wfe[model.num_samples // 2, :], label='WFE')
    plt.plot(model.fn, model.jitter[model.num_samples // 2, :], label='Jitter')
    plt.plot(model.fn, model.smear[model.num_samples // 2, :], label='Smear')
    plt.plot(model.fn, model.mtf_kernel[model.num_samples // 2, :], label='Kernel')
    plt.plot(model.fn, model.mtf_sys[model.num_samples // 2, :], label='System')
    plt.title('MTF Models',fontsize=18)
    plt.xlabel('Normalized Frequency [cyc/pixel]',fontsize=14)
    plt.ylabel('Magnitude',fontsize=14)
    plt.legend(prop={'size': 12})
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)

    plt.show()