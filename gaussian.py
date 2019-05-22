
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt


def gaussian_filter(img, blur_intensity):
    # Prepare an Gaussian convolution kernel

    # First a 1-D  Gaussian
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-blur_intensity*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

    # Implement convolution via FFT
    # Padded fourier transform, with the same shape as the image
    kernel_ft = np.fft.fftn(kernel, s=img.shape[:2], axes=(0, 1))
    img_ft = np.fft.fftn(img, axes=(0, 1))

    # the 'newaxis' is to match to color direction
    img2_ft = kernel_ft[:, :, np.newaxis] * img_ft

    img2 = np.fft.ifft2(img2_ft, axes=(0, 1)).real

    return img2
