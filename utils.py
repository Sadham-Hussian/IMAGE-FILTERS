import numpy as np
import matplotlib.pyplot as plt
import  skimage

'''
    Changes the given Coour Image into GreyScale
'''
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

'''
    Returns the Fourier Transform for the given Image
'''
def fourier(img):
    gray_img = rgb2gray(img)
    four = np.fft.fft2(gray_img)
    shifted_fourier = np.fft.fftshift(four)
    return shifted_fourier

'''
    Plots the Fourier Transform Represented by the given Matrix
'''
def plotFourier(fourier):
    psd2D = np.log(np.abs(fourier)**2+1)
    (height,width) = psd2D.shape
    plt.figure(figsize=(10,10*height/width),facecolor='white')
    plt.clf()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.imshow( psd2D, cmap='Greys_r',extent=[-np.pi,np.pi,-np.pi,np.pi],aspect='auto')
    plt.show()

'''
    Reads Image from the given path
'''
def read_image_from_path(path):
    if path.split('.')[0] == 'png':
        original_image = plt.imread(path).astype(float)
    else:
        original_image = skimage.img_as_float(skimage.io.imread(path))
    return original_image

'''
    Saves the Image at the given path
'''
def save_img_at_path(img, path):
    plt.imsave(path, img)

'''
    Given the intensity the Blur Effect is Applied using Gaussian filter
'''
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
