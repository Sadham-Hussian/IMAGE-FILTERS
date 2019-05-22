import numpy as np
import matplotlib.pyplot as plt

from utils import rgb2gray, gaussian_filter

'''
    Base Class for all Filters
'''
class Filter:

    '''
        Changes the channel with respect to given value
    '''
    def channel_adjust(self,channel, values):
        orig_size = channel.shape
        flat_channel = channel.flatten()
        adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)

        return adjusted.reshape(orig_size)

    '''
        function applies the the according filter
    '''
    def apply_filter(self,img, *args):
        raise NotImplementedError

'''
    Implementation of an Instagram Filter named Gotham filter
        -> Channels down the r-channel
            and increases the b-channel
        -> Blurs the given Image to an appropriate Proportion 
'''
class GothamFilter(Filter):

    '''
        Applies the default filter
    '''
    def apply_filter(self, original_image, *args):
        r = original_image[:, :, 0]
        b = original_image[:, :, 2]
        r_boost_lower = self.channel_adjust(r, [
            0, 0.05, 0.1, 0.2, 0.3,
            0.5, 0.7, 0.8, 0.9,
            0.95, 1.0])
        b_more = np.clip(b + 0.03, 0, 1.0)
        merged = np.stack([r_boost_lower, original_image[:, :, 1], b_more], axis=2)

        '''
            Blurs the image. FFT is used here.
        '''
        blurred = gaussian_filter(merged, 0.001)

        final = np.clip(merged * 1.3 - blurred * 0.3, 0, 1.0)
        b = final[:, :, 2]
        b_adjusted = self.channel_adjust(b, [
            0, 0.047, 0.198, 0.251, 0.318,
            0.392, 0.42, 0.439, 0.475,
            0.561, 0.58, 0.627, 0.671,
            0.733, 0.847, 0.925, 1])

        final[:, :, 2] = b_adjusted
        return final

class RiverdaleFilter(Filter):
    def apply_filter(self, original_image, *args):
        r = original_image[:, :, 0]
        b = original_image[:, :, 2]
        r_boost_lower = self.channel_adjust(r, [
            0, 0.05, 0.1, 0.2, 0.3,
            0.5, 0.7, 0.8, 0.9,
            0.95, 1.0])
        b_more = np.clip(b + 0.2, 0, 1.0)
        merged = np.stack([r_boost_lower, original_image[:, :, 1], b_more], axis=2)


        ## Note: This has been changed to use the custom-defined Gaussian filter using FFT
        blurred = gaussian_filter(merged, 0.1)

        final = np.clip(merged + blurred*0.3, 0, 1.0)
        b = final[:, :,1]
        return final

class RandomFilter(Filter):
    def apply_filter(self, original_image, *args):
        r = original_image[100:, :, 0]
        b = original_image[100:, :, 2]
        r_boost_lower = self.channel_adjust(r, [
            0, 0.05, 0.1, 0.2, 0.3,
            0.5, 0.7, 0.8, 0.9,
            0.95, 1.0])
        # b_more = np.clip(b + 0.2, 0, 1.0)
        merged = np.stack([r_boost_lower, original_image[100:, :, 1], b], axis=2)


        # Apply the Gaussian Filter in the frequency domain to average the color values
        blurred = gaussian_filter(merged, 0.1)

        final = np.clip(merged + blurred*0.3, 0, 1.0)
        b = final[100:, :,1]
        return final


class GrayscaleFilter(Filter):
    def apply_filter(self, original_image, *args):
        r = original_image[:, :, 0]
        g = original_image[:, :, 1]
        b = original_image[:, :, 2]
        gray = rgb2gray(original_image)
        r = g = b = gray
        merged = np.stack([r,g, b], axis=2)


        # Apply the Gaussian Filter in the frequency domain to average the color values
        blurred = gaussian_filter(merged, 0.1)

        final = np.clip(merged + blurred*0.3, 0, 1.0)
        return final


class BlurFilter(Filter):
    def apply_filter(self, original_image, amount, *args):
        blurred = gaussian_filter(original_image, 1/amount)
        return blurred


class MultipleFilter(Filter):
    def apply_filter(self, original_image, *args):
        final = original_image
        for custom_filter in args:
            f = custom_filter()
            final = f.apply_filter(final)

        return final
