import matplotlib.pyplot as pl_
import numpy as np_
from scipy.signal import convolve2d
from skimage import exposure, filters






def ContrastNormalized(frame, percentile : list) -> np_.ndarray:
    
    """
    ContrastNormalized function 
    This function normalizes the background using the contrast streching method,
    the histogram of the pixel intesities is normalized between [min_percentile, max_percentile]
    """

    kernel = np_.array([[1,2,1],[2,4,2],[1,2,1]]) 
    kernel = kernel/np_.sum(kernel) # normalized gaussian filter
    
    #convolution
    edges = convolve2d(frame, kernel, mode='same') 
    

    # contrast stretching
    p_inf = np_.percentile(edges, percentile[0])
    p_sup = np_.percentile(edges, percentile[1])
    img = exposure.rescale_intensity(frame, in_range=(p_inf, p_sup)) #stretching image intensity
    
    smooth_frm = filters.gaussian(img, sigma=(5,3), multichannel=None) # smooth image with a gaussian filter
    

    #pl_.matshow(smooth_frm)
        
    return smooth_frm



