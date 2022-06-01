import numpy as np


def mask_image(image, sigma=2,):
    """
    Mask out pixels in image with value in mask.
    """
    from kuaizi.detection import vanilla_detection
    from astropy.convolution import convolve, Gaussian2DKernel
    _, segmap = vanilla_detection(image, sigma=sigma, conv_radius=1, convolve=True, show_fig=False, verbose=False)
    cenid = segmap[segmap.shape[0] // 2, segmap.shape[1] // 2] - 1
    segmap[segmap == (cenid + 1)] = 0.0

    smooth_radius = 3
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    mask = (mask_conv >= gaussian_threshold)
    return mask

def SBP_star(model_dict, hsc_data, ind=0):
    from kuaizi.tractor.utils import HiddenPrints
    from photutils.isophote import Ellipse, EllipseGeometry

    gal = model_dict['i'].catalog[model_dict['i'].target_ind]
    geometry = EllipseGeometry(x0=gal.pos.x, 
                               y0=gal.pos.y, 
                               sma=1, eps=0,
                               pa=0)
    channels = list('grizy') + ['N708', 'N540']
    with HiddenPrints():
        model_img = np.asarray([
            model_dict[key].getModelImage(
                0, srcs=model_dict[key].catalog[model_dict[key].target_ind:model_dict[key].target_ind+1]
                                         ) for key in channels])
    
    ellipse = Ellipse(model_img[ind], geometry)
    try:
        isolist_model = ellipse.fit_image(
    #                                  sma0=10, minsma=0.,
                                    nclip=3, integrmode='median',
                                    maxit=10, maxgerr=1.0,
                                    fix_center=True, fix_pa=True, fix_eps=True)
    except:
        pass
    try:
        isolist_model = ellipse.fit_image(fix_center=True, fix_pa=True, fix_eps=True)
    except:
        print('Failed to fit data')
        return 

    _mean_x = np.mean(isolist_model.x0[isolist_model.sma > 10])
    _mean_y = np.mean(isolist_model.y0[isolist_model.sma > 10])
    
    # Data
    mask = mask_image(hsc_data.images[ind])
    
    ellipse = Ellipse(hsc_data.images[ind] * (~mask), geometry)
    try:
        isolist_data = ellipse.fit_image(
    #                                  sma0=10, minsma=0.,
                                    nclip=3, integrmode='median',
                                    maxit=10, maxgerr=1.0,
                                    fix_center=False, fix_pa=True, fix_eps=True)
    except:
        pass
    try:
        isolist_data = ellipse.fit_image(fix_pa=True, fix_eps=True)
    except:
        print('Failed to fit data')
        isolist_data = None
    return (isolist_model, isolist_data)