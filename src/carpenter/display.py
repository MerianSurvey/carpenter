import os
import copy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from astropy.visualization import AsymmetricPercentileInterval, ZScaleInterval
from astropy.visualization.lupton_rgb import AsinhMapping, LinearMapping
from matplotlib import colors

from matplotlib.ticker import (AutoMinorLocator, FormatStrFormatter,
                               MaxNLocator, NullFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.sequential import (Blues_9, Greys_9, OrRd_9,
                                               Purples_9, YlGn_9)


def random_cmap(ncolors=256, background_color='white'):
    """Random color maps, from ``kungpao`` https://github.com/dr-guangtou/kungpao.

    Generate a matplotlib colormap consisting of random (muted) colors.
    A random colormap is very useful for plotting segmentation images.

    Parameters
        ncolors : int, optional
            The number of colors in the colormap.  The default is 256.
        random_state : int or ``~numpy.random.RandomState``, optional
            The pseudo-random number generator state used for random
            sampling.  Separate function calls with the same
            ``random_state`` will generate the same colormap.

    Returns
        cmap : `matplotlib.colors.Colormap`
            The matplotlib colormap with random colors.

    Notes
        Based on: colormaps.py in photutils

    """
    prng = np.random.mtrand._rand

    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)

    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    if background_color is not None:
        if background_color not in colors.cnames:
            raise ValueError('"{0}" is not a valid background color '
                             'name'.format(background_color))
        rgb[0] = colors.hex2color(colors.cnames[background_color])

    return colors.ListedColormap(rgb)


# About the Colormaps
IMG_CMAP = copy.copy(matplotlib.cm.get_cmap("viridis"))
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')

BLK = Greys_9.mpl_colormap
ORG = OrRd_9.mpl_colormap
BLU = Blues_9.mpl_colormap
GRN = YlGn_9.mpl_colormap
PUR = Purples_9.mpl_colormap


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xysize=(8, 8),
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   alpha=1.0,
                   no_negative=False,
                   percentiles=[1.0, 99.0],
                   cmap=IMG_CMAP,
                   norm=None,
                   scale_bar=True,
                   scale_bar_kwargs=None,
                   color_bar=False,
                   color_bar_kwargs=None,
                   add_text=None,
                   usetex=True,
                   text_fontsize=30,
                   text_y_offset=0.80,
                   text_color='w'):
    """
    Display single image. We use ``arcsinh`` stretching, "zscale" scaling 
    and ``viridis`` colormap as default. This function is borrowed from 
    ``kungpao`` https://github.com/dr-guangtou/kungpao.

    Parameters
    ----------
    img : numpy.ndarray. The image.
    pixel_scale : float, optional. The pixel scale in arcsec/pixel.
    physical_scale : float, optional. The physical scale in kpc/arcsec.
    xysize : tuple, optional. The size of the image in inches.
    ax : matplotlib.axes.Axes, optional. The axes to plot on.
    stretch : str, optional. The stretch function to use. 
        Options are "arcsinh", "log", "log10" and "linear".
    scale : str, optional. The scale function to use
        Options are "zscale" and "percentile".
    contrast : float, optional. The contrast of the image. 
        Only valid when using "zscale".
    alpha : float, optional. The alpha value of the image.
    no_negative : bool, optional. Whether to clip the negative values.
    percentiles : float, optional. The lower and upper percentile 
        used to scale the image if ``scale="percentile"``.
    cmap : matplotlib.colors.Colormap, optional. The colormap to use.
    norm : matplotlib.colors.Normalize, optional. The normalization to use.
    scale_bar : bool, optional. Whether to add a scale bar.
    scale_bar_kwargs : dict, optional. The keyword arguments for the scale bar.
        e.g., {"color": "white", "fontsize": 15, "y_offset": 0.5, "length": 5.0}
    color_bar : bool, optional. Whether to add a color bar.
    color_bar_kwargs : dict, optional. The keyword arguments for the color bar.
        e.g., {"loc": "1", "width": "75%", "height": "5%", "fontsize": 15, "color": "white"}
    add_text : str, optional. The text to add to the image.
    usetex : bool, optional. Whether to use latex to render the text.
    text_fontsize : int, optional. The font size of the text.
    text_y_offset : float, optional. The y offset of the text.
    text_color : str, optional. The color of the text.

    Returns
    -------
    ax : matplotlib.axes.Axes. The axes object, if ``ax`` is not ``None``.
    """

    if ax is None:
        fig = plt.figure(figsize=xysize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=percentiles[0],
                upper_percentile=percentiles[1]).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap, norm=norm,
                      vmin=zmin, vmax=zmax, alpha=alpha)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        scale_bar_length = scale_bar_kwargs.get('length', 5.0)
        scale_bar_fontsize = scale_bar_kwargs.get('fontsize', 20)
        scale_bar_y_offset = scale_bar_kwargs.get('y_offset', 0.5)
        scale_bar_color = scale_bar_kwargs.get('color', 'w')
        scale_bar_loc = scale_bar_kwargs.get('loc', 'left')

        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)
        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x * 0.08)
        text_y_0 = int(img_size_y * text_y_offset)
        if usetex:
            ax.text(text_x_0, text_y_0,
                    r'$\mathrm{' + add_text + '}$',
                    fontsize=text_fontsize, color=text_color)
        else:
            ax.text(text_x_0, text_y_0, add_text,
                    fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        color_bar_loc = color_bar_kwargs.get('loc', 1)
        color_bar_width = color_bar_kwargs.get('width', "75%")
        color_bar_height = color_bar_kwargs.get('height', "5%")
        color_bar_fontsize = color_bar_kwargs.get('fontsize', 18)
        color_bar_color = color_bar_kwargs.get('color', 'w')

        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def channels_to_rgb(channels):
    """
    From ``scarlet.display``.

    Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the the channels are ordered in wavelength
    direction, starting with the shortest wavelength. The mapping seeks to produce
    a relatively even weights for across all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.

    Parameters
    ----------
    channels: int in range(0,7)
        Number of channels

    Returns
    -------
    array (3, channels) to map onto RGB
    """
    assert channels in range(
        0, 8
    ), "No mapping has been implemented for more than {} channels".format(channels)

    channel_map = np.zeros((3, channels))
    if channels == 1:
        channel_map[0, 0] = channel_map[1, 0] = channel_map[2, 0] = 1
    elif channels == 2:
        channel_map[0, 1] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[1, 0] = 0.333
        channel_map[2, 0] = 0.667
        channel_map /= 0.667
    elif channels == 3:
        channel_map[0, 2] = 1
        channel_map[1, 1] = 1
        channel_map[2, 0] = 1
    elif channels == 4:
        channel_map[0, 3] = 1
        channel_map[0, 2] = 0.333
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.667
        channel_map[2, 1] = 0.333
        channel_map[2, 0] = 1
        channel_map /= 1.333
    elif channels == 5:
        channel_map[0, 4] = 1
        channel_map[0, 3] = 0.667
        channel_map[1, 3] = 0.333
        channel_map[1, 2] = 1
        channel_map[1, 1] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 1.667
    elif channels == 6:
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    elif channels == 7:
        channel_map[:, 6] = 2 / 3.
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    return channel_map


def img_to_3channel(img, channel_map=None, fill_value=0):
    """
    From ``scarlet.display``.

    Convert multi-band image cube into 3 RGB channels.

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    fill_value: float, default=`0`
        Value to use for any masked pixels.

    Returns
    -------
    RGB: numpy array with dtype float
    """
    # expand single img into cube
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    C = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(C)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(channel_map, img_.reshape(C, -1)).reshape(3, ny, nx)

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(img, channel_map=None, fill_value=0, norm=None, mask=None):
    """
    From ``scarlet.display``.

    Convert images to normalized RGB.
    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    norm: `scarlet.display.Norm`, default `None`
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.
    mask: array_like
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.

    Returns
    -------
    rgb: numpy array with dimensions (3, height, width) and dtype uint8
    """
    RGB = img_to_3channel(img, channel_map=channel_map)
    if norm is None:
        norm = LinearMapping(image=RGB)
    rgb = norm.make_rgb_image(*RGB)
    if mask is not None:
        rgb = np.dstack([rgb, ~mask * 255])
    return rgb


def display_merian_cutout_rgb(images, filters=list('griz') + ['N708'],
                              ax=None, half_width=None,
                              minimum=-0.15, stretch=1.2, Q=3,
                              color_norm=None, channel_map=None, 
                              N708_strength=0.7
                              ):
    """
    Display RGB color-composite image for HSC + Merian cutout. 
    Currently Merian N708 is highlighted in purple. N504 is not highlighted.

    Parameters
    ----------
    images: numpy array, shape (n_bands, height, width).
        Images to display.
    filters: list of str, default ['g', 'r', 'i', 'z', 'N708', 'N504']
        List of filters corresponding to the images.
    ax: matplotlib.axes.Axes, default None
        Axes to plot on.
    half_width: int, default None.
        If provided, the image will be cropped to this width.
    minimum: float, default -0.15.
        Minimum value to use for the color-mapping.
    N708_strength: float, default 0.7.
        Higher values will make the N708 band more prominent.
    stretch: float, default 1.2. See `astropy.visualization.make_lupton_rgb`.
    Q: float, default 3. See `astropy.visualization.make_lupton_rgb`.
    color_norm: `scarlet.display.Norm`, default None.

    Returns
    -------
    ax: matplotlib.axes.Axes.
    img_rgb: numpy array of the output RGB image (0-255), shape (height, width, 3).
    """
    

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Crop
    if half_width is not None:
        if half_width * 2 < min(images.shape[1], images.shape[2]):
            cen = (images.shape[1] // 2, images.shape[2] // 2)
            images = images[:, cen[0] - half_width:cen[0] +
                            half_width, cen[1] - half_width:cen[1] + half_width]

    # Construct the color matrix
    if color_norm is None:
        color_norm = {'g': 1.9, 'r': 1.2, 'i': 1.0,
                      'z': 0.85, 'y': 0.5, 'N708': 1.2, 'N540': 1.0}

    if channel_map is None:
        channel_map = channels_to_rgb(len('griz'))
        _map = np.zeros((3, len(filters)))
        _map[:, :4] = channel_map
        if 'N708' in filters:
            _map[0, 4] = N708_strength  # N708 for red
        # _map[0, 5] = 0.2
        _map /= _map.sum(axis=1)[:, None]
    else:
        _map = channel_map 
    # Now the color matrix is actually a multiplication between ``_map`` and ``f_c``.

    f_c = np.array([color_norm[filt] for filt in filters])
    _images = images * f_c[:, np.newaxis, np.newaxis]

    # Display
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
    img_rgb = img_to_rgb(_images, norm=norm, channel_map=_map)

    ax.imshow(img_rgb, origin='lower')
    ax.axis('off')

    if ax is None:
        return fig, img_rgb
    return ax, img_rgb
