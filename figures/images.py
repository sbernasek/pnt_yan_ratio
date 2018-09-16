__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt

from .base import Base
from .settings import *


class Projection(Base):
    """
    Object for plotting maximum intensity projection.

    Attributes:
    projection (data.image.Image) - maximum intensity projection across layers spanning multipotent cells

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    # set class path and name
    path = 'graphics/images'

    def __init__(self, disc, **kwargs):
        """
        Instantiate object for plotting maximum intensity projection.

        Args:
        disc (data.discs.Disc)
        kwargs: keyword arguments for constructing maximum intensity projection
        """

        # load imagestack
        stack = disc.load_imagestack()

        # get multipotent layers
        layers = disc.get_multipotent_layers()

        # perform maximum intensity projection
        self.projection = stack.project_max(*layers, **kwargs)

        # apply smoothing
        self.projection.smooth()

        self.fig = None

    def render(self, species, figsize=(2, 2)):
        """
        Render image.

        Args:
        species (str) - species to be plotted, either 'pnt', 'yan', or 'both'
        figsize (tuple) - figure size
        """
        self.fig = self.create_figure(figsize=figsize)
        self.plot(self.fig.axes[0], species=species)
        plt.tight_layout()

    def render_all(self, figsize=(6, 6)):
        """
        Render a composite image of Pnt and Yan, an image of the pixelwise difference, and standalone images for each on four adjacent axes.

        Args:
        figsize (tuple) - figure size
        """

        # create figure
        self.fig = self.create_figure(nrows=2, ncols=2, figsize=figsize)
        ax00, ax01, ax10, ax11 = self.fig.axes

        # render images
        self.plot(ax00, 'both')
        self.plot(ax01, 'difference')
        self.plot(ax10, 'pnt')
        self.plot(ax11, 'yan')
        plt.tight_layout()

    def plot(self, ax, species):
        """
        Plot image on axis.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        species (str) - species to be plotted, either 'pnt', 'yan', 'both', or 'difference'
        """

        # visualize Pnt
        if species == 'pnt':
            self.projection['g'].render(ax=ax)
            self.label(ax, s='PntGFP', c='k')

        # visualize Yan
        elif species == 'yan':
            self.projection['r'].render(ax=ax)
            self.label(ax, s='AntiYan', c='k')

        # visualize Pnt and Yan using magenta/green colorscheme
        elif species == 'both':
            self.projection.render(scheme='mg',
                                   channels='mg',
                                   reference='b', ax=ax)
            self.label(ax, s='PntGFP', c='g')
            self.label(ax, s='\nAntiYan', c='m')

        # visualize difference between Pnt and Yan
        else:
            pnt = self.projection['g']
            yan = self.projection['r']
            difference = pnt - yan
            difference.render(ax=ax, cmap=plt.cm.PiYG, vmin=-.3, vmax=.3)
            self.label(ax, s='Difference \n(Pnt - Yan)', c='k')

    def label(self, ax, s, c='k', pad=25):
        """
        Add label to upper right corner of image.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        s (str) - label text
        c (str) - label color
        pad (int) - distance from upper corner, in pixels
        """
        xpos = self.projection.im.shape[0] - pad
        ax.text(xpos, pad, s, ha='right', va='top', color=c, weight='bold')
