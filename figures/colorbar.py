import numpy as np
from matplotlib.colors import ColorbarBase
from .base import Base


class ColorBar(Base):
    """
    Standalone colorbar.
    """

    def __init__(self, figsize=(5, 1), **kwargs):
        """
        Instantiate standalone colorbar.

        Args:
        figsize (tuple) - dimensions
        vmin, vmax (float) - bounds for colorscale
        cmap (matplotlib.colors.ColorMap) - color map
        """
        self.fig = self.create_figure(figsize=figsize)
        self.render(**kwargs)

    def render(self,
                vmin=0, vmax=1,
                cmap=plt.cm.plasma,
                label=None):
        """
        Plot standalone colorbar.

        Args:
        vmin, vmax (float) - bounds for colorscale
        cmap (matplotlib.colors.ColorMap) - color map
        """

        # get axis
        ax = self.fig.axes[0]

        # plot colorbar
        cbar = ColorbarBase(ax,
                            cmap=cmap,
                            norm=Normalize(vmin, vmax),
                            orientation='horizontal')

        # format ticks
        ax.xaxis.set_ticks_position('top')
        cbar.set_ticks(np.arange(0, 1.1, .2))
        ax.tick_params(pad=1)

        # add label
        if label is not None:
            cbar.set_label(label, labelpad=5)
