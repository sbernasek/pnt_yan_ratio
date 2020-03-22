__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt

from .base import Base
from .settings import *

from binding.analysis.titration import Grid


class PhaseDiagram(Base):
    """
    Object for plotting phase diagram of equilibrium binding site occupancies across a two-dimensional concentration grid.

    Attributes:
    element (binding.model.elements.Element) - binding element

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, element):
        """
        Instantiate object for plotting binding site occupancy phase diagram.

        Args:
        element (binding.model.elements.Element) - binding element
        """

        # store element
        self.element = element

        # initialize figure
        self.fig = None

    def render(self,
                cmin=(0, 0),
                cmax=(100, 100),
                Nc=(25, 25),
                figsize=(2, 2),
                **kwargs):
        """
        Render equilibrium binding site occupancy phase diagram.

        Args:
        cmin, cmax (tuple) - concentration bounds for phase diagram
        Nc (tuple) - sampling density for each concentration axis
        figsize (tuple) - figure size
        kwargs: keyword arguments for binding model
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # evaluate and plot phase diagram
        self.plot(cmin=cmin,
                  cmax=cmax,
                  Nc=Nc,
                  **kwargs)

    def plot(self,
            cmin=(0, 0),
            cmax=(100, 100),
            Nc=(25, 25),
            **kwargs):
        """
        Plot expression dynamics for each cell type.

        Args:
        cmin, cmax (tuple) - concentration bounds for phase diagram
        Nc (tuple) - sampling density for each concentration axis
        kwargs: keyword aguments for binding model
        """

        # construct concentration grid and evaluate binding site occupancies
        grid = Grid(cmin=cmin, cmax=cmax, Nc=Nc)
        if self.element is None:
            model = grid.run_simple_model(**kwargs)
        else:
            model = grid.run_binding_model(self.element, **kwargs)
        self.model = model

        # plot phase diagram
        model.plot_phase_diagram('Pnt',
                                 cmap=plt.cm.PiYG,
                                 ax=self.fig.axes[0])

    @staticmethod
    def _format_ax(ax):
        """
        Format axis.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        """
        pass


class TitrationContours(PhaseDiagram):
    """
    Object for visualizing a titration contour. Figure shows equilibrium Pnt binding site occupancies for each binding site position as a function of Pnt protein concentration. Yan levels are held constant at a specified value.

    Attributes:
    element (binding.model.elements.Element) - binding element

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def render(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                figsize=(3, 2),
                **kwargs):
        """
        Render titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        kwargs: keyword aguments for binding model
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # evaluate and plot phase diagram
        self.plot(yan_level=yan_level,
                  cmin=cmin,
                  cmax=cmax,
                  Nc=Nc,
                  **kwargs)

    def plot(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                **kwargs):

        """
        Plot titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        kwargs: keyword aguments for binding model
        """

        # construct concentration grid and evaluate binding site occupancies
        grid = Grid(cmin=(yan_level, cmin), cmax=(yan_level, cmax), Nc=(1, Nc))
        model = grid.run_binding_model(self.element, **kwargs)
        self.model = model

        # plot titration contours
        _ = model.plot_titration_contours(fixed=0, fig=self.fig)


class OverallTitrationContours(PhaseDiagram):
    """
    Object for visualizing an overall titration contour. Figure shows equilibrium Pnt binding site occupancies averaged across all binding site positions as a function of Pnt protein concentration. Yan levels are held constant at a specified value.

    Attributes:
    element (binding.model.elements.Element) - binding element

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, *elements):
        """
        Instantiate object for plotting overall titration contours.

        Args:
        elements (binding.model.elements.Element) - binding element(s)
        """

        # store binding elements
        self.elements = elements

        # instantiate model list
        self.models = []

        # initialize figure
        self.fig = None

    def render(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                figsize=(3, 2),
                **kwargs):
        """
        Render titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        kwargs: keyword aguments for binding model
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # evaluate and plot phase diagram
        self.plot(yan_level=yan_level,
                  cmin=cmin,
                  cmax=cmax,
                  Nc=Nc,
                  **kwargs)

    def plot(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                colors=None,
                **kwargs):

        """
        Plot titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        colors (iterable) - line colors, must be same length as self.elements
        kwargs: keyword aguments for binding model
        """

        # construct concentration grid
        grid = Grid(cmin=(yan_level, cmin), cmax=(yan_level, cmax), Nc=(1, Nc))

        # make sure color vector is the correct size
        if colors is not None:
            assert len(colors) == len(self.elements), 'Wrong number of colors.'
        else:
            colors = ['k' for k in range(len(self.elements))]

        # evaluate binding site occupancies
        for i, element in enumerate(self.elements):
            model = grid.run_binding_model(element, **kwargs)
            self.models.append(model)
    
            # plot titration contours
            _ = model.plot_overall_titration_contour(species='Pnt', 
                                                     variable='Pnt', 
                                                     fixed=0,
                                                     color=colors[i], 
                                                     ax=self.fig.axes[0])

        ax = self.fig.axes[0]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
