__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from .base import Base
from .settings import *
from .palettes import Palette, line_colors


class Expression(Base):
    """
    Object for plotting expression dynamics.

    Attributes:
    experiment (data.experiments.Experiment)
    colorer (figures.palettes.Palette) - color palette for cell types
    greys (figures.palettes.Palette) - color palette for progenitors

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    # set class path and name
    path = 'graphics/expression'

    def __init__(self, experiment, **kwargs):
        """
        Instantiate object for plotting expression dynamics.

        Args:
        experiment (data.experiments.Experiment)
        include_neurons (bool) - if True, compile young R cell data for annotating time windows
        kwargs: keyword arguments for early R cell selection
        """

        # store experiment
        self.experiment = experiment

        # assign color palettes
        self.colorer = Palette()
        self.greys = Palette({'grey': 'grey'})

        # initialize figure
        self.fig = None

    def render(self, channel,
                cell_types=None,
                scatter=False,
                interval=False,
                marker_kw={},
                line_kw={},
                interval_kw={},
                ma_kw={},
                shading=None,
                figsize=(2, 1),
                ax_kw={},
                **kwargs):
        """
        Plot expression dynamics for single channel.

        Progenitor expression dynamics are shown by default. Additional cell types may be added via the cell_types argument.

        Args:
        channel (str) - fluorescence channel
        cell_types (list) - included cell types to be added
        scatter (bool) - if True, add markers for each measurement
        interval - if True, add confidence interval for moving average
        marker_kw (dict) - keyword arguments for marker formatting
        line_kw (dict) - keyword arguments for line formatting
        interval_kw (dict) - keyword arguments for interval formatting
        ma_kw (dict) - keyword arguments for interval construction
        shading (str) - color used to shade time window of young cells
        fig_size (tuple) - figure size
        ax_kw (dict) - keyword arguments for axis formatting
        kwargs: keyword arguments for plot function
        """

        # create figure
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig

        # plot expression dynamics
        self.plot(ax, channel,
                cell_types=cell_types,
                scatter=scatter,
                interval=interval,
                marker_kw=marker_kw,
                line_kw=line_kw,
                interval_kw=interval_kw,
                ma_kw=ma_kw,
                shading=shading,
                **kwargs)

        # format axis labels
        ax.set_ylabel('')
        ax.set_xlabel('Time (h)')

        # format axis
        self._format_ax(ax, **ax_kw)

    def render_all_channels(self,
                            cell_types=None,
                            scatter=False,
                            interval=False,
                            marker_kw={},
                            line_kw={},
                            interval_kw={},
                            ma_kw={},
                            shading=None,
                            figsize=(2.5, 4.5),
                            xlim=(-25, 55),
                            **kwargs):
        """
        Plot stacked Pnt/Yan/Ratio expression dynamics.

        Progenitor expression dynamics are shown by default. Additional cell types may be added via the cell_types argument.

        Args:
        cell_types (list) - included cell types to be added
        scatter (bool) - if True, add markers for each measurement
        interval - if True, add confidence interval for moving average
        marker_kw (dict) - keyword arguments for marker formatting
        line_kw (dict) - keyword arguments for line formatting
        interval_kw (dict) - keyword arguments for interval formatting
        ma_kw (dict) - keyword arguments for interval construction
        shading (str) - color used to shade time window of young cells
        fig_size (tuple) - figure size
        xlim (tuple) - bounds for x-axis
        kwargs: keyword arguments for plot function
        """

        # set formatting
        kw = dict(cell_types=cell_types,
                scatter=scatter,
                interval=interval,
                marker_kw=marker_kw,
                line_kw=line_kw,
                interval_kw=interval_kw,
                ma_kw=ma_kw,
                shading=shading,
                **kwargs)

        # create figure
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=figsize)
        (ax0, ax1, ax2) = axes
        self.fig = fig

        # plot pnt dynamics
        self.plot(ax0, 'ch1_normalized', **kw)

        # plot yan dynamics
        self.plot(ax1, 'ch2_normalized', **kw)

        # plot ratio dynamics
        self.plot(ax2, 'logratio', **kw)

        # format axis labels
        ax0.set_xlabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('Time (h)')
        ax0.set_ylabel('Pnt (a.u.)')
        ax1.set_ylabel('Yan (a.u.)')
        ax2.set_ylabel('Ratio')

        # format axes (good defaults for wildtype data)
        self._format_ax(ax0, xlim, ylim=(0.1,2.1), yticks=np.arange(.2,2.2,.3))
        self._format_ax(ax1, xlim, ylim=(0.1,2.3), yticks=np.arange(.2,2.4,.3))
        self._format_ax(ax2, xlim, ylim=(-2, 2.1), yticks=np.arange(-2,2.1,.5))
        ax2.spines['bottom'].set_position(('outward', 0))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # adjust spacing
        plt.subplots_adjust(hspace=0.15)

    def plot(self, ax, channel,
                cell_types=None,
                scatter=False,
                interval=False,
                marker_kw={},
                line_kw={},
                interval_kw={},
                ma_kw={},
                shading=None,
                **kwargs):

        """
        Plot expression dynamics for each cell type.

        Args:
        ax (plt.Axis instance) - if None, create axes
        channel (str) - fluorescence channel
        cell_types (list) - reference cell types to be added
        scatter (bool) - if True, add markers for each measurement
        interval - if True, add confidence interval for moving average
        marker_kw (dict) - keyword arguments for marker formatting
        line_kw (dict) - keyword arguments for line formatting
        interval_kw (dict) - keyword arguments for interval formatting
        ma_kw (dict) - keyword arguments for interval construction
        shading (str) - color used to shade time window of young cells
        kwargs: keyword arguments for Cells.plot_dynamics

        Returns:
        ax (plt.Axis instance)
        """

        # select progenitors
        pre = self.experiment.get_cells('pre')

        # format markers
        grey_shade = 4
        marker_kw['color'] = self.greys('grey', 'light', grey_shade)
        marker_kw['s'] = 0.5
        marker_kw['rasterized'] = True

        # format line
        line_kw['color'] = 'black'
        interval_kw['color'] = 'black'

        # set window size
        if channel[-4:] == 'flux':
            ma_kw['window_size'] = 500
        else:
            ma_kw['window_size'] = 250

        # plot progenitor expression
        pre.plot_dynamics(channel, ax,
                          scatter=scatter,
                          interval=interval,
                          marker_kw=marker_kw,
                          line_kw=line_kw,
                          interval_kw=interval_kw,
                          ma_kw=ma_kw)

        # add neuron expression
        if cell_types is None:
            cell_types = []
        for types in cell_types:

            # format markers and lines
            marker_kw['color'] = self.colorer(types[0])
            marker_kw['s'] = 2
            marker_kw['rasterized'] = True

            # if scatter is True, use line_colors for line/interval
            if scatter:
                line_color = line_colors[types[0]]
            else:
                line_color = self.colorer(types[0])
            line_kw['color'] = line_color
            interval_kw['color'] = line_color

            # set moving average resolution
            # set window size
            if channel[-4:] == 'flux':
                ma_kw['window_size'] = 150
            else:
                ma_kw['window_size'] = 75
            ma_kw['resolution'] = 5

            # select cells
            cells = self.experiment.get_cells(types)

            # plot dynamics
            cells.plot_dynamics(channel, ax,
                          scatter=scatter,
                          interval=interval,
                          marker_kw=marker_kw,
                          line_kw=line_kw,
                          interval_kw=interval_kw,
                          ma_kw=ma_kw,
                          **kwargs)

            # shade early R cell region
            if shading is not None:
                if channel == 'logratio':
                    self.shade_window(ax, types, color=shading, ymin=-2.75, ymax=2.75, alpha=0.25)
                else:
                    self.shade_window(ax, types, color=shading, alpha=0.25)

        return ax

    def shade_window(self, ax, reference,
                     color='orange',
                     alpha=0.5,
                     ymin=-2.5,
                     ymax=2.5):
        """
        Shade time window corresponding to first ten cells of reference type.

        Args:
        ax (plt.axis instance)
        reference (list) - reference cell type
        color (str) - shading color
        ymin, ymax (float) - shading boundaries
        """

        # select reference cells
        data = self.experiment.select_by_concurrency(reference, 10, 0, 1)
        data = data[data.label.isin(reference)]

        # shade time window
        tmin, tmax = data.t.min(), data.t.max()
        ax.fill_between([tmin, tmax],
                        [ymin, ymin],
                        [ymax, ymax],
                        color=color,
                        alpha=0.5,
                        zorder=0)

    @staticmethod
    def _format_ax(ax,
                xlim=(-25, 55),
                xticks=None,
                ylim=(0, 2.5),
                yticks=None,
                yspine_lim=None):
        """
        Format axis limits, spine limits, and tick positions.

        Args:
        ax (plt.axis instance)
        xlim (tuple) - limits for x-axis
        xticks (array like) - tick positions for x-axis
        ylim (tuple) - limits for y-axis
        yticks (array like) - tick positions for y-axis
        yspine_lim (tuple) - limits for y-axis spines
        """

        # format x axis
        ax.set_xlim(*xlim)
        if xticks is None:
            xticks = np.arange(xlim[0]+5, xlim[1]+5, 10)
        ax.set_xticks(xticks)
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_bounds(*xlim)

        # format y axis
        ax.set_ylim(*ylim)
        if yticks is None:
            yticks = np.arange(0., 2.75, .5)
        ax.set_yticks(yticks)
        ax.yaxis.set_ticks_position('left')
        if yspine_lim is None:
            yspine_lim = ylim
        ax.spines['left'].set_bounds(*yspine_lim)

        # format spines
        ax.spines['left'].set_position(('outward', 0))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


class DualExpression(Expression):
    """
    Object for comparing expression dynamics between two experimental conditions.

    Attributes:
    experiments (dict) - {condition: data.experiments.Experiment} pairs for the control and perturbation conditions.
    colors (figures.palettes.Palette) - color palettes for control and perturbation conditions.

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, control, perturbation,
                 control_color='k',
                 perturbation_color='r'):
        """
        Instantiate object for comparing expression dynamics between a control and perturbation condition.

        Args:
        control (data.experiments.Experiment) - control data
        perturbation (data.experiments.Experiment) - perturbation data
        control_color (str) - control color
        perturbation_color (str) - perturbation color
        """

        # define experiments
        self.experiments = {'control': control,
                            'perturbation': perturbation}

        # define color palettes
        self.colors = Palette({'control': control_color,
                               'perturbation': perturbation_color})

        # initialize figure
        self.fig = None

    def render(self,
                channel,
                cell_types=['pre'],
                figsize=(2, 1),
                ax_kw={},
                **kwargs):
        """
        Render expression timeseries comparison figure.

        Args:
        channel (str) - fluorescence channel
        cell_types (list) - cell types to be included
        figsize (tuple) - figure size
        ax_kw (dict) - keywoard arguments for format_axis
        kwargs: keyword arguments for plotting
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)
        ax = self.fig.axes[0]

        # plot expression for each experiment
        for exp in self.experiments.keys():
            self.plot(ax, exp, channel, cell_types=cell_types, **kwargs)

        # format axis
        self._format_ax(ax, **ax_kw)

    def plot(self, ax, exp, channel,
                cell_types=['pre'],
                scatter=False,
                interval=False,
                marker_kw={},
                line_kw={},
                interval_kw={},
                ma_kw={},
                **kwargs):

        """
        Plot expression dynamics for single experiment, channel, and cell type.

        Args:
        ax (plt.Axis instance) - if None, create axes
        exp (str) - experiment key
        channel (str) - fluorescence channel
        cell_types (list) - cell types to be included
        scatter (bool) - if True, add markers for each measurement
        interval - if True, add confidence interval for moving average
        marker_kw (dict) - keyword arguments for marker formatting
        line_kw (dict) - keyword arguments for line formatting
        interval_kw (dict) - keyword arguments for interval formatting
        ma_kw (dict) - keyword arguments for interval construction
        kwargs: keyword arguments for Cells.plot_dynamics

        Returns:
        ax (plt.Axis instance)
        """

        # select cells of specified type
        cells = self.experiments[exp].get_cells(cell_types)

        # define linestyle
        if exp == 'perturbation':
            line_kw['linestyle'] = 'dashed'
        elif exp == 'wildtype':
            line_kw['linestyle'] = 'dashed'
        else:
            line_kw['linestyle'] = 'solid'

        # define colors
        marker_kw['color'] = self.colors[exp]
        line_kw['color'] = 'k'
        interval_kw['color'] = self.colors[exp]
        ma_kw['window_size'] = 250

        # plot expression dynamics
        cells.plot_dynamics(channel, ax,
                          scatter=scatter,
                          interval=interval,
                          marker_kw=marker_kw,
                          line_kw=line_kw,
                          interval_kw=interval_kw,
                          ma_kw=ma_kw)

        ax.set_xlabel('Time (h)')

        return ax


class MultiExpression(DualExpression):
    """
    Object for comparing expression dynamics between multiple experimental conditions.

    Inherited attributes:
    experiments (dict) - {condition: data.experiments.Experiment} pairs for all conditions
    colors (figures.palettes.Palette) - color palette for all conditions
    fig (matplotlib.figure.Figure)
    """
    def __init__(self, *experiments):
        """
        Instantiate object for comparing expression dynamics between multiple conditions.

        Args:
        experiments (*tuple) - list of (measurement data, color) pairs
        """
        experiments, colors = list(zip(experiments))
        self.experiments = {k: v for k, v in enumerate(experiments)}
        self.colors = {k: v for k, v in enumerate(colors)}
        self.fig = None
