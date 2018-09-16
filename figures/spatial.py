__author__ = 'Sebastian Bernasek'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .base import Base
from .settings import *
from .palettes import Palette

from flyeye.analysis.spectrogram import Spectrogram
from flyeye.analysis.correlation import SpatialCorrelation


class Correlation(Base):
    """
    Object for visualizing spatial correlation function.

    Attributes:
    num_discs (int) - number of separate eye discs in experiment

    Inherited attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    # set class path and name
    path = 'graphics/spatial'

    def __init__(self, experiment):
        """
        Instantiate spatial correlation figure from Experiment instance. Data are automatically limited to progenitors.

        Args:
        experiment (data.experiments.Experiment)
        """
        super().__init__(experiment.get_cells('pre').df)
        self.num_discs = len(experiment.discs)

    def render(self,
               channel='ratio',
               tmin=0,
               tmax=1.75):
        """
        Render all spatial correlation figures.

        Args:
        channel (str) - expressional channel
        tmin, tmax (float) - bounds for cell selection time window
        """

        # create figure
        figsize = (1.5, 0.5*self.num_discs)
        self.fig = self.create_figure(nrows=self.num_discs, figsize=figsize)

        # plot autocorrelations
        self.plot(channel=channel, tmin=tmin, tmax=tmax)

        # adjust spacing
        plt.subplots_adjust(hspace=0)

    def plot(self,
             channel='ratio',
             tmin=0,
             tmax=1.75):
        """
        Plot spatial correlation function.

        Args:
        channel (str) - expression channel
        tmin, tmax (float) - bounds for cell selection time window
        """

        # iterate across discs
        for disc_id, ax in enumerate(self.fig.axes):

            # select cells
            cells = self.df[self.df.disc_id==disc_id]
            cells = cells[cells.t.between(tmin, tmax)]

            # instantiate correlation object
            corr = SpatialCorrelation(cells, channel, y_only=True, raw=False)

            # specify smoothing parameters
            ma_kw = dict(window_size=50,
                         resolution=5,
                         ma_type='savgol')

            # plot correlation functions
            corr.visualize(ax=ax,
                          null_model=False,
                          scatter=False,
                          confidence=True,
                          zero=True,
                          ma_kw=ma_kw,
                          nbootstraps=1000,
                          color='k',
                          max_distance=500)

            # # add bin statistics
            # mean, std = cells[channel].mean(), cells[channel].std()
            # if channel == 'ratio':
            #     cov = std
            # else:
            #     cov = std/mean

            # annotate sample size
            s = 'N = {:d} cells'.format(len(cells))
            ax.text(400, 0.6, s=s, ha='right', va='center', fontsize=7)

            # format axis
            last = (disc_id == self.num_discs-1)
            self._format_ax(ax, last=last)

    @staticmethod
    def _format_ax(ax, last=False):
        """
        Format axis.

        Args:
        ax (matplotlib.axes._subplots.AxesSublot)
        last (bool) - if True, this is the bottom axis in the figure
        """

        # temporarily change labelsize
        ax.tick_params(labelsize=7)

        # format y axis
        ax.set_ylabel('')
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-.8, .8)
        ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
        ax.set_yticklabels([-.6, '', 0, '', .6])
        ax.spines['left'].set_bounds(-.6, .6)

        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('outward', 5))

        # format x axis
        ax.set_xlim(0, 400)
        ax.spines['top'].set_visible(False)
        if last:
            ax.set_xticks(np.arange(0, 450, 100))
            ax.set_xlabel('Dorsoventral distance (px)', fontsize=8)
            ax.spines['bottom'].set_position(('outward', 0))
            ax.xaxis.set_ticks_position('bottom')
        else:
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)


class Periodogram(Correlation):
    """
    Object for visualizing spectral power density of periodic spatial patterns within each eye disc of an experiment.

    Inherited attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    num_discs (int) - number of separate eye discs in experiment
    """

    def render(self,
               channel='ratio',
               tmin=0,
               tmax=1.75,
               period_range=(50, 200)):
        """
        Render all spatial correlation figures.

        Args:
        channel (str) - expressional channel
        tmin, tmax (float) - bounds for cell selection time window
        period_range (tuple) - bounds for spectral components
        """

        # create figure
        figsize = (1.25, 0.5*self.num_discs)
        self.fig = self.create_figure(nrows=self.num_discs, figsize=figsize)

        # plot spectrograms
        self.plot(channel, tmin=tmin, tmax=tmax, period_range=period_range)

    def plot(self,
             channel='ratio',
             tmin=0,
             tmax=1.75,
             period_range=(50, 200)):
        """
        Plot periodogram.

        Args:
        channel (str) - expression channel
        tmin, tmax (float) - bounds for cell selection time window
        period_range (tuple) - bounds for spectral components
        """

        # define spectral components to be tested
        lb, ub = np.log10(period_range[0]), np.log10(period_range[1])
        periods = 10 ** np.linspace(lb, ub, 1000)

        # iterate across discs
        for disc_id, ax in enumerate(self.fig.axes):

            # select cells
            cells = self.df[self.df.disc_id==disc_id]
            cells = cells[cells.t.between(tmin, tmax)]

            # plot spectrogram
            spectrogram = Spectrogram(cells.centroid_y.values,
                                      cells[channel].values,
                                      periods=periods)
            spectrogram.simple_visualization(ax=ax,
                                             nbootstraps=1000,
                                             confidence=[95])

            # format axis
            last = (disc_id == self.num_discs-1)
            self._format_ax(ax, period_range, last=last)

    @staticmethod
    def _format_ax(ax, period_range, last=False):
        """
        Format axis.

        Args:
        ax (matplotlib.axes._subplots.AxesSublot)
        period_range (tuple) - bounds for spectral components
        last (bool) - if True, this is the bottom axis in the figure
        """

        # temporarily change labelsize
        ax.tick_params(labelsize=6)

        # format x axis
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 3))

        # format y axis
        ax.set_ylabel('')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('outward', 3))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels([0, '', '', '', 1])
        ax.set_ylim(-0.1, 1.1)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['right'].set_visible(False)

        # remove x-axis for intermediate rows
        if last:
            lb, ub = period_range[0], period_range[1]+10
            ax.set_xticks(np.arange(lb, ub, 25))
            ax.set_xlabel('Oscillation period (px)', fontsize=7)
            ax.xaxis.set_ticks_position('bottom')
        else:
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)


class R8Spacing(Base):
    """
    Object for visualizing dorsoventral spacing between adjacent R8 neurons.

    Distances are obtained by sliding a narrow (~1 hr developmental time) window along the anterior-posterior axis and computing the distance between adjacent R8 neurons within the window.

    Inherited attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, experiment, dt=1.75):
        """
        Instantiate R8 spacing visualization from Experiment instance.

        Data are automatically limited to R8 neurons. Distances are evaluated and aggregated upon instantiation.

        Args:
        experiment (data.experiments.Experiment)
        dt (float) - time interval between adjacent columns, in hours
        """

        # evaluate separation distances for all discs
        distances = {}
        for i, disc in experiment.discs.items():
            distances[i] = self._evaluate_distances(disc, dt=dt)

        # pad with NaNs to generate equal length arrays
        size = max([len(dist) for dist in distances.values()])
        for i, dist in distances.items():
            nans = np.empty(size-dist.size)
            nans[:] = np.nan
            distances[i] = np.hstack((dist, nans))

        # compile dataframe of distances
        df = pd.DataFrame.from_dict(distances)

        # call base class instantiation
        super().__init__(df)

    @staticmethod
    def _evaluate_distances(disc, dt=1.75):
        """
        Construct distirbution of separation distances between adjacent rows of R8 neurons.

        Distances are obtained by sliding a narrow (~1 hr developmental time) window along the anterior-posterior axis and computing the distance between adjacent R8 neurons within the window.

        Args:
        disc (data.discs.Disc) - individual eye disc
        dt (float) - time interval between adjacent columns

        Returns
        distances (np.ndarray) - distribution of dorso-ventral distances
        """
        r8s = disc.select_cell_type('r8')
        distances = np.array([])
        for tmin in np.arange(r8s.df.t.min(), r8s.df.t.max(), dt/2):
            cells = r8s.select_by_position(tmin=tmin, tmax=tmin+dt)
            ycoords = cells.df[['centroid_y']].values.flatten()
            distances = np.hstack((distances, np.diff(sorted(ycoords))))
        return distances[1:]

    def render(self):
        """ Render boxplots. """

        # create figure
        self.fig = self.create_figure(figsize=(1, .75))

        # plot boxplots
        self.plot()

    def plot(self):
        """
        Plot distribution of dorsoventral R8 separation distances.

        Distances are obtained by sliding a narrow (~1 hr developmental time) window along the anterior-posterior axis and computing the distance between adjacent R8 neurons within the window.
        """

        # get axis
        ax = self.fig.axes[0]

        # plot distributions
        sns.boxplot(data=self.df,
                    orient='h',
                    fliersize=0,
                    notch=True,
                    color='w',
                    linewidth=1,
                    width=0.5,
                    ax=ax)

        # format axis
        self._format_ax(ax)

    @staticmethod
    def _format_ax(ax):
        """
        Format axis.

        Args:
        ax (matplotlib.axes._subplots.AxesSublot)
        """

        # format y axis
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

        # format x axis
        ax.set_xlabel('R8 separation (px)')
        ax.set_xlim(0, 150)
        ax.set_xticks(np.arange(0, 200, 50))
        ax.spines['bottom'].set_position(('outward', 2))
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['top'].set_visible(False)

