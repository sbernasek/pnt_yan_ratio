__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from operator import add

from .base import Base
from .settings import *
from .palettes import Palette


class JointDistribution(Base):
    """
    Object for constructing joint distribution plots.

    Attributes:
    pre_color (tuple) - grey color for progenitors

    Inherited attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    # set class path and name
    path = 'graphics/jointdistribution'

    def __init__(self, df):
        """
        Initialize object for constructing joint distribution plots.

        Args:
        df (pd.DataFrame)
        """
        super().__init__(df)

        # get color for progenitors
        greys = Palette({'grey': 'grey'})
        self.pre_color = greys('grey', 'light', shade=5)

    @classmethod
    def from_experiment(cls, experiment, cell_types='pre'):
        """
        Instantiate from Experiment instance with optional constraint on included cell types.

        Args:
        experiment (data.experiments.Experiment)
        cell_types (list or str) - included cell types

        Returns:
        fig_obj (figures.base.Base derivative)
        """

        # select measurement data
        cells = reduce(add, experiment.discs.values())
        df = cells.select_cell_type(cell_types).df

        return cls(df)

    def render(self,
               cell_types=['pre'],
               height=2.65):
        """
        Create figure and plot joint distribution.

        Args:
        cell_types (list) - included cell types
        height (float) - figure height argument
        """

        # plot progenitor joint distribution
        self.fig = self.plot(cell_types, height=height, color=self.pre_color)

    def plot(self,
             cell_types=['pre'],
             color='grey',
             height=2.65):
        """
        Plot joint distribution for specific cell type using sns.jointplot

        Args:
        cell_types (list) - included cell types
        color (str or RGB tuple) - color for markers
        height (float) - figure height argument

        Returns:
        fig (matplotlib.figure.Figure)
        """

        # select cells
        cells = self.df[self.df.label.isin(cell_types)]

        # define formatting arguments
        joint_kws = {'edgecolor': 'white',
                     'linewidth': 0.25,
                     'facecolor': color,
                     's': 10}
        hist_kws = dict(edgecolor='black',
                        facecolor=color,
                        linewidth=0,
                        alpha=0.75)
        marginal_kws = {'bins': np.arange(0, 2., .05),
                        'norm_hist': True,
                        'hist_kws': hist_kws}

        # create jointplot
        fig = sns.jointplot('blue', 'green',
                          data=cells,
                          height=height,
                          ratio=3,
                          #color=color,
                          stat_func=None,
                          joint_kws=joint_kws,
                          marginal_kws=marginal_kws,
                          xlim=(0, 2),
                          ylim=(0, 2))

        # add diagonal with slope equivalent to median ratio
        median_ratio = 2**(cells.ratio.median())
        fig.ax_joint.plot([0, 1.5], [0, 1.5*median_ratio], '-',
                        linewidth=1, color='k', zorder=0)
        #angle = np.arctan(median_ratio) * 180/np.pi
        #g.ax_joint.text(1.75, 1.75*median_ratio, 'median ratio', rotation=angle, va='center', ha='center')

        # format axes
        fig.ax_joint.set_ylabel('Pnt (a.u.)')
        fig.ax_joint.set_xlabel('Yan (a.u.)')

        return fig


class DualJointDistribution(JointDistribution):
    """
    Object for constructing joint distribution plots overlayed with second cell type.

    Attributes:
    reference (list) - reference cell types
    reference_color (RGB tuple) - color for reference cell type

    Inherited attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    pre_color (RGB tuple) - color for progenitors
    """
    def __init__(self, df, reference):
        """
        Instantiate joint distribution figure with two cell types.

        Args:
        df (pd.DataFrame) - data containing Control/Perturbation labels
        reference (list) - reference cell types
        """
        super().__init__(df)
        self.reference = reference

        # get color for reference cell type
        colorer = Palette()
        self.reference_color = colorer(reference[0].lower())

    @classmethod
    def from_experiment(cls, experiment, reference, **kwargs):
        """
        Instantiate from Experiment instance with cells limited to those concurrent with a reference population.

        Args:
        experiment (data.experiments.Experiment)
        reference (list) - reference cell types
        kwargs: keyword arguments for concurrent cell selection

        Returns:
        fig_obj (figures.base.Base derivative)
        """

        # select cells concurrent with reference
        df = experiment.select_by_concurrency(reference, **kwargs)

        return cls(df, reference)

    def add_reference_to_fig(self):
        """ Add reference cell type to joint and marginal axes. """

        # select reference cells
        df = self.df[self.df.Population=='Differentiated']

        # scatter neurons on joint axes
        self.fig.x = df['blue']
        self.fig.y = df['green']
        self.fig.plot_joint(plt.scatter,
                            c=self.reference_color,
                            linewidth=0.25,
                            s=10,
                            edgecolor='w')

        # get marginal axes limits
        axx, axy = self.fig.ax_marg_x, self.fig.ax_marg_y
        xlim = axx.get_xlim()
        ylim = axy.get_ylim()

        # add reference cell distributions to marginal x axis
        axx.hist(df['blue'],
                 bins=np.arange(0, 2, 0.05),
                 color=self.reference_color,
                 alpha=0.75,
                 density=True,
                 linewidth=0)
        axx.set_xlim(*xlim)

        # add reference cell distributions to marginal y axis
        axy.hist(df['green'],
                 orientation='horizontal',
                 bins=np.arange(0, 2, 0.05),
                 color=self.reference_color,
                 alpha=0.75,
                 density=True,
                 linewidth=0)
        axy.set_ylim(*ylim)

    def render(self,
               height=2.65,
               **kwargs):
        """
        Create figure and plot progenitor joint distribution overlayed with reference cell type.

        Args:
        height (float) - figure height argument for sns.jointplot
        """

        # plot progenitor joint distribution
        self.fig = self.plot(height=height, color=self.pre_color)

        # add reference cells
        self.add_reference_to_fig()

        # format axes
        self.format_axes()

    def format_axes(self):
        """ Format all axes. """

        # get axes
        ax_joint = self.fig.ax_joint
        axx, axy = self.fig.ax_marg_x, self.fig.ax_marg_y

        # add data labels
        label = 'Young ' + self.df['ReferenceType'].unique()[0]
        ax_joint.text(0.1, 2, label, ha='left', va='top', color=self.reference_color, fontsize=7)
        ax_joint.text(0.1, 2, '\nConcurrent multipotents', ha='left', va='top', color=self.pre_color, fontsize=7)

        # format ticks
        axx.tick_params(length=0)
        axy.tick_params(length=0)

        # set axes shift
        axes_shift = 0
        ax_joint.spines['bottom'].set_position(('outward', axes_shift))
        ax_joint.spines['left'].set_position(('outward', axes_shift))
