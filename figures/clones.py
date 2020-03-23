__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt

from flyqma.data import Experiment
from flyqma.analysis.statistics import PairwiseCelltypeComparison

from .base import Base
from .settings import *


class CloneComparison(Base):

    def __init__(self, data):
        """
        Args:

            data (pd.DataFrame)

        """
        super().__init__(data)
        self.ch2 = PairwiseCelltypeComparison(data, 0, 1, 'ch2_normalized', 'genotype')
        self.ch1c = PairwiseCelltypeComparison(data, 0, 1, 'ch1c_normalized', 'genotype')

    @property
    def ch2_pvalue(self):
        """ Comparison p-value for Channel 2. """
        return self.ch2.compare()[0]

    @property
    def ch1c_pvalue(self):
        """ Comparison p-value for Channel 2. """
        return self.ch1c.compare()[0]
    
    @staticmethod
    def from_flyqma(path, exclude_boundary=True):
        """
        Instantiate from Fly-QMA experiment directory.

        Args:

            path (str) - path to Fly-QMA experiment directory

        """

        # load Fly-QMA data
        experiment = Experiment(path)
        
        # filter measurements using inclusion criteria
        options = dict(selected_only=True, exclude_boundary=exclude_boundary)
        data = experiment.aggregate_measurements(**options)

        return CloneComparison(data)

    def render(self,
               height=1.5,
               aspect=1.25,
               width=.75,
               lw=0.75,
               cell_types=None):
        """
        Render boxplot.

        Args:
        height (float) - figure height passed to seaborn.FacetGrid
        aspect (float) - figure aspect ratio passed to seaborn.FacetGrid
        width (float) - artist patch width
        lw (float) - artist line width
        cell_types (list of lists) - early R cell types to be included
        """

        # create figure
        fig = self.create_figure(figsize=(3, 1.5), nrows=1, ncols=2)
        ax0, ax1 = fig.axes

        # plot clonal marker comparison
        self.ch2.plot(ax=ax0, ylabel='UbiRFP (a.u.)', cut=1)
        
        # plot PntGFP comparison
        self.ch1c.plot(ax=ax1, ylabel='PntGFP (a.u.)', cut=1)

        # format axes
        self.format_axes(ax0, ax1)

        # store output
        self.fig = fig

    @staticmethod
    def format_axes(ax0, ax1):
        """
        Format <ax0> and <ax1>.
        """

        ax0.set_ylim(-0.1, 2.5)
        ax0.set_yticks(np.arange(0, 2.5+.1, .5))
      
        ax1.set_ylim(-0.1, 1.2)
        ax1.set_yticks(np.arange(0, 1.2+.1, .2))
        
        # enforce tight layout
        plt.tight_layout()
