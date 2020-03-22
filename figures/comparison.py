__author__ = 'Sebastian Bernasek'

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp

from .base import Base
from .settings import *
from .palettes import Palette

from flyeye.processing.alignment import MultiExperimentAlignment


class HorizontalBoxplot(Base):
    """
    Object for constructing horizontal boxplot comparisons.

    Attributes:
    orientation (str) - indicates whether figure is horizontal/vertical

    Inherited attributes:
    data (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    # set class path and name
    path = 'graphics/comparison'

    @classmethod
    def from_experiment(cls, experiment, **kwargs):
        """
        Instantiate from Experiment instance. Cells concurrent with early R-cells are selected by default.

        Args:
        experiment (data.experiments.Experiment)
        kwargs: keyword arguments for cell selection

        Returns:
        figure_obj (figures.base.Base derivative)
        """

        # determine which channel corresponds to yan
        yan_channel = cls.get_yan_channel(experiment)

        # get early neuron data and delineate metrics
        data = experiment.get_early_neuron_data(**kwargs)
        data = cls._delineate_metrics(data, yan_channel=yan_channel)

        # instantiate figure object
        figure_obj = cls(data=data)

        return figure_obj

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

        # get neuron classes and corresponding labels
        if cell_types is None:
            cell_types = [['r8'],['r2','r5'],['r3', 'r4'],['r1','r6'],['r7']]
        r_cell_types = ['/'.join(types).upper() for types in cell_types]

        # get data
        data = self.data[self.data['ReferenceType'].isin(r_cell_types)]

        # create facetgrid and plot data
        fig = self.plot(data, height=height, aspect=aspect, width=width, lw=lw)
        self.set_patch_colors(fig, cell_types=cell_types)

        # format axes
        self.format_axes(fig)
        self.format_ticklabels(fig, cell_types, self.orientation)

        # store output
        self.fig = fig

    def plot(self,
             data,
             height=1.5,
             aspect=1.25,
             width=0.75,
             lw=0.75):
        """
        Create facetgrid and plot data using seaborn.boxplot method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

        Args:
        data (pd.DataFrame) - data
        height (float) - figure height passed to seaborn.FacetGrid
        aspect (float) - figure aspect ratio passed to seaborn.FacetGrid
        width (float) - artist patch width
        lw (float) - artist line width

        Returns:
        fig (matplotlib.figure.Figure)
        """
        order = ['R8', 'R2/R5', 'R3/R4', 'R1/R6', 'R7']
        hue_order = ['Multipotent', 'Differentiated']
        grid = sns.FacetGrid(data, col="Metric", height=height, aspect=aspect, sharey=False)
        fig = grid.map(sns.boxplot,
                        "ReferenceType", "Score", "Population",
                        order=order,
                        hue_order=hue_order,
                        orient='v',
                        fliersize=0,
                        linewidth=lw,
                        notch=True,
                        width=width)
        fig.despine(left=False, right=False, top=False)
        self.orientation = 'v'
        return fig

    @staticmethod
    def _delineate_metrics(data0, yan_channel='ch2_normalized'):
        """ Delineate early neuron data with Metric/Score attributes. """

        data0['Metric'] = 'Pnt'
        data0['Score'] = data0.ch1_normalized

        data1 = deepcopy(data0)
        data1['Metric'] = 'Yan'
        data1['Score'] = data1[yan_channel]

        data2 = deepcopy(data0)
        data2['Metric'] = 'Ratio'
        data2['Score'] = data2.logratio

        delineated_data = pd.concat((data0, data1, data2))

        return delineated_data

    @staticmethod
    def set_patch_colors(fig, cell_types, lw=1):
        """
        Set colors for patch objects (boxes).

        Args:
        fig (matplotlib.figure.Figure)
        cell_types (list of lists) - cell types used to define colors
        colorer (figures.palettes.Palette) - colormap for cell types
        lw (float) - patch line width
        """

        colorer = Palette()

        for ax in fig.axes.ravel():

            for i, artist in enumerate(ax.artists):

                # get color from colorer
                neuron = cell_types[i // 2][0]
                if i % 2 == 0:
                    color = colorer(neuron)
                    greys = Palette({'grey': 'grey'})
                    artist.set_facecolor(greys('grey', 'light', shade=1))
                    artist.set_edgecolor(color)
                    artist.set_linewidth(lw)
                else:
                    color = colorer(neuron)
                    artist.set_facecolor(color)

    @staticmethod
    def format_ticklabels(fig, cell_types, orientation='v'):
        """ Set neuron type labels and label colors. """

        colorer = Palette()

        # get axes iterable
        if type(fig.axes) == list:
            axes = fig.axes
        else:
            axes = fig.axes.ravel()

        # format each axis
        for ax in axes:

            # set axis label
            labels = ['{:s}'.format('/'.join(types).upper()) for types in cell_types]

            if orientation == 'v':
                ax.set_xticklabels(labels)
                ticks = ax.xaxis.get_ticklabels()
            else:
                ax.set_yticklabels(labels)
                ticks = ax.yaxis.get_ticklabels()

            for i, label in enumerate(ticks):

                # get color from colorer
                types = cell_types[i]
                cell_type = types[0]
                color = colorer(cell_type)

                # set text and color
                label.set_color(color)

    @staticmethod
    def format_axes(fig):
        """ Format figure axes. """

        # get axes objects
        axes = fig.axes[0]
        ax0, ax1, ax2 = axes

        # set axis limits
        for ax in axes[0:2]:
            ax.set_ylim(0, 2)
            ax.set_yticks(np.arange(0, 2.5, 0.5))
        axes[-1].set_ylim(-2, 2)
        axes[-1].set_yticks(np.arange(-2, 2.5, 1))

        # set axis labels
        ax0.set_ylabel('Pnt Expression (a.u.)')
        ax1.set_ylabel('Yan Expression, (a.u.)')
        ax2.set_ylabel('log2 Ratio')
        fig.set_xlabels('')
        fig.set_titles('')

        # set spacing betwee subplots
        plt.subplots_adjust(wspace=0.5)

        # format ticks
        for ax in axes:
            ax.tick_params(axis='both', direction='in', length=3)

    def get_statistics(self,
                       test='ks_2samp',
                       rounding=1):
        """
        Perform statistical test to determine whether or not expression levels differ between cell types.

        Args:
        test (str) - statistical test used
        rounding (int) - rounding applied to log10(pvalues)

        Returns:
        pvalues (dict) - nested {metric: {cell_type: {test: pvalue}}} pairs where metric is the quantity being compared, cell type indicates the concurrent population of early R cells, test denotes the type of statistical test, and pvalue is the resultant pvalue.
        """

        channel_names = dict(ch1_normalized='PntGFP', ch2_normalized='Yan', logratio='P:Y ratio')

        # determine unique populations
        cell_types = self.data['ReferenceType'].unique()

        # get test
        if test == 'ks_2samp':
            test = lambda x, y: ks_2samp(x, y)
        elif test == 'welch':
            test = lambda x, y: ttest_ind(x, y, equal_var=False)
        elif test == 't':
            test = lambda x, y: ttest_ind(x, y, equal_var=True)

        # iterate across channels to be compared
        stats = {}
        for channel in ('ch1_normalized', 'ch2_normalized', 'logratio'):
            stats[channel_names[channel]] = {}

            # iterate across neuron types
            for types in cell_types:

                # get early neurons and concurrent progenitors
                current = self.data[self.data['ReferenceType']==types]
                multipotent = current[current.Population=='Multipotent'][channel].values
                differentiated = current[current.Population=='Differentiated'][channel].values

                # apply test and store log10 pval
                ks, pval = test(multipotent, differentiated)
                stats[channel_names[channel]][types] = np.log10(pval)

        # compile and sort dataframe
        data = pd.DataFrame.from_dict(stats)
        columns = ['PntGFP', 'Yan', 'P:Y ratio']
        return data.loc[cell_types][columns].round(rounding)


class VerticalBoxplot(HorizontalBoxplot):
    """
    Object for constructing horizontal boxplot comparisons.

    Inherited attributes:
    data (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    orientation (str) - indicates whether figure is horizontal/vertical
    """

    def plot(self,
             data,
             height=1.5,
             aspect=1.25,
             width=.75,
             lw=0.75):
        """
        Create facetgrid and plot data using seaborn.boxplot method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

        Args:
        data (pd.DataFrame) - data containing Metric, Score, ReferenceType, and Population keys.
        height (float) - figure height passed to seaborn.FacetGrid
        aspect (float) - figure aspect ratio passed to seaborn.FacetGrid
        width (float) - artist patch width
        lw (float) - artist line width

        Returns:
        fig (matplotlib.figure.Figure)
        """
        order = ['R8', 'R2/R5', 'R3/R4', 'R1/R6', 'R7']
        hue_order = ['Multipotent', 'Differentiated']

        grid = sns.FacetGrid(data, row="Metric", height=height, aspect=aspect, sharex=False)
        fig = grid.map(sns.boxplot,
                       "Score", "ReferenceType", "Population",
                        order=order,
                        hue_order=hue_order,
                        orient='h',
                        fliersize=0,
                        linewidth=lw,
                        notch=True,
                        width=width)
        fig.despine(left=False, right=False, top=False, bottom=False)
        self.orientation = 'h'
        return fig

    @staticmethod
    def format_axes(fig):

        # get axes objects
        axes = fig.axes.ravel()
        ax0, ax1, ax2 = axes

        # set axis limits
        for ax in axes[0:2]:
            ax.set_xlim(-0, 2)
            ax.set_xticks(np.arange(0., 2.5, 0.5))
        axes[-1].set_xlim(-2, 2)
        axes[-1].set_xticks(np.arange(-2, 3, 1))

        # set axis labels
        ax0.set_xlabel('Pnt Expression (a.u.)')
        ax1.set_xlabel('Yan Expression (a.u.)')
        ax2.set_xlabel('log2 Ratio')
        fig.set_ylabels('')
        fig.set_titles('')

        # set spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        # format axes
        for ax in axes:
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('outward', 5))
            ax.spines['bottom'].set_position(('outward', 5))

            yticks = ax.get_yticks()
            ax.spines['left'].set_bounds(yticks[0], yticks[-1])




class ViolinPlot(VerticalBoxplot):
    """
    Object for constructing vertical violinplot comparisons.

    Creates facetgrid and plot data using seaborn.boxplot method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

    Inherited attributes:
    data (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    orientation (str) - indicates whether figure is horizontal/vertical
    """

    def plot(self,
             data,
             height=4,
             aspect=1,
             lw=1,
             **kwargs):
        """
        Create facetgrid and plot data using seaborn.violin method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

        Args:
        data (pd.DataFrame) - data
        height (float) - figure height passed to seaborn.FacetGrid
        aspect (float) - figure aspect ratio passed to seaborn.FacetGrid
        lw (float) - artist line width

        Returns:
        fig (matplotlib.figure.Figure)
        """

        grid = sns.FacetGrid(data, row="Metric", height=height, aspect=aspect, sharex=False)
        fig = (grid.map(sns.violinplot, "Score", "ReferenceType", "Population", orient='h', split=True, scale="area", inner='quartile', linewidth=lw, bw=.25, scale_hue=False))
        fig.despine(left=False, right=False, top=False, bottom=False)
        self.orientation = 'h'
        return fig


class DosingComparison(VerticalBoxplot):
    """
    Object for constructing boxplot comparison of expression levels between two experiments with different PntGFP dosages.

    Creates facetgrid and plot data using seaborn.boxplot method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

    Attributes:
    pvalues (dict) - comparison statistics

    Inherited attributes:
    data (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    orientation (str) - indicates whether figure is horizontal/vertical
    """

    def __init__(self, data, orientation='v'):
        """
        Instantiate gene dosage comparison figure.

        Args:
        data (pd.DataFrame) - data for figure
        orientation (str) - indicates whether figure is horizontal/vertical
        """
        super().__init__(data)
        self.orientation = orientation

    @classmethod
    def from_experiment(cls, pnt1x, pnt2x, orientation='v', **kwargs):
        """
        Instantiate from data.experiments.Experiment instances. Cells concurrent with early R-cells are selected by default.

        Args:
        pnt1x (data.experiments.Experiment) - first PntGFP dosage condition
        pnt2x (data.experiments.Experiment) - second PntGFP dosage condition
        orientation (str) - indicates whether figure is horizontal/vertical
        kwargs: keyword arguments for early R cell selection

        Returns:
        figure_obj (figures.base.Base)
        """

        # align experiments
        alignment = MultiExperimentAlignment(pnt1x, pnt2x)
        pnt1x_aligned, pnt2x_aligned = alignment.get_aligned_experiments()

        # get early neuron data
        data_1x = pnt1x_aligned.get_early_neuron_data(**kwargs)
        data_2x = pnt2x_aligned.get_early_neuron_data(**kwargs)

        # assign gene dosage labels
        data_1x['Dosing'] = '1X PntGFP'
        data_2x['Dosing'] = '2X PntGFP'
        dosing_data = pd.concat((data_1x, data_2x))

        # only use progenitors
        data = dosing_data[dosing_data.Population=='Multipotent']

        # instantiate DosingComparisonFigure object
        return cls(data=data, orientation=orientation)

    def render(self,
               channel='logratio',
               figsize=(2, 1.5),
               width=.75,
               lw=0.75,
               cell_types=None):
        """
        Render boxplot comparing gene dosage conditions.

        Args:
        channel (str) - expressional channel to be compared
        figsize (tuple) - figure size
        width (float) - artist patch width
        lw (float) - artist line width
        cell_types (list of lists) - early R cell types to be included
        """

        # select data concurrent with specified cell types
        if cell_types is None:
            cell_types = [['r8'],['r2','r5'],['r3', 'r4'],['r1','r6'],['r7']]
        r_cell_types = ['/'.join(types).upper() for types in cell_types]
        data = self.data[self.data['ReferenceType'].isin(r_cell_types)]

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # plot data
        self.plot(data, channel=channel, width=width, lw=lw)
        self.set_patch_colors(self.fig.axes[0], cell_types=cell_types)

        # format axes
        self._format_ax(self.fig.axes[0], channel, self.orientation)
        self.format_ticklabels(self.fig, cell_types, self.orientation)

    def plot(self, data,
             channel='logratio',
             width=0.75,
             lw=1):
        """
        Create facetgrid and plot data using seaborn.boxplot method. Data must contain Metric, Score, ReferenceType, and Population keys as these are used to categorically partition data within the figure.

        Args:
        data (pd.DataFrame) - data containing ReferenceType and Dosing keys.
        channel (str) - expressional channel to be compared
        width (float) - artist patch width
        lw (float) - artist line width
        """

        # define colors for 1x and 2x dosing
        greys = Palette({'grey': 'grey'})
        pal = [greys('grey', 'light', shade=2), greys('grey', 'dark', shade=2)]

        # assemble formatting arguments
        kw = dict(palette=pal,
                  notch=True,
                  fliersize=0,
                  width=width,
                  linewidth=lw)

        # plot distributions
        if self.orientation == 'v':
            sns.boxplot(data=data,
                        x='ReferenceType',
                        y=channel,
                        hue='Dosing',
                        orient='v',
                        ax=self.fig.axes[0],
                        **kw)
        else:
            sns.boxplot(data=data,
                        x=channel,
                        y='ReferenceType',
                        hue='Dosing',
                        orient='h',
                        ax=self.fig.axes[0],
                        **kw)
        # remove legend
        self.fig.axes[0].legend_.remove()

    @staticmethod
    def _format_ax(ax, channel='ch1_normalized', orientation='v'):
        """
        Format figure axes.

        Args:
        fig (matplotlib.axes.AxesSubplot)
        channel (str) - expression channel
        orientation (str) - box orientation
        """

        # format axis

        # get axis limits
        if channel == 'ch1_normalized':
            lim = (0, 6)
            ticks = np.arange(7)
            label = 'Pnt (a.u.)'
        elif channel == 'ch2_normalized':
            lim = (0, 6)
            ticks = np.arange(7)
            label = 'Yan (a.u.)'
        else:
            lim = (-3, 3.5)
            ticks = np.arange(-3, 4)
            label = 'log2 Ratio'

        # horizontal formatting
        if orientation == 'h':
            ax.set_ylabel('')
            ax.set_xlim(*lim)
            ax.set_xticks(ticks)
            ax.set_xlabel(label)
            ax.spines['left'].set_position(('outward', 5))
            ax.spines['bottom'].set_position(('outward', 1))
            ax.plot([0, 0], [-5, 5], '-', color='pink', linewidth=2, zorder=0)

        # vertical formatting
        else:
            ax.set_xlabel('')
            ax.set_ylim(*lim)
            ax.set_yticks(ticks)
            ax.set_ylabel(label)
            ax.spines['left'].set_position(('outward', 1))
            ax.spines['bottom'].set_position(('outward', -5))
            ax.plot([-5, 5], [0, 0], '-', color='pink', linewidth=2, zorder=0)

        # format spines
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.spines['left'].set_bounds(yticks[0], yticks[-1])
        ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    @staticmethod
    def set_patch_colors(ax, cell_types, lw=1):
        """
        Set colors for patch objects (boxes).

        Args:
        ax (matplotlib.axes.AxesSubplot)
        cell_types (list of lists) - cell types used to define colors
        lw (float) - patch line width
        """

        # instantiate cell type colormaps
        colorer = Palette()
        greys = Palette({'grey': 'grey'})

        # iterate across patch artists
        for i, artist in enumerate(ax.artists):

            # set facecolor
            if i % 2 == 0:
                artist.set_facecolor(greys('grey', 'light', shade=2))
            else:
                artist.set_facecolor(greys('grey', 'dark', shade=2))

            # set linecolor
            types = cell_types[i // 2][0]
            color = colorer(types)
            artist.set_edgecolor(color)
            artist.set_linewidth(lw)

    def get_statistics(self,
                       channel='logratio',
                       test='ks_2samp',
                       rounding=2):
        """
        Perform statistical test to determine whether or not expression levels differ between PntGFP dosages.

        Args:
        channel (str) - expression channel
        test (str) - statistical test used
        rounding (int) - rounding applied to log10(pvalues)

        Returns:
        data (pd.DataFrame) - dataframe summarizing comparison statistics
        """

        # get test
        if test == 'ks_2samp':
            test = lambda x, y: ks_2samp(x, y)
        elif test == 'welch':
            test = lambda x, y: ttest_ind(x, y, equal_var=False)
        elif test == 't':
            test = lambda x, y: ttest_ind(x, y, equal_var=True)

        # determine unique reference populations
        cell_types = self.data['ReferenceType'].unique()

        # split data by PntGFP dosage
        pnt1x_data = self.data[self.data.Dosing=='1X PntGFP']
        pnt2x_data = self.data[self.data.Dosing=='2X PntGFP']

        # iterate across metrics and cell types to be compared
        stats = {}

        # iterate across reference cell types
        for types in cell_types:

            # select concurrent progenitors
            p1 = pnt1x_data[pnt1x_data['ReferenceType']==types]
            p2 = pnt2x_data[pnt2x_data['ReferenceType']==types]

            # get scores
            p1scores = p1[channel].values
            p2scores = p2[channel].values

            # apply test and store log10 pval
            _, pval = test(p1scores, p2scores)
            stats[types] = {'log10 pval': np.round(np.log10(pval), rounding)}

        return pd.DataFrame.from_dict(stats, orient='index')


class PerturbationComparison(Base):
    """
    Object for constructing boxplot comparison between control and perturbation experiments.

    Attributes:
    reference_types (list) - reference cell types
    pre (pd.DataFrame) - data for progenitors concurrent with reference
    reference (pd.DataFrame) - data for reference cells

    Inherited attributes:
    data (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, data, reference, **kwargs):
        """
        Instantiate perturbation comparison figure.

        Args:
        data (pd.DataFrame) - data containing Control/Perturbation labels
        reference (list) - reference cell types
        kwargs: keyword arguments for concurrent cell selection
        """

        # define measurement data
        cells = self.select_concurrent(data, reference, **kwargs)
        self.reference_types = reference
        self.pre = cells[cells.label=='pre']
        self.reference = cells[cells.label.isin(reference)]

        # initialize figure
        self.fig = None

    @classmethod
    def from_experiments(cls, control, perturbation, reference, **kwargs):
        """
        Instantiate from Experiment instances. Cells concurrent with early R-cells are selected by default.

        Args:
        control (data.experiments.Experiment) - control data
        perturbation (data.experiments.Experiment) - perturbation data
        reference (list) - reference cell types
        kwargs: keyword arguments for concurrent cell selection

        Returns:
        figure_obj (figures.base.Base derivative)
        """

        # compile control data
        control = pd.concat([d.data for d in control.discs.values()])
        control['experiment'] = 'Control'

        # compile perturbation data
        perturbation = pd.concat([d.data for d in perturbation.discs.values()])
        perturbation['experiment'] = 'Perturbation'

        # concatenate data
        data = pd.concat((control, perturbation))

        return cls(data, reference, **kwargs)

    @staticmethod
    def select_concurrent(data, reference,
                          offset=1,
                          n=25):
        """
        Select all cells concurrent with early cells of a reference cell type.

        Args:
        data (pd.DataFrame) - cell measurement data
        reference (list) - reference cell types
        offset (int) - index of first early reference cell (excludes outliers)
        n (int) - number of early reference cells included

        Returns:
        concurrent (pd.DataFrame) - concurrent cell measurement data
        """
        selected = data[data.label.isin(reference)]
        selected.sort_values(by='t', inplace=True)
        tmin = selected.iloc[offset:n].t.min()
        tmax = selected.iloc[offset:n].t.max()
        concurrent = data[data.t.between(tmin, tmax)]
        return concurrent

    def render(self,
               channel='logratio',
               figsize=(1., 1.5),
               **kwargs):
        """
        Create formatted figure and plot data.

        Args:
        channel (str) - expression channel
        figsize (tuple) - figure size
        kwargs: keyword arguments for plotting
        """
        self.fig = self.create_figure(figsize=figsize)

        # determine marker color
        palette = Palette()
        marker_color = palette(self.reference_types[0])

        # plot boxplots and markers
        self.plot(channel=channel, color=marker_color, **kwargs)
        self._format_ax(self.fig.axes[0])

    def plot(self,
             channel='logratio',
             color='k',
             size=2):
        """
        Plot data using seaborn.boxplot method. Data must contain "experiment" keys as these are used to categorically partition data.

        Progenitors are shown as boxes, overlayed by reference cell markers.

        Args:
        channel (str) - expression channel
        color (str) - marker color
        size (float) - marker size
        """

        # get axis
        ax = self.fig.axes[0]

        # define box order
        order = ('Control', 'Perturbation')

        # plot progenitors
        ax = sns.boxplot(x="experiment", y=channel, data=self.pre, color=".8", order=order, width=0.5, ax=ax, linewidth=1, fliersize=2)

        # overlay reference cells
        ax = sns.stripplot(x="experiment", y=channel, data=self.reference, size=size, jitter=.25, order=order, color=color, edgecolor='w', linewidth=0.)

    @staticmethod
    def _format_ax(ax):
        """
        Format figure axes.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        """

        # set axis limits
        ax.set_ylim(-2, 2)
        ax.set_yticks(np.arange(-2, 2.25, 1))

        # labels
        ax.tick_params(pad=0, labelsize=6, length=2)
        ax.set_ylabel('')
        ax.set_xticklabels(['WT', 'Sev>RasV12'])
        ax.set_xlabel('')

        # hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
