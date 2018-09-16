__author__ = 'Sebastian Bernasek'

import seaborn as sns

# default marker colors
marker_colors = {'pre': 'grey',
                 'r8': 'g',
                 'r2': 'navy',
                 'r5': 'navy',
                 'r1': 'r',
                 'r6': 'r',
                 'r3': 'orange',
                 'r4': 'orange',
                 'r7': 'purple',
                 'c1': 'deepskyblue',
                 'c2': 'deepskyblue',
                 'c3': 'black',
                 'c4': 'black',
                 'n?': 'm',
                 'er': 'deepskyblue',
                 'er7': 'm'}

# default moving average line colors
line_colors = {  'pre': 'k',
                 'r8': 'k',
                 'r2': 'orange',
                 'r5': 'orange',
                 'r1': 'k',
                 'r6': 'k',
                 'r3': 'k',
                 'r4': 'k',
                 'r7': 'k',
                 'c1': 'k',
                 'c2': 'k',
                 'c3': 'k',
                 'c4': 'k',
                 'n?': 'm'}


class Palette:
    """
    Object for generating colormaps for different cell types.

    Attributes:
    palettes (dict) - keys are 'light' and 'dark', values are dictionaries whose keys are cell types and values are seaborn color palettes
    """

    def __init__(self, colors=marker_colors):
        """
        Instantiate color palette from color dictionary.

        Args:
        colors (dict) - keys are cell types, values are colors
        """
        self.palettes = self.set_palettes(colors)

    def __getitem__(self, cell_type, shade=-1):
        """
        Get color from light palette for specified cell type and shade.

        Args:
        cell_type (str) - cell type
        shade (int) - shade, higher is darker

        Returns:
        color (RGB tuple)
        """
        return self.palettes['light'][cell_type][shade]

    def __call__(self, cell_type, palette='light', shade=-1):
        """
        Get color from specified cell type, palette, and shade.

        Args:
        cell_type (str) - cell type
        palette (str) - 'light' or 'dark'
        shade (int) - darkness, higher value is darker

        Returns:
        color (RGB tuple)
        """
        return self.palettes[palette][cell_type][shade]

    def set_palettes(self, colors):
        """
        Construct light and dark palettes for each cell type.

        Args:
        colors (dict) - keys are cell types, values are colors

        Returns:
        palettes (dict) - keys are 'light' and 'dark', values are dictionaries whose keys are cell types and values are seaborn color palettes
        """
        palettes = {
            'light': {cell_type: sns.light_palette(color) for cell_type, color in colors.items()},
            'dark': {cell_type: sns.dark_palette(color) for cell_type, color in colors.items()}}
        return palettes

    def get_color_dict(self, cell_types=None, palette='light'):
        """
        Construct dictionary mapping individual cell type to a color.

        Args:
        cell_types (list) - included cell types
        palette (str) - light or dark

        Returns:
        color_dict (dict) - keys are individual cell types, values are colors
        """
        if cell_types is None:
            cell_types = [['r8'],['r2', 'r5'],['r3', 'r4'],['r1', 'r6'],['r7']]
        return {ct[0]: self.palettes[palette][ct[0]] for ct in cell_types}

