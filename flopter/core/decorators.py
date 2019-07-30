import functools as ft
import matplotlib.pyplot as plt
import collections as coll
import contextlib
import os


def plotmethod(func):  # the decorator
    """
    Decorator function used to add kwargs common to all plotting functions.
    :param func:    Function to be decorated.
    :return:        Decorated function
    """
    @ft.wraps(func)
    def wrapper(*args, style=None, show_fl=False, ax=None, ax_labels=None, ax_range=None, legend_fl=True, **kwargs):
        """
        Wrapper for function which adds minimal plotting functionality of
        selecting a figure to plot on and then whether to plt.show() or not.
        :param style        (string) String of matplotlib style to be switched
                            to for the duration of plotting. Default is no
                            change.
        :param ax:          The matplotlib axis object to be plotted on - useful
                            for when multiple plot functions need to be mixed
                            together onto the same figure. Default is None,
                            which will create a blank figure to plot onto.
        :param ax_labels    (list - str) List of two strings, to be used as
                            the x- and y-axis labels respectively. Default is
                            not applied.
        :param ax_range     (list - float) List of two float pairs, which are
                            unpacked into the x- and y-axis limits
                            (plt.xlim() etc.). Default is not applied.
        :param legend_fl    (boolean) Flag to control whether the legend is
                            shown. Default is True.
        :param show_fl:     (boolean) Flag to control whether to run plt.show()
                            after plotting. Default is False.
        """
        if style is not None:
            plt.style.use(style)

        if not ax:
            plt.figure()

        return_var = func(*args, **kwargs)

        if ax_range and isinstance(ax_range, coll.Iterable) and len(ax_range) == 2:
            x_range, y_range = ax_range
            plt.xlim(*x_range)
            plt.ylim(*y_range)

        if isinstance(ax_labels, coll.Sequence) and all([isinstance(label, str) for label in ax_labels]):
            plt.xlabel(ax_labels[0])
            plt.ylabel(ax_labels[1])

        if legend_fl:
            plt.legend()

        if show_fl:
            plt.show()
        return return_var

    return wrapper


def printmethod(func):
    """
    Decorator function used to add kwargs common to all plotting functions.
    :param func:    Function to be decorated.
    :return:        Decorated function
    """
    def wrapper(*args, print_fl=True, **kwargs):
        """
        Wrapper for function which adds printing functionality.
        :param print_fl:    (boolean) Flag to control whether to run print
                            statement in function is executed. Default is False.
        """
        if not print_fl:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                return_var = func(*args, **kwargs)
        else:
            return_var = func(*args, **kwargs)
        return return_var

    return wrapper
