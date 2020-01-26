import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save():
    pass


def plot(x_list, y_list, range):
    assert len(x_list) == len(y_list)
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", len(x_list))
    with sns.axes_style('darkgrid'):
        pass
