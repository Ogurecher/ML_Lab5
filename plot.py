import matplotlib.pyplot as plt
import numpy as np


def colors (labels):
    return ['green' if label == 'P' else 'red' for label in labels]


def plot_results (x, y, zz, actual_labels, predicted_labels, filename_out):
    plt.scatter(x, y, c=colors(actual_labels))

    errors = np.array([[xi, yi] for xi, yi, actual_label, predicted_label in zip(x, y, actual_labels, predicted_labels) if
              actual_label != predicted_label])

    plt.scatter(errors[:, 0], errors[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')


    plot_step = 0.02

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    ZZ = np.array([1 if label == 'N' else 0 for label in zz])

    ax = plt.gca()
    ax.contour(xx, yy, ZZ.reshape(xx.shape))

    plt.savefig(filename_out)
    plt.close()


def get_z (x, y, plot_step=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    z = np.c_[xx.ravel(), yy.ravel()]

    return z
