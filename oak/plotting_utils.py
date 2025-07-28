# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union
import gpflow
import matplotlib
import numpy as np
import tensorflow as tf
import tikzplotlib
from matplotlib import pyplot as plt
from oak.utils import get_model_sufficient_statistics


# -

@dataclass
class FigureDescription:
    fig: matplotlib.figure.Figure
    description: str


def save_fig_list(
    fig_list: List[FigureDescription],
    dirname: Path,
):
    # save figures to local directory
    dirname.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {dirname}")
    for f in fig_list:
        f.fig.savefig(dirname / (f.description + ".pdf"), bbox_inches="tight")


def plot_single_effect(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR, gpflow.models.SVGP],
    i: int,
    covariate_name: str = "",
    title: str = "",
    x_transform=None,
    y_transform=None,
    semilogy=False,
    plot_corrected_data=False,
    plot_raw_data=False,
    X_list=None,
    fontsize=22,
    tikz_path=None,
    ylim=None,
    quantile_range: Optional[List] = None,
    log_bin=False,
    num_bin: Optional[int] = 100,
):
    """
    :param m: a gpflow GPR or SVGP instance, it is expected to contain an instance of the OAK Kernel.
    :param i: (integer) the  index of the effect to plot
    :param covariate_name: str, used for the plot title
    :param title: title of the plot
    :param x_transform: callable function that maps the i'th column of X back to original coordinates
    :param y_transform: callable function that maps the Y-data back to original coordinates
    :param semilogy: whether to log transform the y-axis
    :param plot_corrected_data: whether to scatter plot corrected data (substract other effects)
    :param plot_raw_data: whether to scatter plot raw data
    :param X_list: optional list of input features [X0, X1] If provided, plot histograms of elements X0 and X1
    :param tikz_path: path to save tikz figure
    :param ylim: range on the y-axis
    :param quantile_range: quantile of range of features to plot, should be in [0,100], e.g. [2,98]
    :param log_bin: whether to use bins on log scale for histograms
    :param num_bin: number of bins for histogram
    :return Make a plot of a single effect (aka main effect).
    """
    matplotlib.rcParams.update({"font.size": fontsize})
    X, Y = m.data

    if isinstance(m, gpflow.models.SVGP):
        posterior = m.posterior()
        alpha, Qinv = posterior.alpha, posterior.Qinv[0]
        # separate condition when we plot the latent effects
        if i == m.data[0].shape[1]:
            Xi = np.linspace(-3, 3, 100)
            X_histogram = np.random.normal(size=1000)
        else:
            Xi = X[:, i].numpy()
    else:
        alpha, L = get_model_sufficient_statistics(m)
        Xi = X[:, i].numpy()

    if isinstance(m, gpflow.models.GPR):
        X_conditioned = X
    elif isinstance(m, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X_conditioned = m.inducing_variable.Z

    if quantile_range is None:
        quantile_range = [0, 100]
    xmin, xmax = np.percentile(Xi, (quantile_range[0], quantile_range[1]))
    xx = np.linspace(xmin, xmax, 100)
    Kxx = (
        m.kernel.kernels[i].K(xx[:, None], X_conditioned[:, i : i + 1])
        * m.kernel.variances[1]
    )
    mu = tf.matmul(Kxx, alpha)[:, 0]
    if isinstance(m, gpflow.models.SVGP):
        Kxx = tf.transpose(Kxx)
        tmp = tf.matmul(Kxx, Qinv @ Kxx, transpose_a=True)
        var = m.kernel.kernels[i].K_diag(xx[:, None]) * m.kernel.variances[
            1
        ] - tf.linalg.diag_part(tmp)
    else:
        tmp = tf.linalg.triangular_solve(L, tf.transpose(Kxx))
        var = m.kernel.kernels[i].K_diag(xx[:, None]) * m.kernel.variances[1] - np.sum(
            tmp ** 2, axis=0
        )
    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    # do "data correction" for what each component is seeing
    if plot_corrected_data:
        K_sub = m.kernel(X, X_conditioned)
        K_sub -= (
            m.kernel.kernels[i].K(X[:, i : i + 1], X_conditioned[:, i : i + 1])
            * m.kernel.variances[1]
        )
        Y_corrected = Y - tf.matmul(K_sub, alpha)

    # rescale the x-data.
    if x_transform is None:
        xx_rescaled = 1.0 * xx
        Xi_rescaled = 1.0 * Xi
    else:
        xx_rescaled = x_transform(xx)
        Xi_rescaled = x_transform(Xi)
        if not isinstance(xx_rescaled, np.ndarray):
            xx_rescaled = xx_rescaled.numpy()
        if not isinstance(Xi_rescaled, np.ndarray):
            Xi_rescaled = Xi_rescaled.numpy()

    # re-scale the predictions and the y-data
    if y_transform is None:
        mu_rescaled = 1.0 * mu
        lower_rescaled = 1.0 * lower
        upper_rescaled = 1.0 * upper
        Y_rescaled = Y * 1.0
        if plot_corrected_data:
            Y_corrected_rescaled = 1.0 * Y_corrected
    else:
        mu_rescaled = y_transform(mu)
        lower_rescaled = y_transform(lower)
        upper_rescaled = y_transform(upper)
        Y_rescaled = y_transform(Y)
        if plot_corrected_data:
            Y_corrected_rescaled = y_transform(Y_corrected)

    # do the actual plotting
    figure = plt.figure(figsize=(8, 4))
    ax1 = figure.add_axes([0.2, 0.2, 0.75, 0.75])
    ax1.plot(xx_rescaled, mu_rescaled, linewidth=1, color="k", zorder=11)
    ax1.plot(xx_rescaled, lower_rescaled, linewidth=0.5, color="k", zorder=11)
    ax1.plot(xx_rescaled, upper_rescaled, linewidth=0.5, color="k", zorder=11)
    ax1.fill_between(xx_rescaled, lower_rescaled, upper_rescaled, alpha=0.2, color="C0")
    if plot_corrected_data:
        ax1.plot(
            Xi_rescaled,
            Y_corrected_rescaled,
            "C0x",
            label="data with other effects removed",
        )
        ax1.set_ylim(*np.percentile(Y_corrected_rescaled, (2, 98)))
    else:
        ax1.set_ylim(
            *np.percentile(Y_rescaled, (0, 98))
        ) if ylim is None else ax1.set_ylim(ylim)
    ax1.set_xlim(xx_rescaled.min(), xx_rescaled.max())

    if plot_raw_data:
        ax1a = ax1.twinx()
        ax1a.plot(Xi_rescaled, Y_rescaled, "C1x")
        ax1a.set_ylabel("Raw data", color="C1")
        ax1a.spines["bottom"].set_visible("False")
        ax1.set_zorder(ax1a.get_zorder() + 1)
        if semilogy:
            ax1a.semilogy()

    ax1.patch.set_visible(False)
    for tick in ax1.get_xticklabels():
        tick.set_visible(False)

    ax1.set_ylabel("$f_{" + covariate_name + "}$")
    ax1.set_title(title)
    ax1.spines["bottom"].set_visible("False")

    ax2 = figure.add_axes([0.2, 0.05, 0.75, 0.15], sharex=ax1)
    bins = (
        num_bin
        if not log_bin
        else np.logspace(
            start=np.log10(Xi_rescaled.min() + 1),
            stop=np.log10(Xi_rescaled.max() + 1),
            num=num_bin,
        )
    )
    if X_list is not None:
        assert len(X_list) == 2
        ax2.hist(X_list[0], alpha=0.3, color="orange", bins=bins, label="data 1")
        ax2.hist(X_list[1], alpha=0.3, color="blue", bins=bins, label="data 2")
        ax2.legend(loc="upper right", prop={"size": 12})
    else:
        ax2.hist(Xi_rescaled.flatten(), alpha=0.2, color="grey", bins=bins)
    ax2.set_yticks([])
    ax2.set_xlabel(covariate_name)
    if semilogy:
        ax1.semilogy()
    fig_list = FigureDescription(fig=figure, description=title)
    if tikz_path is not None:
        tikzplotlib.save(tikz_path + f"{title}.tex")
    return fig_list


def plot_second_order(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR, gpflow.models.SVGP],
    i: int,
    j: int,
    covariate_names: Optional[str] = None,
    x_transforms: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    title: Optional[str] = "",
    tikz_path: Optional[str] = None,
    quantile_range: Optional[List[List]] = [[2, 98], [2, 98]],
    log_axis: Optional[List[bool]] = [False, False],
    xx: Optional[np.ndarray] = None,
    yy: Optional[np.ndarray] = None,
    num_bin: Optional[int] = 100,
):
    """
    :param m: gpflow model
    :param i: index of feature on the x-axis
    :param j: index of feature on the y-axis
    :param covariate_names: list of feature names to label on the axes
    :param x_transforms: inverse transformation of features from the standardized space back to original space
    :param y_transform: transformation of output to the original space
    :param title: title of plot
    :param tikz_path: path to save tikz figure
    :param quantile_range: list of range of features i and j to plot
    :param log_axis: list of boolean indicating whether to plot axis on log(x+1) space
    :param xx: x-value of the grid points to evaluate functions on, if None, use linspace of standardised feature i
    :param yy: y-value of the grid points to evaluate functions on, if None, use linspace of standardised feature j
    :param num_bin: number of bins for histogram
    """
    if covariate_names is None:
        covariate_names = [f"input {i}", f"input {j}"]

    X, Y = m.data

    if isinstance(m, gpflow.models.SVGP):
        posterior = m.posterior()
        alpha = posterior.alpha
    else:
        alpha, _ = get_model_sufficient_statistics(m)
    Xi = X[:, i].numpy()
    Xj = X[:, j].numpy()

    if isinstance(m, gpflow.models.GPR):
        X_conditioned = X
    elif isinstance(m, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X_conditioned = m.inducing_variable.Z

    if quantile_range[0] is not None:
        xmin, xmax = np.percentile(Xi, (quantile_range[0][0], quantile_range[0][1]))
    else:
        xmin, xmax = Xi.min(), Xi.max()
    if quantile_range[1] is not None:
        ymin, ymax = np.percentile(Xj, (quantile_range[1][0], quantile_range[1][1]))
    else:
        ymin, ymax = Xj.min(), Xj.max()

    xx_range = np.linspace(start=xmin, stop=xmax, num=50) if xx is None else xx
    yy_range = np.linspace(start=ymin, stop=ymax, num=50) if yy is None else yy

    xx, yy = np.meshgrid(xx_range, yy_range)
    XX = np.vstack([xx.flatten(), yy.flatten()]).T
    Kxx = (
        m.kernel.kernels[i].K(XX[:, 0:1], X_conditioned[:, i : i + 1])
        * m.kernel.variances[2]
    )
    Kxx *= m.kernel.kernels[j].K(XX[:, 1:2], X_conditioned[:, j : j + 1])
    mu = np.dot(Kxx, alpha)

    # rescale the x- and y-data.
    if x_transforms is None:
        xx_rescaled = 1.0 * xx
        Xi_rescaled = 1.0 * Xi
        yy_rescaled = 1.0 * yy
        Xj_rescaled = 1.0 * Xj
    else:
        xx_rescaled = x_transforms[0](xx)
        Xi_rescaled = x_transforms[0](Xi)
        yy_rescaled = x_transforms[1](yy)
        Xj_rescaled = x_transforms[1](Xj)

        if not isinstance(xx_rescaled, np.ndarray):
            xx_rescaled = xx_rescaled.numpy()
        if not isinstance(Xi_rescaled, np.ndarray):
            Xi_rescaled = Xi_rescaled.numpy()
        if not isinstance(yy_rescaled, np.ndarray):
            yy_rescaled = yy_rescaled.numpy()
        if not isinstance(Xj_rescaled, np.ndarray):
            Xj_rescaled = Xj_rescaled.numpy()

    # re-scale the predictions
    if y_transform is None:
        mu_rescaled = 1.0 * mu
    else:
        mu_rescaled = y_transform(mu)

    # do the actual plotting
    figure = plt.figure(figsize=(8, 4))
    ax1 = figure.add_axes([0.2, 0.2, 0.75, 0.75])
    bins_i = bins_j = num_bin
    if log_axis[0] is True:
        ax1.set_xscale("log")
        # plot log(x+1) if on log scale
        xx_rescaled += 1
        Xi_rescaled += 1
        bins_i = np.logspace(
            start=np.log10(Xi_rescaled.min() + 1),
            stop=np.log10(Xi_rescaled.max() + 1),
            num=num_bin,
        )

    if log_axis[1] is True:
        ax1.set_yscale("log")
        yy_rescaled += 1
        Xj_rescaled += 1
        bins_j = np.logspace(
            start=np.log10(Xj_rescaled.min() + 1),
            stop=np.log10(Xj_rescaled.max() + 1),
            num=num_bin,
        )

    contours = ax1.contour(
        xx_rescaled,
        yy_rescaled,
        mu_rescaled.reshape(*xx.shape),
        linewidths=1.4,
        colors="C0",
    )
    ax1.clabel(contours, inline=1, fontsize=20)
    ax1.set_title(title)

    ax2 = figure.add_axes([0.2, 0.05, 0.75, 0.15], sharex=ax1)
    ax2.hist(Xi_rescaled.flatten(), alpha=0.2, color="grey", bins=bins_i)
    ax2.set_yticks([])
    ax2.set_xlabel(covariate_names[0])

    ax3 = figure.add_axes([0.08, 0.2, 0.12, 0.75], sharey=ax1)
    ax3.hist(
        Xj_rescaled.flatten(),
        alpha=0.2,
        color="grey",
        bins=bins_j,
        orientation="horizontal",
    )
    ax3.set_xticks([])
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_ylabel(covariate_names[1])

    ax1.set_xlim(xx_rescaled.min(), xx_rescaled.max())
    ax1.set_ylim(yy_rescaled.min(), yy_rescaled.max())

    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_visible(False)

    fig_list = FigureDescription(fig=figure, description=title)
    if tikz_path is not None:
        tikzplotlib.save(tikz_path + f"{title}.tex")
    return fig_list


def plot_single_effect_binary(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR, gpflow.models.SVGP],
    i: int,
    binary_name: List[str],
    covariate_name: str = "",
    title: str = "Output Effect",
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    semilogy: bool = False,
    tikz_path: Optional[str] = None,
):
    # 1) unpack data + stats
    X, Y = m.data
    Xi = X[:, i].numpy().squeeze().astype(int)             # shape (N,)
    alpha_tf, L_tf = get_model_sufficient_statistics(m)   # alpha_tf: [M,1], L_tf: [M,M]
    alpha = alpha_tf.numpy().reshape(-1)                  # [M,]
    
    # 2) choose conditioning set
    if isinstance(m, gpflow.models.GPR):
        Xc = X
    else:  # SGPR or SVGP
        Xc = m.inducing_variable.Z
    
    # 3) posterior effect at {0,1}
    xx = np.array([[0.0], [1.0]])                         # shape (2,1)
    # cross‐covariance: (2 × M)
    Kxx = m.kernel.kernels[i].K(xx, Xc[:, i : i+1]).numpy()
    Kxx *= float(m.kernel.variances[1].numpy())
    # mean + variance
    mu = Kxx.dot(alpha)                                    # (2,)
    tmp = tf.linalg.triangular_solve(L_tf, tf.transpose(Kxx))  # (M,2)
    tmp = tmp.numpy()
    var = (m.kernel.kernels[i].K_diag(xx).numpy() * float(m.kernel.variances[1].numpy())
           - np.sum(tmp**2, axis=0))                      # (2,)
    lower = mu - 2*np.sqrt(var)
    upper = mu + 2*np.sqrt(var)

    # 4) compute “corrected” posterior mean at training points
    #    i.e. remove the i-th effect from the full posterior
    K_sub = m.kernel(X, Xc).numpy()                        # (N × M)
    Ki   = m.kernel.kernels[i].K(X[:, i:i+1], Xc[:, i:i+1]).numpy()
    Ki  *= float(m.kernel.variances[1].numpy())
    K_sub -= Ki                                           # remove effect i
    mu_minus_i = K_sub.dot(alpha)                         # (N,)

    # 5) apply y_transform if given
    if y_transform is not None:
        Ycorr = y_transform(mu_minus_i)
        mu_plot   = y_transform(mu)
        lower_plot = y_transform(lower)
        upper_plot = y_transform(upper)
    else:
        Ycorr     = mu_minus_i
        mu_plot   = mu
        lower_plot = lower
        upper_plot = upper

    # 6) plotting
    fig, ax1 = plt.subplots(figsize=(10,6))
    # red bars for uncertainty
    ax1.plot([0,0], [lower_plot[0], upper_plot[0]], lw=6, color="r")
    ax1.plot([1,1], [lower_plot[1], upper_plot[1]], lw=6, color="r")
    # twin‐axis for boxplot
    ax2 = ax1.twinx()
    ax1.get_shared_y_axes().join(ax1, ax2)

    # boxplot of corrected data by group
    data0 = Ycorr[Xi == 0]
    data1 = Ycorr[Xi == 1]
    ax2.boxplot([data0, data1], positions=[0,1])
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(binary_name)
    ax2.set_ylabel("data with other effects removed", color="k")

    # posterior means
    ax1.plot([0,1], mu_plot, "x", color="b", ms=10)
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(binary_name)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylabel(title, color="r")
    ax1.set_title(covariate_name)

    if semilogy:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    plt.tight_layout()
    if tikz_path is not None:
        tikzplotlib.save(f"{tikz_path}{title}.tex")

    return FigureDescription(fig=fig, description=title)


def plot_second_order_binary(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR],
    i: int,
    j: int,
    binary_name: list,
    covariate_names: Optional[list] = None,
    title: str = "",
    x_transforms=None,
    y_transform=None,
    tikz_path=None,
):
    """
    :param m: GP model
    :param i: index of continuous feature
    :param j: index of binary feature
    :param covariate_name: list of continuous and binary feature name
    :param x_transforms: transformation for continuous feature
    :param y_transform: transformation for the output

    """
    if covariate_names is None:
        covariate_names = [f"input {i}", f"input {j}"]

    X, Y = m.data
    Xi = X[:, i].numpy()
    alpha, L = get_model_sufficient_statistics(m)
    if isinstance(m, gpflow.models.GPR):
        X_conditioned = X
    elif isinstance(m, gpflow.models.SGPR):
        X_conditioned = m.inducing_variable.Z

    xmin, xmax = np.percentile(Xi, (2, 98))

    xx, yy = np.mgrid[xmin:xmax:100j, 0:1:2j]
    XX = np.vstack([xx.flatten(), yy.flatten()]).T
    Kxx = (
        m.kernel.kernels[i].K(XX[:, 0:1], X_conditioned[:, i : i + 1])
        * m.kernel.variances[2]
    )
    Kxx *= m.kernel.kernels[j].K(XX[:, 1:2], X_conditioned[:, j : j + 1])
    mu = np.dot(Kxx, alpha)[:, 0]

    tmp = tf.linalg.triangular_solve(L, tf.transpose(Kxx))
    var = m.kernel.kernels[i].K_diag(XX[:, 0:1]) * m.kernel.kernels[j].K_diag(
        XX[:, 1:2]
    ) * m.kernel.variances[2] - np.sum(tmp ** 2, axis=0)

    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    # do "data correction" for what each component is seeing
    K_sub = m.kernel(X, X_conditioned)
    K_sub -= (
        m.kernel.kernels[i].K(X[:, i : i + 1], X_conditioned[:, i : i + 1])
        * m.kernel.kernels[j].K(X[:, j : j + 1], X_conditioned[:, j : j + 1])
        * m.kernel.variances[2]
    )

    if x_transforms is None:
        xx_rescaled = 1.0 * xx[:, 0]
        Xi_rescaled = 1.0 * Xi
    else:
        xx_rescaled = x_transforms[0](xx[:, 0]).numpy()
        Xi_rescaled = x_transforms[0](Xi).numpy()

    # re-scale the predictions and the y-data
    if y_transform is None:
        mu_rescaled = 1.0 * mu
        lower_rescaled = 1.0 * lower
        upper_rescaled = 1.0 * upper
    else:
        mu_rescaled = y_transform(mu)
        lower_rescaled = y_transform(lower)
        upper_rescaled = y_transform(upper)

    # do the actual plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex="col", figsize=(10, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25, right=1)

    ax1 = axes[0]
    ax2 = axes[1]

    mu_rescaled0 = mu_rescaled[yy.flatten() == 0]
    mu_rescaled1 = mu_rescaled[yy.flatten() == 1]
    lower_rescaled0 = lower_rescaled[yy.flatten() == 0]
    lower_rescaled1 = lower_rescaled[yy.flatten() == 1]
    upper_rescaled0 = upper_rescaled[yy.flatten() == 0]
    upper_rescaled1 = upper_rescaled[yy.flatten() == 1]

    ax1.plot(xx_rescaled, lower_rescaled0, linewidth=0.5, color="k", zorder=11)
    ax1.plot(xx_rescaled, upper_rescaled0, linewidth=0.5, color="k", zorder=11)
    ax1.plot(
        xx_rescaled,
        mu_rescaled0,
        linewidth=2,
        color="C0",
        zorder=10,
        label=binary_name[0],
    )
    ax1.fill_between(
        xx_rescaled, lower_rescaled0, upper_rescaled0, alpha=0.2, color="C0"
    )

    ax1.legend()

    ax2.plot(
        xx_rescaled,
        mu_rescaled1,
        linewidth=2,
        color="C0",
        zorder=10,
        label=binary_name[1],
    )
    ax2.plot(xx_rescaled, lower_rescaled1, linewidth=0.5, color="k", zorder=11)
    ax2.plot(xx_rescaled, upper_rescaled1, linewidth=0.5, color="k", zorder=11)
    ax2.fill_between(
        xx_rescaled, lower_rescaled1, upper_rescaled1, alpha=0.2, color="C0"
    )

    ax2.legend()
    ax1.set_title(title)

    ax1.set_xlim(xx_rescaled.min(), xx_rescaled.max())

    ax3 = fig.add_axes([0.25, 0.02, 0.75, 0.15], sharex=ax2)
    ax3.hist(Xi_rescaled.flatten(), alpha=0.2, color="grey", bins=50)
    ax3.set_yticks([])
    ax3.set_xlabel(covariate_names[0])

    for tick in ax1.get_xticklabels() + ax2.get_xticklabels():
        tick.set_visible(False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    Y_dict = {binary_name[0]: mu_rescaled0, binary_name[1]: mu_rescaled1}  # , \

    ax.boxplot(Y_dict.values(), positions=np.array(range(2)))
    ax.set_xticklabels(Y_dict.keys())
    ax.set_ylabel("Predicted Effect", color="k")
    ax.set_title(title)

    fig_list = FigureDescription(fig=fig, description=title)
    if tikz_path is not None:
        tikzplotlib.save(tikz_path + f"{title}.tex")
    return fig_list


def plot_single_effect_categorical(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR, gpflow.models.SVGP],
    i: int,
    categorical_name: list,
    title: str = "Output Effect",
    y_transform=None,
    semilogy=False,
    tikz_path=None,
):
    X, Y = m.data
    alpha, L = get_model_sufficient_statistics(m)
    if isinstance(m, gpflow.models.GPR):
        X_conditioned = X
    elif isinstance(m, (gpflow.models.SVGP, gpflow.models.SGPR)):
        X_conditioned = m.inducing_variable.Z

    num_cat = m.kernel.kernels[i].num_cat
    xx = np.arange(num_cat)
    Kxx = (
        m.kernel.kernels[i].K(xx[:, None], X_conditioned[:, i : i + 1])
        * m.kernel.variances[1]
    )
    mu = tf.matmul(Kxx, alpha)[:, 0]
    tmp = tf.linalg.triangular_solve(L, tf.transpose(Kxx))
    var = m.kernel.kernels[i].K_diag(xx[:, None]) * m.kernel.variances[1] - np.sum(
        tmp ** 2, axis=0
    )

    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    if y_transform is None:
        mu_rescaled = 1.0 * mu
        lower_rescaled = 1.0 * lower
        upper_rescaled = 1.0 * upper
    else:
        mu_rescaled = y_transform(mu)
        lower_rescaled = y_transform(lower)
        upper_rescaled = y_transform(upper)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    for ii in range(num_cat):
        ax1.plot(
            [ii, ii],
            [lower_rescaled[ii], upper_rescaled[ii]],
            linewidth=8,
            color="cornflowerblue",
        )
        ax1.plot(ii, mu_rescaled[ii], "x", linewidth=20, color="r")

    plt.xticks(np.arange(num_cat), [categorical_name[ii] for ii in range(num_cat)])
    plt.xlim([-1, num_cat])
    plt.tight_layout()

    ax1.set_ylabel("Output Effect")
    ax1.set_title(title)

    if semilogy:
        ax1.semilogy()

    fig_list = FigureDescription(fig=fig, description=title)
    if tikz_path is not None:
        tikzplotlib.save(tikz_path + f"{title}.tex")
    return fig_list




def plot_second_order_2dkernel(
    m: Union[gpflow.models.GPR, gpflow.models.SGPR, gpflow.models.SVGP],
    i: int,
    j: int,
    covariate_names: Optional[List[str]] = None,
    x_transforms: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    title: str = "",
    tikz_path: Optional[str] = None,
    quantile_range: Optional[List[List]] = [[2, 98], [2, 98]],
    log_axis: Optional[List[bool]] = [False, False],
    xx: Optional[np.ndarray] = None,
    yy: Optional[np.ndarray] = None,
    num_bin: int = 100,
):
    # ---- 1) Setup names and get alpha ----
    if covariate_names is None:
        covariate_names = [f"input {i}", f"input {j}"]

    X, Y = m.data
    if isinstance(m, gpflow.models.SVGP):
        posterior = m.posterior()
        alpha = posterior.alpha.numpy()
    else:
        alpha, _ = get_model_sufficient_statistics(m)
        alpha = alpha.numpy()

    Xi = X[:, i].numpy().flatten()
    Xj = X[:, j].numpy().flatten()

    # ---- 2) Decide where to evaluate the grid ----
    if quantile_range[0] is not None:
        xmin, xmax = np.percentile(Xi, quantile_range[0])
    else:
        xmin, xmax = Xi.min(), Xi.max()
    if quantile_range[1] is not None:
        ymin, ymax = np.percentile(Xj, quantile_range[1])
    else:
        ymin, ymax = Xj.min(), Xj.max()

    xx_range = xx if xx is not None else np.linspace(xmin, xmax, 50)
    yy_range = yy if yy is not None else np.linspace(ymin, ymax, 50)
    xxg, yyg = np.meshgrid(xx_range, yy_range)
    grid = np.vstack([xxg.ravel(), yyg.ravel()]).T  # shape [G,2]

    # ---- 3) Figure out which kernel to use ----
    # Look for a 2‐D kernel whose active_dims exactly matches {i,j}
    two_d_kernel = None
    for kern in m.kernel.kernels:
        if hasattr(kern, "active_dims") and set(kern.active_dims) == {i, j}:
            two_d_kernel = kern
            break

    # ---- 4) Build K_grid × Z ----
    # We need the “inputs” in the full space of X_conditioned
    if isinstance(m, gpflow.models.GPR):
        Z = X
    else:
        Z = m.inducing_variable.Z

    if two_d_kernel is not None:
        # Build a full G×n grid with two‐dim kernel
        Kgrid = two_d_kernel.K(
            # need to pad grid back into full-dim X format,
            # so insert dummy zeros for other dims
            np.hstack([
                np.zeros((grid.shape[0], X.shape[1] - 2)),  # dummy columns
                grid  # but make sure the two cols align to i,j in ordering
            ])[:, np.arange(X.shape[1])],  # reorder to match X dims
            Z
        )
        # scale by the kernel’s variance if it has one
        if hasattr(two_d_kernel, "variance"):
            Kgrid = Kgrid * two_d_kernel.variance.numpy()
    else:
        # fallback: multiply two 1‐D kernels as before
        Ki = m.kernel.kernels[i].K(
            grid[:, :1], Z[:, i : i + 1]
        ) * m.kernel.variances[i].numpy()
        Kj = m.kernel.kernels[j].K(
            grid[:, 1:2], Z[:, j : j + 1]
        ) * m.kernel.variances[j].numpy()
        Kgrid = Ki * Kj

    # ---- 5) Compute the surface ----
    mu = Kgrid.dot(alpha)  # shape [G,]

    # ---- 6) Rescale back to original coords if needed ----
    if x_transforms:
        xxg = x_transforms[0](xxg)
        Xi = x_transforms[0](Xi)
        yyg = x_transforms[1](yyg)
        Xj = x_transforms[1](Xj)
    if y_transform:
        mu = y_transform(mu)

    # ---- 7) Plot ----
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    cs = ax.contour(
        xxg, yyg, mu.reshape(xxg.shape),
        linewidths=1.2, colors="C0"
    )
    ax.clabel(cs, inline=1, fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(covariate_names[0])
    ax.set_ylabel(covariate_names[1])

    # marginal histograms
    ax_x = fig.add_axes([0.2, 0.05, 0.75, 0.15], sharex=ax)
    ax_x.hist(Xi, bins=num_bin, color="gray", alpha=0.3)
    ax_x.set_yticks([])

    ax_y = fig.add_axes([0.05, 0.2, 0.15, 0.75], sharey=ax)
    ax_y.hist(Xj, bins=num_bin, orientation="horizontal", color="gray", alpha=0.3)
    ax_y.set_xticks([])

    ax.set_xlim(xxg.min(), xxg.max())
    ax.set_ylim(yyg.min(), yyg.max())

    # optional log‐scales
    if log_axis[0]:
        ax.set_xscale("log")
    if log_axis[1]:
        ax.set_yscale("log")

    if tikz_path:
        import tikzplotlib
        tikzplotlib.save(f"{tikz_path}/{title}.tex")

    return FigureDescription(fig=fig, description=title)
