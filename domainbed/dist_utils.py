# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import random
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from functools import partial


pd.set_option('colheader_justify', 'center')
ERRORS = ["MSE", "FVU", "R2"]     # r^2 = 1 - FVU, where FVU is the fraction of variance unexplained
ERROR = ERRORS[1]
USE_FVU = ERROR in ["MSE", "FVU"]
EPS = 1e-6



def analytical_results(x1_var, y_var, x2_var, use_fvu=USE_FVU):
    """
    TODO: Add r_sq_x1x2
    """
    x1_res = x1_var / (x1_var + y_var)
    x2_res = (x1_var + y_var) / (x1_var + y_var + x2_var)

    if use_fvu:
        x1_res, x2_res = 1. - x1_res, 1. - x2_res

    print(f"Analytical results\nX1:{x1_res:.2f};\tX2:{x2_res:.2f}")


def solve_linear_system(x, y):
    coeff = np.linalg.inv(x.T @ x) @ x.T.dot(y)
    mse = np.mean((y - x.dot(coeff)) ** 2)
    r2 = 1. - mse / y.var()
    return coeff, r2


def empirical_results(env_xs, env_ys, use_fvu=USE_FVU, per_env_results=False, print_coeffs=False):
    """
    TODO: results seem slightly off compared to analytical and SGD.
    """
    env_x1s = [x[:, 0:1].numpy() for x in env_xs]
    env_x2s = [x[:, 1:].numpy() for x in env_xs]

    if per_env_results:
        x1_e_res = [solve_linear_system(x, y) for x, y in zip(env_x1s, env_ys)]
        x2_e_res = [solve_linear_system(x, y) for x, y in zip(env_x2s, env_ys)]
        x1x2_e_res = [solve_linear_system(x, y) for x, y in zip(env_xs, env_ys)]

        if use_fvu:
            x1_e_res = [(coeff, 1. - r2) for (coeff, r2) in x1_e_res]
            x2_e_res = [(coeff, 1. - r2) for (coeff, r2) in x2_e_res]
            x1x2_e_res = [(coeff, 1. - r2) for (coeff, r2) in x1x2_e_res]

        print("Empirical results (per env):")
        for i, ((x1_c, x1_r2), (x2_c, x2_r2), (x1x2_c, x1x2_r2)) in enumerate(zip(x1_e_res, x2_e_res, x1x2_e_res)):
            print(f"Env {i}")
            print(f"X1: res={x1_r2:.2f}" + f"; coeff={x1_c}" * print_coeffs)
            print(f"X2: res={x2_r2:.2f}" + f"; coeff={x2_c}" * print_coeffs)
            print(f"[X1,X2]: res={x1x2_r2:.2f}" + f"; coeffs={x1x2_c}" * print_coeffs)

    # Pooled envs
    pooled_x1s, pooled_ys, pooled_x2s = np.vstack(env_x1s), np.vstack(env_ys), np.vstack(env_x2s)
    x1_pooled_res = solve_linear_system(pooled_x1s, pooled_ys)
    x2_pooled_res = solve_linear_system(pooled_x2s, pooled_ys)
    x1x2_pooled_res = solve_linear_system(np.hstack([pooled_x1s, pooled_x2s]), pooled_ys)

    if use_fvu:    # (coeff, r^2)
        x1_pooled_res = (x1_pooled_res[0], 1. - x1_pooled_res[1])
        x2_pooled_res = (x2_pooled_res[0], 1. - x2_pooled_res[1])
        x1x2_pooled_res = (x1x2_pooled_res[0], 1. - x1x2_pooled_res[1])

    print("Empirical results (pooled envs):")
    print(f"X1: res={x1_pooled_res[1]:.2f}" + f"; coeff={x1_pooled_res[0]}" * print_coeffs)
    print(f"X2: res={x2_pooled_res[1]:.2f}" + f"; coeff={x2_pooled_res[0]}" * print_coeffs)
    print(f"[X1,X2]: res={x1x2_pooled_res[1]:.2f}" + f"; coeffs={x1x2_pooled_res[0]}" * print_coeffs)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_error(algorithm, x, y, error=ERROR):
    with torch.no_grad():
        if len(y.unique()) == 2:    # hack to detect classification
            return algorithm.predict(x).gt(0).ne(y).float().mean().item()
        else:                       # regression
            mse = (algorithm.predict(x) - y).pow(2).mean().item()
            if error == "MSE":
                return mse
            var_y = torch.var(y).item()
            fvu = mse / var_y
            if error == "FVU": # lower is better (like MSE), but still normalised by var(y)
                return fvu
            elif error == "R2":
                return 1. - fvu
            else:
                raise ValueError(f"Invalid error {error}. Choose one of: {ERRORS}.")


def compute_errors(model, envs):
    for split in envs.keys():
        if not bool(model.callbacks["errors"][split]):
            model.callbacks["errors"][split] = {
                key: [] for key in envs[split]["keys"]}

        for k, env in zip(envs[split]["keys"], envs[split]["envs"]):
            model.callbacks["errors"][split][k].append(
                compute_error(model, *env))


def sort_models_alpha(model_name):
    if '=' in model_name:
        start_idx = model_name.index('=')
        end_idx = model_name.index('}$')
        m_name = model_name[:start_idx]
        alpha_value = float(model_name[start_idx + 1:end_idx])
    elif '\\approx' in model_name:
        m_name = model_name[:model_name.index('\\approx')]
        alpha_value = 1
    else:
        m_name = model_name
        alpha_value = 0

    return m_name, alpha_value


def sort_models_alpha_only(model_name):
    if '=' in model_name:
        start_idx = model_name.index('=')
        print(model_name)
        m_name = model_name[:start_idx]
        alpha_value = float(model_name[start_idx + 1:-1])
    elif '\\approx' in model_name:
        m_name = model_name[:model_name.index('\\approx')]
        alpha_value = 1
    else:
        m_name = model_name
        alpha_value = 0

    return m_name, alpha_value


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


def powerset_k(s, k):
    x = len(s)
    masks = [1 << k]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def findsubsets(S, m):
    return set(itertools.combinations(S, m))


# ------ KDE PyTorch Implementation ------
# See https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py for main code.
# See https://github.com/mennthor/awkde for code on adaptive width kernels.
# See https://www.jstor.org/stable/2242011?seq=1 for an analysis of variable kernel density estimation. We use
# a "sample smoothing estimator, as in Eq. 1.7.

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=0.05):
        """Initializes a new Kernel.
        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        bandwidth = torch.ones(1) * bandwidth if isinstance(bandwidth, float) else bandwidth
        self._global_bw = bandwidth         # global bandwidth
        self.local_bw_scalings = None       # local bandwidth scalings/weights

    @property
    def bandwidth(self):
        if self.local_bw_scalings is None:
            return self._global_bw
        else:
            return self._global_bw * self.local_bw_scalings

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth ** dim
        return (coef * inside).mean(dim=1)

    def sample(self, train_Xs):
        device = train_Xs.device
        noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
        return train_Xs + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2
        var = self.bandwidth ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)
        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bandwidth
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bandwidth   # fixed variance
        x_ = test_Xs.repeat(len(mus), 1).T               # repeat observations to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel=None, bandwidth_est_method=None, alpha=0, norm_geom_mean=True):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)
        if kernel is None:
            if bandwidth_est_method is None:
                self.kernel = GaussianKernel()
            else:
                bandwidth = estimate_bandwidth(self.train_Xs, bandwidth_est_method)
                self.kernel = GaussianKernel(bandwidth)
        else:
            self.kernel = kernel

        # adaptive bandwidth based on location (sample smoothing estimator)
        self.adaptive = False
        self._kde_values = None
        self._inv_loc_bw = None
        self.alpha = alpha
        self.norm_geom_mean = norm_geom_mean

        if self.adaptive:
            self._calc_local_bandwidth()

    @property
    def device(self):
        return self.train_Xs.device

    @property
    def alpha(self):
        """Alpha=0: no adaptive behaviour. Alpha=1, weight bandwidth h_i by 1/p(x_i)."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1]")

        if alpha > 0:
            self.adaptive = True
        else:
            self.adaptive = False

        self._alpha = alpha

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)

    def _calc_local_bandwidth(self, n_iters=2):
        """See Eq.2 of https://journals.sagepub.com/doi/pdf/10.1177/1536867X0300300204"""
        for _ in range(n_iters):
            # Calculate "pilot" densities at each x_i using adaptive bandwidths
            self._kde_values = self.kernel(self.train_Xs, self.train_Xs)

            # Get local bandwidth from local "density" g
            if self.norm_geom_mean:
                G = torch.exp(torch.sum(torch.log(self._kde_values)) / len(self._kde_values))
            else:
                G = 1
            self._inv_loc_bw = (G / self._kde_values) ** self._alpha

            # Update kernel bandwidths
            self.kernel.local_bw_scalings = self._inv_loc_bw


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)
    # print(sample_std)
    if 'median' in method:
        upper_tri_diff_matrix = (x_.unsqueeze(1) - x_).triu()
        sq_diffs = upper_tri_diff_matrix[upper_tri_diff_matrix != 0] ** 2
        sorted_sq_diffs, _ = torch.sort(sq_diffs)
        if 'HL' in method:
            # Hodges-Lehmann estimator of the median: data-efficient and robust
            # See https://en.wikipedia.org/wiki/Hodges-Lehmann_estimator
            subsets_2_elements = torch.tensor(list(findsubsets(sorted_sq_diffs, 2)))
            mean_subsets_2_elements = torch.mean(subsets_2_elements, dim=1)
            median_sq_diff = torch.median(mean_subsets_2_elements)
        else:
            median_sq_diff = torch.median(sorted_sq_diffs)
        bandwidth = torch.sqrt(median_sq_diff / 2)

    elif method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method == 'scott':
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 1.059 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method == 'Gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected {method}.")

    return bandwidth


def mise_approx(f, f_ns, xs=None):
    if xs is None:
        xs = torch.linspace(-10, 10, 100)
    approx_integrated_sq_errs = torch.vstack([torch.mean((f(xs) - f_n(xs))**2) for f_n in f_ns])  # estimate of integral
    approx_mise = torch.mean(approx_integrated_sq_errs)                             # mean over f_ns (samples of size n)
    return approx_mise


def add_row_to_table(table, table_columns, row):
    assert len(table_columns) == len(row)
    table.update({c:table[c] + [r] for c, r in zip(table_columns, row)})


def _plot_kde(x, kdes, settings, true_dist, true_samples):
    fig, axs = plt.subplots(nrows=2, sharey=True, sharex=True)

    # Plot each kde (different bandwidth selection methods)
    for (bw_method, alpha), kde_m in zip(settings, kdes):
        is_adaptive = alpha > 0
        if is_adaptive:
            axs[1].plot(x, kde_m(x), label=f"{bw_method}")
        else:
            axs[0].plot(x, kde_m(x), label=f"{bw_method}")

    for i, ax in enumerate(axs):
        # Plot true distribution and n samples
        ax.plot(x.numpy(), torch.exp(true_dist.log_prob(x)).numpy(), label="true pdf", color="gray")
        ax.scatter(true_samples, np.zeros(len(true_samples)), color="black", label="samples", s=2)

        # Settings
        ax.legend()
        ax.set_title("Fixed" if i == 0 else "Adaptive")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p(x)$")

    dist_name = type(true_dist).__name__
    n = len(true_samples)
    plt.suptitle(f"{dist_name} | N={n}")
    plt.savefig(f"kde_test_{dist_name}_{n}.pdf")


def sklearn_kde_likelihood(kde, samples):
    return np.exp(kde.score_samples(samples.numpy().reshape(-1, 1)))


def _test_kde(plot=True):
    ns = [10, 20, 50, 100, 1000]                          # sample size
    m = 100                                                     # number of samples (of size n)
    k = 1000                                                    # number of points over which to estimate MISE
    kde_bw_methods = ["silverman", "median", "Gauss-optimal"]
    alphas = [0, 0.5]
    kde_settings = [(bw_m, a) for bw_m in kde_bw_methods for a in alphas] + [("cross-val", 0)] + [("GMM", 0)]
    seed = 12345

    from distributions import Normal, LogNormal, Gumbel, Weibull, GMM
    gmm_weights = 0.5 * torch.ones(2)
    gmm_means = torch.tensor([-10., 10.])
    gmm_stddevs = torch.ones(2)

    dists = [Normal(0, 1), GMM(gmm_weights, gmm_means, gmm_stddevs), LogNormal(0, 0.75), Gumbel(3, 4), Weibull(2, 2)]
    dist_ranges = [(-4, 4), (-25, 25), (EPS, 8), (-5, 25), (EPS, 5)]
    dist_names = [type(d).__name__ for d in dists]

    dist_kdes = []                                                  # store kds objects for plotting
    dist_samples = []
    table_columns = ["N", "Method", "True Dist.", "MISE"]
    table = {c: [] for c in table_columns}                          # build pandas table of results for printing
    for dist, dist_name, dist_range in zip(dists, dist_names, dist_ranges):
        print(f"True dist: {dist_name}")
        true_pdf = lambda y: torch.exp(dist.log_prob(y))
        ys = torch.linspace(*dist_range, k)
        kdes_d = []
        samples_d = []
        for n in ns:
            print(f"{n} samples.")
            kdes_n = []
            for bw_method, alpha in kde_settings:
                # print(f"Bandwidth method: {bw_method}; alpha: {alpha}; norm_gm: {norm_geom_mean}.")
                # Seed to ensure (i) same samples for each method; (ii) nested samples for ns.
                torch.manual_seed(seed)
                kde_m = []
                for _ in range(m):
                    # Draw n iid samples.
                    x, _ = torch.sort(dist.sample(n))

                    # Estimate density from n samples
                    if bw_method == "cross-val":
                        # use grid search cross-validation to select bandwidth
                        params = {'bandwidth': np.logspace(-1, 2, num=10, base=2)}
                        hyper_kde = GridSearchCV(KernelDensity(), params, n_jobs=1, cv=5, verbose=0)
                        hyper_kde.fit(x.numpy().reshape(-1, 1))
                        kde = hyper_kde.best_estimator_

                        # kde = KernelDensity().fit(x.numpy().reshape(-1, 1))
                        kde = partial(sklearn_kde_likelihood, kde)
                        kde_m.append(kde)

                    elif bw_method == "GMM":
                        kde = GaussianMixture(3).fit(x.numpy().reshape(-1, 1))
                        kde = partial(sklearn_kde_likelihood, kde)
                        kde_m.append(kde)
                    else:
                        kde = KernelDensityEstimator(x, bandwidth_est_method=bw_method, alpha=alpha)
                        kde_m.append(kde)

                kdes_n.append(kde)                  # store only the last kde for plotting purposes
                mise = mise_approx(true_pdf, kde_m, ys)
                add_row_to_table(table, table_columns,
                                 [n, f"{bw_method}-alpha={alpha}", dist_name, round(mise.item(), 5)])

            kdes_d.append(kdes_n)
            samples_d.append(x)                     # store only the last samples for plotting purposes

        dist_kdes.append(kdes_d)
        dist_samples.append(samples_d)

    # Aggregate results and print
    df = pd.DataFrame(table)
    df_pivot = df.pivot_table("MISE", ["N", "Method"], "True Dist.")
    df_pivot["Avg."] = df_pivot[dist_names].mean(axis=1).round(4)
    df_pivot.sort_values(["N", "Avg."], inplace=True)
    print("---------- MISEs ----------")
    print(df_pivot)

    # Plot
    if plot:
        print("\nPlotting...")
        for dist, dist_name, dist_range, kdes_d, s_d in zip(dists, dist_names, dist_ranges, dist_kdes, dist_samples):
            for n, kdes_n, s_n in zip(ns, kdes_d, s_d):
                x = torch.linspace(*dist_range, k)
                samples = np.array(s_n)
                fig, axs = plt.subplots(nrows=2, sharey=True, sharex=True)

                # Plot each kde (different bandwidth selection methods)
                for (bw_method, alpha), kde_m in zip(kde_settings, kdes_n):
                    is_adaptive = alpha > 0
                    if is_adaptive:
                        axs[1].plot(x, kde_m(x), label=f"{bw_method}")
                    else:
                        axs[0].plot(x, kde_m(x), label=f"{bw_method}")

                for i, ax in enumerate(axs):
                    # Plot true distribution and n samples
                    ax.plot(x.numpy(), torch.exp(dist.log_prob(x)).numpy(), label="true pdf", color="gray")
                    ax.scatter(samples, np.zeros(len(samples)), color="black", label="samples", s=2)

                    # Settings
                    ax.set_title("Fixed" if i == 0 else "Adaptive")
                    ax.set_xlabel("$x$")
                    ax.set_ylabel("$p(x)$")

                plt.suptitle(f"{dist_name} | N={n}")
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc=7)
                fig.subplots_adjust(right=0.75)
                fig.subplots_adjust(hspace=0.25)
                plt.savefig(f"kde_test_{dist_name}_{n}.pdf")
                plt.close(fig)


if __name__ == "__main__":
    _test_kde()
