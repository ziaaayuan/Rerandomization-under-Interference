"""Redraw every simulation figure of the paper from the saved simulation output (no re-simulation).

Figures are drawn at the size at which they are printed (about 2.3 in wide for one panel), with
9-point text, so that all lettering is at least 8 points in the manuscript (ASA requirement).
Symbol keys are given in the figure captions instead of in-panel legends where panels are small.

Usage (defaults are the output folders written by the notebooks and scripts in this directory):
    python make_figures.py [--vr-dir results_vr_data] [--cv-dir results_cv_data]
                           [--altinf results_altinf/altinf_results.json] [--out figures]
Output (inside --out):
    variance_reduction/vr_covs-<set>_type<T>.pdf            variance ratio panels
    conservative_variance/cv_covs-<set>_type<T>_ratio.pdf    CI-length ratio panels
    conservative_variance/cv_covs-<set>_type<T>_comparison.pdf  conservativeness-ratio panels
    conservative_variance/cv_covs-np_type<T>_coverage.pdf    coverage (three strategies)
    alternative_inference/altinf_type<T>_{variance,cilength}.pdf
    critical_value.pdf                                        q_{1-alpha/2}(R^2; p, a) curves
For each (balancing set, outcome model) the most recent JSON file in the data directory is used.
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PANEL = (2.34, 1.75)        # inches: 0.36\textwidth with 1-inch margins (also used at 0.32\textwidth)
FS = 9                      # points; >= 8 pt even when a panel is shown at 0.32\textwidth
XLABEL = r'$\log_2 \kappa$'
STRATEGY = {                # key -> (color, marker, label)
    'static':  ('#1f77b4', 'o', 'Pretreatment'),
    'dynamic': ('#2ca02c', 's', 'Treatment-dependent'),
    'all':     ('#d62728', '^', 'Joint'),
}


def style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass
    plt.rcParams.update({
        'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
        'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}',
        'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'axes.linewidth': 0.6,
        'grid.linewidth': 0.4, 'grid.linestyle': '--', 'grid.alpha': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'axes.labelpad': 2, 'savefig.dpi': 300,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.01, 'figure.constrained_layout.w_pad': 0.01,
    })


def latest(directory, prefix):
    out = {}
    for f in glob.glob(os.path.join(directory, f'{prefix}_covs-*_type*_*.json')):
        m = re.match(rf'({prefix}_covs-.+?_type\d+)_(\d{{8}}_\d{{6}})\.json$', os.path.basename(f))
        if m and (m.group(1) not in out or m.group(2) > out[m.group(1)][1]):
            out[m.group(1)] = (f, m.group(2))
    return {k: v[0] for k, v in out.items()}


def load(path):
    with open(path) as f:
        return json.load(f)


def xs(res):
    return [s['actual_log_ratio'] for s in res['sweep']]


def finish(ax, xlim=(-4, 4)):
    ax.set_xlim(*xlim)
    ax.set_xticks(range(-4, 5, 2))
    ax.set_xlabel(XLABEL)
    ax.grid(True)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- variance ratio
def plot_vr(res, path):
    fig, ax = plt.subplots(figsize=PANEL)
    for m, (c, mk, lab) in STRATEGY.items():
        ax.plot(xs(res), [s['variance_ratios'][m] for s in res['sweep']], f'{mk}-', color=c, label=lab)
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_ylim(0.0, 1.2)
    ax.set_yticks([0.0, 0.4, 0.8, 1.2])
    ax.set_ylabel('Variance ratio')
    finish(ax)
    save(fig, path)


# ---------------------------------------------------------------- CI length ratio (rho)
def rho_values(res, strat):
    y = []
    for s in res['sweep']:
        sd = s.get(strat, {})
        if not isinstance(sd, dict):
            y.append(np.nan); continue
        rho = sd.get('rho_ci_length', np.nan)
        if rho is None or not np.isfinite(rho):
            v = sd.get('mean_v_hat_n', np.nan)
            u = sd.get('mean_u_hat_n_bern', sd.get('mean_u_hat_n', np.nan))
            rho = np.sqrt(v / u) if (np.isfinite(v) and np.isfinite(u) and u > 0) else np.nan
        y.append(rho)
    return y


def plot_ratio(res, path):
    fig, ax = plt.subplots(figsize=PANEL)
    for m, (c, mk, lab) in STRATEGY.items():
        ax.plot(xs(res), rho_values(res, m), f'{mk}-', color=c, label=lab)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel(r'$\rho = \widehat{L}_{\mathrm{ReM}} / \widehat{L}_{\mathrm{Bern}}$')
    finish(ax)
    save(fig, path)


# ---------------------------------------------------------------- conservativeness-ratio comparison
def plot_comparison(res, path):
    def lg(x):
        return np.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan
    a = [s.get('all', {}) if isinstance(s.get('all', {}), dict) else {} for s in res['sweep']]
    fig, ax = plt.subplots(figsize=PANEL)
    ax.plot(xs(res), [lg(d.get('conserv_cond', np.nan)) for d in a], 'o-', color='#d62728')
    ax.plot(xs(res), [lg(d.get('conserv_cond_uncen', np.nan)) for d in a], '^-.', color='#ff7f0e')
    ax.plot(xs(res), [lg(d.get('conserv_cond_savje', np.nan)) for d in a], 's--', color='#9467bd')
    ax.axhline(0.0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel(r'$\mathrm{CR}$')
    finish(ax)
    save(fig, path)


# ---------------------------------------------------------------- coverage (three strategies)
def plot_coverage(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), sharey=True)
    for ax, (m, (_, _, title)) in zip(axes, STRATEGY.items()):
        cu = [s.get(m, {}).get('coverage_uncond', np.nan) for s in res['sweep']]
        cc = [s.get(m, {}).get('coverage_cond', np.nan) for s in res['sweep']]
        ax.plot(xs(res), cu, 'o-', color='#1f77b4', label='Unconditional CI')
        ax.plot(xs(res), cc, 's-', color='#d62728', label='Conditional CI')
        ax.axhline(1 - res['config']['ci_alpha'], color='gray', ls='--', lw=0.8, label=r'Nominal (95\%)')
        ax.set_title(title)
        ax.set_ylim(0.8, 1.05)
        finish(ax)
    axes[0].set_ylabel('Coverage probability')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False)
    save(fig, path)


# ---------------------------------------------------------------- alternative inference
VAR_STYLE = {'ours': ('#d62728', 'o'), 'bernoulli': ('#1f77b4', 's'), 'plugin': ('#2ca02c', '^')}
CI_STYLE = {'ours': ('#d62728', 'o'), 'bernoulli': ('#1f77b4', 's'), 'dqo': ('#9467bd', 'D'),
            'plugin': ('#2ca02c', '^')}
ALT = (3.1, 2.2)            # inches: subfigure of width 0.48\textwidth


def plot_alt_variance(res, path):
    fig, ax = plt.subplots(figsize=ALT)
    for k, (c, mk) in VAR_STYLE.items():
        ax.plot(xs(res), [s['conserv_ratio'][k] for s in res['sweep']], f'{mk}-', color=c)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{E}[\hat v_n] / \mathrm{Var}(\hat\tau \mid M_n \le a)$')
    finish(ax)
    save(fig, path)


def plot_alt_cilength(res, path):
    fig, ax = plt.subplots(figsize=ALT)
    for k, (c, mk) in CI_STYLE.items():
        ax.plot(xs(res), [s['ci_ratio_to_bern'][k] for s in res['sweep']], f'{mk}-', color=c)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_ylabel(r'$\widehat L / \widehat L_{\mathrm{Bern}}$')
    finish(ax)
    save(fig, path)


# ---------------------------------------------------------------- critical value q_{1-alpha/2}(R^2; p, a)
def simulate_L(p, a, N, rng):
    g = rng.standard_normal((N, p))
    u1 = g[:, 0] / np.sqrt((g ** 2).sum(axis=1))
    U = rng.uniform(0.0, stats.chi2.cdf(a, p), size=N)
    return np.sqrt(stats.chi2.ppf(U, p)) * u1


def q_curve(p, pa, alpha, R2, N=1_000_000, seed=11):
    """Same construction and seed as rerand_quantiles.ipynb (bottom-right panel of quantile_curves.pdf)."""
    rng = np.random.default_rng(seed)
    a = stats.chi2.ppf(pa, p)
    vp = stats.chi2.cdf(a, p + 2) / stats.chi2.cdf(a, p)
    eps0 = rng.standard_normal(N)
    L = simulate_L(p, a, N, rng)
    return np.array([np.quantile(np.sqrt(1 - r) * eps0 + np.sqrt(r) * L, 1 - alpha / 2)
                     / np.sqrt(1 - (1 - vp) * r) for r in R2])


def plot_critical_value(path, pa=0.05, alpha=0.05):
    R2 = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    for p in (2, 5, 10, 20):
        ax.plot(R2, q_curve(p, pa, alpha, R2), label=f'$p={p}$')
    ax.axhline(stats.norm.ppf(1 - alpha / 2), ls='--', color='k', lw=0.8, label=r'$z_{1-\alpha/2}$')
    ax.set_xlabel(r'$R^2$')
    ax.set_ylabel(r'$q_{1-\alpha/2}(R^2;p,a)$')
    ax.grid(True)
    ax.legend(frameon=True, loc='lower left')
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vr-dir', default='results_vr_data')
    ap.add_argument('--cv-dir', default='results_cv_data')
    ap.add_argument('--altinf', default='results_altinf/altinf_results.json')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--only', nargs='*', default=None, help='restrict to these output files (relative to --out)')
    args = ap.parse_args()
    style()
    want = set(args.only) if args.only else None
    def go(rel, fn, *a):
        if want is None or rel in want:
            fn(*a, os.path.join(args.out, rel)); print('wrote', rel)

    for key, f in sorted(latest(args.vr_dir, 'vr').items()):
        go(f'variance_reduction/{key}.pdf', plot_vr, load(f))
    for key, f in sorted(latest(args.cv_dir, 'cv').items()):
        res = load(f)
        go(f'conservative_variance/{key}_ratio.pdf', plot_ratio, res)
        go(f'conservative_variance/{key}_comparison.pdf', plot_comparison, res)
        go(f'conservative_variance/{key}_coverage.pdf', plot_coverage, res)
    alt = load(args.altinf)
    for t in (1, 5):
        go(f'alternative_inference/altinf_type{t}_variance.pdf', plot_alt_variance, alt[f'type{t}'])
        go(f'alternative_inference/altinf_type{t}_cilength.pdf', plot_alt_cilength, alt[f'type{t}'])
    go('critical_value.pdf', plot_critical_value)


if __name__ == '__main__':
    main()
