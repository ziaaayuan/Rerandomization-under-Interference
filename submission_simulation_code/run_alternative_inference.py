import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import asdict
from datetime import datetime
from joblib import Parallel, delayed

from core import (
    Config, Simulator, NumpyEncoder,
    create_shared_experiment, build_sim_kwargs, compute_beta_from_ratio,
    setup_plot_style, XLABEL_LOG_RATIO,
)

OUT_DIR = 'results_altinf'


N_UNITS = 500
N_MC = 2000
N_ACCEPTED = 300
ACCEPT_RATE = 0.05
TOTAL_SIGNAL_STD = 30.0
LOG_RATIOS = np.linspace(-4, 4, 9)
ALPHA = 0.05
K_MC = 10000
N_T = 21
RERAND_COVS = ['neighbor_prop']


def compute_local_dep_degrees(adj):

    adj_sp = csr_matrix(adj)
    adj2 = adj_sp @ adj_sp
    A_dep = ((adj_sp + adj2) > 0).astype(float)
    A_dep.setdiag(1)
    A_dep.eliminate_zeros()
    return np.asarray(A_dep.sum(axis=1)).flatten()


def compute_U_hat(Z, Y, X_rerand, d_local, p_treat, n):

    phi = np.hstack([Y.reshape(-1, 1), X_rerand])
    treated = (Z == 1)
    control = (Z == 0)
    phi_hat = phi.copy()
    if treated.sum() > 0:
        phi_hat[treated] -= phi[treated].mean(axis=0)
    if control.sum() > 0:
        phi_hat[control] -= phi[control].mean(axis=0)
    psi_sq = np.where(Z == 1, 1.0 / p_treat**2, 1.0 / (1 - p_treat)**2)
    scale = np.sqrt(d_local * psi_sq)
    weighted = phi_hat * scale[:, np.newaxis]
    return (1.0 / n**2) * (weighted.T @ weighted)


def _reg_psd(Mat, reg=1e-8):
    p = Mat.shape[0]
    eigs = np.linalg.eigvalsh(Mat)
    if eigs.min() < reg:
        Mat = Mat + (abs(eigs.min()) + reg) * np.eye(p)
    return Mat


def compute_v_hat_n(U_hat, Sigma_xx, v_p, reg=1e-8, n_bisect=60):

    U11 = U_hat[0, 0]
    U12 = U_hat[0, 1:]
    U22 = U_hat[1:, 1:]
    p = U22.shape[0]
    diff = _reg_psd(U22 - Sigma_xx, reg)
    Sxx = _reg_psd(Sigma_xx.copy(), reg)

    def _M(t):
        return Sxx + t * diff

    def _G(t):
        Mi = np.linalg.inv(_M(t))
        return float(U12 @ Mi @ (Sxx + (t ** 2) * diff) @ Mi @ U12)

    def _v_at(t):
        Mi = np.linalg.inv(_M(t))
        inner = (1.0 - v_p) * Sxx + (t ** 2) * diff
        return float(U11 - U12 @ Mi @ inner @ Mi @ U12)

    t_lo = 1.0 - v_p
    if _G(t_lo) <= U11:
        Mi = np.linalg.inv(_M(t_lo))
        return float(U11 - (1.0 - v_p) * (U12 @ Mi @ U12))
    lo, hi = t_lo, 1.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _G(mid) > U11:
            lo = mid
        else:
            hi = mid
    return _v_at(hi)


def sample_L_pa(p, a, K, rng, oversample=40):

    out = []
    while len(out) < K:
        D = rng.standard_normal((K * oversample, p))
        keep = (D ** 2).sum(axis=1) <= a
        out.extend(D[keep, 0].tolist())
    return np.asarray(out[:K])


def mixture_quantile(A, B, E0, La, q):

    A = max(float(A), 0.0)
    B = max(float(B), 0.0)
    return float(np.quantile(np.sqrt(A) * E0 + np.sqrt(B) * La, q))


def compute_qbar(p, a, v_p, E0, La, alpha):

    q_level = 1 - alpha / 2
    best = 0.0
    for R2 in np.linspace(0.0, 1.0, 41):
        omega = float(np.quantile(np.sqrt(1 - R2) * E0 + np.sqrt(R2) * La, q_level))
        denom = np.sqrt(max(1 - (1 - v_p) * R2, 1e-12))
        best = max(best, omega / denom)
    return best


def path_AB(U_hat, Sigma_xx, v_p, n_t=N_T, reg=1e-8):


    U11 = U_hat[0, 0]
    u = U_hat[0, 1:]
    diff = _reg_psd(U_hat[1:, 1:] - Sigma_xx, reg)
    Sxx = _reg_psd(Sigma_xx.copy(), reg)

    def _Mi(t):
        return np.linalg.inv(Sxx + t * diff)

    def _G(t):
        Mi = _Mi(t)
        return float(u @ Mi @ (Sxx + (t ** 2) * diff) @ Mi @ u)


    t_lo = 1.0 - v_p
    if _G(t_lo) <= U11:
        t_start = t_lo
    else:
        lo, hi = t_lo, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _G(mid) > U11:
                lo = mid
            else:
                hi = mid
        t_start = hi

    A_list, B_list = [], []
    for t in np.linspace(t_start, 1.0, n_t):
        Mi = _Mi(t)
        G = float(u @ Mi @ (Sxx + (t ** 2) * diff) @ Mi @ u)
        B = float(u @ Mi @ Sxx @ Mi @ u)
        A_list.append(max(U11 - G, 0.0))
        B_list.append(max(B, 0.0))
    return np.asarray(A_list), np.asarray(B_list)


def _run_single(cfg, shared, target_delta, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    std_pre, std_trt = shared['std_pre'], shared['std_trt']
    beta_pre_val, beta_trt_val = compute_beta_from_ratio(
        target_delta, std_pre, std_trt, TOTAL_SIGNAL_STD)
    actual_ratio = ((beta_trt_val * std_trt) / (beta_pre_val * std_pre)
                    if beta_pre_val * std_pre > 0 else float('inf'))
    actual_log_ratio = (np.log2(actual_ratio)
                        if actual_ratio > 0 and np.isfinite(actual_ratio) else target_delta)

    sim = Simulator(**build_sim_kwargs(cfg, shared, beta_pre_val, beta_trt_val))
    d_local = compute_local_dep_degrees(sim.adj)

    def X_joint(Z):
        return np.hstack([sim.X_pre, sim._get_covs(Z, cfg.rerand_covs)])


    tau_y_list, tau_x_list, sqrt_u_bern = [], [], []
    for _ in range(N_MC):
        Z = np.random.binomial(1, cfg.p_treat, sim.n)
        if not (0 < Z.sum() < sim.n):
            continue
        Y = sim._outcome(Z)
        tau_y_list.append(sim._hajek(Z, Y)[0])
        tau_x_list.append(sim._hajek(Z, X_joint(Z)))
        U_b = compute_U_hat(Z, Y, X_joint(Z), d_local, cfg.p_treat, sim.n)
        sqrt_u_bern.append(np.sqrt(max(float(U_b[0, 0]), 0.0)))

    tau_y_arr = np.array(tau_y_list)
    tau_x_arr = np.array(tau_x_list)
    Sigma_xx = np.cov(tau_x_arr, rowvar=False)
    if Sigma_xx.ndim == 0:
        Sigma_xx = np.array([[float(Sigma_xx)]])
    p_cov = tau_x_arr.shape[1]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx + 1e-8 * np.eye(p_cov))
    thresh = chi2.ppf(ACCEPT_RATE, p_cov)
    v_p = chi2.cdf(thresh, p_cov + 2) / chi2.cdf(thresh, p_cov)
    mean_sqrt_u_bern = float(np.mean(sqrt_u_bern))


    E0 = rng.standard_normal(K_MC)
    La = sample_L_pa(p_cov, thresh, K_MC, rng)
    qbar = compute_qbar(p_cov, thresh, v_p, E0, La, ALPHA)
    z = norm.ppf(1 - ALPHA / 2)
    q_level = 1 - ALPHA / 2


    v_ours, v_bern, v_plugin = [], [], []
    L_ours, L_bern, L_dqo, L_plugin = [], [], [], []
    tau_hat = []
    n_draw = 0
    while len(tau_hat) < N_ACCEPTED and n_draw < N_ACCEPTED * 400:
        n_draw += 1
        Z = np.random.binomial(1, cfg.p_treat, sim.n)
        if not (0 < Z.sum() < sim.n):
            continue
        tx = sim._hajek(Z, X_joint(Z))
        if tx @ Sigma_xx_inv @ tx > thresh:
            continue
        Y = sim._outcome(Z)
        ty = sim._hajek(Z, Y)[0]
        U_hat = compute_U_hat(Z, Y, X_joint(Z), d_local, cfg.p_treat, sim.n)
        U11 = float(U_hat[0, 0])
        U12 = U_hat[0, 1:]
        U22 = U_hat[1:, 1:]
        U22_inv = np.linalg.inv(_reg_psd(U22.copy()))
        explained = float(U12 @ U22_inv @ U12)


        v_o = max(compute_v_hat_n(U_hat, Sigma_xx, v_p), 0.0)
        v_b = U11
        v_pi = U11 * (1.0 + v_p) - explained


        l_o = qbar * np.sqrt(v_o)
        l_b = z * np.sqrt(max(U11, 0.0))

        A_path, B_path = path_AB(U_hat, Sigma_xx, v_p)
        mix = np.sqrt(A_path)[:, None] * E0[None, :] + np.sqrt(B_path)[:, None] * La[None, :]
        l_dqo = float(np.quantile(mix, q_level, axis=1).max())

        A_pi = max(U11 - explained, 0.0)
        l_pi = mixture_quantile(A_pi, U11, E0, La, q_level)

        tau_hat.append(ty)
        v_ours.append(v_o); v_bern.append(v_b); v_plugin.append(max(v_pi, 0.0))
        L_ours.append(l_o); L_bern.append(l_b); L_dqo.append(l_dqo); L_plugin.append(l_pi)

    var_cond = float(np.var(tau_hat)) if tau_hat else float('nan')

    def m(x):
        return float(np.mean(x)) if len(x) else float('nan')

    mean_L_bern = m(L_bern)
    out = {
        'target_log_ratio': float(target_delta),
        'actual_log_ratio': float(actual_log_ratio),
        'n_accepted': len(tau_hat),
        'p_cov': int(p_cov), 'v_p': float(v_p), 'qbar': float(qbar),
        'var_cond': var_cond,

        'var_mean': {'ours': m(v_ours), 'bernoulli': m(v_bern), 'plugin': m(v_plugin)},
        'conserv_ratio': {k: (m(v) / var_cond if var_cond > 0 else float('nan'))
                          for k, v in [('ours', v_ours), ('bernoulli', v_bern), ('plugin', v_plugin)]},

        'ci_mean': {'ours': m(L_ours), 'bernoulli': mean_L_bern,
                    'dqo': m(L_dqo), 'plugin': m(L_plugin)},
        'ci_ratio_to_bern': {k: (m(v) / mean_L_bern if mean_L_bern > 0 else float('nan'))
                             for k, v in [('ours', L_ours), ('bernoulli', L_bern),
                                          ('dqo', L_dqo), ('plugin', L_plugin)]},
    }
    return out


def run_type(cfg, n_jobs=-1):
    shared = create_shared_experiment(cfg)
    print(f"\n=== Outcome type {cfg.outcome_type} | covs={cfg.rerand_covs} | n={cfg.n_units} ===")
    print(f"block stds: pre={shared['std_pre']:.4f}, trt={shared['std_trt']:.4f}")
    sweep = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single)(cfg, shared, float(d), cfg.seed + 7000 + i)
        for i, d in enumerate(LOG_RATIOS))
    sweep.sort(key=lambda s: s['target_log_ratio'])
    for s in sweep:
        cr = s['conserv_ratio']; cir = s['ci_ratio_to_bern']
        print(f"  log2R={s['actual_log_ratio']:5.2f} | "
              f"var E[v]/Vc: ours={cr['ours']:.2f} bern={cr['bernoulli']:.2f} plugin={cr['plugin']:.2f} | "
              f"CI/Lbern: ours={cir['ours']:.2f} dqo={cir['dqo']:.2f} plugin={cir['plugin']:.2f}")
    return {'config': asdict(cfg),
            'model': {'outcome_type': cfg.outcome_type, 'description': shared['description']},
            'sweep': sweep, 'timestamp': datetime.now().isoformat()}


VAR_STYLE = {
    'ours':      ('#d62728', 'o', 'Ours (optimization)'),
    'bernoulli': ('#1f77b4', 's', 'Bernoulli'),
    'plugin':    ('#2ca02c', '^', r'Small-$a$ plug-in'),
}
CI_STYLE = {
    'ours':      ('#d62728', 'o', 'Ours (optimization)'),
    'bernoulli': ('#1f77b4', 's', 'Bernoulli Wald'),
    'dqo':       ('#9467bd', 'D', 'Direct quantile opt.'),
    'plugin':    ('#2ca02c', '^', 'Plug-in rerand.'),
}


def plot_variance(results, save_path):
    setup_plot_style()
    sweep = results['sweep']
    x = [s['actual_log_ratio'] for s in sweep]
    fig, ax = plt.subplots(figsize=(12, 9))
    for m_key, (c, mk, lab) in VAR_STYLE.items():
        y = [s['conserv_ratio'][m_key] for s in sweep]
        ax.plot(x, y, f'{mk}-', color=c, lw=3.0, ms=10, label=lab, alpha=0.9)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel(XLABEL_LOG_RATIO, labelpad=12)
    ax.set_ylabel(r'$\mathrm{E}[\hat v_n]\,/\,\mathrm{Var}(\hat\tau \mid M_n \le a)$', labelpad=12)
    ax.set_xlim(-4, 4)
    ax.legend(frameon=True, framealpha=0.95, loc='best', fontsize=16)
    ax.grid(True, ls='--', alpha=0.6)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.95)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cilength(results, save_path):
    setup_plot_style()
    sweep = results['sweep']
    x = [s['actual_log_ratio'] for s in sweep]
    fig, ax = plt.subplots(figsize=(12, 9))
    for m_key, (c, mk, lab) in CI_STYLE.items():
        y = [s['ci_ratio_to_bern'][m_key] for s in sweep]
        ax.plot(x, y, f'{mk}-', color=c, lw=3.0, ms=10, label=lab, alpha=0.9)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel(XLABEL_LOG_RATIO, labelpad=12)
    ax.set_ylabel(r'$\widehat L \,/\, \widehat L_{\mathrm{Bern}}$', labelpad=12)
    ax.set_xlim(-4, 4)
    ax.legend(frameon=True, framealpha=0.95, loc='best', fontsize=16)
    ax.grid(True, ls='--', alpha=0.6)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.95)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for otype in (1, 5):
        cfg = Config(
            n_units=N_UNITS, outcome_type=otype, n_pre=6, n_hidden=1,
            rerand_covs=RERAND_COVS, acceptance_rate=ACCEPT_RATE,
            n_accepted=N_ACCEPTED, n_mc=N_MC, p_treat=0.5,
            ci_alpha=ALPHA, graph_type='ba', graph_m=5,
        )
        res = run_type(cfg, n_jobs=-1)
        all_results[f'type{otype}'] = res
        plot_variance(res, os.path.join(OUT_DIR, f'altinf_type{otype}_variance.pdf'))
        plot_cilength(res, os.path.join(OUT_DIR, f'altinf_type{otype}_cilength.pdf'))
        print(f"  saved figures for type{otype}")
    with open(os.path.join(OUT_DIR, 'altinf_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == '__main__':
    main()
