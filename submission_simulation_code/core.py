import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import chi2, norm
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json, os
from datetime import datetime
from joblib import Parallel, delayed


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return super().default(obj)


def save_results(results: dict, save_dir: str = "results", prefix: str = "") -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    otype = results['config']['outcome_type']
    filename = f"{prefix}type{otype}_{ts}.json"
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Results saved to: {path}")
    return path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _neighbor_prop(Z, adj, X_pre=None):

    deg = np.maximum(adj.sum(1), 1)
    return (adj @ Z) / deg


def _neighbor_count(Z, adj, X_pre=None):

    return adj @ Z


def _has_treated(Z, adj, X_pre=None):

    return ((adj @ Z) > 0).astype(float)


def _neighbor_prop_sq(Z, adj, X_pre=None):

    return _neighbor_prop(Z, adj) ** 2


def _second_order_prop(Z, adj, X_pre=None):

    adj2 = adj @ adj
    np.fill_diagonal(adj2, 0)
    deg2 = np.maximum(adj2.sum(1), 1)
    return (adj2 @ Z) / deg2


def _neighbor_prop_partial(Z, adj, X_pre=None):

    n = len(Z)
    result = np.zeros(n)
    np.random.seed(2026)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) > 0:
            n_select = max(1, len(neighbors) // 2)
            selected = np.random.choice(neighbors, n_select, replace=False)
            result[i] = Z[selected].mean()
    return result


def _neighbor_weighted_X(Z, adj, X_pre):

    if X_pre is None:
        raise ValueError("X_pre is required for neighbor_weighted_X")
    deg = np.maximum(adj.sum(1), 1).reshape(-1, 1)
    return (adj @ (Z.reshape(-1, 1) * X_pre)) / deg


def _neighbor_weighted_X_squared(Z, adj, X_pre):

    if X_pre is None:
        raise ValueError("X_pre is required")
    deg = np.maximum(adj.sum(1), 1).reshape(-1, 1)
    return (adj @ (Z.reshape(-1, 1) * X_pre ** 2)) / deg


def _second_order_neighbor_weighted_X(Z, adj, X_pre):

    if X_pre is None:
        raise ValueError("X_pre is required")
    adj2 = adj @ adj
    np.fill_diagonal(adj2, 0)
    deg2 = np.maximum(adj2.sum(1), 1).reshape(-1, 1)
    return (adj2 @ (Z.reshape(-1, 1) * X_pre)) / deg2


def _second_order_neighbor_count(Z, adj, X_pre=None):
    adj2 = adj @ adj
    np.fill_diagonal(adj2, 0)
    return adj2 @ Z


def _ego_neighbor_interaction(Z, adj, X_pre=None):
    return Z * _neighbor_prop(Z, adj)


COVARIATE_FUNCS = {
    'neighbor_prop': _neighbor_prop,
    'neighbor_count': _neighbor_count,
    'has_treated_neighbor': _has_treated,
    'neighbor_prop_squared': _neighbor_prop_sq,
    'second_order_neighbor_prop': _second_order_prop,
    'neighbor_prop_partial': _neighbor_prop_partial,
    'neighbor_weighted_X': _neighbor_weighted_X,
    'neighbor_weighted_X_squared': _neighbor_weighted_X_squared,
    'second_order_neighbor_weighted_X': _second_order_neighbor_weighted_X,
    'second_order_neighbor_count': _second_order_neighbor_count,
    'ego_neighbor_interaction': _ego_neighbor_interaction,
}


def get_model_description(outcome_type: int) -> str:

    descriptions = {
        1: r"Type 1 (linear): dyn $= \beta_3 \bar{Z}_{N_i}$",
        2: r"Type 2 (linear): dyn $= \beta_3 \bar{Z}_{N_i} + \beta_4^\top \overline{ZX}_{N_i}$",
        3: r"Type 3 (linear): dyn $= \beta_3^\top \overline{ZX}_{N_i}$",
        4: r"Type 4 (linear): dyn $= \beta_3 \bar{Z}_{N_i} + \beta_4^\top \overline{ZX}_{N_i^{(2)}}$",
        5: r"Type 5 (nonlin, sum$\to$exp): dyn $= \beta_3 \exp(\bar{Z}_{N_i})$",
        6: r"Type 6 (nonlin, exp$\to$sum): dyn $= \beta_3 \sum_{j\in N_i} \exp(Z_j/|N_i|)$",
        7: r"Type 7 (nonlin, sum$\to$exp): dyn $= \beta_3 \exp(\bar{Z}_{N_i}) + \beta_4^\top \exp(\overline{ZX}_{N_i})$",
        8: r"Type 8 (nonlin, exp$\to$sum): dyn $= \beta_3 \sum_j \exp(Z_j/|N_i|) + \beta_4^\top \sum_j \exp(Z_j X_j/|N_i|)$",
    }
    if outcome_type not in descriptions:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")
    return descriptions[outcome_type]


def _outcome_coefs(outcome_type: int, cfg, rng) -> Dict:


    def rand_vec():
        return rng.uniform(0.5, 2.0, cfg.n_pre) * rng.choice([-1, 1], cfg.n_pre)

    if outcome_type == 1:
        return {'beta_3': 5.0}
    elif outcome_type == 2:
        return {'beta_3': 5.0, 'beta_4': rand_vec()}
    elif outcome_type == 3:
        return {'beta_3': rand_vec()}
    elif outcome_type == 4:
        return {'beta_3': 5.0, 'beta_4': rand_vec()}
    elif outcome_type == 5:
        return {'beta_3': 5.0}
    elif outcome_type == 6:
        return {'beta_3': 5.0}
    elif outcome_type == 7:
        return {'beta_3': 5.0, 'beta_4': rand_vec()}
    elif outcome_type == 8:
        return {'beta_3': 5.0, 'beta_4': rand_vec()}
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")


def _static_block(outcome_type: int, X_pre: np.ndarray, beta_xpre: np.ndarray) -> np.ndarray:


    if outcome_type in [1, 2, 3, 4]:
        return X_pre @ beta_xpre
    elif outcome_type in [5, 6, 7, 8]:
        return np.exp(X_pre @ beta_xpre)
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")


def _dynamic_block(outcome_type: int, Z: np.ndarray, adj: np.ndarray,
                   X_pre: np.ndarray, coefs: Dict) -> np.ndarray:


    n = Z.shape[0]
    deg = np.maximum(adj.sum(1), 1)

    if outcome_type == 1:

        neighbor_prop = (adj @ Z) / deg
        return coefs['beta_3'] * neighbor_prop

    elif outcome_type == 2:

        neighbor_prop = (adj @ Z) / deg
        neighbor_wX = (adj @ (Z[:, None] * X_pre)) / deg[:, None]
        return coefs['beta_3'] * neighbor_prop + neighbor_wX @ coefs['beta_4']

    elif outcome_type == 3:

        neighbor_wX = (adj @ (Z[:, None] * X_pre)) / deg[:, None]
        return neighbor_wX @ coefs['beta_3']

    elif outcome_type == 4:

        neighbor_prop = (adj @ Z) / deg
        adj2 = adj @ adj
        np.fill_diagonal(adj2, 0)
        deg2 = np.maximum(adj2.sum(1), 1)
        so_neighbor_wX = (adj2 @ (Z[:, None] * X_pre)) / deg2[:, None]
        return coefs['beta_3'] * neighbor_prop + so_neighbor_wX @ coefs['beta_4']

    elif outcome_type == 5:

        neighbor_prop = (adj @ Z) / deg
        return coefs['beta_3'] * np.exp(neighbor_prop)

    elif outcome_type == 6:


        untreated_count = adj @ (1 - Z)
        treated_count = adj @ Z
        result = untreated_count + treated_count * np.exp(1.0 / deg)
        return coefs['beta_3'] * result

    elif outcome_type == 7:

        neighbor_prop = (adj @ Z) / deg
        neighbor_wX = (adj @ (Z[:, None] * X_pre)) / deg[:, None]
        term1 = coefs['beta_3'] * np.exp(neighbor_prop)
        term2 = np.exp(neighbor_wX) @ coefs['beta_4']
        return term1 + term2

    elif outcome_type == 8:

        untreated_count = adj @ (1 - Z)
        treated_count = adj @ Z
        scalar_sum = untreated_count + treated_count * np.exp(1.0 / deg)

        p = X_pre.shape[1]
        vector_sum = np.tile(untreated_count.reshape(-1, 1), (1, p)).astype(float)
        treated_idx = np.where(Z == 1)[0]
        if len(treated_idx) > 0:
            for i in range(n):
                nbrs = np.where(adj[i] > 0)[0]
                tn = np.intersect1d(nbrs, treated_idx, assume_unique=False)
                if len(tn) > 0:
                    vector_sum[i] += np.sum(np.exp(X_pre[tn] / deg[i]), axis=0)
        return coefs['beta_3'] * scalar_sum + vector_sum @ coefs['beta_4']

    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")


@dataclass
class Config:
    n_units: int = 2000


    graph_type: str = 'ba'
    graph_m: int = 5
    graph_k: int = 15
    graph_p: float = 0.1
    outcome_type: int = 1
    beta0: float = 0.5
    beta_treatment: float = 2.0
    noise_std: float = 1.0
    n_pre: int = 6
    n_hidden: int = 1
    rerand_covs: List[str] = field(default_factory=lambda: ['neighbor_prop', 'neighbor_weighted_X'])
    acceptance_rate: float = 0.05
    n_accepted: int = 2000
    n_mc: int = 5000
    p_treat: float = 0.5
    ci_alpha: float = 0.05
    seed: int = 2026


class Simulator:


    def __init__(self, cfg: Config, beta_pre=1.0, beta_trt=1.0,
                 beta_xpre=None, beta_hidden=None,
                 outcome_coefs=None,
                 adj=None, X_pre=None, X_hidden=None):
        self.cfg = cfg
        self.n = cfg.n_units
        self.beta_pre = beta_pre
        self.beta_trt = beta_trt

        self.adj = adj if adj is not None else self._make_graph()
        self.X_pre = X_pre if X_pre is not None else \
            np.random.randn(self.n, cfg.n_pre) * 0.316
        self.X_hidden = X_hidden if X_hidden is not None else \
            np.random.randn(self.n, cfg.n_hidden) * 0.316

        rng = np.random.default_rng(cfg.seed)
        self.beta_xpre = beta_xpre if beta_xpre is not None else \
            rng.uniform(0.5, 2.0, cfg.n_pre) * rng.choice([-1, 1], cfg.n_pre)
        self.beta_hidden = beta_hidden if beta_hidden is not None else \
            rng.uniform(0.5, 2.0, cfg.n_hidden) * rng.choice([-1, 1], cfg.n_hidden)


        self.outcome_coefs = outcome_coefs if outcome_coefs is not None else {}

        self.p_pre = cfg.n_pre
        Z_test = np.random.binomial(1, 0.5, self.n)
        dyn_test = self._get_covs(Z_test, cfg.rerand_covs)
        self.p_dynamic = dyn_test.shape[1] if dyn_test.ndim == 2 else 1

    def _make_graph(self):

        gt = self.cfg.graph_type
        if gt == 'ba':
            return self._make_ba_graph()
        if gt == 'ws':
            return self._make_ws_graph()
        raise ValueError(f"Unknown graph_type: {gt!r}. Expected 'ba' or 'ws'.")

    def _make_ba_graph(self):


        n, m = self.n, self.cfg.graph_m
        if m < 1:
            raise ValueError(f"graph_m must be >= 1 (got {m})")
        if m >= n:
            raise ValueError(f"graph_m ({m}) must be < n ({n})")
        adj = np.zeros((n, n))

        for i in range(m + 1):
            for j in range(i):
                adj[i, j] = adj[j, i] = 1
        deg = adj.sum(1)
        for new in range(m + 1, n):
            probs = deg[:new] / deg[:new].sum()
            targets = np.random.choice(new, size=m, replace=False, p=probs)
            for t in targets:
                adj[new, t] = adj[t, new] = 1
                deg[t] += 1
            deg[new] = m
        return adj

    def _make_ws_graph(self):

        n, k, p = self.n, self.cfg.graph_k, self.cfg.graph_p
        k = min(k - k % 2, n - 1)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(1, k // 2 + 1):
                adj[i, (i + j) % n] = adj[(i + j) % n, i] = 1
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if np.random.random() < p and adj[i, (i + j) % n]:
                    cands = [x for x in range(n) if x != i and not adj[i, x]]
                    if cands:
                        new = np.random.choice(cands)
                        adj[i, (i + j) % n] = adj[(i + j) % n, i] = 0
                        adj[i, new] = adj[new, i] = 1
        return adj

    def _get_covs(self, Z, names):
        if not names:
            return np.zeros((self.n, 0))
        covs = []
        for name in names:
            result = COVARIATE_FUNCS[name](Z, self.adj, self.X_pre)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            covs.append(result)
        return np.hstack(covs) if covs else np.zeros((self.n, 0))

    def _hajek(self, Z, X):

        n1, n0 = Z.sum(), (1 - Z).sum()
        if n1 == 0 or n0 == 0:
            return np.full(X.shape[1] if X.ndim > 1 else 1, np.nan)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (Z @ X) / n1 - ((1 - Z) @ X) / n0

    def _compute_static_block(self):

        return _static_block(self.cfg.outcome_type, self.X_pre, self.beta_xpre)

    def _compute_dynamic_block(self, Z):

        return _dynamic_block(self.cfg.outcome_type, Z, self.adj,
                              self.X_pre, self.outcome_coefs)

    def _outcome(self, Z):

        Y = self.cfg.beta0 + self.cfg.beta_treatment * Z
        Y += self.beta_pre * self._compute_static_block()
        Y += self.X_hidden @ self.beta_hidden
        Y += self.beta_trt * self._compute_dynamic_block(Z)
        Y += np.random.randn(self.n) * self.cfg.noise_std
        return Y

    def estimate_block_stds(self, n_samples=2000):

        std_pre = np.std(self._compute_static_block())
        dynamic_stds = []
        for _ in range(n_samples):
            Z = np.random.binomial(1, self.cfg.p_treat, self.n)
            if 0 < Z.sum() < self.n:
                dynamic_stds.append(np.std(self._compute_dynamic_block(Z)))
        return std_pre, np.mean(dynamic_stds)


def create_shared_experiment(cfg: Config):


    np.random.seed(cfg.seed)
    ref_sim = Simulator(cfg, beta_pre=1.0, beta_trt=1.0)
    adj = ref_sim.adj
    X_pre = ref_sim.X_pre
    X_hidden = ref_sim.X_hidden
    beta_xpre = ref_sim.beta_xpre
    beta_hidden = ref_sim.beta_hidden


    rng = np.random.default_rng(cfg.seed)
    rng.uniform(0.5, 2.0, cfg.n_pre)
    rng.choice([-1, 1], cfg.n_pre)
    rng.uniform(0.5, 2.0, cfg.n_hidden)
    rng.choice([-1, 1], cfg.n_hidden)
    outcome_coefs = _outcome_coefs(cfg.outcome_type, cfg, rng)


    ref_sim.outcome_coefs = outcome_coefs
    std_pre, std_trt = ref_sim.estimate_block_stds(n_samples=2000)

    return {
        'adj': adj, 'X_pre': X_pre, 'X_hidden': X_hidden,
        'beta_xpre': beta_xpre, 'beta_hidden': beta_hidden,
        'outcome_coefs': outcome_coefs,
        'description': get_model_description(cfg.outcome_type),
        'std_pre': std_pre, 'std_trt': std_trt,
    }


def build_sim_kwargs(cfg, shared, beta_pre_val, beta_trt_val):

    return {
        'cfg': cfg, 'beta_pre': beta_pre_val, 'beta_trt': beta_trt_val,
        'beta_xpre': np.array(shared['beta_xpre']),
        'beta_hidden': np.array(shared['beta_hidden']),
        'outcome_coefs': shared['outcome_coefs'],
        'adj': shared['adj'],
        'X_pre': shared['X_pre'],
        'X_hidden': shared['X_hidden'],
    }


def compute_beta_from_ratio(target_delta, std_pre, std_trt, total_signal_std):

    r = 2.0 ** target_delta
    A = total_signal_std / np.sqrt(1 + r**2)
    B = total_signal_std * r / np.sqrt(1 + r**2)
    beta_pre_val = A / std_pre if std_pre > 0 else 1.0
    beta_trt_val = B / std_trt if std_trt > 0 else 1.0
    return beta_pre_val, beta_trt_val


def setup_plot_style():

    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 16, "axes.titlesize": 20,
    })
    try:
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        })
    except Exception:
        plt.rcParams["text.usetex"] = False
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('ggplot')
        except Exception:
            pass


STRATEGY_STYLES = {
    'static':  ('#1f77b4', 'o', 'Pre-Treatment'),
    'dynamic': ('#2ca02c', 's', 'Treatment-Dependent'),
    'all':     ('#d62728', '^', 'Joint'),
}


XLABEL_LOG_RATIO = r'$\log_2(\beta_{\mathrm{interf}} \cdot \mathrm{std}_{\mathrm{interf}} \,/\, \beta_{\mathrm{base}} \cdot \mathrm{std}_{\mathrm{base}})$'
