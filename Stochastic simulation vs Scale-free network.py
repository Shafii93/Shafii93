import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Set to True to use the literal sign "+2*gamma_h*[I_hI_h]" from the supplied
# equations; False uses the standard "-2*gamma_h".
LITERAL_II_HH_SIGN = True


# ===================================================================
# 1. NETWORK GENERATION  (POWER-LAW CONFIGURATION MODEL)  -- NEW
# ===================================================================
#
# Replaces the bimodal generator. Degrees are drawn i.i.d. from a
# TRUNCATED DISCRETE POWER LAW
#       P(k) = k^{-gamma} / Z ,   k in {k_min, ..., k_max},
# separately for humans (gamma_h, k_min_h, k_max_h) and rodents
# (gamma_r, k_min_r, k_max_r). Edges are then formed configuration-model
# style: a prescribed fraction of stubs is used for human-rodent (cross)
# edges, and the remaining stubs are paired within-species. Self-loops
# are avoided via bounded rejection; multi-edges are allowed (standard
# for the configuration model).

def sample_power_law_degrees(N, gamma, k_min, k_max):
    """Draw N degrees from P(k) ~ k^{-gamma}, k_min <= k <= k_max,
    and enforce an even stub total (configuration-model requirement)."""
    k_vals = np.arange(k_min, k_max + 1)
    probs = k_vals.astype(float) ** (-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k_vals, size=N, p=probs).astype(int)

    if degrees.sum() % 2 != 0:
        # flip parity by bumping one non-saturated node up (or down) by 1
        idx = np.random.randint(0, N)
        tries = 0
        while degrees[idx] >= k_max and tries < 100:
            idx = np.random.randint(0, N)
            tries += 1
        if degrees[idx] < k_max:
            degrees[idx] += 1
        elif degrees[idx] > k_min:
            degrees[idx] -= 1
    return degrees.tolist()


def generate_power_law_network(N_h, N_r,
                               gamma_h, k_min_h, k_max_h,
                               gamma_r, k_min_r, k_max_r,
                               p_hh, p_hr, p_rr):
    """Two-species power-law configuration network.

    p_hr : controls the number of human-rodent (cross) edges,
           n_cross = p_hr * (human stubs) / 2  (same convention as before)
    p_hh : target number of human-human edges = p_hh * (human stubs) / 2
    p_rr : target number of rodent-rodent edges = p_rr * (rodent stubs) / 2
    Targets are capped by stub availability, exactly as in the bimodal code.
    """
    h_degrees = sample_power_law_degrees(N_h, gamma_h, k_min_h, k_max_h)
    r_degrees = sample_power_law_degrees(N_r, gamma_r, k_min_r, k_max_r)

    h_stubs = []
    for i, deg in enumerate(h_degrees):
        h_stubs.extend([('H', i)] * deg)
    r_stubs = []
    for i, deg in enumerate(r_degrees):
        r_stubs.extend([('R', i)] * deg)

    total_h_stubs = len(h_stubs)
    total_r_stubs = len(r_stubs)

    n_cross = int(p_hr * total_h_stubs / 2)
    n_cross = min(n_cross, total_r_stubs // 2, total_h_stubs)

    n_hh = int(p_hh * total_h_stubs / 2)
    n_rr = int(p_rr * total_r_stubs / 2)
    n_hh = min(n_hh, (total_h_stubs - n_cross) // 2)
    n_rr = min(n_rr, (total_r_stubs - n_cross) // 2)

    N = N_h + N_r
    adj = [[] for _ in range(N)]
    species = ['H'] * N_h + ['R'] * N_r

    def h_global(i): return i
    def r_global(j): return N_h + j

    # ---- cross edges (h-r): cannot self-loop by construction ----
    used_h = set()
    used_r = set()
    cross_edges = []
    for _ in range(n_cross):
        hi = random.randint(0, total_h_stubs - 1)
        while hi in used_h:
            hi = random.randint(0, total_h_stubs - 1)
        used_h.add(hi)
        ri = random.randint(0, total_r_stubs - 1)
        while ri in used_r:
            ri = random.randint(0, total_r_stubs - 1)
        used_r.add(ri)
        cross_edges.append((h_global(h_stubs[hi][1]), r_global(r_stubs[ri][1])))

    # ---- within-species edges, with bounded self-loop rejection ----
    def pair_stubs(stubs, available, n_edges, to_global):
        edges = []
        for _ in range(n_edges):
            if len(available) < 2:
                break
            i1 = random.choice(available); available.remove(i1)
            i2 = random.choice(available); available.remove(i2)
            a, b = stubs[i1][1], stubs[i2][1]
            attempts = 0
            while a == b and attempts < 20:
                available.append(i2)
                i2 = random.choice(available); available.remove(i2)
                b = stubs[i2][1]
                attempts += 1
            if a == b:                      # give up on this edge
                available.extend([i1, i2])
                continue
            edges.append((to_global(a), to_global(b)))
        return edges

    h_avail = [i for i in range(total_h_stubs) if i not in used_h]
    r_avail = [i for i in range(total_r_stubs) if i not in used_r]
    hh_edges = pair_stubs(h_stubs, h_avail, n_hh, h_global)
    rr_edges = pair_stubs(r_stubs, r_avail, n_rr, r_global)

    for u, v in cross_edges + hh_edges + rr_edges:
        adj[u].append(v)
        adj[v].append(u)

    return adj, species, N_h, N_r, h_degrees, r_degrees


def network_edge_summary(adj, species, N_h, N_r):
    """Count actual hh / hr / rr edges and average degree by type."""
    edges_hh = edges_hr = edges_rr = 0
    deg_h_total = deg_r_total = 0
    for u in range(N_h + N_r):
        deg_u = len(adj[u])
        if species[u] == 'H':
            deg_h_total += deg_u
        else:
            deg_r_total += deg_u
        for v in adj[u]:
            if v > u:
                if species[u] == 'H' and species[v] == 'H':
                    edges_hh += 1
                elif species[u] == 'R' and species[v] == 'R':
                    edges_rr += 1
                else:
                    edges_hr += 1
    kh = deg_h_total / N_h if N_h > 0 else 0
    kr = deg_r_total / N_r if N_r > 0 else 0
    return edges_hh, edges_hr, edges_rr, kh, kr


def degree_summary(adj, species, N_h, N_r):
    """Print realised degree statistics for each species (heterogeneity
    check: <k^2>/<k> >> <k> indicates a heavy-tailed network)."""
    for name, idx_range in (("human", range(0, N_h)),
                            ("rodent", range(N_h, N_h + N_r))):
        d = np.array([len(adj[i]) for i in idx_range], dtype=float)
        print(f"  {name:<7}: min={int(d.min()):>3}  max={int(d.max()):>3}  "
              f"mean={d.mean():6.2f}  var={d.var():8.2f}  "
              f"<k^2>/<k>={np.mean(d**2)/max(d.mean(),1e-9):6.2f}")


def plot_degree_distributions(adj, species, N_h, N_r, gamma_h, gamma_r):
    """Log-log plot of the realised degree distributions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, idx_range, g, color in (
            ("Human", range(0, N_h), gamma_h, 'tab:blue'),
            ("Rodent", range(N_h, N_h + N_r), gamma_r, 'tab:orange')):
        d = np.array([len(adj[i]) for i in idx_range])
        vals, counts = np.unique(d[d > 0], return_counts=True)
        ax.loglog(vals, counts / counts.sum(), 'o', ms=4, alpha=0.7,
                  color=color, label=f'{name} (target gamma={g})')
    ax.set_xlabel('degree k')
    ax.set_ylabel('P(k)')
    ax.set_title('Realised degree distributions (log-log)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('degree_distributions.png', dpi=150)


# ===================================================================
# 2. STOCHASTIC SIMULATION
# ===================================================================

def simulate_two_species(adj, species, N_h, N_r,
                         alpha_rh, alpha_h, tau_h, q, gamma_h,
                         tau_rr, gamma_r,
                         init_frac_h, init_frac_r, t_max):
    N = N_h + N_r
    human_nodes = [i for i in range(N) if species[i] == 'H']
    rodent_nodes = [i for i in range(N) if species[i] == 'R']
    h_index = {node: i for i, node in enumerate(human_nodes)}
    r_index = {node: i for i, node in enumerate(rodent_nodes)}

    h_status = [0] * N_h
    r_status = [0] * N_r

    for idx in random.sample(range(N_h), int(init_frac_h * N_h)):
        h_status[idx] = 2
    for idx in random.sample(range(N_r), int(init_frac_r * N_r)):
        r_status[idx] = 1

    h_I_list = [i for i in range(N_h) if h_status[i] == 2]
    h_E_list = [i for i in range(N_h) if h_status[i] == 1]
    h_Q_list = [i for i in range(N_h) if h_status[i] == 3]
    r_I_list = [i for i in range(N_r) if r_status[i] == 1]

    t = 0.0

    def snap():
        return (t, len(h_I_list), len(h_E_list), len(h_Q_list),
                sum(1 for s in h_status if s == 4),
                len(r_I_list),
                sum(1 for s in r_status if s == 2))

    history = [snap()]

    while t < t_max:
        rr_SI_edges = []
        for r in r_I_list:
            for nb in adj[rodent_nodes[r]]:
                if species[nb] == 'R':
                    nb_r = r_index[nb]
                    if r_status[nb_r] == 0:
                        rr_SI_edges.append((r, nb_r))
        T_rr = tau_rr * len(rr_SI_edges)

        R_r = gamma_r * len(r_I_list)

        zoonotic_edges = []
        for r in r_I_list:
            for nb in adj[rodent_nodes[r]]:
                if species[nb] == 'H':
                    h_idx = h_index[nb]
                    if h_status[h_idx] == 0:
                        zoonotic_edges.append((h_idx, r))
        Z = alpha_rh * len(zoonotic_edges)

        hh_SI_edges = []
        for h in h_I_list:
            for nb in adj[human_nodes[h]]:
                if species[nb] == 'H':
                    nb_h = h_index[nb]
                    if h_status[nb_h] == 0:
                        hh_SI_edges.append((h, nb_h))
        H = alpha_h * len(hh_SI_edges)

        E_prog = tau_h * len(h_E_list)
        E_quar = q * len(h_E_list)
        I_rec = gamma_h * len(h_I_list)
        Q_rec = gamma_h * len(h_Q_list)

        R_total = T_rr + R_r + Z + H + E_prog + E_quar + I_rec + Q_rec
        if R_total == 0:
            break

        t += -math.log(random.random()) / R_total
        u = random.random()
        cumulative = 0.0

        cumulative += T_rr / R_total
        if u < cumulative:
            r_sus = random.choice(rr_SI_edges)[1]
            r_status[r_sus] = 1
            r_I_list.append(r_sus)
            history.append(snap()); continue

        cumulative += R_r / R_total
        if u < cumulative:
            r_rec = random.choice(r_I_list)
            r_status[r_rec] = 2
            r_I_list.remove(r_rec)
            history.append(snap()); continue

        cumulative += Z / R_total
        if u < cumulative:
            h_sus = random.choice(zoonotic_edges)[0]
            h_status[h_sus] = 1
            h_E_list.append(h_sus)
            history.append(snap()); continue

        cumulative += H / R_total
        if u < cumulative:
            h_sus = random.choice(hh_SI_edges)[1]
            h_status[h_sus] = 1
            h_E_list.append(h_sus)
            history.append(snap()); continue

        cumulative += E_prog / R_total
        if u < cumulative:
            h = random.choice(h_E_list)
            h_status[h] = 2
            h_E_list.remove(h); h_I_list.append(h)
            history.append(snap()); continue

        cumulative += E_quar / R_total
        if u < cumulative:
            h = random.choice(h_E_list)
            h_status[h] = 3
            h_E_list.remove(h); h_Q_list.append(h)
            history.append(snap()); continue

        cumulative += I_rec / R_total
        if u < cumulative:
            h = random.choice(h_I_list)
            h_status[h] = 4
            h_I_list.remove(h)
            history.append(snap()); continue

        cumulative += Q_rec / R_total
        if u < cumulative:
            h = random.choice(h_Q_list)
            h_status[h] = 4
            h_Q_list.remove(h)
            history.append(snap()); continue

    return history


# ===================================================================
# 3a. MEAN-FIELD (SINGLES) ODE MODEL
# ===================================================================

def mean_field_singles_odes(y, t, params):
    (N_h, N_r, kh, kr,
     alpha_rh, alpha_h, tau_h, q, gamma_h, tau_rr, gamma_r,
     delta_hh, delta_rh) = params

    S_h, E_h, I_h, Q_h, R_h, S_r, I_r, R_r = y

    pi_Ih = I_h / N_h
    pi_Ir = I_r / N_r

    SI_hr_mf = delta_rh * kh * S_h * pi_Ir
    SI_hh_mf = delta_hh * kh * S_h * pi_Ih
    SI_rr_mf = kr * S_r * pi_Ir

    dS_h = -alpha_rh * SI_hr_mf - alpha_h * SI_hh_mf
    dE_h = alpha_rh * SI_hr_mf + alpha_h * SI_hh_mf - (tau_h + q) * E_h
    dI_h = tau_h * E_h - gamma_h * I_h
    dQ_h = q * E_h - gamma_h * Q_h
    dR_h = gamma_h * (I_h + Q_h)

    dS_r = -tau_rr * SI_rr_mf
    dI_r = tau_rr * SI_rr_mf - gamma_r * I_r
    dR_r = gamma_r * I_r

    return [dS_h, dE_h, dI_h, dQ_h, dR_h, dS_r, dI_r, dR_r]


# ===================================================================
# 3b. PAIRWISE CLUSTERED-APPROXIMATION ODE MODEL
# ===================================================================
#
# NOTE (power-law network): the ODEs are implemented in the
# single-degree-class form (k = l = kh, x = y = kr) using the REALISED
# mean degrees. On a heterogeneous power-law network this is a cruder
# closure than on the old regular/bimodal graph -- the RMSE tables
# below quantify exactly how much accuracy is lost.

def pairwise_clustered_odes(y, t, params):
    (N_h, N_r, kh, kr,
     alpha_rh, alpha_h, tau_h, q, gamma_h, tau_rr, gamma_r,
     phi, delta_hh, delta_rh, delta_rr) = params

    (SS_hh, SE_hh, SI_hh, EE_hh, EI_hh, II_hh,
     SS_hr, SI_hr,
     SS_rr, SI_rr, II_rr,
     S_h, E_h, I_h, Q_h, R_h,
     S_r, I_r, R_r) = y

    I_h_safe = max(I_h, 1e-9)
    I_r_safe = max(I_r, 1e-9)

    pi_Ih = I_h / N_h
    pi_Ir = I_r / N_r

    n_hh_Nh_over_m = N_h / kh
    n_rr_Nr_over_z = N_r / kr
    n_rh_over_kx = (kh * N_h + kr * N_r) / (kh * kr)

    def bracket_hh(deg):
        return ((1 - phi) * delta_hh * deg * pi_Ih
                + phi * n_hh_Nh_over_m * (delta_hh ** 2) * (pi_Ih ** 2) * deg / I_h_safe)

    def bracket_rh_h(deg):
        return ((1 - phi) * delta_rh * deg * pi_Ir
                + phi * n_rh_over_kx * delta_rh * delta_rr * (pi_Ir ** 2) * deg / I_r_safe)

    def bracket_rh_r(deg):
        return ((1 - phi) * delta_rr * deg * pi_Ir
                + phi * n_rh_over_kx * delta_rh * delta_rr * (pi_Ir ** 2) * deg / I_r_safe)

    def bracket_rr(deg):
        return ((1 - phi) * delta_rr * deg * pi_Ir
                + phi * n_rr_Nr_over_z * (delta_rr ** 2) * (pi_Ir ** 2) * deg / I_r_safe)

    B_hh = bracket_hh(kh)
    B_rh_h = bracket_rh_h(kh)
    B_rh_r = bracket_rh_r(kr)
    B_rr = bracket_rr(kr)

    # ---- Human-human pairs ----
    dSS_hh = -2 * alpha_rh * (kh - 1) * B_rh_h * SS_hh

    dSE_hh = (alpha_rh * (kh - 1) / kh * SS_hh * B_rh_h
              + alpha_h * (kh - 1) / kh * SS_hh * B_hh
              - alpha_h * (kh - 1) / kh * SE_hh * B_hh
              - tau_h * SE_hh - q * SE_hh)

    dSI_hh = tau_h * SE_hh - (alpha_h + gamma_h) * SI_hh

    dEE_hh = 2 * alpha_h * (kh - 1) / kh * SE_hh * B_hh - 2 * tau_h * EE_hh

    dEI_hh = tau_h * (EE_hh - EI_hh) + alpha_h * SI_hh

    ii_sign = +1.0 if LITERAL_II_HH_SIGN else -1.0
    dII_hh = 2 * tau_h * EI_hh + ii_sign * 2 * gamma_h * II_hh

    # ---- Human-rodent pairs ----
    dSS_hr = -(alpha_rh + tau_rr) * ((kr - 1) / kr * SS_hr * B_rh_r
                                     + (kh - 1) / kh * SS_hr * B_rh_h)

    dSI_hr = (alpha_rh + tau_rr) * (kh - 1) / kh * SS_hr * B_rh_h - gamma_r * SI_hr

    # ---- Rodent-rodent pairs ----
    dSS_rr = -2 * tau_rr * (kr - 1) / kr * SS_rr * B_rr

    dSI_rr = tau_rr * (kr - 1) / kr * SS_rr * B_rr - tau_rr * SI_rr - gamma_r * SI_rr

    dII_rr = 2 * (tau_rr - gamma_r) * SI_rr

    # ---- Singles ----
    dS_h = -alpha_rh * SI_hr - alpha_h * SI_hh
    dE_h = alpha_rh * SI_hr + alpha_h * SI_hh - (tau_h + q) * E_h
    dI_h = tau_h * E_h - gamma_h * I_h
    dQ_h = q * E_h - gamma_h * Q_h
    dR_h = gamma_h * (I_h + Q_h)

    dS_r = -tau_rr * SI_rr
    dI_r = tau_rr * SI_rr - gamma_r * I_r
    dR_r = gamma_r * I_r

    return [dSS_hh, dSE_hh, dSI_hh, dEE_hh, dEI_hh, dII_hh,
            dSS_hr, dSI_hr,
            dSS_rr, dSI_rr, dII_rr,
            dS_h, dE_h, dI_h, dQ_h, dR_h,
            dS_r, dI_r, dR_r]


def build_initial_pair_state(N_h, N_r, edges_hh, edges_hr, edges_rr,
                             init_frac_h, init_frac_r):
    s_h, i_h = 1 - init_frac_h, init_frac_h
    s_r, i_r = 1 - init_frac_r, init_frac_r

    return [edges_hh * s_h ** 2, 0.0, edges_hh * 2 * s_h * i_h, 0.0, 0.0, edges_hh * i_h ** 2,
            edges_hr * s_h * s_r, edges_hr * s_h * i_r,
            edges_rr * s_r ** 2, edges_rr * 2 * s_r * i_r, edges_rr * i_r ** 2,
            N_h * s_h, 0.0, N_h * i_h, 0.0, 0.0,
            N_r * s_r, N_r * i_r, 0.0]


# ===================================================================
# 4a. GOODNESS-OF-FIT METRICS
# ===================================================================

def compute_metrics(sim, model):
    sim = np.asarray(sim, dtype=float)
    model = np.asarray(model, dtype=float)
    err = model - sim

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    max_err = float(np.max(np.abs(err)))

    mean_sim = float(np.mean(sim))
    rng_sim = float(np.max(sim) - np.min(sim))

    nrmse_mean = 100.0 * rmse / mean_sim if mean_sim > 1e-12 else np.nan
    nrmse_range = 100.0 * rmse / rng_sim if rng_sim > 1e-12 else np.nan

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((sim - mean_sim) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {'RMSE': rmse, 'NRMSE_mean': nrmse_mean, 'NRMSE_range': nrmse_range,
            'MAE': mae, 'Bias': bias, 'MaxErr': max_err, 'R2': r2}


def peak_metrics(time_grid, sim, model):
    sim = np.asarray(sim, dtype=float)
    model = np.asarray(model, dtype=float)
    i_s, i_m = int(np.argmax(sim)), int(np.argmax(model))
    return {'peak_sim': float(sim[i_s]), 'peak_model': float(model[i_m]),
            'peak_err': float(model[i_m] - sim[i_s]),
            'peak_pct_err': 100.0 * (model[i_m] - sim[i_s]) / sim[i_s] if sim[i_s] > 1e-12 else np.nan,
            't_peak_sim': float(time_grid[i_s]), 't_peak_model': float(time_grid[i_m]),
            't_peak_err': float(time_grid[i_m] - time_grid[i_s])}


def rmse_vs_replicates(replicates, model):
    model = np.asarray(model, dtype=float)
    vals = [float(np.sqrt(np.mean((model - np.asarray(rep, dtype=float)) ** 2)))
            for rep in replicates]
    return float(np.mean(vals)), float(np.std(vals))


def print_rmse_tables(time_grid, series_dict, replicate_dict=None):
    n = len(time_grid)
    line = "=" * 104

    print("\n" + line)
    print(f" TABLE 1 -- RMSE OF ODE MODELS vs STOCHASTIC SIMULATION MEAN "
          f"(n = {n} time points, t in [0, {time_grid[-1]:.2f}])")
    print(line)
    print(f"{'Compartment':<14}{'Model':<22}"
          f"{'RMSE':>10}{'NRMSE%(mean)':>14}{'NRMSE%(range)':>15}"
          f"{'MAE':>10}{'Bias':>10}{'R^2':>9}")
    print("-" * 104)

    summary = {}
    for label, (sim, pw, mf) in series_dict.items():
        m_pw = compute_metrics(sim, pw)
        m_mf = compute_metrics(sim, mf)
        summary[label] = (m_pw, m_mf)
        for name, m in (("Pairwise clustered", m_pw), ("Mean-field", m_mf)):
            print(f"{label if name.startswith('Pairwise') else '':<14}{name:<22}"
                  f"{m['RMSE']:>10.3f}{m['NRMSE_mean']:>14.2f}{m['NRMSE_range']:>15.2f}"
                  f"{m['MAE']:>10.3f}{m['Bias']:>10.3f}{m['R2']:>9.4f}")
        better = "Pairwise" if m_pw['RMSE'] < m_mf['RMSE'] else "Mean-field"
        impr = 100.0 * (m_mf['RMSE'] - m_pw['RMSE']) / m_mf['RMSE'] if m_mf['RMSE'] > 0 else np.nan
        print(f"{'':<14}{'--> lower RMSE:':<22}{better:>10}"
              f"   (pairwise RMSE is {impr:+.1f}% relative to mean-field)")
        print("-" * 104)

    print("\n" + line)
    print(" TABLE 2 -- PEAK HEIGHT AND PEAK TIMING ERRORS")
    print(line)
    print(f"{'Compartment':<14}{'Model':<22}{'Peak(sim)':>12}{'Peak(model)':>13}"
          f"{'PeakErr':>10}{'PeakErr%':>11}{'t_peak(sim)':>13}{'t_peak(mod)':>13}")
    print("-" * 104)
    for label, (sim, pw, mf) in series_dict.items():
        for name, mod in (("Pairwise clustered", pw), ("Mean-field", mf)):
            p = peak_metrics(time_grid, sim, mod)
            print(f"{label if name.startswith('Pairwise') else '':<14}{name:<22}"
                  f"{p['peak_sim']:>12.2f}{p['peak_model']:>13.2f}"
                  f"{p['peak_err']:>10.2f}{p['peak_pct_err']:>11.2f}"
                  f"{p['t_peak_sim']:>13.3f}{p['t_peak_model']:>13.3f}")
        print("-" * 104)

    if replicate_dict is not None:
        print("\n" + line)
        print(" TABLE 3 -- RMSE AGAINST INDIVIDUAL STOCHASTIC REPLICATES (mean +/- sd over runs)")
        print(line)
        print(f"{'Compartment':<14}{'Pairwise clustered':>28}{'Mean-field':>28}"
              f"{'Sim-mean vs replicates':>30}")
        print("-" * 104)
        for label, (sim, pw, mf) in series_dict.items():
            reps = replicate_dict.get(label)
            if reps is None:
                continue
            a_m, a_s = rmse_vs_replicates(reps, pw)
            b_m, b_s = rmse_vs_replicates(reps, mf)
            c_m, c_s = rmse_vs_replicates(reps, sim)
            print(f"{label:<14}{a_m:>18.3f} +/-{a_s:>7.3f}"
                  f"{b_m:>18.3f} +/-{b_s:>7.3f}{c_m:>20.3f} +/-{c_s:>7.3f}")
        print("-" * 104)
        print(" (The last column is the irreducible error floor: the RMSE of the")
        print("  simulation mean itself against individual runs.)")

    tot_pw = sum(m_pw['RMSE'] for m_pw, _ in summary.values())
    tot_mf = sum(m_mf['RMSE'] for _, m_mf in summary.values())
    wins_pw = sum(1 for m_pw, m_mf in summary.values() if m_pw['RMSE'] < m_mf['RMSE'])
    print("\n" + line)
    print(" OVERALL")
    print(line)
    print(f"  Summed RMSE across compartments : pairwise = {tot_pw:.3f}   mean-field = {tot_mf:.3f}")
    print(f"  Compartments where pairwise wins: {wins_pw} / {len(summary)}")
    print(f"  Best overall model              : "
          f"{'Pairwise clustered ODE' if tot_pw < tot_mf else 'Mean-field ODE'}")
    print(line + "\n")

    return summary


# ===================================================================
# 4b. RUN SIMULATIONS AND PLOT
# ===================================================================

def run_and_plot(N_h=500, N_r=500,
                 gamma_h=2.3, k_min_h=2, k_max_h=60,
                 gamma_r=2.3, k_min_r=3, k_max_r=100,
                 p_hh=0.5, p_hr=0.5, p_rr=0.5,
                 alpha_rh=0.1, alpha_h=0.05, tau_h=0.2, q=0.1, gamma_h_ep=0.1,
                 tau_rr=0.15, gamma_r_ep=0.1,
                 init_frac_h=0.01, init_frac_r=0.02, t_max=100,
                 num_sims=20,
                 phi=0.0, delta_hh=1.0, delta_rh=1.0, delta_rr=1.0):

    adj, species, N_h_actual, N_r_actual, h_deg_t, r_deg_t = generate_power_law_network(
        N_h, N_r,
        gamma_h, k_min_h, k_max_h,
        gamma_r, k_min_r, k_max_r,
        p_hh, p_hr, p_rr)

    edges_hh, edges_hr, edges_rr, kh, kr = network_edge_summary(
        adj, species, N_h_actual, N_r_actual)

    print(f"Network generated (power-law configuration model): "
          f"{len(species)} nodes, {sum(len(a) for a in adj)//2} edges")
    print(f"  edges_hh={edges_hh}, edges_hr={edges_hr}, edges_rr={edges_rr}")
    print(f"  avg human degree kh={kh:.3f}, avg rodent degree kr={kr:.3f}")
    print("  Realised degree statistics:")
    degree_summary(adj, species, N_h_actual, N_r_actual)
    plot_degree_distributions(adj, species, N_h_actual, N_r_actual, gamma_h, gamma_r)

    print("Running stochastic simulations...")
    histories = []
    for i in range(num_sims):
        if i % 5 == 0:
            print(f"  Simulation {i+1}/{num_sims}")
        histories.append(simulate_two_species(
            adj, species, N_h_actual, N_r_actual,
            alpha_rh, alpha_h, tau_h, q, gamma_h_ep, tau_rr, gamma_r_ep,
            init_frac_h, init_frac_r, t_max))

    print("Interpolating results...")
    max_time = min(h[-1][0] for h in histories if len(h) > 0)
    time_grid = np.linspace(0, max_time, 200)

    I_h_interp, I_r_interp, E_h_interp = [], [], []
    Q_h_interp, R_h_interp, R_r_interp = [], [], []

    for hist in histories:
        if len(hist) == 0:
            continue
        times = np.array([h[0] for h in hist])
        I_h_interp.append(np.interp(time_grid, times, [h[1] for h in hist]))
        E_h_interp.append(np.interp(time_grid, times, [h[2] for h in hist]))
        Q_h_interp.append(np.interp(time_grid, times, [h[3] for h in hist]))
        R_h_interp.append(np.interp(time_grid, times, [h[4] for h in hist]))
        I_r_interp.append(np.interp(time_grid, times, [h[5] for h in hist]))
        R_r_interp.append(np.interp(time_grid, times, [h[6] for h in hist]))

    I_h_mean, I_h_std = np.mean(I_h_interp, axis=0), np.std(I_h_interp, axis=0)
    I_r_mean, I_r_std = np.mean(I_r_interp, axis=0), np.std(I_r_interp, axis=0)
    E_h_mean = np.mean(E_h_interp, axis=0)
    Q_h_mean = np.mean(Q_h_interp, axis=0)
    R_h_mean = np.mean(R_h_interp, axis=0)
    R_r_mean = np.mean(R_r_interp, axis=0)

    print("Solving mean-field (singles) ODE...")
    y0_mf = [N_h_actual * (1 - init_frac_h), 0.0, N_h_actual * init_frac_h, 0.0, 0.0,
             N_r_actual * (1 - init_frac_r), N_r_actual * init_frac_r, 0.0]
    params_mf = (N_h_actual, N_r_actual, kh, kr,
                 alpha_rh, alpha_h, tau_h, q, gamma_h_ep, tau_rr, gamma_r_ep,
                 delta_hh, delta_rh)
    mf_sol = odeint(mean_field_singles_odes, y0_mf, time_grid, args=(params_mf,))
    E_h_mf, I_h_mf, Q_h_mf, R_h_mf = mf_sol[:, 1], mf_sol[:, 2], mf_sol[:, 3], mf_sol[:, 4]
    I_r_mf, R_r_mf = mf_sol[:, 6], mf_sol[:, 7]

    print("Solving pairwise clustered-approximation ODE...")
    y0 = build_initial_pair_state(N_h_actual, N_r_actual,
                                  edges_hh, edges_hr, edges_rr,
                                  init_frac_h, init_frac_r)
    params = (N_h_actual, N_r_actual, kh, kr,
              alpha_rh, alpha_h, tau_h, q, gamma_h_ep, tau_rr, gamma_r_ep,
              phi, delta_hh, delta_rh, delta_rr)
    ode_sol = odeint(pairwise_clustered_odes, y0, time_grid, args=(params,))

    E_h_ode, I_h_ode, Q_h_ode, R_h_ode = ode_sol[:, 12], ode_sol[:, 13], ode_sol[:, 14], ode_sol[:, 15]
    I_r_ode, R_r_ode = ode_sol[:, 17], ode_sol[:, 18]

    # ---------------- RMSE / goodness-of-fit ----------------
    series_dict = {
        'I_h':     (I_h_mean, I_h_ode, I_h_mf),
        'E_h':     (E_h_mean, E_h_ode, E_h_mf),
        'Q_h':     (Q_h_mean, Q_h_ode, Q_h_mf),
        'R_h':     (R_h_mean, R_h_ode, R_h_mf),
        'I_r':     (I_r_mean, I_r_ode, I_r_mf),
        'R_r':     (R_r_mean, R_r_ode, R_r_mf),
        'I_total': (I_h_mean + I_r_mean, I_h_ode + I_r_ode, I_h_mf + I_r_mf),
    }
    replicate_dict = {
        'I_h': I_h_interp, 'E_h': E_h_interp, 'Q_h': Q_h_interp,
        'R_h': R_h_interp, 'I_r': I_r_interp, 'R_r': R_r_interp,
        'I_total': [a + b for a, b in zip(I_h_interp, I_r_interp)],
    }
    metrics = print_rmse_tables(time_grid, series_dict, replicate_dict)

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def annotate(ax, label):
        m_pw, m_mf = metrics[label]
        ax.text(0.98, 0.97,
                f"RMSE  pairwise = {m_pw['RMSE']:.2f}\nRMSE  mean-field = {m_mf['RMSE']:.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.75))

    ax = axes[0, 0]
    ax.plot(time_grid, I_h_mean, 'b-', label='Stochastic mean')
    ax.fill_between(time_grid, I_h_mean - I_h_std, I_h_mean + I_h_std, alpha=0.2, color='b', label='±1 std')
    ax.plot(time_grid, I_h_ode, 'r--', label='Pairwise clustered ODE')
    ax.plot(time_grid, I_h_mf, 'k:', label='Mean-field ODE')
    ax.set_xlabel('Time'); ax.set_ylabel('Infectious Humans')
    ax.set_title('Human Infectious (I_h)'); ax.legend(); ax.grid(True, alpha=0.3)
    annotate(ax, 'I_h')

    ax = axes[0, 1]
    ax.plot(time_grid, E_h_mean, 'g-', label='E_h (stochastic)')
    ax.plot(time_grid, Q_h_mean, 'm-', label='Q_h (stochastic)')
    ax.plot(time_grid, E_h_ode, 'g--', label='E_h (pairwise ODE)')
    ax.plot(time_grid, Q_h_ode, 'm--', label='Q_h (pairwise ODE)')
    ax.plot(time_grid, E_h_mf, 'g:', label='E_h (mean-field ODE)')
    ax.plot(time_grid, Q_h_mf, 'm:', label='Q_h (mean-field ODE)')
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    ax.set_title('Human Exposed (E_h) and Quarantined (Q_h)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    annotate(ax, 'E_h')

    ax = axes[1, 0]
    ax.plot(time_grid, I_r_mean, 'b-', label='Stochastic mean')
    ax.fill_between(time_grid, I_r_mean - I_r_std, I_r_mean + I_r_std, alpha=0.2, color='b', label='±1 std')
    ax.plot(time_grid, I_r_ode, 'r--', label='Pairwise clustered ODE')
    ax.plot(time_grid, I_r_mf, 'k:', label='Mean-field ODE')
    ax.set_xlabel('Time'); ax.set_ylabel('Infectious Rodents')
    ax.set_title('Rodent Infectious (I_r)'); ax.legend(); ax.grid(True, alpha=0.3)
    annotate(ax, 'I_r')

    ax = axes[1, 1]
    I_total_mean = I_h_mean + I_r_mean
    I_total_std = np.std(np.array(I_h_interp) + np.array(I_r_interp), axis=0)
    ax.plot(time_grid, I_total_mean, 'b-', label='Stochastic mean')
    ax.fill_between(time_grid, I_total_mean - I_total_std, I_total_mean + I_total_std,
                    alpha=0.2, color='b', label='±1 std')
    ax.plot(time_grid, I_h_ode + I_r_ode, 'r--', label='Pairwise clustered ODE')
    ax.plot(time_grid, I_h_mf + I_r_mf, 'k:', label='Mean-field ODE')
    ax.set_xlabel('Time'); ax.set_ylabel('Total Infected')
    ax.set_title('Total Infected (I_h + I_r)'); ax.legend(); ax.grid(True, alpha=0.3)
    annotate(ax, 'I_total')

    plt.tight_layout()
    plt.savefig('two_species_epidemic_comparison.png', dpi=150)

    # ---------------- RMSE bar chart ----------------
    labels = list(series_dict.keys())
    rmse_pw = [metrics[l][0]['RMSE'] for l in labels]
    rmse_mf = [metrics[l][1]['RMSE'] for l in labels]
    xpos = np.arange(len(labels))
    w = 0.38

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(xpos - w / 2, rmse_pw, w, label='Pairwise clustered ODE', color='firebrick')
    ax2.bar(xpos + w / 2, rmse_mf, w, label='Mean-field ODE', color='dimgray')
    ax2.set_xticks(xpos); ax2.set_xticklabels(labels)
    ax2.set_ylabel('RMSE vs stochastic mean')
    ax2.set_title('Model accuracy by compartment (lower is better)')
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)
    for x, v in zip(xpos - w / 2, rmse_pw):
        ax2.text(x, v, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    for x, v in zip(xpos + w / 2, rmse_mf):
        ax2.text(x, v, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('rmse_comparison.png', dpi=150)
    plt.show()

    return histories, time_grid, metrics


# ===================================================================
# 5. RUN THE COMPLETE ANALYSIS
# ===================================================================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    N_h, N_r = 1000, 400

    # ---- Power-law degree parameters (per species) ----
    # P(k) ~ k^{-gamma}, k in [k_min, k_max].
    # These choices give mean degrees of roughly 6 (humans) and 10 (rodents),
    # comparable to the old d_h=6 / d_r=10 settings; the realised means are
    # printed at runtime.
    gamma_h, k_min_h, k_max_h = 2.3, 2, 100
    gamma_r, k_min_r, k_max_r = 2.3, 3, 50

    # Edge-type budget (fractions of stubs, as before)
    p_hh, p_hr, p_rr = 1.0, 0.9, 1.0

    alpha_rh = 1.0
    alpha_h = 1.0
    tau_h = 1.0
    q = 0.5
    gamma_h_ep = 0.7
    tau_rr = 1.0
    gamma_r_ep = 0.7

    init_frac_h, init_frac_r = 0.01, 0.02
    t_max, num_sims = 100, 50

    phi = 0.4
    delta_hh = 1.0
    delta_rh = 0.9
    delta_rr = 1.0

    print("=" * 60)
    print("TWO-SPECIES EPIDEMIC ON POWER-LAW CONFIGURATION NETWORK")
    print("=" * 60)
    print(f"Humans: {N_h}, Rodents: {N_r}")
    print(f"Human degrees : gamma={gamma_h}, k in [{k_min_h}, {k_max_h}]")
    print(f"Rodent degrees: gamma={gamma_r}, k in [{k_min_r}, {k_max_r}]")
    print(f"Rates: alpha_rh={alpha_rh}, alpha_h={alpha_h}, tau_h={tau_h}, q={q}, gamma_h={gamma_h_ep}")
    print(f"Rodent rates: tau_rr={tau_rr}, gamma_r={gamma_r_ep}")
    print(f"Number of simulations: {num_sims}")
    print("=" * 60)

    histories, time_grid, metrics = run_and_plot(
        N_h, N_r,
        gamma_h, k_min_h, k_max_h,
        gamma_r, k_min_r, k_max_r,
        p_hh, p_hr, p_rr,
        alpha_rh, alpha_h, tau_h, q, gamma_h_ep,
        tau_rr, gamma_r_ep,
        init_frac_h, init_frac_r, t_max,
        num_sims,
        phi=phi, delta_hh=delta_hh, delta_rh=delta_rh, delta_rr=delta_rr)

    print("Done! Plots saved as 'degree_distributions.png',")
    print("'two_species_epidemic_comparison.png' and 'rmse_comparison.png'")
