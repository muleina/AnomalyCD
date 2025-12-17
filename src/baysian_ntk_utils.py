
# =============================================================================
# AnomalyCD: Scalable Temporal Anomaly Causality Discovery in Large Systems
# =============================================================================
# Utilities provide commonly shared functions and libraries for GCM and BN modeling
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2022
# =============================================================================
import numpy as np
import pandas as pd
import itertools
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pyvis.network import Network
import networkx as nx
from castle import algorithms as castle_algorithms
from castle.metrics import MetricsDAG as castle_MetricsDAG
os.environ['CASTLE_BACKEND'] = 'pytorch'
import pgmpy
from pgmpy.estimators import PC as pgmpy_PC
from pgmpy.estimators import HillClimbSearch as pgmpy_HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from IPython.display import display, HTML
from pgmpy.inference import VariableElimination, CausalInference
from pgmpy.base import DAG
from cdt.causality.graph import CGNN, PC, GES, GIES, LiNGAM, CAM, GS, IAMB, MMPC, SAM, CCDr
from cdt.metrics import (precision_recall, SID, SHD)
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI as tigramite_PCMCI
from tigramite.lpcmci import LPCMCI as tigramite_LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gsquared import Gsquared
import utilities as util

sns.set()
graph_theme = {
                "dark": {"bgcolor": "black", "font_color": "white"},
                "white": {"bgcolor": "white", "font_color": "black"}
             }

# Data discretization
def boundary_str(start, end, tier):
    return f'{tier}: {start:+0,.4f} to {end:+0,.4f}'

def relabel(v, boundaries, return_num_cat=False, return_int=False):

    if return_num_cat:
        for i, boundary in enumerate(boundaries):
            if (v >= boundary[0]) and (v <= boundary[1]):
                # return f'{boundary[1]:+0,.4f}'
                if return_int:
                    return i
                else:
                    return np.round(boundary[1], 4)

    else:
        tiers = 'ABCDEFGHIJK'
        for i, boundary in enumerate(boundaries):
            if (v >= boundary[0]) and (v <= boundary[1]):
                return boundary_str(boundary[0], boundary[1], tier=tiers[i])
    return np.nan

def get_boundaries(tiers):
    prev_tier = tiers[0]
    boundaries = [(prev_tier[0], prev_tier[prev_tier.shape[0] - 1])]
    for index, tier in enumerate(tiers):
        if index is not 0:
            boundaries.append(
                (prev_tier[prev_tier.shape[0] - 1], tier[tier.shape[0] - 1]))
            prev_tier = tier
    return boundaries

def convert_values_to_discrete(df, TIERS_NUM=3, return_num_cat=False, return_int=False):
    nodes = df.columns.tolist()
    new_columns = {}
    for i, content in enumerate(df.items()):
        (label, series) = content
        # label
        values = np.sort(
            np.array([x for x in series.tolist() if not np.isnan(x)], dtype=float))
        if values.shape[0] < TIERS_NUM:
            print(f'Error: there are not enough data for label {label}')
            break
        # np.nanmin(values), np.nanmax(values)
        boundaries = get_boundaries(tiers=np.array_split(values, TIERS_NUM))
        # boundaries
        new_columns[label] = [relabel(value, boundaries, return_num_cat=return_num_cat, return_int=return_int)
                              for value in series.tolist()]

    df_binned = pd.DataFrame(data=new_columns)
    df_binned.columns = nodes
    # df_binned.index = range(years["min"], years["max"] + 1)
    print(df.head())
    print(df_binned.head())
    return df_binned

def build_link_assumptions(link_assumptions_absent_link_means_no_knowledge,
                           n_component_time_series,
                           tau_max,
                           tau_min=0, num_vars_org=None):

    print("build_init_link_assumptions...")

    if num_vars_org is None:
        num_vars_org = n_component_time_series

    out = {j: {(i, -tau_i): ("o?>" if (tau_i > 0) or (i >= num_vars_org) else("<?o" if (tau_i > 0) or (j >= num_vars_org) else "o?o"))
               for i in range(n_component_time_series) for tau_i in range(tau_min, tau_max+1)
               if (tau_i > 0 or i != j)} for j in range(n_component_time_series)}

    # print(out)
    # util.print_dict(out)
    for j, links_j in link_assumptions_absent_link_means_no_knowledge.items():
        # print(j, links_j)
        for (i, lag_i), link_ij in links_j.items():
            # print(i, lag_i, link_ij)
            if link_ij == "":
                del out[j][(i, lag_i)]
            else:
                out[j][(i, lag_i)] = link_ij

    # util.print_dict(out)
    # link at tlag zero links must be symmetric: adjustment
    visited_links = []
    for j, links_j in out.items():
        for (i, lag_i), link_ij in links_j.items():
            if (lag_i != 0) or ((i, j) in visited_links):
                continue

            # if (j, 0) in out[(i)].keys():
            if link_ij == "<?o":
                # print(f"updating symmetric edge at {i} o?> for ({j}, 0)")
                out[i][(j, 0)] = "o?>"
            elif link_ij == "o?>":
                # print(f"updating symmetric edge at {i} <?o for ({j}, 0)")
                out[i][(j, 0)] = "<?o"
            elif link_ij == "o?o":
                # print(f"updating symmetric edge at {i} o?o for ({j}, 0)")
                out[i][(j, 0)] = "o?o"
            elif link_ij == "-->":
                # print(f"updating symmetric edge at {i} o?o for ({j}, 0)")
                out[i][(j, 0)] = "<--"
            else:
                print(
                    f"undefined link {j}, {i} orientation {link_ij} symmetric adjustment at lag=0!")
                raise f"undefined link orientation {link_ij} symmetric adjustment at lag=0!"

            visited_links.append((j, i))

    return out

def remove_self_lag_causality(num_vars, tau_max, tau_min=0, isblacklist_mode=False, blacklist_tags=[], vars_names=[], isunrolled=False):

    print("removeing self_lag links...")
    if isblacklist_mode:
        blacklist_vars_idx = []
        for blk_tag in blacklist_tags:
            blacklist_vars_idx.extend([vars_names.index(y) for y in list(
                filter(lambda x: blk_tag in x, vars_names))])
        print("blacklist_vars_idx: ", blacklist_vars_idx)

    link_assumptions_absent_link_means_no_knowledge = {}
    # print(num_vars)
    num_vars_org = None
    if not isunrolled:
        for i in range(num_vars):
            # i is destination node and link_i ((j, -t)) is source links from jth node at tlag t
            link_i = {(i, -t): "" for t in range(1, tau_max+1)}
            link_assumptions_absent_link_means_no_knowledge[i] = link_i
    else:
        ts_org_vars = list(filter(lambda x: "__tlag" not in x, vars_names))
        # print(ts_org_vars)
        num_vars_org = len(ts_org_vars)
        for i in range(num_vars):
            vars_names_splited = vars_names[i].split("__tlag_")
            var_name = vars_names_splited[0]
            tlag = 0 if vars_names[i] in ts_org_vars else int(
                vars_names_splited[-1])

            if tlag == 0:
                link_i = {(j, 0): "" for j, col in enumerate(vars_names) if (
                    i != j) and (var_name in col)}  # all nodes except self lags
            else:
                link_i = {(j, 0): "" for j, col in enumerate(vars_names) if (i != j) and (
                    (var_name not in col) or (col in ts_org_vars))}  # nodes with tlag in +ve and tlag have no source node

                link_i_self_lag_j = {(j, 0): "-->" for j, col in enumerate(vars_names) if (i != j) and (col not in ts_org_vars) and (
                    var_name in col) and (int(col.split("__tlag_")[-1]) > tlag)}  # nodes with tlag in +ve and tlag have no source node
                link_i.update(link_i_self_lag_j)

            link_assumptions_absent_link_means_no_knowledge[i] = link_i

    if isblacklist_mode:
        for i in blacklist_vars_idx:
            link_i_blacklist = {(b, -t): "" for t in range(0, tau_max+1)
                                for b in blacklist_vars_idx if i != b}
            link_assumptions_absent_link_means_no_knowledge[i].update(
                link_i_blacklist)

    # util.print_dict(link_assumptions_absent_link_means_no_knowledge)
    link_assumptions_wo_self_lag = build_link_assumptions(
        link_assumptions_absent_link_means_no_knowledge, num_vars, tau_max, tau_min=tau_min, num_vars_org=num_vars_org)

    return link_assumptions_wo_self_lag

def get_sparse_link_assumptions(df_binary, tau_max, link_assumptions, thr=0.00):
    print("removing sparse links...")

    if tau_max <=0 :
        return link_assumptions
    
    df = df_binary.diff(axis=0).fillna(0)
    # df[(df<0) | (df==0)] = np.nan
    df[ df < 0] = 0 # keep only rising edges of df_binary 0 -> 1
    df_taumax_w = (df.rolling(window=tau_max, min_periods=1, center=False).sum()>0)
    robust_link_assumptions = util.copy.deepcopy(link_assumptions)
    for i in range(df_taumax_w.shape[1]):
        for j in range(i+1, df_taumax_w.shape[1]):
            causal_col = df_taumax_w.columns[i]
            effect_var = df_taumax_w.columns[j]
            overlap_bits = df_taumax_w[effect_var].mask(~df_taumax_w[causal_col]).sum()
            if overlap_bits > 0:
                overlap_score_ij = overlap_bits/df_taumax_w[causal_col].sum()
                overlap_score_ji = overlap_bits/df_taumax_w[effect_var].sum()             
                if overlap_score_ij <= thr:
                    print(f"{causal_col} -> {effect_var}-number of overlapped flags: {overlap_bits}, overlap score w.r.t {causal_col}: {overlap_score_ij:0.3f}")
                    [robust_link_assumptions[j].pop((i, -t)) for t in range(0, tau_max+1)]
                if overlap_score_ji <= thr:
                    print(f"{effect_var} -> {causal_col}-number of overlapped flags: {overlap_bits}, overlap score w.r.t {effect_var}: {overlap_score_ji:0.3f}")
                    [robust_link_assumptions[i].pop((j, -t)) for t in range(0, tau_max+1)]  
            else:
                print(f"No overlap: removing links between {causal_col} <-> {effect_var}")
                [robust_link_assumptions[i].pop((j, -t)) for t in range(0, tau_max+1)]
                [robust_link_assumptions[j].pop((i, -t)) for t in range(0, tau_max+1)]  

    N = df_binary.shape[1]
    # n_default_links = N**2*(tau_max+1)
    n_default_links = N*((tau_max + 1) * (N - 1) + tau_max) # updated in 2025, test in cludes vj(t=0) = sum all vi neq j (t=-tmax...0) [means N-1 times] + vj(t=-tmax...-1)
    
    # n_default_links
    n_links = sum([len(v) for _, v in link_assumptions.items()]) 
    n_link_reduced = n_default_links - n_links   # == n*(tau_max+1)
    print(f"n_default_links: {n_default_links}, n_after_self_lag_link_removal: {n_links}, n_link_reduced: {n_link_reduced} ({100*float(n_link_reduced/n_default_links):0.2f}%)")
    n_sparse_links = sum([len(v) for _, v in robust_link_assumptions.items()])
    n_link_reduced = n_default_links - n_sparse_links
    print(f"n_default_links: {n_default_links}, n_after_sparse_link_removal: {n_sparse_links}, n_link_reduced: {n_link_reduced} ({100*float(n_link_reduced/n_default_links):0.2f}%)")

    return robust_link_assumptions

def lagged_src_name(s, t):
    if t == 0:
        return s
    return "{}__tlag_{:02d}".format(s, abs(t))

def prepare_edges_pcmci(results, var_names, tau_max, tau_min=0, significance_level=0.01, isagg=True, weight_sign='both', weight_thr=0.001):
    print("prepare_edges_pcmci...")
    mesh_edge_vars = list(itertools.product(
        var_names, var_names, (-1*np.arange(tau_min, tau_max+1, 1)).tolist()))
    mesh_edge_vars
    edges_df = pd.DataFrame(mesh_edge_vars, columns=["src", "dst", "t"])
    edges_df["pval"] = results['p_matrix'].reshape((-1))
    edges_df["weight"] = results['val_matrix'].reshape((-1))

    # remove self causality at t=0
    edges_df = edges_df[~((edges_df["t"] == 0) & (
        edges_df["src"] == edges_df["dst"]))]
    edges_df = edges_df[~np.isinf(edges_df["weight"].values)]
    print(f"p-value < {significance_level} are selected:")
    edges_df = edges_df[edges_df["pval"] < significance_level]
    print("edges_df: ", edges_df.shape)
    
    edges_df = edges_df[edges_df["weight"].abs() > weight_thr]
    if weight_sign == 'positive':  # for anomaly corr using parcorr scores to keep positive corr only
        print("positive weights are selected!")
        edges_df = edges_df[edges_df["weight"] > 0]
        print("edges_df: ", edges_df.shape)

    if not edges_df.empty:
        edges_df["src_lag"] = edges_df[["src", "t"]].apply(
            lambda x: lagged_src_name(*x), axis=1)

        if isagg:
            edges_agg_df = aggregate_time_tag_edges(edges_df)
            return edges_df, edges_agg_df
        else:
            return edges_df
    else:
        return edges_df

def plot_adjacency_matrix(A, varnames, vmin=0, vmax=1, t_lag=None, figsize=(3, 3), ax=None, isshow=False, **kwargs):
    sns.set()
    iscbar = kwargs.get("iscbar", True)
    isyticks = kwargs.get("isyticks", True)
    fmt = kwargs.get("fmt", '0.2f')
    fontsize = kwargs.get("fontsize", None)
    _cbar_kws = kwargs.get("cbar_kws", {})
    cbar_kws = {"label": "Edge weight",
                    "shrink": 1.0, 
                    "orientation": "vertical", "pad": 0.02
                 }
    cbar_kws.update(**_cbar_kws) 
    
    if isshow:
        plt.figure(figsize=figsize)

    with sns.axes_style("white"):
        ax = sns.heatmap(A, vmin=vmin, vmax=vmax, square=True,
                            annot=True, fmt=fmt, 
                            cbar_kws=cbar_kws, 
                            cbar=iscbar, cmap='Spectral',
                            ax=ax
                        ) 
        print(varnames)
        _ = ax.set_xticklabels(varnames, rotation=90, fontsize=fontsize)
        _ = ax.set_yticklabels(varnames, rotation=0, fontsize=fontsize)

        if t_lag is None:
            _ = ax.set_ylabel(f"Cause", fontsize=fontsize)
            _ = ax.set_xlabel("Effect", fontsize=fontsize)
        else:
            _ = ax.set_ylabel(f"Cause (at $t={-t_lag}$)", fontsize=fontsize)
            _ = ax.set_xlabel("Effect (at $t=0$)", fontsize=fontsize)

        if isyticks:
            ax.figure.axes[-1].yaxis.label.set_size(fontsize)
            ax.figure.axes[-1].yaxis.set_ticklabels(ax.figure.axes[-1].yaxis.get_ticklabels(), fontsize=fontsize)

    if isshow:
        plt.show()
    else:
        return ax

def plot_tlagged_weight_map(results, pc_alpha, varnames, w_thr=0.0, figsize=(3, 3), isshow=True, **kwargs):
    sns.set()

    fontsize = kwargs.get("fontsize", None)
    # _cbar_kws = kwargs.get("cbar_kws", {})
    # cbar_kws = {"label": "Edge weight",
    #                 "shrink": 1.0, 
    #                 "orientation": "vertical", "pad": 0.02
    #              }
    # cbar_kws.update(**_cbar_kws) 

    n = results['val_matrix'].shape[-1]
    fig_cols = n if n <= 5 else 5
    fig_rows = int(np.ceil(n/fig_cols))
    grid_h = figsize[1]*fig_rows
    grid_w = figsize[0]*fig_cols

    fig = plt.figure(figsize=(grid_w, grid_h), constrained_layout=True)
    axs = fig.subplots(fig_rows, fig_cols)
    axs = util.trim_axs(axs, n)

    for i in range(n):
        t_lag = n-i-1
        p = results['p_matrix'][..., t_lag]
        w = results['val_matrix'][..., t_lag]
        w[w < w_thr] = 0
        w[p > pc_alpha] = 0
        ax = axs[i]
        # with sns.axes_style("white"):
        #     ax = sns.heatmap(w, vmin=0, vmax=1, square=True,
        #                     annot=True, fmt=fmt, cbar_kws=cbar_kws, 
        #                    cbar= i==len(axs) - 1, 
        #                    cmap='Spectral', ax=ax
        #                 )  
        ax = plot_adjacency_matrix(w, varnames, vmin=0, vmax=1, square=True, iscbar= i == len(axs) - 1, ax=ax, isshow=False, **kwargs)
        _ = ax.set_xticks(np.arange(len(w)))
        _ = ax.set_yticks(np.arange(len(w)))
        _ = ax.set_xticklabels(varnames, rotation=90, fontsize=fontsize)
        _ = ax.set_yticklabels(varnames, rotation=0, fontsize=fontsize)
        _ = ax.set_ylabel(f"Cause (at $t={-t_lag}$)", fontsize=fontsize)
        _ = ax.set_xlabel("Effect (at $t=0$)", fontsize=fontsize)

        if i==len(axs) - 1:
            ax.figure.axes[-1].yaxis.label.set_size(fontsize)
            ax.figure.axes[-1].yaxis.set_ticklabels(ax.figure.axes[-1].yaxis.get_ticklabels(), fontsize=fontsize)

    if isshow:
        plt.show()
    else:
        return fig, axs

def get_digraph(results, var_names, G, pc_alpha, w_thr=0, tlags=None):
    H = nx.DiGraph()
    H.add_nodes_from(G)
    if tlags is None:
        tau_max_ = results['val_matrix'].shape[-1] - 1
        tlags = np.arange(tau_max_+1)
        print(f"default tlag is set {tlags}. assuming tau_min=0, tau_max={tau_max_}")

    for i, t in enumerate(tlags):
        rval = results['val_matrix'][:, :, i]
        rval[results['p_matrix'][:, :, i] >= pc_alpha] = 0.0
        nodes_cause = [f"{x}, t={-t}" for x in var_names]
        nodes_effect = [f"{x}, t=0" for x in var_names]
        edges_t_df = pd.DataFrame(rval, index=nodes_cause, columns=nodes_effect)
        edges_t_df.index.name = "cause"
        edges_t_df = edges_t_df.reset_index().melt(value_vars=nodes_effect, var_name="effect", value_name="w", id_vars="cause")
        # edges_t_df
        H.add_edges_from(edges_t_df.loc[edges_t_df["w"] > w_thr, ["cause", "effect"]].values.tolist())

    # print(list(H.nodes))
    print(list(H.edges))
    return H

def eval_graph_match(_G, _H, **kwargs):
    double_for_anticausal = kwargs.get("double_for_anticausal", True)
    pos_label = kwargs.get("pos_label", None)
    H = nx.DiGraph()
    H.add_nodes_from(_G)
    H.add_nodes_from(_H)
    H.add_edges_from(_H.edges)
    G = nx.DiGraph()
    G.add_nodes_from(_G)
    G.add_nodes_from(_H)
    G.add_edges_from(_G.edges)
    print(G.edges, H.edges)
    apauc_score = precision_recall(G, H)[0]  # best 1
    shd_score = SHD(G, H, double_for_anticausal=double_for_anticausal) # best is 0
    score_df = pd.DataFrame(index=[0], columns=["APRC", "SHD"])
    score_df["APRC"] = [np.round(apauc_score, 5)]
    score_df["SHD"] = [np.round(shd_score, 5)]
    return score_df

def prepare_digraph(edges_df, var_names, return_summary=False):
    H = nx.DiGraph()
    H.add_nodes_from(var_names)
    if return_summary:
        edges_arr = edges_df[["src", "dst"]].apply(
            tuple, axis=1).values.tolist()
    else:
        edges_arr = edges_df[["src_lag", "dst"]].apply(
            tuple, axis=1).values.tolist()
    edges_arr = list(set(edges_arr))
    H.add_edges_from(edges_arr)
    print("Graph nodes all:", list(H.nodes))
    print("Graph edges:" , list(H.edges))
    return H

def cd_eval_score(G, edges_df, variables, return_summary=True, **kwargs):
    H = prepare_digraph(edges_df, variables,
                        return_summary=return_summary)
    # print("predicted nodes: ", H.nodes)
    print("predicted edges: ", H.edges)
    score_df = eval_graph_match(G, H, **kwargs)
    # print(score_df)
    return score_df

def align_graph_nodes(_G, _H):
    H = nx.DiGraph()
    H.add_nodes_from(_G)
    H.add_nodes_from(_H)
    H.add_edges_from(_H.edges)

    G = nx.DiGraph()
    G.add_nodes_from(_G)
    G.add_nodes_from(_H)
    G.add_edges_from(_G.edges)
    return G, H

def cd_eval_score_extended(G, edges_df, var_names, return_summary=True, isplot=False, figsize=(3, 3)):
    
    G_pred = prepare_digraph(edges_df, var_names, 
                            return_summary=return_summary)
    
    # adding missing nodes to each other from each other
    G, G_pred = align_graph_nodes(G, G_pred)

    edges_ground_truth_Amatrix_arr = np.array(nx.adjacency_matrix(G).todense())
    edges_pred_Amatrix = np.array(nx.adjacency_matrix(G_pred).todense())

    if isplot:
        print("True GCM")
        _ = plot_adjacency_matrix(edges_ground_truth_Amatrix_arr,
                                                G.nodes, vmin=0, vmax=1, t_lag=None, figsize=figsize, iscbar=True, fmt='.0f', 
                                                cbar_kws={"label": "Edge weight",
                                                            "shrink": 0.8, "orientation": "vertical", "pad": 0.02
                                                            },
                                                            isshow=True
                                                )

        print("Estimated GCM")
        _ = plot_adjacency_matrix(edges_pred_Amatrix,
                                                G_pred.nodes, vmin=0, vmax=1, t_lag=None, figsize=figsize, iscbar=True, fmt='.0f', 
                                                cbar_kws={"label": "Edge weight",
                                                            "shrink": 0.8, "orientation": "vertical", "pad": 0.02
                                                            },
                                                            isshow=True
                                               )
    
    # calculate metrics
    mt = castle_MetricsDAG(edges_pred_Amatrix, edges_ground_truth_Amatrix_arr)
    score_df_ = cd_eval_score(
        G, edges_df, var_names, return_summary=return_summary)

    score_dict = mt.metrics
    _ = score_dict.pop('fdr')
    _ = score_dict.pop('tpr')
    _ = score_dict.pop('nnz')
    _ = score_dict.pop('gscore')
    score_dict['shdu'] = score_dict.pop('shd')
    score_dict = {k:np.around(v, 3) for k, v in score_dict.items()}
    score_dict['shd'] = score_df_["SHD"].iloc[0]
    score_dict['aprc'] = score_df_["APRC"].iloc[0].round(3)
    return score_dict

def gcm_flat_to_pcmci_result_format(Gt, num_variables):
    results = {"val_matrix": [], "p_matrix": []}
    edges_Amatrix = np.array(nx.adjacency_matrix(Gt).todense())
    t_dim = edges_Amatrix.shape[0]//num_variables
    results['val_matrix'] = np.array(edges_Amatrix[:, :num_variables]).T.reshape(-1, t_dim, order="F").reshape(num_variables, num_variables, t_dim, order="C")
    results['p_matrix'] = np.zeros_like(results['val_matrix'])
    print(f"Adjacent matrix flat: {edges_Amatrix.shape}, reshaped: {results['val_matrix'].shape}")
    return results

def unrolled_ts(df, tau_max):
    print("unrolled_ts...", df.shape)
    df_unrolled = df.copy()
    # for col in df.columns:
    for t in range(1, tau_max+1):
        for col in df.columns:
        #  for t in range(1, tau_max):
            df_unrolled["{}__tlag_{:02d}".format(col, abs(t))] = df[col].shift(t).fillna(0)
    print(df_unrolled.shape)
    return df_unrolled

def roll_unrolled_pcmci_results(results, unrolled_ncols, rolling_tau_max):
    print("roll_unrolled_pcmci_results...")
    num_vars = results['val_matrix'].shape[0]//(rolling_tau_max+1)
    results_rolled = {} 
    for k, v in results.items():
        if len(v.shape) == 0:
            continue
        rval = v[:, :num_vars, 0]
        # rval[:num_vars, num_vars:] = np.nan
        rval.shape
        for i in range(rval.shape[1]):
            if i == 0:
                rval_unroll = rval[i::rval.shape[1], :].T # tlag for each var is at each rval.shape[1] ste[]
            else:
                rval_unroll = np.vstack((rval_unroll, rval[i::rval.shape[1], :].T))

        results_rolled[k] = rval_unroll.reshape((unrolled_ncols, unrolled_ncols, -1))
    # util.print_dict(results_rolled)
    return results_rolled

def generate_pcmci_gcm_compare_ref_gcm(df_binary, gcm_method, ci_test_method, G, tau_min=0, tau_max=3, pc_alpha=0.05, w_thr=0.001, link_assumptions=None, isunrolled=False, verbosity=0):

    print(f"CD config: gcm_method: {gcm_method}, ci_test_method: {ci_test_method}, tau_min:{tau_min}, tau_max:{tau_max}, p_v:{pc_alpha}")
    
    rolled_col_names = df_binary.columns
    if isunrolled:
        rolling_tau_max = tau_max
        rolled_ncol = df_binary.shape[1]
        tau_max = 0
        df_binary = unrolled_ts(df_binary, rolling_tau_max)
        # print(df_binary_unrolled.head(10))

        g, ax = util.plot_grid(df_binary.rename(columns={f"{col}":col.replace('$__tlag_0', '$ - ') for col in df_binary.columns}), 
                                figsize=(7, 2), ncol_force=True, ncol=rolled_ncol, x_gridno=10,
                                # color="red"
                                iscolor_per_col=True, xlabel="t", ylabel="value",
                                wspace=0.2, hspace=0.6,
                                fontsize=16,
                                labelfontsize=16,
                                tickfontsize=16,
                                isshow=False
                                )
        for ax_sub in ax:
            _ = ax_sub.set_yticks([0, 1])
            plt.show()

    var_names = [rf"{col}" for col in df_binary.columns] # r in  case of $$ in the name
    pp_binary = pp.DataFrame(df_binary.values.astype("float64"),
                             #  mask=mask,
                             datatime={0: np.arange(len(df_binary))},
                             var_names=var_names)
    
    tp.plot_timeseries(pp_binary); plt.show()
    
    if ci_test_method == "parcorr":
        ci_test_func = ParCorr(significance='analytic')
    else:
        raise f"undefined ci_test_method: {ci_test_method}"  
    
    tObj = util.ProcTimer()

    if gcm_method == "pcmci":
        pcmci = tigramite_PCMCI(
                                dataframe=pp_binary,
                                cond_ind_test=ci_test_func,
                                verbosity=verbosity)
        results = pcmci.run_pcmciplus(
                            tau_min=tau_min, tau_max=tau_max, 
                            pc_alpha=pc_alpha,
                            link_assumptions=link_assumptions,
                            # contemp_collider_rule='majority',
                            # reset_lagged_links=True
                            # auto_first=False
                            )
    elif gcm_method == "lpcmci":
        pcmci = tigramite_LPCMCI(
                                dataframe=pp_binary,
                                cond_ind_test=ci_test_func,
                                verbosity=verbosity)    
        results = pcmci.run_lpcmci(
                                tau_min=tau_min, tau_max=tau_max, 
                                pc_alpha=pc_alpha,
                                link_assumptions=link_assumptions,
                                # auto_first=False
                            )
    else:
        raise f"undefined pcmci: {gcm_method}"
    
    proc_t = tObj.get_proctime()
    tObj.display_proctime()

    if isunrolled:
        results['val_matrix'][:rolled_ncol, rolled_ncol:] = np.nan
        results = roll_unrolled_pcmci_results(results, rolled_ncol, rolling_tau_max)
    
    isinf = np.isinf(results["p_matrix"]) | np.isinf(results["val_matrix"])
    results["p_matrix"][isinf] = 0
    results["val_matrix"][isinf] = 0
    
    tp.plot_graph(
            val_matrix=results['val_matrix'],
            graph=results['graph'],
            var_names=var_names,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False
            ); plt.show()

    print("Temporal GCM outline: ", results["graph"])
    print("result matrics shape: ", results['val_matrix'].shape)
    # print(results)
    print("p_matrix: ")
    print(results["p_matrix"].round(2))
    print("val_matrix: ")
    print(results["val_matrix"].round(2))

    plot_tlagged_weight_map(results, pc_alpha, rolled_col_names, w_thr=w_thr)

    H = get_digraph(results, rolled_col_names, G, pc_alpha, w_thr=w_thr)
    score_df = eval_graph_match(G, H)
    score_df["p_value"] = pc_alpha
    score_df["Process Time"] = proc_t

    return score_df, H

def sparse_data_highlighter(df_binary, tau_max=1, **kwargs):
    fontsize = kwargs.get("fontsize", 16)
    legendfontsize = kwargs.get("legendfontsize", 14)
    bbox_to_anchor =  kwargs.get("bbox_to_anchor", (-0.0, 1.07))

    unoin_flag_agg_ = df_binary.astype(str).apply(lambda x: ",".join(x), axis=1)
    unoin_flag_agg = unoin_flag_agg_.apply(lambda x: False).copy()
    idx = 0
    for k, g in itertools.groupby(unoin_flag_agg_.values):
        g_size = len(list(g))
        idx += g_size
        if g_size > tau_max:
            unoin_flag_agg.iloc[np.arange(idx-g_size, idx - tau_max)] = True
    unoin_flag_agg_df = pd.DataFrame(index=df_binary.index, columns=df_binary.columns)

    for col in df_binary.columns:
        unoin_flag_agg_df[col] = unoin_flag_agg

    fig, ax = util.plot_grid(df_binary, 
                    join_df=unoin_flag_agg_df, 
                    join_format="scatter", color_join="gray", marker_size=30,
                    legends=["Anomaly Flag", "Sparse Region"], legend_idxs=[0],
                    **kwargs
                    )
    for ax_sub in ax:
        if ax_sub.legend_: ax_sub.get_legend().remove()
        _ = ax_sub.set_yticks([0, 1])

    leg = plt.figlegend(loc='upper left', bbox_to_anchor=bbox_to_anchor, ncol=2,
                                fancybox=True, frameon=True, shadow=False, fontsize=legendfontsize)

    for leg_line in leg.legendHandles: # for matplotlib <=3.6
        # for leg_line in leg.legend_handles:
        leg_line.set_linewidth(5)
    return fig

def sparse_data_handler(df, data_mode="nan", **kwargs):
    """
    """
    print(f"sparse_data_handler...data_mode: {data_mode}: ",  kwargs)
    if data_mode == "nan":
        return util.nan_handler(df, **kwargs)
    elif data_mode == "flag":
        keep_len = kwargs.get("keep_len", 1)
        _keep_len = max(keep_len, 1)  # at least keep 1 data point
        _df = df.reset_index(drop=True, inplace=False)

        # based on state status across all
        unoin_flag_agg = _df.astype(str).apply(lambda x: ",".join(x), axis=1)

        idx = 0
        for k, g in itertools.groupby(unoin_flag_agg.values):
            g_size = len(list(g))
            idx += g_size
            # if (k != 0) and (g_size > _keep_len):
            if g_size > _keep_len:
                _df.drop(index=np.arange(idx-g_size, idx -
                                         _keep_len).tolist(),  inplace=True)
        
        # pad zero tail _keep_len
        # _df = pd.concat([_df, _df.iloc[:_keep_len, :].apply(lambda x: 0*x)], axis=0)
        _df = _df.shift(_keep_len).fillna(0).reset_index(drop=True, inplace=False)
        print("Input data size: {}, new size: {}. The rate of compression by {:0.4}%".format(df.shape, _df.shape, 100*float(df.shape[0] - _df.shape[0])/df.shape[0]))
        return _df
    else:
        return df

def learn_scm_dag_structure(df, alg="hybrid", **kwargs):
    print("learn_scm_dag_structure...", alg)
    print(kwargs)

    tObj = util.ProcTimer()
    
    verbosity = kwargs.pop("verbosity", 1)
    significance_level = kwargs.pop("significance_level", 0.01)
    variant = kwargs.pop("variant", 'stable')
    # ci_test = kwargs.pop("ci_test", 'pearsonr')  # pearsonr, chi_square
    max_cond_vars = kwargs.pop("max_cond_vars", 5)
    tabu_length = kwargs.pop("tabu_length", 10)
    # BicScore is must faster than BDeuScore and BicScore: speed BicScore >> K2Score > BDeuScore but BicScore is less accurate
    scoring_method = kwargs.pop("scoring_method", BicScore)
    epsilon = kwargs.pop("epsilon", 0.0001)
    max_iter = kwargs.pop("max_iter", 100)
    start_dag = kwargs.pop("start_dag", None)  # e.g. domain knowledge e.g. ngCCM and RM ntk
    fixed_edges = kwargs.pop("fixed_edges", {})
    first_skeleton_alg = kwargs.pop("first_skeleton_alg", "mmpc")
    graph = kwargs.pop("graph", None)  # will convert
    method_indep = kwargs.pop("method_indep", 'corr')
    alpha = significance_level
    njobs = kwargs.pop("njobs", None)
    cutoff = kwargs.pop("cutoff", 0.001)
    linear = kwargs.pop("linear ", True)
    lambda1 = kwargs.pop("lambda1", 10)
    lambda2 = kwargs.pop("lambda2", 0.001)
    train_epochs = kwargs.pop("max_iter", 3000)
    test_epochs = kwargs.pop("test_epochs", 1000)
    batch_size = kwargs.pop("batch_size", -1)
    # fgan’ (default), ‘gan’ or ‘mse’)
    losstype = kwargs.pop("losstype ", 'fgan’')

    ks = list(kwargs.keys())

    if alg == "pc":
        ci_test = kwargs.pop("ci_test", 'pearsonr')  # pearsonr, chi_square
        print(kwargs)
        print(variant, ci_test, max_cond_vars, significance_level)

        tObj.restart()

        pc_est = pgmpy_PC(df)
        model = pc_est.estimate(variant=variant, ci_test=ci_test,
                                max_cond_vars=max_cond_vars, significance_level=significance_level, **kwargs)

        tObj.display_proctime()

    elif alg == "hc":
        for k in ks:
            if k not in ['fixed_edges', 'tabu_length', 'start_dag', ' scoring_method', 'max_iter', 'epsilon']:
                kwargs.pop(k, None)
        
        kwargs.pop("ci_test", None)
        # hcestimate(scoring_method='k2score', start_dag=None, fixed_edges={}, tabu_length=100, max_indegree=None, black_list=None, white_list=None, epsilon=0.0001, max_iter=1000000.0, show_progress=True)
        print(kwargs)

        tObj.restart()

        hc_est = pgmpy_HillClimbSearch(df, use_cache=False)
        model = hc_est.estimate(tabu_length=tabu_length, scoring_method=scoring_method(
            df), start_dag=start_dag, fixed_edges=fixed_edges, max_iter=max_iter, epsilon=epsilon, **kwargs)  # return best model from candidate dags

        tObj.display_proctime()

    elif alg in ["hybrid", "mmhc"]:
        # The MMHC algorithm combines the constraint-based and score-based method. Max-Min Hill-Climbing" (MMHC) algorithm.
        # It has two parts:
            # Learn undirected graph skeleton using the constraint-based construction procedure MMPC
            # Orient edges using score-based optimization (BDeu score + modified hill-climbing)
        for k in ks:
            if k not in ['first_skeleton_alg', 'tabu_length', 'significance_level', ' scoring_method', 'max_iter', 'ci_test']:
                kwargs.pop(k, None) 

        print(kwargs)
        if first_skeleton_alg == "mmpc":
            kwargs.pop("ci_test", None)

            tObj.restart()

            mmhc_est = pgmpy.estimators.MmhcEstimator(df)
            model = mmhc_est.estimate(significance_level=significance_level,
                                      tabu_length=tabu_length, scoring_method=scoring_method(df), **kwargs)

            tObj.display_proctime()

        elif first_skeleton_alg == "pc":
            ci_test = kwargs.pop("ci_test", 'chi_square')
            tObj.restart()

            pc_est = pgmpy_PC(df)
            skeleton = pc_est.estimate(variant=variant, ci_test=ci_test,
                                       max_cond_vars=max_cond_vars, significance_level=significance_level, **kwargs)

            tObj.display_proctime()

            print("Part 1) Skeleton: ", skeleton.edges())

            tObj.restart()

            hc_est = pgmpy_HillClimbSearch(df)
            model = hc_est.estimate(tabu_length=tabu_length, white_list=skeleton.to_directed(
            ).edges(), scoring_method=scoring_method(df), start_dag=start_dag, fixed_edges=fixed_edges, max_iter=max_iter, epsilon=epsilon, **kwargs)

            tObj.display_proctime()

    elif alg == "fges":
        penalty = kwargs.pop("penalty", 60.0)

        tObj.restart()

        model = FGES(df, penalty=penalty)
        model.forward_equivalence_search()

        tObj.display_proctime()

    elif alg.startswith("cdt__"):
        cdt_alg = alg.split("__")[1]
        cdt_alg_dict = {"CGNN":  {"alg": CGNN, "data": ["continues"], "func": "nonlinear"},
                        "PC": {"alg": PC, "data": ["continues", "discrete"], "func": "linear"},
                        "GES": {"alg": GES, "data": ["continues", "categorical"], "func": "linear"},
                        "GIES": {"alg": GIES, "data": ["continues", "categorical"], "func": "linear"},
                        "LINGAM": {"alg": LiNGAM, "data": ["continues"], "func": "linear"},
                        "CAM": {"alg": CAM, "data": ["continues"], "func": "nonlinear"},
                        "GS": {"alg": GS, "data": ["continues", "discrete"], "func": "linear"},
                         "IAMB": {"alg":IAMB, "data":["continues", "discrete"], "func":"linear"},
                        "MMPC": {"alg": MMPC, "data": ["continues", "discrete"], "func": "linear"},
                        "SAM": {"alg": SAM, "data": ["continues", "mixed"], "func": "nonlinear"},
                        "CCDR": {"alg": CCDr, "data": ["continues"], "func": "nonlinear"}}

        cdt_ALG = cdt_alg_dict[cdt_alg.upper()]["alg"]
        print(cdt_ALG)

        if cdt_alg in ["ges", "gies"]:
            """
            Description: 2002 Greedy Equivalence Search algorithm. 
            A score-based Bayesian algorithm that searches heuristically the graph which minimizes a likelihood score on the data.

            Description: 2022 Greedy Interventional Equivalence Search algorithm. 
            A score-based Bayesian algorithm that searches heuristically the graph which minimizes a likelihood score on the data. 
            The main difference with GES is that it accepts interventional data for its inference.

            Assumptions: The output is a Partially Directed Acyclic Graph (PDAG) (A markov equivalence class). 
            The available scores assume linearity of mechanisms and gaussianity of the data.

            """
            # Continuous (score='obs') or Categorical (score='int')
            score = kwargs.pop("score", 'int')

            tObj.restart()

            obj = cdt_ALG(score=score)

        elif cdt_alg in ["pc"]:
            """
            Description: pgmpy_PC (Peter - Clark) One of the most famous score based approaches for causal discovery. 
            Based on conditional tests on variables and sets of variables, it proved itself to be really efficient.

            cdt.causality.graph.pgmpy_PC(CItest='gaussian', method_indep='corr', alpha=0.01, njobs=None, verbose=None)
            """
            ci_test = kwargs.get("ci_test", 'gaussian')  # Available CI tests: binary, discrete, hsic_gamma, hsic_perm, hsic_clust, gaussian, rcit, rcot

            tObj.restart()
            obj = cdt_ALG(CItest=ci_test,
                          method_indep=method_indep,
                          alpha=alpha, njobs=njobs)

        elif cdt_alg in ["cam"]:
            """
            Description: Causal Additive models (2014), a causal discovery algorithm relying on fitting Gaussian Processes on data, while considering all noises additives and additive contributions of variables.
            Assumptions: The data follows a generalized additive noise model: each variable in the graph  is generated following the model  e presenting mutually independent noises variables accounting for unobserved variables.

            """
            tObj.restart()
            obj = cdt_ALG(cutoff=cutoff)

        elif cdt_alg in ["ccdr"]:
            """
            Description: Concave penalized Coordinate Descent with reparametrization) structure learning algorithm as described in Aragam and Zhou (2015). 
            This is a fast, score based method for learning Bayesian networks that uses sparse regularization and block-cyclic coordinate descent.

            Assumptions: This model does not restrict or prune the search space in any way, does not assume faithfulness, does not require a known variable ordering, 
            works on observational data (i.e. without experimental interventions), 
            works effectively in high dimensions, and is capable of handling graphs with several thousand variables. The output of this model is a DAG.
            """
            tObj.restart()
            obj = cdt_ALG()

        elif cdt_alg in ["cgnn"]:
            """
            Causal Generative Neural Netwoks 2017

            Description: Causal Generative Neural Networks. 
            Score-method that evaluates candidate graph by generating data following the topological order of the graph using neural networks, and using MMD for evaluation.            
            """
            njobs = kwargs.get("njobs", None)

            tObj.restart()
            obj = cdt_ALG()

        elif cdt_alg in ["sam"]:
            """
            Description: Structural Agnostic Model 2018 is an causal discovery algorithm for DAG recovery leveraging both distributional asymetries and conditional independencies. 
            the first version of SAM without DAG constraint is available as SAMv1.

            Assumptions: The class of generative models is not restricted with a hard contraint, but with soft constraints parametrized with the lambda1 and lambda2 parameters, with gumbel softmax sampling. This algorithms greatly benefits from bootstrapped runs (nruns >=8 recommended). GPUs are recommended but not compulsory. 
            The output is a DAG, but may need a thresholding as the output is averaged over multiple runs.
            """
            tObj.restart()
            obj = cdt_ALG()

        elif cdt_alg in ["lingam"]:
            """
            Description:2006 Linear Non-Gaussian Acyclic model. LiNGAM handles linear structural equation models.

            Assumptions: The underlying causal model is supposed to be composed of linear mechanisms and non-gaussian data. 
            Under those assumptions, it is shown that causal structure is fully identifiable (even inside the Markov equivalence class).
            """

            tObj.restart()
            obj = cdt_ALG()

        elif cdt_alg in ["mmpc", "gc", "iamb"]:
            """
            Max-Min Parents-Children algorithm 2003, mmhc-2006.

            Description: The Max-Min Parents-Children (MMPC) is a 2-phase algorithm with a forward pass and a backward pass. 
            The forward phase adds recursively the variables that possess the highest association with the target conditionally to the already selected variables. The backward pass tests d-separability of variables conditionally to the set and subsets of the selected variables.

            Assumptions: MMPC outputs markov blankets of nodes, with additional assumptions depending on the conditional test used.
            return The Markov Blanket of node A is the set of nodes composed of A's parents, its children, and its children's other parents (i.e., spouses)
            """

            score = kwargs.pop("score", 'NULL')
            tObj.restart()
            obj = cdt_ALG(score=score, alpha=alpha)

        else:
            """
            Description: Linear Non-Gaussian Acyclic model. LiNGAM handles linear structural equation models.Assumptions: The underlying causal model is supposed to be composed of linear mechanisms and non-gaussian data. 
            Under those assumptions, it is shown that causal structure is fully identifiable (even inside the Markov equivalence class).
            """
            alpha = kwargs.get("alpha", 0.05)
            score = kwargs.get("score", 'corr')

            tObj.restart()
            obj = cdt_ALG(score=score, alpha=alpha)

        # No graph provided as an argument
        model_cdt = obj.predict(df, graph=graph)
        tObj.display_proctime()

        # if cdt_alg in ["pc", "mmpc", "gc", "ges", "gies"]:
        if cdt_alg in ["pc", "gc", "ges", "gies", "iamb", "mmpc"]:
            tObj.restart()
            model_cdt = obj.predict(df, graph=nx.Graph(model_cdt))  # to direct undirected graphs
            tObj.display_proctime()

        # convert to pgmpy model
        model = DAG()
        model.add_nodes_from(nodes=df.columns.tolist())
        model.add_edges_from(ebunch=model_cdt.edges())

    elif alg.startswith("gc__"):
        ci_test = kwargs.pop("ci_test", 'chi2')
        gc_alg_dict = {
                    # 2005
                    "pc": {"name": 'PC', "default_kwargs": dict(variant=variant, ci_test=ci_test, alpha=alpha)},
                    # 2002, not scalable for high dims >20
                    "ges": {"name": 'GES', "default_kwargs": dict(criterion='bdeu', k=0.001, N=10, method='scatter')},
                    "directlingam": {"name": 'DirectLiNGAM', "default_kwargs": dict(measure='pwling', thresh=0.01)},  # 2011
                    # 2006
                    "icalingam": {"name": 'ICALiNGAM', "default_kwargs": dict(random_state=None, max_iter=1000, thresh=0.01)},

                    # very poor fprfromance # 2019, needs gpu slow
                    "mcsl": {"name": 'MCSL', "default_kwargs": dict(device_type='gpu', max_iter=5, iter_step=1000)},
                    "notears": {"name": 'Notears', "default_kwargs": dict(loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.01)},  # 2018
                    "notearslowrank": {"name": 'NotearsLowRank', "default_kwargs": dict(max_iter=15, h_tol=1e-6, rho_max=1e+20, w_threshold=0.01)},
                    "notearsnonlinear": {"name": 'NotearsNonlinear', "default_kwargs": dict(device_type='gpu')},
                    "golem": {"name": 'GOLEM', "default_kwargs": dict(device_type='gpu', num_iter=10000)},  # num_iter=1e+5
                    "grandag": {"name": 'GraNDAG', "default_kwargs": dict(input_dim=df.shape[1], iterations=1000,  hidden_num=2,
                                                    hidden_dim=10,
                                                    batch_size=64,
                                                    lr=0.0001,
                                                    # model_name='NonLinGaussANM',
                                                    nonlinear='leaky-relu',
                                                    optimizer='rmsprop',
                                                    # h_threshold=1e-8,
                                                    device_type='gpu',
                                                    # device_ids='0',
                                                    # use_pns=False,
                                                    # pns_thresh=0.75,
                                                    # num_neighbors=None,
                                                    # normalize=False,
                                                    precision=False,
                                                    random_seed=10,
                                                    # jac_thresh=True,
                                                    # lambda_init=0.0,
                                                    # mu_init=0.001,
                                                    # omega_lambda=0.0001,
                                                    # omega_mu=0.9,
                                                    stop_crit_win=100,
                                                    edge_clamp_range=0.0001)
                                },  # 2020

                    # needs gpu slow
                    "gae": {"name": 'GAE', "default_kwargs": dict(device_type='gpu', update_freq=3000, epochs=3, seed=42)},  # 2019
                    "dag_gnn": {"name": 'DAG_GNN', "default_kwargs": dict(device_type='gpu', epochs=300, batch_size=100, seed=42)},  # 2019
                    "rl": {"name": 'RL', "default_kwargs": dict(device_type='gpu', nb_epoch=1000, seed=42)},  # 2019
                    "corl": {"name": 'CORL', "default_kwargs": dict(device_type='gpu', iteration=1000, seed=42)}  # 2021
        }

        gc_alg = gc_alg_dict[alg.split("__")[1].lower()]

        print(gc_alg)
        gc_alg_kwargs = gc_alg["default_kwargs"]
        # gc_alg_kwargs.update(kwargs)
        for k, v in gc_alg_kwargs.items():
            gc_alg_kwargs[k] = kwargs.pop(k, v)
        print(gc_alg_kwargs)

        if 'device_type' in gc_alg_kwargs.keys():
            if gc_alg_kwargs['device_type'] == 'gpu':
                if not util.torch.cuda.is_available():
                    print(f"GPU not available: switching {alg} device_type to cpu")
                    gc_alg_kwargs['device_type'] = 'cpu'
                    
        gcModel = getattr(castle_algorithms, gc_alg["name"])(**gc_alg_kwargs)

        input_data = df.values.astype("double")
        gcModel.learn(input_data, df.shape[1]) if gc_alg["name"] == 'NotearsLowRank' else gcModel.learn(input_data)

        proc_t = tObj.get_proctime()
        tObj.display_proctime()

        model = DAG()
        model.add_nodes_from(df.columns)
        model.add_edges_from(list(map(lambda x: (
            df.columns[x[0]], df.columns[x[1]]), np.argwhere(np.array(gcModel.causal_matrix) == 1))))
        del gcModel
        util.gc.collect()
    else:
        raise Exception('Unknown dag learning algorithm')

    print("Model: ", model.edges())

    tObj.stop()

    return model

def learn_scm_ts_dag_structure(df, alg="lpcmci", **kwargs):
    print("learn_scm_ts_dag_structure...", alg, kwargs)

    tau_min = kwargs.pop("tau_min", 0)
    tau_max = kwargs.pop("tau_max", 3)
    pc_alpha = kwargs.pop("significance_level", 0.01)
    auto_first = kwargs.pop("auto_first", False)
    link_assumptions = kwargs.pop("link_assumptions", None)
    ci_test = kwargs.pop("ci_test", 'Gsquared')
    verbosity = kwargs.pop("verbosity", 1)
    datatype = kwargs.pop("datatype", "float")
    # 'positive; for anomaly flags with ParCorr() and RobustParCorr()
    weight_sign = kwargs.pop("weight_sign", "both")
    cond_ind_test_dict = {"RobustParCorr": RobustParCorr(),
                          "ParCorr": ParCorr(),
                        #   "GPDC": GPDC(),
                          "CMIknn": CMIknn(),
                          "CMIsymb": CMIsymb(),
                          "Gsquared": Gsquared(significance='analytic')
                          }

    cond_ind_test = cond_ind_test_dict[ci_test]
    print("cond_ind_test: ", cond_ind_test)

    pp_dataframe = pp.DataFrame(df.values.astype(datatype),
                                #  mask=mask,
                                datatime={0: np.arange(len(df))},
                                var_names=df.columns)
    tObj = util.ProcTimer()

    if alg == "pcmci":
        pcmci_ = tigramite_PCMCI(
            dataframe=pp_dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity)

        print(f"tau_min={tau_min}, tau_max={tau_max}")
        tObj.restart()
        results = pcmci_.run_pcmciplus(
            tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,
            reset_lagged_links=True, 
            link_assumptions=link_assumptions
        )
    
        tObj.display_proctime()

    elif alg == "lpcmci":
        '''
        '''
        pcmci_default_kwargs = dict(link_assumptions=link_assumptions, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,                          
                                    auto_first=auto_first,
                                    ) 

        pcmci_default_kwargs.update(
            {k: v for k, v in kwargs.items() if k in pcmci_default_kwargs})

        lpcmci_ = tigramite_LPCMCI(
            dataframe=pp_dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity)

        tObj.restart()
        results = lpcmci_.run_lpcmci(**pcmci_default_kwargs)

        tObj.display_proctime()

    else:
        print("alg: ", alg)
        raise f"Undefined SCM learning \"{alg}\" algorithm."

    if verbosity:
        print("causal graph: ")
        try:
            util.print_dict(results["graph"])
        except:
            print(results["graph"])

    edges_df = prepare_edges_pcmci(
        results, df.columns.tolist(), tau_max, significance_level=pc_alpha, isagg=False, weight_sign=weight_sign)
    
    tObj.stop()

    print("edges_df: ", edges_df.shape)
    if verbosity: print(edges_df.head())
    return edges_df

def learn_scm_dag_bays_parameters(df_input, dag_model, alg="bayesian", **kwargs):
    print("learn_scm_dag_bays_parameters...", alg, df_input.shape)
    # disable text wrapping in output cell
    display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
    active_nodes = list(dag_model.nodes())  # pass only nodes with edges
    print("active_nodes: ", active_nodes)
    print("number of active_nodes:", len(active_nodes))
    df = df_input[active_nodes].astype('category')
    print("data size: ", df.shape)
    print("dag bn parameter estimation...")

    dag_param_model = None

    tObj = util.ProcTimer()

    if alg == "bayesian":
        prior_type = kwargs.get("prior_type", "BDeu")
        equivalent_sample_size = kwargs.get("equivalent_sample_size", 10)
        # supports nan values or unknown
        complete_samples_only = kwargs.get("complete_samples_only", False)

        tObj.restart()

        try:
            dag_param_model = BayesianModel(dag_model.edges())
            dag_param_model.cpds = []
            dag_param_model.fit(data=df,
                                estimator=BayesianEstimator,
                                prior_type=prior_type,
                                equivalent_sample_size=equivalent_sample_size,
                                # complete_samples_only=complete_samples_only,
                                n_jobs=1, # solves serialization issue on paraller processing
                                )
        except Exception as ex:
            print("Exception during BayesianModel fit: ", ex)
            dag_param_model = None
        
        tObj.display_proctime()

    return dag_param_model

def aggregate_time_tag_edges(edges_df):
    if 't' not in edges_df.columns:
        edges_df['t'] = 0
        return edges_df
    
    edges_df['t'] = edges_df['t'].astype(int)
    edges_agg_df = edges_df.groupby(by=["src", "dst"]).agg(
        {"t": list, "weight": sum}).reset_index()[["src", "dst", "weight", "t"]]
    return edges_agg_df

def plot_dag_graph_ntk(dag_skeleton_model, ntk_structure_alg_tag="", **kwargs):

    def node_color_mapper(var_name, isnotation=False):
        if isnotation:
            if ("T" in var_name):
                return "#97fa02"  # "lime"
            elif ("H" in var_name):
                return "#02f2fa"  # light blueish "lightgreen"
            elif ("C" in var_name):
                return "#fcba03"  # "orange"
            elif ("V" in var_name):
                return "#C70039"  # brown
            elif ("PWR" in var_name):
                return "#a302fa"  # "purple"
            elif ("RSSI" in var_name):
                return "#d203fc"  # pink "yellow"
            # else:
            #     return "blue"

        else:
            if ("TEMPERATURE" in var_name) or ("_TEMP_" in var_name):
                return "#97fa02"  # "lime"
            elif ("_RH_" in var_name) or ("HUMIDITY" in var_name):
                return "#02f2fa"  # light blueish "lightgreen"
            elif ("CURRENT" in var_name):
                return "#fcba03"  # "orange"
            elif ("BVIN_" in var_name) or ("VT_" in var_name) or ("VIN_" in var_name) or ("VDD_" in var_name) or ("VOLTAGE" in var_name) or ("BIASMON" in var_name):
                return "#C70039"  # brown
            elif ("POWER" in var_name):
                return "#a302fa"  # "purple"
            elif ("RSSI" in var_name):
                return "#d203fc"  # pink "yellow"
            # else:
            #     return "blue"

    print("plot_dag_graph_ntk...")
    notebook = kwargs.pop("notebook", False)
    select_menu = kwargs.pop("select_menu", False)
    filter_menu = kwargs.pop("filter_menu", False)
    physics = kwargs.pop("physics", True)
    arrowStrikethrough = kwargs.pop("arrowStrikethrough", True)
    show_buttons = kwargs.pop("show_buttons", False)
    height = kwargs.pop("height", '1000px')
    width = kwargs.pop("width", '100%')
    bgcolor = kwargs.pop("bgcolor", '#222222')
    font_color = kwargs.pop("font_color", 'white')
    directed = kwargs.pop("directed", True)
    label_edges = kwargs.pop("label_edges", False)
    isshow = kwargs.pop("isshow", True)
    issave = kwargs.pop("issave", False)
    filepath = kwargs.pop("filepath",None)
    isnotation = kwargs.pop("isnotation", False)
    nodefontsize = kwargs.pop("nodefontsize", 30)

    assert filepath is not None, "add graph store filedirpath"
                           
    print("Warning: input must be [source, target, weight, t]!")
    if not isinstance(dag_skeleton_model, (np.ndarray, pd.DataFrame)):
        ntk_edges = np.array(dag_skeleton_model.edges())
    elif isinstance(dag_skeleton_model, pd.DataFrame):
        ntk_edges = dag_skeleton_model.values
    else:
        ntk_edges = dag_skeleton_model

    net_dict = {
                "source": ntk_edges[:, 0],
                "target": ntk_edges[:, 1],
                "weight": [1]*len(ntk_edges) if ntk_edges.shape[1] < 3 else ntk_edges[:, 2],
                "t": [0]*len(ntk_edges) if ntk_edges.shape[1] < 4 else ntk_edges[:, 3]
                }
    net_df = pd.DataFrame.from_dict(net_dict)

    if (ntk_edges.shape[1] < 4) and ("__tlag" in " ".join(net_df["source"].values.tolist())):
        net_df[["source", "t"]] = net_df["source"].str.split(
            "__tlag", expand=True)
        net_df.loc[~net_df["t"].isna(), "t"] = net_df.loc[~net_df["t"].isna(), "t"].apply(
            lambda x: "[-{}]".format(int(x.strip("_"))))
        net_df["t"] = net_df["t"].fillna('[0]')

    print(net_df.head())
    got_net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color,
                                notebook=notebook,
                                directed=directed,
                                select_menu=select_menu,
                                filter_menu=filter_menu, **kwargs
                                )

    # set the physics layout of the network
    got_net.barnes_hut(gravity=-10000 if not isnotation else -5000,  # The more negative the gravity value is, the stronger the repulsion is.
                       central_gravity=0.001,
                       spring_length=400,
                       spring_strength=0.001,
                       damping=1.0,
                       overlap=0
                       )
    got_net.toggle_physics(physics)

    if show_buttons:
        got_net.show_buttons(filter_=True)  # display editing meanu

    sources = net_df['source']
    targets = net_df['target']
    weights = net_df['weight']
    edge_title = net_df['t']
    edge_data = zip(sources, targets, weights, edge_title)
    labeled_edges = []
    islabeled_rev_edge = False
    for i, e in enumerate(edge_data):
        src = e[0]
        dst = e[1]
        w = e[2]
        t = e[3]
        src_color = node_color_mapper(src, isnotation)
        dst_color = node_color_mapper(dst, isnotation)
        got_net.add_node(src, src, title=src, color=src_color)
        got_net.add_node(dst, dst, title=dst, color=dst_color)  # group=grp_id
        if label_edges:
            islabeled_rev_edge = (dst, src) in labeled_edges
            if not islabeled_rev_edge:
                labeled_edges.append((src, dst))

        got_net.add_edge(src, dst, value=w, color=src_color,
                         arrowStrikethrough=arrowStrikethrough,
                         title=f"causality from {src} to {dst} at time-lag (unit): {t}",
                         label="t={}".format(
                             t) if label_edges else None,
                         physics=islabeled_rev_edge
                         )
    neighbor_map = got_net.get_adj_list()
    # add neighbor data to node hover data
    for node in got_net.nodes:
        node['title'] = "Node: " + node['title'] + '\n' + '__'*20 + '\n Neighbors (Influenced by the Node): \n' + \
            '\n'.join(sorted(list(neighbor_map[node['id']])))
        node['value'] = len(neighbor_map[node['id']])
        node["font"] = {"size": nodefontsize}

    filepath_full = '{}.html'.format(
        os.path.join(*[filepath, ntk_structure_alg_tag]))

    def convert_pyvis_to_nx(pyvis_net):
        # Create an empty NetworkX graph
        nx_graph = nx.DiGraph()
        # Add nodes to the NetworkX graph
        for node in pyvis_net.nodes:
            node_id = node['id']
            # You can extract other attributes if needed, e.g., label, title, size, etc.
            # Be aware that pyvis might add default attributes not present in original NetworkX graph
            node_attributes = {key: value for key, value in node.items() if key != 'id'}
            nx_graph.add_node(node_id, **node_attributes)

        # Add edges to the NetworkX graph
        for edge in pyvis_net.edges:
            source = edge['from']
            target = edge['to']
            # Extract edge attributes like title, weight, etc.
            edge_attributes = {key: value for key, value in edge.items() if key not in ['from', 'to']}
            nx_graph.add_edge(source, target, **edge_attributes)
        return nx_graph

    if isshow:
        #  display on browser, interactive
        got_net.show(filepath_full, notebook=notebook)

        # displays on notebook, non-interactive
        G = convert_pyvis_to_nx(got_net)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, 
                # node_color='skyblue', 
                node_size=1500, 
                edge_color='gray', 
                font_size=10)
    else:
        if issave:
            got_net.save_graph(filepath_full)
            return filepath_full
        else:
            return got_net

def prepare_unrolled_timelag_data(df, edges_agg_df):
    edges_agg_df.head()
    edges_agg_df['t_lag'] = edges_agg_df['t'].astype(
        str).apply(lambda x: x.strip('[').strip(']').strip(' '))
    edges_agg_df.head()
    src_with_lags_df = edges_agg_df[['src', 't_lag']].groupby(
        by=['src']).agg({"t_lag": ','.join}).reset_index()

    src_with_lags_df

    src_with_lags_df['t_lag'] = src_with_lags_df['t_lag'].apply(
        lambda x: ','.join(set(x.strip(' ').split(","))))
    src_with_lags_df
    src_with_lags_df = src_with_lags_df.loc[src_with_lags_df['t_lag'] != '0']

    new_lagged_cols = []
    for _, row in tqdm(src_with_lags_df.iterrows()):
        # row
        src, t_lags = row.values
        src, t_lags
        for t_lag in t_lags.split(','):
            t_lag = t_lag.strip(' ')
            if t_lag == '0':
                continue

            t = int(t_lag)

            new_col = "{}__tlag_{:02d}".format(src, abs(t))
            # new_col
            # new_lagged_cols.append(new_col)
            df[new_col] = df[src].shift(periods=-t)
            df[[src, new_col]].head()
        # break

    new_lagged_cols = sorted(df.filter(regex="\w*__tlag_\w*").columns.tolist())
    return df, new_lagged_cols

def directing_tlag_0(df_aml_rca, bn_edges_df, significance_level=0.01, max_cond_vars=5):
    print("directed_tlag_0...")
    # use lower significance_level to avoid cyclic links
    bn_edges_df_directed = bn_edges_df.loc[bn_edges_df["t"] != 0].reset_index(drop=True, inplace=False)
    bn_edges_df_directed.shape
    bn_edges_df_undirected = bn_edges_df.loc[bn_edges_df["t"] == 0].reset_index(drop=True, inplace=False)
    # keep only bidirecetd edges at t=0
    bn_edges_df_undirected["isbidirected"] = False
    edges_arr = bn_edges_df_undirected[["src", "dst"]].apply(tuple, axis=1).values.tolist()
    for i, edge in bn_edges_df_undirected.iterrows():
            current_edge = tuple(edge.loc[["src", "dst"]].tolist())
            rev_directed_edge = tuple(edge.loc[["dst", "src"]].tolist())
            if rev_directed_edge in edges_arr:
                print(f"{rev_directed_edge} undirected_edge is detected at t==0 for edge {current_edge}")
                bn_edges_df_undirected.iloc[i, -1] = True
            else:
                print(f"{current_edge} is detected at t=0.")

    bn_edges_df_directed = pd.concat([bn_edges_df_directed, bn_edges_df_undirected.loc[~bn_edges_df_undirected["isbidirected"]]], axis=0, ignore_index=True)     
    bn_edges_df_directed.drop(columns=["isbidirected"], inplace=True)
    bn_edges_df_undirected = bn_edges_df_undirected.loc[bn_edges_df_undirected["isbidirected"]]  
    bn_edges_df_undirected.drop(columns=["isbidirected"], inplace=True)
    
    print("At t<=0 and directed: ", bn_edges_df_directed.shape)
    print(bn_edges_df_directed)
    print("At t=0 and undirected: ", bn_edges_df_undirected.shape)
    print(bn_edges_df_undirected)
    pc_est = pgmpy_PC(df_aml_rca[list(filter(lambda x: "tlag" not in x, df_aml_rca.columns))].astype("int8"))
    model_t_0 = pc_est.estimate(variant='stable', ci_test='chi_square',
                                max_cond_vars=max_cond_vars, 
                                significance_level=significance_level)
    G_pred_t_0 = nx.DiGraph()
    G_pred_t_0.add_nodes_from(df_aml_rca.columns)
    G_pred_t_0.add_edges_from(model_t_0.edges())

    edge_undirected_directed_df = pd.DataFrame(G_pred_t_0.edges(), columns=["src", "dst"])
    print("edge_undirected_directed_df: ")
    print(edge_undirected_directed_df)
    edge_undirected_directed_df = pd.merge(bn_edges_df_undirected, 
                                            edge_undirected_directed_df,
                                            how="inner",
                                            left_on=['src', 'dst'],
                                            right_on=['src', 'dst'],
                                            sort=True
                                        )
    print("edge_undirected_directed_df: ")
    print(edge_undirected_directed_df)
    bn_edges_directed_all_df = pd.concat([bn_edges_df_directed, edge_undirected_directed_df], axis=0, ignore_index=True)
    bn_edges_directed_all_df.shape

    print("Directed: ")
    print(bn_edges_directed_all_df)
    return bn_edges_directed_all_df

def prune_binary_ts_edges(edges_df: pd.DataFrame, keep_undirected_at_lagged_zero=False, **kwargs):
    """
    prunes undirected or looped links due to overlapped regions in  binary anomaly flags.
    It keeps the the edge with max lag
    """
    print("#"*80)
    weight_aware = kwargs.get("weight_aware", False)
    isweight_first = kwargs.get("isweight_first", False)
    use_weighted_tlag = kwargs.get("use_weighted_tlag", False)
    use_weighted_tlag_startlag = kwargs.get("use_weighted_tlag_startlag", 0)
    domain_edges = kwargs.get("domain_edges", None)
    verbose = kwargs.get("verbose", False)

    domain_status = domain_edges is not None
    print(f"prune_binary_ts_edges..., keep_undirected_at_lagged_zero: {keep_undirected_at_lagged_zero}, weight_aware: {weight_aware}, use_weighted_tlag:{use_weighted_tlag}")

    if not use_weighted_tlag:
        edges_most_lagged_df = edges_df.groupby(by=["src", "dst"]).agg(
            {"t": min, "weight": sum}).reset_index(drop=False)
    else:
        def get_tlag_maximum_weight(row):
            max_w_val = max(row["weight"])
            max_w_idx = row["weight"].index(max_w_val)
            return pd.Series([row["t"][max_w_idx], row["weight"][max_w_idx]], index=["t", "weight"])

        edges_most_lagged_df = edges_df.groupby(
            by=["src", "dst"]).agg({"t": list, "weight": list})
        edges_most_lagged_df[["t", "weight"]] = edges_most_lagged_df[[
            "t", "weight"]].apply(get_tlag_maximum_weight, axis=1)
        edges_most_lagged_df["t"] = edges_most_lagged_df["t"].astype("int8")
        edges_most_lagged_df = edges_most_lagged_df.reset_index(drop=False)
        edges_most_lagged_df["t"] = edges_most_lagged_df["t"].astype("int8")
        edges_most_lagged_df["weight"] = edges_most_lagged_df["weight"].round(6).astype("float")

    print("Agg edge df:")
    print(edges_most_lagged_df)

    edges_arr = edges_most_lagged_df[["src", "dst"]].apply(
        tuple, axis=1).values.tolist()
    prune_idx = []
    if isweight_first:
        for i, edge in edges_most_lagged_df.iterrows():
            current_edge = tuple(edge.loc[["src", "dst"]].tolist())
            undirected_edge = tuple(edge.loc[["dst", "src"]].tolist())
            # print(edge.tolist())
            if (i not in prune_idx) and (undirected_edge in edges_arr):
                if verbose: print(f"undirected_edge is detected! {undirected_edge}")
                sd_idx = i
                ds_idx = edges_arr.index(undirected_edge)
                t_sd = edges_most_lagged_df.iloc[sd_idx]["t"]
                t_ds = edges_most_lagged_df.iloc[ds_idx]["t"]
                w_sd = edges_most_lagged_df.iloc[sd_idx]["weight"]
                w_ds = edges_most_lagged_df.iloc[ds_idx]["weight"]
                # print(sd_idx, ds_idx, t_sd, t_ds, w_sd, w_ds)
                if weight_aware and (w_sd > w_ds):
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}") 
                        prune_idx.append(ds_idx)
                elif weight_aware and (w_sd < w_ds):
                    if ds_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                elif t_sd < t_ds:
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}")
                        prune_idx.append(ds_idx)
                elif t_sd > t_ds:
                    if ds_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                elif domain_edges is not None:
                    if not keep_undirected_at_lagged_zero:
                        if (current_edge in domain_edges) and (sd_idx not in prune_idx):
                            if verbose: print(
                                f"edge direction using domain edge applied. keep: {current_edge}")
                            prune_idx.append(ds_idx)
                        elif (undirected_edge in domain_edges) and (ds_idx not in prune_idx):
                            if verbose: print(
                                f"edge direction using domain edge applied. keep: {undirected_edge}")
                            prune_idx.append(sd_idx)
                        else:
                            # keep only one of the edges
                            if sd_idx not in prune_idx:
                                if verbose: print(f"edge direction keep: {current_edge}")
                                prune_idx.append(ds_idx)
                            else:
                                if verbose: print(
                                    f"edge direction keep: {undirected_edge}")
                                prune_idx.append(sd_idx)

                elif not keep_undirected_at_lagged_zero:
                    # keep only one of the edges
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}")
                        prune_idx.append(ds_idx)
                    else:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                else:
                    pass
                    # else:
                    #     #keep undirecetd if both are at zero
    else:
        for i, edge in edges_most_lagged_df.iterrows():
            current_edge = tuple(edge.loc[["src", "dst"]].tolist())
            undirected_edge = tuple(edge.loc[["dst", "src"]].tolist())
            # print(edge.tolist())
            if (i not in prune_idx) and (undirected_edge in edges_arr):
                if verbose: print(f"undirected_edge is detected! {undirected_edge}")
                sd_idx = i
                ds_idx = edges_arr.index(undirected_edge)
                t_sd = edges_most_lagged_df.iloc[sd_idx]["t"]
                t_ds = edges_most_lagged_df.iloc[ds_idx]["t"]
                w_sd = edges_most_lagged_df.iloc[sd_idx]["weight"]
                w_ds = edges_most_lagged_df.iloc[ds_idx]["weight"]
                # print(sd_idx, ds_idx, t_sd, t_ds)
                if t_sd < t_ds:
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}")
                        prune_idx.append(ds_idx)
                elif t_sd > t_ds:
                    if ds_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                elif weight_aware and (w_sd > w_ds):
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}")
                        prune_idx.append(ds_idx)
                elif weight_aware and (w_sd < w_ds):
                    if ds_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                elif domain_edges is not None:
                    if not keep_undirected_at_lagged_zero:
                        if (current_edge in domain_edges) and (sd_idx not in prune_idx):
                            if verbose: print(
                                f"edge direction using domain edge applied. keep: {current_edge}")
                            prune_idx.append(ds_idx)
                        elif (undirected_edge in domain_edges) and (ds_idx not in prune_idx):
                            if verbose: print(
                                f"edge direction using domain edge applied. keep: {undirected_edge}")
                            prune_idx.append(sd_idx)
                        else:
                            # keep only one of the edges
                            if sd_idx not in prune_idx:
                                if verbose: print(f"edge direction keep: {current_edge}")
                                prune_idx.append(ds_idx)
                            else:
                                if verbose: print(
                                    f"edge direction keep: {undirected_edge}")
                                prune_idx.append(sd_idx)

                elif not keep_undirected_at_lagged_zero:
                    # keep only one of the edges
                    if sd_idx not in prune_idx:
                        if verbose: print(f"edge direction keep: {current_edge}")
                        prune_idx.append(ds_idx)
                    else:
                        if verbose: print(f"edge direction keep: {undirected_edge}")
                        prune_idx.append(sd_idx)
                else:
                    pass

    if verbose: print("prune_idx: ", prune_idx)
    edges_most_lagged_df.drop(index=prune_idx, inplace=True)
    if verbose: print(edges_most_lagged_df)

    if not use_weighted_tlag:
        edges_raw_arr = edges_df[["src", "dst", "t"]].apply(
            tuple, axis=1).values.tolist()
        edges_directed_arr = edges_most_lagged_df[["src", "dst", "t"]].apply(
            tuple, axis=1).values.tolist()
        org_idx = [e in edges_directed_arr for e in edges_raw_arr]
        edges_most_lagged_df = edges_df.iloc[org_idx][[
            "src", "dst", "weight", "t"]]
        
    edges_most_lagged_df["src_lag"] = edges_most_lagged_df[[
        "src", "t"]].apply(lambda x: lagged_src_name(*x), axis=1)
    print(edges_most_lagged_df)
    print("#"*80)
    return edges_most_lagged_df

def query_report_cond_prob_infer(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)

    start_time = time.time()

    query_result = infer.query(variables=variables,
                               evidence=evidence,
                               elimination_order=elimination_order,
                               show_progress=show_progress)
    print(
        f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')
    print(query_result)
    return query_result

def query_report_causal_infer(infer, variables, do=None, show_progress=False, desc=""):
    if desc:
        print(desc)

    start_time = time.time()

    query_result = infer.query(variables=variables,
                               do=do,
                               show_progress=show_progress)
    print(
        f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')
    print(query_result)
    return query_result

def get_ordering(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    start_time = time.time()
    ordering = infer._get_elimination_order(variables=variables,
                                            evidence=evidence,
                                            elimination_order=elimination_order,
                                            show_progress=show_progress)
    if desc:
        print(desc, ordering, sep='\n')
        print(
            f'--- Ordering found in {time.time() - start_time:0,.4f} seconds ---\n')
    return ordering

def padding(heuristic):
    return (heuristic + ":").ljust(16)

def compare_all_ordering(infer, variables, evidence=None, show_progress=False):
    ord_dict = {
        "MinFill": get_ordering(infer, variables, evidence, "MinFill", show_progress),
        "MinNeighbors": get_ordering(infer, variables, evidence, "MinNeighbors", show_progress),
        "MinWeight": get_ordering(infer, variables, evidence, "MinWeight", show_progress),
        "WeightedMinFill": get_ordering(infer, variables, evidence, "WeightedMinFill", show_progress)
    }
    if not evidence:
        pre = f'elimination order found for probability query of {variables} with no evidence:'
    else:
        pre = f'elimination order found for probability query of {variables} with evidence {evidence}:'
    if ord_dict["MinFill"] == ord_dict["MinNeighbors"] and ord_dict["MinFill"] == ord_dict["MinWeight"] and ord_dict["MinFill"] == ord_dict["WeightedMinFill"]:
        print(f'All heuristics find the same {pre}.\n{ord_dict["MinFill"]}\n')
    else:
        print(f'Different {pre}')
        for heuristic, order in ord_dict.items():
            print(f'{padding(heuristic)} {order}')
        print()

def cond_prob_inference_report(dag_param_model, target_vars, observed_cond_dict={}, elimination_order="MinNeighbors", isprint=True):
    """
    Probabilistic Inference: Given a fully specified BN, the probabilistic inference algorithms allow users to query the model for any conditional distribution
    exact inference: VariableElimination
    """

    print("cond_prob_inference_report...")

    if not isinstance(target_vars, list):
        target_vars = [target_vars]

    # Variable elimination (VE) is a simple and general exact inference algorithm in probabilistic graphical models, such as Bayesian networks and Markov random fields. It can be used for inference of maximum a posteriori (MAP) state or estimation of conditional or marginal distributions over a subset of variables.
    # The elimination method is one of the most widely used techniques for solving systems of equations. Why? Because it enables us to eliminate or get rid of one of the variables, so we can solve a more simplified equation.

    bn_infer_engine = VariableElimination(dag_param_model)

    ordering = get_ordering(bn_infer_engine, variables=target_vars, elimination_order=elimination_order, evidence=observed_cond_dict,
                            desc=f'Elimination order for {target_vars} with no {observed_cond_dict} computed through {elimination_order} heuristic:')

    query_result = query_report_cond_prob_infer(bn_infer_engine, variables=target_vars, evidence=observed_cond_dict, elimination_order=ordering,
                                                desc=f'Probability of {target_vars} change with observed condition of {observed_cond_dict}:')

    return query_result

def causal_inference_report(dag_param_model, target_vars, observed_cond_dict={}):
    """
    return adjusted inference instead of a raw con prob as cond_prob_inference_report
    observed_cond_dict: holds observed evidence or condition e.g.{'SIPM__PELTIERVOLTAGE_F': 'E: +2.6895 to +2.6953'}

    Causal Inference: The causal inference module provides features to estimate the causal
    effect between a given exposure and an outcome variable. 
    """

    print("causal_inference_report...")

    if not isinstance(target_vars, list):
        target_vars = [target_vars]

    bn_infer_engine = CausalInference(dag_param_model)

    query_result = query_report_causal_infer(bn_infer_engine, variables=target_vars, do=observed_cond_dict,
                                             desc=f'Causality Probability of {target_vars} increase with observed condition of {observed_cond_dict}:')
    return query_result

def iscausal(dag_skeleton_model, cause_node, infuenced_node, observed=None):
    """
    Returns True if there is an active trail (i.e. d-connection) between start and end node given that observed is observed.
    check if cause_node and infuenced_node have comman cause
    """
    return dag_skeleton_model.is_dconnected(cause_node, infuenced_node, observed=observed)

def get_noncausality_scenarios(dag_skeleton_model, sel_node):
    return dag_skeleton_model.local_independencies(sel_node)

def get_all_causal_nodes(dag_skeleton_model, sel_node):
    return dag_skeleton_model.get_parents(sel_node)

def get_all_influenced_nodes(dag_skeleton_model):
    return dag_skeleton_model.get_leaves()

def get_all_reachable_nodes(dag_skeleton_model, sel_nodes, observed=None, include_latents=False):
    """
    # Active trail: For any two variables A and B in a network if any change in A influences the values of B then we say
    #               that there is an active trail between A and B.
    # In pgmpy active_trail_nodes gives a set of nodes which are affected (i.e. correlated) by any
    # change in the node passed in the argument.
    """
    if not isinstance(sel_nodes, list):
        sel_nodes = [sel_nodes]

    return dag_skeleton_model.active_trail_nodes(sel_nodes, observed=observed, include_latents=include_latents)

def get_influenced_dag(dag_skeleton_model, cause_nodes):
    """
    get_influenced_dag(x) or get_influenced_dag(X=x)
    Applies the do operator to the graph and returns a new DAG with the transformed graph.
    The do-operator, do(X = x) has the effect of removing all edges from the parents of X and setting X to the given value x.
    """
    return dag_skeleton_model.do(cause_nodes)

def get_causal_dag(dag_skeleton_model, infuenced_nodes):
    if not isinstance(infuenced_nodes, list):
        infuenced_nodes = [infuenced_nodes]
    return dag_skeleton_model.get_ancestral_graph(infuenced_nodes)

def build_cd_ntk(df, dag_skeleton_alg='hc', scoring_method='BDeuScore', dag_params_alg="bayesian", max_iter=100, isplot=False, isshow=False,
                  isfixed_mode=False, isblacklist_mode=False, blacklist_tags=[], isskeletonbn=False, **kwargs):
    print("build_cd_ntk...")
    print(kwargs)

    scoring_method_dicts = {'BDeuScore': BDeuScore,
                            'BicScore': BicScore, 'K2Score': K2Score}
    
    ntk_structure_alg_tag = "{}_{}".format(dag_skeleton_alg, scoring_method)
    scoring_method_func = scoring_method_dicts[scoring_method]
    node_varnames = df.columns.tolist()
    black_list = []
    if isblacklist_mode:
        for blk_tag in blacklist_tags:
            blacklist_vars = list(
                filter(lambda x: blk_tag in x, node_varnames))
            black_list.extend(
                list(itertools.product(blacklist_vars, blacklist_vars)))
        print("black_list: ", black_list)

    dag_skeleton_model = learn_scm_dag_structure(
        df, alg=dag_skeleton_alg, scoring_method=scoring_method_func, max_iter=max_iter, black_list=black_list, **kwargs)

    if isblacklist_mode:
        new_edges = dag_skeleton_model.edges()
        new_edges = list(set(new_edges).difference(set(black_list)))
        dag_skeleton_model_ = DAG()
        dag_skeleton_model_.add_nodes_from(nodes=node_varnames)
        dag_skeleton_model_.add_edges_from(ebunch=new_edges)
        dag_skeleton_model = dag_skeleton_model_

    edges_df = pd.DataFrame(dag_skeleton_model.edges(), columns=["src", "dst"])
    edges_df["t"] = 0
    edges_df["weight"] = 1
    edges_df["pval"] = 0
    edges_df["src_lag"] = edges_df["src"]

    if len(dag_skeleton_model.edges()) > 0:
        if isplot:
            grp_net = plot_dag_graph_ntk(
                dag_skeleton_model, ntk_structure_alg_tag=ntk_structure_alg_tag, isshow=isshow, **kwargs)
    return dag_skeleton_model, edges_df

def build_ts_cd_ntk(df, dag_skeleton_alg='pcmci', scoring_method='ParCorr', dag_params_alg="bayesian", max_iter=100, isplot=False, isshow=False, approach="fullts",
                     isfixed_mode=False, isblacklist_mode=False, blacklist_tags=[], isskeletonbn=False, isremove_self_lags=True, tau_min=0, tau_max=5, domain_edges=None, issparse_link=False, link_assume_thr=0, **kwargs):
    print("build_ts_cd_ntk...")
    print(kwargs)
    verbosity = kwargs.get('verbosity', 0)
    weight_sign = kwargs.get('weight_sign', 'both')
    isprune_undirected = kwargs.pop('isprune_undirected', False)
    weight_aware = kwargs.pop('weight_aware', True)
    use_weighted_tlag = kwargs.pop('use_weighted_tlag', True)
    ntk_structure_alg_tag = "{}".format(dag_skeleton_alg)
    link_assumptions = None
    print(f"tau_max={tau_max}, tau_min={tau_min}")
    if approach == "fullts" or (tau_min == 1):
        if isremove_self_lags:
            link_assumptions = remove_self_lag_causality(
                                                        df.shape[1], tau_max, tau_min=tau_min, 
                                                        isblacklist_mode=isblacklist_mode, 
                                                        blacklist_tags=blacklist_tags, 
                                                        vars_names=df.columns.tolist()
                                                        ) 
        if issparse_link:
            if link_assumptions is None:
                link_assumptions = remove_self_lag_causality(
                                                        df.shape[1], tau_max, tau_min=tau_min, 
                                                        isblacklist_mode=isblacklist_mode,
                                                        blacklist_tags=blacklist_tags, 
                                                        vars_names=df.columns.tolist()
                                                        ) 
                
            link_assumptions = get_sparse_link_assumptions(df, tau_max, link_assumptions, thr=link_assume_thr)

        edges_df = learn_scm_ts_dag_structure(
            df, alg=dag_skeleton_alg, max_iter=max_iter, link_assumptions=link_assumptions, tau_max=tau_max, tau_min=tau_min, **kwargs)
    else:
        print(f"approach: {approach} is not supported for build_ts_cd_ntk.")
        raise NotImplementedError

    if edges_df.empty:
        print("edges_df is empty!")
        dag_skeleton_model = DAG()
        dag_skeleton_model.add_nodes_from(nodes=df.columns.tolist())
        return dag_skeleton_model, edges_df
    else:
        edges_df = edges_df.loc[~np.isinf(edges_df["weight"].values), :]
    
    edges = edges_df[["src_lag", "dst"]].apply(tuple, axis=1).values.tolist()
    print('GCM time lag edges: ', edges)
    dag_skeleton_model = DAG()
    dag_skeleton_model.add_nodes_from(nodes=df.columns.tolist())
    dag_skeleton_model.add_edges_from(ebunch=edges)
    
    if len(dag_skeleton_model.edges()) > 0:
        if isplot:
            edges_agg_df = aggregate_time_tag_edges(edges_df)
            grp_net = plot_dag_graph_ntk(
                edges_agg_df.values, ntk_structure_alg_tag=ntk_structure_alg_tag, isshow=isshow, **kwargs)
    
    return dag_skeleton_model, edges_df
