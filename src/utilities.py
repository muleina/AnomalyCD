# =============================================================================
# AnomalyCD: Scalable Temporal Anomaly Causality Discovery in Large Systems
# =============================================================================
# Utilities module provides commonly shared functions and libraries across the different modules
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2022
# =============================================================================

import sys
import io
import copy
import re
import pickle
import json
import gc
import psutil
import os
import torch
import time
import itertools
import functools
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal, stats
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from claspy.window_size import dominant_fourier_frequency, highest_autocorrelation, suss
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticke
import seaborn as snsr
from pyvis.network import Network
from bokeh.palettes import Category10_10
from tqdm import tqdm
import sranodec_local as sr
import warnings
warnings.filterwarnings("ignore")

sns.set()
plt.style.use("seaborn-v0_8-whitegrid")
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 800)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 300)
pd.options.plotting.backend = 'matplotlib'
FIGURE_HEIGHT = 3
FIGURE_WIDTH = 4.5
FIGURE_LBL_FONTSIZE_MAIN = 16
FIGURE_SAVEFORMAT = ".jpg"

valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if
                  item[1] != 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])
colors = ['b', 'g', 'r', 'c', 'm',  'y', 'k']
filled_markers = ('o', 'v', 'X', '^', 's', '<', '+', '>',
                  'd', '8', 'p', 'h', 'H', 'D', 'P', '*')
linestyles = ('-', '--', '-', '--', '-', '--', '-',
              '--', '-', '--', '-', '--', '-', '--', '-')

class ProcTimer():
    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.proc_time = 0
        self.time_status = True

    def restart(self):
        self.start_time = time.time()
        self.end_time = 0
        self.time_status = True

    def stop(self):
        self.time_status = False
        self.end_time = time.time()
        self.proc_time = self.end_time - self.start_time

    def get_proctime(self, time_format="s"):
        if self.time_status:
            # self.end_time = time.process_time()
            self.end_time = time.time()
            self.proc_time = self.end_time - self.start_time

        if time_format == "s":
            return self.proc_time

    def display_proctime(self, time_format="s"):
        print("process time: {} seconds.".format(
            self.get_proctime(time_format=time_format)))


def sort_columns(df, isreverse=False):
    return df.reindex(sorted(df.columns, reverse=isreverse), axis=1)

def drop_suffix(vars_rbx_sel, suffix=""):
    return list(map(lambda x: x.replace(suffix, ""), vars_rbx_sel))

def remove_prefix_suffix(columns, direction="suffix"):
    if direction == "suffix":
        return np.array(columns.str.split("__").tolist())[:, 0]
    elif direction == "prefix":
        return np.array(columns.str.split("__").tolist())[:, 0]
    else:
        return np.array(columns.str.split("__").tolist())[:, 1]

def removeaffix(s, affix):
    if s is None:
        return ""
    return removesuffix(removeprefix(s, affix), affix)

def removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s[:]
    
def removesuffix(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s[:]

def ts_data_reconstructor(df, timestep=None, t_idx="column", **kwargs):
    print("ts_data_reconstructor...")

    num_missing_seq_limit = kwargs.get("num_missing_seq_limit", None)
    fillna_method = kwargs.get(
        "fillna_method", "keeplast")  # keeplast, rolling
    rolling_window = kwargs.get("rolling_window", None)
    agg_func = kwargs.get("agg_func", None)

    if rolling_window is None:
        rolling_window = num_missing_seq_limit

    print("time resolution adjustment timestep: ", timestep)

    if df.empty:
        raise "no data in df is empty!"

    if not timestep:
        return df

    if t_idx == "column":
        df = df.set_index("t")

    try:
        #  .ffill() has single fill while .asfreq().ffill() fills all gaps, .asfreq() along make jumps if extact stamp is not hit
        # None == None to avoid non None is and non zero are different

        if agg_func is None:
            _df = df.resample(timestep)
        elif agg_func == "mean":
            _df = df.resample(timestep, label='right').mean()
            # return df.resample(timestep, label='right').mean()
        elif agg_func == "sum":
            _df = df.resample(timestep, label='right').sum()
        elif agg_func == "max":
            _df = df.resample(timestep, label='right').max()
        elif callable(agg_func):
            _df = df.resample(timestep, label='right').apply(agg_func)
        else:
            raise "Undefined resampler agg_func!"

        if fillna_method == "keeplast":
            if (num_missing_seq_limit is None) or (num_missing_seq_limit > 0):
                return _df.ffill(limit=num_missing_seq_limit).iloc[1:, :]
            else:
                # return df.resample(timestep).ffill(limit=1).iloc[1:, :]
                return _df.iloc[1:, :]

        elif fillna_method == "rolling":
            if num_missing_seq_limit > 0:
                # min_periods=1 enables num_missing_seq_limit-1 in a sliding window with nan values
                _df = _df.fillna(_df.rolling(rolling_window, min_periods=1).median(
                ), limit=num_missing_seq_limit).iloc[1:, :]
            else:
                return _df.iloc[1:, :]

    except Exception as ex:
        raise ex

def plot_graph(net_df, filename, **kwargs):

    def node_color_mapper(var_name):
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
        else:
            return "#03a5fc"

    height = kwargs.pop("height", '1000px')
    width = kwargs.pop("width", '100%')
    bgcolor = kwargs.pop("bgcolor", '#222222')
    font_color = kwargs.pop("font_color", 'white')
    directed = kwargs.pop("directed", False)
    filepath = kwargs.pop("filepath", result_path)
    notebook = kwargs.pop("notebook", False)
    select_menu = kwargs.pop("select_menu", False)
    filter_menu = kwargs.pop("filter_menu", False)
    physics = kwargs.pop("physics", True)
    isshow = kwargs.pop("isshow", False)
    issave = kwargs.pop("issave", False)
    arrowStrikethrough = kwargs.pop("arrowStrikethrough", True)
    print(type(net_df))

    if not isinstance(net_df, (np.ndarray, pd.DataFrame)):
        ntk_edges = np.array(net_df.edges())
    elif isinstance(net_df, pd.DataFrame):
        ntk_edges = net_df.values
        print(net_df.columns)
    else:
        ntk_edges = net_df
    net_dict = {
        "source": ntk_edges[:, 0],
        "target": ntk_edges[:, 1],
        "weight": [1]*len(ntk_edges) if ntk_edges.shape[1] < 3 else ntk_edges[:, 2],
        "t": [0]*len(ntk_edges) if ntk_edges.shape[1] < 4 else ntk_edges[:, 3]
    }
    net_df = pd.DataFrame.from_dict(net_dict)
    print(net_df.head())

    got_net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color,
                      directed=directed,
                      notebook=notebook,
                      select_menu=select_menu,
                      filter_menu=filter_menu, **kwargs
                      )
    # set the physics layout of the network
    got_net.barnes_hut()
    got_net.toggle_physics(physics)
    sources = net_df['source']
    targets = net_df['target']
    weights = net_df['weight']
    edge_title = net_df['t']
    edge_data = zip(sources, targets, weights, edge_title)
    for e in edge_data:
        # print(e)
        src = e[0]
        dst = e[1]
        w = e[2]
        t = e[3]
        src_color = node_color_mapper(src)
        dst_color = node_color_mapper(dst)

        got_net.add_node(src, src, title=src, color=src_color)
        got_net.add_node(dst, dst, title=dst, color=dst_color)  # group=grp_id

        if directed:
            got_net.add_edge(src, dst, value=w, color=src_color, arrowStrikethrough=arrowStrikethrough,
                             title=f"Link b/n {src} and {dst} at time-lag (unit): {t}")
        else:
            if w > 0:
                got_net.add_edge(src, dst, value=w,
                                 color="#03a5fc",
                                 arrowStrikethrough=arrowStrikethrough,
                                 title=f"Link b/n {src} and {dst} at time-lag (unit): {t}")
            else:
                got_net.add_edge(src, dst, value=w, color="red",  arrowStrikethrough=arrowStrikethrough,
                                 title=f"Link b/n {src} and {dst} at time-lag (unit): {t}")
    neighbor_map = got_net.get_adj_list()
    # add neighbor data to node hover data
    for node in got_net.nodes:
        node['title'] = "Node: " + node['title'] + '\n' + '__'*20 + '\n Neighbors (Linked to the Node): \n' + \
            '\n'.join(sorted(list(neighbor_map[node['id']])))
        node['value'] = len(neighbor_map[node['id']])
    filepath_full = '{}.html'.format(os.path.join(*[filepath, filename]))
    if isshow:
        got_net.show(filepath_full, notebook=notebook)
    else:
        if issave:
            got_net.save_graph(filepath_full)
            return filepath_full
        else:
            return got_net

def plot_grid(input_data_list, label="error", **kwargs):
    """
    Plot grid of time series data.
    """
    
    rbx_sel = kwargs.get("rbx_sel", "")
    ncol = kwargs.get("n_grid_col", 2)
    ncol = kwargs.get("ncol", 2)
    ncol_force = kwargs.get("ncol_force", False)
    use_timestamp = kwargs.get("use_timestamp", False)
    isreset_index = kwargs.get("isreset_index", True)
    kind = kwargs.get("kind", "line")
    y_gridno = kwargs.get("y_gridno", 6)
    x_gridno = kwargs.get("x_gridno", 6)
    force_y_gridno =  kwargs.get("force_y_gridno", False)
    color = kwargs.get("color", 'r')
    overlap_df = kwargs.get("overlap_df", pd.DataFrame())
    join_df = kwargs.get("join_df", pd.DataFrame())
    join_format = kwargs.get("join_format", "marker")
    marker_size = kwargs.get("marker_size", 20)
    color_join = kwargs.get("color_join", "red")
    iscolor_per_col = kwargs.get("iscolor_per_col", False)
    sep = kwargs.get("sep", {})
    issave = kwargs.get("issave", False)
    filename = kwargs.get("filename", "grid_plot")
    filepath = kwargs.get("filepath", None)
    dpi = kwargs.get("dpi", 300)
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    legends = kwargs.get("legends", ["signal", "ref"])
    isshow = kwargs.get("isshow", True)
    wspace = kwargs.get("wspace", 0.1)
    hspace = kwargs.get("hspace", 0.2)
    add_axes_sub = kwargs.get("add_axes_sub", False)
    fontsize = kwargs.get("fontsize", 12)
    labelfontsize = kwargs.get("labelfontsize", 12)
    tickfontsize = kwargs.get("tickfontsize", 12)
    legendfontsize = kwargs.get("legendfontsize", 12)
    x_minticks = kwargs.get("x_minticks", 3)
    legend_idxs = kwargs.get("legend_idxs", None)
    input_data_ref = pd.DataFrame()  # rec signal
    if isinstance(input_data_list, list):
        input_data = input_data_list[0].copy()
        input_data_ref = input_data_list[1].copy()
    else:
        input_data = input_data_list.copy()

    if isinstance(input_data, pd.Series):
        input_data = input_data.to_frame()

    if isinstance(input_data_ref, pd.Series):
        input_data_ref = input_data_ref.to_frame()

    if input_data.empty:
        # raise ValueError("plot_grid: input_data can not be empty!")
        return print("plot_grid: input_data can not be empty!")

    input_data.columns.name = None
    input_data_ref.columns.name = None
    input_data_ref_use = not input_data_ref.empty
    isoverlap = not overlap_df.empty
    isjoin = not join_df.empty

    if (not use_timestamp) and isreset_index:
        input_data.reset_index(drop=True, inplace=True)
        if input_data_ref_use:
            input_data_ref.reset_index(drop=True, inplace=True)
        if isoverlap:
            overlap_df.reset_index(drop=True, inplace=True)

    if xlabel is None:
        xlabel = input_data.index.name

    t_s = time.time()

    if kind == "line":
        figsize = kwargs.get("figsize", (15, 3))
        fig_w, fig_h = figsize
        column_names = input_data.columns.tolist()
        scale = (int(isoverlap)+1)
        n = scale*len(column_names)
        ncol = 2 if isoverlap else ncol
        overlap_split = 2 if isoverlap else 1
        fig_cols = ncol if ncol_force else np.min([int(np.sqrt(n)), ncol])
        fig_rows = int(np.ceil(n/fig_cols))
        grid_h = fig_h*fig_rows
        grid_w = fig_w*fig_cols
        num_major_ticks = grid_w//4
        num_minor_ticks = 5

        fig, ax = plt.subplots(figsize=(grid_w, grid_h), ncols=fig_cols,
                               nrows=fig_rows, sharex=False if add_axes_sub else True, constrained_layout=True)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        if n == 1:
            ax = [ax]
        else:
            ax = trim_axs(ax, n)

        var_idx = 0
        colors = ["gray", "gray", "red", "red", "red", "blue"]
        color_step = 2
        color_step = max(color_step, 1)
        col_colors = Category10_10[::color_step]

        for i in range(fig_rows):
            for j in range(fig_cols):
                if iscolor_per_col:
                    color = col_colors[j]

                if var_idx > n-1:
                    break

                ishide_xaxis = (i < fig_rows-1) and (var_idx <
                                                     n-1) and (not add_axes_sub)
                var_sel = column_names[var_idx//overlap_split]

                if j % overlap_split == 0:
                    ax_sub = plt.subplot(fig_rows, fig_cols, var_idx+1)
                    plt.title("{}".format(var_sel.replace(
                        rbx_sel, "")), fontsize=fontsize)
                    islegend_active = legends[0] if legend_idxs is None else (legends[0] if var_idx in legend_idxs else None)
                    plt.plot(input_data[var_sel].index, input_data[var_sel].values,
                             color=color, label=legends[0] if legend_idxs is None else (legends[0] if var_idx in legend_idxs else None))

                    if input_data_ref_use:
                        plt.plot(input_data_ref[var_sel].index, input_data_ref[var_sel].values,
                                 color="gray", label=legends[1] if legend_idxs is None else (legends[1] if var_idx in legend_idxs else None))

                        if islegend_active is not None: plt.legend(fontsize=legendfontsize)
                        
                    plot_sep(sep=sep)

                    if isjoin:
                        if join_format == "line":
                            plt.plot(
                                join_df[var_sel].index, join_df[var_sel].values, color=color_join, label=legends[1] if legend_idxs is None else (legends[1] if var_idx in legend_idxs else None))
                        elif join_format == "marker":
                            plt.plot(input_data[var_sel].index, input_data[var_sel].mask(
                                ~join_df[var_sel], np.nan).values, color=color_join, linewidth=2, label=legends[1] if legend_idxs is None else (legends[1] if var_idx in legend_idxs else None))
                        elif join_format == "scatter":
                            plt.scatter(input_data[var_sel].index.values, input_data[var_sel].mask(
                                ~join_df[var_sel], np.nan), color=color_join, s=marker_size, label=legends[1] if legend_idxs is None else (legends[1] if var_idx in legend_idxs else None))
          
                        if islegend_active is not None: plt.legend(fontsize=legendfontsize)


                    if use_timestamp:
                        if input_data.index.dtype == "<M8[ns]":
                            locator = mdates.AutoDateLocator(
                                minticks=min(x_minticks, x_gridno), maxticks=x_gridno)
                            formatter = mdates.ConciseDateFormatter(locator)
                            ax_sub.xaxis.set_major_locator(locator)
                            ax_sub.xaxis.set_major_formatter(formatter)
                        else:
                            locator = ticker.MaxNLocator(x_gridno)
                            ax_sub.xaxis.set_major_locator(locator)
                    else:
                        locator = ticker.MaxNLocator(x_gridno)
                        ax_sub.xaxis.set_major_locator(locator)

                    if input_data[var_sel].dtype == "bool":
                        input_data[var_sel] = input_data[var_sel].astype(
                            "int8")

                    if input_data[var_sel].dtype in ("int", "float"):
                        nunique = input_data[var_sel].nunique()
                        if nunique > y_gridno or force_y_gridno:
                            plt.locator_params(
                                tight=True, axis="y", nbins=y_gridno)

                        elif input_data[var_sel].max() - input_data[var_sel].min() > y_gridno:
                            plt.locator_params(
                                tight=True, axis="y", nbins=y_gridno)
                        else:
                            plt.locator_params(
                                tight=True, axis="y", nbins=nunique)
                        
                    if ishide_xaxis:
                        plt.xlabel(None)
                        ax_sub.tick_params(axis='x', colors='white')
                    else:
                        plt.xlabel(xlabel, fontsize=labelfontsize)

                    if var_idx%fig_cols==0:
                        plt.ylabel(ylabel, fontsize=labelfontsize)

                    plt.xticks(fontsize=tickfontsize)
                    plt.yticks(fontsize=tickfontsize)

                    var_idx += 1
                else:
                    if isoverlap:
                        ax_overlap = plt.subplot(fig_rows, fig_cols, var_idx+1)
                        plt.plot(
                            overlap_df[var_sel].index, overlap_df[var_sel].values, color="darkviolet")
                        plt.title("{}".format(var_sel.replace(
                            rbx_sel, "")), fontsize=fontsize)

                        if overlap_df[var_sel].dtype == "bool":
                            overlap_df[var_sel] = overlap_df[var_sel].astype(
                                "int8")

                        if overlap_df[var_sel].dtype in ("int", "float") or force_y_gridno:
                            nunique = overlap_df[var_sel].nunique()
                            if nunique > y_gridno:
                                locator = ticker.MaxNLocator(y_gridno)
                                ax_sub.yaxis.set_major_locator(locator)
                            else:
                                plt.locator_params(tight=True, axis="y", nbins=nunique)

                        if ishide_xaxis:
                            plt.xlabel(None)
                            ax_overlap.tick_params(axis='x', colors='white')
                        else:
                            plt.xlabel(overlap_df.index.name,
                                       fontsize=labelfontsize)
                        plt.xticks(fontsize=tickfontsize)
                        plt.yticks(fontsize=tickfontsize)
                        var_idx += 1
                        plot_sep(sep=sep)
        if not isshow:
            return fig, ax

        if issave:
            save_figure(filename, fig, filepath=filepath,
                        isshow=False, issave=True, dpi=dpi)
        else:
            plt.show(block=False)

def trim_axs(axs, n):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    # print(n)
    try:
        axs = axs.flat
        for ax in axs[n:]:
            ax.remove()
        return axs[:n]
    except:
        if n == 1:
            return [axs] if not isinstance(axs, list) else axs

def analyze_missing(df):
    print("number of nan values...")
    isnan = df.isna()
    nan_sum = isnan.sum(axis=0)
    nan_sum = nan_sum.to_frame()
    nan_sum.columns = ["nan_size"]
    nan_sum["data_size"] = df.shape[0]
    nan_sum["nan_size_perc"] = 100 * \
        nan_sum["nan_size"].divide(nan_sum["data_size"])
    print(nan_sum.loc[nan_sum["nan_size"] > 0, :])
    return nan_sum, isnan

def get_corr(df_input, use_cat=False, cat_cols=[], isfeature_matrix=False, affix="", **kwargs):
    isdropna = kwargs.pop("isdropna", True)

    print("df.shape:", df_input.shape)
    df = df_input.copy()
    df = df.reset_index(drop=True).replace(
        [np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = df.columns.str.strip(affix)
    if not isfeature_matrix:
        if use_cat:
            print("categorical variables: {}".format(cat_cols))
            return nominal.associations(df, nominal_columns=list(cat_cols), return_results=True, mark_columns=True, theil_u=True, figsize=(20, 20))
        else:
            print("df.shape:", df.shape)
            return df.corr(method='pearson', min_periods=np.min([df.shape[0], 100000]))
    else:
        df_norm = df.T.divide(df.T.median(
            numeric_only=True).apply(lambda x: x if x > 0 else 1))
        # neg- smaller is weaker
        return -pd.DataFrame(euclidean_distances(df_norm, df_norm), columns=df.columns, index=df.columns)

def estimate_signal_period(data_np, alg="ACF"):
    """
    @params data_np: 1d numpy array
    @params alg: period estimation algorithms. choose from FFT, ACF and SuSS
    """

    print("estimate_signal_period...")

    tw_estimator_dict = {
        "FFT": dominant_fourier_frequency,
        "ACF": highest_autocorrelation,
        "SuSS": suss
    }

    return tw_estimator_dict[alg](data_np)

def trend_drift_detection(ds, var="sensor", **kwargs):

    def clean_slowprogressing(x, thr):
        # print(x, thr)
        # return np.zeros_like(x) if np.nanstd(x) <= istrend_thr_small_med else x
        return 0 if np.nanstd(x.values) <= thr else x.values[len(x)//2]

    print("trend_drift_detection...", kwargs)

    ts = time.time()
    smoothing_kernel_size = kwargs.get("smoothing_kernel_size", 5)

    istrend_thr = kwargs.get("istrend_thr", 10)
    istrend_small_thr = kwargs.get("istrend_small_thr", 0.0001)
    reading_accuray_error_thr = kwargs.get("istrend_small_thr", 0.02) # SH21S Accuracy	±0.3 °C, ±2%RH
    drift_thr = kwargs.get("drift_thr", 5)
    drift_dxn = kwargs.get("drift_dxn", 'both')
    trend_alg = kwargs.get("trend_alg", "linearconv")
    # true past and future, centered filter. False past only
    istrend_twosided = kwargs.get("istrend_twosided", True)
    trend_period = kwargs.get("trend_period", 10)
    isremove_cycle = kwargs.get("isremove_cycle", False)
    drift_score_window = kwargs.get("drift_score_window", 0)
    drift_score_smoothingfilter_window = kwargs.get(
        "drift_score_smoothingfilter_window", 0)
    user_data_cp_idx = kwargs.get("user_data_cp_idx", None)

    use_trend_period_auto = trend_period is None

    drift_tol_scaler = kwargs.get("drift_tol_scaler", 1)

    if drift_tol_scaler is None:
        _var = var.lower()
        drift_tol_scaler = 5 if ("humidity" in _var) or ("rh" in _var) else 1

    if isinstance(ds, pd.Series):
        _df = ds.to_frame()
        _df.columns = [var]
    else:
        _df = ds
        var = _df.columns[0]

    print(var)

    _df.index = pd.DatetimeIndex(_df.index)
    _df.sort_index(inplace=True)

    print(_df.shape)
    isnan_idx = _df[var].isna()
    print("Nan records: ", isnan_idx.sum(axis=0))
    _df_idx_bak = _df.index
    _df.dropna(axis=0, inplace=True)
    print(_df.shape)

    init_resid = np.zeros_like(_df[var].values)
    if smoothing_kernel_size > 0:
        _filtered_data = signal.medfilt(
            _df[var].values, kernel_size=smoothing_kernel_size)
        init_resid = _df[var] - _filtered_data
        _df[var] = _filtered_data
    
    if use_trend_period_auto:
        trend_period = estimate_signal_period(_df[var].values, "ACF")
        print(f"trend_period: {trend_period}")

    if trend_alg == "linearconv":
        decomp_obj = sm.tsa.seasonal_decompose(
            x=_df[var], model='additive', extrapolate_trend='freq', period=trend_period, two_sided=istrend_twosided)
        _df["trend"] = decomp_obj.trend
        _df["seasonal"] = decomp_obj.seasonal
        _df["resid"] = decomp_obj.resid

    elif trend_alg == "stl":
        # smoother trend
        decomp_obj = sm.tsa.seasonal.STL(
            x=_df[var],
            period=trend_period).fit()
        _df["trend"] = decomp_obj.trend
        _df["seasonal"] = decomp_obj.seasonal
        _df["resid"] = decomp_obj.resid

    elif trend_alg == "hpf":
        # The Hodrick-Prescott smoothing filter
        """
        This filter is mainly useful in removing the cyclic component from time-series data. Applying the Hodrick–Prescott filter in time series allows us to obtain a smooth time series from time series that has time series components like trend cycle and noise in large quantities
        lamb: The Hodrick-Prescott smoothing parameter. A value of 1600 is suggested for quarterly data. Ravn and Uhlig suggest using a value of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly data.
        """
        gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(_df[var], lamb=1600)
        decomp_obj = pd.Dataframe()
        decomp_obj["trend"] = gdp_trend
        decomp_obj["resid"] = gdp_cycle
        _df["trend"] = decomp_obj.trend
        _df["seasonal"] = 0
        _df["resid"] = decomp_obj.resid

    if not isremove_cycle: 
        _df["resid"] = _df["resid"] + _df["seasonal"]

    _df["resid"] = _df["resid"] + init_resid
    trend = _df["trend"]
    detrended_ae = (_df[var] - trend).abs().values
    detrended_med = np.nanmedian(detrended_ae[detrended_ae>0])
    detrended_med = detrended_med if detrended_med else 1
    istrend_thr_med = istrend_thr*detrended_med
    istrend_thr_small_med = istrend_small_thr*detrended_med
    print(f"detrended_med: {detrended_med}, istrend_thr: {istrend_thr}, istrend_thr_med: {istrend_thr_med}")
    d_trend = trend.diff().fillna(0).values

    if drift_dxn == 'up':
        d_trend_reset = d_trend > istrend_thr_med
    elif drift_dxn == 'down':
        d_trend_reset = (-d_trend) > istrend_thr_med
    else:
        d_trend_reset = np.abs(d_trend) > istrend_thr_med

    if user_data_cp_idx is not None:
        for idx in user_data_cp_idx:
            start_idx = _df["trend"].index[_df["trend"].index > idx].values[0]
            print(idx, start_idx)
            d_trend_reset[_df["trend"].index == start_idx] = True

    d_trend_reset = d_trend_reset.astype('int8')
    trend_drift_score = np.zeros(d_trend.shape)
    idx = 0
    trend_chunk = None
    for k, g in itertools.groupby(d_trend_reset):
        g_size = len(list(g))
        idx += g_size
        if (k == 0):
            trend_chunk_idx = [idx-g_size, idx]
            trend_chunk = d_trend[trend_chunk_idx[0]:trend_chunk_idx[1]]
            trend_drift_score[trend_chunk_idx[0]:trend_chunk_idx[1]] = signal.medfilt(trend_chunk, kernel_size=drift_score_smoothingfilter_window).cumsum(
            ) if drift_score_smoothingfilter_window > len(trend_chunk) else trend_chunk.cumsum()
            # avoid small dips or ups due to operation change in the LHC such as collision
            trend_chunk_drift_score = trend_drift_score[trend_chunk_idx[0]:trend_chunk_idx[1]]
            trend_drift_score_peak = np.max(
                np.abs(trend_chunk_drift_score))
            if trend_drift_score_peak < istrend_thr_small_med:  # istrend_too_small
                trend_drift_score[trend_chunk_idx[0]:trend_chunk_idx[1]] = 0

    n_nan = len(isnan_idx[~isnan_idx])
    missing_data_scaler = len(isnan_idx)/n_nan if n_nan > 0 else 1
    print(f"drift_tol_scaler:{drift_tol_scaler}, missing_data_scaler: {missing_data_scaler}, detrended_med: {detrended_med}")

    _df[var+"_DRIFT_OL_SCORE"] = missing_data_scaler * \
                                    np.abs(trend_drift_score)/(drift_tol_scaler*detrended_med)
    _df[var+"_DRIFT_OL_LBL"] = _df[var+"_DRIFT_OL_SCORE"] > drift_thr
    _df[var+"_TREND"] = trend
    _df = _df.reindex(_df_idx_bak, axis=0)
    print(f"processing time: {time.time()-ts} secs.")
    return _df, decomp_obj

def get_local_outliers(ts_df, method="stat_sigma", **kwargs):
    def dropna_with_idx(df, isnan_idx):
        df_idx_bak = df.index
        num_nans = isnan_idx.sum(axis=0)
        if num_nans > 0:
            print(df.shape)
            print(f"dropping {num_nans} nans...")
            df.dropna(axis=0, inplace=True)
            print(df.shape)
        return df, df_idx_bak
    
    # anormalies along columns per single variables
    isplot = kwargs.get("isplot", False)
    sr_th = kwargs.get("sr_th", 5)
    sd_th = kwargs.get("sd_th", 3)
    sd_zero_adjust_k = kwargs.get("sd_zero_adjust_k", 0.05)
    istrend_thr = kwargs.get("istrend_thr", 10)
    drift_thr = kwargs.get("drift_thr", 5)
    trend_period = kwargs.get("trend_period", 10)
    istrend_twosided = kwargs.get("istrend_twosided", True)
    isremove_cycle = kwargs.get("isremove_cycle", False)
    mes_smoothing_level = kwargs.get("smoothing_level", 0.2)
    issliding_window = kwargs.get("issliding_window", False)
    sliding_window = kwargs.get("sliding_window", 10)
    use_quantile = kwargs.get("use_quantile", [])
    isdropna = kwargs.get("isdropna", False)
    user_data_cp_idx = kwargs.get("user_data_cp_idx", None)
    
    use_sliding_window_auto = isinstance(sliding_window, str)
    use_trend_period_auto = trend_period is None
    sliding_min_period = kwargs.get(
        "sliding_min_period", max(sliding_window//3 if not use_sliding_window_auto else 0, 5))
    tw_estimator_dict = {
        "FFT": dominant_fourier_frequency,
        "ACF": highest_autocorrelation,
        "SuSS": suss
    }
    if not use_sliding_window_auto:
        if sliding_min_period > sliding_window:
            sliding_min_period = sliding_window
    else:
        sliding_window_alg = tw_estimator_dict[sliding_window]
        print("sliding_window_alg: ", sliding_window_alg)

    print("local outlier detection: {} ...".format(method))
    variables = ts_df.columns
    stat_df = None
    df = pd.DataFrame(index=ts_df.index)

    elif method == "stat_sigma":
        stat_df = ts_df[ts_df > 0].describe().T
        print(stat_df)
        stat_df["median"] = stat_df["50%"]
        # stat_df[stat_df["std"]==0, "std"] = 1
        stat_df.loc[stat_df["std"] == 0, "std"] = sd_zero_adjust_k * \
            stat_df.loc[stat_df["std"] == 0, "median"]
        stat_df = stat_df.T
        for var in tqdm(variables):
            df[var+"_OL_SCORE"] = np.abs(ts_df[var] -
                                         stat_df.loc["median", var])/stat_df.loc["std", var]
            if sd_th > 1:
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
            else:
                sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

    elif method == "msd":
        print("Moving Average Standard Deviation (SD)...")

        for var in tqdm(variables):
            _df = ts_df[[var]]

            if isdropna:
                isnan_idx = _df[var].isna()
                _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)
                
            if user_data_cp_idx is None:
                user_data_cp_idx = [_df.index[0]]

            decomp_mu = pd.Series(index=_df.index)
            decomp_std = pd.Series(index=_df.index)

            prev_i_start = None
            for i, idx in enumerate(user_data_cp_idx + [_df.index[-1]]):
                start_idx = _df.index[_df.index >= idx].values[0]
                if i == 0:
                    i_start = 0
                    i_end = _df.index.get_loc(start_idx)
                else:
                    i_start = prev_i_start
                    i_end = _df.index.get_loc(start_idx)

                if i_start >= i_end:
                    continue

                prev_i_start = i_end

                resid_ = _df[var].iloc[i_start:i_end]
                print(resid_.shape)

                if use_sliding_window_auto:
                    sliding_window = sliding_window_alg(resid_.values)
                    # sliding_window = estimate_signal_period(resid_.values, "ACF")
                    sliding_min_period = max(sliding_window//3, 5)
                    print("sliding_window: ", sliding_window)

                if len(use_quantile):

                    decomp_q1 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                        use_quantile[0], interpolation='midpoint')
                    decomp_q3 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                        use_quantile[1], interpolation='midpoint')

                    resid_.loc[(resid_ < decomp_q1) | (
                        resid_ > decomp_q3)] = np.nan
                    decomp_mu_ = resid_.rolling(
                        window=sliding_window, min_periods=sliding_min_period).median()
                    decomp_std_ = resid_.rolling(
                        window=sliding_window, min_periods=sliding_min_period).std()
                else:
                    decomp_mu_ = resid_.rolling(
                        window=sliding_window, min_periods=sliding_min_period).median()
                    decomp_std_ = resid_.rolling(
                        window=sliding_window, min_periods=sliding_min_period).std()

                decomp_mu_ = decomp_mu_.ffill().bfill().fillna(0)
                decomp_std_ = decomp_std_.ffill().bfill().fillna(0)

                print("number time-windows with std=0: ",
                      (decomp_std_ == 0).sum())
                global_med_ = np.nanmedian(decomp_mu_)
                decomp_std_[decomp_std_ == 0] = decomp_mu_[decomp_std_ == 0].apply(
                    lambda x: sd_zero_adjust_k*x if x != 0 else global_med_)

                decomp_mu.iloc[i_start:i_end] = decomp_mu_
                decomp_std.iloc[i_start:i_end] = decomp_std_

            score = (_df[var] - decomp_mu).abs()/decomp_std

            if isdropna:
                _df = _df.reindex(_df_idx_bak, axis=0)
                score = score.reindex(_df_idx_bak, axis=0)
                score.fillna(0, inplace=True)

            df[var+"_OL_SCORE"] = score
            if sd_th > 1:
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
            else:
                sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

    elif method == "mes":
        print("Exponential Smoothing...")
        """
        Exponential Smoothing is a very popular scheme to produce a smoothed Time Series. Recall in Single Moving Averages the past observations are weighted equally. 
        But isn’t it more intuitive that recent data points should have a stronger influence on today’s data than ancient data? Exponential Smoothing is designed to address this problem. 
        Exponential Smoothing assigns exponentially decreasing weights as the observation gets older. In other words, recent data are given relatively more weight in forecasting than older data.
        """

        for var in tqdm(variables):
            EMAfit = ExponentialSmoothing(
                ts_df[var], damped_trend=True, trend="add").fit()
            scaler = ts_df[var].copy()
            rolled_mu = ts_df[var].rolling(
                window=sliding_window, min_periods=sliding_min_period).median()
            scaler[scaler == 0] = rolled_mu[scaler == 0]

            df[var+"_OL_SCORE"] = (ts_df[var] -
                                   EMAfit.predict(start=0)).abs()/scaler
            if sd_th > 1:
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
            else:
                sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

    elif method == "tsd":
        """
        Seasonal-Trend Decomposition (STD)
        This technique gives you the ability to split your time series signal into three parts: seasonal, trend, and residual, and is suitable for many time series that possess calendar patterns. 
        This model assumes the three components are simply additive, meaning you can simply add them up to get back to the original time series (seasonal + trend + residual = the time series). 
        The algorithm automatically searches for the periodical patterns.
        """

        if issliding_window:

            for var in tqdm(variables):
                print(f"{var}...")
                _df = ts_df[[var]].copy()
                _df.index = pd.DatetimeIndex(_df.index)
                _df.sort_index(inplace=True)

                if isdropna:
                    
                    _df_idx_bak = _df.index
                    isnan_idx = _df[var].isna()
                    _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)
                    
                # decomp_obj = sm.tsa.seasonal_decompose(x=_df[var], model='additive', extrapolate_trend='freq')
                # require hourly samples or peroid adjustment
                if use_trend_period_auto:
                    trend_period = estimate_signal_period(_df[var].values, "ACF")
                    print(f"trend_period: {trend_period}")

                decomp_obj = sm.tsa.seasonal_decompose(
                    x=_df[var], model='additive', extrapolate_trend='freq', period=trend_period, two_sided=istrend_twosided)
                # You need to explicitly set the period when using data that does not have an obvious seasonality."period = 60*60*24 to represent a daily period.
                # decomp_obj = sm.tsa.seasonal_decompose(x=_df[var], model='additive', extrapolate_trend=0)
                # decomp_obj.plot()
                # plt.show()
                
                _df["resid"] = decomp_obj.resid if isremove_cycle else decomp_obj.resid + decomp_obj.seasonal 

                # df[var+"_TREND"] = decomp_obj.trend
                df[var+"_CYCLIC"] = decomp_obj.seasonal

                if user_data_cp_idx is None:
                    user_data_cp_idx = [_df.index[0]]

                decomp_mu = pd.Series(index=_df.index)
                decomp_std = pd.Series(index=_df.index)

                prev_i_start = None
                for i, idx in enumerate(user_data_cp_idx + [_df.index[-1]]):
                    start_idx = _df.index[_df.index >= idx].values[0]
                    if i == 0:
                        i_start = 0
                        i_end = _df.index.get_loc(start_idx)
                    else:
                        i_start = prev_i_start
                        i_end = _df.index.get_loc(start_idx)

                    prev_i_start = i_end

                    if i_start >= i_end:
                        continue

                    resid_ = _df["resid"].iloc[i_start:i_end]
                    print(resid_.shape)

                    if use_sliding_window_auto:
                        sliding_window = sliding_window_alg(resid_.values)
                        sliding_min_period = max(sliding_window//3, 5)
                        print("sliding_window: ", sliding_window)

                    if len(use_quantile):

                        decomp_q1 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                            use_quantile[0], interpolation='midpoint')
                        decomp_q3 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                            use_quantile[1], interpolation='midpoint')

                        resid_.loc[(resid_ < decomp_q1) | (
                            resid_ > decomp_q3)] = np.nan
                        decomp_mu_ = resid_.rolling(
                            window=sliding_window, min_periods=sliding_min_period).median()
                        decomp_std_ = resid_.rolling(
                            window=sliding_window, min_periods=sliding_min_period).std()
                    else:
                        decomp_mu_ = resid_.rolling(
                            window=sliding_window, min_periods=sliding_min_period).median()
                        decomp_std_ = resid_.rolling(
                            window=sliding_window, min_periods=sliding_min_period).std()

                    decomp_mu_ = decomp_mu_.ffill().bfill().fillna(0)
                    decomp_std_ = decomp_std_.ffill().bfill().fillna(0)

                    print("number time-windows with std=0: ",
                          (decomp_std_ == 0).sum())
                    global_med_ = np.nanmedian(decomp_mu_)
                    decomp_std_[decomp_std_ == 0] = decomp_mu_[decomp_std_ == 0].apply(
                        lambda x: sd_zero_adjust_k*x if x != 0 else global_med_)

                    decomp_mu.iloc[i_start:i_end] = decomp_mu_
                    decomp_std.iloc[i_start:i_end] = decomp_std_

                score = (_df["resid"] - decomp_mu).abs()/decomp_std

                if isdropna:
                    _df = _df.reindex(_df_idx_bak, axis=0)
                    score = score.reindex(_df_idx_bak, axis=0)
                    score.fillna(0, inplace=True)

                # df[var+"_OL_SCORE"] =  (_df["resid"] - decomp_mu).abs()/decomp_std
                df[var+"_OL_SCORE"] = score
                if sd_th > 1:
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
                else:
                    sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

        else:
            for var in tqdm(variables):
                print(f"{var}...")
                _df = ts_df[[var]].copy()
                _df.index = pd.DatetimeIndex(_df.index)
                _df.sort_index(inplace=True)
                isnan_idx = _df[var].isna()
                _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)
                
                if use_trend_period_auto:
                    trend_period = estimate_signal_period(_df[var].values, "ACF")
                    print(f"trend_period: {trend_period}")

                decomp_obj = sm.tsa.seasonal_decompose(
                    x=_df[var], model='additive', extrapolate_trend='freq', period=trend_period, two_sided=istrend_twosided)
                _df["resid"] = decomp_obj.resid if isremove_cycle else decomp_obj.resid + decomp_obj.seasonal  
                decomp_mu = np.nanmedian(_df["resid"].values)
                decomp_std = np.nanstd(_df["resid"].values)
                decomp_mu = 0 if np.isnan(decomp_mu) else decomp_mu
                decomp_std = 0 if np.isnan(decomp_std) else decomp_std
                decomp_std = decomp_std if decomp_std > 0 else sd_zero_adjust_k*decomp_mu
                _df = _df.reindex(_df_idx_bak, axis=0)
                df[var+"_OL_SCORE"] = np.abs(_df["resid"] -
                                             decomp_mu)/decomp_std
                df[var+"_OL_SCORE"].fillna(0, inplace=True)
                if sd_th > 1:
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
                else:
                    sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

    elif method.startswith("drift"):

        if issliding_window:
            for var in tqdm(variables):
                _df, decomp_obj = trend_drift_detection(
                    ts_df[var], var=var, **kwargs)

                for col in _df.columns:
                    if col.startswith(f"{var}_") and ("_TREND" in col):
                        df[col] = _df[col]

                if "tsd" in method:
                    if isdropna:
                        isnan_idx = _df[var].isna()
                        _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)

                    if user_data_cp_idx is None:
                        user_data_cp_idx = [_df.index[0]]

                    decomp_mu = pd.Series(index=_df.index)
                    decomp_std = pd.Series(index=_df.index)
                    prev_i_start = None
                    for i, idx in enumerate(user_data_cp_idx + [_df.index[-1]]):
                        start_idx = _df.index[_df.index >= idx].values[0]
                        if i == 0:
                            i_start = 0
                            i_end = _df.index.get_loc(start_idx)
                        else:
                            i_start = prev_i_start
                            i_end = _df.index.get_loc(start_idx)

                        prev_i_start = i_end

                        if i_start >= i_end:
                            continue
                        
                        resid_ = _df["resid"].iloc[i_start:i_end]
                        print(resid_.shape)

                        if use_sliding_window_auto:
                            sliding_window = sliding_window_alg(resid_.values)
                            sliding_min_period = max(sliding_window//3, 5)
                            print("sliding_window: ", sliding_window)

                        if len(use_quantile):

                            decomp_q1 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                                use_quantile[0], interpolation='midpoint')
                            decomp_q3 = resid_.rolling(window=sliding_window, min_periods=sliding_min_period).quantile(
                                use_quantile[1], interpolation='midpoint')

                            resid_.loc[(resid_ < decomp_q1) | (
                                resid_ > decomp_q3)] = np.nan
                            decomp_mu_ = resid_.rolling(
                                window=sliding_window, min_periods=sliding_min_period).median()
                            decomp_std_ = resid_.rolling(
                                window=sliding_window, min_periods=sliding_min_period).std()
                        else:
                            decomp_mu_ = resid_.rolling(
                                window=sliding_window, min_periods=sliding_min_period).median()
                            decomp_std_ = resid_.rolling(
                                window=sliding_window, min_periods=sliding_min_period).std()

                        decomp_mu_ = decomp_mu_.ffill().bfill().fillna(0)
                        decomp_std_ = decomp_std_.ffill().bfill().fillna(0)
                        print("number time-windows with std=0: ",
                              (decomp_std_ == 0).sum())
                        global_med_ = np.nanmedian(decomp_mu_)
                        decomp_std_[decomp_std_ == 0] = decomp_mu_[decomp_std_ == 0].apply(
                            lambda x: sd_zero_adjust_k*x if x != 0 else global_med_)
                        decomp_mu.iloc[i_start:i_end] = decomp_mu_
                        decomp_std.iloc[i_start:i_end] = decomp_std_

                    score = (_df["resid"] - decomp_mu).abs()/decomp_std

                    if isdropna:
                        _df = _df.reindex(_df_idx_bak, axis=0)
                        score = score.reindex(_df_idx_bak, axis=0)
                        score.fillna(0, inplace=True)

                    df[var+"_OL_SCORE"] = score
                    if sd_th > 1:
                        df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
                    else:
                        sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                        df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

                for col in _df.columns:
                    if col.startswith(f"{var}_") and ("_TREND" not in col):
                        df[col] = _df[col]
                    if col.startswith(f"{var}_") and ("_SCORE" in col):
                        df[col].fillna(0, inplace=True)
                    elif col.startswith(f"{var}_") and ("_LBL" in col):
                        df[col].fillna(False, inplace=True)

        else:
            for var in tqdm(variables):
                _df, decomp_obj = trend_drift_detection(
                    ts_df[var], var=var, **kwargs)

                if "tsd" in method:
                    for col in _df.columns:
                        if col.startswith(f"{var}_") and ("_TREND" in col):
                            df[col] = _df[col]

                    isnan_idx = _df[var].isna()
                    _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)
                    decomp_mu = np.nanmedian(_df["resid"].values)
                    decomp_std = np.nanstd(_df["resid"].values)
                    decomp_mu = 0 if np.isnan(decomp_mu) else decomp_mu
                    decomp_std = 0 if np.isnan(decomp_std) else decomp_std
                    decomp_std = decomp_std if decomp_std > 0 else sd_zero_adjust_k*decomp_mu
                    _df = _df.reindex(_df_idx_bak, axis=0)
                    df[var+"_OL_SCORE"] = np.abs(_df["resid"] -
                                                 decomp_mu)/decomp_std
                    df[var+"_OL_SCORE"].fillna(0, inplace=True)
                    if sd_th > 1:
                        df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th
                    else:
                        sd_th_q = np.percentile(df[var+"_OL_SCORE"], 100*sd_th)
                        df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sd_th_q

                for col in _df.columns:
                    if col.startswith(f"{var}_") and ("_TREND" not in col):
                        df[col] = _df[col]
                    if col.startswith(f"{var}_") and ("_SCORE" in col):
                        df[col].fillna(0, inplace=True)
                    elif col.startswith(f"{var}_") and ("_LBL" in col):
                        df[col].fillna(False, inplace=True)

    elif method.startswith("sr"):
        method_th = method.split("_")
        method_th = method_th[1] if len(method_th) > 1 else "fixed"
        stat_df = None
        sr_sampling_rate = kwargs.get("sr_sampling_rate", 60)
        score_window_size = kwargs.get("score_window_size", 10)
        series_window_size = kwargs.get("series_window_size", sr_sampling_rate)
        pad_len = kwargs.get("pad_len", 0)
        timewindow_slice = kwargs.get("timewindow_slice", 0)
        istrend_signal = kwargs.get("istrend_signal", False)
        user_data_cp_idx = kwargs.get("user_data_cp_idx", None)

        score_norm_type = "abs"

        if pad_len is None:
            pad_len = score_window_size

        sr_sampling_rate = 1*1*sr_sampling_rate
        amp_window_size = 1*sr_sampling_rate
        print("sampling_rate={}, amp_window_size={}, series_window_size={}, score_window_size={}".format(
            sr_sampling_rate, amp_window_size, series_window_size, score_window_size))

        for var in tqdm(variables):

            if ts_df[var].nunique() > 1:

                _df = ts_df[[var]].copy()
                isnan_idx = _df[var].isna()
                _df, _df_idx_bak = dropna_with_idx(_df, isnan_idx)
                uts_data_np = _df[var].values

                if istrend_signal:
                    trend_period = kwargs.get(
                        "trend_period", score_window_size)
                    istrend_twosided = kwargs.get("trend_period", True)
                    decomp_obj = sm.tsa.seasonal_decompose(
                        x=_df[var], model='additive', extrapolate_trend='freq', period=trend_period, two_sided=istrend_twosided)
                    _df["trend"] = decomp_obj.trend
                    uts_data_np = _df["trend"].values
                    score_norm_type = "trend"

                N = len(uts_data_np)
                if not timewindow_slice:
                    timewindow_slice = N

                sr_score = np.zeros((1, N)).squeeze(0)
                num_w = N//timewindow_slice

                if user_data_cp_idx is not None:
                    num_cps = len(user_data_cp_idx)
                    prev_i_start = None
                    for i, idx in enumerate(user_data_cp_idx + [_df.index[-1]]):
                        start_idx = _df.index[_df.index >= idx].values[0]
                        print(idx, start_idx)
                        if i == 0:
                            i_start = 0
                            i_end = _df.index.get_loc(start_idx)
                        # elif i == num_cps:
                        #     i_start = _df.index.get_loc(start_idx)
                        #     i_end = N
                        else:
                            i_start = prev_i_start
                            i_end = _df.index.get_loc(start_idx)

                        prev_i_start = i_end

                        # print(idx, start_idx, i_start, i_end)
                        # print(_df.iloc[i_start:i_end].head())
                        # print(_df.iloc[i_start:i_end].tail())
                        uts_data_np_ = uts_data_np[i_start:i_end]

                        if pad_len > 0:
                            uts_data_np_ = np.concatenate(
                                (uts_data_np_[:pad_len], uts_data_np_, uts_data_np_[-pad_len:]))

                        spec = sr.Silency(
                            amp_window_size, series_window_size, score_window_size)

                        sr_score_ = spec.generate_anomaly_score(
                            uts_data_np_, type=score_norm_type)

                        if pad_len > 0:
                            # type="abs" May 25, 2023, resturn +ve scores
                            sr_score_ = sr_score_[pad_len:-pad_len]

                        # sr_score_[sr_score_ < 0] = 0 # added May 25, 2023, skip down step changes
                        sr_score_[0] = 0
                        sr_score[i_start:i_end] = sr_score_

                else:

                    for i in range(num_w):
                        i_start = i*timewindow_slice
                        i_end = (i + 1)*timewindow_slice if i == num_w-1 else N
                        uts_data_np_ = uts_data_np[i_start:i_end]

                        if pad_len > 0:
                            uts_data_np_ = np.concatenate(
                                (uts_data_np_[:pad_len], uts_data_np_, uts_data_np_[-pad_len:]))

                        spec = sr.Silency(
                            amp_window_size, series_window_size, score_window_size)

                        sr_score_ = spec.generate_anomaly_score(
                            uts_data_np_, type=score_norm_type)

                        if pad_len > 0:
                            # type="abs" May 25, 2023, resturn +ve scores
                            sr_score_ = sr_score_[pad_len:-pad_len]

                        # sr_score_[sr_score_ < 0] = 0 # added May 25, 2023, skip down step changes
                        sr_score_[0] = 0
                        sr_score[i_start:i_end] = sr_score_

                _df["OL_SCORE"] = sr_score
                _df = _df.reindex(_df_idx_bak, axis=0)

                if method_th == "fixed":
                    df[var+"_OL_SCORE"] = _df["OL_SCORE"].values
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sr_th
                elif method_th == "stat":
                    sr_th = np.quantile(_df["OL_SCORE"].values, 0.999)
                    df[var+"_OL_SCORE"] = _df["OL_SCORE"].values
                    df[var+"_OL_LBL"] = df[var+"_OL_SCORE"] > sr_th
                elif method_th == "ml":
                    #  sr score is local and global od may not relevant due to similar but large sr score will be detected as normal.
                    sr_score = np.expand_dims(_df["OL_SCORE"].values, axis=1)

                    ol_model = LocalOutlierFactor(
                        n_neighbors=100, contamination="auto", novelty=False, n_jobs=-1)
                    ol_score = ol_model.fit_predict(sr_score)
                    df[var+"_OL_SCORE"] = -1*ol_model.negative_outlier_factor_
                    df[var+"_OL_LBL"] = ol_score != 1
            else:
                # constant values will generate an error for sr analysis
                df[var+"_OL_SCORE"] = 0
                df[var+"_OL_LBL"] = False

            if isplot:
                sr_score[ts_df[var].isna()] = np.nan
                fig, ax = plt.subplots(
                    ncols=1, nrows=2, figsize=(15, 4), sharex=True, constrained_layout=True)
                ax1 = plt.subplot(211)
                plt.plot(ts_df[var].values, label="signal")
                plt.scatter(np.arange(ts_df[var].shape[0]), ts_df[var].mask(
                    ~df[var+"_OL_LBL"], np.nan).values, color="red", s=20, label="oulier")
                ax1.locator_params(axis="x", nbins=10)
                ax1.locator_params(axis="y", nbins=4)
                plt.legend()
                plt.title(var)

                ax2 = plt.subplot(212)
                plt.plot(np.arange(ts_df[var].shape[0]),
                         sr_score, label="outlier score", color="red")
                ax2.locator_params(axis="x", nbins=10)
                ax2.locator_params(axis="y", nbins=4)
                plt.legend()
                plt.show()

    lbl_cols = df.filter(regex="_OL_LBL$").columns
    df[lbl_cols] = df[lbl_cols].astype('int8')

    return df, stat_df

def uts_anomaly_detection(ts_dfc, method="sr", **kwargs):
    # prepares sensor signal without anomalies for modeling the normal characteristics
    print("preparing normal signal after removing outliers...")
    print("algorithms: stat_sigma, sr, msd, mes, tsd, prophet, drift_prophet",
          "drift_tsd", "drift")
    imputation_outlier = kwargs.get("imputation_outlier", True)
    verbose = kwargs.get("verbose", 1)
    ts_df = ts_dfc.copy()
    col_vars = ts_df.columns
    print("missing data (before outlier cleaning): {}".format(
        ts_df.loc[ts_df.isna().sum(axis=1) > 0].shape))
    ts_lo_df = None
    if method.startswith(("stat_sigma", "sr", "ml", "pca", "msd", "mes", "tsd", "prophet", "drift_prophet", "drift_tsd", "drift")):
        ts_lo_df, stat_df = get_local_outliers(ts_df, method=method, **kwargs)
        if verbose:
            lbl_cols = ts_lo_df.filter(regex="_OL_LBL$").columns
            for col in tqdm(lbl_cols):
                print(ts_lo_df[col].value_counts())

    print("missing data (after outlier cleaning): {}".format(
        ts_df.loc[ts_df.isna().sum(axis=1) > 0].shape))

    return ts_df, ts_lo_df

def nan_handler(df, isdropna=False, isdrop_all_mode=True, isfillna=False, method=None, fill_value=0, issparse_gap_opt=False, **kwargs):
    print("nan_handler...", kwargs)
    timestep = kwargs.pop("timestep", "60S")
    num_missing_seq_limit = kwargs.get("num_missing_seq_limit", 10)
    isclean_outlier = kwargs.get("isclean_outlier", False)
    min_gap_len = kwargs.pop("min_gap_len", 60)
    nan_gap_keep_len = kwargs.pop("nan_gap_keep_len", 1)
    iseq_sampled = kwargs.pop("iseq_sampled", True)

    fillna_method = kwargs.get("fillna_method", "keepval")  # keepval, rolling
    rolling_window = kwargs.get("rolling_window", None)
    if isdropna:
        print("dropping nan before: ", df.shape)
        if not issparse_gap_opt:
            print("num of nan records: ",
                  df.isna().sum(axis=0).sum(axis=0))

            if isdrop_all_mode:
                df.dropna(axis=0, inplace=True)
            else:
                # clean all nan regions only
                all_zero_region = df.isna().sum(axis=1) == df.shape[1]
                print("number of all nan idx: ", all_zero_region.sum())
                df = df.loc[~all_zero_region]
        else:
            print("num of nan records: ",
                  df.isna().sum(axis=0).sum(axis=0))
            df = sparse_time_gap_handler(df, timestep=timestep, min_gap_len=min_gap_len,
                                         nan_gap_keep_len=nan_gap_keep_len, iseq_sampled=True, isdrop_all_mode=isdrop_all_mode, **kwargs)

    if isfillna:
        if fillna_method == "rolling":
            if rolling_window is None:
                rolling_window = num_missing_seq_limit
        if method is None:
            if num_missing_seq_limit != 0:
                if fillna_method == "rolling":
                    fill_value_ = df.rolling(
                        rolling_window, min_periods=1).median()
                    fill_value_[fill_value_.isna()] = fill_value
                    df = df.fillna(fill_value_, limit=num_missing_seq_limit)
                else:
                    df = df.fillna(fill_value, limit=num_missing_seq_limit)
        else:
            if num_missing_seq_limit != 0:
                if fillna_method == "rolling":
                    df = df.fillna(df.rolling(
                        rolling_window, min_periods=1).median(), limit=num_missing_seq_limit)

                if 'ffill' in method:
                    df = df.ffill(limit=num_missing_seq_limit)
                if 'bfill' in method:
                    df = df.bfill(limit=num_missing_seq_limit)

    print("dropping nan after: ", df.shape)
    print("num of nan records: ", df.isna().sum(axis=0).sum(axis=0))

    print(df.head())
    return df

def get_object_columns(df):
    cols = df.select_dtypes("object").columns.tolist()
    return cols

def sparse_time_gap_handler(df, timestep="60S", min_gap_len=60, nan_gap_keep_len=60, iseq_sampled=False, isdrop_all_mode=True, **kwargs):
    """
    Cleans long nan gaps but by keeping smaller section from the begining yo avoid timestamp misss during timewindows
    df_eq_sample: equaly sampled data from df using timestep
    min_gap_len: minimum gap size in the number of samples, e.g min_gap_len=60 for 1h gap in minute (timestep=60S) sampling rate
    min_gap_len >= nan_gap_keep_len
    """

    print("sparse_time_gap_handler...")

    exclude_cols = kwargs.get("exclude_cols", [])

    print(df.shape)
    if not iseq_sampled:
        df_eq_sample = ts_data_reconstructor(
            df, timestep=timestep, t_idx="index", num_missing_seq_limit=1, **kwargs)
        print("After equal sampling interval :", df_eq_sample.shape)
    else:
        df_eq_sample = df

    if nan_gap_keep_len > min_gap_len:
        nan_gap_keep_len = min_gap_len

    print(f"nan_gap_keep_len: {nan_gap_keep_len}, min_gap_len: {min_gap_len}")
    # cleaning nan gaps (at least one column) but of the data
    num_cols = list(filter(lambda x: x not in get_object_columns(
        df_eq_sample), df_eq_sample.columns.tolist()))
    num_cols = list(filter(lambda x: x not in exclude_cols, num_cols))
    print(num_cols)
    if isdrop_all_mode:
        isnan_idx = df_eq_sample[num_cols].isna().sum(axis=1) > 0
    else:
        # clean all nan regions only
        isnan_idx = df_eq_sample[num_cols].isna().sum(axis=1) == len(num_cols)

    i = 0
    t_gap_all = []
    for k, g in itertools.groupby(isnan_idx.values.astype("int")):
        g_len = len(list(g))
        if k == 1:
            if g_len >= min_gap_len:
                t_gap = df_eq_sample.iloc[i:i+g_len].index
                # t_gap_all.append(([t_gap[0], t_gap[-1]], g_len))
                t_gap_all.append((t_gap, g_len))
        i = i + g_len

    print("Number of gaps: ", len(t_gap_all))

    for i, (t_gap, g_len) in enumerate(t_gap_all):
        df_eq_sample.drop(
            axis=0, index=t_gap[nan_gap_keep_len:], inplace=True)

    print(df_eq_sample.shape)

    return df_eq_sample

def plot_sep(sep={}):
    xmin, xmax, ymin, ymax = plt.gca().axis()
    # print(sep)
    for key, val in sep.items():
        plt.plot([val, val], [ymin, ymax], label=key,
                 color="black", linestyle="--")

def print_dict(dc):
    print("\n{}".format("*"*40))
    for key, value in dc.items():
        print("{}: {}".format(key, value))
    print("{}".format("*"*40))

def join_path(path_list=[]):
    # return "//".join([removesuffix(path, "//") for path in path_list])+"//"
    return os.path.join(*[removeaffix(removeaffix(path, "//"), "/") if i != 0 else removesuffix(removesuffix(path, "//"), "/") for i, path in enumerate(path_list)])+"//"

def get_filepath(dirpath, filename):
    # adjusts for linux and window paths
    return [os.path.join(dirpath, d) for d in os.listdir(dirpath) if d == filename][0]

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_figure(filename, fig, filepath=None, isshow=False, issave=True, dpi=300):
    filepath = filepath.rstrip("//")
    filepath = "{}//{}{}".format(filepath, filename,
                                 FIGURE_SAVEFORMAT).lstrip("//")
    print("saving ", filepath)
    if issave:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        if isshow:
            plt.show(fig)
        else:
            plt.close()

    if isshow:
        plt.show(fig)

def save_json(filename, datadic, filepath=None):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.json".format(removesuffix(filename, ".json"))

    print("saving: ", filepath)
    with open(filepath + filename, 'w') as fhandle:
        fhandle.write(json.dumps(datadic, indent=4,
                                 sort_keys=True, cls=MyEncoder))
        fhandle.close()

def load_json(filename, filepath=None):
    if not filepath:
        filepath = load_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.json".format(removesuffix(filename, ".json"))
    # print("loading: ", filepath + filename)
    print('loading ', filepath + filename, end='\r')
    with open(filepath + filename, 'r') as fhandle:
        data_dict = json.load(fhandle)
        fhandle.close()
    # print(data_dict)
    return data_dict

def save_csv(filename, df, filepath=None, index=True, ignore_format=False):
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.csv".format(removesuffix(filename, ".csv"))

    print('saving ', filepath + filename)
    if not ignore_format:
        df.to_csv(filepath + filename, float_format='%6.5f', index=index)
    else:
        df.to_csv(filepath + filename, index=index)

def load_csv(filename=None, filepath=None, index_col=None, filepath_full=None, **kwargs):
    if not filepath:
        filepath = load_dir
    if filepath_full is None:
        filepath = "{}//".format(removesuffix(filepath, "//"))
        filename = "{}.csv".format(removesuffix(filename, ".csv"))
        filepath_full = filepath + filename
    else:
        filepath_full = "{}.csv".format(removesuffix(filepath_full, ".csv"))

    # print('loading ', filepath)
    return pd.read_csv(filepath_full, index_col=index_col, **kwargs)

def save_npdata(filename, data, filepath=None):
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.npy".format(removesuffix(filename, ".npy"))
    print(filepath + filename)
    np.save(filepath + filename, data)

def load_npdata(filename, filepath=None):
    if not filepath:
        filepath = load_dir
    filepath = "{}//".format(removesuffix(filepath, "//"))
    filename = "{}.npy".format(removesuffix(filename, ".npy"))

    print(filepath + filename)
    # to allow Object arrays containing string
    return np.load(filepath + filename, allow_pickle=True)

