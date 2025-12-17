# =============================================================================
# AnomalyCD: Scalable Temporal Anomaly Causality Discovery in Large Systems
# =============================================================================
# This script provides an integrated pipeline for for online-AD to generate binary anomaly data sets.

# Main Features:
#   - Visualization and saving of results.
#   - Flexible interface for configuration.
#
# Author: Mulugeta Weldezgina Asres
# Email: muleina2000@gmail.com
# Date: May 2025
# =============================================================================

import numpy as np
import pandas as pd
import itertools
import datetime
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use("seaborn-v0_8-whitegrid")
import utilities as util

class OnlineTSAD():
    """
    Online Time Series Anomaly Detection class 
    """
    def __init__(self):
        pass

    def uts_outlier_detection(self, ts_df, method="sr", **kwargs):
        print("ts outliers detection...", method)
        sr_sampling_rate = kwargs.pop("sr_sampling_rate", 6) # kernel_size for edge detection in spectrum (f)
        score_window_size = kwargs.pop("score_window_size", 10) # kernel_size on time doemain of the score to smooth adjucent scores
        isplot = kwargs.get("isplot", False)
        _, df_outlier = util.uts_anomaly_detection(
            ts_df, method=method, sr_sampling_rate=sr_sampling_rate, score_window_size=score_window_size, **kwargs)
        print("Number of missing data in df_outlier: ")
        print(df_outlier.isna().sum(axis=1).value_counts())
        lbl_cols = df_outlier.filter(regex="_OL_LBL$").columns
        df_outlier[lbl_cols] = df_outlier[lbl_cols].astype("int8")
        print("Number of OD flags in df_outlier: ")
        print(df_outlier[lbl_cols].sum(axis=0))
        return df_outlier
        
    def online_outlier_detection(self, ts_df, **kwargs):
        isscaler = kwargs.pop("isscaler", False)
        isdiff_norm = kwargs.pop("isdiff_norm", False)

        if not isinstance(kwargs["od_alg"], list):
            kwargs["od_alg"] = [kwargs["od_alg"]]
        od_alg = kwargs["od_alg"] 

        if isscaler:
            _ts_df_med = ts_df.apply(lambda x: np.nanmedian(x, axis=0))
            print(_ts_df_med)
            _ts_df_med[_ts_df_med==0] = 1
            _ts_df = ts_df.divide(_ts_df_med)
        else:
            _ts_df = ts_df.copy()

        if isdiff_norm:
            _ts_df = _ts_df.diff(axis=0) 
            _ts_df.iloc[0, :] = 0
        df_outliers = [ts_df.copy()]
        index_col = ts_df.index.name
        for method in od_alg:  
            print("#"*80)
            print(f"{od_alg}...")
            _df_outlier = self.uts_outlier_detection(_ts_df, method=method, **kwargs)
            df_outliers.append(_df_outlier.add_prefix(f"{method.upper()}__"))
            print("#"*80)
        df_outlier = pd.concat(df_outliers, axis=1)
        
        print("aggregating outlier flags (sum)...")
        for col in tqdm(ts_df.columns):
            col_vars = df_outlier.filter(regex=f"\w*{col}\w*").columns.tolist()
            lbl_col_vars = df_outlier[col_vars].filter(regex="_OL_LBL$").columns.tolist()
            print(lbl_col_vars)
            df_outlier[f"{col}_OL_LBL"] = df_outlier[lbl_col_vars].sum(axis=1).astype("int8")
            df_outlier.drop(columns=lbl_col_vars, inplace=True)
            print(df_outlier[f"{col}_OL_LBL"].value_counts())
        print(df_outlier.head())
        return df_outlier

    def uts_anomaly_detection_analysis(self, ts_df, df_outlier, **kwargs):
        print("uts_anomaly_detection_analysis...: ", ts_df.columns)
        print("df_outlier...: ", df_outlier.columns)
        print("set nan for missing values for none-flag columns..")
        for col in tqdm(ts_df.columns):
            col_vars = df_outlier.filter(regex=f"\w*{col}\w*").columns.tolist()
            lbl_cols = list(filter(lambda x: x.endswith("_OL_LBL"), col_vars))
            non_lbl_cols = list(filter(lambda x: x not in lbl_cols, col_vars))
            print(lbl_cols)
            nan_idx = ts_df[col].isna()
            df_outlier.loc[nan_idx, non_lbl_cols] = np.nan # to avoid convert label to float64
            df_outlier[non_lbl_cols] = df_outlier[non_lbl_cols].astype("float32")
            df_outlier.loc[nan_idx, lbl_cols] = 0 # convert label to float64
            num_outlier = (df_outlier[f"{col}_OL_LBL"] > 0).sum()
            print(f"{col}: num_outlier: {num_outlier}")
        return df_outlier
    
    def inference(self, data_df, **kwargs):
        df_raw = util.add_prefix_sortednumeric(data_df.copy())

        if not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index = pd.date_range(datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(minutes=df_raw.shape[0]-1), freq='T')
        
        print(df_raw.head())
        print("data shape: ", df_raw.shape)

        tObj = util.ProcTimer()
        tObj.restart()

        #  ts od inference
        df_outlier_rm = self.online_outlier_detection(df_raw, **kwargs)
        df_outlier_rm = self.uts_anomaly_detection_analysis(df_raw, df_outlier_rm, **kwargs)
        df_raw = util.remove_prefix_sortednumeric(df_raw)
        df_outlier_rm = util.remove_prefix_sortednumeric(df_outlier_rm)
        df_raw.index = data_df.index
        df_raw.index.name = "t"
        df_outlier_rm.index = data_df.index
        df_raw.index.name = "t"
        
        tObj.display_proctime()
        tObj.stop()

        return df_outlier_rm, df_raw
    
    def prepare_plot_dfs(self, df_raw, df_outlier_rm, isfragment_flag_handle=True):
        plot_dfs = []
        marker_dfs = []
        for c, col in enumerate(df_raw.columns[:]):
            cols_per_sensor = df_outlier_rm.filter(
                regex=f"\w*{col}\w*").columns.tolist()
            df = df_outlier_rm[cols_per_sensor]
            non_lbl_cols = cols_per_sensor[:]
            non_lbl_cols.remove(col + "_OL_LBL")

            df.rename(columns={col + "_OL_LBL": "OD FLAG",
                                col + "_OL_SCORE": col + "_OD_SCORE"}, inplace=True)
            
            if isfragment_flag_handle:
                _df = df.reset_index(drop=True)
                anml_idx_list = _df.loc[_df["OD FLAG"] > 0, "OD FLAG"].index
                for i, anml_idx in enumerate(anml_idx_list):
                    if anml_idx == anml_idx_list[-1]:
                        break
                    flag_gap = anml_idx_list[i+1] - anml_idx_list[i]
                    if (flag_gap > 1) and (flag_gap < 5):
                        df.loc[df.index[anml_idx_list[i]]
                            :df.index[anml_idx_list[i+1]], "OD FLAG"] = 1
            marker_df = df.copy()
            marker_df = marker_df.apply(lambda x: df["OD FLAG"] > 0)
            df.drop(columns=["OD FLAG"], inplace=True)
            marker_df.drop(columns=["OD FLAG"], inplace=True)
            plot_dfs.append(df)
            marker_dfs.append(marker_df)

        plot_dfs = pd.concat(plot_dfs, axis=1)
        marker_dfs = pd.concat(marker_dfs, axis=1).fillna(0).astype(bool)

        # renaming variables
        col_list = [util.removesuffix(col, "_SCORE") for col in plot_dfs.columns]
        col_list = [col.replace("DRIFT_TSD__", "") for col in col_list]
        col_list = [col.replace("_OL", "_OD") for col in col_list]
        for i, col in enumerate(col_list):
            if "SR__" in col: col_list[i] = col.replace("SR__", "").replace("_OD", "_SR_OD")
            elif f"_OD" in col and "_DRIFT_OD" not in col and "_SR_OD" not in col: col_list[i] = col.replace("_OD" , "_STD_OD")

        plot_dfs.columns = col_list
        marker_dfs.columns = col_list
        
        return plot_dfs, marker_dfs, non_lbl_cols
    
    def get_agg_multi_od_flags(self, df_raw, df_outlier_rm, ops_mask=None, dropna=False):
        aml_dfs = []
        for col in tqdm(df_raw.columns):
            cols_per_sensor = df_outlier_rm.filter(
                regex=f"\w*{col}\w*").columns.tolist()
            df = df_outlier_rm[cols_per_sensor]
            num_outlier = (df[col + "_OL_LBL"] > 0).fillna(0).sum() if col + \
                "_OL_LBL" in cols_per_sensor else 0
            if (num_outlier == 0):
                continue
            df.rename(columns={col + "_OL_LBL": "OD_FLAG"}, inplace=True)
            aml_dfs.append(df[["OD_FLAG"]].add_prefix(f"{col}__"))
        aml_dfs = pd.concat(aml_dfs, axis=1)
        if ops_mask is not None: aml_dfs = aml_dfs.mask(~ops_mask)
        aml_dfs.columns = aml_dfs.columns.str.replace("__OD_FLAG", "")
        aml_dfs[df_raw.isna()] = np.nan
        if dropna:
            # aml_dfs[df_raw.isna()] = np.nan
            aml_dfs = aml_dfs.dropna(axis=0)
        df_aml = (aml_dfs > 0).astype(int)
        return df_aml
    
    @staticmethod
    def plot_ts_data_multi_system(df_raw_ts_data, system_col=None, ncol=3, figsize=(6,3.5), hspace=0.8, wspace=0.25, **kwargs):
        legendfontsize = kwargs.get("legendfontsize", 16)
        bbox_to_anchor =  kwargs.get("bbox_to_anchor", (0.1, 0.95))
        sns.set()
        sns.set_context("notebook", font_scale=2.8, rc={"lines.linewidth": 1.})

        plot_df = df_raw_ts_data.reset_index(drop=False).copy()
        if system_col is None:
            system_col = "_SYSTEM"
            plot_df[system_col] = ""

        subsys_quantity_list = plot_df.columns.tolist()
        if "t" in subsys_quantity_list: subsys_quantity_list.remove("t") 
        subsys_quantity_list.remove(system_col)
        
        n = len(subsys_quantity_list)
        nrow = int(np.ceil(n/ncol))
        fig = plt.figure(figsize=(figsize[0]*ncol, figsize[1]*nrow), constrained_layout=True)
        axs = fig.subplots(nrow, ncol)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        axs = util.trim_axs(axs, n+1)
        
        for i, var in enumerate(subsys_quantity_list):
            ax_sub = axs[i]
            _ = sns.lineplot(ax=ax_sub, x="t", y=var, 
                            data=plot_df,
                            hue=system_col,
                            markers=True, 
                            dashes=False, markersize=4, markeredgecolor=None, legend=i==0, style=system_col)
            _ = ax_sub.set_title(var)
            if i == 0:
                _ = ax_sub.set_ylabel("value")
                _ = ax_sub.legend(fontsize=legendfontsize)
            elif i%ncol==0:
                _ = ax_sub.set_ylabel("value")
            else:
                _ = ax_sub.set_ylabel(None)
            locator = util.ticker.MaxNLocator(4)
            ax_sub.yaxis.set_major_locator(locator)
            locator = util.mdates.AutoDateLocator(minticks=3, maxticks=8)
            formatter = util.mdates.ConciseDateFormatter(locator)
            ax_sub.xaxis.set_major_locator(locator)
            ax_sub.xaxis.set_major_formatter(formatter)

        axs[0].get_legend().remove()
        leg = plt.figlegend(loc='upper left',
                            bbox_to_anchor=bbox_to_anchor, ncol=4,
                                fancybox=True, frameon=True, shadow=False
                                )
        for leg_line in leg.legendHandles: # for matplotlib <=3.6
        # for leg_line in leg.legend_handles:
            leg_line.set_linewidth(10)
        sns.set()
        return fig

    @staticmethod
    def plot_od_results_ts_with_markers(df_raw_ts_data, df_od_flag_ts_data, system_col=None, sel_system=None, sel_sensor=None, data_gap_dates=None, od_kwargs={}):
        print("plotting... may take few minutes depending on data size. Disable use_timestamp=False for further processing with index based x-axis.")
        sns.set()
        sns.set_context("notebook", font_scale=1.9, rc={"lines.linewidth": 1.})
        
        plot_dfs = []
        marker_dfs = []
        if system_col is None:
            system_col = "_SYSTEM"
            df_raw_ts_data[system_col] = ""
            df_od_flag_ts_data[system_col] = ""
        if sel_system is None:
            sel_system = ""
        if sel_sensor is None:
            sel_sensor = ""

        for data_source_sysid in df_raw_ts_data[system_col].unique():
            if sel_system not in data_source_sysid:
                continue
            _df_raw = df_raw_ts_data.loc[df_raw_ts_data[system_col] == data_source_sysid]
            _df_raw.drop(columns=[system_col], inplace=True)
            _df_outlier = df_od_flag_ts_data.loc[df_od_flag_ts_data[system_col] == data_source_sysid]
            _df_outlier.drop(columns=[system_col], inplace=True)
            for _, col in enumerate(_df_raw.columns):
                if sel_sensor not in col:
                    continue

                cols_per_sensor = _df_outlier.filter(regex=f"\w*{col}\w*").columns.tolist()
                cols_per_sensor = [l for l in cols_per_sensor if (l == col) or (f"_{col}" in l) or (l.startswith(col))]
                _df = _df_outlier[cols_per_sensor]
                non_lbl_cols = cols_per_sensor[:]
                _ = [non_lbl_cols.remove(l) for l in non_lbl_cols if "_OL_LBL" in l]
                _df.rename(columns={col + "_OL_LBL": "OD FLAG"}, 
                                    inplace=True)
                marker_df = _df.copy()
                marker_df = marker_df.apply(lambda x: _df["OD FLAG"] > 0)
                _df.drop(columns=["OD FLAG"], inplace=True)
                marker_df.drop(columns=["OD FLAG"], inplace=True)
                if data_source_sysid != "":
                    col_name = f"{data_source_sysid}: "
                else:
                    col_name = ""
                plot_dfs.append(_df.add_prefix(col_name))
                marker_dfs.append(marker_df.add_prefix(col_name))

        if len(plot_dfs) == 0:
            return ValueError("No data found for the selected system or sensor.")
          
        plot_dfs = pd.concat(plot_dfs, axis=1)
        marker_dfs = pd.concat(marker_dfs, axis=1).fillna(0).astype(bool)

        # renaming variables
        col_list = [util.removesuffix(col, "_SCORE") for col in plot_dfs.columns]
        col_list = [col.replace("DRIFT_TSD__", "") for col in col_list]
        col_list = [col.replace("_OL", "_OD") for col in col_list]
        for i, col in enumerate(col_list):
            if "SR__" in col: 
                col_list[i] = col.replace("SR__", "").replace("_OD", "_SR_OD")
            elif ("_OD" in col) and ("_DRIFT_OD" not in col) and ("_SR_OD" not in col): 
                col_list[i] = col.replace("_OD" , "_STD_OD")

        plot_dfs.columns = col_list
        marker_dfs.columns = col_list

        ops_mask = None
        if data_gap_dates is not None:
            ops_mask = pd.Series(index=plot_dfs.index, name="OPS_MASK")
            ops_mask = ops_mask.apply(lambda x: False)
            for ops_t_range in data_gap_dates:
                ops_mask.loc[ops_t_range[0]:ops_t_range[1]] = True

        fig, ax = util.plot_grid(
                                plot_dfs.ffill() if ops_mask is None  else plot_dfs.ffill().mask(~ops_mask) ,
                                join_df=marker_dfs, join_format="scatter",
                                figsize=(6.5, 3.0), ncol=len(non_lbl_cols), color="blue",
                                x_gridno=8,
                                y_gridno=4,
                                wspace=0.22,
                                hspace=0.3,
                                fontsize=24,
                                legendfontsize=18,
                                labelfontsize=24,
                                tickfontsize=24,
                                ylabel="value",
                                ncol_force=True, iscolor_per_col=True, use_timestamp=True, isshow=False,
                                legends=["SIGNAL", "FLAG"]
                                )
        o_idx = 2 if len([col for col in plot_dfs.columns if "TREND" in col]) else 1
        for od_thr_name in ["sd_th", "drift_thr", "sr_th"]:
            if od_thr_name in od_kwargs.keys():
                _ = [ax[i].plot(plot_dfs.index, [od_kwargs[od_thr_name]]*plot_dfs.shape[0], label="THRESHOLD", color="gray") for i in np.arange(o_idx, len(ax), step=len(non_lbl_cols))]
                o_idx = o_idx + 1

        for ax_sub in ax:
            locator = util.ticker.MaxNLocator(5)
            ax_sub.yaxis.set_major_locator(locator)
            ax_sub.tick_params(axis='y', which='major', pad=1)
        return fig
    
# objODEngine = OnlineTSAD()
# df_outlier_rm, df_raw = objODEngine.inference(data_df, **kwargs)
# plot_dfs, marker_dfs, non_lbl_cols = objODEngine.prepare_plot_dfs(df_raw, df_outlier_rm, isfragment_flag_handle=True)
# df_aml = objODEngine.get_agg_multi_od_flags(df_raw, df_outlier_rm, ops_mask=None)