# =============================================================================
# AnomalyCD: Scalable Temporal Anomaly Causality Discovery in Large Systems
# =============================================================================
# This script provides an integrated pipeline for computationally efficient tools for graphical causal discovery from binary anomaly data sets.
# It supports multiple CD methods methods (PCMC variants etc.)
#
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
import os
import utilities as util
import baysian_ntk_utils as bays_util

class AnomalyCD():
    """
    Anomaly Causality Discovery class
    """
    def __init__(self):
        pass

    def cd_preprocess_sparse_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocesses sparse time series data for causal discovery.
        Args:
            df (pd.DataFrame): Input data.
            istodiscrete (bool, optional): If True, discretize data.
            iscontinues (bool, optional): If True, treat as continuous.
            keep_len (int, optional): Length for sparse handler.
            isflag_sparse_handle (bool, optional): If True, handle flag sparse.
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        istodiscrete = kwargs.get("istodiscrete", False)
        iscontinues = kwargs.get("iscontinues", False)
        keep_len = kwargs.pop("keep_len", 10)
        isflag_sparse_handle = kwargs.pop("isflag_sparse_handle", True)

        tObj = util.ProcTimer()

        if istodiscrete:
            df_binned_num = bays_util.convert_values_to_discrete(
                df, TIERS_NUM=4, return_num_cat=True, return_int=False)
            kwargs["iscontinues"] = False
            df = df_binned_num.astype("float16")

        df = df.copy().astype("float32")

        if (not iscontinues) and (not istodiscrete):
            fillna_method = None
            fill_value = 0.0
            isdrop_all_mode = False

        else:
            fillna_method = ['ffill', 'bfill']
            fill_value = None
            isdrop_all_mode = False

        _df = bays_util.sparse_data_handler(df, data_mode="nan", method=fillna_method, fill_value=fill_value,
                                isdrop_all_mode=isdrop_all_mode, keep_len=keep_len, **kwargs)

        if (not iscontinues) and (not istodiscrete):
            _df = _df.fillna(0)
            if isflag_sparse_handle:
                _df = bays_util.sparse_data_handler(
                    _df, data_mode="flag", keep_len=keep_len)  # prob changes with for non-ts due to keep_len
    
        tObj.stop()
        tObj.display_proctime()
        return _df

    def cd_structure_analysis(
        self,
        df: pd.DataFrame,
        cd_sel: str = "cd_nonts",
        dag_skeleton_alg: str = 'lpcmci',
        tau_min: int = 0,
        tau_max: int = 5,
        significance_level: float = 0.01,
        isfixed_mode: bool = False,
        isplot: bool = False,
        dag_params_alg: str = 'bayesian',
        **kwargs
        ):    
        """
        Performs causal structure analysis on the input data.
        Args:
            df (pd.DataFrame): Input data.
            cd_sel (str): RCA selection mode.
            dag_skeleton_alg (str): Algorithm for DAG skeleton.
            tau_min (int): Minimum lag.
            tau_max (int): Maximum lag.
            significance_level (float): Significance level for tests.
            ishrch (bool): Hierarchical flag.
            isfixed_mode (bool): Fixed mode flag.
            isplot (bool): Plot flag.
            dag_params_alg (str): DAG parameter algorithm.
            **kwargs: Additional arguments.
        Returns:
            tuple: (dag_skeleton_model, edges_df, net_df, )
        """
        print("cd_structure_analysis...")
        approach = kwargs.pop("approach", "fullts")  # "fullts"
        influence_dxn = kwargs.pop("influence_dxn", 'both') # 'positive' for anomaly flags
        isremove_self_lags = kwargs.pop("isremove_self_lags", True)
        max_iter = kwargs.pop("max_iter", 100)
        isplot = kwargs.pop("isplot", False)
        rand_seed = kwargs.pop("rand_seed", 42)
        ci_test = kwargs.pop('ci_test', 'ParCorr')
        
        sel_cols = df.columns.tolist()[:]
        indices = slice(0, df.shape[0])
        
        np.random.seed(rand_seed)
        if cd_sel == "cd_ts":
            dag_skeleton_model, edges_df = bays_util.build_ts_cd_ntk(df.iloc[indices][sel_cols], 
                                                                    dag_skeleton_alg=dag_skeleton_alg,
                                                                    dag_params_alg=None,
                                                                    max_iter=max_iter, 
                                                                    isplot=isplot, 
                                                                    isfixed_mode=isfixed_mode, 
                                                                    isblacklist_mode=True, 
                                                                    isskeletonbn=False, 
                                                                    isremove_self_lags=isremove_self_lags, 
                                                                    tau_min=tau_min, 
                                                                    tau_max=tau_max, 
                                                                    filter_menu=False,          
                                                                    approach=approach,
                                                                    significance_level=significance_level,
                                                                    ci_test=ci_test, 
                                                                    weight_sign=influence_dxn, 
                                                                    **kwargs
                                                                    )
        else:
            if influence_dxn == 'both':
                dag_skeleton_model, edges_df = bays_util.build_cd_ntk(df.iloc[indices][sel_cols],
                                                                                        dag_skeleton_alg=dag_skeleton_alg, 
                                                                                        dag_params_alg=None, 
                                                                                        max_iter=max_iter, 
                                                                                        isplot=isplot,
                                                                                        isfixed_mode=isfixed_mode,
                                                                                        isblacklist_mode=True,
                                                                                        isskeletonbn=False, 
                                                                                        significance_level=significance_level, 
                                                                                        ci_test=ci_test, 
                                                                                        isshow=False, 
                                                                                        weight_sign=influence_dxn, 
                                                                                        **kwargs
                                                                                    )
            else:
                tau_min, tau_max = 0, 0
                ci_test = 'ParCorr'
                ci_test = kwargs.pop('ci_test', 'ParCorr')
                isremove_self_lags = kwargs.pop("isremove_self_lags", True)
                dag_skeleton_alg = 'pcmci' 
                dag_skeleton_model, edges_df = bays_util.build_ts_cd_ntk(df.iloc[indices][sel_cols], dag_skeleton_alg=dag_skeleton_alg,
                                                                                                        dag_params_alg=None,
                                                                                                        max_iter=max_iter, 
                                                                                                        isplot=False, 
                                                                                                        isfixed_mode=isfixed_mode, 
                                                                                                        isblacklist_mode=True, 
                                                                                                        isskeletonbn=False, 
                                                                                                        isremove_self_lags=isremove_self_lags, 
                                                                                                        tau_min=tau_min, tau_max=tau_max, 
                                                                                                        filter_menu=False, 
                                                                                                        approach="tslaggedonly", 
                                                                                                        significance_level=significance_level, 
                                                                                                        ci_test=ci_test, 
                                                                                                        weight_sign=influence_dxn,
                                                                                                          **kwargs
                                                                                                        )
        return dag_skeleton_model, edges_df

    def cd_bn_model_build(self, df_binary, edges_df, dag_params_alg='bayesian', **kwargs):
        print("cd_bn_analysis...", kwargs)

        iscontinues = kwargs.pop("iscontinues", False)
        weight_aware = kwargs.pop('weight_aware', False)
        use_weighted_tlag = kwargs.pop('use_weighted_tlag', False)
        keep_undirected_at_lagged_zero = kwargs.pop('keep_undirected_at_lagged_zero', False)
        domain_edges = kwargs.pop('domain_edges', None)
        significance_level = kwargs.get('significance_level', 0.01)
        
        if iscontinues:
            dag_params_alg = None
        else:
            dag_params_alg = dag_params_alg

        if (dag_params_alg is None) or edges_df.empty:
            return None, None

        edges_agg_df = bays_util.aggregate_time_tag_edges(edges_df)
        df_unrolled_ts, new_lagged_cols = bays_util.prepare_unrolled_timelag_data(df_binary, edges_agg_df)

        # print("may remove one of the edges of undirected edge with keep_undirected_at_lagged_zero=True. To avoid error for BN model building.")
        bn_edges_df = bays_util.prune_binary_ts_edges(edges_df, 
                                                    keep_undirected_at_lagged_zero=keep_undirected_at_lagged_zero,
                                                    weight_aware=weight_aware, 
                                                    use_weighted_tlag=use_weighted_tlag, 
                                                    domain_edges=domain_edges, 
                                                    **kwargs
                                                    )
        if keep_undirected_at_lagged_zero:
            bn_edges_df = bays_util.directing_tlag_0(df_unrolled_ts, bn_edges_df, significance_level=significance_level)
        
        edges = bn_edges_df[["src_lag", "dst"]].apply(tuple, axis=1).values.tolist()
        dag_skeleton_model = bays_util.DAG()
        dag_skeleton_model.add_nodes_from(nodes=set(np.array(bn_edges_df[["src_lag", "dst"]].apply(tuple, axis=1).values.tolist()).reshape(-1).tolist()))
        dag_skeleton_model.add_edges_from(ebunch=edges, weights=bn_edges_df["weight"].values.tolist())

        dag_param_model = bays_util.learn_scm_dag_bays_parameters(df_unrolled_ts.fillna(0).astype("int8").astype("str"), 
                                                                  dag_skeleton_model, 
                                                                  alg=dag_params_alg, 
                                                                  **kwargs
                                                                  )

        return dag_param_model, dag_skeleton_model, bn_edges_df[["src", "dst", "weight", "t"]]

    def cd_gcm_build(self, df, **kwargs):
        "Causal Discovery"
        print("#"*80)
        significance_level = kwargs.pop("significance_level", 0.01)
        tau_min = kwargs.pop("tau_min", 0)
        cd_sel = kwargs.pop("cd_sel", "cd_nonts")
        maxlag = kwargs.pop("maxlag", 5)
        keep_len = kwargs.pop("keep_len", None)
        ispreprocess_sparse = kwargs.pop("ispreprocess_sparse", True)

        keep_len = 2*maxlag if keep_len is None else keep_len
        nq_df = df.nunique()
        nonconstant_sensors = nq_df[nq_df > 1].index
        print("nonconstant_sensors: ", nonconstant_sensors)
        df = df.copy().astype("float32")

        tObj = util.ProcTimer()
        tObj.restart()

        if ispreprocess_sparse:
            df_compressed = self.cd_preprocess_sparse_data(df.copy(), isdropna=True,
                                            issparse_gap_opt=True,
                                            isfillna=True,
                                            keep_len=keep_len,  # to give weight to trend
                                            min_gap_len=maxlag,
                                            nan_gap_keep_len=maxlag
                                            )
        else:
            df_compressed = df.copy()

        if cd_sel == "cd_ts":
            dag_skeleton_alg = kwargs.pop("dag_skeleton_alg", 'pcmci')
        else:
            dag_skeleton_alg = kwargs.pop("dag_skeleton_alg", 'cdt__ccdr')
        
        dag_skeleton_model, edges_df = self.cd_structure_analysis(df_compressed.fillna(0), cd_sel=cd_sel, 
                                                                                dag_skeleton_alg=dag_skeleton_alg,
                                                                                tau_min=tau_min,
                                                                                tau_max=maxlag,
                                                                                significance_level=significance_level, 
                                                                                ishrch=False, 
                                                                                isfixed_mode=False, 
                                                                                **kwargs
                                                                                )
        tObj.stop()
        proc_t = tObj.get_proctime()
        tObj.display_proctime()

        edges_agg_df = bays_util.aggregate_time_tag_edges(edges_df)
        net_df = edges_agg_df.values
        print(edges_agg_df.head(50))
        if net_df.shape[0] == 0:
            print("No relevant connection is detected among the variable.")
        print("Global roots: ", dag_skeleton_model.get_roots())
        print("#"*80)

        return df_compressed, dag_skeleton_model, edges_df, net_df, proc_t

    def cd_bn_inference(self, dag_param_model, dag_skeleton_model=None, cd_opt="Probabilistic Inference Report", **kwargs):
        
        cd_opts_dict = {
            "cd_cond_prob_infer_report": "Probabilistic Inference Report",
            "cd_all_causal_nodes": "All Causal Nodes",
            "cd_connected": "Connected (Reachable) Nodes",
            "cd_d_connected": "Check Causal Connection",
            "cd_plt_causal": "Plot Causal Network",
            "cd_plt_effect": "Plot Influenced Network"
        }

        st_cd_opts_sbox = list(
            filter(lambda x: cd_opts_dict[x] == cd_opt, cd_opts_dict))[0]
        print(st_cd_opts_sbox)

        bn_vars = sorted(list(dag_param_model.nodes()))

        if dag_param_model is not None:

            try:
                if st_cd_opts_sbox in ["cd_cond_prob_infer_report", "cd_causal_infer_report"]:
                    if st_cd_opts_sbox == "cd_cond_prob_infer_report":
                        print(
                            "Retrieves conditional probability of anomaly flags on the given condition.")
                    else:
                        print(
                            "Retrieves causality inference probability of anomaly flags on the given condition.")

                    st_target_vars_mbox = kwargs.get("st_target_vars_mbox", [])

                    print("active_trail_nodes: ")
                    print([dag_param_model.active_trail_nodes(target)
                        for target in st_target_vars_mbox])

                    bn_vars_list = list(
                        filter(lambda x: x not in st_target_vars_mbox, bn_vars))

                    st_observe_vars_aml_mbox = kwargs.get(
                        "st_observe_vars_aml_mbox", [])

                    bn_vars_list = list(
                        filter(lambda x: x not in st_observe_vars_aml_mbox, bn_vars_list))

                    st_observed_vars_wo_aml_mbox = kwargs.get(
                        "st_observed_vars_wo_aml_mbox", [])

                    print("st_target_vars_mbox: ", st_target_vars_mbox)
                    print("st_observe_vars_aml_mbox: ", st_observe_vars_aml_mbox)
                    print("st_observed_vars_wo_aml_mbox: ",
                        st_observed_vars_wo_aml_mbox)
                    observe_vars_dict = {
                        k: '1' for k in st_observe_vars_aml_mbox}
                    observe_vars_dict.update(
                        {k: '0' for k in st_observed_vars_wo_aml_mbox})

                    print("observe_vars_dict:", observe_vars_dict)

                    if st_target_vars_mbox:
                        if st_cd_opts_sbox == "cd_cond_prob_infer_report":
                            bn_result = bays_util.cond_prob_inference_report(
                                dag_param_model, st_target_vars_mbox, observed_cond_dict=observe_vars_dict, elimination_order="MinNeighbors")
                        else:
                            bn_result = bays_util.causal_inference_report(
                                dag_param_model, st_target_vars_mbox, observed_cond_dict=observe_vars_dict)

                        # print(bn_result)

                elif st_cd_opts_sbox == "cd_all_causal_nodes":
                    print("Retrieves root-causes or parent nodes.")
                    st_target_var_sbox = kwargs.get("st_target_var_sbox", [])
                    if st_target_var_sbox:
                        bn_result = bays_util.get_all_causal_nodes(
                            dag_skeleton_model, st_target_var_sbox)
                        print(bn_result)

                elif st_cd_opts_sbox == "cd_connected":
                    print("Gives a set of nodes which are affected (i.e. correlated) by any change in the given target nodes given an effect is oberved on the given observed variables.")

                    st_target_vars_mbox = kwargs.get("st_target_vars_mbox", [])
                    bn_vars_list = list(
                        filter(lambda x: x not in st_target_vars_mbox, bn_vars))
                    st_observed_vars_mbox = kwargs.get("st_observed_vars_mbox", [])

                    bn_result = bays_util.get_all_reachable_nodes(dag_skeleton_model, st_target_vars_mbox,
                                                                observed=st_observed_vars_mbox, include_latents=False)
                    print(bn_result)

                elif st_cd_opts_sbox == "cd_d_connected":
                    print(
                        "Determines if the given target node affects the influenced node or have a comman cause.")

                    st_target_var_sbox = kwargs.get("st_target_var_sbox", [])

                    bn_vars_list = list(
                        filter(lambda x: x not in st_target_var_sbox, bn_vars))
                    st_influenced_var_sbox = kwargs.get(
                        "st_influenced_var_sbox", [])

                    bn_result = bays_util.iscausal(dag_skeleton_model, st_target_var_sbox, st_influenced_var_sbox
                                                # observed=None
                                                )
                    if bn_result:
                        print(
                            f"{st_target_var_sbox} and {st_influenced_var_sbox} are causally connected.")
                    else:
                        print(
                            f"{st_target_var_sbox} and {st_influenced_var_sbox} are NOT causally connected.")

                elif st_cd_opts_sbox in ["cd_plt_causal", "cd_plt_effect"]:
                    if st_cd_opts_sbox == "cd_plt_causal":
                        print("Retrieves and plots graph network of their root-causes.")
                        st_target_vars_mbox = kwargs.get("st_target_vars_mbox", [])

                        if st_target_vars_mbox:
                            dag = bays_util.get_causal_dag(
                                dag_skeleton_model, st_target_vars_mbox)

                    elif st_cd_opts_sbox == "cd_plt_effect":
                        print(
                            "Retrieves and plots graph network of their influenced (affected) nodes.")
                        st_target_vars_mbox = kwargs.get("st_target_vars_mbox", [])
                        if st_target_vars_mbox:
                            dag = bays_util.get_influenced_dag(
                                dag_skeleton_model, st_target_vars_mbox)

                    if st_target_vars_mbox:
                        net_df = np.array(dag.edges())

                        if net_df.shape[0] == 0:
                            print(
                                "No relevant connection is detected among the variable.")
                            return None
                        else:
                            return net_df
            except Exception as ex:
                print(ex)
                print("oops! AnomalyCD got internal bug during inferencing!")

    def plotter_graph(net_df, graph_html_filename, main_path, **kwargs):
        graph_html_filepath = os.path.join(*[main_path, "tmp"])
        grp_src = kwargs.pop("grp_src", None)
        if grp_src is None:
            return util.plot_graph(net_df, graph_html_filename, filepath=graph_html_filepath, **kwargs)
        elif grp_src == "rca":
            return bays_util.plot_dag_graph_ntk(
                net_df, ntk_structure_alg_tag=graph_html_filename, filepath=graph_html_filepath, **kwargs)

    def cd_bn_top_root_causes(self, dag_param_model):
        """
        In the following cell it is evaluated which are the nodes that have the maximum and the minimum number of appearance in independence assertions as independent variable or evidence. 
        It can be notice that the closer a node is to the core of the network, the less are the independence assertions in which it is the independent variable and the more are the ones in which it is given as evidence.
        """

        def independent_assertions_score_function(model, node):
            return len([a for a in model.get_independencies().get_assertions() if node in a.event1])

        def evidence_assertions_score_function(model, node):
            return len([a for a in model.get_independencies().get_assertions() if node in a.event3])

        def update(assertion_dict, node, score_function, *args):
            tmp_score = score_function(*args, node)
            if tmp_score == assertion_dict["max"]["score"]:
                assertion_dict["max"]["nodes"].append(node)
            elif tmp_score > assertion_dict["max"]["score"]:
                assertion_dict["max"]["nodes"] = [node]
                assertion_dict["max"]["score"] = tmp_score
            if tmp_score == assertion_dict["min"]["score"]:
                assertion_dict["min"]["nodes"].append(node)
            elif tmp_score < assertion_dict["min"]["score"]:
                assertion_dict["min"]["nodes"] = [node]
                assertion_dict["min"]["score"] = tmp_score

        bn_vars = sorted(list(dag_param_model.nodes()))
        print("bn_vars: ", bn_vars)

        nodes = bn_vars[:]
        if len(nodes) > 1:
            evidence_init = evidence_assertions_score_function(
                dag_param_model, nodes[0])
            evidence_dict = {"max": {"nodes": [nodes[0]], "score": evidence_init},
                            "min": {"nodes": [nodes[0]], "score": evidence_init}}
            for node in nodes[1:]:
                update(evidence_dict, node,
                    evidence_assertions_score_function, dag_param_model)

        # most influential global root-causes
        print("most influential: ")

        print(f'Nodes which appear most ({evidence_dict["max"]["score"]} times) in independence assertions',
            f'as evidence are:\n{set(evidence_dict["max"]["nodes"])}')

        # least influential
        print("least influentials: ")

        print(f'Nodes which appear least ({evidence_dict["min"]["score"]} times) in independence assertions',
            f'as evidence are:\n{set(evidence_dict["min"]["nodes"])}')

        return list(set(evidence_dict["max"]["nodes"]))

    def plot_dag_graph_ntk(self, dag_skeleton_model, ntk_structure_alg_tag="", **kwargs):

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
        filepath = kwargs.pop("filepath", util.result_path)

        isnotation = kwargs.pop("isnotation", False)
        # nodesize = kwargs.pop("nodesize", 20)
        nodefontsize = kwargs.pop("nodefontsize", 30)

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
        # net_dict
        net_df = pd.DataFrame.from_dict(net_dict)

        if (ntk_edges.shape[1] < 4) and ("__tlag" in " ".join(net_df["source"].values.tolist())):
            net_df[["source", "t"]] = net_df["source"].str.split(
                "__tlag", expand=True)
            net_df.loc[~net_df["t"].isna(), "t"] = net_df.loc[~net_df["t"].isna(), "t"].apply(
                lambda x: "[-{}]".format(int(x.strip("_"))))
            net_df["t"] = net_df["t"].fillna('[0]')

        got_net = bays_util.Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color,
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
        # got_net.repulsion(node_distance=100, spring_length=200)
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

            # node["size"] = 100
            node["font"] = {"size": nodefontsize}

        filepath_full = '{}.html'.format(
            os.path.join(*[filepath, ntk_structure_alg_tag]))
        if isshow:
            got_net.show(filepath_full, notebook=notebook)
        else:
            if issave:
                got_net.save_graph(filepath_full)
                return filepath_full
            else:
                return got_net

# objAnmCD = AnomalyCD()