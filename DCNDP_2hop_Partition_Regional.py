import warnings
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import time
import random
from gurobi_solvers import solve_2hop_DCND
from utils import make_dir, sol_to_txt, get_total_2hop_connectivity,\
    calc_graph_connectivity, agile_HDA, fix_simplicials, partition_G
import multiprocessing
import shutil
import json
import argparse

import pandas as pd
random.seed(42)


warnings.filterwarnings("ignore")
#root_export_dir = os.path.join(".", "DCNDP_morPOP_hybrid_HDA_sols", 'morPOP2022-08-06-18-16')


gp_env = gp.Env(empty="gurobi.log")
# env.setParam("OutputFlag",0)
gp_env.start()






def solve_attributed_hybrid_DNCDP(G, partitions, export_path, hybrid_rate):

    root_export_dir = os.path.join(
        export_path, "hybrid_rate_"+str(hybrid_rate))
    partition_export_dir = os.path.join(root_export_dir, dividing_col)
    edge_cut_export_dir = os.path.join(root_export_dir, 'edge_cut_sols')

    partition_results_df_path = os.path.join(root_export_dir,
              "partition_results.csv")
    
    if(os.path.exists(partition_results_df_path)):
        df = pd.read_csv(partition_results_df_path)
    else:
        df = pd.DataFrame()

    make_dir(partition_export_dir)
    make_dir(edge_cut_export_dir)
    for identifier in partitions.keys():
        print("=======================================")
        current_community = partitions[identifier]
        print("Commmunity for {}={}: ".format(dividing_col, identifier))
        num_partition = len(current_community)
        print("Number of partitions: ", num_partition)
        for _part in range(num_partition):
            print("-----------------------------------")
            _part_nodes = [str(x) for x in current_community[_part]]
            #_part_nodes = [int(x) for x in _part_nodes]
            print("Len of current part nodes: ", len(_part_nodes))        

            H = G.subgraph(_part_nodes)
            print("\tSize of current partition", H.number_of_nodes())
            print("\t# of connected components: ",
              nx.number_connected_components(H))
            
            for c_count, component in enumerate(nx.connected_components(H)):
                print("\tsize of the connected component: ", len(list(component)))
                if len(list(component)) > 50:
                    exp_export_dir = os.path.join(
                        partition_export_dir, f"{identifier}_{str(_part)}_{str(c_count)}")
                    
                    overall_sol_path = os.path.join(
                    exp_export_dir, 'overall_sol.txt')
                    if os.path.exists(overall_sol_path):
                        print("Solution already exists for identifier: {} p:{} c:{} hr:{}".format(
                            identifier, _part, c_count, hybrid_rate))
                        continue

                    make_dir(exp_export_dir)

                    H_prime = G.subgraph(list(component)).copy()
                    H_prime_pc = calc_graph_connectivity(
                        H_prime, experiment_type='CN')
                    nx.write_edgelist(
                        H_prime, os.path.join(exp_export_dir, "G.el"))
                    N = H_prime.number_of_nodes()
                    M = H_prime.number_of_edges()

                    K = int(PARTITION_REMOVE_BUDGET * N)

                    print("\t\tSubgraph nodes: ", N)
                    print("\t\tSubgraph edges: ", H_prime.number_of_edges())
                    print("\t\tSubgraph budget: ", K)

                    heuristic_K = int(hybrid_rate * K)
                    gurobi_K = K - heuristic_K

                    print("\t\tGurobi budget: ", gurobi_K)
                    print("\t\tHeuristic budget: ", heuristic_K)

                    pc = 1
                    overall_sol = []
                    heuristic_approach = "HDA"
                    heuristic_time = 0
                    gurobi_time = 0

                    # create power graph
                    pre_opt_2hop_conn = get_total_2hop_connectivity(H_prime)

                    if(heuristic_K > 0):

                        #_iter_k = int(heuristic_K / 10)
                        heurstic_start_time = time.time()
                        heuristic_sol = agile_HDA(H_prime, heuristic_K)
                        #H_prime, s_d, heuristic_sol = AdaptiveBaselines(H_prime, k=_iter_k, experiment_types=['CN'],
                        #                                                node_limit=heuristic_K, ret_sol=True, approach=heuristic_approach)
                        heurstic_end_time = time.time()
                        heuristic_time = heurstic_end_time - heurstic_start_time

                        sol_to_txt(heuristic_sol, os.path.join(
                            exp_export_dir, 'heuristic_sol.txt'))
                        overall_sol += heuristic_sol
                        pc = calc_graph_connectivity(H_prime, experiment_type='CN')
                        print("\t\tHDA sol pairwise connectivity: ", pc)
                        print("\t\tG nodes after Heuristic: ",
                            H_prime.number_of_nodes())
                    
                    pre_gurobi_2hop_conn = get_total_2hop_connectivity(H_prime)
                    gurobi_time = 0
                    if(gurobi_K > 0):
                        H_prime_copy = H_prime.copy()
                        gurobi_warm_sol = agile_HDA(H_prime_copy, gurobi_K)
                        #H_prime_copy, _, gurobi_warm_sol = AdaptiveBaselines(H_prime_copy, k = 10, experiment_types=['CN'],
                        #        node_limit=gurobi_K, ret_sol=True)

                        gurobi_warm_sol = gurobi_warm_sol[0:gurobi_K]
                        gurobi_warm_sol = None #no HDA warm start
                        fixed_simplicial_nodes = None
                        num_simplicials = 0

                        if(simplicial_fixing):
                            fixed_simplicial_nodes = fix_simplicials(H_prime_copy)
                            #for i in range(len(fixed_simplicial_nodes)):
                            #    fixed_simplicial_nodes[i] = str(fixed_simplicial_nodes[i])
                            print("Number of simplicial nodes fixed: ", len(fixed_simplicial_nodes))
                            num_simplicials = len(fixed_simplicial_nodes)

                        gurobi_start_time = time.time()
                        m, gurobi_sol, new_short_pairs, original_short_pairs = solve_2hop_DCND(H_prime, gurobi_K, timelimit=TIMELIMIT,
                                            MIPgap=0.0, warm_start=gurobi_warm_sol, env=None,
                                            threads=num_threads, fixed_simplicial=fixed_simplicial_nodes,
                                            budget_constr='leq', X_binary=True,  aggregated_constraints=aggregated_constraints)
                        
                        
                    
                    
                    
                        gurobi_end_time = time.time()
                        gurobi_time = gurobi_end_time - gurobi_start_time

                        H_prime.remove_nodes_from(gurobi_sol)
                        pc = calc_graph_connectivity(H_prime, experiment_type='CN')
                        print("\t\Gurobi sol pairwise connectivity: ", pc)
                        sol_to_txt(gurobi_sol, os.path.join(
                            exp_export_dir, 'gurobi_sol.txt'))
                        overall_sol += gurobi_sol

                    post_gurobi_2hop_conn = get_total_2hop_connectivity(H_prime)
                    sol_to_txt(overall_sol, overall_sol_path)

                    new_row = {dividing_col: identifier, 'partition': _part,
                            'c_count': c_count, 'component_size': N, 'component_edges': M,
                            'budget': K,'heuristic_K': heuristic_K, 'gurobi_K': gurobi_K,
                            'pre_optimization_short_pairs': pre_opt_2hop_conn,'pre_gurobi_short_pairs': pre_gurobi_2hop_conn,
                            'post_gurobi_short_pairs': post_gurobi_2hop_conn,'c_pairwise_connectivity': H_prime_pc, 
                            'c_new_pairwise_connectivity': pc, "heuristic_approach": heuristic_approach, "heuristic_time": heuristic_time,
                            "gurobi_time": gurobi_time, "total_time": heuristic_time + gurobi_time, 'num_simplicials': num_simplicials}
                    if(gurobi_K > 0):
                        new_row['m_status'] = m.status
                        new_row['m_obj'] = m.ObjVal
                        new_row['m_MIPGap'] = m.MIPGap
                        new_row['m_sol_count'] = m.SolCount
                        new_row['m_runtime'] = m.Runtime
                        m.dispose()

                    df = df.append(new_row, ignore_index=True)
                    df.to_csv(partition_results_df_path, index=False)

    df.to_csv(partition_results_df_path, index=False)


def solve_min_edge_cover(G, export_path, hybrid_rate):
    df = pd.DataFrame()
    root_export_dir = os.path.join(
        export_path, "hybrid_rate_"+str(hybrid_rate))
    partition_export_dir = os.path.join(root_export_dir, dividing_col)
    edge_cut_export_dir = os.path.join(root_export_dir, 'edge_cut_sols')
    # load report dataframe
    report_df = pd.read_csv(os.path.join(
        root_export_dir, "partition_results.csv"))

    sol_nodes = []
    for index, row in report_df.iterrows():
        subg_dir = os.path.join(f"{row[dividing_col]}_{str(int(row['partition']))}_{str(int(row['c_count']))}")
    
        subg_path = os.path.join(partition_export_dir, subg_dir, "G.el")
        sol_path = os.path.join(partition_export_dir,
                                subg_dir, "overall_sol.txt")

        curr_sol = list(np.loadtxt(sol_path, dtype=int, delimiter=','))
        sol_nodes += curr_sol

    print(len(sol_nodes))

    print("Original graph before partition solutions:")
    print("\tNumber of nodes: {}".format(G.number_of_nodes()))
    print("\tNumber of edges: {}".format(G.number_of_edges()))
    print("\tPariwise connectivity: {}".format(
        calc_graph_connectivity(G, experiment_type='CN')))

    G.remove_nodes_from(sol_nodes)

    print("Original graph after partition solutions:")
    print("\tNumber of nodes: {}".format(G.number_of_nodes()))
    print("\tNumber of edges: {}".format(G.number_of_edges()))
    print("\tPariwise connectivity: {}".format(
        calc_graph_connectivity(G, experiment_type='CN')))

    cut_edge_set = []

    for (u, v) in G.edges():
        if G.nodes[u][dividing_col] != G.nodes[v][dividing_col]:
            cut_edge_set.append((u, v))

    print(len(cut_edge_set))
    _cut_graph = G.edge_subgraph(cut_edge_set)
    cut_graph = _cut_graph.copy()
    print("Cut graph loaded, number of nodes: {}".format(
        cut_graph.number_of_nodes()))
    print("Cut graph loaded, number of edges: {}".format(
        cut_graph.number_of_edges()))

    # get the median of degrees
    degrees = [d for n, d in cut_graph.degree()]
    degrees.sort()
    median_degree = degrees[int(len(degrees)/2)]

    print(median_degree)

    dominating_set = nx.dominating_set(cut_graph)

    print("Length of dominating set: {}".format(len(dominating_set)))

    print("Solve the min dominating set on the graph induced by edges ...")

    m = gp.Model(env=gp_env)
    X = m.addVars(cut_graph.nodes(), vtype=GRB.BINARY)
    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)
    m.addConstrs(X[u]+X[v] >= 1 for (u, v) in cut_graph.edges())
    #m.addConstrs( X[v] + gp.quicksum(X[u] for u in cut_graph.neighbors(v)) >= 1 for v in cut_graph.nodes())
    m.Params.Threads = num_threads
    m.setParam('MIPGap', 0.025)
    m.setParam('TimeLimit', 60*60)
    # optimize
    m.optimize()

    edge_cut_sol_arr = []
    for i in cut_graph.nodes():
        if(X[i].X == 1):
            edge_cut_sol_arr.append(int(i))
            G.remove_node(i)

    new_pairwise_connectivity = calc_graph_connectivity(
        G, experiment_type='CN')
    new_2hop_conn = get_total_2hop_connectivity(G)

    print("Final 2-hop connectivity: {}".format(new_2hop_conn))
    print("\Final Pariwise connectivity: {}".format(new_pairwise_connectivity))

    np.savetxt(os.path.join(edge_cut_export_dir, "cut_edge_sol.txt"),
               edge_cut_sol_arr, fmt='%i', delimiter=',')
    print("Number of nodes in solutions: ", len(edge_cut_sol_arr))
    print("Number of remaining nodes: ", G.number_of_nodes())

    print("Number of remaining edges: ", G.number_of_edges())

def get_solution_results(G, export_path, hybrid_rate):
    root_export_dir = os.path.join(
        export_path, "hybrid_rate_"+str(hybrid_rate))
    partition_export_dir = os.path.join(root_export_dir, dividing_col)
    edge_cut_export_dir = os.path.join(root_export_dir, 'edge_cut_sols')
    # load report dataframe
    report_df = pd.read_csv(os.path.join(
        root_export_dir, "partition_results.csv"))

    sol_nodes = []
    for index, row in report_df.iterrows():
        subg_dir = os.path.join(f"{row[dividing_col]}_{str(int(row['partition']))}_{str(int(row['c_count']))}")
        subg_path = os.path.join(partition_export_dir, subg_dir, "G.el")
        sol_path = os.path.join(partition_export_dir,
                                subg_dir, "overall_sol.txt")

        curr_sol = list(np.loadtxt(sol_path, dtype=int, delimiter=','))
        sol_nodes += curr_sol

    edge_cut_sol_path = os.path.join(edge_cut_export_dir, "cut_edge_sol.txt")
    edge_cut_sol = list(np.loadtxt(
        edge_cut_sol_path, dtype=int, delimiter=','))

    sol_nodes += edge_cut_sol

    len_sol_nodes = len(sol_nodes)
    return {"len_solution": len_sol_nodes}

def solve_attributed_DCNDP(G, G_attributes_df, export_path):
    
    row_dict = {}
    
    if(os.path.exists(os.path.join(export_path, "partitions.json"))):
        G = nx.read_gml(os.path.join(export_path, "G_partitioned.gml"))
        partitions = json.load(open(os.path.join(export_path, "partitions.json")))

        print("====== Loaded G, and Partitions dict from before ======")

    else:
        partitions = {}

        #drop rows where dividing column is nan
        G_attributes_df[dividing_col] = G_attributes_df[dividing_col].fillna(-1)
        G_attributes_df[dividing_col] = G_attributes_df[dividing_col].astype(int)

        # add dividing column as attribute
        id_attribute_dict = G_attributes_df.set_index('id')[dividing_col].to_dict()

        nx.set_node_attributes(G, id_attribute_dict, dividing_col)

        #get all unique values in dividing column
        part_start_time = time.time()
        unique_dividing_values = set(id_attribute_dict.values())
        #print(unique_dividing_values)
        for unique_val in unique_dividing_values:
            if(unique_val == -1):
                continue
            subgraph = G.subgraph([n for n, attr in G.nodes(data=True) if attr[dividing_col] == unique_val])
            

            if subgraph.number_of_nodes() < 1.5 * partition_avg_size:
                for node, attr in subgraph.nodes(data=True):
                    G.nodes[node]["partition"] = 0
                partitions[unique_val] = [list(subgraph.nodes())]
            else:
                num_partition = int(subgraph.number_of_nodes() / partition_avg_size) + 1
                subgraph, _partitions = partition_G(subgraph, num_partition)
                print("Calculating parts for {}={}, number of nodes: {}, numbder of partitions: {}".format(dividing_col, unique_val, subgraph.number_of_nodes(), num_partition))
                # Update the original graph with the partition attribute from the subgraph
                for node, attr in subgraph.nodes(data=True):
                    G.nodes[node]["partition"] = attr["partition"]
            
                partitions[unique_val] = _partitions[1]

        part_end_time = time.time()
        row_dict["partitioning_time"] = part_end_time - part_start_time
        nx.write_gml(G, path=os.path.join(export_path, "G_partitioned.gml"))
        #write partitions to json file
        with open(os.path.join(export_path, "partitions.json"), 'w') as f:
            json.dump(partitions, f)
    
        G = nx.read_gml(os.path.join(export_path, "G_partitioned.gml"))
        partitions = json.load(open(os.path.join(export_path, "partitions.json")))
        print("====== Loaded G, and Partitions dict again to make sure things work ======")
    


    #hybrid_rates = [0.8, 0]
    hybrid_rates = [0]
    #hybrid_rates = [0]

    hybrid_row_dicts_l = []
    for hybrid_rate in hybrid_rates:
        curr_row = row_dict.copy()
        print("Hybrid rate: ", hybrid_rate)
        curr_row["hybrid_rate"] = hybrid_rate

        copy_G = G.copy()
        hybrid_DNCDP_start_time = time.time()
        solve_attributed_hybrid_DNCDP(G, partitions, export_path, hybrid_rate)
        hybrid_DNCDP_end_time = time.time()
        curr_row["hybrid_DNCDP_time"] = hybrid_DNCDP_end_time - \
            hybrid_DNCDP_start_time

        copy_G = G.copy()
        print("Working on min edge cover problem")

        min_edge_cover_start_time = time.time()
        min_edge_cover_sol = solve_min_edge_cover(
            copy_G, export_path, hybrid_rate)
        min_edge_cover_end_time = time.time()
        curr_row["min_edge_cover_time"] = min_edge_cover_end_time - \
            min_edge_cover_start_time

        print("Getting solution analytics....")
        solution_results = get_solution_results(G, export_path, hybrid_rate)
        curr_row = {**curr_row, **solution_results}

        hybrid_row_dicts_l.append(curr_row)
    return hybrid_row_dicts_l


def main(DATA_PATH, meta_root_export_dir):
    meta_report_df_path = os.path.join(meta_root_export_dir, "report.csv")

    if os.path.exists(meta_report_df_path):
        meta_report_df = pd.read_csv(meta_report_df_path)
    else:
        meta_report_df = pd.DataFrame(columns=["exp_label"])

    completed_exps = meta_report_df['exp_label'].tolist()

    for g_name in os.listdir(DATA_PATH):
        # if g_name does not end with el
        if not g_name.endswith(".el"):
            continue

        g_path = os.path.join(DATA_PATH, g_name)
        g_subd_name = g_name.split(".")[0]
        g_export_path = os.path.join(meta_root_export_dir, g_subd_name)
        make_dir(g_export_path)

        shutil.copyfile(os.path.join(DATA_PATH, f"{g_subd_name}.csv"), os.path.join(meta_root_export_dir, f"G.csv"))
        G_attributes_df = pd.read_csv(os.path.join(meta_root_export_dir, f"G.csv"))


        if g_subd_name in completed_exps:
            print("Skipping", g_name)
            continue

        orig_G = nx.read_edgelist(g_path, nodetype=int)
        orig_G.node = orig_G.nodes
        print("=========================================")
        print(g_path)
        print("G name: ", g_name, "G size: ", orig_G.number_of_nodes())

        nx.write_edgelist(orig_G, os.path.join(g_export_path, "G.el"))

        hybrid_results_l = solve_attributed_DCNDP(orig_G, G_attributes_df, g_export_path)

        for hybrid_results in hybrid_results_l:
            hybrid_results["exp_label"] = g_export_path
            hybrid_results["graph_name"] = g_name

            meta_report_df = meta_report_df.append(
                hybrid_results, ignore_index=True)
        meta_report_df.to_csv(meta_report_df_path, index=False)

        completed_exps.append(g_export_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dividing_col', type=str, default='healthAuthority_ID', help='What column to divide the graph by (default: healthAuthority_ID)')
    parser.add_argument('--export_path', type=str, required=True, help='Path to the output solution')
    parser.add_argument('--input_path', type=str, default=os.path.join(".", "DCNDP_Datasets", 'NL_Day2'), help='Input graph path')

    parser.add_argument('--partition_avg_size', type=int, default=2500, help='partition avg size')
    parser.add_argument('--partition_remove_budget', type=float, default=0.2, help='Node budget for each partitions')
    parser.add_argument('--no_simplicial_fixing', action='store_true', help='fix simplicials?')
    parser.add_argument('--disaggragated_constraints', action='store_true', help='Use disaggregated constraints?')
    parser.add_argument('--timelimit', type=int, default=int(60) , help='Time limit for the solver')
    parser.add_argument('--num_threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads to use')



    args = parser.parse_args()
    
    DATA_PATH  = args.input_path
    dividing_col = args.dividing_col
    num_threads = args.num_threads
    TIMELIMIT = args.timelimit
    partition_avg_size = args.partition_avg_size
    PARTITION_REMOVE_BUDGET = args.partition_remove_budget
    simplicial_fixing = not args.no_simplicial_fixing
    aggregated_constraints = not args.disaggragated_constraints
    X_binary = False
    export_path = args.export_path

    meta_root_export_dir = os.path.join(
        ".", export_path)  # , getDTString())
    make_dir(meta_root_export_dir)

    with open(os.path.join(meta_root_export_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(DATA_PATH, meta_root_export_dir)

    # write a simple command to run the script
    # python DCNDP_2hop_Partition_Regional.py --export_path DCNDP_NL_D2_2023_06_14_healthAuthority_ID --input_path DCNDP_Datasets/NL_Day2 --dividing_col healthAuthority_ID --partition_avg_size 2500 --partition_remove_budget 0.2 --timelimit 3600 >> DCNDP_NL_D2_2023_06_14_healthAuthority_ID.txt
    # python DCNDP_2hop_Partition_Regional.py --export_path DCNDP_NL_D2_2023_06_14_healthAuthority_ID_disagg --input_path DCNDP_Datasets/NL_Day2 --dividing_col healthAuthority_ID --partition_avg_size 2500 --partition_remove_budget 0.2 --timelimit 3600 --disaggragated_constraints >> DCNDP_NL_D2_2023_06_14_healthAuthority_ID_disagg.txt