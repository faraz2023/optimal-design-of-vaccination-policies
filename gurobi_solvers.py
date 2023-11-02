from gurobipy import GRB
import gurobipy as gp
import networkx as nx
import os
import numpy as np


def sol_to_txt(sol_arr, export_path):
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',')


def make_dir(path):
    try:
        os.makedirs(path)
    except:
        return -1

def solve_1hop_DCND(G, k, timelimit=None, MIPgap=None, threads=16, warm_start=None, \
    env=None):
    print("Solving 1 hop DCNDP ...")

    m = gp.Model(env=env)
    m.Params.Threads = threads

    if(timelimit):
        m.setParam('TimeLimit', timelimit)
    # gap = | ObjBound - ObjVal | / | ObjVal |
    if(MIPgap):
        m.setParam('MIPGap', MIPgap)



    Y = m.addVars(G.nodes, vtype=GRB.BINARY)
    X = m.addVars(G.edges, vtype=GRB.BINARY)
    
    if(warm_start):
        for v in warm_start:
            Y[v].Start = 1
    # 1 − yu − yv ≤ xuv ∀{u, v} ∈ E
    m.addConstrs(1 - Y[u] - Y[v] <= X[u, v] for u, v in G.edges)
    m.addConstr(gp.quicksum(Y) <= k)
    print("All constaints added")
    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    print("Start optimizing...")
    #m.optimize()
    #added callback:
    m.optimize()

    critical_nodes = [vertex for vertex in G.nodes if Y[vertex].x > 0.5]
    print("# of critical nodes: ", len(critical_nodes))

    return m, critical_nodes



# Updated and debugged
def solve_2hop_DCND(G, k, timelimit=None, MIPgap=None, threads=10,
                warm_start=None, env=None, budget_constr="leq",
                  X_binary=True, aggregated_constraints=True,
                 fixed_simplicial=None, new_ver=False):

    print("Solving 2hop DCND ...")

    # create power graph
    PG = nx.power(G, 2)
    m = gp.Model(env=env)
    m.Params.Threads = threads

    if(timelimit):
        m.setParam('TimeLimit', timelimit)
    # gap = | ObjBound - ObjVal | / | ObjVal |
    if(MIPgap):
        m.setParam('MIPGap', MIPgap)

    m.setParam('Method', 3)

    # create y variables
    Y = m.addVars(PG.nodes, vtype=GRB.BINARY)

    if(warm_start):
        for v in warm_start:
            Y[v].Start = 1

    if(fixed_simplicial):
        for v in fixed_simplicial:
            Y[v].UB = 0


    G_edges = []
    for edge in G.edges:
        if(edge[0] < edge[1]):
            G_edges.append(tuple(edge))
        else:
            G_edges.append((edge[1], edge[0]))

    PG_edges = []
    for edge in PG.edges:
        if(edge[0] < edge[1]):
            PG_edges.append(tuple(edge))
        else:
            PG_edges.append((edge[1], edge[0]))

    new_edges_temp = [edge for edge in PG.edges if edge not in G.edges]
    new_edges = []
    for edge in new_edges_temp:
        if(edge[0] < edge[1]):
            new_edges.append(tuple(edge))
        else:
            new_edges.append((edge[1], edge[0]))

    original_short_pairs = len(PG_edges)
    print("Number of original edges in the power graph: ", len(PG_edges))

    if(X_binary):
        X = m.addVars(PG_edges, vtype=GRB.BINARY)
    else:
        X = m.addVars(PG_edges, vtype=GRB.CONTINUOUS)

    # minimize the number of short connections
    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)
    m._X = X
    m._Y = Y
    m._new_edges = new_edges
    m._G = G

    small_common_neighbours = 0
    print("!!! Aggregated constraints: ", aggregated_constraints)
    for edge in new_edges:
        common_neighbors = list(nx.common_neighbors(G, edge[0], edge[1]))
        if(len(common_neighbors) <= 2):
            small_common_neighbours += 1
        if(not aggregated_constraints):
        
            m.addConstrs(1 - Y[edge[0]] - Y[edge[1]] - Y[vertex] <= X[edge] for vertex in common_neighbors)
        
        if(aggregated_constraints): # this is currently the best improvment
            m.addConstr( (1/len(common_neighbors)) * gp.quicksum((1 - Y[vertex])
                    for vertex in common_neighbors) - Y[edge[0]] - Y[edge[1]] <= X[edge])
                

          

    # add constraints for edges in G

    print("!! Small common neighbourhoods: ", small_common_neighbours)
    m.addConstrs(X[edge] + Y[edge[0]] + Y[edge[1]] >= 1 for edge in G_edges)
    if(not new_ver):
        m.addConstrs(X[edge] + Y[edge[0]] <= 1 for edge in G_edges)
        m.addConstrs(X[edge] + Y[edge[1]] <= 1 for edge in G_edges)

    print("Loop constaints added")
    # add budget constraints
    #print("BUDGET: ", k)
    if(budget_constr == "leq"):
        m.addConstr(gp.quicksum(Y) <= k)
    elif(budget_constr == "eq"):
        m.addConstr(gp.quicksum(Y) == k)
    print("All constaints added")
    #Y[0].lb = 1

    
    print("Start optimizing...")
    #m.optimize()
    #added callback:
    m.optimize()

    # print(m.display())

    # retrieve solutions
    critical_nodes = [vertex for vertex in PG.nodes if Y[vertex].x > 0.5]

    print("# of critical nodes: ", len(critical_nodes))

    #short_pairs = [ edge for edge in new_edges if X[edge].x > 0.5 ]
    short_pairs = [edge for edge in PG_edges if X[edge].x > 0.5]

    print("# of remaining short pairs: ", len(short_pairs))

    return m, critical_nodes, short_pairs, original_short_pairs




# solve_DCNDP_upperBmin
def solve_DCNDP_upperBmin(G, k, timelimit=None, MIPgap=None, threads=10,
                warm_start=None, env=None, callback_export_dir=None,
                budget_constr="leq", X_binary=True,
                 fixed_simplicial=None, add_constraints_onthefly=False):

    print("Solving DCNP Upper Bound minimization ...")

    # create power graph
    PG = nx.power(G, 2)
    m = gp.Model(env=env)
    m.Params.Threads = threads

    if(timelimit):
        m.setParam('TimeLimit', timelimit)
    # gap = | ObjBound - ObjVal | / | ObjVal |
    if(MIPgap):
        m.setParam('MIPGap', MIPgap)

    # create y variables
    Y = m.addVars(PG.nodes, vtype=GRB.BINARY)

    if(warm_start):
        for v in warm_start:
            Y[v].Start = 1

    if(fixed_simplicial):
        for v in fixed_simplicial:
            Y[v].UB = 0


    # create x variables
    # if(include_1_hop):
    #    new_edges = [edge for edge in PG.edges]
    # else:
    #    new_edges = [edge for edge in PG.edges if edge not in G.edges]
    G_edges = []
    for edge in G.edges:
        if(edge[0] < edge[1]):
            G_edges.append(tuple(edge))
        else:
            G_edges.append((edge[1], edge[0]))

    PG_edges = []
    for edge in PG.edges:
        if(edge[0] < edge[1]):
            PG_edges.append(tuple(edge))
        else:
            PG_edges.append((edge[1], edge[0]))

    new_edges_temp = [edge for edge in PG.edges if edge not in G.edges]
    new_edges = []
    for edge in new_edges_temp:
        if(edge[0] < edge[1]):
            new_edges.append(tuple(edge))
        else:
            new_edges.append((edge[1], edge[0]))

    original_short_pairs = len(PG_edges)
    print("Number of original edges in the power graph: ", len(PG_edges))
    print("Number of new edges in the power graph: ", len(new_edges))
    # print("# of short pairs: ", original_short_pairs)

    #X = m.addVars(new_edges, vtype=GRB.BINARY)

    #X = m.addVars(PG_edges, vtype=GRB.CONTINUOUS)
    if(X_binary):
        X = m.addVars(G_edges, vtype=GRB.BINARY)
    else:
        X = m.addVars(G_edges, vtype=GRB.CONTINUOUS)

    obj = gp.quicksum(X)
    for edge in new_edges:
        common_neighbors = list(nx.common_neighbors(G, edge[0], edge[1]))
        obj += (1/ len(common_neighbors)) * gp.quicksum((1 - Y[vertex]) for vertex in common_neighbors)
    m.setObjective(obj, GRB.MINIMIZE)

    m._X = X
    m._Y = Y
    m._new_edges = new_edges
    m._G = G
    if(add_constraints_onthefly):
        m.Params.LazyConstraints = 1
    if(not add_constraints_onthefly):
        # add covering constraints
        for edge in new_edges:
            common_neighbors = list(nx.common_neighbors(G, edge[0], edge[1]))
            # print(common_neighbors)
            #if(set_new==False): #reduce 2b constraints
            #    m.addConstr(X[edge] + Y[edge[0]] <= 1)
            #    m.addConstr(X[edge] + Y[edge[1]] <= 1)
            

            m.addConstrs(X[min(edge[0], vertex), max(edge[0], vertex)] +
                        X[min(edge[1], vertex), max(edge[1], vertex)] +
                        Y[vertex] - 1 <= gp.quicksum((1 - Y[vertex])
                    for vertex in common_neighbors) for vertex in common_neighbors)

            for vertex in common_neighbors:
                m.addConstr(X[min(edge[0], vertex), max(edge[0], vertex)] -
                    Y[vertex] <= gp.quicksum((1 - Y[vertex])
                        for vertex in common_neighbors))
                
                m.addConstr(X[min(vertex, edge[1]), max(vertex, edge[1])] -
                    Y[vertex] <= gp.quicksum((1 - Y[vertex])
                        for vertex in common_neighbors))

            #m.addConstr( X[edge] <= X[(min(edge[1]), edge[0])])
            #    m.addConstr(X[edge] + Y[vertex] <= 1)

    # add constraints for edges in G
    m.addConstrs(X[edge] + Y[edge[0]] + Y[edge[1]] >= 1 for edge in G_edges)


    print("Loop constaints added")
    # add budget constraints
    #print("BUDGET: ", k)
    if(budget_constr == "leq"):
        m.addConstr(gp.quicksum(Y) <= k)
    elif(budget_constr == "eq"):
        m.addConstr(gp.quicksum(Y) == k)
    print("All constaints added")
    #Y[0].lb = 1

    # Optimize model
    if(add_constraints_onthefly):
        m.optimize(onthefly_constraints)
    if(callback_export_dir):
        make_dir(callback_export_dir)
        # write csv file with just headers
        with open(os.path.join(callback_export_dir, "MIP_sols_report.csv"), "w") as f:
            f.write("nodecnt,obj,solcnt,sol_phase,gap,runtime\n")
        with open(os.path.join(callback_export_dir, "MIP_report.csv"), "w") as f:
            f.write(f"nodecnt,objbst,objbnd,solcnt,gap,runtime\n")
        m._export_path = callback_export_dir
        m.optimize(mycallback)
    else:
        print("Start optimizing...")
        #m.optimize()
        #added callback:
        m.optimize()

    # print(m.display())

    # retrieve solutions
    #critical_nodes = [vertex for vertex in PG.nodes if Y[vertex].x > 0.5]
    common_neighbor_dict = {edge: [vertex for vertex in nx.common_neighbors(G, edge[0], edge[1]) if Y[vertex].x > 0.5] for edge in new_edges}
    #print(common_neighbor_dict)
    removed_edges = [edge for edge in new_edges if len(common_neighbor_dict[edge]) == len(list(nx.common_neighbors(G, edge[0], edge[1])))]
    print("Len of removed new edges: ", len(removed_edges))

    critical_nodes = [vertex for vertex in G.nodes if Y[vertex].x > 0.5]
    
    print("# of critical nodes: ", len(critical_nodes))

    short_pairs = [edge for edge in G_edges if X[edge].x > 0.5]
    remaining_edges = len(new_edges) - len(removed_edges) + len(short_pairs)
    #print("# of remaining short pairs: ", len(short_pairs))

    #return m, critical_nodes, short_pairs, original_short_pairs
    return m, critical_nodes, remaining_edges
