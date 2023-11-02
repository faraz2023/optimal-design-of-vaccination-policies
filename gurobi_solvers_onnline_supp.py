import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def solve_modified_1hop_DCND(G, b, timelimit=None, MIPgap=None, threads=16, warm_start=None, env=None, ineffective_coefficient=0.05):
    print("Solving updated DCNDP ...")

    m = gp.Model(env=env)
    m.Params.Threads = threads

    if timelimit:
        m.setParam('TimeLimit', timelimit)
    if MIPgap:
        m.setParam('MIPGap', MIPgap)

    G_edges_min_ordering = [(min(e), max(e)) for e in G.edges]
    # Define variables
    Y = m.addVars(G.nodes, vtype=GRB.BINARY, name="Y")
    X = m.addVars(G_edges_min_ordering, vtype=GRB.BINARY, name="X") # Binary or continous?
    Q = m.addVars(G.edges, vtype=GRB.CONTINUOUS, name="Q")
    Z = m.addVars(G.edges, vtype=GRB.CONTINUOUS, name="Z")
    W = {e: ineffective_coefficient for e in G.edges} # vaccine inffectiveness coeficient

    # Warm start
    if warm_start:
        for v in warm_start:
            Y[v].Start = 1

    # Constraints
    print("-=======")

    m.addConstrs(1 - Y[u] - Y[v] <= X[(min([u,v]), max([u,v]))] for u, v in G.edges)
    m.addConstr(gp.quicksum(Y) <= b)
    m.addConstrs(Y[v] <= 1 - X[(min(e), max(e))] for v in G.nodes for e in G.edges(v))
    m.addConstrs(1 - X[(min(e), max(e))] == Q[e] + Z[e] for e in G.edges)
    m.addConstrs(Y[u] + Y[v] <= 1 + Z[u, v] for u, v in G.edges)
    m.addConstrs(Z[u, v] <= Y[u] for u, v in G.edges)
    m.addConstrs(Z[u, v] <= Y[v] for u, v in G.edges)

    # Objective function
    obj = gp.quicksum(X[(min(e), max(e))] for e in G.edges) + gp.quicksum(W[e] * Q[e] for e in G.edges) + gp.quicksum(W[e]**2 * Z[e] for e in G.edges)
    m.setObjective(obj, GRB.MINIMIZE)

    print("All constraints added")
    print("Start optimizing...")

    m.optimize()

    critical_nodes = [vertex for vertex in G.nodes if Y[vertex].x > 0.5]
    print("# of critical nodes: ", len(critical_nodes))

    return m, critical_nodes

def solve_modified_2hop_DCND(G, b, timelimit=None, MIPgap=None, threads=10,
                    warm_start=None, env=None, budget_constr="leq",
                    aggregated_constraints=True, X_binary=False,
                    fixed_simplicial=None, new_ver=False, ineffective_coefficient=0.05):
    
    print("Solving 2hop DCND ...")

    # create power graph
    PG = nx.power(G, 2)
    m = gp.Model(env=env)
    m.Params.Threads = threads

    if timelimit:
        m.setParam('TimeLimit', timelimit)
    if MIPgap:
        m.setParam('MIPGap', MIPgap)

    m.setParam('Method', 3)

    # create y variables
    Y = m.addVars(PG.nodes, vtype=GRB.BINARY)
    if(X_binary):
        X = m.addVars(PG.edges, vtype=GRB.BINARY)
    else:
        X = m.addVars(PG.edges, vtype=GRB.CONTINUOUS)
    Q = m.addVars(PG.edges, vtype=GRB.CONTINUOUS)
    Z = m.addVars(PG.edges, vtype=GRB.CONTINUOUS)
    W = {e: ineffective_coefficient for e in PG.edges}  # vaccine inffectiveness coeficient

    if warm_start:
        for v in warm_start:
            Y[v].Start = 1

    if fixed_simplicial:
        for v in fixed_simplicial:
            Y[v].UB = 0

    # Constraints
    pg_minus_g_edges = set(PG.edges) - set(G.edges)

    #2b and 2f
    for u, v in pg_minus_g_edges:
        common_neighbors = list(nx.common_neighbors(G, u, v))
        common_neighbors_count = len(common_neighbors)
        
        # b part
        quicksum_common_neighbors = gp.quicksum((1 - Y[i]) for i in common_neighbors)
        m.addConstr((1 / common_neighbors_count) * quicksum_common_neighbors - Y[u] - Y[v] <= X[u, v])

        # f part
        quicksum_common_neighbors = gp.quicksum(Y[i] for i in common_neighbors)
        m.addConstr(quicksum_common_neighbors - Y[u] - Y[v] <= common_neighbors_count - X[u, v])

    
    #2c
    m.addConstrs(1 - Y[u] - Y[v] <= X[u, v] for u, v in G.edges)
    #2d
    m.addConstr(gp.quicksum(Y) <= b)

    #2e
    m.addConstrs(Y[v] <= 1 - X[(min(e), max(e))] for v in G.nodes for e in PG.edges(v))
        

    #2g
    m.addConstrs(1 - X[e] == Q[e] + Z[e] for e in PG.edges)

    #2h
    m.addConstrs(Y[u] + Y[v] <= 1 + Z[u, v] for u, v in PG.edges)
    m.addConstrs(Z[u, v] <= Y[u] for u, v in PG.edges)
    m.addConstrs(Z[u, v] <= Y[v] for u, v in PG.edges)

    # Objective function
    obj = gp.quicksum(X[e] for e in PG.edges) + gp.quicksum(W[e] * Q[e] + W[e]**2 * Z[e] for e in G.edges) + gp.quicksum(W[e]**2 * Q[e] + W[e]**4 * Z[e] for e in pg_minus_g_edges)
    m.setObjective(obj, GRB.MINIMIZE)

    print("All constraints added")
    print("Start optimizing...")

    m.optimize()

    critical_nodes = [vertex for vertex in PG.nodes if Y[vertex].x > 0.5]
    print("# of critical nodes: ", len(critical_nodes))

    short_pairs = [edge for edge in PG.edges if X[edge].x > 0.5]
    print("# of remaining short pairs: ", len(short_pairs))

    return m, critical_nodes, short_pairs, len(PG.edges)

