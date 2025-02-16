# ontology_builder.py

import networkx as nx

def build_graph():
    G = nx.DiGraph()
    nodes = ['character', 'background', 'belongings', 'cloth', 'action', 'details']
    G.add_nodes_from(nodes)

    edges = [
        ('belongings', 'character'),
        ('cloth', 'character'),
        ('action', 'character'),
        ('action', 'belongings'),
        ('details', 'character'),
        ('details', 'belongings')
    ]
    G.add_edges_from(edges)

    return G
