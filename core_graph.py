import networkx as nx
import numpy as np

def generate_dummy_trees(num_trees=50, grid_size=100):
    
    coords = np.random.rand(num_trees, 2) * grid_size
    return coords

def build_ecological_graph(tree_coords, jump_radius=15.0):
   
    G = nx.Graph()
    
    #(x,y) coords 
    for i, (x, y) in enumerate(tree_coords):
        G.add_node(i, pos=(x, y))
        
    
    for i in range(len(tree_coords)):
        for j in range(i + 1, len(tree_coords)):
            dist = np.linalg.norm(tree_coords[i] - tree_coords[j])
            if dist <= jump_radius:
                G.add_edge(i, j, weight=dist)
                
    return G

def get_network_health(G):
    
    if len(G.nodes) == 0:
        return 0, 0
        
    
    connected_components = nx.number_connected_components(G)
    
    
    largest_cc = max(nx.connected_components(G), key=len)
    largest_habitat_size = len(largest_cc)
    
    return connected_components, largest_habitat_size

def remove_tree(G, node_id):
    
    if G.has_node(node_id):
        G.remove_node(node_id)
    return G