import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import os
import core_graph as cg
import core_vision as cv

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ArborNet ", layout="wide")
st.title("ArborNet _-_-_-  Ecological Connectivity Simulator")
st.markdown("Analyzing urban canopy fragmentation. *Green nodes = Trees, Lines = Navigable animal pathways.*")

IMAGE_PATH = "data/test_image.jpg"

# -----------------------------------------------------------------------------
# 2. State Management
# -----------------------------------------------------------------------------
if 'tree_coords' not in st.session_state:
    if not os.path.exists(IMAGE_PATH):
        st.error(f"❌ Could not find image at {IMAGE_PATH}. Please check the filename!")
        st.stop()
        
    with st.spinner("Analyzing satellite imagery..."):
        st.session_state.tree_coords = cv.get_real_trees(IMAGE_PATH)
    
if 'G' not in st.session_state:
    if len(st.session_state.tree_coords) == 0:
        st.warning("uyarı:  0 trees found in this image.")
        st.session_state.G = nx.Graph()
    else:
        st.session_state.G = cg.build_ecological_graph(st.session_state.tree_coords, jump_radius=15.0)
        for node in st.session_state.G.nodes:
            st.session_state.G.nodes[node]['type'] = 'natural'

if 'baseline_components' not in st.session_state:
    comp, hab = cg.get_network_health(st.session_state.G)
    st.session_state.baseline_components = comp
    st.session_state.baseline_habitat = hab

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Sim Controls")

# THE FIX: Blast a horizontal highway through the middle to FORCE a split
if st.sidebar.button(" Simulate Infrastructure (Power Line Clear-Cut)"):
    if len(st.session_state.G.nodes) > 0:
        pos = nx.get_node_attributes(st.session_state.G, 'pos')
        if pos:
            all_y = [p[1] for p in pos.values()]
            mid_y = sum(all_y) / len(all_y)
            # Identify all trees in the "highway zone" and clear-cut them
            doomed_nodes = [n for n, p in pos.items() if mid_y - 12 < p[1] < mid_y + 12]
            st.session_state.G.remove_nodes_from(doomed_nodes)
            st.rerun()

if st.sidebar.button(" Construct Wildlife Overpass (Canopy Bridge)"):
    components = list(nx.connected_components(st.session_state.G))
    
    # THE FIX: Warn the user if they haven't broken the map yet
    if len(components) <= 1:
        st.sidebar.warning("The ecosystem is already fully connected! Build a highway first.")
    else:
        # Sort to find the two largest isolated chunks
        components = sorted(components, key=len, reverse=True)
        comp1, comp2 = components[0], components[1]
        
        pos = nx.get_node_attributes(st.session_state.G, 'pos')
        min_dist = float('inf')
        best_pair = None
        
        # Find the shortest gap between the two separate forests
        for n1 in comp1:
            for n2 in comp2:
                p1, p2 = pos[n1], pos[n2]
                dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (n1, n2)
        
        if best_pair:
            n1_id, n2_id = best_pair
            p1, p2 = pos[n1_id], pos[n2_id]
            steps = max(2, int(min_dist / 8))
            
            # Plant neon pink trees along the gap
            for i in range(1, steps):
                t = i / steps
                new_x = p1[0] + t * (p2[0] - p1[0])
                new_y = p1[1] + t * (p2[1] - p1[1])
                
                new_node_id = f"planted_{len(st.session_state.G.nodes)}"
                st.session_state.G.add_node(new_node_id, pos=(new_x, new_y), type='planted')
            
            # Re-wire the ecosystem with the new trees
            all_pos = nx.get_node_attributes(st.session_state.G, 'pos')
            nodes_list = list(st.session_state.G.nodes)
            
            for i, node1 in enumerate(nodes_list):
                for j, node2 in enumerate(nodes_list):
                    if i < j:
                        pt1, pt2 = all_pos[node1], all_pos[node2]
                        d = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
                        if d <= 15.0:
                            st.session_state.G.add_edge(node1, node2)
            st.rerun()

if st.sidebar.button("Reset Forest"):
    del st.session_state['tree_coords']
    del st.session_state['G']
    del st.session_state['baseline_components']
    st.rerun()

# -----------------------------------------------------------------------------
# 4. Metrics Dashboard
# -----------------------------------------------------------------------------
current_comp, current_hab = cg.get_network_health(st.session_state.G)

m1, m2, m3 = st.columns(3)
m1.metric("Total Surviving Trees", len(st.session_state.G.nodes))
m2.metric("Isolated Sub-networks", current_comp, 
            delta=current_comp - st.session_state.baseline_components, delta_color="inverse")
m3.metric("Largest Connected Habitat Size", current_hab, 
            delta=current_hab - st.session_state.baseline_habitat, delta_color="normal")

st.divider()

# -----------------------------------------------------------------------------
# 5. Dual Visualization (Augmented Reality Mode)
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Imagery")
    st.image(IMAGE_PATH, use_column_width=True)

with col2:
    st.subheader("Actionable Real-World Overlay")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # THE FIX: Load the real image as the background for our graph!
    img = plt.imread(IMAGE_PATH)
    # Map the image to our 0-100 coordinate grid so the nodes align perfectly
    ax.imshow(img, extent=[0, 100, 0, 100])

    pos = nx.get_node_attributes(st.session_state.G, 'pos')

    if len(pos) > 0:
        natural_nodes = [n for n in st.session_state.G.nodes if st.session_state.G.nodes[n].get('type', 'natural') == 'natural']
        planted_nodes = [n for n in st.session_state.G.nodes if st.session_state.G.nodes[n].get('type') == 'planted']
        
        # Draw the natural green graph (made slightly transparent so you can still see the city)
        nx.draw_networkx_nodes(st.session_state.G, pos, nodelist=natural_nodes, ax=ax, node_color='#00FFAA', node_size=20, alpha=0.5)
        nx.draw_networkx_edges(st.session_state.G, pos, ax=ax, edge_color='#FFFFFF', width=0.8, alpha=0.3)
        
        # Draw the neon pink intervention overlay (High contrast, zero transparency)
        if planted_nodes:
            nx.draw_networkx_nodes(st.session_state.G, pos, nodelist=planted_nodes, ax=ax, node_color='#FF007F', node_size=80, edgecolors='#FFFFFF', linewidths=1.5)
            # Find and highlight the specific paths that use the planted nodes
            connected_to_planted = [(u, v) for u, v in st.session_state.G.edges if (u in planted_nodes or v in planted_nodes)]
            nx.draw_networkx_edges(st.session_state.G, pos, edgelist=connected_to_planted, ax=ax, edge_color='#FF007F', width=3.5, style='dashed', alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)