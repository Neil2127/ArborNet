import streamlit as st

# CRUCIAL: This must happen before ANY other custom imports execute
st.set_page_config(page_title="ArborNet MVP", layout="wide")

import matplotlib.pyplot as plt
import networkx as nx
import os
import core_graph as cg
import core_vision as cv

# -----------------------------------------------------------------------------
# 1. Page Title & Setup
# -----------------------------------------------------------------------------
st.title("ArborNet: Ecological Connectivity Simulator")
st.markdown("Analyzing urban canopy fragmentation. *Green nodes = Trees, Lines = Navigable animal pathways.*")

IMAGE_PATH = "data/test_image1.jpg"

# -----------------------------------------------------------------------------
# 2. State Management
# -----------------------------------------------------------------------------
if 'tree_coords' not in st.session_state:
    if not os.path.exists(IMAGE_PATH):
        st.error(f" Could not find image at {IMAGE_PATH}. Please check the filename!")
        st.stop()
        
    with st.spinner("Analyzing satellite imagery..."):
        st.session_state.tree_coords = cv.get_real_trees(IMAGE_PATH)
    
if 'G' not in st.session_state:
    if len(st.session_state.tree_coords) == 0:
        st.warning("uyar :  0 trees found in this image.")
        st.session_state.G = nx.Graph()
    else:
        st.session_state.G = cg.build_ecological_graph(st.session_state.tree_coords, jump_radius=15.0)
        for node in st.session_state.G.nodes:
            st.session_state.G.nodes[node]['type'] = 'natural'

if 'baseline_components' not in st.session_state:
    # ADD 'fied' HERE TO CATCH THE THIRD VALUE
    comp, hab, fied = cg.get_network_health(st.session_state.G)
    st.session_state.baseline_components = comp
    st.session_state.baseline_habitat = hab
    st.session_state.baseline_fiedler = fied  # Save the baseline Fiedler value

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Simulation Controls")

if st.sidebar.button(" Simulate Infrastructure dev()"):
    if len(st.session_state.G.nodes) > 0:
        pos = nx.get_node_attributes(st.session_state.G, 'pos')
        if pos:
            all_y = [p[1] for p in pos.values()]
            mid_y = sum(all_y) / len(all_y)
            doomed_nodes = [n for n, p in pos.items() if mid_y - 12 < p[1] < mid_y + 12]
            st.session_state.G.remove_nodes_from(doomed_nodes)
            st.rerun()
if st.sidebar.button("🌉 Deploy Optimized Eco-Corridor (Miyawaki Method)"):
    # Get all disconnected chunks
    all_components = list(nx.connected_components(st.session_state.G))
    
    # FILTER: Ignore tiny isolated noise (components with less than 3 trees)
    components = [c for c in all_components if len(c) >= 3]
    
    # Fallback to all components if filtering empties the list
    if len(components) < 2:
        components = all_components

    if len(components) <= 1:
        st.sidebar.warning("The main ecosystem blocks are already connected!")
    else:
        # Sort to get the two largest remaining habitat blocks
        components = sorted(components, key=len, reverse=True)
        comp1, comp2 = components[0], components[1]
        
        pos = nx.get_node_attributes(st.session_state.G, 'pos')
        
        # 1.  scan ALL crossing pairs to find the absolute true gap distance
        all_pairs = []
        for n1 in comp1:
            for n2 in comp2:
                p1, p2 = pos[n1], pos[n2]
                dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                all_pairs.append((dist, n1, n2))
        
        # Sort by distance to find the actual structural divide
        all_pairs.sort(key=lambda x: x[0])
        
        # Filter out micro-glitches under 15 pixels, then grab the true closest connection
        valid_pairs = [p for p in all_pairs if p[0] > 15.0]
        
        if valid_pairs:
            # Target the absolute closest points across the real barrier
            chosen_dist, n1_id, n2_id = valid_pairs[0]
            
            p1, p2 = pos[n1_id], pos[n2_id]
            # Determine step count based on the true size of the gap
            steps = max(2, int(chosen_dist / 10))
            
            # 2. Deploy a single clean linear corridor directly across the void
            for i in range(1, steps):
                t = i / steps
                new_x = p1[0] + t * (p2[0] - p1[0])
                new_y = p1[1] + t * (p2[1] - p1[1])
                
                new_node_id = f"planted_{len(st.session_state.G.nodes)}"
                st.session_state.G.add_node(new_node_id, pos=(new_x, new_y), type='planted')
            
            # 3. Re-mesh the network paths globally using the new nodes
            all_pos = nx.get_node_attributes(st.session_state.G, 'pos')
            nodes_list = list(st.session_state.G.nodes)
            
            # Clear old edges and rebuild to ensure global network consistency
            st.session_state.G.clear_edges()
            for i, node1 in enumerate(nodes_list):
                for j, node2 in enumerate(nodes_list):
                    if i < j:
                        pt1, pt2 = all_pos[node1], all_pos[node2]
                        d = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
                        # Use a slightly forgiving radius to ensure the new bridge hooks in perfectly
                        if d <= 16.5: 
                            st.session_state.G.add_edge(node1, node2)
            st.rerun()
        else:
            st.sidebar.error("No valid fragmentation gap detected across the main sub-networks.")

if st.sidebar.button("Reset Forest"):
    del st.session_state['tree_coords']
    del st.session_state['G']
    del st.session_state['baseline_components']
    st.rerun()

# -----------------------------------------------------------------------------
# 4. Metrics Dashboard
# -----------------------------------------------------------------------------
# Ensure this line is catching all 3 variables too
current_comp, current_hab, current_fiedler = cg.get_network_health(st.session_state.G)

m1, m2, m3 = st.columns(3)
m1.metric("Total Surviving Trees", len(st.session_state.G.nodes))

m2.metric("Isolated Sub-networks", current_comp, 
            delta=current_comp - st.session_state.baseline_components, delta_color="inverse")

# connection or resilience gradient 
fiedler_delta = current_fiedler - st.session_state.baseline_fiedler
m3.metric("Ecosystem Resilience (Fiedler Value)", f"{current_fiedler:.3f}", 
            delta=f"{fiedler_delta:.3f}", delta_color="normal")

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
    
    img = plt.imread(IMAGE_PATH)
    ax.imshow(img, extent=[0, 100, 0, 100])

    pos = nx.get_node_attributes(st.session_state.G, 'pos')

    if len(pos) > 0:
        natural_nodes = [n for n in st.session_state.G.nodes if st.session_state.G.nodes[n].get('type', 'natural') == 'natural']
        planted_nodes = [n for n in st.session_state.G.nodes if st.session_state.G.nodes[n].get('type') == 'planted']
        
        nx.draw_networkx_nodes(st.session_state.G, pos, nodelist=natural_nodes, ax=ax, node_color='#00FFAA', node_size=20, alpha=0.5)
        nx.draw_networkx_edges(st.session_state.G, pos, ax=ax, edge_color='#FFFFFF', width=0.8, alpha=0.3)
        
        if planted_nodes:
            nx.draw_networkx_nodes(st.session_state.G, pos, nodelist=planted_nodes, ax=ax, node_color='#FF007F', node_size=80, edgecolors='#FFFFFF', linewidths=1.5)
            connected_to_planted = [(u, v) for u, v in st.session_state.G.edges if (u in planted_nodes or v in planted_nodes)]
            nx.draw_networkx_edges(st.session_state.G, pos, edgelist=connected_to_planted, ax=ax, edge_color='#FF007F', width=3.5, style='dashed', alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)