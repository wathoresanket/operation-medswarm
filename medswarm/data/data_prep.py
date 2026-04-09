"""
data_prep.py
------------
Downloads the road network for Connaught Place from OpenStreetMap
and builds two distance matrices:
  - ambulance_matrix: shortest road distance between all node pairs
  - drone_matrix:     straight-line (Euclidean) distance between all node pairs

If the internet is unavailable, a synthetic fallback is generated automatically.
"""

import os
import pickle
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def prepare_data(
    location: str = "Connaught Place, New Delhi, India",
    num_zones: int = 12,
    seed: int = 42,
    save_path: str = "data/medswarm_data.pkl",
) -> dict:
    """
    Downloads the real road network and builds everything the env needs.
    Falls back to synthetic data if OSM download fails.

    Returns a dict with keys:
      - hospital_node:      int, node ID of the base hospital
      - zone_nodes:         list of 12 int node IDs (triage zones)
      - ambulance_matrix:   (13x13) road distances in meters
      - drone_matrix:       (13x13) Euclidean distances in meters
      - node_coords:        dict {node_id: (lat, lon)}
      - graph:              networkx graph (or None if synthetic)
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[data_prep] Trying to download OpenStreetMap data...")
    data = _try_osm_download(location, num_zones, seed)

    if data is None:
        print("[data_prep] OSM download failed — using synthetic data instead.")
        data = _build_synthetic_data(num_zones, seed)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[data_prep] Saved to {save_path}")
    _print_summary(data)
    return data


def load_data(path: str = "data/medswarm_data.pkl") -> dict:
    """Loads previously saved data from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# OSM download path
# ─────────────────────────────────────────────

def _try_osm_download(location, num_zones, seed):
    """
    Downloads the road graph from OSM and picks nodes for the hospital + zones.
    Returns None if anything goes wrong (no internet, timeout, etc).
    """
    try:
        import osmnx as ox
        import networkx as nx

        print(f"[data_prep] Downloading: {location}")
        G = ox.graph_from_place(location, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        nodes = list(G.nodes())
        random.seed(seed)

        # pick hospital + 12 zone nodes — make sure they're all different
        chosen = random.sample(nodes, num_zones + 1)
        hospital_node = chosen[0]
        zone_nodes = chosen[1:]

        all_nodes = [hospital_node] + zone_nodes  # index 0 = hospital, 1-12 = zones
        n = len(all_nodes)

        # build road distance matrix using Dijkstra on real road network
        print("[data_prep] Computing road distance matrix (this takes ~30s)...")
        amb_matrix = np.zeros((n, n))
        for i, src in enumerate(all_nodes):
            for j, dst in enumerate(all_nodes):
                if i == j:
                    continue
                try:
                    length = nx.shortest_path_length(G, src, dst, weight="length")
                    amb_matrix[i][j] = length
                except nx.NetworkXNoPath:
                    # fallback: use Euclidean if no road path exists
                    amb_matrix[i][j] = _euclidean(G, src, dst)

        # build drone distance matrix (straight-line, no roads)
        print("[data_prep] Computing drone (Euclidean) distance matrix...")
        drone_matrix = np.zeros((n, n))
        for i, src in enumerate(all_nodes):
            for j, dst in enumerate(all_nodes):
                if i != j:
                    drone_matrix[i][j] = _euclidean(G, src, dst)

        # store lat/lon for each node (used by the dashboard map)
        node_coords = {
            node: (G.nodes[node]["y"], G.nodes[node]["x"])
            for node in all_nodes
        }

        return {
            "hospital_node": hospital_node,
            "zone_nodes": zone_nodes,
            "ambulance_matrix": amb_matrix,
            "drone_matrix": drone_matrix,
            "node_coords": node_coords,
            "graph": G,
            "all_nodes": all_nodes,
            "source": "osm",
        }

    except Exception as e:
        print(f"[data_prep] OSM error: {e}")
        return None


def _euclidean(G, node_a, node_b):
    """Haversine distance between two graph nodes, converted to meters."""
    import math
    lat1 = math.radians(G.nodes[node_a]["y"])
    lon1 = math.radians(G.nodes[node_a]["x"])
    lat2 = math.radians(G.nodes[node_b]["y"])
    lon2 = math.radians(G.nodes[node_b]["x"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────

def _build_synthetic_data(num_zones, seed):
    """
    Builds a fake but plausible city grid — 13 nodes scattered across
    a ~2km x 2km area. Used when OSM is unavailable (e.g., no internet).
    """
    np.random.seed(seed)
    n = num_zones + 1  # hospital + zones

    # random 2D positions in a 2000m x 2000m box
    positions = np.random.uniform(0, 2000, size=(n, 2))

    # drone matrix = straight-line distances
    drone_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                drone_matrix[i][j] = np.sqrt(dx ** 2 + dy ** 2)

    # ambulance matrix = road distances (30-60% longer than straight line)
    amb_matrix = drone_matrix * np.random.uniform(1.3, 1.6, size=(n, n))
    np.fill_diagonal(amb_matrix, 0)

    # fake lat/lon around Connaught Place
    base_lat, base_lon = 28.6315, 77.2167
    node_coords = {}
    for i in range(n):
        lat = base_lat + positions[i][0] / 111000
        lon = base_lon + positions[i][1] / (111000 * np.cos(np.radians(base_lat)))
        node_coords[i] = (lat, lon)

    return {
        "hospital_node": 0,
        "zone_nodes": list(range(1, n)),
        "ambulance_matrix": amb_matrix,
        "drone_matrix": drone_matrix,
        "node_coords": node_coords,
        "graph": None,
        "all_nodes": list(range(n)),
        "source": "synthetic",
    }


def _print_summary(data):
    amb = data["ambulance_matrix"]
    drone = data["drone_matrix"]
    nonzero_amb = amb[amb > 0]
    nonzero_drone = drone[drone > 0]
    print(f"\n--- Data Summary ---")
    print(f"  Source:              {data['source']}")
    print(f"  Hospital node:       {data['hospital_node']}")
    print(f"  Triage zones:        {len(data['zone_nodes'])}")
    print(f"  Avg road distance:   {nonzero_amb.mean():.0f}m")
    print(f"  Avg drone distance:  {nonzero_drone.mean():.0f}m")
    print(f"  Max drone distance:  {nonzero_drone.max():.0f}m")
    print(f"  Matrix shape:        {amb.shape}")