from typing import TYPE_CHECKING

import networkx as nx
from bs4 import BeautifulSoup
from networkx.algorithms.community import louvain_communities
from pyvis.network import Network

if TYPE_CHECKING:
    from pathlib import Path

    from discograph.data import MutualFriends, User

"List of distinct colors for community visualization."
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

DARK_MODE_CSS = """
    <style>
    body { background: black !important; color: white !important; }
    #mynetwork { background: black !important; }
    .vis-configuration {
        background: black !important;
        color: white !important;
        border-color: #444 !important;
    }
    </style>
"""


def expr_size(x: float, s_min: float, k: float) -> float:
    """Calculate the expression size based on input value, minimum size,
    and a scaling factor (affine transformation).

    Parameters
    ----------
    x : float
        The input value to scale.
    s_min : float
        The minimum size to add.
    k : float
        The scaling factor to multiply with x.

    Returns
    -------
    float
        The calculated expression size as (x * k + s_min).

    """
    return x * k + s_min


def add_friend_node(
    graph: nx.Graph,
    friend: User,
    nb_connections: int,
    color: str | None = None,
) -> None:
    """Add a friend node to the given NetworkX graph with custom
    visualization attributes.

    The node is added with visualization attributes such as label, title,
    shape, image, size, font, color, and border width for selected state.

    Parameters
    ----------
    graph : networkx.Graph
        The NetworkX graph to which the friend node will be added.
    friend : User
        The user object representing the friend.
    nb_connections: int
        The number of connections the friend has, used to determine
        node size and display information.
    color : str
        The color (in hex format) to use for the node. Defaults is blue
        for highlight and lightblue for background.

    """
    default_border_color = color or "blue"
    default_bg_color = color or "lightblue"
    default_font_color = color or "white"
    node_color = {
        "border": "black",
        "highlight": {
            "border": default_border_color,
            "background": default_bg_color,
        },
    }

    graph.add_node(
        friend.id,
        label=friend.display_name,
        title=(
            f"{friend.display_name}({friend.username}) has {nb_connections} connections"
        ),
        shape="circularImage",
        image=friend.avatar_url() or "",
        size=expr_size(nb_connections, s_min=5, k=2),
        font={
            "color": default_font_color,
            "size": 50,
            "strokeWidth": 2,
        },
        color=node_color,
        borderWidthSelected=5,
    )


def combine_hex_values(d: dict[str, float]) -> str:
    """Combine multiple hex color values weighted by their values.

    Parameters
    ----------
    d : dict[str, float]
        A dictionary where keys are hex color strings (without #) and
        values are their corresponding weights.

    Returns
    -------
    str
        Combined hex color value without # prefix.

    """
    tot_weight = sum(d.values())
    if tot_weight == 0:
        return "000000"

    sorted_items = sorted(d.items())

    # weighted average for each channel
    red = int(sum(int(k[:2], 16) * v for k, v in sorted_items) / tot_weight)
    green = int(sum(int(k[2:4], 16) * v for k, v in sorted_items) / tot_weight)
    blue = int(sum(int(k[4:6], 16) * v for k, v in sorted_items) / tot_weight)

    return f"{red:02x}{green:02x}{blue:02x}"


def decorate_friend_connection(
    graph: nx.Graph,
    friend1_id: str,
    friend2_id: str,
    color1: str | None,
    color2: str | None,
) -> None:
    """Decorate an existing friendship connection (edge) between two users
    in the given graph.

    The edge will be updated to include visual attributes such as color and
    selection width for visualization purposes.

    Parameters
    ----------
    graph: networkx.Graph
        The NetworkX graph containing the friendship connection.
    friend1_id: str
        The ID of the first user in the friendship connection.
    friend2_id: str
        The ID of the second user in the friendship connection.
    color1: str
        The color (in hex format) associated with the first user. Defaults to white.
    color2: str
        The color (in hex format) associated with the second user. Defaults to white.

    """
    edge_data = graph.get_edge_data(friend1_id, friend2_id)
    if edge_data is None:
        return
    color1 = color1 or "#ffffff"
    color2 = color2 or "#ffffff"
    if color1 == color2:
        color = color1
    else:
        combined = combine_hex_values(
            {
                color1.lstrip("#"): 1,
                color2.lstrip("#"): 1,
            },
        )
        color = f"#{combined}"

    # add some transparency
    color += "BF"

    edge_data.update(
        {
            "color": {"color": color, "highlight": "white"},
            "selectionWidth": edge_data.get("selectionWidth", 4) + 2,
        },
    )


def create_graph(mutual_friends: MutualFriends) -> tuple[nx.Graph, dict[str, str]]:
    """Create a NetworkX graph from mutual friends with community detection.

    Parameters
    ----------
    mutual_friends : MutualFriends
        The mutual friends data used to build the graph.

    Returns
    -------
    tuple[nx.Graph, dict[str, str]]
        A tuple containing the created NetworkX graph and a dictionary
        mapping friend IDs to their community colors.

    Raises
    ------
    ValueError
        If the number of detected communities exceeds the number of available colors.

    """
    graph: nx.Graph = nx.Graph()
    edges = [
        (friend_id, mutual_friend_id)
        for friend_id, mutuals in mutual_friends.friends.items()
        for mutual_friend_id in mutuals.mutual_ids
        if mutual_friend_id > friend_id  # Avoid duplicate edges
    ]
    graph.add_edges_from(edges)

    communities = list(louvain_communities(graph, resolution=1.2))
    if len(communities) > len(COLORS):
        msg = (
            f"Number of detected communities {(len(communities))} "
            f"exceeds available colors {len(COLORS)}. "
        )
        raise ValueError(msg)

    community_colors = {
        member_id: COLORS[i]
        for i, community in enumerate(communities)
        for member_id in community
    }

    for friend_id, mutuals in mutual_friends.friends.items():
        friend = mutuals.user
        nb_connections = len(mutuals.mutual_ids)
        color = community_colors.get(friend_id)
        add_friend_node(graph, friend, nb_connections, color)

    for edge in graph.edges:
        friend1_id, friend2_id = edge
        color1 = community_colors.get(friend1_id)
        color2 = community_colors.get(friend2_id)
        decorate_friend_connection(
            graph,
            friend1_id,
            friend2_id,
            color1,
            color2,
        )
    return graph, community_colors


def create_network(graph: nx.Graph, *, notebook=False) -> Network:
    """Create a pyvis Network object from a NetworkX graph with specific
    visualization settings.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to be converted into a pyvis Network.
    notebook : bool, default: False
        Whether the network is to be displayed in a Jupyter notebook.

    Returns
    -------
    Network
        The pyvis Network object representing the graph.

    """
    nt: Network = Network(
        width="100%",
        notebook=notebook,
        cdn_resources="in_line" if notebook else "local",
    )
    nt.from_nx(graph)
    nt.toggle_physics(status=False)
    nt.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0,
    )
    nt.show_buttons(filter_=["manipulation", "physics", "selection", "renderer"])
    return nt


def write_html_graph(network: Network, path: Path) -> None:
    """Write the HTML representation of the network graph to the specified path,
    with dark mode CSS injected.

    Parameters
    ----------
    network : Network
        The pyvis Network object representing the graph to export.
    path : Path
        The file path where the HTML graph will be saved.

    """
    file = network.generate_html()
    file = BeautifulSoup(file, "html.parser")
    file_div = file.div
    if file_div is not None:
        file_div.unwrap()
    if file.head is not None:
        file.head.append(BeautifulSoup(DARK_MODE_CSS, "html.parser"))
    path.write_text(str(file), encoding="utf-8")
