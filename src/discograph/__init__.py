import asyncio
import functools
import getpass
import logging
import random
import shutil
import warnings
import webbrowser
from collections.abc import Sized
from enum import IntEnum
from os import getenv
from typing import Annotated, Final, Literal, Self

import aiohttp
import networkx as nx
from attr import dataclass
from bs4 import BeautifulSoup
from cyclopts import App, Parameter
from cyclopts.types import StdioPath
from networkx.algorithms.community import louvain_communities
from platformdirs import PlatformDirs
from pydantic import BaseModel, TypeAdapter
from pyvis.network import Network
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

APP_NAME = "discograph"
__version__ = "0.2.3"

DIRS = PlatformDirs(APP_NAME, version=__version__)
DIRS.user_state_path.mkdir(parents=True, exist_ok=True)
DIRS.user_cache_path.mkdir(parents=True, exist_ok=True)

DISCORD_API_LINK: Final = "https://discord.com/api/v10"

# ---------------------------------------------------------------------------- #
#                                    Logging                                   #
# ---------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(DIRS.user_state_path / "discord_graph.log"),
        # logging.StreamHandler(),
    ],
)
logger = logging.getLogger(APP_NAME)

# ---------------------------------------------------------------------------- #
#                                 Data Fetching                                #
# ---------------------------------------------------------------------------- #


def get_config_value(
    cli_value: str | None,
    env_var: str,
    value_name: str,
    *,
    password_like: bool = False,
) -> str | None:
    if cli_value is not None:
        logger.debug(
            "Using CLI provided value for %s",
            value_name,
        )
        return cli_value

    env_value = getenv(env_var)
    if env_value is not None:
        logger.debug(
            "Using environment variable value for %s",
            value_name,
        )
        return env_value

    input_func = (
        functools.partial(getpass.getpass, echo_char="*") if password_like else input
    )
    input_value = input_func(f"Enter value for {value_name}: ")
    if input_value:
        logger.debug(
            "Using user input value for %s",
            value_name,
        )
        return input_value

    msg = f"No value provided for {value_name}"
    logger.error(msg)
    return None


type AvatarSize = Literal[16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


class RelationShipType(IntEnum):
    none = 0
    friend = 1
    blocked = 2
    implicit = 3
    pending_incoming = 4
    pending_outgoing = 5


class UserResponse(BaseModel):
    id: str
    nickname: str | None
    user: User
    type: RelationShipType


UsersResponse = TypeAdapter(list[UserResponse])


class User(BaseModel):
    """The user information. Is equatable and hashable by ID."""

    id: str
    avatar: str | None
    discriminator: str
    username: str
    global_name: str | None

    def avatar_url(self, size: AvatarSize = 128) -> str | None:
        if self.avatar:
            url = f"https://cdn.discordapp.com/avatars/{self.id}/{self.avatar}.png?size={size}"
            logger.debug(
                "Generated avatar URL",
                extra={"username": self.username, "url": url},
            )
            return url
        logger.debug("No avatar available for user", extra={"username": self.username})
        return None

    async def download_avatar_bytes(
        self,
        session: aiohttp.ClientSession,
        size: AvatarSize = 128,
    ) -> bytes | None:
        url = self.avatar_url(size=size)
        if not url:
            logger.warning(
                "No avatar URL available for user",
                extra={"username": self.username},
            )
            return None

        logger.debug(
            "Downloading avatar",
            extra={"username": self.username, "url": url},
        )
        async with session.get(url) as response:
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                logger.exception(
                    "Failed to download avatar",
                    extra={
                        "username": self.username,
                        "status": response.status,
                        "error": str(e),
                    },
                )
                return None
            data = await response.read()
            logger.info(
                "Successfully downloaded avatar",
                extra={"username": self.username, "bytes": len(data)},
            )
            return data

    async def fetch_mutual_friends(
        self,
        session: aiohttp.ClientSession,
        user_secret: str,
    ) -> set[User]:
        link = f"{DISCORD_API_LINK}/users/{self.id}/relationships"
        headers = {"authorization": user_secret}
        logger.debug(
            "Fetching mutual friends",
            extra={"username": self.username, "user_id": self.id},
        )

        retry_count = 0
        while True:
            async with session.get(link, headers=headers) as response:
                try:
                    friends_data = await response.json()
                except aiohttp.ContentTypeError:
                    text = await response.text()
                    logger.exception(
                        "Failed to parse JSON for user",
                        extra={"username": self.username, "response_text": text},
                    )
                    raise

                if isinstance(friends_data, dict) and (
                    wait := friends_data.get("retry_after")
                ):
                    retry_count += 1
                    wait_time = wait * 1.1  # 10% buffer
                    logger.warning(
                        "Rate limited while fetching friends. Retrying after delay",
                        extra={
                            "username": self.username,
                            "retry_count": retry_count,
                            "wait_seconds": round(wait_time, 2),
                        },
                    )
                    await asyncio.sleep(wait_time)
                    continue
                break

        mutual_friends = {User.model_validate(fd) for fd in friends_data}
        logger.info(
            "Fetched mutual friends",
            extra={"username": self.username, "count": len(mutual_friends)},
        )
        return mutual_friends

    def __eq__(self, other) -> bool:
        if not isinstance(other, User):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def display_name(self) -> str:
        return self.global_name or self.username


async def get_friends_dict(
    session: aiohttp.ClientSession,
    user_secret: str,
) -> dict[str, User] | None:
    """Fetch the user's friends list from the Discord API and returns a
    dictionary mapping user IDs to friend (user) objects.

    Args:
        session (aiohttp.ClientSession): An active aiohttp session used
        to make HTTP requests.
        user_secret (str): The user's authorization token for the Discord API.

    Returns:
        dict: A dictionary where the keys are user IDs
        and the values are friend objects.

    """
    link = rf"{DISCORD_API_LINK}/users/@me/relationships"
    headers = {"authorization": user_secret}
    logger.info("Fetching friends list from Discord API")

    async with session.get(link, headers=headers) as response:
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError as e:
            logger.exception(
                "Failed to fetch friends list",
                extra={"status": response.status, "error": str(e)},
            )
            print("Error fetching friends list:", str(e))
            return None
        friends_data = await response.json()

    user_with_relationship = UsersResponse.validate_python(friends_data)
    friends = [
        fr for fr in user_with_relationship if fr.type == RelationShipType.friend
    ]
    logger.info(
        "Successfully fetched friends",
        extra={"count": len(friends)},
    )
    return {friend.user.id: friend.user for friend in friends}


class MutualFriends(BaseModel, Sized):
    """Represent a structure for managing users and their mutual friends
    within a network. It contains two tables: one for storing user
    information and another for storing mutual friends. Is a sized
    collection, the size being the number of users.

    Attributes:
        friends_table (dict[str, User]): A mapping from user IDs to User objects.
        mutual_friends (dict[str, set[str]]): A mapping from user IDs to sets of user
        IDs representing their mutual friends.

    Methods:
        from_friends(session, friends, *, progress=True):
            Asynchronously constructs a MutualFriends instance from a dictionary of
            UserResponse objects, fetching each user's mutual friends using the
            provided aiohttp session. Optionally displays a progress bar.

        fetch(session, *, progress=True):
            Asynchronously fetches the user's friends and their mutual friends,
            returning a MutualFriends instance. Optionally displays a progress bar.

    """

    friends_table: dict[str, User]
    mutual_friends: dict[str, set[str]]

    @classmethod
    async def from_friends(
        cls,
        session: aiohttp.ClientSession,
        user_secret: str,
        friends: dict[str, User],
        *,
        progress: bool = True,
    ) -> Self:
        """Asynchronously construct a MutualFriends instance from a dictionary of
        UserResponse objects, fetching each user's mutual friends using the
        provided aiohttp session. Optionally displays a progress bar.

        The fetching is done sequentially over each user in the friends dictionary,
        to avoid hitting rate limits imposed by the Discord API
        (e.g. cloudflare ip ban).

        Args:
            session (aiohttp.ClientSession): An active aiohttp session used
            to make HTTP requests.
            friends (dict[str, UserResponse]): A dictionary mapping user IDs to
            UserResponse objects (taken from get_friends_dict).
            progress (bool, optional): Whether to display a progress bar during
            the fetching process. Defaults to True.

        Returns:
            MutualFriends: An instance of MutualFriends containing the users
            and their mutual friends.

        """
        logger.info(
            "Starting to fetch mutual friends",
            extra={"friends_count": len(friends)},
        )
        self = cls(
            friends_table={},
            mutual_friends={},
        )
        progress_iter = (
            tqdm(
                friends.values(),
                total=len(friends),
                desc="Fetching mutual friends",
                unit="friend",
            )
            if progress
            else friends.values()
        )

        success_count = 0
        error_count = 0

        for friend in progress_iter:
            try:
                mutual_friends = await friend.fetch_mutual_friends(session, user_secret)
                success_count += 1
            except Exception:
                error_count += 1
                logger.exception(
                    "Error fetching mutual friends",
                    extra={"username": friend.username},
                )
                continue
            self.friends_table[friend.id] = friend
            self.mutual_friends[friend.id] = {cf.id for cf in mutual_friends}

        logger.info(
            "Completed fetching mutual friends",
            extra={"success": success_count, "errors": error_count},
        )
        return self

    @classmethod
    async def fetch(
        cls,
        session: aiohttp.ClientSession,
        user_secret: str,
        *,
        progress: bool = True,
    ) -> Self | None:
        """Asynchronously fetch and construct an instance of the class using
        friends data.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for
                HTTP requests.
            user_secret (str): The user's authorization token for the Discord API.
            progress (bool, optional): Whether to display progress information.
                Defaults to True.

        Returns:
            An instance of the class constructed from the fetched friends data.

        Logs:
            Logs the start of the fetch operation.

        """
        logger.info("Starting MutualFriends.fetch()")
        friends = await get_friends_dict(session, user_secret)
        if friends is None:
            logger.error("Failed to fetch friends dictionary")
            return None
        return await cls.from_friends(
            session,
            user_secret,
            friends,
            progress=progress,
        )

    def __len__(self):
        return len(self.mutual_friends)


# ---------------------------------------------------------------------------- #
#                                     Graph                                    #
# ---------------------------------------------------------------------------- #


def expr_size(x: float, s_min: float, k: float = 2) -> float:
    """Calculate the expression size based on input value, minimum size,
    and a scaling factor.

    Args:
        x (int or float): The input value to scale.
        s_min (int or float): The minimum size to add.
        k (int or float, optional): The scaling factor to multiply with x.
            Defaults to 2.

    Returns:
        int or float: The calculated expression size as (x * k + s_min).

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

    Parameters
    ----------
        graph (nx.Graph): The NetworkX graph to which the friend node will be added.
        friend (User): The user object representing the friend.
        nb_connections (int): The number of connections the friend has,
            used to determine node size and display information.

    The node is added with visualization attributes such as label, title,
    shape, image, size, font, color, and border width for selected state.

    """
    node_color = {
        "border": "black",
        "highlight": {
            "border": color or "blue",
            "background": color or "lightblue",
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
            "color": color or "white",
            "size": 50,
            "strokeWidth": 3,
        },
        color=node_color,
        borderWidthSelected=5,
    )


def combine_hex_values(d: dict[str, float]) -> str:
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    if tot_weight == 0:
        return "000000"
    red = int(sum(int(k[:2], 16) * v for k, v in d_items) / tot_weight)
    green = int(sum(int(k[2:4], 16) * v for k, v in d_items) / tot_weight)
    blue = int(sum(int(k[4:6], 16) * v for k, v in d_items) / tot_weight)

    def zpad(x):
        return x if len(x) == 2 else "0" + x

    return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])  # noqa: FURB116


def add_friends_connection(
    graph: nx.Graph,
    friend1: User,
    friend2: User,
    color1: str | None,
    color2: str | None,
) -> None:
    """Add a friendship connection (edge) between two users in the given
    graph.

    Parameters
    ----------
        graph (nx.Graph): The NetworkX graph to which the friendship
            connection will be added.
        friend1 (User): The first user in the friendship connection.
        friend2 (User): The second user in the friendship connection.
        color1 (str | None): The color associated with the first user.
            If None, defaults to white.
        color2 (str | None): The color associated with the second user.
            If None, defaults to white.

    The edge will include visual attributes such as color and selection
    width for visualization purposes.

    """
    color1 = color1 or "#ffffff"
    color2 = color2 or "#ffffff"
    if color1 and color2 and color1 == color2:
        color = color1
    else:
        # mixed colors (hex)
        color = "#" + combine_hex_values({color1.lstrip("#"): 1, color2.lstrip("#"): 1})
    color += "BF"
    graph.add_edge(
        friend1.id,
        friend2.id,
        color={"color": color, "highlight": "white"},
        selectionWidth=4,
    )


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


def create_graph(
    mutual_friends: MutualFriends,
    community_colors: dict[str, str],
) -> nx.Graph:
    """Create a NetworkX graph representing the network of friends and
    their mutual connections.
    Each friend is added as a node to the graph, and an edge is created
    between friends who share a mutual connection.

    Args:
        mutual_friends (MutualFriends): An object containing the friends
        table and their mutual friends mapping.
        community_colors (dict[str, str]): Mapping from user IDs to their
        community colors.

    Returns:
        nx.Graph: A NetworkX graph representing the friends network.

    """
    logger.info("Creating graph", extra={"friends_count": len(mutual_friends)})
    graph: nx.Graph = nx.Graph()

    for friend_id, friend in mutual_friends.friends_table.items():
        mutuals = mutual_friends.mutual_friends.get(friend_id, set())
        color = community_colors.get(friend_id)
        add_friend_node(graph, friend, nb_connections=len(mutuals), color=color)

        for mutual_friend_id in mutuals:
            if mutual_friend_id not in mutual_friends.friends_table:
                logger.warning(
                    "mutual friend not in friends table",
                    extra={
                        "friend_id": friend_id,
                        "mutual_friend_id": mutual_friend_id,
                    },
                )
                continue
            if mutual_friend_id > friend_id:
                continue  # avoid duplicate edges (A-B and B-A)

            mutual_friend = mutual_friends.friends_table[mutual_friend_id]
            add_friends_connection(
                graph,
                friend,
                mutual_friend,
                color,
                community_colors.get(mutual_friend_id),
            )

    logger.info(
        "Graph created",
        extra={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
    )
    return graph


def detect_communities(mutual_friends: MutualFriends) -> dict[str, str]:
    """Detect communities in the mutual friends network using Louvain algorithm.

    Args:
        mutual_friends (MutualFriends): An object containing the friends
            table and their mutual friends mapping.

    Returns:
        Mapping from user IDs to their community colors

    """
    temp_graph: nx.Graph = nx.Graph()
    for friend_id in mutual_friends.friends_table:
        if len(mutual_friends.mutual_friends.get(friend_id, set())) != 0:
            temp_graph.add_node(friend_id)

    for friend_id, mutuals in mutual_friends.mutual_friends.items():
        for mutual_friend_id in mutuals:
            if (
                mutual_friend_id in mutual_friends.friends_table
                and mutual_friend_id > friend_id
            ):
                temp_graph.add_edge(friend_id, mutual_friend_id)

    communities = list(louvain_communities(temp_graph, resolution=1.2))

    if len(communities) > len(COLORS):
        msg = (
            "Number of detected communities exceeds available colors."
            f" Detected: {len(communities)}, Available: {len(COLORS)}"
        )
        raise ValueError(msg)

    community_colors = {}
    for i, community in enumerate(communities):
        for member_id in community:
            community_colors[member_id] = COLORS[i]

    return community_colors


def create_network_graph(
    mutual_friends: MutualFriends,
    *,
    notebook=False,
) -> Network:
    logger.info("Creating network visualization")

    community_colors = detect_communities(mutual_friends)
    graph = create_graph(mutual_friends, community_colors)

    # Create network
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
    nt.show_buttons(
        filter_=[
            "manipulation",
            "physics",
            "selection",
            "renderer",
        ],
    )
    logger.info("Network visualization created successfully")
    return nt


def write_html_graph(network: Network, path: StdioPath) -> None:
    """Generate the HTML representation of the network graph and writes
    it to the specified file path.

    Parameters
    ----------
        network (Network): The network graph to be converted to HTML.
        path (Path): The file path where the HTML content will be saved.

    """
    logger.info("Writing HTML graph", extra={"path": str(path)})
    file = network.generate_html()
    file = BeautifulSoup(file, "html.parser")
    file_div = file.div
    if file_div is not None:
        file_div.unwrap()

    dark_css = """
    <style>
    body {
        background: black !important;
        color: white !important;
    }
    #mynetwork {
        background: black !important;
    }
    .vis-configuration {
        background: black !important;
        color: white !important;
        border-color: #444 !important;
    }
    </style>
    """
    if file.head is not None:
        file.head.append(BeautifulSoup(dark_css, "html.parser"))

    path.write_text(str(file), encoding="utf-8")
    logger.info("Successfully wrote HTML graph", extra={"path": str(path)})


# ---------------------------------------------------------------------------- #
#                                      App                                     #
# ---------------------------------------------------------------------------- #
app = App()


def set_seed(seed: int) -> None:
    random.seed(seed)


# Flatten the namespace, i.e. will be "--logging-level"
# instead of "--common-params.logging-level"
@Parameter(name="*")
@dataclass
class CommonParams:
    """Args:
    logging_level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
        The logging level to use for the application.
    seed (int): The random seed to use for reproducibility.

    """

    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    seed: int = 1


class FailedToDownloadError(Exception):
    """Raised when the download of mutual friends data fails."""


@app.command
def download(
    path: Annotated[StdioPath | None, Parameter(name=["--path", "-p"])] = None,
    user_secret: str | None = None,
    *,
    progress: bool = True,
    common_params: CommonParams | None = None,
) -> None:
    """Download the mutual friends data from the Discord API and
    save it to JSON.

    Parameters
    ----------
    path:
        The path to save the mutual friends JSON data. Default is in the
        operating system's cache directory. Can use "-" for stdout.
    progress:
        Whether to show progress bars during data fetching.

    """
    if common_params is None:
        common_params = CommonParams()
    set_seed(common_params.seed)

    logger.setLevel(common_params.logging_level)

    if path is None:
        logger.debug("No path provided, using default cache path")
        path = StdioPath(DIRS.user_cache_path / "mutual_friends.json")

    secret = get_config_value(
        user_secret,
        env_var="DISCORD_USER_SECRET",
        value_name="User Secret",
        password_like=True,
    )
    if secret is None:
        return download(
            path,
            user_secret,
            progress=progress,
            common_params=common_params,
        )

    async def runner() -> None:
        async with aiohttp.ClientSession() as session:
            mutual_friends = await MutualFriends.fetch(
                session,
                progress=progress,
                user_secret=secret,
            )
            if mutual_friends is None:
                logger.error("Failed to fetch mutual friends data")
                raise FailedToDownloadError
            logger.info(
                "Fetched mutual friends for processing",
                extra={"count": len(mutual_friends)},
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                mutual_friends.model_dump_json(),
                encoding="utf-8",
            )
            logger.info(
                "Saved mutual friends data to JSON",
                extra={"path": str(path)},
            )

    try:
        asyncio.run(runner())
    except FailedToDownloadError:
        download(path, user_secret, progress=progress, common_params=common_params)
    return None


@app.command
def graph(
    input_: Annotated[
        StdioPath | None,
        Parameter(name=["--input", "-i"]),
    ] = None,
    output: Annotated[
        StdioPath | None,
        Parameter(name=["--output", "-o"]),
    ] = None,
    common_params: CommonParams | None = None,
) -> None:
    """Create the friends network graph HTML from the mutual friends JSON data.

    Parameters
    ----------
    input:
        The path to read the mutual friends JSON data. Default is in the
        operating system's cache directory. Can use "-" for stdin.
    output:
        The path to save the generated HTML graph. Default is in the operating
        system's cache directory. Can use "-" for stdout.

    """  # noqa: DOC102
    if common_params is None:
        common_params = CommonParams()
    set_seed(common_params.seed)
    logger.setLevel(common_params.logging_level)

    if input_ is None:
        input_ = StdioPath(DIRS.user_cache_path / "mutual_friends.json")
    if output is None:
        output = StdioPath(DIRS.user_cache_path / "graph.html")

    mutual_friends = MutualFriends.model_validate_json(
        input_.read_text(encoding="utf-8"),
    )

    network = create_network_graph(mutual_friends)
    write_html_graph(network, output)


@app.command
def clear(*, all_version: bool = False) -> None:
    """Clear the cached data files.

    Parameters
    ----------
    all_version:
        Clear everything, including files from previous versions.

    """
    if all_version:
        dir_path = DIRS.user_cache_path.parent
        logger.info("Clearing all cached data", extra={"path": str(dir_path)})
        shutil.rmtree(dir_path)
        return

    dir_path = DIRS.user_cache_path
    logger.info("Clearing cached data", extra={"path": str(dir_path)})
    shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


@app.default
def default_command(  # noqa: PLR0913
    data_path: Annotated[
        StdioPath | None,
        Parameter(name=["--data-path", "--input", "-i"]),
    ] = None,
    html_path: Annotated[
        StdioPath | None,
        Parameter(name=["--html-path", "--output", "-o"]),
    ] = None,
    *,
    user_secret: str | None = None,
    redownload: Annotated[bool, Parameter(name=["--redownload", "--update"])] = False,
    progress: bool = True,
    common_params: CommonParams | None = None,
) -> None:
    """Save your Discord mutual friends and visualize them as a graph.

    The default behavior is to download the mutual friends data
    (if not already present) and generate the HTML graph, then open it
    in the default web browser.

    Parameters
    ----------
    data_path:
        Where to read or save the mutual friends JSON data. Default is
        in the operating system's cache directory. Can use "-" for stdin.
        Does not supports writing to stdout. Use the `download` command
        instead for that.
    html_path:
        Where to save the generated HTML graph. Default is in the operating
        system's cache directory. Can use "-" for stdout.
    redownload:
        Redownload the data from the Discord API, even if it already exists.
    user_secret:
        Set the user secret to use for fetching data. Takes precedence over
        the shell variable.
    progress:
        Show progress bars during data fetching.

    """
    if common_params is None:
        common_params = CommonParams()
    set_seed(common_params.seed)
    logger.setLevel(common_params.logging_level)

    if data_path is None:
        data_path = StdioPath(DIRS.user_cache_path / "mutual_friends.json")

    if html_path is None:
        html_path = StdioPath(DIRS.user_cache_path / "friends_network_graph.html")

    if redownload or not data_path.exists():
        download(data_path, user_secret, progress=progress, common_params=common_params)
    graph(data_path, html_path, common_params=common_params)

    if not html_path.is_stdio:
        webbrowser.open_new_tab(f"file://{html_path.resolve()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
