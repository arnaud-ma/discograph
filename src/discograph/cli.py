import asyncio
import random
import shutil
import webbrowser
from dataclasses import dataclass
from typing import Annotated, Literal

import aiohttp
from cyclopts import App, Parameter
from cyclopts.types import StdioPath

from discograph.config import DIRS, get_config_value, setup_logging
from discograph.data import MutualFriends
from discograph.graph import create_graph, create_network, write_html_graph

app = App()


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)


# Flatten the namespace, i.e. will be "--loglevel"
# instead of "--common-params.loglevel"
@Parameter(name="*")
@dataclass
class CommonParams:
    """Args:
    loglevel (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
        The logging level to use for the application.
    seed (int): The random seed to use for reproducibility.

    """

    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
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
    user_secret:
        The discord user secret token to use for fetching data. Takes precedence over
        the shell `DISCORD_USER_SECRET` variable.
    progress:
        Whether to show progress bars during data fetching.
    common_params:
        Common parameters used across commands.


    """
    if common_params is None:
        common_params = CommonParams()
    set_seed(common_params.seed)

    logger = setup_logging(common_params.loglevel)
    logger.info("Download command started")

    if path is None:
        logger.debug("No path provided, using default cache path")
        path = StdioPath(DIRS.user_cache_path / "mutual_friends.json")
    logger.info("Mutual friends data will be saved to %s", path)

    logger.debug("Starting to acquire user secret token")

    secret = get_config_value(
        user_secret,
        env_var="DISCORD_USER_SECRET",
        value_name="User Secret",
        password_like=True,
    )
    if secret is None:
        logger.warning(
            "No or bad user secret provided, cannot download mutual friends data. "
            "Retrying...",
        )
        return download(
            path,
            user_secret,
            progress=progress,
            common_params=common_params,
        )

    logger.info("Token acquired, starting download of mutual friends data")

    async def runner() -> MutualFriends:
        async with aiohttp.ClientSession() as session:
            mutual_friends = await MutualFriends.fetch(
                session,
                progress=progress,
                user_secret=secret,
                logger=logger,
            )
            if mutual_friends is None:
                logger.error("Failed to fetch mutual friends data")
                raise FailedToDownloadError
            logger.info(
                "Successfully fetched %d mutual friends. Saving to JSON...",
                len(mutual_friends),
            )
        return mutual_friends

    try:
        mutual_friends = asyncio.run(runner())
    except FailedToDownloadError:
        logger.warning("Failed to download mutual friends data, retrying...")
        download(path, user_secret, progress=progress, common_params=common_params)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        mutual_friends.model_dump_json(),
        encoding="utf-8",
    )
    logger.info(
        "Saved %d mutual friends to %s",
        len(mutual_friends),
        path,
    )
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
    input_:
        The path to read the mutual friends JSON data. Default is in the
        operating system's cache directory. Can use "-" for stdin.
    output:
        The path to save the generated HTML graph. Default is in the operating
        system's cache directory. Can use "-" for stdout.
    common_params:
        Common parameters used across commands.

    """
    if common_params is None:
        common_params = CommonParams()

    logger = setup_logging(common_params.loglevel)
    logger.info("Graph command started")
    logger.debug("input_: %s, output: %s", input_, output)

    set_seed(common_params.seed)
    logger.debug("Random seed set to %d", common_params.seed)

    if input_ is None:
        input_ = StdioPath(DIRS.user_cache_path / "mutual_friends.json")
    logger.info("Reading mutual friends data from %s", input_)

    if output is None:
        output = StdioPath(DIRS.user_cache_path / "graph.html")
    logger.info("Output graph will be saved to %s", output)

    try:
        mutual_friends = MutualFriends.model_validate_json(
            input_.read_text(encoding="utf-8"),
        )
    except Exception:
        logger.exception(
            "Failed to read mutual friends data from %s",
            input_,
        )
        return

    graph_, _ = create_graph(mutual_friends)
    network = create_network(graph_, notebook=False)
    write_html_graph(network, output)
    logger.info("Saved HTML graph to %s", output)
    return


@app.command
def clear(
    common_params: CommonParams | None = None,
    *,
    all_version: bool = False,
) -> None:
    """Clear the cached data files.

    Parameters
    ----------
    all_version:
        Clear everything, including files from previous versions.

    """
    if common_params is None:
        common_params = CommonParams()
    logger = setup_logging(common_params.loglevel)

    if all_version:
        dir_path = DIRS.user_cache_path.parent
        logger.info("Clearing all cached data from %s", dir_path)
        shutil.rmtree(dir_path)
        logger.info("Cleaned all cached data from %s", dir_path)
        return

    dir_path = DIRS.user_cache_path
    logger.info("Clearing cached data from %s", dir_path)
    shutil.rmtree(dir_path)
    logger.info("Cleaned cached data from %s", dir_path)
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
    common_params:
        Common parameters used across commands.

    """
    if common_params is None:
        common_params = CommonParams()
    logger = setup_logging(common_params.loglevel)
    set_seed(common_params.seed)
    logger.debug("Random seed set to %d", common_params.seed)

    if data_path is None:
        data_path = StdioPath(DIRS.user_cache_path / "mutual_friends.json")

    if html_path is None:
        html_path = StdioPath(DIRS.user_cache_path / "friends_network_graph.html")

    if redownload or not data_path.exists():
        download(data_path, user_secret, progress=progress, common_params=common_params)
    graph(data_path, html_path, common_params=common_params)

    if not html_path.is_stdio:
        webbrowser.open_new_tab(f"file://{html_path.resolve()}")
        logger.info("Opened graph HTML in web browser from %s", html_path)


def main() -> None:
    app()
