from __future__ import annotations

import functools
import getpass
import logging
import os

from platformdirs import PlatformDirs

APP_NAME = "discograph"
__version__ = "0.2.3"

DIRS = PlatformDirs(APP_NAME, version=__version__)
DIRS.user_state_path.mkdir(parents=True, exist_ok=True)
DIRS.user_cache_path.mkdir(parents=True, exist_ok=True)

DISCORD_API_LINK = "https://discord.com/api/v10"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(DIRS.user_state_path / "discord_graph.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("discograph")
    logger.info("Logging initialized at level: %s", level)
    logger.info("Application version: %s", __version__)
    logger.info("Log file located at: %s", DIRS.user_state_path / "discord_graph.log")

    return logging.getLogger(APP_NAME)


def get_config_value(
    cli_value: str | None,
    env_var: str,
    value_name: str,
    *,
    password_like: bool = False,
) -> str | None:
    """Retrieve a configuration value from CLI argument, environment variable,
    or user input.

    It tries to get the value in the following order:
    1. Command-line interface argument (`cli_value`).
    2. Environment variable (`env_var`).
    3. User input prompt with message "Enter value for {value_name}:".

    Parameters
    ----------
    cli_value : str | None
        The value provided via command-line interface.
    env_var : str
        The name of the environment variable to check.
    value_name : str
        A human-readable name for the value, used in prompts and logging.
    password_like : bool, optional
        If True, the user input will be hidden (suitable for passwords).
        Defaults to False.

    Returns
    -------
    str | None
        The retrieved configuration value, or None if not provided.

    """
    if cli_value is not None:
        return cli_value

    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value

    input_func = (
        functools.partial(getpass.getpass, echo_char="*") if password_like else input
    )
    input_value = input_func(f"Enter value for {value_name}: ")
    if input_value:
        return input_value

    return None
