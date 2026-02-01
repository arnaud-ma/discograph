import asyncio
import warnings
from collections.abc import Sized
from enum import IntEnum
from typing import Literal, NamedTuple, Self, TYPE_CHECKING

import aiohttp
from pydantic import BaseModel, TypeAdapter

from .config import DISCORD_API_LINK

if TYPE_CHECKING:
    import logging

type AvatarSize = Literal[16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


class RelationShipType(IntEnum):
    """Type of relationship taken from [Discord API docs](https://discord-api-types.dev/api/discord-api-types-v10/enum/RelationshipType)."""

    none = 0
    friend = 1
    blocked = 2
    implicit = 3
    pending_incoming = 4
    pending_outgoing = 5


class UserResponse(BaseModel):
    """Response from Discord API for user relationship.

    There is much more fields returned by the api, only a few are modeled here.
    """

    id: str
    nickname: str | None
    user: User
    type: RelationShipType


UsersResponse = TypeAdapter(list[UserResponse])


class FetchResponseError(Exception):
    """Raised when fetching mutual friends fails."""


class User(BaseModel):
    """Discord user model.

    Is a subset of fields returned by the Discord API when fetching user relationships.
    """

    id: str
    avatar: str | None
    discriminator: str
    username: str
    global_name: str | None

    def avatar_url(self, size: AvatarSize = 128) -> str | None:
        """Get the URL of the user's avatar.

        The url can then be used to download the avatar image.

        Parameters
        ----------
        size : AvatarSize, optional
            The size of the avatar image. Must be one of the following values:
            16, 32, 64, 128, 256, 512, 1024, 2048, 4096. Default is 128.

        Returns
        -------
        str | None
            The URL of the user's avatar, or None if the user has no avatar.

        """
        if self.avatar:
            return f"https://cdn.discordapp.com/avatars/{self.id}/{self.avatar}.png?size={size}"
        return None

    async def download_avatar_bytes(
        self,
        session: aiohttp.ClientSession,
        size: AvatarSize = 128,
    ) -> bytes | None:
        """Download the user's avatar image (profile picture) as bytes (png format).

        Parameters
        ----------
        session : aiohttp.ClientSession
            The aiohttp session to use for the request.
        size : AvatarSize, optional
            The size of the avatar image. Must be one of the following values:
            16, 32, 64, 128, 256, 512, 1024, 2048, 4096. Default is 128.

        Returns
        -------
        bytes | None
            The avatar image as bytes, or None if the user has no avatar
            or the download fails.

        """
        url = self.avatar_url(size=size)
        if not url:
            return None
        async with session.get(url) as response:
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError:
                return None
            return await response.read()

    async def fetch_mutual_friends(
        self,
        session: aiohttp.ClientSession,
        user_secret: str,
    ) -> set[User]:
        """Fetch mutual friends of this user.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The aiohttp session to use for the request.
        user_secret : str
            The user's authorization token.

        Returns
        -------
        set[User]
            A set of User objects representing the mutual friends.

        Raises
        ------
        FetchResponseError
            If the response from the API is invalid.

        Warnings
        --------
        It is using the personal user's token to fetch the data. Using this
        method may violate Discord's Terms of Service. Use at your own risk.
        Also, excessive requests in a very short time may lead to rate limiting
        or temporary bans. So always use this method sequentially with some delay
        between requests to be safe.

        """
        link = f"{DISCORD_API_LINK}/users/{self.id}/relationships"
        headers = {"authorization": user_secret}
        while True:
            async with session.get(link, headers=headers) as response:
                try:
                    friends_data = await response.json()
                except aiohttp.ContentTypeError as e:
                    msg = f"Failed to fetch mutual friends for user {self.id}"
                    raise FetchResponseError(msg) from e
                if isinstance(friends_data, dict) and (
                    wait := friends_data.get("retry_after")
                ):
                    await asyncio.sleep(wait * 1.1)
                    continue
                break
        return {User.model_validate(fd) for fd in friends_data}

    def __eq__(self, other) -> bool:
        return isinstance(other, User) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def display_name(self) -> str:
        """The display name of the user (can be different from username)."""
        return self.global_name or self.username


async def get_friends_dict(
    session: aiohttp.ClientSession,
    user_secret: str,
) -> dict[str, User] | None:
    """Fetch the friends of the user.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp session to use for the request.
    user_secret : str
        The user's authorization token.

    Returns
    -------
    dict[str, User] | None
        A dictionary mapping user IDs to User objects representing the friends,
        or None if the request fails.


    Warnings
    --------
    It uses the personal user's token to fetch the data. Using this
    method may violate Discord's Terms of Service. Use at your own risk. Also,
    excessive requests in a very short time may lead to rate limiting
    or temporary bans. So always use this method sequentially with some delay
    between requests to be safe.

    """
    link = f"{DISCORD_API_LINK}/users/@me/relationships"
    headers = {"authorization": user_secret}
    async with session.get(link, headers=headers) as response:
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError:
            return None
        friends_data = await response.json()
    user_with_relationship = UsersResponse.validate_python(friends_data)
    friends = [
        fr for fr in user_with_relationship if fr.type == RelationShipType.friend
    ]
    return {friend.user.id: friend.user for friend in friends}


class FriendMutuals(NamedTuple):
    user: User
    mutual_ids: set[str]


class MutualFriends(BaseModel, Sized):
    friends: dict[str, FriendMutuals]

    @classmethod
    async def from_friends(
        cls,
        session: aiohttp.ClientSession,
        user_secret: str,
        friends: dict[str, User],
        *,
        progress: bool = True,
        logger: logging.Logger | None = None,
    ) -> Self:
        """Create a MutualFriends instance by fetching mutual friends for each friend.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The aiohttp session to use for the requests.
        user_secret : str
            The user's authorization token.
        friends : dict[str, User]
            A dictionary mapping user IDs to User objects representing the friends.
        progress : bool, default: True
            Whether to show progress bars during data fetching.
        logger : logging.Logger | None, optional
            If provided, log debug information about the fetching process.

        Returns
        -------
        Self
            An instance of MutualFriends containing the fetched data.

        Warnings
        --------
        It uses the personal user's token to fetch the data. Using this
        method may violate Discord's Terms of Service. Use at your own risk. Also,
        excessive requests in a very short time may lead to rate limiting
        or temporary bans. So always use this method sequentially with some delay
        between requests to be safe.

        """
        from tqdm.rich import tqdm  # noqa: PLC0415
        from tqdm.std import TqdmExperimentalWarning  # noqa: PLC0415

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        friends_data: dict[str, FriendMutuals] = {}
        mutuals_by_friend = {}
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
        for friend in progress_iter:
            try:
                mutual_friends = await friend.fetch_mutual_friends(session, user_secret)
            except FetchResponseError:
                mutual_friends = set()
                # TODO: proper warning here
            friends_data[friend.id] = FriendMutuals(
                user=friend,
                mutual_ids={mf.id for mf in mutual_friends},
            )
            mutuals_by_friend[friend.id] = friends_data[friend.id].mutual_ids
            if logger:
                logger.debug(
                    "Fetched %d mutual friends for user %s (%s)",
                    len(mutual_friends),
                    friend.id,
                    friend.display_name,
                )
        return cls(friends=friends_data)

    @classmethod
    async def fetch(
        cls,
        session: aiohttp.ClientSession,
        user_secret: str,
        *,
        progress: bool = True,
        logger: logging.Logger | None = None,
    ) -> Self | None:
        """Fetch mutual friends data for the user.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The aiohttp session to use for the requests.
        user_secret : str
            The user's authorization token.
        progress : bool, optional
            Whether to show progress bars during data fetching. Default is True.
        logger : logging.Logger | None, optional
            If provided, log debug information about the fetching process.

        Returns
        -------
        Self | None
            An instance of MutualFriends containing the fetched data,
            or None if fetching friends fails.

        Warnings
        --------
        It uses the personal user's token to fetch the data. Using this
        method may violate Discord's Terms of Service. Use at your own risk. Also,
        excessive requests in a very short time may lead to rate limiting
        or temporary bans. So always use this method sequentially with some delay
        between requests to be safe.

        """
        friends = await get_friends_dict(session, user_secret)
        if friends is None:
            return None
        return await cls.from_friends(
            session,
            user_secret,
            friends,
            progress=progress,
            logger=logger,
        )

    def __len__(self) -> int:
        return len(self.friends)
