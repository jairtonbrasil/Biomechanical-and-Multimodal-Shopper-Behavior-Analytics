import ipaddress
import re
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from pydantic import SecretStr

if TYPE_CHECKING:
    from app.core.config import Settings


class StreamPolicy:
    def __init__(self, settings: "Settings") -> None:
        self.networks = tuple(
            ipaddress.ip_network(network, strict=True) for network in settings.rtsp_allowed_networks
        )
        self.ports = frozenset(settings.rtsp_allowed_ports)

    def validate(self, url: SecretStr | str) -> None:
        value = url.get_secret_value() if isinstance(url, SecretStr) else url
        try:
            self._validate(value)
        except (ValueError, UnicodeError):
            raise ValueError("Stream destination is not allowed") from None

    def _validate(self, value: str) -> None:
        if (
            len(value) > 4096
            or not value.isascii()
            or any(ord(character) <= 32 or ord(character) == 127 for character in value)
            or "\\" in value
            or re.search(r"%(?![0-9A-Fa-f]{2})", value)
        ):
            raise ValueError
        parsed = urlsplit(value)
        if parsed.scheme not in {"rtsp", "rtsps"} or "#" in value or "?" in value:
            raise ValueError
        if not parsed.hostname or "%" in parsed.hostname or parsed.netloc.count("@") > 1:
            raise ValueError
        if parsed.username is not None and (not parsed.username or not parsed.password):
            raise ValueError
        decoded = unquote(value, errors="strict")
        if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
            raise ValueError
        address = ipaddress.ip_address(parsed.hostname)
        if (
            address.is_multicast
            or address.is_unspecified
            or address.is_link_local
            or getattr(address, "ipv4_mapped", None) is not None
        ):
            raise ValueError
        default_port = 322 if parsed.scheme == "rtsps" else 554
        port = default_port if parsed.port is None else parsed.port
        if port not in self.ports:
            raise ValueError
        if not any(address in network for network in self.networks):
            raise ValueError
