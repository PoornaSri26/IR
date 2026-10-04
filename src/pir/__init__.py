"""pir — a pure-IR positional index system: phrase, proximity, explain, eval."""

__version__ = "1.0.0"

from .index import PositionalIndex, build_index  # noqa: F401
from .query_parser import Query, QueryParseError, parse  # noqa: F401
