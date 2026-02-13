import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
) -> None:
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
