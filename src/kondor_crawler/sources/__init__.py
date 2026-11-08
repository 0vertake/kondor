"""Import sources so they self-register on `SOURCES`."""

from kondor_crawler.sources import civitai as civitai
from kondor_crawler.sources import github as github
from kondor_crawler.sources import huggingface as huggingface

__all__ = ["civitai", "github", "huggingface"]
