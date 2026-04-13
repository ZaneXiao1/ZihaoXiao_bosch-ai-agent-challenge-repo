"""
Document loading and chunking for ECU manuals.

Design Decision: Section-based chunking preserves the integrity of
specification tables and diagnostic blocks. Each chunk is prefixed
with its series and model context so embeddings carry that metadata
without relying on LangChain metadata fields (which are invisible to
the embedding model).
"""
import re
import logging
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Split ECU-700 on bold numbered sections: **1. Title**
_ECU700_SECTION_RE = re.compile(r"(?m)^(?=\*\*\d+\.)")

# Split ECU-800 on markdown level-2 headings: ## Title
_ECU800_SECTION_RE = re.compile(r"(?m)^(?=## )")

# Maps filename substrings to (series, models_covered)
_FILE_METADATA: dict[str, tuple[str, list[str]]] = {
    "ECU-700": ("ECU-700", ["ECU-750"]),
    "ECU-800_Series_Base": ("ECU-800", ["ECU-850"]),
    "ECU-800_Series_Plus": ("ECU-800", ["ECU-850b"]),
}


def _get_file_metadata(filename: str) -> tuple[str, list[str]]:
    """Determine series and models_covered from the markdown filename.

    Args:
        filename: Base filename of the markdown document.

    Returns:
        Tuple of (series_str, models_list).
    """
    for pattern, meta in _FILE_METADATA.items():
        if pattern in filename:
            return meta
    logger.warning("Unrecognised filename pattern: %s — defaulting to ECU-800", filename)
    return ("ECU-800", ["ECU-850", "ECU-850b"])


def _extract_section_title(text: str) -> str:
    """Extract a clean section title from the first line of a chunk.

    Handles both **N. Title** (ECU-700) and ## Title (ECU-800) formats.

    Args:
        text: Raw section text starting with the heading line.

    Returns:
        Clean title string without numbering or markdown syntax.
    """
    first_line = text.split("\n")[0].strip()
    # **N. Title** or **N. Title: subtitle** → Title or Title: subtitle
    match = re.match(r"^\*\*\d+\.\s+(.+?)\*\*\s*$", first_line)
    if match:
        return match.group(1).strip()
    # ## Title → Title
    match = re.match(r"^#+\s+(.+)$", first_line)
    if match:
        return match.group(1).strip()
    return first_line.strip("*#").strip()


def _split_ecu700(content: str) -> list[str]:
    """Split an ECU-700 document on bold-numbered section markers.

    The preamble (document header containing Document ID, etc.) is
    included as the first chunk if it has meaningful content.

    Args:
        content: Full markdown file content.

    Returns:
        List of section text strings, including the preamble.
    """
    parts = _ECU700_SECTION_RE.split(content)
    result = []
    for p in parts:
        stripped = p.strip()
        if not stripped:
            continue
        # Numbered sections like **1. Introduction**
        if re.match(r"^\*\*\d+\.", stripped):
            result.append(stripped)
        # Preamble: keep if it has substantive content (e.g. Document ID)
        elif len(stripped) > 30:
            result.append(stripped)
    return result


def _split_ecu800(content: str) -> list[str]:
    """Split an ECU-800 document on level-2 (##) headings.

    Args:
        content: Full markdown file content.

    Returns:
        List of section text strings (document title excluded).
    """
    parts = _ECU800_SECTION_RE.split(content)
    return [p.strip() for p in parts if p.strip().startswith("## ")]


def load_documents(data_dir: Optional[str] = None) -> list[Document]:
    """Load and chunk ECU markdown documents from *data_dir*.

    Each chunk receives:
    - A context prefix injected into ``page_content`` so the embedding is
      self-contained (e.g. ``[ECU-700 Series | ECU-750] Diagnostics\\n…``).
    - Metadata fields: ``source``, ``series``, ``models_covered``, ``section``.

    Args:
        data_dir: Directory containing the three ECU markdown files.
                  Defaults to the package's own ``data/`` directory.

    Returns:
        Flat list of :class:`~langchain_core.documents.Document` objects.

    Raises:
        FileNotFoundError: If no markdown files are found in *data_dir*.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data")

    data_path = Path(data_dir)
    md_files = sorted(data_path.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {data_dir}")

    documents: list[Document] = []

    for md_file in md_files:
        filename = md_file.name
        content = md_file.read_text(encoding="utf-8")
        series, models = _get_file_metadata(filename)

        raw_sections = (
            _split_ecu700(content) if "ECU-700" in filename else _split_ecu800(content)
        )

        logger.debug("File %s → %d sections", filename, len(raw_sections))

        for section_text in raw_sections:
            title = _extract_section_title(section_text)
            models_str = ", ".join(models)
            prefix = f"[{series} Series | {models_str}] {title}\n"

            doc = Document(
                page_content=prefix + section_text,
                metadata={
                    "source": filename,
                    "series": series,
                    "models_covered": models,
                    "section": title,
                },
            )
            documents.append(doc)

    logger.info("Loaded %d document chunks from %s", len(documents), data_dir)
    return documents
