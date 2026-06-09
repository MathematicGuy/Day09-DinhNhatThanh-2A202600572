from __future__ import annotations


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    chunks: list[dict] = []
    section_h2: str | None = None
    section_h3: str | None = None
    content_lines: list[str] = []

    def flush_chunk() -> None:
        if section_h2 is None or section_h3 is None:
            return
        content = "\n".join(line for line in content_lines).strip()
        if not content:
            return
        chunks.append(
            {
                "section_h2": section_h2,
                "section_h3": section_h3,
                "citation": f"{section_h2} > {section_h3}",
                "rendered_text": f"{section_h2}\n{section_h3}\n{content}",
            }
        )

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            flush_chunk()
            section_h2 = line[3:].strip()
            section_h3 = None
            content_lines = []
            continue
        if line.startswith("### "):
            flush_chunk()
            section_h3 = line[4:].strip()
            content_lines = []
            continue
        if section_h3 is not None:
            content_lines.append(line)

    flush_chunk()
    return chunks
