"""Generate the code reference pages."""

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"
src_ragas = root / "src" / "ragas"

logger.info("Generating code reference pages for %s with root %s", src, root)
for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src_ragas).with_suffix(".md")
    full_doc_path = Path("references", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        # doc_path = doc_path.with_name("index.md")
        # full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    elif parts[-1][0] == "_":  # Skip private modules
        continue

    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("references/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
