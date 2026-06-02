# scripts/show_project_tree.py

from __future__ import annotations

from pathlib import Path


IGNORE_DIRS = {
    ".git",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "checkpoints",
    "logs",
    "data/raw",
}

IGNORE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".wav",
    ".flac",
    ".mp3",
}

IGNORE_FILES = {
    ".DS_Store",
}


def should_ignore(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()

    if path.name in IGNORE_FILES:
        return True

    if path.suffix in IGNORE_SUFFIXES:
        return True

    parts = set(path.relative_to(root).parts)

    for ignored in IGNORE_DIRS:
        ignored_parts = ignored.split("/")
        rel_parts = path.relative_to(root).parts

        if len(ignored_parts) == 1:
            if ignored in parts:
                return True
        else:
            if tuple(ignored_parts) == rel_parts[: len(ignored_parts)]:
                return True

    return False


def print_tree(root: Path, prefix: str = "", max_depth: int = 5, current_depth: int = 0) -> None:
    if current_depth > max_depth:
        return

    items = sorted(
        [p for p in root.iterdir() if not should_ignore(p, ROOT)],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )

    for idx, path in enumerate(items):
        connector = "└── " if idx == len(items) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if idx == len(items) - 1 else "│   "
            print_tree(
                path,
                prefix + extension,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )


if __name__ == "__main__":
    ROOT = Path.cwd().resolve()

    print(ROOT.name + "/")
    print_tree(ROOT, max_depth=6)