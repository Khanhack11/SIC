"""Inventory storage and tracked source without importing code or following directory links."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inventory(root, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folders = defaultdict(lambda: {"files": 0, "bytes": 0})
    errors, links = [], []
    files = 0
    source_candidates = []
    with (output / "files.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "top_level"])
        for directory, directories, names in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
            parent = Path(directory)
            directories.sort()
            for name in directories[:]:
                path = parent / name
                if path.is_symlink() or path.is_junction():
                    links.append(path.relative_to(root).as_posix())
                    directories.remove(name)
                elif path.resolve() == output:
                    directories.remove(name)
            for name in sorted(names):
                path = parent / name
                try:
                    if path.is_symlink():
                        links.append(path.relative_to(root).as_posix())
                        continue
                    relative = path.relative_to(root)
                    size = path.stat().st_size
                    top = relative.parts[0] if len(relative.parts) > 1 else "(root files)"
                    writer.writerow([relative.as_posix(), size, top])
                    folders[top]["files"] += 1
                    folders[top]["bytes"] += size
                    files += 1
                    if not {".git", ".venv", ".venv-gpu", "node_modules", "__pycache__", "artifacts", "raw", "processed", "results"}.intersection(relative.parts):
                        source_candidates.append(relative.as_posix())
                except OSError as error:
                    errors.append(f"{path}: {error}")
    try:
        if (root / ".git").exists():
            git = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
            tracked = git.stdout.decode("utf-8").split("\0") if git.returncode == 0 else source_candidates
            basis = "Git index" if git.returncode == 0 else "Filesystem source candidates (no Git repository)"
        else:
            tracked, basis = source_candidates, "Filesystem source candidates (no Git repository)"
    except OSError:
        tracked, basis = source_candidates, "Filesystem source candidates (Git unavailable)"
    sources, duplicates = [], defaultdict(list)
    extensions = Counter()
    for name in filter(None, tracked):
        path = root / name
        if not path.is_file():
            continue
        extensions[path.suffix.lower() or "(no extension)"] += 1
        if path.suffix.lower() not in {".py", ".js", ".cjs", ".html", ".md", ".cmd", ".txt", ".ipynb"}:
            continue
        content = path.read_bytes()
        duplicates[hashlib.sha256(content).hexdigest()].append(name)
        text = content.decode("utf-8-sig")
        item = {"path": name, "lines": len(text.splitlines()), "bytes": len(content)}
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=name)
                item.update(description=ast.get_docstring(tree),
                            definitions=[node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                            imports=sorted({node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
                                           | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}))
            except SyntaxError as error:
                item["syntax_error"] = str(error)
        elif path.suffix == ".ipynb":
            notebook = json.loads(text)
            item["cells"] = dict(Counter(cell["cell_type"] for cell in notebook["cells"]))
            item["code_preview"] = ["".join(cell["source"])[:600] for cell in notebook["cells"] if cell["cell_type"] == "code"]
        sources.append(item)
    summary = {"root": str(root), "files": files, "bytes": sum(row["bytes"] for row in folders.values()),
               "folders": dict(sorted(folders.items(), key=lambda entry: -entry[1]["bytes"])),
               "tracked_files": sum(extensions.values()), "tracked_extensions": dict(extensions),
               "source_inventory_basis": basis,
               "read_errors": errors, "directory_links_not_followed": links,
               "source_syntax_errors": [row for row in sources if "syntax_error" in row],
               "identical_tracked_text_groups": [group for group in duplicates.values() if len(group) > 1],
               "scope": "Logical file sizes; generated report directory excluded; no imports or binary model execution."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "source_catalog.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": files, "GB": round(summary["bytes"] / 1e9, 3), "folders": summary["folders"],
                      "read_errors": len(errors), "python_syntax_errors": len(summary["source_syntax_errors"])}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    inventory(ROOT, arguments.output)
