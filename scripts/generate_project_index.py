"""
generate_project_index.py — Generates project_index.md.
Includes file tree, exact token counts (using tiktoken), file sizes, and summaries.
"""

from __future__ import annotations

import os
from pathlib import Path
import tiktoken

def get_ignore_list(project_root: Path) -> list[str]:
    # Standard directories to ignore
    ignore = {".git", ".pytest_cache", "__pycache__", "venv_cuda", "mlruns", "graphify-out", "runs", "outputs", "dataset", "datasets"}
    
    # Read .gitignore if available
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore.add(line.replace("/", "").replace("*", ""))
                
    return list(ignore)

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback to word-based estimate if tiktoken fails
        return len(text.split())

def generate_file_tree(dir_path: Path, prefix: str = "", ignore: list[str] | None = None) -> list[str]:
    if ignore is None:
        ignore = []
    lines = []
    # Sort files and directories for consistent output
    items = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    
    for i, item in enumerate(items):
        if item.name in ignore or any(item.name.startswith(p) for p in [".", "venv"]):
            continue
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir():
            sub_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(generate_file_tree(item, sub_prefix, ignore))
            
    return lines

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ignore = get_ignore_list(project_root)
    ignore.append("repomix-output.xml")
    ignore.append("project_index.md")
    
    print("Generating codebase index and counting tokens...")
    
    # Build tree
    tree_lines = [project_root.name + "/"]
    tree_lines.extend(generate_file_tree(project_root, "", ignore))
    tree_str = "\n".join(tree_lines)
    
    # Get stats for Python and config files
    file_stats = []
    total_tokens = 0
    total_lines = 0
    
    extensions = {".py", ".yaml", ".yml", ".toml", ".txt", ".md"}
    
    for root, dirs, files in os.walk(project_root):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in ignore and not d.startswith(".") and "venv" not in d]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix not in extensions:
                continue
            if file == "repomix-output.xml" or file == "project_index.md":
                continue
                
            rel_path = file_path.relative_to(project_root).as_posix()
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()
                line_count = len(lines)
                token_count = count_tokens(content)
                size_kb = file_path.stat().st_size / 1024
                
                # Simple file summary based on header or docstring
                summary = "Source code module"
                if file_path.suffix == ".py":
                    for line in lines[:5]:
                        if '"""' in line or "'''" in line:
                            summary = line.replace('"""', '').replace("'''", '').strip()
                            if not summary:
                                continue
                            break
                elif file_path.suffix == ".md":
                    for line in lines:
                        if line.startswith("#"):
                            summary = line.replace("#", "").strip()
                            break
                            
                file_stats.append({
                    "path": rel_path,
                    "tokens": token_count,
                    "lines": line_count,
                    "size_kb": size_kb,
                    "summary": summary
                })
                total_tokens += token_count
                total_lines += line_count
            except Exception as e:
                print(f"Skipping stats for {rel_path}: {e}")

    # Write report
    report_path = project_root / "project_index.md"
    
    markdown_out = []
    markdown_out.append("# Project Index")
    markdown_out.append(f"\nThis document maps the files, structure, and token metrics of the DroneVision repository.")
    
    markdown_out.append("\n## Repository Metrics")
    markdown_out.append(f"- **Total Indexed Files**: {len(file_stats)}")
    markdown_out.append(f"- **Total Code Lines**: {total_lines}")
    markdown_out.append(f"- **Estimated Token Count (cl100k_base)**: {total_tokens:,} tokens")
    
    markdown_out.append("\n## File Tree Layout")
    markdown_out.append("```\n" + tree_str + "\n```")
    
    markdown_out.append("\n## Detailed File Inventory")
    markdown_out.append("| File Path | Tokens | Lines | Size (KB) | Purpose / Summary |")
    markdown_out.append("|---|---|---|---|---|")
    for stat in sorted(file_stats, key=lambda x: x["path"]):
        markdown_out.append(
            f"| [{stat['path']}](file:///{project_root.as_posix()}/{stat['path']}) | "
            f"{stat['tokens']} | {stat['lines']} | {stat['size_kb']:.2f} | {stat['summary']} |"
        )
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_out))
        
    print(f"Project index successfully written to {report_path}")

if __name__ == "__main__":
    main()
