"""
generate_graphs.py — Repository graph generation script for DroneVision.
Parses the codebase structure and generates:
- graph/code_graph.json (classes, methods, functions and their connections)
- graph/dependency_graph.json (file-to-file and module imports)
- graph/api_graph.json (REST API endpoint representations)
- graph/architecture_graph.json (high-level structural pipelines)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

def extract_code_elements(project_root: Path) -> dict:
    code_nodes = []
    code_edges = []
    dep_nodes = set()
    dep_edges = []

    # Traverse dronevision source files
    src_dir = project_root / "dronevision"
    py_files = list(src_dir.rglob("*.py"))

    for py_file in py_files:
        rel_path = py_file.relative_to(project_root).as_posix()
        file_node_id = f"file:{rel_path}"
        code_nodes.append({
            "id": file_node_id,
            "type": "file",
            "name": py_file.name,
            "path": rel_path
        })
        dep_nodes.add(rel_path)

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
        except Exception as e:
            print(f"Skipping {rel_path} due to parse error: {e}")
            continue

        current_class = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"class:{rel_path}:{node.name}"
                code_nodes.append({
                    "id": class_id,
                    "type": "class",
                    "name": node.name,
                    "file": rel_path,
                    "bases": [ast.unparse(b) for b in node.bases]
                })
                # Edge from file to class
                code_edges.append({
                    "source": file_node_id,
                    "target": class_id,
                    "relation": "defines"
                })
                current_class = node.name

            elif isinstance(node, ast.FunctionDef):
                func_name = f"{current_class}.{node.name}" if current_class else node.name
                func_id = f"func:{rel_path}:{func_name}"
                code_nodes.append({
                    "id": func_id,
                    "type": "function",
                    "name": func_name,
                    "file": rel_path,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node) or ""
                })
                parent_id = f"class:{rel_path}:{current_class}" if current_class else file_node_id
                code_edges.append({
                    "source": parent_id,
                    "target": func_id,
                    "relation": "defines"
                })

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Handle imports for dependency_graph.json
                modules = []
                if isinstance(node, ast.Import):
                    for name in node.names:
                        modules.append(name.name)
                else:
                    if node.module:
                        modules.append(node.module)
                
                for mod in modules:
                    if mod.startswith("dronevision"):
                        # Convert module import to file path
                        mod_parts = mod.split(".")
                        target_rel_path = "/".join(mod_parts) + ".py"
                        if (project_root / target_rel_path).exists():
                            dep_edges.append({
                                "source": rel_path,
                                "target": target_rel_path,
                                "type": "internal_import"
                            })
                        else:
                            # Might be a package directory with __init__.py
                            target_init_path = "/".join(mod_parts) + "/__init__.py"
                            if (project_root / target_init_path).exists():
                                dep_edges.append({
                                    "source": rel_path,
                                    "target": target_init_path,
                                    "type": "internal_import"
                                })
                    else:
                        dep_edges.append({
                            "source": rel_path,
                            "target": mod,
                            "type": "external_import"
                        })
                        dep_nodes.add(mod)

    return {
        "code_graph": {"nodes": code_nodes, "edges": code_edges},
        "dependency_graph": {
            "nodes": [{"id": node, "type": "module" if "dronevision" in node or node.endswith(".py") else "external"} for node in dep_nodes],
            "edges": dep_edges
        }
    }

def generate_api_graph() -> dict:
    # Modeled REST API endpoints from TRD.md
    nodes = [
        {"id": "api:image_prediction", "path": "/predict/image", "method": "POST", "summary": "Runs single image drone detection"},
        {"id": "api:video_prediction", "path": "/predict/video", "method": "POST", "summary": "Runs video stream drone tracking/counting"},
        {"id": "handler:image", "type": "handler", "name": "predict_image_handler"},
        {"id": "handler:video", "type": "handler", "name": "predict_video_handler"}
    ]
    edges = [
        {"source": "api:image_prediction", "target": "handler:image", "relation": "routes_to"},
        {"source": "api:video_prediction", "target": "handler:video", "relation": "routes_to"},
        {"source": "handler:image", "target": "class:dronevision/inference/predictor.py:DronePredictor", "relation": "calls"},
        {"source": "handler:video", "target": "class:dronevision/inference/predictor.py:DronePredictor", "relation": "calls"}
    ]
    return {"nodes": nodes, "edges": edges}

def generate_architecture_graph() -> dict:
    nodes = [
        {"id": "arch:input", "name": "Image / Video Input", "category": "Input"},
        {"id": "arch:backbone", "name": "DroneBackbone (Feature Extractor)", "category": "Model"},
        {"id": "arch:neck", "name": "DroneNeck (FPN Fusion)", "category": "Model"},
        {"id": "arch:head", "name": "DroneHead (Multi-scale Predictor)", "category": "Model"},
        {"id": "arch:loss", "name": "DetectionLoss (Target Assignment + CIoU)", "category": "Loss"},
        {"id": "arch:nms", "name": "NMS (Post-processing)", "category": "Post-processing"},
        {"id": "arch:output", "name": "Count & Bounding Boxes", "category": "Output"}
    ]
    edges = [
        {"source": "arch:input", "target": "arch:backbone", "label": "Feed image"},
        {"source": "arch:backbone", "target": "arch:neck", "label": "Pass P3, P4, P5 scales"},
        {"source": "arch:neck", "target": "arch:head", "label": "Pass fused scales N3, N4, N5"},
        {"source": "arch:head", "target": "arch:loss", "label": "Calculate box/obj/cls losses (Training)"},
        {"source": "arch:head", "target": "arch:nms", "label": "Filter low-confidence predictions (Inference)"},
        {"source": "arch:nms", "target": "arch:output", "label": "Count valid boxes"}
    ]
    return {"nodes": nodes, "edges": edges}

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    graph_dir = project_root / "graph"
    graph_dir.mkdir(exist_ok=True)

    print("Analyzing codebase and generating dependency graphs...")
    extracted = extract_code_elements(project_root)

    # Write code_graph.json
    code_graph_path = graph_dir / "code_graph.json"
    with open(code_graph_path, "w", encoding="utf-8") as f:
        json.dump(extracted["code_graph"], f, indent=2)
    print(f"Generated {code_graph_path}")

    # Write dependency_graph.json
    dep_graph_path = graph_dir / "dependency_graph.json"
    with open(dep_graph_path, "w", encoding="utf-8") as f:
        json.dump(extracted["dependency_graph"], f, indent=2)
    print(f"Generated {dep_graph_path}")

    # Check if FastAPI endpoints exist in any source code
    fastapi_exists = False
    src_dir = project_root / "dronevision"
    scripts_dir = project_root / "scripts"
    all_py_files = list(src_dir.rglob("*.py")) + list(scripts_dir.glob("*.py"))
    for py_file in all_py_files:
        if py_file.name == "generate_graphs.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if "fastapi" in content.lower():
                fastapi_exists = True
                break
        except Exception:
            continue

    api_graph_path = graph_dir / "api_graph.json"
    if fastapi_exists:
        # Write api_graph.json
        with open(api_graph_path, "w", encoding="utf-8") as f:
            json.dump(generate_api_graph(), f, indent=2)
        print(f"Generated {api_graph_path}")
    else:
        # Delete api_graph.json if it exists
        if api_graph_path.exists():
            api_graph_path.unlink()
            print("Removed obsolete api_graph.json since FastAPI endpoints are absent.")

    # Write architecture_graph.json
    arch_graph_path = graph_dir / "architecture_graph.json"
    with open(arch_graph_path, "w", encoding="utf-8") as f:
        json.dump(generate_architecture_graph(), f, indent=2)
    print(f"Generated {arch_graph_path}")

if __name__ == "__main__":
    main()
