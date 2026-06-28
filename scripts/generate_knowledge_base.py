"""
generate_knowledge_base.py — Creates a Code Knowledge Base in docs/knowledge/.
Generates markdown files summarising modules, classes, functions, inputs, outputs, and dependencies.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

def parse_python_file(file_path: Path) -> dict:
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}

    module_doc = ast.get_docstring(tree) or "No module docstring available."
    classes = []
    functions = []
    imports = []
    class_methods = set()

    # Pre-collect all methods inside class definitions to differentiate from top-level functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_methods.add(item)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            else:
                if node.module:
                    imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or "No class docstring available."
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_doc = ast.get_docstring(item) or ""
                    args = [arg.arg for arg in item.args.args]
                    methods.append({
                        "name": item.name,
                        "args": args,
                        "docstring": method_doc
                    })
            classes.append({
                "name": node.name,
                "bases": [ast.unparse(b) for b in node.bases],
                "docstring": class_doc,
                "methods": methods
            })
        elif isinstance(node, ast.FunctionDef) and node not in class_methods:
            # Top-level functions
            func_doc = ast.get_docstring(node) or "No function docstring available."
            args = [arg.arg for arg in node.args.args]
            functions.append({
                "name": node.name,
                "args": args,
                "docstring": func_doc
            })

    return {
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
        "imports": list(set(imports))
    }

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "dronevision"
    
    # 1. Generate docs/knowledge/ folder summaries
    kb_dir = project_root / "docs" / "knowledge"
    kb_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting code knowledge base into docs/knowledge/...")
    
    subfolders = ["data", "models", "loss", "engine", "inference", "utils"]
    for folder in subfolders:
        folder_path = src_dir / folder
        if not folder_path.exists():
            continue
            
        kb_file_path = kb_dir / f"{folder}.md"
        markdown_out = []
        markdown_out.append(f"# Codebase Knowledge Base — `dronevision/{folder}`\n")
        markdown_out.append(f"This document contains extracted API summaries, inputs, outputs, and dependencies for the `{folder}` component.\n")
        
        py_files = sorted(list(folder_path.glob("*.py")))
        if not py_files:
            continue
            
        for py_file in py_files:
            rel_path = py_file.relative_to(project_root).as_posix()
            parsed = parse_python_file(py_file)
            if not parsed:
                continue
                
            markdown_out.append(f"## Module: [{py_file.name}](file:///{project_root.as_posix()}/{rel_path})")
            markdown_out.append(f"**Path**: `{rel_path}`")
            markdown_out.append(f"**Imports**: {', '.join(parsed['imports']) if parsed['imports'] else 'None'}\n")
            markdown_out.append(f"### Description\n{parsed['module_doc']}\n")
            
            if parsed["classes"]:
                markdown_out.append("### Classes")
                for cls in parsed["classes"]:
                    bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
                    markdown_out.append(f"#### class `{cls['name']}{bases_str}`")
                    markdown_out.append(f"{cls['docstring']}\n")
                    
                    if cls["methods"]:
                        markdown_out.append("##### Methods")
                        for m in cls["methods"]:
                            markdown_out.append(f"- **`{m['name']}({', '.join(m['args'])})`**: {m['docstring'].splitlines()[0] if m['docstring'] else 'No docstring.'}")
                    markdown_out.append("")
                    
            if parsed["functions"]:
                markdown_out.append("### Functions")
                for func in parsed["functions"]:
                    markdown_out.append(f"#### def `{func['name']}({', '.join(func['args'])}`")
                    markdown_out.append(f"{func['docstring']}\n")
            
            markdown_out.append("---\n")
            
        with open(kb_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_out))
        print(f"Generated knowledge base index: {kb_file_path}")

    # 2. Generate docs/modules/ file-specific summaries
    modules_dir = project_root / "docs" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    print("Generating modular summaries into docs/modules/...")

    module_mappings = {
        "backbone": {
            "title": "Backbone Module (Feature Extraction)",
            "files": ["dronevision/models/backbone.py"],
            "related": ["dronevision/models/blocks.py"],
            "known_issues": "VRAM constraint when loading large image resolutions.",
            "future_improvements": "Add support for dual-stream feature fusion (RGB + Thermal)."
        },
        "neck": {
            "title": "Neck Module (Feature Pyramid Network)",
            "files": ["dronevision/models/neck.py"],
            "related": ["dronevision/models/blocks.py"],
            "known_issues": "Top-down FPN logic only, doesn't contain bottom-up PAN path.",
            "future_improvements": "Implement a BiFPN or PAFPN path for bidirectional scale fusion."
        },
        "head": {
            "title": "Head Module (Detection Heads)",
            "files": ["dronevision/models/head.py"],
            "related": ["dronevision/utils/anchors.py"],
            "known_issues": "Decoupled predictions can have high redundancy before NMS.",
            "future_improvements": "Support anchor-free predictions to reduce hyperparameter tuning complexity."
        },
        "loss": {
            "title": "Loss Module (CIoU & Objectness/Class Loss)",
            "files": ["dronevision/loss/detection_loss.py", "dronevision/loss/iou_loss.py"],
            "related": ["dronevision/utils/bbox.py"],
            "known_issues": "CIoU division-by-zero risk if bounding box dimensions become zero during training.",
            "future_improvements": "Incorporate DIoU / GIoU fallback losses and Focal Loss for objectness."
        },
        "dataset": {
            "title": "Dataset Module (Pre-processing & Batches)",
            "files": ["dronevision/data/dataset.py"],
            "related": ["dronevision/data/transforms.py", "dronevision/data/collate.py", "dronevision/data/augmentation.py"],
            "known_issues": "Standard transformations can overfit on very small training subsets.",
            "future_improvements": "Integrate Mosaic and MixUp augmentations directly in PyTorch data pipeline."
        },
        "trainer": {
            "title": "Trainer Module (Training Engine)",
            "files": ["dronevision/engine/trainer.py"],
            "related": ["dronevision/engine/callbacks.py", "dronevision/utils/logger.py"],
            "known_issues": "High VRAM usage during validation phase.",
            "future_improvements": "Add automated checkpoint uploading to cloud storage or DVC remote push."
        },
        "inference": {
            "title": "Inference Module (Predictor & Post-processing)",
            "files": ["dronevision/inference/predictor.py", "dronevision/inference/nms.py"],
            "related": ["dronevision/inference/visualizer.py"],
            "known_issues": "Standard NMS can be slow when processing thousands of predictions on CPU.",
            "future_improvements": "Implement batched GPU-based NMS via Torchvision utilities."
        },
        "evaluation": {
            "title": "Evaluation Module (mAP Metrics)",
            "files": ["dronevision/engine/evaluator.py"],
            "related": ["dronevision/utils/bbox.py"],
            "known_issues": "Metrics computation can consume high memory on large validation loops.",
            "future_improvements": "Add average precision (AP) curves plotting and export to MLflow artifacts."
        }
    }

    for name, info in module_mappings.items():
        module_path = modules_dir / f"{name}.md"
        markdown_out = []
        markdown_out.append(f"# Module Summary: {info['title']}\n")
        
        all_imports = []
        purposes = []
        details = []

        for fpath in info["files"]:
            file_ref = project_root / fpath
            if file_ref.exists():
                rel_path = file_ref.relative_to(project_root).as_posix()
                parsed = parse_python_file(file_ref)
                if parsed:
                    all_imports.extend(parsed["imports"])
                    purposes.append(parsed["module_doc"])
                    
                    details.append(f"### File: [{file_ref.name}](file:///{project_root.as_posix()}/{rel_path})")
                    details.append(f"**Path**: `{rel_path}`")
                    
                    if parsed["classes"]:
                        details.append("#### Classes")
                        for cls in parsed["classes"]:
                            bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
                            details.append(f"- **class `{cls['name']}{bases_str}`**: {cls['docstring'].splitlines()[0] if cls['docstring'] else 'No docstring.'}")
                            if cls["methods"]:
                                for m in cls["methods"]:
                                    details.append(f"  - `def {m['name']}({', '.join(m['args'])})`")
                    if parsed["functions"]:
                        details.append("#### Functions")
                        for func in parsed["functions"]:
                            details.append(f"- `def {func['name']}({', '.join(func['args'])}`")
                    details.append("")

        markdown_out.append("## Purpose")
        markdown_out.append("\n\n".join(purposes) if purposes else "No module descriptions available.")
        markdown_out.append("")
        
        markdown_out.append("## Inputs & Outputs")
        markdown_out.append("For detailed function inputs and outputs, check the definitions below:\n")
        markdown_out.append("\n".join(details))
        
        markdown_out.append("## Dependencies")
        markdown_out.append(f"- **Imports**: {', '.join(sorted(list(set(all_imports)))) if all_imports else 'None'}")
        markdown_out.append("")

        markdown_out.append("## Related Files")
        markdown_out.append("\n".join([f"- [{Path(rf).name}](file:///{project_root.as_posix()}/{rf})" for rf in info["files"] + info["related"]]))
        markdown_out.append("")

        markdown_out.append("## Known Issues")
        markdown_out.append(info["known_issues"])
        markdown_out.append("")

        markdown_out.append("## Future Improvements")
        markdown_out.append(info["future_improvements"])
        markdown_out.append("")

        with open(module_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_out))
        print(f"Generated module summary: {module_path}")

if __name__ == "__main__":
    main()
