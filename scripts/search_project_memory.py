"""
search_project_memory.py — SQLite FTS5 Memory Layer search client.
Indexes metadata profiles and modular summaries in project_memory.db.
Supports fast, BM25-ranked full-text queries over project documents.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

DB_FILE = "project_memory.db"

def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

class SQLiteMemoryEngine:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db_path = project_root / DB_FILE

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def setup_database(self) -> None:
        with self.get_connection() as conn:
            # Drop existing tables to ensure clean schema rebuild
            conn.execute("DROP TABLE IF EXISTS project_memory")
            # Create FTS5 virtual table
            conn.execute(
                "CREATE VIRTUAL TABLE project_memory USING fts5("
                "   filepath, "
                "   header, "
                "   content, "
                "   tokenize='unicode61'"
                ")"
            )
            conn.commit()

    def parse_markdown_file(self, file_path: Path) -> list[tuple[str, str, str]]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
            return []

        rel_path = file_path.relative_to(self.project_root).as_posix()
        sections = []

        # Split content by any level of markdown header (H1 to H4)
        header_splits = re.split(r'\n(?=#{1,4}\s+)', content)
        
        for section in header_splits:
            section = section.strip()
            if not section:
                continue
                
            lines = section.splitlines()
            # Clean header name
            header = re.sub(r'^#{1,4}\s+', '', lines[0].strip()) if lines else "General"
            
            sections.append((rel_path, header, section))
            
        return sections

    def reindex(self) -> None:
        print("Initializing SQLite FTS5 search database...")
        self.setup_database()
        
        indexed_files_count = 0
        indexed_sections_count = 0

        # Files to parse
        files_to_index = [
            "SESSION_START.md",
            "AGENTS.md",
            "PROJECT_CONTEXT.md",
            "TRAINING_STATUS.md",
            "ARCHITECTURE.md",
            "TRD.md",
            "DECISIONS.md",
            "EXPERIMENTS.md",
            "ERRORS_AND_FIXES.md",
            "DATASET_PROFILE.md",
            "MODEL_PROFILE.md",
            "PROJECT_HEALTH.md",
            "CHANGELOG_AI.md"
        ]

        with self.get_connection() as conn:
            # 1. Index root markdown files
            for fname in files_to_index:
                fpath = self.project_root / fname
                if fpath.exists():
                    sections = self.parse_markdown_file(fpath)
                    if sections:
                        conn.executemany(
                            "INSERT INTO project_memory(filepath, header, content) VALUES(?, ?, ?)",
                            sections
                        )
                        indexed_files_count += 1
                        indexed_sections_count += len(sections)

            # 2. Index docs/modules/*.md files
            modules_dir = self.project_root / "docs" / "modules"
            if modules_dir.exists():
                for f in modules_dir.glob("*.md"):
                    sections = self.parse_markdown_file(f)
                    if sections:
                        conn.executemany(
                            "INSERT INTO project_memory(filepath, header, content) VALUES(?, ?, ?)",
                            sections
                        )
                        indexed_files_count += 1
                        indexed_sections_count += len(sections)

            # 3. Index docs/knowledge/*.md files
            kb_dir = self.project_root / "docs" / "knowledge"
            if kb_dir.exists():
                for f in kb_dir.glob("*.md"):
                    sections = self.parse_markdown_file(f)
                    if sections:
                        conn.executemany(
                            "INSERT INTO project_memory(filepath, header, content) VALUES(?, ?, ?)",
                            sections
                        )
                        indexed_files_count += 1
                        indexed_sections_count += len(sections)

            conn.commit()

        print(f"Successfully indexed {indexed_sections_count} sections across {indexed_files_count} files in project_memory.db.")

    def search(self, query: str, limit: int = 3) -> list[dict]:
        # If database is missing, run reindex first
        if not self.db_path.exists():
            self.reindex()

        # SQLite FTS5 MATCH syntax escaping (simple cleaning for search safety)
        cleaned_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not cleaned_query:
            return []

        # Connect and search using BM25 rank (lower is better in FTS5)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT filepath, header, content, rank "
                "FROM project_memory "
                "WHERE project_memory MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (cleaned_query, limit)
            )
            rows = cursor.fetchall()
            
        results = []
        for row in rows:
            results.append({
                "filepath": row[0],
                "header": row[1],
                "content": row[2],
                "rank": row[3]
            })
        return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Query SQLite FTS5 project memory layer.")
    parser.add_argument("--query", type=str, help="Full-text search query (e.g. 'NaN loss' or 'backbone channels')")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the FTS5 search index database")
    parser.add_argument("--limit", type=int, default=3, help="Number of search results to return")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    engine = SQLiteMemoryEngine(project_root)

    if args.reindex:
        engine.reindex()
        return

    if not args.query:
        print("Please specify a search --query or pass --reindex to rebuild the index database.")
        return

    results = engine.search(args.query, limit=args.limit)

    if not results:
        safe_print(f"\nNo documentation matches found for: '{args.query}'")
        return

    safe_print(f"\nTop BM25-ranked SQLite FTS5 memory matches for: '{args.query}'\n")
    for i, res in enumerate(results, 1):
        safe_print("=" * 80)
        safe_print(f"[{i}] FILE: {res['filepath']} | SECTION: {res['header']} (FTS5 BM25 Rank: {res['rank']:.4f})")
        safe_print("-" * 80)
        # Truncate to first 12 lines for display
        lines = res["content"].splitlines()
        safe_print("\n".join(lines[:12]))
        if len(lines) > 12:
            safe_print("...")
        safe_print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
