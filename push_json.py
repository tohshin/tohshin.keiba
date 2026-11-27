#!/usr/bin/env python3
"""
指定した jsons/data_YYYY-MM-DD.json のみを GitHub Pages (origin/main) に反映するスクリプト。

ローカルの作業ツリーや未プッシュコミットに影響を与えずに、
最新の origin/main をベースにした一時 worktree 上で対象ファイルだけをコミット & プッシュします。

使い方:
    python push_json.py data_2026-11-22.json
    python push_json.py 2026-11-22
    python push_json.py 20261122
    python push_json.py 2026-11-21 2026-11-22
    python push_json.py data_2026-11-22.json --dry-run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
JSONS_DIR = REPO_DIR / "jsons"
REMOTE = "origin"
BRANCH = "main"


def run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed (code {proc.returncode}): {msg}")
    return proc


def resolve_target_file(raw: str) -> tuple[Path, str | None]:
    """
    引数（日付 / ファイル名 / パス）を jsons/ 配下の実ファイルパスに解決する。
    戻り値: (実ファイルパス, data_YYYY-MM-DD.json の場合は YYYY-MM-DD 文字列 / それ以外は None)
    """
    s = raw.strip().strip('"').strip("'")

    # YYYYMMDD -> data_YYYY-MM-DD.json
    m_compact = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m_compact:
        d_str = f"{m_compact.group(1)}-{m_compact.group(2)}-{m_compact.group(3)}"
        candidate = JSONS_DIR / f"data_{d_str}.json"
        return candidate, d_str

    # YYYY-MM-DD -> data_YYYY-MM-DD.json
    m_iso = re.fullmatch(r"(\d{4}-\d{2}-\d{2})", s)
    if m_iso:
        d_str = m_iso.group(1)
        candidate = JSONS_DIR / f"data_{d_str}.json"
        return candidate, d_str

    # ファイル名またはパス指定
    p = Path(s)
    if not p.is_absolute():
        if (JSONS_DIR / p.name).exists() and (len(p.parts) == 1 or p.parts[0].lower() == "jsons"):
            p = JSONS_DIR / p.name
        else:
            p = (REPO_DIR / p).resolve()
    else:
        p = p.resolve()

    # jsons/ 配下であることを確認
    try:
        p.relative_to(JSONS_DIR.resolve())
    except ValueError:
        raise ValueError(f"jsons/ 配下のファイルを指定してください: {raw}")

    m_name = re.fullmatch(r"data_(\d{4}-\d{2}-\d{2})\.json", p.name)
    d_str = m_name.group(1) if m_name else None
    return p, d_str


def validate_json_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONの構文エラーです ({path.name}): {e}")


def update_meta_json_if_needed(worktree_jsons: Path, dates_to_ensure: list[str]) -> bool:
    """worktree 側の meta.json に指定日付が含まれていなければ追加する。変更があれば True。"""
    if not dates_to_ensure:
        return False
    meta_path = worktree_jsons / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"dates": [], "latest": ""}

    existing = set(meta.get("dates") or [])
    new_dates = [d for d in dates_to_ensure if d not in existing]
    if not new_dates:
        return False

    all_dates = sorted(existing | set(new_dates))
    meta["dates"] = all_dates
    meta["latest"] = max(all_dates) if all_dates else ""
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return True


def push_selected_jsons(
    targets: list[tuple[Path, str | None]],
    commit_message: str | None = None,
    update_meta: bool = True,
    dry_run: bool = False,
    max_retries: int = 2,
) -> int:
    rel_names = [f"jsons/{p.name}" for p, _ in targets]
    dates = [d for _, d in targets if d]

    if not commit_message:
        if len(targets) == 1:
            commit_message = f"Update {rel_names[0]}"
        else:
            commit_message = f"Update {', '.join(p.name for p, _ in targets)}"

    for attempt in range(1, max_retries + 1):
        print(f"[1/4] {REMOTE}/{BRANCH} の最新状態を取得中...")
        run_git(["fetch", REMOTE, BRANCH], cwd=REPO_DIR)

        tmp_dir = Path(tempfile.mkdtemp(prefix="push_json_wt_"))
        try:
            print(f"[2/4] 一時ワークツリーを作成中 ({REMOTE}/{BRANCH})...")
            run_git(["worktree", "add", "--detach", str(tmp_dir), f"{REMOTE}/{BRANCH}"], cwd=REPO_DIR)

            wt_jsons = tmp_dir / "jsons"
            wt_jsons.mkdir(parents=True, exist_ok=True)

            staged_paths: list[str] = []
            for src_path, _ in targets:
                dst_path = wt_jsons / src_path.name
                shutil.copy2(src_path, dst_path)
                staged_paths.append(f"jsons/{src_path.name}")

            if update_meta and update_meta_json_if_needed(wt_jsons, dates):
                staged_paths.append("jsons/meta.json")
                print("      jsons/meta.json に新規日付を追加しました")

            run_git(["add", "--", *staged_paths], cwd=tmp_dir)

            diff_check = run_git(["diff", "--cached", "--quiet"], cwd=tmp_dir, check=False)
            if diff_check.returncode == 0:
                print(f"変更はありませんでした（{REMOTE}/{BRANCH} と同一内容です）。プッシュをスキップします。")
                return 0

            stat = run_git(["diff", "--cached", "--stat"], cwd=tmp_dir).stdout.strip()
            print("[3/4] 反映対象の変更:")
            for line in stat.splitlines():
                print(f"      {line}")

            if dry_run:
                print("[dry-run] コミット・プッシュは実行しませんでした。")
                return 0

            run_git(["commit", "-m", commit_message], cwd=tmp_dir)
            print(f"[4/4] {REMOTE}/{BRANCH} へプッシュ中...")
            push_res = run_git(["push", REMOTE, f"HEAD:{BRANCH}"], cwd=tmp_dir, check=False)
            if push_res.returncode == 0:
                print(f"完了しました: {', '.join(staged_paths)} -> {REMOTE}/{BRANCH}")
                return 0

            err_msg = (push_res.stderr or push_res.stdout).strip()
            if attempt < max_retries and ("non-fast-forward" in err_msg or "fetch first" in err_msg):
                print(f"リモートが更新されたため再試行します ({attempt}/{max_retries})...")
                continue
            raise RuntimeError(f"git push に失敗しました: {err_msg}")
        finally:
            run_git(["worktree", "remove", "--force", str(tmp_dir)], cwd=REPO_DIR, check=False)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="指定した jsons/data_YYYY-MM-DD.json だけを GitHub Pages (origin/main) に反映します。"
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="対象ファイルまたは日付（例: data_2026-11-22.json, 2026-11-22, 20261122）",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="本日の日付の data_YYYY-MM-DD.json を対象に追加する",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        help="コミットメッセージ（省略時は自動生成）",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="jsons/meta.json の自動更新を行わない",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コミット・プッシュを行わず差分確認のみ実行する",
    )
    args = parser.parse_args()

    raw_targets = list(args.targets)
    if args.today:
        raw_targets.append(date.today().strftime("%Y-%m-%d"))

    if not raw_targets:
        parser.print_help()
        return 1

    # 重複除去しつつ解決・検証
    seen: set[Path] = set()
    resolved: list[tuple[Path, str | None]] = []
    try:
        for raw in raw_targets:
            p, d_str = resolve_target_file(raw)
            if p not in seen:
                validate_json_file(p)
                seen.add(p)
                resolved.append((p, d_str))
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    try:
        return push_selected_jsons(
            resolved,
            commit_message=args.message,
            update_meta=not args.no_meta,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
