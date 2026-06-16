# src/tools/github_storage_v1.py
"""
Bombay GitHub Storage v1

Purpose
-------
Use GitHub as persistent storage for JSON reports.

Why:
Render free storage is temporary.
Every deploy/restart can lose uploaded logs.

This tool can:
- upload JSON file from Render logs/data to GitHub repo
- download JSON file from GitHub repo back to Render
- check if a file exists on GitHub

Required Render environment variables:
- GITHUB_TOKEN
- GITHUB_REPO

Example:
GITHUB_REPO=gianniskosmadakisgk-ops/bombay-engine
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


GITHUB_API_BASE = "https://api.github.com"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def github_token() -> str:
    return _env("GITHUB_TOKEN")


def github_repo() -> str:
    return _env("GITHUB_REPO")


def github_branch() -> str:
    return _env("GITHUB_BRANCH", "main")


def storage_prefix() -> str:
    """
    Folder inside GitHub repo where persistent reports are stored.
    """
    return _env("GITHUB_STORAGE_PREFIX", "persistent")


def ok_config() -> Tuple[bool, str]:
    if not github_token():
        return False, "Missing GITHUB_TOKEN"
    if not github_repo():
        return False, "Missing GITHUB_REPO"
    return True, "ok"


def headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def api_url(path: str) -> str:
    repo = github_repo()
    clean_path = path.strip().lstrip("/")
    return f"{GITHUB_API_BASE}/repos/{repo}/contents/{clean_path}"


def persistent_path(logs_or_data_path: str) -> str:
    """
    Converts local path like:
      logs/thursday_report_v3.json

    into GitHub path:
      persistent/logs/thursday_report_v3.json
    """
    clean = logs_or_data_path.strip().lstrip("/")
    return f"{storage_prefix().strip('/')}/{clean}"


def read_local_json(local_path: str | Path) -> Any:
    p = Path(local_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_local_json(local_path: str | Path, payload: Any) -> None:
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def github_get_file(github_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Returns:
    - ok
    - GitHub file metadata or None
    - message
    """
    good, msg = ok_config()
    if not good:
        return False, None, msg

    url = api_url(github_path)
    params = {"ref": github_branch()}

    try:
        r = requests.get(url, headers=headers(), params=params, timeout=25)
    except Exception as e:
        return False, None, f"GitHub GET error: {e}"

    if r.status_code == 404:
        return False, None, "not_found"

    if r.status_code >= 300:
        return False, None, f"GitHub GET failed {r.status_code}: {r.text[:500]}"

    try:
        return True, r.json(), "ok"
    except Exception as e:
        return False, None, f"Invalid GitHub JSON response: {e}"


def github_upload_json(
    local_path: str | Path,
    target_relative_path: str,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uploads a local JSON file to GitHub persistent storage.

    local_path:
      /opt/render/project/src/logs/thursday_report_v3.json

    target_relative_path:
      logs/thursday_report_v3.json
    """
    good, msg = ok_config()
    if not good:
        return {"status": "error", "message": msg}

    local = Path(local_path)
    if not local.exists():
        return {
            "status": "error",
            "message": "local_file_missing",
            "local_path": str(local),
        }

    try:
        raw = local.read_bytes()
    except Exception as e:
        return {
            "status": "error",
            "message": f"failed_read_local_file: {e}",
            "local_path": str(local),
        }

    github_path = persistent_path(target_relative_path)

    exists, meta, get_msg = github_get_file(github_path)
    sha = None
    if exists and isinstance(meta, dict):
        sha = meta.get("sha")

    content_b64 = base64.b64encode(raw).decode("utf-8")

    body = {
        "message": message or f"Bombay persistent save: {target_relative_path}",
        "content": content_b64,
        "branch": github_branch(),
    }

    if sha:
        body["sha"] = sha

    try:
        r = requests.put(api_url(github_path), headers=headers(), json=body, timeout=30)
    except Exception as e:
        return {
            "status": "error",
            "message": f"GitHub PUT error: {e}",
            "github_path": github_path,
            "local_path": str(local),
        }

    if r.status_code >= 300:
        return {
            "status": "error",
            "message": f"GitHub PUT failed {r.status_code}",
            "details": r.text[:1000],
            "github_path": github_path,
            "local_path": str(local),
        }

    return {
        "status": "ok",
        "action": "updated" if sha else "created",
        "github_path": github_path,
        "local_path": str(local),
        "branch": github_branch(),
        "repo": github_repo(),
    }


def github_download_json(
    source_relative_path: str,
    local_path: str | Path,
) -> Dict[str, Any]:
    """
    Downloads JSON from GitHub persistent storage to local Render path.

    source_relative_path:
      logs/thursday_report_v3.json

    local_path:
      /opt/render/project/src/logs/thursday_report_v3.json
    """
    good, msg = ok_config()
    if not good:
        return {"status": "error", "message": msg}

    github_path = persistent_path(source_relative_path)

    exists, meta, get_msg = github_get_file(github_path)
    if not exists or not isinstance(meta, dict):
        return {
            "status": "error",
            "message": get_msg,
            "github_path": github_path,
            "local_path": str(local_path),
        }

    encoded = meta.get("content")
    if not encoded:
        return {
            "status": "error",
            "message": "github_file_has_no_content",
            "github_path": github_path,
        }

    try:
        raw = base64.b64decode(encoded)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {
            "status": "error",
            "message": f"decode_json_failed: {e}",
            "github_path": github_path,
        }

    try:
        write_local_json(local_path, payload)
    except Exception as e:
        return {
            "status": "error",
            "message": f"write_local_failed: {e}",
            "github_path": github_path,
            "local_path": str(local_path),
        }

    return {
        "status": "ok",
        "github_path": github_path,
        "local_path": str(local_path),
        "branch": github_branch(),
        "repo": github_repo(),
    }


def github_file_exists(source_relative_path: str) -> Dict[str, Any]:
    github_path = persistent_path(source_relative_path)
    exists, meta, msg = github_get_file(github_path)

    return {
        "status": "ok" if exists else "missing",
        "exists": bool(exists),
        "message": msg,
        "github_path": github_path,
        "repo": github_repo(),
        "branch": github_branch(),
        "sha": meta.get("sha") if isinstance(meta, dict) else None,
        "size": meta.get("size") if isinstance(meta, dict) else None,
    }


if __name__ == "__main__":
    print(json.dumps({
        "config_ok": ok_config(),
        "repo": github_repo(),
        "branch": github_branch(),
        "storage_prefix": storage_prefix(),
    }, ensure_ascii=False, indent=2))
