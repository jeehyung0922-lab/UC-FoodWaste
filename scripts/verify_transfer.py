"""Create and compare read-only SHA-256 manifests for a project transfer."""

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
EXCLUDED = (".venv*", ".git", "__pycache__", "runs", ".pytest_cache", ".ipynb_checkpoints")
STATUSES = ("match", "missing_on_server", "extra_on_server", "content_mismatch",
            "unverified_due_to_read_error")


def excluded(name):
    return any(fnmatch.fnmatchcase(name.casefold(), pattern) for pattern in EXCLUDED)


def read_signature(info):
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def hash_file(path, original):
    with path.open("rb") as stream:
        if read_signature(os.fstat(stream.fileno())) != read_signature(original):
            raise OSError("File changed before hashing")
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        final = os.fstat(stream.fileno())
    if (size != original.st_size or read_signature(final) != read_signature(original)
            or read_signature(path.lstat()) != read_signature(original)):
        raise OSError("File changed during hashing")
    return {"size": size, "sha256": digest.hexdigest()}


def scan(root):
    files, errors = {}, []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            errors.append({"path": directory.relative_to(root).as_posix(), "scope": "tree",
                           "operation": "list_directory", "error": str(exc)})
            continue
        for entry in children:
            if excluded(entry.name):
                continue
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                # Windows DirEntry.stat() can omit device and inode information.
                info = path.lstat()
            except OSError as exc:
                errors.append({"path": relative, "scope": "tree", "operation": "stat", "error": str(exc)})
                continue
            reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            if stat.S_ISLNK(info.st_mode) or reparse:
                errors.append({"path": relative, "scope": "tree", "operation": "inspect",
                               "error": "Symlink or reparse point was not followed"})
            elif stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                try:
                    files[relative] = hash_file(path, info)
                except OSError as exc:
                    errors.append({"path": relative, "scope": "file", "operation": "hash", "error": str(exc)})
            else:
                errors.append({"path": relative, "scope": "file", "operation": "inspect",
                               "error": "Unsupported file type"})
    return sorted(files.items()), sorted(errors, key=lambda row: row["path"])


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def create_manifest(root, output):
    root = root.resolve(strict=True)
    output = output.resolve()
    if not root.is_dir():
        raise ValueError(f"Project root is not a directory: {root}")
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(root) and not any(excluded(p) for p in output.relative_to(root).parts[:-1]):
        raise ValueError("Place the manifest outside the project or inside an excluded directory such as runs")
    files, errors = scan(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        "schema": SCHEMA, "algorithm": "sha256", "excluded_names": list(EXCLUDED),
        "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "files": [{"path": name, **metadata} for name, metadata in files], "errors": errors,
    })
    print(f"{output}: {len(files)} files, {len(errors)} scan errors")
    return 2 if errors else 0


def valid_path(name):
    return (isinstance(name, str) and not any(c in name for c in "\\:\r\n\x00")
            and all(p not in ("", ".", "..") for p in name.split("/")))


def load_manifest(path):
    with path.open(encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if (manifest.get("schema"), manifest.get("algorithm"), manifest.get("excluded_names")) != (
            SCHEMA, "sha256", list(EXCLUDED)):
        raise ValueError(f"Incompatible manifest format or exclusions: {path}")
    files = {}
    for row in manifest["files"]:
        name, size, sha256 = row["path"], row["size"], row["sha256"]
        if (not valid_path(name) or name in files or type(size) is not int or size < 0
                or not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256)):
            raise ValueError(f"Invalid or duplicate file record: {path}: {name!r}")
        files[name] = row
    errors = manifest["errors"]
    for row in errors:
        name, scope = row["path"], row["scope"]
        if (scope not in ("file", "tree")
                or not (valid_path(name) or (name == "." and scope == "tree"))
                or not isinstance(row["operation"], str) or not isinstance(row["error"], str)):
            raise ValueError(f"Invalid scan error record: {path}")
    return manifest, files, errors


def error_paths(errors):
    return ({row["path"] for row in errors if row["scope"] == "file"},
            {row["path"] for row in errors if row["scope"] == "tree"})


def affected(name, blocked):
    files, trees = blocked
    return name in files or "." in trees or any(name == tree or name.startswith(tree + "/") for tree in trees)


def compare(local_path, server_path, output):
    local_doc, local_files, local_errors = load_manifest(local_path)
    server_doc, server_files, server_errors = load_manifest(server_path)
    local_blocked, server_blocked = error_paths(local_errors), error_paths(server_errors)
    counts = Counter(dict.fromkeys(STATUSES, 0))
    errors = [{**error, "side": side} for side, records in
              (("local", local_errors), ("server", server_errors)) for error in records]
    all_paths = set(local_files) | set(server_files) | {error["path"] for error in errors}
    rows, resend = [], []
    for name in sorted(all_paths):
        local = local_files.get(name, {})
        server = server_files.get(name, {})
        if affected(name, local_blocked) or affected(name, server_blocked):
            status = "unverified_due_to_read_error"
        elif not server:
            status = "missing_on_server"
        elif not local:
            status = "extra_on_server"
        elif (local["size"], local["sha256"]) != (server["size"], server["sha256"]):
            status = "content_mismatch"
        else:
            status = "match"
        counts[status] += 1
        rows.append({"path": name, "status": status,
                     "local_size": local.get("size"), "server_size": server.get("size"),
                     "local_sha256": local.get("sha256"), "server_sha256": server.get("sha256")})
        if status in ("missing_on_server", "content_mismatch"):
            resend.append(name)
    output.mkdir(parents=True, exist_ok=False)
    for filename, fields, entries in (
        ("comparison.csv", ("path", "status", "local_size", "server_size", "local_sha256", "server_sha256"), rows),
        ("read_errors.csv", ("side", "path", "scope", "operation", "error"), errors),
    ):
        with (output / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
    with (output / "resend_paths.txt").open("x", encoding="utf-8", newline="\n") as stream:
        for name in resend:
            stream.write(name + "\n")
    summary = {"local_root": local_doc["root"], "server_root": server_doc["root"],
               "counts": dict(counts), "scan_errors": len(errors), "files_to_resend": len(resend)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if errors else (1 if any(counts[s] for s in STATUSES[1:]) else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("manifest", help="Hash project files")
    generate.add_argument("--root", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    comparison = commands.add_parser("compare", help="Compare local source and server destination manifests")
    comparison.add_argument("--local", type=Path, required=True)
    comparison.add_argument("--server", type=Path, required=True)
    comparison.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "manifest":
            return create_manifest(args.root, args.out)
        return compare(args.local, args.server, args.out_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
