"""Copies the Helios engine into a web site, and keeps ittkezdjed/API.md up to date.

    python D:/AI/Unreal/ittkezdjed/sync.py SITE/helios           copy the engine into SITE/helios
    python D:/AI/Unreal/ittkezdjed/sync.py SITE/helios --build   rebuild the wasm first if it is stale
    python D:/AI/Unreal/ittkezdjed/sync.py --check SITE/helios   only report what a sync would change
    python D:/AI/Unreal/ittkezdjed/sync.py --api                 regenerate ittkezdjed/API.md

A sync writes into the target folder:
    pkg/                 the engine (helios_demo.js, helios_demo_bg.wasm, snippets/ with the bake worker)
    helios-boot.js       the start-up helper (see README.md)
    helios-engine.json   which engine this is: commit, date, wasm hash, the page API it exports
It mirrors pkg/ (files the engine no longer ships are removed). It touches nothing else in the folder.
If the folder already holds an engine, it compares the page API and prints the CHANGELOG entries
and engine commits since then. Use those to find what the site has to change.

Only the Python standard library is needed.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "web", "pkg")
WASM = os.path.join(PKG, "helios_demo_bg.wasm")
SOURCES = [os.path.join(ROOT, "crates", "helios-demo", "src", f) for f in ("web.rs", "web_editor.rs")]
OPS = os.path.join(ROOT, "crates", "helios-demo", "src", "editor", "ops.rs")
DOC = os.path.join(ROOT, "crates", "helios-demo", "src", "editor", "doc.rs")
MANIFEST = "helios-engine.json"
# Exports a page never calls (the bake workers use it).
INTERNAL = {"heliosBakeTask"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip() if r.returncode == 0 else ""


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---- Rust source → API description --------------------------------------------------------------

JS_TYPES = {
    "String": "string", "&str": "string", "bool": "boolean", "f32": "number", "f64": "number", "u32": "number",
    "i32": "number", "usize": "number", "Vec<f32>": "number[]", "Vec<u32>": "number[]", "Vec<u8>": "Uint8Array",
    "&[u8]": "Uint8Array", "js_sys::Promise": "Promise", "Option<js_sys::Uint8Array>": "Uint8Array?",
}


def js_type(rust):
    rust = rust.strip()
    return JS_TYPES.get(rust, rust)


def exports():
    """Every #[wasm_bindgen(js_name = X)] function: name, signature, doc, source file."""
    out = []
    pattern = re.compile(
        r"((?:[ \t]*///[^\n]*\n)*)[ \t]*#\[wasm_bindgen\(js_name = (\w+)\)\]\s*\n\s*pub (async )?fn \w+\(([^)]*)\)(?:\s*->\s*([^{]+?))?\s*\{")
    for path in SOURCES:
        for m in pattern.finditer(read(path)):
            doc, name, is_async, args, ret = m.groups()
            if name in INTERNAL:
                continue
            params = []
            for arg in filter(None, (a.strip() for a in args.split(","))):
                pname, ptype = arg.split(":", 1)
                pname = re.sub(r"_(\w)", lambda c: c.group(1).upper(), pname.strip().lstrip("_"))
                params.append(f"{pname}: {js_type(ptype)}")
            ret = (ret or "").strip()
            if ret.startswith("Result<"):
                ret = "Promise<" + js_type(ret[7:].split(",")[0]) + "> (rejects on error)"
            elif is_async:
                ret = f"Promise<{js_type(ret) if ret else 'void'}>"
            else:
                ret = js_type(ret) if ret else "void"
            text = " ".join(line.strip()[3:].strip() for line in doc.strip().splitlines())
            out.append({"name": name, "signature": f"{name}({', '.join(params)}) → {ret}", "doc": text,
                        "group": "editor" if path.endswith("web_editor.rs") else "page"})
    return out


def start_options():
    """heliosConfig keys with their URL parameter, and URL-only parameters."""
    src = read(SOURCES[0])
    config = sorted(set(re.findall(r'(?:option_string|number)\("(\w+)", "([\w-]+)"\)', src)))
    config += [(k, "") for k in sorted(set(re.findall(r'page_config\("(\w+)"\)', src))) if k not in dict(config)]
    url_only = sorted(set(re.findall(r'query_param\("([\w-]+)"\)', src)) - {p for _, p in config})
    return config, url_only


def rust_items(path, names):
    """Fields of the named structs / variants of the named enums, with their docs."""
    src = read(path).splitlines()
    items = {}
    i = 0
    while i < len(src):
        m = re.match(r"\s*pub (struct|enum) (\w+)\s*\{", src[i])
        if not m or m.group(2) not in names:
            i += 1
            continue
        kind, name = m.groups()
        # Docs and serde attributes above the item.
        j, head = i - 1, []
        while j >= 0 and (src[j].strip().startswith("///") or src[j].strip().startswith("#[")):
            head.insert(0, src[j].strip())
            j -= 1
        body, depth = [], 1
        i += 1
        while i < len(src) and depth > 0:
            depth += src[i].count("{") - src[i].count("}")
            if depth > 0 or src[i].strip() not in ("}",):
                body.append(src[i])
            i += 1
        items[name] = {"kind": kind, "head": head, "body": body}
    return items


def field_lines(body, variant_depth=False):
    """(name, type, optional, doc) per field of a struct body or of one enum variant's braces."""
    fields, doc, optional = [], [], False
    for line in body:
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        if s.startswith("#["):
            optional = optional or "default" in s
            if "flatten" in s:
                doc.append("(flattened: its fields sit in this object)")
            continue
        m = re.match(r"(?:pub\s+)?(\w+)\s*:\s*(.+?),?$", s)
        if m:
            name, typ = m.groups()
            if typ.startswith("Option<"):
                typ, optional = typ[7:-1], True
            fields.append((name, typ, optional, " ".join(doc)))
        doc, optional = [], False
    return fields


def snake(name):
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def enum_variants(body):
    """(variant, fields, doc) per variant of an enum body."""
    out, doc, i = [], [], 0
    while i < len(body):
        s = body[i].strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
        elif s.startswith("#["):
            pass
        else:
            m = re.match(r"(\w+)\s*\{(.*)\},?$", s)
            if m:
                inner = [f.strip() for f in m.group(2).split(",") if f.strip()]
                out.append((m.group(1), field_lines(inner), " ".join(doc)))
                doc = []
            elif re.match(r"(\w+)\s*\{$", s):
                name, inner, depth = s.split()[0], [], 1
                i += 1
                while depth > 0:
                    t = body[i].strip()
                    depth += t.count("{") - t.count("}")
                    if depth > 0:
                        inner.append(t)
                    i += 1
                out.append((name, field_lines(inner), " ".join(doc)))
                doc = []
                continue
            elif re.match(r"(\w+),?$", s):
                out.append((s.rstrip(","), [], " ".join(doc)))
                doc = []
            elif re.match(r"(\w+)\((.*)\),?$", s):
                n, t = re.match(r"(\w+)\((.*)\),?$", s).groups()
                out.append((n, [("value", t, False, "")], " ".join(doc)))
                doc = []
        i += 1
    return out


def json_type(rust):
    """A Rust field type as the JSON it (de)serialises to."""
    t = rust.strip()
    m = re.fullmatch(r"\[(\w+);\s*(\d+)\]", t)
    if m:
        return f"[number × {m.group(2)}]"
    m = re.fullmatch(r"Vec<(.+)>", t)
    if m:
        return json_type(m.group(1)) + "[]"
    m = re.fullmatch(r"BTreeMap<(\w+),\s*(.+)>", t)
    if m:
        return "{ [" + ("name" if m.group(1) == "String" else "index") + "]: " + json_type(m.group(2)) + " }"
    m = re.fullmatch(r"\((.+)\)", t)
    if m:
        return "[" + ", ".join(json_type(p) for p in m.group(1).split(",")) + "]"
    t = t.split("::")[-1]
    return {"String": "string", "bool": "boolean", "f32": "number", "u32": "number", "Value": "any JSON value",
            "MaterialGraph": "MaterialGraph (copy one from heliosEditorState().graphs or heliosMaterialNodes().presets)",
            "Graph": "PCG graph (copy one from heliosEditorState().pcg)",
            "Volume": "PCG volume box (copy one from heliosEditorState().pcg)"}.get(t, t)


def fields_md(fields):
    rows = []
    for name, typ, optional, doc in fields:
        if doc.startswith("Version 1"):
            continue  # migrated on load, never written
        rows.append(f"  * `{name}{'?' if optional else ''}`: {json_type(typ)}" + (f": {doc}" if doc else ""))
    return "\n".join(rows)


def js_names(text):
    """Rust names of exports in doc comments (`[`helios_frames`]`) → their JavaScript names."""
    return re.sub(r"\[`helios_(\w+)`\]", lambda m: "`helios" + "".join(p.title() for p in m.group(1).split("_")) + "`", text)


DOC_TYPES = ["Trs", "ComponentRef", "LightDef", "LightShape", "MaterialParams", "SlabDef", "DisplacementDef",
             "ParameterDef", "Environment", "FogDef", "CloudsDef", "WindDef", "SceneDoc", "BaseScene", "CameraDef",
             "ImportDef", "ActorEdit", "ComponentEdit", "AddedActor", "ComponentDef", "ComponentKindDef"]


def api_markdown():
    commit = git("log", "-1", "--format=%h %cs")
    ex = exports()
    config, url_only = start_options()
    md = [
        "# Helios page API (generated)",
        "",
        f"Generated by `python ittkezdjed/sync.py --api` from `crates/helios-demo/src/web.rs`, `web_editor.rs`,",
        f"`editor/ops.rs` and `editor/doc.rs` at engine commit {commit}. **Do not edit by hand**; regenerate.",
        "The how-to is [README.md](README.md).",
        "",
        "## Start options",
        "",
        "Set `globalThis.heliosConfig = { … }` before the engine module loads (`startHelios({ config })` does it).",
        "Each key can also come from the page URL's query parameter in the second column. The config takes",
        "precedence over the URL.",
        "",
        "| `heliosConfig` key | URL parameter |",
        "|---|---|",
    ]
    md += [f"| `{k}` | {('`?' + p + '=`') if p else '(config only)'} |" for k, p in config]
    md += ["", "URL-only parameters (diagnostics and comparisons): " + ", ".join(f"`?{p}=`" for p in url_only) + ".", ""]
    for group, title in (("page", "Page API"), ("editor", "Editor API (scene editing; see Editor operations below)")):
        md += [f"## {title}", ""]
        for e in sorted((e for e in ex if e["group"] == group), key=lambda e: e["name"]):
            md += [f"### `{e['signature']}`", "", js_names(e["doc"]) or "(undocumented)", ""]

    ops_src = rust_items(OPS, {"EditOp", "ComponentRef"})
    md += ["## Editor operations", "",
           "`heliosEditorApply(JSON.stringify({ op: \"<name>\", …fields }))` applies one as an undo step and returns",
           "`{\"created\": {actors, instances, lights, pcg}}` or `{\"error\": \"…\"}` (helios-boot.js `apply()` throws",
           "instead). Actors are addressed by live id (from `heliosEditorState().actors`), components by",
           "`{actor, component}`, instances and lights by scene index. An op may carry `merge: \"key\"` to join the",
           "previous step with the same key (slider drags).", ""]
    for name, fields, doc in enum_variants(ops_src["EditOp"]["body"]):
        md += [f"* **`{snake(name)}`**" + (f": {doc}" if doc else "")] + ([fields_md(fields)] if fields else [])
    md += ["", "## Data types (scene files and editor operations)", "",
           "Scene files (`.helios.json`, `heliosConfig.sceneFile`) are a `SceneDoc` with `\"format\": \"helios-scene\"` and",
           "`\"version\": 2`. Optional fields (`?`) may be left out. The editor's Save (Ctrl+S in `editor.html`)",
           "writes one, the easiest way to get a valid file to start from.", ""]
    items = rust_items(DOC, set(DOC_TYPES))
    items.update({k: v for k, v in ops_src.items() if k == "ComponentRef"})
    for name in DOC_TYPES:
        item = items.get(name)
        if not item:
            continue
        head_doc = " ".join(h[3:].strip() for h in item["head"] if h.startswith("///"))
        tag = re.search(r'tag = "(\w+)"', " ".join(item["head"]))
        untagged = "untagged" in " ".join(item["head"])
        md += [f"### `{name}`", ""]
        if head_doc:
            md += [head_doc, ""]
        if item["kind"] == "struct":
            md += [fields_md(field_lines(item["body"])), ""]
        else:
            note = (f"Tagged by `\"{tag.group(1)}\"`: one of" if tag else "Untagged: one of" if untagged else "One of")
            md += [note + ":"]
            for vname, fields, vdoc in enum_variants(item["body"]):
                label = f'`"{snake(vname)}"`' if not untagged else f"`{vname}`"
                md += [f"* {label}" + (f": {vdoc}" if vdoc else "")] + ([fields_md(fields)] if fields else [])
            md += [""]
    text = "\n".join(line for line in md if line is not None)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


# ---- Sync -------------------------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_source():
    newest, which = 0.0, ""
    for base in (os.path.join(ROOT, "crates"),):
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.endswith((".rs", ".wgsl", ".toml", ".js")):
                    t = os.path.getmtime(os.path.join(dirpath, f))
                    if t > newest:
                        newest, which = t, os.path.join(dirpath, f)
    return newest, which


def manifest():
    head = git("rev-parse", "HEAD")
    return {
        "engine": "Helios",
        "source": ROOT,
        "commit": head,
        "commitDate": git("log", "-1", "--format=%cs"),
        "dirty": bool(git("status", "--porcelain", "--", "crates", "web", "ittkezdjed")),
        "wasmBuilt": datetime.datetime.fromtimestamp(os.path.getmtime(WASM)).isoformat(timespec="seconds"),
        "wasmSha256": sha256(WASM),
        "synced": datetime.datetime.now().isoformat(timespec="seconds"),
        "api": {e["name"]: e["signature"] for e in exports()},
        "config": [k for k, _ in start_options()[0]],
    }


def changelog_since(date):
    """CHANGELOG.md entries (## YYYY-MM-DD sections) on or after `date`."""
    path = os.path.join(HERE, "CHANGELOG.md")
    if not os.path.exists(path):
        return []
    out, keep = [], False
    for line in read(path).splitlines():
        m = re.match(r"## (\d{4}-\d{2}-\d{2})", line)
        if m:
            keep = not date or m.group(1) >= date
        if keep:
            out.append(line)
    return out


def report(old, new):
    if not old:
        print("First sync into this folder.")
        return
    print(f"Before: {old.get('commit', '?')[:9]} ({old.get('commitDate', '?')}), now: {new['commit'][:9]} ({new['commitDate']})")
    if old.get("wasmSha256") == new["wasmSha256"]:
        print("The engine is unchanged (same wasm).")
        return
    oa, na = old.get("api", {}), new["api"]
    added = sorted(set(na) - set(oa))
    removed = sorted(set(oa) - set(na))
    changed = sorted(k for k in set(na) & set(oa) if na[k] != oa[k])
    if added or removed or changed:
        print("Page API changes:")
        for k in added:
            print(f"  + {na[k]}")
        for k in removed:
            print(f"  - {oa[k]}   (REMOVED: the site must stop calling it)")
        for k in changed:
            print(f"  ~ {oa[k]}\n    → {na[k]}")
    else:
        print("Page API: same functions and signatures.")
    oc, nc = set(old.get("config", [])), set(new["config"])
    if oc != nc:
        print(f"heliosConfig keys: + {sorted(nc - oc)}  - {sorted(oc - nc)}")
    notes = changelog_since(old.get("commitDate"))
    if notes:
        print("\nCHANGELOG entries since the last sync (read them; they say what a site has to change):")
        print("\n".join("  " + n for n in notes))
    if old.get("commit") and git("cat-file", "-t", old["commit"]) == "commit":
        log = git("log", "--oneline", f"{old['commit']}..HEAD")
        if log:
            print(f"\nEngine commits since the last sync ({len(log.splitlines())}):")
            print("\n".join("  " + l for l in log.splitlines()[:40]))


def sync(target, build, check):
    target = os.path.abspath(target)
    try:
        inside = os.path.commonpath([os.path.normcase(target), os.path.normcase(ROOT)]) == os.path.normcase(ROOT)
    except ValueError:  # another drive
        inside = False
    if inside:
        sys.exit(f"{target} is inside the engine repository; sync into the web site's own folder.")
    newest, which = newest_source()
    if not os.path.exists(WASM) or os.path.getmtime(WASM) < newest:
        if build:
            shell = shutil.which("pwsh") or "powershell"
            print("The wasm is older than the sources: building (./build-web.ps1)…", flush=True)
            subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(ROOT, "build-web.ps1")],
                           cwd=ROOT, check=True)
        else:
            print(f"WARNING: web/pkg is older than {os.path.relpath(which, ROOT)}; run with --build for the current engine.")
    new = manifest()
    old_path = os.path.join(target, MANIFEST)
    old = json.loads(read(old_path)) if os.path.exists(old_path) else None
    report(old, new)
    if new["dirty"]:
        print("NOTE: the engine has uncommitted changes; the manifest's commit is not the whole story.")
    if check:
        return
    pkg = os.path.join(target, "pkg")
    if os.path.exists(pkg):
        if not os.path.exists(os.path.join(pkg, "helios_demo.js")):
            sys.exit(f"{pkg} exists but holds no Helios engine; not overwriting it.")
        shutil.rmtree(pkg)
    os.makedirs(target, exist_ok=True)
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns(".gitignore"))
    shutil.copy2(os.path.join(HERE, "helios-boot.js"), os.path.join(target, "helios-boot.js"))
    with open(old_path, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2)
        f.write("\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(pkg) for f in fs) / 2**20
    print(f"\nSynced into {target}: pkg/ ({size:.1f} MiB), helios-boot.js, {MANIFEST}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="the site's engine folder, e.g. D:/AI/MySite/helios")
    ap.add_argument("--build", action="store_true", help="rebuild web/pkg first if it is older than the sources")
    ap.add_argument("--check", action="store_true", help="report only, copy nothing")
    ap.add_argument("--api", action="store_true", help="regenerate ittkezdjed/API.md")
    args = ap.parse_args()
    if args.api:
        with open(os.path.join(HERE, "API.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(api_markdown())
        print("wrote ittkezdjed/API.md")
    if args.target:
        sync(args.target, args.build, args.check)
    elif not args.api:
        ap.print_help()


if __name__ == "__main__":
    main()
