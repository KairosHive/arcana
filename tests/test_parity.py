"""
The app acts only through the ops.

If the web UI could change something the CLI or an agent cannot, or changed it
by another route, the one vocabulary would split in two. So the app's modules
may not call into the modules that change state or make pictures. The allowed
path into those is arcana.ops. This walks their syntax trees rather than
grepping, so a comment or a string that mentions a module does not count.

Run with:  python -m pytest tests/test_parity.py -q
"""

import ast
import os

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arcana")

APP_MODULES = ("arcana.py", "ui_datasets.py")

# Modules the app may not import or call. Each phase of OPS_PLAN.md that moves
# a tab onto the ops adds its modules here.
FORBIDDEN = {"boards",                                              # P1
             "color_transfer", "lab_transfer", "classic_transfer",   # P2
             "transfer_methods", "style_transfer", "lut", "generate"}

# Modules the app reads from freely but must not write through. P4 moved the
# Datasets tab onto the dataset ops, so these functions are the ops' alone.
FORBIDDEN_CALLS = {
    "db": {"index_dataset", "extend_dataset", "rework_dataset", "finish_dataset",
           "refresh_marks", "warm_label_cache"},
    "relocate": {"relocate_legacy", "relocate_bundle"},
    "legacy": {"delete_dataset", "convert"},
}


def _imports(tree):
    """Local names bound to a forbidden module, and from-imports of one."""
    bound, direct = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mod = a.name.split(".")[-1]
                if mod in FORBIDDEN:
                    bound.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[-1]
            if mod in FORBIDDEN:
                direct.append(f"from {node.module} import ... (line {node.lineno})")
            for a in node.names:
                if a.name in FORBIDDEN:
                    bound.add(a.asname or a.name)
    return bound, direct


def test_the_app_changes_state_only_through_the_ops():
    offenders = []
    for name in APP_MODULES:
        tree = ast.parse(open(os.path.join(PKG, name), encoding="utf-8").read())
        bound, direct = _imports(tree)
        offenders += [f"{name}: {d}" for d in direct]
        offenders += [f"{name}: imports {b}" for b in sorted(bound)]
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in bound:
                offenders.append(f"{name}:{node.lineno} uses {node.value.id}.{node.attr}")
    assert not offenders, "Go through arcana.ops instead:\n  " + "\n  ".join(offenders)


def _write_calls(tree):
    """Writing functions of db, relocate or legacy, by module alias or imported by name."""
    aliases = {}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mod = a.name.split(".")[-1]
                if mod in FORBIDDEN_CALLS:
                    aliases[a.asname or a.name] = mod
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[-1]
            for a in node.names:
                if a.name in FORBIDDEN_CALLS:
                    aliases[a.asname or a.name] = a.name
                elif mod in FORBIDDEN_CALLS and a.name in FORBIDDEN_CALLS[mod]:
                    found.append(f"imports {mod}.{a.name} (line {node.lineno})")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in aliases and node.attr in FORBIDDEN_CALLS[aliases[node.value.id]]:
            found.append(f"calls {aliases[node.value.id]}.{node.attr} (line {node.lineno})")
    return found


def test_the_app_writes_datasets_only_through_the_ops():
    offenders = []
    for name in APP_MODULES:
        tree = ast.parse(open(os.path.join(PKG, name), encoding="utf-8").read())
        offenders += [f"{name}: {f}" for f in _write_calls(tree)]
    assert not offenders, "Go through the dataset ops instead:\n  " + "\n  ".join(offenders)


def test_the_dataset_check_would_catch_a_write():
    src = ("from . import db as _db\n_db.index_dataset('x', 'y')\n"
           "from .legacy import delete_dataset\n")
    assert len(_write_calls(ast.parse(src))) == 2


def test_the_check_would_catch_a_call():
    src = "from . import boards as _boards\n_boards.save('x', [])\n"
    bound, _ = _imports(ast.parse(src))
    assert bound == {"_boards"}
