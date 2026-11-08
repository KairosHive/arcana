# arcana.ops -- one vocabulary for everything Arcana can do
#
# Every capability is an op, declared once in registry terms. The CLI
# (arcana/cli.py), the MCP server and the tests reach the same ops through
# call() and exec_lines(), and the text grammar, help, completion and schema
# are all generated from the declarations. See OPS_PLAN.md.
#
#     from arcana import ops
#     ops.call("search text", {"dataset": "japan", "phrase": ["rain"]}, actor="cli")
#     ops.exec_lines(['search text japan --phrase "rain"', "item show japan:12"])

from . import registry, grammar, render, archive          # noqa: F401
from .registry import (                                     # noqa: F401
    ACTORS, Args, Attachment, Batch, Context, Op, Refusal, Result, Step, TextResult,
    all_ops, call, context, exec_lines, get, op, positional, register,
)
from .grammar import help_text, parse_line, parse_tokens   # noqa: F401

# Importing the op modules registers their ops, in the order `op list` shows.
from . import system, dataset, search, item, alchemy       # noqa: F401,E402
