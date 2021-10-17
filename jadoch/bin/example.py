#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
from jadoch.core.context import Context
from jadoch.core.app import harness, slurmify


def main(ctx: Context) -> None:
    ctx.parser.add_argument("--foo", action="store_true")
    ctx.parser.set_defaults(sb_partition="short-24core", modules=["shared"])
    args = slurmify(ctx.parser)
    ctx.log.info("example")


if __name__ == "__main__":
    harness(main)
