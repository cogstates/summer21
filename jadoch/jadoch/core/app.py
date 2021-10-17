# -*- coding: utf-8 -*
import argparse
import logging
import os
import subprocess
import sys
from typing import Any, Callable

from .context import Context, get_context
from .slurm import sbatch


def slurmify(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Add slurmified arguments.
    grp = parser.add_argument_group("slurmified arguments")
    grp.add_argument("-m", "--modules", nargs="*")
    grp.add_argument("-l", "--local", action="store_true", help="run locally")
    grp.add_argument("-y", "--dryrun", action="store_true", help="don't send to slurm")

    # Add sbatch arguments.
    tkns = (
        subprocess.run(["sbatch", "--help"], check=True, capture_output=True)
        .stdout.decode("utf-8")
        .split()
    )
    flags = set(
        [tkn.split("=")[0][2:].replace("[", "") for tkn in tkns if tkn.startswith("--")]
    )
    grp = parser.add_argument_group("sbatch arguments")
    for flag in sorted(list(flags)):
        grp.add_argument("--sb-" + flag, help=argparse.SUPPRESS)
    parser.epilog = "\n".join(
        [
            parser.epilog or "",
            "sbatch arguments can be modified using --sb-<argument_name>",
        ]
    )
    args = parser.parse_args()

    # Send the job to slurm.
    if not args.local:
        sbflags = {flg: getattr(args, "sb_" + flg.replace("-", "_")) for flg in flags}
        sbflags = {key: val for key, val in sbflags.items() if val is not None}
        sys.exit(
            sbatch(
                f"python -u {os.path.abspath(sys.argv[0])} {' '.join(sys.argv[1:])} --local",
                flags=sbflags,
                modules=args.modules,
                dryrun=args.dryrun,
            ).returncode
        )

    # Run the script locally.
    return args


def harness(main: Callable[[Context], Any]) -> int:
    # Create a context.
    ctx = get_context()

    # Run main.
    ctx.log.debug("Prelude complete.")
    ctx.log.info("Starting main.")
    exit_status = None
    try:
        main(ctx)
    except (Exception, KeyboardInterrupt) as exc:
        ctx.log.error(exc, exc_info=True)
        exit_status = 1
    except SystemExit as exc:
        exit_status = exc.args[0] if exc.args else 0
    else:
        exit_status = 0

    # Epilogue
    ctx.log.info(f"Main complete. [exit {exit_status}]")
    ctx.log.debug("Epilogue complete.")
    return exit_status
