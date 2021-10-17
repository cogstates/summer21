# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import sys
from typing import Dict, Iterable, Optional

from .functional import safe_iter


_FLAGS = {
    "job-name": os.path.basename(sys.argv[0]),
    "output": "/gpfs/scratch/%u/%x.%j.out",
    "partition": "medium-28core",
    "time": "08:00:00",
}


def sbatch(
    cmds: Iterable[str],
    flags: Optional[Dict[str, str]] = None,
    modules: Optional[Iterable[str]] = None,
    dryrun: bool = False,
) -> "subprocess.CompletedProcess[bytes]":
    # Parse inputs.
    cmds = safe_iter(cmds)
    flags = flags or _FLAGS
    for key in _FLAGS:
        if key not in flags:
            flags[key] = _FLAGS[key]
    modules = modules or []
    # Prepare batch script.
    stdin = ["#!/bin/bash"]
    for key, val in flags.items():
        stdin.append(f"#SBATCH --{key}={val}")
    if modules:
        stdin.append("")
    for module in modules:
        stdin.append(f"module load {module}")
    stdin.append("")
    stdin += cmds
    log = logging.getLogger(__name__)
    # Handle dryruns.
    if dryrun:
        stdin.append("=" * 32 + " DRYRUN " + "=" * 32)
        stdin.insert(0, "=" * 32 + " DRYRUN " + "=" * 32)
        log.info("Would submit to slurm.\n%s", "\n".join(stdin))
        return subprocess.CompletedProcess("", 0)
    log.info("Submitting to slurm.\n%s", "\n".join(stdin))
    # Submit the job.
    return subprocess.run(
        ["sbatch"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        input="\n".join(stdin).encode("utf-8"),
        check=True,
    )
