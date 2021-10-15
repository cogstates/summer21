# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from typing import Dict, Iterable, Optional

from .functional import safe_iter


_FLAGS = {
    "job-name": os.path.basename(sys.argv[0]),
    "output": "/gpfs/scratch/%u/%x.%j.out",
    "partition": "medium-28core",
}


def sbatch(
    cmds: Iterable[str],
    flags: Optional[Dict[str, str]] = None,
    modules: Optional[Iterable[str]] = None,
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
    print("\n".join(stdin))
    # Submit the job.
    return subprocess.run(
        ["sbatch"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        input="\n".join(stdin).encode("utf-8"),
        check=True,
    )
