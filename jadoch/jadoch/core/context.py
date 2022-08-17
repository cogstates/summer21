# -*- coding: utf-8 -*-
import getpass
import inspect
import logging.handlers
import os
import sys
import time
from argparse import ArgumentParser
from types import ModuleType
from typing import Any, Optional


# TODO: This belongs somewhere else?
def module_from_path(path: str) -> ModuleType:
    from importlib.abc import Loader
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location(os.path.basename(path).replace(".py", ""), path)
    assert spec
    mod = module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(mod)
    return mod


# TODO: Not sure about this whole idea.
class Context:
    def __init__(self) -> None:
        # Initialize argparser.
        self.parser = ArgumentParser(description=module_from_path(sys.argv[0]).__doc__)
        self.parser.add_argument(
            "-v", "--verbose", action="store_true", help="turn on verbose logging"
        )
        if any(flg in sys.argv for flg in ("-h", "--help")):
            return  # Don't intialize logging if we are just printing help.
        # Intialize logging.
        scriptname = os.path.basename(sys.argv[0]).replace(".py", "")
        logpath = time.strftime(
            #  f"/gpfs/scratch/{getpass.getuser()}/logs/{scriptname}.%Y%m%d.%H%M%S.log"
            f"/home/{getpass.getuser()}/scratch/logs/{scriptname}.%Y%m%d.%H%M%S.log"
        )
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        logging.basicConfig(
            format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(logpath)],
        )
        self.log.info("Initialized logging: %s", logpath)
        # Handle default arguments.
        if any(flg in sys.argv for flg in ("-v", "--verbose")):
            logging.getLogger().setLevel(logging.DEBUG)

    @property
    def log(self) -> logging.Logger:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        return logging.getLogger(module.__name__ if module else frame.filename)


_CONTEXT = None


def get_context() -> Context:
    global _CONTEXT
    if not _CONTEXT:
        _CONTEXT = Context()
    return _CONTEXT
