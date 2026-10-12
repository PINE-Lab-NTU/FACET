"""Redirect ``lerobot.common`` imports to ``lerobot`` so openpi runs on LeRobot >= 0.3."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_OLD, _NEW = "lerobot.common", "lerobot"


class _Redirect(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD and not fullname.startswith(_OLD + "."):
            return None
        try:
            module = importlib.import_module(_NEW + fullname[len(_OLD) :])
        except Exception:
            return None
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return sys.modules[spec.name]

    def exec_module(self, module):
        pass


def install_lerobot_common_redirect() -> bool:
    if any(isinstance(f, _Redirect) for f in sys.meta_path):
        return True
    if importlib.util.find_spec("lerobot") is None:
        return False
    try:
        importlib.import_module(_OLD)
        return True
    except Exception:
        pass
    if importlib.util.find_spec("lerobot.datasets") is None:
        return False
    sys.meta_path.insert(0, _Redirect())
    return True

