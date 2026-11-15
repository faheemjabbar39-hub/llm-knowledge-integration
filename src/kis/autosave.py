"""Save results to GitHub automatically while a long run is going.

Colab and Kaggle wipe the machine when a session disconnects, so anything not on GitHub is
lost. When the notebook's Setup A has stored a GitHub token in the GITHUB_TOKEN environment
variable, the run scripts call `autosave()` every few dozen questions and at the end. Without
a token it does nothing, so local runs and tests are unaffected.

An autosave never stops the experiment: if pushing fails, it prints a warning and the run
carries on (the next autosave tries again).
"""
from __future__ import annotations

import os
import subprocess
import time

from kis.common import ROOT

REPO = os.environ.get("KIS_REPO", "github.com/faheemjabbar39-hub/llm-knowledge-integration.git")
_last = 0.0


def _git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=timeout)


def autosave(message: str, min_interval_s: float = 0) -> bool:
    """Commit results/ and adapters/ and push them. Returns True if GitHub is up to date."""
    global _last
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token or (time.time() - _last) < min_interval_s:
        return False
    _last = time.time()
    url = f"https://{token}@{REPO}"
    try:
        paths = [p for p in ("results", "adapters") if (ROOT / p).exists()]
        _git("add", *paths)
        if _git("diff", "--cached", "--quiet").returncode != 0:
            _git("-c", "user.name=Faheem Jabbar", "-c", "user.email=faheemjabbar39@gmail.com",
                 "commit", "-q", "-m", f"Autosave: {message}")
        _git("pull", "-q", "--rebase", "--autostash", "-X", "theirs", url, "main")
        push = _git("push", "-q", url, "HEAD:main")
        if push.returncode != 0:
            raise RuntimeError(push.stderr.replace(token, "***").strip()[-300:])
        print(f"  [autosave] saved to GitHub: {message}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001  (never stop the experiment because of a save problem)
        print(f"  [autosave] could not save to GitHub ({str(e).replace(token, '***')}); "
              "the run continues and will try again.", flush=True)
        return False
