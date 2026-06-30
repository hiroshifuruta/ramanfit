"""Export a notebook to PDF from within the notebook itself.

Call :func:`export_pdf` from the last cell so a fresh PDF is produced every time
the notebook is run top-to-bottom.  PDF export goes through nbconvert ->
LaTeX, which needs both ``pandoc`` and a TeX engine (``xelatex``); this helper
locates them on common Windows install paths if they are not already on PATH,
and falls back to an HTML export if the PDF toolchain is unavailable.
"""
import os
import shutil
import subprocess
import sys

# Common per-user install locations not always on PATH (Windows).
_EXTRA_DIRS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Pandoc"),
    r"E:\Apps\texlive\2025\bin\windows",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"),
]


def _augmented_env():
    """Return an environment whose PATH includes the extra tool directories."""
    env = dict(os.environ)
    extra = [d for d in _EXTRA_DIRS if os.path.isdir(d)]
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _have(tool, env):
    """True if ``tool`` is resolvable on the (augmented) PATH."""
    return shutil.which(tool, path=env.get("PATH")) is not None


def export_pdf(notebook_path, to="pdf"):
    """Convert ``notebook_path`` to PDF (or HTML fallback) via nbconvert.

    Returns the path of the file produced.  ``to="html"`` forces HTML; the
    default ``"pdf"`` falls back to HTML automatically if pandoc/xelatex are
    missing so the call never hard-fails inside a notebook run.
    """
    env = _augmented_env()
    if to == "pdf" and not (_have("pandoc", env) and _have("xelatex", env)):
        print("export_pdf: pandoc/xelatex not found -> exporting HTML instead")
        to = "html"

    cmd = [sys.executable, "-m", "nbconvert", "--to", to, notebook_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = os.path.splitext(notebook_path)[0] + (".pdf" if to == "pdf" else ".html")

    if proc.returncode != 0 or not os.path.exists(out):
        if to == "pdf":                       # retry as HTML on any PDF failure
            print("export_pdf: PDF export failed -> exporting HTML instead")
            return export_pdf(notebook_path, to="html")
        raise RuntimeError(f"nbconvert failed:\n{proc.stderr[-2000:]}")

    print(f"export_pdf: wrote {out}")
    return out
