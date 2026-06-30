"""Robust loading of two-column Raman spectra (Raman shift, intensity)."""
import io as _io
import numpy as np


def _read_text(source):
    """Return the file contents as text from a path or a file-like object."""
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        return raw
    with open(source, "r", encoding="utf-8-sig") as f:
        return f.read()


def choose_spectrum_file(default=None, initialdir="data", title="Select a spectrum file"):
    """Open a native file-picker dialog and return the chosen path.

    Uses ``tkinter`` (always available with CPython, works in Jupyter / VS Code
    notebooks).  Returns the selected path; if the user cancels, or no GUI is
    available (e.g. a headless run), falls back to ``default`` so notebooks stay
    runnable non-interactively.  Set ``RAMANFIT_NO_DIALOG=1`` to skip the dialog
    entirely (used when re-executing the notebook with nbconvert).
    """
    import os
    if os.environ.get("RAMANFIT_NO_DIALOG"):
        return default
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:                       # no Tk -> headless; use the default
        return default
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # bring the dialog to the front
        path = filedialog.askopenfilename(
            title=title, initialdir=initialdir,
            filetypes=[("Spectra", "*.tsv *.csv *.txt *.dat"), ("All files", "*.*")])
        root.destroy()
    except Exception:                       # display not available
        return default
    return path or default                  # "" on cancel -> default


def load_spectrum(source):
    """Load a two-column spectrum from a path or file-like object.

    Handles tab- or comma-delimited data, Windows (CRLF) line endings and a
    UTF-8 BOM.  Returns ``(x, y)`` as float arrays sorted by ascending Raman
    shift.
    """
    text = _read_text(source).lstrip("﻿")
    data = None
    last_err = None
    for delim in ("\t", ",", None):  # None -> any whitespace
        try:
            data = np.loadtxt(_io.StringIO(text), delimiter=delim)
        except Exception as e:  # noqa: BLE001 - try the next delimiter
            last_err = e
            continue
        if data.ndim == 2 and data.shape[1] >= 2:
            break
        data = None
    if data is None:
        raise ValueError(f"Could not parse two-column data: {last_err}")

    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]
