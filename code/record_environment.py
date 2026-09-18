"""Record the computing environment, so the methods text can cite a results file.

The hardware is not incidental here. Every training run in both papers fit inside
one laptop GPU's memory, and that ceiling is what sets the top rung of the ladder;
a reader deciding whether the scaling range is a choice or a constraint needs the
number. Typing it into the prose would leave it to go stale silently the first
time the work moves to another machine, so it is collected here and reaches the
manuscript through macros like every other number.
"""
import os, json, sys, time, platform

import numpy as np
import scipy
import torch
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def host_memory_gb():
    """Total physical RAM, or None if the platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, AttributeError):
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 2 ** 30
    return None


def main():
    out = {}

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu_name"] = p.name
        out["gpu_memory_gib"] = round(p.total_memory / 2 ** 30, 1)
        out["gpu_count"] = torch.cuda.device_count()
        out["gpu_capability"] = f"{p.major}.{p.minor}"
    else:
        # Recorded rather than guessed: a CPU-only host would make the timings
        # in the methods meaningless, and the gate below should say so loudly.
        out["gpu_name"] = None
        out["gpu_memory_gib"] = None
        out["gpu_count"] = 0
        out["gpu_capability"] = None

    mem = host_memory_gb()
    out["host_memory_gb"] = round(mem, 1) if mem is not None else None
    out["os"] = f"{platform.system()} {platform.release()}"
    out["os_build"] = platform.version()
    out["cpu_count"] = os.cpu_count()
    out["python"] = platform.python_version()
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["numpy"] = np.__version__
    out["scipy"] = scipy.__version__
    out["python_chess"] = chess.__version__

    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(out, open(os.path.join(RES, "environment.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:18s} {v}")

    if out["gpu_count"] == 0:
        print("\n  WARNING: no GPU visible; this is not the environment the runs used.")


if __name__ == "__main__":
    main()
