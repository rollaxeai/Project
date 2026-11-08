"""Пиковая память (commit) одного куска генератора."""
import ctypes
from ctypes import wintypes
from dataclasses import asdict

from amc.generator import CLASSES, GenConfig, _chunk


class PMC(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS (WinAPI)."""

    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                       "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def peak() -> str:
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return f"peak WS {c.PeakWorkingSetSize / 2**20:.0f} MB, peak commit {c.PeakPagefileUsage / 2**20:.0f} MB"


print("после импорта:", peak())
for k in range(3):
    _chunk((0, CLASSES * 46, 1, k, asdict(GenConfig())))
    print("после куска", k, peak())
