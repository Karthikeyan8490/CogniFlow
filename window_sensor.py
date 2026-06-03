"""
CogniFlow — Window Sensor
Detects the active foreground application and maps it to a focus context score.
"""
import psutil
import ctypes

CONTEXT_SCORES = {
    "code": 100, "terminal": 95, "document": 80,
    "browser_work": 60, "email": 55, "media": 30,
    "social": 20, "idle": 10
}

def get_active_window():
    """Returns (app_name, section) of the foreground window."""
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = psutil.Process(pid.value)
        return proc.name(), ""
    except Exception:
        return "unknown", ""

def get_activity_label(app, section):
    """Maps app name to activity category."""
    app = app.lower()
    if any(k in app for k in ["code", "pycharm", "vim", "nvim"]):
        return "coding"
    if any(k in app for k in ["chrome", "firefox", "edge", "brave"]):
        return "browsing"
    if "explorer" in app or app == "":
        return "idle"
    return "other"

