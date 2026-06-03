"""
CogniFlow — Mouse Sensor
Tracks mouse movement and click activity as a supplementary focus signal.
"""
import threading
import time
from pynput import mouse

class MouseSensor:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._last_move = time.time()
        self._listener = None

    def start(self):
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click
        )
        self._listener.start()

    def _on_move(self, x, y):
        with self._lock:
            self._active = True
            self._last_move = time.time()

    def _on_click(self, x, y, button, pressed):
        with self._lock:
            self._active = True

    def is_active(self):
        with self._lock:
            return (time.time() - self._last_move) < 3.0

    def stop(self):
        if self._listener:
            self._listener.stop()

