"""
CogniFlow — Keyboard Sensor
Measures typing flight-time variance as a proxy for cognitive engagement.
"""
import threading
import time
import statistics
from pynput import keyboard

class KeyboardSensor:
    def __init__(self):
        self._lock = threading.Lock()
        self._press_times = {}
        self._flight_times = []
        self._last_release = None
        self._state = "idle"

    def start(self):
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        listener.daemon = True
        listener.start()

    def _on_press(self, key):
        now = time.time()
        with self._lock:
            if self._last_release:
                ft = now - self._last_release
                if 0.01 < ft < 2.0:
                    self._flight_times.append(ft)
                    if len(self._flight_times) > 50:
                        self._flight_times.pop(0)
            self._state = "typing"

    def _on_release(self, key):
        with self._lock:
            self._last_release = time.time()

    def get_variance(self):
        with self._lock:
            if len(self._flight_times) < 3:
                return 0.0
            return statistics.variance(self._flight_times)

    def get_state(self):
        with self._lock:
            return self._state

