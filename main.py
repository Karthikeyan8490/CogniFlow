"""
CogniFlow — Entry Point
Starts all sensors and launches the focus engine.
"""
import threading
from engine import CogniFlowEngine
from tray import TrayApp

if __name__ == "__main__":
    shutdown_event = threading.Event()
    engine = CogniFlowEngine(shutdown_event)
    tray = TrayApp(engine, shutdown_event)

    engine_thread = threading.Thread(target=engine.run, daemon=True)
    engine_thread.start()
    tray.run()

