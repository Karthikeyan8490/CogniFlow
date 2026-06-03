"""
CogniFlow - Entry Point
Starts all sensors and launches the focus engine.
Run: python main.py
"""
import threading
import signal
import sys
from engine import CogniFlowEngine

def main():
    print("=" * 45)
    print("   CogniFlow - Cognitive Focus Monitor")
    print("   Press Ctrl+C to stop")
    print("=" * 45)

    shutdown_event = threading.Event()

    def handle_exit(sig, frame):
        print("\n[CogniFlow] Shutting down...")
        shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    engine = CogniFlowEngine(shutdown_event)

    # Try to launch tray (optional - works without it)
    try:
        from tray import TrayApp
        tray = TrayApp(engine, shutdown_event)
        engine_thread = threading.Thread(target=engine.run, daemon=True)
        engine_thread.start()
        tray.run()
    except Exception:
        # Run in console-only mode if tray unavailable
        print("[CogniFlow] Running in console mode (no tray)")
        engine.run()

    # Print session summary on exit
    summary = engine.summary()
    if summary:
        print("\n--- Session Summary ---")
        print(f"  Duration : {summary['duration_min']} min")
        print(f"  Average  : {summary['average']}/100")
        print(f"  Peak     : {summary['peak_score']} at {summary['peak_time']}")
        print(f"  Low      : {summary['low_score']} at {summary['low_time']}")
        print(f"  Dominant : {summary['dominant']}")

if __name__ == "__main__":
    main()
