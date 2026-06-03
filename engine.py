"""
CogniFlow - Main Engine
Orchestrates all sensors and calculates focus score every 2 seconds.
Clean shutdown via shared threading.Event.
"""

import time
import statistics
import csv
import threading
import os
import queue

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from webcam_sensor import WebcamSensor
from keyboard_sensor import KeyboardSensor
from mouse_sensor import MouseSensor
from window_sensor import get_active_window, get_activity_label

API_URL = "http://localhost:8000/api/telemetry"

WEIGHTS = {
    "ear":    0.35,
    "typing": 0.25,
    "state":  0.20,
    "yawn":   0.15,
    "mouse":  0.05,
}

CALIBRATION_SECONDS  = 30
EAR_FALLBACK         = 0.30
SMOOTH_WINDOW        = 8
ALERT_COOLDOWN       = 60
LOW_SCORE_THRESHOLD  = 40
YAWN_ALERT_COUNT     = 3
EYE_CLOSE_ALERT      = 3
CSV_FILE             = "cogniflow_log.csv"


class CogniFlowEngine:

    def __init__(self, shutdown_event):
        self.shutdown_event = shutdown_event
        self.webcam   = WebcamSensor()
        self.keyboard = KeyboardSensor()
        self.mouse    = MouseSensor()

        self._cal_samples  = []
        self._cal_start    = time.time()
        self.is_calibrated = False
        self.baseline_ear  = EAR_FALLBACK
        self.baseline_var  = 0.01

        self._score_history  = []
        self.current_score   = 100
        self.current_label   = "Calibrating..."
        self.current_color   = "#90A4AE"
        self.current_state   = "thinking"
        self.current_app     = ""
        self.current_section = ""
        self.current_activity= ""

        self._last_alert_at  = 0
        self.alert_message   = None
        self.total_yawns     = 0
        self.total_eye_closes= 0
        self.session_log     = []

        self._setup_csv()

        # ML Integration
        self.rf_model    = None
        self.le_state    = None
        self.le_activity = None
        self.ml_label    = "N/A"
        self._load_ml_models()

        # Cloud Telemetry (silent - fails gracefully if no server)
        self.telemetry_queue = queue.Queue()
        threading.Thread(target=self._telemetry_worker, daemon=True).start()

    # ── ML ─────────────────────────────────────────────────────────────────────

    def _load_ml_models(self):
        if not JOBLIB_AVAILABLE:
            print("[Engine] joblib not available - running rule-based only.")
            return
        try:
            if (os.path.exists('models/cogniflow_rf_model.pkl') and
                    os.path.exists('models/le_state.pkl') and
                    os.path.exists('models/le_activity.pkl')):
                self.rf_model    = joblib.load('models/cogniflow_rf_model.pkl')
                self.le_state    = joblib.load('models/le_state.pkl')
                self.le_activity = joblib.load('models/le_activity.pkl')
                print("[Engine] ML Validation Layer loaded.")
            else:
                print("[Engine] ML models not found - running rule-based only.")
        except Exception as e:
            print(f"[Engine] Error loading ML models: {e}")

    # ── Telemetry ──────────────────────────────────────────────────────────────

    def _stream_telemetry(self, score, label, state, ear,
                          variance, yawns, eye_closes, activity, ml_label):
        if not REQUESTS_AVAILABLE:
            return
        payload = {
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "score":      score,
            "rule_label": label,
            "ml_label":   ml_label,
            "state":      state,
            "ear":        round(ear, 3),
            "variance":   round(variance, 4),
            "yawns":      yawns,
            "eye_closes": eye_closes,
            "activity":   activity,
        }
        self.telemetry_queue.put(payload)

    def _telemetry_worker(self):
        while not self.shutdown_event.is_set():
            try:
                payload = self.telemetry_queue.get(timeout=1.0)
                try:
                    if REQUESTS_AVAILABLE:
                        requests.post(API_URL, json=payload, timeout=2.0)
                except Exception:
                    pass
                finally:
                    self.telemetry_queue.task_done()
            except queue.Empty:
                continue

    # ── CSV ────────────────────────────────────────────────────────────────────

    def _setup_csv(self):
        try:
            with open(CSV_FILE, 'x', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["Time", "Score", "Label", "State",
                             "EAR", "Variance", "Yawns", "EyeCloses",
                             "Activity", "ML_Label", "Type"])
        except FileExistsError:
            pass
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "", "", "", "", "", "", "", "", "", "SESSION_START"
            ])

    def _log_csv(self, score, label, state, ear, variance,
                 yawns, eye_closes, activity, ml_label):
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                score, label, state,
                round(ear, 3), round(variance, 4),
                yawns, eye_closes, activity, ml_label, "DATA"
            ])

    # ── Calibration ────────────────────────────────────────────────────────────

    def _calibrate(self, ear, variance):
        if self.is_calibrated:
            return
        if ear > 0:
            self._cal_samples.append({"ear": ear, "var": variance})
        elapsed = time.time() - self._cal_start
        if elapsed >= CALIBRATION_SECONDS and len(self._cal_samples) >= 5:
            ears  = [s["ear"] for s in self._cal_samples]
            vars_ = [s["var"] for s in self._cal_samples if s["var"] > 0]
            self.baseline_ear = statistics.mean(ears)
            self.baseline_var = statistics.mean(vars_) if vars_ else 0.01
            self.is_calibrated = True
            print(f"[Engine] Calibrated | EAR={self.baseline_ear:.3f} "
                  f"VAR={self.baseline_var:.4f}")

    def calibration_progress(self):
        elapsed = time.time() - self._cal_start
        return min(100, int((elapsed / CALIBRATION_SECONDS) * 100))

    # ── Scorers ────────────────────────────────────────────────────────────────

    def _score_ear(self, ear):
        if ear <= 0:
            return 50
        baseline   = self.baseline_ear
        ear_closed = baseline * 0.65
        if ear >= baseline:
            return 100
        if ear <= ear_closed:
            return 0
        return max(0, min(100, int(
            ((ear - ear_closed) / (baseline - ear_closed)) * 100
        )))

    def _score_typing(self, variance, state):
        if state in ("reading", "thinking"):
            return 60
        if variance <= 0:
            return 50
        ratio = variance / self.baseline_var
        if ratio < 0.5:  return 65
        if ratio <= 1.5: return 100
        if ratio <= 3.0: return max(30, int(100 - ((ratio - 1.5) / 1.5) * 35))
        return max(10, int(65 - ((ratio - 3.0) * 10)))

    def _score_state(self, state):
        return {"typing": 95, "reading": 70, "thinking": 40}.get(state, 50)

    def _score_yawn(self, yawn_count):
        return max(0, 100 - (yawn_count * 20))

    def _score_mouse(self, active):
        return 80 if active else 40

    # ── Score calculation ──────────────────────────────────────────────────────

    def _calculate(self, ear, variance, yawn_count, state, mouse_active):
        raw_scores = {
            "ear":    self._score_ear(ear),
            "typing": self._score_typing(variance, state),
            "state":  self._score_state(state),
            "yawn":   self._score_yawn(yawn_count),
            "mouse":  self._score_mouse(mouse_active),
        }
        raw = sum(raw_scores[k] * WEIGHTS[k] for k in raw_scores)
        self._score_history.append(raw)
        if len(self._score_history) > SMOOTH_WINDOW:
            self._score_history = self._score_history[-SMOOTH_WINDOW:]
        smoothed = max(0, min(100, int(statistics.mean(self._score_history))))
        return smoothed, self._label(smoothed), self._color(smoothed), raw_scores

    def _label(self, s):
        if s >= 85: return "Deep Focus"
        if s >= 70: return "Focused"
        if s >= 55: return "Moderate"
        if s >= 40: return "Drifting"
        if s >= 25: return "Fatigued"
        return "Exhausted"

    def _color(self, s):
        if s >= 75: return "#00C853"
        if s >= 50: return "#FFD600"
        if s >= 30: return "#FF6D00"
        return "#D50000"

    # ── Alerts ─────────────────────────────────────────────────────────────────

    def _check_alerts(self, score, yawn_count, eye_close_count,
                      eyes_closed_now, cam_blocked):
        now = time.time()
        if cam_blocked:
            if now - self._last_alert_at > 10:
                self.alert_message = (
                    "Webcam is blocked!\n"
                    "Something is covering the camera.\n"
                    "Focus tracking has been paused."
                )
                self._last_alert_at = now
            return
        if now - self._last_alert_at < ALERT_COOLDOWN:
            return
        if eyes_closed_now:
            self.alert_message = (
                "Eyes closed detected!\n"
                "Are you falling asleep? Take a break!"
            )
            self._last_alert_at = now
            return
        if eye_close_count >= EYE_CLOSE_ALERT:
            self.alert_message = (
                f"Eyes closed {eye_close_count} times this session.\n"
                f"You're showing signs of fatigue. Rest up!"
            )
            self._last_alert_at = now
            self.webcam.reset_eye_close_count()
            self.total_eye_closes = 0
            return
        if yawn_count >= YAWN_ALERT_COUNT:
            self.alert_message = (
                f"Yawned {yawn_count} times this session.\n"
                f"Time for a short break!"
            )
            self._last_alert_at = now
            self.webcam.reset_yawn_count()
            self.total_yawns = 0
            return
        if score < LOW_SCORE_THRESHOLD:
            self.alert_message = (
                f"Focus Score: {score}/100 ({self._label(score)})\n"
                f"You seem fatigued. Consider a short break!"
            )
            self._last_alert_at = now

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self):
        self.webcam.start()
        self.keyboard.start()
        self.mouse.start()
        print("[Engine] All sensors started. Calibrating for 30 seconds...")
        print("[Engine] Sit normally and look at the screen during calibration.")

        while not self.shutdown_event.is_set():
            ear         = self.webcam.get_ear()
            variance    = self.keyboard.get_variance()
            state       = self.keyboard.get_state()
            active      = self.mouse.is_active()
            app, section= get_active_window()
            activity    = get_activity_label(app, section)
            yawns       = self.webcam.get_yawn_count()
            eye_closes  = self.webcam.get_eye_close_count()
            eyes_closed = self.webcam.is_eyes_closed()
            cam_blocked = self.webcam.is_cam_blocked()

            self.total_yawns      = yawns
            self.total_eye_closes = eye_closes

            self._calibrate(ear, variance)

            if not self.is_calibrated:
                prog = self.calibration_progress()
                self.current_label = f"Calibrating {prog}%"
                print(f"  Calibrating... {prog}%  EAR:{ear:.3f}")
                time.sleep(2)
                continue

            if cam_blocked:
                self.current_label = "Cam Blocked"
                self.current_color = "#B71C1C"
                self._check_alerts(0, yawns, eye_closes, False, True)
                print("  Camera blocked - tracking paused.")
                time.sleep(2)
                continue

            score, label, color, breakdown = self._calculate(
                ear, variance, yawns, state, active)

            # ML Prediction
            if (self.rf_model and self.le_state and self.le_activity):
                try:
                    s_idx = (self.le_state.transform([state])[0]
                             if state in self.le_state.classes_ else 0)
                    a_idx = (self.le_activity.transform([activity])[0]
                             if activity in self.le_activity.classes_ else 0)
                    features = [[ear, yawns, variance, a_idx, s_idx, eye_closes]]
                    self.ml_label = self.rf_model.predict(features)[0]
                except Exception:
                    self.ml_label = "Error"
            else:
                self.ml_label = "N/A"

            self.current_score    = score
            self.current_label    = label
            self.current_color    = color
            self.current_state    = state
            self.current_app      = app
            self.current_section  = section
            self.current_activity = activity

            self._check_alerts(score, yawns, eye_closes, eyes_closed, cam_blocked)
            self._log_csv(score, label, state, ear, variance,
                          yawns, eye_closes, activity, self.ml_label)
            self._stream_telemetry(score, label, state, ear, variance,
                                   yawns, eye_closes, activity, self.ml_label)

            self.session_log.append({
                "time":  time.strftime("%H:%M:%S"),
                "score": score,
                "state": state,
                "ear":   round(ear, 3),
            })

            bar = "\u2588" * int(score / 5) + "\u2591" * (20 - int(score / 5))
            eyes_status = "CLOSED" if eyes_closed else f"{ear:.3f}"
            print(f"  [{bar}] {score:3d}  Rule:{label:<12} ML:{self.ml_label:<12} "
                  f"EAR:{eyes_status}  Yawns:{yawns}  {activity}")

            time.sleep(2)

        print("[Engine] Stopped.")

    # ── Session summary ────────────────────────────────────────────────────────

    def summary(self):
        if not self.session_log:
            return None
        scores   = [e["score"] for e in self.session_log]
        peak     = max(self.session_log, key=lambda x: x["score"])
        low      = min(self.session_log, key=lambda x: x["score"])
        duration = round((time.time() - self._cal_start) / 60, 1)
        states   = [e["state"] for e in self.session_log]
        dominant = max(set(states), key=states.count)
        return {
            "duration_min": duration,
            "average":      round(statistics.mean(scores)),
            "peak_score":   peak["score"],
            "peak_time":    peak["time"],
            "low_score":    low["score"],
            "low_time":     low["time"],
            "dominant":     dominant,
        }
