"""
CogniFlow - Webcam Sensor
Single camera instance shared with face mesh viewer.
Tracks EAR, MAR, yawn (4-state FSM), eye closure, and camera blocking.
Calibrates to personal baseline - no hardcoded thresholds.
"""

import cv2
import mediapipe as mp
import math
import time
import threading
import statistics
import numpy as np

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP_INNER = 13
MOUTH_BOT_INNER = 14
MOUTH_LEFT      = 78
MOUTH_RIGHT     = 308

CALIBRATION_SECONDS      = 30
EAR_CLOSED_FRACTION      = 0.65
EYE_CLOSE_HOLD_SECONDS   = 2.0
YAWN_HOLD_SECONDS        = 3.0
YAWN_MARGIN              = 0.35
YAWN_ABSOLUTE_MIN        = 0.55
YAWN_DEBOUNCE_SECONDS    = 5.0
BLOCK_BRIGHTNESS_THRESHOLD = 20
BLOCK_STDDEV_THRESHOLD   = 12
BLOCK_HOLD_SECONDS       = 2.0


class WebcamSensor:

    def __init__(self):
        self._lock    = threading.Lock()
        self._running = False

        # Public outputs
        self.ear            = 0.30
        self.mar            = 0.0
        self.yawn_count     = 0
        self.face_visible   = False
        self.eyes_closed    = False
        self.eye_close_count= 0
        self.is_blocked     = False
        self.block_count    = 0

        # Shared frame for viewer
        self._frame_lock      = threading.Lock()
        self._latest_frame    = None
        self._latest_landmarks= None
        self._frame_ready     = False

        # Calibration
        self._cal_ear_samples = []
        self._cal_mar_samples = []
        self._cal_start       = time.time()
        self._is_calibrated   = False
        self._baseline_ear    = 0.30
        self._ear_closed_thr  = 0.20
        self._baseline_mar    = 0.0
        self._yawn_threshold  = YAWN_ABSOLUTE_MIN

        # Eye close state
        self._eyes_closing       = False
        self._eyes_close_start   = 0.0
        self._last_eye_close_at  = 0.0

        # Yawn FSM state
        self._yawning       = False
        self._yawn_start    = 0.0
        self._last_yawn_at  = 0.0
        self._yawn_peak_mar = 0.0

        # Blocking state
        self._block_suspect   = False
        self._block_start     = 0.0
        self._block_confirmed = False

        self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _calc_ear(self, landmarks, indices):
        v = math.dist(
            [landmarks[indices[1]].x, landmarks[indices[1]].y],
            [landmarks[indices[5]].x, landmarks[indices[5]].y],
        )
        h = math.dist(
            [landmarks[indices[0]].x, landmarks[indices[0]].y],
            [landmarks[indices[3]].x, landmarks[indices[3]].y],
        )
        return v / h if h else 0.0

    def _calc_mar(self, landmarks):
        top   = landmarks[MOUTH_TOP_INNER]
        bottom= landmarks[MOUTH_BOT_INNER]
        left  = landmarks[MOUTH_LEFT]
        right = landmarks[MOUTH_RIGHT]
        v = math.dist([top.x, top.y],  [bottom.x, bottom.y])
        h = math.dist([left.x, left.y],[right.x,  right.y])
        return v / h if h else 0.0

    # ── Calibration ────────────────────────────────────────────────────────────

    def _calibrate(self, ear, mar):
        if self._is_calibrated:
            return
        if ear > 0.15:
            self._cal_ear_samples.append(ear)
        if 0.01 < mar < 0.25:
            self._cal_mar_samples.append(mar)
        elapsed = time.time() - self._cal_start
        if (elapsed >= CALIBRATION_SECONDS and
                len(self._cal_ear_samples) >= 10 and
                len(self._cal_mar_samples) >= 15):
            self._baseline_ear  = statistics.mean(self._cal_ear_samples)
            self._ear_closed_thr= self._baseline_ear * EAR_CLOSED_FRACTION
            sorted_mar = sorted(self._cal_mar_samples)
            p90 = sorted_mar[int(len(sorted_mar) * 0.90)]
            self._baseline_mar   = p90
            self._yawn_threshold = max(YAWN_ABSOLUTE_MIN, p90 + YAWN_MARGIN)
            self._is_calibrated  = True
            print("[Webcam] Calibrated!")
            print(f"  EAR baseline:      {self._baseline_ear:.3f}")
            print(f"  Eye-close thresh:  {self._ear_closed_thr:.3f}")
            print(f"  MAR baseline(p90): {self._baseline_mar:.3f}")
            print(f"  Yawn threshold:    {self._yawn_threshold:.3f}")

    # ── Yawn FSM (IDLE -> OPENING -> CONFIRMED -> DEBOUNCE -> IDLE) ────────────

    def _check_yawn(self, mar):
        now = time.time()
        if mar > self._yawn_threshold:
            if not self._yawning:
                self._yawning       = True
                self._yawn_start    = now
                self._yawn_peak_mar = mar
            else:
                self._yawn_peak_mar = max(self._yawn_peak_mar, mar)
                held = now - self._yawn_start
                if held >= YAWN_HOLD_SECONDS:
                    if now - self._last_yawn_at >= YAWN_DEBOUNCE_SECONDS:
                        if self._yawn_peak_mar >= self._yawn_threshold * 1.1:
                            with self._lock:
                                self.yawn_count += 1
                            self._last_yawn_at  = now
                            self._yawning       = False
                            self._yawn_peak_mar = 0.0
                            print(f"[Webcam] Yawn #{self.yawn_count} confirmed")
        else:
            self._yawning       = False
            self._yawn_peak_mar = 0.0

    # ── Eye close detection ────────────────────────────────────────────────────

    def _check_eye_close(self, ear):
        now = time.time()
        if ear < self._ear_closed_thr:
            if not self._eyes_closing:
                self._eyes_closing      = True
                self._eyes_close_start  = now
            else:
                held = now - self._eyes_close_start
                if held >= EYE_CLOSE_HOLD_SECONDS:
                    if now - self._last_eye_close_at > (EYE_CLOSE_HOLD_SECONDS + 1.0):
                        with self._lock:
                            self.eyes_closed      = True
                            self.eye_close_count += 1
                        self._last_eye_close_at = now
                        print(f"[Webcam] Eyes closed #{self.eye_close_count} "
                              f"EAR={ear:.3f}")
        else:
            self._eyes_closing = False
            with self._lock:
                self.eyes_closed = False

    # ── Blocking detection ─────────────────────────────────────────────────────

    def _check_blocking(self, frame, face_detected):
        now = time.time()
        if face_detected:
            self._block_suspect   = False
            self._block_confirmed = False
            with self._lock:
                self.is_blocked = False
            return
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        stddev     = float(np.std(gray))
        is_suspect = (brightness < BLOCK_BRIGHTNESS_THRESHOLD or
                      stddev < BLOCK_STDDEV_THRESHOLD)
        if is_suspect:
            if not self._block_suspect:
                self._block_suspect = True
                self._block_start   = now
            elif (now - self._block_start >= BLOCK_HOLD_SECONDS and
                  not self._block_confirmed):
                self._block_confirmed = True
                with self._lock:
                    self.is_blocked  = True
                    self.block_count += 1
                print(f"[Webcam] Camera blocked "
                      f"brightness={brightness:.1f} std={stddev:.1f}")
        else:
            if self._block_confirmed:
                print("[Webcam] Camera unblocked.")
            self._block_suspect   = False
            self._block_confirmed = False
            with self._lock:
                self.is_blocked = False

    # ── Camera loop ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Webcam] ERROR: Could not open camera.")
            return
        print("[Webcam] Camera started.")
        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results  = self._mp_mesh.process(rgb)
            face_det = bool(results.multi_face_landmarks)

            with self._frame_lock:
                self._latest_frame     = frame.copy()
                self._latest_landmarks = (
                    results.multi_face_landmarks[0].landmark
                    if face_det else None
                )
                self._frame_ready = True

            self._check_blocking(frame, face_det)

            if face_det and not self.is_blocked:
                lm         = results.multi_face_landmarks[0].landmark
                left_ear   = self._calc_ear(lm, LEFT_EYE)
                right_ear  = self._calc_ear(lm, RIGHT_EYE)
                avg_ear    = (left_ear + right_ear) / 2.0
                mar_val    = self._calc_mar(lm)
                with self._lock:
                    self.ear          = avg_ear
                    self.mar          = mar_val
                    self.face_visible = True
                self._calibrate(avg_ear, mar_val)
                if self._is_calibrated:
                    self._check_eye_close(avg_ear)
                    self._check_yawn(mar_val)
            else:
                with self._lock:
                    self.face_visible = False
                    self.eyes_closed  = False
        cap.release()

    # ── Frame getter for viewer ────────────────────────────────────────────────

    def get_latest_frame(self):
        with self._frame_lock:
            if not self._frame_ready:
                return None, None
            return self._latest_frame.copy(), self._latest_landmarks

    # ── Public getters ─────────────────────────────────────────────────────────

    def get_ear(self):             
        with self._lock: return self.ear
    def get_mar(self):             
        with self._lock: return self.mar
    def get_yawn_count(self):      
        with self._lock: return self.yawn_count
    def get_eye_close_count(self): 
        with self._lock: return self.eye_close_count
    def get_block_count(self):     
        with self._lock: return self.block_count
    def is_eyes_closed(self):      
        with self._lock: return self.eyes_closed
    def is_cam_blocked(self):      
        with self._lock: return self.is_blocked
    def is_face_visible(self):     
        with self._lock: return self.face_visible
    def is_calibrated(self):       
        return self._is_calibrated

    def reset_yawn_count(self):
        with self._lock:
            c = self.yawn_count
            self.yawn_count = 0
            return c

    def reset_eye_close_count(self):
        with self._lock:
            c = self.eye_close_count
            self.eye_close_count = 0
            return c

    def get_thresholds(self):
        return {
            "ear_baseline":   round(self._baseline_ear, 3),
            "ear_closed":     round(self._ear_closed_thr, 3),
            "mar_baseline":   round(self._baseline_mar, 3),
            "yawn_threshold": round(self._yawn_threshold, 3),
        }
