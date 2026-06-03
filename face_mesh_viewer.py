"""
CogniFlow — Face Mesh Viewer
Optional floating window showing live webcam feed with MediaPipe FaceMesh overlay.
Eye landmarks are highlighted green (open) or red (closed).
EAR and MAR values annotated in real time.
"""
# Implementation: Uses webcam_sensor shared frame buffer
# See webcam_sensor.py -> get_latest_frame()

