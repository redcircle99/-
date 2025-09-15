# ─────────────────────────────────────────────────────────────
# [추가] UI를 마음대로 바꾸는 두 가지 방법
#   A) Qt Designer(.ui) 방식: UI는 디자이너에서 자유롭게 편집, 파이썬은 로직만 담당
#   B) JSON 선언형 방식: ui_config.json으로 컨트롤을 선언하면, 앱이 자동으로 폼을 생성
# 아래에 두 방식 모두 “바로 실행” 가능한 최소 예시를 제공한다.
# ─────────────────────────────────────────────────────────────

"""
A) Qt Designer(.ui) + 동적 바인딩 로더
--------------------------------------
장점
 - UI 배치는 전부 디자이너에서 드래그&드롭으로 수정
 - 파이썬은 objectName 기준으로 컨트롤을 찾아 로직만 연결
사용법
 1) 아래 viewer.ui 내용을 파일로 저장(또는 Qt Designer로 새로 만들기)
 2) 아래 viewer_loader.py를 실행 → .ui를 로드하여 동작
 3) UI 바꾸고 싶을 때는 viewer.ui만 수정하면 됨(파이썬은 그대로)
"""

from __future__ import annotations
import sys, time, threading
from typing import Optional
import cv2, numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, uic

class CameraBackendUIOpenCV:
    def __init__(self, source=0):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = source
        self._lock = threading.Lock()
        self._latest = None
        self._running = False

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def close(self):
        self._running = False
        if self.cap:
            self.cap.release(); self.cap = None

    def start(self):
        if not self.cap or self._running: return
        self._running = True
        def loop():
            while self._running and self.cap:
                ok, f = self.cap.read()
                if ok:
                    with self._lock: self._latest = f
                else:
                    time.sleep(0.01)
        threading.Thread(target=loop, daemon=True).start()

    def get(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("viewer.ui", self)  # Qt Designer에서 만든 .ui 파일 로드

        # ── objectName으로 위젯 찾기 (Designer에서 마음대로 배치/이름 변경 가능) ──
        self.btnOpen: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnOpen")
        self.btnClose: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnClose")
        self.comboStage: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "comboStage")
        self.lblView: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "lblView")

        # 파라미터 위젯(있으면 바인딩, 없으면 건너뜀)
        self.spinThresh: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, "spinThresh")
        self.spinC1: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, "spinC1")
        self.spinC2: QtWidgets.QSpinBox = self.findChild(QtWidgets.QSpinBox, "spinC2")

        # 백엔드
        self.cam = CameraBackendUIOpenCV(0)

        # 시그널 연결 (UI에 해당 컨트롤이 없으면 None이므로 체크)
        if self.btnOpen: self.btnOpen.clicked.connect(self.on_open)
        if self.btnClose: self.btnClose.clicked.connect(self.on_close)

        # 표시 타이머
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

    def on_open(self):
        if self.cam.open():
            self.cam.start()

    def on_close(self):
        self.cam.close()

    def closeEvent(self, e):
        self.on_close(); super().closeEvent(e)

    def on_tick(self):
        f = self.cam.get()
        if f is None: return
        stage = self.comboStage.currentText() if self.comboStage else "Input"
        disp = f
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if stage == "Gray":
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif stage == "Binary" and self.spinThresh:
            _, thr = cv2.threshold(gray, self.spinThresh.value(), 255, cv2.THRESH_BINARY)
            disp = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        elif stage == "Canny" and self.spinC1 and self.spinC2:
            ed = cv2.Canny(gray, self.spinC1.value(), self.spinC2.value())
            disp = cv2.cvtColor(ed, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        q = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
        if self.lblView:
            self.lblView.setPixmap(QtGui.QPixmap.fromImage(q))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.show(); sys.exit(app.exec())