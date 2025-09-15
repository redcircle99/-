"""
SDK-교체형 카메라 뷰어 (학습용 주석 버전)
--------------------------------------
목적:
  - 카메라 SDK만 바꿔 끼우면 바로 쓸 수 있는 최소 뼈대
  - 소프트웨어 트리거(주기/수동), 연속 그랩, UI 업데이트 흐름을 명확히 이해
  - 기본 영상 파이프라인(디노이즈 → 콘트라스트 → 이진화 → 엣지 → 형태학)을 실습

권장 학습 루트:
 1) OpenCV 웹캠으로 실행해 구조 익히기
 2) CameraBackend를 MVS/pylon 등으로 교체
 3) 파이프라인 파라미터를 바꿔가며 결과 비교

필수 패키지: PyQt6, opencv-python
  pip install PyQt6 opencv-python

실행:
  python SDK-교체형_카메라_뷰어_파이프라인_주석버전.py
"""
from __future__ import annotations
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


# ─────────────────────────────────────────────────────────────
# 1) Backend 인터페이스 (교체 포인트)
#    - 실제 벤더 SDK(MVS, pylon 등)로 교체할 때 아래 메서드만 채우면 됨
#    - 지금은 OpenCV(VideoCapture)로 동작하게 하여 구조를 학습
# ─────────────────────────────────────────────────────────────
class CameraBackend:
    """학습용 백엔드.

    실제 SDK로 바꿀 때 필요한 공통 인터페이스를 최대한 단순화했다.
    - open/close: 디바이스 열고 닫기
    - set_trigger_mode: 연속 그랩 vs 트리거 기반 캡처 전환
    - software_trigger: 트리거 1회 발생
    - grab_continuous_start/stop: 연속 그랩 시작/정지
    - get_latest_frame: 최신 프레임 전달 (UI Thread가 호출)

    주의: 실제 SDK에서는 픽셀 포맷, 버퍼 획득, 쓰레드/콜백 모델이 다르다.
         핵심은 "UI 스레드와 캡처 스레드를 분리"하는 패턴을 익히는 것.
    """
    def __init__(self, source: int | str = 0):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = source
        self._lock = threading.Lock()
        self._running = False
        self._trigger_mode = False
        self._latest: Optional[np.ndarray] = None

    # --- 디바이스 열기/닫기 ---
    def open(self) -> bool:
        # TODO: MVS라면 MV_CC_CreateHandle → MV_CC_OpenDevice 등으로 대체
        self.cap = cv2.VideoCapture(self.source)
        return bool(self.cap and self.cap.isOpened())

    def close(self) -> None:
        self._running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # --- 노출/게인 (OpenCV에선 제한적) ---
    def set_exposure_us(self, exposure_us: int) -> None:
        # TODO: SDK 노드에 맞춰 구현 (예: ExposureAuto=Off, ExposureTime=exposure_us)
        if self.cap is not None:
            # 일부 드라이버는 음수로 log-scale, 또는 무시될 수 있음
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_us) / 1000.0)

    def set_gain(self, gain: float) -> None:
        # TODO: SDK 게인 노드 설정 (GainAuto=Off, Gain=...)
        pass

    # --- 트리거 모드 전환 ---
    def set_trigger_mode(self, on: bool) -> None:
        # TODO: SDK에서는 TriggerMode=On/Off, TriggerSource=Software 등 설정
        self._trigger_mode = on
        if on:
            # 트리거 모드: 연속 그랩 중지
            self.grab_continuous_stop()
        else:
            # 연속 그랩: 프레임 루프 시작
            self.grab_continuous_start()

    # --- 소프트웨어 트리거 1회 ---
    def software_trigger(self) -> None:
        # TODO: SDK에서는 ExecuteSoftwareTrigger() 호출
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            with self._lock:
                self._latest = frame

    # --- 연속 그랩 시작/정지 ---
    def grab_continuous_start(self) -> None:
        if self.cap is None:
            return
        if self._running:
            return
        self._running = True

        def _loop():
            # 트리거 모드일 때는 루프가 쉬도록 구성 (이 샘플에서는 보호적 동작)
            while self._running and self.cap is not None:
                if self._trigger_mode:
                    time.sleep(0.003)
                    continue
                ok, frame = self.cap.read()
                if ok:
                    with self._lock:
                        self._latest = frame
                else:
                    time.sleep(0.003)

        threading.Thread(target=_loop, daemon=True).start()

    def grab_continuous_stop(self) -> None:
        self._running = False

    # --- 최신 프레임 획득 (깊은 복사로 UI Thread 안전 보장) ---
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()


# ─────────────────────────────────────────────────────────────
# 2) 영상 파이프라인 (학습 핵심)
#    - Denoise → Contrast → Threshold → Edge → Morphology
#    - 모든 스테이지는 Optional, 파라미터를 UI에서 조절
# ─────────────────────────────────────────────────────────────
@dataclass
class PipelineParams:
    # Denoise
    denoise: str = "none"            # none/gaussian/median/bilateral
    denoise_ksize: int = 3            # 홀수

    # Contrast
    contrast: str = "none"            # none/auto/clahe
    clahe_clip: float = 2.0
    clahe_tiles: int = 8

    # Threshold
    thresh_mode: str = "none"         # none/binary/otsu/adaptive
    thresh_value: int = 128
    adaptive_block: int = 15          # 홀수
    adaptive_C: int = 2

    # Edge
    edge_mode: str = "none"            # none/canny/sobel
    canny_low: int = 80
    canny_high: int = 160
    sobel_ksize: int = 3

    # Morphology
    morph_mode: str = "none"           # none/open/close/erode/dilate
    morph_ksize: int = 3               # 홀수
    morph_iter: int = 1


class ProcessingPipeline:
    """입력 프레임(BGR 또는 Gray)을 받아 지정된 스테이지를 순차 적용한다."""
    def __init__(self, params: PipelineParams):
        self.p = params

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        if self.p.denoise == "gaussian":
            k = max(1, self.p.denoise_ksize | 1)  # 홀수 보정
            return cv2.GaussianBlur(img, (k, k), 0)
        if self.p.denoise == "median":
            k = max(1, self.p.denoise_ksize | 1)
            return cv2.medianBlur(img, k)
        if self.p.denoise == "bilateral":
            # d, sigmaColor, sigmaSpace 간단 설정 (학습용)
            d = max(3, self.p.denoise_ksize)
            return cv2.bilateralFilter(img, d, 75, 75)
        return img

    def _contrast(self, img_gray: np.ndarray) -> np.ndarray:
        if self.p.contrast == "auto":
            # 히스토그램 평활화 (8비트 그레이 가정)
            return cv2.equalizeHist(img_gray)
        if self.p.contrast == "clahe":
            clip = max(0.1, float(self.p.clahe_clip))
            tiles = max(2, int(self.p.clahe_tiles))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
            return clahe.apply(img_gray)
        return img_gray

    def _threshold(self, img_gray: np.ndarray) -> np.ndarray:
        if self.p.thresh_mode == "binary":
            _, thr = cv2.threshold(img_gray, int(self.p.thresh_value), 255, cv2.THRESH_BINARY)
            return thr
        if self.p.thresh_mode == "otsu":
            _, thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return thr
        if self.p.thresh_mode == "adaptive":
            blk = max(3, self.p.adaptive_block | 1)
            C = int(self.p.adaptive_C)
            return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, blk, C)
        return img_gray

    def _edge(self, img_gray_or_bin: np.ndarray) -> np.ndarray:
        if self.p.edge_mode == "canny":
            low, high = int(self.p.canny_low), int(self.p.canny_high)
            return cv2.Canny(img_gray_or_bin, low, high)
        if self.p.edge_mode == "sobel":
            k = max(1, self.p.sobel_ksize | 1)
            sx = cv2.Sobel(img_gray_or_bin, cv2.CV_16S, 1, 0, ksize=k)
            sy = cv2.Sobel(img_gray_or_bin, cv2.CV_16S, 0, 1, ksize=k)
            mag = cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32))
            mag = np.clip(mag, 0, 255).astype(np.uint8)
            return mag
        return img_gray_or_bin

    def _morph(self, img_bin_or_edge: np.ndarray) -> np.ndarray:
        if self.p.morph_mode == "none":
            return img_bin_or_edge
        k = max(1, self.p.morph_ksize | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        it = max(1, int(self.p.morph_iter))
        if self.p.morph_mode == "open":
            return cv2.morphologyEx(img_bin_or_edge, cv2.MORPH_OPEN, kernel, iterations=it)
        if self.p.morph_mode == "close":
            return cv2.morphologyEx(img_bin_or_edge, cv2.MORPH_CLOSE, kernel, iterations=it)
        if self.p.morph_mode == "erode":
            return cv2.erode(img_bin_or_edge, kernel, iterations=it)
        if self.p.morph_mode == "dilate":
            return cv2.dilate(img_bin_or_edge, kernel, iterations=it)
        return img_bin_or_edge

    # --- 파이프라인 엔드투엔드 ---
    def run(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """프레임에 파이프라인을 적용하고 중간 단계를 함께 반환.
        반환값: (최종 출력 이미지, 디버그 딕셔너리)
        """
        dbg = {}
        # 1) 그레이 변환 (대부분의 고전 필터는 그레이 입력을 가정)
        gray = self._to_gray(frame_bgr)
        dbg["gray"] = gray

        # 2) 디노이즈
        dn = self._denoise(gray)
        dbg["denoise"] = dn

        # 3) 콘트라스트
        ct = self._contrast(dn)
        dbg["contrast"] = ct

        # 4) 이진화
        th = self._threshold(ct)
        dbg["threshold"] = th

        # 5) 엣지
        ed = self._edge(th)
        dbg["edge"] = ed

        # 6) 형태학
        mo = self._morph(ed)
        dbg["morph"] = mo

        return mo, dbg


# ─────────────────────────────────────────────────────────────
# 3) PyQt 뷰어
#    - 트리거 모드/주기
#    - 파이프라인 파라미터 UI
#    - 표시 단계 선택(Input/Gray/Threshold/Edge/Morph)
# ─────────────────────────────────────────────────────────────
class Viewer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SDK-교체형 카메라 뷰어 (학습용)")
        self.resize(1080, 760)

        # 백엔드 & 파이프라인
        self.cam = CameraBackend(source=0)
        self.params = PipelineParams()
        self.pipe = ProcessingPipeline(self.params)
        self._last_dbg = {}

        # --- 레이아웃 ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # 상단: 카메라/트리거
        h_cam = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("카메라 열기")
        self.btn_close = QtWidgets.QPushButton("닫기")
        self.chk_trigger = QtWidgets.QCheckBox("소프트 트리거 모드")
        self.spin_period = QtWidgets.QSpinBox()
        self.spin_period.setRange(5, 5000)
        self.spin_period.setValue(200)
        self.chk_auto = QtWidgets.QCheckBox("자동(주기) 트리거")
        self.btn_once = QtWidgets.QPushButton("트리거 1회")
        h_cam.addWidget(self.btn_open)
        h_cam.addWidget(self.btn_close)
        h_cam.addSpacing(16)
        h_cam.addWidget(self.chk_trigger)
        h_cam.addWidget(QtWidgets.QLabel("주기(ms)"))
        h_cam.addWidget(self.spin_period)
        h_cam.addWidget(self.chk_auto)
        h_cam.addWidget(self.btn_once)
        h_cam.addStretch(1)
        vbox.addLayout(h_cam)

        # 중단: 파이프라인 파라미터들
        grid = QtWidgets.QGridLayout()
        r = 0
        # Denoise
        self.cb_denoise = QtWidgets.QComboBox(); self.cb_denoise.addItems(["none","gaussian","median","bilateral"])
        self.spin_dn_ksize = QtWidgets.QSpinBox(); self.spin_dn_ksize.setRange(1, 31); self.spin_dn_ksize.setValue(self.params.denoise_ksize)
        grid.addWidget(QtWidgets.QLabel("Denoise"), r, 0); grid.addWidget(self.cb_denoise, r, 1)
        grid.addWidget(QtWidgets.QLabel("ksize"), r, 2); grid.addWidget(self.spin_dn_ksize, r, 3); r += 1

        # Contrast
        self.cb_contrast = QtWidgets.QComboBox(); self.cb_contrast.addItems(["none","auto","clahe"])
        self.dbl_clahe_clip = QtWidgets.QDoubleSpinBox(); self.dbl_clahe_clip.setRange(0.1, 40.0); self.dbl_clahe_clip.setDecimals(1); self.dbl_clahe_clip.setValue(self.params.clahe_clip)
        self.spin_clahe_tiles = QtWidgets.QSpinBox(); self.spin_clahe_tiles.setRange(2, 32); self.spin_clahe_tiles.setValue(self.params.clahe_tiles)
        grid.addWidget(QtWidgets.QLabel("Contrast"), r, 0); grid.addWidget(self.cb_contrast, r, 1)
        grid.addWidget(QtWidgets.QLabel("CLAHE clip"), r, 2); grid.addWidget(self.dbl_clahe_clip, r, 3)
        grid.addWidget(QtWidgets.QLabel("tiles"), r, 4); grid.addWidget(self.spin_clahe_tiles, r, 5); r += 1

        # Threshold
        self.cb_thresh = QtWidgets.QComboBox(); self.cb_thresh.addItems(["none","binary","otsu","adaptive"])
        self.spin_thresh = QtWidgets.QSpinBox(); self.spin_thresh.setRange(0,255); self.spin_thresh.setValue(self.params.thresh_value)
        self.spin_ad_block = QtWidgets.QSpinBox(); self.spin_ad_block.setRange(3, 99); self.spin_ad_block.setValue(self.params.adaptive_block)
        self.spin_ad_C = QtWidgets.QSpinBox(); self.spin_ad_C.setRange(-20, 20); self.spin_ad_C.setValue(self.params.adaptive_C)
        grid.addWidget(QtWidgets.QLabel("Threshold"), r, 0); grid.addWidget(self.cb_thresh, r, 1)
        grid.addWidget(QtWidgets.QLabel("value"), r, 2); grid.addWidget(self.spin_thresh, r, 3)
        grid.addWidget(QtWidgets.QLabel("adaptive block"), r, 4); grid.addWidget(self.spin_ad_block, r, 5)
        grid.addWidget(QtWidgets.QLabel("C"), r, 6); grid.addWidget(self.spin_ad_C, r, 7); r += 1

        # Edge
        self.cb_edge = QtWidgets.QComboBox(); self.cb_edge.addItems(["none","canny","sobel"])
        self.spin_canny_low = QtWidgets.QSpinBox(); self.spin_canny_low.setRange(0,255); self.spin_canny_low.setValue(self.params.canny_low)
        self.spin_canny_high = QtWidgets.QSpinBox(); self.spin_canny_high.setRange(0,255); self.spin_canny_high.setValue(self.params.canny_high)
        self.spin_sobel_k = QtWidgets.QSpinBox(); self.spin_sobel_k.setRange(1, 31); self.spin_sobel_k.setValue(self.params.sobel_ksize)
        grid.addWidget(QtWidgets.QLabel("Edge"), r, 0); grid.addWidget(self.cb_edge, r, 1)
        grid.addWidget(QtWidgets.QLabel("canny low/high"), r, 2)
        grid.addWidget(self.spin_canny_low, r, 3); grid.addWidget(self.spin_canny_high, r, 4)
        grid.addWidget(QtWidgets.QLabel("sobel k"), r, 5); grid.addWidget(self.spin_sobel_k, r, 6); r += 1

        # Morphology
        self.cb_morph = QtWidgets.QComboBox(); self.cb_morph.addItems(["none","open","close","erode","dilate"])
        self.spin_morph_k = QtWidgets.QSpinBox(); self.spin_morph_k.setRange(1, 31); self.spin_morph_k.setValue(self.params.morph_ksize)
        self.spin_morph_it = QtWidgets.QSpinBox(); self.spin_morph_it.setRange(1, 10); self.spin_morph_it.setValue(self.params.morph_iter)
        grid.addWidget(QtWidgets.QLabel("Morph"), r, 0); grid.addWidget(self.cb_morph, r, 1)
        grid.addWidget(QtWidgets.QLabel("ksize"), r, 2); grid.addWidget(self.spin_morph_k, r, 3)
        grid.addWidget(QtWidgets.QLabel("iter"), r, 4); grid.addWidget(self.spin_morph_it, r, 5); r += 1

        vbox.addLayout(grid)

        # 하단: 표시 단계 선택 + 캔버스
        h_disp = QtWidgets.QHBoxLayout()
        self.cb_stage = QtWidgets.QComboBox(); self.cb_stage.addItems(["Input","gray","denoise","contrast","threshold","edge","morph"])
        h_disp.addWidget(QtWidgets.QLabel("표시 단계")); h_disp.addWidget(self.cb_stage)
        h_disp.addStretch(1)
        vbox.addLayout(h_disp)

        self.lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.lbl, 1)

        # 타이머: UI 갱신 (화면 렌더)
        self.ui_timer = QtWidgets.QTimer(self)
        self.ui_timer.setInterval(30)
        self.ui_timer.timeout.connect(self.update_view)
        self.ui_timer.start()

        # 타이머: 주기 트리거
        self.trig_timer = QtWidgets.QTimer(self)
        self.trig_timer.timeout.connect(self.on_trigger_tick)

        # 시그널 연결
        self.btn_open.clicked.connect(self.on_open)
        self.btn_close.clicked.connect(self.on_close)
        self.chk_trigger.toggled.connect(self.on_trigger_mode)
        self.chk_auto.toggled.connect(self.on_auto_toggle)
        self.spin_period.valueChanged.connect(self.on_period_change)
        self.btn_once.clicked.connect(self.on_once)

        # 파라미터 변경 → dataclass 반영
        self.cb_denoise.currentTextChanged.connect(self.sync_params)
        self.spin_dn_ksize.valueChanged.connect(self.sync_params)
        self.cb_contrast.currentTextChanged.connect(self.sync_params)
        self.dbl_clahe_clip.valueChanged.connect(self.sync_params)
        self.spin_clahe_tiles.valueChanged.connect(self.sync_params)
        self.cb_thresh.currentTextChanged.connect(self.sync_params)
        self.spin_thresh.valueChanged.connect(self.sync_params)
        self.spin_ad_block.valueChanged.connect(self.sync_params)
        self.spin_ad_C.valueChanged.connect(self.sync_params)
        self.cb_edge.currentTextChanged.connect(self.sync_params)
        self.spin_canny_low.valueChanged.connect(self.sync_params)
        self.spin_canny_high.valueChanged.connect(self.sync_params)
        self.spin_sobel_k.valueChanged.connect(self.sync_params)
        self.cb_morph.currentTextChanged.connect(self.sync_params)
        self.spin_morph_k.valueChanged.connect(self.sync_params)
        self.spin_morph_it.valueChanged.connect(self.sync_params)

    # ── 시그널 핸들러 ─────────────────────────────────────────
    def on_open(self) -> None:
        if self.cam.open():
            if not self.chk_trigger.isChecked():
                self.cam.grab_continuous_start()

    def on_close(self) -> None:
        self.trig_timer.stop()
        self.cam.grab_continuous_stop()
        self.cam.close()

    def on_trigger_mode(self, on: bool) -> None:
        self.cam.set_trigger_mode(on)

    def on_auto_toggle(self, on: bool) -> None:
        if on:
            self.trig_timer.start(self.spin_period.value())
        else:
            self.trig_timer.stop()

    def on_period_change(self, _val: int) -> None:
        if self.chk_auto.isChecked():
            self.trig_timer.start(self.spin_period.value())

    def on_once(self) -> None:
        self.cam.software_trigger()

    def on_trigger_tick(self) -> None:
        self.cam.software_trigger()

    # ── 파라미터 동기화 ───────────────────────────────────────
    def sync_params(self) -> None:
        self.params.denoise = self.cb_denoise.currentText()
        self.params.denoise_ksize = self.spin_dn_ksize.value()
        self.params.contrast = self.cb_contrast.currentText()
        self.params.clahe_clip = self.dbl_clahe_clip.value()
        self.params.clahe_tiles = self.spin_clahe_tiles.value()
        self.params.thresh_mode = self.cb_thresh.currentText()
        self.params.thresh_value = self.spin_thresh.value()
        self.params.adaptive_block = self.spin_ad_block.value()
        self.params.adaptive_C = self.spin_ad_C.value()
        self.params.edge_mode = self.cb_edge.currentText()
        self.params.canny_low = self.spin_canny_low.value()
        self.params.canny_high = self.spin_canny_high.value()
        self.params.sobel_ksize = self.spin_sobel_k.value()
        self.params.morph_mode = self.cb_morph.currentText()
        self.params.morph_ksize = self.spin_morph_k.value()
        self.params.morph_iter = self.spin_morph_it.value()

    # ── 화면 갱신 ─────────────────────────────────────────────
    def update_view(self) -> None:
        frame = self.cam.get_latest_frame()
        if frame is None:
            return

        # 파이프라인 실행
        out, dbg = self.pipe.run(frame)
        self._last_dbg = dbg

        # 표시 단계 선택
        stage = self.cb_stage.currentText()
        if stage == "Input":
            disp = frame
        else:
            disp = dbg.get(stage, out)
            if disp.ndim == 2:
                # 그레이/바이너리/엣지 등은 컬러로 변환하여 보기 좋게
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        # QImage 변환
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],QtGui.QImage.Format.Format_RGB888)
        self.lbl.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # ── 안전한 종료 ───────────────────────────────────────────
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.on_close()
        super().closeEvent(e)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()