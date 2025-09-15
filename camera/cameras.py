# cameras.py
import cv2
import numpy as np
from threading import Event, Thread
import time
import sys
import os
import threading
import queue  # queue 모듈 추가

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DictKey import DictKey
from PySide6.QtCore import QTimer
if DictKey.UseNoviCam:
    import ctypes  
from camera.NovitecCameraAPIWrapper import NovitecCameraAPIWrapper, CError, Image

# from genicam import gentl  # 필요에 따라 import
if DictKey.UseMVSCam:
    from camera.CameraParams_header import *
    from camera.MvCameraControl_class import *
    
    from ctypes import *
    from camera.MvCameraControl_class import *
    from camera.CameraParams_header import *
    from camera.MvErrorDefine_const import *
    from camera.CamOperation_class import *

if DictKey.UseBaslerCam:
    from camera.basler_camera import BaslerCamera
    
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr



class RGB8Image:
    def __init__(self, width, height, data_format, image_data):
        self.image_data = self._process_image(image_data, data_format, width, height)

    def _process_image(self, image_data, data_format, width, height) -> np.ndarray:
        if data_format == "Mono8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_GRAY2RGB)
        elif data_format == "BayerRG8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_BayerRG2RGB)
        elif data_format == "BayerGR8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_BayerGR2RGB)
        elif data_format == "BayerGB8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_BayerGB2RGB)
        elif data_format == "BayerBG8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_BayerBG2RGB)
        elif data_format == "RGB8":
            return image_data.reshape(height, width, 3)
        elif data_format == "BGR8":
            return cv2.cvtColor(image_data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported pixel format: %s" % data_format)

class CameraManager:
    def __init__(self, st_device_list=None):
        # self.harvester_manager = harvester_manager
        self.st_device_list = self.init_device_list(st_device_list)  # 초기화
        self.camera = None
        self.camera_type = None
        self.camera_source = None
        self.timer = None
        self._stop_event = Event()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time  # 마지막 fps 업데이트 시간 초기화
        self.last_fps = 0.0
        self.camera_config = {}
        self.realtimeFps = 0
        self.fps = 60
        self.is_running = True
        self.acquisition_thread = None
        self.frame_queue = queue.Queue()  # threading.Queue() -> queue.Queue() 변경
        self.current_fps = 0.0

    def init_device_list(self, st_device_list):
        if DictKey.UseMVSCam:
            if st_device_list is None or isinstance(st_device_list, list):
                # `st_device_list`가 `MV_CC_DEVICE_INFO_LIST` 타입이 아닌 경우 초기화 필요
                st_device_list = MV_CC_DEVICE_INFO_LIST()  # 새 인스턴스 생성
            self.enumerate_mvs_devices(st_device_list)  # 수정된 호출 방식
        return st_device_list

    def enumerate_mvs_devices(self, device_list):
        # MVS 카메라가 활성화되지 않은 경우
        if not DictKey.UseMVSCam:
            return False
            
        # MVS 카메라 목록 열거
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != MV_OK:
            print(f"장치 검색 실패: {ToHexStr(ret)}")
            return False
        print(f"발견된 장치 수: {device_list.nDeviceNum}")
        
        # 발견된 각 장치의 정보 출력
        for i in range(device_list.nDeviceNum):
            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print(f"장치 {i}: GigE 카메라")
                gige_info = cast(pointer(mvcc_dev_info.SpecialInfo.stGigEInfo), POINTER(MV_GIGE_DEVICE_INFO)).contents
                print(f"  IP: {hex(gige_info.nCurrentIp)[2:]}")
                
                # 문자열 변환을 위해 bytes 객체로 변환
                manufacturer = bytes(bytearray(gige_info.chManufacturerName)).decode('ascii', errors='ignore').strip('\0')
                model = bytes(bytearray(gige_info.chModelName)).decode('ascii', errors='ignore').strip('\0')
                serial = bytes(bytearray(gige_info.chSerialNumber)).decode('ascii', errors='ignore').strip('\0')
                
                print(f"  제조사: {manufacturer}")
                print(f"  모델: {model}")
                print(f"  시리얼: {serial}")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print(f"장치 {i}: USB 카메라")
                usb_info = cast(pointer(mvcc_dev_info.SpecialInfo.stUsb3VInfo), POINTER(MV_USB3_DEVICE_INFO)).contents
                
                # 문자열 변환을 위해 bytes 객체로 변환
                serial = bytes(bytearray(usb_info.chSerialNumber)).decode('ascii', errors='ignore').strip('\0')
                manufacturer = bytes(bytearray(usb_info.chManufacturerName)).decode('ascii', errors='ignore').strip('\0')
                
                print(f"  시리얼: {serial}")
                print(f"  제조사: {manufacturer}")
        
        return True

    def create_camera(self, camera_type, source):
        self.camera_type = camera_type
        self.camera_source = source
        # if camera_type == DictKey.GenICam and self.harvester_manager:
        #     self.camera = GenICamCamera(index=int(source), harvester_manager=self.harvester_manager)
        # el
        if camera_type == DictKey.USBCam:
            self.camera = USBCamera(index=int(source))
        elif camera_type == DictKey.IMGFile:
            self.camera = ImageCamera(source)
        elif camera_type == DictKey.VODFile:
            self.camera = VideoCamera(source)
        elif camera_type == DictKey.NoviCam:
            self.camera = NovitecCamera(serial_number=source)
        elif camera_type == DictKey.MVSCam and DictKey.UseMVSCam and self.st_device_list is not None:
            self.camera = MVSCamera(st_device_list=self.st_device_list, n_connect_num=int(source))
        elif camera_type == DictKey.BaslerCam and DictKey.UseBaslerCam:
            self.camera = BaslerCamera(index=int(source))
        else:
            raise ValueError(f"Unknown camera type or disabled: {camera_type}")

    def connect(self):
        if not self.camera:
            self.create_camera(self.camera_type, self.camera_source)
        if self.camera:
            try:
                is_connected = self.camera.connect()
                if is_connected:
                    print("Camera connected successfully")
                else:
                    print(f"Failed to connect camera: {self.camera_type}")
                return is_connected
            except Exception as e:
                print(f"Error connecting camera: {e}")
                return False
        return False

    def start_acquisition(self, callback_func=None):
        # 프레임 획득용 스레드 구현
        self.is_running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_thread, args=(callback_func,))
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
        return True
        
    def _acquisition_thread(self, callback_func):
        # 카메라에서 프레임만 빠르게 가져오는 전용 스레드
        last_time = time.time()
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while self.is_running:
            # 프레임 획득 - 가능한 한 최소한의 처리만 수행
            frame = self._get_frame_fast()  # 최적화된 프레임 획득 함수
            
            if frame is None:
                error_count += 1
                if error_count > max_errors:
                    print("연속된 프레임 획득 실패, 카메라 재연결 시도...")
                    self.disconnect()
                    time.sleep(1)
                    self.connect()
                    self.camera.start_acquisition()
                    error_count = 0
                time.sleep(0.1)  # 실패 시 잠시 대기
                continue
            
            error_count = 0  # 성공하면 오류 카운터 초기화
            
            # FPS 계산
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= 1.0:  # 1초마다 FPS 업데이트
                self.current_fps = frame_count / elapsed
                frame_count = 0
                last_time = current_time
                print(f"현재 FPS: {self.current_fps:.2f}")
            
            # 큐에 프레임 저장 (콜백 함수는 별도 스레드에서 실행)
            if self.frame_queue.qsize() < 2:  # 큐 크기 제한
                self.frame_queue.put((frame, self.current_fps))
                
    def _get_frame_fast(self):
        # 카메라에서 최대한 빠르게 프레임 획득
        if self.camera_type == DictKey.MVSCam:
            # 메모리 할당 최소화 및 복사 연산 최적화
            # 카메라 SDK의 고속 프레임 획득 기능 활용
            if self.camera:
                return self.camera.get_frame()
            return None
        elif self.camera_type == DictKey.BaslerCam:
            # Basler 카메라용 고속 프레임 획득
            if self.camera:
                return self.camera.get_frame()
            return None
        return self.camera.get_frame() if self.camera else None  # 다른 카메라 타입 처리

    def _calculate_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:  # Ensure elapsed_time is not zero
            if current_time - self.last_update_time >= 1.0:
                self.last_fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = current_time
                self.last_update_time = current_time
        else:
            self.last_fps = 0  # Set fps to 0 if elapsed_time is zero

        return self.last_fps

    def get_frame(self):
        if self.camera:
            try:
                return self.camera.get_frame()
            except Exception as e:
                print(f"Error getting frame: {e}")
                return None
        return None

    def stop_acquisition(self):
        self.is_running = False
        if self.camera:
            self.camera.stop_acquisition()
        if self.timer:
            self.timer.stop()
            self.timer = None

    def disconnect(self):
        self.stop_acquisition()
        if self.camera:
            self.camera.disconnect()
            self.camera = None

    def configure_camera(self):
        if self.camera:
            try:
                self.camera_config = self.camera.configure() 
                return self.camera_config
            except Exception as e:
                print(f"Error configuring camera: {e}")
                # 기본값 반환
                return {
                    "fps": 30,
                    "exposure": 10000,
                    "gain": 1.0,
                    "trigger_mode": False
                }
        return {}

    def set_trigger(self, mode):
        if self.camera:
            try:
                self.camera.set_trigger(mode)
                self.configure_camera()
            except Exception as e:
                print(f"Error setting trigger mode: {e}")
    
    def set_trigger_source(self, value):
        if self.camera:
            try:
                self.camera.set_trigger_source(value)
                self.configure_camera()
            except Exception as e:
                print(f"Error setting trigger source: {e}")

    def TriggerSoftwareExecute(self):
        if self.camera:
            try:
                self.camera.TriggerSoftwareExecute()
            except Exception as e:
                print(f"Error executing software trigger: {e}")

    def set_exposure(self, exposure):
        if self.camera:
            try:
                self.camera.set_exposure(exposure)
                self.configure_camera()
            except Exception as e:
                print(f"Error setting exposure: {e}")

    def set_fps(self, fps):
        if self.camera:
            try:
                self.camera.set_fps(fps)
                self.configure_camera()
            except Exception as e:
                print(f"Error setting fps: {e}")

    def set_gain(self, gain):
        if self.camera:
            try:
                self.camera.set_gain(gain)
                self.configure_camera()
            except Exception as e:
                print(f"Error setting gain: {e}")
    
    def get_trigger_mode(self):
        return self.camera_config.get("trigger_mode", False)
    
    def get_fps(self):
        return self.camera_config.get("fps", 12)
    
    def get_exposure(self):
        return self.camera_config.get("exposure", 100)
    
    def get_gain(self):
        return self.camera_config.get("gain", 1)

class MVSCamera:
    def __init__(self, st_device_list, n_connect_num):
        self.camera = MvCamera()
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.connected = False
        self.grabbing = False
        self.buffer = None
        self.buffer_size = 0

    def _get_optimal_packet_size(self):
        """GigE 카메라의 최적 패킷 크기를 가져옵니다."""
        nPacketSize = self.camera.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            return int(nPacketSize)
        return 1500  # 기본 패킷 크기

    def connect(self):
        if self.connected:
            print("MVS 카메라가 이미 연결되어 있습니다.")
            return True

        # 장치 목록 확인
        if self.st_device_list.nDeviceNum == 0:
            print("연결된 카메라가 없습니다.")
            return False

        if self.n_connect_num >= self.st_device_list.nDeviceNum:
            print(f"잘못된 카메라 인덱스입니다. 사용 가능한 카메라 수: {self.st_device_list.nDeviceNum}")
            return False

        # Handle 초기화 및 장치 열기
        try:
            selected_device_info = cast(self.st_device_list.pDeviceInfo[self.n_connect_num], 
                                      POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self.camera.MV_CC_CreateHandle(selected_device_info)
            if ret != MV_OK:
                print(f"핸들 생성 실패: {ToHexStr(ret)}")
                return False

            ret = self.camera.MV_CC_OpenDevice()
            if ret != MV_OK:
                print(f"장치 열기 실패: {ToHexStr(ret)}")
                self.camera.MV_CC_DestroyHandle()
                return False

            # 연결 성공 표시
            self.connected = True
            
            # 스트림 전송 최적화 - 연결 성공 후에 설정
            ret = self.camera.MV_CC_SetIntValue("GevSCPD", 0)  # 패킷 지연 최소화 (GigE 카메라용)
            
            # 데이터 전송 큐 크기 증가
            ret = self.camera.MV_CC_SetIntValue("TLParamsLocked", 0)  # 설정 잠금 해제
            ret = self.camera.MV_CC_SetIntValue("StreamTransferSize", 1048576)  # 1MB로 설정
            ret = self.camera.MV_CC_SetIntValue("StreamTransferNumberUrb", 64)  # USB 전송 버퍼 수
            
            # 트리거 모드 확인
            ret = self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            
            # 네트워크 MTU 크기 최적화
            if hasattr(self, '_get_optimal_packet_size'):
                packet_size = self._get_optimal_packet_size()
                if packet_size > 0:
                    ret = self.camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

            print("MVS 카메라가 성공적으로 연결되었습니다.")
            return True

        except Exception as e:
            print(f"카메라 연결 중 오류 발생: {str(e)}")
            if hasattr(self.camera, 'MV_CC_DestroyHandle'):
                self.camera.MV_CC_DestroyHandle()
            self.connected = False
            return False

    def configure(self):
        if not self.connected:
            print("Camera is not connected")
            return {}

        frame_rate = MVCC_FLOATVALUE()
        exposure = MVCC_FLOATVALUE()
        trigger_mode = MVCC_ENUMVALUE()
        gain = MVCC_FLOATVALUE()

        self.camera.MV_CC_GetFloatValue("AcquisitionFrameRate", frame_rate)
        self.camera.MV_CC_GetFloatValue("ExposureTime", exposure)
        self.camera.MV_CC_GetEnumValue("TriggerMode", trigger_mode)
        self.camera.MV_CC_GetFloatValue("Gain", gain)

        config = {
            "fps": frame_rate.fCurValue,
            "exposure": exposure.fCurValue,
            "trigger_mode": (trigger_mode.nCurValue == MV_TRIGGER_MODE_ON),
            "gain": gain.fCurValue
        }
        
        # 프레임 속도 최적화
        try:
            # 프레임 속도 제한 해제
            self.camera.MV_CC_SetEnumValue("AcquisitionFrameRateMode", 2)  # 제어 가능 모드
            self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", 70.0)  # 원하는 fps 값으로 설정
            
            # 자동 노출 끄기 (더 빠른 프레임 율을 위해)
            self.camera.MV_CC_SetEnumValue("ExposureAuto", 0)  # Off
            self.camera.MV_CC_SetFloatValue("ExposureTime", exposure.fCurValue)  # 빠른 노출 시간 (마이크로초)
        except Exception as e:
            print(f"프레임 속도 최적화 중 오류: {e}")
        
        # print("Camera configuration:", config)
        return config
    
    def stream_to_gstreamer(self):
        # GStreamer 파이프라인 설정
        gst_pipeline = "appsrc ! videoconvert ! autovideosink"
        cap = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30, (1060, 1060), True)

        while self.grabbing:
            frame = self.get_frame()
            if frame is not None:
                cap.write(frame)
            time.sleep(1 / 70)  # 60 FPS로 전송

        cap.release()

    def start_acquisition(self):
        if not self.connected:
            self.connect()
            print("MVS: Camera re connected")

        if not self.connected:
            print("Camera not connected")
            return False

        ret = self.camera.MV_CC_StartGrabbing()
        if ret != MV_OK:
            print(f"Failed to start grabbing: {ToHexStr(ret)}")
            return False

        print("Camera started grabbing")
        self.grabbing = True
        return True

    def stop_acquisition(self):
        if not self.grabbing:
            print("Camera not grabbing")
            return False

        ret = self.camera.MV_CC_StopGrabbing()
        if ret != MV_OK:
            print(f"Failed to stop grabbing: {ToHexStr(ret)}")
            return False

        print("Camera stopped grabbing")
        self.grabbing = False
        return True

    def get_frame(self): # todo: 카메라 매니저에 있는 클래스 타서 실행되도록 해야함 로직은 비슷하기에 주석 처리 후 변경
        if not self.grabbing:
            # 카메라가 프레임을 획득하고 있지 않은 경우
            if not self.start_acquisition():
                print("카메라가 그래빙 상태가 아니어서 그래빙을 시작합니다.")
            else:
                print("그래빙 시작에 실패했습니다.")
                return None

        # try:
        #     # 간단한 재시도 메커니즘
        #     max_retries = 3
        #     for retry in range(max_retries):
        #         stFrameInfo = MV_FRAME_OUT_INFO_EX()
                
        #         # 버퍼 크기 확인
        #         stPayloadSize = MVCC_INTVALUE_EX()
        #         ret_temp = self.camera.MV_CC_GetIntValueEx("PayloadSize", stPayloadSize)
        #         if ret_temp != MV_OK:
        #             print(f"페이로드 크기 가져오기 실패: {ToHexStr(ret_temp)}")
        #             time.sleep(0.01)
        #             continue
                
        #         NeedBufSize = int(stPayloadSize.nCurValue)
                
        #         # 버퍼 크기 확인 및 조정
        #         if not hasattr(self, 'buffer') or self.buffer is None or self.buffer_size < NeedBufSize:
        #             self.buffer = (c_ubyte * NeedBufSize)()
        #             self.buffer_size = NeedBufSize
                
        #         # 타임아웃 시간을 1초로 설정하여 안정성 확보
        #         ret = self.camera.MV_CC_GetOneFrameTimeout(self.buffer, self.buffer_size, stFrameInfo, 1000)
                
        #         if ret == MV_OK:
        #             # 성공적으로 프레임 획득
        #             frame_data = np.ctypeslib.as_array(self.buffer)
        #             frame_data = frame_data[:stFrameInfo.nHeight * stFrameInfo.nWidth].reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    
        #             # 픽셀 포맷 처리
        #             if stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
        #                 frame = cv2.cvtColor(frame_data, cv2.COLOR_BayerBG2RGB)
        #             elif stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
        #                 frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
        #             else:
        #                 frame = frame_data
                    
        #             return frame
        #         else:
        #             # 실패시 짧은 대기 후 재시도
        #             time.sleep(0.01)
            
        #     return None
        # except Exception as e:
        #     print(f"프레임 획득 중 예외 발생: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return None
        
        try:
            # 프레임 획득 시도
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            
            # 버퍼 크기 확인
            stPayloadSize = MVCC_INTVALUE_EX()
            ret_temp = self.camera.MV_CC_GetIntValueEx("PayloadSize", stPayloadSize)
            if ret_temp != MV_OK:
                print(f"페이로드 크기 가져오기 실패: {ret_temp}")
                return None
            
            NeedBufSize = int(stPayloadSize.nCurValue)
            
            # 버퍼 크기 확인 및 조정
            if not hasattr(self, 'buffer') or not self.buffer or self.buffer_size < NeedBufSize:
                self.buffer = (c_ubyte * NeedBufSize)()
                self.buffer_size = NeedBufSize
                print(f"버퍼 크기 조정: {NeedBufSize}")
            
            # 타임아웃 시간 증가 (1000ms -> 5000ms)
            ret = self.camera.MV_CC_GetOneFrameTimeout(self.buffer, self.buffer_size, stFrameInfo, 1000)
            
            if ret == MV_OK:
                #print(f"프레임 획득 성공: {stFrameInfo.nWidth}x{stFrameInfo.nHeight}")
                # 성공적으로 프레임 획득
                frame_data = np.ctypeslib.as_array(self.buffer).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                
                # 픽셀 포맷 변환
                if stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
                    frame = cv2.cvtColor(frame_data, cv2.COLOR_BayerBG2RGB)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
                    frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)  # 그레이스케일을 RGB로 변환
                else:
                    print(f"알 수 없는 픽셀 형식: {stFrameInfo.enPixelType}")
                    frame = frame_data  # 기본 프레임 반환
                
                # 배열 형태 확인
                #print(f"변환된 프레임 형태: {frame.shape}, 타입: {frame.dtype}") 타입 볼때 열기
                
                return frame
            else:
                print(f"프레임 획득 실패: 오류 코드 {ret}")
                return None
                
        except Exception as e:
            print(f"프레임 획득 중 예외 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def disconnect(self):
        if not self.connected:
            print("MVSCamera: Camera not connected")
            return

        try:
            if self.grabbing:
                self.stop_acquisition()  # Ensure grabbing stops

            ret = self.camera.MV_CC_CloseDevice()
            if ret != MV_OK:
                print(f"MVSCamera: Failed to close device: {ToHexStr(ret)}")
            
            self.camera.MV_CC_DestroyHandle()
            self.connected = False
            print("MVSCamera: Camera disconnected")
        except Exception as e:
            print(f"MVSCamera disconnect error: {e}")

    def set_trigger(self, mode):
        if not self.connected:
            print("Camera not connected")
            return False

        try:
            if not mode:
                ret = self.camera.MV_CC_SetEnumValue("TriggerMode", 0)
            else:
                ret = self.camera.MV_CC_SetEnumValue("TriggerMode", 1)
                ret = self.camera.MV_CC_SetEnumValue("TriggerSource", 0)  # Line0
            
            return ret == MV_OK
        except Exception as e:
            print(f"Set trigger mode error: {e}")
            return False

    def set_trigger_source(self, source):
        if not self.connected:
            print("Camera not connected")
            return False
        
        try:
            source_value = 7  # Default to software trigger
            if source == "Software":
                source_value = 7
            elif source == "Line0":
                source_value = 0
            elif source == "Line1":
                source_value = 1
            elif source == "Line2":
                source_value = 2
            elif source == "Line3":
                source_value = 3

            ret = self.camera.MV_CC_SetEnumValue("TriggerSource", source_value)
            if ret != MV_OK:
                print(f"Failed to set trigger source: {ToHexStr(ret)}")
                return False

            print(f"Trigger source set to {source}")
            return True
        except Exception as e:
            print(f"Set trigger source error: {e}")
            return False

    def TriggerSoftwareExecute(self):
        if not self.connected:
            print("Camera not connected")
            return False

        try:
            ret = self.camera.MV_CC_SetCommandValue("TriggerSoftware")
            if ret != MV_OK:
                print(f"Failed to execute software trigger: {ToHexStr(ret)}")
                return False

            print("Software trigger executed")
            return True
        except Exception as e:
            print(f"Software trigger error: {e}")
            return False

    def set_fps(self, fps):
        if not self.connected:
            print("Camera is not connected")
            return False
        try:
            ret = self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
            if ret != MV_OK:
                print(f"Failed to set FPS: {ToHexStr(ret)}")
                return False
            print(f"FPS set to {fps}")
            return True
        except Exception as e:
            print(f"Set FPS error: {e}")
            return False

    def set_exposure(self, exposure):
        if not self.connected:
            print("Camera is not connected")
            return False
        try:
            ret = self.camera.MV_CC_SetFloatValue("ExposureTime", exposure)
            if ret != MV_OK:
                print(f"Failed to set exposure: {ToHexStr(ret)}")
                return False
            print(f"Exposure set to {exposure}")
            return True
        except Exception as e:
            print(f"Set exposure error: {e}")
            return False

    def set_gain(self, gain):
        if not self.connected:
            print("Camera is not connected")
            return False
        try:
            ret = self.camera.MV_CC_SetFloatValue("Gain", gain)
            if ret != MV_OK:
                print(f"Failed to set gain: {ToHexStr(ret)}")
                return False
            print(f"Gain set to {gain}")
            return True
        except Exception as e:
            print(f"Set gain error: {e}")
            return False

# Novitec Camera 클래스
class NovitecCamera:
    def __init__(self, serial_number):
        
        self.serial_number = serial_number
        self.connected = False
        self.grabbing = False

    def connect(self):
        # Novitec API를 통해 카메라 연결
        self.api = NovitecCameraAPIWrapper()  # Novitec API 인스턴스 생성
        result = self.api.connect_by_serial_number(self.serial_number)
        if result.errCode == 0:
            self.connected = True

             
            return self.connected
        else:
            self.connected = False
            print(f"Failed to connect to Novitec camera: {result.errMessage}")
            return self.connected
    
    def configure(self):
        if not self.connected:
            return {}

        fps = self.api.get_feature_value_float("AcquisitionFrameRate")[1]
        exposure = self.api.get_feature_value_int("ExposureTime")[1]
        trigger_mode = bool(self.api.get_feature_value_bool("TriggerMode")[1])
         
        options = {
            "fps": fps,
            "exposure": exposure,
            "trigger_mode": trigger_mode
        }
        return options

    def start_acquisition(self):

        time.sleep(3)
        if not self.connected:
            print("[❌] Camera not connected")
            return False

        if self.grabbing:
            print("[⚠️] Camera is already streaming")
            return True

        ret = self.api.start()
        if ret.errCode != 0:
            print(f"[❌] Starting the stream failed: {ret.errMessage}")
            self.grabbing = False
            return False
        else:
            self.grabbing = True
            print("[✅] Stream started successfully.")

        return self.grabbing

    
    def get_frame(self):
        if not self.connected:
            print("⚠️ Camera not connected")
            return None

        if not self.grabbing:
            print("⚠️ Camera is not grabbing frames")
            return None

        try:
            result, image = self.api.get_image()

            if result.errCode != 0:
                print(f"❌ Failed to retrieve image: {result.errMessage}")
                return None

            # 포인터와 데이터 크기 체크
            if image is None or image.data is None or image.dataSize == 0:
                print("⚠️ No valid image data received")
                return None

            if image.payloadType == 1:  # Raw 이미지 타입
                byte_data = ctypes.string_at(image.data, image.dataSize)
                cv_img = np.frombuffer(byte_data, dtype=np.uint8).reshape((image.height, image.width))
                return cv2.cvtColor(cv_img, cv2.COLOR_BayerGB2RGB)
            elif image.payloadType == 5:  # JPEG 이미지 타입
                byte_data = ctypes.string_at(image.data, image.dataSize)
                jpeg_data = np.frombuffer(byte_data, dtype=np.uint8)
                return cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
            return None
        except Exception as e:
            print(f"⚠️ Exception while processing image: {e}")
            return None

    def stop_acquisition(self):
        # Novitec API 스트림 종료
        if not self.grabbing:
            return True
            
        try:
            result = self.api.stop()
            if result.errCode != 0:
                print(f"Failed to stop Novitec camera stream: {result.errMessage}")
                return False
            self.grabbing = False
            return True
        except Exception as e:
            print(f"Stop acquisition error: {e}")
            return False

    def disconnect(self):
        # Novitec API 연결 해제
        if not self.connected:
            return
            
        try:
            self.stop_acquisition()
            result = self.api.disconnect()
            if result.errCode == 0:
                self.connected = False
                print("Novitec Camera disconnected successfully.")
                return True
            else:
                print(f"Failed to disconnect Novitec camera: {result.errMessage}")
                return False
        except Exception as e:
            print(f"Disconnect error: {e}")
            return False

    def set_trigger(self, mode):
        if not self.connected:
            print("Camera not connected")
            return False
            
        try:
            result = self.api.set_feature_value_bool("TriggerMode", bool(mode))
            if result.errCode != 0:
                print(f"Failed to set trigger mode: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Set trigger mode error: {e}")
            return False

    def set_trigger_source(self, source):
        if not self.connected:
            print("Camera not connected")
            return False
            
        try:
            result = self.api.set_feature_value_enum("TriggerSource", source)
            if result.errCode != 0:
                print(f"Failed to set trigger source: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Set trigger source error: {e}")
            return False

    def TriggerSoftwareExecute(self):
        if not self.connected:
            print("Camera not connected")
            return False
            
        try:
            result = self.api.execute_feature("TriggerSoftware")
            if result.errCode != 0:
                print(f"Failed to execute software trigger: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Software trigger error: {e}")
            return False

    def set_fps(self, fps):
        if not self.connected:
            return False
            
        try:
            result = self.api.set_feature_value_float("AcquisitionFrameRate", float(fps))
            if result.errCode != 0:
                print(f"Failed to set frame rate: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Set fps error: {e}")
            return False

    def set_exposure(self, exposure):
        if not self.connected:
            return False
            
        try:
            result = self.api.set_feature_value_int("ExposureTime", int(exposure))
            if result.errCode != 0:
                print(f"Failed to set exposure: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Set exposure error: {e}")
            return False

    def set_gain(self, gain):
        if not self.connected:
            return False
            
        try:
            result = self.api.set_feature_value_float("Gain", float(gain))
            if result.errCode != 0:
                print(f"Failed to set gain: {result.errMessage}")
                return False
            return True
        except Exception as e:
            print(f"Set gain error: {e}")
            return False

class ImageCamera:
    def __init__(self, file_path):
        self.file_path = file_path
        self.frame = cv2.imread(self.file_path)
        self.connected = self.frame is not None
        self.grabbing = False  # 추가: grabbing 상태 플래그

    def connect(self):
        if not self.connected and self.file_path:
            self.frame = cv2.imread(self.file_path)
            self.connected = self.frame is not None
        return self.connected
    
    def configure(self):
        if not self.connected:
            return {}

        configs = {
            "fps": 30,
            "min_fps": 1,
            "max_fps": 60,
            "exposure": 100,
            "min_exposure": 1,
            "max_exposure": 100,
            "trigger_mode": False,
            "gain": 1.0
        }
        return configs

    def start_acquisition(self):
        if not self.connected:
            print("ImageCamera: Camera not connected")
            return False
        self.grabbing = True
        print("ImageCamera: Camera started grabbing")
        return True

    def get_frame(self):
        if not self.grabbing or self.frame is None:
            return None
        return self.frame.copy()

    def stop_acquisition(self):
        if self.grabbing:
            self.grabbing = False
            print("ImageCamera: Camera stopped grabbing")
        return True

    def disconnect(self):
        self.frame = None
        self.connected = False
        self.grabbing = False
        print("ImageCamera: Camera disconnected")
        return True
    
    def set_trigger(self, mode):
        return True  # 지원하지 않음

    def set_trigger_source(self, source):
        return True  # 지원하지 않음

    def TriggerSoftwareExecute(self):
        return True  # 지원하지 않음

    def set_fps(self, fps):
        return True  # 지원하지 않음

    def set_exposure(self, exposure):
        return True  # 지원하지 않음

    def set_gain(self, gain):
        return True  # 지원하지 않음

class VideoCamera:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        self.connected = self.cap.isOpened()
        self.grabbing = False  # 추가: grabbing 상태 플래그

    def connect(self):
        if not self.connected and self.file_path:
            self.cap = cv2.VideoCapture(self.file_path)
            self.connected = self.cap.isOpened()
        return self.connected

    def configure(self):
        if not self.connected:
            return {}

        configs = {
            "fps": round(self.cap.get(cv2.CAP_PROP_FPS), 1),
            "min_fps": 1,
            "max_fps": 60,
            "exposure": round(self.cap.get(cv2.CAP_PROP_EXPOSURE), -2),
            "min_exposure": round(-11),
            "max_exposure": round(-2),
            "trigger_mode": False,
            "gain": 1.0
        }
        return configs

    def start_acquisition(self):
        if not self.connected:
            print("VideoCamera: Camera not connected")
            return False
        self.grabbing = True
        print("VideoCamera: Camera started grabbing")
        return True

    def get_frame(self):
        if not self.grabbing:
            print("VideoCamera: Camera not grabbing")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset if video ends
                ret, frame = self.cap.read()
                if not ret:
                    return None
            return frame
        except Exception as e:
            print(f"VideoCamera frame error: {e}")
            return None

    def stop_acquisition(self):
        if self.grabbing:
            self.grabbing = False
            print("VideoCamera: Camera stopped grabbing")
        return True

    def disconnect(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.connected = False
        self.grabbing = False
        print("VideoCamera: Camera disconnected")
        return True

    def set_trigger(self, mode):
        return True  # 지원하지 않음

    def set_trigger_source(self, source):
        return True  # 지원하지 않음

    def TriggerSoftwareExecute(self):
        return True  # 지원하지 않음

    def set_fps(self, fps):
        if self.cap and self.cap.isOpened():
            return self.cap.set(cv2.CAP_PROP_FPS, fps)
        return False

    def set_exposure(self, exposure):
        if self.cap and self.cap.isOpened():
            return self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        return False

    def set_gain(self, gain):
        return True  # 대부분의 비디오 파일에서 지원하지 않음

class USBCamera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.connected = False
        self.grabbing = False  # 추가: grabbing 상태 플래그

    def connect(self):
        try:
            self.cap = cv2.VideoCapture(self.index + cv2.CAP_DSHOW) #30fps 가 나옴=>그나마 제일 낫군
            self.connected = self.cap.isOpened()
            if not self.connected:
                print(f"[USBCamera] Failed to open camera index {self.index}")
            return self.connected
        except Exception as e:
            print(f"[USBCamera] Connection error: {e}")
            self.connected = False
            return False

    def configure(self):
        if not self.connected:
            return {}

        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            configs = {
                "fps": fps if fps > 0 else 30.0,
                "min_fps": 1,
                "max_fps": 60,
                "exposure": exposure,
                "min_exposure": -11,
                "max_exposure": -2,
                "trigger_mode": False,
                "gain": 1.0
            }
            return configs
        except Exception as e:
            print(f"Configure error: {e}")
            return {
                "fps": 30.0,
                "exposure": -6,
                "gain": 1.0,
                "trigger_mode": False
            }

    def start_acquisition(self):
        if not self.connected:
            self.connect()
            print("USBCamera: Camera re connected")
             
        if not self.connected:
            print("USBCamera: Camera not connected")
            return False
            
        self.grabbing = True  # 프레임 수집 시작
        print("USBCamera: Camera started grabbing")
        return True

    def get_frame(self):
        if not self.grabbing:
            print("USBCamera: Camera not grabbing")
            return None
            
        try:
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            print(f"Get frame error: {e}")
            return None

    def stop_acquisition(self):
        if self.cap and self.grabbing:    
            self.grabbing = False  # 프레임 수집 중단
            print("USBCamera: Camera stopped grabbing")
        return True

    def disconnect(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.connected = False
            self.grabbing = False
            print("USBCamera: Camera disconnected")
            return True
        except Exception as e:
            print(f"Disconnect error: {e}")
            return False

    def set_fps(self, fps):
        if self.cap and self.connected:
            try:
                return self.cap.set(cv2.CAP_PROP_FPS, fps)
            except Exception as e:
                print(f"Set FPS error: {e}")
                return False
        return False

    def set_exposure(self, exposure):
        if self.cap and self.connected:
            try:
                return self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            except Exception as e:
                print(f"Set exposure error: {e}")
                return False
        return False

    def set_gain(self, value):
        # OpenCV에서 USB 카메라의 게인 제어는 종종 지원되지 않습니다.
        return True
        
    def set_trigger(self, value):
        # 대부분의 일반 USB 카메라는 하드웨어 트리거를 지원하지 않습니다.
        return True

    def set_trigger_source(self, value):
        # 대부분의 일반 USB 카메라는 트리거 소스 설정을 지원하지 않습니다.
        return True

    def TriggerSoftwareExecute(self):
        # 대부분의 일반 USB 카메라는 소프트웨어 트리거를 지원하지 않습니다.
        return True