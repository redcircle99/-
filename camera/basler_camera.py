# basler_camera.py
# -*- coding: utf-8 -*-
import time
import traceback
from pypylon import pylon

class BaslerCamera:
    def __init__(self, index: int = 0):
        self.index = index
        self.camera = None
        self.connected = False
        self.grabbing = False
        self.device_info = None

    def connect(self):
        """카메라를 열고 feature 를 읽어들이며 연결 상태를 설정합니다."""
        try:
            # 이미 연결된 경우 처리
            if self.connected and self.camera is not None and self.camera.IsOpen():
                print("[BaslerCamera] Already connected")
                return True
                
            # 기존 연결이 있으면 정리
            if self.camera is not None:
                try:
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    if self.camera.IsOpen():
                        self.camera.Close()
                except:
                    pass
                self.camera = None
                
            # 디바이스 리스트 조회
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices:
                print("[BaslerCamera] No camera found")
                return False
                
            # 유효한 인덱스인지 확인
            if self.index >= len(devices):
                print(f"[BaslerCamera] Invalid index {self.index}. Available cameras: {len(devices)}")
                return False
                
            # 카메라 생성 및 연결
            self.device_info = devices[self.index]
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(self.device_info))
            
            # 카메라 열기
            self.camera.Open()
            
            # 연결 확인
            if not self.camera.IsOpen():
                print("[BaslerCamera] Failed to open camera")
                return False
            
            # 기본 설정 적용
            try:
                # 연속 모드로 설정 (기본)
                if hasattr(self.camera, "AcquisitionMode"):
                    self.camera.AcquisitionMode.SetValue("Continuous")
                
                # 트리거 모드 비활성화 (기본)
                if hasattr(self.camera, "TriggerMode"):
                    self.camera.TriggerMode.SetValue("Off")
                    
                # 프레임 레이트 활성화
                if hasattr(self.camera, "AcquisitionFrameRateEnable"):
                    self.camera.AcquisitionFrameRateEnable.SetValue(True)
                    
            except Exception as e:
                print(f"[BaslerCamera] Warning: Could not apply default settings: {e}")
            
            self.connected = True
            self.grabbing = False
            print(f"[BaslerCamera] Connected to {self.device_info.GetModelName()} (SN: {self.device_info.GetSerialNumber()})")
            return True
            
        except Exception as e:
            print(f"[BaslerCamera] Connection error: {str(e)}")
            traceback.print_exc()
            self.connected = False
            if self.camera:
                try:
                    self.camera.Close()
                except:
                    pass
                self.camera = None
            return False

    def configure(self) -> dict:
        """카메라 current 설정 및 지원 범위를 dict 로 반환합니다."""
        if not self.connected or not self.camera or not self.camera.IsOpen():
            print("[BaslerCamera] Cannot configure: camera not connected")
            return {}

        try:
            config = {}
            
            # ExposureTime 가져오기
            if hasattr(self.camera, "ExposureTime"):
                try:
                    min_e = float(self.camera.ExposureTime.GetMin())
                    max_e = float(self.camera.ExposureTime.GetMax())
                    cur_e = float(self.camera.ExposureTime.GetValue())
                    config.update({
                        "min_exposure": min_e,
                        "max_exposure": max_e,
                        "exposure": cur_e
                    })
                except Exception as e:
                    print(f"[BaslerCamera] Failed to get ExposureTime: {e}")
                    config.update({
                        "exposure": 1000,
                        "min_exposure": 100,
                        "max_exposure": 10000
                    })
            else:
                config.update({
                    "exposure": 1000,
                    "min_exposure": 100,
                    "max_exposure": 10000
                })

            # Gain 가져오기
            if hasattr(self.camera, "Gain"):
                try:
                    min_g = float(self.camera.Gain.GetMin())
                    max_g = float(self.camera.Gain.GetMax())
                    cur_g = float(self.camera.Gain.GetValue())
                    config.update({
                        "min_gain": min_g,
                        "max_gain": max_g,
                        "gain": cur_g
                    })
                except Exception as e:
                    print(f"[BaslerCamera] Failed to get Gain: {e}")
                    config.update({
                        "gain": 0.0,
                        "min_gain": 0.0,
                        "max_gain": 10.0
                    })
            else:
                config.update({
                    "gain": 0.0,
                    "min_gain": 0.0,
                    "max_gain": 10.0
                })

            # FrameRate 가져오기
            if hasattr(self.camera, "AcquisitionFrameRate"):
                try:
                    min_f = float(self.camera.AcquisitionFrameRate.GetMin())
                    max_f = float(self.camera.AcquisitionFrameRate.GetMax())
                    cur_f = float(self.camera.AcquisitionFrameRate.GetValue())
                    config.update({
                        "min_fps": min_f,
                        "max_fps": max_f,
                        "fps": cur_f
                    })
                except Exception as e:
                    print(f"[BaslerCamera] Failed to get FrameRate: {e}")
                    config.update({
                        "fps": 30.0,
                        "min_fps": 1.0,
                        "max_fps": 60.0
                    })
            else:
                config.update({
                    "fps": 30.0, 
                    "min_fps": 1.0,
                    "max_fps": 60.0
                })

            # TriggerMode 가져오기
            if hasattr(self.camera, "TriggerMode"):
                try:
                    tm = self.camera.TriggerMode.GetValue()
                    config["trigger_mode"] = (tm == "On")
                except Exception as e:
                    print(f"[BaslerCamera] Failed to get TriggerMode: {e}")
                    config["trigger_mode"] = False
            else:
                config["trigger_mode"] = False

            return config
            
        except Exception as e:
            print(f"[BaslerCamera] Configuration error: {str(e)}")
            traceback.print_exc()
            # 기본 설정값 반환
            return {
                "fps": 30.0,
                "min_fps": 1.0,
                "max_fps": 60.0,
                "exposure": 10000,
                "min_exposure": 100,
                "max_exposure": 100000,
                "gain": 0.0,
                "min_gain": 0.0,
                "max_gain": 10.0,
                "trigger_mode": False
            }

    def start_acquisition(self) -> bool:
        """프레임 수집을 시작합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot start acquisition: camera not connected")
            return False
            
        try:
            # 이미 grabbing 중인 경우 먼저 정지
            if self.camera.IsGrabbing():
                print("[BaslerCamera] Already grabbing, stopping first...")
                self.camera.StopGrabbing()
                time.sleep(0.1)  # 정지 대기
            
            # grabbing 시작
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.grabbing = True
            print("[BaslerCamera] Started grabbing")
            return True
        except Exception as e:
            print(f"[BaslerCamera] Start acquisition error: {str(e)}")
            traceback.print_exc()
            self.grabbing = False
            return False

    def get_frame(self, timeout=2000):
        """가장 최신 프레임을 가져와 numpy array로 반환합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot get frame: camera not connected")
            return None
            
        if not self.grabbing:
            if not self.start_acquisition():
                print("[BaslerCamera] Cannot get frame: failed to start acquisition")
                return None
        
        result = None
        try:
            # Timeout을 조절하여 성능과 안정성 사이의 균형을 유지
            result = self.camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
            
            # result가 유효한지 먼저 확인
            if result is None:
                print("[BaslerCamera] Failed to retrieve result: result is None")
                return None
                
            # result가 유효한 데이터를 포함하는지 확인
            try:
                if not result.IsValid():
                    print("[BaslerCamera] Failed to grab frame: result is not valid")
                    if result:
                        result.Release()
                    return None
            except Exception as e:
                print(f"[BaslerCamera] Error checking result validity: {e}")
                if result:
                    result.Release()
                return None
            
            # grab이 성공했는지 확인
            try:
                if not result.GrabSucceeded():
                    print(f"[BaslerCamera] Failed to grab frame: {result.GetErrorCode()}, {result.GetErrorDescription()}")
                    result.Release()
                    return None
            except Exception as e:
                print(f"[BaslerCamera] Error checking grab success: {e}")
                if result:
                    result.Release()
                return None
                
            # 성공적으로 이미지 획득
            try:
                img = result.Array
                result.Release()
                return img
            except Exception as e:
                print(f"[BaslerCamera] Error accessing image array: {e}")
                if result:
                    result.Release()
                return None
                
        except pylon.TimeoutException:
            print("[BaslerCamera] Timeout while grabbing frame")
            if result:
                result.Release()
            return None
        except Exception as e:
            print(f"[BaslerCamera] Error grabbing frame: {str(e)}")
            traceback.print_exc()
            if result:
                try:
                    result.Release()
                except:
                    pass  # Release 실패해도 계속 진행
            return None

    def stop_acquisition(self):
        """프레임 수집을 중지합니다."""
        if not self.connected or not self.grabbing:
            return True
            
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self.grabbing = False
            print("[BaslerCamera] Stopped grabbing")
            return True
        except Exception as e:
            print(f"[BaslerCamera] Stop acquisition error: {str(e)}")
            traceback.print_exc()
            return False

    def disconnect(self):
        """카메라 연결을 해제합니다."""
        if not self.connected:
            return True
            
        try:
            self.stop_acquisition()
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
            self.connected = False
            print("[BaslerCamera] Disconnected")
            return True
        except Exception as e:
            print(f"[BaslerCamera] Disconnect error: {str(e)}")
            traceback.print_exc()
            return False

    # --- 공통 인터페이스 맞춤 메서드들 ---
    def set_trigger(self, mode: bool):
        """트리거 모드를 설정합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot set trigger: camera not connected")
            return False
            
        try:
            if hasattr(self.camera, "TriggerMode"):
                self.camera.TriggerMode.SetValue("On" if mode else "Off")
                print(f"[BaslerCamera] Trigger mode set to {'On' if mode else 'Off'}")
                return True
            else:
                print("[BaslerCamera] TriggerMode not supported")
                return False
        except Exception as e:
            print(f"[BaslerCamera] Set trigger error: {str(e)}")
            traceback.print_exc()
            return False

    def set_trigger_source(self, source: str):
        """트리거 소스를 설정합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot set trigger source: camera not connected")
            return False
            
        try:
            if hasattr(self.camera, "TriggerSource"):
                # source 값 검증
                valid_sources = ["Software", "Line1", "Line2", "Line3", "Line4"]
                if source not in valid_sources:
                    source = "Software"  # 기본값
                    
                self.camera.TriggerSource.SetValue(source)
                print(f"[BaslerCamera] Trigger source set to {source}")
                return True
            else:
                print("[BaslerCamera] TriggerSource not supported")
                return False
        except Exception as e:
            print(f"[BaslerCamera] Set trigger source error: {str(e)}")
            traceback.print_exc()
            return False

    def TriggerSoftwareExecute(self):
        """소프트웨어 트리거를 실행합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot execute trigger: camera not connected")
            return False
            
        try:
            if hasattr(self.camera, "TriggerSoftware"):
                self.camera.TriggerSoftware.Execute()
                print("[BaslerCamera] Software trigger executed")
                return True
            else:
                print("[BaslerCamera] TriggerSoftware not supported")
                return False
        except Exception as e:
            print(f"[BaslerCamera] Software trigger error: {str(e)}")
            traceback.print_exc()
            return False

    def set_exposure(self, value: float):
        """노출 시간을 설정합니다. (개선된 버전)"""
        if not self.connected:
            print("[BaslerCamera] Cannot set exposure: camera not connected")
            return False
            
        try:
            if not hasattr(self.camera, "ExposureTime"):
                print("[BaslerCamera] ExposureTime not supported")
                return False
            
            # 카메라가 grabbing 중인지 확인
            was_grabbing = self.grabbing and self.camera.IsGrabbing()
            
            # grabbing 중이면 잠시 정지
            if was_grabbing:
                try:
                    self.camera.StopGrabbing()
                    time.sleep(0.2)  # 완전히 정지될 때까지 대기
                except Exception as e:
                    print(f"[BaslerCamera] Warning: Could not stop grabbing: {e}")
            
            # 범위 내 값으로 제한
            min_exp = self.camera.ExposureTime.GetMin()
            max_exp = self.camera.ExposureTime.GetMax()
            if value < min_exp:
                value = min_exp
                print(f"[BaslerCamera] Exposure value clamped to minimum: {min_exp}")
            elif value > max_exp:
                value = max_exp
                print(f"[BaslerCamera] Exposure value clamped to maximum: {max_exp}")
            
            # 노출 시간 설정 시도 (재시도 메커니즘 포함)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.camera.ExposureTime.SetValue(value)
                    
                    # 설정 확인
                    actual_value = self.camera.ExposureTime.GetValue()
                    if abs(actual_value - value) > 1.0:  # 1μs 오차 허용
                        print(f"[BaslerCamera] Warning: Requested {value}, got {actual_value}")
                    
                    print(f"[BaslerCamera] Exposure set to {actual_value}")
                    
                    # grabbing이 활성화되어 있었다면 다시 시작
                    if was_grabbing:
                        try:
                            time.sleep(0.1)  # 설정이 적용될 시간
                            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        except Exception as e:
                            print(f"[BaslerCamera] Warning: Could not restart grabbing: {e}")
                            self.grabbing = False
                    
                    return True
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[BaslerCamera] Exposure setting attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(0.5)  # 재시도 전 대기
                    else:
                        print(f"[BaslerCamera] Set exposure error after {max_retries} attempts: {str(e)}")
                        traceback.print_exc()
                        
                        # grabbing 상태 복원 시도
                        if was_grabbing:
                            try:
                                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                            except:
                                self.grabbing = False
                        
                        return False
                        
        except Exception as e:
            print(f"[BaslerCamera] Set exposure error: {str(e)}")
            traceback.print_exc()
            return False

    def set_gain(self, value: float):
        """게인을 설정합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot set gain: camera not connected")
            return False
            
        try:
            if hasattr(self.camera, "Gain"):
                # 범위 내 값으로 제한
                min_gain = self.camera.Gain.GetMin()
                max_gain = self.camera.Gain.GetMax()
                if value < min_gain:
                    value = min_gain
                elif value > max_gain:
                    value = max_gain
                    
                self.camera.Gain.SetValue(value)
                print(f"[BaslerCamera] Gain set to {value}")
                return True
            else:
                print("[BaslerCamera] Gain not supported")
                return False
        except Exception as e:
            print(f"[BaslerCamera] Set gain error: {str(e)}")
            traceback.print_exc()
            return False

    def set_fps(self, value: float):
        """프레임 레이트를 설정합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot set FPS: camera not connected")
            return False
            
        try:
            # 프레임 레이트 설정을 위해 필요한 단계들
            if hasattr(self.camera, "AcquisitionFrameRateEnable"):
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                
            if hasattr(self.camera, "AcquisitionFrameRate"):
                # 범위 내 값으로 제한
                min_fps = self.camera.AcquisitionFrameRate.GetMin()
                max_fps = self.camera.AcquisitionFrameRate.GetMax()
                if value < min_fps:
                    value = min_fps
                elif value > max_fps:
                    value = max_fps
                    
                self.camera.AcquisitionFrameRate.SetValue(value)
                print(f"[BaslerCamera] Frame rate set to {value}")
                return True
            else:
                print("[BaslerCamera] AcquisitionFrameRate not supported")
                return False
        except Exception as e:
            print(f"[BaslerCamera] Set FPS error: {str(e)}")
            traceback.print_exc()
            return False

    def get_width(self):
        """카메라 이미지 폭을 반환합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot get width: camera not connected")
            return 0
            
        try:
            if hasattr(self.camera, "Width"):
                width = int(self.camera.Width.GetValue())
                return width
            else:
                print("[BaslerCamera] Width feature not supported")
                return 0
        except Exception as e:
            print(f"[BaslerCamera] Get width error: {str(e)}")
            traceback.print_exc()
            return 0

    def get_height(self):
        """카메라 이미지 높이를 반환합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot get height: camera not connected")
            return 0
            
        try:
            if hasattr(self.camera, "Height"):
                height = int(self.camera.Height.GetValue())
                return height
            else:
                print("[BaslerCamera] Height feature not supported")
                return 0
        except Exception as e:
            print(f"[BaslerCamera] Get height error: {str(e)}")
            traceback.print_exc()
            return 0

    def get_pixel_format(self):
        """카메라 픽셀 포맷을 반환합니다."""
        if not self.connected:
            print("[BaslerCamera] Cannot get pixel format: camera not connected")
            return ""
            
        try:
            if hasattr(self.camera, "PixelFormat"):
                pixel_format = str(self.camera.PixelFormat.GetValue())
                return pixel_format
            else:
                print("[BaslerCamera] PixelFormat feature not supported")
                return ""
        except Exception as e:
            print(f"[BaslerCamera] Get pixel format error: {str(e)}")
            traceback.print_exc()
            return ""