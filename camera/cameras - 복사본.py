# cameras.py
import cv2
import numpy as np
from threading import Event, Thread
import time
from PySide6.QtCore import QTimer

import cv2
import numpy as np
from threading import Event, Thread
import time
from PySide6.QtCore import QTimer

# cameras.py
import cv2
import numpy as np
from threading import Event, Thread
import time
from PySide6.QtCore import QTimer
from camera.NovitecCameraAPIWrapper import NovitecCameraAPIWrapper, CError, Image
import sys
import cv2
import numpy as np
import ctypes
from DictKey import DictKey


# cameras.py

import cv2
import numpy as np
from threading import Event, Thread
import time
from PySide6.QtCore import QTimer
 
from genicam import gentl  # 필요에 따라 import


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
    def __init__(self, harvester_manager=None):
        self.harvester_manager = harvester_manager
        self.camera = None
        self.camera_type = None
        self.camera_source = None
        self.timer = None
        self._stop_event = Event()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time  # 마지막 fps 업데이트 시간
        self.last_fps = 0.0
        self.camera_config = {}
        self.realtimeFps = 0

        # self.exposure = 0 
        # self.gain = 0
        # self.trigger_mode = False

        # self.min_fps = 0
        # self.max_fps = 0
        # self.min_exposure = 0
        # self.max_exposure = 0

    

    def create_camera(self, camera_type, source):
        self.camera_type = camera_type
        self.camera_source = source
        if camera_type == DictKey.GenICam and self.harvester_manager:
            self.camera = GenICamCamera(index=int(self.camera_source), harvester_manager=self.harvester_manager)
        elif camera_type == DictKey.USBCam:
            self.camera = USBCamera(index=int(self.camera_source))
        elif camera_type == DictKey.IMGFile:
            self.camera = ImageCamera(self.camera_source)
        elif camera_type == DictKey.VODFile:
            self.camera = VideoCamera(self.camera_source)
        elif camera_type == DictKey.NoviCam:
            self.camera = NovitecCamera(serial_number=self.camera_source)  # Novitec 카메라 인스턴스 생성
        else:
            raise ValueError("Unknown camera type")

    def connect(self):
        if self.camera is None:
            self.create_camera(self.camera_type, self.camera_source)
        if self.camera:
            return self.camera.connect()
        return False
    
    def configure_camera(self):
        if self.camera:
            self.camera_config = self.camera.configure()
            return self.camera_config
        return {}
        
    def start_acquisition(self, update_frame_callback):
        self.frame_count = 0
        self.start_time = time.time()
        
        if self.camera_type ==DictKey.GenICam or self.camera_type == DictKey.NoviCam:
            self._stop_event.clear()

            # threading.Thread(target=self.acquire_images, daemon=True).start()
            self.thread = Thread(target=self._acquire_frames_async, args=(update_frame_callback,), daemon=True)
            self.thread.start()
        else:
            self.timer = QTimer()
            self.timer.timeout.connect(lambda: self._acquire_frame(update_frame_callback))
            self.timer.start(int(1000 / 30))  # 30 FPS

    def _acquire_frames_async(self, update_frame_callback):
        while not self._stop_event.is_set():
            frame = self.get_frame()
            if frame is not None:
                self.realtimeFps = self._calculate_fps()
                update_frame_callback(frame, self.realtimeFps)
            # time.sleep(1.0 / 30)

    def _acquire_frame(self, update_frame_callback):
        frame = self.get_frame()
        if frame is not None:
            fps = self._calculate_fps()
            update_frame_callback(frame, fps)
    
    def _calculate_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # 1초가 지났을 때만 fps 계산
        if current_time - self.last_update_time >= 1.0:
            self.last_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time  # 프레임 측정 시간 갱신
            self.last_update_time = current_time  # 마지막 업데이트 시간 갱신

        return self.last_fps
 
    def get_frame(self):
        if self.camera:
            return self.camera.get_frame()
        return None

    def stop_acquisition(self):
        if self.camera_type ==DictKey.GenICam or self.camera_type == DictKey.NoviCam:
            self._stop_event.set()
            if self.camera:
                self.camera.stop_acquisition()
        elif self.timer:
            self.timer.stop()
            self.timer = None

    def disconnect(self):
        self.stop_acquisition()
        if self.camera:
            self.camera.disconnect()
            self.camera = None

    def set_trigger(self, mode):
        if self.camera:
            self.camera.set_trigger(mode)
            # self.configure_camera()
    
    def set_trigger_source(self, value):
        if self.camera:
            self.camera.set_trigger_source(value)
            # self.configure_camera()

    def TriggerSoftwareExecute(self):
        if self.camera:
            self.camera.TriggerSoftwareExecute()
            # self.configure_camera()

    def set_exposure(self, exposure):
        if self.camera:
            self.camera.set_exposure(exposure)
            # self.configure_camera()

    def set_fps(self, fps):
        if self.camera:
            self.camera.set_fps(fps)
            # self.configure_camera()

    def set_gain(self, gain):
        if self.camera:
            self.camera.set_gain(gain)
            # self.configure_camera()
    
    
    def get_trigger_mode(self):
        return self.camera_config.get("trigger_mode", None)
    
    def get_fps(self):
        return self.camera_config.get("fps", None)
    
    def get_exposure(self):
        return self.camera_config.get("exposure", None)
    
    def get_gain(self):
        return self.camera_config.get("gain", None)

    


    # def get_trigger_mode(self):
    #     return self.trigger_mode 
    
    
    
    # def get_fps(self):
    #     return self.fps
    
    # def get_exposure(self):
    #     return self.exposure
    
    # def get_gain(self):
    #     return self.gain

    
# Novitec Camera 클래스
class NovitecCamera:
    def __init__(self, serial_number):
        self.api = NovitecCameraAPIWrapper()  # Novitec API 인스턴스 생성
        self.serial_number = serial_number
        self.connected = False

    def connect(self):
        # Novitec API를 통해 카메라 연결
        result = self.api.connect_by_serial_number(self.serial_number)
        if result.errCode == 0:
            
            print(f"Novitec Camera connected with serial number: {self.serial_number}")
            
            ret = self.api.start()
            if ret.errCode != 0:
                print(f"Starting the stream failed: {ret.errMessage}")
                self.connected = False
                return False
            else:
                self.connected = True
                print("Stream started successfully.")

            return True
        else:
            print(f"Failed to connect to Novitec camera: {result.errMessage}")
            return False

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

    def get_frame(self):
        if not self.connected:
            return None
        
        # Novitec API를 통해 이미지 가져오기
        result, image = self.api.get_image()
        if result.errCode != 0:
            print(f"Failed to retrieve image: {result.errMessage}")
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

    def stop_acquisition(self):
        # Novitec API 스트림 종료
        result = self.api.stop()
        if result.errCode != 0:
            print(f"Failed to stop Novitec camera stream: {result.errMessage}")

    def disconnect(self):
        # Novitec API 연결 해제
        self.stop_acquisition()
        result = self.api.disconnect()
        if result.errCode == 0:
            self.connected = False
            print("Novitec Camera disconnected successfully.")
        else:
            print(f"Failed to disconnect Novitec camera: {result.errMessage}")

class GenICamCamera:
    def __init__(self, index=0, harvester_manager=None):
        self.harvester_manager = harvester_manager
        self.acquirer = None
        self.connected = False
        
        if self.harvester_manager:
            self.acquirer = self.harvester_manager.create_image_acquirer(index)
            self.connected = self.acquirer is not None
            self.stop_acquisition()

    def connect(self):
        if self.acquirer and not self.connected:
            try:
                self.acquirer.start_image_acquisition()
                self.connected = True
            except Exception as e:
                print("Failed to start acquisition:", e)
                self.connected = False
        return self.connected

    def configure(self):
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            configs = {
                "fps": round(node_map.AcquisitionFrameRate.value, 1),
                "min_fps": round(node_map.AcquisitionFrameRate.min, 1),
                "max_fps": round(node_map.AcquisitionFrameRate.max, 1),
                "exposure": round(node_map.ExposureTime.value, 1),
                "min_exposure": round(node_map.ExposureTime.min, 1),
                "max_exposure": round(node_map.ExposureTime.max, 1),
                "trigger_mode": node_map.TriggerMode.value == "On"
            }
            return configs
        else:
            return {}
        

    def get_frame(self):
        if not self.connected:
            return None

        try:
            # Fetch buffer with a specific timeout (e.g., 2000 milliseconds)
            # with self.acquirer.fetch_buffer() as buffer:
            with self.acquirer.fetch_buffer(timeout=10000) as buffer:
                component = buffer.payload.components[0]
                processed_img = RGB8Image(
                    component.width, 
                    component.height, 
                    component.data_format, 
                    component.data.copy()
                )
                # TriggerMode = self.acquirer.remote_device.node_map.TriggerMode.value
                # print(f"node_map.TriggerMode.value ={node_map.TriggerMode.value }")
                # print(f"component._buffer.image_offset={component._buffer.image_offset}")
                # print(f"component._buffer.payload_type={component._buffer.payload_type}")
                print(f"component._buffer.frame_id={component._buffer.frame_id}")
                return processed_img.image_data
        except gentl.TimeoutException:
            print("Warning: Frame fetch timed out, retrying...")
            return None  # Return None if a timeout occurs so the loop can retry

    def stop_acquisition(self):
        if self.acquirer:
            self.acquirer.stop_image_acquisition()
            

    def disconnect(self):
        self.stop_acquisition()
        if self.acquirer:
            self.acquirer.destroy()
            self.acquirer = None
            self.connected = False
    
    def set_trigger(self, value):
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            if value:
                node_map.TriggerMode.value = "On"
            else:
                node_map.TriggerMode.value = "Off"
           
    def set_trigger_source(self, value):
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            node_map.TriggerSource.value = value
            

    def TriggerSoftwareExecute(self):
        node_map = self.acquirer.remote_device.node_map
        node_map.TriggerSoftware.execute()

    def set_fps(self, value):
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            node_map.AcquisitionFrameRateEnable.value = True
            node_map.AcquisitionFrameRate.value = float(value)
            # self.fps = value

    def set_exposure(self, value):
        # GenICam 카메라의 노출 설정 코드 추가
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            node_map.ExposureTime.value = float(value)
            # self.exposure =  float(value)

    def set_gain(self, value):
        if self.acquirer:
            node_map = self.acquirer.remote_device.node_map
            node_map.Gain.value = float(value)
            # self.camera_config['gain'] = value     
    

class ImageCamera:
    def __init__(self, file_path):
        self.file_path = file_path
        self.frame = cv2.imread(self.file_path)
        self.connected = self.frame is not None

    def connect(self):
        return self.connected
    
    def configure(self):
        pass

    def get_frame(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def disconnect(self):
        self.frame = None
        self.connected = False
    
    def set_fps(self, fps):
        # 정적 이미지의 FPS는 설정할 수 없습니다.
        pass

    def set_exposure(self, exposure):
        # 정적 이미지의 노출 설정은 지원되지 않습니다.
        pass

class VideoCamera:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        self.connected = self.cap.isOpened()

    def connect(self):
        return self.connected

    def configure(self):
        pass

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Reset to the beginning if the video ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        if frame is not None:
            # Ensure BGR to RGB conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def disconnect(self):
        if self.cap:
            self.cap.release()
        self.connected = False

    def set_fps(self, fps):
        # 정적 이미지의 FPS는 설정할 수 없습니다.
        pass

    def set_exposure(self, exposure):
        # 정적 이미지의 노출 설정은 지원되지 않습니다.
        pass


class USBCamera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.connected = False

    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        self.connected = self.cap.isOpened()
        return self.connected

    def configure(self):
        pass
    
    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def disconnect(self):
        if self.cap:
            self.cap.release()
        self.connected = False

    def set_fps(self, fps):
        # USB 카메라의 FPS를 설정하는 코드를 추가
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def set_exposure(self, exposure):
        # USB 카메라의 노출을 설정하는 코드를 추가
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))