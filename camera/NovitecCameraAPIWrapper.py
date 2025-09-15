import ctypes
from ctypes import *
from typing import Tuple


class CError(ctypes.Structure):
    _fields_ = [
        ("errCode", ctypes.c_int),
        ("errMessage", ctypes.c_char_p)
    ]


class Image(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("bpp", ctypes.c_int),
        ("timestamp", ctypes.c_uint),
        ("frameNum", ctypes.c_uint),
        ("payloadType", ctypes.c_int),
        ("pixelFormat", ctypes.c_int),
        ("dataSize", ctypes.c_uint),
        ('data', POINTER(c_ubyte)),
    ]


class DiscoverDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("modelName", ctypes.c_char * 32),
        ("serialNumber", ctypes.c_char * 16),
        ("firmwareVersion", ctypes.c_char * 32),
        ("macAddress", ctypes.c_ubyte * 6),
        ("ipAddress", ctypes.c_ubyte * 4),
        ("subnetMask", ctypes.c_ubyte * 4),
        ("defaultGateway", ctypes.c_ubyte * 4),
        ("isNetworkCompatible", ctypes.c_bool)
    ]


class NovitecCameraAPIWrapper:
    def __init__(self):
        self.lib = windll.LoadLibrary("./libs/NOVITECCAMERAAPIC.dll")
        self.lib.Discover.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.Discover.restype = CError

        self.lib.GetDeviceInfo.argtypes = [ctypes.c_uint, ctypes.POINTER(DiscoverDeviceInfo)]
        self.lib.GetDeviceInfo.restype = CError

        self.lib.ConnectBySerialNumber.argtypes = [ctypes.c_char_p]
        self.lib.ConnectBySerialNumber.restype = CError

        self.lib.Start.argtypes = []
        self.lib.Start.restype = CError

        self.lib.Stop.argtypes = []
        self.lib.Stop.restype = CError

        self.lib.Disconnect.argtypes = []
        self.lib.Disconnect.restype = CError

        self.lib.GetImage.argtypes = [ctypes.POINTER(Image)]
        self.lib.GetImage.restype = CError

        self.lib.SetFeatureValueInt.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.SetFeatureValueInt.restype = CError

        self.lib.SetFeatureValueFloat.argtypes = [ctypes.c_char_p, ctypes.c_float]
        self.lib.SetFeatureValueFloat.restype = CError

        self.lib.SetFeatureValueBool.argtypes = [ctypes.c_char_p, ctypes.c_bool]
        self.lib.SetFeatureValueBool.restype = CError

        self.lib.SetFeatureValueEnum.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.SetFeatureValueEnum.restype = CError

        self.lib.SetFeatureValueString.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.SetFeatureValueString.restype = CError

        self.lib.ExecuteFeature.argtypes = [ctypes.c_char_p]
        self.lib.ExecuteFeature.restype = CError

        self.lib.WriteMemory.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        self.lib.WriteMemory.restype = CError

        self.lib.GetFeatureValueInt.argtypes = [ctypes.c_char_p, POINTER(ctypes.c_int)]
        self.lib.GetFeatureValueInt.restype = CError

        self.lib.GetFeatureValueFloat.argtypes = [ctypes.c_char_p, POINTER(ctypes.c_float)]
        self.lib.GetFeatureValueFloat.restype = CError

        self.lib.GetFeatureValueBool.argtypes = [ctypes.c_char_p, POINTER(ctypes.c_bool)]
        self.lib.GetFeatureValueBool.restype = CError

        self.lib.GetFeatureValueEnum.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.GetFeatureValueEnum.restype = CError

        self.lib.GetFeatureValueString.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.GetFeatureValueString.restype = CError

        self.lib.GetFeatureMinMaxValueInt.argtypes = [ctypes.c_char_p, POINTER(ctypes.c_int), POINTER(ctypes.c_int)]
        self.lib.GetFeatureMinMaxValueInt.restype = CError

        self.lib.GetFeatureMinMaxValueFloat.argtypes = [ctypes.c_char_p, POINTER(ctypes.c_float), POINTER(ctypes.c_float)]
        self.lib.GetFeatureMinMaxValueFloat.restype = CError

    def discover(self) -> Tuple[CError, int]:
        num_of_device = ctypes.c_uint()
        result = self.lib.Discover(ctypes.byref(num_of_device))
        return result, num_of_device.value

    def get_device_info(self, device_index: int) -> Tuple[CError, DiscoverDeviceInfo]:
        device_info = DiscoverDeviceInfo()
        result = self.lib.GetDeviceInfo(ctypes.c_uint(device_index), ctypes.byref(device_info))
        return result, device_info

    # NovitecCameraAPIWrapper.py의 connect_by_serial_number 수정 예시
    def connect_by_serial_number(self, serial_number):
        # serial_number가 bytes인지 str인지 확인 후 처리
        if isinstance(serial_number, str):
            serial_number = serial_number.encode("utf-8")
        result = self.lib.ConnectBySerialNumber(serial_number)
        return result


    def start(self) -> CError:
        result = self.lib.Start()
        return result

    def stop(self) -> CError:
        result = self.lib.Stop()
        return result

    def disconnect(self) -> CError:
        result = self.lib.Disconnect()
        return result

    def get_image(self) -> Tuple[CError, Image]:
        # 🛠 이미지 객체를 명확하게 초기화
        image = Image()
        image.width = 0
        image.height = 0
        image.bpp = 0
        image.timestamp = 0
        image.frameNum = 0
        image.payloadType = 0
        image.pixelFormat = 0
        image.dataSize = 0
        image.data = ctypes.POINTER(ctypes.c_ubyte)()  # NULL 포인터로 초기화

        result = self.lib.GetImage(ctypes.byref(image))

        # 🛠 데이터 포인터 및 크기 검증
        if image.data is None or image.dataSize == 0:
            print("⚠️ No valid image data received")
            return result, None

        return result, image


    def set_feature_value_int(self, feature_name: str, value: int) -> CError:
        result = self.lib.SetFeatureValueInt(feature_name.encode("utf-8"), ctypes.c_int(value))
        return result

    def set_feature_value_bool(self, feature_name: str, value: bool) -> CError:
        result = self.lib.SetFeatureValueBool(feature_name.encode("utf-8"), ctypes.c_bool(value))
        return result

    def set_feature_value_float(self, feature_name: str, value: float) -> CError:
        result = self.lib.SetFeatureValueFloat(feature_name.encode("utf-8"), ctypes.c_float(value))
        return result

    def set_feature_value_enum(self, feature_name: str, value: str) -> CError:
        result = self.lib.SetFeatureValueEnum(feature_name.encode("utf-8"), value.encode("utf-8"))
        return result

    def set_feature_value_string(self, feature_name: str, value: str) -> CError:
        result = self.lib.SetFeatureValueString(feature_name.encode("utf-8"), value.encode("utf-8"))
        return result

    def execute_feature(self, feature_name: str) -> CError:
        result = self.lib.ExecuteFeature(feature_name.encode("utf-8"))
        return result
    
    def write_memory(self, address: int, data, length: int) -> CError:
        if isinstance(data, int):
            data_c = ctypes.c_int(data)
            data_ptr = ctypes.byref(data_c)
        elif isinstance(data, float):
            data_c = ctypes.c_float(data)
            data_ptr = ctypes.byref(data_c)
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
            data_c = ctypes.create_string_buffer(data_bytes)
            length = len(data_bytes)
            data_ptr = data_c
        elif isinstance(data, bytes):
            data_c = ctypes.create_string_buffer(data)
            data_ptr = data_c
        else:
            result = CError()
            result.errCode = -1
            result.errorMessage = b"Unsupported data type"
            return result

        result = self.lib.WriteMemory(ctypes.c_uint(address), data_ptr, ctypes.c_uint(length))
        return result

    def get_feature_value_int(self, feature_name: str) -> Tuple[CError, int]:
        c_int_value = ctypes.c_int()
        result = self.lib.GetFeatureValueInt(feature_name.encode("utf-8"), ctypes.byref(c_int_value))
        return result, c_int_value.value

    def get_feature_value_bool(self, feature_name: str) -> Tuple[CError, bool]:
        c_bool_value = ctypes.c_bool()
        result = self.lib.GetFeatureValueBool(feature_name.encode("utf-8"), ctypes.byref(c_bool_value))
        return result, c_bool_value.value

    def get_feature_value_float(self, feature_name: str) -> Tuple[CError, float]:
        c_float_value = ctypes.c_float()
        result = self.lib.GetFeatureValueFloat(feature_name.encode("utf-8"), ctypes.byref(c_float_value))
        return result, c_float_value.value

    def get_feature_value_enum(self, feature_name: str) -> Tuple[CError, str]:
        symbolic_value = ctypes.create_string_buffer(128)
        result = self.lib.GetFeatureValueEnum(feature_name.encode("utf-8"), symbolic_value)
        return result, symbolic_value.value.decode('utf-8').strip()

    def get_feature_value_string(self, feature_name: str) -> Tuple[CError, str]:
        string_value = ctypes.create_string_buffer(256)
        result = self.lib.GetFeatureValueString(feature_name.encode("utf-8"), string_value)
        return result, string_value.value.decode('utf-8').strip()

    def get_feature_min_max_value_int(self, feature_name: str) -> Tuple[CError, int, int]:
        c_min = ctypes.c_int()
        c_max = ctypes.c_int()
        result = self.lib.GetFeatureMinMaxValueInt(feature_name.encode("utf-8"), ctypes.byref(c_min), ctypes.byref(c_max))
        return result, c_min.value, c_max.value

    def get_feature_min_max_value_float(self, feature_name: str) -> Tuple[CError, float, float]:
        c_min = ctypes.c_float()
        c_max = ctypes.c_float()
        result = self.lib.GetFeatureMinMaxValueFloat(feature_name.encode("utf-8"), ctypes.byref(c_min), ctypes.byref(c_max))
        return result, c_min.value, c_max.value
