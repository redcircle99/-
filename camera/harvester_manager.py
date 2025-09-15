from harvesters.core import Harvester

class HarvesterManager:
    _instance = None

    def __new__(cls, cti_path):
        if cls._instance is None:
            cls._instance = super(HarvesterManager, cls).__new__(cls)
            cls._instance._initialize(cti_path)
        return cls._instance

    def _initialize(self, cti_path):
        self.harvester = Harvester()
        self.harvester.add_cti_file(cti_path)
        self.harvester.update_device_info_list()
        self.device_info_list = self.harvester.device_info_list

    def get_device_info_list(self):
        return self.device_info_list

    def create_image_acquirer(self, index):
        if 0 <= index < len(self.device_info_list):
            return self.harvester.create_image_acquirer(list_index=index)
        else:
            raise IndexError("Index out of range for available GenICam devices")
