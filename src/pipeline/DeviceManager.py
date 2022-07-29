
class DeviceManager:
    """
    handles Pytorch GPU <--> CPU data & model movement
    """

    def __init__(self, gpu_device_name : str = "cuda") -> None:
        self.gpu_device_name = gpu_device_name

    def move_to_gpu(self, obj):
        return obj.to(device=self.gpu_device_name)

    def move_to_cpu(self, obj):
        return obj.cpu()
