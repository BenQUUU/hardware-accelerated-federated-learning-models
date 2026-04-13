import threading
import time
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    NVML_AVAILABLE = False

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

class HardwareProfiler:
    def __init__(self, device_type):
        self.device_type = device_type 
        self.running = False
        self.thread = None
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.vram_usage = []

        if self.device_type == 'cuda' and NVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

    def _sample_metrics(self):
        if self.device_type in ['cpu', 'cuda']:
            while self.running:
                self.cpu_usage.append(psutil.cpu_percent(interval=None))
                self.ram_usage.append(psutil.virtual_memory().percent)

                if self.device_type == 'cuda' and NVML_AVAILABLE:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.vram_usage.append((info.used / info.total) * 100)
                    self.gpu_usage.append(rates.gpu)
                else:
                    self.vram_usage.append(0.0)
                    self.gpu_usage.append(0.0)
                time.sleep(0.5)

        elif self.device_type == 'jetson':
            if not JTOP_AVAILABLE:
                print("[Błąd Profilera] Biblioteka jtop nie jest zainstalowana na tym urządzeniu.")
                return
            with jtop() as jetson:
                while self.running and jetson.ok():
                    self.cpu_usage.append(psutil.cpu_percent(interval=None))
                    ram_percent = psutil.virtual_memory().percent
                    self.ram_usage.append(ram_percent)
                    self.vram_usage.append(ram_percent)
                    
                    stats = jetson.stats
                    self.gpu_usage.append(stats.get('GPU', 0))
                    time.sleep(0.5)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample_metrics)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        def avg(lst): return sum(lst) / len(lst) if lst else 0.0
        return {
            "avg_cpu_percent": avg(self.cpu_usage),
            "avg_ram_percent": avg(self.ram_usage),
            "avg_gpu_percent": avg(self.gpu_usage),
            "avg_vram_percent": avg(self.vram_usage)
        }