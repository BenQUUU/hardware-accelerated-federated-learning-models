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

        # Inicjalizacja list z metrykami
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        self.power_usage = []

        if self.device_type == 'cuda' and NVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _sample_metrics(self):
        if self.device_type in ['cpu', 'cuda']:
            while self.running:
                # 1. Pobieramy obciążenie CPU raz, by użyć go ew. do estymacji
                current_cpu = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(current_cpu)
                self.ram_usage.append(psutil.virtual_memory().percent)

                if self.device_type == 'cuda' and NVML_AVAILABLE:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.vram_usage.append((info.used / info.total) * 100)
                    self.gpu_usage.append(rates.gpu)

                    # Pobór mocy dla CUDA (NVML zwraca miliwaty)
                    try:
                        power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                        self.power_usage.append(power_w)
                    except pynvml.NVMLError:
                        self.power_usage.append(0.0)
                else:
                    self.vram_usage.append(0.0)
                    self.gpu_usage.append(0.0)

                    # Estymacja mocy dla CPU / Maliny (Zakładamy 3W idle, 9W max)
                    estimated_power_w = 3.0 + (9.0 - 3.0) * (current_cpu / 100.0)
                    self.power_usage.append(estimated_power_w)

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

                    # Pomiary mocy z Jetsona (INA3221) z zabezpieczeniem struktury
                    try:
                        power_w = jetson.power['tot']['power'] / 1000.0
                        self.power_usage.append(power_w)
                    except Exception:
                        self.power_usage.append(0.0)

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
            "avg_vram_percent": avg(self.vram_usage),
            "avg_power_w": avg(self.power_usage)
        }