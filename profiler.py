import os
import threading
import time
import subprocess
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

# Intel/AMD RAPL: sprzetowy licznik energii CPU x86 -- WYLACZNIE Linux.
# Na Windows ta sciezka nie istnieje (RAPL_AVAILABLE = False), wiec moc CPU
# pozostaje swiadomie niezmierzona. Zgodnie z zalozeniem badawczym wezly
# 'bez akceleracji' zyja na Linuxie (Raspberry Pi 5 / Jetson CPU-only), gdzie
# moc jest mierzona realnie -- RAPL to tylko opcjonalny bonus dla x86/Linux.
RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
RAPL_AVAILABLE = os.path.exists(RAPL_ENERGY_PATH)


def _read_rapl_energy_uj():
    with open(RAPL_ENERGY_PATH, "r") as f:
        return int(f.read())


def _read_rapl_max_uj():
    try:
        with open(RAPL_MAX_PATH, "r") as f:
            return int(f.read())
    except Exception:
        return 0


def _read_rpi_power():
    """Realny pobor mocy plytki Raspberry Pi 5 z ukladu PMIC.

    'vcgencmd pmic_read_adc' zwraca napiecie (_V) i prad (_A) per szyna zasilajaca.
    Moc calkowita = suma V*I po wszystkich szynach. To pomiar sprzetowy, nie estymacja.
    """
    out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)
    volts = {}
    amps = {}
    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        name_part, val_part = line.split("=", 1)
        name = name_part.split()[0]  # np. VDD_CORE_A / VDD_CORE_V
        try:
            val = float(val_part.strip().rstrip("AV"))
        except ValueError:
            continue
        if name.endswith("_A"):
            amps[name[:-2]] = val
        elif name.endswith("_V"):
            volts[name[:-2]] = val

    return sum(volts[k] * amps[k] for k in amps if k in volts)


class HardwareProfiler:
    def __init__(self, device_type):
        self.device_type = device_type
        self.running = False
        self.thread = None

        # Initialize metric lists
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        self.power_usage = []

        if self.device_type == 'cuda' and NVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _sample_metrics(self):
        if self.device_type == 'jetson':
            self._sample_jetson()
            return

        # Priming: pierwszy odczyt cpu_percent(interval=None) zawsze zwraca 0.0.
        psutil.cpu_percent(interval=None)

        # Stan poczatkowy licznika RAPL (tylko sciezka 'cpu' na x86/Linux).
        prev_energy = None
        prev_ts = None
        rapl_max = 0
        if self.device_type == 'cpu' and RAPL_AVAILABLE:
            prev_energy = _read_rapl_energy_uj()
            prev_ts = time.time()
            rapl_max = _read_rapl_max_uj()

        while self.running:
            current_cpu = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(current_cpu)
            self.ram_usage.append(psutil.virtual_memory().percent)

            if self.device_type == 'cuda' and NVML_AVAILABLE:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.vram_usage.append((info.used / info.total) * 100)
                self.gpu_usage.append(rates.gpu)

                # Realny pobor mocy GPU (NVML zwraca miliwaty) -- dziala na Windows i Linux
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.power_usage.append(power_w)
                except pynvml.NVMLError:
                    self.power_usage.append(0.0)

            elif self.device_type == 'rpi':
                self.vram_usage.append(0.0)
                self.gpu_usage.append(0.0)
                try:
                    self.power_usage.append(_read_rpi_power())
                except Exception:
                    self.power_usage.append(0.0)

            else:  # 'cpu' (x86)
                self.vram_usage.append(0.0)
                self.gpu_usage.append(0.0)

                if RAPL_AVAILABLE and prev_energy is not None:
                    # x86/Linux: realny pomiar z licznika energii RAPL
                    now = time.time()
                    energy = _read_rapl_energy_uj()
                    dt = now - prev_ts
                    delta = energy - prev_energy
                    if delta < 0:  # licznik energii przewinal sie
                        delta += rapl_max
                    power_w = (delta / 1e6) / dt if dt > 0 else 0.0
                    self.power_usage.append(power_w)
                    prev_energy, prev_ts = energy, now
                else:
                    # Windows/x86 bez RAPL => moc CPU swiadomie niezmierzona (0.0),
                    # zamiast zmyslonej estymacji. Ten wezel nie jest baseline'em energii.
                    self.power_usage.append(0.0)

            time.sleep(0.5)

    def _sample_jetson(self):
        if not JTOP_AVAILABLE:
            print("[Profiler Error] The jtop library is not installed on this device.")
            return
        with jtop() as jetson:
            while self.running and jetson.ok():
                self.cpu_usage.append(psutil.cpu_percent(interval=None))

                ram_percent = psutil.virtual_memory().percent
                self.ram_usage.append(ram_percent)
                self.vram_usage.append(ram_percent)  # Orin: pamiec zunifikowana

                stats = jetson.stats
                self.gpu_usage.append(stats.get('GPU', 0))

                # Realny pomiar mocy z Jetsona (INA3221) z ochrona struktury
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
