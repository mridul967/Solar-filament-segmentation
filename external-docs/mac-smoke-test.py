import platform, subprocess, shutil, sys

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"

print("=" * 60)
print("SYSTEM")
print("=" * 60)
print(f"macOS version   : {platform.mac_ver()[0]}")
print(f"Architecture    : {platform.machine()}")
print(f"Chip / CPU      : {sh('sysctl -n machdep.cpu.brand_string')}")
print(f"CPU cores       : {sh('sysctl -n hw.ncpu')} "
      f"(perf: {sh('sysctl -n hw.perflevel0.physicalcpu')}, "
      f"eff: {sh('sysctl -n hw.perflevel1.physicalcpu')})")

ram_bytes = int(sh('sysctl -n hw.memsize') or 0)
print(f"Unified memory  : {ram_bytes / 1024**3:.1f} GB  <-- this is your 'VRAM' pool")

print("\n" + "=" * 60)
print("GPU HARDWARE")
print("=" * 60)
gpu = sh("system_profiler SPDisplaysDataType")
for line in gpu.splitlines():
    if any(k in line for k in ("Chipset", "Type", "Vendor", "Total Number of Cores", "Metal")):
        print(line.strip())

print("\n" + "=" * 60)
print("LIVE MEMORY PRESSURE")
print("=" * 60)
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"Total     : {vm.total / 1024**3:.1f} GB")
    print(f"Used      : {vm.used / 1024**3:.1f} GB")
    print(f"Available : {vm.available / 1024**3:.1f} GB")
    print(f"Percent   : {vm.percent}%")
except ImportError:
    print("pip install psutil for live memory stats")

print("\n" + "=" * 60)
print("PYTORCH / MPS BACKEND")
print("=" * 60)
try:
    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"MPS built       : {torch.backends.mps.is_built()}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")
    print(f"CUDA available  : {torch.cuda.is_available()}  (always False on Mac)")

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print(f"Device          : {dev}")
        print(f"Recommended max working set: "
              f"{torch.mps.recommended_max_memory() / 1024**3:.2f} GB")
        print(f"Currently allocated by torch: "
              f"{torch.mps.current_allocated_memory() / 1024**3:.3f} GB")

        # Quick compute benchmark
        import time
        n = 4096
        a = torch.randn(n, n, device=dev, dtype=torch.float32)
        b = torch.randn(n, n, device=dev, dtype=torch.float32)
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(20):
            c = a @ b
        torch.mps.synchronize()
        dt = (time.time() - t0) / 20
        tflops = (2 * n**3) / dt / 1e12
        print(f"\nMatmul {n}x{n} FP32 : {dt*1000:.2f} ms  ->  {tflops:.2f} TFLOPS")
        print(f"Peak torch memory  : "
              f"{torch.mps.current_allocated_memory() / 1024**3:.3f} GB")
    else:
        print("MPS not available — you'd be on CPU only.")
except ImportError:
    print("PyTorch not installed: pip install torch")

print("\n" + "=" * 60)
print("DISK")
print("=" * 60)
total, used, free = shutil.disk_usage("/")
print(f"Total: {total/1024**3:.0f} GB | Used: {used/1024**3:.0f} GB | Free: {free/1024**3:.0f} GB")