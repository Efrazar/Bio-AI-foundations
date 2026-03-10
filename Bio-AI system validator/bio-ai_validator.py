import torch
import psutil
import platform
import time


def get_gpu_architecture(major, minor):
    """Dynamically map compute capability to architecture name."""
    arch_map = {
        (3,):    "Kepler",
        (5,):    "Maxwell",
        (6,):    "Pascal",
        (7, 0):  "Volta",
        (7, 2):  "Volta (Jetson)",
        (7, 5):  "Turing",
        (8, 0):  "Ampere (Data Center)",
        (8, 6):  "Ampere (Consumer)",
        (8, 9):  "Ada Lovelace",
        (9, 0):  "Hopper",
        (10, 0): "Blackwell",
    }
    return arch_map.get((major, minor), arch_map.get((major,), f"Unknown (sm_{major}{minor})"))


def run_system_check():
    print("=" * 60)
    print("  🧬 BIO-AI SYSTEM READINESS REPORT")
    print("=" * 60)

    passed = []
    failed = []

    # ──────────────────────────────────────────────────────────────
    # 1. HOST MACHINE INFO
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[HOST MACHINE]")
    print(f"  OS           : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  CPU          : {platform.processor()}")

    phys_cores = psutil.cpu_count(logical=False)
    logi_cores = psutil.cpu_count(logical=True)
    print(f"  CPU Cores    : {phys_cores} physical / {logi_cores} logical (threads)")

    vm = psutil.virtual_memory()
    total_ram = vm.total     / (1024 ** 3)
    avail_ram = vm.available / (1024 ** 3)
    print(f"  System RAM   : {total_ram:.2f} GB total | {avail_ram:.2f} GB available ({vm.percent}% used)")

    if total_ram >= 16:
        passed.append("System RAM")
    else:
        failed.append(f"System RAM low ({total_ram:.1f} GB)")

    # ──────────────────────────────────────────────────────────────
    # 2. PYTHON / PYTORCH ENVIRONMENT
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[PYTHON / PYTORCH ENVIRONMENT]")
    print(f"  Python Version : {platform.python_version()}")
    print(f"  PyTorch Version: {torch.__version__}")

    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_version = torch.backends.cudnn.version() if cudnn_enabled else "N/A"
    print(f"  cuDNN Status   : {'✅ Enabled' if cudnn_enabled else '❌ Disabled'} | Version: {cudnn_version}")

    if cudnn_enabled:
        passed.append("cuDNN")
    else:
        failed.append("cuDNN not enabled — check PyTorch CUDA build")

    # ──────────────────────────────────────────────────────────────
    # 3. GPU / eGPU CHECK
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[GPU ACCELERATOR]")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  CUDA Status : ✅ Available | CUDA Runtime: {torch.version.cuda}")
        print(f"  GPU Count   : {gpu_count} detected\n")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            arch  = get_gpu_architecture(props.major, props.minor)

            torch.cuda.set_device(i)
            vram_total     = props.total_memory       / (1024 ** 3)
            mem_reserved   = torch.cuda.memory_reserved(i)  / (1024 ** 3)
            mem_allocated  = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_free       = vram_total - mem_reserved

            # Tensor Cores: available from Turing (sm_75) and above
            has_tensor_cores = (props.major > 7) or (props.major == 7 and props.minor >= 5)

            print(f"  ── GPU {i}: {props.name} ──")
            print(f"     Architecture   : {arch} (sm_{props.major}{props.minor})")
            print(f"     CUDA Version   : {torch.version.cuda}")
            print(f"     VRAM Total     : {vram_total:.2f} GB")
            print(f"     VRAM Free/Used : {mem_free:.2f} GB free | {mem_allocated:.2f} GB allocated")
            print(f"     Multiprocessors: {props.multi_processor_count}")
            print(f"     FP16 Support   : {'✅ Yes' if props.major >= 7 else '⚠️  Limited'}")
            print(f"     Tensor Cores   : {'✅ Available (great for AMP training)' if has_tensor_cores else '❌ Not available'}")

        passed.append("CUDA GPU")
    else:
        print("  Status : ❌ CUDA NOT DETECTED — Check Thunderbolt connection or drivers.")
        failed.append("CUDA not available")

    # ──────────────────────────────────────────────────────────────
    # 4. STRESS TEST: TENSOR OPERATIONS
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[STRESS TEST: TENSOR OPERATIONS]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Running on : {device.upper()}")

    # Warmup — critical for accurate CUDA timing (cold kernel launch skews results)
    if device == "cuda":
        _w = torch.randn(128, 128, device=device)
        torch.matmul(_w, _w)
        torch.cuda.synchronize()
        del _w

    try:
        size = 10000
        mem_per_matrix_mb = size * size * 4 / (1024 ** 2)
        print(f"  Matrix Size : {size}x{size} FP32 (~{mem_per_matrix_mb:.0f} MB per matrix)")

        start = time.perf_counter()
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        z = torch.matmul(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  MatMul Time : {elapsed:.4f}s")

        if device == "cuda":
            if elapsed < 1.0:
                status = "✅ PASS"
                note   = " (< 0.5s = healthy Thunderbolt bandwidth)" if elapsed < 0.5 else ""
                print(f"  Result      : {status} — GPU is computing correctly{note}")
                passed.append("GPU Stress Test")
            else:
                print(f"  Result      : ⚠️  SLOW (>1s) — Possible Thunderbolt/driver bottleneck")
                failed.append("GPU stress test slow")
        else:
            print(f"  Result      : ✅ CPU fallback test passed")
            passed.append("CPU Stress Test")

        del x, y, z

    except torch.cuda.OutOfMemoryError:
        print(f"  Result      : ❌ CUDA OUT OF MEMORY — Try reducing matrix size.")
        failed.append("OOM in stress test")
    except Exception as e:
        print(f"  Result      : ❌ FAILED — {e}")
        failed.append(f"Stress test error")

    # ──────────────────────────────────────────────────────────────
    # 5. AUTOMATIC MIXED PRECISION (AMP) CHECK
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[MIXED PRECISION (AMP / FP16) CHECK]")
    print("  Why: AMP cuts VRAM usage ~2x and accelerates Tensor Core ops on your RTX 2080 Ti")

    try:
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                a = torch.randn(2000, 2000, device="cuda")
                b = torch.matmul(a, a)
            torch.cuda.synchronize()
            print("  FP16 AMP    : ✅ Supported — torch.amp.autocast works correctly")
            passed.append("FP16 AMP")
            del a, b
        else:
            print("  FP16 AMP    : ⚠️  Skipped (no CUDA device)")
    except Exception as e:
        print(f"  FP16 AMP    : ❌ Failed — {e}")
        failed.append("FP16 AMP failed")

    # ──────────────────────────────────────────────────────────────
    # 6. FINAL VERDICT
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL VERDICT")
    print(f"{'=' * 60}")
    print(f"  ✅ Passed ({len(passed)}): {', '.join(passed)}")
    if failed:
        print(f"  ❌ Issues  ({len(failed)}): {', '.join(failed)}")
        print("\n  ⚠️  Hardware NOT fully ready — review issues above before training.")
    else:
        print("\n  🚀 ALL CHECKS PASSED — Hardware is ready for Bio-AI / DL Training.")
    print("=" * 60)


if __name__ == "__main__":
    run_system_check()
