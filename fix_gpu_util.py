#!/usr/bin/env python3
"""Refresh a GPU stuck at 100% utilization with no memory or compute processes.

Usage: fix_gpu_util.py [GPU_INDEX] [--check]
Uses Python's standard library and libcuda; does not kill tasks or reset GPUs.
This is a workaround for the observed symptom, not a driver root-cause fix.
Exit codes: 0 = idle/recovered/check complete, 1 = failed, 2 = busy/skipped.
"""

import argparse
import csv
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


class BusyGPU(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    index: int
    uuid: str
    bus: str
    utilization: int
    memory: int
    pids: tuple


def smi(gpu, query):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), query, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))


def snapshot(gpu):
    rows = smi(gpu, "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used")
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("无法唯一识别目标 GPU，停止处理。")
    index, uuid, bus, utilization, memory = (value.strip() for value in rows[0])
    processes = smi(uuid, "--query-compute-apps=gpu_uuid,pid")
    pids = []
    for row in processes:
        if len(row) != 2 or row[0].strip() != uuid:
            raise RuntimeError("无法确认 GPU 计算进程列表，停止处理。")
        pids.append(int(row[1].strip()))
    # Unsupported/N/A counters must fail instead of being interpreted as zero.
    return Snapshot(int(index), uuid, bus, int(utilization), int(memory), tuple(pids))


def show(state):
    pids = ", ".join(map(str, state.pids)) or "无"
    print(
        f"GPU {state.index}: 利用率 {state.utilization}%，"
        f"显存 {state.memory} MiB，计算进程 {pids}", flush=True,
    )


def ensure_unused(state):
    if state.memory != 0 or state.pids:
        raise BusyGPU(f"GPU {state.index} 存在显存或计算进程占用，停止处理。")


def cuda_poke(uuid):
    # Check again inside the child just before CUDA initialization.
    state = snapshot(uuid)
    ensure_unused(state)
    if state.utilization != 100:
        print("利用率已发生变化，跳过 CUDA 操作。", flush=True)
        return
    lib = C.CDLL("libcuda.so.1")

    def invoke(name, types, *args):
        fn = getattr(lib, name)
        fn.argtypes = types
        fn.restype = C.c_int
        code = fn(*args)
        if code:
            details = []
            for getter in ("cuGetErrorName", "cuGetErrorString"):
                value = C.c_char_p()
                error_fn = getattr(lib, getter)
                error_fn.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
                error_fn.restype = C.c_int
                if error_fn(code, C.byref(value)) == 0 and value.value:
                    details.append(value.value.decode(errors="replace"))
            raise RuntimeError(f"{name}: CUDA 错误 {code} ({'; '.join(details)})")

    invoke("cuInit", [C.c_uint], 0)
    domain, bus, slot = state.bus.split(":")
    pci_bus = f"{int(domain, 16):04x}:{bus}:{slot}".encode("ascii")
    device = C.c_int()
    invoke("cuDeviceGetByPCIBusId", [C.POINTER(C.c_int), C.c_char_p],
           C.byref(device), pci_bus)
    context = C.c_void_p()
    invoke("cuCtxCreate_v2", [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
           C.byref(context), 0, device)
    pointer = C.c_uint64()
    size = 1024 * 1024
    try:
        invoke("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t],
               C.byref(pointer), size)
        try:
            invoke("cuMemsetD8_v2", [C.c_uint64, C.c_ubyte, C.c_size_t],
                   pointer, 0x5A, size)
            buffer = (C.c_ubyte * 64)()
            invoke("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t],
                   buffer, pointer, C.sizeof(buffer))
            invoke("cuCtxSynchronize", [])
            if bytes(buffer) != bytes([0x5A]) * 64:
                raise RuntimeError("CUDA 显存读写校验失败。")
        finally:
            invoke("cuMemFree_v2", [C.c_uint64], pointer)
    finally:
        invoke("cuCtxDestroy_v2", [C.c_void_p], context)
    print("CUDA 检查通过，1 MiB 测试显存和上下文已释放。", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="处理无计算任务、显存为零却显示 100% 利用率的 GPU。",
        epilog="示例：fix-gpu-util 0；fix-gpu-util 1 --check。仅按需运行，不常驻。",
    )
    parser.add_argument("gpu", type=int, nargs="?", default=0,
                        help="nvidia-smi 中的物理 GPU 编号，默认 0")
    parser.add_argument("--check", action="store_true", help="仅查看状态，不执行 CUDA 操作")
    parser.add_argument("--_cuda-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU 编号必须大于等于 0")
    if args._cuda_worker:
        cuda_poke(args._cuda_worker)
        return 0

    state = snapshot(args.gpu)
    show(state)
    if args.check:
        return 0
    ensure_unused(state)
    if state.utilization == 0:
        print("GPU 已空闲，无需处理。", flush=True)
        return 0
    if state.utilization != 100:
        print("不符合无任务却显示 100% 的症状，跳过处理。", flush=True)
        return 2

    time.sleep(1)
    state = snapshot(state.uuid)
    ensure_unused(state)
    if state.utilization != 100:
        show(state)
        print("状态已发生变化，跳过处理。", flush=True)
        return 0 if state.utilization == 0 else 2
    print("确认异常持续，执行轻量 CUDA 检查……", flush=True)
    # A stuck driver call must not block the CLI indefinitely.
    worker = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--_cuda-worker", state.uuid],
        timeout=20,
    )
    if worker.returncode:
        return worker.returncode if worker.returncode > 0 else 1

    consecutive_idle = 0
    for _ in range(8):
        time.sleep(1)
        state = snapshot(state.uuid)
        show(state)
        ensure_unused(state)
        consecutive_idle = consecutive_idle + 1 if state.utilization == 0 else 0
    if consecutive_idle < 3:
        raise RuntimeError("利用率未稳定恢复为 0%；本次方法未能解决。")
    print(f"GPU {state.index} 已恢复：利用率 0%，显存 0 MiB。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BusyGPU as exc:
        print(f"跳过：{exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    except subprocess.TimeoutExpired as exc:
        print(f"操作超时（{exc.timeout} 秒），停止处理。", file=sys.stderr, flush=True)
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr, flush=True)
        sys.exit(130)
