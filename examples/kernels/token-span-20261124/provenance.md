# Provenance

Facts about the setup that no other file here records. Lines in code blocks were copied from command output during the session of 2026-11-24; nothing else was kept of those commands.

## VM and toolchain

- Boot id `41119ca1-3818-4901-a08e-de3c859ff473` (`/proc/sys/kernel/random/boot_id`), kernel `6.12.93-android16-6-g3a7d1771d492-ab16496487-4k`, 7 vCPUs, `MemTotal: 10873112 kB`.
- `gcc (Debian 14.2.0-19) 14.2.0`. The shim was built with `gcc -O2 -Wall -Wextra -shared -fPIC vkshim.c -o vkshim.so -ldl`, without warnings.

`vulkaninfo`, filtered to the device names, timestamp periods and queue families:

```
GPU0:
	deviceName        = PowerVR C-Series CXTP-48-1536 MC1
	timestampPeriod                                 = 26.0417
	queueProperties[0]:
		queueCount                  = 2
		queueFlags                  = QUEUE_GRAPHICS_BIT | QUEUE_COMPUTE_BIT | QUEUE_TRANSFER_BIT | QUEUE_SPARSE_BINDING_BIT | QUEUE_PROTECTED_BIT
		timestampValidBits          = 64
GPU1:
	deviceName        = llvmpipe (LLVM 19.1.7, 128 bits)
	timestampPeriod                                 = 1
	queueProperties[0]:
		queueCount                  = 1
		queueFlags                  = QUEUE_GRAPHICS_BIT | QUEUE_COMPUTE_BIT | QUEUE_TRANSFER_BIT | QUEUE_SPARSE_BINDING_BIT
		timestampValidBits          = 64
```

## Shim

| Build | Source sha256 | `vkshim.so` sha256 | Used for |
| --- | --- | --- | --- |
| first | `0dcf2632295cdd216bc506ede29ecbd14b79862016e343f5367045ffc4c67f8e` | `32fbcc51107ed92c17a1546e55b8bb6ba8bb70bb02ca28d0ac5d8161d761404f` | `rehearsal/rehearsal1-lvp.*` (turned itself off: no `vkGetDeviceQueue2` wrapper) |
| second | `277159b6aa0fba980b069606494955db03db0a62d5bbd60623c2e6e1e665077d` | `996500581def9b8ae2a09d15abbcf6d0803e7de8d6e930473bf4257784e00ecc` | `rehearsal/lvp.*` and every run in `gpu-runs/` |

After the measurements, `guest/probes/vkshim.c` changed in comments and in two lines that do not affect these runs. `TSLOTS` can now be set at build time. `vkGetDeviceQueue` and `vkGetDeviceQueue2` no longer set the device that `VKSHIM_GPU_TS` uses; the token mode never used it. That version has not been built.

## lavapipe rehearsals

In the VM, from `~/.local/share/SpiralPixcel/kernel-study/token-span-v2-20261124` (`$D`):

```
R=~/.local/share/SpiralPixcel/runtimes/llama-b11960-pvr-decode-r7
M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf
systemd-run --user --scope --quiet -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M \
  env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json GGML_VK_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$R \
  LD_PRELOAD=$D/vkshim.so VKSHIM_TOKEN_TS=1 VKSHIM_OUT=$D/lvp.json VKSHIM_TOKEN_OUT=$D/lvp.spans \
  GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 \
  GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 \
  timeout 600 $R/llama-bench -m $M -p 0 -n 64 -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 \
  --repack 1 --no-warmup -o json > $D/lvp.stdout 2> $D/lvp.err
```

`llama-bench --list-devices` under the same `VK_DRIVER_FILES` and `GGML_VK_VISIBLE_DEVICES` listed only `Vulkan0: llvmpipe (LLVM 19.1.7, 128 bits)`. The first rehearsal ran the same command with the first build (its outputs were renamed `rehearsal1-*`).

## Runtimes

Each run's stderr (`gpu-runs/*.err`) names the `libggml-vulkan.so` it loaded: `runtimes/llama-b11960-pvr-decode-r6/` for the `-r6-` runs and `runtimes/llama-b11960-pvr-decode-r7/` for the `-r7-` runs. The libraries were not hashed on 2026-11-24; the phone dropped off USB before that could be done. Their build-time hashes are `b1985926c4a4c27b647f300a5a5a8d64c9466ec96f89f55c029302fb31b878fc` (r6, [`../../inference/decode-study/runtime/r6/`](../../inference/decode-study/runtime/r6/)) and `59c4dbf0c52b1e822e062ad1b12a0720817f6e81a79720b40e35834d6a3931b7` (r7, [`../../inference/q50-study/runtime/r7/`](../../inference/q50-study/runtime/r7/)).

## Phone state

At 10:57 JST and again just before the smoke run (about 11:11 JST): `Thermal Status: 0`, and the Terminal activity on top:

```
    topResumedActivity=ActivityRecord{183587351 u0 com.android.virtualization.terminal/.new2.ui.MainActivity t1428}
```

During the fourth or fifth run of `ab1` (about 11:14 JST), `top` and `dumpsys power`:

```
 3044 u0_a303      20   0  23G 1.4G 1.4G S 42.4   9.6   9:59.31 crosvm_debian --extended-status ...
20196 audioserver  20   0  11G  25M  18M S 36.3   0.1   4:14.99 audioserver
4.71 3.95 3.07 3/11616 4196
  mWakefulness=Dreaming
  mIsPowered=true
```

At 11:15:07 JST, during the seventh run:

```
  mWakefulness=Dreaming
  mStayOn=false
  mLastWakeTime=97285223 (265922 ms ago)
  mMaximumScreenOffTimeoutFromDeviceAdmin=60000 (enforced=true)
  mScreenOffTimeoutSetting=1800000
mCurrentDream=DreamRecord{..., name=ComponentInfo{com.google.assistant.hubui/com.google.assistant.hubui.dream.EverydayClockDreamService}...
```

The last wake was at about 11:10:41 JST. With the 60 s cap enforced, the screensaver could not have started before about 11:11:41, after the smoke run (`smoke-r7-111127`, 11:11:27) had begun. Whether the screen was still on for its whole decode was not recorded.
