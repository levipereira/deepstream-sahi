# Installation Guide

This guide covers cloning the repository, launching the DeepStream container, and running the one-step installer.

## 1. Clone the Repository

```bash
git clone https://github.com/levipereira/deepstream-sahi.git
cd deepstream-sahi
```

## 2. Launch the DeepStream Container

Choose the command that matches your environment. Both DeepStream 8.x and 9.x are supported — the installer auto-detects the version.

| DeepStream | Container Image |
|------------|----------------|
| 9.x | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| 8.x | `nvcr.io/nvidia/deepstream:8.0-triton-multiarch` |

### WSL2 (Windows Subsystem for Linux)

```bash
# Replace the image tag with the desired DeepStream version
docker run \
    -it \
    --name deepstream-sahi \
    --net=host \
    --gpus all \
    -v `pwd`:/apps/deepstream-sahi \
    -w /apps/deepstream-sahi \
    -e CUDA_CACHE_DISABLE=0 \
    --device /dev/snd \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

> **Display output on WSL2:** The pipeline uses `fakesink` by default, so display is not required. If you want visual output with `--display`, there is a [known issue](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_WSL2_FAQ.html#while-using-pipelines-involving-nveglglessink-or-any-other-display-sinks-black-screen-is-coming-with-mesa-error-failed-to-attach-to-x11-shm-on-terminal) where `nveglglessink` shows a black screen on Ubuntu 24-based Docker on WSL2. The workaround is X11 forwarding over SSH as described in [Enabling Display on WSL2](#enabling-display-on-wsl2). Without a display server, the pipeline automatically falls back to `fakesink` with a warning.

### Native Linux (with display forwarding)

```bash
xhost +
docker run \
    -it \
    --name deepstream-sahi \
    --net=host \
    --ipc=host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v `pwd`:/apps/deepstream-sahi \
    -w /apps/deepstream-sahi \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

> **Re-attaching to an existing container:**
> If the container already exists, start and attach to it:
> ```bash
> docker start -ai deepstream-sahi
> ```

## 3. Run the Installer (inside the container)

A single command installs everything:

```bash
/apps/deepstream-sahi/install.sh
```

The script auto-detects the DeepStream version and performs four steps:

| Step | What it does |
|------|-------------|
| **1/4** | Installs DeepStream additional components (`user_additional_install.sh`) |
| **2/4** | Installs DeepStream Python bindings — **DS 8.x:** pre-built pyds 1.2.2 via `--version`; **DS 9.x:** built from source via `--build-bindings -r master` |
| **3/4** | Backs up original libs, copies sources, builds and installs plugins. **DS 8.x:** also builds the modified `nvdsinfer`; **DS 9.x:** uses the stock `nvdsinfer` |
| **4/4** | Installs Python test dependencies (`pandas`, `matplotlib`, `numpy`) into the `pyds` virtualenv |

Steps that have already been completed are detected and skipped automatically.

### Plugins-Only Mode

If the DeepStream dependencies and Python bindings are already set up and you only need to rebuild the SAHI plugins and modified libraries (e.g. after code changes), use:

```bash
/apps/deepstream-sahi/install.sh --plugins-only
```

This runs only step 3 (build + install), skipping steps 1, 2, and 4.

### Installed Libraries

| Library | Description |
|---------|-------------|
| `libnvds_infer.so` | Modified nvdsinfer with smart engine caching |
| `libnvds_infer_yolo.so` | YOLO custom bounding-box parser |
| `libnvdsgst_sahipreprocess.so` | SAHI dynamic-slice pre-process plugin |
| `libnvdsgst_sahipostprocess.so` | SAHI GreedyNMM post-process plugin |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DS_ROOT` | `/opt/nvidia/deepstream/deepstream` | DeepStream installation root |
| `CUDA_VER` | auto-detected from `nvcc` | CUDA version (e.g. `12.6`, `13.1`) |
| `PYDS_VERSION` | `1.2.2` (DS 8.x only) | DeepStream Python bindings version. Ignored on DS 9.x (bindings are built from source) |

## 4. Download Test Videos

The test videos are not included in the repository. Download them from Google Drive and place them in `python_test/videos/`:

**[Download test videos](https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4)**

| File | Size | Scene |
|------|------|-------|
| `aerial_crowding_01.mp4` | 274.6 MB | Dense pedestrian crowd |
| `aerial_crowding_02.mp4` | 20.1 MB | Very dense crowd with motorcycles |
| `aerial_vehicles.mp4` | 8.5 MB | Dense vehicle traffic |

```bash
# Place the downloaded .mp4 files in:
/apps/deepstream-sahi/python_test/videos/
```

> You can also use your own videos — just pass the file path as argument to the test scripts.

## 5. Activate the Python Virtual Environment

All Python test scripts **must** run inside the `pyds` virtualenv:

```bash
source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
```

> **Tip:** Add this line to your `~/.bashrc` inside the container to activate it automatically on every session.

## Verifying the Plugins

After completing all steps, verify that the plugins are registered:

```bash
gst-inspect-1.0 nvsahipreprocess
gst-inspect-1.0 nvsahipostprocess
```

Both commands should print the plugin details (name, description, properties) without errors.

## Enabling Display on WSL2

Due to a [known NVIDIA bug](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_WSL2_FAQ.html#while-using-pipelines-involving-nveglglessink-or-any-other-display-sinks-black-screen-is-coming-with-mesa-error-failed-to-attach-to-x11-shm-on-terminal), display sinks (`nveglglessink`) produce a black screen on Ubuntu 24-based Docker containers running on WSL2. The workaround is to use X11 forwarding over SSH.

### Step 1 — Configure SSH inside the container

```bash
apt update
apt install -y openssh-server xauth

mkdir -p /run/sshd

cat >> /etc/ssh/sshd_config << EOF
Port 2222
X11Forwarding yes
X11UseLocalhost no
PermitRootLogin yes
EOF

passwd root
# Set a password when prompted

/usr/sbin/sshd
```

### Step 2 — Connect from WSL (outside the container)

Open a new WSL terminal (not inside the container) and SSH into it with X11 forwarding:

```bash
ssh -Y -p 2222 root@localhost
```

From this SSH session, display sinks will work correctly through X11 forwarding.

> **Note:** If you don't need display output, you can skip this section entirely. The pipeline uses `fakesink` by default — display is only enabled with the explicit `--display` flag. Alternatively, use `--output-mp4` to save results to a file.

## Restoring Original Libraries

`install.sh` backs up the original directories before overwriting. To restore:

```bash
DS=/opt/nvidia/deepstream/deepstream/sources/libs
mv "$DS/nvdsinfer.bak" "$DS/nvdsinfer"
mv "$DS/nvdsinfer_yolo.bak" "$DS/nvdsinfer_yolo"
```

Then rebuild:

```bash
make -C "$DS/nvdsinfer" clean && make -C "$DS/nvdsinfer" -j$(nproc) && make -C "$DS/nvdsinfer" install
make -C "$DS/nvdsinfer_yolo" clean && make -C "$DS/nvdsinfer_yolo" -j$(nproc) && make -C "$DS/nvdsinfer_yolo" install
```
