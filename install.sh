#!/usr/bin/env bash
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install DeepStream SAHI plugins and all required dependencies.

Options:
  --plugins-only   Build and install only the SAHI plugins and modified libs
                   (skip DeepStream deps, Python bindings, and pip packages)
  -h, --help       Show this help message

Environment variables:
  DS_ROOT          DeepStream root (default: /opt/nvidia/deepstream/deepstream)
  CUDA_VER         CUDA version (default: auto-detected from nvcc)
  PYDS_VERSION     DeepStream Python bindings version
                   (default: 1.2.2 for DS 8.x, ignored for DS 9.x)

Supported DeepStream versions: 8.x, 9.x
EOF
    exit 0
}

PLUGINS_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plugins-only) PLUGINS_ONLY=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1" >&2; usage ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()  { echo -e "\n${CYAN}══════════════════════════════════════════════════════════════${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}\n"; }

# ── Paths ─────────────────────────────────────────────────────────────────────

DS_ROOT="${DS_ROOT:-/opt/nvidia/deepstream/deepstream}"
DS_SOURCES="${DS_ROOT}/sources"
DS_LIB="${DS_ROOT}/lib"
DS_GST="${DS_LIB}/gst-plugins"
DS_PYTHON_APPS="${DS_SOURCES}/deepstream_python_apps"
PYDS_VENV="${DS_PYTHON_APPS}/pyds"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SOURCES="${SCRIPT_DIR}/deepstream_source"
TEST_DIR="${SCRIPT_DIR}/python_test/deepstream-test-sahi"

# ── DeepStream version detection ─────────────────────────────────────────────

if command -v deepstream-app &>/dev/null; then
    DS_VERSION=$(deepstream-app --version 2>&1 | grep -oP 'DeepStreamSDK \K[0-9]+\.[0-9]+' || true)
fi

if [[ -z "${DS_VERSION:-}" ]]; then
    DS_VERSION=$(basename "$(readlink -f "$DS_ROOT")" | grep -oP '[0-9]+\.[0-9]+' || true)
fi

[[ -n "${DS_VERSION:-}" ]] || error "Could not detect DeepStream version"

DS_MAJOR=$(echo "$DS_VERSION" | cut -d. -f1)

if [[ "$DS_MAJOR" -ne 8 && "$DS_MAJOR" -ne 9 ]]; then
    error "Unsupported DeepStream version ${DS_VERSION}. Only 8.x and 9.x are supported."
fi

info "Detected DeepStream ${DS_VERSION} (major=${DS_MAJOR})"

if [[ "$DS_MAJOR" -eq 8 ]]; then
    PYDS_VERSION="${PYDS_VERSION:-1.2.2}"
fi

# ── Prerequisites ─────────────────────────────────────────────────────────────

[[ -d "$DS_ROOT" ]] || error "DeepStream not found at ${DS_ROOT}. Are you inside the DeepStream container?"
[[ -d "$DS_SOURCES" ]] || error "DeepStream sources not found at ${DS_SOURCES}"

if [[ -n "${CUDA_VER:-}" ]]; then
    info "Using CUDA_VER=${CUDA_VER} (from environment)"
elif command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    info "Detected CUDA_VER=${CUDA_VER}"
else
    error "CUDA_VER not set and nvcc not found. Export CUDA_VER (e.g. 12.6)"
fi
export CUDA_VER

if $PLUGINS_ONLY; then
    info "Running in --plugins-only mode (steps 1, 2, 4 will be skipped)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: DeepStream additional components
# ══════════════════════════════════════════════════════════════════════════════

if ! $PLUGINS_ONLY; then
    step "Step 1/4 — Installing DeepStream additional components"

    ADDITIONAL_INSTALL="${DS_ROOT}/user_additional_install.sh"
    if [[ -f "$ADDITIONAL_INSTALL" ]]; then
        if dpkg -s librdkafka122 &>/dev/null 2>&1; then
            info "Additional components already installed, skipping"
        else
            info "Running ${ADDITIONAL_INSTALL}..."
            bash "$ADDITIONAL_INSTALL"
        fi
    else
        warn "user_additional_install.sh not found at ${ADDITIONAL_INSTALL}, skipping"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: DeepStream Python bindings (pyds virtualenv)
# ══════════════════════════════════════════════════════════════════════════════

if ! $PLUGINS_ONLY; then
    PYTHON_INSTALL="${DS_ROOT}/user_deepstream_python_apps_install.sh"

    if [[ "$DS_MAJOR" -eq 8 ]]; then
        step "Step 2/4 — Installing DeepStream Python bindings (pyds ${PYDS_VERSION})"

        if [[ -d "$DS_PYTHON_APPS" ]]; then
            info "deepstream_python_apps already exists at ${DS_PYTHON_APPS}, skipping"
            info "(${PYTHON_INSTALL} would fail if this directory already exists)"
        else
            if [[ -f "$PYTHON_INSTALL" ]]; then
                info "Running ${PYTHON_INSTALL} --version ${PYDS_VERSION}..."
                bash "$PYTHON_INSTALL" --version "$PYDS_VERSION"
            else
                warn "user_deepstream_python_apps_install.sh not found at ${PYTHON_INSTALL}, skipping"
            fi
        fi
    else
        step "Step 2/4 — Installing DeepStream Python bindings (DS ${DS_VERSION}, build from source)"

        if [[ -d "$DS_PYTHON_APPS" ]]; then
            info "deepstream_python_apps already exists at ${DS_PYTHON_APPS}, skipping"
        else
            if [[ -f "$PYTHON_INSTALL" ]]; then
                info "Running ${PYTHON_INSTALL} --build-bindings -r master..."
                bash "$PYTHON_INSTALL" --build-bindings -r master
            else
                warn "user_deepstream_python_apps_install.sh not found at ${PYTHON_INSTALL}, skipping"
            fi
        fi
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build and install SAHI plugins + modified libs
# ══════════════════════════════════════════════════════════════════════════════

if $PLUGINS_ONLY; then
    step "Building SAHI plugins and modified libraries"
else
    step "Step 3/4 — Building SAHI plugins and modified libraries"
fi

backup_if_exists() {
    local target="$1"
    if [[ -d "$target" && ! -d "${target}.bak" ]]; then
        warn "Backing up original $(basename "$target") -> $(basename "$target").bak"
        cp -r "$target" "${target}.bak"
    fi
}

if [[ "$DS_MAJOR" -eq 8 ]]; then
    backup_if_exists "${DS_SOURCES}/libs/nvdsinfer"
fi
backup_if_exists "${DS_SOURCES}/libs/nvdsinfer_yolo"

info "Copying SAHI plugins to ${DS_SOURCES}/gst-plugins/"
cp -r "${REPO_SOURCES}/gst-plugins/gst-nvsahipreprocess"  "${DS_SOURCES}/gst-plugins/"
cp -r "${REPO_SOURCES}/gst-plugins/gst-nvsahipostprocess" "${DS_SOURCES}/gst-plugins/"

info "Copying modified libs to ${DS_SOURCES}/libs/"
if [[ "$DS_MAJOR" -eq 8 ]]; then
    cp -r "${REPO_SOURCES}/libs/nvdsinfer"      "${DS_SOURCES}/libs/"
fi
cp -r "${REPO_SOURCES}/libs/nvdsinfer_yolo"  "${DS_SOURCES}/libs/"

if [[ "$DS_MAJOR" -eq 8 ]]; then
    info "Building nvdsinfer... (DS 8.x only)"
    make -C "${DS_SOURCES}/libs/nvdsinfer" clean
    make -C "${DS_SOURCES}/libs/nvdsinfer" -j"$(nproc)" CUDA_VER="${CUDA_VER}"
    make -C "${DS_SOURCES}/libs/nvdsinfer" install CUDA_VER="${CUDA_VER}"
else
    info "Skipping nvdsinfer build (not needed on DS ${DS_VERSION})"
fi

info "Building nvdsinfer_yolo..."
make -C "${DS_SOURCES}/libs/nvdsinfer_yolo" clean
make -C "${DS_SOURCES}/libs/nvdsinfer_yolo" -j"$(nproc)" CUDA_VER="${CUDA_VER}"
make -C "${DS_SOURCES}/libs/nvdsinfer_yolo" install CUDA_VER="${CUDA_VER}"

info "Building gst-nvsahipreprocess..."
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipreprocess" clean
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipreprocess" -j"$(nproc)" CUDA_VER="${CUDA_VER}"
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipreprocess" install CUDA_VER="${CUDA_VER}"

info "Building gst-nvsahipostprocess..."
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipostprocess" clean
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipostprocess" -j"$(nproc)" CUDA_VER="${CUDA_VER}"
make -C "${DS_SOURCES}/gst-plugins/gst-nvsahipostprocess" install CUDA_VER="${CUDA_VER}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Install Python test dependencies
# ══════════════════════════════════════════════════════════════════════════════

if ! $PLUGINS_ONLY; then
    step "Step 4/4 — Installing Python test dependencies"

    if [[ -d "$PYDS_VENV" ]]; then
        info "Activating pyds virtualenv..."
        source "${PYDS_VENV}/bin/activate"

        REQ_FILE="${TEST_DIR}/requirements_compare.txt"
        if [[ -f "$REQ_FILE" ]]; then
            info "Installing comparison tool dependencies..."
            pip install -r "$REQ_FILE"
        fi

        deactivate
    else
        warn "pyds virtualenv not found at ${PYDS_VENV}, skipping Python dependencies"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════

echo ""
step "Installation complete!"

info "Installed libraries:"
ls -lh "${DS_LIB}/libnvds_infer.so"                    2>/dev/null || true
ls -lh "${DS_LIB}/libnvds_infer_yolo.so"               2>/dev/null || true
ls -lh "${DS_GST}/libnvdsgst_sahipreprocess.so"         2>/dev/null || true
ls -lh "${DS_GST}/libnvdsgst_sahipostprocess.so"        2>/dev/null || true

if ! $PLUGINS_ONLY; then
    echo ""
    info "To run the test pipelines:"
    echo -e "  ${CYAN}source ${PYDS_VENV}/bin/activate${NC}"
    echo -e "  ${CYAN}cd ${TEST_DIR}${NC}"
    echo -e "  ${CYAN}python3 deepstream_test_sahi.py --model visdrone-full-640 --no-display --csv ../videos/aerial_vehicles.mp4${NC}"
fi
echo ""
