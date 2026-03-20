# SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# This software contains source code provided by NVIDIA Corporation.
#
# nvsahipreprocess — SAHI dynamic slice preprocessor for DeepStream
#
# Requires the same custom library (e.g. libcustom2d_preprocess.so) as nvdspreprocess
# for tensor preparation. The config file now only needs the [property] section
# (no [group-N] or ROI definitions — slices are computed dynamically).

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
WSL:=$(shell uname -a | grep -io "microsoft" | tr '[:upper:]' '[:lower:]')

SRCS:= gstnvsahipreprocess.cpp gstnvsahipreprocess_allocator.cpp \
       nvsahipreprocess_property_parser.cpp

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_sahipreprocess.so

DS_ROOT:=/opt/nvidia/deepstream/deepstream
NVDS_VERSION:=$(shell basename $(shell readlink -f $(DS_ROOT)) | sed 's/deepstream-//')

CFLAGS+= -fPIC -std=c++17 -Wall -Werror -DDS_VERSION=\"$(NVDS_VERSION).0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I include \
	 -I ../../includes \
	 -I ../gst-nvdspreprocess/include

GST_INSTALL_DIR?=$(DS_ROOT)/lib/gst-plugins/
LIB_INSTALL_DIR?=$(DS_ROOT)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
	-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ("microsoft", "$(WSL)")
  LIBS+= -L/usr/lib/wsl/lib/
endif

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
