# SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# nvsahipostprocess — SAHI GreedyNMM post-process plugin for DeepStream

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++

SRCS:= gstnvsahipostprocess.cpp
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_sahipostprocess.so

DS_ROOT:=/opt/nvidia/deepstream/deepstream

CFLAGS+= -fPIC -std=c++17 -Wall -Werror \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes

GST_INSTALL_DIR?=$(DS_ROOT)/lib/gst-plugins/
LIB_INSTALL_DIR?=$(DS_ROOT)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

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
