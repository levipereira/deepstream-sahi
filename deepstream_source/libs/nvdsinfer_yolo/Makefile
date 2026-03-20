################################################################################
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Author: Levi Pereira 
# https://forums.developer.nvidia.com/u/levi_pereira/activity
# https://github.com/levipereira
################################################################################

CC := g++
CFLAGS := -Wall -std=c++11
CFLAGS += -shared -fPIC
CFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes \
          -I/usr/local/cuda-$(CUDA_VER)/include

CUDA_MAJOR_VER := $(shell echo $(CUDA_VER) | cut -d. -f1)
CUDA_MINOR_VER := $(shell echo $(CUDA_VER) | cut -d. -f2)

ifeq ($(shell [ $(CUDA_MAJOR_VER) -gt 12 ] && echo true || { [ $(CUDA_MAJOR_VER) -eq 12 ] && [ $(CUDA_MINOR_VER) -ge 2 ] && echo true; }), true)
    LIBS := -lnvinfer
else
    LIBS := -lnvinfer -lnvparsers
endif

LFLAGS := -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES := nvdsinfer_yolo.cpp 
TARGET_LIB := libnvds_infer_yolo.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) /opt/nvidia/deepstream/deepstream/lib/

clean:
	rm -rf $(TARGET_LIB)
