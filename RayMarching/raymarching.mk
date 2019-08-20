OBJS := raymarching.o
HDRS := raymarching.cuh
CFLAGS += -Wall -fPIC

include ../nvcc.mk

CUFLAGS += -O3 --relocatable-device-code=true -I../SimpleDrawCUDA -I../SimpleDraw


libRayMarching.a: $(OBJS)
	$(NVCC) -lib -o $@ $(OBJS)
