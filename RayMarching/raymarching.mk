OBJS := raymarching.o
HDRS := raymarching.cuh vecops.cuh dynmath.cuh
CFLAGS += -Wall -fPIC

include ../nvcc.mk

CUFLAGS += -O3 --relocatable-device-code=true -I../SimpleDrawCUDA


libRayMarching.a: $(OBJS)
	$(NVCC) -lib -o $@ $(OBJS)
