OBJS := simpledrawCUDA.o simpledraw.o assert_cuda.o
HDRS := simpledrawCUDA.cuh simpledraw.h
CFLAGS += -Wall -fPIC

include ../nvcc.mk

CUFLAGS += -lineinfo -pg -O2 --relocatable-device-code=true


libSimpleDrawCUDA.a: $(OBJS)
	$(NVCC) -lib -o $@ $(OBJS)
