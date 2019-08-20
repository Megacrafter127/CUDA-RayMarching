OBJS := simpledrawCUDA.o
HDRS := simpledrawCUDA.cuh
CFLAGS += -Wall -fPIC

include ../nvcc.mk

CUFLAGS += -O3 --relocatable-device-code=true -I../SimpleDraw


libSimpleDrawCUDA.a: $(OBJS)
	$(NVCC) -lib -o $@ $(OBJS)
