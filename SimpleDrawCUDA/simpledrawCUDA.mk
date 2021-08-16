OBJS := assert_cuda.o simpledraw.o simpledrawCUDA.o
HDRS := assert_cuda.h simpledraw.h
ARTIFACTS := libSimpleDrawCUDA.a libSimpleDrawCUDA.so
LDLIBS += -lSDL2 -lcudart
CFLAGS += -Wall -fPIC -O3 -g
CXXFLAGS += -Wall -fPIC -O3 -g


libSimpleDrawCUDA.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $?

libSimpleDrawCUDA.so: $(OBJS) $(LDLIBS)
	$(CC) $(LDFLAGS) -shared -o $@ $+

