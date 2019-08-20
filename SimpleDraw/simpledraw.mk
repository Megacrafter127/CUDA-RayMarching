OBJS := simpledraw.o
HDRS := simpledraw.h
LDLIBS += -lSDL2 -lm
CFLAGS += -O3 -Wall -fPIC

libSimpleDraw.so: $(OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) $(LDFLAGS) -shared -o $@ $(OBJS) $(LDLIBS)
libSimpleDraw.a: $(OBJS) $(LDLIBS)
	$(AR) $(ARFLAGS) "$@" $(OBJS)
