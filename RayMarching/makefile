all: libRayMarching.a

include raymarching.mk

clean:
	-$(RM) libRayMarching.a $(OBJS)

-include ../installs.mk
install: /usr/local/lib/libRayMarching.a $(HDRS:%=/usr/local/include/%)
uninstall:
	-$(RM) /usr/local/lib/libRayMarching.a $(HDRS:%=/usr/local/include/%)
.PHONY: all clean install uninstall