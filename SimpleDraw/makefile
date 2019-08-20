all: libSimpleDraw.so libSimpleDraw.a

include simpledraw.mk

clean:
	-$(RM) libSimpleDraw.so libSimpleDraw.a $(OBJS)

-include ../installs.mk
install: /usr/local/lib/libSimpleDraw.so /usr/local/lib/libSimpleDraw.a $(HDRS:%=/usr/local/include/%)
uninstall:
	-$(RM) /usr/local/lib/libSimpleDraw.so /usr/local/lib/libSimpleDraw.a $(HDRS:%=/usr/local/include/%)
.PHONY: all clean install uninstall