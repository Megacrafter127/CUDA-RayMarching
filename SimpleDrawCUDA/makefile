all: libSimpleDrawCUDA.a

include simpledrawCUDA.mk

clean:
	-$(RM) libSimpleDrawCUDA.a $(OBJS)

-include ../installs.mk
install: /usr/local/lib/libSimpleDrawCUDA.a $(HDRS:%=/usr/local/include/%)
uninstall:
	-$(RM) /usr/local/lib/libSimpleDrawCUDA.a $(HDRS:%=/usr/local/include/%)
.PHONY: all clean install uninstall