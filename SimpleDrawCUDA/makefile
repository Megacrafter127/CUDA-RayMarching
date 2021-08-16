include simpledrawCUDA.mk

all: $(ARTIFACTS)

clean:
	-$(RM) $(ARTIFACTS) $(OBJS)

-include ../installs.mk
install: $(ARTIFACTS:%=/usr/local/lib/%) $(HDRS:%=/usr/local/include/%)
uninstall:
	-$(RM) $(ARTIFACTS:%=/usr/local/lib/%) $(HDRS:%=/usr/local/include/%)
.PHONY: all clean install uninstall
