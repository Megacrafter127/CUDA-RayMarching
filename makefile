all: StaticRayMarchingTest

VPATH = src

SUBPROJECTS := SimpleDrawCUDA RayMarching

-l%: %/lib%.a
	$(MAKE) -C $< install
-l%: %/lib%.so
	$(MAKE) -C $< install

SimpleDrawCUDA/libSimpleDrawCUDA.a:
	$(MAKE) -C SimpleDrawCUDA libSimpleDrawCUDA.a

RayMarching/libRayMarching.a:
	$(MAKE) -C RayMarching libRayMarching.a

include nvcc.mk

OBJS := main.o
CFLAGS += -Wall -fPIC
CUFLAGS += -O3 --cudart static --relocatable-device-code=true
LDLIBS += $(SUBPROJECTS:%=-l%) -lSDL2 -lGL -lGLU -lglut

StaticRayMarchingTest: CUFLAGS += $(SUBPROJECTS:%=-I%)
StaticRayMarchingTest: LDFLAGS += $(SUBPROJECTS:%=-L%/)
StaticRayMarchingTest: $(OBJS) SimpleDrawCUDA/libSimpleDrawCUDA.a RayMarching/libRayMarching.a
	$(NVCC) $(CUFLAGS) $(LDFLAGS) -link -o $@ $(OBJS) -Xlinker -Bstatic $(LDLIBS) -Xlinker -Bdynamic
RayMarchingTest: LDFLAGS += -L/usr/local/lib
RayMarchingTest: $(OBJS) $(LDLIBS)
	$(NVCC) $(CUFLAGS) $(LDFLAGS) -link -o $@ $(OBJS) $(LDLIBS)

-include installs.mk
install: /usr/local/sbin/RayMarchingTest
uninstallSimpleDrawCUDA:
	-$(MAKE) -C SimpleDrawCUDA uninstall
uninstallRayMarching:
	-$(MAKE) -C RayMarching uninstall
uninstall:
	-$(RM) /usr/local/sbin/RayMarchingTest
uninstallall: uninstall $(SUBPROJECTS:%=uninstall%)
cleanSimpleDrawCUDA:
	-$(MAKE) -C SimpleDrawCUDA clean
cleanRayMarching:
	-$(MAKE) -C RayMarching clean
clean:
	-$(RM) RayMarchingTest StaticRayMarchingTest $(OBJS)
cleanall: clean $(SUBPROJECTS:%=clean%)
.PHONY: all $(SUBPROJECTS:%=clean%) clean cleanall install $(SUBPROJECTS:%=uninstall%) uninstall uninstallall