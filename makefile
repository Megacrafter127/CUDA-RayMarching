all: StaticRayMarchingTest

VPATH = src

SUBPROJECTS := SimpleDraw SimpleDrawCUDA RayMarching

-l%: %/lib%.a
	$(MAKE) -C $< install
-l%: %/lib%.so
	$(MAKE) -C $< install

SimpleDraw/libSimpleDraw.a:
	$(MAKE) -C SimpleDraw libSimpleDraw.a

SimpleDrawCUDA/libSimpleDrawCUDA.a:
	$(MAKE) -C SimpleDrawCUDA libSimpleDrawCUDA.a

RayMarching/libRayMarching.a:
	$(MAKE) -C RayMarching libRayMarching.a

include nvcc.mk

OBJS := main.o
CFLAGS += -Wall -fPIC
CUFLAGS += -O3 --cudart static --relocatable-device-code=true
LDLIBS += $(SUBPROJECTS:%=-l%) -lSDL2

StaticRayMarchingTest: CUFLAGS += $(SUBPROJECTS:%=-I%)
StaticRayMarchingTest: LDFLAGS += $(SUBPROJECTS:%=-L%/)
StaticRayMarchingTest: $(OBJS) SimpleDraw/libSimpleDraw.a SimpleDrawCUDA/libSimpleDrawCUDA.a RayMarching/libRayMarching.a
	$(NVCC) $(CUFLAGS) $(LDFLAGS) -link -o $@ $(OBJS) -Xlinker -Bstatic $(LDLIBS) -Xlinker -Bdynamic
RayMarchingTest: LDFLAGS += -L/usr/local/lib
RayMarchingTest: $(OBJS) $(LDLIBS)
	$(NVCC) $(CUFLAGS) $(LDFLAGS) -link -o $@ $(OBJS) $(LDLIBS)


-include installs.mk
install: /usr/local/sbin/RayMarchingTest
uninstallSimpleDraw:
	-$(MAKE) -C SimpleDraw uninstall
uninstallSimpleDrawCUDA:
	-$(MAKE) -C SimpleDrawCUDA uninstall
uninstallRayMarching:
	-$(MAKE) -C RayMarching uninstall
uninstall:
	-$(RM) /usr/local/sbin/RayMarchingTest
uninstallall: uninstall $(SUBPROJECTS:%=uninstall%)
cleanSimpleDraw:
	-$(MAKE) -C SimpleDraw clean
cleanSimpleDrawCUDA:
	-$(MAKE) -C SimpleDrawCUDA clean
cleanRayMarching:
	-$(MAKE) -C RayMarching clean
clean:
	-$(RM) RayMarchingTest RayMarchingTestStatic $(OBJS)
cleanall: clean $(SUBPROJECTS:%=clean%)
.PHONY: all $(SUBPROJECTS:%=clean%) clean cleanall install $(SUBPROJECTS:%=uninstall%) uninstall uninstallall