NVCC = nvcc
CUFLAGS = 
CFLAGS += -I /usr/local/cuda-11.0/include/


%.o: %.cu
	$(NVCC) $(CFLAGS:%=-Xcompiler %) $(CXXFLAGS:%=-Xcompiler %) $(CUFLAGS) -dc -o $@ $<
