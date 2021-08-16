NVCC = nvcc
CUFLAGS = 


%.o: %.cu
	$(NVCC) $(CFLAGS:%=-Xcompiler %) $(CXXFLAGS:%=-Xcompiler %) $(CUFLAGS) -dc -o $@ $^
