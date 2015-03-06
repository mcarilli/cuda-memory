all:  main.x

NVCC_FLAGS := -arch=sm_30 -lineinfo # --ptxas-options=-v

srcfiles := main.cpp KernelLauncher.cu DataHolder.cpp
objfiles := main.o KernelLauncher.o DataHolder.o
depfile := dependencies.d
objdir := obj
srcdir := src
objects := $(addprefix $(objdir)/,$(objfiles))
deps := $(addprefix $(srcdir)/,$(depfile))
srcs := $(addprefix $(srcdir)/,$(srcfiles)) 

depend:
	cp src/KernelLauncher.cu src/KernelLauncher.cpp; \
	g++ -MM $(patsubst %.cu,%.cpp,$(srcs)) > $(deps); \
	sed -i "s/\(.*.o:\)/$(objdir)\/\1/g" $(deps); \
	sed -i "s/KernelLauncher.cpp/KernelLauncher.cu/g" $(deps); \
	rm src/KernelLauncher.cpp 

include $(deps)

main.x: $(objects)
	nvcc $^ -o $@

$(objdir)/%.o: $(srcdir)/%.cu 
	nvcc -c $(NVCC_FLAGS) $< -o $@

$(objdir)/%.o: $(srcdir)/%.cpp 
	nvcc -c $(NVCC_FLAGS) $< -o $@

clean:
	rm $(objdir)/*;
	rm main.x

