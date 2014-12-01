all:  main.x

NVCC_FLAGS := -arch=sm_20

objfiles := main.o KernelLauncher.o FloatHolder.o
objdir := obj
objects := $(addprefix $(objdir)/,$(objfiles))
srcdir := src

# depend: FORCE
#	cp KernelLauncher.cu KernelLauncher.cpp; \
#	g++ -MM 
#	rm KernelLauncher.cpp
#	# ugly for now, figure out later
#
# FORCE:
#
# -include $(srcdir)/%.d

main.x: $(objects)
	nvcc $^ -o $@

$(objdir)/%.o: $(srcdir)/%.cu 
	nvcc -c $(NVCC_FLAGS) $< -o $@

$(objdir)/%.o: $(srcdir)/%.cpp $(srcdir)/%.h
	nvcc -c $(NVCC_FLAGS) $< -o $@

clean:
	rm $(objdir)/*;
	rm main.x
