all:  main.x

NVCC_FLAGS := -arch=sm_20

srcfiles := main.cpp KernelLauncher.cu FloatHolder.cpp
objfiles := main.o KernelLauncher.o FloatHolder.o
depfiles := $(patsubst %.o,%.d,$(objfiles))
objdir := obj
objects := $(addprefix $(objdir)/,$(objfiles))
srcdir := src

# depend: $(srcdir)/$(depfiles)
#
# $(srcdir)/%.d: FORCE
# 	# cp src/KernelLauncher.cu src/KernelLauncher.cpp;
#	# g++ -MM $(patsubst %.d,%.cpp,$@) > $@; \
#	rm src/KernelLauncher.cpp 
#	# ugly for now, figure out later
#
# FORCE:
#
# -include $(srcdir)/%.d

main.x: $(objects)
	nvcc $^ -o $@

$(objdir)/%.o: $(srcdir)/%.cu 
	nvcc -c $(NVCC_FLAGS) $< -o $@

$(objdir)/%.o: $(srcdir)/%.h
$(objdir)/%.o: $(srcdir)/%.cpp 
	nvcc -c $(NVCC_FLAGS) $< -o $@

clean:
	rm $(objdir)/*;
	rm main.x
