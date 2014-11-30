all:  main.x

objfiles := main.o KernelLauncher.o FloatHolder.o
objdir := obj
objects := $(addprefix $(objdir)/,$(objfiles))
srcdir := src

main.x: $(objects)
	nvcc $^ -o $@

$(objdir)/%.o: $(srcdir)/%.cu 
	nvcc -c -arch=sm_20 $< -o $@

$(objdir)/%.o: $(srcdir)/%.cpp
	nvcc -c -arch=sm_20 $< -o $@

clean:
	rm $(objdir)/*;
	rm main.x
