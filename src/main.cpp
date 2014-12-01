#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "FloatHolder.h"
#include "KernelLauncher.h"

int main()
{
  runSaxpyTest();
  runTransposeNaiveTest();
  runTransposeFastTest();

  return 0;
}  


void runSaxpyTest() 
{  
  int N = 1<<20;
  float a = 2.0;
  FloatHolder fhx(N);
  FloatHolder fhy(N);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int k=0; k<fhx.nz(); k++)
      for (int j=0; j<fhx.ny(); j++)
	for (int i=0; i<fhx.nx(); i++)
	{
	  fhx[indx] = 1.0f;
	  fhy[indx] = 2.0f;
	  indx++;
	}
  }

  fhx.copyCPUtoGPU();
  fhy.copyCPUtoGPU();

  kernelLauncher.saxpy(fhx, fhy, a);

  fhy.copyGPUtoCPU();
  
  { 
    float maxError = 0.0f;
    int indx = 0;
    for (int k=0; k<fhx.nz(); k++)
      for (int j=0; j<fhx.ny(); j++)
	for (int i=0; i<fhx.nx(); i++)
	{
          if (fhy[indx] != 4.0)
	    std::cout << "Index " << indx << ": fhy[indx]: " << std::endl;
	  indx++;
	}
  }
}

void runTransposeNaiveTest()
{
}

void runTransposeFastTest()
{
}
