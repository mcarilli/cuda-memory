#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "FloatHolder.h"
#include "KernelLauncher.h"

#define MATDIMX 1024
#define MATDIMY 1024

using namespace std;

int main()
{
  void runTestCopy();
  void runTestSaxpy();
  void runTestTransposeNaive();
  void runTestTransposeFast();

  cout << fixed << setprecision(2);
  cout << endl;
  runTestCopy();
  cout << endl;
  runTestSaxpy();
  cout << endl;
  runTestTransposeNaive();
  cout << endl;
  runTestTransposeFast();
  cout << endl;

  return 0;
}  

void runTestCopy()
{  
  cout << "Testing copyKernel" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,j,k) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.copy(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  { 
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
        {
          if (fhin(i,j,k) != fhout(i,j,k)) 
          {
            cout << "Transpose failed!" << endl;
            cout << "fhin ("<< i << "," << j << "," << k << "):  "
                << setw(10) << fhin(i,j,k) << std::endl;
            cout << "fhout("<< i << "," << j << "," << k << "):  "
                << setw(10) << fhout(i,j,k) << std::endl;
            break;
          }
        }
  }
}


void runTestSaxpy() 
{  
  cout << "Testing saxpyKernel" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhx(Nx,Ny);
  FloatHolder fhy(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhx.nz(); i++)
      for (int j=0; j<fhx.ny(); j++)
	for (int k=0; k<fhx.nx(); k++)
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
    int indx = 0;
    for (int i=0; i<fhx.nz(); i++)
      for (int j=0; j<fhx.ny(); j++)
	for (int k=0; k<fhx.nx(); k++)
	{
          if (fhy[indx] != 4.0)
          {
            cout << "Saxpy failed!" << endl; 
	    cout << "Index " << indx << ": fhy[indx]: " << fhy[indx] << std::endl;
            break;
          }
	  indx++;
	}
  }
}

void runTestTransposeNaive()
{  
  cout << "Testing transposeNaiveKernel" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,j,k) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.transposeNaive(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  { 
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
        {
          if(fhin(i,j,k) != fhout(i,k,j)) 
          {
            cout << "TransposeNaive failed!" << endl;
            cout << "fhin ("<< i << "," << j << "," << k << "):  "
                << setw(10) << fhin(i,j,k) << std::endl;
            cout << "fhout("<< i << "," << k << "," << j << "):  "
                << setw(10) << fhout(i,k,j) << std::endl;
            break;
          }
        }
  }
}

void runTestTransposeFast()
{  
  cout << "Testing transposeFastKernel" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,j,k) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();
  // fhout.copyCPUtoGPU();

  kernelLauncher.transposeFast(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  { 
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
          if(fhin(i,j,k) != fhout(i,k,j)) 
          {
            cout << "TransposeFast failed!" << endl;
	    cout << "fhin ("<< i << "," << j << "," << k << "):  "
		<< setw(10) << fhin(i,j,k) << std::endl;
	    cout << "fhout("<< i << "," << k << "," << j << "):  "
		<< setw(10) << fhout(i,k,j) << std::endl;
            break;
          }
	}
  }
}
