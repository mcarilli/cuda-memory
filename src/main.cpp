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

class Error3d
{
  public:
  int i,j,k;
  Error3d(int ii, int jj, int kk) : i(ii), j(jj), k(kk) {}
};

int main()
{
  void runTestCopy();
  void runTestSaxpy();
  void runTestTransposeNaive();
  void runTestTransposeFast();
  void runTestTransposeFastNoBankConf();
  void runTestMatxmatNaive();
  void runTestMatxmatTiles();
  void runTestReduceY();

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
  runTestTransposeFastNoBankConf();
  cout << endl;
  runTestMatxmatNaive();
  cout << endl;
  runTestMatxmatTiles();
  cout << endl;
  runTestReduceY();
  cout << endl;

  return 0;
}  

void runTestCopy()
{  
  cout << "Testing copy" << std::endl;

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
	  fhout(i,j,k) = 0.;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.copy(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
        for (int k=0; k<fhin.nx(); k++)
          if(fhin(i,j,k) != fhout(i,j,k))
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "copy failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhin(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }  
}


void runTestSaxpy() 
{  
  cout << "Testing saxpy" << std::endl;

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
	  fhx[indx] = 1.0;
	  fhy[indx] = 2.0;
	  indx++;
	}
  }

  fhx.copyCPUtoGPU();
  fhy.copyCPUtoGPU();

  kernelLauncher.saxpy(fhx, fhy, a);

  fhy.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<fhy.nz(); i++)
      for (int j=0; j<fhy.ny(); j++)
        for (int k=0; k<fhy.nx(); k++)
          if(fhy(i,j,k) != 4.0)
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "saxpy failed!" << endl;
    cout << "fhy ("<< i << "," << j << "," << k << "):  "
        << setw(10) << fhy(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
        << setw(10) << fhy(i,k,j) << std::endl;
  }
}

void runTestTransposeNaive()
{  
  cout << "Testing transposeNaive" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Ny,Nx);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,k,j) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.transposeNaive(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
        for (int k=0; k<fhin.nx(); k++)
          if(fhin(i,j,k) != fhout(i,k,j))
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeNaive failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhin(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }
}

void runTestTransposeFast()
{  
  cout << "Testing transposeFast" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Ny,Nx);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,k,j) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.transposeFast(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
          if(fhin(i,j,k) != fhout(i,k,j)) 
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFast failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhin(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }
}

void runTestTransposeFastNoBankConf() 
{  
  cout << "Testing transposeFastNoBankConf" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fhin(Nx,Ny);
  FloatHolder fhout(Ny,Nx);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
	{
	  fhin(i,j,k) = indx;
	  fhout(i,k,j) = 0;
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.transposeFastNoBankConf(fhin, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fhin.nz(); i++)
      for (int j=0; j<fhin.ny(); j++)
	for (int k=0; k<fhin.nx(); k++)
          if(fhin(i,j,k) != fhout(i,k,j)) 
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFastNoBankConf failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhin(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }
}

void runTestMatxmatNaive()
{  
  cout << "Testing matxmatNaive" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fha(Nx,Ny);
  FloatHolder fhb(Ny,Nx);
  FloatHolder fhout(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fha.nx(); k++)
	{
	  fha(i,j,k) = indx;
          fhb(i,k,j) = indx;
	  fhout(i,k,j) = 0;
	  indx++;
	}
  }

  fha.copyCPUtoGPU();
  fhb.copyCPUtoGPU();

  kernelLauncher.matxmatNaive(fha, fhb, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fha.nx(); k++)
	{
          // if(fhin(i,j,k) != fhout(i,k,j)) 
          //  throw Error3d(i,j,k);
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFast failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhout(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }
}

void runTestMatxmatTiles()
{  
  cout << "Testing matxmatTiles" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fha(Nx,Ny);
  FloatHolder fhb(Ny,Nx);
  FloatHolder fhout(Nx,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fha.nx(); k++)
	{
	  fha(i,j,k) = 1;
          fhb(i,k,j) = 1;
	  fhout(i,k,j) = 0;
	  indx++;
	}
  }

  fha.copyCPUtoGPU();
  fhb.copyCPUtoGPU();

  kernelLauncher.matxmatTiles(fha, fhb, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fhb.nx(); k++)
	{
          if(fhout(i,j,k) != MATDIMX) 
            throw Error3d(i,j,k);
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFast failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhout(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << fhout(i,k,j) << std::endl;
  }
}

void runTestReduceY()
{  
  cout << "Testing reduceY" << std::endl;

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
	  fhin(i,j,k) = 1;  // Warning:  try big numbers here and you will
	  fhout(i,j,k) = 0; // exceed floating-point precision!
	  indx++;
	}
  }

  fhin.copyCPUtoGPU();

  kernelLauncher.reduceY(fhin, fhout);

  fhout.copyGPUtoCPU();

  for (int i=0; i<fhin.nz(); i++)
    for (int k=0; k<fhin.nx(); k++)
      for (int j=1; j<fhin.ny(); j++)
	  fhin(i,0,k) += fhin(i,j,k);

  try 
  {
    for (int i=0; i<fhin.nz(); i++)
      for (int k=0; k<fhin.nx(); k++)
	if (fhin(i,0,k) != fhout(i,0,k))
	  throw Error3d(i,0,k); 
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "reduceY failed!" << endl;
    cout << "fhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhin(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhout(i,j,k) << std::endl;
  }
}


