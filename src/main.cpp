#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "FloatHolder.h"
#include "KernelLauncher.h"

#define MATDIMX 64
#define MATDIMY 64

using namespace std;

class Error3d
{
  public:
  int i,j,k;
  Error3d(int ii, int jj, int kk) : i(ii), j(jj), k(kk) {}
};

int main()
{
  void runTestcopy();
  void runTestsaxpy();
  void runTesttransposeNaive();
  void runTesttransposeFast();
  void runTesttransposeFastNoBankConf();
  void runTestmatxmatNaive();
  void runTestmatxmatTiles();
  void runTestreduceY();

  cout << fixed << setprecision(2);
  cout << endl;
  runTestcopy();
  cout << endl;
  runTestsaxpy();
  cout << endl;
  runTesttransposeNaive();
  cout << endl;
  runTesttransposeFast();
  cout << endl;
  runTesttransposeFastNoBankConf();
  cout << endl;
  runTestmatxmatNaive();
  cout << endl;
  runTestmatxmatTiles();
  cout << endl;
  runTestreduceY();
  cout << endl;

  return 0;
}  

void runTestcopy()
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


void runTestsaxpy() 
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

void runTesttransposeNaive()
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

void runTesttransposeFast()
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

void runTesttransposeFastNoBankConf() 
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

void runTestmatxmatNaive()
{  
  cout << "Testing matxmatNaive" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fha(Nx,Ny);
  FloatHolder fhb(Ny,Nx);
  FloatHolder fhout(Ny,Ny);
  FloatHolder fhsoln(Ny,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fha.nx(); k++)
	{
	  fha(i,j,k) = indx;
          fhb(i,k,j) = indx;
	  indx++;
	}
  }

  {
    for (int i=0; i<fhout.nz(); i++)
      for (int j=0; j<fhout.ny(); j++)
	for (int k=0; k<fhout.nx(); k++)
	{
	  fhout(i,j,k) = 0;
	  fhsoln(i,j,k) = 0;
	}
  }

  fha.copyCPUtoGPU();
  fhb.copyCPUtoGPU();

  kernelLauncher.matxmatNaive(fha, fhb, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fhout.nz(); i++)
      for (int j=0; j<fhout.ny(); j++)
	for (int k=0; k<fhout.nx(); k++)
	{
          for (int x=0; x<fha.nx(); x++)
            fhsoln(i,j,k) += fha(i,j,x)*fhb(i,x,k);
          if(fhout(i,j,k) != fhsoln(i,j,k))
            throw Error3d(i,j,k);
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "matxmatNaive failed!" << endl;
    cout << "fhsoln ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhsoln(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhout(i,j,k) << std::endl;
  }
}

void runTestmatxmatTiles()
{  
  cout << "Testing matxmatTiles" << std::endl;

  // float sum = 0;
  // for (int i=0; i<64; i++)
  //   sum += (64+i)*i;
  // std::cout << sum << endl; 

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  float a = 2.0;
  FloatHolder fha(Nx,Ny);
  FloatHolder fhb(Ny,Nx);
  FloatHolder fhout(Ny,Ny);
  FloatHolder fhsoln(Ny,Ny);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  {
    int indx = 0;
    for (int i=0; i<fha.nz(); i++)
      for (int j=0; j<fha.ny(); j++)
	for (int k=0; k<fha.nx(); k++)
	{
	  fha(i,j,k) = indx;
          fhb(i,k,j) = indx;
	  indx++;
	}
  }

  {
    for (int i=0; i<fhout.nz(); i++)
      for (int j=0; j<fhout.ny(); j++)
	for (int k=0; k<fhout.nx(); k++)
	{
	  fhout(i,j,k) = 0;
	  fhsoln(i,j,k) = 0;
	}
  }

  fha.copyCPUtoGPU();
  fhb.copyCPUtoGPU();

  kernelLauncher.matxmatTiles(fha, fhb, fhout);

  fhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<fhout.nz(); i++)
      for (int j=0; j<fhout.ny(); j++)
	for (int k=0; k<fhout.nx(); k++)
	{
          for (int x=0; x<fha.nx(); x++)
            fhsoln(i,j,k) += fha(i,j,x)*fhb(i,x,k);
          if(fhout(i,j,k) != fhsoln(i,j,k))
          {
            throw Error3d(i,j,k);
          }
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "matxmatTiles failed!" << endl;
    cout << "fhsoln ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhsoln(i,j,k) << std::endl;
    cout << "fhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << fhout(i,j,k) << std::endl;
  }
}

void runTestreduceY()
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


