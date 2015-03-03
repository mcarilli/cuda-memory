#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "datatype.h"
#include "DataHolder.h"
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

template<class T> void runTestcopy();
template<class T> void runTestsaxpy();
template<class T> void runTesttransposeNaive();
template<class T> void runTesttransposeFast();
template<class T> void runTesttransposeFastNoBankConf();
template<class T> void runTestmatxmatNaive();
template<class T> void runTestmatxmatTiles();
template<class T> void runTestreduceY();
template<class T> void runTestparallelPrefixSum();

int main()
{
  cout << fixed << setprecision(2);
  cout << endl;
  runTestcopy<datatype>();
  cout << endl;
  runTestsaxpy<datatype>();
  cout << endl;
  runTesttransposeNaive<datatype>();
  cout << endl;
  runTesttransposeFast<datatype>();
  cout << endl;
  runTesttransposeFastNoBankConf<datatype>();
  cout << endl;
  runTestmatxmatNaive<datatype>();
  cout << endl;
  runTestmatxmatTiles<datatype>();
  cout << endl;
  runTestreduceY<datatype>();
  cout << endl;
  runTestparallelPrefixSum<datatype>();
  cout << endl;

  return 0;
}  

template<class T> void runTestcopy()
{  
  cout << "Testing copy" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dhin(Nx,Ny);
  DataHolder<T> dhout(Nx,Ny);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = indx;
	  dhout(i,j,k) = 0;
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.copy(dhin, dhout);

  dhout.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
        for (int k=0; k<dhin.nx(); k++)
          if(dhin(i,j,k) != dhout(i,j,k))
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "copy failed!" << endl;
    cout << "dhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << dhout(i,k,j) << std::endl;
  }  
}


template<class T> void runTestsaxpy() 
{  
  cout << "Testing saxpy" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  T a = 2;
  DataHolder<T> dhx(Nx,Ny);
  DataHolder<T> dhy(Nx,Ny);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhx.nz(); i++)
      for (int j=0; j<dhx.ny(); j++)
	for (int k=0; k<dhx.nx(); k++)
	{
	  dhx[indx] = 1;
	  dhy[indx] = 2;
	  indx++;
	}
  }

  dhx.copyCPUtoGPU();
  dhy.copyCPUtoGPU();

  kernelLauncher.saxpy(dhx, dhy, a);

  dhy.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<dhy.nz(); i++)
      for (int j=0; j<dhy.ny(); j++)
        for (int k=0; k<dhy.nx(); k++)
          if(dhy(i,j,k) != 4)
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "saxpy failed!" << endl;
    cout << "dhy ("<< i << "," << j << "," << k << "):  "
        << setw(10) << dhy(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << k << "," << j << "):  "
        << setw(10) << dhy(i,k,j) << std::endl;
  }
}

template<class T> void runTesttransposeNaive()
{  
  cout << "Testing transposeNaive" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dhin(Nx,Ny);
  DataHolder<T> dhout(Ny,Nx);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = indx;
	  dhout(i,k,j) = 0;
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.transposeNaive(dhin, dhout);

  dhout.copyGPUtoCPU();
  
  try
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
        for (int k=0; k<dhin.nx(); k++)
          if(dhin(i,j,k) != dhout(i,k,j))
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeNaive failed!" << endl;
    cout << "dhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << dhout(i,k,j) << std::endl;
  }
}

template<class T> void runTesttransposeFast()
{  
  cout << "Testing transposeFast" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dhin(Nx,Ny);
  DataHolder<T> dhout(Ny,Nx);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = indx;
	  dhout(i,k,j) = 0;
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.transposeFast(dhin, dhout);

  dhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
          if(dhin(i,j,k) != dhout(i,k,j)) 
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFast failed!" << endl;
    cout << "dhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << dhout(i,k,j) << std::endl;
  }
}

template<class T> void runTesttransposeFastNoBankConf() 
{  
  cout << "Testing transposeFastNoBankConf" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dhin(Nx,Ny);
  DataHolder<T> dhout(Ny,Nx);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = indx;
	  dhout(i,k,j) = 0;
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.transposeFastNoBankConf(dhin, dhout);

  dhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
          if(dhin(i,j,k) != dhout(i,k,j)) 
            throw Error3d(i,j,k);
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "transposeFastNoBankConf failed!" << endl;
    cout << "dhin ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << k << "," << j << "):  "
	<< setw(10) << dhout(i,k,j) << std::endl;
  }
}

template<class T> void runTestmatxmatNaive()
{  
  cout << "Testing matxmatNaive" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dha(Nx,Ny);
  DataHolder<T> dhb(Ny,Nx);
  DataHolder<T> dhout(Ny,Ny);
  DataHolder<T> dhsoln(Ny,Ny);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dha.nz(); i++)
      for (int j=0; j<dha.ny(); j++)
	for (int k=0; k<dha.nx(); k++)
	{
	  dha(i,j,k) = indx;
          dhb(i,k,j) = indx;
	  indx++;
	}
  }

  {
    for (int i=0; i<dhout.nz(); i++)
      for (int j=0; j<dhout.ny(); j++)
	for (int k=0; k<dhout.nx(); k++)
	{
	  dhout(i,j,k) = 0;
	  dhsoln(i,j,k) = 0;
	}
  }

  dha.copyCPUtoGPU();
  dhb.copyCPUtoGPU();

  kernelLauncher.matxmatNaive(dha, dhb, dhout);

  dhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<dhout.nz(); i++)
      for (int j=0; j<dhout.ny(); j++)
	for (int k=0; k<dhout.nx(); k++)
	{
          for (int x=0; x<dha.nx(); x++)
            dhsoln(i,j,k) += dha(i,j,x)*dhb(i,x,k);
          if(dhout(i,j,k) != dhsoln(i,j,k))
            throw Error3d(i,j,k);
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "matxmatNaive failed!" << endl;
    cout << "dhsoln ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhsoln(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhout(i,j,k) << std::endl;
  }
}

template<class T> void runTestmatxmatTiles()
{  
  cout << "Testing matxmatTiles" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dha(Nx,Ny);
  DataHolder<T> dhb(Ny,Nx);
  DataHolder<T> dhout(Ny,Ny);
  DataHolder<T> dhsoln(Ny,Ny);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dha.nz(); i++)
      for (int j=0; j<dha.ny(); j++)
	for (int k=0; k<dha.nx(); k++)
	{
	  dha(i,j,k) = indx;
          dhb(i,k,j) = indx;
	  indx++;
	}
  }

  {
    for (int i=0; i<dhout.nz(); i++)
      for (int j=0; j<dhout.ny(); j++)
	for (int k=0; k<dhout.nx(); k++)
	{
	  dhout(i,j,k) = 0;
	  dhsoln(i,j,k) = 0;
	}
  }

  dha.copyCPUtoGPU();
  dhb.copyCPUtoGPU();

  kernelLauncher.matxmatTiles(dha, dhb, dhout);

  dhout.copyGPUtoCPU();
  
  try 
  {
    for (int i=0; i<dhout.nz(); i++)
      for (int j=0; j<dhout.ny(); j++)
	for (int k=0; k<dhout.nx(); k++)
	{
          for (int x=0; x<dha.nx(); x++)
            dhsoln(i,j,k) += dha(i,j,x)*dhb(i,x,k);
          if(dhout(i,j,k) != dhsoln(i,j,k))
          {
            throw Error3d(i,j,k);
          }
	}
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "matxmatTiles failed!" << endl;
    cout << "dhsoln ("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhsoln(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhout(i,j,k) << std::endl;
  }
}

template<class T> void runTestreduceY()
{  
  cout << "Testing reduceY" << std::endl;

  int Nx = MATDIMX;
  int Ny = MATDIMY;
  DataHolder<T> dhin(Nx,Ny);
  DataHolder<T> dhout(Nx,Ny);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = 1;  // Warning:  try big numbers here and you will
	  dhout(i,j,k) = 0; // exceed floating-point precision!
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.reduceY(dhin, dhout);

  dhout.copyGPUtoCPU();

  for (int i=0; i<dhin.nz(); i++)
    for (int k=0; k<dhin.nx(); k++)
      for (int j=1; j<dhin.ny(); j++)
	  dhin(i,0,k) += dhin(i,j,k);

  try 
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int k=0; k<dhin.nx(); k++)
	if (dhin(i,0,k) != dhout(i,0,k))
	  throw Error3d(i,0,k); 
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "reduceY failed!" << endl;
    cout << "dhin("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhout(i,j,k) << std::endl;
  }
}

template<class T> void runTestparallelPrefixSum()
{  
  cout << "Testing parallelPrefixSum" << std::endl;

  int Nx = MATDIMX*MATDIMY;
  DataHolder<T> dhin(Nx);
  DataHolder<T> dhout(Nx);
  KernelLauncher<T>& kernelLauncher = KernelLauncher<T>::instance();

  {
    int indx = 0;
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	{
	  dhin(i,j,k) = rand()%2;  // Warning:  try big numbers here and you will
	  dhout(i,j,k) = 0;        // exceed floating-point precision!
	  indx++;
	}
  }

  dhin.copyCPUtoGPU();

  kernelLauncher.parallelPrefixSum(dhin, dhout);

  dhout.copyGPUtoCPU();
  
  unsigned int sum = 0;
  for (int i=0; i<dhin.nz(); i++)
    for (int j=0; j<dhin.ny(); j++)
      for (int k=0; k<dhin.nx(); k++)
      {
	  dhin(i,j,k) += sum;
          sum += dhin(i,j,k);
      }

  try 
  {
    for (int i=0; i<dhin.nz(); i++)
      for (int j=0; j<dhin.ny(); j++)
	for (int k=0; k<dhin.nx(); k++)
	  if (dhin(i,j,k) != dhout(i,j,k))
	    throw Error3d(i,j,k); 
  }
  catch(Error3d& error)
  {
    int i(error.i), j(error.j), k(error.k);
    cout << "parallelPrefixSum failed!" << endl;
    cout << "dhin("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhin(i,j,k) << std::endl;
    cout << "dhout("<< i << "," << j << "," << k << "):  "
	<< setw(10) << dhout(i,j,k) << std::endl;
  }
}
