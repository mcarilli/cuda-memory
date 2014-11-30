#pragma once
#include <cstdlib>

class FloatHolder;

class KernelLauncher 
{
  static KernelLauncher kl;
  KernelLauncher() {}
  KernelLauncher& operator=(KernelLauncher&); 
  KernelLauncher(const KernelLauncher&);
  public:
  static KernelLauncher& instance() { return kl; }
  void saxpy(FloatHolder& fhx, FloatHolder& fhy, float a);
};
