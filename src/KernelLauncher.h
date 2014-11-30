#pragma once
#include <cstdlib>

class FloatHolder;

class KernelLauncher 
{
  KernelLauncher() {}
  KernelLauncher& operator=(KernelLauncher&); 
  KernelLauncher(const KernelLauncher&);
  public:
  // static KernelLauncher& instance() { return kl; }
  static KernelLauncher kl;
  void saxpy(FloatHolder, FloatHolder, float);
};
