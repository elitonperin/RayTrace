#ifndef __Utils_h
#define __Utils_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                  GVSG Foundation Classes (CUDA Library)                  |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2009, Paulo Pagliosa                           |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
// OVERVIEW: Util.h
// ========
// Definitions for timer, etc.

#ifndef __Typedefs_h
#include "Typedefs.h"
#endif

namespace Utils
{ // begin namespace Utils


//////////////////////////////////////////////////////////
//
// Utils:Flags: flags class
// ===========
class __align__(4) Flags
{
public:
  // Constructors
  __host__ __device__
  Flags()
  {
    bits = 0;
  }

  __host__ __device__
  Flags(int mask)
  {
    bits = mask;
  }

  __host__ __device__
  Flags& operator =(int mask)
  {
    bits = mask;
    return *this;
  }

  __host__ __device__
  void set(int mask)
  {
    bits |= mask;
  }

  __host__ __device__
  void reset(int mask)
  {
    bits &= ~mask;
  }

  __host__ __device__
  void clear()
  {
    bits = 0;
  }

  __host__ __device__
  void enable(int mask, bool state)
  {
    state ? set(mask) : reset(mask);
  }

  __host__ __device__
  operator int() const
  {
    return bits;
  }

  __host__ __device__ bool
  isSet(int mask) const
  {
    return (bits & mask) == mask;
  }

  __host__ __device__
  bool test(int mask) const
  {
    return (bits & mask) != 0;
  }

private:
  int bits;

}; // Flags

} // end namespace Utils

#endif // __Utils_h
