#ifndef __Real_h
#define __Real_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                        GVSG Foundation Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007, Paulo Aristarco Pagliosa                 |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
// OVERVIEW: Real.h
// ========
// Mathematical macros/inline functions.

#include <float.h>
#include <math.h>

#undef min
#undef max

// #define __CUDA

#ifdef __CUDA
#include "host_defines.h"
#else
#define __host__
#define __device__
#define __align__(i)
#endif

#ifdef __DOUBLE_FP
#define REAL double
#define M_EPS DBL_EPSILON
#define M_INF DBL_MAX
#else
#define REAL float
#define M_EPS FLT_EPSILON
#define M_INF FLT_MAX
#endif

//
// Math macros
//
#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#define M_PI_2  (M_PI * 0.5)
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887242097
#endif
#define M_SQRT3 1.7320508075688772935274463415059

namespace Math
{ // begin namespace Math

template <typename real> real zero();

template <>
__host__ __device__ inline float
zero<float>()
{
  return FLT_EPSILON;
}

template <>
__host__ __device__ inline double
zero<double>()
{
  return DBL_EPSILON;
}

template <typename real> real infinity();

template <>
__host__ __device__ inline float
infinity<float>()
{
  return FLT_MAX;
}

template <>
__host__ __device__ inline double
infinity<double>()
{
  return DBL_MAX;
}

template <typename real>
__host__ __device__ inline bool
isZero(real a, real eps = zero<real>())
{
  return fabs(a) <= eps;
}

template <typename real>
__host__ __device__ inline bool
isEqual(real a, real b, real eps = zero<real>())
{
  return isZero<real>(a - b, eps);
}

template <typename real>
__host__ __device__ inline bool
isPositive(real a, real eps = zero<real>())
{
  return a > +eps;
}

template <typename real>
__host__ __device__ inline bool
isNegative(real a, real eps = zero<real>())
{
  return a < -eps;
}

template <typename real>
__host__ __device__ inline bool
isNull(real a, real b, real eps)
{
  return isZero<real>(a, eps) && isZero<real>(b, eps);
}

template <typename real>
__host__ __device__ inline bool
isNull(real a, real b, real c, real eps)
{
  return isNull<real>(a, b, eps) && isZero<real>(c, eps);
}

template <typename real>
__host__ __device__ inline bool
  isNull(real a, real b, real c, real d, real eps)
{
  return isNull<real>(a, b, c, eps) && isZero<real>(d, eps);
}

template <typename real>
__host__ __device__ inline real
inverse(real a)
{
  return (real)(1.0 / a);
}

template <typename real>
__host__ __device__ inline real
toRadians(real a)
{
  return (real)((a) * M_PI / 180);
}

template <typename real>
__host__ __device__ inline real
toDegrees(real a)
{
  return (real)((a) * 180 / M_PI);
}

template <typename real>
__host__ __device__ inline real
sqr(real a)
{
  return a * a;
}

template <typename real>
__host__ __device__ inline real
cube(real a)
{
  return a * a * a;
}

template <typename T>
__host__ __device__ inline T
max(const T& a, const T& b)
{
  return a > b ? a : b;
}

template <typename T>
__host__ __device__ inline T
min(const T& a, const T& b)
{
  return a < b ? a : b;
}

template <typename real>
__host__ __device__ inline int
realAsInt(real f)
{
  union
  {
    real f;
    int i;
  } u;

  u.f = f;
  return u.i;
}

template <typename real>
__host__ __device__ inline real
intAsReal(int i)
{
  union
  {
    real f;
    int i;
  } u;

  u.i = i;
  return u.f;
}

} // end namespace Math

//
// Axis enumeration
//
enum { _X, _Y, _Z, _W };

#endif // __Real_h
