#ifndef __Vector3_h
#define __Vector3_h

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
// OVERVIEW: Vector3.h
// ========
// Class definition for 3D vector.

#include <stdio.h>

#ifndef __Real_h
#include "Real.h"
#endif

using namespace Math;


//////////////////////////////////////////////////////////
//
// Vector3: 3D vector class
// =======
template <typename real>
class Vector3
{
public:
  real x;
  real y;
  real z;

  // Constructors
  __host__ __device__
  Vector3()
  {
    // do nothing
  }

  __host__ __device__
  Vector3(real a, real b, real c)
  {
    set(a, b, c);
  }

  __host__ __device__
  Vector3(const real* v)
  {
    set(v);
  }

  __host__ __device__
  void set(const Vector3<real>& v)
  {
    x = v.x;
    y = v.y;
    z = v.z;
  }

  __host__ __device__
  void set(real a, real b, real c)
  {
    x = a;
    y = b;
    z = c;
  }

  __host__ __device__
  void set(const real* v)
  {
    x = v[_X];
    y = v[_Y];
    z = v[_Z];
  }

  __host__ __device__
  Vector3<real>& zero()
  {
    x = y = z = 0;
    return *this;
  }

  __host__ __device__
  bool operator ==(const Vector3<real>& v) const
  {
    return Math::isNull<real>(x - v.x, y - v.y, z - v.z, Math::zero<real>());
  }

  __host__ __device__
  bool operator !=(const Vector3<real>& v) const
  {
    return !operator ==(v);
  }

  __host__ __device__
  Vector3<real>& operator +=(const Vector3<real>& b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  __host__ __device__
  Vector3<real>& operator -=(const Vector3<real>& b)
  {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  __host__ __device__
  Vector3<real>& operator *=(real s)
  {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }

  __host__ __device__
  real& operator [](int i)
  {
    return (&x)[i];
  }

  __host__ __device__
  const real& operator [](int i) const
  {
    return (&x)[i];
  }

  __host__ __device__
  Vector3<real> operator +(const Vector3<real>& b) const
  {
    return Vector3<real>(x + b.x, y + b.y, z + b.z);
  }

  __host__ __device__
  Vector3<real> operator -(const Vector3<real>& b) const
  {
    return Vector3<real>(x - b.x, y - b.y, z - b.z);
  }

  __host__ __device__
  Vector3<real> operator -() const
  {
    return Vector3<real>(-x, -y, -z);
  }

  __host__ __device__
  real operator *(const Vector3<real>& b) const
  {
    return x * b.x + y * b.y + z * b.z;
  }

  __host__ __device__
  Vector3<real> operator *(real s) const
  {
    return Vector3<real>(x * s, y * s, z * s);
  }

  __host__ __device__
  bool isNull() const
  {
    return Math::isNull<real>(x, y, z, Math::zero<real>());
  }

  __host__ __device__
  real norm() const
  {
    return sqr(x) + sqr(y) + sqr(z);
  }

  __host__ __device__
  real length() const
  {
    return sqrt(norm());
  }

  __host__ __device__
  real max() const
  {
    return Math::max<real>(x, Math::max<real>(y, z));
  }

  __host__ __device__
  real min() const
  {
    return Math::min<real>(x, Math::min<real>(y, z));
  }

  __host__ __device__
  Vector3<real> inverse() const
  {
    return Vector3<real>(1 / x, 1 / y, 1 / z);
  }

  __host__ __device__
  Vector3<real>& negate()
  {
    x = -x;
    y = -y;
    z = -z;
    return *this;
  }

  __host__ __device__
  Vector3<real>& normalize()
  {
    real len = length();

    if (!Math::isZero<real>(len))
      operator *=(Math::inverse<real>(len));
    return *this;
  }

  __host__ __device__
  Vector3<real> versor() const
  {
    return Vector3<real>(*this).normalize();
  }

  __host__ __device__
  real inner(const Vector3<real>& v) const
  {
    return operator*(v);
  }

  __host__ __device__
  real inner(real x, real y, real z) const
  {
    return inner(Vector3<real>(x, y, z));
  }

  __host__ __device__
  Vector3<real> cross(const Vector3<real>& v) const
  {
    real a = y * v.z - z * v.y;
    real b = z * v.x - x * v.z;
    real c = x * v.y - y * v.x;

    return Vector3<real>(a, b, c);
  }

  __host__ __device__
  Vector3<real> cross(real x, real y, real z) const
  {
    return cross(Vector3<real>(x, y, z));
  }

  void print(const char* s, FILE* f = stdout) const
  {
    fprintf(f, "%s<%f,%f,%f>\n", s, x, y, z);
  }

}; // Vector3<real>

//
// Auxiliary function
//
template <typename real>
__host__ __device__ inline Vector3<real>
operator *(double s, const Vector3<real>& v)
{
  return v * (real)s;
}

typedef Vector3<REAL> Vec3;

#endif // __Vector3_h
