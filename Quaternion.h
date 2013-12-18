#ifndef __Quaternion_h
#define __Quaternion_h

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
// OVERVIEW: Quaternion.h
// ========
// Class definition for quaternion.

#ifndef __Vector3_h
#include "Vector3.h"
#endif

//
// Forward definition
//
template <typename real> class Matrix33;


//////////////////////////////////////////////////////////
//
// Quaternion: quaternion class
// ==========
template <typename real>
class Quaternion
{
public:
  Vector3<real> v;
  real w;

  // Constructors
  __host__ __device__
  Quaternion()
  {
    // do nothing
  }

  __host__ __device__
  Quaternion(real x, real y, real z, real w)
  {
    set(x, y, z, w);
  }

  __host__ __device__
  explicit Quaternion(const Vector3<real>& v, real w = 0)
  {
    set(v, w);
  }

  __host__ __device__
  Quaternion(const real* q)
  {
    set(q);
  }

  __host__ __device__
  Quaternion(real angle, const Vector3<real>& axis)
  {
    set(angle, axis);
  }

  __host__ __device__
  explicit Quaternion(const Matrix33<real>& m)
  {
    set(m);
  }

  __host__ __device__
  void set(const Quaternion<real> q)
  {
    v = q.v;
    w = q.w;
  }

  __host__ __device__
  void set(const Vector3<real>& v, real w = 0)
  {
    this->v = v;
    this->w = w;
  }

  __host__ __device__
  void set(real x, real y, real z, real w)
  {
    this->v.set(x, y, z);
    this->w = w;
  }

  __host__ __device__
  void set(const real* q)
  {
    v.set(q);
    w = q[3];
  }

  __host__ __device__
  void set(real angle, const Vector3<real>& axis)
  {
    real aBy2 = angle * (real)0.5;

    v = axis.versor() * sin(aBy2);
    w = cos(aBy2);
  }

  // Implemented in Matrix33.h
  __host__ __device__
  void set(const Matrix33<real>&);

  __host__ __device__
  Quaternion<real>& operator =(const Matrix33<real>& m)
  {
    set(m);
    return *this;
  }

  __host__ __device__
  Quaternion<real>& zero()
  {
    v.zero();
    w = 0;
    return *this;
  }

  __host__ __device__
  Quaternion<real>& identity()
  {
    v.zero();
    w = 1;
    return *this;
  }

  __host__ __device__
  bool operator ==(const Quaternion<real>& q) const
  {
    return v == q.v && Math::isEqual(w, q.w);
  }

  __host__ __device__
  bool operator !=(const Quaternion<real>& q) const
  {
    return !operator ==(q);
  }

  __host__ __device__
  Quaternion<real>& operator +=(const Quaternion<real>& q)
  {
    v += q.v;
    w += q.w;
    return *this;
  }

  __host__ __device__
  Quaternion<real>& operator -=(const Quaternion<real>& q)
  {
    v -= q.v;
    w -= q.w;
    return *this;
  }

  __host__ __device__
  Quaternion<real>& operator *=(real s)
  {
    v *= s;
    w *= s;
    return *this;
  }

  __host__ __device__
  Quaternion<real>& operator *=(const Quaternion<real>& q)
  {
    set(operator *(q));
    return *this;
  }

  __host__ __device__
  Quaternion<real> operator +(const Quaternion<real>& q) const
  {
    return Quaternion<real>(v + q.v, w + q.w);
  }

  __host__ __device__
  Quaternion<real> operator -(const Quaternion<real>& q) const
  {
    return Quaternion<real>(v - q.v, w - q.w);
  }

  __host__ __device__
  Quaternion<real> operator *(real s) const
  {
    return Quaternion<real>(v * s, w * s);
  }

  __host__ __device__
  Quaternion<real> operator *(const Quaternion<real>& q) const
  {
    Vec3 a = w * q.v + q.w * v + v.cross(q.v);
    real b = w * q.w - v.inner(q.v);

    return Quaternion<real>(a, b);
  }

  __host__ __device__
  Quaternion<real> operator -() const
  {
    return Quaternion<real>(-v, -w);
  }

  __host__ __device__
  Quaternion<real> operator ~() const
  {
    return Quaternion<real>(-v, +w);
  }

  __host__ __device__
  bool isNull(real eps = Math::zero<real>()) const
  {
    return v.isNull(eps) && Math::isZero(w, eps);
  }

  __host__ __device__
  real norm() const
  {
    return v.norm() + real(sqr(w));
  }

  __host__ __device__
  real length() const
  {
    return real(sqrt(norm()));
  }

  __host__ __device__
  Quaternion<real>& normalize(real eps = Math::zero<real>())
  {
    real len = length();

    if (!Math::isZero(len, eps))
      operator *=((real)Math::inverse(len));
    return *this;
  }

  __host__ __device__
  Quaternion<real>& negate()
  {
    v.negate();
    w = -w;
    return *this;
  }

  __host__ __device__
  Quaternion<real>& invert()
  {
    v.negate();
    return normalize();
  }

  __host__ __device__
  Quaternion<real> conjugate() const
  {
    return operator ~();
  }

  __host__ __device__
  Quaternion<real> inverse() const
  {
    return conjugate().normalize();
  }

  __host__ __device__
  Vec3 rotate(const Vector3<real>& p) const
  {
    return (p * (w * w - (real)0.5) + v.cross(p) * w + v * v.inner(p)) * 2;
  }

  void print(const char* s, FILE* f = stdout) const
  {
    fprintf(f, "%s[<%f,%f,%f>,%f]\n", s, v.x, v.y, v.z, w);
  }

}; // Quaternion

//
// Auxiliary function
//
template <typename real>
__host__ __device__ inline Quaternion<real>
operator *(double s, const Quaternion<real>& q)
{
  return q * (real)s;
}

typedef Quaternion<REAL> Quat;

#endif // __Quaternion_h
