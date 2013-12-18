#ifndef __BoundingBox_h
#define __BoundingBox_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007, Paulo Aristarco Pagliosa                 |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: BoundingBox.h
//  ========
//  Class definition for bounding box.

#include <stdlib.h>

#ifndef __Ray_h
#include "Ray.h"
#endif

//
// Auxiliary function
//
__host__ __device__ inline void
inflateAABB(Vec3& p1, Vec3& p2, const Vec3& p)
{
  if (p.x < p1.x)
    p1.x = p.x;
  if (p.x > p2.x)
    p2.x = p.x;
  if (p.y < p1.y)
    p1.y = p.y;
  if (p.y > p2.y)
    p2.y = p.y;
  if (p.z < p1.z)
    p1.z = p.z;
  if (p.z > p2.z)
    p2.z = p.z;
}


//////////////////////////////////////////////////////////
//
// BoundingBox: axis-aligned bounding box class
// ===========
class __align__(16) BoundingBox
{
public:
  // Constructors
  __host__ __device__
  BoundingBox()
  {
    setEmpty();
  }

  BoundingBox(const Vec3& p1, const Vec3& p2)
  {
    set(p1, p2);
  }

  BoundingBox(const BoundingBox& b, const Transf3& m):
    p1(b.p1),
    p2(b.p2)
  {
    transform(m);
  }

  __host__ __device__
  Vec3 getCenter() const
  {
    return (p1 + p2) * 0.5;
  }

  __host__ __device__
  REAL getDiagonalLength() const
  {
    return (p2 - p1).length();
  }

  __host__ __device__
  Vec3 getSize() const
  {
    return p2 - p1;
  }

  __host__ __device__
  REAL getMaxSize() const
  {
    return getSize().max();
  }

  __host__ __device__
  REAL getArea() const
  {
    Vec3 s = getSize();
    REAL a = s.x * s.y + s.y * s.z + s.z * s.x;

    return a + a;
  }

  __host__ __device__
  bool isEmpty() const
  {
    return p1.x >= p2.x || p1.y >= p2.y || p1.z >= p2.z;
  }

  __host__ __device__
  const Vec3& getP1() const
  {
    return p1;
  }

  __host__ __device__
  const Vec3& getP2() const
  {
    return p2;
  }

  __host__ __device__
  void setEmpty()
  {
    p1.x = p1.y = p1.z = +Math::infinity<REAL>();
    p2.x = p2.y = p2.z = -Math::infinity<REAL>();
  }

  void set(const Vec3&, const Vec3&);

  __host__ __device__
  void inflate(const Vec3& p)
  {
    inflateAABB(p1, p2, p);
  }

  __host__ __device__
  void inflate(REAL s)
  {
    if (Math::isPositive(s))
    {
      Vec3 c = getCenter() * (1 - s);

      p1 = p1 * s + c;
      p2 = p2 * s + c;
    }
  }

  __host__ __device__
  void inflate(const BoundingBox& b)
  {
    inflate(b.p1);
    inflate(b.p2);
  }

  __host__ __device__
  bool intersect(const RayInfo& r, REAL& d) const
  {
    REAL tmin;
    REAL tmax;
    REAL amin;
    REAL amax;

    tmin = ((&p1)[r.negative[_X]].x - r.origin.x) * r.inverseDirection.x;
    tmax = ((&p1)[1 - r.negative[_X]].x - r.origin.x) * r.inverseDirection.x;
    amin = ((&p1)[r.negative[_Y]].y - r.origin.y) * r.inverseDirection.y;
    amax = ((&p1)[1 - r.negative[_Y]].y - r.origin.y) * r.inverseDirection.y;
    if (tmin > amax || amin > tmax)
      return false;
    if (amin > tmin)
      tmin = amin;
    if (amax < tmax)
      tmax = amax;
    amin = ((&p1)[r.negative[_Z]].z - r.origin.z) * r.inverseDirection.z;
    amax = ((&p1)[1 - r.negative[_Z]].z - r.origin.z) * r.inverseDirection.z;
    if (tmin > amax || amin > tmax)
      return false;
    if (amin > tmin)
      tmin = amin;
    if (amax < tmax)
      tmax = amax;
    if (tmin > 0)
      d = tmin;
    else if (tmax > 0)
      d = tmax;
    else
      return false;
    return true;
  }

  void transform(const Transf3&);

protected:
  Vec3 p1;
  Vec3 p2;

}; // BoundingBox

#endif // __BoundingBox_h
