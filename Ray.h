#ifndef __Ray_h
#define __Ray_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                        GVSG Foundation Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007-2009, Paulo Aristarco Pagliosa            |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: Ray.h
//  ========
//  Class definition for ray.

#ifndef __Transform3_h
#include "Transform3.h"
#endif
#ifndef __Utils_h
#include "Utils.h"
#endif

using namespace Utils;


//////////////////////////////////////////////////////////
//
// Ray: ray class
// ===
struct Ray
{
  Vec3 origin;
  Vec3 direction;

  // Constructors
  __host__ __device__
  Ray()
  {
    // do nothing
  }

  __host__ __device__
  Ray(const Vec3& origin, const Vec3& direction)
  {
    set(origin, direction);
  }

  __host__ __device__
  Ray(const Ray& ray, const Transf3& m)
  {
    set(m.transform(ray.origin), m.transformVector(ray.direction));
  }

  __host__ __device__
  void set(const Vec3& origin, const Vec3& direction)
  {
    this->origin = origin;
    this->direction = direction;
  }

  __host__ __device__
  void transform(const Transf3& m)
  {
    m.transformRef(origin);
    m.transformVectorRef(direction);
  }

}; // Ray

//
// Make ray point
//
__host__ __device__
inline Vec3
makeRayPoint(const Vec3& origin, const Vec3& direction, REAL t)
{
  return origin + direction * t;
}

__host__ __device__
inline Vec3
makeRayPoint(const Ray& ray, REAL t)
{
  return makeRayPoint(ray.origin, ray.direction, t);
}


//////////////////////////////////////////////////////////
//
// RayInfo: ray info class
// =======
struct RayInfo
{
public:
  Vec3 origin;
  Vec3 inverseDirection;
  uint negative[3];

  // Constructors
  __host__ __device__
  RayInfo()
  {
    // do nothing
  }

  __host__ __device__
  RayInfo(const Ray& ray)
  {
    set(ray);
  }

  __host__ __device__
  void set(const Ray& ray)
  {
    set(ray.origin, ray.direction);
  }

  __host__ __device__
  void set(const Vec3& origin, const Vec3& direction)
  {
    this->origin = origin;
    inverseDirection.set(direction.inverse());
    negative[_X] = direction.x < 0;
    negative[_Y] = direction.y < 0;
    negative[_Z] = direction.z < 0;
  }

}; // RayInfo

namespace Graphics
{ // begin namespace Graphics

//
// Forward definition
//
class Model;


//////////////////////////////////////////////////////////
//
// IntersectInfo: intersection ray/object info class
// =============
struct IntersectInfo
{
  // The intersection point
  Vec3 p;
  // The distance from the ray's origin to the intersection point
  REAL distance;
  // The object intercepted by the ray
  Model* object;
  // The object's material
  int materialIndex;
  // Flags
  Flags flags;
  // Any user data
  void* userData;

}; // IntersectInfo

} // end namespace Graphics

#endif // __Ray_h
