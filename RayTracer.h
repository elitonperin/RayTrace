#ifndef __RayTracer_h
#define __RayTracer_h

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
//  OVERVIEW: RayTracer.h
//  ========
//  Class definition for simple ray tracer.

#ifndef __BVH_h
#include "BVH.h"
#endif
#ifndef __GLImage_h
#include "GLImage.h"
#endif
#ifndef __Renderer_h
#include "Renderer.h"
#endif

namespace Graphics
{ // begin namespace Graphics

//
// Ray tracer settings struct
//
struct RayTracerSettings
{
  uint H;
  uint W;
  uint maxRecursionLevel;
  REAL minWeight;

}; // RayTracerSettings


//////////////////////////////////////////////////////////
//
// RayTracer: simple ray tracer class
// =========
class RayTracer: public Renderer
{
public:
  // Constructor
  RayTracer(Scene&, Camera* = 0, Intersector* = new BVH());

  int getMaxRecursionLevel() const;
  REAL getMinWeight() const;

  void setMaxRecursionLevel(int);
  void setMinWeight(REAL);

  void render();
  void renderImage(Image&);

protected:
  Ray pixelRay;
  // Interreflection parameters
  int maxRecursionLevel;
  REAL minWeight;
  // Scene intersector
  ObjectPtr<Intersector> intersector;

  void scan(Image&);
  void setPixelRay(REAL, REAL);
  Color shoot(REAL, REAL);
  Color trace(const Ray&, REAL, int, REAL);
  bool intersect(const Ray&, IntersectInfo&, REAL);

  /*
   * TODO: declare your methods here
   */

}; // RayTracer


//////////////////////////////////////////////////////////
//
// RayTracer inline implementation
// =========
inline REAL
RayTracer::getMinWeight() const
{
  return minWeight;
}

inline void
RayTracer::setMinWeight(REAL minWeight)
{
  this->minWeight = minWeight;
}

inline int
RayTracer::getMaxRecursionLevel() const
{
  return maxRecursionLevel;
}

inline void
RayTracer::setMaxRecursionLevel(int maxRecursionLevel)
{
  this->maxRecursionLevel = maxRecursionLevel;
}

} // end namespace Graphics

#endif // __RayTracer_h
