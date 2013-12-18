#ifndef __Intersector_h
#define __Intersector_h

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
//  OVERVIEW: Intersector.h
//  ========
//  Class definition for generic ray/object intersector.

#ifndef __Scene_h
#include "Scene.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// Intersector: generic ray/object intersector class
// ===========
class Intersector: public Object
{
public:
  // Destructor
  virtual ~Intersector()
  {
    // do nothing
  }

  virtual void init(const Scene&) = 0;
  virtual bool intersect(const Ray&, IntersectInfo&, REAL) = 0;

}; // Intersector

#endif // __Intersector_h
