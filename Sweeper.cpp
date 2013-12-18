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
//  OVERVIEW: Sweeper.cpp
//  ========
//  Source file for generic sweeper.

#ifndef __Sweeper_h
#include "Sweeper.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// Sweeper::Polyline implementation
// =================
void
Sweeper::Polyline::transform(const Transf3& m)
//[]----------------------------------------------------[]
//|  Transform                                           |
//[]----------------------------------------------------[]
{
  for (VertexIterator vit(getVertexIterator()); vit;)
    (vit++).transform(m);
}

Vec3
Sweeper::Polyline::normal() const
//[]----------------------------------------------------[]
//|  Normal                                              |
//[]----------------------------------------------------[]
{
  Vec3 N(0, 0, 0);
  VertexIterator vit(getVertexIterator());

  if (vit)
  {
    Vec3* first = &(vit++).position;
    Vec3* p = first;

    for (Vec3* q = 0; q != first;)
    {
      q = vit ? &(vit++).position : first;
      N.x += (p->y - q->y) * (p->z + q->z);
      N.y += (p->z - q->z) * (p->x + q->x);
      N.z += (p->x - q->x) * (p->y + q->y);
      p = q;
    }
  }
  return N;
}

//
// Auxiliary function
//
Vec3
getFirstPoint(const Vec3& center, REAL radius, const Vec3& normal)
{
  Vec3 p;

  if (fabs(normal.z) > M_SQRT2 * 0.5)
  {
      // Choose p in y-z plane
    REAL a = normal.y * normal.y + normal.z * normal.z;
    REAL k = Math::inverse<REAL>(sqrt(a));

    p.set(0, -normal.z * k, normal.y * k);
  }
  else
  {
    // Choose p in x-y plane
    REAL a = normal.x * normal.x + normal.y * normal.y;
    REAL k = Math::inverse<REAL>(sqrt(a));

    p.set(-normal.y * k, normal.x * k, 0);
  }
  return center + radius * p;
}


//////////////////////////////////////////////////////////
//
// Sweeper implementation
// =======
Sweeper::Polyline
Sweeper::makeArc(const Vec3& center,
  REAL radius,
  const Vec3& normal,
  REAL angle,
  int segments)
//[]----------------------------------------------------[]
//|  Make arc                                            |
//[]----------------------------------------------------[]
{
  Sweeper::Polyline poly;
  Transf3 m;
  Vec3 p = getFirstPoint(center, radius, normal);

  m.rotation(center, normal, Math::toRadians<REAL>(angle) / segments);
  poly.mv(p);
  for (int i = 1; i <= segments; i++)
  {
    m.transformRef(p);
    poly.mv(p);
  }
  return poly;
}

Sweeper::Polyline
Sweeper::makeCircle(const Vec3& center,
  REAL radius,
  const Vec3& normal,
  int points)
//[]----------------------------------------------------[]
//|  Make circle                                         |
//[]----------------------------------------------------[]
{
  Sweeper::Polyline poly;
  Transf3 m;
  Vec3 p = getFirstPoint(center, radius, normal);

  m.rotation(center, normal, Math::toRadians<REAL>(360) / points);
  poly.mv(p);
  for (int i = 1; i < points; i++)
  {
    m.transformRef(p);
    poly.mv(p);
  }
  poly.close();
  return poly;
}

