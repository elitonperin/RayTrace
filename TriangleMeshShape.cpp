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
//  OVERVIEW: TriangleMeshShape.cpp
//  ========
//  Source file for triangle mesh shape.

#ifndef __TriangleMeshShape_h
#include "TriangleMeshShape.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// TriangleMeshShape implementation
// =================
Object*
TriangleMeshShape::clone() const
//[]---------------------------------------------------[]
//|  Make copy                                          |
//[]---------------------------------------------------[]
{
  return new TriangleMeshShape(this->data.copy());
}

bool
TriangleMeshShape::intersect(const Ray& , IntersectInfo& ) const
//[]---------------------------------------------------[]
//|  Intersect       TODO                              |
//[]---------------------------------------------------[]
{

	return false;
}

Vec3
TriangleMeshShape::normal(const IntersectInfo& hit) const
//[]---------------------------------------------------[]
//|  Normal                                             |
//[]---------------------------------------------------[]
{
  return data.normalAt((Triangle*)hit.userData, hit.p);
}

const TriangleMesh*
TriangleMeshShape::triangleMesh() const
//[]---------------------------------------------------[]
//|  Triangle mesh                                      |
//[]---------------------------------------------------[]
{
  return this;
}

BoundingBox
TriangleMeshShape::boundingBox() const
//[]---------------------------------------------------[]
//|  Bounding box                                       |
//[]---------------------------------------------------[]
{
  BoundingBox box;

  for (int i = 0; i < data.numberOfVertices; i++)
    box.inflate(data.vertices[i]);
  return box;
}

void
TriangleMeshShape::transform(const Transf3& t)
//[]---------------------------------------------------[]
//|  Transform                                          |
//[]---------------------------------------------------[]
{
  for (int i = 0; i < data.numberOfVertices; i++)
    t.transformRef(data.vertices[i]);
  if (data.normals != 0)
    for (int i = 0; i < data.numberOfNormals; i++)
      t.transformVectorRef(data.normals[i]);
}

void
TriangleMeshShape::setMaterial(Material& m)
//[]---------------------------------------------------[]
//|  Set material                                       |
//[]---------------------------------------------------[]
{
  Primitive::setMaterial(m);
  for (int i = 0; i < data.numberOfTriangles; i++)
    data.triangles[i].v[3] = m.getIndex();
}
