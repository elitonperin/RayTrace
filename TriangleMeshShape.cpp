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
TriangleMeshShape::intersect(const Ray& ray, IntersectInfo& intersectInfo) const
//[]---------------------------------------------------[]
//|  Intersect       TODO                              |
//[]---------------------------------------------------[]
{
	/*
	 
bool TriangleMesh::intersect(const Ray& ray, IntersectInfo& info, float tmax)const
{
	bool hit = false;
	float distance = tmax;
	for(int index=0; index < numberOfTriangles; index++)
	{
	  const Vec3& p0 = v[triangles[index].v[0]];
	  const Vec3& p1 = v[triangles[index].v[1]];
	  const Vec3& p2 = v[triangles[index].v[2]];
	  float t, b1, b2;
	    if(intersectTri(ray, p0, p1, p2, t, b1, b2, tmax))
	    {
	      if(t>0.01f){
		hit = true;
		if(t<distance)
		{
		  info.index = index;
		  info.b1 = b1;
		  info.b2 = b2;
		  info.t = t;
		  distance = t;
		}
	      }
	    }
	}
	if(hit)
		info.model = (Model*)this;
	return hit;
}
	

	bool hit = false;
	REAL distance = intersectInfo.distance;

	for(int i=0; i < data.numberOfTriangles; i++)
	{
		const Vec3& v0 = data.vertices[data.triangles[i].v[0]];
		const Vec3& v1 = data.vertices[data.triangles[i].v[1]];
		const Vec3& v2 = data.vertices[data.triangles[i].v[2]];
		
		Vec3 * vector = new Vec3(v0, v1, v2);

		REAL t, b1, b2;

		
		if()
	    {
	      if(t>0.01f){
		hit = true;
		if(t<distance)
		{
		  intersectInfo. = index;
		  intersectInfo.b1 = b1;
		  info.b2 = b2;
		  info.t = t;
		  distance = t;
		}
	      }
	    }
	}
	if(hit)
		info.model = (Model*)this;
	return hit;
	return false;
	*/
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
