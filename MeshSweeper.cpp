//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2009, Paulo Aristarco Pagliosa                 |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: MeshSweeper.cpp
//  ========
//  Source file for mesh sweeper.

#include <stdio.h>

#ifndef __MeshSweeper_h
#include "MeshSweeper.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// MeshSweeper implementation
// ===========
//

TriangleMeshShape*
	MeshSweeper::makeCone(const Polyline& circle, const Vec3& path, bool flat)
//[]----------------------------------------------------[]
//|  Make cone                                       |
//[]----------------------------------------------------[]
{
	int np = circle.getNumberOfVertices();
	int nv = np *2; // number of vertices
	int nb = np - 2; // number of triangles of the base
	int nt = np + nb; // number of triangles
	TriangleMesh::Arrays data;
	
	data.vertices = new Vec3[data.numberOfVertices = nv];
	data.normals = new Vec3[data.numberOfNormals = np + 2];
	data.triangles = new TriangleMesh::Triangle[data.numberOfTriangles = nt];

	Vec3 c(0, 0, 0);

	if (true)
	{
		Polyline::VertexIterator vit(circle.getVertexIterator());

		for (int i = 0; i < np; i++)
		{
			const Vec3 p = vit++.position;
			data.vertices[i] = p;
			c+=p;
		}

		c *= Math::inverse<REAL>((REAL)np);
		
		// Adiciona vértice superior do cone
		data.vertices[np+1] =  path + data.vertices[np].normalize();

	}

	if (true)
	{
		Vec3 N = triangleNormal(data.vertices);

		data.normals[np] = N;
		data.normals[np+1] = -N;
	}

	TriangleMesh::Triangle* triangle = data.triangles;

	int npm = np + 1;

	for (int i = 0; i < np; i++)
	{
		int k = (i + 1) % np;
		triangle->setVertices(i, npm, k);

		if (flat)
		{
		  data.normals[i] = triangleNormal(data.vertices, i, npm, k);
		  triangle->setNormal(i);
		}
		else
		{
		  data.normals[i] = (data.vertices[i + np] - c).versor();
		  triangle->setNormals(i, i, k);
		}

		triangle++;
	}

	int v0 = 0;
	int v1 = 1;
	int v2 = 2;

	int aux = np - 2;

	for (int i = 0; i < aux; i++)
	{
		triangle->setVertices(v0, v1, v2);
		triangle->setNormal(np);
		triangle++;
		v1 = (v1+1) % np;
		v2 = (v2+1) % np;
	}

	return new TriangleMeshShape(data);

}


TriangleMeshShape*
MeshSweeper::makeCylinder(const Polyline& circle, const Vec3& path, bool flat)
//[]----------------------------------------------------[]
//|  Make cylinder                                       |
//[]----------------------------------------------------[]
{
  int np = circle.getNumberOfVertices();
  int nv = np * 2; // number of vertices
  int nb = np - 2; // number of triangles of the base
  int nt = nv + 2 * nb; // number of triangles
  TriangleMesh::Arrays data;

  data.vertices = new Vec3[data.numberOfVertices = nv];
  data.normals = new Vec3[data.numberOfNormals = np + 2];
  data.triangles = new TriangleMesh::Triangle[data.numberOfTriangles = nt];

  Vec3 c(0, 0, 0);

  if (true)
  {
    Polyline::VertexIterator vit(circle.getVertexIterator());

    for (int i = 0; i < np; i++)
    {
      const Vec3 p = vit++.position;

      c += p;
      data.vertices[i + np] = p;
      data.vertices[i] = p + path;
    }
    c *= Math::inverse<REAL>((REAL)np);
  }
  if (true)
  {
    Vec3 N = triangleNormal(data.vertices); // base normal

    data.normals[np] = N;
    data.normals[np + 1] = -N;
  }

  TriangleMesh::Triangle* triangle = data.triangles;

  for (int i = 0; i < np; i++)
  {
    int j = (i + np);
    int k = (i + 1) % np;

    triangle->setVertices(i, j, k);
    triangle[1].setVertices(j, k + np, k);
    if (flat)
    {
      data.normals[i] = triangleNormal(data.vertices, i, j, k);
      triangle->setNormal(i);
      triangle[1].setNormal(i);
    }
    else
    {
      data.normals[i] = (data.vertices[i + np] - c).versor();
      triangle->setNormals(i, i, k);
      triangle[1].setNormals(i, k, k);
    }
    triangle += 2;
  }

  int v0 = 0;
  int v1 = 1;
  int v2 = 2;

  for (int i = 0; i < nb; i++)
  {
    triangle->setVertices(v0, v1, v2);
    triangle[nb].setVertices(v0 + np, v2 + np, v1 + np);
    triangle->setNormal(np);
    triangle[nb].setNormal(np + 1);
    triangle++;
    //v2 = ((v1 = (v0 = v2) + 1) + 1) % np;
	v1 = (v1+1) % np;
	v2 = (v2+1) % np;
  }
  return new TriangleMeshShape(data);
}

TriangleMeshShape*
MeshSweeper::makeCone(const Vec3& center, REAL radius, const Vec3& normal, const REAL height, int segments)
//[]----------------------------------------------------[]
//|  Make cone                                       |
//[]----------------------------------------------------[]
{
	Polyline circle = makeCircle(center, radius, normal, segments);
	return makeCone(circle, normal * -height, false);
}

TriangleMeshShape*
MeshSweeper::makeBox(const Vec3& center,
  const Vec3& normal,
  const Vec3& up,
  const Vec3& size)
//[]----------------------------------------------------[]
//|  Make box 1                                          |
//[]----------------------------------------------------[]
{
  Polyline poly;
  Vec3 N(normal.versor());
  Vec3 U(up.cross(normal).versor());
  Vec3 V(N.cross(U));

  N *= size.z * (REAL)0.5;
  U *= size.x * (REAL)0.5;
  V *= size.y * (REAL)0.5;
  poly.mv(center - U - V - N);
  poly.mv(center + U - V - N);
  poly.mv(center + U + V - N);
  poly.mv(center - U + V - N);
  poly.close();
  return makeCylinder(poly, 2 * N, true);
}

TriangleMeshShape*
MeshSweeper::makeBox(const Vec3& center,
  const Vec3& orientation,
  const Vec3& scale)
//[]----------------------------------------------------[]
//|  Make box 2                                          |
//[]----------------------------------------------------[]
{
#define BOX_O  Vec3(0,0,0)
#define BOX_X  Vec3(1,0,0)
#define BOX_Y  Vec3(0,1,0)
#define BOX_Z  Vec3(0,0,1)
#define BOX_S  Vec3(1,1,1)

  TriangleMeshShape* box = makeBox(BOX_O, BOX_Z, BOX_Y, BOX_S);
  Transf3 m2g, temp;

  m2g.scale(scale);
  temp.rotationY(Math::toRadians<REAL>(orientation.y));
  m2g.compose(temp);
  temp.rotationX(Math::toRadians<REAL>(orientation.x));
  m2g.compose(temp);
  temp.rotationZ(Math::toRadians<REAL>(orientation.z));
  m2g.compose(temp);
  temp.translation(center);
  m2g.compose(temp);
  box->transform(m2g);
  return box;

#undef BOX_O
#undef BOX_X
#undef BOX_Y
#undef BOX_Z
#undef BOX_S
}

TriangleMeshShape*
MeshSweeper::makeCylinder(const Vec3& center,
  REAL radius,
  const Vec3& normal,
  REAL height,
  int segments)
//[]----------------------------------------------------[]
//|  Make cylinder                                       |
//[]----------------------------------------------------[]
{
  Polyline circle = makeCircle(center, radius, normal, segments);

  return makeCylinder(circle, normal * -height, false);
}
/*
TriangleMeshShape*
MeshSweeper::makeCone(const Vec3& center,
  REAL radius,
  const Vec3& normal,
  REAL height,
  int segments)
//[]----------------------------------------------------[]
//|  Make cylinder                                       |
//[]----------------------------------------------------[]
{
  Polyline circle = makeCircle(center, radius, normal, segments);

  return makeCone(circle, normal * -height);
}
*/
TriangleMeshShape*
MeshSweeper::makeSphere(const Vec3& center, REAL radius, int mers)
//[]----------------------------------------------------[]
//|  Make sphere                                         |
//[]----------------------------------------------------[]
{
  if (mers < 6)
    mers = 6;

  int sections = mers;
  int nv = sections * mers + 2; // number of vertices (and normals)
  int nt = 2 * mers * sections; // number of triangles
  TriangleMesh::Arrays data;

  data.vertices = new Vec3[data.numberOfVertices = nv];
  data.normals = new Vec3[data.numberOfNormals = nv];
  data.triangles = new TriangleMesh::Triangle[data.numberOfTriangles = nt];
  {
    Polyline arc = makeArc(center, radius, Vec3(0, 0, 1), 180, sections + 1);
    Polyline::VertexIterator vit = arc.getVertexIterator();
    Transf3 rot;
    Vec3* vertex = data.vertices;
    Vec3* normal = data.normals;
    REAL invRadius = Math::inverse<REAL>(radius);

    *normal = ((*vertex = (vit++).position) - center) * invRadius;
    rot.rotation(center, *normal, Math::toRadians<REAL>(360) / mers);
    vertex++;
    normal++;
    for (int s = 0; s < sections; s++)
    {
      Vec3 p = *vertex = (vit++).position;

      *normal = (p - center) * invRadius;
      vertex++;
      normal++;
      for (int m = 1; m < mers; m++)
      {
        *vertex = rot.transformRef(p);
        *normal = (p - center) * invRadius;
        vertex++;
        normal++;
      }
    }
    *normal = ((*vertex = (vit++).position) - center) * invRadius;
  }

  TriangleMesh::Triangle* triangle = data.triangles;

  for (int i = 1; i <= mers; i++)
  {
    int j = i % mers + 1;

    triangle->setVertices(0, i, j);
    triangle->setNormals(0, i, j);
    triangle++;
  }
  for (int s = 1; s < sections; s++)
    for (int m = 0, b = (s - 1) * mers + 1; m < mers;)
    {
      int i = b + m;
      int k = b + ++m % mers;
      int j = i + mers;
      int l = k + mers;

      triangle->setVertices(i, j, k);
      triangle->setNormals(i, j, k);
      triangle[1].setVertices(k, j, l);
      triangle[1].setNormals(k, j, l);
      triangle += 2;
    }
  for (int m = 0, b = (sections - 1) * mers + 1, j = nv - 1; m < mers;)
  {
    int i = b + m;
    int k = b + ++m % mers;

    triangle->setVertices(i, j, k);
    triangle->setNormals(i, j, k);
    triangle++;
  }
  return new TriangleMeshShape(data);
}
