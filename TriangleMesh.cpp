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
//  OVERVIEW: TriangleMesh.cpp
//  ========
//  Source file for simple triangle mesh.

#include <memory.h>

#ifndef __TriangleMesh_h
#include "TriangleMesh.h"
#endif

//
// Auxiliary functions
//
template <typename T>
inline void
copyArray(T* dst, const T* src, int n)
{
  memcpy(dst, src, n * sizeof(T));
}

template <typename T>
inline void
copyNewArray(T*& dst, const T* src, int n)
{
  copyArray<T>(dst = new T[n], src, n);
}

using namespace Graphics;

//
// Auxiliary function
//
inline void
printVec3(FILE*f, const char* s, const Vec3& p)
{
  fprintf(f, "%s<%g, %g, %g>\n", s, p.x, p.y, p.z);
}


//////////////////////////////////////////////////////////
//
// TriangleMesh implementation
// ============
TriangleMesh::Arrays
TriangleMesh::Arrays::copy() const
//[]---------------------------------------------------[]
//|  Copy data                                          |
//[]---------------------------------------------------[]
{
  Arrays c;

  if (vertices != 0)
  {
    c.numberOfVertices = numberOfVertices;
    ::copyNewArray(c.vertices, vertices, numberOfVertices);
  }
  if (normals != 0)
  {
    c.numberOfNormals = numberOfNormals;
    ::copyNewArray(c.normals, normals, numberOfNormals);
  }
  if (triangles != 0)
  {
    c.numberOfTriangles = numberOfTriangles;
    ::copyNewArray(c.triangles, triangles, numberOfTriangles);
  }
  if (colors != 0)
  {
    c.numberOfColors = numberOfColors;
    ::copyNewArray(c.colors, colors, numberOfColors);
  }
  return c;
}
