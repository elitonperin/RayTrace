#ifndef __TriangleMesh_h
#define __TriangleMesh_h

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
//  OVERVIEW: TriangleMesh.h
//  ========
//  Class definition for simple triangle mesh.

#ifndef __Color_h
#include "Color.h"
#endif
#ifndef __Ray_h
#include "Ray.h"
#endif

namespace Graphics
{ // begin namespace Graphics

//
// Auxiliary functions
//

	
//
// Auxiliary functions
//
inline void
eliminateDominant(Vec3 &v, int d)
{
	if (d == 0)
		v.x = v.z;
	else if (d == 1)
		v.y = v.z;
}
__host__ __device__ inline Vec3
triangleNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
  return ((v1 - v0).cross(v2 - v0)).versor();
}

__host__ __device__ inline Vec3
triangleNormal(Vec3* v)
{
  return triangleNormal(v[0], v[1], v[2]);
}

__host__ __device__ inline Vec3
triangleNormal(Vec3* v, int i, int j, int k)
{
  return triangleNormal(v[i], v[j], v[k]);
}

__host__ __device__ inline Vec3
triangleNormal(Vec3* v, int i[3])
{
  return triangleNormal(v[i[0]], v[i[1]], v[i[2]]);
}

__host__ __device__ inline Vec3
triangleCenter(const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
  return (v0 + v1 + v2) * Math::inverse<REAL>(3);
}

__host__ __device__ inline Vec3
triangleCenter(Vec3* v)
{
  return triangleCenter(v[0], v[1], v[2]);
}

__host__ __device__ inline Vec3
triangleCenter(Vec3* v, int i, int j, int k)
{
  return triangleCenter(v[i], v[j], v[k]);
}

__host__ __device__ inline Vec3
triangleCenter(Vec3* v, int i[3])
{
  return triangleCenter(v[i[0]], v[i[1]], v[i[2]]);
}


//////////////////////////////////////////////////////////
//
// Triangle: simple triangle mesh class
// ========
struct __align__(16) Triangle
{
  Vec3 v0, v1, v2;
  Vec3 N;

  // Constructors
  __host__ __device__
  Triangle()
  {
    // do nothing
  }

  __host__ __device__
  Triangle(const Vec3& p0, const Vec3& p1, const Vec3& p2)
  {
    v0 = p0;
    v1 = p1;
    v2 = p2;
    computeNormal();
  }

  __host__ __device__
  Triangle(Vec3* p)
  {
    v0 = p[0];
    v1 = p[1];
    v2 = p[2];
    computeNormal();
  }

  __host__ __device__
  Triangle(Vec3* p, int i[3])
  {
    v0 = p[i[0]];
    v1 = p[i[1]];
    v2 = p[i[2]];
    computeNormal();
  }

  __host__ __device__
  Vec3 center() const
  {
    return triangleCenter(v0, v1, v2);
  }

  template <typename T>
  __host__ __device__
  static T interpolate(const Vec3& p, const T& v0, const T& v1, const T& v2)
  {
    return v0 * p.x + v1 * p.y + v2 * p.z;
  }

  template <typename T>
  __host__ __device__
  static T interpolate(const Vec3& p, T v[3])
  {
    return interpolate<T>(p, v[0], v[1], v[2]);
  }

  __host__ __device__
  bool intersect(const Ray& ray, Vec3& p, REAL& t)
  {
    /*
    * TODO: insert your code here
	*	Ray intersect triangle
	* **** conforme visto em sala C[] = A*B[]
	*/

	Vec3 e1 = v1 - v0;
	Vec3 e2 = v2 - v0;

	Vec3 s1 = ray.direction.cross(e2);

	REAL aux = s1.inner(e1);

	if(s1.isNull() || Math::isZero(aux))
	  return false;

	// Parte A da equação
	aux = Math::inverse(aux);

	// Vetor S
	Vec3 s = ray.origin - v0;
	Vec3 s2 = s.cross(e1);

	// Parte B[] da equação
	REAL b1 = s1.inner(s) * aux;
	REAL b2 = s2.inner(ray.direction) * aux;

	REAL one = 1.f;

	// Se B1 for zero, Se B2 for zero ou se a soma dos dois for maior que 1
	if(Math::isZero(b1) || Math::isZero(b2) ||  Math::isEqual(b1+b2, one)  ) {
		return false;
	}

	t = s2.inner(e2)*aux;

	if(Math::isZero(t) || Math::isNegative(t)) {
		return false;
	}

	return true;
  }

private:
  __host__ __device__
  void computeNormal()
  {
    N = triangleNormal(v0, v1, v2);
  }

}; // Triangle


//////////////////////////////////////////////////////////
//
// TriangleMesh: simple triangle mesh class
// ============
class TriangleMesh
{
public:
  struct __align__(16) Triangle
  {
    int v[4]; // v[3] = material index
    int n[4];

    // Constructor
    Triangle()
    {
      v[3] = 0; n[3] = -1;
    }

    void setVertices(int v0, int v1, int v2)
    {
      v[0] = v0;
      v[1] = v1;
      v[2] = v2;
    }

    void setNormal(int i)
    {
      n[0] = n[1] = n[2] = n[3] = i;
    }

    void setNormals(int n0, int n1, int n2)
    {
      n[0] = n0;
      n[1] = n1;
      n[2] = n[3] = n2;
    }

    __host__ __device__
    int materialIndex() const
    {
      return v[3];
    }

  }; // Triangle

  struct Data
  {
    Vec3* vertices;
    Vec3* normals;
    Triangle* triangles;
    Color* colors;

    __host__ __device__
    bool intersect(int i, const Ray& ray, Vec3& p, REAL& d) const
    {
      Graphics::Triangle t(vertices, triangles[i].v);
      return t.intersect(ray, p, d);
    }

    __host__ __device__
    Vec3 normalAt(Triangle* t, const Vec3& p) const
    {
      if (t->n[3] == -1)
        return triangleNormal(vertices, t->v);

      Vec3 N0 = normals[t->n[0]];
      Vec3 N1 = normals[t->n[1]];
      Vec3 N2 = normals[t->n[2]];

      return Graphics::Triangle::interpolate<Vec3>(p, N0, N1, N2);
    }

  }; // Data

  struct Arrays: public Data
  {
    int numberOfVertices;
    int numberOfNormals;
    int numberOfTriangles;
    int numberOfColors;

    // Constructor
    Arrays():
      numberOfVertices(0),
      numberOfNormals(0),
      numberOfTriangles(0),
      numberOfColors(0)
    {
      vertices = 0;
      normals = 0;
      triangles = 0;
      colors = 0;
    }

    Arrays copy() const;

  }; // Arrays

  // Constructor
  TriangleMesh(const Arrays& aData):
    data(aData)
  {
    // do nothing
  }

  // Destructor
  ~TriangleMesh()
  {
    delete data.vertices;
    delete data.normals;
    delete data.triangles;
    delete data.colors;
  }

  void setColors(Color* colors, int n)
  {
    delete data.colors;
    data.colors = colors;
    data.numberOfColors = n;
  }

  const Arrays& getData() const
  {
    return data;
  }

protected:
  Arrays data;

}; // TriangleMesh

} // end namespace Graphics

#endif // __TriangleMesh_h
