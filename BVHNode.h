#ifndef __BVHNode_h
#define __BVHNode_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2009, Dilson e Paulo                           |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: BVHNode.h
//  ========
//  Class definition for BVH node.

#ifndef __BoundingBox_h
#include "BoundingBox.h"
#endif
#ifndef __TriangleMesh_h
#include "TriangleMesh.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// BVHNode: BVH node class
// =======
struct BVHNode: public BoundingBox
{
  REAL w1;
  REAL w2;

  __host__ __device__
  void lChild(int i)
  {
    w1 = intAsReal<REAL>(i);
  }

  __host__ __device__
  void rChild(int i)
  {
    w2 = intAsReal<REAL>(i);
  }

  __host__ __device__
  void begin(int i)
  {
    w1 = intAsReal<REAL>(-1 - i);
  }

  __host__ __device__
  void end(int i)
  {
    w2 = intAsReal<REAL>(-1 - i);
  }

  __host__ __device__
  int lChild() const
  {
    return realAsInt<REAL>(w1);
  }

  __host__ __device__
  int rChild() const
  {
    return realAsInt<REAL>(w2);
  }

  __host__ __device__
  int begin() const
  {
    return -1 - realAsInt<REAL>(w1);
  }

  __host__ __device__
  int end() const
  {
    return -1 - realAsInt<REAL>(w2);
  }

}; // BVHNode

inline __host__ __device__ REAL
intersectLeaf(BVHNode* leaf,
  const TriangleMesh::Data& geometry,
  const Ray& ray,
  IntersectInfo& hit)
{
  for (int e = leaf->end(), i = leaf->begin(); i <= e; i++)
  {
    Vec3 p;
    REAL d;

    if (!geometry.intersect(i, ray, p, d))
      continue;
    if (zero<REAL>() < d && d < hit.distance)
    {
      TriangleMesh::Triangle* t = geometry.triangles + i;

      hit.p = p;
      hit.distance = d;
      hit.materialIndex = t->materialIndex();
      hit.userData = t;
    }
  }
  return hit.distance;
}

#define BVH_STACK_SIZE 30

inline __host__ __device__ bool
intersectBVH(BVHNode* bvh,
  const TriangleMesh::Data& geometry,
  const Ray& ray,
  IntersectInfo& hit,
  REAL dmax)
{
  RayInfo r(ray);

  hit.distance = dmax;
  hit.userData = 0;
  {
    REAL d;

    if (!bvh[0].intersect(r, d))
      return false;
  }

  int boxStack[BVH_STACK_SIZE];
  int top = 0;
  BVHNode* node = bvh;

  boxStack[top++] = -1;
  while (top != 0)
  {
    if (node->lChild() < 0)
    {
      IntersectInfo h;

      h.distance = hit.distance;
      if (intersectLeaf(node, geometry, ray, h) < hit.distance)
      {
        hit.p = h.p;
        hit.distance = h.distance;
        hit.materialIndex = h.materialIndex;
        hit.userData = h.userData;
      }
      node = bvh + boxStack[--top];
      continue;
    }

    do
    {
      int lChild = node->lChild();
      int rChild = node->rChild();
      REAL d1;
      REAL d2;
      bool inter1 = bvh[lChild].intersect(r, d1);
      bool inter2 = bvh[rChild].intersect(r, d2);

      if (inter1)
      {
        if (inter2)
        {
          if (d2 < d1)
          {
            int temp = lChild;

            lChild = rChild;
            rChild = temp;
          }
          boxStack[top++] = rChild;
        }
        node = bvh + lChild;
      }
      else if (inter2)
        node = bvh + rChild;
      else
        node = bvh + boxStack[--top];
    } while (top != 0 && node->lChild() >= 0);
  }
  return hit.userData != 0;
}

extern void printBVH(const BVHNode*, int, FILE*);

inline void
printBVH(const char* fileName, const BVHNode* nodes)
{
  FILE* file = fopen(fileName, "w");

  printBVH(nodes, 0, file);
  fclose(file);
}

#endif // __BVHNode_h
