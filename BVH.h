#ifndef __BVH_h
#define __BVH_h

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
//  OVERVIEW: BVH.h
//  ========
//  Class definition for BVH.

#ifndef __BVHNode_h
#include "BVHNode.h"
#endif
#ifndef __Intersector_h
#include "Intersector.h"
#endif
#ifndef __TriangleMeshShape_h
#include "TriangleMeshShape.h"
#endif

struct GeometryData: public TriangleMesh::Arrays
{
  int numberOfStaticTriangles;

  // Constructor
  GeometryData()
  {
    numberOfStaticTriangles = 0;
  }

}; // GeometryData

//
// Auxiliary function
//
extern GeometryData mergeGeometry(const Scene&);


//////////////////////////////////////////////////////////
//
// BVH: BVH class
// ===
class BVH: public Intersector
{
public:
  int maxLevel;

  // Constructor
  BVH():
    mesh(0),
    nodes(0),
    maxLevel(-1)
  {
    numberOfNodes = numberOfStaticNodes = 0;
  }

  // Destructor
  ~BVH();

  void init(const Scene&);
  bool intersect(const Ray&, IntersectInfo&, REAL);

  const GeometryData& getGeometry() const
  {
    return geometry;
  }

  int getNumberOfNodes() const
  {
    return numberOfNodes;
  }

  int getNumberOfStaticNodes() const
  {
    return numberOfStaticNodes;
  }

  BVHNode* getNodes() const
  {
    return nodes;
  }

protected:
  GeometryData geometry;
  TriangleMeshShape* mesh;
  BVHNode* nodes;
  int numberOfNodes;
  int numberOfStaticNodes;

  void build(BVHNode&, int, int);
  void split(BVHNode&, int);

}; // BVH

#endif // __BVH_h
