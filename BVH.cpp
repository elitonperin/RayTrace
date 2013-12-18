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
//  OVERVIEW: BVH.cpp
//  ========
//  Source file for BVH.

#include <memory.h>

#ifndef __BVH_h
#include "BVH.h"
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

GeometryData
mergeGeometry(const Scene& scene)
{
  GeometryData data;
  ActorIterator ait = scene.getActorIterator();

  for (Actor* actor; ait;)
  {
    if (!(actor = ait++)->isVisible())
      continue;

    const TriangleMesh* mesh = actor->getModel()->triangleMesh();

    if (mesh == 0)
      continue;

    const TriangleMesh::Arrays& meshData = mesh->getData();
    int nt = meshData.numberOfTriangles;

    if (!actor->isDynamic())
      data.numberOfStaticTriangles += nt;
    data.numberOfVertices += meshData.numberOfVertices;
    data.numberOfNormals += meshData.numberOfNormals;
    data.numberOfTriangles += nt;
  }
  data.vertices = new Vec3[data.numberOfVertices];
  if (data.numberOfNormals != 0)
    data.normals = new Vec3[data.numberOfNormals];
  data.triangles = new TriangleMesh::Triangle[data.numberOfTriangles];

  Vec3* vertices = data.vertices;
  Vec3* normals = data.normals;
  int vidx = 0;
  int nidx = 0;
  TriangleMesh::Triangle* triangles[2];

  triangles[0] = data.triangles;
  triangles[1] = data.triangles + data.numberOfStaticTriangles;
  ait.restart();
  for (Actor* actor; ait;)
  {
    if (!(actor = ait++)->isVisible())
      continue;

    const TriangleMesh* mesh = actor->getModel()->triangleMesh();

    if (mesh == 0)
      continue;

    const TriangleMesh::Arrays& meshData = mesh->getData();
    int nv = meshData.numberOfVertices;
    int nn = meshData.numberOfNormals;
    int nt = meshData.numberOfTriangles;
    TriangleMesh::Triangle*& t = triangles[actor->isDynamic()];

    copyArray(vertices, meshData.vertices, nv);
    if (nn > 0)
      copyArray(normals, meshData.normals, nn);
    copyArray(t, meshData.triangles, nt);
    if (vidx == 0)
      t += nt;
    else
      for (int i = 0; i < nt; i++, t++)
    {
      t->v[0] += vidx;
      t->v[1] += vidx;
      t->v[2] += vidx;
      if (t->n[3] == -1)
        continue;
      t->n[0] += nidx;
      t->n[1] += nidx;
      t->n[2] += nidx;
    }
    vertices += nv;
    vidx += nv;
    normals += nn;
    nidx += nn;
  }
  return data;
}

inline void
inflate(BoundingBox& b, const TriangleMesh::Data& g, int i)
{
  TriangleMesh::Triangle& t = g.triangles[i];

  b.inflate(g.vertices[t.v[0]]);
  b.inflate(g.vertices[t.v[1]]);
  b.inflate(g.vertices[t.v[2]]);
}

inline Vec3
getCenter(const TriangleMesh::Data& g, int i)
{
  return triangleCenter(g.vertices, g.triangles[i].v);
}


//////////////////////////////////////////////////////////
//
// BVH implementation
// ===
static const int binDim = 32;
static int* bin;
static int* nextBin;

BVH::~BVH()
//[]---------------------------------------------------[]
//|  Destructor                                         |
//[]---------------------------------------------------[]
{
  delete nodes;
  delete mesh;
}

FILE* d;

void
BVH::init(const Scene& scene)
//[]---------------------------------------------------[]
//|  Init                                               |
//[]---------------------------------------------------[]
{
  mesh = new TriangleMeshShape(geometry = mergeGeometry(scene));

  int nt = geometry.numberOfTriangles;
  int ns = geometry.numberOfStaticTriangles;

  printf("Scene geometry merging (CPU): %d triangles\n", nt);
  nodes = new BVHNode[2 * nt];
  if (ns > 0)
  {
    numberOfNodes = 1;
    build(nodes[1], 0, ns - 1);
  }
  if (nt - ns > 0)
  {
    numberOfStaticNodes = numberOfNodes++;
    build(nodes[numberOfNodes], ns, nt - 1);
  }
  numberOfNodes++;
  numberOfStaticNodes++;
  if (ns == 0 || nt == ns)
    nodes[0] = nodes[1];
  else
  {
    nodes->inflate(nodes[1]);
    nodes->lChild(1);
    nodes->inflate(nodes[numberOfStaticNodes]);
    nodes->rChild(numberOfStaticNodes);
  }
  printf("BVH building (CPU): %d nodes\n", numberOfNodes);
#if 0
  printBVH("bvh-cpu.dbg", nodes);
#endif
}

bool
BVH::intersect(const Ray& ray, IntersectInfo& hit, REAL dmax)
//[]---------------------------------------------------[]
//|  Intersect                                          |
//[]---------------------------------------------------[]
{
  return intersectBVH(nodes, geometry, ray, hit, dmax) ?
    hit.object = mesh, true : false;
}

void
BVH::build(BVHNode& node, int begin, int end)
//[]---------------------------------------------------[]
//|  Build                                              |
//[]---------------------------------------------------[]
{
  int nt = end + begin + 1;

  node.begin(begin);
  node.end(end);
  bin = new int[2 * nt];
  nextBin = bin + nt;
  split(node, 0);
  delete bin;
}

void
BVH::split(BVHNode& node, int level)
//[]---------------------------------------------------[]
//|  Split                                              |
//[]---------------------------------------------------[]
{
  int begin = node.begin();
  int end = node.end();
  int numberOfTriangles = end - begin + 1;

  for (int i = begin; i <= end; i++)
    inflate(node, geometry, i);
  if (numberOfTriangles <= 8)
    return;
  if (maxLevel > 0 && level == maxLevel)
    return;

  Vec3 size = node.getSize();
  REAL minCost = Math::infinity<REAL>();
  REAL minFirstPlane;
  REAL k[3];
  int minCostPlane = binDim + 1;
  int minAxis;
  int splitPoint;

  for (int axis = 0; axis < 3; axis++)
  {
    /*
    REAL fPlane(+M_INFINITY);
    REAL lPlane(-M_INFINITY);

    for (int i = begin; i <= end; i++)
    {
      REAL c = getCenter(geometry, i)[axis];

      if (c < fPlane)
        fPlane = c;
      if (c > lPlane)
        lPlane = c;
    }
    */
    REAL fPlane = node.getP1()[axis];
    REAL lPlane = node.getP2()[axis];

    if (Math::isZero(lPlane - fPlane))
    {
      /*
      printf("split BVH: degenerated AABB: %f-%f triangles: %d, axis: %d\n",
        fPlane,
        lPlane,
        numberOfTriangles,
        axis);

      REAL size = node.getMaxSize() * 0.5f;

      fPlane -= size;
      lPlane += size;
      */
      return;
    }

    int listNodes = 0;
    int binsSize[binDim];
    int binsHead[binDim];
    int binsTail[binDim];

    for (int i = 0; i < binDim; i++)
    {
      binsSize[i] = 0;
      binsHead[i] = binsTail[i] = -1;
    }
    k[axis] = binDim * (1.0f - 1e-6f) / (lPlane - fPlane);

    for (int i = begin; i <= end; i++)
    {
      int bid = (int)(k[axis] * (getCenter(geometry, i)[axis] - fPlane));

      if (bid >= binDim)
        bid = binDim - 1;
      try
      {
        PRECONDITION(bid >= 0 && bid < binDim);
      }
      catch (...)
      {
        printf("split BVH: invalid bin id: %d axis: %d\n", bid, axis);
        exit(0);
      }
      bin[listNodes] = i;
      nextBin[listNodes] = -1;
      if (binsHead[bid] == -1)
        binsHead[bid] = listNodes;
      else
        nextBin[binsTail[bid]] = listNodes;
      binsTail[bid] = listNodes;
      listNodes++;
    }

    REAL sap[binDim];
    REAL sas[binDim];
    REAL saInv = Math::inverse<REAL>(node.getArea());
    BoundingBox temp;

    for (int i = 0; i < binDim; i++)
    {
      int size = 0;

      for (int bid = binsHead[i]; bid != -1; bid = nextBin[bid], size++)
        inflate(temp, geometry, bin[bid]);
      sap[i] = temp.getArea();
      binsSize[i] = size;
      if (i > 0)
        binsSize[i] += binsSize[i - 1];
    }
    temp.setEmpty();
    for (int i = binDim - 1; i >= 0; i--)
    {
      for (int bid = binsHead[i]; bid != -1; bid = nextBin[bid])
        inflate(temp, geometry, bin[bid]);
      sas[i] = temp.getArea();
    }

    REAL minLocalCost = saInv * sap[binDim - 1] * numberOfTriangles;
    int minLocalCostPlane = -1;

    for (int i = 0; i < binDim; i++)
    {
      REAL cost = (sap[i] * binsSize[i] + sas[i] * (binsSize[binDim - 1] -
        binsSize[i])) * saInv;

      if (cost < minLocalCost)
      {
        minLocalCost = cost;
        minLocalCostPlane = i;
      }
    }
    if (minLocalCostPlane == -1)
    {
      /*
      if (numberOfTriangles <= 8)
        return;
      */
      int i = 0;

      while (i < binDim && binsSize[i] < binsSize[binDim - 1] - binsSize[i])
        i++;
      minLocalCostPlane = i;
    }
    if (minLocalCost < minCost)
    {
      minCost = minLocalCost;
      minCostPlane = minLocalCostPlane;
      minFirstPlane = fPlane;
      minAxis = axis;
      splitPoint = binsSize[minCostPlane] + begin - 1;
    }
  }

  int l = begin;
  int r = end;

  while (l < r)
  {
    while (l < r &&
      (int)(k[minAxis] * (getCenter(geometry, l)[minAxis] -
      minFirstPlane)) <= minCostPlane)
      l++;
    while (l < r &&
      (int)(k[minAxis] * (getCenter(geometry, r)[minAxis] -
      minFirstPlane)) >  minCostPlane)
      r--;
    System::swap(geometry.triangles[l], geometry.triangles[r]);
  }
  node.lChild(++numberOfNodes);
  node.rChild(++numberOfNodes);

  int lChild = node.lChild();
  int rChild = node.rChild();

  nodes[lChild].begin(begin);
  nodes[lChild].end(splitPoint);
  nodes[rChild].begin(splitPoint + 1);
  nodes[rChild].end(end);
  level++;
  split(nodes[lChild], level);
  split(nodes[rChild], level);
}

void
printBVH(const BVHNode* bvh, int id, FILE* file)
{
  const BVHNode* node = bvh + id;
  Vec3 p1 = node->getP1();
  Vec3 p2 = node->getP2();

  fprintf(file,
    "Node %d [<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>] ",
    id, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  if (node->lChild() < 0)
    fprintf(file, "begin: %d end: %d\n", node->begin(), node->end());
  else
  {
    int lChild = node->lChild();
    int rChild = node->rChild();

    fprintf(file, "lChild: %d rChild: %d\n", lChild, rChild);
    printBVH(bvh, lChild, file);
    printBVH(bvh, rChild, file);
  }
}
