#ifndef __MeshSweeper_h
#define __MeshSweeper_h

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
//  OVERVIEW: MeshSweeper.h
//  ========
//  Class definition for mesh sweeper.

#ifndef __Sweeper_h
#include "Sweeper.h"
#endif
#ifndef __TriangleMeshShape_h
#include "TriangleMeshShape.h"
#endif

namespace Graphics
{ // begin namespace Graphics


//////////////////////////////////////////////////////////
//
// MeshSweeper: mesh sweeper class
// ===========
class MeshSweeper: public Sweeper
{
public:
  // Make box
  static TriangleMeshShape* makeBox(const Vec3& center,
    const Vec3& normal,
    const Vec3& up,
    const Vec3& size);

  static TriangleMeshShape* makeBox(const Vec3& center,
    const Vec3& orientation,
    const Vec3& scale);

  // Make cube
  static TriangleMeshShape* makeCube(const Vec3& center,
    const Vec3& normal,
    const Vec3& up,
    REAL size)
  {
    return makeBox(center, normal, up, Vec3(size, size, size));
  }

  static TriangleMeshShape* makeCube(const Vec3& center,
    const Vec3& orientation,
    REAL scale)
  {
    return makeBox(center, orientation, Vec3(scale, scale, scale));
  }

  static TriangleMeshShape* makeCube()
  {
    return makeCube(Vec3(0, 0, 0), Vec3(0, 0, 0), 1);
  }

  // Make sphere
  static TriangleMeshShape* makeSphere(const Vec3& center,
    REAL radius,
    int meridians = 16);

  static TriangleMeshShape* makeSphere()
  {
    return makeSphere(Vec3(0, 0, 0), 1);
  }

  // Make cylinder
  static TriangleMeshShape* makeCylinder(const Vec3& center,
    REAL radius,
    const Vec3& normal,
    REAL height,
    int segments = 16);

  // Make cone
  static TriangleMeshShape* makeCone(const Vec3& center,
    REAL radius,
    const Vec3& normal,
    REAL height,
    int segments = 16);

private:
  // Make cylinder
  static TriangleMeshShape* makeCylinder(const Polyline& circle,
    const Vec3& path,
    bool flat);

  //Make cone
  static TriangleMeshShape* makeCone(const Polyline& circle,
    const Vec3& path, bool flat);
	

}; // MeshSweeper

} // end namespace Graphics

#endif // __MeshSweeper_h
