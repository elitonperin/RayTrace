#ifndef __Model_h
#define __Model_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Library                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007, Paulo Aristarco Pagliosa                 |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: Model.h
//  ========
//  Class definition for generic model.

#ifndef __BoundingBox_h
#include "BoundingBox.h"
#endif
#ifndef __Material_h
#include "Material.h"
#endif

namespace Graphics
{ // begin namespace Graphics

class TriangleMesh;


//////////////////////////////////////////////////////////
//
// Model: generic simple model class
// =====
class Model: public Object
{
public:
  // Destructor
  virtual ~Model()
  {
    // do nothing
  }

  virtual bool intersect(const Ray&, IntersectInfo&) const = 0;
  virtual Vec3 normal(const IntersectInfo&) const = 0;
  virtual int getMaterialIndex() const = 0;
  virtual BoundingBox boundingBox() const = 0;
  virtual const TriangleMesh* triangleMesh() const = 0;

  virtual void transform(const Transf3&) = 0;
  virtual void setMaterial(Material&) = 0;

}; // Model


//////////////////////////////////////////////////////////
//
// Primitive: generic primitive model class
// =========
class Primitive: public Model
{
public:
  int getMaterialIndex() const
  {
    return material->getIndex();
  }

  void setMaterial(Material& material)
  {
    this->material = &material;
  }

protected:
  ObjectPtr<Material> material;

  // Protected constructors
  Primitive():
    material(Material::getDefault())
  {
    // do nothing
  }

  Primitive(const Primitive& primitive):
    material(primitive.material)
  {
    // do nothing
  }

}; // Primitive

} // end namespace Graphics

#endif // __Model_h
