#ifndef __Material_h
#define __Material_h

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
// OVERVIEW: Material.h
// ========
// Class definition for material.

#ifndef __Array_h
#include "Array.h"
#endif
#ifndef __Color_h
#include "Color.h"
#endif
#ifndef __NameableObject_h
#include "NameableObject.h"
#endif

using namespace System;
using namespace System::Collections;

namespace Graphics
{ // begin namespace Graphics


//////////////////////////////////////////////////////////
//
// Finish: finish class
// ======
class Finish
{
public:
  // Constructor
  Finish():
    ambient(0.2f),
    diffuse(0.8f),
    specular(0),
    shine(0),
    spot(0),
    transparency(0),
    IOR(1)
  {
    // do nothing
  }

  float ambient;
  float diffuse;
  float specular;
  float shine;
  float spot;
  float transparency;
  float IOR;

}; // Finish


//////////////////////////////////////////////////////////
//
// Material: material class
// ========
class Material: public NameableObject
{
public:
  class Surface
  {
  public:
    Color ambient;      // ambient color
    Color diffuse;      // diffuse color
    float shine;        // specular spot exponent
    Color spot;         // specular spot color
    Color specular;     // specular color
    Color transparency; // transparency color
    float IOR;          // index of refraction

  }; // Surface

  Surface surface;

  int getIndex() const
  {
    return index;
  }

  static Material* getDefault();

  void setSurface(const Color&, const Finish&);

protected:
  // Constructor
  Material(const string&);

private:
  int index;

  friend class MaterialFactory;

}; // Material

typedef PointerArrayIterator<Material> MaterialIterator;


//////////////////////////////////////////////////////////
//
// MaterialFactory: material factory class
// ===============
class MaterialFactory
{
public:
  class Materials: public PointerArray<Material>
  {
  public:
    Material* defaultMaterial;

    // Constructor
    Materials()
    {
      defaultMaterial = MaterialFactory::New("default");
    }

  }; // Materials

  static Material* New();
  static Material* New(const string&);

  static Material* get(const string&);

  static Material* get(int id)
  {
    return materials[id];
  }

  static Material* getDefaultMaterial()
  {
    return materials.defaultMaterial;
  }

  static int size()
  {
    return materials.size();
  }

  static MaterialIterator iterator()
  {
    return MaterialIterator(materials);
  }

private:
  static Materials materials;

  static void add(Material* material)
  {
    material->index = materials.size();
    materials.add(makeUse(material));
  }

}; // MaterialFactory


//////////////////////////////////////////////////////////
//
// Material inline implementtaion
// ========
inline Material*
Material::getDefault()
{
  return MaterialFactory::getDefaultMaterial();
}

} // end namespace Graphics

#endif // __Material_h
