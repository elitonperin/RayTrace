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
// OVERVIEW: Material.cpp
// ========
// Source file for material.

#ifndef __Material_h
#include "Material.h"
#endif

using namespace Graphics;


//////////////////////////////////////////////////////////
//
// Material implementation
// ========
Material::Material(const string& name):
  NameableObject(name)
//[]---------------------------------------------------[]
//|  Constructor                                        |
//[]---------------------------------------------------[]
{
  setSurface(Color(255, 255, 255), Finish());
}

void
Material::setSurface(const Color& color, const Finish& finish)
//[]---------------------------------------------------[]
//|  Set surface                                        |
//[]---------------------------------------------------[]
{
  surface.ambient = color * finish.ambient;
  surface.diffuse = color * finish.diffuse;
  surface.shine = finish.shine;
  surface.spot = color * finish.spot;
  surface.specular = color * finish.specular;
  surface.transparency = color * finish.transparency;
  surface.IOR = finish.IOR;
}

//////////////////////////////////////////////////////////
//
// MaterialFactory implementation
// ===============
MaterialFactory::Materials MaterialFactory::materials;

Material*
MaterialFactory::New()
//[]---------------------------------------------------[]
//|  Create material                                    |
//[]---------------------------------------------------[]
{
  int id = materials.size();
  char name[16];

  sprintf(name, "mat%d", id);

  Material* material = new Material(name);

  add(material);
  return material;
}

Material*
MaterialFactory::New(const string& name)
//[]---------------------------------------------------[]
//|  Create material                                    |
//[]---------------------------------------------------[]
{
  Material* material = get(name);
  
  if (material == 0)
    add(material = new Material(name));
  return material;
}

Material*
MaterialFactory::get(const string& name)
//[]---------------------------------------------------[]
//|  Get material                                       |
//[]---------------------------------------------------[]
{
  for (MaterialIterator mit(materials); mit; ++mit)
    if (name == mit.current()->getName())
      return mit.current();
  return 0;
}
