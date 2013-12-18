#ifndef __Light_h
#define __Light_h

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
//  OVERVIEW: Light.h
//  ========
//  Class definition for light.

#ifndef __Color_h
#include "Color.h"
#endif
#ifndef __DoubleList_h
#include "DoubleList.h"
#endif
#ifndef __SceneComponent_h
#include "SceneComponent.h"
#endif
#ifndef __Transform3_h
#include "Transform3.h"
#endif
#ifndef __Utils_h
#include "Utils.h"
#endif

using namespace Utils;

namespace Graphics
{ // begin namespace Graphics


//////////////////////////////////////////////////////////
//
// Light: light class
// =====
class Light: public SceneComponent
{
public:
  enum
  {
    Linear = 1,
    Squared = 2,
    Directional = 4,
    TurnedOn = 8
  };

  Vec3 position;
  Color color;
  Flags flags;

  // Constructor
  Light(const Vec3& p, const Color& c = Color::white):
    position(p),
    color(c),
    flags(TurnedOn)
  {
    // do nothing
  }

  bool isDirectional() const
  {
    return flags.isSet(Directional);
  }

  void setDirectional(bool state)
  {
    flags.enable(Directional, state);
  }

  bool isTurnedOn() const
  {
    return flags.isSet(TurnedOn);
  }

  void setSwitch(bool state)
  {
    flags.enable(TurnedOn, state);
  }

  Color getScaledColor(REAL) const;
  void lightVector(const Vec3&, Vec3&, REAL&) const;

private:
  DECLARE_DOUBLE_LIST_ELEMENT(Light);

  friend class Scene;

}; // Light

typedef DoubleListImp<Light> Lights;
typedef DoubleListIteratorImp<Light> LightIterator;

__host__ __device__ inline Color
getLightScaledColor(const Color& color, Flags flags, REAL distance)
{
  if (flags.isSet(Light::Directional))
    return color;
  if (!flags.test(Light::Linear | Light::Squared))
    return color;

  REAL f = Math::inverse<REAL>(distance);

  if (flags.isSet(Light::Squared))
    f *= f;
  return color * f;
}

__host__ __device__ inline void
lightVector(const Vec3& position,
  Flags flags,
  const Vec3& P,
  Vec3& L,
  REAL& distance)
{
  if (flags.isSet(Light::Directional))
  {
    L = position.versor();
    distance = Math::infinity<REAL>();
  }
  else
  {
    distance = (L = position - P).length();
    if (Math::isZero(distance))
      return;
    L *= Math::inverse<REAL>(distance);
  }
}


//////////////////////////////////////////////////////////
//
// Light inline implementation
// =====
inline Color
Light::getScaledColor(REAL distance) const
{
  return getLightScaledColor(color, flags, distance);
}

inline void
Light::lightVector(const Vec3& P, Vec3& L, REAL& distance) const
{
  return Graphics::lightVector(position, flags, P, L, distance);
}

} // end namespace Graphics

#endif // __Light_h
