#ifndef __Actor_h
#define __Actor_h

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
//  OVERVIEW: Actor.h
//  ========
//  Class definition for actor.

#ifndef __DoubleList_h
#include "DoubleList.h"
#endif
#ifndef __Model_h
#include "Model.h"
#endif
#ifndef __SceneComponent_h
#include "SceneComponent.h"
#endif

using namespace System;
using namespace System::Collections;

namespace Graphics
{ // begin namespace Graphics


//////////////////////////////////////////////////////////
//
// Actor: actor class
// =====
class Actor: public SceneComponent
{
public:
  enum
  {
    Visible = 1,
    Dynamic = 2
  };

  Flags flags;

  // Constructor
  Actor(Model& aModel):
    flags(Visible),
    model(&aModel)
  {
    // do nothing
  }

  bool isVisible() const
  {
    return flags.isSet(Visible);
  }

  void setVisible(bool state)
  {
    flags.enable(Visible, state);
  }

  bool isDynamic() const
  {
    return flags.isSet(Dynamic);
  }

  void setDynamic(bool state)
  {
    flags.enable(Dynamic, state);
  }

  Model* getModel() const
  {
    return model;
  }

  void setModel(Model& model)
  {
    this->model = &model;
  }

protected:
  ObjectPtr<Model> model;

  DECLARE_DOUBLE_LIST_ELEMENT(Actor);

  friend class Scene;

}; // Actor

typedef DoubleListImp<Actor> Actors;
typedef DoubleListIteratorImp<Actor> ActorIterator;

} // end namespace Graphics

#endif // __Actor_h
