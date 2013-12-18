#ifndef __Sweeper_h
#define __Sweeper_h

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
//  OVERVIEW: Sweeper.h
//  ========
//  Class definition for generic sweeper.

#ifndef __Array_h
#include "Array.h"
#endif
#ifndef __List_h
#include "List.h"
#endif
#ifndef __Object_h
#include "Object.h"
#endif
#ifndef __Transform3_h
#include "Transform3.h"
#endif
#ifndef __Utils_h
#include "Utils.h"
#endif

using namespace System;
using namespace System::Collections;

namespace Graphics
{ // begin namespace Graphics

using namespace Utils;


//////////////////////////////////////////////////////////
//
// Sweeper: generic sweeper class
// =======
class Sweeper
{
public:
  class Polyline
  {
  public:
    enum
    {
      closed = 1
    };

    class Vertex
    {
    public:
      Vec3 position;

      // Constructors
      Vertex()
      {
        // do nothing
      }

      Vertex(const Vec3& p):
        position(p)
      {
        // do nothing
      }

      void transform(const Transf3& m)
      {
        m.transformRef(position);
      }

      bool operator ==(const Vertex& vertex) const
      {
        return position == vertex.position;
      }

    }; // Sweeper::Polyline::Vertex

    typedef List<Vertex> Vertices;
    typedef ListIterator<Vertex> VertexIterator;

    // Constructors
    Polyline():
      body(new Body())
    {
      makeUse(body);
    }

    Polyline(const Polyline& polyline):
      body(makeUse(polyline.body))
    {
      // do nothing
    }

    // Destructor
    ~Polyline()
    {
      body->release();
    }

    void mv(const Vec3& position)
    {
      body->vertices.add(Vertex(position));
    }

    void transform(const Transf3&);

    void invert()
    {
      body->vertices.invert();
    }

    void open()
    {
      body->flags.reset(closed);
    }

    void close()
    {
      body->flags.set(closed);
    }

    int getNumberOfVertices() const
    {
      return body->vertices.size();
    }

    VertexIterator getVertexIterator() const
    {
      return VertexIterator(body->vertices);
    }

    Flags getFlags() const
    {
      return body->flags;
    }

    bool isClosed() const
    {
      return body->flags.isSet(closed);
    }

    Vec3 normal() const;

    Polyline& operator =(const Polyline& polyline)
    {
      body->release();
      body = makeUse(polyline.body);
    }

    bool operator ==(const Polyline& polyline) const
    {
      return body == polyline.body;
    }

  private:
    class Body: public Object
    {
    private:
      Polyline::Vertices vertices;
      Flags flags;

      // Constructor
      Body()
      {
        // do nothing
      }

      friend class Polyline;

    }; // Sweeper::Polyline::tBody

    Body* body;

  }; // Sweeper::Polyline

  typedef Array<Polyline> PolylineArray;

  static Polyline makeArc(const Vec3&, REAL, const Vec3&, REAL, int = 16);
  static Polyline makeCircle(const Vec3&, REAL, const Vec3&, int = 16);

}; // Sweeper

} // end namespace Graphics

#endif // __Sweeper_h
