#ifndef __GLRenderer_h
#define __GLRenderer_h

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
//  OVERVIEW: GLRenderer.h
//  ========
//  Class definition for GL renderer.

#ifndef __Renderer_h
#include "Renderer.h"
#endif
#ifndef __TriangleMeshShape_h
#include "TriangleMeshShape.h"
#endif

namespace Graphics
{ // begin namespace Graphics


//////////////////////////////////////////////////////////
//
// GLRenderer: GL renderer class
// ==========
class GLRenderer: public Renderer
{
public:
  enum RenderMode
  {
    Wireframe = 1,
    HiddenLines = 2,
    Flat = 4,
    Smooth = 0
  };

  // Flags
  enum
  {
    useLights = 1,
    drawSceneBoundingBox = 2,
    useVertexColors = 4
  };

  RenderMode renderMode;
  Flags flags;

  // Constructor
  GLRenderer(Scene&, Camera*);

  void render();

protected:
  virtual void startRender();
  virtual void endRender();
  virtual void renderWireframe();
  virtual void renderFaces();

  virtual void drawLine(const Vec3&, const Vec3&) const;
  virtual void drawAABB(const BoundingBox&) const;

  virtual Light* makeDefaultLight() const;

private:
  void setProjectionMatrix();
  void renderLights();

}; // GLRenderer

} // end namespace Graphics

#endif // __GLRenderer_h
