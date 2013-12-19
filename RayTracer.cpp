//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                        GVSG Foundation Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007-2009, Paulo Aristarco Pagliosa            |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: RayTracer.cpp
//  ========
//  Source file for simple ray tracer.

#include <stdlib.h>

#ifndef __RayTracer_h
#include "RayTracer.h"
#endif


//////////////////////////////////////////////////////////
//
// RayTracer implementation
// =========
RayTracer::RayTracer(Scene& scene, Camera* camera, Intersector* i):
  Renderer(scene, camera),
  maxRecursionLevel(6),
  minWeight(0.001f),
	intersector(i)
//[]---------------------------------------------------[]
//|  Constructor                                        |
//[]---------------------------------------------------[]
{
  if (i != 0)
    i->init(scene);
}

//
// Auxiliary VRC
//
static Vec3 VRC_u;
static Vec3 VRC_v;
static Vec3 VRC_n;

//
// Auxiliary mapping variables
//
static REAL VW_h;
static REAL VW_w;
static REAL II_h;
static REAL II_w;

void
RayTracer::render()
//[]---------------------------------------------------[]
//|  Render                                             |
//[]---------------------------------------------------[]
{
  printf("Invoke renderImage(image) to run the ray tracer\n");
}

static long numberOfRays;
static long numberOfHits;

void
RayTracer::renderImage(Image& image)
//[]---------------------------------------------------[]
//|  Run the ray tracer                                 |
//[]---------------------------------------------------[]
{
  image.getSize(W, H);
  // init auxiliary VRC
  VRC_n = camera->getViewPlaneNormal();
  VRC_v = camera->getViewUp();
  VRC_u = VRC_v.cross(VRC_n);
  // init auxiliary mapping variables
  II_w = Math::inverse<REAL>(REAL(W));
  II_h = Math::inverse<REAL>(REAL(H));

  REAL height = camera->windowHeight();

  if (W >= H)
    VW_w = (VW_h = height) * W * II_h;
  else
    VW_h = (VW_w = height) * H * II_w;
  // init pixel ray
  pixelRay.origin = camera->getPosition();
  pixelRay.direction = -VRC_n;
  scan(image);
  printf("\nNumber of rays: %lu", numberOfRays);
  printf("\nNumber of hits: %lu", numberOfHits);
}

void
RayTracer::scan(Image& image)
//[]---------------------------------------------------[]
//|  Basic scan with optional jitter                    |
//[]---------------------------------------------------[]
{
  Pixel* pixels = new Pixel[W];

  numberOfRays = numberOfHits = 0;
  for (int j = 0; j < H; j++)
  {
    REAL y = j + 0.5;

    printf("Scanning line %d of %d\r", j + 1, H);
    for (int i = 0; i < W; i++)
      pixels[i] = shoot(i + 0.5, y);
    image.write(j, pixels);
  }
  delete []pixels;
}

Color
RayTracer::shoot(REAL x, REAL y)
//[]---------------------------------------------------[]
//|  Shoot a pixel ray                                  |
//|  @param x coordinate of the pixel                   |
//|  @param y cordinates of the pixel                   |
//|  @return RGB color of the pixel                     |
//[]---------------------------------------------------[]
{
  // set pixel ray
  setPixelRay(x, y);

  // trace pixel ray
  Color color = trace(pixelRay, Math::infinity<REAL>(), 0, 1.0f);

  // adjust RGB color
  if (color.r > 1.0f)
    color.r = 1.0f;
  if (color.g > 1.0f)
    color.g = 1.0f;
  if (color.b > 1.0f)
    color.b = 1.0f;
  // return pixel color
  return color;
}

void
RayTracer::setPixelRay(REAL x, REAL y)
//[]---------------------------------------------------[]
//|  Set pixel ray                                      |
//|  @param x coordinate of the pixel                   |
//|  @param y cordinates of the pixel                   |
//[]---------------------------------------------------[]
{
  Vec3 p;

  p = VW_w * (x * II_w - 0.5) * VRC_u + VW_h * (y * II_h - 0.5) * VRC_v;
  switch (camera->getProjectionType())
  {
    case Camera::Perspective:
      pixelRay.direction = (p - camera->getDistance() * VRC_n).versor();
      break;

    case Camera::Parallel:
      pixelRay.origin = camera->getPosition() + p;
      break;
  }
}

bool
RayTracer::intersect(const Ray& ray, IntersectInfo& info, REAL maxDist)
//[]---------------------------------------------------[]
//|  Ray/object intersection                            |
//|  @param the ray (input)                             |
//|  @param information on intersection (output)        |
//|  @param background distance                         |
//|  @return true if the ray intersects an object       |
//[]---------------------------------------------------[]
{
  if (intersector != 0)
    if (!intersector->intersect(ray, info, maxDist)){
      return false;
	}
    else
    {
      numberOfHits++;
      return true;
    }
  info.distance = maxDist;
  info.object = 0;

  IntersectInfo hit;

  for (ActorIterator ait(scene->getActorIterator()); ait; ++ait)
    if (ait.current()->isVisible())
    {
      Model* object = ait.current()->getModel();

      if (object->intersect(ray, hit))
        if (hit.distance < info.distance)
          info = hit;
    }
    return info.object != 0;
}

Color
RayTracer::trace(const Ray& ray, REAL maxDist, int level, REAL weight)
//[]---------------------------------------------------[]
//|  Trace a ray                                        |
//|  @param the ray                                     |
//|  @param max distance
//|  @param recursion level                             |
//|  @param ray weight                                  |
//|  @return color of the ray                           |
//[]---------------------------------------------------[]
{
  /*
   * TODO: insert your code here
   */

	REAL diference_weight = weight - getMinWeight();

	if(level >= maxRecursionLevel){
		return Color::black;
	}
	IntersectInfo info;
	if(intersect(ray, info, maxDist)){
		return Color::blue;
	}
	
	return Color::black;
}
