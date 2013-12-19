#include <stdio.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glut.h>

#ifndef __GLRenderer_h
#include "GLRenderer.h"
#endif
#ifndef __MeshSweeper_h
#include "MeshSweeper.h"
#endif
#ifndef __RayTracer_h
#include "RayTracer.h"
#endif

using namespace Graphics;

// Globals
const float CAMERA_RES = 1.0f / 50;
const float ZOOM_SCALE = 1.01f;
static Scene* scene;
static Camera* camera;
static GLRenderer* renderer;
static TriangleMeshShape* mesh;
Transf3 MTM;
static RayTracer *raytracer;
GLImage *frame;
int w, h;
bool useGL = true;
bool modifiedView;
bool pause = true;

// Mouse globals
static int mx = 0;
static int my = 0;

// Keyboard globals
#define MAX_KEYS 256
static bool keys[MAX_KEYS];

static void
printControls()
{
  printf("\n camera controls:\n"
    " ----------------\n"
    "(w) pan forward  (s) pan backward\n"
    "(q) pan up       (z) pan down\n"
    "(a) pan left     (d) pan right\n"
    "(+) zoom in      (-) zoom out\n"
    " Render mode controls:\n"
    " -----------======----\n"
    " (,) wireframe    (/) smooth\n"
    " Other controls:\n"
    " ---------------\n"
    " (p) pause rotation on/off\n");
}

inline void
printControls2()
{
  printf("\n"
    "Camera controls:\n"
    "----------------\n"
    "(w) pan forward (s) pan backward\n"
    "(q) pan up      (z) pan down\n"
    "(a) pan left    (d) pan right\n"
    "(+) zoom in     (-) zoom out\n\n"
    "Projection type:\n"
    "----------------\n"
    "(,) wireframe   (v) smooth\n"
    "(p) perspective (o) ortographic\n\n");
}

static void
processKeys()
{
  // Process keys
  for (int i = 0; i < MAX_KEYS; i++)
  {
    if (!keys[i])
      continue;

    float len = camera->getDistance() * 2 * CAMERA_RES;

    switch (i)
    {
      // Camera controls
      case 'w':
        camera->move(0, 0, -len);
        break;
      case 's':
        camera->move(0, 0, +len);
        break;
      case 'q':
        camera->move(0, +len, 0);
        break;
      case 'z':
        camera->move(0, -len, 0);
        break;
      case 'a':
        camera->move(-len, 0, 0);
        break;
      case 'd':
        camera->move(+len, 0, 0);
        break;
      case '-':
        camera->zoom(1.0f / ZOOM_SCALE);
        keys[i] = false;
        break;
      case '+':
        camera->zoom(ZOOM_SCALE);
        keys[i] = false;
        break;
      case ',':
        renderer->renderMode = GLRenderer::Wireframe;
        break;
      case '/':
        renderer->renderMode = GLRenderer::Smooth;
        break;
    }
  }
}


void
processKeys2()
{
  // Process keys
  for (int i = 0; i < MAX_KEYS; i++)
  {
    if (!keys[i])
      continue;

    float len = camera->getDistance() * CAMERA_RES;

    switch (i)
    {
      // Camera controls
    case 'w':
      camera->move(0, 0, -len);
      modifiedView = true;
      break;

    case 's':
      camera->move(0, 0, +len);
      modifiedView = true;
      break;

    case 'q':
      camera->move(0, +len, 0);
      modifiedView = true;
      break;

    case 'z':
      camera->move(0, -len, 0);
      modifiedView = true;
      break;

    case 'a':
      camera->move(-len, 0, 0);
      modifiedView = true;
      break;

    case 'd':
      camera->move(+len, 0, 0);
      modifiedView = true;
      break;

    case '-':
      camera->zoom(1.0f / ZOOM_SCALE);
      modifiedView = true;
      keys[i] = false;
      break;

    case '+':
      camera->zoom(ZOOM_SCALE);
      modifiedView = true;
      keys[i] = false;
      break;

    case 'p':
      camera->setProjectionType(Camera::Perspective);
      modifiedView = true;
      break;

    case 'o':
      camera->setProjectionType(Camera::Parallel);
      modifiedView = true;
      break;

    case 't':
      useGL ^= true;
      break;
    case ',':
      renderer->renderMode = GLRenderer::Wireframe;
      break;
	case 'v':
		renderer->renderMode = GLRenderer::Smooth;
	  break;
    }
  }
}

/**
 * Callback function responsible for rendering the scene.
 */
static void
renderCallback2()
{
  // Process user input
  processKeys2();

  
  // Render scene
  if (useGL)
    renderer->render();
  else
  {
    if (frame == 0)
      frame = new GLImage(roundupImageWidth(w), h);
    if (modifiedView)
    {
      raytracer->renderImage(*frame);
      modifiedView = false;
    }
    frame->draw();
  }
  // Swap buffers
  glutSwapBuffers();
}

/**
 * Callback function responsible for rendering the scene.
 */
static void
renderCallback()
{
  // Process user input
  processKeys();
  // Render scene
  renderer->render();
  // Swap buffers
  glutSwapBuffers();
}

/**
 * Callback function responsible for setting the aspect ratio.
 */
static void
reshapeCallback(int width, int height)
{
  renderer->setImageSize(width, height);
  camera->setAspectRatio(REAL(width) / REAL(height));
  glViewport(0, 0, width, height);
}

/*
 * Callback function responsible for setting the aspect ratio.
 */
void
reshapeCallback2(int width, int height)
{
  if (w == width && h == height)
    return;
  if (frame != 0)
  {
    delete frame;
    frame = 0;
  }
  renderer->setImageSize(w = width, h = height);
  modifiedView = true;
}

/**
 * Callback function called when application is idle.
 */
static void
idleCallback()
{
  static int count = -1;

  if (!pause && (++count % 80000) == 0)
  {
    mesh->transform(MTM);
    glutPostRedisplay();
  }
}

/**
 * Callback function responsible for handling of pressed keys.
 */
static void
keyboardCallback(unsigned char key, int /*x*/, int /*y*/)
{
  keys[key] = true;
  glutPostRedisplay();
}

/**
 * Callback function responsible for handling of released keys.
 */
static void
keyboardUpCallback(unsigned char key, int /*x*/, int /*y*/)
{
  keys[key] = false;
  switch (key)
  {
    case 'p':
      pause ^= true;
      break;
    case 27:
      exit(0);
      break;
    default:
      break;
  }
}

/**
 * Callback function responsible for handling of mouse clicks.
 */
static void
mouseCallback(int /*button*/, int state, int x, int y)
{
  mx = x;
  my = y;
}

/**
 * Callback function responsible for handling of mouse motion.
 */
static void
motionCallback(int x, int y)
{
  float da = camera->getViewAngle() * CAMERA_RES;
  float ay = (mx - x) * da;
  float ax = (my - y) * da;

  camera->rotateYX(ay, ax);
  mx = x;
  my = y;
  pause = true;
  glutPostRedisplay();

  mx = x;
  my = y;
}

/**
 * Function responsible for initializing GLUT.
 */
static void
initGlut(int argc, char** argv)
{
/**
 * All the functions in GLUT have the prefix glut, and those which perform
 * some kind of initialization have the prefix glutInit. The first thing
 * we must do is call the function glutInit().
 * --------------------------------------------------------------------------
 * void glutInit(int *argc, char **argv);
 *
 * Parameters:
 * argc: a pointer to the unmodified argc variable from the main function.
 * argv: a pointer to the unmodified argv variable from the main function.
 * --------------------------------------------------------------------------
 */
  glutInit(&argc, argv);
/**
 * After initializing GLUT itself, we're going to define our window.
 * First we establish the window's position, i.e. its top left corner.
 * In order to do this we use the function glutInitWindowPosition().
 * --------------------------------------------------------------------------
 * void glutInitWindowPosition(int x, int y);
 *
 * Parameters:
 * x: the number of pixels from the left of the screen. -1 is the default
 *    value, meaning it is up to the window manager to decide where the
 *    window will appear. If not using the default values then we should
 *    pick a positive value, preferably one that will fit in our screen.
 * y: the number of pixels from the top of the screen. The comments
 *    mentioned for the x parameter also apply in here.
 * --------------------------------------------------------------------------
 * Note that these parameters are only a suggestion to the window manager.
 * The window returned may be in a different position, although if we
 * choose them wisely we'll usually get what we want.
 */
  glutInitWindowPosition(-1, -1);
/**
 * Next we'll choose the window size. In order to do this we use the
 * function glutInitWindowSize().
 * --------------------------------------------------------------------------
 * void glutInitWindowSize(int width, int height);
 *
 * Parameters:
 * width: the width of the window.
 * height: the height of the window.
 * --------------------------------------------------------------------------
 * Again the values for width and height are only a suggestion, so avoid
 * choosing negative values.
 */
  glutInitWindowSize(512, 512);
/**
 * Then we should define the display mode using the function
 * glutInitDisplayMode().
 * --------------------------------------------------------------------------
 * void glutInitDisplayMode(unsigned int mode);
 *
 * Parameters:
 * mode: specifies the display mode.
 * --------------------------------------------------------------------------
 * The mode parameter is a Boolean combination (OR bit wise) of the possible
 * predefined values in the GLUT library. We use mode to specify the color
 * model, and the number and type of buffers. The predefined constants to
 * specify the color model are:
 * GLUT_RGBA or GLUT_RGB: selects a RGBA window (default color mode).
 * GLUT_INDEX: selects a color index mode. 
 * The display mode also allows us to select either a single or double
 * buffer window. The predefined constants for this are:
 * GLUT_SINGLE: single buffer window.
 * GLUT_DOUBLE: double buffer window. Double buffering allows for smooth
 * animation by keeping the drawing in a back buffer and swapping the back
 * with the front buffer (the visible one) when the rendering is complete.
 * Using double buffering prevents flickering.
 * There is more, we can specify if we want a window with a particular set
 * of buffers. The most common are:
 * GLUT_ACCUM: the accumulation buffer.
 * GLUT_STENCIL: the stencil buffer.
 * GLUT_DEPTH: the depth buffer.
 */
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
/**
 * After all the above steps, the window can be created with
 * glutCreateWindow().
 * --------------------------------------------------------------------------
 * int glutCreateWindow(char *title);
 *
 * Parameters:
 * title: sets the window title.
 * --------------------------------------------------------------------------
 * The return value of glutCreateWindow() is the window identifier.
 */
  glutCreateWindow("Rt");
/**
 * Whenever we want to take control of the processing of an event (key
 * pressed, window resized, etc.) we have to tell GLUT in advance which
 * function is going to perform such task. This procedure of notifying that
 * when an event occurs we want to execute a particular function is also
 * called "registering a callback function".
 *
 * Now we must tell GLUT which function should be used for the rendering.
 * GLUT will call the function we supply whenever rendering is in order.
 * Lets tell GLUT that the function RenderCallback() should be used whenever
 * the window is reported by the window system to be damaged. GLUT has a
 * function that takes as a parameter the name of the function to use when
 * redrawing is needed. Note: the function we supply as a parameter is also
 * called the first time the window is created.
 * --------------------------------------------------------------------------
 * void glutDisplayFunc(void (*func)(void)); 
 *
 * Parameters:
 * func: the name of the function to be called when the window needs to be
 *    redrawn. Note: it is illegal to pass NULL as the argument to this
 *    function.
 * --------------------------------------------------------------------------
 */
  //glutDisplayFunc(renderCallback);
  glutDisplayFunc(renderCallback2);
/**
 * GLUT provides a way to define which function should be called when the
 * window is resized, i.e. to register a callback for recomputing the
 * aspect ratio. Furthermore, this function will also be called when the
 * window is initially created so that even if we're initial window is not
 * square things will look OK. GLUT achieves this using the function
 * glutReshapeFunc().
 * --------------------------------------------------------------------------
 * void glutReshapeFunc(void (*func)(int width, int height)); 
 *
 * Parameters:
 * func: the name of the function that will be responsible for setting the
 *    correct aspect ratio when the window changes size. 
 * --------------------------------------------------------------------------
 */
  // glutReshapeFunc(reshapeCallback);
  glutReshapeFunc(reshapeCallback2);
/**
 * Now we want to tell GLUT that when the application is idle the render
 * function should be called. This causes GLUT to keep calling our rendering
 * function therefore enabling animation. GLUT provides a function,
 * glutIdleFunc(), that lets us register a callback function to be called
 * when the application is idle.
 * --------------------------------------------------------------------------
 * void glutIdleFunc(void (*func)(void));
 *
 * Parameters:
 * func: the name of the function that will be called whenever the
 *    application is idle. 
 * --------------------------------------------------------------------------
 * In our case, when the application is idle we want to call the previously
 * defined function responsible for the actual rendering. This is done by
 * registering the function IdleCallback(), which calls the GLUT function
 * glutPostRedisplay(). This one, in its turn, calls RenderCallback().
 */
  //glutIdleFunc(idleCallback);
/**
 * Up until now we used GLUT to tell the windows system which functions we
 * wanted to do the rendering when the window needed to be repainted, which
 * function to call when the system was idle, and which function to call
 * when the window was resized. Similarly we must do the same thing for
 * keyboard events. We must notify the windows system, using GLUT, which
 * function(s) will perform the required processing when a key is pressed.
 * GLUT allows us to build applications that detect keyboard input using
 * either the "normal" keys, or the special keys like F1 and Up.
 *
 * GLUT provides two functions to register callbacks for keyboard events
 * that occur when you press a key. The first one, glutKeyboardFunc(), is
 * used to tell the windows system which function we want to process the
 * "normal" key events. By "normal" keys, we mean letters, numbers, anything
 * that has an ASCII code.
 * --------------------------------------------------------------------------
 * void glutKeyboardFunc(void (*func)(unsigned char key, int x, int y));
 *
 * Parameters:
 * func: the name of the function that will process the "normal" keyboard
 *    events. Passing NULL as an argument causes GLUT to ignore "normal"
 *    keys.
 * --------------------------------------------------------------------------
 * The function used as an argument to glutKeyboardFunc() needs to have
 * three arguments. The first indicates the ASCII code of the key pressed,
 * the remaining two arguments provide the mouse position when the key is
 * pressed. The mouse position is relative to the top left corner of the
 * client area of the window.
 *
 * GLUT provides the function glutSpecialFunc() so that we can register our
 * function for special key events processing.
 * --------------------------------------------------------------------------
 * void glutSpecialFunc(void (*func)(int key, int x, int y)); 
 *
 * Parameters:
 * func: the name of the function that will process the special keyboard
 *    events. Passing NULL as an argument causes GLUT to ignore the special
 *    keys.
 * --------------------------------------------------------------------------
 * The special keys are predefined constants in glut.h The full set of
 * GLUT_KEY_* constants is:
 * GLUT_KEY_F1: F1 function key
 * GLUT_KEY_F2: F2 function key
 * GLUT_KEY_F3: F3 function key
 * GLUT_KEY_F4: F4 function key
 * GLUT_KEY_F5: F5 function key
 * GLUT_KEY_F6: F6 function key
 * GLUT_KEY_F7: F7 function key
 * GLUT_KEY_F8: F8 function key
 * GLUT_KEY_F9: F9 function key
 * GLUT_KEY_F10: F10 function key
 * GLUT_KEY_F11: F11 function key
 * GLUT_KEY_F12: F12 function key
 * GLUT_KEY_LEFT: Left function key
 * GLUT_KEY_RIGHT: Right function key
 * GLUT_KEY_UP: Up function key
 * GLUT_KEY_DOWN: Down function key
 * GLUT_KEY_PAGE_UP: Page Up function key
 * GLUT_KEY_PAGE_DOWN: Page Down function key
 * GLUT_KEY_HOME: Home function key
 * GLUT_KEY_END: End function key
 * GLUT_KEY_INSERT: Insert function key
 *
 * Sometimes we may want to know if one modifier key, i.e. CTRL, ALT or SHIFT
 * is being pressed. GLUT provides a function that detects if any modifier is
 * being pressed. This function should only be called inside the functions
 * that process keyboard or mouse input events.
 * --------------------------------------------------------------------------
 * int glutGetModifiers();
 * --------------------------------------------------------------------------
 * The return value for this function is either one of three predefined
 * constants (in glut.h), or any bitwise OR combination of them.
 * The constants are:
 *
 * GLUT_ACTIVE_SHIFT: set if either you press the SHIFT key, or Caps Lock
 * is on. Note that if they are both on then the constant is not set.
 * GLUT_ACTIVE_CTRL: set if you press the CTRL key.
 * GLUT_ACTIVE_ALT: set if you press the ALT key.
 */
  glutKeyboardFunc(keyboardCallback);
/**
 * GLUT provides two functions that register callbacks when a key is
 * released. 
 * --------------------------------------------------------------------------
 * void glutKeyboardUpFunc(void (*func)(unsigned char key,int x,int y));
 * void glutSpecialUpFunc(void (*func)(int key,int x, int y));
 *
 * Parameters:
 * func: the name of the callback function. 
 * --------------------------------------------------------------------------
 * The parameters are the same as for when the user presses a key.
 */
  glutKeyboardUpFunc(keyboardUpCallback);
/**
 * As in the keyboard version, GLUT provides a way for us to register the
 * function that will be responsable for processing events generated by mouse
 * clicks. The name of this function is glutMouseFunc(),and it is commonly
 * called in the initialization phase of the application.
 * --------------------------------------------------------------------------
 * void glutMouseFunc(void (*func)(int button, int state, int x, int y));
 *
 * Parameters:
 * func: the name of the function that will handle mouse click events.
 * --------------------------------------------------------------------------
 * The function that will handle the mouse click events must have four
 * parameters. The first relates to which button was pressed, or released.
 * This argument can have one of three values: GLUT_LEFT_BUTTON,
 * GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON. The second argument relates to the
 * state of the button when the callback was generated, i.e. pressed or
 * released. The possible values are: GLUT_DOWN, GLUT_UP. When a callback is
 * generated with the state GLUT_DOWN, the application can assume that a
 * GLUT_UP will come afterwards even if the mouse moves outside the window.
 * However if the application calls glutMouseFunc() again with NULL as
 * argument then GLUT will stop sending mouse state changes.
 *
 * The remaining two parameters provide the (x,y) coordinates of the mouse
 * relatively to the upper left corner of the client area of the window.
 *
 * GLUT provides mouse motion detection capabilities to an application.
 * There are two types of motion that GLUT handles: active and passive
 * motion. Active motion occurs when the mouse is moved and a button is
 * pressed. Passive motion is when the mouse is moving but no buttons are
 * pressed. If an application is tracking motion, an event will be generated
 * per frame during the period that the mouse is moving. As usual we must
 * register with GLUT the function that will be responsable for handling the
 * motion events. GLUT allows us to specify two different functions: one for
 * tracking passive motion, and another to track active motion. 
 * --------------------------------------------------------------------------
 * void glutMotionFunc(void (*func)(int x,int y));
 * void glutPassiveMotionFunc(void (*func)(int x, int y));
 * Parameters:
 * func: the function that will be responsible for the respective type of
 *   motion.
 * --------------------------------------------------------------------------
 * The parameters for the motion processing function are the (x,y)
 * coordinates of the mouse relatively to the upper left corner of the
 * window's client area.
 */
  glutMouseFunc(mouseCallback);
  glutMotionFunc(motionCallback);
}

static void
createRGBCube()
{
  // Create the model
  TriangleMeshShape* t = MeshSweeper::makeCube();
  Color* c = new Color[8];

  c[0] = Color::blue;
  c[1] = Color::magenta;
  c[2] = Color::white;
  c[3] = Color::cyan;
  c[4] = Color::black;
  c[5] = Color::red;
  c[6] = Color::yellow;
  c[7] = Color::green;
  t->setColors(c, 8);
  // Create an actor from model and add it into the scene
  scene->addActor(new Actor(*t));
  mesh = t;

  // Set model transformation matrix
  REAL a = Math::toRadians<REAL>(5);

  MTM.rotationY(a);

  Transf3 temp;

  temp.rotationX(a);
  MTM.compose(temp);
  temp.rotationZ(a);
  MTM.compose(temp);
}

int
main(int argc, char** argv)
{
/*
 * Print controls.
 */
  printControls2();
/**
 * Call function for initializing GLUT.
 */
  initGlut(argc, argv);
  
/**
 * Create scene, camera, and OpenGL renderer.
 */
  scene = Scene::New();
  camera = new Camera();
  camera->setPosition(Vec3(0, 0, 5));
  camera->setViewAngle(30);

  createRGBCube();

  renderer = new GLRenderer(*scene, camera);
  renderer->flags.set(GLRenderer::useVertexColors);

  raytracer = new RayTracer(*scene, camera);
  
/**
 * Create an actor and add it into the scene.
 */
/**
 * Lets tell GLUT that we're ready to get in the application event
 * processing loop. GLUT provides a function that gets the application
 * in a never ending loop, always waiting for the next event to process.
 * The GLUT function is glutMainLoop(). 
 * --------------------------------------------------------------------------
 * void glutMainLoop()
 * --------------------------------------------------------------------------
 */
  glutMainLoop();
/**
 * Destroy OpenGL renderer, camera, and scene.
 */
  return 0;
}
