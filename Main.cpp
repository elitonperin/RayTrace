#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glut.h>

#include <iostream>
#include <string>

#ifndef __GLRenderer_h
#include "GLRenderer.h"
#endif
#ifndef __MeshSweeper_h
#include "MeshSweeper.h"
#endif

#include "pugixml.hpp"

using namespace pugi;
using namespace std;
using namespace Graphics;

// Globals
const float CAMERA_RES = 1.0f / 50;
const float ZOOM_SCALE = 1.01f;
static Scene* scene;
static Camera* camera; 
static GLRenderer* renderer;
static TriangleMeshShape* mesh;
Transf3 MTM;
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
		" (,) wireframe    (.) smooth\n"
		" Other controls:\n"
		" ---------------\n"
		" (p) pause rotation on/off\n");
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

/**
Retorna Vec3 a partir de Strin do XMl
*/
static Vec3 StringToIntVec3(const char * position){
	char pos1_c = position[0];
	char pos2_c = position[2];
	char pos3_c = position[4];

	int pos1 = pos1_c + '0';
	int pos2 = pos2_c + '0';
	int pos3 = pos3_c + '0';

	return Vec3(pos1, pos2, pos3);
}

/**
Retorna Vec3 FLOAT a partir de Strin do XMl
*/
static Vec3 StringToFloatVec3(const char * position) {

	REAL pos1, pos2, pos3;

	sscanf(position, "%f %f %f", &pos1, &pos2, &pos3);

	return Vec3(pos1, pos2, pos3);
}

static int
	verifyLightFlag(const char * LightValue){
		return 1;
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
	glutCreateWindow("RGB Cube");
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
	glutDisplayFunc(renderCallback);
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
	glutReshapeFunc(reshapeCallback);
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
	glutIdleFunc(idleCallback);
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

void readXML(char * filename){

	/* De acordo com documentação pugixml*/
	xml_document document_xml;
	xml_node node, node_father, node_aux, node_in, node_more_in;
	xml_attribute att;
	
	// Size Image
	int w, h;

	// Camera
	camera = new Camera();
	Vec3 position_camera(), position_to();
	REAL camera_angle;
	REAL camera_aspect_w, camera_aspect_h;
	int camera_projection = 0;
	TriangleMeshShape *t;
	Actor *actor;
	Light *light;

	// Scene
	const char *scene_name;

	xml_parse_result parser_result = document_xml.load_file(filename);
	printf("NOME %s\n", filename);
	printf("DECRICAO %s\n", parser_result.description());
	node_father = document_xml.child("rt");

	if(node_father) {
		node = node_father.child("image");
		if(node) {
			node = node.child("width");
			w = atoi(node.value());
			node = node.child("height");
			h = atoi(node.value());
		}

		node_aux = node_father.child("camera");

		if(node_aux) {
			camera = new Camera();
			node = node_aux.child("position");
			camera->setPosition(StringToFloatVec3(node.value()));

			node = node_aux.child("to");
			//TODO
			//position_to(StringToFloatVec3(node.value()));

			node = node_aux.child("up");
			// TODO 
			//position_up(StringToFloatVec3(node.value()));

			//camera->setPosition(StringToFloatVec3(node.value()));

			node = node_aux.child("angle");
			if(node) {
				  REAL camera_angle = Math::zero<REAL>();
				  sscanf(node.value(), "%f", &camera_angle);
				  camera->setViewAngle(camera_angle);
			}

			node = node_aux.child("aspect");
			if(node){
				  sscanf(node.value(), "%f:%f", &camera_aspect_w, &camera_aspect_h);
				  camera->setAspectRatio(camera_aspect_w/camera_aspect_h);
			}

			node = node_aux.child("projection");
			if(node) {
				if(strcmp(node.value(), "Perspective")){
					camera->setProjectionType(Graphics::Camera::Perspective);
				} else {
					camera->setProjectionType(Graphics::Camera::Parallel);
				}
			}
		}

		node_aux = node_father.child("scene");
		
		att = node_aux.attribute("name");

		scene = Scene::New();

		if(att){
			scene_name = att.value();
		}

		if(node_aux) {
			node = node_aux.child("background");
			if(node) {
				REAL r_back = 0, g_back = 0, b_back= 0 ;
				sscanf(node.value(), "%f %f %f", &r_back, &g_back, &b_back);
				scene->backgroundColor.setRGB(r_back, g_back, b_back); 
			}

			node = node_aux.child("ambience");
			if(node) {
				REAL r_ambience, g_ambience, b_ambience;
				sscanf(node.value(), "%f %f %f", &r_ambience, &g_ambience, &b_ambience);
				scene->backgroundColor.setRGB(r_ambience, g_ambience, b_ambience);
			}


			// Verifica cada um dos possíveis atributos 
			for (node = node_aux.child("box"); node; node = node_aux.next_sibling("box")) {
				
				// Box variables
				Vec3 center_box(0,0,0);
				Vec3 orientation_box(0,0,0);
				REAL scale_box;
				REAL x, y, z;

				node_in = node.child("center");
				if(node_in){
					sscanf( node_in.value() , "%f %f %f" , &x, &y, &z);
					center_box.set(x, y , z);
				}

				node_in = node.child("orientation");
				if(node_in) {
					sscanf( node_in.value() , "%f %f %f" , &x, &y, &z);
					orientation_box.set(x, y, z);
				}

				node_in = node.child("scale");
				if(node_in){
					(sscanf(node_in.value(), "%f", scale_box));
				}
				
				// TODO
				//t = MeshSweeper::makeBox(Vec3(center_box), Vec3(orientation_box), Vec3(scale_box));
				scene->addActor(new Actor(*t));

				node_in = node.child("transform");
				if(node_in){
					for(node_more_in = node_in.first_child(); node_more_in; node_more_in = node_more_in.next_sibling()){
						
						if(strcmp(node_more_in.name(),"translation")==0){
							//TODO
						}
						if(strcmp(node_more_in.name(),"scale")==0){
							// TODO
						}
						if(strcmp(node_more_in.name(),"rotation")==0){
							//TODO
						}
					
					}
				}

				node_in = node.child("material");

				node_more_in = node_in.child("diffuse");

				if(node_more_in){
					// TODO 
					
				}

				node_more_in = node_in.child("spot");

				if(node_more_in){
					// TODO 
					
				}

				node_more_in = node_in.child("shine");

				if(node_more_in){
					// TODO 
				
				}

				node_more_in = node_in.child("specular");

				if(node_more_in){
					// TODO 
				
				}
			}

			// 
			for (node = node_aux.child("sphere"); node; node = node_aux.next_sibling("sphere")) {

				Vec3 center_sphere(0, 0 , 0);
				REAL x, y, z;
				REAL radius_sphere;

				node_in = node.child("center");
				if(node_in){
					sscanf(node_in.value(), "%f %f %f", &x, &y, &z);
					center_sphere.set(x, y, z); 
				}

				node_in = node.child("radius");
				if(node_in){
					radius_sphere = atof(node_in.value()); 
				}

				t = MeshSweeper::makeSphere(center_sphere, radius_sphere);

				scene->addActor(new Actor(*t));

				node_in = node.child("transform");
				if(node_in){
					for(node_more_in = node_in.first_child(); node_more_in; node_more_in = node_more_in.next_sibling()){
						if(strcmp(node_more_in.name(),"translation")==0){
							//TODO
						}
						if(strcmp(node_more_in.name(),"scale")==0){
							// TODO
						}
						if(strcmp(node_more_in.name(),"rotation")==0){
							//TODO
						}
					
					}
				}

				node_in = node.child("material");

				node_more_in = node_in.child("diffuse");

				if(node_more_in){
					// TODO 
					
				}

				node_more_in = node_in.child("spot");

				if(node_more_in){
					// TODO 
				
				}

				node_more_in = node_in.child("shine");

				if(node_more_in){
					// TODO 
				
				}

				node_more_in = node_in.child("specular");

				if(node_more_in){
					// TODO 
				
				}				
			}

			for (node = node_aux.child("cone"); node; node = node_aux.next_sibling("cone")) {

				node_in = node.child("center");
				if(node_in){
					// TODO
				}

				node_in = node.child("radius");
				if(node_in){
					// TODO
				}

				node_in = node.child("normal");
				if(node_in){
					// TODO
				}

				node_in = node.child("heigth");
				if(node_in){
					// TODO
				}

				node_in = node.child("segments");
				if(node_in){
					// TODO
				}

				node_in = node.child("transform");
				if(node_in){
					for(node_more_in = node_in.first_child(); node_more_in; node_more_in = node_more_in.next_sibling()){
						if(strcmp(node_more_in.name(),"translation")==0){
							//TODO
						}
						if(strcmp(node_more_in.name(),"scale")==0){
							// TODO
						}
						if(strcmp(node_more_in.name(),"rotation")==0){
							//TODO
						}
					
					}
				}

				node_in = node.child("material");

				node_more_in = node_in.child("diffuse");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("spot");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("shine");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("specular");

				if(node_more_in){
					// TODO 
				}

			}

			// CYLINDER
			for (node = node_aux.child("cylinder"); node; node = node_aux.next_sibling("cylinder")) {

				REAL x, y, z;
				Vec3 center_cylinder(0,0,0);
				Vec3 normal_cylinder(0,0,0);
				REAL radius_cylinder, height_cylinder;
				int segments_cylinder = 16;

				//light = new Light();

				node_in = node.child("center");
				if(node_in){
					sscanf(node_in.value(), "%f %f %f", &x, &y, &z);
					center_cylinder.set(x, y, z);
				}

				node_in = node.child("radius");
				if(node_in){
					sscanf(node_in.value(), "%f", &radius_cylinder);
				}

				node_in = node.child("normal");
				if(node_in){
					sscanf(node_in.value(), "%f %f %f", &x, &y, &z);
					normal_cylinder.set(x, y, z);
				}

				node_in = node.child("heigth");
				if(node_in){
					sscanf(node_in.value(), "%f", &height_cylinder);
				}

				node_in = node.child("segments");
				if(node_in){
					sscanf(node_in.value(), "%d", &segments_cylinder);
				}

				t =  MeshSweeper::makeCylinder(center_cylinder,radius_cylinder, normal_cylinder, height_cylinder, segments_cylinder);

				scene->addActor(new Actor(*t));

				node_in = node.child("transform");
				if(node_in){
					for(node_more_in = node_in.first_child(); node_more_in; node_more_in = node_more_in.next_sibling()){
						if(strcmp(node_more_in.name(),"translation")==0){
							//TODO
						}
						if(strcmp(node_more_in.name(),"scale")==0){
							// TODO
						}
						if(strcmp(node_more_in.name(),"rotation")==0){
							//TODO
						}
					
					}
				}

				node_in = node.child("material");

				node_more_in = node_in.child("diffuse");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("spot");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("shine");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("specular");

				if(node_more_in){
					// TODO 
				}

			}

			for (node = node_aux.child("light"); node; node = node_aux.next_sibling("light")) {
				node_more_in = node_in.child("position");
				REAL x, y , z, color_aux;
				Vec3 color_light(0,0,0);

				light = new Light(Vec3(1,1,1), Color::white);

				// TODO POSITION
				
				node_more_in = node_in.child("color");

				if(node_more_in) {
					sscanf(node_more_in.value(), "%f %f %f %f" , &x, &y, &z, &color_aux);
					if(Math::isZero(color_aux)){
						light->setDirectional(true);
					} else {
						light->setDirectional(false);
					}

					light->color = Vec3(x, y, z);
				}

				node_more_in = node_in.child("fallof");

				if(scene->getNumberOfLights() == 0){
					scene->addLight(light);
				}
				// Add light
				scene->addLight(light);
							
			}

			for (node = node_aux.child("mesh"); node; node = node_aux.next_sibling("mesh")) {
				
				att = node.attribute("name");

				if(att) {

					// TODO READOBJ
					//readOBJ(att);
				}

				node_in = node.child("transform");
				if(node_in){
					for(node_more_in = node_in.first_child(); node_more_in; node_more_in = node_more_in.next_sibling()){
						if(strcmp(node_more_in.name(),"translation")==0){
							//TODO
						}
						if(strcmp(node_more_in.name(),"scale")==0){
							// TODO
						}
						if(strcmp(node_more_in.name(),"rotation")==0){
							//TODO
						}
					
					}
				}

				node_in = node.child("material");

				node_more_in = node_in.child("diffuse");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("spot");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("shine");

				if(node_more_in){
					// TODO 
				}

				node_more_in = node_in.child("specular");

				if(node_more_in){
					// TODO 
				}
				
			}

		} else {
			printf("Error - TAG <scene> not found\n");
			exit(0);
		}
		} else {
			printf("Error - TAG <rt> not found\n");
			exit(0);
		}
	

}

int
	main(int argc, char** argv)
{
	/**
	* Create scene, camera, and OpenGL renderer.
	*/
	
	scene = Scene::New();
	camera = new Camera();

	camera->setPosition(Vec3(0, 0, 10));
	camera->setViewAngle(30);
	//camera->rotateYX(0, -0,4);
	
	/**
	* Create an actor and add it into the scene.
	*/
	//	createRGBCube();
	/*
	Graphics::MeshSweeper::Polyline circlet = MeshSweeper::makeCircle(Vec3(0,0,0), 1 ,Vec3(0,-1,0), 16);
	TriangleMeshShape* t = MeshSweeper::makeCone(Vec3(0,0,0),1,Vec3(0,-1,0), 5, 32);
	
	Color* c = new Color[4];
	
	c[0] = Color::blue;
	c[1] = Color::magenta;
	c[2] = Color::white;
	c[3] = Color::cyan;
	t->setColors(c, 4);

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
	*/

	readXML(argv[1]);
	renderer = new GLRenderer(*scene, camera);
	renderer->flags.set(GLRenderer::useVertexColors);

	// Create the model
	/*
	TiXmlDocument *doc = new TiXmlDocument("simple-scene.xml");
	doc->LoadFile();

	// <rt>
	TiXmlElement* root = doc->FirstChildElement( "rt" );
	if ( root ) // verifica se o valor nao eh nulo
	{
		// <image>
		TiXmlElement* imageXML = root->FirstChildElement( "image" );
		if ( imageXML ) // verifica se o valor nao eh nulo
		{
			// <width>
			TiXmlElement* widthXML = imageXML->FirstChildElement( "width" );
			if ( widthXML ) // verifica se o valor nao eh nulo
			{
				int width = atoi(widthXML->GetText());
			}

			TiXmlElement* heightXML = imageXML->FirstChildElement( "height" );

			if ( heightXML ) // verifica se o valor nao eh nulo
			{
				int height = atoi(heightXML->GetText());
			}
		}

		TiXmlElement* cameraXML = root->FirstChildElement( "camera" );
		if( cameraXML )
		{
			// <position>
			TiXmlElement* positionXML = cameraXML->FirstChildElement( "position" );
			const char * position;
			if( positionXML ) {
				position = positionXML->GetText();
				camera->setPosition(StringToIntVec3(position));				 
			}

			TiXmlElement* toXML = cameraXML->FirstChildElement( "to" );
			if( toXML ) {
				position = toXML->GetText();
			}

			TiXmlElement* upXML = cameraXML->FirstChildElement( "up" );
			if( upXML ){
				position = upXML->GetText();
			}

			TiXmlElement* angleXML = cameraXML->FirstChildElement( "angle" );

			if( angleXML ) {
				int angle_camera = atoi(angleXML->GetText());
				camera->setViewAngle(angle_camera);	
			}

			TiXmlElement* aspectXML = cameraXML->FirstChildElement( "aspect" );

			if( aspectXML ) { // TODO
				float aspect_ratio = 16/9; 
				camera->setAspectRatio(aspect_ratio);
			}

			TiXmlElement* projectionXML = cameraXML->FirstChildElement( "projection" );
			if( projectionXML ) { // TODO
				if(projectionXML->GetText() == "parallel") {
					camera->setProjectionType(Graphics::Camera::Parallel);
				} else {
					camera->setProjectionType(Graphics::Camera::Perspective);
				}
			}
		}

		TiXmlElement* sceneXML = root->FirstChildElement( "scene" );
		if( sceneXML )
		{
			if(sceneXML->Attribute("name")){
				string scene_name = sceneXML->Attribute("name");
			}

			TiXmlElement* elem = sceneXML->FirstChildElement();
			static TriangleMeshShape* t;
			static Light* l;

			while(elem)
			{
				if(elem->Value() == "background") {
					scene->backgroundColor = (StringToFloatVec3(elem->GetText()));
				}
				else if (elem->Value() == "ambient" ) {
					scene->ambientLight = (StringToFloatVec3(elem->GetText()));
				}
				else if (elem->Value() == "mesh" ) {
					if(elem->Attribute("file")){

						const char * obj_filename = sceneXML->Attribute("file");

						TiXmlElement* mesh_tranform = sceneXML->FirstChildElement("transform");

						if(mesh_tranform){

							TiXmlElement* mesh_elem = sceneXML->FirstChildElement();

							while(mesh_elem) {
								if(mesh_elem->Value() == "translation"){
									char const * mesh_translation = mesh_elem->GetText() ;

									// TODO tranformation

								} else if (mesh_elem->Value() == "scale"){
									char const * mesh_scale = mesh_elem->GetText();

									// TODO scale

								} else if (mesh_elem->Value() == "rotation"){
									TiXmlElement* mesh_rotation_axis = mesh_elem->FirstChildElement("axis");
									TiXmlElement* mesh_rotation_angle = mesh_elem->FirstChildElement("angle");

									// TODO rotation
								}

								mesh_elem = mesh_elem->FirstChildElement();
							}
						}

						TiXmlElement* mesh_material = elem->FirstChildElement("material");

						TiXmlElement* mesh_material_ambient = mesh_material->FirstChildElement("ambient");

						if(mesh_material_ambient){
							const char* mesh_material_ambient_value = mesh_material_ambient->GetText();
							// TODO
						}

						TiXmlElement* mesh_material_diffuse = mesh_material->FirstChildElement("diffuse");

						if(mesh_material_diffuse) {
							const char* mesh_material_diffuse_value = mesh_material_diffuse->GetText();
							// TODO
						}

						TiXmlElement* mesh_material_spot = mesh_material->FirstChildElement("spot");

						if(mesh_material_spot) {
							const char* mesh_material_spot_value = mesh_material_spot->GetText();
							// TODO
						}				

						TiXmlElement* mesh_material_specular = mesh_material->FirstChildElement("shine");

						if(mesh_material_specular) {
							const char* mesh_material_specular_value = mesh_material_specular->GetText();
							// TODO
						}

					} else {
						printf("ERRRRRRRROOO - ARQUVIO OBJ NAO INFORMADO");
					}

				}

				//TiXmlElement* translation = transform->FirstChildElement(  ); 
				else if (elem->Value() == "sphere" )
				{
					const char * sphere_center_value = "0 0 0";
					const char * sphere_radius_value = "1";
					float sphere_radius_float_value; 

					TiXmlElement* sphere_center = elem->FirstChildElement( "center" );
					TiXmlElement* sphere_radius = elem->FirstChildElement( "radius" );


					if( sphere_center ){
						sphere_center_value = sphere_center->GetText();
						if( sphere_radius ){
							sphere_radius_value = sphere_radius->GetText();
						}
					}

					sphere_radius_float_value = atof(sphere_radius_value);

					// TODO IDE FILHO DA PUTA
					t = MeshSweeper::makeSphere(StringToIntVec3(sphere_center_value), sphere_radius_float_value);

					scene->addActor(new Actor(*t));

				}
				else if (elem->Value() == "box" )
				{

					const char * box_center_value = "0 0 0";
					const char * box_scale_value = "0, 0, 0";
					const char * box_orientation_value = "0, 0, 0";

					TiXmlElement* box_center = elem->FirstChildElement( "center" );
					TiXmlElement* box_orientation = elem->FirstChildElement( "orientation" );
					TiXmlElement* box_scale = elem->FirstChildElement( "scale" );

					if( box_center ) {
						box_center_value = box_center->GetText();		
					}
					if( box_orientation ){
						box_orientation_value = box_orientation->GetText();
					}
					if( box_scale ){
						box_scale_value = box_scale->GetText();
					}

					t = MeshSweeper::makeBox(StringToIntVec3(box_center_value), StringToIntVec3(box_orientation_value), StringToIntVec3(box_scale_value));							  
					scene->addActor(new Actor(*t));
				}
				else if (elem->Value() == "cone" )
				{

					TiXmlElement* cone_center = elem->FirstChildElement( "center" );
					TiXmlElement* cone_radius = elem->FirstChildElement( "radius" );
					TiXmlElement* cone_normal = elem->FirstChildElement( "normal" );
					TiXmlElement* cone_height = elem->FirstChildElement( "height" );
					TiXmlElement* cone_segments = elem->FirstChildElement( "segments" );


					// TODO

					const char * cone_center_value = "0 0 0";
					const char * cone_radius_value = "0.0";
					const char * cone_normal_value = "0, 0, 0";
					const char * cone_height_value = "0.0";
					const char * cone_segments_value = "0";

					if(cone_center) {
						cone_center_value = cone_center->GetText();
					}
					if(cone_radius) {
						cone_radius_value = cone_radius->GetText();
					}
					if(cone_normal) {
						cone_normal_value = cone_normal->GetText();
					}
					if(cone_height) {
						cone_height_value = cone_height->GetText();
					}
					if(cone_segments) {
						cone_segments_value = cone_segments->GetText();
					}


					t = MeshSweeper::makeCone(StringToIntVec3(cone_center_value), atof(cone_radius_value), StringToIntVec3(cone_normal_value), atof(cone_height_value), atoi(cone_segments_value));							  
					scene->addActor(new Actor(*t));
				}
				else if ("cylinder"){
					TiXmlElement* cylinder_center = elem->FirstChildElement( "center" );
					TiXmlElement* cylinder_radius = elem->FirstChildElement( "radius" );
					TiXmlElement* cylinder_normal = elem->FirstChildElement( "normal" );
					TiXmlElement* cylinder_height = elem->FirstChildElement( "height" );
					TiXmlElement* cylinder_segments = elem->FirstChildElement( "segments" );


					// TODO

					const char * cylinder_center_value = "0 0 0";
					const char * cylinder_radius_value = "0.0";
					const char * cylinder_normal_value = "0, 0, 0";
					const char * cylinder_height_value = "0.0";
					const char * cylinder_segments_value = "0";

					if(cylinder_center) {
						cylinder_center_value = cylinder_center->GetText();
					}
					if(cylinder_radius) {
						cylinder_radius_value = cylinder_radius->GetText();
					}
					if(cylinder_normal) {
						cylinder_normal_value = cylinder_normal->GetText();
					}
					if(cylinder_height) {
						cylinder_height_value = cylinder_height->GetText();
					}
					if(cylinder_segments) {
						cylinder_segments_value = cylinder_segments->GetText();
					}

					t = MeshSweeper::makeCone(StringToIntVec3(cylinder_center_value), atof(cylinder_radius_value), StringToIntVec3(cylinder_normal_value), atof(cylinder_height_value), atoi(cylinder_segments_value));							  
					scene->addActor(new Actor(*t));

				} else if ("screw") {
					TiXmlElement* screw_center = elem->FirstChildElement( "center" );
					TiXmlElement* screw_first = elem->FirstChildElement( "first" );
					TiXmlElement* screw_normal = elem->FirstChildElement( "normal" );
					TiXmlElement* screw_distance = elem->FirstChildElement( "distance" );
					TiXmlElement* screw_angle = elem->FirstChildElement( "angle" );
					TiXmlElement* screw_height = elem->FirstChildElement( "height" );
					TiXmlElement* screw_delta = elem->FirstChildElement( "delta" );
					TiXmlElement* screw_steps = elem->FirstChildElement( "steps" );

					// TODO

					const char * screw_center_value = "0 0 0";
					const char * screw_first_value = "0 0 0";
					const char * screw_normal_value = "0, 0, 0";
					const char * screw_distance_value = "0.0";
					const char * screw_angle_value = "0.0";
					const char * screw_height_value = "0.0";
					const char * screw_delta_value = "0.0";
					const char * screw_steps_value = "0.0";

					const char * cylinder_segments_value = "0";

					if(screw_center) {
						screw_center_value = screw_center->GetText();
					}
					if(screw_first) {
						screw_first_value = screw_first->GetText();
					}
					if(screw_normal) {
						screw_normal_value = screw_normal->GetText();
					}
					if(screw_distance) {
						screw_distance_value = screw_distance->GetText();
					}
					if(screw_angle) {
						screw_angle_value = screw_angle->GetText();
					}
					if(screw_height) {
						screw_height_value = screw_height->GetText();
					}
					if(screw_delta) {
						screw_delta_value = screw_delta->GetText();
					}
					if(screw_steps) {
						screw_steps_value = screw_steps->GetText();
					}
					// TODO
					printf("Parafuso não implementado por enquanto\n");
					scene->addActor(new Actor(*t));
				}
				else if (elem->Value() == "light" )
				{
					TiXmlElement* position_light = elem->FirstChildElement( "position" );
					if( position_light ){

						const char * position_light_value = position_light->GetText();

						// Verificar se o ultimo valor é zero ou nao
						// Se nao for 0, luz é pontual
						// Se for 0 é direcional

						//Retorna 1 se for pontual, retorna 4 se for direcional
						verifyLightFlag(position_light_value);

						l->position = StringToFloatVec3(position_light_value);

						TiXmlElement* color_light = elem->FirstChildElement( "color" );
						TiXmlElement* falloff_light = elem->FirstChildElement( "falloff" );

						l->color = StringToFloatVec3(color_light->GetText());


					} else {			
						printf("ERRO - LIGHT DEVE POSSUI POSITION\n");
					}
				}
				elem = elem->NextSiblingElement();
			}
		} else {
			printf("Error: <scene> not found\n");
		}
	} else {
		printf("Error: <rt> element not found.\n");
		return 0;
	}
	*/
	
	/**
	* Print controls.
	*/
	printControls();
	/**
	* Call function for initializing GLUT.
	*/
	initGlut(argc, argv);
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
