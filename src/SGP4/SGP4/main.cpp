#include "common.h"

const std::string tle = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753\n\
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667     0.00      4320.0        360.00\n\
#                       ## fig show lyddane fix error with gsfc ver \n\
1 04632U 70093B   04031.91070959 -.00000084  00000-0  10000-3 0  9955 \n\
2 04632  11.4628 273.1101 1450506 207.6000 143.9350  1.20231981 44145  -5184.0     -4896.0        120.00\n\
#                       # simplified drag eq\n\
1 29238U 06022G   06177.28732010  .00766286  10823-4  13334-2 0   101\n\
2 29238  51.5595 213.7903 0202579  95.2503 267.9010 15.73823839  1061      0.0      1440.0        120.00\n";

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/version.h>

#include "tle.h"
#include "sgp4CUDA.cuh"


static void handle_error(cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
    	std::cout <<  file << ", line " << line << ": " << cudaGetErrorString(error) << "\n";
        exit(EXIT_FAILURE);
    }
}

// window
int window_width = 512;
int window_height = 512;

GLuint vbo_pos[1];
// Device buffer variables
float4* d_pos;
float4* d_vel;

//delta time
float deltatime = 0.01f;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

// GL functionality
bool initGL();
void createVBOs( GLuint* vbo);
void deleteVBOs( GLuint* vbo);


// rendering callbacks
void display(void);
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void idle(void);

//  printf prints to file. printw prints to window
void printw (float x, float y, float z, char* format, ...);
GLvoid *font_style = GLUT_BITMAP_TIMES_ROMAN_24;

// Cuda functionality
void runCuda( GLuint* vbo, float dt);


std::vector<satelliterecord_aos_t> SatRec;

size_t numberSatellites = 0;

void calculateFPS();
void drawFPS();
//Number of frames
unsigned int frame = 0;
//Frames per second
float fps = 0.0;
//Time
int currentTime = 0, previousTime = 0;


void main(int argc, char **argv){
	shrQAStart(argc, argv);

	cutilChooseCudaDevice(argc, argv);
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0) );

	//int thrust_major_version = THRUST_MAJOR_VERSION;
	//int thrust_minor_version = THRUST_MINOR_VERSION;

	//Read in TLEs and Initialize SGP4
	std::ifstream tle_file("D:\\School\\cis565\\cis565-project\\src\\SGP4\\SGP4\\catalog_3l_2012_03_26_am.txt");
	twolineelement2rv(tle_file, SatRec);
	numberSatellites = SatRec.size();
	initSGP4CUDA(wgs84, SatRec, numberSatellites);

	SatRec.erase(SatRec.begin(), SatRec.end());

	glutInit(&argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( " SGP4 CUDA Enabled ");

	    // initialize GL
    if( false == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);
	glutIdleFunc( idle);
	
    // create VBO
    createVBOs( vbo_pos );

    glutMainLoop();
	return;
}


////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return false;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    
    // TODO (maybe) :: depending on your parameters, you may need to change
    // near and far view distances (1, 500), to better see the simulation.
    // If you do this, probably also change translate_z initial value at top.
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1, 500.0);

    //CUT_CHECK_ERROR_GL();

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda( GLuint * vbo, float dt)
{
	// map OpenGL buffer object for writing from CUDA
    float4* positions;

    // Map opengl buffers to CUDA.
	cutilSafeCall(cudaGLMapBufferObject( (void**)&positions, vbo[0]));

	ComputeSGP4CUDA(positions, dt, numberSatellites);

    // unmap buffer objects from cuda.
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo[0]));
}

int count = 0;
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display(void)
{
    runCuda(vbo_pos, deltatime);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo with newPos
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 1.0, 0.0);
    glDrawArrays(GL_POINTS, 0, numberSatellites);
    glDisableClientState(GL_VERTEX_ARRAY);
	
    drawFPS();
    glutSwapBuffers();
    //glutPostRedisplay();
	//count++;
	//if(count == 60 * 10){
 //       deleteVBOs( vbo_pos );
	//	FreeVariables();
	//	exit(0);
	//}
}

////////////////////////////////////////////////////////////////////////////////
//! Draw FPS
////////////////////////////////////////////////////////////////////////////////
void drawFPS()
{
    //  Load the identity matrix so that FPS string being drawn
    //  won't get animates
	glLoadIdentity ();

	//  Print the FPS to the window
	printw (-17, 16, -30, "FPS: %4.2f", fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Draw string to screen
////////////////////////////////////////////////////////////////////////////////
void printw (float x, float y, float z, char* format, ...)
{
	va_list args;	//  Variable argument list
	int len;		//	String length
	int i;			//  Iterator
	char * text;	//	Text

	//  Initialize a variable argument list
	va_start(args, format);

	//  Return the number of characters in the string referenced the list of arguments.
	//  _vscprintf doesn't count terminating '\0' (that's why +1)
	len = _vscprintf(format, args) + 1; 

	//  Allocate memory for a string of the specified size
	text = (char *)malloc(len * sizeof(char));

	//  Write formatted output using a pointer to the list of arguments
	vsprintf_s(text, len, format, args);

	//  End using variable argument list 
	va_end(args);

	//  Specify the raster position for pixel operations.
	glRasterPos3f (x, y, z);

	//  Draw the characters one by one
    for (i = 0; text[i] != '\0'; i++)
        glutBitmapCharacter(font_style, text[i]);

	//  Free the allocated memory for the string
	free(text);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOs(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo[0]);

    // initialize buffer object; this will be used as 'oldPos' initially
    unsigned int size = numberSatellites * 4 * sizeof( float);

    // TODO :: Modify initial positions!
    float4* temppos = (float4*)malloc( numberSatellites * 4 * sizeof(float));
    for(int i = 0; i < numberSatellites; ++i)
    {
			temppos[i].x = temppos[i].y = temppos[i].z = 0.0f;
			temppos[i].w = 1.;
    }

    // Notice only vbo[0] has initial data!
    glBufferData( GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    free(temppos);

    //// Create initial 'newPos' buffer
    //glBindBuffer( GL_ARRAY_BUFFER, vbo[1]);
    //glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);


    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer objects with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo[0]));

    //CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOs( GLuint* vbo)
{
    glBindBuffer( 1, vbo[0]);
    glDeleteBuffers( 1, &vbo[0]);
    /*glBindBuffer( 1, vbo[1]);
    glDeleteBuffers( 1, &vbo[1]);*/

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo[0]));
    //CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo[1]));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        deleteVBOs( vbo_pos );
        //deleteDeviceData();
        exit( 0);
	case( 43):
		deltatime += 0.05f;
		break;
	case( 45):
		deltatime -= 0.05f;
		break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Idle
////////////////////////////////////////////////////////////////////////////////
void idle(void){
	// Calculate FPS
	calculateFPS();

	//
	glutPostRedisplay ();
}

//-------------------------------------------------------------------------
// Calculates the frames per second
//-------------------------------------------------------------------------
void calculateFPS()
{
    //  Increase frame count
    frame++;
	
    //  Get the number of milliseconds since glutInit called 
    //  (or first call to glutGet(GLUT ELAPSED TIME)).
    currentTime = glutGet(GLUT_ELAPSED_TIME);

    //  Calculate time passed
    int timeInterval = currentTime - previousTime;

    if(timeInterval > 1000)
    {
        //  calculate the number of frames per second
        fps = frame / (timeInterval / 1000.0f);

        //  Set time
        previousTime = currentTime;

        //  Reset frame count
        frame = 0;
    }
}