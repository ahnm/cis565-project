//#include <cutil_inline.h>
//#include "tle.h"
//#include "functions.h"
//
//
//struct SRCmpOrbitalPeriod {
//  __host__ __device__
//  bool operator()(const satellite_record& sr1, const satellite_record& sr2) {
//	  return (2 * 3.14 / sr1.n) < (2 * 3.14 / sr2.n);
//  }
//};
//
//
//
//

///////////////////////////////////////////////////////////////////////////////
// Header for common includes and utility functions
///////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H

///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

//#include <stdlib.h>
//#include <stdio.h>
//#include <time.h>
//#include <memory.h>

#define HANDLE_ERROR(error) (handle_error(error, __FILE__, __LINE__ ))
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include <cutil_inline.h>
#include <cutil_math.h>
#include <cuda_gl_interop.h>
#include <shrQATest.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "constants.h"
#include "satelliterecord.h"

///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
static const int StrideAlignment = 32;

#define NUM_THREADS 512
//int *numberSatellites;


//float4 *d_position, *d_velocity;

///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////


#endif