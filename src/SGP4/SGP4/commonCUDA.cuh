#ifndef __COMMON_CUDA_CUH
#define __COMMON_CUDA_CUH

#if _SINGLE_PRECISION
#define t_var float
#define t_var3 float3
#define t_var4 float4
#define GLUT_DEFINE_MODE GLUT_SINGLE
#elif _DOUBLE_PRECISION
#define t_var double
#define t_var3 double3
#define t_var4 double4
#define GLUT_DEFINE_MODE GLUT_DOUBLE
#endif

#include "common.h"

__device__ __constant__ gravconstant_t gravity_constants;


#endif