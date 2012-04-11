#include "common.h"
#include "constants.h"
#include "satelliterecord.h"

#ifndef SGP4_CUDA_H
#define SGP4_CUDA_H

void ComputeSGP4CUDA(	gravconsttype whichconst,	
						std::vector<satelliterecord_aos_t> &SatRec,
						double starttime,
						double endtime,
						double deltatime	);


//void ComputeFlowCUDA(const float *I0,  // source frame
//                     const float *I1,  // tracked frame
//                     int width,        // frame width
//					 int height,       // frame height
//					 int stride,       // row access stride
//                     float alpha,      // smoothness coefficient
//					 int nLevels,      // number of levels in pyramid
//                     int nWarpIters,   // number of warping iterations per pyramid level
//					 int nSolverIters, // number of solver iterations (for linear system)
//                     float *u,         // output horizontal flow
//					 float *v);        // output vertical flow

#endif