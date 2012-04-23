#ifndef SGP4_CUDA_H
#define SGP4_CUDA_H

#include "commonCUDA.cuh"
#include "tle.h"



void initSGP4CUDA(	gravconsttype whichconst,
					std::vector<satelliterecord_aos_t> &SatRecAoS,
					int numberSatellites	);

void ComputeSGP4CUDA(	float4 *positions,
						double deltatime,
						int numberSatellites	);
void FreeVariables();
#endif