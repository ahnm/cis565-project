//#include "sgp4CUDA.cuh"

#include "commonCUDA.cuh"
#include "tle.h"
#include "sgp4initKernel.cu"
#include "sgp4Kernel.cu"


double currenttime = 0.0;

struct satelliterecord_soa_t *d_satrec, *h_satrec;

void initSGP4CUDA( 	gravconsttype whichconst, std::vector<satelliterecord_aos_t> &SatRecAoS, int numberSatellites ){
	//Set Gravitational Constants
	gravconstant_t gravconstant;
	setGravConstant(whichconst, gravconstant);
	cudaMemcpyToSymbol("gravity_constants", &gravconstant, sizeof(gravconstant_t), 0, cudaMemcpyHostToDevice);

	satelliterecord_soa_t *SatRecSoA = (satelliterecord_soa_t*) malloc(sizeof(satelliterecord_soa_t) * numberSatellites);

	satelliteRecordConvert(SatRecAoS, SatRecSoA);

	cutilSafeCall(cudaMalloc((void **) &d_satrec, sizeof(satelliterecord_soa_t) * numberSatellites));
	cutilSafeCall(cudaMemcpy(d_satrec, SatRecSoA, sizeof(satelliterecord_soa_t) * numberSatellites, cudaMemcpyHostToDevice));

	free(SatRecSoA);

	dim3 threadsperblock( NUM_THREADS , 1 );
	dim3 blockspergrid(  numberSatellites / NUM_THREADS + (!(numberSatellites % NUM_THREADS) ? 0 : 1), 1 , 1);
	sgp4initkernel<<< blockspergrid, threadsperblock >>>(d_satrec, numberSatellites);
	cudaError_t STATUS = cudaGetLastError();

}

void ComputeSGP4CUDA(	float4 *positions, double deltatime, int numberSatellites	){	
	currenttime += deltatime;
	
	dim3 threadsperblock( NUM_THREADS , 1 );
	dim3 blockspergrid(  numberSatellites / NUM_THREADS + (!(numberSatellites % NUM_THREADS) ? 0 : 1), 1 , 1);

	sgp4<<< blockspergrid, threadsperblock >>>(d_satrec, numberSatellites, currenttime, positions);
}

void FreeVariables(){
	cutilSafeCall(cudaFree(d_satrec));
}