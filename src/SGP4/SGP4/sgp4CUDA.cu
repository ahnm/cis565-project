//#include "sgp4CUDA.h"

#include "common.h"
#include "constants.h"
#include "satelliterecord.h"
#include "tle.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\device_vector.h>
#include <thrust\sort.h>

#include "commonCUDA.cuh"
#include "sgp4initKernel.cu"
#include "sgp4Kernel.cu"

//struct SRCmpOrbitalPeriod {
//  __host__ __device__
//	  bool operator()(const satelliterecord_soa_t& sr1, const satelliterecord_soa_t& sr2) {
//		  return (2 * 3.14 / sr1.no) < (2 * 3.14 / sr2.no);
//  }
//};
//__device__ __constant__ gravconstant_t gravity_constants;

void ComputeSGP4CUDA(	gravconsttype whichconst, std::vector<satelliterecord_aos_t> &SatRecAoS,
						double starttime, double endtime, double deltatime ){
	
	int threads = 512;
	int N = SatRecAoS.size();

	unsigned int kernelTime;
	cutCreateTimer(&kernelTime);
	cutResetTimer(kernelTime);

	gravconstant_t gravconstant;
	setGravConstant(whichconst, gravconstant);

	cutStartTimer(kernelTime);
	cudaMemcpyToSymbol("gravity_constants", &gravconstant, sizeof(gravconstant_t), 0, cudaMemcpyHostToDevice);
	
	satelliterecord_soa_t *SatRecSoA = (satelliterecord_soa_t*) malloc(sizeof(satelliterecord_soa_t) * N);

	satelliteRecordConvert(SatRecAoS, SatRecSoA);

	struct satelliterecord_soa_t *d_satrec, *h_satrec;
	cutilSafeCall(cudaMalloc((void **) &d_satrec, sizeof(satelliterecord_soa_t) * N));
	cutilSafeCall(cudaMemcpy(d_satrec, SatRecSoA, sizeof(satelliterecord_soa_t) * N, cudaMemcpyHostToDevice));

	dim3 threadsperblock( threads, 1 );
	dim3 blockspergrid(  N / threads + (!(N % threads) ? 0 : 1), 1 , 1);
	sgp4initkernel<<< blockspergrid, threadsperblock >>>(d_satrec, N);
	cudaError_t STATUS = cudaGetLastError();

	double3 *h_r, *h_v;
	double3 *d_r, *d_v;

	h_r = (double3*)malloc(sizeof(double3) * N);
	h_v = (double3*)malloc(sizeof(double3) * N);
	cutilSafeCall(cudaMalloc((void **) &d_r, sizeof(double3) * N));
	cutilSafeCall(cudaMalloc((void **) &d_v, sizeof(double3) * N));
	//std::ofstream output, output2;
	//output.open("sat1.txt");
	//output2.open("sat2.txt");
	//output << "xyz = [";
	//output2 << "xyz = [";
	for(int i = 0; i < 1000; i++){
		sgp4<<< blockspergrid, threadsperblock >>>(d_satrec, N, i/2, d_r, d_v);
		//cutilSafeCall(cudaMemcpy(h_r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost));
		//printf("%f %f %f\n", h_r[0].x, h_r[0].y, h_r[0].z);
		//output << "[" << h_r[0].x << "," << h_r[0].y << "," << h_r[0].z << "];";
		//output2 << "[" << h_r[1].x << "," << h_r[1].y << "," << h_r[1].z << "];";
	}
	//output << "];";
	//output.close();
	//output2 << "];";
	//output2.close();
	//h_satrec = (satelliterecord_soa_t*) malloc(sizeof(satelliterecord_soa_t) * N);
	//cutilSafeCall(cudaMemcpy(h_satrec, d_satrec, sizeof(satelliterecord_soa_t) * N, cudaMemcpyDeviceToHost));

	printf ("Time for the kernel: %f ms\n", cutGetTimerValue(kernelTime));

	cutilSafeCall(cudaFree(d_satrec));
	cutilSafeCall(cudaFree(d_r));
	cutilSafeCall(cudaFree(d_v));
	/*int i = 10 + (10 % 0);
	std::cout << i << std::endl;*/
	//std::cout << sizeof(satelliterecord_soa_t) << std::endl;
	//std::cout << h_satrec[0].init << std::endl;
	//std::cout << h_satrec[N-1].init << std::endl;
	//std::cout << h_satrec[1000].a << std::endl;
/*
	thrust::host_vector<satelliterecord_soa_t> h_vector;
	h_vector.push_back(SatRecSoA);*/
/*
	thrust::device_vector<satelliterecord_soa_t> d_vector = h_vector;*/
	//thrust::sort(d_vector.begin(), d_vector.end(), SRCmpOrbitalPeriod());
	
    //struct satelliterecord_soa_t* _array = thrust::raw_pointer_cast( &d_vector[0] );
	/*std::cout << sizeof(satelliterecord_soa_t) << std::endl;
	int threads = 16;
	int blocks = (int)std::ceil((float)SatRecAoS.size() / (float)threads);
	sgp4initkernel<<<threads, blocks>>>(_array);

	thrust::host_vector<satelliterecord_soa_t> h_vec = d_vector;
	std::cout << h_vec[0].t[0] << std::endl;
	struct satelliterecord_soa_t* u;
	cutilSafeCall(cudaMemcpy(u, _array, sizeof(satelliterecord_soa_t) * SatRecAoS.size(), cudaMemcpyDeviceToHost));
	std::cout << u->init[0] << std::endl;
	std::cout << "test" << std::endl;*/
}