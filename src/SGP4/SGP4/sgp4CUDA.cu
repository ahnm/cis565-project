//#include "sgp4CUDA.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "constants.h"
#include "satelliterecord.h"
#include <thrust\device_vector.h>
#include <thrust\sort.h>

struct SRCmpOrbitalPeriod {
  __host__ __device__
  bool operator()(const satelliterecord_t& sr1, const satelliterecord_t& sr2) {
	  return (2 * 3.14 / sr1.n) < (2 * 3.14 / sr2.n);
  }
};

void ComputeSGP4CUDA(	gravconsttype whichconst, std::vector<satelliterecord_t> &SatRec,
						double starttime, double endtime, double deltatime ){
	
	gravconstant_t gravconstant;
	setGravConstant(whichconst, gravconstant);

	thrust::host_vector<satelliterecord_t> h_vector;
	for(int i = 0; i < SatRec.size(); i++){
		h_vector.push_back(SatRec[i]);
	}

	thrust::device_vector<satelliterecord_t> d_vector = h_vector;
	thrust::sort(d_vector.begin(), d_vector.end(), SRCmpOrbitalPeriod());

}