#ifndef SATELLITE_RECORD
#define SATELLITE_RECORD

#include "common.h"
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

struct satelliterecord_t{
	int		satellite_num;
	
	double	a,				//semi-major axis
			e,				//eccentricity
			i,				//inclination (radians)
			raan,			//right ascension of the ascending node
			w,				//argument of periapsis
			M,				//Mean Anomaly

			epoch_jd,		//epoch Julian Days
			
			n,				//mean motion
			n_dt,			//first derivative of mean motion
			n_ddt,			//second derivative of mean motion
			
			bstar,			//Drag coefficient B = C_D A/m  -> B* = B * rho_0 / 2

			nu				//True Anomaly
			;

	double3	r,			//Position
			v;			//Velocity
};

// Template structure to pass to kernel
struct SatelliteRecordArray
{
    struct satelliterecord_t* _array;
    int _size;
};


//SatelliteRecordArray convertToKernel( thrust::device_vector< satelliterecord_t >& dVec )
//{
//    SatelliteRecordArray SRArray;
//    SRArray._array = thrust::raw_pointer_cast( &dVec[0] );
//    SRArray._size  = ( int ) dVec.size();
//
//    return SRArray;
//};

#endif