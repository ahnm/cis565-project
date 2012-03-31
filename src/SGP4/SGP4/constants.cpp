#include "constants.h"


void setGravConstant(gravconsttype whichconst, gravconstant_t &gravconst){
	switch (whichconst)
	{
		// -- wgs-72 low precision str#3 constants --
	case wgs72old:
		gravconst.mu     = 398600.79964;        // in km3 / s2
		gravconst.radiusearthkm = 6378.135;     // km
		gravconst.xke    = 0.0743669161;
		gravconst.tumin  = 1.0 / xke;
		gravconst.j2     =   0.001082616;
		gravconst.j3     =  -0.00000253881;
		gravconst.j4     =  -0.00000165597;
		gravconst.j3oj2  =  j3 / j2;
		break;
		// ------------ wgs-72 constants ------------
	case wgs72:
		gravconst.mu     = 398600.8;            // in km3 / s2
		gravconst.radiusearthkm = 6378.135;     // km
		gravconst.xke    = 60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
		gravconst.tumin  = 1.0 / xke;
		gravconst.j2     =   0.001082616;
		gravconst.j3     =  -0.00000253881;
		gravconst.j4     =  -0.00000165597;
		gravconst.j3oj2  =  j3 / j2;
		break;
	case wgs84:
		// ------------ wgs-84 constants ------------
		gravconst.mu     = 398600.5;            // in km3 / s2
		gravconst.radiusearthkm = 6378.137;     // km
		gravconst.xke    = 60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
		gravconst.tumin  = 1.0 / xke;
		gravconst.j2     =   0.00108262998905;
		gravconst.j3     =  -0.00000253215306;
		gravconst.j4     =  -0.00000161098761;
		gravconst.j3oj2  =  j3 / j2;
		break;
	default:
		fprintf(stderr,"unknown gravity option (%d)\n",whichconst);
		break;
	}


}