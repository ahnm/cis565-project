#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "common.h"

enum gravconsttype
{
  wgs72old,
  wgs72,
  wgs84
};

struct gravconstant_t{
	double	mu,
			radiusearthkm,
			xke,
			tumin,
			j2,
			j3,
			j4,
			j3oj2;
};

void setGravConstant(gravconsttype whichconst, gravconstant_t &gravconst);

const double PI				=	std::atan(1.0) * 4.0;

const double xpdotp			=	1440.0 / (2.0 * PI);  // 229.1831180523293
// ------------ wgs-84 constants ------------ //
const double mu				=	398600.5;           // in km3 / s2
const double radiusearthkm	=	6378.137;			// km
const double xke			=	60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
const double tumin			=	1.0 / xke;
const double j2				=	0.00108262998905;
const double j3				=	-0.00000253215306;
const double j4				=	-0.00000161098761;
const double j3oj2			=	j3 / j2;


#endif