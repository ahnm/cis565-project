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
	t_var	mu,
			radiusearthkm,
			xke,
			tumin,
			j2,
			j3,
			j4,
			j3oj2;
};

void setGravConstant(gravconsttype whichconst, gravconstant_t &gravconst);

const t_var PI				=	std::atan(1.0) * 4.0;

const t_var xpdotp			=	1440.0 / (2.0 * PI);  // 229.1831180523293
// ------------ wgs-84 constants ------------ //
const t_var mu				=	398600.5;           // in km3 / s2
const t_var radiusearthkm	=	6378.137;			// km
const t_var xke			=	60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
const t_var tumin			=	1.0 / xke;
const t_var j2				=	0.00108262998905;
const t_var j3				=	-0.00000253215306;
const t_var j4				=	-0.00000161098761;
const t_var j3oj2			=	j3 / j2;


#endif