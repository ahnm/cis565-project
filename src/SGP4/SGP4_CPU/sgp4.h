#pragma once
#ifndef SGP4_CUDA_H
#define SGP4_CUDA_H

#include "common.h"
#include "functions.h"


void initSGP4(	gravconsttype whichconst,
				std::vector<satelliterecord_aos_t> &SatRecAoS,
				int numberSatellites	);

void ComputeSGP4(	float4 *positions,
					t_var deltatime,
					int numberSatellites	);



static void dspace
(
int irez,
t_var d2201,  t_var d2211,  t_var d3210,   t_var d3222,  t_var d4410,
t_var d4422,  t_var d5220,  t_var d5232,   t_var d5421,  t_var d5433,
t_var dedt,   t_var del1,   t_var del2,    t_var del3,   t_var didt,
t_var dmdt,   t_var dnodt,  t_var domdt,   t_var argpo,  t_var argpdot,
t_var t,      t_var tc,     t_var gsto,    t_var xfact,  t_var xlamo,
t_var no,
t_var& atime, t_var& em,    t_var& argpm,  t_var& inclm, t_var& xli,
t_var& mm,    t_var& xni,   t_var& nodem,  t_var& dndt,  t_var& nm
);


static void dscom
(
t_var epoch,  t_var ep,     t_var argpp,   t_var tc,     t_var inclp,
t_var nodep,  t_var np,
t_var& snodm, t_var& cnodm, t_var& sinim,  t_var& cosim, t_var& sinomm,
t_var& cosomm,t_var& day,   t_var& e3,     t_var& ee2,   t_var& em,
t_var& emsq,  t_var& gam,   t_var& peo,    t_var& pgho,  t_var& pho,
t_var& pinco, t_var& plo,   t_var& rtemsq, t_var& se2,   t_var& se3,
t_var& sgh2,  t_var& sgh3,  t_var& sgh4,   t_var& sh2,   t_var& sh3,
t_var& si2,   t_var& si3,   t_var& sl2,    t_var& sl3,   t_var& sl4,
t_var& s1,    t_var& s2,    t_var& s3,     t_var& s4,    t_var& s5,
t_var& s6,    t_var& s7,    t_var& ss1,    t_var& ss2,   t_var& ss3,
t_var& ss4,   t_var& ss5,   t_var& ss6,    t_var& ss7,   t_var& sz1,
t_var& sz2,   t_var& sz3,   t_var& sz11,   t_var& sz12,  t_var& sz13,
t_var& sz21,  t_var& sz22,  t_var& sz23,   t_var& sz31,  t_var& sz32,
t_var& sz33,  t_var& xgh2,  t_var& xgh3,   t_var& xgh4,  t_var& xh2,
t_var& xh3,   t_var& xi2,   t_var& xi3,    t_var& xl2,   t_var& xl3,
t_var& xl4,   t_var& nm,    t_var& z1,     t_var& z2,    t_var& z3,
t_var& z11,   t_var& z12,   t_var& z13,    t_var& z21,   t_var& z22,
t_var& z23,   t_var& z31,   t_var& z32,    t_var& z33,   t_var& zmol,
t_var& zmos
);

static void dpper
(
t_var e3,     t_var ee2,    t_var peo,     t_var pgho,   t_var pho,
t_var pinco,  t_var plo,    t_var se2,     t_var se3,    t_var sgh2,
t_var sgh3,   t_var sgh4,   t_var sh2,     t_var sh3,    t_var si2,
t_var si3,    t_var sl2,    t_var sl3,     t_var sl4,    t_var t,
t_var xgh2,   t_var xgh3,   t_var xgh4,    t_var xh2,    t_var xh3,
t_var xi2,    t_var xi3,    t_var xl2,     t_var xl3,    t_var xl4,
t_var zmol,   t_var zmos,   t_var inclo,
char init,
t_var& ep,    t_var& inclp, t_var& nodep,  t_var& argpp, t_var& mp
);
static void dsinit
(
gravconstant_t gravity_constants,
t_var cosim,  t_var emsq,   t_var argpo,   t_var s1,     t_var s2,
t_var s3,     t_var s4,     t_var s5,      t_var sinim,  t_var ss1,
t_var ss2,    t_var ss3,    t_var ss4,     t_var ss5,    t_var sz1,
t_var sz3,    t_var sz11,   t_var sz13,    t_var sz21,   t_var sz23,
t_var sz31,   t_var sz33,   t_var t,       t_var tc,     t_var gsto,
t_var mo,     t_var mdot,   t_var no,      t_var nodeo,  t_var nodedot,
t_var xpidot, t_var z1,     t_var z3,      t_var z11,    t_var z13,
t_var z21,    t_var z23,    t_var z31,     t_var z33,    t_var ecco,
t_var eccsq,  t_var& em,    t_var& argpm,  t_var& inclm, t_var& mm,
t_var& nm,    t_var& nodem,
int& irez,
t_var& atime, t_var& d2201, t_var& d2211,  t_var& d3210, t_var& d3222,
t_var& d4410, t_var& d4422, t_var& d5220,  t_var& d5232, t_var& d5421,
t_var& d5433, t_var& dedt,  t_var& didt,   t_var& dmdt,  t_var& dndt,
t_var& dnodt, t_var& domdt, t_var& del1,   t_var& del2,  t_var& del3,
t_var& xfact, t_var& xlamo, t_var& xli,    t_var& xni
);

void sgp4initkernel(gravconstant_t gravity_constants, satelliterecord_soa_t *satrec, int tid);

void FreeVariables();
#endif