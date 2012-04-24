#pragma once
#ifndef SGP4_CUDA_H
#define SGP4_CUDA_H

#include "common.h"
#include "functions.h"


void initSGP4(	gravconsttype whichconst,
				std::vector<satelliterecord_aos_t> &SatRecAoS,
				int numberSatellites	);

void ComputeSGP4(	float4 *positions,
					double deltatime,
					int numberSatellites	);



static void dspace
(
int irez,
double d2201,  double d2211,  double d3210,   double d3222,  double d4410,
double d4422,  double d5220,  double d5232,   double d5421,  double d5433,
double dedt,   double del1,   double del2,    double del3,   double didt,
double dmdt,   double dnodt,  double domdt,   double argpo,  double argpdot,
double t,      double tc,     double gsto,    double xfact,  double xlamo,
double no,
double& atime, double& em,    double& argpm,  double& inclm, double& xli,
double& mm,    double& xni,   double& nodem,  double& dndt,  double& nm
);


static void dscom
(
double epoch,  double ep,     double argpp,   double tc,     double inclp,
double nodep,  double np,
double& snodm, double& cnodm, double& sinim,  double& cosim, double& sinomm,
double& cosomm,double& day,   double& e3,     double& ee2,   double& em,
double& emsq,  double& gam,   double& peo,    double& pgho,  double& pho,
double& pinco, double& plo,   double& rtemsq, double& se2,   double& se3,
double& sgh2,  double& sgh3,  double& sgh4,   double& sh2,   double& sh3,
double& si2,   double& si3,   double& sl2,    double& sl3,   double& sl4,
double& s1,    double& s2,    double& s3,     double& s4,    double& s5,
double& s6,    double& s7,    double& ss1,    double& ss2,   double& ss3,
double& ss4,   double& ss5,   double& ss6,    double& ss7,   double& sz1,
double& sz2,   double& sz3,   double& sz11,   double& sz12,  double& sz13,
double& sz21,  double& sz22,  double& sz23,   double& sz31,  double& sz32,
double& sz33,  double& xgh2,  double& xgh3,   double& xgh4,  double& xh2,
double& xh3,   double& xi2,   double& xi3,    double& xl2,   double& xl3,
double& xl4,   double& nm,    double& z1,     double& z2,    double& z3,
double& z11,   double& z12,   double& z13,    double& z21,   double& z22,
double& z23,   double& z31,   double& z32,    double& z33,   double& zmol,
double& zmos
);

static void dpper
(
double e3,     double ee2,    double peo,     double pgho,   double pho,
double pinco,  double plo,    double se2,     double se3,    double sgh2,
double sgh3,   double sgh4,   double sh2,     double sh3,    double si2,
double si3,    double sl2,    double sl3,     double sl4,    double t,
double xgh2,   double xgh3,   double xgh4,    double xh2,    double xh3,
double xi2,    double xi3,    double xl2,     double xl3,    double xl4,
double zmol,   double zmos,   double inclo,
char init,
double& ep,    double& inclp, double& nodep,  double& argpp, double& mp
);
static void dsinit
(
gravconstant_t gravity_constants,
double cosim,  double emsq,   double argpo,   double s1,     double s2,
double s3,     double s4,     double s5,      double sinim,  double ss1,
double ss2,    double ss3,    double ss4,     double ss5,    double sz1,
double sz3,    double sz11,   double sz13,    double sz21,   double sz23,
double sz31,   double sz33,   double t,       double tc,     double gsto,
double mo,     double mdot,   double no,      double nodeo,  double nodedot,
double xpidot, double z1,     double z3,      double z11,    double z13,
double z21,    double z23,    double z31,     double z33,    double ecco,
double eccsq,  double& em,    double& argpm,  double& inclm, double& mm,
double& nm,    double& nodem,
int& irez,
double& atime, double& d2201, double& d2211,  double& d3210, double& d3222,
double& d4410, double& d4422, double& d5220,  double& d5232, double& d5421,
double& d5433, double& dedt,  double& didt,   double& dmdt,  double& dndt,
double& dnodt, double& domdt, double& del1,   double& del2,  double& del3,
double& xfact, double& xlamo, double& xli,    double& xni
);

void sgp4initkernel(gravconstant_t gravity_constants, satelliterecord_soa_t *satrec, int tid);

void FreeVariables();
#endif