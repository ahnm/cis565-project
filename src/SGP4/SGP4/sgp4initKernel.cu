/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "common.h"
#include "commonCUDA.cu"
#include "satelliterecord.h"

__global__ void sgp4init
	(gravconsttype whichconst,       const int satn,     const double epoch,
	const double xbstar,  const double xecco, const double xargpo,
	const double xinclo,  const double xmo,   const double xno,
	const double xnodeo,  elsetrec& satrec
	)
{
	/* --------------------- local variables ------------------------ */
	double ao, ainv,   con42, cosio, sinio, cosio2, eccsq,
		omeosq, posq,   rp,     rteosq,
		cnodm , snodm , cosim , sinim , cosomm, sinomm, cc1sq ,
		cc2   , cc3   , coef  , coef1 , cosio4, day   , dndt  ,
		em    , emsq  , eeta  , etasq , gam   , argpm , nodem ,
		inclm , mm    , nm    , perige, pinvsq, psisq , qzms24,
		rtemsq, s1    , s2    , s3    , s4    , s5    , s6    ,
		s7    , sfour , ss1   , ss2   , ss3   , ss4   , ss5   ,
		ss6   , ss7   , sz1   , sz2   , sz3   , sz11  , sz12  ,
		sz13  , sz21  , sz22  , sz23  , sz31  , sz32  , sz33  ,
		tc    , temp  , temp1 , temp2 , temp3 , tsi   , xpidot,
		xhdot1, z1    , z2    , z3    , z11   , z12   , z13   ,
		z21   , z22   , z23   , z31   , z32   , z33,
		qzms2t, ss, j2, j3oj2, j4, x2o3, r[3], v[3],
		tumin, mu, radiusearthkm, xke, j3;

	/* ------------------------ initialization --------------------- */
	// sgp4fix divisor for divide by zero check on inclination
	const double temp4    =   1.0 + cos(pi-1.0e-9);

	/* ----------- set all near earth variables to zero ------------ */
	satrec.isimp   = 0;   satrec.method = 'n'; satrec.aycof    = 0.0;
	satrec.con41   = 0.0; satrec.cc1    = 0.0; satrec.cc4      = 0.0;
	satrec.cc5     = 0.0; satrec.d2     = 0.0; satrec.d3       = 0.0;
	satrec.d4      = 0.0; satrec.delmo  = 0.0; satrec.eta      = 0.0;
	satrec.argpdot = 0.0; satrec.omgcof = 0.0; satrec.sinmao   = 0.0;
	satrec.t       = 0.0; satrec.t2cof  = 0.0; satrec.t3cof    = 0.0;
	satrec.t4cof   = 0.0; satrec.t5cof  = 0.0; satrec.x1mth2   = 0.0;
	satrec.x7thm1  = 0.0; satrec.mdot   = 0.0; satrec.nodedot  = 0.0;
	satrec.xlcof   = 0.0; satrec.xmcof  = 0.0; satrec.nodecf   = 0.0;

	/* ----------- set all deep space variables to zero ------------ */
	satrec.irez  = 0;   satrec.d2201 = 0.0; satrec.d2211 = 0.0;
	satrec.d3210 = 0.0; satrec.d3222 = 0.0; satrec.d4410 = 0.0;
	satrec.d4422 = 0.0; satrec.d5220 = 0.0; satrec.d5232 = 0.0;
	satrec.d5421 = 0.0; satrec.d5433 = 0.0; satrec.dedt  = 0.0;
	satrec.del1  = 0.0; satrec.del2  = 0.0; satrec.del3  = 0.0;
	satrec.didt  = 0.0; satrec.dmdt  = 0.0; satrec.dnodt = 0.0;
	satrec.domdt = 0.0; satrec.e3    = 0.0; satrec.ee2   = 0.0;
	satrec.peo   = 0.0; satrec.pgho  = 0.0; satrec.pho   = 0.0;
	satrec.pinco = 0.0; satrec.plo   = 0.0; satrec.se2   = 0.0;
	satrec.se3   = 0.0; satrec.sgh2  = 0.0; satrec.sgh3  = 0.0;
	satrec.sgh4  = 0.0; satrec.sh2   = 0.0; satrec.sh3   = 0.0;
	satrec.si2   = 0.0; satrec.si3   = 0.0; satrec.sl2   = 0.0;
	satrec.sl3   = 0.0; satrec.sl4   = 0.0; satrec.gsto  = 0.0;
	satrec.xfact = 0.0; satrec.xgh2  = 0.0; satrec.xgh3  = 0.0;
	satrec.xgh4  = 0.0; satrec.xh2   = 0.0; satrec.xh3   = 0.0;
	satrec.xi2   = 0.0; satrec.xi3   = 0.0; satrec.xl2   = 0.0;
	satrec.xl3   = 0.0; satrec.xl4   = 0.0; satrec.xlamo = 0.0;
	satrec.zmol  = 0.0; satrec.zmos  = 0.0; satrec.atime = 0.0;
	satrec.xli   = 0.0; satrec.xni   = 0.0;

	// sgp4fix - note the following variables are also passed directly via satrec.
	// it is possible to streamline the sgp4init call by deleting the "x"
	// variables, but the user would need to set the satrec.* values first. we
	// include the additional assignments in case twoline2rv is not used.
	satrec.bstar   = xbstar;
	satrec.ecco    = xecco;
	satrec.argpo   = xargpo;
	satrec.inclo   = xinclo;
	satrec.mo	    = xmo;
	satrec.no	    = xno;
	satrec.nodeo   = xnodeo;

	/* ------------------------ earth constants ----------------------- */
	// sgp4fix identify constants and allow alternate values
	getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
	ss     = 78.0 / radiusearthkm + 1.0;
	qzms2t = pow(((120.0 - 78.0) / radiusearthkm), 4);
	x2o3   =  2.0 / 3.0;

	satrec.init = 'y';
	satrec.t	 = 0.0;

	initl
		(
		satn, whichconst, satrec.ecco, epoch, satrec.inclo, satrec.no, satrec.method,
		ainv, ao, satrec.con41, con42, cosio, cosio2, eccsq, omeosq,
		posq, rp, rteosq, sinio, satrec.gsto
		);
	satrec.error = 0;

	if (rp < 1.0)
	{
		//         printf("# *** satn%d epoch elts sub-orbital ***\n", satn);
		satrec.error = 5;
	}

	if ((omeosq >= 0.0 ) || ( satrec.no >= 0.0))
	{
		satrec.isimp = 0;
		if (rp < (220.0 / radiusearthkm + 1.0))
			satrec.isimp = 1;
		sfour  = ss;
		qzms24 = qzms2t;
		perige = (rp - 1.0) * radiusearthkm;

		/* - for perigees below 156 km, s and qoms2t are altered - */
		if (perige < 156.0)
		{
			sfour = perige - 78.0;
			if (perige < 98.0)
				sfour = 20.0;
			qzms24 = pow(((120.0 - sfour) / radiusearthkm), 4.0);
			sfour  = sfour / radiusearthkm + 1.0;
		}
		pinvsq = 1.0 / posq;

		tsi  = 1.0 / (ao - sfour);
		satrec.eta  = ao * satrec.ecco * tsi;
		etasq = satrec.eta * satrec.eta;
		eeta  = satrec.ecco * satrec.eta;
		psisq = fabs(1.0 - etasq);
		coef  = qzms24 * pow(tsi, 4.0);
		coef1 = coef / pow(psisq, 3.5);
		cc2   = coef1 * satrec.no * (ao * (1.0 + 1.5 * etasq + eeta *
			(4.0 + etasq)) + 0.375 * j2 * tsi / psisq * satrec.con41 *
			(8.0 + 3.0 * etasq * (8.0 + etasq)));
		satrec.cc1   = satrec.bstar * cc2;
		cc3   = 0.0;
		if (satrec.ecco > 1.0e-4)
			cc3 = -2.0 * coef * tsi * j3oj2 * satrec.no * sinio / satrec.ecco;
		satrec.x1mth2 = 1.0 - cosio2;
		satrec.cc4    = 2.0* satrec.no * coef1 * ao * omeosq *
			(satrec.eta * (2.0 + 0.5 * etasq) + satrec.ecco *
			(0.5 + 2.0 * etasq) - j2 * tsi / (ao * psisq) *
			(-3.0 * satrec.con41 * (1.0 - 2.0 * eeta + etasq *
			(1.5 - 0.5 * eeta)) + 0.75 * satrec.x1mth2 *
			(2.0 * etasq - eeta * (1.0 + etasq)) * cos(2.0 * satrec.argpo)));
		satrec.cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
			(etasq + eeta) + eeta * etasq);
		cosio4 = cosio2 * cosio2;
		//temp1  = 1.5 * j2 * pinvsq * satrec.no;
		temp2  = 0.5 * 1.5 * j2 * pinvsq * satrec.no * j2 * pinvsq;
		temp3  = -0.46875 * j4 * pinvsq * pinvsq * satrec.no;
		satrec.mdot     = satrec.no + 0.5 * 1.5 * j2 * pinvsq * satrec.no * rteosq * satrec.con41 + 0.0625 *
			temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
		satrec.argpdot  = -0.5 * 1.5 * j2 * pinvsq * satrec.no * con42 + 0.0625 * temp2 *
			(7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
			temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
		xhdot1            = -1.5 * j2 * pinvsq * satrec.no * cosio;
		satrec.nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
			2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
		xpidot            =  satrec.argpdot+ satrec.nodedot;
		satrec.omgcof   = satrec.bstar * cc3 * cos(satrec.argpo);
		satrec.xmcof    = 0.0;
		if (satrec.ecco > 1.0e-4)
			satrec.xmcof = -x2o3 * coef * satrec.bstar / eeta;
		satrec.nodecf = 3.5 * omeosq * xhdot1 * satrec.cc1;
		satrec.t2cof   = 1.5 * satrec.cc1;
		// sgp4fix for divide by zero with xinco = 180 deg
		if (fabs(cosio+1.0) > 1.5e-12)
			satrec.xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
		else
			satrec.xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
		satrec.aycof   = -0.5 * j3oj2 * sinio;
		satrec.delmo   = pow((1.0 + satrec.eta * cos(satrec.mo)), 3);
		satrec.sinmao  = sin(satrec.mo);
		satrec.x7thm1  = 7.0 * cosio2 - 1.0;

		/* --------------- deep space initialization ------------- */
		if ((2*pi / satrec.no) >= 225.0)
		{
			satrec.method = 'd';
			satrec.isimp  = 1;
			tc    =  0.0;
			inclm = satrec.inclo;

			dscom
				(
				epoch, satrec.ecco, satrec.argpo, tc, satrec.inclo, satrec.nodeo,
				satrec.no, snodm, cnodm,  sinim, cosim,sinomm,     cosomm,
				day, satrec.e3, satrec.ee2, em,         emsq, gam,
				satrec.peo,  satrec.pgho,   satrec.pho, satrec.pinco,
				satrec.plo,  rtemsq,        satrec.se2, satrec.se3,
				satrec.sgh2, satrec.sgh3,   satrec.sgh4,
				satrec.sh2,  satrec.sh3,    satrec.si2, satrec.si3,
				satrec.sl2,  satrec.sl3,    satrec.sl4, s1, s2, s3, s4, s5,
				s6,   s7,   ss1,  ss2,  ss3,  ss4,  ss5,  ss6,  ss7, sz1, sz2, sz3,
				sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33,
				satrec.xgh2, satrec.xgh3,   satrec.xgh4, satrec.xh2,
				satrec.xh3,  satrec.xi2,    satrec.xi3,  satrec.xl2,
				satrec.xl3,  satrec.xl4,    nm, z1, z2, z3, z11,
				z12, z13, z21, z22, z23, z31, z32, z33,
				satrec.zmol, satrec.zmos
				);
			dpper
				(
				satrec.e3, satrec.ee2, satrec.peo, satrec.pgho,
				satrec.pho, satrec.pinco, satrec.plo, satrec.se2,
				satrec.se3, satrec.sgh2, satrec.sgh3, satrec.sgh4,
				satrec.sh2, satrec.sh3, satrec.si2,  satrec.si3,
				satrec.sl2, satrec.sl3, satrec.sl4,  satrec.t,
				satrec.xgh2,satrec.xgh3,satrec.xgh4, satrec.xh2,
				satrec.xh3, satrec.xi2, satrec.xi3,  satrec.xl2,
				satrec.xl3, satrec.xl4, satrec.zmol, satrec.zmos, inclm, satrec.init,
				satrec.ecco, satrec.inclo, satrec.nodeo, satrec.argpo, satrec.mo
				);

			argpm  = 0.0;
			nodem  = 0.0;
			mm     = 0.0;

			dsinit
				(
				whichconst,
				cosim, emsq, satrec.argpo, s1, s2, s3, s4, s5, sinim, ss1, ss2, ss3, ss4,
				ss5, sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33, satrec.t, tc,
				satrec.gsto, satrec.mo, satrec.mdot, satrec.no, satrec.nodeo,
				satrec.nodedot, xpidot, z1, z3, z11, z13, z21, z23, z31, z33,
				satrec.ecco, eccsq, em, argpm, inclm, mm, nm, nodem,
				satrec.irez,  satrec.atime,
				satrec.d2201, satrec.d2211, satrec.d3210, satrec.d3222 ,
				satrec.d4410, satrec.d4422, satrec.d5220, satrec.d5232,
				satrec.d5421, satrec.d5433, satrec.dedt,  satrec.didt,
				satrec.dmdt,  dndt,         satrec.dnodt, satrec.domdt ,
				satrec.del1,  satrec.del2,  satrec.del3,  satrec.xfact,
				satrec.xlamo, satrec.xli,   satrec.xni
				);
		}

		/* ----------- set variables if not deep space ----------- */
		if (satrec.isimp != 1)
		{
			cc1sq          = satrec.cc1 * satrec.cc1;
			satrec.d2    = 4.0 * ao * tsi * cc1sq;
			temp           = satrec.d2 * tsi * satrec.cc1 / 3.0;
			satrec.d3    = (17.0 * ao + sfour) * temp;
			satrec.d4    = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) *
				satrec.cc1;
			satrec.t3cof = satrec.d2 + 2.0 * cc1sq;
			satrec.t4cof = 0.25 * (3.0 * satrec.d3 + satrec.cc1 *
				(12.0 * satrec.d2 + 10.0 * cc1sq));
			satrec.t5cof = 0.2 * (3.0 * satrec.d4 +
				12.0 * satrec.cc1 * satrec.d3 +
				6.0 * satrec.d2 * satrec.d2 +
				15.0 * cc1sq * (2.0 * satrec.d2 + cc1sq));
		}
	} // if omeosq = 0 ...

	/* finally propogate to zero epoch to initialise all others. */
	if(satrec.error == 0)
		sgp4(whichconst, satrec, 0.0, r, v);

	satrec.init = 'n';

	//#include "debug6.cpp"
	return satrec.error;
}  // end sgp4init


///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
///
/// CUDA kernel
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
__global__ 
void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= count) return;

    sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static
void Add(const float *op1, const float *op2, int count, float *sum)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(count, threads.x));

    AddKernel<<<blocks, threads>>>(op1, op2, count, sum);
}



static void initl
	(
	int satn,      gravconsttype whichconst,
	double ecco,   double epoch,  double inclo,   double& no,
	char& method,
	double& ainv,  double& ao,    double& con41,  double& con42, double& cosio,
	double& cosio2,double& eccsq, double& omeosq, double& posq,
	double& rp,    double& rteosq,double& sinio , double& gsto
	)
{
	/* --------------------- local variables ------------------------ */
	double ak, d1, del, adel, po, x2o3, j2, xke,
		tumin, mu, radiusearthkm, j3, j4, j3oj2;

	// sgp4fix use old way of finding gst
	int ids70;
	double ts70, ds70, tfrac, c1, thgr70, fk5r, c1p2p, thgr, thgro;
	const double twopi = 2.0 * pi;

	/* ----------------------- earth constants ---------------------- */
	// sgp4fix identify constants and allow alternate values
	getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
	x2o3   = 2.0 / 3.0;

	/* ------------- calculate auxillary epoch quantities ---------- */
	eccsq  = ecco * ecco;
	omeosq = 1.0 - eccsq;
	rteosq = sqrt(omeosq);
	cosio  = cos(inclo);
	cosio2 = cosio * cosio;

	/* ------------------ un-kozai the mean motion ----------------- */
	ak    = pow(xke / no, x2o3);
	d1    = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
	del   = d1 / (ak * ak);
	adel  = ak * (1.0 - del * del - del *
		(1.0 / 3.0 + 134.0 * del * del / 81.0));
	del   = d1/(adel * adel);
	no    = no / (1.0 + del);

	ao    = pow(xke / no, x2o3);
	sinio = sin(inclo);
	po    = ao * omeosq;
	con42 = 1.0 - 5.0 * cosio2;
	con41 = -con42-cosio2-cosio2;
	ainv  = 1.0 / ao;
	posq  = po * po;
	rp    = ao * (1.0 - ecco);
	method = 'n';

	// sgp4fix modern approach to finding sidereal timew
	// gsto = gstime(epoch + 2433281.5);

	// sgp4fix use old way of finding gst
	// count integer number of days from 0 jan 1970
	ts70  = epoch - 7305.0;
	ids70 = floor(ts70 + 1.0e-8);
	ds70  = ids70;
	tfrac = ts70 - ds70;
	// find greenwich location at epoch
	c1    = 1.72027916940703639e-2;
	thgr70= 1.7321343856509374;
	fk5r  = 5.07551419432269442e-15;
	c1p2p = c1 + twopi;
	gsto  = fmod( thgr70 + c1*ds70 + c1p2p*tfrac + ts70*ts70*fk5r, twopi);
	if ( gsto < 0.0 )
		gsto = gsto + twopi;

	//#include "debug5.cpp"
}  // end initl