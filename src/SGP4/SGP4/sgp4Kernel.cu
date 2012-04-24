#include "common.h"
#include "commonCUDA.cuh"
#include <math_constants.h>

__device__ static void dspace
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


__global__ void sgp4(satelliterecord_soa_t *satrec, int N, t_var tsince, float4 *r)
{
#define STRIDE		0
#define OFFSET		0
#define GROUP_SIZE	512
	int block_start_idx = blockIdx.x * blockDim.x;
	//int tid = block_start_idx + ((threadIdx.x + OFFSET) % STRIDE);
	int tid = block_start_idx + threadIdx.x;
	if(tid < N){
		t_var	am		,	cos2u	,	coseo1	,	cosip	,	cosisq	,	delm	,	delomg	,	em		,	
				el2		,	eo1		,	ep		,	esine	,	argpm	,	pl		,	mrt		,	rl		,
				sin2u	,	sineo1	,	sinip	,	su		,	tem5	,	tempa	,
				tempe	,	templ	,	u		,	ux		,	uy		,	uz		,	inclm	,	mm		,
				nm		,	nodem	,	xinc	,	xincp	,	xl		,	xlm		,	mp		,	xnode	,	
				nodep	,	tc		,	dndt	;
		int ktr;

		/* ------------------ set mathematical constants --------------- */
		// sgp4fix divisor for divide by zero check on inclination
		//const double temp4    =   1.0 + cos(CUDART_PI-1.0e-9);
		//twopi = 2.0 * CUDART_PI;
		//x2o3  = 2.0 / 3.0;
		// sgp4fix identify constants and allow alternate values
		//vkmpersec     = gravity_constants.radiusearthkm * gravity_constants.xke/60.0;
	
		/* --------------------- clear sgp4 error flag ----------------- */
		satrec[tid].t     = tsince;
		satrec[tid].error = 0;

		/* ------- update for secular gravity and atmospheric drag ----- */
		//xmdf    = satrec[tid].mo + satrec[tid].mdot * satrec[tid].t;
		//argpdf  = satrec[tid].argpo + satrec[tid].argpdot * satrec[tid].t;
		//nodedf  = satrec[tid].nodeo + satrec[tid].nodedot * satrec[tid].t;
		argpm   = satrec[tid].argpo + satrec[tid].argpdot * satrec[tid].t;
		mm      = satrec[tid].mo + satrec[tid].mdot * satrec[tid].t;
		//t2      = satrec[tid].t * satrec[tid].t;
		//nodem   = nodedf + satrec[tid].nodecf * t2;
		//nodem   = nodedf + satrec[tid].nodecf * pow(satrec[tid].t, 2.0);
		nodem   = (satrec[tid].nodeo + satrec[tid].nodedot * satrec[tid].t) + satrec[tid].nodecf * pow(satrec[tid].t, (t_var)2.0);
		tempa   = 1.0 - satrec[tid].cc1 * satrec[tid].t;
		tempe   = satrec[tid].bstar * satrec[tid].cc4 * satrec[tid].t;
		templ   = satrec[tid].t2cof * pow(satrec[tid].t, (t_var)2.0);

		if (satrec[tid].isimp != 1)
		{
			delomg = satrec[tid].omgcof * satrec[tid].t;
			delm   = satrec[tid].xmcof *
				(pow((1.0 + satrec[tid].eta * cos(satrec[tid].mo + satrec[tid].mdot * satrec[tid].t)), 3) -
				satrec[tid].delmo);
			//temp   = delomg + delm;
			mm     = satrec[tid].mo + satrec[tid].mdot * satrec[tid].t + delomg + delm;
			argpm  = (satrec[tid].argpo + satrec[tid].argpdot * satrec[tid].t) - (delomg + delm);
			//t3     = t2 * satrec[tid].t;
			//t3     = pow(satrec[tid].t, 3.0);
			//t4     = t3 * satrec[tid].t;
			//t4     = pow(satrec[tid].t, 4.0);
			//tempa  = tempa - satrec[tid].d2 * t2 - satrec[tid].d3 * t3 - satrec[tid].d4 * t4;
			tempa  = tempa - satrec[tid].d2 * pow(satrec[tid].t, (t_var)2.0) - satrec[tid].d3 * pow(satrec[tid].t, (t_var)3.0) - satrec[tid].d4 * pow(satrec[tid].t, (t_var)4.0);
			tempe  = tempe + satrec[tid].bstar * satrec[tid].cc5 * (sin(mm) - satrec[tid].sinmao);
			//templ  = templ + satrec[tid].t3cof * t3 + t4 * (satrec[tid].t4cof + satrec[tid].t * satrec[tid].t5cof);
			templ  = templ + satrec[tid].t3cof * pow(satrec[tid].t, (t_var)3.0) + pow(satrec[tid].t, (t_var)4.0) * (satrec[tid].t4cof + satrec[tid].t * satrec[tid].t5cof);
		}

		nm    = satrec[tid].no;
		em    = satrec[tid].ecco;
		inclm = satrec[tid].inclo;
		if (satrec[tid].method == 'd')
		{
			tc = satrec[tid].t;
			dspace
				(
				satrec[tid].irez,
				satrec[tid].d2201, satrec[tid].d2211, satrec[tid].d3210,
				satrec[tid].d3222, satrec[tid].d4410, satrec[tid].d4422,
				satrec[tid].d5220, satrec[tid].d5232, satrec[tid].d5421,
				satrec[tid].d5433, satrec[tid].dedt,  satrec[tid].del1,
				satrec[tid].del2,  satrec[tid].del3,  satrec[tid].didt,
				satrec[tid].dmdt,  satrec[tid].dnodt, satrec[tid].domdt,
				satrec[tid].argpo, satrec[tid].argpdot, satrec[tid].t, tc,
				satrec[tid].gsto, satrec[tid].xfact, satrec[tid].xlamo,
				satrec[tid].no, satrec[tid].atime,
				em, argpm, inclm, satrec[tid].xli, mm, satrec[tid].xni,
				nodem, dndt, nm
				);
		} // if method = d

		if (nm <= 0.0)
		{
			//         printf("# error nm %f\n", nm);
			satrec[tid].error = 2;
		}
		am = pow((gravity_constants.xke / nm),(t_var)(2.0 / 3.0)) * tempa * tempa;
		nm = gravity_constants.xke / pow(am, (t_var)1.5);
		em = em - tempe;

		// fix tolerance for error recognition
		if ((em >= 1.0) || (em < -0.001) || (am < 0.95))
		{
			//         printf("# error em %f\n", em);
			satrec[tid].error = 1;
		}
		if (em < 0.0)
			em  = 1.0e-6;
		mm     = mm + satrec[tid].no * templ;
		xlm    = mm + argpm + nodem;
		//emsq   = em * em;
		//temp   = 1.0 - emsq;

		nodem  = fmod(nodem, (t_var)(2.0 * CUDART_PI));
		argpm  = fmod(argpm, (t_var)(2.0 * CUDART_PI));
		xlm    = fmod(xlm, (t_var)(2.0 * CUDART_PI));
		mm     = fmod(xlm - argpm - nodem, (t_var)(2.0 * CUDART_PI));

		/* ----------------- compute extra mean quantities ------------- */
		/*sinim = sin(inclm);
		cosim = cos(inclm);*/

		/* -------------------- add lunar-solar periodics -------------- */
		ep     = em;
		xincp  = inclm;
		//argpp  = argpm;
		nodep  = nodem;
		mp     = mm;
		sinip  = sin(inclm);
		cosip  = cos(inclm);
		if (satrec[tid].method == 'd')
		{
			dpper
				(
				satrec[tid].e3,   satrec[tid].ee2,  satrec[tid].peo,
				satrec[tid].pgho, satrec[tid].pho,  satrec[tid].pinco,
				satrec[tid].plo,  satrec[tid].se2,  satrec[tid].se3,
				satrec[tid].sgh2, satrec[tid].sgh3, satrec[tid].sgh4,
				satrec[tid].sh2,  satrec[tid].sh3,  satrec[tid].si2,
				satrec[tid].si3,  satrec[tid].sl2,  satrec[tid].sl3,
				satrec[tid].sl4,  satrec[tid].t,    satrec[tid].xgh2,
				satrec[tid].xgh3, satrec[tid].xgh4, satrec[tid].xh2,
				satrec[tid].xh3,  satrec[tid].xi2,  satrec[tid].xi3,
				satrec[tid].xl2,  satrec[tid].xl3,  satrec[tid].xl4,
				satrec[tid].zmol, satrec[tid].zmos, satrec[tid].inclo,
				'n', ep, xincp, nodep, argpm, mp
				);
			if (xincp < 0.0)
			{
				xincp  = -xincp;
				nodep = nodep + CUDART_PI;
				argpm  = argpm - CUDART_PI;
			}
			if ((ep < 0.0 ) || ( ep > 1.0))
			{
				//            printf("# error ep %f\n", ep);
				satrec[tid].error = 3;
			}
		} // if method = d

		/* -------------------- long period periodics ------------------ */
		if (satrec[tid].method == 'd')
		{
			sinip =  sin(xincp);
			cosip =  cos(xincp);
			satrec[tid].aycof = -0.5*gravity_constants.j3oj2*sinip;
			// sgp4fix for divide by zero for xincp = 180 deg
			if (fabs(cosip+1.0) > 1.5e-12)
				satrec[tid].xlcof = -0.25 * gravity_constants.j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
			else
				satrec[tid].xlcof = -0.25 * gravity_constants.j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cos(CUDART_PI-1.0e-9));
		}
		//axnl = ep * cos(argpp);
		//temp = 1.0 / (am * (1.0 - ep * ep));
		//aynl = ep* sin(argpp) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof;
		xl   = mp + argpm + nodep + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].xlcof * (ep * cos(argpm));

		/* --------------------- solve kepler's equation --------------- */
		u    = fmod(xl - nodep, (t_var)(2.0 * CUDART_PI));
		eo1  = u;
		tem5 = 9999.9;
		ktr = 1;
		//   sgp4fix for kepler iteration
		//   the following iteration needs better limits on corrections
		while (( fabs(tem5) >= 1.0e-12) && (ktr <= 10) )
		{
			sineo1 = sin(eo1);
			coseo1 = cos(eo1);
			tem5   = 1.0 - coseo1 * (ep * cos(argpm)) - sineo1 * (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof);
			tem5   = (u - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) * coseo1 + (ep * cos(argpm)) * sineo1 - eo1) / tem5;
			if(fabs(tem5) >= 0.95)

				tem5 = tem5 > 0.0 ? 0.95 : -0.95;
			eo1    = eo1 + tem5;
			//ktr = ktr + 1;
			ktr++;
		}

		/* ------------- short period preliminary quantities ----------- */
		//ecose = (ep * cos(argpp))*coseo1 + (ep* sin(argpp) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof)*sineo1;
		esine = (ep * cos(argpm))*sineo1 - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof)*coseo1;
		el2   = (ep * cos(argpm))*(ep * cos(argpm)) + (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof)*(ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof);
		pl    = am*(1.0-el2);
		if (pl < 0.0)
		{
			//         printf("# error pl %f\n", pl);
			satrec[tid].error = 4;
		}
		else
		{
			rl     = am * (1.0 - (ep * cos(argpm))*coseo1 + (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof)*sineo1);
			//rdotl  = sqrt(am) * esine/rl;
			//rvdotl = sqrt(pl) / rl;
			//betal  = sqrt(1.0 - el2);
			//temp   = esine / (1.0 + betal);
			//sinu   = am / rl * (sineo1 - aynl - axnl * temp);
			//cosu   = am / rl * (coseo1 - axnl + aynl * temp);
			su     = atan2(	am / rl * (sineo1 - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) - (ep * cos(argpm)) * esine / (1.0 + sqrt(1.0 - el2))),
							am / rl * (coseo1 - (ep * cos(argpm)) + (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) * esine / (1.0 + sqrt(1.0 - el2))));
			sin2u  = ( am / rl * (coseo1 - (ep * cos(argpm)) + (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) * esine / (1.0 + sqrt(1.0 - el2))) +  am / rl * (coseo1 - (ep * cos(argpm)) + (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) * esine / (1.0 + sqrt(1.0 - el2)))) * (am / rl * (sineo1 - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) - (ep * cos(argpm)) * esine / (1.0 + sqrt(1.0 - el2))));
			cos2u  = 1.0 - 2.0 * (am / rl * (sineo1 - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) - (ep * cos(argpm)) * esine / (1.0 + sqrt(1.0 - el2)))) * (am / rl * (sineo1 - (ep* sin(argpm) + (1.0 / (am * (1.0 - ep * ep))) * satrec[tid].aycof) - (ep * cos(argpm)) * esine / (1.0 + sqrt(1.0 - el2))));
			//temp   = 1.0 / pl;
			//temp1  = 0.5 * gravity_constants.j2 * (1.0 / pl);
			//temp2  = (0.5 * gravity_constants.j2 * (1.0 / pl)) * (1.0 / pl);
			//betal  = sqrt(1.0 - el2);
			//temp   = esine / (1.0 + betal);
			//sinu   = am / rl * (sineo1 - aynl - axnl * temp);
			//cosu   = am / rl * (coseo1 - axnl + aynl * temp);
			//su     = atan2(sinu, cosu);
			//sin2u  = (cosu + cosu) * sinu;
			//cos2u  = 1.0 - 2.0 * sinu * sinu;
			//temp   = 1.0 / pl;
			//temp1  = 0.5 * gravity_constants.j2 * temp;
			//temp2  = temp1 * temp;

			/* -------------- update for short period periodics ------------ */
			if (satrec[tid].method == 'd')
			{
				cosisq                 = cosip * cosip;
				satrec[tid].con41  = 3.0*cosisq - 1.0;
				satrec[tid].x1mth2 = 1.0 - cosisq;
				satrec[tid].x7thm1 = 7.0*cosisq - 1.0;
			}
			mrt   = rl * (1.0 - 1.5 * ((0.5 * gravity_constants.j2 * (1.0 / pl)) * (1.0 / pl)) * sqrt(1.0 - el2) * satrec[tid].con41) + 0.5 * (0.5 * gravity_constants.j2 * (1.0 / pl)) * satrec[tid].x1mth2 * cos2u;
			su    = su - 0.25 * ((0.5 * gravity_constants.j2 * (1.0 / pl)) * (1.0 / pl)) * satrec[tid].x7thm1 * sin2u;
			xnode = nodep + 1.5 * ((0.5 * gravity_constants.j2 * (1.0 / pl)) * (1.0 / pl)) * cosip * sin2u;
			xinc  = xincp + 1.5 * ((0.5 * gravity_constants.j2 * (1.0 / pl)) * (1.0 / pl)) * cosip * sinip * cos2u;
			//mvt   = rdotl - nm * temp1 * satrec[tid].x1mth2 * sin2u / gravity_constants.xke;
			//rvdot = rvdotl + nm * temp1 * (satrec[tid].x1mth2 * cos2u + 1.5 * satrec[tid].con41) / gravity_constants.xke;

			/* --------------------- orientation vectors ------------------- */
			/*sinsu =  sin(su);
			cossu =  cos(su);
			snod  =  sin(xnode);
			cnod  =  cos(xnode);
			sini  =  sin(xinc);
			cosi  =  cos(xinc);*/
			/*xmx   = -sin(xnode) * cos(xinc);
			xmy   =  cos(xnode) * cos(xinc);*/
			ux    =  (-sin(xnode) * cos(xinc)) * sin(su) + cos(xnode) * cos(su);
			uy    =  (cos(xnode) * cos(xinc)) * sin(su) + sin(xnode) * cos(su);
			uz    =  sin(xinc) * sin(su);
			/*sinsu =  sin(su);
			cossu =  cos(su);
			snod  =  sin(xnode);
			cnod  =  cos(xnode);
			sini  =  sin(xinc);
			cosi  =  cos(xinc);
			xmx   = -snod * cosi;
			xmy   =  cnod * cosi;
			ux    =  xmx * sinsu + cnod * cossu;
			uy    =  xmy * sinsu + snod * cossu;
			uz    =  sini * sinsu;*/
			//vx    =  xmx * cossu - cnod * sinsu;
			//vy    =  xmy * cossu - snod * sinsu;
			//vz    =  sini * cossu;

			/* --------- position and velocity (in km and km/sec) ---------- */
			//r[0] = (mrt * ux)* gravity_constants.radiusearthkm;
			//r[1] = (mrt * uy)* gravity_constants.radiusearthkm;
			//r[2] = (mrt * uz)* gravity_constants.radiusearthkm;
			//v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
			//v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
			//v[2] = (mvt * uz + rvdot * vz) * vkmpersec;
			//r[tid].x = (mrt * ux)* gravity_constants.radiusearthkm;
			//r[tid].y = (mrt * uy)* gravity_constants.radiusearthkm;
			//r[tid].z = (mrt * uz)* gravity_constants.radiusearthkm;
			//v[tid].x = (mvt * ux + rvdot * vx) * vkmpersec;
			//v[tid].y = (mvt * uy + rvdot * vy) * vkmpersec;
			//v[tid].z = (mvt * uz + rvdot * vz) * vkmpersec;
			r[tid].x = (mrt * ux);
			r[tid].y = (mrt * uy);
			r[tid].z = (mrt * uz);
			r[tid].w = 1.0;
		}  // if pl > 0

		// sgp4fix for decaying satellites
		if (mrt < 1.0)
		{
			//         printf("# decay condition %11.6f \n",mrt);
			satrec[tid].error = 6;
		}

	}
	////#include "debug7.cpp"
	//return satrec[tid].error;
}  // end sgp4

__device__ static void dspace
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
	)
{
	//const double twopi = 2.0 * CUDART_PI;
	int iretn , iret;
	t_var delt, ft, theta, x2li, x2omi, xl, xldot , xnddt, xndt, xomi, g22, g32,
		g44, g52, g54, fasx2, fasx4, fasx6, rptim , step2, stepn , stepp;

	ft    = 0.0;
	fasx2 = 0.13130908;
	fasx4 = 2.8843198;
	fasx6 = 0.37448087;
	g22   = 5.7686396;
	g32   = 0.95240898;
	g44   = 1.8014998;
	g52   = 1.0508330;
	g54   = 4.4108898;
	rptim = 4.37526908801129966e-3; // this equates to 7.29211514668855e-5 rad/sec
	stepp =    720.0;
	stepn =   -720.0;
	step2 = 259200.0;

	/* ----------- calculate deep space resonance effects ----------- */
	dndt   = 0.0;
	theta  = fmod(gsto + tc * rptim, (t_var)(2.0 * CUDART_PI));
	em     = em + dedt * t;

	inclm  = inclm + didt * t;
	argpm  = argpm + domdt * t;
	nodem  = nodem + dnodt * t;
	mm     = mm + dmdt * t;

	//   sgp4fix for negative inclinations
	//   the following if statement should be commented out
	//  if (inclm < 0.0)
	// {
	//    inclm = -inclm;
	//    argpm = argpm - pi;
	//    nodem = nodem + pi;
	//  }

	/* - update resonances : numerical (euler-maclaurin) integration - */
	/* ------------------------- epoch restart ----------------------  */
	//   sgp4fix for propagator problems
	//   the following integration works for negative time steps and periods
	//   the specific changes are unknown because the original code was so convoluted

	ft    = 0.0;
	atime = 0.0;
	if (irez != 0)
	{
		if ((atime == 0.0) || ((t >= 0.0) && (atime < 0.0)) ||
			((t < 0.0) && (atime >= 0.0)))
		{
			if (t >= 0.0)
				delt = stepp;
			else
				delt = stepn;
			atime  = 0.0;
			xni    = no;
			xli    = xlamo;
		}
		iretn = 381; // added for do loop
		iret  =   0; // added for loop
		while (iretn == 381)
		{
			if ((fabs(t) < fabs(atime)) || (iret == 351))
			{
				if (t >= 0.0)
					delt = stepn;
				else
					delt = stepp;
				iret  = 351;
				iretn = 381;
			}
			else
			{
				if (t > 0.0)  // error if prev if has atime:=0.0 and t:=0.0 (ge)
					delt = stepp;
				else
					delt = stepn;
				if (fabs(t - atime) >= stepp)
				{
					iret  = 0;
					iretn = 381;
				}
				else
				{
					ft    = t - atime;
					iretn = 0;
				}
			}

			/* ------------------- dot terms calculated ------------- */
			/* ----------- near - synchronous resonance terms ------- */
			if (irez != 2)
			{
				xndt  = del1 * sin(xli - fasx2) + del2 * sin(2.0 * (xli - fasx4)) +
					del3 * sin(3.0 * (xli - fasx6));
				xldot = xni + xfact;
				xnddt = del1 * cos(xli - fasx2) +
					2.0 * del2 * cos(2.0 * (xli - fasx4)) +
					3.0 * del3 * cos(3.0 * (xli - fasx6));
				xnddt = xnddt * xldot;
			}
			else
			{
				/* --------- near - half-day resonance terms -------- */
				xomi  = argpo + argpdot * atime;
				x2omi = xomi + xomi;
				x2li  = xli + xli;
				xndt  = d2201 * sin(x2omi + xli - g22) + d2211 * sin(xli - g22) +
					d3210 * sin(xomi + xli - g32)  + d3222 * sin(-xomi + xli - g32)+
					d4410 * sin(x2omi + x2li - g44)+ d4422 * sin(x2li - g44) +
					d5220 * sin(xomi + xli - g52)  + d5232 * sin(-xomi + xli - g52)+
					d5421 * sin(xomi + x2li - g54) + d5433 * sin(-xomi + x2li - g54);
				xldot = xni + xfact;
				xnddt = d2201 * cos(x2omi + xli - g22) + d2211 * cos(xli - g22) +
					d3210 * cos(xomi + xli - g32) + d3222 * cos(-xomi + xli - g32) +
					d5220 * cos(xomi + xli - g52) + d5232 * cos(-xomi + xli - g52) +
					2.0 * (d4410 * cos(x2omi + x2li - g44) +
					d4422 * cos(x2li - g44) + d5421 * cos(xomi + x2li - g54) +
					d5433 * cos(-xomi + x2li - g54));
				xnddt = xnddt * xldot;
			}

			/* ----------------------- integrator ------------------- */
			if (iretn == 381)
			{
				xli   = xli + xldot * delt + xndt * step2;
				xni   = xni + xndt * delt + xnddt * step2;
				atime = atime + delt;
			}
		}  // while iretn = 381

		nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
		xl = xli + xldot * ft + xndt * ft * ft * 0.5;
		if (irez != 1)
		{
			mm   = xl - 2.0 * nodem + 2.0 * theta;
			dndt = nm - no;
		}
		else
		{
			mm   = xl - nodem - argpm + theta;
			dndt = nm - no;
		}
		nm = no + dndt;
	}

	//#include "debug4.cpp"
}  // end dsspace
