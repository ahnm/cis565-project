#ifndef SATELLITE_RECORD
#define SATELLITE_RECORD

#include "common.h"

struct satelliterecord_aos_t{
	int		satellite_num;
	
	t_var	a,				//semi-major axis
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

typedef struct satelliterecord_soa_t
{
  long int		satnum;
  //int			epochyr, epochtynumrev;
  int			error;
  char			init, method;

  /* Near Earth */
  int			isimp;
  t_var		aycof  , con41  , cc1    , cc4      , cc5    , d2      , d3   , d4    ,
				delmo  , eta    , argpdot, omgcof   , sinmao , t       , t2cof, t3cof ,
				t4cof  , t5cof  , x1mth2 , x7thm1   , mdot   , nodedot, xlcof , xmcof ,
				nodecf;

  /* Deep Space */
  int			irez;
  t_var		d2201  , d2211  , d3210  , d3222    , d4410  , d4422   , d5220 , d5232 ,
				d5421  , d5433  , dedt   , del1     , del2   , del3    , didt  , dmdt  ,
				dnodt  , domdt  , e3     , ee2      , peo    , pgho    , pho   , pinco ,
				plo    , se2    , se3    , sgh2     , sgh3   , sgh4    , sh2   , sh3   ,
				si2    , si3    , sl2    , sl3      , sl4    , gsto    , xfact , xgh2  ,
				xgh3   , xgh4   , xh2    , xh3      , xi2    , xi3     , xl2   , xl3   ,
				xl4    , xlamo  , zmol   , zmos     , atime  , xli     , xni;

  t_var		a      , altp   , alta   , epochdays, jdsatepoch       , nddot , ndot  ,
				bstar  , rcse   , inclo  , nodeo    , ecco             , argpo , mo    ,
				no;
} elsetrec;

// Template structure to pass to kernel
struct SatelliteRecordArray
{
    struct satelliterecord_soa_t _array;
    int _size;
};

#endif