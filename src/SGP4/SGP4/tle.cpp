#include "tle.h"
#include "satelliterecord.h"
#include "constants.h"
#include "functions.h"

void twolineelement2rv(std::ifstream &tle_read, std::vector<satelliterecord_aos_t> &SatRec){

	std::string line;
	while(!tle_read.eof()){
		std::getline(tle_read, line);
		std::stringstream ss(std::stringstream::in | std::stringstream::out);
		std::string line_num;
		ss << line;
		ss >> std::setw(1) >> line_num;

		if(line_num.compare("1") == 0){
			satelliterecord_aos_t newSatelliteRecord;
			SatRec.push_back(newSatelliteRecord);
			std::string satellite_num_str, classification_str, launch_year_str,
				launch_num_str, launch_piece_str, epoch_year_str, epoch_day_str,
				mean_motion_dt_str, mean_motion_ddt_str, bstar_str,
				ephemeris_type_str, element_number_str, checksum_str;

			ss	>>	std::setw(5)	>>	satellite_num_str
				>>	std::setw(1)	>>	classification_str
				>>	std::setw(2)	>>	launch_year_str
				>>	std::setw(3)	>>	launch_num_str
				>>	std::setw(3)	>>	launch_piece_str
				>>	std::setw(2)	>>	epoch_year_str
				>>	std::setw(12)	>>	epoch_day_str
				>>	std::setw(10)	>>	mean_motion_dt_str
				>>	std::setw(8)	>>	mean_motion_ddt_str
				>>	std::setw(8)	>>	bstar_str
				>>	std::setw(1)	>>	ephemeris_type_str
				>>	std::setw(4)	>>	element_number_str
				>>	checksum_str;
			
			SatRec[SatRec.size()-1].satellite_num = StringToNumber<int>(satellite_num_str);
			
			/* Determine Julian Day */
			/* Begin */
			int epochyr;
			epochyr = StringToNumber<int>(epoch_year_str);
			epochyr = ( epochyr < 57 ) ? epochyr + 2000 : epochyr + 1900;

			double epochdayofyear;
			epochdayofyear = StringToNumber<double>(epoch_day_str);

			int month, day, hour, minute;
			double second, jd;

			days2mdhms( epochyr, epochdayofyear, month, day, hour, minute, second );
			jday(epochyr, month, day, hour, minute, second, jd);
			SatRec[SatRec.size()-1].epoch_jd = jd;
			/* End */

			SatRec[SatRec.size()-1].n_dt = StringToNumber<double>(mean_motion_dt_str) / (xpdotp * 1440.0);
			SatRec[SatRec.size()-1].n_ddt = StringExpToNumber<double>(mean_motion_ddt_str.insert(0,".")) / (xpdotp * 1440.0 * 1440.0);
			SatRec[SatRec.size()-1].bstar = StringExpToNumber<double>(bstar_str.insert(0,"."));


		}else if(line_num.compare("2") == 0){
			std::string satellite_num_str, inclination_str, raan_str, eccentricity_str,
				argument_of_perigee_str, mean_anomaly_str, mean_motion_str,
				revs_str, checksum_str;

			ss	>>	std::setw(5)	>>	satellite_num_str
				>>	std::setw(8)	>>	inclination_str
				>>	std::setw(8)	>>	raan_str
				>>	std::setw(7)	>>	eccentricity_str
				>>	std::setw(8)	>>	argument_of_perigee_str
				>>	std::setw(8)	>>	mean_anomaly_str
				>>	std::setw(11)	>>	mean_motion_str
				>>	std::setw(5)	>>	revs_str
				>>	checksum_str;

			int satellite_num = StringToNumber<int>(satellite_num_str);
			if(SatRec[SatRec.size()-1].satellite_num != satellite_num){
				SatRec.pop_back();
				continue;
			}
			
			SatRec[SatRec.size()-1].e		=	StringToNumber<double>(eccentricity_str.insert(0,"."));
			SatRec[SatRec.size()-1].i		=	deg2rad<double>(StringToNumber<double>(inclination_str));
			SatRec[SatRec.size()-1].raan	=	deg2rad<double>(StringToNumber<double>(raan_str));
			SatRec[SatRec.size()-1].w		=	deg2rad<double>(StringToNumber<double>(argument_of_perigee_str));
			SatRec[SatRec.size()-1].M		=	deg2rad<double>(StringToNumber<double>(mean_anomaly_str));
			SatRec[SatRec.size()-1].n		=	StringToNumber<double>(mean_motion_str) / xpdotp;
			SatRec[SatRec.size()-1].a		=	pow( SatRec[SatRec.size()-1].n * tumin, (-2.0/3.0) );
		}
	}
}

void satelliteRecordConvert(std::vector<satelliterecord_aos_t> &SatRecAoS, satelliterecord_soa_t *SatRecSoA){
	size_t numSatellites = SatRecAoS.size();
	//SatRecSoA.satnum			=	new long int[numSatellites];
	////SatRecSoA.epochyr			=	new int[numSatellites];
	////SatRecSoA.epochtynumrev		=	new int[numSatellites];
	//SatRecSoA.error				=	new int[numSatellites];
	//SatRecSoA.init				=	new char[numSatellites];
	//SatRecSoA.method			=	new char[numSatellites];

	//SatRecSoA.isimp				=	new int[numSatellites];
	//SatRecSoA.aycof				=	new double[numSatellites];
	//SatRecSoA.con41				=	new double[numSatellites];
	//SatRecSoA.cc1				=	new double[numSatellites];
	//SatRecSoA.cc4				=	new double[numSatellites];
	//SatRecSoA.cc5				=	new double[numSatellites];
	//SatRecSoA.d2				=	new double[numSatellites];
	//SatRecSoA.d3				=	new double[numSatellites];
	//SatRecSoA.d4				=	new double[numSatellites];
	//SatRecSoA.delmo				=	new double[numSatellites];
	//SatRecSoA.eta				=	new double[numSatellites];
	//SatRecSoA.argpdot			=	new double[numSatellites];
	//SatRecSoA.omgcof			=	new double[numSatellites];
	//SatRecSoA.sinmao			=	new double[numSatellites];
	//SatRecSoA.t					=	new double[numSatellites];
	//SatRecSoA.t2cof				=	new double[numSatellites];
	//SatRecSoA.t3cof				=	new double[numSatellites];
	//SatRecSoA.t4cof				=	new double[numSatellites];
	//SatRecSoA.t5cof				=	new double[numSatellites];
	//SatRecSoA.x1mth2			=	new double[numSatellites];
	//SatRecSoA.x7thm1			=	new double[numSatellites];
	//SatRecSoA.mdot				=	new double[numSatellites];
	//SatRecSoA.nodedot			=	new double[numSatellites];
	//SatRecSoA.xlcof				=	new double[numSatellites];
	//SatRecSoA.xmcof				=	new double[numSatellites];
	//SatRecSoA.nodecf			=	new double[numSatellites];

	//SatRecSoA.irez			=	new int[numSatellites];
	//SatRecSoA.d2201			=	new double[numSatellites];
	//SatRecSoA.d2211			=	new double[numSatellites];
	//SatRecSoA.d3210 		=	new double[numSatellites];
	//SatRecSoA.d3222			=	new double[numSatellites];
	//SatRecSoA.d4410			=	new double[numSatellites];
	//SatRecSoA.d4422			=	new double[numSatellites];
	//SatRecSoA.d5220			=	new double[numSatellites];
	//SatRecSoA.d5232			=	new double[numSatellites];
	//SatRecSoA.d5421			=	new double[numSatellites];
	//SatRecSoA.d5433			=	new double[numSatellites];
	//SatRecSoA.dedt			=	new double[numSatellites];
	//SatRecSoA.del1			=	new double[numSatellites];
	//SatRecSoA.del2			=	new double[numSatellites];
	//SatRecSoA.del3			=	new double[numSatellites];
	//SatRecSoA.didt			=	new double[numSatellites];
	//SatRecSoA.dmdt			=	new double[numSatellites];
	//SatRecSoA.dnodt			=	new double[numSatellites];
	//SatRecSoA.domdt			=	new double[numSatellites];
	//SatRecSoA.e3			=	new double[numSatellites];
	//SatRecSoA.ee2			=	new double[numSatellites];
	//SatRecSoA.peo			=	new double[numSatellites];
	//SatRecSoA.pgho			=	new double[numSatellites];
	//SatRecSoA.pho			=	new double[numSatellites];
	//SatRecSoA.pinco			=	new double[numSatellites];
	//SatRecSoA.plo			=	new double[numSatellites];
	//SatRecSoA.se2			=	new double[numSatellites];
	//SatRecSoA.se3			=	new double[numSatellites];
	//SatRecSoA.sgh2			=	new double[numSatellites];
	//SatRecSoA.sgh3			=	new double[numSatellites];
	//SatRecSoA.sgh4			=	new double[numSatellites];
	//SatRecSoA.sh2			=	new double[numSatellites];
	//SatRecSoA.sh3			=	new double[numSatellites];
	//SatRecSoA.si2			=	new double[numSatellites];
	//SatRecSoA.si3			=	new double[numSatellites];
	//SatRecSoA.sl2			=	new double[numSatellites];
	//SatRecSoA.sl3			=	new double[numSatellites];
	//SatRecSoA.sl4			=	new double[numSatellites];
	//SatRecSoA.gsto			=	new double[numSatellites];
	//SatRecSoA.xfact			=	new double[numSatellites];
	//SatRecSoA.xgh2			=	new double[numSatellites];
	//SatRecSoA.xgh3			=	new double[numSatellites];
	//SatRecSoA.xgh4			=	new double[numSatellites];
	//SatRecSoA.xh2			=	new double[numSatellites];
	//SatRecSoA.xh3			=	new double[numSatellites];
	//SatRecSoA.xi2			=	new double[numSatellites];
	//SatRecSoA.xi3			=	new double[numSatellites];
	//SatRecSoA.xl2			=	new double[numSatellites];
	//SatRecSoA.xl3			=	new double[numSatellites];
	//SatRecSoA.xl4			=	new double[numSatellites];
	//SatRecSoA.xlamo			=	new double[numSatellites];
	//SatRecSoA.zmol			=	new double[numSatellites];
	//SatRecSoA.zmos			=	new double[numSatellites];
	//SatRecSoA.atime			=	new double[numSatellites];
	//SatRecSoA.xli			=	new double[numSatellites];
	//SatRecSoA.xni			=	new double[numSatellites];

	//SatRecSoA.a				=	new double[numSatellites];
	//SatRecSoA.altp			=	new double[numSatellites];
	//SatRecSoA.alta			=	new double[numSatellites];
	//SatRecSoA.epochdays		=	new double[numSatellites];
	//SatRecSoA.jdsatepoch	=	new double[numSatellites];
	//SatRecSoA.nddot			=	new double[numSatellites];
	//SatRecSoA.ndot			=	new double[numSatellites];
	//SatRecSoA.bstar			=	new double[numSatellites];
	//SatRecSoA.rcse			=	new double[numSatellites];
	//SatRecSoA.inclo			=	new double[numSatellites];
	//SatRecSoA.nodeo			=	new double[numSatellites];
	//SatRecSoA.ecco			=	new double[numSatellites];
	//SatRecSoA.argpo			=	new double[numSatellites];
	//SatRecSoA.mo			=	new double[numSatellites];
	//SatRecSoA.no			=	new double[numSatellites];

	for(int i = 0; i < SatRecAoS.size(); i++){
		SatRecSoA[i].satnum	=	SatRecAoS[i].satellite_num;
		SatRecSoA[i].a		=	SatRecAoS[i].a;
		SatRecSoA[i].ecco	=	SatRecAoS[i].e;
		SatRecSoA[i].inclo	=	SatRecAoS[i].i;
		SatRecSoA[i].nodeo	=	SatRecAoS[i].raan;
		SatRecSoA[i].argpo	=	SatRecAoS[i].w;
		SatRecSoA[i].mo		=	SatRecAoS[i].M;

		SatRecSoA[i].no		=	SatRecAoS[i].n;
		SatRecSoA[i].ndot	=	SatRecAoS[i].n_dt;
		SatRecSoA[i].nddot	=	SatRecAoS[i].n_ddt;

		SatRecSoA[i].bstar	=	SatRecAoS[i].bstar;

		SatRecSoA[i].alta	=	SatRecAoS[i].a * (1.0 + SatRecAoS[i].e);
		SatRecSoA[i].altp	=	SatRecAoS[i].a * (1.0 - SatRecAoS[i].e);

		SatRecSoA[i].jdsatepoch	=	SatRecAoS[i].epoch_jd;

	}
}