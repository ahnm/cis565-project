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

			t_var epochdayofyear;
			epochdayofyear = StringToNumber<t_var>(epoch_day_str);

			int month, day, hour, minute;
			t_var second, jd;

			days2mdhms( epochyr, epochdayofyear, month, day, hour, minute, second );
			jday(epochyr, month, day, hour, minute, second, jd);
			SatRec[SatRec.size()-1].epoch_jd = jd;
			/* End */

			SatRec[SatRec.size()-1].n_dt = StringToNumber<t_var>(mean_motion_dt_str) / (xpdotp * 1440.0);
			SatRec[SatRec.size()-1].n_ddt = StringExpToNumber<t_var>(mean_motion_ddt_str.insert(0,".")) / (xpdotp * 1440.0 * 1440.0);
			SatRec[SatRec.size()-1].bstar = StringExpToNumber<t_var>(bstar_str.insert(0,"."));


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
			
			SatRec[SatRec.size()-1].e		=	StringToNumber<t_var>(eccentricity_str.insert(0,"."));
			SatRec[SatRec.size()-1].i		=	deg2rad<t_var>(StringToNumber<t_var>(inclination_str));
			SatRec[SatRec.size()-1].raan	=	deg2rad<t_var>(StringToNumber<t_var>(raan_str));
			SatRec[SatRec.size()-1].w		=	deg2rad<t_var>(StringToNumber<t_var>(argument_of_perigee_str));
			SatRec[SatRec.size()-1].M		=	deg2rad<t_var>(StringToNumber<t_var>(mean_anomaly_str));
			SatRec[SatRec.size()-1].n		=	StringToNumber<t_var>(mean_motion_str) / xpdotp;
			SatRec[SatRec.size()-1].a		=	pow( SatRec[SatRec.size()-1].n * tumin, (t_var)(-2.0/3.0) );
		}
	}
}

void satelliteRecordConvert(std::vector<satelliterecord_aos_t> &SatRecAoS, satelliterecord_soa_t *SatRecSoA){
	size_t numSatellites = SatRecAoS.size();
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