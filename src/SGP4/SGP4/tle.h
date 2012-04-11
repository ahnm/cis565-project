#ifndef TLE_H
#define TLE_H

#include "common.h"
#include "satelliterecord.h"

void twolineelement2rv(std::ifstream &tle_read, std::vector<satelliterecord_aos_t> &SatRec);

void satelliteRecordConvert(std::vector<satelliterecord_aos_t> &SatRecAoS, satelliterecord_soa_t *SatRecSoA);

#endif