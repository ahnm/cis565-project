#ifndef TLE_H
#define TLE_H

#include "common.h"
#include "satelliterecord.h"

void twolineelement2rv(std::ifstream &tle_read, std::vector<satelliterecord_t> &SatRec);

#endif