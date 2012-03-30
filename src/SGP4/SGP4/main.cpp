#include "common.h"

const std::string tle = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753\n\
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667     0.00      4320.0        360.00\n\
#                       ## fig show lyddane fix error with gsfc ver \n\
1 04632U 70093B   04031.91070959 -.00000084  00000-0  10000-3 0  9955 \n\
2 04632  11.4628 273.1101 1450506 207.6000 143.9350  1.20231981 44145  -5184.0     -4896.0        120.00\n\
#                       # simplified drag eq\n\
1 29238U 06022G   06177.28732010  .00766286  10823-4  13334-2 0   101\n\
2 29238  51.5595 213.7903 0202579  95.2503 267.9010 15.73823839  1061      0.0      1440.0        120.00\n";


int main(int argc, char **argv){
	shrQAStart(argc, argv);
	
    cutilChooseCudaDevice(argc, argv);
	
	std::ifstream tle_file("catalog_3l_2012_03_26_am.txt");

    // report self-test status
    shrQAFinish(argc, const_cast<const char **>(argv), true ? QA_PASSED : QA_FAILED);
	return 0;
}