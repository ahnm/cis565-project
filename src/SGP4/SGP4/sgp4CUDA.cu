//#include "sgp4CUDA.h"
//
//void test(){
//	
//	thrust::host_vector<satellite_record> satrec;
//	std::ifstream tle_file("catalog_3l_2012_03_26_am.txt");
//	//std::string tle((std::istreambuf_iterator<char>(tle_file)), std::istreambuf_iterator<char>());
//	startTime();
//	twolineelement2rv(tle_file, satrec);
//	std::cout << calculateElapsedTime() << std::endl;
//	//cout << satrec[0].satellite_num << endl;
//	//cout << satrec[1].satellite_num << endl;
//
//	//cout << sizeof(satrec) << endl;
//	for(int i = 0; i < satrec.size(); i++){
//		std::cout << satrec[i].satellite_num << std::endl;
//	}
//
//	thrust::device_vector<satellite_record> satrec_device = satrec;
//	thrust::sort(satrec_device.begin(), satrec_device.end(), SRCmpOrbitalPeriod());
//
//
//
//}