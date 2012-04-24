#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "common.h"

template <typename T>
T deg2rad(T deg){
	return (deg * PI / 180.0);
};

template <typename T>
T StringToNumber ( const std::string &Text )
{
	std::istringstream ss(Text);
	T result;
	return ss >> result ? result : 0;
};

template <typename T>
T StringExpToNumber ( const std::string &Text )
{
	std::string coeff_str = Text.substr(0,Text.length() - 2);
	std::string exp_str = Text.substr(Text.length() - 2,2);
	std::stringstream ss;
	ss << coeff_str << exp_str;
	T coeff, exp, result;
	ss >> coeff >> exp;
	result = coeff * pow(10, exp);
	return result;
};

void days2mdhms( int year, t_var days,	int& mon, int& day, int& hr, int& minute, t_var& sec );

void jday(int year, int mon, int day, int hr, int minute, t_var sec, t_var& jd);

#endif