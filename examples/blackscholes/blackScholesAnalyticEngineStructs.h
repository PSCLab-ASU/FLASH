#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
//#include <cuda.h>

//blackScholesAnalyticEngineStructs.cuh
//Scott Grauer-Gray
//Structs for running black scholes using the analytic engine (from quantlib) on the GPU

#ifndef BLACK_SCHOLES_ANALYTIC_ENGINE_STRUCTS_CUH
#define BLACK_SCHOLES_ANALYTIC_ENGINE_STRUCTS_CUH

//define the total number of samples
#define NUM_SAMPLES_BLACK_SCHOLES_ANALYTIC 200000

//define the thread block size
#define THREAD_BLOCK_SIZE 256

#define NUM_DIFF_SETTINGS 37

typedef struct
{
	int day;
	int month;
	int year;
} dateStruct;


typedef struct
{
	int type;
	float strike;
} payoffStruct;


typedef struct
{
	float typeExercise;
	float yearFractionTime;
} exerciseStruct;

typedef struct
{
	float rate;
	float freq;
	int comp;
} interestRateStruct;

typedef struct
{
	float timeYearFraction;
	float forward;
	float compounding;
	float frequency;
	float intRate;
} yieldTermStruct;

typedef struct
{
	float timeYearFraction;
	float following;
	float volatility;
} blackVolStruct;

typedef struct
{
	float x0;
	yieldTermStruct dividendTS;
	yieldTermStruct riskFreeTS;
	blackVolStruct blackVolTS;
} blackScholesMertStruct;

typedef struct
{
	blackScholesMertStruct process;
	float tGrid;
	float xGrid;
	float dampingSteps;
	float schemeDesc;
	float localVol;
} engineStruct;


typedef struct
{
	payoffStruct payoff;
	float yearFractionTime;
	blackScholesMertStruct pricingEngine;
} optionStruct;

typedef struct
{
	float strike;
	float forward;
	float stdDev;
	float discount;
	float variance;
	float d1;
	float d2;
	float alpha;
	float beta;
	float DalphaDd1;
	float DbetaDd2;
	float n_d1;
	float cum_d1;
	float n_d2;
	float cum_d2;
	float x;
	float DxDs;
	float DxDstrike;
} blackCalcStruct;

typedef struct
{
	float average;
	float sigma;
	float denominator;
	float derNormalizationFactor;
	float normalizationFactor;
} normalDistStruct;

//define into for each type of option
#define CALL 0
#define PUT 1

typedef struct
{ 
	int type;
	float strike;
	float spot;
	float q;
	float r;
	float t;
	float vol;
	float value;
	float tol;
} optionInputStruct;

inline void prepare_input( std::vector<optionInputStruct>& input, int numVals ) 
{
  for (int numOption = 0; numOption < numVals; numOption++)
    {
      if ((numOption % NUM_DIFF_SETTINGS) == 0)
        input.push_back({ CALL,  40.00,  42.00, 0.08, 0.04, 0.75, 0.35,  5.0975, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 1)
        input.push_back({ CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  0.0205, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 2)
        input.push_back({ CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 3)
        input.push_back({ CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  9.9413, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 4)
        input.push_back({ CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25,  0.3150, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 5)
        input.push_back({ CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 6)
        input.push_back({ CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25, 10.3556, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 7)
        input.push_back( { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35,  0.9474, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 8)
        input.push_back({ CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 9)
        input.push_back({ CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35, 11.1381, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 10)
        input.push_back( { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15,  0.8069, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 11)
        input.push_back( { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 12)
        input.push_back( { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15, 10.5769, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 13)
        input.push_back(  { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25,  2.7026, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 14)
        input.push_back(  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 15)
        input.push_back(  { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25, 12.7857, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 16)
        input.push_back(  { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35,  4.9329, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 17)
        input.push_back( { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 18)
        input.push_back({ CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35, 15.3086, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 19)
        input.push_back( { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  9.9210, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 20)
        input.push_back(  { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 21)
        input.push_back(  { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  0.0408, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 22)
        input.push_back( { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25, 10.2155, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 23)
        input.push_back(  { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 24)
        input.push_back(   { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25,  0.4551, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 25)
        input.push_back( { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35, 10.8479, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 26)
        input.push_back(  { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 27)
        input.push_back( { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35,  1.2376, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 28)
        input.push_back( { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15, 10.3192, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 29)
        input.push_back(  { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 30)
        input.push_back( { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15,  1.0646, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 31)
        input.push_back( { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25, 12.2149, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 32)
        input.push_back(  { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 33)
        input.push_back(  { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25,  3.2734, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 34)
        input.push_back(  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35, 14.4452, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 35)
        input.push_back( { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4});
      if ((numOption % NUM_DIFF_SETTINGS) == 36)
        input.push_back(  { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35,  5.7963, 1.0e-4});
    }
}

#endif //BLACK_SCHOLES_ANALYTIC_ENGINE_STRUCTS_CUH
