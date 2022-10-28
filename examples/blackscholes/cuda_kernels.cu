#include <blackScholesAnalyticEngineStructs.h>


//constants used in this code
#define M_1_SQRTPI  0.564189583547756286948f
#define M_SQRT_2    0.7071067811865475244008443621048490392848359376887f

#ifndef M_1_PI
    #define M_1_PI      0.318309886183790671538f
#endif

#ifndef M_2_PI
    #define M_2_PI      0.636619772367581343076f
#endif

#ifndef M_1_SQRTPI
    #define M_1_SQRTPI  0.564189583547756286948f
#endif

#ifndef M_2_SQRTPI
    #define M_2_SQRTPI  1.12837916709551257390f
#endif

#ifndef M_SQRT2
    #define M_SQRT2     1.41421356237309504880f
#endif

#ifndef M_SQRT_2
    #define M_SQRT_2    0.7071067811865475244008443621048490392848359376887f
#endif

#ifndef M_SQRT1_2
    #define M_SQRT1_2   0.7071067811865475244008443621048490392848359376887f
#endif


#define ERROR_FUNCT_tiny 0.000000000000000000001f //QL_EPSILON
#define ERROR_FUNCT_one 1.00000000000000000000e+00f
        /* c = (float)0.84506291151 */
#define ERROR_FUNCT_erx 8.45062911510467529297e-01f /* 0x3FEB0AC1, 0x60000000 */
        //
        // Coefficients for approximation to  erf on [0,0.84375]
        //
#define ERROR_FUNCT_efx  1.28379167095512586316e-01f /* 0x3FC06EBA, 0x8214DB69 */
#define ERROR_FUNCT_efx8 1.02703333676410069053e+00f /* 0x3FF06EBA, 0x8214DB69 */
#define ERROR_FUNCT_pp0 1.28379167095512558561e-01f /* 0x3FC06EBA, 0x8214DB68 */
#define ERROR_FUNCT_pp1 -3.25042107247001499370e-01f /* 0xBFD4CD7D, 0x691CB913 */
#define ERROR_FUNCT_pp2 -2.84817495755985104766e-02f /* 0xBF9D2A51, 0xDBD7194F */
#define ERROR_FUNCT_pp3 -5.77027029648944159157e-03f /* 0xBF77A291, 0x236668E4 */
#define ERROR_FUNCT_pp4 -2.37630166566501626084e-05f /* 0xBEF8EAD6, 0x120016AC */
#define ERROR_FUNCT_qq1 3.97917223959155352819e-01f /* 0x3FD97779, 0xCDDADC09 */
#define ERROR_FUNCT_qq2 6.50222499887672944485e-02f /* 0x3FB0A54C, 0x5536CEBA */
#define ERROR_FUNCT_qq3 5.08130628187576562776e-03f /* 0x3F74D022, 0xC4D36B0F */
#define ERROR_FUNCT_qq4 1.32494738004321644526e-04f /* 0x3F215DC9, 0x221C1A10 */
#define ERROR_FUNCT_qq5 -3.96022827877536812320e-06f /* 0xBED09C43, 0x42A26120 */
        //
        // Coefficients for approximation to  erf  in [0.84375,1.25]
        //
#define ERROR_FUNCT_pa0 -2.36211856075265944077e-03f /* 0xBF6359B8, 0xBEF77538 */
#define ERROR_FUNCT_pa1 4.14856118683748331666e-01f /* 0x3FDA8D00, 0xAD92B34D */
#define ERROR_FUNCT_pa2 -3.72207876035701323847e-01f /* 0xBFD7D240, 0xFBB8C3F1 */
#define ERROR_FUNCT_pa3 3.18346619901161753674e-01f /* 0x3FD45FCA, 0x805120E4 */
#define ERROR_FUNCT_pa4 -1.10894694282396677476e-01f /* 0xBFBC6398, 0x3D3E28EC */
#define ERROR_FUNCT_pa5 3.54783043256182359371e-02f /* 0x3FA22A36, 0x599795EB */
#define ERROR_FUNCT_pa6 -2.16637559486879084300e-03f /* 0xBF61BF38, 0x0A96073F */
#define ERROR_FUNCT_qa1 1.06420880400844228286e-01f /* 0x3FBB3E66, 0x18EEE323 */
#define ERROR_FUNCT_qa2 5.40397917702171048937e-01f /* 0x3FE14AF0, 0x92EB6F33 */
#define ERROR_FUNCT_qa3 7.18286544141962662868e-02f /* 0x3FB2635C, 0xD99FE9A7 */
#define ERROR_FUNCT_qa4 1.26171219808761642112e-01f /* 0x3FC02660, 0xE763351F */
#define ERROR_FUNCT_qa5 1.36370839120290507362e-02f /* 0x3F8BEDC2, 0x6B51DD1C */
#define ERROR_FUNCT_qa6 1.19844998467991074170e-02f /* 0x3F888B54, 0x5735151D */
//
// Coefficients for approximation to  erfc in [1.25,1/0.35]
//
#define ERROR_FUNCT_ra0 -9.86494403484714822705e-03f /* 0xBF843412, 0x600D6435 */
#define ERROR_FUNCT_ra1 -6.93858572707181764372e-01f /* 0xBFE63416, 0xE4BA7360 */
#define ERROR_FUNCT_ra2 -1.05586262253232909814e+01f /* 0xC0251E04, 0x41B0E726 */
#define ERROR_FUNCT_ra3 -6.23753324503260060396e+01f /* 0xC04F300A, 0xE4CBA38D */
#define ERROR_FUNCT_ra4 -1.62396669462573470355e+02f /* 0xC0644CB1, 0x84282266 */
#define ERROR_FUNCT_ra5 -1.84605092906711035994e+02f /* 0xC067135C, 0xEBCCABB2 */
#define ERROR_FUNCT_ra6 -8.12874355063065934246e+01f /* 0xC0545265, 0x57E4D2F2 */
#define ERROR_FUNCT_ra7 -9.81432934416914548592e+00f /* 0xC023A0EF, 0xC69AC25C */
#define ERROR_FUNCT_sa1 1.96512716674392571292e+01f /* 0x4033A6B9, 0xBD707687 */
#define ERROR_FUNCT_sa2 1.37657754143519042600e+02f /* 0x4061350C, 0x526AE721 */
#define ERROR_FUNCT_sa3 4.34565877475229228821e+02f /* 0x407B290D, 0xD58A1A71 */
#define ERROR_FUNCT_sa4 6.45387271733267880336e+02f /* 0x40842B19, 0x21EC2868 */
#define ERROR_FUNCT_sa5 4.29008140027567833386e+02f /* 0x407AD021, 0x57700314 */
#define ERROR_FUNCT_sa6 1.08635005541779435134e+02f /* 0x405B28A3, 0xEE48AE2C */
#define ERROR_FUNCT_sa7 6.57024977031928170135e+00f /* 0x401A47EF, 0x8E484A93 */
#define ERROR_FUNCT_sa8 -6.04244152148580987438e-02f /* 0xBFAEEFF2, 0xEE749A62 */
//
// Coefficients for approximation to  erfc in [1/.35,28]
//
#define ERROR_FUNCT_rb0 -9.86494292470009928597e-03f /* 0xBF843412, 0x39E86F4A */
#define ERROR_FUNCT_rb1 -7.99283237680523006574e-01f /* 0xBFE993BA, 0x70C285DE */
#define ERROR_FUNCT_rb2 -1.77579549177547519889e+01f /* 0xC031C209, 0x555F995A */
#define ERROR_FUNCT_rb3 -1.60636384855821916062e+02f /* 0xC064145D, 0x43C5ED98 */
#define ERROR_FUNCT_rb4 -6.37566443368389627722e+02f /* 0xC083EC88, 0x1375F228 */
#define ERROR_FUNCT_rb5 -1.02509513161107724954e+03f /* 0xC0900461, 0x6A2E5992 */
#define ERROR_FUNCT_rb6 -4.83519191608651397019e+02f /* 0xC07E384E, 0x9BDC383F */
#define ERROR_FUNCT_sb1 3.03380607434824582924e+01f /* 0x403E568B, 0x261D5190 */
#define ERROR_FUNCT_sb2 3.25792512996573918826e+02f /* 0x40745CAE, 0x221B9F0A */
#define ERROR_FUNCT_sb3 1.53672958608443695994e+03f /* 0x409802EB, 0x189D5118 */
#define ERROR_FUNCT_sb4 3.19985821950859553908e+03f /* 0x40A8FFB7, 0x688C246A */
#define ERROR_FUNCT_sb5 2.55305040643316442583e+03f /* 0x40A3F219, 0xCEDF3BE6 */
#define ERROR_FUNCT_sb6 4.74528541206955367215e+02f /* 0x407DA874, 0xE79FE763 */
#define ERROR_FUNCT_sb7 -2.24409524465858183362e+01f /* 0xC03670E2, 0x42712D62 */

//device kernel to retrieve the compound factor in interestRate
__device__ float interestRateCompoundFactor(float t, yieldTermStruct currYieldTermStruct)
{
  return (expf((currYieldTermStruct.forward)*t));
}


//device kernel to retrieve the discount factor in interestRate
__device__ float interestRateDiscountFactor(float t, yieldTermStruct currYieldTermStruct)
{
  return 1.0f / interestRateCompoundFactor(t, currYieldTermStruct);
}


//device function to get the variance of the black volatility function
__device__ float getBlackVolBlackVar(blackVolStruct volTS)
{
  float vol = volTS.volatility;
  return vol*vol*volTS.timeYearFraction;
}


//device function to get the discount on a dividend yield
__device__ float getDiscountOnDividendYield(float yearFraction, yieldTermStruct dividendYieldTermStruct)
{
  float intDiscountFactor = interestRateDiscountFactor(yearFraction, dividendYieldTermStruct);
  return intDiscountFactor;
}


//device function to get the discount on the risk free rate
__device__ float getDiscountOnRiskFreeRate(float yearFraction, yieldTermStruct riskFreeRateYieldTermStruct)
{
  return interestRateDiscountFactor(yearFraction, riskFreeRateYieldTermStruct);
}


//device kernel to run the error function
__device__ float errorFunct(normalDistStruct normDist, float x)
{
  float R,S,P,Q,s,y,z,r, ax;

  ax = fabsf(x);

  if(ax < 0.84375f) 
  {      
    if(ax < 3.7252902984e-09f) 
    { 
      if (ax < FLT_MIN*16.0f)
        return 0.125f*(8.0f*x+ (ERROR_FUNCT_efx8)*x);  /*avoid underflow */
      return x + (ERROR_FUNCT_efx)*x;
    }
    z = x*x;
    r = ERROR_FUNCT_pp0+z*(ERROR_FUNCT_pp1+z*(ERROR_FUNCT_pp2+z*(ERROR_FUNCT_pp3+z*ERROR_FUNCT_pp4)));
    s = ERROR_FUNCT_one+z*(ERROR_FUNCT_qq1+z*(ERROR_FUNCT_qq2+z*(ERROR_FUNCT_qq3+z*(ERROR_FUNCT_qq4+z*ERROR_FUNCT_qq5))));
    y = r/s;
    return x + x*y;
  }
  if(ax <1.25f) 
  {      
    s = ax-ERROR_FUNCT_one;
    P = ERROR_FUNCT_pa0+s*(ERROR_FUNCT_pa1+s*(ERROR_FUNCT_pa2+s*(ERROR_FUNCT_pa3+s*(ERROR_FUNCT_pa4+s*(ERROR_FUNCT_pa5+s*ERROR_FUNCT_pa6)))));
    Q = ERROR_FUNCT_one+s*(ERROR_FUNCT_qa1+s*(ERROR_FUNCT_qa2+s*(ERROR_FUNCT_qa3+s*(ERROR_FUNCT_qa4+s*(ERROR_FUNCT_qa5+s*ERROR_FUNCT_qa6)))));
    if(x>=0.0f) return ERROR_FUNCT_erx + P/Q; else return -1.0f*ERROR_FUNCT_erx - P/Q;
  }
  if (ax >= 6.0f) 
  {      
    if(x>=0.0f) 
      return ERROR_FUNCT_one-ERROR_FUNCT_tiny; 
    else 
      return ERROR_FUNCT_tiny-ERROR_FUNCT_one;
  }

  /* Starts to lose accuracy when ax~5 */
  s = ERROR_FUNCT_one/(ax*ax);

  if(ax < 2.85714285714285f) { /* |x| < 1/0.35 */
    R = ERROR_FUNCT_ra0+s*(ERROR_FUNCT_ra1+s*(ERROR_FUNCT_ra2+s*(ERROR_FUNCT_ra3+s*(ERROR_FUNCT_ra4+s*(ERROR_FUNCT_ra5+s*(ERROR_FUNCT_ra6+s*ERROR_FUNCT_ra7))))));
    S = ERROR_FUNCT_one+s*(ERROR_FUNCT_sa1+s*(ERROR_FUNCT_sa2+s*(ERROR_FUNCT_sa3+s*(ERROR_FUNCT_sa4+s*(ERROR_FUNCT_sa5+s*(ERROR_FUNCT_sa6+s*(ERROR_FUNCT_sa7+s*ERROR_FUNCT_sa8)))))));
  } else {    /* |x| >= 1/0.35 */
    R=ERROR_FUNCT_rb0+s*(ERROR_FUNCT_rb1+s*(ERROR_FUNCT_rb2+s*(ERROR_FUNCT_rb3+s*(ERROR_FUNCT_rb4+s*(ERROR_FUNCT_rb5+s*ERROR_FUNCT_rb6)))));
    S=ERROR_FUNCT_one+s*(ERROR_FUNCT_sb1+s*(ERROR_FUNCT_sb2+s*(ERROR_FUNCT_sb3+s*(ERROR_FUNCT_sb4+s*(ERROR_FUNCT_sb5+s*(ERROR_FUNCT_sb6+s*ERROR_FUNCT_sb7))))));
  }

  r = expf( -ax*ax-0.5625f +R/S);
  if(x>=0.0f) 
    return ERROR_FUNCT_one-r/ax; 
  else 
    return r/ax-ERROR_FUNCT_one;
}



//device kernel to run the operator function in cumulative normal distribution
__device__ float cumNormDistOp(normalDistStruct normDist, float z)
{
  z = (z - normDist.average) / normDist.sigma;
  float result = 0.5f * ( 1.0f + errorFunct(normDist, z*M_SQRT_2 ) );
  return result;
}


//device kernel to run the gaussian function in the normal distribution
__device__ float gaussianFunctNormDist(normalDistStruct normDist, float x)
{
  float deltax = x - normDist.average;
  float exponent = -(deltax*deltax)/normDist.denominator;

  // debian alpha had some strange problem in the very-low range
  return exponent <= -690.0f ? 0.0f :  // exp(x) < 1.0e-300 anyway
    normDist.normalizationFactor * expf(exponent);
}


//device kernel to retrieve the derivative in a cumulative normal distribution
__device__ float cumNormDistDeriv(normalDistStruct normDist, float x)
{
  float xn = (x - normDist.average) / normDist.sigma;
  return gaussianFunctNormDist(normDist, xn) / normDist.sigma;
}


//device function to initialize the cumulative normal distribution structure
__device__ void initCumNormDist(normalDistStruct& currCumNormDist)
{
  currCumNormDist.average = 0.0f;
  currCumNormDist.sigma = 1.0f;
  currCumNormDist.normalizationFactor = M_SQRT_2*M_1_SQRTPI/currCumNormDist.sigma;
  currCumNormDist.derNormalizationFactor = currCumNormDist.sigma*currCumNormDist.sigma;
  currCumNormDist.denominator = 2.0f*currCumNormDist.derNormalizationFactor;
}


//device function to initialize variable in the black calculator
__device__ void initBlackCalcVars(blackCalcStruct& blackCalculator, payoffStruct payoff)
{
  blackCalculator.d1 = log(blackCalculator.forward / blackCalculator.strike)/blackCalculator.stdDev + 0.5f*blackCalculator.stdDev;
  blackCalculator.d2 = blackCalculator.d1 - blackCalculator.stdDev;

  //initialize the cumulative normal distribution structure
  normalDistStruct currCumNormDist;
  initCumNormDist(currCumNormDist);

  blackCalculator.cum_d1 = cumNormDistOp(currCumNormDist, blackCalculator.d1);
  blackCalculator.cum_d2 = cumNormDistOp(currCumNormDist, blackCalculator.d2);
  blackCalculator.n_d1 = cumNormDistDeriv(currCumNormDist, blackCalculator.d1);
  blackCalculator.n_d2 = cumNormDistDeriv(currCumNormDist, blackCalculator.d2);

  blackCalculator.x = payoff.strike;
  blackCalculator.DxDstrike = 1.0f;

  // the following one will probably disappear as soon as
  // super-share will be properly handled
  blackCalculator.DxDs = 0.0f;

  // this part is always executed.
  // in case of plain-vanilla payoffs, it is also the only part
  // which is executed.
  switch (payoff.type) 
  {
    case CALL:
      blackCalculator.alpha     = blackCalculator.cum_d1;//  N(d1)
      blackCalculator.DalphaDd1 = blackCalculator.n_d1;//  n(d1)
      blackCalculator.beta      = -1.0f*blackCalculator.cum_d2;// -N(d2)
      blackCalculator.DbetaDd2  = -1.0f*blackCalculator.n_d2;// -n(d2)
      break;
    case PUT:
      blackCalculator.alpha     = -1.0f+blackCalculator.cum_d1;// -N(-d1)
      blackCalculator.DalphaDd1 = blackCalculator.n_d1;//  n( d1)
      blackCalculator.beta      = 1.0f-blackCalculator.cum_d2;//  N(-d2)
      blackCalculator.DbetaDd2  = -1.0f* blackCalculator.n_d2;// -n( d2)
      break;
  }
}


//device function to initialize the black calculator
__device__ void initBlackCalculator(blackCalcStruct& blackCalc, payoffStruct payoff, float forwardPrice, float stdDev, float riskFreeDiscount)
{
  blackCalc.strike = payoff.strike;
  blackCalc.forward = forwardPrice;
  blackCalc.stdDev = stdDev;
  blackCalc.discount = riskFreeDiscount;
  blackCalc.variance = stdDev * stdDev;

  initBlackCalcVars(blackCalc, payoff);
}


//device function to retrieve the output resulting value
__device__ float getResultVal(blackCalcStruct blackCalculator)
{
  float result = blackCalculator.discount * (blackCalculator.forward * 
      blackCalculator.alpha + blackCalculator.x * blackCalculator.beta);
  return result;
}


//global function to retrieve the output value for an option
__global__ void getOutValOption(int numVals, optionInputStruct* options, float* outputVals)
{
  int optionNum = blockIdx.x * blockDim.x + threadIdx.x;

  //check if within current options
  if (optionNum < numVals)
  {
    optionInputStruct threadOption = options[optionNum];

    payoffStruct currPayoff;
    currPayoff.type = threadOption.type;
    currPayoff.strike = threadOption.strike;

    yieldTermStruct qTS;
    qTS.timeYearFraction = threadOption.t;
    qTS.forward = threadOption.q;

    yieldTermStruct rTS;
    rTS.timeYearFraction = threadOption.t;
    rTS.forward = threadOption.r;

    blackVolStruct volTS;
    volTS.timeYearFraction = threadOption.t;
    volTS.volatility = threadOption.vol;

    blackScholesMertStruct stochProcess;
    stochProcess.x0 = threadOption.spot;
    stochProcess.dividendTS = qTS;
    stochProcess.riskFreeTS = rTS;
    stochProcess.blackVolTS = volTS;

    optionStruct currOption;
    currOption.payoff = currPayoff;
    currOption.yearFractionTime = threadOption.t;
    currOption.pricingEngine = stochProcess; 

    float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
    float dividendDiscount = getDiscountOnDividendYield(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
    float riskFreeDiscount = getDiscountOnRiskFreeRate(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
    float spot = currOption.pricingEngine.x0; 

    float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

    //declare the blackCalcStruct
    blackCalcStruct blackCalc;

    //initialize the calculator
    initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sqrt(variance), riskFreeDiscount);

    //retrieve the results values
    float resultVal = getResultVal(blackCalc);

    //write the resulting value to global memory
    outputVals[optionNum] = resultVal;
  }
}

