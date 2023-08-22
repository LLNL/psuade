// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free 
// Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class OneSampleAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2007
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "OneSampleAnalyzer.h"
#include "TwoSampleAnalyzer.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "psMatrix.h"
#include "psVector.h"
#include "Sampling.h"
#include "PDFManager.h"
#include "PrintingTS.h"

// ************************************************************************
// chi-squqred test table
// ------------------------------------------------------------------------
#define CSMaxSampleSize        100
#define CSMaxSignificanceLevels  5
//**/  Upper 0.1 0.05 0.025 0.01 0.001 (one-sided)
static double CSUTable[CSMaxSampleSize][CSMaxSignificanceLevels] =
{
  {   2.706,   3.841,   5.024,   6.635,  10.828},
  {   4.605,   5.991,   7.378,   9.210,  13.816},
  {   6.251,   7.815,   9.348,  11.345,  16.266},
  {   7.779,   9.488,  11.143,  13.277,  18.467},
  {   9.236,  11.070,  12.833,  15.086,  20.515},
  {  10.645,  12.592,  14.449,  16.812,  22.458},
  {  12.017,  14.067,  16.013,  18.475,  24.322},
  {  13.362,  15.507,  17.535,  20.090,  26.125},
  {  14.684,  16.919,  19.023,  21.666,  27.877},
  {  15.987,  18.307,  20.483,  23.209,  29.588},
  {  17.275,  19.675,  21.920,  24.725,  31.264},
  {  18.549,  21.026,  23.337,  26.217,  32.910},
  {  19.812,  22.362,  24.736,  27.688,  34.528},
  {  21.064,  23.685,  26.119,  29.141,  36.123},
  {  22.307,  24.996,  27.488,  30.578,  37.697},
  {  23.542,  26.296,  28.845,  32.000,  39.252},
  {  24.769,  27.587,  30.191,  33.409,  40.790},
  {  25.989,  28.869,  31.526,  34.805,  42.312},
  {  27.204,  30.144,  32.852,  36.191,  43.820},
  {  28.412,  31.410,  34.170,  37.566,  45.315},
  {  29.615,  32.671,  35.479,  38.932,  46.797},
  {  30.813,  33.924,  36.781,  40.289,  48.268},
  {  32.007,  35.172,  38.076,  41.638,  49.728},
  {  33.196,  36.415,  39.364,  42.980,  51.179},
  {  34.382,  37.652,  40.646,  44.314,  52.620},
  {  35.563,  38.885,  41.923,  45.642,  54.052},
  {  36.741,  40.113,  43.195,  46.963,  55.476},
  {  37.916,  41.337,  44.461,  48.278,  56.892},
  {  39.087,  42.557,  45.722,  49.588,  58.301},
  {  40.256,  43.773,  46.979,  50.892,  59.703},
  {  41.422,  44.985,  48.232,  52.191,  61.098},
  {  42.585,  46.194,  49.480,  53.486,  62.487},
  {  43.745,  47.400,  50.725,  54.776,  63.870},
  {  44.903,  48.602,  51.966,  56.061,  65.247},
  {  46.059,  49.802,  53.203,  57.342,  66.619},
  {  47.212,  50.998,  54.437,  58.619,  67.985},
  {  48.363,  52.192,  55.668,  59.893,  69.347},
  {  49.513,  53.384,  56.896,  61.162,  70.703},
  {  50.660,  54.572,  58.120,  62.428,  72.055},
  {  51.805,  55.758,  59.342,  63.691,  73.402},
  {  52.949,  56.942,  60.561,  64.950,  74.745},
  {  54.090,  58.124,  61.777,  66.206,  76.084},
  {  55.230,  59.304,  62.990,  67.459,  77.419},
  {  56.369,  60.481,  64.201,  68.710,  78.750},
  {  57.505,  61.656,  65.410,  69.957,  80.077},
  {  58.641,  62.830,  66.617,  71.201,  81.400},
  {  59.774,  64.001,  67.821,  72.443,  82.720},
  {  60.907,  65.171,  69.023,  73.683,  84.037},
  {  62.038,  66.339,  70.222,  74.919,  85.351},
  {  63.167,  67.505,  71.420,  76.154,  86.661},
  {  64.295,  68.669,  72.616,  77.386,  87.968},
  {  65.422,  69.832,  73.810,  78.616,  89.272},
  {  66.548,  70.993,  75.002,  79.843,  90.573},
  {  67.673,  72.153,  76.192,  81.069,  91.872},
  {  68.796,  73.311,  77.380,  82.292,  93.168},
  {  69.919,  74.468,  78.567,  83.513,  94.461},
  {  71.040,  75.624,  79.752,  84.733,  95.751},
  {  72.160,  76.778,  80.936,  85.950,  97.039},
  {  73.279,  77.931,  82.117,  87.166,  98.324},
  {  74.397,  79.082,  83.298,  88.379,  99.607},
  {  75.514,  80.232,  84.476,  89.591, 100.888},
  {  76.630,  81.381,  85.654,  90.802, 102.166},
  {  77.745,  82.529,  86.830,  92.010, 103.442},
  {  78.860,  83.675,  88.004,  93.217, 104.716},
  {  79.973,  84.821,  89.177,  94.422, 105.988},
  {  81.085,  85.965,  90.349,  95.626, 107.258},
  {  82.197,  87.108,  91.519,  96.828, 108.526},
  {  83.308,  88.250,  92.689,  98.028, 109.791},
  {  84.418,  89.391,  93.856,  99.228, 111.055},
  {  85.527,  90.531,  95.023, 100.425, 112.317},
  {  86.635,  91.670,  96.189, 101.621, 113.577},
  {  87.743,  92.808,  97.353, 102.816, 114.835},
  {  88.850,  93.945,  98.516, 104.010, 116.092},
  {  89.956,  95.081,  99.678, 105.202, 117.346},
  {  91.061,  96.217, 100.839, 106.393, 118.599},
  {  92.166,  97.351, 101.999, 107.583, 119.850},
  {  93.270,  98.484, 103.158, 108.771, 121.100},
  {  94.374,  99.617, 104.316, 109.958, 122.348},
  {  95.476, 100.749, 105.473, 111.144, 123.594},
  {  96.578, 101.879, 106.629, 112.329, 124.839},
  {  97.680, 103.010, 107.783, 113.512, 126.083},
  {  98.780, 104.139, 108.937, 114.695, 127.324},
  {  99.880, 105.267, 110.090, 115.876, 128.565},
  { 100.980, 106.395, 111.242, 117.057, 129.804},
  { 102.079, 107.522, 112.393, 118.236, 131.041},
  { 103.177, 108.648, 113.544, 119.414, 132.277},
  { 104.275, 109.773, 114.693, 120.591, 133.512},
  { 105.372, 110.898, 115.841, 121.767, 134.746},
  { 106.469, 112.022, 116.989, 122.942, 135.978},
  { 107.565, 113.145, 118.136, 124.116, 137.208},
  { 108.661, 114.268, 119.282, 125.289, 138.438},
  { 109.756, 115.390, 120.427, 126.462, 139.666},
  { 110.850, 116.511, 121.571, 127.633, 140.893},
  { 111.944, 117.632, 122.715, 128.803, 142.119},
  { 113.038, 118.752, 123.858, 129.973, 143.344},
  { 114.131, 119.871, 125.000, 131.141, 144.567},
  { 115.223, 120.990, 126.141, 132.309, 145.789},
  { 116.315, 122.108, 127.282, 133.476, 147.010},
  { 117.407, 123.225, 128.422, 134.642, 148.230},
  { 118.498, 124.342, 129.561, 135.807, 149.449},
};

//**/  Lower 0.9 0.95 0.975 0.99 0.999 (one-sided)
static double CSLTable[CSMaxSampleSize][CSMaxSignificanceLevels] =
{
   {       .016,     .004,     .001,     .000,     .000},
   {       .211,     .103,     .051,     .020,     .002},
   {       .584,     .352,     .216,     .115,     .024},
   {      1.064,     .711,     .484,     .297,     .091},
   {      1.610,    1.145,     .831,     .554,     .210},
   {      2.204,    1.635,    1.237,     .872,     .381},
   {      2.833,    2.167,    1.690,    1.239,     .598},
   {      3.490,    2.733,    2.180,    1.646,     .857},
   {      4.168,    3.325,    2.700,    2.088,    1.152},
   {      4.865,    3.940,    3.247,    2.558,    1.479},
   {      5.578,    4.575,    3.816,    3.053,    1.834},
   {      6.304,    5.226,    4.404,    3.571,    2.214},
   {      7.042,    5.892,    5.009,    4.107,    2.617},
   {      7.790,    6.571,    5.629,    4.660,    3.041},
   {      8.547,    7.261,    6.262,    5.229,    3.483},
   {      9.312,    7.962,    6.908,    5.812,    3.942},
   {     10.085,    8.672,    7.564,    6.408,    4.416},
   {     10.865,    9.390,    8.231,    7.015,    4.905},
   {     11.651,   10.117,    8.907,    7.633,    5.407},
   {     12.443,   10.851,    9.591,    8.260,    5.921},
   {     13.240,   11.591,   10.283,    8.897,    6.447},
   {     14.041,   12.338,   10.982,    9.542,    6.983},
   {     14.848,   13.091,   11.689,   10.196,    7.529},
   {     15.659,   13.848,   12.401,   10.856,    8.085},
   {     16.473,   14.611,   13.120,   11.524,    8.649},
   {     17.292,   15.379,   13.844,   12.198,    9.222},
   {     18.114,   16.151,   14.573,   12.879,    9.803},
   {     18.939,   16.928,   15.308,   13.565,   10.391},
   {     19.768,   17.708,   16.047,   14.256,   10.986},
   {     20.599,   18.493,   16.791,   14.953,   11.588},
   {     21.434,   19.281,   17.539,   15.655,   12.196},
   {     22.271,   20.072,   18.291,   16.362,   12.811},
   {     23.110,   20.867,   19.047,   17.074,   13.431},
   {     23.952,   21.664,   19.806,   17.789,   14.057},
   {     24.797,   22.465,   20.569,   18.509,   14.688},
   {     25.643,   23.269,   21.336,   19.233,   15.324},
   {     26.492,   24.075,   22.106,   19.960,   15.965},
   {     27.343,   24.884,   22.878,   20.691,   16.611},
   {     28.196,   25.695,   23.654,   21.426,   17.262},
   {     29.051,   26.509,   24.433,   22.164,   17.916},
   {     29.907,   27.326,   25.215,   22.906,   18.575},
   {     30.765,   28.144,   25.999,   23.650,   19.239},
   {     31.625,   28.965,   26.785,   24.398,   19.906},
   {     32.487,   29.787,   27.575,   25.148,   20.576},
   {     33.350,   30.612,   28.366,   25.901,   21.251},
   {     34.215,   31.439,   29.160,   26.657,   21.929},
   {     35.081,   32.268,   29.956,   27.416,   22.610},
   {     35.949,   33.098,   30.755,   28.177,   23.295},
   {     36.818,   33.930,   31.555,   28.941,   23.983},
   {     37.689,   34.764,   32.357,   29.707,   24.674},
   {     38.560,   35.600,   33.162,   30.475,   25.368},
   {     39.433,   36.437,   33.968,   31.246,   26.065},
   {     40.308,   37.276,   34.776,   32.018,   26.765},
   {     41.183,   38.116,   35.586,   32.793,   27.468},
   {     42.060,   38.958,   36.398,   33.570,   28.173},
   {     42.937,   39.801,   37.212,   34.350,   28.881},
   {     43.816,   40.646,   38.027,   35.131,   29.592},
   {     44.696,   41.492,   38.844,   35.913,   30.305},
   {     45.577,   42.339,   39.662,   36.698,   31.020},
   {     46.459,   43.188,   40.482,   37.485,   31.738},
   {     47.342,   44.038,   41.303,   38.273,   32.459},
   {     48.226,   44.889,   42.126,   39.063,   33.181},
   {     49.111,   45.741,   42.950,   39.855,   33.906},
   {     49.996,   46.595,   43.776,   40.649,   34.633},
   {     50.883,   47.450,   44.603,   41.444,   35.362},
   {     51.770,   48.305,   45.431,   42.240,   36.093},
   {     52.659,   49.162,   46.261,   43.038,   36.826},
   {     53.548,   50.020,   47.092,   43.838,   37.561},
   {     54.438,   50.879,   47.924,   44.639,   38.298},
   {     55.329,   51.739,   48.758,   45.442,   39.036},
   {     56.221,   52.600,   49.592,   46.246,   39.777},
   {     57.113,   53.462,   50.428,   47.051,   40.519},
   {     58.006,   54.325,   51.265,   47.858,   41.264},
   {     58.900,   55.189,   52.103,   48.666,   42.010},
   {     59.795,   56.054,   52.942,   49.475,   42.757},
   {     60.690,   56.920,   53.782,   50.286,   43.507},
   {     61.586,   57.786,   54.623,   51.097,   44.258},
   {     62.483,   58.654,   55.466,   51.910,   45.010},
   {     63.380,   59.522,   56.309,   52.725,   45.764},
   {     64.278,   60.391,   57.153,   53.540,   46.520},
   {     65.176,   61.261,   57.998,   54.357,   47.277},
   {     66.076,   62.132,   58.845,   55.174,   48.036},
   {     66.976,   63.004,   59.692,   55.993,   48.796},
   {     67.876,   63.876,   60.540,   56.813,   49.557},
   {     68.777,   64.749,   61.389,   57.634,   50.320},
   {     69.679,   65.623,   62.239,   58.456,   51.085},
   {     70.581,   66.498,   63.089,   59.279,   51.850},
   {     71.484,   67.373,   63.941,   60.103,   52.617},
   {     72.387,   68.249,   64.793,   60.928,   53.386},
   {     73.291,   69.126,   65.647,   61.754,   54.155},
   {     74.196,   70.003,   66.501,   62.581,   54.926},
   {     75.100,   70.882,   67.356,   63.409,   55.698},
   {     76.006,   71.760,   68.211,   64.238,   56.472},
   {     76.912,   72.640,   69.068,   65.068,   57.246},
   {     77.818,   73.520,   69.925,   65.898,   58.022},
   {     78.725,   74.401,   70.783,   66.730,   58.799},
   {     79.633,   75.282,   71.642,   67.562,   59.577},
   {     80.541,   76.164,   72.501,   68.396,   60.356},
   {     81.449,   77.046,   73.361,   69.230,   61.137},
   {     82.358,   77.929,   74.222,   70.065,   61.918},
};

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
OneSampleAnalyzer::OneSampleAnalyzer() : Analyzer()
{
  setName("1SAMPLE");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OneSampleAnalyzer::~OneSampleAnalyzer()
{
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double OneSampleAnalyzer::analyze(aData &adata)
{
  int    testOption=-1, length, ss;
  double ddata;
  char   filename[501], lineIn[501];
  FILE   *file;
                                                                                
  (void) adata;

  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "* The available 1-sample tests are: \n");
  printEquals(PL_INFO, 0);
  while (testOption < 1 || testOption > 2)
  {
    printf( "Choose your test: \n");
    printf("(1) Chi-squared test (sample std dev test)\n");
    printf("(2) Distribution fit (sample mean and/or std dev test)\n");
    printf("Your choice (1 or 2): ");
    scanf("%d", &testOption);
  }

  //**/ ---------------------------------------------------------------
  //**/ check and read input data file
  //**/ ---------------------------------------------------------------
  printf("Enter your data file (format: N Y1 Y2 .. YN) : ");
  scanf("%s", filename);
  fgets(lineIn,500,stdin);
  file = fopen(filename, "r");
  if (file == NULL)
  {
    printOutTS(PL_INFO, 
       "OneSampleAnalyzer ERROR: file %s does not exist.\n",filename);
    return PSUADE_UNDEFINED;
  }
  fscanf(file, "%d", &length);
  if (length <= 1)
  {
    printOutTS(PL_ERROR, 
      "OneSampleAnalyzer ERROR: file has invalid length %d.\n",length);
    fclose(file);
    return PSUADE_UNDEFINED;
  }
  psVector vecYT;
  vecYT.setLength(length);
  for (ss = 0; ss < length; ss++) 
  {
    fscanf(file, "%lg", &ddata);
    vecYT[ss] = ddata;
  }
  fclose(file);

  //**/ ---------------------------------------------------------------
  //**/ call corresponding function 
  //**/ ---------------------------------------------------------------
  switch(testOption)
  {
    case 1:
         ddata = CSAnalyze(length, vecYT.getDVector(), 2);
         break;
    case 2:
         ddata = DFAnalyze(length, vecYT.getDVector(), 2);
         break;
  }
  return ddata;
}

// *************************************************************************
// Chi-squared test (test standard deviation)
// -------------------------------------------------------------------------
double OneSampleAnalyzer::CSAnalyze(int length, double *Y, int pLevel)
{
  double tval, cvalU, cvalL, dtemp, retval, newStd, mean, var;
  char   lineIn[500];

  //**/ ---------------------------------------------------------------
  //**/ get target standard deviation
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "* CHI-SQUARED TEST: \n");
  printOutTS(PL_INFO, 
    "*    NULL HYPOTHESIS H0: dataset std dev differs from a target\n");
  printEquals(PL_INFO, 0);

  if (length > 100)
  {
    printOutTS(PL_ERROR, 
       "Chi-squared test ERROR: sample size needs to be <= 100\n");
    printOutTS(PL_ERROR, 
       "       because the table for larger sample is not ready.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ compute statistic (2-sided test with alpha =0.05) 
  //**/ ---------------------------------------------------------------
  computeMeanVariance(length, Y, &mean, &var);
  newStd = 0.0;
  while (newStd <= 0.0)
  {
    printf("Sample std deviation  = %13.5e\n", sqrt(var));
    printf("What is the target standard deviation? (> 0) ");
    scanf("%lg", &newStd);
  }
  fgets(lineIn,500,stdin);
  dtemp = var / (newStd * newStd);
  tval = (length - 1.0) * dtemp;
  if (pLevel > 0)
  {
    printOutTS(PL_INFO, "Sample mean           = %13.5e\n", mean);
    printOutTS(PL_INFO, "Sample std deviation  = %13.5e\n", sqrt(var));
    printOutTS(PL_INFO, "Target std deviation  = %13.5e\n", newStd);
    printOutTS(PL_INFO, "Size of dataset       = %d\n", length);
    printOutTS(PL_INFO, "Chi-squared statistic = %13.5e\n", tval);
    printOutTS(PL_INFO, 
     "(statistic is (N-1) (ssd/tsd)^2 where ssd is the sample standard\n");
    printOutTS(PL_INFO, 
     " deviation and tsd is the target standard deviation.)\n");
    printEquals(PL_INFO, 0);
  }
  printOutTS(PL_INFO, "Significance level    = 0.1 \n");
  if (length > CSMaxSampleSize+1) cvalU = CSUTable[CSMaxSampleSize-1][1];
  else                            cvalU = CSUTable[length-2][1];
  if (length > CSMaxSampleSize+1) cvalL = CSLTable[CSMaxSampleSize-1][1];
  else                            cvalL = CSLTable[length-2][1];
  printOutTS(PL_INFO, 
     "Upper, lower bounds of chi-squared = %12.4e, %12.4e\n",cvalU,cvalL);
  if (tval > cvalU || tval < cvalL )
  {
    printOutTS(PL_INFO, "ACCEPT NULL HYPOTHESIS (stdev ~ %e.)\n", newStd);
    retval = 0.0;
  }
  else
  {
    printOutTS(PL_INFO, "REJECT NULL HYPOTHESIS (stdev ~ %e).\n", newStd);
    retval = 1.0;
  }
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO, "Significance level    = 0.05 \n");
  if (length > CSMaxSampleSize+1) cvalU = CSUTable[CSMaxSampleSize-1][2];
  else                            cvalU = CSUTable[length-2][2];
  if (length > CSMaxSampleSize+1) cvalL = CSLTable[CSMaxSampleSize-1][2];
  else                            cvalL = CSLTable[length-2][2];
  printOutTS(PL_INFO,"Upper, lower bounds of chi-squared = %12.4e, %12.4e\n",
             cvalU,cvalL);
  if (tval > cvalU || tval < cvalL )
  {
    printOutTS(PL_INFO, "ACCEPT NULL HYPOTHESIS (stdev ~ %e.)\n", newStd);
    retval = 0.0;
  }
  else
  {
    printOutTS(PL_INFO, "REJECT NULL HYPOTHESIS (stdev ~ %e).\n", newStd);
    retval = 1.0;
  }
  if (pLevel > 0)
  {
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }
  return retval;
}

// *************************************************************************
// Distribution fit analysis
// -------------------------------------------------------------------------
double OneSampleAnalyzer::DFAnalyze(int length, double *Y, int pLevel)
{
  int    meanNpts, stdevNpts, im, is, gtype, nSamp, inc, imKeep, isKeep;
  double mean, var, stdev, meanL, meanU, stdevL, stdevU, delMean, delStd;
  double gLower, gUpper, curMean, curStdev;
  double targetMean, targetStdev, minRetVal, retval;
  char   pString[500], lineIn[500];
  FILE   *fp=NULL;
  PDFManager *pdfman;
  psMatrix   corMat;
  psVector   vecOut, vecUpper, vecLower;
  TwoSampleAnalyzer *tsPtr=NULL;

  //**/ ---------------------------------------------------------------
  //**/ first compute basic statistic and ask for ranges to search
  //**/ ---------------------------------------------------------------
  computeMeanVariance(length, Y, &mean, &var);
  stdev = sqrt(var);
  printOutTS(PL_INFO, "Sample mean               = %13.5e\n", mean);
  printOutTS(PL_INFO, "Sample standard deviation = %13.5e\n", stdev);
  meanL = meanU = 0.0;
  meanNpts = 0;
  while (meanL >= meanU || meanNpts <= 1)
  {
    printf("Enter the lower bound of mean to search : ");
    scanf("%lg", &meanL);
    printf("Enter the upper bound of mean to search : ");
    scanf("%lg", &meanU);
    printf("Enter number of points for the mean search (e.g., 100): ");
    scanf("%d", &meanNpts);
    if (meanL >= meanU) 
      printOutTS(PL_INFO, "lower bound %e >= upper bound %e\n",meanL,meanU);
    if (meanNpts <= 1)
    {
      printOutTS(PL_INFO, "number of points has to be > 1.\n");
      fgets(lineIn,500,stdin);
    }
  }
  stdevL = stdevU = 0.0;
  stdevNpts = 0;
  while (stdevL >= stdevU || stdevNpts <= 1)
  {
    printf("Enter the lower bound of std dev to search : ");
    scanf("%lg", &stdevL);
    printf("Enter the upper bound of std dev to search : ");
    scanf("%lg", &stdevU);
    printf("Enter number of points for std dev search (e.g., 100): ");
    scanf("%d", &stdevNpts);
    if (stdevL >= stdevU) 
      printOutTS(PL_INFO, "lower bound %e >= upper bound %e\n",stdevL,stdevU);
    if (stdevNpts <= 1)
    {
      printOutTS(PL_INFO, "number of points has to be > 1.\n");
      fgets(lineIn,500,stdin);
    }
  }
  fgets(lineIn,500,stdin);

  //**/ ---------------------------------------------------------------
  //**/ set up sampler to search
  //**/ ---------------------------------------------------------------
  delMean = (meanU - meanL) / (double) meanNpts;
  delStd  = (stdevU - stdevL) / (double) stdevNpts;
  tsPtr = new TwoSampleAnalyzer();
  snprintf(pString,100,"Distribution type = (1) Normal (2) Lognormal : ");
  gtype = getInt(1, 2, pString);
  if      (gtype == 1) gtype = PSUADE_PDF_NORMAL;
  else if (gtype == 2) gtype = PSUADE_PDF_LOGNORMAL;
  nSamp = length;

  //**/ ---------------------------------------------------------------
  //**/ search for the best distribution parameters (mean, std dev)
  //**/ ---------------------------------------------------------------
  if (pLevel > 0)
  {
    if (plotMatlab()) fp = fopen("matlabdf.m", "w");
    else              fp = fopen("scilabdf.sci", "w");
  }
  if (fp != NULL) fprintf(fp, "Amat = [\n");
  targetMean = 0.0;
  targetStdev = 0.0;
  minRetVal = PSUADE_UNDEFINED;
  inc = meanNpts * stdevNpts / 100;
  if (inc <= 0) inc = 1;
  if (pLevel > 0) printOutTS(PL_INFO, "Begin processing ... \n");
  imKeep = isKeep = 0;
  for (im = 0; im < meanNpts; im++)
  {
    for (is = 0; is < stdevNpts; is++)
    {
      if ((im*stdevNpts+is) % inc == 0) printOutTS(PL_INFO, ".");
      curMean = meanL + im * (meanU - meanL) / (double) meanNpts; 
      curStdev = stdevL + is * (stdevU - stdevL) / (double) stdevNpts; 
      gLower = curMean - 5.0 * curStdev; 
      gUpper = curMean + 5.0 * curStdev; 
      corMat.setDim(1,1);
      corMat.setEntry(0,0, 1.0e0);
      pdfman = new PDFManager();
      pdfman->initialize(1,&gtype,&curMean,&curStdev,corMat,NULL,NULL);
      vecOut.setLength(nSamp);
      vecUpper.load(1, &gUpper);
      vecLower.load(1, &gLower);
      pdfman->genSample(nSamp, vecOut, vecLower, vecUpper);
      delete pdfman;
      retval = tsPtr->KSAnalyze(length, Y, nSamp, vecOut.getDVector(), 0);
      if (fp != NULL) fprintf(fp, "%e\n", retval);
      if (retval < minRetVal)
      {
        isKeep = is;
        imKeep = im;
        minRetVal = retval;
        targetMean = curMean;
        targetStdev = curStdev;
        //printf("Current best mean, std = %e %e : KS metric = %e\n",
        //       curMean,curStdev,retval);
      }
    }
  }
  printOutTS(PL_INFO, "\n");

  //**/ ---------------------------------------------------------------
  //**/ once found, output the K-S statistic by re-run
  //**/ ---------------------------------------------------------------
  curMean = meanL + imKeep * (meanU - meanL) / (double) meanNpts; 
  curStdev = stdevL + isKeep * (stdevU - stdevL) / (double) stdevNpts; 
  gLower = curMean - 5.0 * curStdev; 
  gUpper = curMean + 5.0 * curStdev; 
  corMat.setDim(1,1);
  corMat.setEntry(0,0, 1.0e0);
  pdfman = new PDFManager();
  pdfman->initialize(1, &gtype, &curMean, &curStdev, corMat, NULL,NULL);
  vecOut.setLength(nSamp);
  vecUpper.load(1, &gUpper);
  vecLower.load(1, &gLower);
  pdfman->genSample(nSamp, vecOut, vecLower, vecUpper);
  delete pdfman;
  retval = tsPtr->KSAnalyze(length, Y, nSamp, vecOut.getDVector(), 0);

  //**/ ---------------------------------------------------------------
  //**/ output results and create matlab response surface file
  //**/ ---------------------------------------------------------------
  if (pLevel > 0)
  {
    printOutTS(PL_INFO, 
       "Distribution fitting: best mean    = %13.5e\n", targetMean);
    printOutTS(PL_INFO, 
       "Distribution fitting: best std dev = %13.5e\n", targetStdev);
  }
  if (fp != NULL)
  {
    fprintf(fp, "];\n");
    fprintf(fp, "Amat = reshape(Amat,%d,%d);\n",meanNpts,stdevNpts);
    fprintf(fp, "M = [\n");
    for (im = 0; im < meanNpts; im++)
    {
      curMean = meanL + im * (meanU - meanL) / (double) meanNpts; 
      fprintf(fp, "%e\n", curMean);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "S = [\n");
    for (is = 0; is < stdevNpts; is++)
    {
      curStdev = stdevL + is * (stdevU - stdevL) / (double) stdevNpts; 
      fprintf(fp, "%e\n", curStdev);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "mesh(M,S,Amat)\n");
    fwritePlotAxes(fp);
    strcpy(pString, "target means");
    fwritePlotXLabel(fp, pString);
    strcpy(pString, "target standard deviations");
    fwritePlotYLabel(fp, pString);
    strcpy(pString, "Excess KS D-statistic");
    fwritePlotZLabel(fp, pString);
    if (plotMatlab())
      printOutTS(PL_INFO, 
       "The response surface for excess D-statistic is in matlabdf.m\n");
    else
      printOutTS(PL_INFO, 
       "The response surface for excess D-statistic is in scilabdf.sci\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  delete tsPtr;
  if (fp != NULL) fclose(fp);
  return 0.0;
}

// *************************************************************************
// Compute agglomerated mean and variance
// -------------------------------------------------------------------------
int OneSampleAnalyzer::computeMeanVariance(int nSamples, double *Y, 
                                           double *mean, double *variance)
{
  int ss;
  double meanTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++) meanTmp += Y[ss];
  meanTmp /= (double) nSamples;
  double varTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++)
    varTmp += ((Y[ss] - meanTmp) * (Y[ss] - meanTmp));
  varTmp /= (double) (nSamples - 1);
  (*mean)     = meanTmp;
  (*variance) = varTmp;
  return 0;
}

// *************************************************************************
// Compute inverse chi-squared information
//**/ (May, 2009: This works only for dofs <= 250)
//**/ converted from James Filliben's fortran code chsppf.f
// -------------------------------------------------------------------------
int OneSampleAnalyzer::inverseChiSquared(double percent, int dofs, 
                                         double &retdata)
{
  int    icount, info;
  double C, D[10], dnu, gamma, dp, z, den, z2, z3, z4, z5, A, B, G;
  double xmin, xmin0, xmax, pcalc, xlower, xupper, xmid, xdel, dx;

  C = 0.918938533204672741;
  D[0] = 0.0833333333333333333;
  D[1] = -0.00277777777777777778;
  D[2] = 0.000793650793650793651;
  D[3] = -0.000595238095238095238;
  D[4] = 0.0008417508417508417151;
  D[5] = -0.00191752691752691753;
  D[6] = 0.00641025641025641025;
  D[7] = -0.02955065359147712418;
  D[8] = 0.179644372368830573;
  D[9] = -1.39243221690590111;

  dnu   = (double) dofs; 
  gamma = 0.5 * dnu;
  dp    = percent;

  z = gamma;
  den = 1.0;
  while (z < 10.0)
  {
    den *= z;
    z = z + 1.0;
  }
  z2 = z * z;
  z3 = z * z2;
  z4 = z2 * z2;
  z5 = z2 * z3;
  A  = (z - 0.5) * log(z) - z + C;
  B  = D[0]/z + D[1]/z3 + D[2]/z5 + D[3]/(z2*z5) + D[4]/(z4*z5) +
       D[5]/(z*z5*z5)+D[6]/(z3*z5*z5)+D[7]/(z5*z5*z5)+D[8]/(z2*z5*z5*z5);
  G  = exp(A+B)/den;

  xmin0 = pow(dp * gamma * G, 1.0/gamma);
  xmin  = xmin0;
  icount = 1;
  while (icount <= 30000)
  {
    xmax = xmin0 * (double) icount;
    dx = xmax;
    info = iterate(gamma, dx, G, pcalc);
    if (info == 1) return 0;

    if (pcalc >= dp)
    {
      xmid = 0.5 * (xmin + xmax);
      xlower = xmin;
      xupper = xmax;
      icount = 0;
      while (1)
      {
        dx = xmid;
        info = iterate(gamma, dx, G, pcalc);
        if (info == 1)
        {
          retdata = 0.0; 
          return 1;
        }
        if (pcalc == dp) 
        {
          retdata = 2.0*xmid;
          return 0;
        }
        else if (pcalc >  dp)
        {
          xupper = xmid;
          xmid = 0.5 * (xmid + xlower);
        } 
        else
        {
          xlower = xmid;
          xmid = 0.5 * (xmid + xupper);
        }
        xdel = xmid - xlower;
        if (xdel < 0) xdel = - xdel;
        icount++;
        if (xdel < 1.0e-10 || icount > 100)
        {
          retdata = 2.0*xmid;
          return 0;
        }
      }
    }
    else
    {
      xmin = xmax;
      icount++;
    }
  }
  xmid = 0.5 * (xmin + xmax);
  xlower = xmin;
  xupper = xmax;
  icount = 0;
  while (1)
  {
    dx = xmid;
    info = iterate(gamma, dx, G, pcalc);
    if (info == 1)
    {
      retdata = 0.0;
      return 1;
    }
    if (pcalc == dp) 
    {
      retdata = 2.0*xmid;
      return 0;
    }
    else if (pcalc >  dp)
    {
      xupper = xmid;
      xmid = 0.5 * (xmid + xlower);
    } 
    else
    {
      xlower = xmid;
      xmid = 0.5 * (xmid + xupper);
    }
    xdel = xmid - xlower;
    if (xdel < 0) xdel = - xdel;
    icount++;
    if (xdel < 1.0e-10 || icount > 100)
    {
      retdata = 2.0*xmid;
      return 0;
    }
  }
}

// *************************************************************************
// Compute inverse chi-squared information
// -------------------------------------------------------------------------
int OneSampleAnalyzer::iterate(double gamma, double delta, double gamma2, 
                               double &retdata)
{
  int    jj, maxit=100000;
  double sum, term, cut1, cut2, cutoff;

  sum  = 1.0 / gamma;
  term = 1.0 / gamma;
  cut1 = delta - gamma;
  cut2 = delta * 1.0e10;
  for (jj = 1; jj <= maxit; jj++)
  {
    term = delta * term / (gamma + jj);
    sum += term;
    cutoff = cut1 + (cut2 * term / sum);
    if (jj > cutoff) break;
  }
  if (jj == maxit+1)
  {
    retdata = 0.0;
    return 1;
  }
  else
  {
    retdata = pow(delta, gamma) * exp(-delta) * sum / gamma2;
    return 0;
  }
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
OneSampleAnalyzer& OneSampleAnalyzer::operator=(const OneSampleAnalyzer &)
{
  printOutTS(PL_ERROR, 
       "OneSampleAnalyzer operator= ERROR: operation not supported.\n");
  exit(1);
  return (*this);
}

