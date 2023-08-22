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
// Functions for the class TwoSampleAnalyzer (T-test, K-S test)
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONG
// DATE   : 2007
// ************************************************************************

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "TwoSampleAnalyzer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "psMatrix.h"
#include "psVector.h"
#include "sysdef.h"
#include "PDFManager.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Student's T-test table
// ------------------------------------------------------------------------

#define TMaxSampleSize        100
#define TMaxSignificanceLevels  6
static double TTable[TMaxSampleSize][TMaxSignificanceLevels] =
{
  {3.078, 6.314, 12.706,  31.821,  63.657, 318.313},
  {1.886, 2.920,  4.303,   6.965,   9.925,  22.327},
  {1.638, 2.353,  3.182,   4.541,   5.841,  10.215},
  {1.533, 2.132,  2.776,   3.747,   4.604,   7.173},
  {1.476, 2.015,  2.571,   3.365,   4.032,   5.893},
  {1.440, 1.943,  2.447,   3.143,   3.707,   5.208},
  {1.415, 1.895,  2.365,   2.998,   3.499,   4.782},
  {1.397, 1.860,  2.306,   2.896,   3.355,   4.499},
  {1.383, 1.833,  2.262,   2.821,   3.250,   4.296},
  {1.372, 1.812,  2.228,   2.764,   3.169,   4.143},
  {1.363, 1.796,  2.201,   2.718,   3.106,   4.024},
  {1.356, 1.782,  2.179,   2.681,   3.055,   3.929},
  {1.350, 1.771,  2.160,   2.650,   3.012,   3.852},
  {1.345, 1.761,  2.145,   2.624,   2.977,   3.787},
  {1.341, 1.753,  2.131,   2.602,   2.947,   3.733},
  {1.337, 1.746,  2.120,   2.583,   2.921,   3.686},
  {1.333, 1.740,  2.110,   2.567,   2.898,   3.646},
  {1.330, 1.734,  2.101,   2.552,   2.878,   3.610},
  {1.328, 1.729,  2.093,   2.539,   2.861,   3.579},
  {1.325, 1.725,  2.086,   2.528,   2.845,   3.552},
  {1.323, 1.721,  2.080,   2.518,   2.831,   3.527},
  {1.321, 1.717,  2.074,   2.508,   2.819,   3.505},
  {1.319, 1.714,  2.069,   2.500,   2.807,   3.485},
  {1.318, 1.711,  2.064,   2.492,   2.797,   3.467},
  {1.316, 1.708,  2.060,   2.485,   2.787,   3.450},
  {1.315, 1.706,  2.056,   2.479,   2.779,   3.435},
  {1.314, 1.703,  2.052,   2.473,   2.771,   3.421},
  {1.313, 1.701,  2.048,   2.467,   2.763,   3.408},
  {1.311, 1.699,  2.045,   2.462,   2.756,   3.396},
  {1.310, 1.697,  2.042,   2.457,   2.750,   3.385},
  {1.309, 1.696,  2.040,   2.453,   2.744,   3.375},
  {1.309, 1.694,  2.037,   2.449,   2.738,   3.365},
  {1.308, 1.692,  2.035,   2.445,   2.733,   3.356},
  {1.307, 1.691,  2.032,   2.441,   2.728,   3.348},
  {1.306, 1.690,  2.030,   2.438,   2.724,   3.340},
  {1.306, 1.688,  2.028,   2.434,   2.719,   3.333},
  {1.305, 1.687,  2.026,   2.431,   2.715,   3.326},
  {1.304, 1.686,  2.024,   2.429,   2.712,   3.319},
  {1.304, 1.685,  2.023,   2.426,   2.708,   3.313},
  {1.303, 1.684,  2.021,   2.423,   2.704,   3.307},
  {1.303, 1.683,  2.020,   2.421,   2.701,   3.301},
  {1.302, 1.682,  2.018,   2.418,   2.698,   3.296},
  {1.302, 1.681,  2.017,   2.416,   2.695,   3.291},
  {1.301, 1.680,  2.015,   2.414,   2.692,   3.286},
  {1.301, 1.679,  2.014,   2.412,   2.690,   3.281},
  {1.300, 1.679,  2.013,   2.410,   2.687,   3.277},
  {1.300, 1.678,  2.012,   2.408,   2.685,   3.273},
  {1.299, 1.677,  2.011,   2.407,   2.682,   3.269},
  {1.299, 1.677,  2.010,   2.405,   2.680,   3.265},
  {1.299, 1.676,  2.009,   2.403,   2.678,   3.261},
  {1.298, 1.675,  2.008,   2.402,   2.676,   3.258},
  {1.298, 1.675,  2.007,   2.400,   2.674,   3.255},
  {1.298, 1.674,  2.006,   2.399,   2.672,   3.251},
  {1.297, 1.674,  2.005,   2.397,   2.670,   3.248},
  {1.297, 1.673,  2.004,   2.396,   2.668,   3.245},
  {1.297, 1.673,  2.003,   2.395,   2.667,   3.242},
  {1.297, 1.672,  2.002,   2.394,   2.665,   3.239},
  {1.296, 1.672,  2.002,   2.392,   2.663,   3.237},
  {1.296, 1.671,  2.001,   2.391,   2.662,   3.234},
  {1.296, 1.671,  2.000,   2.390,   2.660,   3.232},
  {1.296, 1.670,  2.000,   2.389,   2.659,   3.229},
  {1.295, 1.670,  1.999,   2.388,   2.657,   3.227},
  {1.295, 1.669,  1.998,   2.387,   2.656,   3.225},
  {1.295, 1.669,  1.998,   2.386,   2.655,   3.223},
  {1.295, 1.669,  1.997,   2.385,   2.654,   3.220},
  {1.295, 1.668,  1.997,   2.384,   2.652,   3.218},
  {1.294, 1.668,  1.996,   2.383,   2.651,   3.216},
  {1.294, 1.668,  1.995,   2.382,   2.650,   3.214},
  {1.294, 1.667,  1.995,   2.382,   2.649,   3.213},
  {1.294, 1.667,  1.994,   2.381,   2.648,   3.211},
  {1.294, 1.667,  1.994,   2.380,   2.647,   3.209},
  {1.293, 1.666,  1.993,   2.379,   2.646,   3.207},
  {1.293, 1.666,  1.993,   2.379,   2.645,   3.206},
  {1.293, 1.666,  1.993,   2.378,   2.644,   3.204},
  {1.293, 1.665,  1.992,   2.377,   2.643,   3.202},
  {1.293, 1.665,  1.992,   2.376,   2.642,   3.201},
  {1.293, 1.665,  1.991,   2.376,   2.641,   3.199},
  {1.292, 1.665,  1.991,   2.375,   2.640,   3.198},
  {1.292, 1.664,  1.990,   2.374,   2.640,   3.197},
  {1.292, 1.664,  1.990,   2.374,   2.639,   3.195},
  {1.292, 1.664,  1.990,   2.373,   2.638,   3.194},
  {1.292, 1.664,  1.989,   2.373,   2.637,   3.193},
  {1.292, 1.663,  1.989,   2.372,   2.636,   3.191},
  {1.292, 1.663,  1.989,   2.372,   2.636,   3.190},
  {1.292, 1.663,  1.988,   2.371,   2.635,   3.189},
  {1.291, 1.663,  1.988,   2.370,   2.634,   3.188},
  {1.291, 1.663,  1.988,   2.370,   2.634,   3.187},
  {1.291, 1.662,  1.987,   2.369,   2.633,   3.185},
  {1.291, 1.662,  1.987,   2.369,   2.632,   3.184},
  {1.291, 1.662,  1.987,   2.368,   2.632,   3.183},
  {1.291, 1.662,  1.986,   2.368,   2.631,   3.182},
  {1.291, 1.662,  1.986,   2.368,   2.630,   3.181},
  {1.291, 1.661,  1.986,   2.367,   2.630,   3.180},
  {1.291, 1.661,  1.986,   2.367,   2.629,   3.179},
  {1.291, 1.661,  1.985,   2.366,   2.629,   3.178},
  {1.290, 1.661,  1.985,   2.366,   2.628,   3.177},
  {1.290, 1.661,  1.985,   2.365,   2.627,   3.176},
  {1.290, 1.661,  1.984,   2.365,   2.627,   3.175},
  {1.290, 1.660,  1.984,   2.365,   2.626,   3.175},
  {1.290, 1.660,  1.984,   2.364,   2.626,   3.174}
};

// ************************************************************************
// Kolmogorov Smirnov table
// ------------------------------------------------------------------------

#define KSMaxSignificanceLevels 5
#define KSMaxSampleSize         35
static double KSTable[KSMaxSampleSize][KSMaxSignificanceLevels] =
{
  {0.900,   0.925,   0.950,    0.975,   0.995},
  {0.684,   0.726,   0.776,    0.842,   0.929},
  {0.565,   0.597,   0.642,    0.708,   0.828},
  {0.494,   0.525,   0.564,    0.624,   0.733},
  {0.446,   0.474,   0.510,    0.565,   0.669},
  {0.410,   0.436,   0.470,    0.521,   0.618},
  {0.381,   0.405,   0.438,    0.486,   0.577},
  {0.358,   0.381,   0.411,    0.457,   0.543},
  {0.339,   0.360,   0.388,    0.432,   0.514},
  {0.322,   0.342,   0.368,    0.410,   0.490},
  {0.307,   0.326,   0.352,    0.391,   0.468},
  {0.295,   0.313,   0.338,    0.375,   0.450},
  {0.284,   0.302,   0.325,    0.361,   0.433},
  {0.274,   0.292,   0.314,    0.349,   0.418},
  {0.266,   0.283,   0.304,    0.338,   0.404},
  {0.258,   0.274,   0.295,    0.328,   0.392},
  {0.250,   0.266,   0.286,    0.318,   0.381},
  {0.244,   0.259,   0.278,    0.309,   0.371},
  {0.237,   0.252,   0.271,    0.301,   0.363},
  {0.231,   0.246,   0.264,    0.295,   0.355},
  {0.226,   0.240,   0.258,    0.290,   0.348},
  {0.222,   0.235,   0.253,    0.285,   0.341},
  {0.218,   0.230,   0.248,    0.280,   0.334},
  {0.214,   0.225,   0.244,    0.275,   0.327},
  {0.210,   0.220,   0.240,    0.270,   0.320},
  {0.206,   0.215,   0.236,    0.264,   0.313},
  {0.202,   0.211,   0.232,    0.258,   0.306},
  {0.198,   0.207,   0.228,    0.252,   0.300},
  {0.194,   0.203,   0.224,    0.246,   0.295},
  {0.190,   0.200,   0.220,    0.240,   0.290},
  {0.187,   0.197,   0.217,    0.237,   0.286},
  {0.185,   0.195,   0.215,    0.235,   0.282},
  {0.183,   0.193,   0.213,    0.233,   0.278},
  {0.181,   0.191,   0.211,    0.231,   0.274},
  {0.180,   0.190,   0.210,    0.230,   0.270}
};

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
TwoSampleAnalyzer::TwoSampleAnalyzer() : Analyzer()
{
  setName("TwoSample");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
TwoSampleAnalyzer::~TwoSampleAnalyzer()
{
}

// ************************************************************************
// perform the choice of a number of two-sample tests
// ------------------------------------------------------------------------
double TwoSampleAnalyzer::analyze(aData &adata)
{
  int    testOption=-1, length1, length2, ss;
  double retval, ddata;
  char   filename1[500], filename2[500];

  (void) adata;

  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "* The available 2-sample tests are: \n");
  printOutTS(PL_INFO, "* (1) Student's T-test: \n");
  printOutTS(PL_INFO, 
     "*          DO two NORMALLY DISTRIBUTED populations differ?\n");
  printOutTS(PL_INFO, 
     "* (2) Mann-Whitney test: (non-parametric) \n");
  printOutTS(PL_INFO, 
     "*          Do the two samples come from the same population \n");
  printOutTS(PL_INFO, 
     "*          (drawn from the same probability distribution)? \n");
  printOutTS(PL_INFO, 
     "*          The MW test, roughly speaking, compares medians. Therefore,\n");
  printOutTS(PL_INFO, 
     "*          if one sample occupies the high and low ends and the other\n");
  printOutTS(PL_INFO, 
     "*          the middle, the MW test may not be effective. Also, the MW\n");
  printOutTS(PL_INFO, 
     "*          test many not be effective if there are too many ties. Use\n");
  printOutTS(PL_INFO, 
     "*          the K-S test instead. The MW test is thus more sensitive to\n");
  printOutTS(PL_INFO, 
     "*          differences in means or medians and less so for differences\n");
  printOutTS(PL_INFO, 
     "*          in shapes.\n");
  printOutTS(PL_INFO, 
     "* (3) Kolmogorov Smirnov test: (non-parametric)\n");
  printOutTS(PL_INFO, 
     "*          Does the sample come from a population with a specific\n");
  printOutTS(PL_INFO, 
     "*          distribution (with given mean/stdev)? \n");
  printOutTS(PL_INFO, 
     "*          The K-S test is less likely to detect small differences in\n");
  printOutTS(PL_INFO, 
     "*          the mean. It is sensitive to large data differences ==> \n");
  printOutTS(PL_INFO, 
     "*          shape of the distribution.\n");
  printAsterisks(PL_INFO, 0);
  while (testOption < 1 || testOption > 3)
  {
    printf( "Choose your test: \n");
    printf( "(1) Student's t-test\n");
    printf("(2) Kolmogorov Smirnov test\n");
    printf("(3) Mann-Whitney test\n");
    printf("Your choice (1, 2, or 3): ");
    scanf("%d", &testOption);
  }

  //**/ ---------------------------------------------------------------
  //**/ useful information 
  //**/ ---------------------------------------------------------------
  printf("Enter your data file 1 (format: N Y1 Y2 .. YN) : ");
  scanf("%s", filename1); 
  printf("Enter your data file 2 (format: N Y1 Y2 .. YN) : ");
  scanf("%s", filename2); 
  FILE *file1 = fopen(filename1, "r");
  if (file1 == NULL)
  {
    printOutTS(PL_ERROR, 
      "TwoSampleAnalyzer ERROR: file1 %s does not exist.\n", filename1);
    return PSUADE_UNDEFINED;
  }
  fscanf(file1, "%d", &length1); 
  if (length1 <= 0)
  {
    printOutTS(PL_ERROR, 
      "TwoSampleAnalyzer ERROR: file1 has invalid length %d.\n",length1);
    fclose(file1);
    return PSUADE_UNDEFINED;
  }
  fclose(file1);

  FILE *file2 = fopen(filename2, "r");
  if (file2 == NULL)
  {
    printOutTS(PL_ERROR, 
      "TwoSampleAnalyzer ERROR: file2 %s does not exist.\n", filename2);
    return PSUADE_UNDEFINED;
  }
  fscanf(file2, "%d", &length2); 
  if (length2 <= 0)
  {
    printOutTS(PL_ERROR, 
      "TwoSampleAnalyzer ERROR: file2 has invalid length %d.\n",length2);
    fclose(file2);
    return PSUADE_UNDEFINED;
  }
  fclose(file2);

  //**/ ---------------------------------------------------------------
  //**/ allocate memory and read data
  //**/ ---------------------------------------------------------------
  file1 = fopen(filename1, "r");
  if (file1 != NULL)
  {
    fscanf(file1, "%d", &length1);
  }
  else
  {
    printOutTS(PL_ERROR, 
      "file %s does not exist when trying to open in file %s, LINE %d\n",
      filename1, __FILE__, __LINE__);
    exit(1);
  }
  psVector vecYT;
  vecYT.setLength(length1);
  for (ss = 0; ss < length1; ss++) 
  {
    fscanf(file1, "%lg", &ddata);
    vecYT[ss] = ddata; 
  }
  fclose(file1);

  file2 = fopen(filename2, "r");
  if(file2 != NULL)
  { 
    fscanf(file2, "%d", &length2);
  }
  else
  {
    printOutTS(PL_ERROR, 
       "file %s does not exist when trying to open in file %s, Line %d\n",
       filename2, __FILE__, __LINE__);
    exit(1);
  }
  psVector vecY2;
  vecY2.setLength(length2);
  for (ss = 0; ss < length2; ss++) 
  {
    fscanf(file2, "%lg", &ddata);
    vecY2[ss] = ddata; 
  }
  fclose(file2);
   
  //**/ ---------------------------------------------------------------
  //**/ call corresponding function 
  //**/ ---------------------------------------------------------------
  switch(testOption)
  {
    case 1: retval = TAnalyze(length1, vecYT.getDVector(), length2,
                              vecY2.getDVector(), 2);  
            break; 
    case 2: retval = KSAnalyze(length1, vecYT.getDVector(), length2,
                               vecY2.getDVector(), 2); 
            break; 
    case 3: retval = MWAnalyze(length1, vecYT.getDVector(), length2, 
                               vecY2.getDVector(), 2); 
            break; 
  }
  return retval;
}

// ************************************************************************
// perform T-Test
// ------------------------------------------------------------------------
double TwoSampleAnalyzer::TAnalyze(int length1, double *Y1, int length2,
                                   double *Y2, int pLevel)
{
  int    ss, ii, dofs;
  double mean1, mean2, stdev, var1, var2, tval, retval;
  double localTable[TMaxSignificanceLevels];

  //**/ ---------------------------------------------------------------
  //**/ compute means 
  //**/ ---------------------------------------------------------------
  mean1 = mean2 = 0.0;
  for (ss = 0; ss < length1; ss++) mean1 += Y1[ss];
  for (ss = 0; ss < length2; ss++) mean2 += Y2[ss];
  mean1 /= (1.0 * length1);
  mean2 /= (1.0 * length2);
  var1 = var2 = 0.0;
  for (ss = 0; ss < length1; ss++) var1 += pow(Y1[ss] - mean1, 2.0);
  for (ss = 0; ss < length2; ss++) var2 += pow(Y2[ss] - mean2, 2.0);
  var1 = var1 / (length1 - 1.0);
  var2 = var2 / (length2 - 1.0);
  //**/stdev = (var1*(length1-1.0)+var2*(length2-1.0))/(length1+length2-2.0);
  //**/stdev = stdev * (1.0 / length1 + 1.0 / length2);
  //**/stdev = sqrt(stdev);
  stdev = sqrt(var1/length1+var2/length2);
  tval = (mean1 - mean2) / stdev;
  if (tval < 0) tval = - tval;

  //**/ ---------------------------------------------------------------
  //**/ calculate test-statistic 
  //**/ ---------------------------------------------------------------
  //dofs = length1 + length2 - 2;
  dofs = (int) (pow(stdev, 4.0) / (pow(var1/length1,2.0)/(length1-1.0)+
         pow(var2/length2,2.0)/(length2-1.0))+0.5);
  if (dofs > TMaxSampleSize) dofs = TMaxSampleSize;
  for (ii = 0; ii < TMaxSignificanceLevels; ii++)
    localTable[ii] = TTable[dofs-1][ii];

  if (pLevel > 1)
  {
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO," STUDENT'S (Welch) T-TEST:\n\n");
    printOutTS(PL_INFO," (Assume unequal sample sizes and variances)\n\n");
    printOutTS(PL_INFO," NULL HYPOTHESIS H0: dataset means do not ");
    printOutTS(PL_INFO,"differ significantly.\n");
    printOutTS(PL_INFO," (This test is applied when sample sizes ");
    printOutTS(PL_INFO,"are small enough that using\n");
    printOutTS(PL_INFO,"  normality assumption and the ");
    printOutTS(PL_INFO,"associated z-test leads to incorrect\n");
    printOutTS(PL_INFO,"  inference. For this test, the ");
    printOutTS(PL_INFO,"variance of the two populations\n");
    printOutTS(PL_INFO,"  do not need to be equal.\n");
    printOutTS(PL_INFO,"  T-value is calculated and used to find ");
    printOutTS(PL_INFO,"the p-value from the lookup\n");
    printOutTS(PL_INFO,"  table. If, e.g., p-value > 0.05, then reject ");
    printOutTS(PL_INFO,"the NULL HYPOTHESIS.)\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO," SIZE OF DATASET 1 = %d\n", length1);
    printOutTS(PL_INFO,"      mean = %e\n", mean1);
    printOutTS(PL_INFO,"      s.d. = %e\n", sqrt(var1));
    printOutTS(PL_INFO," SIZE OF DATASET 2 = %d\n", length2);
    printOutTS(PL_INFO,"      mean = %e\n", mean2);
    printOutTS(PL_INFO,"      s.d. = %e\n", sqrt(var1));
    printOutTS(PL_INFO," DEGREE OF FREEDOM = %d\n", dofs);
    printOutTS(PL_INFO," Std. Err. of mean = %e\n", stdev);
    printOutTS(PL_INFO," T-statistic       = %e\n", tval);
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
       " What ALPHA means: larger ALPHA ==> wider acceptance interval\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
       " ALPHA LEVEL (2-sided)   CUTOFF         CONCLUSION\n");
    printOutTS(PL_INFO, " 0.05                     %5.3f",localTable[0]);
    if (tval <= localTable[0]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printOutTS(PL_INFO, " 0.025                    %5.3f",localTable[1]);
    if (tval <= localTable[1]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printOutTS(PL_INFO, " 0.0125                   %5.3f",localTable[2]);
    if (tval <= localTable[2]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printOutTS(PL_INFO, " 0.005                    %5.3f",localTable[3]);
    if (tval <= localTable[3]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printOutTS(PL_INFO, " 0.0025                   %5.3f",localTable[4]);
    if (tval <= localTable[4]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printOutTS(PL_INFO, " 0.0005                   %5.3f",localTable[5]);
    if (tval <= localTable[5]) 
         printOutTS(PL_INFO,"          ACCEPT H0\n");
    else printOutTS(PL_INFO,"          REJECT H0\n");
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }
  else if (pLevel > 0)
  {
    if (tval <= localTable[0]) 
    {
      printOutTS(PL_INFO, 
         "          ACCEPT H0 (%e < %e)\n",tval,localTable[0]);
      printOutTS(PL_INFO, 
         "          Mean, Std. dev 1 = %e  %e\n",mean1,sqrt(var1));
      printOutTS(PL_INFO, 
         "          Mean, Std. dev 2 = %e  %e\n",mean2,sqrt(var2));
    }
    else
    {
      printOutTS(PL_INFO, 
         "          REJECT H0 (%e > %e)\n",tval,localTable[0]);
      printOutTS(PL_INFO, 
         "          Mean, Std. dev 1 = %e  %e\n",mean1,sqrt(var1));
      printOutTS(PL_INFO, 
         "          Mean, Std. dev 2 = %e  %e\n",mean2,sqrt(var2));
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ return (return value = 0 ==> accept hypothesis) 
  //**/ ---------------------------------------------------------------
  if      (tval < -localTable[0]) retval = -1.0;
  else if (tval >  localTable[0]) retval = 1.0;
  else                            retval = 0.0;
  return retval;
}

// ************************************************************************
// perform Kolmogorov-Smirnov Test
// ------------------------------------------------------------------------
double TwoSampleAnalyzer::KSAnalyze(int length1, double *Y1, int length2,
                                    double *Y2, int pLevel)
{
  int    ss, ii, nCount, index;
  double DStat, dtemp, localTable[KSMaxSignificanceLevels];
  FILE   *outfile;

  //**/ ---------------------------------------------------------------
  //**/ sort the two list
  //**/ ---------------------------------------------------------------
  sortDbleList(length1, Y1);
  sortDbleList(length2, Y2);

  //**/ ---------------------------------------------------------------
  //**/ write to matlab/scilab file
  //**/ ---------------------------------------------------------------
  if (pLevel > 0)
  {
    if (plotScilab())
    {
      outfile = fopen("scilabks.sci", "w");
      if (outfile != NULL)
           fprintf(outfile, "// Kolmogorov Smirnov test\n");
      else printOutTS(PL_INFO, "INFO: cannot open file scilabks.sci\n");
    }
    else
    {
      outfile = fopen("matlabks.m", "w");
      if (outfile != NULL)
           fprintf(outfile, "%% Kolmogorov Smirnov test\n");
      else printOutTS(PL_INFO, "INFO: cannot open file matlabks.m\n");
    }
    if (outfile != NULL)
    { 
      fprintf(outfile, "X1 = [\n");
      fprintf(outfile, "%e\n", Y1[0]);
      for (ss = 1; ss < length1; ss++) 
      {
        fprintf(outfile, "%e\n", Y1[ss-1]);
        fprintf(outfile, "%e\n", Y1[ss]);
      }
      fprintf(outfile, "];\n");
      fprintf(outfile, "C1 = [\n");
      nCount = 0;
      fprintf(outfile, "%d\n", nCount);
      for (ss = 1; ss < length1; ss++) 
      {
        fprintf(outfile, "%e\n", nCount / (double) length1);
        fprintf(outfile, "%e\n", nCount / (double) length1);
        nCount++;
      }
      fprintf(outfile, "];\n");
      fprintf(outfile, "X2 = [\n");
      fprintf(outfile, "%e\n", Y2[0]);
      for (ss = 1; ss < length2; ss++) 
      {
        fprintf(outfile, "%e\n", Y2[ss-1]);
        fprintf(outfile, "%e\n", Y2[ss]);
      }
      fprintf(outfile, "];\n");
      fprintf(outfile, "C2 = [\n");
      nCount = 0;
      fprintf(outfile, "%d\n", nCount);
      for (ss = 1; ss < length2; ss++) 
      {
        fprintf(outfile, "%e\n", nCount / (double) length2);
        fprintf(outfile, "%e\n", nCount / (double) length2);
        nCount++;
      }
      fprintf(outfile, "];\n");
      fprintf(outfile, "plot(X1,C1,X2,C2)\n");
      fwritePlotAxes(outfile);
      fwritePlotXLabel(outfile, "Input Value");
      fwritePlotYLabel(outfile, "Cumulative Probability");
      fwritePlotTitle(outfile, "Kolgomorov Smirnov Test");
      fclose(outfile);
      if (plotScilab())
           printOutTS(PL_INFO, "KSAnalyzer: scilabks.sci created.\n");
      else printOutTS(PL_INFO, "KSAnalyzer: matlabks.m created.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ calculate D-statistic 
  //**/ ---------------------------------------------------------------
  DStat = 0.0;
  for (ss = 0; ss < length1; ss++) 
  {
    index = binarySearchDble(Y1[ss], Y2, length2);
    if (index < 0) index = - index - 1;
    dtemp = ss / (double) length1 - index / (double) length2;
    if (dtemp < 0.0) dtemp = - dtemp;
    if (dtemp > DStat) DStat = dtemp;
  }
  if (length1 > 35)
  {
    localTable[0] = 1.07 / sqrt((double) length1);
    localTable[1] = 1.14 / sqrt((double) length1);
    localTable[2] = 1.22 / sqrt((double) length1);
    localTable[3] = 1.36 / sqrt((double) length1);
    localTable[4] = 1.63 / sqrt((double) length1);
  }
  else
  {
    for (ii = 0; ii < KSMaxSignificanceLevels; ii++)
      localTable[ii] = KSTable[length1-1][ii];
  }
  if (pLevel > 0)
  {
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, 
        " KOLMOGOROV SMIRNOV TWO-SAMPLE TEST NONPARAMETRIC TEST: \n\n");
    printOutTS(PL_INFO, 
        " NULL HYPOTHESIS H0 : datasets do not differ significantly.\n");
    printOutTS(PL_INFO, 
        " ALTERNATIVE HYPOTHESIS : datasets differ significantly.\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
        " ALPHA LEVEL : level of significance (the higher the better).\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, " SIZE OF DATASET 1 = %d\n", length1);
    printOutTS(PL_INFO, " SIZE OF DATASET 2 = %d\n", length2);
    printOutTS(PL_INFO, " KS D-Statistic    = %e\n", DStat);
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, " ALPHA LEVEL       CUTOFF         CONCLUSION\n");
    printOutTS(PL_INFO, " 20%%              %5.3f", localTable[0]);
    if (DStat <= localTable[0]) 
         printOutTS(PL_INFO,"           ACCEPT H0\n");
    else printOutTS(PL_INFO,"           REJECT H0\n");
    printOutTS(PL_INFO, " 15%%              %5.3f",localTable[1]);
    if (DStat <= localTable[1]) 
         printOutTS(PL_INFO,"           ACCEPT H0\n");
    else printOutTS(PL_INFO,"           REJECT H0\n");
    printOutTS(PL_INFO, " 10%%              %5.3f",localTable[2]);
    if (DStat <= localTable[2]) 
         printOutTS(PL_INFO,"           ACCEPT H0\n");
    else printOutTS(PL_INFO,"           REJECT H0\n");
    printOutTS(PL_INFO, "  5%%              %5.3f",localTable[3]);
    if (DStat <= localTable[3]) 
         printOutTS(PL_INFO,"           ACCEPT H0\n");
    else printOutTS(PL_INFO,"           REJECT H0\n");
    printOutTS(PL_INFO, "  1%%              %5.3f",localTable[4]);
    if (DStat <= localTable[4]) 
         printOutTS(PL_INFO,"           ACCEPT H0\n");
    else printOutTS(PL_INFO,"           REJECT H0\n");
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ return D-statistic
  //**/ ---------------------------------------------------------------
  return (DStat);
}

// ************************************************************************
// perform Mann-Whitney Test
// ------------------------------------------------------------------------
double TwoSampleAnalyzer::MWAnalyze(int length1, double *Y1, int length2,
                                    double *Y2, int pLevel)
{
  int    ss, length, index, ptype;
  double *Y, S1, S2, U1, U2, sig12, cval, U, Um, Uh, Z, alpha;
  double *dSortList, mean, std, dOne=1.0, dZero=0.0, *Y1L, *Y2L;
  PDFManager *pdfman;
  psMatrix   corMat;
  psVector   vecIn, vecOut, vecUpper, vecLower, vecY, vecY1L, vecY2L;
  psVector   vecDSort;

  //**/ ---------------------------------------------------------------
  //**/ sort the two lists
  //**/ ---------------------------------------------------------------
  vecY.setLength(length1+length2);
  vecY1L.setLength(length1);
  vecY2L.setLength(length2);
  for (ss = 0; ss < length1; ss++) vecY[ss] = vecY1L[ss];
  for (ss = 0; ss < length2; ss++) vecY[length1+ss] = vecY2L[ss];

  vecDSort.setLength(length1+length2);
  for (ss = 0; ss < length1+length2; ss++) vecDSort[ss] = (double) ss;
  sortDbleList2(length1+length2,vecY.getDVector(),vecDSort.getDVector());
  index = 0;
  for (ss = 0; ss < length1+length2; ss++) 
    if (vecDSort[ss] < length1) vecY1L[index++] = ss + 1;
  index = 0;
  for (ss = 0; ss < length1+length2; ss++) 
    if (vecDSort[ss] >= length1) vecY2L[index++] = ss + 1;

  //**/ ---------------------------------------------------------------
  //**/ compute statistics 
  //**/ ---------------------------------------------------------------
  length = length1 + length2;
  S1 = S2 = 0.0;
  for (ss = 0; ss < length1; ss++) S1 += vecY1L[ss]; 
  for (ss = 0; ss < length2; ss++) S2 += vecY2L[ss]; 
  U1 = S1 - 0.5 * length1 * (length1 + 1);
  U2 = S2 - 0.5 * length2 * (length2 + 1);
  U  = (U1 < U2) ? U1 : U2; 
  Uh = (U1 > U2) ? U1 : U2; 
  Um = 0.5 * length1 * length2;
  sig12 = sqrt(length1 * length2 * (length + 1.0) / 12.0);
  Z  = (U - Um) / sig12;
  alpha = 0.05;
  cval = 1.0 - 0.5 * alpha;
  corMat.setDim(1,1);
  corMat.setEntry(0,0, 1.0e0);
  pdfman = new PDFManager();
  ptype = PSUADE_PDF_NORMAL;
  mean = dZero;
  std  = dOne;
  pdfman->initialize(1, &ptype, &mean, &std, corMat, NULL, NULL);
  vecOut.setLength(1);
  vecIn.load(1, &Z);
  vecUpper.load(1, &dOne);
  vecLower.load(1, &dZero);
  pdfman->getCDF(1, vecIn, vecOut, vecLower, vecUpper);
  delete pdfman;
  cval = vecOut[0];
  //**/ why here
  //**/ cval = cval * 10.0 - 5.0;
  if (pLevel > 0)
  {
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, " MANN-WHITNEY TEST:\n");
    printOutTS(PL_INFO, 
       " NULL HYPOTHESIS H0 : datasets do not differ significantly.\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO," SIZE OF DATASET 1   = %d\n", length1);
    printOutTS(PL_INFO," SIZE OF DATASET 2   = %d\n", length2);
    printOutTS(PL_INFO," MW TEST U statistic = %e (high = %e) \n",U,Uh);
    printOutTS(PL_INFO," MW TEST Z statistic = %e\n", Z);
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
       " Significance level                = 0.05 (one-sided)\n");
    printOutTS(PL_INFO," Critical value (Prob[0.05,0.95])  = %e\n",cval);
    if (cval >= 0.05 && cval <= 0.95)
         printOutTS(PL_INFO, " ACCEPT the NULL HYPOTHESIS.\n");
    else printOutTS(PL_INFO, " REJECT the NULL HYPOTHESIS.\n");
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }
  return (PABS(Z)- cval);
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
TwoSampleAnalyzer& TwoSampleAnalyzer::operator=(const TwoSampleAnalyzer &)
{
  printOutTS(PL_ERROR, 
       "TwoSampleAnalyzer operator= ERROR: operation not supported.\n");
  exit(1);
   return (*this);
}

