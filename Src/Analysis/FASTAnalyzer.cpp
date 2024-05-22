// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public 
// License.
//
// PSUADE is free software; you can redistribute it and/or modify it under 
// the terms of the GNU Lesser General Public License (as published by the 
// Free Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of 
// the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class FASTAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "FASTAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

//**/ ---------------------------------------------------------------------
//**/ al definitions
//**/ ---------------------------------------------------------------------
#define  PSUADE_FAST_MaxDimension  50

static unsigned long
PSUADE_FAST_OMEGA[PSUADE_FAST_MaxDimension] =
{
      1,    3,    1,    5,   11,    1,   17,   23,   19,   25,
     41,   31,   23,   87,   67,   73,   85,  143,  149,   99,
    119,  237,  267,  283,  151,  385,  157,  215,  449,  163,
    337,  253,  375,  441,  673,  773,  875,  873,  587,  849,
    623,  637,  891,  943, 1171, 1225, 1335, 1725, 1663, 2019
};

static unsigned long
PSUADE_FAST_DELTA[PSUADE_FAST_MaxDimension] =
{
      4,    8,    6,   10,   20,   22,   32,   40,   38,   26,
     56,   62,   46,   76,   96,   60,   86,  126,  134,  112,
     92,  128,  154,  196,   34,  416,  106,  208,  328,  198,
    382,   88,  348,  186,  140,  170,  284,  568,  302,  438,
    410,  248,  448,  388,  596,  216,  100,  488,  166,    0
};
                                                                                
// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
FASTAnalyzer::FASTAnalyzer() : Analyzer()
{
  nInputs_      = 0;
  M_            = 0;
  FASTvariance_ = 0.0; 
  setName("FAST");
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
FASTAnalyzer::~FASTAnalyzer() 
{ 
  VecFourierCoefs_.clean();
} 

// ************************************************************************ 
// perform analysis 
// ------------------------------------------------------------------------
double FASTAnalyzer::analyze(aData &adata)
{
  int    ii, N2, maxInd, ss, count;
  double *fourierCoefs, *fourierCoefs2, retdata, maxData, fsum;
  psVector vecYY, vecFourierCoefs, vecFourierCoefs2;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs    = adata.nInputs_;
  nInputs_       = nInputs;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *Y      = adata.sampleOutputs_;
  int outputID   = adata.outputID_;
  if (adata.inputPDFs_ != NULL)
  {
    count = 0;
    for (ii = 0; ii < nInputs; ii++) count += adata.inputPDFs_[ii];
    if (count > 0)
    {
      printOutTS(PL_WARN,"FAST INFO: Some inputs have non-uniform ");
      printOutTS(PL_WARN,"PDFs, but they are not relevant\n");
      printOutTS(PL_WARN,"           in this analysis.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking and diagnostics
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR,"FAST ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs > PSUADE_FAST_MaxDimension)
  {
    printOutTS(PL_ERROR,
         "FAST ERROR: Input dimension needs to be <= 50.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    return PSUADE_UNDEFINED;
  } 
  if (outputID < 0 || outputID >= nOutputs)
  {
    printOutTS(PL_ERROR, "FAST ERROR: Invalid outputID.\n");
    printOutTS(PL_ERROR, "    outputID = %d\n", outputID+1);
    return PSUADE_UNDEFINED;
  } 
  VecFourierCoefs_.clean();
   
  //**/ ---------------------------------------------------------------
  //**/ compute Fourier coeficients 
  //**/ ---------------------------------------------------------------
  vecYY.setLength(nSamples);
  vecFourierCoefs.setLength(nInputs+1);
  for (ss = 0; ss < nSamples; ss++) vecYY[ss] = Y[ss*nOutputs+outputID];
  printf("WARNING: No checking is done to ensure this sample is FAST.\n");
  int M = computeCoefficents(nSamples, nInputs, vecYY.getDVector(), 
                             vecFourierCoefs, printLevel);
  if (M < 0) return 0.0; 
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO, 
   "Fourier Amplitude Sensitivity Test (FAST) coefficients (Normalized)\n");
  printOutTS(PL_INFO, 
   "(to estimate the Sobol' first-order sensitivity indices)\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INTERACTIVE, "* M = %d\n", M);
  fsum = 0.0;
  for (ii = 0; ii < nInputs; ii++)
  {
     printOutTS(PL_INFO,
       "Input %3d: sensitivity index = %10.3e (unnormalized = %10.3e)\n",
               ii+1, vecFourierCoefs[ii],
               vecFourierCoefs[ii]*vecFourierCoefs[nInputs]);
     fsum += vecFourierCoefs[ii];
  }
  printOutTS(PL_INFO,"Sum of FAST coefficients = %11.3e\n", fsum);
  printOutTS(PL_INFO, 
       "Output variance          = %11.3e\n", 
       vecFourierCoefs[nInputs]);

  //save Fourier coefficients
  VecFourierCoefs_.setLength(nInputs_);
  M_ = M;
  for (ii=0; ii<nInputs_; ii++) 
    VecFourierCoefs_[ii] = vecFourierCoefs[ii];
  FASTvariance_ = vecFourierCoefs[nInputs];
  printEquals(PL_INFO, 0);

  return retdata;
}

// ************************************************************************
// calculate frequencies
// ------------------------------------------------------------------------
int FASTAnalyzer::calculateOmegas(int nInputs, int nSamples, int *omegas)
{
  omegas[0] = PSUADE_FAST_OMEGA[nInputs-1];
  for (int ii = 1; ii < nInputs; ii++)
    omegas[ii] = omegas[ii-1] + PSUADE_FAST_DELTA[nInputs-1-ii];
  if (nInputs == 1) omegas[0] = 11;
  else if (nInputs == 2)
  {
    omegas[0] = 11;
    omegas[1] = 21;
  }
  else if (nInputs == 3)
  {
    //**/ this set will make sure that
    //**/ sum_{i=1}^3 a_i omegas[i] != 0 for sum_{i=1}^3 |a_i| <= M+1
    omegas[0] = 11;
    omegas[1] = 21;
    omegas[2] = 29;
  }
  return 0;
}

// ************************************************************************ 
// compute Fourier coefficients
// ------------------------------------------------------------------------
int FASTAnalyzer::computeCoefficents(int nSamples, int nInputs, double *Y,
                                     psVector &vecFourierCoefs, int flag)
{
  int    M=4, ii, jj, ss, N;
  double ps_pi=3.14159, ds, freq;
  double fastReal, fastImag, fastCoef, dataReal, dataImag;
  psIVector vecOmegas;
  psVector  vecFastReal, vecFastImag;

  //**/ ---------------------------------------------------------------
  //**/ compute frequencies
  //**/ ---------------------------------------------------------------
  vecOmegas.setLength(nInputs);
  calculateOmegas(nInputs, nSamples, vecOmegas.getIVector());
  if ((nSamples - 1) / (2 * vecOmegas[nInputs-1]) < 4)
  {
    printOutTS(PL_ERROR,"nSamples = %4d too small\n", nSamples);
    printOutTS(PL_ERROR,"Is it a FAST sample? \n");
    return -1;
  } 
   
  //**/ ---------------------------------------------------------------
  //**/ compute coefficients
  //**/ ---------------------------------------------------------------
  vecFastReal.setLength(M*nInputs);
  vecFastImag.setLength(M*nInputs);
  for (ii = 0; ii < M*nInputs; ii++)
    vecFastReal[ii] = vecFastImag[ii] = 0.0;
  ds = ps_pi / (double) nSamples;
  
  for (ii = 0; ii < M; ii++)
  {
    for (jj = 0; jj < nInputs; jj++)
    {
      for (ss = 0; ss < nSamples; ss++)
      {
        freq = - ps_pi / 2.0 + ds * 0.5 * (2 * ss + 1);
        vecFastReal[ii*nInputs+jj] += 
           Y[ss]*cos((ii+1)*vecOmegas[jj]*freq)*ds;
        vecFastImag[ii*nInputs+jj] += 
           Y[ss]*sin((ii+1)*vecOmegas[jj]*freq)*ds;
      }
      for (ss = 0; ss < (nSamples-1)/2; ss++)
      {
        freq = - ps_pi + ds * (ss + 1);
        vecFastReal[ii*nInputs+jj] += 
           Y[(nSamples+1)/2-ss-2]*cos((ii+1)*vecOmegas[jj]*freq)*ds;
        vecFastImag[ii*nInputs+jj] += 
           Y[(nSamples+1)/2-ss-2]*sin((ii+1)*vecOmegas[jj]*freq)*ds;
      }
      for (ss = 0; ss < (nSamples-1)/2; ss++)
      {
        freq = ps_pi - ds * (ss + 1);
        vecFastReal[ii*nInputs+jj] += 
           Y[(nSamples+1)/2+ss]*cos((ii+1)*vecOmegas[jj]*freq)*ds;
        vecFastImag[ii*nInputs+jj] += 
           Y[(nSamples+1)/2+ss]*sin((ii+1)*vecOmegas[jj]*freq)*ds;
      }
    }
  }
  for (ii = 0; ii < M*nInputs; ii++)
  {
    vecFastReal[ii] /= (2.0 * ps_pi);
    vecFastImag[ii] /= (2.0 * ps_pi);
  }
  for (jj = 0; jj < nInputs; jj++)
  {
    vecFourierCoefs[jj] = 0.0;
    for (ii = 0; ii < M; ii++)
      vecFourierCoefs[jj] += 
          (vecFastReal[ii*nInputs+jj]*vecFastReal[ii*nInputs+jj]);
    for (ii = 0; ii < M; ii++)
      vecFourierCoefs[jj] += 
          (vecFastImag[ii*nInputs+jj]*vecFastImag[ii*nInputs+jj]);
    vecFourierCoefs[jj] *= 2.0;
  }

  //**/ ---------------------------------------------------------------
  //**/ compute total contribution
  //**/ ---------------------------------------------------------------
  N = M * vecOmegas[nInputs-1];
  fastReal = fastImag = 0.0;
  if (flag >= 3)
  {
    for (jj = 0; jj < nInputs; jj++)
      printOutTS(PL_INFO,"FAST: input %4d fundamental frequency = %d\n",
                 jj+1, vecOmegas[jj]);
  }
  for (ii = 0; ii < N; ii++)
  {
    dataReal = dataImag = 0.0;
    for (ss = 0; ss < nSamples; ss++)
    {
      freq = - ps_pi / 2.0 + ds * 0.5 * (2 * ss + 1);
      dataReal += Y[ss]*cos((ii+1)*freq)*ds;
      dataImag += Y[ss]*sin((ii+1)*freq)*ds;
    }
    for (ss = 0; ss < (nSamples-1)/2; ss++)
    {
      freq = - ps_pi + ds * (ss + 1);
      dataReal += Y[(nSamples+1)/2-ss-2]*cos((ii+1)*freq)*ds;
      dataImag += Y[(nSamples+1)/2-ss-2]*sin((ii+1)*freq)*ds;
    }
    for (ss = 0; ss < (nSamples-1)/2; ss++)
    {
      freq = ps_pi - ds * (ss + 1);
      dataReal += Y[(nSamples+1)/2+ss]*cos((ii+1)*freq)*ds;
      dataImag += Y[(nSamples+1)/2+ss]*sin((ii+1)*freq)*ds;
    }
    dataReal /= (2.0 * ps_pi);
    dataImag /= (2.0 * ps_pi);
    fastReal += dataReal * dataReal;
    fastImag += dataImag * dataImag;
    if (flag >= 3)
    {
      printOutTS(PL_INFO,
         "FAST: frequency %5d : data = %9.1e (%9.1e %9.1e) ",
         ii+1,dataReal*dataReal+dataImag*dataImag,dataReal,dataImag);
      for (jj = 0; jj < nInputs; jj++)
        if ((ii + 1) / vecOmegas[jj] * vecOmegas[jj] == (ii + 1))
          printOutTS(PL_INFO, "(%4d) ", jj+1);
      printOutTS(PL_INFO, "\n");
    }
  }
  fastCoef = 2.0 * (fastReal + fastImag);

  //**/ ---------------------------------------------------------------
  //**/ scale and clean up
  //**/ ---------------------------------------------------------------
  for (jj = 0; jj < nInputs; jj++) vecFourierCoefs[jj] /= fastCoef;
  vecFourierCoefs[nInputs] = fastCoef;
  return M;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
FASTAnalyzer& FASTAnalyzer::operator=(const FASTAnalyzer &)
{
  printOutTS(PL_ERROR, "FAST operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int FASTAnalyzer::get_nInputs()
{
  return nInputs_;
}
int FASTAnalyzer::get_M()
{
  return M_;
}
double *FASTAnalyzer::get_fourierCoefs()
{
  psVector vecT;
  vecT = VecFourierCoefs_;
  return vecT.takeDVector();
}
double FASTAnalyzer::get_FASTvariance()
{
  return FASTvariance_;
}

