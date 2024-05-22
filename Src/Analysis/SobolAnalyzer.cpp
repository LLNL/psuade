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
// Functions for the class SobolAnalyzer (TS, S, interactions)  
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "SobolAnalyzer.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
SobolAnalyzer::SobolAnalyzer() : Analyzer(), nInputs_(0)
{
  setName("SOBOL");
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
SobolAnalyzer::~SobolAnalyzer()
{
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
//**/ Note: The following algorithm uses the Sobol' method based on Monte
//**/    Carlo simulations. The formula is:
//**/      VCE(X) = sum_{i=1}^N [f(X,Z)f(X,Z') - [1/N sum_{i=1}^N f(X,Z)]^2
//**/    where N is the sample size
//**/          Z are inputs other than X
//**/    (ref: "Different numerical estimators for main effect global 
//**/          sensitivity indices by Sergei Kucherenko and Shufang Song).
// ------------------------------------------------------------------------
//**/ Improved formula by Kucherenko (not implemented)
//**/      VCE(X) = sum_{i=1}^N [f(X,Z) (f(X,Z') - f(X',Z'))]
//**/ using the idea that f0^2 = 1/N sum_{i=1}^N f(X,Z) f(X',Z')
// ------------------------------------------------------------------------
double SobolAnalyzer::analyze(aData &adata)
{
  int     count, repID, iD, ii, nReps, ss, errCount;
  double  xtemp1, xtemp2, tau, dtemp;
  psVector VecY, vecMeans, vecStds, vecModMeans;

  //**/ ---------------------------------------------------------------
  //**/ display header
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
  {
    printAsterisks(PL_INFO, 0);
    if (order_ == 1)
      printf("*             Sensitivity Analysis on Sobol Samples\n"); 
    if (order_ == 2)
    {
      printf("*             Sensitivity Analysis (2nd-order) ");
      printf("on Sobol Samples\n"); 
    }
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  nInputs_ = nInputs;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *xIn    = adata.sampleInputs_;
  double *yIn    = adata.sampleOutputs_;
  int    *sIn    = adata.sampleStates_;
  int    outputID = adata.outputID_;
  int    printLevel_ = adata.printLevel_;

  //**/ ---------------------------------------------------------------
  //**/ write sample data to a file for diagnostics
  //**/ ---------------------------------------------------------------
  if (psConfig_.DiagnosticsIsOn())
  {
    FILE *fp = fopen("SobolSample","w");
    for (iD = 0; iD < nSamples; iD++)
    {
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fp,"%10.4e ",xIn[iD*nInputs+ii]);
      fprintf(fp,"%10.4e\n",yIn[iD]);
    }
    fclose(fp);
    printf("**** SobolSample has the Sobol sample (inputs/outputs)\n");
    printf("NOTE: This file is created in diagnostics mode.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0 || 
      outputID < 0 || outputID >= nOutputs)
  {
    printOutTS(PL_ERROR,"SobolAnalyzer ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    printOutTS(PL_ERROR,"    outputID = %d\n", outputID+1);
    return PSUADE_UNDEFINED;
  } 

  //**/ ---------------------------------------------------------------
  //**/ clean up first
  //**/ ---------------------------------------------------------------
  VecModMeans_.clean();
  VecStds_.clean();
  VecS_.clean();
  VecST_.clean();
  VecPE_.clean();

  //**/ ---------------------------------------------------------------
  //**/ check for valid samples (if some sample points are invalid, 
  //**/ that is okay - maybe those are not feasible points)
  //**/ ---------------------------------------------------------------
  VecY.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) VecY[ss] = yIn[nOutputs*ss+outputID];
  count = nSamples - VecY.countUndefined();
  if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
  {
    printOutTS(PL_INFO,
       "SobolAnalyzer INFO: There are %d sample points.\n",nSamples);
  }
  if ((psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn()) &&
      count < nSamples)
    printOutTS(PL_INFO,
      "SobolAnalyzer INFO: There are %d valid sample points.\n",count);

  //**/ ---------------------------------------------------------------
  //**/ if second order, call another function
  //**/ ---------------------------------------------------------------
  if (order_ == 2) return analyze2(adata);

  //**/ ---------------------------------------------------------------
  //**/ check if the Sobol sampling plan has been used
  //**/ ---------------------------------------------------------------
  nReps = nSamples / (nInputs + 2);
  if ((nReps * (nInputs+2)) == nSamples)
  {
    //**/ there should be nReps blocks of nInputs+2 samples each
    for (ss = 0; ss < nSamples; ss+=(nInputs+2))
    {
      errCount = 0;
      //**/ starting from the second sample in the block
      for (iD = 1; iD <= nInputs; iD++)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          //**/ xtemp1 = either the ii-th input of the first sample
          //**/          or the ii-th input of the last sample 
          if (ii == (iD-1))
               xtemp1 = xIn[(ss+nInputs+1)*nInputs+ii]; 
          else xtemp1 = xIn[ss*nInputs+ii]; 
          //**/ xtemp2 = the i-th input in the current sample
          xtemp2 = xIn[(ss+iD)*nInputs+ii]; 
          //**/ the ii-th input of the current sample must be the same
          //**/ as the ii-th input of the first or the last sample in
          //**/ the block
          if (xtemp1 != xtemp2) errCount++;
        }
      }
      if (errCount > 0)
      {
        printOutTS(PL_ERROR,
             "SobolAnalyzer ERROR: Invalid sample (%d,%d)\n",
             ss, errCount);
        printOutTS(PL_ERROR, 
             "SobolAnalyzer requires Sobol samples.\n");
        return PSUADE_UNDEFINED;
      }
    }
  }
  else
  {
    printOutTS(PL_ERROR,"SobolAnalyzer ERROR: Invalid sample size.\n");
    printOutTS(PL_ERROR,"SobolAnalyzer requires Sobol samples.\n");
    return PSUADE_UNDEFINED;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ set up for and call MOAT analysis
  //**/ (Since Sobol sample satisfies MOAT property, it makes sense to
  //**/ run it through MOAT analysis)
  //**/ NOTE: This is performed only if there is no undefined points
  //**/ ---------------------------------------------------------------
  vecMeans.setLength(nInputs);
  vecModMeans.setLength(nInputs);
  vecStds.setLength(nInputs);

  //**/ do it if no sample has been filtered out and print level > 1
  if (count == nSamples && printLevel_ > 1)
  {
    MOATAnalyze(nInputs,nSamples,xIn,VecY.getDVector(),xLower,xUpper,
                vecMeans.getDVector(), vecModMeans.getDVector(),
                vecStds.getDVector());

    if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
    {
      printOutTS(PL_INFO, "Sobol-OAT (One-at-a-time) Analysis : \n");
      for (ii = 0; ii < nInputs; ii++)
        printOutTS(PL_INFO, 
             "Input %3d (mod. mean & std) = %12.4e %12.4e\n",
             ii+1, vecModMeans[ii], vecStds[ii]);
      printEquals(PL_INFO, 0);
    }
  }
  //**/ save MOAT means & stds (even though they may not be computed)
  VecStds_ = vecStds;
  VecModMeans_ = vecModMeans;

  //**/ ---------------------------------------------------------------
  //**/ compute mean (based on M2) and variance
  //**/ (meanSq is needed to estimate E^2 for first-order indices)
  //**/ ---------------------------------------------------------------
  //**/ need to reload VecY since it has been modified by MOATAnalysis
  VecY.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) 
    VecY[ss] = yIn[nOutputs*ss+outputID];
  double meanM2 = 0.0;
  count = 0;
  for (repID = 0;  repID < nReps; repID++)
  {
    if (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED) 
    {
      meanM2 += VecY[repID*(nInputs+2)]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
               "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanM2 /= ((double) (count));
  double varM2 = 0.0;
  for (repID = 0;  repID < nReps; repID++)
    if (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED) 
      varM2 += ((VecY[repID*(nInputs+2)] - meanM2) * 
                (VecY[repID*(nInputs+2)] - meanM2)); 
  varM2 = varM2 / (double) (count-1.0);
  if (varM2 == 0)
  {
    printOutTS(PL_ERROR, 
               "SobolAnalyzer ERROR: Sample variance = 0.0.\n");
    exit(1);
  }
  double meanSq = 0.0;
  count = 0;
  for (repID = 0;  repID < nReps; repID++)
  {
    if (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED && 
        VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED) 
    {
      meanSq += VecY[repID*(nInputs+2)] * 
                VecY[repID*(nInputs+2)+nInputs+1]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanSq /= ((double) (count));

  //**/ ---------------------------------------------------------------
  //**/ perform Sobol analysis
  //**/ ---------------------------------------------------------------
  double meanSq2, var2;
  VecS_.setLength(nInputs_);
  VecST_.setLength(nInputs_);
  VecPE_.setLength(nInputs_);
  Variance_ = varM2;
  for (ii = 0; ii < nInputs; ii++)
  {
    //**/------------------------------------------------------------
    //**/ compute total sensitivity index (TSI)
    //**/------------------------------------------------------------
    //**/ for each block of nInputs+2 points
    //**/ the first sample in the block is M2
    //**/ the second row is M2 with the first input perturbed by M1
    //**/ ...
    //**/ the last sample in the block is M1
    //**/ tau = sum (F(x1,.. xi, ..xn) F(x1, ...xi*, ..xn)) 
    tau = 0.0;
    count = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED))
      {
        tau += VecY[repID*(nInputs+2)] * VecY[repID*(nInputs+2)+ii+1];
        count++;
      }
    }
    if (count <= 1)
    {
      printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: too few valid sample points for TSI.\n");
      exit(1);
    }
    //**/ compute total index (normalized)
    tau /= ((double) count);
    VecST_[ii] = 1.0 - (tau - meanM2 * meanM2) / varM2; 
    //if (VecST_[ii] < 0) VecST_[ii] = 0;

    //**/------------------------------------------------------------
    //**/ compute first-order sensitivity index VCE(Y)
    //**/------------------------------------------------------------
#if 0
    //**/ original Sobol' method
    //**/ VCE(X) = sum_{i=1}^N [f(X,Z)f(X,Z') - [1/N sum_{i=1}^N f(X,Z)]^2
    tau = 0.0;
    count = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      //**/ tau = sum (F(x1,.. xi, ..xn) F(x1*, ...xi, ..xn*)) 
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        tau += VecY[repID*(nInputs+2)+nInputs+1] * 
               VecY[repID*(nInputs+2)+ii+1]; 
        count++;
      }
    }
    if (count <= 0)
    {
      printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: too few valid sample points for VCE.\n");
      exit(1);
    }
    //**/ compute main effect index (normalized)
    tau /= ((double) count);
    VecS_[ii] = (tau - meanSq) / varM2; 
    if (VecS_[ii] < 0) VecS_[ii] = 0;
#else
    //**/ Kucherncho improved method
    //**/ VCE(X) = sum_{i=1}^N [f(X,Z) (f(X,Z') - f(X',Z'))]
    tau = 0.0;
    count = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      //**/ tau = sum (F(x1,.xi,.xn) [F(x1*,.xi,.xn*)-f(x1,.xi,.xn)] 
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED))
      {
        tau += VecY[repID*(nInputs+2)+nInputs+1] * 
               (VecY[repID*(nInputs+2)+ii+1] -  
                VecY[repID*(nInputs+2)]);  
        count++;
      }
    }
    if (count <= 0)
    {
      printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: too few valid sample points for VCE.\n");
      exit(1);
    }
    tau /= ((double) count);
    VecS_[ii] = tau / varM2; 
    if (VecS_[ii] < 0) VecS_[ii] = 0;
#endif

    //**/------------------------------------------------------------
    //**/ compute probable error (the formula in Sobol is wrong) 
    //**/------------------------------------------------------------
    meanSq2 = 0.0;
    count  = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        meanSq2 += (VecY[repID*(nInputs+2)+nInputs+1]*
                    VecY[repID*(nInputs+2)+ii+1]);
        count++;
      }
    }
    meanSq2 = meanSq2 / (double) count;
    var2 = 0.0;
    for (repID = 0;  repID < nReps; repID++)
    {
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        dtemp = VecY[repID*(nInputs+2)+ii+1] * 
                VecY[repID*(nInputs+2)+nInputs+1];
        var2 += pow(dtemp - meanSq2, 2.0);
      }
    }
    var2 = var2 / (double) (count - 1.0);
    VecPE_[ii] = 0.6745 * sqrt(var2) / sqrt(1.0 * count);
  }

  //printOutTS(PL_INFO, 
  //     "Sobol Analysis (ST: total sensitivity, PE: probable error):\n");
  //for (ii = 0; ii < nInputs; ii++)
  //  printOutTS(PL_INFO, "Input %3d (S, ST, PE) = %12.4e %12.4e %12.4e\n",
  //         ii+1, VecS_[ii], VecST_[ii], VecPE_[ii]);
  if (psConfig_.InteractiveIsOn() && psConfig_.DiagnosticsIsOn())
  {
    printOutTS(PL_INFO, 
         "Sobol Analysis (analyze S: First-Order; ST: Total-Order):\n");
    for (ii = 0; ii < nInputs; ii++)
      printf("Input %3d (S, ST) = %12.4e %12.4e (normalized)\n",
             ii+1, VecS_[ii], VecST_[ii]);
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printf("Input %3d (S, ST) = %12.4e %12.4e (unnormalized)\n",
             ii+1, VecS_[ii]*varM2, VecST_[ii]*varM2);
    printf("Variance = %12.4e\n",varM2);
    printAsterisks(PL_INFO, 0);
  }
  return 0.0;
}

// ************************************************************************
// perform analysis for input pairs
// ------------------------------------------------------------------------
double SobolAnalyzer::analyze2(aData &adata)
{
  int count, repID, iD, iD2, ii, ss, errCount;

  if (psConfig_.DiagnosticsIsOn())
    printOutTS(PL_INFO,"Entering SobolAnalyzer analyze2\n");

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  int outputID = adata.outputID_;
  double *xIn  = adata.sampleInputs_;
  double *yIn  = adata.sampleOutputs_;

  //**/ ---------------------------------------------------------------
  //**/ check if the 2nd-order Sobol sampling plan has been used
  //**/ ---------------------------------------------------------------
  int blkSize = nInputs * (nInputs - 1) / 2 + 2;
  int nReps = nSamples / blkSize;
  int blkLast, rowCnt;
  double xtemp1, dtemp;
  psVector vecMeanChk;
  vecMeanChk.setLength(blkSize);
  if ((nReps * blkSize) == nSamples)
  {
    //**/ there should be nReps blocks of blkSize samples each
    for (ss = 0; ss < nSamples; ss+=blkSize)
    {
      rowCnt   = ss;
      errCount = 0;
      blkLast  = ss + blkSize - 1;
      vecMeanChk[rowCnt-ss] += yIn[rowCnt];
      //**/ starting from the second sample in the block
      for (iD = 0; iD < nInputs; iD++)
      {
        for (iD2 = iD+1; iD2 < nInputs; iD2++)
        {
          rowCnt++;
          vecMeanChk[rowCnt-ss] += yIn[rowCnt];
          //**/ the current row (rowCnt) should differ from the
          //**/ first row in input iD and iD2 only
          for (ii = 0; ii < nInputs; ii++)
          {
            //**/ take the sample input ii
            xtemp1 = xIn[rowCnt*nInputs+ii];
            //**/ for input ii=iD, its value should be from M2
            if (ii == iD  && xtemp1 != xIn[blkLast*nInputs+ii]) 
              errCount++;
            //**/ for input ii=iD2, its value should be from M2
            else if (ii == iD2 && xtemp1 != xIn[blkLast*nInputs+ii]) 
              errCount++;
            //**/ for input ii!=iD,iD2, its value should be from M1
            else if (ii != iD && ii != iD2 && 
                     xtemp1 != xIn[ss*nInputs+ii]) 
              errCount++;
          }
        }
      }
      vecMeanChk[rowCnt+1-ss] += yIn[rowCnt+1];
      if (errCount > 0)
      {
        printOutTS(PL_ERROR,
             "SobolAnalyzer ERROR: Invalid sample (%d,%d)\n",
             ss, errCount);
        printOutTS(PL_ERROR, 
             "SobolAnalyzer requires Sobol samples.\n");
        return PSUADE_UNDEFINED;
      }
    }
    if (psConfig_.DiagnosticsIsOn())
    {
      for (ii = 0; ii < blkSize; ii++)
      {
        vecMeanChk[ii] /= (double) nReps;
        printf("Sobol: Sample mean check %d = %e\n",ii,vecMeanChk[ii]);
      }
      printf("NOTE: All the above means should be about the same.\n");
      printf("NOTE: There are %d blocks each with %d samples.\n",nReps,
             blkSize);
      printf("NOTE: So there are %d means.\n",blkSize);
    }
  }
  else
  {
    printOutTS(PL_ERROR,"SobolAnalyzer ERROR: Invalid sample size.\n");
    printOutTS(PL_ERROR,"SobolAnalyzer requires Sobol samples.\n");
    return PSUADE_UNDEFINED;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ compute mean (based on M2) and variance
  //**/ (meanSq is also needed to estimate E^2 for Sobol' indices)
  //**/ ---------------------------------------------------------------
  psVector VecY;
  VecY.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) VecY[ss] = yIn[nOutputs*ss+outputID];
  double meanM2 = 0.0;
  count = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED) 
    {
      meanM2 += VecY[repID*(nInputs+2)]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
               "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanM2 /= ((double) (count));
  double varM2 = 0.0;
  for (repID = 0; repID < nReps; repID++)
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED) 
      varM2 += ((VecY[repID*blkSize] - meanM2) * 
                (VecY[repID*blkSize] - meanM2)); 
  varM2 = varM2 / (double) (count-1.0);
  if (varM2 == 0)
  {
    printOutTS(PL_ERROR, 
               "SobolAnalyzer ERROR: Sample variance = 0.0.\n");
    exit(1);
  }
  double meanSq = 0.0;
  count = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED && 
        VecY[(repID+1)*blkSize-1] < 0.9*PSUADE_UNDEFINED) 
    {
      meanSq += VecY[repID*blkSize] * 
                VecY[(repID+1)*blkSize-1]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanSq /= ((double) (count));

  //**/ ---------------------------------------------------------------
  //**/ perform Sobol analysis
  //**/ ---------------------------------------------------------------
  int    offset=1;
  double tau;
  VecStds_.setLength(nInputs);
  VecModMeans_.setLength(nInputs);
  VecS_.setLength(nInputs_);
  VecST_.setLength(nInputs_);
  VecS2_.setLength(nInputs_*nInputs);
  VecPE_.setLength(nInputs_*nInputs);
  Variance_ = varM2;
  for (iD = 0; iD < nInputs; iD++)
  {
    for (iD2 = iD+1; iD2 < nInputs; iD2++)
    {
      //**/----------------------------------------------------------
      //**/ compute second-order sensitivity index VCE(Y)
      //**/----------------------------------------------------------
      tau = 0.0;
      count = 0;
      for (repID = 0; repID < nReps; repID++)
      {
        //**/ tau=(F(x1,..,xi,..,xj,..,xn) F(x1*, ...xi,..,xj,...xn*)) 
        if ((VecY[(repID+1)*blkSize-1] < 0.9*PSUADE_UNDEFINED) &&
            (VecY[repID*blkSize+offset] < 0.9*PSUADE_UNDEFINED))
        {
          tau += VecY[(repID+1)*blkSize-1]*VecY[repID*blkSize+offset]; 
          count++;
        }
      }
      if (count <= 0)
      {
        printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: Too few valid sample points for VCE.\n");
        exit(1);
      }
      //**/ compute pair effect index
      tau /= ((double) count);
      VecS2_[iD*nInputs+iD2] = (tau - meanSq) / varM2; 
      if (VecS2_[iD*nInputs+iD2] < 0) VecS2_[iD*nInputs+iD2] = 0;
      VecS2_[iD2*nInputs+iD] = VecS2_[iD*nInputs+iD2];

      offset++;
    }
  }
  if (psConfig_.InteractiveIsOn() && psConfig_.DiagnosticsIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"Sobol Analysis (analyze2: Input Pair Effect):\n");
    for (iD = 0; iD < nInputs; iD++)
      for (iD2 = iD+1; iD2 < nInputs; iD2++)
        printOutTS(PL_INFO, "Input %3d,%3d = %12.4e (normalized)\n",
             iD+1, iD2+1, VecS2_[iD*nInputs+iD2]);
    printAsterisks(PL_INFO, 0);
  }
  if (psConfig_.DiagnosticsIsOn())
    printOutTS(PL_INFO,"Exiting SobolAnalyzer analyze2\n");
  return 0.0;
}

// ************************************************************************
// perform analysis for group
// ------------------------------------------------------------------------
double SobolAnalyzer::analyze3(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  if (psConfig_.DiagnosticsIsOn())
    printOutTS(PL_INFO,"Entering SobolAnalyzer analyze3\n");
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  int outputID = adata.outputID_;
  double *xIn  = adata.sampleInputs_;
  double *yIn  = adata.sampleOutputs_;

  //**/ ---------------------------------------------------------------
  //**/ check group file 
  //**/ ---------------------------------------------------------------
  char cfname[1001],pString[1001],*cString,winput1[1001],winput2[1001];
  FILE *fp = fopen(adata.grpFileName_, "r");
  if (fp != NULL)
  {
    printOutTS(PL_INFO,"SOBOLAnalyzer: group file found.\n");
    int length = strlen(adata.grpFileName_); 
    strncpy(cfname, adata.grpFileName_, length);
    cfname[length] = '\0';
    fclose(fp);
  }
  psIMatrix matGrpMembers;
  matGrpMembers.setFormat(PS_MAT2D);
  int errFlag = readGrpInfoFile(cfname, nInputs, matGrpMembers);
  if (errFlag != 0)
  { 
    printOutTS(PL_ERROR,
      "ERROR: Failed to read group information file.\n");
    exit(1); 
  }
  int nGroups = matGrpMembers.nrows();

  //**/ ---------------------------------------------------------------
  //**/ check if the 2nd-order Sobol sampling plan has been used
  //**/ ---------------------------------------------------------------
  int count, repID, iD, iD2, ii, ss, kk, errCount;
  int blkSize = nGroups + 2;
  int nReps = nSamples / blkSize;
  int blkLast, rowCnt;
  double xtemp1, dtemp;
  psVector vecMeanChk;
  vecMeanChk.setLength(blkSize);
  if ((nReps * blkSize) == nSamples)
  {
    //**/ there should be nReps blocks of blkSize samples each
    for (ss = 0; ss < nSamples; ss+=blkSize)
    {
      rowCnt   = ss;
      errCount = 0;
      blkLast  = ss + blkSize - 1;
      vecMeanChk[rowCnt-ss] += yIn[rowCnt];
      //**/ starting from the second sample in the block
      for (iD = 0; iD < blkSize-2; iD++)
      {
        rowCnt++;
        vecMeanChk[rowCnt-ss] += yIn[rowCnt];
        //**/ the current row (rowCnt) should differ from the
        //**/ first row in certain position
        for (ii = 0; ii < nInputs; ii++)
        {
          //**/ take the sample input ii
          xtemp1 = xIn[rowCnt*nInputs+ii];
          //**/ for input ii=group, its value should be from M2
          kk = matGrpMembers.getEntry(iD,ii);
          if (kk != 0  && xtemp1 != xIn[blkLast*nInputs+ii]) 
            errCount++;
          //**/ for input ii!=iD,iD2, its value should be from M1
          else if (kk == 0 && xtemp1 != xIn[ss*nInputs+ii]) 
            errCount++;
        }
      }
      vecMeanChk[rowCnt+1-ss] += yIn[rowCnt+1];
      if (errCount > 0)
      {
        printOutTS(PL_ERROR,
             "SobolAnalyzer ERROR: Invalid sample (%d,%d)\n",
             ss, errCount);
        printOutTS(PL_ERROR, 
             "SobolAnalyzer requires Sobol samples.\n");
        return PSUADE_UNDEFINED;
      }
    }
    if (psConfig_.DiagnosticsIsOn())
    {
      for (ii = 0; ii < blkSize; ii++)
      {
        vecMeanChk[ii] /= (double) nReps;
        printf("Sobol: Sample mean check %d = %e\n",ii,vecMeanChk[ii]);
      }
      printf("NOTE: All the above means should be about the same.\n");
      printf("NOTE: There are %d blocks each with %d samples.\n",nReps,
             blkSize);
      printf("NOTE: So there are %d means.\n",blkSize);
    }
  }
  else
  {
    printOutTS(PL_ERROR,"SobolAnalyzer ERROR: Invalid sample size.\n");
    printOutTS(PL_ERROR,"SobolAnalyzer requires Sobol samples.\n");
    return PSUADE_UNDEFINED;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ compute mean (based on M2) and variance
  //**/ (meanSq is also needed to estimate E^2 for Sobol' indices)
  //**/ ---------------------------------------------------------------
  psVector VecY;
  VecY.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) VecY[ss] = yIn[nOutputs*ss+outputID];
  double meanM2 = 0.0;
  count = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED) 
    {
      meanM2 += VecY[repID*blkSize]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
               "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanM2 /= ((double) (count));
  double varM2 = 0.0;
  for (repID = 0; repID < nReps; repID++)
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED) 
      varM2 += ((VecY[repID*blkSize] - meanM2) * 
                (VecY[repID*blkSize] - meanM2)); 
  varM2 = varM2 / (double) (count-1.0);
  if (varM2 == 0)
  {
    printOutTS(PL_ERROR, 
               "SobolAnalyzer ERROR: Sample variance = 0.0.\n");
    exit(1);
  }
  double meanSq = 0.0;
  count = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    if (VecY[repID*blkSize] < 0.9*PSUADE_UNDEFINED && 
        VecY[(repID+1)*blkSize-1] < 0.9*PSUADE_UNDEFINED) 
    {
      meanSq += VecY[repID*blkSize] * 
                VecY[(repID+1)*blkSize-1]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  meanSq /= ((double) (count));

  //**/ ---------------------------------------------------------------
  //**/ perform Sobol group analysis
  //**/ ---------------------------------------------------------------
  int    offset=1;
  double tau;
  VecStds_.setLength(nInputs);
  VecModMeans_.setLength(nInputs);
  VecSG_.setLength(nGroups);
  Variance_ = varM2;
  for (iD = 0; iD < nGroups; iD++)
  {
    //if (psConfig_.DiagnosticsIsOn())
      printf("Sobol Analysis analyze3: processing group %d\n",iD+1);
    //**/----------------------------------------------------------
    //**/ compute group-order sensitivity index VCE(Y)
    //**/----------------------------------------------------------
    tau = 0.0;
    count = 0;
    for (repID = 0; repID < nReps; repID++)
    {
      //**/ tau=(F(x1,..,xi,..,xj,..,xn) F(x1*, ...xi,..,xj,...xn*)) 
      if ((VecY[(repID+1)*blkSize-1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*blkSize+offset] < 0.9*PSUADE_UNDEFINED))
      {
        tau += VecY[(repID+1)*blkSize-1]*VecY[repID*blkSize+offset]; 
        count++;
      }
    }
    if (count <= 0)
    {
      printOutTS(PL_ERROR,
       "SobolAnalyzer ERROR: Too few valid sample points for VCE.\n");
      exit(1);
    }
    //**/ compute group effect index
    tau /= ((double) count);
    VecSG_[iD] = (tau - meanSq) / varM2; 
    if (VecSG_[iD] < 0) VecSG_[iD] = 0;
    offset++;
  }
  if (psConfig_.InteractiveIsOn() && psConfig_.DiagnosticsIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"Sobol Analysis (analyze3: Group Effect):\n");
    for (iD = 0; iD < nGroups; iD++)
      printOutTS(PL_INFO, "  Group %3d = %12.4e (normalized)\n",
           iD+1, VecSG_[iD]);
    printAsterisks(PL_INFO, 0);
  }
  if (psConfig_.DiagnosticsIsOn())
    printOutTS(PL_INFO,"Exiting SobolAnalyzer analyze3\n");
  return 0.0;
}

// ************************************************************************
// perform analysis similar to MOAT analysis
// ------------------------------------------------------------------------
int SobolAnalyzer::MOATAnalyze(int nInputs, int nSamples, double *xIn,
                       double *yIn, double *xLower, double *xUpper,
                       double *means, double *modifiedMeans, double *stds)
{
  int    ss, ii;
  double xtemp1, xtemp2, ytemp1, ytemp2, scale;
  FILE   *fp;
  psIVector vecCounts;

  //**/ ---------------------------------------------------------------
  //**/ first compute the approximate gradients
  //**/ ---------------------------------------------------------------
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      ytemp1 = yIn[ss+ii]; 
      ytemp2 = yIn[ss]; 
      xtemp1 = xIn[(ss+ii)*nInputs+ii-1]; 
      xtemp2 = xIn[ss*nInputs+ii-1]; 
      scale  = xUpper[ii-1] - xLower[ii-1];
      if (xtemp1 != xtemp2)
      {
        if (ytemp1+ytemp2 < 0.9*PSUADE_UNDEFINED)
          yIn[ss+ii] = (ytemp2-ytemp1)/(xtemp2-xtemp1)*scale;
        else
          yIn[ss+ii] = PSUADE_UNDEFINED;
      }
      else
      {
        printOutTS(PL_ERROR, "SobolAnalyzer ERROR: divide by 0.\n");
        printOutTS(PL_ERROR, "     Check sample (Is this Sobol?) \n");
        printf("Replicate = %d (out of %d)\n",ss/(nInputs+2)+1, 
               nSamples/(nInputs+2));
        printf("Point 1: (Should differ from Point 2 at input = %d)\n",
               ii);
        for (int jj = 1; jj <= nInputs; jj++)
          printf("Input %d = %e\n",jj,xIn[(ss+ii)*nInputs+jj-1]); 
        printf("Point 2: \n");
        for (int jj = 1; jj <= nInputs; jj++)
          printf("Input %d = %e\n",jj,xIn[ss*nInputs+jj-1]); 
        exit(1);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ next compute the basic statistics
  //**/ ---------------------------------------------------------------
  vecCounts.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecCounts[ii] = 0;
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      if (yIn[ss+ii] < 0.9*PSUADE_UNDEFINED)
      {
        means[ii-1] += yIn[ss+ii];
        modifiedMeans[ii-1] += PABS(yIn[ss+ii]);
        vecCounts[ii-1]++;
      }
    }
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (vecCounts[ii] > 0)
    {
      means[ii] /= (double) (vecCounts[ii]);
      modifiedMeans[ii] /= (double) (vecCounts[ii]);
    }
  }
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      if (yIn[ss+ii] < 0.9*PSUADE_UNDEFINED)
        stds[ii-1] += (yIn[ss+ii] - means[ii-1]) *
                      (yIn[ss+ii] - means[ii-1]);
    }
  }
  for (ii = 0; ii < nInputs; ii++)
    if (vecCounts[ii] > 0)
      stds[ii] /= (double) (vecCounts[ii]);
  for (ii = 0; ii < nInputs; ii++) stds[ii] = sqrt(stds[ii]);

  //**/ ---------------------------------------------------------------
  //**/ write results to file for visualization
  //**/ ---------------------------------------------------------------
#if 0
  printDashes(PL_INFO, 0);
  if (plotScilab()) fp = fopen("scilabsobol.sci", "w");
  else              fp = fopen("matlabsobol.m", "w");
  if (fp == NULL)
  {
    printOutTS(PL_ERROR, "ERROR: cannot open file to write statistics.\n");
    return 0;
  }
  fprintf(fp, "Y = [\n");
  for (ii = 0; ii < nInputs; ii++) fprintf(fp, "%24.16e\n", stds[ii]);
  fprintf(fp, "];\n");
  fprintf(fp, "X = [\n");
  for (ii = 0; ii < nInputs; ii++) 
  fprintf(fp, "%24.16e\n",modifiedMeans[ii]);
  fprintf(fp, "];\n");
  fprintf(fp, "xh = max(X) - min(X);\n");
  fprintf(fp, "yh = max(Y) - min(Y);\n");
  fprintf(fp, "plot(X,Y,'*','MarkerSize',12)\n");
  fwritePlotAxes(fp);
  fwritePlotXLabel(fp, "Modified Means");
  fwritePlotYLabel(fp, "Std Devs");
  fprintf(fp, "text(X+0.01*xh,Y+0.01*yh,{");
  for (ii = 0; ii < nInputs-1; ii++) fprintf(fp, "'X%d',",ii+1);
  fprintf(fp, "'X%d'},'FontWeight','bold','FontSize',12)\n",nInputs);
  fprintf(fp, "title('Std Devs vs Modified mean Plot')\n");
  fclose(fp);
  if (plotScilab()) 
       printf("FILE scilabsobol.sci has results for plotting\n");
  else printf("FILE matlabsobol.m has results for plotting\n");
  printDashes(PL_INFO, 0);
#endif
  return 0;
}

// ************************************************************************
// set first or second order 
// ------------------------------------------------------------------------
int SobolAnalyzer::setOrder(int order)
{
  order_ = order;
  if (order != 1 && order != 2)
  {
    printf("SobolAnalyzer ERROR: Wrong order. Default to 1.\n");
    order_ = 1;
  }
  return 0; 
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
SobolAnalyzer& SobolAnalyzer::operator=(const SobolAnalyzer &)
{
  printOutTS(PL_ERROR,
           "SobolAnalyzer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int SobolAnalyzer::get_nInputs()
{
  return nInputs_;
}
int SobolAnalyzer::get_nGroups()
{
  return nGroups_;
}
double SobolAnalyzer::get_variance()
{
  return Variance_;
}
double SobolAnalyzer::get_modifiedMeans(int ind)
{
  if (ind >= 0 && ind < VecModMeans_.length()) 
       return VecModMeans_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_stds(int ind)
{
  if (ind >= 0 && ind < VecStds_.length()) 
       return VecStds_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_S(int ind)
{
  if (ind >= 0 && ind < VecS_.length()) 
       return VecS_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_S2(int ind)
{
  if (ind >= 0 && ind < VecS2_.length()) 
       return VecS2_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_SG(int ind)
{
  if (ind >= 0 && ind < VecSG_.length()) 
       return VecSG_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_ST(int ind)
{
  if (ind >= 0 && ind < VecST_.length()) 
       return VecST_[ind];
  else return 0.0;
}
double SobolAnalyzer::get_PE(int ind)
{
  if (ind >= 0 && ind < VecPE_.length()) 
       return VecPE_[ind];
  else return 0.0;
}

