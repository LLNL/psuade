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
double SobolAnalyzer::analyze(aData &adata)
{
  int     count, errFlag, repID, iD, ii, nReps, ss, errCount;
  double  xtemp1, xtemp2, tau, sMean, sVar, sMean2, sVar2, dtemp;
  psVector VecY, vecS, vecMeans, vecST, vecStds, vecPE, vecModMeans;

  printAsterisks(PL_INFO, 0);
  printf("*             Sensitivity Analysis on Sobol Samples\n"); 
  printEquals(PL_INFO, 0);

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

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0 || 
      outputID < 0 || outputID >= nOutputs)
  {
    printOutTS(PL_ERROR, "SobolAnalyzer ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "    nSamples = %d\n", nSamples);
    printOutTS(PL_ERROR, "    outputID = %d\n", outputID+1);
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
  //**/ check for valid samples
  //**/ ---------------------------------------------------------------
  count = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    errFlag = 0;
    if (sIn[ss] != 1) errFlag = 1;
    for (ii = 0; ii < nOutputs; ii++)
      if (yIn[ss*nOutputs+ii] > 0.99*PSUADE_UNDEFINED) errFlag = 1;
    if (errFlag == 0) count++;
  }
  printOutTS(PL_INFO,"SobolAnalyzer INFO: There are %d sample points.\n",
         nSamples);
  printOutTS(PL_INFO,
       "SobolAnalyzer INFO: There are %d valid sample points.\n",count);

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
        printOutTS(PL_ERROR,"SobolAnalyzer ERROR: invalid sample (%d,%d)\n",
                   ss, errCount);
        printOutTS(PL_ERROR, "SobolAnalyzer requires Sobol samples.\n");
        return PSUADE_UNDEFINED;
      }
    }
  }
  else
  {
    printOutTS(PL_ERROR, "SobolAnalyzer ERROR: invalid sample size.\n");
    printOutTS(PL_ERROR, "SobolAnalyzer requires Sobol samples.\n");
    return PSUADE_UNDEFINED;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ set up for and call MOAT analysis
  //**/ (Since Sobol sample satisfies MOAT property, it makes sense to
  //**/ run it through MOAT analysis)
  //**/ ---------------------------------------------------------------
  VecY.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) VecY[ss] = yIn[nOutputs*ss+outputID];
  vecMeans.setLength(nInputs);
  vecModMeans.setLength(nInputs);
  vecStds.setLength(nInputs);

  MOATAnalyze(nInputs,nSamples,xIn,VecY.getDVector(),xLower,xUpper,
              vecMeans.getDVector(), vecModMeans.getDVector(),
              vecStds.getDVector());

  printOutTS(PL_INFO, "Sobol-OAT (One-at-a-time) Analysis : \n");
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO, "Input %3d (mod. mean & std) = %12.4e %12.4e\n",
           ii+1, vecModMeans[ii], vecStds[ii]);
  printEquals(PL_INFO, 0);

  //save means & stds
  VecStds_ = vecStds;
  VecModMeans_ = vecModMeans;

  //**/ ---------------------------------------------------------------
  //**/ perform Sobol analysis
  //**/ ---------------------------------------------------------------

  //**/ make a copy of the outputs
  for (ss = 0; ss < nSamples; ss++) VecY[ss] = yIn[nOutputs*ss+outputID];

  //**/ compute mean and variance
  sMean = 0.0;
  count = 0;
  for (repID = 0;  repID < nReps; repID++)
  {
    if (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED) 
    {
      sMean += VecY[repID*(nInputs+2)+nInputs+1]; 
      count++;
    }
  }
  if (count <= 1)
  {
    printOutTS(PL_ERROR,
               "SobolAnalyzer ERROR: Too few valid sample points.\n");
    exit(1);
  }
  sMean /= ((double) (count));
  sVar = 0.0;
  for (repID = 0;  repID < nReps; repID++)
    if (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED) 
      sVar += ((VecY[repID*(nInputs+2)+nInputs+1] - sMean) * 
               (VecY[repID*(nInputs+2)+nInputs+1] - sMean)); 
  sVar = sVar / (double) (count-1.0);
  if (sVar == 0)
  {
    printOutTS(PL_ERROR, "SobolAnalyzer ERROR: sample variance = 0.0.\n");
    exit(1);
  }

  //**/ compute sensitivity indices
  vecS.setLength(nInputs_);
  vecST.setLength(nInputs_);
  vecPE.setLength(nInputs_);
  for (ii = 0; ii < nInputs; ii++)
  {
    //**/------------------------------------------------------------
    //**/ compute total sensitivity index
    //**/------------------------------------------------------------
    tau = 0.0;
    count = 0;
    //**/ for each block of nInputs+2 points
    //**/ tau = sum (F(x1,.. xi, ..xn) - mu) (F(x1, ...xi*, ..xn)-mu) 
    for (repID = 0;  repID < nReps; repID++)
    {
       if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
           (VecY[repID*(nInputs+2)] < 0.9*PSUADE_UNDEFINED))
       {
          tau += ((VecY[repID*(nInputs+2)] - sMean) *
                  (VecY[repID*(nInputs+2)+ii+1] - sMean)); 
          count++;
       }
    }
    if (count <= 0)
    {
       printOutTS(PL_ERROR,
          "SobolAnalyzer ERROR: too few valid sample points for TSI.\n");
       exit(1);
    }
    //**/ compute total index
    tau /= ((double) count);
    vecST[ii] = 1.0 - tau / sVar; 
    if (vecST[ii] < 0) vecST[ii] = 0;

    //**/------------------------------------------------------------
    //**/ compute sensitivity index V(Y)
    //**/------------------------------------------------------------
    tau = 0.0;
    count = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      //**/ tau = sum (F(x1,.. xi, ..xn) - mu) (F(x1*, ...xi, ..xn*)-mu) 
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        tau += ((VecY[repID*(nInputs+2)+nInputs+1] - sMean) *
                (VecY[repID*(nInputs+2)+ii+1] - sMean)); 
        count++;
      }
    }
    if (count <= 0)
    {
      printOutTS(PL_ERROR,
         "SobolAnalyzer ERROR: too few valid sample points for VCE.\n");
      exit(1);
    }
    //**/ compute main effect index
    tau /= ((double) count);
    vecS[ii] = tau / sVar; 
    if (vecS[ii] < 0) vecS[ii] = 0;

    //**/------------------------------------------------------------
    //**/ compute probable error (the formula in Sobol is wrong) 
    //**/------------------------------------------------------------
    sMean2 = 0.0;
    count  = 0;
    for (repID = 0;  repID < nReps; repID++)
    {
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        sMean2 += (VecY[repID*(nInputs+2)+nInputs+1]*
                   VecY[repID*(nInputs+2)+ii+1]);
        count++;
      }
    }
    sMean2 = sMean2 / (double) count;
    sVar2 = 0.0;
    for (repID = 0;  repID < nReps; repID++)
    {
      if ((VecY[repID*(nInputs+2)+ii+1] < 0.9*PSUADE_UNDEFINED) &&
          (VecY[repID*(nInputs+2)+nInputs+1] < 0.9*PSUADE_UNDEFINED))
      {
        dtemp = VecY[repID*(nInputs+2)+ii+1] * 
                VecY[repID*(nInputs+2)+nInputs+1];
        sVar2 += pow(dtemp - sMean2, 2.0);
      }
    }
    sVar2 = sVar2 / count;
    vecPE[ii] = 0.6745 * sqrt(sVar2) / sqrt(1.0 * count);
  }

  //printOutTS(PL_INFO, 
  //     "Sobol Analysis (ST: total sensitivity, PE: probable error):\n");
  //for (ii = 0; ii < nInputs; ii++)
  //  printOutTS(PL_INFO, "Input %3d (S, ST, PE) = %12.4e %12.4e %12.4e\n",
  //         ii+1, vecS[ii], vecST[ii], vecPE[ii]);
  printOutTS(PL_INFO, 
       "Sobol Analysis (S: Main Effect; ST: Total Sensitivity):\n");
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO, "Input %3d (S, ST) = %12.4e %12.4e\n",
           ii+1, vecS[ii], vecST[ii]);

  //save sensitivities
  VecS_ = vecS;
  VecST_ = vecST;
  VecPE_ = vecPE;
  printAsterisks(PL_INFO, 0);

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
        yIn[ss+ii] = (ytemp2-ytemp1)/(xtemp2-xtemp1)*scale;
      else
      {
        printOutTS(PL_ERROR, "SobolAnalyzer ERROR: divide by 0.\n");
        printOutTS(PL_ERROR, "     Check sample (Is this Sobol?) \n");
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
double *SobolAnalyzer::get_modifiedMeans()
{
  psVector vecModMeans;
  vecModMeans = VecModMeans_;
  double *retVal = vecModMeans.takeDVector();
  return retVal;
}
double *SobolAnalyzer::get_stds()
{
  psVector vecStds;
  vecStds = VecStds_;
  double *retVal = vecStds.takeDVector();
  return retVal;
}
double *SobolAnalyzer::get_S()
{
  psVector vecS;
  vecS = VecS_;
  double *retVal = vecS.takeDVector();
  return retVal;
}
double *SobolAnalyzer::get_ST()
{
  psVector vecST;
  vecST = VecST_;
  double *retVal = vecST.takeDVector();
  return retVal;
}
double *SobolAnalyzer::get_PE()
{
  psVector vecPE;
  vecPE = VecPE_;
  double *retVal = vecPE.takeDVector();
  return retVal;
}

