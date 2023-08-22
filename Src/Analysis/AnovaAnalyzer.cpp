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
// Functions for the class AnovaAnalyzer (modified from my work in DDace) 
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "FuncApprox.h"
#include "AnovaAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
AnovaAnalyzer::AnovaAnalyzer() : Analyzer(), tableLeng_(0) 
{
  setName("ANOVA");
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
AnovaAnalyzer::~AnovaAnalyzer()
{
  VecDofs_.clean();
  VecSumOfSqs_.clean();
  VecMeanSqs_.clean();
  VecFVals_.clean();
  MatCode_.clean();
}

// ************************************************************************
// perform Anova
// ------------------------------------------------------------------------
double AnovaAnalyzer::analyze(aData &adata)
{
  int    ii, jj, kk, ss, nPtsPerDim=256, ncount, index, nDataPts, faType;
  int    inCnt, inCnt2, inCnt3, nLevels=10, ncnt, cind1, cind2, cind3;
  int    dimPts;
  double ymax, dvalue;
  char   pString[1000];
  FuncApprox *fa;
  psVector   vecYY, vecXInterp, vecYInterp;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *xLower   = adata.iLowerB_;
  double *xUpper   = adata.iUpperB_;
  double *X        = adata.sampleInputs_;
  double *Y        = adata.sampleOutputs_;
  int  outputID = adata.outputID_;
  if (adata.inputPDFs_ != NULL)
  {
    ncount = 0;
    for (ii = 0; ii < nInputs; ii++) ncount += adata.inputPDFs_[ii];
    if (ncount > 0)
    {
      printOutTS(PL_WARN,
           "ANOVA INFO: some inputs have non-uniform PDFs.\n");
      printOutTS(PL_WARN,
           "      However, they are not relevant in this analysis\n");
      printOutTS(PL_WARN,
           "      (the sample should have been generated with\n");
      printOutTS(PL_WARN, "       the desired distributions.)\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR, "ANOVA ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  for (ii = 0; ii < nInputs; ii++)
  {
    if (xLower[ii] >= xUpper[ii])
    {
      printOutTS(PL_ERROR, "ANOVA ERROR: invalid input bounds.\n");
      printOutTS(PL_ERROR, "             Input %3d bounds = %e %e\n", 
                 ii+1, xLower[ii], xUpper[ii]);
      return PSUADE_UNDEFINED;
    }
  }
   
  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  VecSumOfSqs_.clean();
  VecMeanSqs_.clean();
  VecFVals_.clean();
  VecDofs_.clean();
  MatCode_.clean();
 
  //**/ ---------------------------------------------------------------
  //**/ generate response surface
  //**/ ---------------------------------------------------------------
  vecYY.setLength(nSamples);

  for (ss = 0; ss < nSamples; ss++) 
    vecYY[ss] = Y[nOutputs*ss+outputID];
  dimPts = 1000001;
  while (dimPts > 1000000)
  {
    nPtsPerDim = nPtsPerDim / 2;
    dimPts = 1;
    for (ii = 0; ii < nInputs; ii++)
    {
      dimPts *= nPtsPerDim;
      if (dimPts > 1000000) break;
    }
  }
  printf("This analysis uses a response surface ");
  printf("constructed from your sample.\n"); 
  printf("The available response surfaces are:\n");
  printf("1. MARS\n");
  printf("2. Legendre polynomial\n");
  printf("3. Gaussian process\n");
  faType = -1;
  snprintf(pString,100,"Enter your response surface choice ? ");
  while (faType < 0 || faType > 3) faType = getInt(1, 3, pString);
  if      (faType == 1) faType = PSUADE_RS_MARS;
  else if (faType == 2) faType = PSUADE_RS_REGRL;
  else if (faType == 3) faType = PSUADE_RS_GP3;
  fa = genFA(faType, nInputs, 1, nSamples);
  fa->setNPtsPerDim(nPtsPerDim);
  fa->setBounds(xLower, xUpper);
  fa->initialize(X, vecYY.getDVector());
   
  //**/ -------------------------------------------------------------
  //**/ obtain number of levels for each factors and generate estimates
  //**/ -------------------------------------------------------------
  nLevels  = nPtsPerDim;
  nDataPts = nLevels;
  for (ii = 1; ii < nInputs; ii++) nDataPts *= nLevels;
  vecXInterp.setLength(nDataPts*nInputs);
  vecYInterp.setLength(nDataPts);
  ncount = nDataPts / nLevels;
  for (ii = 0; ii < nInputs; ii++)
  {
    for (jj = 0; jj < nDataPts; jj+=ncount)
    {
      index = (jj / ncount) % nLevels;
      dvalue = (xUpper[ii]-xLower[ii]) / (nLevels-1)*index +
                xLower[ii]; 
      for (kk = 0; kk < ncount; kk++) 
        vecXInterp[(jj+kk)*nInputs+ii] = dvalue;
    }
    ncount /= nLevels;
  }

  ymax = -1.0E20;
  double *dptr = vecXInterp.getDVector();
  for (jj = 0; jj < nDataPts; jj++)
  {
    vecYInterp[jj] = fa->evaluatePoint(&dptr[jj*nInputs]);
    if (vecYInterp[jj] > ymax) ymax = vecYInterp[jj];
  }
  if (ymax != 0.0)
  {
    for (jj = 0; jj < nDataPts; jj++) vecYInterp[jj] /= ymax;
  }

  //**/ -------------------------------------------------------------
  //**/ calculate degree of freedom for each input and total
  //**/ -------------------------------------------------------------
  tableLeng_ = nInputs;
  tableLeng_ += ((nInputs - 1) * nInputs / 2);
  for (ii = 0; ii < nInputs; ii++)
    for (jj = ii+1; jj < nInputs; jj++)
      for (kk = jj+1; kk < nInputs; kk++) tableLeng_++;
  tableLeng_ += 2;
  VecDofs_.setLength(tableLeng_);
  MatCode_.setFormat(PS_MAT2D);
  MatCode_.setDim(tableLeng_, 3);
  for (ii = 0; ii < nInputs; ii++) VecDofs_[ii] = nLevels - 1;

  ncnt = nInputs;
  for (ii = 0; ii < nInputs; ii++)
    for (jj = ii+1; jj < nInputs; jj++)
      VecDofs_[ncnt++] = VecDofs_[ii] * VecDofs_[jj];
  for (ii = 0; ii < nInputs; ii++)
    for (jj = ii+1; jj < nInputs; jj++)
      for (kk = jj+1; kk < nInputs; kk++)
        VecDofs_[ncnt++] = VecDofs_[ii] * VecDofs_[jj] * VecDofs_[kk];
  VecDofs_[tableLeng_-1] = nDataPts - 1;
  VecDofs_[tableLeng_-2] = nDataPts - 1;
  ncnt = nInputs;
  ncnt += ((nInputs - 1) * nInputs / 2);
  for (jj = 0; jj < ncnt; jj++) VecDofs_[tableLeng_-2] -= VecDofs_[jj];

  //**/ -------------------------------------------------------------
  //**/ process for each input
  //**/ -------------------------------------------------------------
  VecSumOfSqs_.setLength(tableLeng_);
  VecMeanSqs_.setLength(tableLeng_);
  VecFVals_.setLength(tableLeng_);
  ncnt = 0;
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "*                       ANOVA table\n");
  printOutTS(PL_INFO, "*              (based on RS interpolation)\n");
  printEquals(PL_INFO, 0);

  for (inCnt = 0; inCnt < nInputs; inCnt++)
  {
    VecSumOfSqs_[inCnt] = computeSumSquares1(nDataPts, nInputs, inCnt,
                     VecDofs_[inCnt], 1, 0, vecXInterp.getDVector(),
                     vecYInterp.getDVector());
    MatCode_.setEntry(ncnt, 0, inCnt);
    MatCode_.setEntry(ncnt, 1, 0);
    MatCode_.setEntry(ncnt, 2, 0);
    ncnt++;
  }
  for (inCnt = 0; inCnt < nInputs; inCnt++)
  {
    for (inCnt2 = inCnt+1; inCnt2 < nInputs; inCnt2++)
    {
      VecSumOfSqs_[ncnt] = computeSumSquares2(nDataPts, nInputs,
                      inCnt, inCnt2, VecDofs_[inCnt],
                      VecDofs_[inCnt2], 1, 0, vecXInterp.getDVector(),
                      vecYInterp.getDVector(), VecSumOfSqs_[inCnt],
                      VecSumOfSqs_[inCnt2]);
      MatCode_.setEntry(ncnt, 0, inCnt);
      MatCode_.setEntry(ncnt, 1, inCnt2);
      MatCode_.setEntry(ncnt, 2, 0);
      ncnt++;
    }
  }
  for (inCnt = 0; inCnt < nInputs; inCnt++)
  {
    for (inCnt2 = inCnt+1; inCnt2 < nInputs; inCnt2++)
    {
      for (inCnt3 = inCnt2+1; inCnt3 < nInputs; inCnt3++)
      {
        kk = nInputs;
        for (ii = 0; ii < nInputs; ii++)
        {
          for (jj = ii+1; jj < nInputs; jj++)
          {
            if (ii == inCnt && jj == inCnt2)
            {
              cind1 = kk; break;
            } else kk++;
          }
        }
        kk = nInputs;
        for (ii = 0; ii < nInputs; ii++)
        {
          for (jj = ii+1; jj < nInputs; jj++)
          {
            if (ii == inCnt && jj == inCnt3)
            {
              cind2 = kk; break;
            } else kk++;
          }
        }
        kk = nInputs;
        for (ii = 0; ii < nInputs; ii++)
        {
          for (jj = ii+1; jj < nInputs; jj++)
          {
            if (ii == inCnt2 && jj == inCnt3)
            {
              cind3 = kk; break;
            } else kk++;
          }
        }
        if (cind1 < 0 || cind2 < 0 || cind3 < 0 || cind1 >= tableLeng_ ||
            cind2 >= tableLeng_ || cind3 >= tableLeng_)
        {
          printOutTS(PL_ERROR, "ANOVA ERROR: unrecoverable error.\n");
          printOutTS(PL_INFO, "Consult PSUADE developers\n");
          delete fa;
          return -1;
        }
        VecSumOfSqs_[ncnt] = computeSumSquares3(nDataPts, nInputs,
                    inCnt, inCnt2, inCnt3, VecDofs_[inCnt],
                    VecDofs_[inCnt2], VecDofs_[inCnt3], 1, 0,
                    vecXInterp.getDVector(), vecYInterp.getDVector(),
                    VecSumOfSqs_[inCnt], VecSumOfSqs_[inCnt2],
                    VecSumOfSqs_[inCnt3], VecSumOfSqs_[cind1],
                    VecSumOfSqs_[cind2], VecSumOfSqs_[cind3]);
        MatCode_.setEntry(ncnt, 0, inCnt);
        MatCode_.setEntry(ncnt, 1, inCnt2);
        MatCode_.setEntry(ncnt, 2, inCnt3);
        ncnt++;
      }
    }
  }
  //**/ compute SSTotal = variance * N
  double dmean = 0.0;
  for (jj = 0; jj < nDataPts; jj++) dmean += vecYInterp[jj];
  dmean /= (double) nDataPts;
  double SSTotal = 0.0;
  for (jj = 0; jj < nDataPts; jj++)
    SSTotal += (vecYInterp[jj] - dmean) * (vecYInterp[jj] - dmean);
  VecSumOfSqs_[tableLeng_-1] = SSTotal;

  //**/ SSE = SSTotal - SST (sum(sum(Yij-mean(Yi)))) 
  VecSumOfSqs_[tableLeng_-2] = VecSumOfSqs_[tableLeng_-1];
  ncnt = nInputs;
  ncnt += ((nInputs - 1) * nInputs / 2);
  for (jj = 0; jj < ncnt; jj++)
    VecSumOfSqs_[tableLeng_-2] -= VecSumOfSqs_[jj];

  //**/ VecMeanSqs = SST / (k-1)
  //**/ k = no. of treatments (or different values)
  for (jj = 0; jj < tableLeng_; jj++)
  {
    if (VecDofs_[jj] > 0)
      VecMeanSqs_[jj] = VecSumOfSqs_[jj] / (double) VecDofs_[jj];
    else
      VecMeanSqs_[jj] = 0.0;
  }
  //**/ compute F value = SST/(nLevels-1) / (SSE/(N-nLevels))
  //**/ SST/(nLevels-1) is in VecMeanSqs
  //**/ SSTotal is in VecSumOfSqs_[tableLeng_-1]
  for (jj = 0; jj < tableLeng_-2; jj++)
  {
    //**/ compute SSE = SSTotal - SST
    dvalue = VecSumOfSqs_[tableLeng_-1] - VecSumOfSqs_[jj]; 
    dvalue = dvalue / (double) (nDataPts - nLevels);
    if (dvalue != 0) VecFVals_[jj] = VecMeanSqs_[jj] / dvalue;
    else             VecFVals_[jj] = 0;
    //if (VecMeanSqs_[tableLeng_-2] != 0.0)
    //  VecFVals_[jj] = VecMeanSqs_[jj] / VecMeanSqs_[tableLeng_-2];
    //else
    //  VecFVals_[jj] = 0.0;
  }

  printEquals(PL_INFO, 65);
  printOutTS(PL_INFO, 
    "|  source of | deg. of|   sum of    |   mean      |            |\n");
  printOutTS(PL_INFO, 
    "|  variation | freedom|   squares   |   square    |       F    |\n");
  printDashes(PL_INFO, 65);
  for (jj = 0; jj < tableLeng_-2; jj++)
  {
    if (jj < nInputs)
    {
      printOutTS(PL_WARN,"|    %7d |%7d | %11.4e | %11.4e | %11.4e|\n",
        jj+1, VecDofs_[jj], VecSumOfSqs_[jj], VecMeanSqs_[jj],VecFVals_[jj]);
    } 
    else if (MatCode_.getEntry(jj, 2) == 0)
    {
      printOutTS(PL_WARN,"|    %3d,%3d |%7d | %11.4e | %11.4e | %11.4e|\n",
        MatCode_.getEntry(jj,0)+1,MatCode_.getEntry(jj,1)+1,
        VecDofs_[jj],VecSumOfSqs_[jj], VecMeanSqs_[jj], VecFVals_[jj]);
    } 
#if 0
    else
    {
      printOutTS(PL_INFO,
         "| %3d,%3d,%3d|   %6d | %11.4e | %11.4e | %11.4e|\n",
         MatCode_.getEntry(jj,0)+1,MatCode_.getEntry(jj,1)+1,
         MatCode_.getEntry(jj,2),VecDofs_[jj],
         VecSumOfSqs_[jj],VecMeanSqs_[jj],VecFVals_[jj]);
    }
#endif
  }
  //printOutTS(PL_INFO,"|    SSE     |%7d | %11.4e | %11.4e |    -----   |\n",
  //        VecDofs_[tableLeng_-2],VecSumOfSqs_[tableLeng_-2],
  //        VecMeanSqs_[tableLeng_-2]);
  printOutTS(PL_INFO,"|   total    |%7d | %11.4e | %11.4e |    -----   |\n",
          VecDofs_[tableLeng_-1],VecSumOfSqs_[tableLeng_-1],
          VecMeanSqs_[tableLeng_-1]);
  printAsterisks(PL_INFO, 0);
  printf("* Mean square : importance indicator of the source\n");
  printf("* F value     : large ==> reject null hypothesis ==> \n");
  printf("*           ==> significant differences among population means.\n");
  printf("*               (or variance of conditional mean is high)\n");
  printf("* Note: pairwise sources of variation exclude individual sources.\n");
  printAsterisks(PL_INFO, 0);

  //**/ -------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------
  delete fa;
  return 0.0;
}

// *************************************************************************
// Compute sum of squares for one factor
// -------------------------------------------------------------------------
double AnovaAnalyzer::computeSumSquares1(int nSamples, int nInputs, 
                       int source, int dof, int nOutputs, int outindex, 
                       double *X, double *Y)
{
  int    ss, jj, nLevels, xcnt, found;
  double mean, SST;
  psVector vecGSum, vecXVals;

  //**/ -------------------------------------------------------------
  //**/ compute overall mean
  //**/ -------------------------------------------------------------
  mean = 0.0;
  for (ss = 0; ss < nSamples; ss++) mean += Y[nOutputs*ss+outindex];
  mean /= (double) nSamples;

  //**/ -------------------------------------------------------------
  //**/ check number of input levels is correct
  //**/ -------------------------------------------------------------
  nLevels = dof + 1;
  xcnt    = 0;
  vecXVals.setLength(nLevels);
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt; jj++)
    {
      if (PABS(vecXVals[jj] - X[nInputs*ss+source]) < 1.0E-12)
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals[xcnt++] = X[nInputs*ss+source];
  }
  if (xcnt != nLevels) 
  {
    printOutTS(PL_INFO, "ANOVA ERROR (1): %d\n",xcnt);
    printOutTS(PL_INFO, "Consult PSUADE developers\n");
    exit(0);
  }

  //**/ -------------------------------------------------------------
  //**/ compute the sum of squares 
  //**/ -------------------------------------------------------------
  vecGSum.setLength(nLevels);
  for (jj = 0; jj < nLevels; jj++) vecGSum[jj] = 0.0;
  for (ss = 0; ss < nSamples; ss++)
  {
    for (jj = 0; jj < xcnt; jj++)
      if (PABS(vecXVals[jj]-X[nInputs*ss+source]) < 1.0E-12) break;
    vecGSum[jj] += Y[nOutputs*ss+outindex];
  }
  SST = 0.0;
  for (jj = 0; jj < nLevels; jj++)
  {
    //**/ compute group mean (mean at each level)
    vecGSum[jj] /= (nSamples / nLevels);
    SST += ((vecGSum[jj] - mean) * (vecGSum[jj] - mean));
  }
  //**/ compute final SST
  SST = SST / nLevels * nSamples;

  return SST;
}

// *************************************************************************
// Compute sum of squares for two factors
// -------------------------------------------------------------------------
double AnovaAnalyzer::computeSumSquares2(int nSamples, int nInputs, 
                       int input1, int input2, int dof1, int dof2, 
                       int nOutputs, int outindex, double *X, double *Y,
                       double ss1, double ss2)
{
  int ss, jj, kk, nlevels1, nlevels2, ncnt, xcnt1, xcnt2, found;
  psVector vecGSum1, vecGSum2, vecGSum12, vecXVals1, vecXVals2;

  //**/ -------------------------------------------------------------
  //**/ compute mean
  //**/ -------------------------------------------------------------
  double mean = 0.0;
  for (ss = 0; ss < nSamples; ss++) mean += Y[nOutputs*ss+outindex];
  mean = mean / nSamples;

  //**/ -------------------------------------------------------------
  //**/ find the input levels
  //**/ -------------------------------------------------------------
  nlevels1 = dof1 + 1;
  nlevels2 = dof2 + 1;
  xcnt1    = 0;
  xcnt2    = 0;
  vecXVals1.setLength(nlevels1);
  vecXVals2.setLength(nlevels2);
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt1; jj++)
    {
      if (PABS(vecXVals1[jj] - X[nInputs*ss+input1]) < 1.0E-8)
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals1[xcnt1++] = X[nInputs*ss+input1];
  }
  if (xcnt1 != nlevels1)
  {
    printOutTS(PL_INFO, "ANOVA ERROR (2a): %d\n",nlevels1);
    printOutTS(PL_INFO, "Consult PSUADE developers\n");
    exit(0);
  }
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt2; jj++)
    {
      if (PABS(vecXVals2[jj] - X[nInputs*ss+input2]) < 1.0E-8) 
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals2[xcnt2++] = X[nInputs*ss+input2];
  }
  if (xcnt2 != nlevels2)
  {
    printOutTS(PL_INFO, "AnovaAnalyzer ERROR (2b): %d\n",nlevels2);
    exit(0);
  }

  //**/ -------------------------------------------------------------
  //**/ compute the double-term sum of squares
  //**/ -------------------------------------------------------------
  vecGSum1.setLength(nlevels1);
  for (jj = 0; jj < nlevels1; jj++) vecGSum1[jj] = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    for (jj = 0; jj < xcnt1; jj++)
      if (PABS(vecXVals1[jj]-X[nInputs*ss+input1]) < 1.0E-12) break;
    vecGSum1[jj] += Y[nOutputs*ss+outindex];
  }
  for (jj = 0; jj < nlevels1; jj++) 
    vecGSum1[jj] = vecGSum1[jj] / (nSamples / nlevels1);

  vecGSum2.setLength(nlevels2);
  for (jj = 0; jj < nlevels2; jj++) vecGSum2[jj] = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    for (jj = 0; jj < xcnt2; jj++)
      if (PABS(vecXVals2[jj]-X[nInputs*ss+input2]) < 1.0E-12) break;
    vecGSum2[jj] += Y[nOutputs*ss+outindex];
  }
  for (jj = 0; jj < nlevels2; jj++) 
    vecGSum2[jj] = vecGSum2[jj] / (nSamples / nlevels2);

  ncnt = nlevels1 * nlevels2;
  vecGSum12.setLength(ncnt);
  for (jj = 0; jj < ncnt; jj++) vecGSum12[jj] = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    kk = -1;
    for (jj = 0; jj < xcnt1; jj++)
    {
      found = 0;
      if (PABS(vecXVals1[jj] - X[nInputs*ss+input1]) < 1.0E-12)
      {
        for (kk = 0; kk < xcnt2; kk++)
        {
          if (PABS(vecXVals2[kk]-X[nInputs*ss+input2]) < 1.0E-12)
          {
            found = 1;
            break;
          }
        }
      }
      if (found == 1) break;
    }
    if (kk >= 0) vecGSum12[jj*xcnt2+kk] += Y[nOutputs*ss+outindex];
  }
  for (jj = 0; jj < nlevels1*nlevels2; jj++) 
    vecGSum12[jj] = vecGSum12[jj] / (nSamples / (nlevels1 * nlevels2));

  //**/ -------------------------------------------------------------
  //**/ finally subtract from it the single terms 
  //**/ -------------------------------------------------------------
  double SST = 0.0;
  for (jj = 0; jj < nlevels1; jj++) 
  {
    for (kk = 0; kk < nlevels2; kk++) 
      SST += 
        ((vecGSum12[jj*nlevels2+kk]-vecGSum1[jj]-vecGSum2[kk]+mean)*
         (vecGSum12[jj*nlevels2+kk]-vecGSum1[jj]-vecGSum2[kk]+mean));
  }
  SST = SST / (nlevels1 * nlevels2) * nSamples;
  return SST;
}

// *************************************************************************
// Compute sum of squares for three factors
// -------------------------------------------------------------------------
double AnovaAnalyzer::computeSumSquares3(int nSamples, int nInputs, 
                          int input1, int input2, int input3, int dof1,
                          int dof2, int dof3, int nOutputs,
                          int output, double *X, double *Y,
                          double ss1, double ss2, double ss3, double ss4,
                          double ss5, double ss6)
{
  int    nlevels1, nlevels2, nlevels3, ncnt, xcnt1, xcnt2;
  int    xcnt3, found, ss, jj, kk, mm;
  double mean, sumT2, varN, sss1, sss2, sss3, sss4, sss5, sss6, diff, CT;
  psVector vecGSum, vecXVals1, vecXVals2, vecXVals3;

  //**/ -------------------------------------------------------------
  //**/ compute CT
  //**/ -------------------------------------------------------------
  CT = 0.0;
  for (ss = 0; ss < nSamples; ss++) CT += Y[nOutputs*ss+output];
  mean = CT / nSamples;
  CT = (CT * CT) / nSamples;

  //**/ -------------------------------------------------------------
  //**/ find the input levels
  //**/ -------------------------------------------------------------
  nlevels1 = dof1 + 1;
  nlevels2 = dof2 + 1;
  nlevels3 = dof3 + 1;
  xcnt1    = 0;
  xcnt2    = 0;
  xcnt3    = 0;
  vecXVals1.setLength(nlevels1);
  vecXVals2.setLength(nlevels2);
  vecXVals3.setLength(nlevels3);
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt1; jj++)
    {
      if (PABS(vecXVals1[jj]-X[nInputs*ss+input1]) < 1.0E-12) 
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals1[xcnt1++] = X[nInputs*ss+input1];
  }
  if (xcnt1 != nlevels1) 
  {
    printOutTS(PL_WARN, "ANOVA ERROR (3a): %d\n",nlevels1);
    printOutTS(PL_INFO, "Consult PSUADE developers\n");
    exit(0);
  }
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt2; jj++)
    {
      if (PABS(vecXVals2[jj]-X[nInputs*ss+input2]) < 1.0E-12) 
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals2[xcnt2++] = X[nInputs*ss+input2];
  }
  if (xcnt2 != nlevels2) 
  {
    printOutTS(PL_INFO, "ANOVA ERROR (3b): %d\n",nlevels2);
    printOutTS(PL_INFO, "Consult PSUADE developers\n");
    exit(1);
  }
  for (ss = 0; ss < nSamples; ss++)
  {
    found = 0;
    for (jj = 0; jj < xcnt3; jj++)
    {
      if (PABS(vecXVals3[jj]-X[nInputs*ss+input3]) < 1.0E-12) 
      {
        found = 1;
        break;
      }
    }
    if (found == 0) vecXVals3[xcnt3++] = X[nInputs*ss+input3];
  }
  if (xcnt3 != nlevels3) 
  {
    printOutTS(PL_INFO, "ANOVA ERROR (3c): %d\n",nlevels3);
    printOutTS(PL_INFO, "Consult PSUADE developers\n");
    exit(0);
  }

  //**/ -------------------------------------------------------------
  //**/ compute the double-term sum of squares
  //**/ -------------------------------------------------------------
  ncnt = nlevels1 * nlevels2 * nlevels3;
  vecGSum.setLength(ncnt);
  for (jj = 0; jj < ncnt; jj++) vecGSum[jj] = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    kk = mm = -1;
    for (jj = 0; jj < xcnt1; jj++)
    {
      found = 0;
      if (PABS(vecXVals1[jj]-X[nInputs*ss+input1]) < 1.0E-12)
      {
        for (kk = 0; kk < xcnt2; kk++)
        {
          if (PABS(vecXVals2[kk]-X[nInputs*ss+input2]) < 1.0E-12)
          {
            for (mm = 0; mm < xcnt3; mm++)
            {
              diff = vecXVals3[mm] - X[nInputs*ss+input3];
              if (PABS(diff) < 1.0E-12)
              {
                found = 1;
                break;
              }
            }
            if (found == 1) break;
          }
        }
      }
      if (found == 1) break;
    }
    if (kk < 0 || mm < 0) printOutTS(PL_INFO, "ANOVA ERROR (4).\n");
    else
      vecGSum[jj*xcnt2*xcnt3+kk*xcnt3+mm] += Y[nOutputs*ss+output];
  }
  sumT2 = 0.0;
  for (jj = 0; jj < ncnt; jj++) sumT2 += (vecGSum[jj] * vecGSum[jj]);
  sumT2 = sumT2 * nlevels1 * nlevels2 * nlevels3 / nSamples;

  //**/ -------------------------------------------------------------
  //**/ finally subtract from it the single terms and CT
  //**/ -------------------------------------------------------------
  sss1  = ss1 + CT;
  sss2  = ss2 + CT;
  sss3  = ss3 + CT;
  sss4  = ss4 + sss1 + sss2 - CT;
  sss5  = ss5 + sss1 + sss3 - CT;
  sss6  = ss6 + sss2 + sss3 - CT;
  sumT2 = sumT2 + sss1 + sss2 + sss3 - sss4 - sss5 - sss6 - CT;

  return sumT2;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
AnovaAnalyzer& AnovaAnalyzer::operator=(const AnovaAnalyzer &)
{
  printOutTS(PL_ERROR, "ANOVA operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}
// ************************************************************************
// Getters for analysis results
// ------------------------------------------------------------------------
int AnovaAnalyzer::get_tableLeng()
{
  return tableLeng_;
}

int *AnovaAnalyzer::get_dofs()
{
  psIVector vecT;
  vecT = VecDofs_;
  return vecT.takeIVector();
}

double *AnovaAnalyzer::get_sumSquare()
{
  psVector vecT;
  vecT = VecSumOfSqs_;
  return vecT.takeDVector();
}

double *AnovaAnalyzer::get_meanSquares()
{
  psVector vecT;
  vecT = VecMeanSqs_;
  return vecT.takeDVector();
}

double *AnovaAnalyzer::get_fValues()
{
  psVector vecT;
  vecT = VecFVals_;
  return vecT.takeDVector();
}

int **AnovaAnalyzer::get_code()
{
  psIMatrix matT;
  matT = MatCode_;
  return matT.takeIMatrix2D();
}

