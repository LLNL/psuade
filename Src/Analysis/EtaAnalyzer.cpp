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
// Functions for the class EtaAnalyzer (Eta test)
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONGSnow
// DATE   : 2009
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "EtaAnalyzer.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
EtaAnalyzer::EtaAnalyzer() : Analyzer(), nInputs_(0), totalViolation_(0)
{
  setName("ETATEST");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
EtaAnalyzer::~EtaAnalyzer()
{
  VecOrders_.clean();
  VecRanks_.clean();
  VecInpViolations_.clean();
}

// ************************************************************************
// perform test
// ------------------------------------------------------------------------
double EtaAnalyzer::analyze(aData &adata)
{
  int    ss, ss2, ii2, ii, jj, kk, ii3, status, ind, nNeighs=5, imax, count;
  double dist, dmax, ddata, ***minDistMap, mean, stdev, dtemp, *minNeighs;
  char   pString[500];
  FILE   *fp;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  nInputs_ = adata.nInputs_;
  int printLevel = adata.printLevel_;
  int nSamples   = adata.nSamples_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int outputID   = adata.outputID_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  double *iLowerB = adata.iLowerB_;
  double *iUpperB = adata.iUpperB_;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nSamples < 100)
  {
    printOutTS(PL_ERROR,"EtaTest INFO: not meaningful to do this test");
    printOutTS(PL_ERROR,"when nSamples<100.\n");
    return PSUADE_UNDEFINED;
  }
  if (XIn == NULL || YIn == NULL)
  {
    printOutTS(PL_ERROR, "EtaTest ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  }
  status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YIn[nOutputs*ii+outputID] == PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR, "EtaTest: Some outputs are undefined.\n");
    printOutTS(PL_ERROR, "         Prune the undefined's first.\n");
    return PSUADE_UNDEFINED;
  }
  
  //**/ ---------------------------------------------------------------
  //**/ compute input ranges for later scaling
  //**/ ---------------------------------------------------------------
  psVector vecRangesInv2;
  vecRangesInv2.setLength(nInputs);

  for (ii = 0; ii < nInputs; ii++) 
  {
    if ((iUpperB[ii] - iLowerB[ii]) > 0)
      vecRangesInv2[ii] = 1.0 / (iUpperB[ii] - iLowerB[ii]) /
                                (iUpperB[ii] - iLowerB[ii]);
    else
    {
      printOutTS(PL_ERROR, "EtaTest ERROR: problem with input range.\n");
      return PSUADE_UNDEFINED;
    }
  }
 
  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  VecOrders_.clean();
  VecRanks_.clean();
  VecInpViolations_.clean();

  //**/ ---------------------------------------------------------------
  //**/ get user information
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "EtaTest for variable selection\n");
  printDashes(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,"EtaTest default number of neighbors k = 5.\n");
    printOutTS(PL_INFO, 
       " * The larger k is, the larger the distinguishing power is\n");
    printOutTS(PL_INFO, 
       " * The larger k is, the more expensive it is computationally\n");
    snprintf(pString,100,"What is your choice of k (>= 1, <= 20)? ");
    nNeighs = getInt(1, 20, pString);
  }
  if (nNeighs > nSamples/2)
  {
    nNeighs = nSamples / 2; 
    printOutTS(PL_INFO, 
         "EtaTest INFO: number of neighbors reset to be %d.\n",nNeighs);
  }  
  printOutTS(PL_INFO,"EtaTest INFO: number of neighbors = %d.\n",nNeighs);
  printOutTS(PL_INFO,
             "        To change, set analysis expert mode and re-do.\n");
 
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ prepare to run
  //**/ ---------------------------------------------------------------
  psIVector vecInpBins;
  vecInpBins.setLength(nInputs);

  minDistMap = new double**[nSamples];
  checkAllocate(minDistMap, "minDistMap in etaTest::analyze");
  for (ii = 0; ii < nSamples; ii++)
  {
    minDistMap[ii] = new double*[nInputs];
    for (jj = 0; jj < nInputs; jj++)
    {
      minDistMap[ii][jj] = new double[nNeighs];
      for (ss = 0; ss < nNeighs; ss++)
        minDistMap[ii][jj][ss] = 0.0;
    }
  }

  psVector vecMinNeighs;
  vecMinNeighs.setLength(nNeighs);

  VecRanks_.setLength(nInputs_);
  psVector vecRanks;
  vecRanks.setLength(nInputs);
  double *dRanks = vecRanks.getDVector();

  psVector vecMeans;
  vecMeans.setLength(nInputs);

  psIVector vecInds;
  vecInds.setLength(nNeighs);

  VecInpViolations_.setLength(nInputs);
  psIVector vecInpViolate;
  vecInpViolate.setLength(nInputs);
  int *inputViolations = vecInpViolate.getIVector();

  VecOrders_.setLength(nInputs_);
  psVector vecOrders;
  vecOrders.setLength(nInputs);
  double *dOrder = vecOrders.getDVector();

  //**/ ---------------------------------------------------------------
  //**/ search for min distance 
  //**/ ---------------------------------------------------------------
  srand(time(NULL));  
  for (ii = 0; ii < nInputs; ii++)
  {
    vecInpBins[ii] = 1;
    inputViolations[ii] = 0;
  }
  for (ss = 0; ss < nSamples; ss++)
  {
    if (printLevel > 1)
      printOutTS(PL_INFO, 
         "EtaTest: processing sample %5d (of %d)\n",ss+1,nSamples);
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ deactivate the current input
      vecInpBins[ii] = 0;
      //**/ initialize tracker
      for (jj = 0; jj < nNeighs; jj++)
      {
        vecMinNeighs[jj] = PSUADE_UNDEFINED;
        vecInds[jj] = -1;
      }
      //**/ examine with all other sample points
      //**/ and find a few nearest neighbors in all but i-th 
      //**/ dimension (that is, remove i-th input and find
      //**/ nearest neighbors
      for (ss2 = 0; ss2 < nSamples; ss2++)
      {
        if (ss != ss2)
        {
          //**/ compute distance for all inputs except ii
          dist = 0.0;
          for (jj = 0; jj < nInputs; jj++)
          {
            if (vecInpBins[jj] == 1)
            {
              dtemp = XIn[ss*nInputs+jj] - XIn[ss2*nInputs+jj];
              dist += dtemp * dtemp * vecRangesInv2[jj];
            }
          }
          //**/ keep the sample ss2 with the smallest distances
          if (dist > 0.0)
          {
            //**/ if current distance is smaller than some in store
            if (dist < vecMinNeighs[0])
            {
              //**/ replace the largest (first one) with it
              vecMinNeighs[0] = dist; 
              vecInds[0] = ss2; 
              for (jj = 1; jj < nNeighs; jj++)
              {
                if (vecMinNeighs[jj] > vecMinNeighs[jj-1])
                {
                  dmax = vecMinNeighs[jj];
                  vecMinNeighs[jj] = vecMinNeighs[jj-1];
                  vecMinNeighs[jj-1] = dmax;
                  imax = vecInds[jj];
                  vecInds[jj] = vecInds[jj-1];
                  vecInds[jj-1] = imax;
                }
              }
            }
          }
        }
      }
      for (jj = 0; jj < nNeighs; jj++) if (vecInds[jj] == -1) break;
      if (jj != nNeighs)
      {
        printOutTS(PL_ERROR, 
           "EtaTest ERROR: cannot find enough (%d) neighbors.\n",nNeighs);
        printOutTS(PL_ERROR, 
             "    Sample %d: nNeigh = %d (%d)\n",ss+1,jj,nNeighs);
        exit(1);
      }
      //**/ for sample ss and input ii, compute Ydiff from each
      //**/ nearest neighbor
      for (jj = 0; jj < nNeighs; jj++)
      {
        ind = vecInds[jj];
        minDistMap[ss][ii][jj] = 
           PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
        ddata = PABS((XIn[ind*nInputs]-XIn[ss*nInputs]))/
                     (iUpperB[0]-iLowerB[0]);
        if (ii == 0) ddata = 0.0;
        //**/ find for neighbor jj which input has largest distance
        ii3 = 0;
        for (kk = 1; kk < nInputs; kk++)
        {
          dtemp = PABS((XIn[ind*nInputs+kk]-XIn[ss*nInputs+kk]))/
                       (iUpperB[kk]-iLowerB[kk]);
          if (kk != ii && dtemp > ddata)
          {
            ddata = dtemp;
            ii3 = kk;
          }
        }
        dtemp = PABS((XIn[ind*nInputs+ii]-XIn[ss*nInputs+ii]))/
                     (iUpperB[ii]-iLowerB[ii]);
        ddata /= dtemp;
        //**/ if all other dimensions are further away than i-th
        //**/ dimension, flag error for the min dimension
        if (ddata < 1) inputViolations[ii3]++;
        if (printLevel > 3)
        {
          printOutTS(PL_INFO, 
               "EtaTest: sample %5d, input %3d (%3d), dist = %e\n",
               ss+1, ii3+1, ii+1, ddata);
        }
      }
      vecInpBins[ii] = 1;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ process and output the results
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++) dRanks[ii] = 0;
  for (jj = 0; jj < nNeighs; jj++)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ compute the mean of gradients with my neighbors in all
      //**/ dimensions
      mean = stdev = 0.0;
      for (ss = 0; ss < nSamples; ss++) mean += minDistMap[ss][ii][jj];
      mean /= (double) (nSamples);
      for (ss = 0; ss < nSamples; ss++) 
        stdev += pow(minDistMap[ss][ii][jj]-mean,2.0);
      stdev = sqrt(stdev / nSamples);
      vecMeans[ii] = mean;
    }
    for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
    sortDbleList2(nInputs, vecMeans.getDVector(), dOrder);
    //**/ accumulate the ranks of each input
    for (ii = 0; ii < nInputs; ii++)
    {
      kk = (int) dOrder[ii];
      dRanks[kk] += ii;
    } 
  }
  dist = dRanks[0];
  for (ii = 1; ii < nInputs; ii++) 
    if (dRanks[ii] > dist) dist = dRanks[ii];
  if (dist > 0.0) 
    for (ii = 0; ii < nInputs; ii++) dRanks[ii] /= dist;
  for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
  sortDbleList2(nInputs, dRanks, dOrder);
  printOutTS(PL_INFO, 
      "Order of importance based on %d nearest neighbors single effect:\n",
      nNeighs);
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO, 
         "(E1)Rank %4d: input %4d (score = %d)\n", ii+1, (int)
         dOrder[nInputs-ii-1]+1, (int) (dRanks[nInputs-ii-1]*100));
#if 0
  printEquals(PL_INFO, 0);
  kk = 0;
  for (ii = 0; ii < nInputs; ii++)
  {
    printOutTS(PL_INFO, "(E1) input %4d violation score = %5.2f\n", 
         ii+1, 100.0*inputViolations[ii]/nNeighs/nSamples);
    kk += inputViolations[ii];
  }
  totalViolation_ = 100.0*kk/nSamples/nInputs/nNeighs;
  printOutTS(PL_INFO, "(E1) Violation score = %5.2f\n",totalViolation_) ;
#endif
  printAsterisks(PL_INFO, 0);

  //save results
  for (ii = 0; ii < nInputs_; ii++)
  {
    VecOrders_[ii] = dOrder[ii];
    VecRanks_[ii]  = dRanks[ii];
    VecInpViolations_[ii] = inputViolations[ii];
  }

  //**/ ---------------------------------------------------------------
  //**/ search for min distance: two parameter effect
  //**/ ---------------------------------------------------------------
  //**/ if (psConfig_.AnaExpertModeIsOn())
  //**/ {
  //**/   printf("Perform second order analysis ? (y or n) ");
  //**/   scanf("%s", pString);
  //**/ }
  //**/ else pString[0] = 'n';
  pString[0] = 'n';

  if (pString[0] == 'y')
  {
    for (ii = 0; ii < nInputs; ii++) vecInpBins[ii] = 1;
    for (ss = 0; ss < nSamples; ss++)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        //**/ reset the current input
        vecInpBins[ii] = 0;
        for (ii2 = ii+1; ii2 < nInputs; ii2++)
        {
          vecInpBins[ii2] = 0;
          //**/ initialize tracker
          for (jj = 0; jj < nNeighs; jj++)
          {
            vecMinNeighs[jj] = PSUADE_UNDEFINED;
            vecInds[jj] = -1;
          }
          //**/ examine with all other sample points
          for (ss2 = 0; ss2 < nSamples; ss2++)
          {
            if (ss != ss2)
            {
              //**/ compute distance
              dist = 0.0;
              for (jj = 0; jj < nInputs; jj++)
              {
                if (vecInpBins[jj] == 1)
                {
                  dtemp = XIn[ss*nInputs+jj] - XIn[ss2*nInputs+jj];
                  dist += dtemp * dtemp * vecRangesInv2[jj];
                }
              }
              if (dist > 0.0)
              {
                //**/ if current distance is < some in store
                if (dist < vecMinNeighs[0])
                {
                  //**/ replace the largest with it
                  vecMinNeighs[0] = dist; 
                  vecInds[0] = ss2; 
                  for (jj = 1; jj < nNeighs; jj++)
                  {
                    if (vecMinNeighs[jj] > vecMinNeighs[jj-1])
                    {
                      dmax = vecMinNeighs[jj];
                      vecMinNeighs[jj] = vecMinNeighs[jj-1];
                      vecMinNeighs[jj-1] = dmax;
                      imax = vecInds[jj];
                      vecInds[jj] = vecInds[jj-1];
                      vecInds[jj-1] = imax;
                    }
                  }
                }
              }
            }
          }
          for (jj = 0; jj < nNeighs; jj++) if (vecInds[jj] == -1) break;
          if (jj != nNeighs)
          {
            printOutTS(PL_ERROR, "EtaTest ERROR: cannot find neighbor.\n");
            printOutTS(PL_ERROR, 
                 "    Sample %d: nNeigh = %d (%d)\n",ss+1,jj,nNeighs);
            exit(1);
          }
          for (jj = 0; jj < nNeighs; jj++)
          {
            ind = vecInds[jj];
            minDistMap[ss][ii][jj] += 
               PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
            minDistMap[ss][ii2][jj] += 
               PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
          }
          vecInpBins[ii2] = 1;
        }
        vecInpBins[ii] = 1;
      }
    }

    //**/ ------------------------------------------------------------
    //**/ process and output the results
    //**/ ------------------------------------------------------------
    for (ii = 0; ii < nInputs; ii++) dRanks[ii] = 0;
    for (jj = 0; jj < nNeighs; jj++)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        mean = stdev = 0.0;
        for (ss = 0; ss < nSamples; ss++) mean += minDistMap[ss][ii][jj];
        mean /= (double) (nSamples);
        for (ss = 0; ss < nSamples; ss++) 
          stdev += pow(minDistMap[ss][ii][jj]-mean,2.0);
        stdev = sqrt(stdev / nSamples);
        vecMeans[ii] = mean;
      }
      for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
      sortDbleList2(nInputs, vecMeans.getDVector(), dOrder);
      for (ii = 0; ii < nInputs; ii++)
      {
        kk = (int) dOrder[ii];
        dRanks[kk] += ii;
      } 
    }
    dist = dRanks[0];
    for (ii = 1; ii < nInputs; ii++) 
      if (dRanks[ii] > dist) dist = dRanks[ii];
    if (dist > 0.0) 
      for (ii = 0; ii < nInputs; ii++) dRanks[ii] /= dist;
    for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
    sortDbleList2(nInputs, dRanks, dOrder);
    printOutTS(PL_INFO, 
       "Order of importance based on %d nearest neighbors twin effect:\n",
       nNeighs);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO, "(E2)Rank %4d: input %4d (score = %d)\n", ii+1, 
           (int) dOrder[nInputs-ii-1]+1,(int) (dRanks[nInputs-ii-1]*100));
    printAsterisks(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ search for min distance: triple parameter effect
  //**/ ---------------------------------------------------------------
  //if (psConfig_.AnaExpertModeIsOn())
  //{
  //  printf("Perform third order analysis ? (y or n) ");
  //  scanf("%s", pString);
  //}
  //else pString[0] = 'n';
  pString[0] = 'n';
  if (pString[0] == 'y')
  {
    for (ii = 0; ii < nInputs; ii++) vecInpBins[ii] = 1;
    for (ss = 0; ss < nSamples; ss++)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        //**/ reset the current input
        vecInpBins[ii] = 0;
        for (ii2 = ii+1; ii2 < nInputs; ii2++)
        {
          vecInpBins[ii2] = 0;
          for (ii3 = ii2+1; ii3 < nInputs; ii3++)
          {
            vecInpBins[ii3] = 0;
            //**/ initialize tracker
            for (jj = 0; jj < nNeighs; jj++)
            {
              vecMinNeighs[jj] = PSUADE_UNDEFINED;
              vecInds[jj] = -1;
            }
            //**/ examine with all other sample points
            for (ss2 = 0; ss2 < nSamples; ss2++)
            {
              if (ss != ss2)
              {
                //**/ compute distance
                dist = 0.0;
                for (jj = 0; jj < nInputs; jj++)
                {
                  if (vecInpBins[jj] == 1)
                  {
                    dtemp = XIn[ss*nInputs+jj]-XIn[ss2*nInputs+jj];
                    dist += dtemp * dtemp * vecRangesInv2[jj];
                  }
                }
                if (dist > 0.0)
                {
                  //**/ if current distance is smaller 
                  if (dist < vecMinNeighs[0])
                  {
                    //**/ replace the largest with it
                    vecMinNeighs[0] = dist; 
                    vecInds[0] = ss2; 
                    for (jj = 1; jj < nNeighs; jj++)
                    {
                      if (vecMinNeighs[jj] > vecMinNeighs[jj-1])
                      {
                        dmax = vecMinNeighs[jj];
                        vecMinNeighs[jj] = vecMinNeighs[jj-1];
                        vecMinNeighs[jj-1] = dmax;
                        imax = vecInds[jj];
                        vecInds[jj] = vecInds[jj-1];
                        vecInds[jj-1] = imax;
                      }
                    }
                  }
                }
              }
            }
            for (jj = 0; jj < nNeighs; jj++) 
              if (vecInds[jj] == -1) break;
            if (jj != nNeighs)
            {
              printOutTS(PL_ERROR, 
                   "EtaTest ERROR: cannot find neighbor.\n");
              printOutTS(PL_ERROR, 
                   "   Sample %d: nNeigh = %d (%d)\n",ss+1,jj,nNeighs);
              exit(1);
            }
            for (jj = 0; jj < nNeighs; jj++)
            {
              ind = vecInds[jj];
              minDistMap[ss][ii][jj] += 
                PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
              minDistMap[ss][ii2][jj] += 
                PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
              minDistMap[ss][ii3][jj] += 
                PABS((YIn[ind*nOutputs+outputID]-YIn[ss*nOutputs+outputID]));
            }
            vecInpBins[ii3] = 1;
          }
          vecInpBins[ii2] = 1;
        }
        vecInpBins[ii] = 1;
      }
    }

    //**/ ------------------------------------------------------------
    //**/ process and output the results
    //**/ ------------------------------------------------------------
    for (ii = 0; ii < nInputs; ii++) dRanks[ii] = 0;
    for (jj = 0; jj < nNeighs; jj++)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        mean = stdev = 0.0;
        for (ss = 0; ss < nSamples; ss++) mean += minDistMap[ss][ii][jj];
        mean /= (double) (nSamples);
        for (ss = 0; ss < nSamples; ss++) 
          stdev += pow(minDistMap[ss][ii][jj]-mean,2.0);
        stdev = sqrt(stdev / nSamples);
        vecMeans[ii] = mean;
      }
      for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
      sortDbleList2(nInputs, vecMeans.getDVector(), dOrder);
      for (ii = 0; ii < nInputs; ii++)
      {
        kk = (int) dOrder[ii];
        dRanks[kk] += ii;
      } 
    }
    dist = dRanks[0];
    for (ii = 1; ii < nInputs; ii++) 
      if (dRanks[ii] > dist) dist = dRanks[ii];
    if (dist > 0.0) 
      for (ii = 0; ii < nInputs; ii++) dRanks[ii] /= dist;
    for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
    sortDbleList2(nInputs, dRanks, dOrder);
    printOutTS(PL_INFO, 
       "Order of importance based on %d nearest neighbors 3-way effect:\n",
       nNeighs);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO, "(E3)Rank %4d: input %4d (score = %d)\n", ii+1, 
          (int) dOrder[nInputs-ii-1]+1, (int) (dRanks[nInputs-ii-1]*100));
    printAsterisks(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ create matlab file
  //**/ ---------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn())
  {
    fp = fopen("psEtaData.m", "w");
    if (fp != NULL)
    {
      fprintf(fp, "clf\n");
      fprintf(fp, "n = %d;\n", nInputs);
      fprintf(fp, "X = 1:n;\n");
      for (jj = 0; jj < nNeighs; jj++) 
      {
        fprintf(fp, "A%d = [\n", jj+1);
        for (ss = 0; ss < nSamples; ss++) 
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%13.5e ", minDistMap[ss][ii][jj]);
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        if (jj % 5 == 0)
          fprintf(fp, "plot(X,A%d,'bx')\n",jj+1);
        else if (jj % 5 == 1)
          fprintf(fp, "plot(X,A%d,'rx')\n",jj+1);
        else if (jj % 5 == 2)
          fprintf(fp, "plot(X,A%d,'kx')\n",jj+1);
        else if (jj % 5 == 3)
          fprintf(fp, "plot(X,A%d,'gx')\n",jj+1);
        else if (jj % 5 == 4)
          fprintf(fp, "plot(X,A%d,'cx')\n",jj+1);
        fprintf(fp, "hold on;\n");
        fprintf(fp, "pause\n");
      }
      fclose(fp);
      printOutTS(PL_INFO, 
           "Use the psEtaData.m file to examine the distributions.\n");
    }
  }
  printAsterisks(PL_INFO, 0);
 
  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nSamples; ii++)
  {
    for (jj = 0; jj < nInputs; jj++)
      delete [] minDistMap[ii][jj];
    delete [] minDistMap[ii];
  }
  delete [] minDistMap;
  return 0.0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
EtaAnalyzer& EtaAnalyzer::operator=(const EtaAnalyzer &)
{
  printOutTS(PL_ERROR, "EtaTest operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int EtaAnalyzer::get_nInputs()
{
  return nInputs_;
}
double *EtaAnalyzer::get_dOrder()
{
  psVector vecDT;
  vecDT = VecOrders_;
  return vecDT.takeDVector();
}
double *EtaAnalyzer::get_dRanks()
{
  psVector vecDT;
  vecDT = VecRanks_;
  return vecDT.takeDVector();
}
int *EtaAnalyzer::get_inputViolations()
{
  psIVector vecIT;
  vecIT = VecInpViolations_;
  return vecIT.takeIVector();
}
double EtaAnalyzer::get_totalViolation()
{
  return totalViolation_;
}

