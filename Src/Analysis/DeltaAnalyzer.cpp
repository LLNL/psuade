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
// Functions for the class DeltaAnalyzer (Delta test)
//**/ Reference: "Variable selection for Financial Modeling"
//**/            by Q. Yu, E. Severin and A. Lendasse 
//**/            "Using the Delta Test for Variable Selection"
//**/            by E. Eirola, E. Liitiainen, A. Lendasse, F. Corona, M.
//**/            Verleysen
//**/ A method based on nearest neighbors to evaluate input selection using
//**/ noise variance estimator (the study o fhow to give an a priori 
//**/ estimate for noise variance given some data. The Delta test estimates
//**/ this variance by:
//**/   * define delta(S) = 1/2M  sum_{i=1}^M (y_i - y_{ns(i)})^2
//**/   * ns(i) = arg min_{j!=i) ||x_i - x_j||_S^2 (in subspace S)
//**/   * ||x_i - x_j||_S^2 = sum_{k in S} (x_i^k - x_j^k)^2
//**/   * objective: D* = arg min_{D in [1,..n]} delta(D)  
//**/ Intuition: e.g. linear problem y = a_0 + sum_{k=1}^n a_k x_k 
//**/ delta(S)= E{(y-y_{ns})^2} = E{sum_{k=1}^n a_k^2 (x^k-x_{ns}^k)^2}
//**/ (1)     = E{ sum_{k in [1,..d]&S} a_k^2 (x^k - x_{ns}^k)^2 } +
//**/ (2)       E{ sum_{k in [1,..d]\S} a_k^2 (x^k - x_{ns}^k)^2 }
//**/ Term (2) becomes E {sum_{k in [1,..d]\S} a_k^2 [1/6] } 
//**/     (1/6 because both x^k and x_{ns}^ are r.v. in [0,1]) because
//**/     in subspace [1,..d]\S, the NN is different from NN in S. For
//**/     the subspace, the NN may be further away in the entire space.
//**/     However, for linear problems and due to the additive nature
//**/     of the terms to form F(X), if an input is allowed to vary
//**/     unrestricted (to subspace), F(X) for its NN may be large.
//**/     Hence, it makes sense to put a sensitive input to S. 
//**/     E.g. 2D at X=(0,0), X(1,0.1) is closer than X(0.5,0.9) but
//**/     if restricted to dimension 1, X(0.5,0.9) is closer. Note that
//**/     when restricted to dimension 1, the distance in dimension 2 
//**/     is ignored in calculating NN, hence 1/6 if k not in S (so 
//**/     x_ns^k varies from [0,1] unrestricted). For linear problems,
//**/     the less sensitive inputs (a_i^2 small) should be here. Also,
//**/     we want to select S as large as possible.
//**/ Term (1) <inside the summation> is an increasing function of size 
//**/     of S (subspace is larger and thus more terms in the linear
//**/     equation, assume a_i>0). Hence, we want to choose S to be
//**/     as small as possible, and include all sensitive inputs 
//**/     (large a_i^2). 
//**/ Based on Term (1) and (2) analysis, we should select S to be
//**/ those inputs such that a_i != 0 (given we want to select S
//**/ to be as small as possible but delta(S) is minimized.
//**/ For nonlinear problems, it is more complicated because it is
//**/ possible that, for a given point X_j, the function value F(X)
//**/ at its NN in S may be larger than that for the space [1,..d].
//**/ However, similar intuition as in the linear case can be drawn
//**/ here so the best S (that minimizes delta(S)) strikes a balance
//**/ between the 2 terms - implicitly.
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONG/Michael Snow
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
#include "DeltaAnalyzer.h"
#include "PrintingTS.h"
#ifdef PSUADE_OMP
#include <omp.h>
#endif

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
DeltaAnalyzer::DeltaAnalyzer(): Analyzer(),nBins_(1000),nInputs_(0)
{
  setName("DELTATEST");
  //**/ Note: nBins needs to be large to give reasonable statistics
  //**/       1000 seems to be okay for now.
  nBins_ = 1000;
  //**/ Note: the larger nNeigh is, more distinguishable power?
  nNeigh_ = 3;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
DeltaAnalyzer::~DeltaAnalyzer()
{
  VecMinDeltas_.clean();
  MatDeltaBins_.clean();
  VecOrders_.clean();
  VecRanks_.clean();
}

// ************************************************************************
// perform delta test
// ------------------------------------------------------------------------
double DeltaAnalyzer::analyze(aData &adata)
{
  int    ss, ss2, ii, jj, kk, ll, *iPtr, converged=0, newMinFlag;
  int    nSelected=0, status, place, uniqueFlag=0, reverseCnt=0;
  double distance, delta, minDist, dtemp, minDelta, deltaSave, auxMin;
  double temperature=.0001, oldDelta=PSUADE_UNDEFINED;
  double bestDelta=PSUADE_UNDEFINED, alpha=0.98, r, ddata, accum;
  char   pString[500], lineIn[5001];
  FILE   *fp;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  int outputID   = adata.outputID_;
  double *XX     = adata.sampleInputs_;
  double *YY     = adata.sampleOutputs_;
  double *iLowerB = adata.iLowerB_;
  double *iUpperB = adata.iUpperB_;
  nInputs_ = nInputs;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nSamples <= 100)
  {
    printOutTS(PL_ERROR, 
         "DeltaTest INFO: test not meaningful for nSamples <= 100.\n");
    return PSUADE_UNDEFINED;
  }
  if (XX == NULL || YY == NULL)
  {
    printOutTS(PL_ERROR, "DeltaTest ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  }
  status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YY[nOutputs*ii+outputID] == PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR, 
         "DeltaTest ERROR: Some outputs are undefined. Prune them.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  VecMinDeltas_.clean();
  MatDeltaBins_.clean();
  VecOrders_.clean();
  VecRanks_.clean();
 
  //**/ ---------------------------------------------------------------
  //**/ prepare to run
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"          Delta Test for Variable Selection\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"This test has the characteristics that ");
  printOutTS(PL_INFO,"the more important a parameter\n");
  printOutTS(PL_INFO,"is relative to the others, the smaller the ");
  printOutTS(PL_INFO,"subset is at the end of the\n");
  printOutTS(PL_INFO,"the test (sharp zoom into the ");
  printOutTS(PL_INFO,"most important subset).\n");
  printOutTS(PL_INFO,"Thus, the purpose of this test is to ");
  printOutTS(PL_INFO,"identify a subset of important\n");
  printOutTS(PL_INFO,"parameters.\n");
  printOutTS(PL_INFO,"NOTE: If both nInputs and nSamples ");
  printOutTS(PL_INFO,"are large, this test may take a\n");
  printOutTS(PL_INFO,"      long time to run. So, be patient.)\n");
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ get user information
  //**/ ---------------------------------------------------------------
  int maxIter = 20000;
  psIVector vecAuxBins; /* for storing pre-selected inputs */
  vecAuxBins.setLength(nInputs);
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,
         "DeltaTest Option: Set the number of neighbors K.\n");
    printOutTS(PL_INFO, 
         " * The larger K is, the larger the distinguishing power is.\n");
    printOutTS(PL_INFO, 
         " * However, larger K means more computational cost.\n");
    snprintf(pString,100,
         "Which K value would you select (>= 1, <= 20, default=3)? ");
    nNeigh_ = getInt(1, 20, pString);
    printOutTS(PL_INFO,
         "DeltaTest Option: Pre-select certain inputs as important.\n");
    snprintf(pString,
         100,"How many inputs to select FOR SURE? (0 if none) ");
    nSelected = getInt(0, nInputs-1, pString);
    for (ii = 0; ii < nInputs; ii++) vecAuxBins[ii] = 0;
    for (ii = 0; ii < nSelected; ii++)
    {
      snprintf(pString,
               100,"Enter the %d-th input to be selected : ", ii+1);
      kk = getInt(1, nInputs, pString);
      vecAuxBins[kk-1] = 1;
    }

    printOutTS(PL_INFO,
      "DeltaTest Option: Set maximum optimization iterations.\n");
    snprintf(pString,100,
      "Maximum iterations for optimization? (20000 - 1000000) ");
    maxIter = getInt(20000, 1000000, pString);

    printOutTS(PL_INFO,
      "DeltaTest Option: Set number of bins for computing statistics.\n");
    printOutTS(PL_INFO,
      "                  (The larger the better. Default = 1000.)\n");
    snprintf(pString,100,
      "Number of bins for computing statistics? (100 - 10000) ");
    nBins_ = getInt(100, 10000, pString);
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ initialize internal variables
  //**/ ---------------------------------------------------------------
  //**/ calculate ranges for scaling later
  psVector vecRangesInv2;
  vecRangesInv2.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
  {
    if ((iUpperB[ii] - iLowerB[ii]) > 0)
      vecRangesInv2[ii] = 1.0 / (iUpperB[ii] - iLowerB[ii]) /
                          (iUpperB[ii] - iLowerB[ii]);
    else
    {
      printOutTS(PL_ERROR, 
                 "DeltaTest ERROR: problem with input range.\n");
      exit(1);
    }
  }

  //**/ for storing the pairwise distances in a given subspace
  psVector vecDistPairs;
  vecDistPairs.setLength(nSamples*(nSamples-1)/2);

  //**/ the current subset under analysis
  psIVector vecInpBins;
  vecInpBins.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecInpBins[ii] = 0;

  //**/ keep track of the smallest deltas (nBins of them) and the 
  //**/ corresponding subset
  VecMinDeltas_.setLength(nBins_);
  psVector vecLMinDeltas;
  vecLMinDeltas.setLength(nBins_);
  double *minDeltas = vecLMinDeltas.getDVector();
  for (ii = 0; ii < nBins_; ii++) minDeltas[ii] = PSUADE_UNDEFINED;
  psIMatrix matLDeltaBins;
  MatDeltaBins_.setFormat(PS_MAT2D);
  MatDeltaBins_.setDim(nBins_, nInputs_);
  matLDeltaBins = MatDeltaBins_;
  int **deltaBins = matLDeltaBins.getIMatrix2D();

  //**/ keep the input set that gives minimum delta
  psIVector vecMinInds;
  vecMinInds.setLength(nNeigh_);

  //**/ extract the output that is to be analyzed
  psVector vecYT;
  vecYT.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) vecYT[ii] = YY[ii*nOutputs+outputID];

  //**/ ---------------------------------------------------------------
  //**/ generate initial configuration and distances
  //**/ ---------------------------------------------------------------
  if (nSelected == 0)
    for (ii = 0; ii < nInputs;ii++) vecInpBins[ii] = PSUADE_rand()%2;
  else
    for (ii = 0; ii < nInputs;ii++) vecInpBins[ii] = vecAuxBins[ii];

  for (ss = 1; ss < nSamples; ss++)
  {
    for (ss2 = 0; ss2 < ss; ss2++)
    {
      distance = 0.0;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (vecInpBins[ii] == 1)
        {
          dtemp = XX[ss*nInputs+ii] - XX[ss2*nInputs+ii];
          distance += dtemp * dtemp * vecRangesInv2[ii];
        }
      }
      vecDistPairs[ss*(ss-1)/2+ss2] = distance;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ search for initial min distance 
  //**/ ---------------------------------------------------------------
  int ind;
  delta = 0.0;
#pragma omp parallel shared(ss,nSamples,vecDistPairs,vecYT,delta) \
    private(jj,ddata,minDist,vecMinInds,ss2,kk,ll,ind)
{
#pragma omp for
  for (ss = 0; ss < nSamples; ss++)
  {
    ddata = 0.0;
    vecMinInds.setLength(nNeigh_);
    //**/ for each sample point, examine all other sample points in 
    //**/ order to identify a number of neighbors in the subspace
    for (jj = 0; jj < nNeigh_; jj++)
    {
      minDist = PSUADE_UNDEFINED;
      vecMinInds[jj] = -1;
      //**/ examine all other sample points up to s-1
      for (ss2 = 0; ss2 < ss; ss2++)
      {
        kk = ss * (ss - 1) / 2 + ss2;
        //**/ if this sample point ss2 is close, treat it as neighbor
        if (vecDistPairs[kk] < minDist)
        {
          //**/ search to see if this is already on the list
          for (ll = 0; ll < jj; ll++)
            if (ss2 == vecMinInds[ll]) break;
          //**/ if not, keep it
          if (jj == 0 || ll == jj)
          {
            minDist = vecDistPairs[kk];
            vecMinInds[jj] = ss2;
          }
        }
      }
      //**/ examine all other sample points from s+1 to the end
      for (ss2 = ss+1; ss2 < nSamples; ss2++)
      {
        kk = ss2 * (ss2 - 1) / 2 + ss;
        //**/ if this sample point ss2 is close, treat it as neighbor
        if (vecDistPairs[kk] < minDist)
        {
          //**/ search to see if this is already on the list
          for (ll = 0; ll < jj; ll++)
            if (ss2 == vecMinInds[ll]) break;
          if (jj == 0 || ll == jj)
          {
            minDist = vecDistPairs[kk];
            //**/ if not, keep it
            vecMinInds[jj] = ss2;
          }
        }
      }
      //**/ error checking: should never happen
      if (vecMinInds[jj] == -1)
      {
        printOutTS(PL_ERROR, "DeltaTest ERROR (1).\n");
        exit(1);
      }
      //**/ sum the differences
      ind = vecMinInds[jj];
      ddata += pow(vecYT[ss] - vecYT[ind], 2.0);
    }
    //**/ take average of sum of neighbors and sum the deltas for all 
    //**/ sample points
#pragma omp critical
    delta += ddata / (double) nNeigh_;
  }
} /* omp */
  printOutTS(PL_INFO,"Current best solution for output %d:\n",outputID+1);
  printOutTS(PL_INFO,
     "To stop the search, create a psuade_stop file in local directory.\n");
  printDashes(PL_INFO, 0);
  printf("When ready to proceed, enter any alphabet and return ");
  scanf("%s", lineIn);
  fgets(lineIn,5000,stdin);

  //**/ take mean as delta
  delta /= (2.0 * nSamples);
  for (ii = 0; ii < nInputs; ii++) 
    printOutTS(PL_INFO, "%d ", vecInpBins[ii]);
  printOutTS(PL_INFO, " = %e\n", delta);
 
  //**/ ---------------------------------------------------------------
  //**/ going through all combinations
  //**/ ---------------------------------------------------------------
  int iterCnt= 1;
  auxMin = - PSUADE_UNDEFINED;
  int stagnateIts = 10 * nInputs;
  if (stagnateIts < 100) stagnateIts = 100;
  //while (iterCnt <= 3*maxIter*nInputs)
  while (converged <= stagnateIts && iterCnt < maxIter)
  {
    fflush(stdout);
    if (printLevel > 1) printf("Iteration %d (of %d) : \n",iterCnt,maxIter);
    iterCnt++;

    //**/ select or de-select the next input
    if (reverseCnt >= 4*nInputs)
    {	
      //**/printOutTS(PL_WARN, "local minima: kickstarted");
      //**/ re-initialize some of the inputs
      temperature *= nInputs * nInputs;
      for (ii = 0; ii <= nInputs/5; ii++) 
        vecInpBins[PSUADE_rand()%nInputs] ^=1;

      //**/ compute pairwise distances in the subspace
      if (printLevel > 1) printf("   Re-compute distance matrix\n");
      for (ss = 1; ss < nSamples; ss++)
      {
        for (ss2 = 0; ss2 < ss; ss2++)
        {
          distance = 0.0;
          for (ii = 0; ii < nInputs; ii++)
          {
            if (vecInpBins[ii] == 1)
            {
              dtemp = XX[ss*nInputs+ii] - XX[ss2*nInputs+ii];
              distance += dtemp * dtemp * vecRangesInv2[ii];
            }
          }
          vecDistPairs[ss*(ss-1)/2+ss2] = distance;
        }
      }
      reverseCnt = 0;
      place = PSUADE_rand()%(nInputs);
    }
    else 
    {
      if (reverseCnt >= 3*nInputs)
      {
        //printOutTS(PL_WARN, "suspected local minima, checking");
        place = reverseCnt - 3 * nInputs;
      }
      else 
      {
        place = PSUADE_rand()%(nInputs);
      }
    }
    temperature *= alpha;
    vecInpBins[place] ^= 1;

    //**/ update the distance with adding/subtracting an input
    if (printLevel > 1) printf("   Update distance matrix\n");
    delta = 0.0;
    for (ss = 1; ss < nSamples; ss++)
    {
      for (ss2 = 0; ss2 < ss; ss2++)
      {
        kk = ss * (ss - 1) / 2 + ss2;
        dtemp = XX[ss*nInputs+place] - XX[ss2*nInputs+place];
        if (vecInpBins[place] == 1)
             vecDistPairs[kk] += dtemp * dtemp * vecRangesInv2[place];
        else vecDistPairs[kk] -= dtemp * dtemp * vecRangesInv2[place];
      }
    }

    //**/ search for min distance 
    if (printLevel > 1) printf("   Searching for minimum distance\n");
    int once=0;
    delta = 0.0;
#pragma omp parallel shared(nSamples,vecDistPairs,vecYT,delta,iterCnt) \
    private(ss,jj,ddata,minDist,vecMinInds,ss2,kk,ll,ind,once)
{
    once = 0;
#pragma omp for
    for (ss = 0; ss < nSamples; ss++)
    {
#ifdef PSUADE_OMP
      if (once == 0 && iterCnt <= 2)
      {
        printf("Delta %d: Running sample %d at thread %d (numThreads=%d)\n",
               iterCnt,ss+1, omp_get_thread_num(), omp_get_num_threads());
        once = 1;
      }
#endif
      ddata = 0.0;
      vecMinInds.setLength(nNeigh_);
      for (jj = 0; jj < nNeigh_; jj++)
      {
        minDist = PSUADE_UNDEFINED;
        vecMinInds[jj] = -1;
        //**/ examine all other sample points up to s-1
        for (ss2 = 0; ss2 < ss; ss2++)
        {
          kk = ss * (ss - 1) / 2 + ss2;
          //**/ if this sample point ss2 is close, treat it as neighbor
          if (vecDistPairs[kk] < minDist)
          {
            //**/ search to see if this is already on the list
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              //**/ if not, keep it
              minDist = vecDistPairs[kk];
              vecMinInds[jj] = ss2;
            }
          }
        }
        //**/ examine all other sample points from s+1 up
        for (ss2 = ss+1; ss2 < nSamples; ss2++)
        {
          kk = ss2 * (ss2 - 1) / 2 + ss;
          //**/ if this sample point ss2 is close, treat it as neighbor
          if (vecDistPairs[kk] < minDist)
          {
            //**/ search to see if this is already on the list
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              //**/ if not, keep it
              minDist = vecDistPairs[kk];
              vecMinInds[jj] = ss2;
            }
          }
        }
        if (vecMinInds[jj] == -1)
        {
          printOutTS(PL_ERROR, "DeltaTest ERROR (2).\n");
          exit(1);
        }
        //**/ sum the differences
        ind = vecMinInds[jj];
        ddata += pow(vecYT[ss] - vecYT[ind], 2.0);
      }
      //**/ take average of distance-squared and add to delta
#pragma omp critical
      delta += ddata / (double) nNeigh_;
    }
} /* end pragma omp parallel */

    //**/ compute mean to get delta
    delta /= (2.0 * nSamples);

    //**/ store away the minimum values and configurations
    //**/ minDeltas initially all undefined and then filled with
    //**/ largest values at the beginning
    newMinFlag = 0;
    if (delta < minDeltas[0])
    {
      //**/ don't find unique, it will mess up ranking
      uniqueFlag = 1;
      //for (ii = 0; ii < nBins_; ii++)
      //{
      //  if (minDeltas[ii] != PSUADE_UNDEFINED)
      //  {
      //    for (jj = 0; jj < nInputs; jj++)
      //      if (vecInpBins[jj] != deltaBins[ii][jj]) break;
      //    if (jj == nInputs) {uniqueFlag = 0; break;}
      //  }
      //}
      if (uniqueFlag == 1)
      {
        //**/ restart convergence count whenever seeing a delta
        //**/ that is less than the 100-th in the min list
        //**/ Note: make sure nBins > 100
        //if (nBins_ >= 100 && delta < minDeltas[nBins_-99]) 
        //  newMinFlag = 1;
        //else if (nBins_ < 100 && delta < minDeltas[nBins_-1]) 
        //  newMinFlag = 1;
        if (delta < minDeltas[nBins_-1]) newMinFlag = 1;
        //**/ put the new minimum at the first position (replace old)
        minDeltas[0] = delta;
        for (ii = 0; ii < nInputs; ii++) 
          deltaBins[0][ii] = vecInpBins[ii];

        //**/ now sort (compare and swap)
        for (ii = 1; ii < nBins_; ii++)
        {
          if (minDeltas[ii] > minDeltas[ii-1])
          {
            dtemp = minDeltas[ii];
            minDeltas[ii] = minDeltas[ii-1];
            minDeltas[ii-1] = dtemp;
            iPtr = deltaBins[ii];
            deltaBins[ii] = deltaBins[ii-1];
            deltaBins[ii-1] = iPtr;
          }
        }
      }
    }
    for (ii = 0; ii < nInputs-1; ii++) 
      printOutTS(PL_INFO, "%d ", vecInpBins[ii]);
    printOutTS(PL_INFO, "%d", vecInpBins[nInputs-1]);
    if (newMinFlag == 1) printOutTS(PL_INFO, "* ");
    else                 printOutTS(PL_INFO, "  ");
    //printOutTS(PL_INFO,"= %e (its=%d, cf=%3.1f)\n",delta,iterCnt-1,
    //          1.0*converged/stagnateIts);
    printOutTS(PL_INFO,"= %e (its=%d)\n",delta,iterCnt-1);
              

    //**/ check convergence
    if (minDeltas[nBins_-1] == auxMin) converged++;
    else
    {
      converged = 0;
      auxMin = minDeltas[nBins_-1];
    }
    if (newMinFlag == 1) converged = 0;
    if (converged > stagnateIts)
    {
      printOutTS(PL_INFO,"DeltaTest: no improvement for %d iterations, ", 
                 stagnateIts);
      printOutTS(PL_INFO, "considered converged.\n");
      break;
    }

    //**/ if current minimum is larger than the previous one,
    //**/ with a certain probability, reverse the selection
    if (delta >= oldDelta) 
    {
      r = PSUADE_rand()%100000;
      r /= 100000;
      //**/ printOutTS(PL_WARN, "\n dice roll r %e and comparison %e and
      //**/ temperature %e \n",r,
      //**/ exp(10000*(oldDelta-delta)/temperature),temperature);

      if (r>=exp(-.1*(delta-oldDelta)/(temperature)))
      {
        vecInpBins[place] ^=1;
        reverseCnt++;
        for (ss = 1; ss < nSamples; ss++)
        {
          for (ss2 = 0; ss2 < ss; ss2++)
          {
            kk = ss * (ss - 1) / 2 + ss2;
            dtemp = XX[ss*nInputs+place] - XX[ss2*nInputs+place];
            if (vecInpBins[place] == 1)
                 vecDistPairs[kk] += dtemp * dtemp * vecRangesInv2[place];
            else vecDistPairs[kk] -= dtemp * dtemp * vecRangesInv2[place];
          }
        }
      }
      else
      {
        oldDelta = delta;
        reverseCnt = 0;
      }
    }
    else 
    {
      oldDelta = delta;
      reverseCnt = 0;
    }
    if (oldDelta <= bestDelta) bestDelta = oldDelta;
    fp = fopen("psuade_stop","r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO, "psuade_stop file found ==> terminate.\n");
      printOutTS(PL_INFO, "To restart, delete psuade_stop first.\n");
      fclose(fp);
      break;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ output the results
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, 
       "Final Selections (based on %d neighbors) = \n", nNeigh_);
  printOutTS(PL_INFO, 
       "(The top is the best and then the next few best.)\n");
  printDashes(PL_INFO, 0);

  //**/ save min Deltas and deltaBins
  for (ii=0; ii < nBins_; ii++)
  {
    VecMinDeltas_[ii] = minDeltas[ii];
    for (kk = 0; kk < nInputs_; kk++)
      MatDeltaBins_.setEntry(ii, kk, deltaBins[ii][kk]);
  }

  //**/ print ranks
  int count = 0;
  kk = 0;
  while (count < 10 && kk < nBins_-1)
  {
    uniqueFlag = 1;
    for (ii = nBins_-1; ii >= nBins_-kk; ii--)
    {
      for (jj = 0; jj < nInputs; jj++)
        if (deltaBins[nBins_-kk-1][jj] != deltaBins[ii][jj]) break;
      if (jj == nInputs) {uniqueFlag = 0; break;}
    } 
    if (uniqueFlag ==1 && minDeltas[nBins_-kk-1] < 0.99 * PSUADE_UNDEFINED)
    {
      printOutTS(PL_INFO, "Rank %2d => ", count+1);
      for (ii = 0; ii < nInputs; ii++) 
        printOutTS(PL_INFO, "%d ", deltaBins[nBins_-kk-1][ii]);
      printOutTS(PL_INFO, ": delta = %11.4e\n",minDeltas[nBins_-kk-1]);
      count++;
    }
    kk++;
  }
  printDashes(PL_INFO, 0);

  //**/ print ranks
  count = 0;
  psIVector vecRanks;
  vecRanks.setLength(nInputs); 
  int *ranks = vecRanks.getIVector();
  for (ii = 0; ii < nInputs; ii++)
  {
    ddata = 0;
    //**/for (kk = 0; kk < nBins_; kk++) 
    //**/  count += deltaBins[nBins_-kk-1][ii];
    //**/ Oct 2009: smallest data weighted more heavily
    accum = 0.0;
    for (kk = 0; kk < nBins_; kk++)
    {
      if (minDeltas[nBins_-kk-1] != PSUADE_UNDEFINED)
      {
        ddata += (minDeltas[nBins_-1]*deltaBins[nBins_-kk-1][ii]/
                  minDeltas[nBins_-kk-1]);
        accum += (minDeltas[nBins_-1]/minDeltas[nBins_-kk-1]);
      }
    }
    //**/ printOutTS(PL_WARN, "%2d ", count);
    ranks[ii] = (int) (ddata / accum * 100);
  }

  //**/ output to a matlab file
  if (plotScilab()) fp = fopen("scilabdelta.sci", "w");
  else              fp = fopen("matlabdelta.m", "w");
  if (fp == NULL)
  {
    printOutTS(PL_INFO,"Delta test ERROR: cannot open graphics files.\n");
    printOutTS(PL_INFO,"                  ==> graphics not generated.\n");
  }
  else
  {
    fwritePlotCLF(fp);
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nInputs; ii++)
      fprintf(fp, "%e\n", 0.01 * ranks[ii]);
    fprintf(fp, "];\n");
    fprintf(fp, "bar(A, 0.8);\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Delta Test Rankings");
    fwritePlotXLabel(fp, "Input parameters");
    fwritePlotYLabel(fp, "Delta Metric (normalized)");
    fclose(fp);
    if (plotScilab()) 
      printOutTS(PL_INFO,
         "Delta test ranking is now in scilabdelta.sci.\n");
    else 
      printOutTS(PL_INFO,"Delta test ranking is now in matlabdelta.m.\n");
  }

  //**/ output the ranks in order
  psVector vecOrders;
  vecOrders.setLength(nInputs);
  double *dOrder = vecOrders.getDVector(); 

  for (ii = 0; ii < nInputs; ii++) dOrder[ii] = 1.0 * ii;
  sortIntList2a(nInputs, ranks, dOrder);
  printOutTS(PL_INFO,
    "Order of importance (based on frequencies of appearance in search)\n");
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO, "(Delta) Rank %4d : input %4d (score = %d )\n", ii+1, 
               (int) dOrder[nInputs-ii-1]+1, ranks[nInputs-ii-1]);
  printAsterisks(PL_INFO, 0);

  //save dOrder and ranks
  VecOrders_.setLength(nInputs_);
  VecRanks_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    VecOrders_[ii] = dOrder[ii];
    VecRanks_[ii]  = ranks[ii];
  }

  //**/ ---------------------------------------------------------------
  //**/ test delta values with most important parameters 
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO, 
    "Final test adding the most important parameters incrementally:\n");
  printOutTS(PL_INFO,"You should see the rightmost values ");
  printOutTS(PL_INFO,"decreasing and then increasing.\n");
  printOutTS(PL_INFO,"The lowest point can be used as a separator ");
  printOutTS(PL_INFO,"for classifying important\n");
  printOutTS(PL_INFO,"and less important parameters.\n");
  printDashes(PL_INFO, 0);
  for (ii = 0; ii < nInputs; ii++) vecInpBins[ii] = 0;
  //**/ since it is not possible to compute the delta reduction due to
  //**/ the first input, the following code does the 2nd one first 
  for (ii = 1; ii >= 0; ii--)
  {
    vecInpBins[(int) dOrder[nInputs-ii-1]] = 1;
    delta = 0.0;
    for (ss = 0; ss < nSamples; ss++)
    {
      ddata = 0.0;
      for (jj = 0; jj < nNeigh_; jj++)
      {
        minDist = PSUADE_UNDEFINED;
        vecMinInds[jj] = -1;
        for (ss2 = 0; ss2 < ss; ss2++)
        {
          distance = 0.0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (vecInpBins[kk] == 1)
            {
              dtemp = XX[ss*nInputs+kk] - XX[ss2*nInputs+kk];
              distance += dtemp * dtemp * vecRangesInv2[kk];
            }
          }
          if (distance < minDist)
          {
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              minDist = distance;
              vecMinInds[jj] = ss2;
            }
          }
        }
        for (ss2 = ss+1; ss2 < nSamples; ss2++)
        {
          distance = 0.0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (vecInpBins[kk] == 1)
            {
              dtemp = XX[ss*nInputs+kk] - XX[ss2*nInputs+kk];
              distance += dtemp * dtemp * vecRangesInv2[kk];
            }
          }
          if (distance < minDist)
          {
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              minDist = distance;
              vecMinInds[jj] = ss2;
            }
          }
        }
        ddata += pow(vecYT[ss] - vecYT[vecMinInds[jj]], 2.0);
      }
      delta += ddata / (double) nNeigh_;
    }
    delta /= (2.0 * nSamples);
    minDeltas[ii] = delta;
  }
  deltaSave = minDeltas[1] - minDeltas[0];
  vecInpBins[(int) dOrder[nInputs-2]] = 0;
  vecInpBins[(int) dOrder[nInputs-1]] = 0;
  //**/ now start from the beginning
  minDelta = 1.0e35;
  for (ii = 0; ii < nInputs; ii++)
  {
    vecInpBins[(int) dOrder[nInputs-ii-1]] = 1;
#pragma omp parallel shared(ss,nSamples,XX,vecRangesInv2,vecInpBins,vecYT,delta) \
    private(ddata,vecMinInds,jj,minDist,ss2,distance,kk,dtemp,ll,ind)
{
    delta = 0.0;
#pragma omp for
    for (ss = 0; ss < nSamples; ss++)
    {
      ddata = 0.0;
      vecMinInds.setLength(nNeigh_);
      for (jj = 0; jj < nNeigh_; jj++)
      {
        minDist = PSUADE_UNDEFINED;
        vecMinInds[jj] = -1;
        for (ss2 = 0; ss2 < ss; ss2++)
        {
          distance = 0.0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (vecInpBins[kk] == 1)
            {
              dtemp = XX[ss*nInputs+kk] - XX[ss2*nInputs+kk];
              distance += dtemp * dtemp * vecRangesInv2[kk];
            }
          }
          if (distance < minDist)
          {
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              minDist = distance;
              vecMinInds[jj] = ss2;
            }
          }
        }
        for (ss2 = ss+1; ss2 < nSamples; ss2++)
        {
          distance = 0.0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (vecInpBins[kk] == 1)
            {
              dtemp = XX[ss*nInputs+kk] - XX[ss2*nInputs+kk];
              distance += dtemp * dtemp * vecRangesInv2[kk];
            }
          }
          if (distance < minDist)
          {
            for (ll = 0; ll < jj; ll++)
              if (ss2 == vecMinInds[ll]) break;
            if (jj == 0 || ll == jj)
            {
              minDist = distance;
              vecMinInds[jj] = ss2;
            }
          }
        }
        ind = vecMinInds[jj];
        ddata += pow(vecYT[ss] - vecYT[ind], 2.0);
      }
#pragma omp critical
      delta += ddata / (double) nNeigh_;
    }
} /* pragma omp parallel */ 
    delta /= (2.0 * nSamples);
    if (ii == 0)
    {
      deltaSave += delta;
      for (kk = 0; kk < nInputs; kk++) printOutTS(PL_INFO, "0 ");
      printOutTS(PL_INFO, " = %e\n", deltaSave);
    }
    for (kk = 0; kk < nInputs; kk++) 
      printOutTS(PL_INFO, "%d ", vecInpBins[kk]);
    printOutTS(PL_INFO, " = %e\n", delta);
    minDeltas[ii] = delta;
    if (delta < minDelta) minDelta = delta;
  }
  printAsterisks(PL_INFO, 0);
  return minDelta;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
DeltaAnalyzer& DeltaAnalyzer::operator=(const DeltaAnalyzer &)
{
  printOutTS(PL_ERROR,"DeltaTest operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int DeltaAnalyzer::get_nBins()
{
  return nBins_;
}
int DeltaAnalyzer::get_nInputs()
{
  return nInputs_;
}
double *DeltaAnalyzer::get_minDeltas()
{
  psVector vecDT;
  vecDT = VecMinDeltas_;
  return vecDT.getDVector();
}
int **DeltaAnalyzer::get_deltaBins()
{
  psIMatrix matIT;
  matIT = MatDeltaBins_;
  return matIT.takeIMatrix2D();
}
double *DeltaAnalyzer::get_dOrder()
{
  psVector vecDT;
  vecDT = VecOrders_;
  return vecDT.getDVector();
}
int *DeltaAnalyzer::get_ranks()
{
  psIVector vecIT;
  vecIT = VecRanks_;
  return vecIT.getIVector();
}

