// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class SCEPOptimizer (Parallel SCE)
// AUTHOR : Charles Tong 
// DATE   : 2016
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "PsuadeUtil.h"
#include "SCEPOptimizer.h"
#include "PsuadeUtil.h"
#include "Sampling.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PrintingTS.h"

// ------------------------------------------------------------------------
#include <math.h> // for standev and georange functions
#include <time.h> // for random number generator
// ------------------------------------------------------------------------

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
SCEPOptimizer::SCEPOptimizer()
{
   psSCEPInputTypes_ = NULL;
   psSCEPnInputs_ = 0;
   psSCEPHistCnt_ = 0;
   psSCEPHistIndex_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
SCEPOptimizer::~SCEPOptimizer()
{
   if (psSCEPInputTypes_ != NULL) delete [] psSCEPInputTypes_;
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void SCEPOptimizer::optimize(oData *odata) 
{
   int    nInputs, printLevel=0, ii, jj, kk, maxfun, nPts, isum, iCall=0;
   int    nComplex=4, includeInitialPoint=1, maxEvoLoop=10, nOutputs;
   double dtemp, sum, Mean;
   double paramSpaceConvergence, percentChange=1;
   char   pString[1000], *cString, winput[1000];

   //**/ ------ prepare object variables ------
   nInputs  = odata->nInputs_;
   nOutputs = odata->nOutputs_;
   odata_   = odata;
   printLevel = odata->outputLevel_;
   if (nOutputs > 1)
   {
      printOutTS(PL_INFO,"INFO: nOutputs = %d.\n",nOutputs);
      printOutTS(PL_INFO,"      There will be %d constraints.\n",
                 nOutputs-1);
   }
   for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
   odata->optimalY_ = PSUADE_UNDEFINED;
   odata->numFuncEvals_ = 0;
   maxfun = odata->maxFEval_;
   paramSpaceConvergence = odata->tolerance_;

   //**/ ask for which option
   psSCEPInputTypes_ = new int[nInputs];
   for (ii = 0; ii < nInputs; ii++) psSCEPInputTypes_[ii] = 0;
   for (ii = 0; ii < nInputs; ii++)
   {
      snprintf(pString,100,"iiscrete%d", ii+1);
      cString = psConfig_.getParameter(pString);
      if (cString != NULL) 
      {
         psSCEPInputTypes_[ii] = 1;
         if (printLevel > 0)
            printf("SCE input %4d is discrete\n",ii+1);
      }
   }
   isum = 0;
   for (ii = 0; ii < nInputs; ii++) isum += psSCEPInputTypes_[ii];
   if (isum == 0 && psSCEPnInputs_ == 0)
   {
      printf("SCE can solve either \n");
      printf("1. continuous \n");
      printf("2. mixed-integer optimization.\n");
      snprintf(pString,100,"Please select (1) or (2) : ");
      jj = getInt(1, 2, pString);
      if (jj == 2)
      {
         psSCEPInputTypes_ = new int[nInputs];
         printf("SCE can handle 2 input types: \n");
         printf("1. Real (or R) - real/continuous\n");
         printf("2. Int  (or I) - integer\n");
         kk = 0;
         for (ii = 0; ii < nInputs; ii++)
         {
           snprintf(pString,100,"Please enter type for input %d : ",ii+1);
           psSCEPInputTypes_[ii] = getInt(1, 2, pString);
           if (psSCEPInputTypes_[ii] == 1) kk++;
         }
         if (kk == nInputs) 
         {
            delete [] psSCEPInputTypes_;
            psSCEPInputTypes_ = NULL;
         }
      }
   }

   // Initialize SCE parameters
   int nMemPerComplex = 2 * nInputs + 1; 
   int nMemPerSimplex = nInputs + 1;
   int nEvoStep = nMemPerComplex;
   int nPtsMultiple;
   nPts = nPtsMultiple = nMemPerComplex * nComplex;
   if (nOutputs > 1) nPtsMultiple = nPts * 5;
   psSCEPHistCnt_ = 0;
   psSCEPHistIndex_ = -1;

   // For each parameter, determine range of points from which we can sample
   double *ranges = new double[nInputs];
   for (ii = 0; ii < nInputs; ii++)
      ranges[ii] = odata->upperBounds_[ii] - odata->lowerBounds_[ii];

   // Create an initial population to fill array Xinit[npt*nInputs]
   // Use PSUADE sampling method of choice
   Sampling *sampler;
   sampler = SamplingCreateFromID(PSUADE_SAMP_MC);
   sampler->setInputBounds(nInputs,odata->lowerBounds_,odata->upperBounds_);
   sampler->setSamplingParams (nPtsMultiple, 1, 0);
   sampler->setOutputParams (1);
   sampler->initialize(0);
   nPtsMultiple = sampler->getNumSamples();
   double *Xinit = new double[nPtsMultiple*nInputs];
   double *Yinit = new double[nPtsMultiple*nOutputs];
   int    *Sinit = new int[nPtsMultiple];
   sampler -> getSamples(nPtsMultiple, nInputs, 1, Xinit, Yinit, Sinit);
   delete [] Sinit;

   //**/ determine whether to include initial point in starting population  
   if ( includeInitialPoint == 1 )
      for (ii = 0; ii < nInputs; ++ii) Xinit[ii] = odata->initialX_[ii];

   //**/ massage integer variables
   if (psSCEPInputTypes_ != NULL)
   {
      for (ii = 0; ii < nInputs; ii++)
      {
         if (psSCEPInputTypes_[ii] == 2)
         {
            for (jj = 0; jj < nPtsMultiple; jj++)
            {
               kk = (int) (Xinit[jj*nInputs+ii] + 0.5); 
               Xinit[jj*nInputs+ii] = (double) kk;
            }
         }
      }
   }
   odata->funcIO_->setDriver(1);

   //**/ determine function value at each sample point
   //**/ discard samples that violates constraints
   int nPtsTrue=0, numRuns, numFails=0;
   while ((nPtsTrue < nPts) && ((nPtsTrue+numFails) < nPtsMultiple))
   {
      numRuns = nPts - nPtsTrue;
      evaluateFunction(numRuns, &Xinit[nPtsTrue*nInputs],
                       &Yinit[nPtsTrue*nOutputs]);
      for (ii = 0; ii < numRuns; ++ii) 
      {
         for (jj = 1; jj < nOutputs; jj++) 
            if (Yinit[(nPtsTrue+ii)*nOutputs+jj] > 0) break;
         if (jj != nOutputs)
         {
            for (kk = (nPtsTrue+1)*nInputs; kk < nPtsMultiple*nInputs; kk++) 
               Xinit[kk-nInputs] = Xinit[kk];
            for (kk = (nPtsTrue+1)*nOutputs; kk < nPtsMultiple*nOutputs; kk++) 
               Yinit[kk-nOutputs] = Yinit[kk];
            numFails++;
         }
         else nPtsTrue++;
      }
   }
   if (nPtsTrue < nPts)
   {
      printf("SCEP WARNING: number of initial points = %d\n",nPtsTrue);
      printf("              Expected number          = %d\n",nPts);
      if (nPtsTrue < nMemPerComplex)
      {
         printf("SCEP ERROR: number of initial points too small.\n");
         printf("            ABORT.\n");
         exit(1);
      }
   }
   nPts = nPtsTrue;
   nComplex = nPts / nMemPerComplex;
   printf("SCEP INFO: current number of complexes = %d\n", nComplex);

   //**/ Sort the population in order of increasing function values;
   //**/ Reorganize x so that order of points corresponds to increasing 
   //**/ function value
   int    *sortArray = new int[nPts];
   double *dsortArray = new double[nPts];
   for (ii = 0; ii < nPts; ++ii) 
   {
      sortArray[ii] = ii;
      dsortArray[ii] = Yinit[ii*nOutputs];;
   }
   sortDbleList2a(nPts, dsortArray, sortArray); 

   //**/ Use sortArray as indices for sorting x1 into x
   double *X = new double[nPts*nInputs];
   double *Y = new double[nPts*nInputs];
   for (ii = 0; ii < nPts; ii++)
   {
      for (jj = 0; jj < nInputs; jj++)
         X[ii*nInputs + jj] = Xinit[sortArray[ii]*nInputs + jj];
      for (jj = 0; jj < nOutputs; jj++)
         Y[ii*nOutputs + jj] = Yinit[sortArray[ii]*nOutputs + jj];
   }
   delete [] Xinit;
   delete [] Yinit;
   Xinit = NULL;
   Yinit = NULL;
   delete [] sortArray;
   delete [] dsortArray;
   sortArray = NULL;
   dsortArray = NULL;

   //**/ Record the best and worst points
   double Xbest[nInputs]; 
   double Xworst[nInputs]; 
   for (ii = 0; ii < nInputs; ++ii)
   {
      Xbest[ii] = X[ii];
      Xworst[ii] = X[(nPts-1)*nInputs+ii];
   }
   double Ybest  = Y[0];
   double Yworst = Y[(nPts-1)*nOutputs];

   //**/ Compute the normalized geometric range of the parameters
   double geoRange = georange(nInputs, nPts, X, ranges);

   // Check for convergency 
   if (printLevel > 0 && odata->numFuncEvals_ >= maxfun)
   {
      printf("Optimization search terminated because the limit on the\n");
      printf("maximum number of trials %d has been exceeded. Search\n",
             maxfun);
      printf("was stopped at trial number %d of the initial loop!\n", 
             iCall);
   }
   if (printLevel > 0 && geoRange < paramSpaceConvergence)
      printf("Population has converged to a small parameter space (%e).\n",
             geoRange);

   if (printLevel > 0 && psSCEPnInputs_ != nInputs)
   {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO,"* SCE optimizer\n");
      printOutTS(PL_INFO,"  max fevals = %d\n", odata->maxFEval_);
      printOutTS(PL_INFO,"  tolerance  = %e\n", odata->tolerance_);
      printOutTS(PL_INFO,"     Note: this is space convergence tolerance.\n");
      printOutTS(PL_INFO,"  Also: terminate if stagnant for %d iterations\n",
                 maxfun/10);
      printEquals(PL_INFO, 0);
   }

   //**/ -----------------------------------------------------------------
   //**/ -----------------------------------------------------------------
   //**/ evolveComplex
   //**/ Begin Evolution Loops!!!!!
   //**/ XC: coordinates of complex
   //**/ YC: function value at each point of complex
   //**/ s:  coordinates of simplex
   //**/ sf: function value at each point of simplex
   double criter[1000];
   double criter_change = 100000;
   int    **partition, **simIndex, ic, flag, iter, iz, nLoop=0;
   double **XC, **YC, **XS, **YS, *XNew, *YNew, simPosition;
   double *XC2 = new double[nMemPerComplex*nInputs];
   double *X2 = new double[nInputs*nPts];

   //**/ memory allocation
   XC = new double*[nComplex];
   YC = new double*[nComplex];
   partition = new int*[nComplex];
   XS = new double*[nComplex];
   YS = new double*[nComplex];
   simIndex = new int*[nComplex];
   XNew = new double[nComplex*nInputs];
   YNew = new double[nComplex*nOutputs];
   for (ic = 0; ic < nComplex; ic++)
   {
      XC[ic] = new double[nMemPerComplex*nInputs];
      YC[ic] = new double[nMemPerComplex];
      XS[ic] = new double[nMemPerComplex*nInputs];
      YS[ic] = new double[nMemPerComplex];
      partition[ic] = new int[nMemPerComplex];
      for (ii = 0; ii < nMemPerComplex; ii++)
         partition[ic][ii] = (ii * nComplex) + ic;
      simIndex[ic] = new int[nMemPerSimplex];
   }
   sortArray = new int[nPts];

   //**/ iterate
   while ((odata->numFuncEvals_ < maxfun) && 
          (geoRange > paramSpaceConvergence) && 
          (criter_change > percentChange)) 
   {
      nLoop += 1;

      //**/ ------------- LOOP ON COMPLEXES ------------// 
      for (ic = 0; ic < nComplex; ++ic)
      {
         //**/ Partition the population into complexes (sub-populations)
       
         for (ii = 0; ii < nMemPerComplex; ++ii)
         {
            YC[ic][ii] = Y[partition[ic][ii]*nOutputs];
            for (jj = 0; jj < nInputs; ++jj)
              XC[ic][ii*nInputs + jj] = X[partition[ic][ii]*nInputs+jj];
         }
      }

      //**/ -----EVOLVE SIMPLEX WITHIN COMPLEX FOR nEvoStep STEPS
      for (ii = 0; ii < nEvoStep; ++ii)
      {
         for (ic = 0; ic < nComplex; ++ic)
         {
            //**/ Select simplex by sampling the complex according to a 
            //**/ linear probability distribution
            //**/ simIndex = array containing indices of simplex points 
            //**/            from XC
            //**/ simPosition = sampled position for simplex
            simIndex[ic][0] = 0; 
            flag = 0;
            for (jj = 1; jj < nMemPerSimplex; ++jj)
            {
               for (iter = 0; iter < 1000; ++iter)
               {
                  simPosition = floor(nMemPerComplex+0.5-sqrt((nMemPerComplex + 
                                0.5)*(nMemPerComplex+0.5)-nMemPerComplex * 
                                (nMemPerComplex + 1)*PSUADE_drand()));
 	          for (iz = 0; iz <= jj-1; ++iz)
                  {
 	             if (simIndex[ic][iz] == simPosition)
                     {
                         flag = 1;
                         break;
                     }
 	          }
 	          if (flag != 1) break;
 	       }
 	       simIndex[ic][jj] = (int) simPosition;
            }
            //**/ Sort simIndex positions
            sortIntList(nMemPerSimplex, simIndex[ic]);
	
            //**/ Construct the simplex
            //**/ XS = simplex coordinates
            //**/ YS = simplex point function values 
            for (jj = 0; jj != nMemPerSimplex; ++jj)
            {
	       YS[ic][jj] = YC[ic][simIndex[ic][jj]];
	       for (kk = 0; kk != nInputs; ++kk)
	          XS[ic][jj*nInputs + kk] = 
                     XC[ic][simIndex[ic][jj]*nInputs+kk];
            }
         }

         //**/ Call function that generates new point in a simplex
         newPoint(YNew,&iCall,nComplex,nInputs,nOutputs,nMemPerSimplex,
                  maxfun, XS, YS, XNew, odata);
      
         for (ic = 0; ic < nComplex; ++ic)
         {
            //**/ Replace the worst point in the Simplex with the new point
            for (jj = 0; jj < nInputs; ++jj)
               XS[ic][nInputs*(nMemPerSimplex-1) + jj] = XNew[ic*nInputs+jj];
            YS[ic][nMemPerSimplex-1] = YNew[ic];  

            //**/ Replace the simplex into the complex
            for (jj = 0; jj < nMemPerSimplex; jj++)
            {
	       YC[ic][simIndex[ic][jj]] = YS[ic][jj];
	       for (kk = 0; kk != nInputs; ++kk)
	          XC[ic][simIndex[ic][jj]*nInputs+kk] = 
                      XS[ic][jj*nInputs + kk];
            }

            //**/ Sort the complex
            for (jj = 0; jj < nPts; ++jj) sortArray[jj] = jj;
            sortDbleList2a(nMemPerComplex, YC[ic], sortArray);

            //**/ Use sortArray as indices for sorting XC
            //**/ Transfer contents of XC to XC2 so that we don't 
            //**/ overwrite values while sorting
            for (jj = 0; jj < nMemPerComplex*nInputs; jj++) 
               XC2[jj] = XC[ic][jj];

            for (jj = 0; jj < nMemPerComplex; jj++)
            {
	       for (kk = 0; kk < nInputs; kk++)
	          XC[ic][jj*nInputs + kk] = XC2[sortArray[jj]*nInputs + kk];
            }
            //**/ End of inner loop for competitive evolution of simplexes
         }
      }
	  
      //**/ Replace the complex back into the population;
      for (ic = 0; ic < nComplex; ++ic)
      {
         for (ii = 0; ii < nMemPerComplex; ii++)
         {
            Y[partition[ic][ii]] = YC[ic][ii];
	    for (jj = 0; jj != nInputs; ++jj)
	       X[partition[ic][ii]*nInputs+jj] = XC[ic][ii*nInputs + jj];
         }
      }

      //**/ Shuffle the complexes by sorting Y and X
      for (ii = 0; ii < nPts; ++ii) sortArray[ii] = ii;
      sortDbleList2a(nPts, Y, sortArray); 

      //**/ Transfer contents of X to X2 so that we don't overwrite 
      //**/ values while sorting
      for (ii = 0; ii < nPts*nInputs; ii++) X2[ii] = X[ii];

      //**/ Use sortArray as indices for sorting x
      for (ii = 0; ii != nPts; ++ii)
      {
         for (jj = 0; jj != nInputs; ++jj)
            X[ii*nInputs + jj] = X2[sortArray[ii]*nInputs + jj];
      }

      // Record the best and worst points
      for (ii = 0; ii < nInputs; ii++)
      {
          Xbest[ii] = X[ii];
          Xworst[ii] = X[(nPts-1)*nInputs+ii];
      }

      Ybest = Y[0];
      Yworst = Y[(nPts-1)*nOutputs];

      //**/ Compute the normalized geometric range of the parameters
      geoRange = georange(nInputs, nPts, X,  ranges);

      //**/ Check for convergency
      if (printLevel > 0 && odata->numFuncEvals_ >= maxfun)
      {
         printf("*** Optimization search terminated because the limit on\n");
	 printf("maximum number of trials %d has been exceeded.\n", maxfun);
      }
      if (printLevel > 0 && geoRange < paramSpaceConvergence)
      {
         printf("Population has converged to a small parameter space (%e).\n",
                geoRange);
         break;
      }
      if (psSCEPHistCnt_ == psSCEPnHist_)
      {
         sum = 0.0;
         for (ii = 0; ii < psSCEPnHist_; ++ii) sum += PABS(psSCEPHistory_[ii]);
         sum /= (double) psSCEPnHist_;
         for (ii = psSCEPnHist_-3; ii < psSCEPnHist_; ii++)
         {
            dtemp = PABS(psSCEPHistory_[ii] - psSCEPHistory_[ii-1]);
            if (sum != 0) dtemp /= sum;
            if (dtemp > odata->tolerance_) break;
         }
         if (ii == psSCEPnHist_) break;
      }
      if (printLevel > 0 && (psSCEPHistIndex_ >= 0) && 
          (psSCEPHistCnt_ != psSCEPnHist_) &&
          (odata->numFuncEvals_-psSCEPHistIndex_ > maxfun/10))
      {
         printf("*** Optimization search terminated due to stagnation\n");
         break;
      }
    
      criter[nLoop-1] = Ybest;
      if (odata->numFuncEvals_ < maxfun && nLoop >= maxEvoLoop)
      {
         if (Ybest != criter[nLoop - maxEvoLoop])
         {
	    criter_change = fabs(criter[nLoop-1]-criter[nLoop-maxEvoLoop])*100;
            sum = 0.0; 
	    for (ii = nLoop-maxEvoLoop; ii != nLoop-1; ++ii)
	       sum += fabs(criter[ii]);
	    Mean = sum/maxEvoLoop;
	    criter_change = criter_change/Mean;
	    if (printLevel > 0 && criter_change < percentChange)
            {
	       printf("The best point has improved in last %d loops by less\n",
                      maxEvoLoop);
	       printf("than the threshold %e => convergence achieved.\n",
                      percentChange);
	    }
         }
      }
   } 
   //**/ End of the while loop
   delete [] sortArray;
   delete [] XC2;
   for (ic = 0; ic < nComplex; ic++)
   {
      delete [] XC[ic];
      delete [] YC[ic];
      delete [] partition[ic];
      delete [] simIndex[ic];
   }
   delete [] XC;
   delete [] YC;
   delete [] partition;
   delete [] simIndex;
   delete [] X2;
   delete [] ranges;
   delete [] XNew;
   delete [] YNew;
 
   if (printLevel > 0)
   {
      for (ii = 0; ii < 60; ii++) printf("*");
      printf("\n");
      printf("Search was stopped at trial number %d\n", iCall);
      printf("Normalized geometric range = %e\n", geoRange);
      printf("Calculated global optimum = %e\n", Ybest);
   }
   psSCEPnInputs_ = nInputs;
   if (printLevel > 0)
   {
      printf("SCEPOptimizer: number of function evaluations = %d\n",
             odata->numFuncEvals_);
   }
}

// ***********************************************************************
// georange function: calculates geometric range of each parameter
// -----------------------------------------------------------------------
double SCEPOptimizer::georange(int nInputs,int nPts,double* x,double* ranges)
{
   //**/ Step 1: determine max and min of each parameter
   int    ii, jj, count=0;
   double y[nInputs], maxval, minval, sum, mean;
   for (ii = 0; ii != nInputs; ++ii)
   {
      maxval = x[ii];
      minval = x[ii];
      for (jj = ii; jj < (nPts*nInputs); jj += nInputs)
      {
         if      (x[jj] > maxval) maxval = x[jj];
         else if (x[jj] < minval) minval = x[jj];
      }
      y[ii] = maxval - minval;
      if (y[ii] > 0) count++;
   }

   //**/ Step 2: divide each element of y by each corresponding 
   //**/ element of ranges
   for (ii = 0; ii != nInputs; ++ii) y[ii] = y[ii] / ranges[ii];

   //**/ Step 3: compute log of each element of y
   for (ii = 0; ii != nInputs; ++ii) 
      if (y[ii] > 0) y[ii] = log(y[ii]);

   //**/ Step 4: Compute mean of elements in y and exponentiate the result
   sum = 0.0;
   for (ii = 0; ii != nInputs; ++ii) 
      if (y[ii] > 0) sum += y[ii];
   if (count > 0) mean = sum / (double) count;
   else           mean = -50.0;
   return exp(mean);
}

// **********************************************************************
// newPoint function: generates a new point in a simplex
// ----------------------------------------------------------------------
//**/ s = the sorted simplex in order of increasing function values
//**/ sf = function values in increasing order
//**/LIST OF LOCAL VARIABLES
//**/ XWorst = the worst point of the simplex
//**/ secWorst = the second worst point of the simplex
//**/ YBest = function value of the best point
//**/ YWorst = function value of the worst point
//**/ XNew = new generated simplex point
//**/ YNew = function value of new point
//**/ centroid = centroid of simplex excluding worst point
// ----------------------------------------------------------------------
void SCEPOptimizer::newPoint(double *YNew, int* iCall, int nComplex, 
                             int nInputs, int nOutputs, int nMemPerSimplex, 
                             int maxfun, double **XS, double **YS, 
                             double *XNew, oData *odata)
{
   int    ii, jj, ic, outOfBound = 0;
   double alpha = 1.0, sum, beta = 0.5;
   double XWorst[nInputs*nComplex], YWorst[nComplex];
   double YNew1[nComplex*nOutputs], centroid[nComplex*nInputs];
   double X1[nInputs], X2[nInputs];

   for (ic = 0; ic < nComplex; ic++)
   {
      //**/ Assign the best and worst points
      for (ii = 0; ii < nInputs; ii++)
         XWorst[ic*nInputs+ii] = XS[ic][nInputs*(nMemPerSimplex-1)+ii];
      YWorst[ic] = YS[ic][nMemPerSimplex-1];

      //**/ Compute the centroid of the simplex excluding the worst point
      for (ii = 0; ii != nInputs; ++ii)
      { 
         sum = 0.0;
         for (jj = ii; jj < ((nMemPerSimplex-1)*nInputs); jj += nInputs)
            sum += XS[ic][jj];
         centroid[ic*nComplex+ii]= sum/(nMemPerSimplex-1);
      }

      //**/ Attempt a reflection point
      for (ii = 0; ii < nInputs; ii++)
         XNew[ic*nInputs+ii] = centroid[ic*nComplex+ii]+alpha*
                              (centroid[ic*nComplex+ii]-XWorst[ic*nInputs+ii]);

      //**/ Check if it is outside the bounds
      for (ii = 0; ii != nInputs; ++ii)
      {
         X1[ii] = XNew[ii] - odata->lowerBounds_[ii];
         X2[ii] = odata->upperBounds_[ii] - XNew[ii];
      }

      for (ii = 0; ii < nInputs; ii++)
      {
         if ((X1[ii]<0) || (X2[ii]<0))
         {
            outOfBound = 1;
            break;
         }
      }
      if (outOfBound == 1)
      {
         for (ii = 0; ii != nInputs ; ++ii)
            XNew[ic*nInputs+ii] = odata->lowerBounds_[ii] + PSUADE_drand() *
                       (odata->upperBounds_[ii] - odata->lowerBounds_[ii]);
      }

      //**/ massage integer variables before evaluation
      if (psSCEPInputTypes_ != NULL)
      {
         for (ii = 0; ii < nInputs; ii++)
         {
            if (psSCEPInputTypes_[ii] == 2)
            {
               jj = (int) (XNew[ii] + 0.5); 
               XNew[ic*nInputs+ii] = (double) jj;
            }
         }
      }
   }
   evaluateFunction(nComplex, XNew, YNew1);
   *iCall += nComplex;

   //**/ Reflection failed; now attempt a contraction step
   for (ic = 0; ic < nComplex; ic++)
   {
      if (YNew1[ic*nOutputs] > YWorst[ic])
      {
         for (ii = 0; ii < nInputs; ii++)
            XNew[ic*nInputs+ii] = XWorst[ic*nInputs+ii]+beta*
                       (centroid[ic*nComplex+ii]-XWorst[ic*nInputs+ii]);
         //**/ massage integer variables
         if (psSCEPInputTypes_ != NULL)
         {
            for (ii = 0; ii < nInputs; ii++)
            {
               if (psSCEPInputTypes_[ii] == 2)
               {
                  jj = (int) (XNew[ic*nInputs+ii] + 0.5); 
                  XNew[ic*nInputs+ii] = (double) jj;
               }
            }
         }
         evaluateFunction(1, &XNew[ic*nInputs], &YNew1[ic]);
         *iCall += 1;

         //**/ Both reflection and contraction have failed, try a random point
         if (YNew1[ic] > YWorst[ic])
         {
            for (ii = 0; ii < nInputs ; ii++)
               XNew[ii] = odata->lowerBounds_[ii] + PSUADE_drand() *
                       (odata->upperBounds_[ii] - odata->lowerBounds_[ii]);
            if (psSCEPInputTypes_ != NULL)
            {
               for (ii = 0; ii < nInputs; ii++)
               {
                  if (psSCEPInputTypes_[ii] == 2)
                  {
                     jj = (int) (XNew[ic*nInputs+ii] + 0.5); 
                     XNew[ii] = (double) jj;
                  }
               }
            }
            evaluateFunction(1, &XNew[ic*nInputs], &YNew1[ic]);
            *iCall += 1;
         }
      }
      YNew[ic] = YNew1[ic];
   }
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
SCEPOptimizer& SCEPOptimizer::operator=(const SCEPOptimizer &)
{
   printf("SCEPOptimizer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

// ************************************************************************
// evaluate function operator
// ------------------------------------------------------------------------
void SCEPOptimizer::evaluateFunction(int nSamp, double *XValues, 
                                     double *YValues)
{
   int    ii, jj, index, funcID, nInputs, nOutputs;
   double Ymin;

   //**/ ------ fetch data ------
   nInputs  = odata_->nInputs_;
   nOutputs = odata_->nOutputs_;

   //**/ ------ run simulation ------
   funcID = odata_->numFuncEvals_;
   odata_->funcIO_->evaluate(nSamp,nInputs,XValues,nOutputs,YValues,funcID);
   odata_->numFuncEvals_ += nSamp;

   //**/ ------ store optimal information ------
   Ymin  = PSUADE_UNDEFINED;
   index = -1;
   for (ii = 0; ii < nSamp; ii++)
   {
      if (YValues[ii*nOutputs] < Ymin) 
      {
         for (jj = 0; jj < nOutputs-1; jj++)
            if (YValues[ii*nOutputs+jj+1] > 0) break;
         if (jj == nOutputs-1)
         {
            Ymin = YValues[ii*nOutputs];
            index = ii;
         }
      }
   }
   if (index >= 0 && Ymin < odata_->optimalY_)
   {
      odata_->optimalY_ = Ymin;
      for (ii = 0; ii < nInputs; ii++) 
         odata_->optimalX_[ii] = XValues[index*nInputs+ii];
      psSCEPHistIndex_ = funcID;
      if (odata_->outputLevel_ > 0)
      {
         printf("SCEPOptimizer %6d : \n", odata_->numFuncEvals_);
         if (odata_->outputLevel_ > 1)
         {
            for (ii = 0; ii < nInputs; ii++)
               printf("    X %6d = %16.8e\n", ii+1, XValues[ii]);
            printf("    Ymin  = %16.8e\n", odata_->optimalY_);
         }
         if (psSCEPHistCnt_ < psSCEPnHist_) 
            psSCEPHistory_[psSCEPHistCnt_++] = Ymin;
         else
         {
            for (ii = 1; ii < psSCEPnHist_; ii++) 
               psSCEPHistory_[ii-1] = psSCEPHistory_[ii];
            psSCEPHistory_[psSCEPnHist_-1] = Ymin;
            psSCEPHistCnt_++;
         }
      }
   }
}

