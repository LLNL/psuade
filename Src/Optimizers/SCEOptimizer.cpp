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
// Functions for the class SCEOptimizer
// AUTHOR : Amer Abdulla 
// DATE   : 2009
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "PsuadeUtil.h"
#include "SCEOptimizer.h"
#include "PsuadeUtil.h"
#include "Sampling.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PrintingTS.h"
#include "IntOptimizer.h"

// ------------------------------------------------------------------------
#include <math.h> // for standev and georange functions
#include <time.h> // for random number generator
// ------------------------------------------------------------------------

#define psSCEMaxSaved_ 100000
int     psSCENSaved_=0;
double  psSCESaveX_[psSCEMaxSaved_*10];
double  psSCESaveY_[psSCEMaxSaved_];
void    *psSCEObj_=NULL;
psIVector VecPsSCEInpTypes_;
int     psSCEnInputs_=0;
#define psSCEnHist_ 10
double  psSCEHistory_[psSCEnHist_];
int     psSCEHistCnt_=0;
int     psSCEHistIndex_=0;
int     psSCECurrDriver_=-1;

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" 
{
#endif
  void *SCEevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    ii, jj, kk, funcID, nInputs, nOutputs, outputID, found;
    double *localY;
    oData  *odata;
    FILE   *infile;

    //**/ ------ get history ------
    nInputs = (*nInps);
    if (psConfig_.OptExpertModeIsOn() && psSCENSaved_ == 0)
    {
      infile = fopen("psuade_sce_data","r");
      if (infile != NULL)
      {
        fscanf(infile, "%d %d", &psSCENSaved_, &kk);
        if ((psSCENSaved_ <= 0) ||
            (psSCENSaved_+1 > psSCEMaxSaved_*10/nInputs))
        {
          printf("PSUADE SCE: history file has too much data.\n");
          printf("            Ask PSUADE developer for help.\n"); 
          fclose(infile);
          psSCENSaved_ = 0;
        }
        else if (kk != nInputs)
        {
          printf("PSUADE SCE: history file has invalid input count.\n");
          fclose(infile);
          psSCENSaved_ = 0;
        }
        else
        {
          for (ii = 0; ii < psSCENSaved_; ii++)
          {
            fscanf(infile, "%d", &kk);
            if (kk != ii+1)
            {
              printf("PSUADE SCE: data index mismatch.\n");
              psSCENSaved_ = 0;
              break;
            }
            for (jj = 0; jj < nInputs; jj++)
              fscanf(infile, "%lg", &psSCESaveX_[ii*nInputs+jj]);
            fscanf(infile, "%lg", &psSCESaveY_[ii]);
          }
          fclose(infile);
        }
      }
    }

    //**/ ------ fetch data ------
    odata    = (oData *) psSCEObj_;
    nOutputs = odata->nOutputs_;
    localY   = (double *) malloc(nOutputs * sizeof(double));
    outputID = odata->outputID_;

    //**/ ------ search to see if it has already been evaluated ------
    found = 0;
    for (ii = 0; ii < psSCENSaved_; ii++)
    {
      for (jj = 0; jj < nInputs; jj++)
        if (PABS(psSCESaveX_[ii*nInputs+jj]-XValues[jj])>1.0e-14) break;
      if (jj == nInputs)
      {
        found = 1;
        break;
      }
    }

    //**/ ------ run simulation ------
    funcID = odata->numFuncEvals_;
    if (found == 0)
    {
      odata->funcIO_->evaluate(funcID,nInputs,XValues,nOutputs,localY,0);
      odata->numFuncEvals_++;
    }
    else localY[outputID] = psSCESaveY_[ii];
    (*YValue) = localY[outputID];

    //**/ ------ save the data ------
    if (psConfig_.OptExpertModeIsOn() && found == 0)
    {
      for (jj = 0; jj < nInputs; jj++)
        psSCESaveX_[psSCENSaved_*nInputs+jj] = XValues[jj];
      psSCESaveY_[psSCENSaved_] = localY[outputID];
      psSCENSaved_++;
    }
    // Use for Diagnostics
    //for (ii = 0; ii < nInputs; ii++)
    //printf("    X %6d = %16.8e\n", ii+1, XValues[ii]);
    //printf("    Y = %16.8e\n", localY[outputID]);

    //**/ ------ store optimal information ------
    if ((*YValue) < odata->optimalY_)
    {
      odata->optimalY_ = (*YValue);
      for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = XValues[ii];
      psSCEHistIndex_ = funcID;
      if (psConfig_.InteractiveIsOn() && odata->outputLevel_ > 0)
      {
        printf("SCEOptimizer %6d : \n", odata->numFuncEvals_);
        for (ii = 0; ii < nInputs; ii++)
          printf("    X %6d = %16.8e\n", ii+1, XValues[ii]);
        printf("    Ymin  = %16.8e\n", odata->optimalY_);
      }
      if (psSCEHistCnt_ < psSCEnHist_) 
        psSCEHistory_[psSCEHistCnt_++] = (*YValue); 
      else
      {
        for (ii = 1; ii < psSCEnHist_; ii++) 
          psSCEHistory_[ii-1] = psSCEHistory_[ii];
        psSCEHistory_[psSCEnHist_-1] = (*YValue);
        psSCEHistCnt_++;
      }
    }
    free(localY);

    //**/ ------ save history ------
    if (psConfig_.OptExpertModeIsOn() && found == 0)
    {
      infile = fopen("psuade_sce_data","w");
      if (infile != NULL)
      {
        fprintf(infile, "%d %d\n", psSCENSaved_, nInputs);
        for (ii = 0; ii < psSCENSaved_; ii++)
        {
          fprintf(infile, "%d ", ii+1);
          for (jj = 0; jj < nInputs; jj++)
            fprintf(infile, "%24.16e ", psSCESaveX_[ii*nInputs+jj]);
          fprintf(infile, "%24.16e\n", psSCESaveY_[ii]);
        }
        fclose(infile);
      }
    }
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
SCEOptimizer::SCEOptimizer()
{
  if (psConfig_.InteractiveIsOn())
  { 
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printf("*   SCE Optimization (Pattern Search, Mixed Integer option)\n");
    printEquals(PL_INFO, 0);
    printf("* - To run this optimizer in batch mode, first make sure\n");
    printf("*   opt_driver in your PSUADE input file has been set to\n");
    printf("*   point to your objective function evaluator.\n");
    printf("* - Set optimization tolerance in your PSUADE input file\n");
    printf("* - Set maximum number of iterations in PSUADE input file\n");
    printf("* - Set num_local_minima to perform multistart optimization\n");
    printf("* - Set optimization print_level to give more screen outputs\n");
    printf("* - In opt_expert mode, you can \n");
    printf("*    + choose between continuous or discrete variables\n");
    printf("*    + choose another integer optimizer or the discrete SCE\n");
    printf("*      version if all variables are integers.\n"); 
    printf("*    + the number of complexes (more is better but slower)\n");
    printf("* - If your opt_driver is a response surface which has more\n");
    printf("*   inputs than the number of optimization inputs, you can\n");
    printf("*   fix some driver inputs by creating an rs_index_file.\n");
    printAsterisks(PL_INFO, 0);
  }
  nComplex_ = 2;
  psSCENSaved_ = 0;
  psSCEObj_ = NULL;
  psSCEnInputs_ = 0;
  psSCEHistCnt_ = 0;
  psSCEHistIndex_ = 0;
  psSCECurrDriver_ = -1;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
SCEOptimizer::~SCEOptimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void SCEOptimizer::optimize(oData *odata) 
{
  int    nInputs, printLevel=0, ii, jj, kk, maxfun, nPts, isum;
  int    includeInitialPoint=1, maxEvoLoop=10, nOutputs, iZero=0;
  double dtemp, sum, Mean;
  double paramSpaceConvergence=0.0001, percentChange=0.1;
  char   pString[1000], *cString, winput[1000];
  FILE   *fp=NULL;

  //**/---------------------------------------------------------
  //**/ ------ prepare object variables ------
  //**/---------------------------------------------------------
  printLevel = odata->outputLevel_;
  nInputs  = odata->nInputs_;
  nOutputs = odata->nOutputs_;
  if (nOutputs > 1)
  {
    printOutTS(PL_ERROR,"SCE ERROR: nOutputs = %d.\n",nOutputs);
    printOutTS(PL_ERROR,"       Only nOutputs=1 is allowed.\n");
    printOutTS(PL_ERROR,"       SCE cannot handle constraints.\n");
    printOutTS(PL_ERROR,"       Suggestion: use penalty term.\n");
    exit(1);
  }
  for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
  maxfun = odata->maxFEval_;
  odata->optimalY_ = 1.0e50;
  int lastNumFuncEvals = odata->numFuncEvals_;

  //**/---------------------------------------------------------
  //**/ set optimization flag
  //**/---------------------------------------------------------
  if ((odata->setOptDriver_ & 1))
  {
    psSCECurrDriver_ = odata->funcIO_->getDriver();
    odata->funcIO_->setDriver(1);
  }
  psSCEObj_= (void *) odata;
  percentChange = odata->tolerance_;

  //**/---------------------------------------------------------
  //**/ initially, look to see if there are discrete variables
  //**/---------------------------------------------------------
  if (psSCEnInputs_ == 0)
  {
    VecPsSCEInpTypes_.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++) VecPsSCEInpTypes_[ii] = 1;
    for (ii = 0; ii < nInputs; ii++)
    {
      snprintf(pString,100,"iDiscrete%d", ii+1);
      cString = psConfig_.getParameter(pString);
      if (cString != NULL) 
      {
        VecPsSCEInpTypes_[ii] = 2;
        if (printLevel > 0)
          printf("SCE input %4d is discrete\n",ii+1);
      }
    }
  }

  //**/---------------------------------------------------------
  //**/ if there are no discrete variable from the config object,
  //**/ and opt expert mode is on, ask individually
  //**/---------------------------------------------------------
  isum = 0;
  for (ii = 0; ii < nInputs; ii++) isum += VecPsSCEInpTypes_[ii];
  if (isum == nInputs && psSCEnInputs_ == 0 && 
      psConfig_.OptExpertModeIsOn())
  {
    printf("SCE can solve either \n");
    printf("1. continuous \n");
    printf("2. mixed-integer optimization.\n");
    snprintf(pString,100,"Please select (1) or (2) : ");
    jj = getInt(1, 2, pString);
    if (jj == 2)
    {
      VecPsSCEInpTypes_.setLength(nInputs);
      printf("SCE can handle 2 input types: \n");
      printf("1. Real (or R) - real/continuous\n");
      printf("2. Int  (or I) - integer\n");
      for (ii = 0; ii < nInputs; ii++)
      {
        snprintf(pString,100,"Please enter type for input %d : ",ii+1);
        VecPsSCEInpTypes_[ii] = getInt(1, 2, pString);
      }
    }
    snprintf(pString,100,"Select number of complexes (2-10, default=4): ");
    nComplex_ = getInt(1, 10, pString);
  }

  //**/---------------------------------------------------------
  //**/ if all discrete variables, ask if want to use another optimizer
  //**/---------------------------------------------------------
  isum = 0;
  for (ii = 0; ii < nInputs; ii++) isum += VecPsSCEInpTypes_[ii];
  if (isum == 2*nInputs &&  psConfig_.OptExpertModeIsOn() &&
      psConfig_.InteractiveIsOn())
  {
    printf("All inputs are discrete. You can stay with the discrete\n");
    printf("version of this optimizer, or you can use another integer\n");
    printf("optimizer.\n");
    printf("Want to use another integer optimizer? (y or n) ");
    scanf("%s", pString);
    fgets(winput, 100, stdin);
    if (pString[0] == 'y')
    {
      printf("use INT optimizer\n");
      INTOptimizer *intOpt = new INTOptimizer();
      intOpt->optimize(odata);
      return;
    }
  }

  //**/---------------------------------------------------------
  //**/ else if all real variables, delete input type array
  //**/---------------------------------------------------------
  if (isum == nInputs) VecPsSCEInpTypes_.setConstant(iZero);

  //**/---------------------------------------------------------
  //**/ if stay with SCE, ask more information
  //**/---------------------------------------------------------
  if (isum == nInputs && psSCEnInputs_ == 0 && 
      psConfig_.OptExpertModeIsOn())
  {
    snprintf(pString,100,"Select number of complexes (2-10, default=4): ");
    nComplex_ = getInt(1, 10, pString);
  }

  //**/---------------------------------------------------------
  // Initialize SCE parameters
  //**/---------------------------------------------------------
  int nMemPerComplex = 2 * nInputs + 1; 
  int nMemPerSimplex = nInputs + 1;
  int nEvoStep = nMemPerComplex;
  nPts = nMemPerComplex * nComplex_;
  psSCEHistCnt_ = 0;
  psSCEHistIndex_ = -1;

  //**/---------------------------------------------------------
  // For each parameter, determine range of points from which 
  // we can sample
  //**/---------------------------------------------------------
  psVector vecRanges;
  vecRanges.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
    vecRanges[ii] = odata->upperBounds_[ii] - odata->lowerBounds_[ii];

  //**/---------------------------------------------------------
  // Create an initial population to fill array XT[npt*nInputs]
  // Use PSUADE sampling method of choice
  //**/---------------------------------------------------------
  Sampling *sampler;
  sampler = SamplingCreateFromID(PSUADE_SAMP_MC);
  sampler->setInputBounds(nInputs,odata->lowerBounds_,
                          odata->upperBounds_);
  sampler->setSamplingParams (nPts, 1, 0);
  sampler->setOutputParams (1);
  sampler->initialize(0);
  nPts = sampler->getNumSamples();
  psVector  vecXT, vecYT;
  psIVector vecST;
  vecXT.setLength(nPts*nInputs);
  vecYT.setLength(nPts);
  vecST.setLength(nPts);
  double *XT = vecXT.getDVector();
  double *YT = vecYT.getDVector();
  sampler->getSamples(nPts,nInputs,1,vecXT.getDVector(),
                      vecYT.getDVector(),vecST.getIVector());

  //**/---------------------------------------------------------
  //**/ determine whether to include initial point in 
  //**/ starting population  
  //**/---------------------------------------------------------
  if (includeInitialPoint == 1)
    for (ii = 0; ii < nInputs; ++ii) XT[ii] = odata->initialX_[ii];

  //**/---------------------------------------------------------
  //**/ massage integer variables
  //**/---------------------------------------------------------
  if (VecPsSCEInpTypes_.length() > 0)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      if (VecPsSCEInpTypes_[ii] == 2)
      {
        for (jj = 0; jj < nPts; jj++)
        {
          kk = (int) (XT[jj*nInputs+ii] + 0.5); 
          vecXT[jj*nInputs+ii] = (double) kk;
        }
      }
    }
  }

  //**/---------------------------------------------------------
  //**/ determine function value at each sample point
  //**/---------------------------------------------------------
  int iCall = 0;
  int nLoop = 0;
  for (ii = 0; ii < nPts; ++ii)
  {
    SCEevalfunc_(&nInputs, &(XT[ii*nInputs]), &(YT[ii]));
    iCall += 1;
  }

  //**/---------------------------------------------------------
  //**/ Sort the population in order of increasing function values;
  //**/ Reorganize vecXX so that order of points corresponds to 
  //**/ increasing function value
  //**/---------------------------------------------------------
  vecST.setLength(nPts);
  for (ii = 0; ii < nPts; ++ii) vecST[ii] = ii;
  sortDbleList2a(nPts, vecYT.getDVector(), vecST.getIVector()); 

  //**/---------------------------------------------------------
  //**/ Use vecST as indices for sorting XT into vecXX
  //**/---------------------------------------------------------
  psVector vecXX;
  vecXX.setLength(nPts*nInputs);
  for (ii = 0; ii < nPts; ii++)
    for (jj = 0; jj < nInputs; jj++)
      vecXX[ii*nInputs + jj] = XT[vecST[ii]*nInputs + jj];

  //**/---------------------------------------------------------
  //**/ Record the best and worst points
  //**/---------------------------------------------------------
  double bestx[nInputs]; 
  double worstx[nInputs]; 
  for (ii = 0; ii < nInputs; ++ii)
  {
    bestx[ii]  = vecXX[ii];
    worstx[ii] = vecXX[(nPts-1)*nInputs+ii];
  }

  double bestf  = YT[0];
  double worstf = YT[nPts -1];

  //**/---------------------------------------------------------
  //**/ Compute the normalized geometric range of the parameters
  //**/---------------------------------------------------------
  double geoRange = georange(nInputs, nPts, vecXX.getDVector(), 
                             vecRanges.getDVector());

  //**/---------------------------------------------------------
  //**/ Display status of algorithm to let us know our progress
  //**/printf("The Initial Loop: 0\n");
  //**/printf("Number of function evaluations = %d\n", iCall);
  //**/printf("Current best function value = %e\n", bestf);
  //**/printf(" X at current best function value = \n");
  //**/for (ii = 0; ii != nInputs; ++ii) printf("%e ", bestx[ii]);
  //**/printf("\n");
  //**/printf("Current worst function value = %e\n", worstf);
  //**/printf(" X at current worst function value = \n");
  //**/for (ii = 0; ii != nInputs; ++ii) printf("%e ", worstx[ii]);
  //**/printf("\n");
  //**/---------------------------------------------------------

  //**/---------------------------------------------------------
  // Check for convergence 
  //**/---------------------------------------------------------
  if (printLevel > 0 && (odata->numFuncEvals_-lastNumFuncEvals) >= maxfun)
  {
    printf("Optimization search terminated because the limit on\n");
    printf("the maximum number of trials %d has been exceeded.\n",
           maxfun);
    printf("Search was stopped at trial number %d of the initial loop!\n", 
           iCall);
  }

  if (printLevel > 0 && geoRange < paramSpaceConvergence)
    printf("Population has converged to a small parameter space (%e).\n",
           geoRange);

  if (psConfig_.InteractiveIsOn() && printLevel > 0 && 
      psSCEnInputs_ != nInputs)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* SCE optimizer\n");
    printOutTS(PL_INFO,"  max fevals = %d\n", odata->maxFEval_);
    printOutTS(PL_INFO,"  tolerance  = %e\n", odata->tolerance_);
    printOutTS(PL_INFO,"     Note: this is space convergence tolerance.\n");
    printOutTS(PL_INFO,"  Also: terminate if stagnant for %d iterations\n",
               maxfun/2);
    printEquals(PL_INFO, 0);
  }

  //**/---------------------------------------------------------
  //**/ evolveComplex
  //**/ Begin Evolution Loops!!!!!
  //**/ vecCX: coordinates of complex
  //**/ vecXF: function value at each point of complex
  //**/ vecSX: coordinates of simplex
  //**/ vecSF: function value at each point of simplex
  //**/---------------------------------------------------------
  psVector vecCriteria;
  vecCriteria.setLength(maxfun);
  double criter_change = 100000;
  int    ic, partitions[nMemPerComplex], flag, iter,iz;
  psVector vecCX;
  vecCX.setLength(nMemPerComplex*nInputs);
  psVector vecCF;
  vecCF.setLength(nMemPerComplex);
  psIVector vecSimIndices;
  vecSimIndices.setLength(nMemPerSimplex);
  double simPosition;
  psVector vecSX, vecSF;
  vecSX.setLength(nMemPerComplex*nInputs);
  vecSF.setLength(nMemPerComplex);
  psVector vecSXNew;
  vecSXNew.setLength(nInputs);
  double sfNew;
  psVector vecCX2;
  vecCX2.setLength(nMemPerComplex*nInputs);
  psVector vecX2;
  vecX2.setLength(nInputs*nPts);
  vecST.setLength(nPts);

  if (psConfig_.InteractiveIsOn() && printLevel > 0)
  {
    printf("SCE Optimization search begins ... \n");
    printf("** To terminate gracefully, create a file called\n");
    printf("** psuade_stop (e.g. 'touch psuade_stop') in the\n");
    printf("** run directory.\n");
    printf("** For more screen dump, create a psuade_print file.\n");
    printf("** For less screen dump, remove psuade_print.\n");
  }
  while ((odata->numFuncEvals_-lastNumFuncEvals < maxfun) && 
         (geoRange > paramSpaceConvergence) && 
         (criter_change > percentChange)) 
  {
    nLoop += 1;

    //**/ ------------- LOOP ON COMPLEXES ------------// 
    for (ic = 0; ic < nComplex_; ++ic)
    {
      //**/ Partition the population into complexes (sub-populations)
      for (ii = 0; ii < nMemPerComplex; ++ii)
        partitions[ii] = (ii * nComplex_) + ic;
     
      for (ii = 0; ii < nMemPerComplex; ++ii)
      {
        vecCF[ii] = YT[partitions[ii]];
        for (jj = 0; jj < nInputs; ++jj)
          vecCX[ii*nInputs + jj] = vecXX[partitions[ii]*nInputs+jj];
      }

      //**/ -----EVOLVE SIMPLEX WITHIN COMPLEX FOR nEvoStep STEPS
      for (ii = 0; ii < nEvoStep; ++ii)
      {
        //**/ Select simplex by sampling the complex according to a 
        //**/ linear probability distribution
        //**/ vecSimIndices = array containing indices of simplex 
        //**/                 points from CX
        //**/ simPosition = sampled position for simplex
        vecSimIndices[0] = 0; 
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
 	      if (vecSimIndices[iz] == simPosition)
              {
                flag = 1;
                break;
              }
 	    }
 	    if (flag != 1) break;
 	  }
 	  vecSimIndices[jj] = (int) simPosition;
        }
        //**/ Sort vecSimIndices positions
        sortIntList(nMemPerSimplex, vecSimIndices.getIVector());
	
        //**/ Construct the simplex
        //**/ vecSX = simplex coordinates
        //**/ vecSF = simplex point function values 
        for (jj = 0; jj != nMemPerSimplex; ++jj)
        {
	  vecSF[jj] = vecCF[vecSimIndices[jj]];
	  for (kk = 0; kk != nInputs; ++kk)
	    vecSX[jj*nInputs + kk] = vecCX[vecSimIndices[jj]*nInputs+kk];
        }
        //**/ Call function that generates new point in a simplex
        newPoint(&sfNew,&iCall,nInputs,nMemPerSimplex,maxfun, 
                 vecSX.getDVector(), vecSF.getDVector(), 
                 vecSXNew.getDVector(), odata);
      
        //**/ Replace the worst point in the Simplex with the new point
        for (jj = 0; jj < nInputs; ++jj)
          vecSX[nInputs*(nMemPerSimplex-1) + jj] = vecSXNew[jj];
        vecSF[nMemPerSimplex-1] = sfNew;  

        //**/ Replace the simplex into the complex
        for (jj = 0; jj != nMemPerSimplex; ++jj)
        {
	  vecCF[vecSimIndices[jj]] = vecSF[jj];
	  for (kk = 0; kk != nInputs; ++kk)
	    vecCX[vecSimIndices[jj]*nInputs+kk] = vecSX[jj*nInputs + kk];
        }

        for (jj = 0; jj < nPts; ++jj) vecST[jj] = jj;

        //**/ Sort the complex
        sortDbleList2a(nMemPerComplex,vecCF.getDVector(),
                       vecST.getIVector());

        //**/ Use vecST as indices for sorting vecCX
        //**/ Transfer contents of vecCX to vecCX2 so that we don't 
        //**/ overwrite values while sorting
        for (jj = 0; jj != nMemPerComplex*nInputs; ++jj) 
          vecCX2[jj] = vecCX[jj];

        for (jj = 0; jj != nMemPerComplex; ++jj)
        {
	  for (kk = 0; kk != nInputs; ++kk)
	    vecCX[jj*nInputs + kk] = vecCX2[vecST[jj]*nInputs + kk];
        }
        //**/ End of inner loop for competitive evolution of simplexes
      }

      //**/ Replace the complex back into the population;
      for (ii = 0; ii != nMemPerComplex; ++ii)
      {
        YT[partitions[ii]] = vecCF[ii];
        for (jj = 0; jj != nInputs; ++jj)
	  vecXX[partitions[ii]*nInputs+jj] = vecCX[ii*nInputs + jj];
      }
      //**/ End of Loop on Complex Evolution
    }

    //**/ Shuffle the complexes by sorting vecYT and vecXX
    for (ii = 0; ii < nPts; ++ii) vecST[ii] = ii;
    sortDbleList2a(nPts, YT, vecST.getIVector()); 

    //**/ Transfer contents of vecXX to vecX2 so that we don't overwrite 
    //**/ values while sorting
    for (ii = 0; ii < nPts*nInputs; ii++) vecX2[ii] = vecXX[ii];

    //**/ Use vecST as indices for sorting vecXX
    for (ii = 0; ii < nPts; ii++)
    {
      for (jj = 0; jj != nInputs; ++jj)
        vecXX[ii*nInputs + jj] = vecX2[vecST[ii]*nInputs + jj];
    }

    // Record the best and worst points
    for (ii = 0; ii != nInputs; ++ii)
    {
      bestx[ii] = vecXX[ii];
      worstx[ii] = vecXX[(nPts-1)*nInputs+ii];
    }

    bestf = YT[0];
    worstf = YT[nPts -1];

    //**/ Compute the normalized geometric range of the parameters
    geoRange = georange(nInputs, nPts, vecXX.getDVector(),  
                        vecRanges.getDVector());

    //**/Display status of algorithm to let us know our progress
    if (psConfig_.InteractiveIsOn() && printLevel > 1)
    {
      printf("SCE: Evolution Loop = %d\n", nLoop);
      printf("     Current number of function evaluations = %d\n",iCall);
      printf("     Current best function value = %e\n", bestf);
      printf("     X at current best function value = ");
      for (ii = 0; ii != nInputs; ++ii) 
      {  
        if (VecPsSCEInpTypes_[ii] == 2) printf("%d ", (int) bestx[ii]);
        else                            printf("%e ", bestx[ii]);
      }
      printf("\n");
    }
    //**/printf(" Current worst function value = %e\n", worstf);
    //**/printf(" X at current worst function value = \n");
    //**/for (ii = 0; ii != nInputs; ++ii) printf("%e ", worstx[ii]);
    //**/printf("\n");

    //**/ Check for convergency
    if (psConfig_.InteractiveIsOn() && printLevel > 1 && 
        odata->numFuncEvals_-lastNumFuncEvals >= maxfun)
    {
      printf("*** Optimization search terminated because the limit on\n");
      printf("maximum number of trials %d has been exceeded.\n", maxfun);
    }
    if (psConfig_.InteractiveIsOn() && printLevel > 2)
      printf("Convergence check 1: %e <? %e\n",geoRange,
             paramSpaceConvergence);
    if (psConfig_.InteractiveIsOn() && printLevel > 1 && 
        geoRange < paramSpaceConvergence)
    {
      printf("The population has converged to a small parameter space.\n");
      break;
    }
    if (psSCEHistCnt_ == psSCEnHist_)
    {
      sum = 0.0;
      for (ii = 0; ii < psSCEnHist_; ++ii) sum += PABS(psSCEHistory_[ii]);
      sum /= (double) psSCEnHist_;
      for (ii = psSCEnHist_-3; ii < psSCEnHist_; ii++)
      {
        dtemp = PABS(psSCEHistory_[ii] - psSCEHistory_[ii-1]);
        if (sum != 0) dtemp /= sum;
          if (dtemp > odata->tolerance_) break;
      }
      if (ii == psSCEnHist_) 
      {
	if (psConfig_.InteractiveIsOn() && printLevel > 2)
          printf("Convergence check 2: converged\n");
        break;
      }
    }
    if ((psSCEHistIndex_ >= 0) && 
        (psSCEHistCnt_ != psSCEnHist_) &&
        (odata->numFuncEvals_-lastNumFuncEvals-psSCEHistIndex_ > maxfun/2))
    {
      printf("*** Optimization search terminated due to stagnation\n");
      break;
    }
    if ((fp=fopen("psuade_stop","r")) != NULL)
    {
      fclose(fp);
      printf("*** psuade_stop file found: optimization terminated \n");
      break;
    }
    if ((fp=fopen("psuade_print","r")) != NULL)
    {
      fclose(fp);
      if (printLevel < 1)
      {
        printLevel = 1;
        printf("*** psuade_print file found: print level set to 1\n");
      }
    }
    else printLevel = 0;
    
    vecCriteria[nLoop-1] = bestf;
    if (odata->numFuncEvals_-lastNumFuncEvals < maxfun && 
        nLoop >= maxEvoLoop)
    {
      if (bestf != vecCriteria[nLoop - maxEvoLoop])
      {
        criter_change = fabs(vecCriteria[nLoop-1]-
                             vecCriteria[nLoop-maxEvoLoop])*100;
        sum = 0.0; 
	for (ii = nLoop-maxEvoLoop; ii != nLoop-1; ++ii)
	  sum += fabs(vecCriteria[ii]);
	Mean = sum/maxEvoLoop;
        if (Mean < 1) Mean = 1;
	criter_change = criter_change/Mean;
	if (psConfig_.InteractiveIsOn() && printLevel > 2)
          printf("Convergence check 2: %e <? %e\n",criter_change,
                 percentChange);
	if (psConfig_.InteractiveIsOn() && printLevel > 1 && 
            criter_change < percentChange)
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

  if (psConfig_.InteractiveIsOn() && printLevel > 1)
  {
    for (ii = 0; ii < 60; ii++) printf("*");
    printf("\n");
    printf("Search was stopped at trial number %d\n", iCall);
    printf("Normalized geometric range = %e\n", geoRange);
    printf("Calculated global optimum = %e\n", bestf);
  }
  if (psConfig_.InteractiveIsOn())
  {
    printf("Global optimum occurs at: \n");
    printf(" X at current best function value = \n");
    for (ii = 0; ii != nInputs; ++ii) 
    {
      if (VecPsSCEInpTypes_[ii] == 2)
           printf("%d ", (int) bestx[ii]);
      else printf("%e ", bestx[ii]);
    }
    printf("\n");
  }

  psSCEnInputs_ = nInputs;
  if (psConfig_.InteractiveIsOn() && printLevel >= 0)
  {
    printf("SCEOptimizer: number of function evaluations = %d\n",
           odata->numFuncEvals_-lastNumFuncEvals);
  }

  //**/ ------ reset things ------
  if ((odata->setOptDriver_ & 2) && psSCECurrDriver_ >= 0)
  {
    odata->funcIO_->setDriver(psSCECurrDriver_);
  }
}

// ***********************************************************************
// georange function: calculates geometric range of each parameter
// -----------------------------------------------------------------------
double SCEOptimizer::georange(int nInputs,int nPts,double* x,double* ranges)
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
//**/ sBest = the best point of the simplex
//**/ sWorst = the worst point of the simplex
//**/ secWorst = the second worst point of the simplex
//**/ sfBest = function value of the best point
//**/ sfWorst = function value of the worst point
//**/ sNew = new generated simplex point
//**/ sfNew = function value of new point
//**/ centroid = centroid of simplex excluding worst point
// ----------------------------------------------------------------------
void SCEOptimizer::newPoint(double *sfNew, int* iCall, int nInputs,
                            int nMemPerSimplex, int maxfun, double* s, 
                            double* sf, double* sNew, oData *odata)
{
  int    ii, jj, outOfBound = 0;
  double alpha = 1.0, beta = 0.5, sum, sfNew1;
  double sBest[nInputs], sWorst[nInputs], centroid[nInputs];

  //**/ Assign the best and worst points
  for (ii = 0; ii < nInputs; ii++)
  {
    sBest[ii]  = s[ii];
    sWorst[ii] = s[nInputs*(nMemPerSimplex-1)+ii];
  }
  //**/double sfBest = sf[0];
  double sfWorst = sf[nMemPerSimplex-1];

  //**/ Compute the centroid of the simplex excluding the worst point
  for (ii = 0; ii < nInputs; ii++)
  { 
    sum = 0.0;
    for (jj = ii; jj < ((nMemPerSimplex-1)*nInputs); jj += nInputs)
      sum += s[jj];
    centroid[ii]= sum/(nMemPerSimplex-1);
  }

  //**/ Attempt a reflection point
  for (ii = 0; ii != nInputs; ++ii)
    sNew[ii] = centroid[ii] + alpha *(centroid[ii] - sWorst[ii]);

  //**/ Check if it is outside the bounds
  double s1[nInputs]; // double* s1 = new double[nInputs];
  double s2[nInputs]; // double* s2 = new double[nInputs]; 
  for (ii = 0; ii < nInputs; ii++)
  {
    s1[ii] = sNew[ii] - odata->lowerBounds_[ii];
    s2[ii] = odata->upperBounds_[ii] - sNew[ii];
  }

  for (ii = 0; ii < nInputs; ii++)
  {
    if ((s1[ii] < 0) || (s2[ii] < 0))
    {
      outOfBound = 1;
      break;
    }
  }
  if (outOfBound == 1)
  {
    for (ii = 0; ii != nInputs ; ++ii)
      sNew[ii] = odata->lowerBounds_[ii] + PSUADE_drand() *
                 (odata->upperBounds_[ii] - odata->lowerBounds_[ii]);
  }

  //**/ massage integer variables before evaluation
  if (VecPsSCEInpTypes_.length() > 0)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      if (VecPsSCEInpTypes_[ii] == 2)
      {
        jj = (int) (sNew[ii] + 0.5); 
        sNew[ii] = (double) jj;
      }
    }
  }
  SCEevalfunc_(&nInputs, sNew, &sfNew1);
  *iCall += 1;

  //**/ Reflection failed; now attempt a contraction step
  if (sfNew1 > sfWorst)
  {
    for (ii = 0; ii != nInputs; ++ii)
      sNew[ii] = sWorst[ii] + beta *(centroid[ii] - sWorst[ii]);
    //**/ massage integer variables
    if (VecPsSCEInpTypes_.length() > 0)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        if (VecPsSCEInpTypes_[ii] == 2)
        {
          jj = (int) (sNew[ii] + 0.5); 
          sNew[ii] = (double) jj;
        }
      }
    }
    SCEevalfunc_(&nInputs, sNew, &sfNew1);
    *iCall += 1;

    //**/ Both reflection and contraction have failed, try a random point
    if (sfNew1 > sfWorst)
    {
      for (ii = 0; ii != nInputs ; ++ii)
        sNew[ii] = odata->lowerBounds_[ii] + PSUADE_drand() *
                   (odata->upperBounds_[ii] - odata->lowerBounds_[ii]);
      if (VecPsSCEInpTypes_.length() > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          if (VecPsSCEInpTypes_[ii] == 2)
          {
            jj = (int) (sNew[ii] + 0.5); 
            sNew[ii] = (double) jj;
          }
        }
      }
      SCEevalfunc_(&nInputs, sNew, &sfNew1);
      *iCall += 1;
    }
  }
  *sfNew = sfNew1;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
SCEOptimizer& SCEOptimizer::operator=(const SCEOptimizer &)
{
  printf("SCEOptimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

