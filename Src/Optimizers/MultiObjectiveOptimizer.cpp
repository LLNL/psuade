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
// Functions for the class BobyqaOptimizer
// AUTHOR : CHARLES TONG
// DATE   : 2010
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include "MultiObjectiveOptimizer.h"
#include "Sampling.h"
#include "PsuadeData.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "PrintingTS.h"

extern "C" void bobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);

void *psMOOObj_=NULL;
int  MO_NumVars_=0;
char MO_PythonFile_[1001];
psVector VecMOVars_;
psVector VecMOOuts_;

#define PABS(x)  ((x) > 0 ? x : -(x))
// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" 
{
#endif
  void *moobobyqaevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    ii, funcID, nInputs, nOutputs;
    double *localY, ddata, dsum;
    char   fName[500], runLine[500];
    oData  *odata;
    FILE   *fp = NULL;

    //**/ ------ get history ------
    nInputs = (*nInps);
    odata    = (oData *) psMOOObj_;
    nOutputs = odata->nOutputs_;
    localY   = (double *) malloc(nOutputs * sizeof(double));

    //**/ ------ run simulation ------
    funcID = odata->numFuncEvals_;
    odata->funcIO_->evaluate(funcID,nInputs,XValues,nOutputs,localY,0);
    funcID = odata->numFuncEvals_++;
    if (strcmp(MO_PythonFile_, "NULL"))
    {
      snprintf(fName,100,"MOO_params.in.%d", funcID);
      fp = fopen(fName, "w");
      if (fp == NULL)
      {
        printf("MultiObjectOptimizer ERROR: cannot open file %s\n",fName);
        exit(1);
      }
      fprintf(fp, "%d\n", nOutputs+MO_NumVars_);
      for (ii = 0; ii < nInputs; ii++) fprintf(fp,"%e\n", VecMOVars_[ii]);
      for (ii = 0; ii < nOutputs; ii++) fprintf(fp,"%e\n", localY[ii]);
      fprintf(fp, "# line 1: number of design variables + number of outputs\n");
      fprintf(fp, "# line 2-: design variables\n");
      fprintf(fp, "# line --: output variables\n");
      fclose(fp);
      snprintf(runLine,100,"%s MOO_params.in.%d MOO_params.out.%d",
               MO_PythonFile_, funcID, funcID);
      system(runLine);
      snprintf(fName,100,"MOO_params.out.%d", funcID);
      fp = fopen(fName, "r");
      if (fp == NULL)
      {
        printf("MultiObjectOptimizer ERROR: cannot open file %s\n",fName);
        exit(1);
      }
      fscanf(fp, "%lg", &ddata);
    }
    else
    {
      if (nOutputs != MO_NumVars_+1)
      {
        printf("MultiObjectOptimizer ERROR: nOutputs != numVars+1\n");
        exit(1);
      }
      if (MO_NumVars_ == nOutputs-1)
      {
        ddata = 0.0;
        dsum  = 0.0;
        for (ii = 0; ii < MO_NumVars_; ii++)
        {
          ddata += VecMOVars_[ii] * localY[ii];
          dsum  += VecMOVars_[ii];
        }
        ddata += (1.0 - dsum) * localY[MO_NumVars_];
      }
      else
      {
        ddata = 0.0;
        for (ii = 0; ii < MO_NumVars_; ii++)
          ddata += VecMOVars_[ii] * localY[ii];
      }
    }
    (*YValue) = ddata;

    //**/ ------ store optimal information ------
    if ((*YValue) < odata->optimalY_)
    {
      odata->optimalY_ = (*YValue);
      for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = XValues[ii];
      for (ii = 0; ii < nOutputs; ii++) VecMOOuts_[ii] = localY[ii];

    }
    free(localY);
    if(fp != NULL) fclose(fp);
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
MultiObjectiveOptimizer::MultiObjectiveOptimizer()
{
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MultiObjectiveOptimizer::~MultiObjectiveOptimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void MultiObjectiveOptimizer::optimize(oData *odata)
{
  int    nInputs, printLevel=0, ii, maxfun, currDriver, nPts=0, nn;
  int    nOutputs, iOne=1, length, numVars, nSamples, pLevel, resolution;
  double rhobeg=1.0, rhoend=1.0e-4, dtemp, MO_OptY;
  char   filename[501], lineIn[501], pString[101];
  string   sfname, iline;
  size_t   compFlag;
  ifstream ifile, ifile2;
  Sampling *samPtr;
  psVector  vecXVals, vecWT, vecLBs, vecUBs, vecSamInps, vecSamOuts;
  psVector  vecOptX;
  psIVector vecSamStas;

  //**/ --------------------------------------------------------
  //**/ display banner (if printLevel == 999, means banner only)
  //**/ --------------------------------------------------------
  printLevel = odata->outputLevel_;
  printAsterisks(PL_INFO, 0);
  printf("Surrogate-based Multi-objective optimization (MOO): \n");
  printDashes(PL_INFO, 0);
  printf("This multi-objective function is based on optimizing ");
  printf("some objective\n");
  printf("function constructed from all sample outputs.  As ");
  printf("such, it is quite\n");
  printf("different from other MOO approaches that construct ");
  printf("Pareto fronts\n");
  printf("(i.e. this MOO uses the objective\n\n");
  printf("  Z = sum_{i=1}^{k-1} w_i O_i + (1-sum_{i=1}^{k-1} w_i) O_k\n");
  printf("\nwhere\n");
  printf("   w_i's are the MOO variables (weights for different outputs)\n");
  printf("   O_i's are the model outputs\n");
  printf("   k     is the number of outputs\n\n");
  printf("Here, Z is cast as a linear combination of the outputs ");
  printf("O_i's and so\n");
  printf("the number of MOO variables = k-1, since the sum of ");
  printf("all weights is\n");
  printf("1.  MOO creates a factorial design for the MOO ");
  printf("variables. For each\n");
  printf("    factorial sample point, it finds the optimal point ");
  printf("with respect to\n");
  printf("    the model inputs. Finally, MOO finds the point in ");
  printf("the design that\n");
  printf("    gives the best overall Z.\n");
  printEquals(PL_INFO, 0);
  if (printLevel == 999) return;

  //**/ --------------------------------------------------------
  //**/ collect information from user 
  //**/ --------------------------------------------------------
  nInputs  = odata->nInputs_;
  nOutputs = odata->nOutputs_;
  printf("To complete MOO setup, provide a MOO configuration file in the\n");
  printf("following format:\n");
  printf("line 1: PSUADE_BEGIN\n");
  printf("line 2: number of MOO variables (normally = %d)\n",nOutputs-1);
  printf("line 3: 1  lbound ubound <lower/upper bounds of variable 1>\n");
  printf("line 4: 2  lbound ubound <lower/upper bounds of variable 2>\n");
  printf("        (These bounds should be in the range [0,1])\n");
  printf("....\n");
  printf("line n: Python file name for evaluating the objective function\n");
  printf("line n+1: PSUADE_END\n");
  printf("NOTE: If the objective function is just a linear combination\n");
  printf("      of the outputs, the Python function line should be just\n");
  printf("      a 'NULL', and the number of design variables should be\n");
  printf("      nOutputs-1 (since sum of weights=1).\n");
  printf("An Example: \n");
  printDashes(PL_INFO, 0);
  printf("PSUADE_BEGIN\n");
  printf("2\n");
  printf("1 0 1\n");
  printf("2 0 1\n");
  printf("objfcn.py (NULL if objective function is a linear combination)\n");
  printf("PSUADE_END\n");
  printDashes(PL_INFO, 0);
  printf("NOTE: the optimizer will evaluate the multi-objective function by\n");
  printf("      using the calling sequence:\n");
  printf("          <pythonFile> <paramFile> <objFile>\n");
  printf("where:\n");
  printf("  <paramFile> contains a sample point to evaluate the function\n");
  printf("      line 1: number of MOO variables + number of outputs\n");
  printf("      line 2: MOO variable w_1 value\n");
  printf("      line 3: MOO variable w_2 value\n");
  printf("              ....\n");
  printf("      line x: Output O_1 \n");
  printf("      line x: Output O_2 \n");
  printf("      line x: ...\n");
  printf("  <objFile> the Python file should write the overall objective\n");
  printf("            function value (Z) to this file.\n");
  printf("NOTE: MAKE SURE the <pythonFile> HAS EXECUTE PERMISSION.\n");
  printf("Enter the name of the configuration file: ");
  cin >> sfname;
  fgets(lineIn, 500, stdin);
  printEquals(PL_INFO, 0);
  length = sfname.size();

  //**/ --------------------------------------------------------
  //**/ read configuration file
  //**/ --------------------------------------------------------
  if (length < 500)
  {
    sfname.copy(filename, length, 0);
    filename[length] = '\0';
    ifile.open(filename);
    if (! ifile.is_open())
    {
      printf("MOO ERROR: cannot open configuration file = %s\n",filename);
      return;
    }
  }
  else
  {
    printf("MOO ERROR: configuration file name too long.\n");
    return;
  }
  getline (ifile, iline);
  compFlag = iline.compare("PSUADE_BEGIN");
  if (compFlag == 0)
  {
    ifile >> numVars;
    if (numVars <= 0)
    {
      printf("MOO configuration file ERROR: numVars <= 0\n");
      ifile.close();
      return;
    }
    vecLBs.setLength(numVars);
    vecUBs.setLength(numVars);
    for (ii = 0; ii < numVars; ii++)
    {
      ifile >> nn;
      if (nn != (ii+1))
      {
        printf("MOO config file ERROR: variable index mismatch (%d != %d).\n",
               nn, ii+1);
        ifile.close();
        return;
      }
      ifile >> vecLBs[ii];
      ifile >> vecUBs[ii];
      if (vecLBs[ii] < 0)
      {
        printf("MOO config file ERROR: lbound  has to be >= 0.\n");
        ifile.close();
        return;
      }
      if (vecUBs[ii] > 1)
      {
        printf("MOO config file ERROR: ubound  has to be <= 1.\n");
        ifile.close();
        return;
      }
      if (vecLBs[ii] > vecUBs[ii])
      {
        printf("MOO config file ERROR: lbound > ubound (input %d).\n",ii+1);
        ifile.close();
        return;
      }
    } 
    getline (ifile, iline);
    ifile >> MO_PythonFile_;
    if (strcmp(MO_PythonFile_, "NULL"))
    {
      ifile2.open(MO_PythonFile_);
      if (! ifile2.is_open())
      {
        printf("MOO config file ERROR: python file not found.\n");
        ifile.close();
        return;
      }
      ifile2.close();
    }
    else
    {
      strcpy(MO_PythonFile_, "NULL");
      if (numVars != nOutputs-1)
      {
        printf("MOO config file ERROR: if no python file, the number of\n");
        printf("   variables in the configuration file should be equal\n");
        printf("   to nOutputs-1, which is %d\n", nOutputs-1);
        return;
      }
    }
  }
  else
  {
    printf("MOO config file ERROR: PSUADE_BEGIN not found.\n");
    ifile.close();
    return;
  }
  getline (ifile, iline);
  getline (ifile, iline);
  compFlag = iline.compare("PSUADE_END");
  if (compFlag != 0)
  {
    printf("MOO config file ERROR: PSUADE_END not found.\n");
    ifile.close();
    return;
  }
  ifile.close();
  if (numVars > 4) 
  {
    printf("MOO currently cannot handle numVars>4.\n");
    return;
  }

  //**/ --------------------------------------------------------
  //**/ Create lattice (factorial design) for variables 
  //**/ --------------------------------------------------------
  printf("A full factorial design is to be generated to explore ");
  printf("the MOO variable\n");
  printf("space (of dimension %d).  To do this, users need to ",
         numVars);
  printf("choose the sample\n");
  printf("resolution (i.e. number of symbols in the factorial ");
  printf("design.  E.g. if\n");
  printf("the resolution is 10 for 3 MOO variables, the total ");
  printf("number of Bobyqa\n");
  printf("optimizations to be performed is 10x10x10=1000 (so ");
  printf("it is expensive).\n");
  if (numVars == 1)
  {
    snprintf(pString,100,
            "Enter the desired resolution (>2,<100,suggested: 11): ");
    resolution = getInt(3, 99, pString);
  }
  else if (numVars == 2)
  {
    snprintf(pString,100,
            "Enter the desired resolution (>2,<50,suggested: 11): ");
    resolution = getInt(3, 49, pString);
  }
  else
  {
    snprintf(pString,100,
            "Enter the desired resolution (>2,<20,suggested: 11): ");
    resolution = getInt(3, 19, pString);
  }
  samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FACT);
  samPtr->setPrintLevel(0);
  samPtr->setInputBounds(numVars, vecLBs.getDVector(), vecUBs.getDVector());
  samPtr->setOutputParams(iOne);
  nSamples = 1;
  for (ii = 0; ii < numVars; ii++) nSamples *= resolution;
  samPtr->setSamplingParams(nSamples, -1, 0);
  samPtr->initialize(0);
  nSamples = samPtr->getNumSamples();
  vecSamInps.setLength(nSamples * numVars);
  vecSamOuts.setLength(nSamples * (nInputs+nOutputs+1));
  vecSamStas.setLength(nSamples);
  samPtr->getSamples(nSamples,numVars,iOne,vecSamInps.getDVector(),
                     vecSamOuts.getDVector(),vecSamStas.getIVector());
  delete samPtr;

  //**/ --------------------------------------------------------
  //**/ prepare for optimization 
  //**/ --------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
  maxfun = odata->maxFEval_;
  vecXVals.setLength(nInputs+1);
  for (ii = 1; ii < nInputs; ii++) 
  {
    dtemp = odata->upperBounds_[ii] - odata->lowerBounds_[ii];
    if (dtemp < rhobeg) rhobeg = dtemp;
  }
  rhobeg = odata->upperBounds_[0] - odata->lowerBounds_[0];
  rhobeg *= 0.5;
  rhoend = rhobeg * odata->tolerance_;
  if (rhobeg < rhoend)
  {
    printf("MOO WARNING: tolerance too large.\n");
    printf("             tolerance reset to 1.0e-6.\n");
    rhoend = rhobeg * 1.0e-6;
  }
  currDriver = odata->funcIO_->getDriver();
  odata->funcIO_->setDriver(1);
  psMOOObj_= (void *) odata;
  if (printLevel > 0)
  {
    printf("MultiObj optimizer: max fevals = %d\n", odata->maxFEval_);
    printf("MultiObj optimizer: tolerance  = %e\n", odata->tolerance_);
  }
  nPts = (nInputs + 1) * (nInputs + 2) / 2;
  vecWT.setLength((nPts+5)*(nPts+nInputs)+3*nInputs*(nInputs+5)/2+1);
  MO_NumVars_ = numVars;
  VecMOVars_.setLength(MO_NumVars_);
  VecMOOuts_.setLength(nOutputs);
  odata->numFuncEvals_ = 0;
  //**/ use pLevel=9999 to tell bobyqa that it is from MOO 
  vecOptX.setLength(numVars);
  MO_OptY = PSUADE_UNDEFINED;

  //**/ --------------------------------------------------------
  //**/ call optimizer 
  //**/ --------------------------------------------------------
  MO_OptY = PSUADE_UNDEFINED;
  printEquals(PL_INFO, 0);
#if 1
  for (nn = 0; nn < nSamples; nn++)
  {
    printf("Running Optimization #%d out of #%d\n",nn+1,nSamples);
    for (ii = 0; ii < numVars; ii++)
      VecMOVars_[ii] = vecSamInps[nn*numVars+ii];
    for (ii = 0; ii < nInputs; ii++) vecXVals[ii] = odata->initialX_[ii];
    odata->optimalY_ = 1.0e50;
#ifdef HAVE_BOBYQA
    pLevel = 9999;
    bobyqa_(&nInputs, &nPts, vecXVals.getDVector(), odata->lowerBounds_,
            odata->upperBounds_, &rhobeg, &rhoend, &pLevel, &maxfun, 
            vecWT.getDVector());
#else
    printf("ERROR: Bobyqa optimizer not installed.\n");
    exit(1);
#endif
    vecSamOuts[nn*(nInputs+nOutputs+1)] = odata->optimalY_;
    for (ii = 0; ii < nInputs; ii++)
      vecSamOuts[nn*(nInputs+nOutputs+1)+ii+1] = odata->optimalX_[ii];
    for (ii = 0; ii < nOutputs; ii++)
      vecSamOuts[nn*(nInputs+nOutputs+1)+nInputs+ii+1] = VecMOOuts_[ii];
    if (printLevel > 2)
    {
      printf("Iteration %5d (%5d) : inputs =", nn+1,nSamples);
      for (ii = 0; ii < numVars; ii++)
        printf(" %e", vecSamInps[nn*numVars+ii]);
      printf(", min = %e\n",odata->optimalY_);
    }
    if (odata->optimalY_ < MO_OptY)
    {
      MO_OptY = odata->optimalY_;
      for (ii = 0; ii < numVars; ii++)
        vecOptX[ii] = vecSamInps[nn*numVars+ii];
    }
  }
#else
  //**/ I don't remember what I did here
  int kk, numPCEs, **PCEs, pOrder, flag, ind, isum, itmp, ind2;
  pOrder = (resolution - 1) * numVars;
  GenPermutations(numVars, pOrder, &numPCEs, &PCEs); 
  ind = 1;
  for (nn = 1; nn < 3*(resolution-1); nn++)
  {
    isum = nn;
    ind2 = ind;
    while (isum == nn && ind2 < numPCEs-1)
    {
      ind2++;
      isum = PCEs[ind2][0];
      for (ii = 1; ii < numVars; ii++) isum += PCEs[ind2][ii];
      if (isum != nn) break;
    }
    if (ind2 == (numPCEs-1)) ind2++;
    for (kk = ind; kk < (ind+ind2)/2; kk++)
    {
      for (ii = 0; ii < numVars; ii++)
      {
        itmp = PCEs[kk][ii];
        PCEs[kk][ii] = PCEs[ind2+ind-kk-1][ii];
        PCEs[ind2+ind-kk-1][ii] = itmp;
      }
    }
    ind = ind2;
  }
  for (nn = 0; nn < numPCEs; nn++)
  {
    flag = 1;
    for (ii = 0; ii < numVars; ii++)
      if (PCEs[nn][ii] >= resolution) flag = 0;
    if (flag == 1)
    {
      ind = PCEs[nn][numVars-1];
      for (ii = numVars-2; ii >= 0; ii--)
        ind = ind * resolution + PCEs[nn][ii]; 
      for (ii = 0; ii < numVars; ii++)
        VecMOVars_[ii] = vecSamInps[ind*numVars+ii];
      if (nn == 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          vecXVals[ii] = odata->initialX_[ii];
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          vecXVals[ii] = odata->optimalX_[ii];
      }
      odata->optimalY_ = 1.0e50;
#ifdef HAVE_BOBYQA
      pLevel = 9999;
      bobyqa_(&nInputs, &nPts, vecXVals.getDVector(), odata->lowerBounds_,
              odata->upperBounds_, &rhobeg, &rhoend, &pLevel, 
              &maxfun, vecWT.getDVector());
#else
      printf("ERROR: Bobyqa optimizer not installed.\n");
      exit(1);
#endif
      vecSamOuts[ind*(nInputs+nOutputs+1)] = odata->optimalY_;
      for (ii = 0; ii < nInputs; ii++)
        vecSamOuts[ind*(nInputs+nOutputs+1)+ii+1] = odata->optimalX_[ii];
      for (ii = 0; ii < nOutputs; ii++)
         vecSamOuts[ind*(nInputs+nOutputs+1)+nInputs+ii+1] = VecMOOuts_[ii];
      if (odata->optimalY_ < MO_OptY)
      {
        MO_OptY = odata->optimalY_;
        for (ii = 0; ii < numVars; ii++)
          vecOptX[ii] = vecSamInps[ind*numVars+ii];
      }
    }
  }
  for (ii = 0; ii < numPCEs; ii++) delete [] PCEs[ii];
  delete [] PCEs;
#endif
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,
       "MOO Summary (X is the set of MOO variables) :\n");
  printAsterisks(PL_INFO, 0);
  dtemp = 0.0;
  for (ii = 0; ii < numVars; ii++)
  {
    printf("MOO OptimalX %2d = %e\n", ii+1, vecOptX[ii]);
    dtemp += vecOptX[ii];
  }
  if (numVars == (nOutputs-1))
    printf("MOO OptimalX %2d = %e\n", numVars+1, 1.0-dtemp);
  printf("MOO OptimalY    = %e\n", MO_OptY);
  printf("MOO nFuncEval   = %d\n", odata->numFuncEvals_);
  printAsterisks(PL_INFO, 0);

  //**/ ------ store results ------
  PsuadeData *ioPtr;
  char       **iNames, **oNames;
  iNames = new char*[numVars];
  for (ii = 0; ii < numVars; ii++)
  {
    iNames[ii] = new char[100];
    snprintf(iNames[ii],100,"X%d", ii+1);
  }
  oNames = new char*[nInputs+nOutputs+1];
  for (ii = 0; ii < nInputs+nOutputs+1; ii++)
  {
    oNames[ii] = new char[100];
    if (ii == 0)            snprintf(oNames[ii],100,"Y");
    else if (ii <= nInputs) snprintf(oNames[ii],100,"X%d", ii);
    else                    snprintf(oNames[ii],100,"F%d", ii);
  }
  for (ii = 0; ii < nSamples; ii++) vecSamStas[ii] = 1;
  ioPtr = new PsuadeData();
  ioPtr->updateInputSection(nSamples, numVars, NULL, vecLBs.getDVector(),
                 vecUBs.getDVector(),vecSamInps.getDVector(), iNames,
                 NULL,NULL,NULL,NULL);
  ioPtr->updateOutputSection(nSamples,nInputs+nOutputs+1,
               vecSamOuts.getDVector(),vecSamStas.getIVector(),oNames);
  ioPtr->updateMethodSection(PSUADE_SAMP_MC, nSamples, 1, -1, -1);
  printEquals(PL_INFO, 0);
  ioPtr->writePsuadeFile("psuade_moo_sample",0);
  printf("A MOO sample has been created in file psuade_moo_sample, where\n");
  printf("Inputs  are: the MOO variables (= %d) in this sample.\n",numVars);
  printf("Outputs are: \n");
  printf("  Output 1: optimal Z for the MOO inputs (size=%d)\n",numVars);
  printf("  Output 2: X1 values for optimal Z given MOO inputs (weights)\n");
  printf("  Output 3: X2 values for optimal Z given MOO inputs (weights)\n");
  printf("  ...\n");
  printf("  Output ?: F1 values for optimal Z given MOO inputs (weights)\n");
  printf("  Output ?: F2 values for optimal Z given MOO inputs (weights)\n");
  printf("  ...\n");
  printf("where F1 is the first objective of the multiobjective function.\n");
  printf("\n");
  printf("Since the MOO variables are explored by a factorial sample, the results\n");
  printf("can be visualized using Matlab functions.  Alternatively, the results\n");
  printf("can be used to construct a response surface (e.g. GP) so that the MOO\n");
  printf("space can be explored further.\n"); 

  //**/ ------ set return values and clean up ------
  for (ii = 0; ii < numVars; ii++) delete [] iNames[ii];
  delete [] iNames;
  for (ii = 0; ii < nInputs+nOutputs+1; ii++) delete [] oNames[ii];
  delete [] oNames;
  odata->funcIO_->setDriver(currDriver);
  delete ioPtr;
  MO_NumVars_ = 0;
}

// *************************************************************************
// generate all combinations of a multivariate PCE expansion
// This code is a direct translation from Burkardt's matlab code)
// -------------------------------------------------------------------------
int MultiObjectiveOptimizer::GenPermutations(int nInputs, int pOrder,
                                            int *nPerms, int ***pPerms)
{
  int  ii, kk, numPerms, orderTmp, rvTmp, **pcePerms;

  //**/ search for maximum order
  numPerms = 1;
  for (ii = nInputs+pOrder; ii > pOrder; ii--) numPerms *= ii;
  for (ii = 2; ii <= nInputs; ii++) numPerms /= ii;

  //**/ construct the permutations
  pcePerms = new int*[numPerms];
  for (ii = 0; ii < numPerms; ii++) pcePerms[ii] = new int[nInputs];

  numPerms = 0;
  for (kk = 0; kk <= pOrder; kk++)
  {
    orderTmp = kk;
    rvTmp = 0;
    pcePerms[numPerms][0] = orderTmp;
    for (ii = 1; ii < nInputs; ii++) pcePerms[numPerms][ii] = 0;
    while (pcePerms[numPerms][nInputs-1] != kk)
    {
      numPerms++;
      for (ii = 0; ii < nInputs; ii++)
        pcePerms[numPerms][ii] = pcePerms[numPerms-1][ii];
      if (orderTmp > 1) rvTmp = 1;
      else              rvTmp++;
      pcePerms[numPerms][rvTmp-1] = 0;
      orderTmp = pcePerms[numPerms-1][rvTmp-1];
      pcePerms[numPerms][0] = orderTmp - 1;
      pcePerms[numPerms][rvTmp] = pcePerms[numPerms-1][rvTmp] + 1;
    }
    numPerms++;
  }
  (*nPerms) = numPerms;
  (*pPerms) = pcePerms;
  return 0;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
MultiObjectiveOptimizer& MultiObjectiveOptimizer::operator=(const 
                                                 MultiObjectiveOptimizer &)
{
  printf("MultiObjectiveOptimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

