// ************************************************************************
// Copyright (c) 2015   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// DASSI is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// DASSI is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// KPCA functions (for Xiao Chen's LDRD project)
// DATE   : 2015
// ************************************************************************
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "normal.h"

#include "MetropolisHastingsMCMC.h"

using namespace std;

// ********************************************************************
// constructor
// ********************************************************************
MetropolisHastingsMCMC::MetropolisHastingsMCMC()
{
  strcpy(PosteriorFile_, "none");
  strcpy(likelihoodFunction_, "none");
  MCMCNDim_ = -1;
  breakPtFlag_ = 0;
}

// ********************************************************************
// set initial guess 
// ********************************************************************
int MetropolisHastingsMCMC::setInitialGuess(psVector vecIG)
{
  VecIG_ = vecIG;
  return 0;
}

// ********************************************************************
// set posterior file
// ********************************************************************
int MetropolisHastingsMCMC::setPosteriorFile(char *postfile)
{
  strcpy(PosteriorFile_, postfile);
  return 0;
}

// ************************************************************************
// set negative log likelihood operator
// ------------------------------------------------------------------------
int MetropolisHastingsMCMC::setLikelihoodFunction(char *func)
{  
  strcpy(likelihoodFunction_, func);
  FILE *fp = fopen(likelihoodFunction_, "r");
  if (fp == NULL)
  {  
    printf("setLikelihoodFunction ERROR: function (%s) not found\n",
           likelihoodFunction_);
    return -1;
  }
  fclose(fp);
  return 0;
}

// ********************************************************************
// Initialize MCMC (an even longer version)
// ********************************************************************
void MetropolisHastingsMCMC::initialize(int nInps, int randseed,
                                        int MCMC_nSamples)
{
  MCMCRandSeed_ = randseed;
  srand48((long) randseed);
  srand((unsigned int) randseed);
  burnInSize_ = 0;
  MCMCMaxSamples_ = MCMC_nSamples;
  if (MCMCNDim_ != -1 && MCMCNDim_ != nInps)
  {
    printf("MHMCMC initialize ERROR: nInputs != nInputs from DRModel\n"); 
    exit(1);
  }
  MCMCNDim_ = nInps;
  MatMcmcChain_.setDim(MCMCNDim_, MCMC_nSamples);
}

// ********************************************************************
// run MCMC
// ********************************************************************
void MetropolisHastingsMCMC::runMCMC()
{
  int    ii, jj, mcmc_iter, iaccept, isample;
  double c1,c2,c11,c12,alpha,uniform_v,ddata,scale=1000,propScale=0.1;
  char   cString[1000];
  FILE   *fp=NULL;
  psVector vecCandidate, vecCurrent;

  // ===========================================================
  // error checking 
  // ===========================================================
  if (!strcmp(likelihoodFunction_, "none"))
  {
    printf("MHMCMC run ERROR: costFunction has not been set\n");
    exit(1);
  } 
  if (VecIG_.length() != MCMCNDim_)
  {
    printf("MHMCMC run ERROR: initial guess size %d not correct\n",
           VecIG_.length());
    exit(1);
  } 

  // ===========================================================
  // check and reset stop function
  // ===========================================================
  strcpy(cString, "psuade_stop");
  fp = fopen(cString, "r");
  if (fp != NULL)
  {
    fclose(fp);
    remove(cString);
  }
  if (breakPtFlag_ == 1) printf("MHMCMC: break point on.\n");

  // ===========================================================
  // compute posterior of initial guess
  // ===========================================================
  vecCurrent = VecIG_;
  c12 = computePosterior(vecCurrent,scale);
  printf("====> MHMC iteration 0 (c12 = %e)\n", c12);
  likelihoodEvaluationCount_ = 0;

  // ===========================================================
  // run mcmc 
  // ===========================================================
  mcmc_iter = iaccept = isample = 0;
  while (isample < MCMCMaxSamples_) 
  {
    isample++;
    if (isample > burnInSize_) 
         printf("====> MHMCMC iteration : %d\n", mcmc_iter++);
    else printf("====> MHMCMC Burn-in iteration : %d\n", isample);

    // sample from the proposal density function 
    vecCandidate = genSampleFromPrior(vecCurrent, propScale);

    // compute posterior of current candidate
    c11 = computePosterior(vecCandidate,scale);
    c1  = (c11  - c12);
    printf("MHMCMC: c1 = %e (c11 = %e)\n", c1, c11);

    // determine whether to accept or reject current candidate
    c2  = log(1);
    alpha = min (0.0, c1+c2);
    uniform_v = drand48();
    if (log(uniform_v) < alpha) 
    {
      vecCurrent = vecCandidate;
      c12 = c11;

      // recording the mcmc sample after the burn in number 
      if (isample > burnInSize_) 
      {
	if (iaccept <= MCMCMaxSamples_) 
          MatMcmcChain_.loadCol(iaccept,vecCurrent.length(), 
                                vecCurrent.getDVector());
	iaccept = iaccept + 1;
      }
      printf("Accepted :) log(uniform_v) %12.4e < alpha %12.4e\n", 
             log(uniform_v), alpha);
      printf("Success percentage = %e\n", 100.0 * iaccept / isample);
    }
    else 
    {
      printf("Rejected :( log(uniform_v) %12.4e > alpha %12.4e\n",
             log(uniform_v), alpha);
      printf("Success percentage = %e\n",100.0*iaccept/isample);
    }
      
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      fclose(fp);
      printf("psuade_stop found ==> terminate\n");
      break;
    }
  } // end do while 

  // ===========================================================
  // store posterior sample 
  // ===========================================================
  fp = NULL;
  if (strcmp(PosteriorFile_,"none")) 
    fp = fopen(PosteriorFile_,"w");
  if (fp != NULL)
  {
    for (ii = 0; ii < iaccept; ii++) 
    {
      for (jj = 0; jj < MatMcmcChain_.nrows(); jj++) 
        fprintf(fp, "%16.8e ", MatMcmcChain_.getEntry(jj,ii));
      fprintf(fp, "\n");
    }
  }
  fclose(fp);

  printf("MHMCMC acceptance  rate  = %e\n", 
          double (iaccept) / double (mcmc_iter) *100);
  printf("MHMCMC completed\n"); 
}

// ********************************************************************
// random proposal distribution
// ********************************************************************
psVector MetropolisHastingsMCMC::genSampleFromPrior(psVector &vCurrent, 
                                                    double propScale) 
{
  psVector randn;
  randn.setLength(MCMCNDim_);
  for (int ii=0; ii < MCMCNDim_; ii++) 
    randn[ii] = r8_normal_ab(vCurrent[ii],propScale,MCMCRandSeed_); 
  return randn;
}

// ************************************************************************
// run simulation to compute likelihood
// ------------------------------------------------------------------------
double MetropolisHastingsMCMC::computeLikelihood(psVector &vecIn,
                                                 double scale) 
{
  int  ii;
  char sysInput[1000], sysOutput[1000], sysCmd[10000], winput[1000];
  FILE *fp;
  psVector vecOut;

  //**/ create simulator input file
  likelihoodEvaluationCount_++;
  snprintf(sysInput,100,"ps.inps.%d", likelihoodEvaluationCount_);
  fp = fopen(sysInput, "w");
  if (fp == NULL) 
  {
    printf("MHMCMC ERROR: cannot open simulator input file (%s).\n", 
           sysInput);
    exit(1);
  }
  for (int ii = 0; ii < vecIn.length(); ii++) 
    fprintf(fp, "%24.16e\n", vecIn[ii]);
  fclose(fp);

  //**/ run likelihood function
  snprintf(sysOutput,100,"ps.outputs.%d",likelihoodEvaluationCount_);
  snprintf(sysCmd,100,"%s %s %s",likelihoodFunction_,sysInput,sysOutput);
  system(sysCmd);
 
  //**/ read likelihood value
  fp = fopen(sysOutput, "r");
  if (fp == NULL) 
  {
    printf("MHMCMC ERROR: cannot open output file (%s).\n", 
           sysOutput);
    exit(1);
  }
  double likelihood;
  fscanf(fp, "%lg", &likelihood);
  fclose(fp);

  //**/ check breakpoint, if so, break
  if (breakPtFlag_ == 0)
  {
    fp = fopen("psuade_setbp", "r");
    if (fp != NULL)
    {
      breakPtFlag_ = 1;
      fclose(fp);
    }
  }
  else
  {
    fp = fopen("psuade_resetbp", "r");
    if (fp != NULL)
    {
      breakPtFlag_ = 0;
      fclose(fp);
    }
  }
  if (breakPtFlag_ == 1)
  {
    printf("Simulation input  file = ps.inps\n");
    printf("Simulation output file = ps.outputs\n");
    printf("Likelihood obtained    = %e\n", likelihood);
    printf("Type 'y' to continue : ");
    scanf("%s", winput);
  }

  //**/ compute log likelihood 
  // - 1/2 || f(\mu) - u_measure||_2^2 *scale
  double logLikelihood = 0.5 * log(likelihood) * scale;
  printf("MHMCMC: log likelihood = %e\n", logLikelihood);

  //**/ clean up
  unlink(sysInput);
  unlink(sysOutput);
  return logLikelihood; 
}

// ************************************************************************
// compute posterior
// ------------------------------------------------------------------------
double MetropolisHastingsMCMC::computePosterior(psVector &vecXi, 
                                                double scale)
{
  double prior, likelihood, posterior;
  //prior = prior_pdf(eta);
  prior = 0.0;
  likelihood = computeLikelihood(vecXi,scale);
  posterior  =  prior + likelihood;
  printf("loglikelihood = %e, prior = %e, RSS = %e\n",
         -likelihood, prior, -likelihood/scale);
  //posterior = exp( posterior ) ;
  return posterior;
}

// ************************************************************************
// set break point
// ------------------------------------------------------------------------
int MetropolisHastingsMCMC::setBreakPoint()
{
  breakPtFlag_ = 1;
  return 0;
}

