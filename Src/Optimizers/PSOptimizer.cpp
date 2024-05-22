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
// Functions for the class PSOptimizer
// AUTHOR : Charles Tong
// DATE   : 2023
// ************************************************************************
// Notes:
// Advantages: derivative-free, very few parameters, easily parallelizable
// Disadvantages: weal local search ability
// Enhancements (that can be added):
//   - swarm collapse
//   - accelerated local search
//   - adaptive inertia (W): to control velocity
//   - asynchronous optimization
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "PsuadeUtil.h"
#include "PSOptimizer.h"
#include "PsuadeUtil.h"
#include "Sampling.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PrintingTS.h"
#include "PDFManager.h"
#include <math.h>
#ifdef PSUADE_OMP
#include <omp.h>
#endif

// ------------------------------------------------------------------------
// reference to optimization object that is passed to evaluation function
// temporarily store current driver to switch to optimization driver
// ------------------------------------------------------------------------
void  *psPSOObj_=NULL;
int   psPSOCurrDriver_ = -1;
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" 
{
#endif
  void *PSOevalfunc_(int *nInps, double *XValues, double *YValue, 
                     int funcID)
  {
    //**/ ------ fetch data ------
    oData *odata  = (oData *) psPSOObj_;
    int nOutputs = odata->nOutputs_;
    int outputID = odata->outputID_;

    //**/ ------ run simulation ------
    double *localY = (double *) malloc(nOutputs * sizeof(double));
    odata->funcIO_->evaluate(funcID,*nInps,XValues,nOutputs,localY,0);
    (*YValue) = localY[outputID];
    free(localY);
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
PSOptimizer::PSOptimizer()
{
  if (psConfig_.InteractiveIsOn())
  { 
    printAsterisks(PL_INFO, 0);
#ifdef PSUADE_OMP
    printf("*   Asynchronous Particle Swarm Optimization (APSO)\n");
    printDashes(PL_INFO, 0);
    printf("*   APSO is slower than PSO because the best current ");
    printf("solution used in\n");
    printf("*   APSO lags that of PSO due to asynchrony.\n");
#else
    printf("*   Particle Swarm Optimization\n");
#endif
    printEquals(PL_INFO, 0);
    printf("* - To run this optimizer in batch mode, first make ");
    printf("sure opt_driver\n");
    printf("*   in your PSUADE input file has been set to point to ");
    printf("your objective\n");
    printf("*   objective function evaluator or response surface.\n");
    printf("* - Internal parameters\n");
    printf("*   + Optimization tolerance (set in your PSUADE input file)\n");
    printf("*   + Maximum number of iterations (set in PSUADE input file)\n");
    printf("*   + Number of particles (set in optimization expert mode)\n");
    printf("*   + Cognition of particles C1 (set in optimization ");
    printf("expert mode)\n");
    printf("*   + Social inference C2 (set in optimization expert mode)\n");
    printf("*   + Inertia W (set in optimization expert mode)\n");
    printf("* - Set optimization print_level to give more screen outputs\n");
    printAsterisks(PL_INFO, 0);
  }
  Swarm_ = NULL;
  nParticles_ = 50;  /* number of particles */
  C1_ = 1.5;         /* cognition of particles */
  C2_ = 1.5;         /* social influence of swarm */
  W_  = 0.8;         /* inertia */
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PSOptimizer::~PSOptimizer()
{
  if (Swarm_ != NULL) 
  {
    for (int ii = 0; ii < nParticles_; ii++) delete Swarm_[ii];
    delete [] Swarm_;
  }
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void PSOptimizer::optimize(oData *odata) 
{
  //**/---------------------------------------------------------
  //**/ some initialization
  //**/---------------------------------------------------------
  int printLevel = odata->outputLevel_;
  int nInputs  = odata->nInputs_;
  int nOutputs = odata->nOutputs_;
  if (nOutputs > 1)
  {
    printOutTS(PL_ERROR,"PSO ERROR: nOutputs = %d.\n",nOutputs);
    printOutTS(PL_ERROR,"       Only nOutputs=1 is allowed.\n");
    exit(1);
  }
  int ii;
  psVector vecLBs, vecUBs;
  vecLBs.setLength(nInputs);
  vecUBs.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
    vecLBs[ii] = odata->lowerBounds_[ii];
  for (ii = 0; ii < nInputs; ii++)
    vecUBs[ii] = odata->upperBounds_[ii];
  for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
  odata->optimalY_ = 1.0e50;

  //**/---------------------------------------------------------
  //**/ set optimization driver
  //**/---------------------------------------------------------
  if ((odata->setOptDriver_ & 1))
  {
    psPSOCurrDriver_ = odata->funcIO_->getDriver();
    odata->funcIO_->setDriver(1);
  }

  //**/---------------------------------------------------------
  //**/ ask for internal parameters
  //**/---------------------------------------------------------
  char pString[1000];
  if (psConfig_.OptExpertModeIsOn())
  {
    snprintf(pString,100,"Select c1 (default = %e) : ",C1_);
    C1_ = getDouble(pString);
    snprintf(pString,100,"Select c2 (default = %e) : ",C2_);
    C2_ = getDouble(pString);
    snprintf(pString,100,"Select W (default = %e) : ",W_);
    W_ = getDouble(pString);
    snprintf(pString,
         100,"Number of particles (10-100, default=50) : ");
    nParticles_ = getInt(10, 1000, pString);
  }

  //**/---------------------------------------------------------
  //**/ set up a swarm (particle initial positions/velocities)
  //**/ initial position is random inside the range
  //**/ initial velocity is random draw from a standard normal
  //**/---------------------------------------------------------
  //**/--------------------------------------
  //**/ So first draw nParticles velocities 
  //**/--------------------------------------
  int    iOne=1;
  double dmean=0, dstd=1, ddata;
  psMatrix corMat;
  corMat.setDim(iOne,iOne);
  corMat.setEntry(0,0,dstd);
  PDFManager *pdfman = new PDFManager();
  ii = 1; /* 1 = normal distribution */
  pdfman->initialize(iOne,&ii,&dmean,&dstd,corMat,NULL,NULL);
  psVector vLB, vUB, vecInitVel;
  vLB.setLength(iOne);
  vUB.setLength(iOne);
  vLB[0] = -4; /* cover +/- 4 standard deviations */
  vUB[0] =  4;
  vecInitVel.setLength(nParticles_*nInputs);
  pdfman->genSample(nParticles_*nInputs,vecInitVel,vLB,vUB);
  delete pdfman;

  //**/--------------------------------------
  //**/ clean up
  //**/--------------------------------------
  int nn;
  if (Swarm_ != NULL) 
  {
    for (nn = 0; nn < nParticles_; nn++) delete Swarm_[nn];
    delete [] Swarm_;
  }

  //**/--------------------------------------
  //**/ create swarm 
  //**/--------------------------------------
  int jj;
  Swarm_ = new PSOParticle*[nParticles_];
  for (nn = 0; nn < nParticles_; nn++)
  {
    Swarm_[nn] = new PSOParticle();
    Swarm_[nn]->VecPos_.setLength(nInputs);
    Swarm_[nn]->VecVel_.setLength(nInputs);
    Swarm_[nn]->VecBestPos_.setLength(nInputs);
    //**/ set initial position and velocity
    for (jj = 0; jj < nInputs; jj++)
    {
      ddata = vecUBs[jj] - vecLBs[jj];
      Swarm_[nn]->VecPos_[jj] = PSUADE_drand() * ddata + vecLBs[jj];
#if 1
      //**/ this option seems to work better
      Swarm_[nn]->VecVel_[jj] = PSUADE_drand() * ddata + vecLBs[jj];
#else
      ddata = vecInitVel[nn*nInputs+jj]; /* from standard normal */
      ddata *= 0.5 * (vecUBs[jj] - vecLBs[jj]);
      Swarm_[nn]->VecVel_[jj] = ddata;   /* 0.1 * scaled normal */
#endif
    }
    Swarm_[nn]->bestFitness_ = 1e50;
  }

  //**/---------------------------------------------------------
  //**/ get ready for optimization
  //**/---------------------------------------------------------
  psVector vecGBestPos;  /* store global best position */
  vecGBestPos.setLength(nInputs);
  int    maxIter = odata->maxFEval_;
  double bestFitness = 1e50; /* global best fitness */
  double lastFitness = 1e35, diffWithMin=1e35;
  int    epoch=0, nStagnates=0;
  double *XX, curPos, curVel, curBestX, rand1, rand2;
  double tol = odata->tolerance_;
  odata->numFuncEvals_ = 0;
  psPSOObj_ = odata;
  double expectedYmin = odata->targetY_;
  //**/ create an array of random numbers
  psVector vecRandom;
  vecRandom.setLength((maxIter+nParticles_)*nInputs);
  for (ii = 0; ii < (maxIter+nParticles_)*2; ii++)
    vecRandom[ii] = PSUADE_drand();

  //**/---------------------------------------------------------
  //**/ looping (3 conditions for termination: stagnation for 
  //**/ too long, sufficiently close to a given fmin, and 
  //**/ maximum number of function evaluations has been reached)
  //**/---------------------------------------------------------
  while (nStagnates < 20*nParticles_ && 
         PABS(diffWithMin) > tol && epoch*nParticles_ < maxIter)
  {
#pragma omp parallel shared(epoch,vecLBs,vecUBs,nInputs,\
    vecGBestPos,bestFitness,nStagnates) \
    private(nn,rand1,rand2,curPos,curVel,curBestX,ddata,ii,XX)
{
#ifdef PSUADE_OMP
#endif
#pragma omp for
    for (nn = 0; nn < nParticles_; nn++)
    {
#ifdef PSUADE_OMP
      if (epoch == 0)
        printf("  Processing particle %d at thread %d\n",nn+1,
               omp_get_thread_num());
#endif
      //**/ generate 2 random numbers
      rand1 = vecRandom[2*(epoch*nParticles_+nn)];
      rand2 = vecRandom[2*(epoch*nParticles_+nn)+1];

      //**/ compute new velocities of particle ii (not the first
      //**/ iteration
      if (epoch > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          curPos = Swarm_[nn]->VecPos_[ii];
          curVel = Swarm_[nn]->VecVel_[ii];
          curBestX = Swarm_[nn]->VecBestPos_[ii];
          ddata = ((W_*curVel) + (C1_*rand1*(curBestX-curPos)) + 
                   (C2_ * rand2 * (vecGBestPos[ii] - curPos)));
          if (ddata < vecLBs[ii]) ddata = vecLBs[ii];
          if (ddata > vecUBs[ii]) ddata = vecUBs[ii];
          Swarm_[nn]->VecVel_[ii] = ddata;
        }
        //**/ compute new position of particle ii
        for (ii = 0; ii < nInputs; ii++)
        { 
          curPos = Swarm_[nn]->VecPos_[ii];
          curVel = Swarm_[nn]->VecVel_[ii];
          ddata = curPos + curVel;
          if (ddata < vecLBs[ii]) ddata = vecLBs[ii];
          if (ddata > vecUBs[ii]) ddata = vecUBs[ii];
          Swarm_[nn]->VecPos_[ii] = ddata;
        }
      }
      //**/ run simulation for current particle
      XX = Swarm_[nn]->VecPos_.getDVector();
      PSOevalfunc_(&nInputs, XX, &ddata, epoch*nParticles_+nn);
#ifndef PSUADE_OMP
      if (epoch == 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("Particle %3d: Initial input %d = %12.5e (V=%12.5e)\n",
                 nn+1,ii+1,XX[ii],Swarm_[nn]->VecVel_[ii]);
        printf("              Initial output = %12.5e\n",ddata);
      }
#endif

      //**/ update errors 
      if (ddata < Swarm_[nn]->bestFitness_)
      {
        Swarm_[nn]->bestFitness_ = ddata;
        Swarm_[nn]->VecBestPos_.load(nInputs, XX);
      }
#pragma omp critical
{
      //printf("Thread %d entering critical\n",omp_get_thread_num());
      if (ddata < bestFitness)
      {
        bestFitness = ddata;
        for (ii = 0; ii < nInputs; ii++)
          vecGBestPos[ii] = Swarm_[nn]->VecBestPos_[ii];
        nStagnates = 0;
      }
      else nStagnates++;
      odata->numFuncEvals_++;
      //printf("Thread %d exiting critical\n",omp_get_thread_num());
} /* OMP critical */
    }
} /* OMP parallel */
    epoch++;
    if (psConfig_.InteractiveIsOn()) 
      printf("Epoch %d current best values : \n",epoch);
    if (psConfig_.InteractiveIsOn() && printLevel > 2) 
    {
      for (ii = 0; ii < nInputs; ii++)
        printf("   Input %d = %12.5e\n",ii+1, vecGBestPos[ii]);
    }
    if (psConfig_.InteractiveIsOn()) 
      printf("   Best function value = %12.5e\n",bestFitness);
    diffWithMin = PABS(bestFitness - expectedYmin);
  }
  if (nStagnates > 20*nParticles_)
    printf("INFO: Optimization terminates due to stagnation in %d runs.\n",
           nStagnates);
  else if (epoch*nParticles_ >= maxIter)
  {
    printf("INFO: Optimization terminates - max function ");
    printf("evaluation (%d) reached.\n",maxIter);
  }
  else if (diffWithMin < tol)
  {
    printf("INFO: Optimization terminates - best function ");
    printf("value within tolerance\n");
    printf("      (%11.4e) of the given expected fmin (%12.5e)\n",tol,
           expectedYmin);
  }
  if (psConfig_.InteractiveIsOn() && printLevel > 2) 
  {
    printAsterisks(PL_INFO, 0);
    printf("Final swarm information : \n");
    for (nn = 0; nn < nParticles_; nn++)
    {
      ii = 0;
      printf("  Particle %3d: input %d = %12.5e (V=%12.5e)\n",nn+1,
             ii+1,Swarm_[nn]->VecBestPos_[ii],
             Swarm_[nn]->VecVel_[ii]);
      for (ii = 1; ii < nInputs; ii++)
        printf("                Input %d = %12.5e (V=%12.5e)\n",
               ii+1,Swarm_[nn]->VecBestPos_[ii],
               Swarm_[nn]->VecVel_[ii]);
      printf("                Value = %12.5e\n",Swarm_[nn]->bestFitness_);
    }
    printAsterisks(PL_INFO, 0);
  }
  FILE *fp = fopen("pso_particles_pos", "w");
  if (fp != NULL)
  {
    fprintf(fp,"# Best position and fitness of %d particles\n",nParticles_);
    for (nn = 0; nn < nParticles_; nn++)
    {
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fp,"%24.16e ",Swarm_[nn]->VecBestPos_[ii]);
      fprintf(fp,"%24.16e\n",Swarm_[nn]->bestFitness_);
    }
    fclose(fp);
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printf("NOTE: Best positions and fitnesses of %d ",nParticles_);
    printf("particles have been stored in\n");
    printf("      a file called pso_particles_pos.\n");
    printf("      Since PSO has been known to be slow in local ");
    printf("convergence, these\n");
    printf("      information may be used to initial guesses ");
    printf("for subsquent multi-\n");
    printf("      start optimization using fast local convergence ");
    printf("optimizers.\n");
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }
  odata->optimalY_ = bestFitness;
  for (ii = 0; ii < nInputs; ii++) 
    odata->optimalX_[ii] = vecGBestPos[ii];

  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"* Particle Swarm Optimizer\n");
    printf("Current optimum is \n");
    for (ii = 0; ii < nInputs; ii++) 
      printf("    Input %d = %12.5e\n", ii+1, vecGBestPos[ii]);
    printf("    Optimal function value = %12.5e\n", bestFitness);
    printf("Total number of function evaluations = %d\n", 
           odata->numFuncEvals_);
  }

  //**/ ------ reset things ------
  if ((odata->setOptDriver_ & 2) && psPSOCurrDriver_ >= 0)
  {
    odata->funcIO_->setDriver(psPSOCurrDriver_);
  }
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
PSOptimizer& PSOptimizer::operator=(const PSOptimizer &)
{
  printf("PSOptimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

