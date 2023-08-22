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
// Functions for the class TGP
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_TGP
extern "C"
{
#include "matrix.h"
#include "rand_draws.h"
#include "rhelp.h"
#include "predict.h"
}
#include "model.h"
#include "params.h"
#include "mstructs.h"
#endif

#include "TBGP.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class TGP
// ------------------------------------------------------------------------
TGP::TGP(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_TGP;
#ifdef HAVE_TGP
  tgp_ = NULL;
#endif
  vecZpMean_.setLength(nSamples_);
  vecZpQ_.setLength(nSamples_);
  vecZpS2_.setLength(nSamples_);
  vecZpQ1_.setLength(nSamples_);
  vecZpMedian_.setLength(nSamples_);
  vecZpQ2_.setLength(nSamples_);
  vecZZQ_.setLength(nSamples_);
  vecZZS2_.setLength(nSamples_);
  vecZZQ1_.setLength(nSamples_);
  vecZZMedian_.setLength(nSamples_);
  vecZZQ2_.setLength(nSamples_);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
TGP::~TGP()
{
#ifdef HAVE_TGP
  if (tgp_ != NULL) delete tgp_;
#endif
  vecZpMean_.clean();
  vecZpQ_.clean();
  vecZpS2_.clean();
  vecZpQ1_.clean();
  vecZpMedian_.clean();
  vecZpQ2_.clean();
  vecZZQ_.clean();
  vecZZQ1_.clean();
  vecZZMedian_.clean();
  vecZZQ2_.clean();
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int TGP::initialize(double *X, double *Y)
{
  int status=-999;
  genNDGridData(X, Y, &status, NULL, NULL);
  if (psConfig_.RSCodeGenIsOn())
    printf("TGP INFO: response surface stand-alone code not available.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int TGP::genNDGridData(double *X, double *Y, int *N, double **XOut, 
                       double **YOut)
{
#ifdef HAVE_TGP
  int  BTE[3], R=1, linBurn=1, PS_FALSE=0, PS_TRUE=1;
  int  totPts, ii, ss, count, bte0, bte1;
  void *tgp_state=NULL;
  char lineOut[101];
  psVector  vecDParms, vecHX, vecXT, vecYOut, vecGpcs, vecXL, vecTParms;
  psIVector vecStateIn;

  //**/ ------------------------------------------------------------
  //**/ interactive query
  //**/ ------------------------------------------------------------
  bte0 = 2000;
  bte1 = 7000;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("TGP: default burn-in sample size = 2000\n");
    printf("TGP: default MCMC    sample size = 7000\n");
    snprintf(lineOut,100,"TGP: enter burn-in sample size (e.g. 500-5000): ");
    bte0 = getInt(500, 5000, lineOut); 
    snprintf(lineOut,100,"TGP: enter MCMC sample size (e.g. %d-20000): ",
             2000+bte0);
    bte1 = getInt(2000+bte0, 20000, lineOut); 
  }

  //**/ ------------------------------------------------------------
  //**/ preparation
  //**/ ------------------------------------------------------------
  if (outputLevel_ >= 1) printf("TGP training begins....\n");
  if (tgp_ != NULL) delete tgp_;

  //**/ ------------------------------------------------------------
  //**/ create the RNG state 
  //**/ ------------------------------------------------------------
  vecStateIn.setLength(3);
  vecStateIn[0] = 782;
  vecStateIn[1] = 267;
  vecStateIn[2] = 218;
  unsigned int lstate = three2lstate(vecStateIn.getIVector());
  tgp_state = newRNGstate(lstate);

  //**/ ------------------------------------------------------------
  //**/ create parameter lists
  //**/ ------------------------------------------------------------
  BTE[0] = bte0;
  BTE[1] = bte1;
  BTE[2] = 2;
  vecDParms.setLength((nInputs_+1)*(nInputs_+1)+nInputs_+45);
  vecDParms[0] = 0.5;  // tree prior alpha
  vecDParms[1] = 2.0;  // tree prior beta
  vecDParms[2] = 10.0; // tree prior minpart
  if ((nInputs_ + 2) > 10) vecDParms[2] = 1.0 * (nInputs_ + 2.0);
  vecDParms[3] = 1.0;  // tree prior splitmin
  vecDParms[4] = 1.0 * nInputs_;  // tree prior basemax
  vecDParms[5] = 0.0;  // meanfn:  0 - linear, 1 : constant
  vecDParms[6] = 2.0;  // bflat in {b0, bmle, bflat, b0not,..}
  //**/ beta (nInputs+1 zeros)
  for (ii = 0; ii <= nInputs_; ii++) vecDParms[7+ii] = 0.0;
  count = nInputs_ + 8;
  vecDParms[count] = 1.0;  // Wi (1, (nInputs+1)^2 zeros)
  for (ii = 0; ii < (nInputs_+1)*(nInputs_+1)-1; ii++)
    vecDParms[count+1+ii] = 0.0;
  for (ii = 1; ii < nInputs_; ii++) 
    vecDParms[count+ii*(nInputs_+1)] = 1.0;
  count += (nInputs_ + 1) * (nInputs_ + 1);
  vecDParms[count++] = 1.0;   // s2tau2 = c(1,1)
  vecDParms[count++] = 1.0;
  vecDParms[count++] = 5.0;   // s2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // s2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 5.0;   // tau2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // tau2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 0.1;
  vecDParms[count++] = 1;     // "expsep"
  vecDParms[count++] = 0.1;   // gd (1)
  vecDParms[count++] = 0.5;   // gd (2)
  vecDParms[count++] = 1;     // nug.p (1)
  vecDParms[count++] = 1;     // nug.p (2)
  vecDParms[count++] = 1;     // nug.p (3)
  vecDParms[count++] = 1;     // nug.p (4)
  vecDParms[count++] = -1;    // nug.lam (1)
  vecDParms[count++] = -1;    // nug.lam (2)
  vecDParms[count++] = -1;    // nug.lam (3)
  vecDParms[count++] = -1;    // nug.lam (4)
  vecDParms[count++] = 10.0;  // gamma(1)
  vecDParms[count++] = 0.2;   // gamma(2)
  vecDParms[count++] = 0.7;   // gamma(3)
  vecDParms[count++] = 1.0;   // d.p(1)
  vecDParms[count++] = 20.0;  // d.p(2)
  vecDParms[count++] = 10.0;  // d.p(3)
  vecDParms[count++] = 10.0;  // d.p(4)
  vecDParms[count++] = -1;    // d.lam (1)
  vecDParms[count++] = -1;    // d.lam (2)
  vecDParms[count++] = -1;    // d.lam (3)
  vecDParms[count++] = -1;    // d.lam (4)
  vecDParms[count++] = 0;     // nu?
  vecTParms.setLength(7);
  vecTParms[0] = 1;
  vecTParms[1] = 0;
  vecTParms[2] = 0;
  vecTParms[3] = 1;
  vecTParms[4] = 1;
  vecTParms[5] = 0;
  vecTParms[6] = 1;

  //**/ ------------------------------------------------------------
  //**/ create function 
  //**/ ------------------------------------------------------------
  if ((*N) != -999 && XOut != NULL && YOut != NULL)
  {
    //**/ set up for generating regular grid data
    totPts = nPtsPerDim_;
    for (ii = 1; ii < nInputs_; ii++) totPts = totPts * nPtsPerDim_;
    vecHX.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) 
       vecHX[ii] = (VecUBs_[ii] - VecLBs_[ii]) /
                   (double) (nPtsPerDim_ - 1); 
    //**/ allocate storage for the data points
    vecXT.setLength(totPts*nInputs_);
    vecXL.setLength(nInputs_);
    //**/ generate the data points 
    for (ii = 0; ii < nInputs_; ii++) vecXL[ii] = VecLBs_[ii];

    for (ss = 0; ss < totPts; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++) 
        vecXT[ss*nInputs_+ii] = vecXL[ii];
      for (ii = 0; ii < nInputs_; ii++) 
      {
        vecXL[ii] += vecHX[ii];
        if (vecXL[ii] < VecUBs_[ii] || 
            PABS(vecXL[ii] - VecUBs_[ii]) < 1.0E-7) break;
        else vecXL[ii] = VecLBs_[ii];
      }
    }
    //**/ call TGP 
    tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                   (int) totPts, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                   (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_TRUE, 
                   (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                   X, Y, vecXT.getDVector(), vecDParms.getDVector(), 
                   vecTParms.getDVector(), (bool) PS_FALSE, (int) PS_FALSE, 
                   (double *) NULL, (double *) NULL);
  }
  else
  {
    //**/ use original points
    totPts = nSamples_;
    vecXT.setLength(totPts*nInputs_);
    for (ss = 0; ss < totPts*nInputs_; ss++) vecXT[ss] = X[ss];
    //**/ call TGP 
    tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                   (int) totPts, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                   (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_FALSE, 
                   (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                   X, Y, vecXT.getDVector(), vecDParms.getDVector(), 
                   vecTParms.getDVector(), (bool) PS_FALSE, (int) PS_FALSE, 
                   (double *) NULL, (double *) NULL);
  }
  tgp_->Init();

  //**/ tgp MCMC rounds are done here 
  tgp_->Rounds();

  //**/ get (possibly unchanged) pseudo-prior 
  tgp_->GetPseudoPrior(vecTParms.getDVector());

  //**/ get the (tree) acceptance rates
  vecGpcs.setLength(4);
  tgp_->GetTreeStats(vecGpcs.getDVector());

  //**/ destroy the RNG 
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  if (outputLevel_ >= 1) printf("TGP training ends.\n");
  if ((*N) == -999 || XOut == NULL || YOut == NULL) 
  {
    (*N) = 0;
    return 0;
  }

  //**/ evaluate NGrid points
  vecYOut.setLength(totPts);
  evaluatePoint(totPts, vecXT.getDVector(), vecYOut.getDVector());

  (*N) = totPts;
  (*XOut) = vecXT.takeDVector();
  (*YOut) = vecYOut.takeDVector();
#else
  printf("PSUADE ERROR : TGP not installed.\n");
  (*N) = 0;
  (*XOut) = NULL;
  (*YOut) = NULL;
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int TGP::gen1DGridData(double *X, double *Y, int ind1, double *settings, 
                       int *n, double **XOut, double **YOut)
{
#ifdef HAVE_TGP
  int    BTE[3], R=1, linBurn=1, PS_FALSE=0, PS_TRUE=1;
  int    totPts, ii, kk, count, bte0, bte1;
  double HX;
  void   *tgp_state=NULL;
  char   lineOut[101];
  psVector  vecDParms, vecTParms, vecGpcs, vecXOut, vecYOut, vecXT;
  psIVector vecStateIn;

  //**/ ------------------------------------------------------------
  //**/ interactive query
  //**/ ------------------------------------------------------------
  bte0 = 2000;
  bte1 = 7000;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("TGP: default burn-in sample size = 2000\n");
    printf("TGP: default MCMC    sample size = 7000\n");
    snprintf(lineOut,100,"TGP: enter burn-in sample size (e.g. 500-5000): ");
    bte0 = getInt(500, 5000, lineOut); 
    snprintf(lineOut,100,"TGP: enter MCMC sample size (e.g. %d-20000): ",
             2000+bte0);
    bte1 = getInt(2000+bte0, 20000, lineOut); 
  }

  //**/ ------------------------------------------------------------
  //**/ preparation
  //**/ ------------------------------------------------------------
  if (outputLevel_ >= 1) printf("TGP training begins....\n");
  if (tgp_ != NULL) delete tgp_;

  //**/ ------------------------------------------------------------
  //**/ create the RNG state 
  //**/ ------------------------------------------------------------
  vecStateIn.setLength(3);
  vecStateIn[0] = 782;
  vecStateIn[1] = 267;
  vecStateIn[2] = 218;
  unsigned int lstate = three2lstate(vecStateIn.getIVector());
  tgp_state = newRNGstate(lstate);

  //**/ ------------------------------------------------------------
  //**/ create parameter lists
  //**/ ------------------------------------------------------------
  BTE[0] = bte0;
  BTE[1] = bte1;
  BTE[2] = 2;
  vecDParms.setLength((nInputs_+1)*(nInputs_+1)+nInputs_+45);
  vecDParms[0] = 0.5;  // tree prior alpha
  vecDParms[1] = 2.0;  // tree prior beta
  vecDParms[2] = 10.0; // tree prior minpart
  if ((nInputs_ + 2) > 10) vecDParms[2] = 1.0 * (nInputs_ + 2.0);
  vecDParms[3] = 1.0;  // tree prior splitmin
  vecDParms[4] = 1.0 * nInputs_;  // tree prior basemax
  vecDParms[5] = 0.0;  // meanfn:  0 - linear, 1 : constant
  vecDParms[6] = 2.0;  // bflat in {b0, bmle, bflat, b0not,..}
  //**/ beta (nInputs+1 zeros)
  for (ii = 0; ii <= nInputs_; ii++) vecDParms[7+ii] = 0.0;
  count = nInputs_ + 8;
  vecDParms[count] = 1.0;  // Wi (1, (nInputs+1)^2 zeros)
  for (ii = 0; ii < (nInputs_+1)*(nInputs_+1)-1; ii++)
    vecDParms[count+1+ii] = 0.0;
  for (ii = 1; ii < nInputs_; ii++) 
    vecDParms[count+ii*(nInputs_+1)] = 1.0;
  count += (nInputs_ + 1) * (nInputs_ + 1);
  vecDParms[count++] = 1.0;   // s2tau2 = c(1,1)
  vecDParms[count++] = 1.0;
  vecDParms[count++] = 5.0;   // s2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // s2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 5.0;   // tau2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // tau2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 0.1;
  vecDParms[count++] = 1;     // "expsep"
  vecDParms[count++] = 0.1;   // gd (1)
  vecDParms[count++] = 0.5;   // gd (2)
  vecDParms[count++] = 1;     // nug.p (1)
  vecDParms[count++] = 1;     // nug.p (2)
  vecDParms[count++] = 1;     // nug.p (3)
  vecDParms[count++] = 1;     // nug.p (4)
  vecDParms[count++] = -1;    // nug.lam (1)
  vecDParms[count++] = -1;    // nug.lam (2)
  vecDParms[count++] = -1;    // nug.lam (3)
  vecDParms[count++] = -1;    // nug.lam (4)
  vecDParms[count++] = 10.0;  // gamma(1)
  vecDParms[count++] = 0.2;   // gamma(2)
  vecDParms[count++] = 0.7;   // gamma(3)
  vecDParms[count++] = 1.0;   // d.p(1)
  vecDParms[count++] = 20.0;  // d.p(2)
  vecDParms[count++] = 10.0;  // d.p(3)
  vecDParms[count++] = 10.0;  // d.p(4)
  vecDParms[count++] = -1;    // d.lam (1)
  vecDParms[count++] = -1;    // d.lam (2)
  vecDParms[count++] = -1;    // d.lam (3)
  vecDParms[count++] = -1;    // d.lam (4)
  vecDParms[count++] = 0;     // nu?
  vecTParms.setLength(7);
  vecTParms[0] = 1;
  vecTParms[1] = 0;
  vecTParms[2] = 0;
  vecTParms[3] = 1;
  vecTParms[4] = 1;
  vecTParms[5] = 0;
  vecTParms[6] = 1;

  //**/ ------------------------------------------------------------
  //**/ create predictions 
  //**/ ------------------------------------------------------------

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
    
  //**/ call TGP 
  tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                 (int) nSamples_, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                 (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_TRUE, 
                 (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                 X, Y, X, vecDParms.getDVector(), vecTParms.getDVector(), 
                 (bool) PS_FALSE, (int) PS_FALSE, (double *) NULL, 
                 (double *) NULL);
   
  //**/ initialize 
  tgp_->Init();

  //**/ tgp MCMC rounds are done here 
  tgp_->Rounds();

  //**/ interpolate
  if (outputLevel_ >= 1) printf("TGP interpolation begins....\n");
  vecYOut.setLength(totPts);
  evaluatePoint(totPts, vecXT.getDVector(), vecYOut.getDVector());

  //**/ get (possibly unchanged) pseudo--prior 
  tgp_->GetPseudoPrior(vecTParms.getDVector());

  //**/ get the (tree) acceptance rates
  vecGpcs.setLength(4);
  tgp_->GetTreeStats(vecGpcs.getDVector());

  //**/ destroy the RNG 
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  if (outputLevel_ >= 1) printf("TGP interpolation completed.\n");
  (*n) = totPts;
  (*YOut) = vecYOut.takeDVector();
#else
  printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int TGP::gen2DGridData(double *X, double *Y, int ind1, int ind2, 
                 double *settings, int *n, double **XOut, double **YOut)
{
#ifdef HAVE_TGP
  int    BTE[3], R=1, linBurn=1, PS_FALSE=0, PS_TRUE=1, bte0, bte1;
  int    totPts, ii, jj, kk, count, index;
  void   *tgp_state=NULL;
  char   lineOut[101];
  psVector vecDParms, vecTParms, vecHX, vecXT, vecXOut, vecYOut, vecGpcs;
  psIVector vecStateIn;

  //**/ ------------------------------------------------------------
  //**/ interactive query
  //**/ ------------------------------------------------------------
  bte0 = 2000;
  bte1 = 7000;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("TGP: default burn-in sample size = 2000\n");
    printf("TGP: default MCMC    sample size = 7000\n");
    snprintf(lineOut,100, "TGP: enter burn-in sample size (e.g. 500-5000): ");
    bte0 = getInt(500, 5000, lineOut); 
    snprintf(lineOut,100,"TGP: enter MCMC sample size (e.g. %d-20000): ",
            2000+bte0);
    bte1 = getInt(2000+bte0, 20000, lineOut); 
  }

  //**/ ------------------------------------------------------------
  //**/ preparation
  //**/ ------------------------------------------------------------
  if (outputLevel_ >= 1) printf("TGP training begins....\n");
  if (tgp_ != NULL) delete tgp_;

  //**/ ------------------------------------------------------------
  //**/ create the RNG state 
  //**/ ------------------------------------------------------------
  vecStateIn.setLength(3);
  vecStateIn[0] = 782;
  vecStateIn[1] = 267;
  vecStateIn[2] = 218;
  unsigned int lstate = three2lstate(vecStateIn.getIVector());
  tgp_state = newRNGstate(lstate);

  //**/ ------------------------------------------------------------
  //**/ create parameter lists
  //**/ ------------------------------------------------------------
  BTE[0] = bte0;
  BTE[1] = bte1;
  BTE[2] = 2;
  vecDParms.setLength((nInputs_+1)*(nInputs_+1)+nInputs_+45);
  vecDParms[0] = 0.5;  // tree prior alpha
  vecDParms[1] = 2.0;  // tree prior beta
  vecDParms[2] = 10.0; // tree prior minpart
  if ((nInputs_ + 2) > 10) vecDParms[2] = 1.0 * (nInputs_ + 2.0);
  vecDParms[3] = 1.0;  // tree prior splitmin
  vecDParms[4] = 1.0 * nInputs_;  // tree prior basemax
  vecDParms[5] = 0.0;  // meanfn:  0 - linear, 1 : constant
  vecDParms[6] = 2.0;  // bflat in {b0, bmle, bflat, b0not,..}
  //**/ beta (nInputs+1 zeros)
  for (ii = 0; ii <= nInputs_; ii++) vecDParms[7+ii] = 0.0;
  count = nInputs_ + 8;
  vecDParms[count] = 1.0;  // Wi (1, (nInputs+1)^2 zeros)
  for (ii = 0; ii < (nInputs_+1)*(nInputs_+1)-1; ii++)
    vecDParms[count+1+ii] = 0.0;
  for (ii = 1; ii < nInputs_; ii++) 
    vecDParms[count+ii*(nInputs_+1)] = 1.0;
  count += (nInputs_ + 1) * (nInputs_ + 1);
  vecDParms[count++] = 1.0;   // s2tau2 = c(1,1)
  vecDParms[count++] = 1.0;
  vecDParms[count++] = 5.0;   // s2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // s2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 5.0;   // tau2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // tau2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 0.1;
  vecDParms[count++] = 1;     // "expsep"
  vecDParms[count++] = 0.1;   // gd (1)
  vecDParms[count++] = 0.5;   // gd (2)
  vecDParms[count++] = 1;     // nug.p (1)
  vecDParms[count++] = 1;     // nug.p (2)
  vecDParms[count++] = 1;     // nug.p (3)
  vecDParms[count++] = 1;     // nug.p (4)
  vecDParms[count++] = -1;    // nug.lam (1)
  vecDParms[count++] = -1;    // nug.lam (2)
  vecDParms[count++] = -1;    // nug.lam (3)
  vecDParms[count++] = -1;    // nug.lam (4)
  vecDParms[count++] = 10.0;  // gamma(1)
  vecDParms[count++] = 0.2;   // gamma(2)
  vecDParms[count++] = 0.7;   // gamma(3)
  vecDParms[count++] = 1.0;   // d.p(1)
  vecDParms[count++] = 20.0;  // d.p(2)
  vecDParms[count++] = 10.0;  // d.p(3)
  vecDParms[count++] = 10.0;  // d.p(4)
  vecDParms[count++] = -1;    // d.lam (1)
  vecDParms[count++] = -1;    // d.lam (2)
  vecDParms[count++] = -1;    // d.lam (3)
  vecDParms[count++] = -1;    // d.lam (4)
  vecDParms[count++] = 0;     // nu?
  vecTParms.setLength(7);
  vecTParms[0] = 1;
  vecTParms[1] = 0;
  vecTParms[2] = 0;
  vecTParms[3] = 1;
  vecTParms[4] = 1;
  vecTParms[5] = 0;
  vecTParms[6] = 1;

  //**/ ------------------------------------------------------------
  //**/ create function 
  //**/ ------------------------------------------------------------

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        vecXT[index*nInputs_+kk] = settings[kk]; 
      vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }
   
  //**/ call TGP 
  tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                 (int) nSamples_, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                 (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_TRUE, 
                 (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                 X, Y, X, vecDParms.getDVector(), vecTParms.getDVector(), 
                 (bool) PS_FALSE, (int) PS_FALSE, (double *) NULL, 
                 (double *) NULL);

  //**/ initialize 
  tgp_->Init();

  //**/ tgp MCMC rounds are done here 
  tgp_->Rounds();

  //**/ interpolate
  if (outputLevel_ >= 1) printf("TGP interpolation begins....\n");
  vecYOut.setLength(totPts);
  evaluatePoint(totPts, vecXT.getDVector(), vecYOut.getDVector());

  //**/ get (possibly unchanged) pseudo--prior 
  tgp_->GetPseudoPrior(vecTParms.getDVector());

  //**/ get the (tree) acceptance rates
  vecGpcs.setLength(4);
  tgp_->GetTreeStats(vecGpcs.getDVector());

  //**/ destroy the RNG 
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  if (outputLevel_ >= 1) printf("TGP interpolation completed.\n");
  (*n) = totPts;
  (*YOut) = vecYOut.takeDVector();
#else
  printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int TGP::gen3DGridData(double *X, double *Y, int ind1, int ind2, int ind3,
                  double *settings, int *n, double **XOut, double **YOut)
{
#ifdef HAVE_TGP
  int    BTE[3], R=1, linBurn=1, PS_FALSE=0, PS_TRUE=1, bte0, bte1;
  int    totPts, ii, jj, ll, kk, count, index;
  void   *tgp_state=NULL;
  char   lineOut[101];
  psVector vecHX, vecXT, vecDParms, vecTParms, vecGpcs, vecXOut, vecYOut;
  psIVector vecStateIn;

  //**/ ------------------------------------------------------------
  //**/ interactive query
  //**/ ------------------------------------------------------------
  bte0 = 2000;
  bte1 = 7000;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("TGP: default burn-in sample size = 2000\n");
    printf("TGP: default MCMC    sample size = 7000\n");
    snprintf(lineOut,100,"TGP: enter burn-in sample size (e.g. 500-5000): ");
    bte0 = getInt(500, 5000, lineOut); 
    snprintf(lineOut,100,"TGP: enter MCMC sample size (e.g. %d-20000): ",
            2000+bte0);
    bte1 = getInt(2000+bte0, 20000, lineOut); 
  }

  //**/ ------------------------------------------------------------
  //**/ preparation
  //**/ ------------------------------------------------------------
  if (outputLevel_ >= 1) printf("TGP training begins....\n");
  if (tgp_ != NULL) delete tgp_;

  //**/ ------------------------------------------------------------
  //**/ create the RNG state 
  //**/ ------------------------------------------------------------
  vecStateIn.setLength(3);
  vecStateIn[0] = 782;
  vecStateIn[1] = 267;
  vecStateIn[2] = 218;
  unsigned int lstate = three2lstate(vecStateIn.getIVector());
  tgp_state = newRNGstate(lstate);

  //**/ ------------------------------------------------------------
  //**/ create parameter lists
  //**/ ------------------------------------------------------------
  BTE[0] = bte0;
  BTE[1] = bte1;
  BTE[2] = 2;
  vecDParms.setLength((nInputs_+1)*(nInputs_+1)+nInputs_+45);
  vecDParms[0] = 0.5;  // tree prior alpha
  vecDParms[1] = 2.0;  // tree prior beta
  vecDParms[2] = 10.0; // tree prior minpart
  if ((nInputs_ + 2) > 10) vecDParms[2] = 1.0 * (nInputs_ + 2.0);
  vecDParms[3] = 1.0;  // tree prior splitmin
  vecDParms[4] = 1.0 * nInputs_;  // tree prior basemax
  vecDParms[5] = 0.0;  // meanfn:  0 - linear, 1 : constant
  vecDParms[6] = 2.0;  // bflat in {b0, bmle, bflat, b0not,..}
  //**/ beta (nInputs+1 zeros)
  for (ii = 0; ii <= nInputs_; ii++) vecDParms[7+ii] = 0.0;
  count = nInputs_ + 8;
  vecDParms[count] = 1.0;  // Wi (1, (nInputs+1)^2 zeros)
  for (ii = 0; ii < (nInputs_+1)*(nInputs_+1)-1; ii++)
    vecDParms[count+1+ii] = 0.0;
  for (ii = 1; ii < nInputs_; ii++) 
    vecDParms[count+ii*(nInputs_+1)] = 1.0;
  count += (nInputs_ + 1) * (nInputs_ + 1);
  vecDParms[count++] = 1.0;   // s2tau2 = c(1,1)
  vecDParms[count++] = 1.0;
  vecDParms[count++] = 5.0;   // s2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // s2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 5.0;   // tau2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // tau2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 0.1;
  vecDParms[count++] = 1;     // "expsep"
  vecDParms[count++] = 0.1;   // gd (1)
  vecDParms[count++] = 0.5;   // gd (2)
  vecDParms[count++] = 1;     // nug.p (1)
  vecDParms[count++] = 1;     // nug.p (2)
  vecDParms[count++] = 1;     // nug.p (3)
  vecDParms[count++] = 1;     // nug.p (4)
  vecDParms[count++] = -1;    // nug.lam (1)
  vecDParms[count++] = -1;    // nug.lam (2)
  vecDParms[count++] = -1;    // nug.lam (3)
  vecDParms[count++] = -1;    // nug.lam (4)
  vecDParms[count++] = 10.0;  // gamma(1)
  vecDParms[count++] = 0.2;   // gamma(2)
  vecDParms[count++] = 0.7;   // gamma(3)
  vecDParms[count++] = 1.0;   // d.p(1)
  vecDParms[count++] = 20.0;  // d.p(2)
  vecDParms[count++] = 10.0;  // d.p(3)
  vecDParms[count++] = 10.0;  // d.p(4)
  vecDParms[count++] = -1;    // d.lam (1)
  vecDParms[count++] = -1;    // d.lam (2)
  vecDParms[count++] = -1;    // d.lam (3)
  vecDParms[count++] = -1;    // d.lam (4)
  vecDParms[count++] = 0;     // nu?
  vecTParms.setLength(7);
  vecTParms[0] = 1;
  vecTParms[1] = 0;
  vecTParms[2] = 0;
  vecTParms[3] = 1;
  vecTParms[4] = 1;
  vecTParms[5] = 0;
  vecTParms[6] = 1;

  //**/ ------------------------------------------------------------
  //**/ create function 
  //**/ ------------------------------------------------------------

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (kk = 0; kk < nInputs_; kk++) 
          vecXT[index*nInputs_+kk] = settings[kk]; 
        vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }
    
  //**/ call TGP 
  tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                 (int) nSamples_, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                 (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_TRUE, 
                 (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                 X, Y, X, vecDParms.getDVector(), vecTParms.getDVector(), 
                 (bool) PS_FALSE, (int) PS_FALSE, (double *) NULL, 
                 (double *) NULL);

  //**/ initialize 
  tgp_->Init();

  //**/ tgp MCMC rounds are done here 
  tgp_->Rounds();

  //**/ interpolate
  if (outputLevel_ >= 1) printf("TGP interpolation begins....\n");
  vecYOut.setLength(totPts);
  evaluatePoint(totPts, vecXT.getDVector(), vecYOut.getDVector());

  //**/ get (possibly unchanged) pseudo--prior 
  tgp_->GetPseudoPrior(vecTParms.getDVector());

  //**/ get the (tree) acceptance rates
  vecGpcs.setLength(4);
  tgp_->GetTreeStats(vecGpcs.getDVector());

  //**/ destroy the RNG 
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  if (outputLevel_ >= 1) printf("TGP interpolation completed.\n");
  (*n) = totPts;
  (*YOut) = vecYOut.takeDVector();
#else
  printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int TGP::gen4DGridData(double *X, double *Y, int ind1, int ind2, int ind3,
                       int ind4, double *settings, int *n, double **XOut, 
                       double **YOut)
{
#ifdef HAVE_TGP
  int    BTE[3], R=1, linBurn=1, PS_FALSE=0, PS_TRUE=1, bte0, bte1;
  int    totPts, ii, jj, ll, mm, kk, count, index;
  void   *tgp_state=NULL;
  char   lineOut[101];
  psVector vecHX, vecXT, vecDParms, vecTParms, vecGpcs, vecXOut, vecYOut;
  psIVector vecStateIn;

  //**/ ------------------------------------------------------------
  //**/ interactive query
  //**/ ------------------------------------------------------------
  bte0 = 2000;
  bte1 = 7000;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("TGP: default burn-in sample size = 2000\n");
    printf("TGP: default MCMC    sample size = 7000\n");
    snprintf(lineOut,100,"TGP: enter burn-in sample size (e.g. 500-5000): ");
    bte0 = getInt(500, 5000, lineOut); 
    snprintf(lineOut,100,"TGP: enter MCMC sample size (e.g. %d-20000): ",
             2000+bte0);
    bte1 = getInt(2000+bte0, 20000, lineOut); 
  }

  //**/ ------------------------------------------------------------
  //**/ preparation
  //**/ ------------------------------------------------------------
  if (outputLevel_ >= 1) printf("TGP training begins....\n");
  if (tgp_ != NULL) delete tgp_;

  //**/ ------------------------------------------------------------
  //**/ create the RNG state 
  //**/ ------------------------------------------------------------
  vecStateIn.setLength(3);
  vecStateIn[0] = 782;
  vecStateIn[1] = 267;
  vecStateIn[2] = 218;
  unsigned int lstate = three2lstate(vecStateIn.getIVector());
  tgp_state = newRNGstate(lstate);

  //**/ ------------------------------------------------------------
  //**/ create parameter lists
  //**/ ------------------------------------------------------------
  BTE[0] = bte0;
  BTE[1] = bte1;
  BTE[2] = 2;
  vecDParms.setLength((nInputs_+1)*(nInputs_+1)+nInputs_+45);
  vecDParms[0] = 0.5;  // tree prior alpha
  vecDParms[1] = 2.0;  // tree prior beta
  vecDParms[2] = 10.0; // tree prior minpart
  if ((nInputs_ + 2) > 10) vecDParms[2] = 1.0 * (nInputs_ + 2.0);
  vecDParms[3] = 1.0;  // tree prior splitmin
  vecDParms[4] = 1.0 * nInputs_;  // tree prior basemax
  vecDParms[5] = 0.0;  // meanfn:  0 - linear, 1 : constant
  vecDParms[6] = 2.0;  // bflat in {b0, bmle, bflat, b0not,..}
  //**/ beta (nInputs+1 zeros)
  for (ii = 0; ii <= nInputs_; ii++) vecDParms[7+ii] = 0.0;
  count = nInputs_ + 8;
  vecDParms[count] = 1.0;  // Wi (1, (nInputs+1)^2 zeros)
  for (ii = 0; ii < (nInputs_+1)*(nInputs_+1)-1; ii++)
    vecDParms[count+1+ii] = 0.0;
  for (ii = 1; ii < nInputs_; ii++) 
    vecDParms[count+ii*(nInputs_+1)] = 1.0;
  count += (nInputs_ + 1) * (nInputs_ + 1);
  vecDParms[count++] = 1.0;   // s2tau2 = c(1,1)
  vecDParms[count++] = 1.0;
  vecDParms[count++] = 5.0;   // s2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // s2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 5.0;   // tau2 prior (a0, g0) = c(5,10)
  vecDParms[count++] = 10.0;
  vecDParms[count++] = 0.2;   // tau2 hierar inv-gamma prior = c(5,10)
  vecDParms[count++] = 0.1;
  vecDParms[count++] = 1;     // "expsep"
  vecDParms[count++] = 0.1;   // gd (1)
  vecDParms[count++] = 0.5;   // gd (2)
  vecDParms[count++] = 1;     // nug.p (1)
  vecDParms[count++] = 1;     // nug.p (2)
  vecDParms[count++] = 1;     // nug.p (3)
  vecDParms[count++] = 1;     // nug.p (4)
  vecDParms[count++] = -1;    // nug.lam (1)
  vecDParms[count++] = -1;    // nug.lam (2)
  vecDParms[count++] = -1;    // nug.lam (3)
  vecDParms[count++] = -1;    // nug.lam (4)
  vecDParms[count++] = 10.0;  // gamma(1)
  vecDParms[count++] = 0.2;   // gamma(2)
  vecDParms[count++] = 0.7;   // gamma(3)
  vecDParms[count++] = 1.0;   // d.p(1)
  vecDParms[count++] = 20.0;  // d.p(2)
  vecDParms[count++] = 10.0;  // d.p(3)
  vecDParms[count++] = 10.0;  // d.p(4)
  vecDParms[count++] = -1;    // d.lam (1)
  vecDParms[count++] = -1;    // d.lam (2)
  vecDParms[count++] = -1;    // d.lam (3)
  vecDParms[count++] = -1;    // d.lam (4)
  vecDParms[count++] = 0;     // nu?
  vecTParms.setLength(7);
  vecTParms[0] = 1;
  vecTParms[1] = 0;
  vecTParms[2] = 0;
  vecTParms[3] = 1;
  vecTParms[4] = 1;
  vecTParms[5] = 0;
  vecTParms[6] = 1;

  //**/ ------------------------------------------------------------
  //**/ create function 
  //**/ ------------------------------------------------------------

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        for (mm = 0; mm < nPtsPerDim_; mm++) 
        {
          index = ii*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                  jj* nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          for (kk = 0; kk < nInputs_; kk++) 
            vecXT[index*nInputs_+kk] = settings[kk]; 
          vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
          vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
          vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
          vecXT[index*nInputs_+ind4] = vecHX[3] * mm + VecLBs_[ind4];
          (*XOut)[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }
    
  //**/ call TGP 
  tgp_ = new Tgp((void*) tgp_state, (int) nSamples_, (int) nInputs_, 
                 (int) nSamples_, (int) BTE[0], (int) BTE[1], (int) BTE[2], 
                 (int) R, (int) linBurn, (bool) PS_TRUE, (bool) PS_TRUE, 
                 (bool) PS_FALSE, (int) PS_FALSE, (bool) PS_FALSE, 
                 X, Y, X, vecDParms.getDVector(), vecTParms.getDVector(), 
                 (bool) PS_FALSE, (int) PS_FALSE, (double *) NULL, 
                 (double *) NULL);

  //**/ initialize 
  tgp_->Init();

  //**/ tgp MCMC rounds are done here 
  tgp_->Rounds();

  //**/ interpolate
  if (outputLevel_ >= 1) printf("TGP interpolation begins....\n");
  vecYOut.setLength(totPts);
  evaluatePoint(totPts, vecXT.getDVector(), vecYOut.getDVector());

  //**/ get (possibly unchanged) pseudo--prior 
  tgp_->GetPseudoPrior(vecTParms.getDVector());

  //**/ get the (tree) acceptance rates
  vecGpcs.setLength(4);
  tgp_->GetTreeStats(vecGpcs.getDVector());

  //**/ destroy the RNG 
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  if (outputLevel_ >= 1) printf("TGP interpolation completed.\n");
  (*n) = totPts;
  (*YOut) = vecYOut.takeDVector();
#else
  printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double TGP::evaluatePoint(double *X)
{
#ifdef HAVE_TGP
  int    iOne=1;
  double Y;
  evaluatePoint(iOne, X, &Y);
  return Y;
#else
  printf("PSUADE ERROR : TGP not installed.\n");
  return 0.0;
#endif
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double TGP::evaluatePoint(int npts, double *X, double *Y)
{
#ifdef HAVE_TGP
  int    iZero=0, PS_FALSE=0, chunksize, ii, jj, kk, nChunks;
  psVector vecEssOut, vecXX, vecYY;

  nChunks = npts / nSamples_;
  if (nChunks == 0) nChunks = 1;

  vecEssOut.setLength(3);
  vecXX.setLength(nSamples_*nInputs_);
  vecYY.setLength(nSamples_);
   
  for (ii = 0; ii < nChunks; ii++)
  {
    if (outputLevel_ > 1 && nChunks > 1)
      printf("TGP: evaluate chunk %d (out of %d)\n",ii+1,nChunks);
    chunksize = nSamples_;
    if (ii*nSamples_+chunksize > npts) chunksize = npts - ii * nSamples_;
    //**/ fill in the chunk 
    for (jj = 0; jj < chunksize*nInputs_; jj++)
      vecXX[jj] = X[ii*nSamples_*nInputs_+jj]; 
    //**/ just fill in the empty space if chunksize < nSamples
    for (jj = chunksize; jj < nSamples_; jj++)
      for (kk = 0; kk < nInputs_; kk++)
        vecXX[jj*nInputs_+kk] = 
          X[ii*nSamples_*nInputs_+(chunksize-1)*nInputs_+kk]; 
    tgp_->LoadInterpolatedPts(nSamples_, vecXX.getDVector());
    tgp_->Predict();
    tgp_->GetStats(iZero,vecZpMean_.getDVector(),vecYY.getDVector(),
             NULL,NULL,vecZpQ_.getDVector(),vecZZQ_.getDVector(),
             PS_FALSE,vecZpS2_.getDVector(),vecZZS2_.getDVector(),
             NULL,NULL,NULL,vecZpQ1_.getDVector(),
             vecZpMedian_.getDVector(),vecZpQ2_.getDVector(),
             vecZZQ1_.getDVector(),vecZZMedian_.getDVector(),
             vecZZQ2_.getDVector(),NULL,NULL,NULL,vecEssOut.getDVector());
    for (jj = 0; jj < chunksize; jj++) Y[ii*nSamples_+jj] = vecYY[jj]; 
  }
  if (nSamples_*nChunks < npts)
  {
    for (ii = nSamples_*nChunks; ii < npts; ii++)
      for (jj = 0; jj < nInputs_; jj++) 
        vecXX[(ii-nSamples_*nChunks)*nInputs_+jj] = X[ii*nInputs_+jj];
    for (ii = npts-nSamples_*nChunks; ii < nSamples_; ii++)
      for (jj = 0; jj < nInputs_; jj++) 
        vecXX[ii*nInputs_+jj] = vecXX[(ii-1)*nInputs_+jj];
    tgp_->LoadInterpolatedPts(nSamples_, vecXX.getDVector());
    tgp_->Predict();
    tgp_->GetStats(iZero,vecZpMean_.getDVector(),vecYY.getDVector(),
             NULL,NULL,vecZpQ_.getDVector(),vecZZQ_.getDVector(),
             PS_FALSE,vecZpS2_.getDVector(),vecZZS2_.getDVector(),NULL,
             NULL,NULL,vecZpQ1_.getDVector(),vecZpMedian_.getDVector(),
             vecZpQ2_.getDVector(),vecZZQ1_.getDVector(),
             vecZZMedian_.getDVector(),vecZZQ2_.getDVector(),NULL,NULL,
             NULL,vecEssOut.getDVector());
    for (jj = 0; jj < npts-nChunks*nSamples_; jj++)
      Y[nChunks*nSamples_+jj] = vecYY[jj]; 
  }
#else
   printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and return also the standard deviation 
// ------------------------------------------------------------------------
double TGP::evaluatePointFuzzy(double *X, double &std)
{
#ifdef HAVE_TGP
  int    iOne=1;
  double Y;
  evaluatePointFuzzy(iOne, X, &Y, &std);
  return Y;
#else
  printf("PSUADE ERROR : TGP not installed.\n");
  return 0.0;
#endif
}

// ************************************************************************
// Evaluate a number of points and return also the standard deviations 
// ------------------------------------------------------------------------
double TGP::evaluatePointFuzzy(int npts,double *X, double *Y, double *Ystd)
{
#ifdef HAVE_TGP
  int    iZero=0, PS_FALSE=0, ii, jj, kk, nChunks, chunkSize;
  psVector vecEssOut, vecXX, vecYY, vecYS;

  nChunks = npts / nSamples_;
  if (nChunks == 0) nChunks = 1;
  vecXX.setLength(nSamples_*nInputs_);
  vecYY.setLength(nSamples_);
  vecYS.setLength(nSamples_);
  vecEssOut.setLength(3);
   
  for (ii = 0; ii < nChunks; ii++)
  {
    if (outputLevel_ > 1 && nChunks > 1)
      printf("TGP: evaluate chunk %d (out of %d)\n",ii+1,nChunks);
    chunkSize = nSamples_;
    if (ii*nSamples_+chunkSize > npts) chunkSize = npts - ii * nSamples_;
    //**/ fill in the chunk 
    for (jj = 0; jj < chunkSize*nInputs_; jj++)
      vecXX[jj] = X[ii*nSamples_*nInputs_+jj]; 
    //**/ just fill in the empty space if chunkSize < nSamples
    for (jj = chunkSize; jj < nSamples_; jj++)
      for (kk = 0; kk < nInputs_; kk++)
        vecXX[jj*nInputs_+kk] = 
          X[ii*nSamples_*nInputs_+(chunkSize-1)*nInputs_+kk]; 
    tgp_->LoadInterpolatedPts(nSamples_, vecXX.getDVector());
    tgp_->Predict();
    tgp_->GetStats(iZero,vecZpMean_.getDVector(),vecYY.getDVector(),
             NULL,NULL,vecZpQ_.getDVector(),vecZZQ_.getDVector(),
             PS_FALSE,vecZpS2_.getDVector(),vecYS.getDVector(),
             NULL,NULL,NULL,vecZpQ1_.getDVector(),
             vecZpMedian_.getDVector(),vecZpQ2_.getDVector(),
             vecZZQ1_.getDVector(),vecZZMedian_.getDVector(),
             vecZZQ2_.getDVector(),NULL,NULL,NULL,vecEssOut.getDVector());
    for (jj = 0; jj < chunkSize; jj++) Y[ii*nSamples_+jj] = vecYY[jj]; 
    for (jj = 0; jj < chunkSize; jj++) Ystd[ii*nSamples_+jj] = vecYS[jj]; 
  }
  if (nSamples_*nChunks < npts)
  {
    for (ii = nSamples_*nChunks; ii < npts; ii++)
      for (jj = 0; jj < nInputs_; jj++) 
        vecXX[(ii-nSamples_*nChunks)*nInputs_+jj] = X[ii*nInputs_+jj];
    for (ii = npts-nSamples_*nChunks; ii < nSamples_; ii++)
      for (jj = 0; jj < nInputs_; jj++) 
        vecXX[ii*nInputs_+jj] = vecXX[(ii-1)*nInputs_+jj];
    tgp_->LoadInterpolatedPts(nSamples_, vecXX.getDVector());
    tgp_->Predict();
    tgp_->GetStats(iZero,vecZpMean_.getDVector(),vecYY.getDVector(),
             NULL,NULL,vecZpQ_.getDVector(),vecZZQ_.getDVector(),
             PS_FALSE,vecZpS2_.getDVector(),vecYS.getDVector(),NULL,
             NULL,NULL,vecZpQ1_.getDVector(),vecZpMedian_.getDVector(),
             vecZpQ2_.getDVector(),vecZZQ1_.getDVector(),
             vecZZMedian_.getDVector(),vecZZQ2_.getDVector(),NULL,NULL,
             NULL,vecEssOut.getDVector());
    for (jj = 0; jj < npts-nChunks*nSamples_; jj++)
      Y[nChunks*nSamples_+jj] = vecYY[jj]; 
    for (jj = 0; jj < npts-nChunks*nSamples_; jj++)
      Ystd[nChunks*nSamples_+jj] = vecYS[jj]; 
  }
#else
  printf("PSUADE ERROR : TGP not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double TGP::setParams(int targc, char **targv)
{
  if (targc > 0 && !strcmp(targv[0], "improv"))
  {
#ifdef HAVE_TGP
#else
     printf("PSUADE ERROR : TGP not installed.\n");
#endif
  }
  return 0.0;
}

