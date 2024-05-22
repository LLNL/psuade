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
// Functions for the class DNN
// AUTHOR : Charles Tong 
// DATE   : March, 2020
// ************************************************************************
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "sysdef.h"
#include "DNN.h"
#include "Globals.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"
#include "Sampling.h"
#include "PDFNormal.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))
// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
#ifdef HAVE_LBFGS
#include "../../External/L-BFGS-B-C/src/lbfgsb.h"
#endif
}

//*************************************************************************
// Constructor
//*************************************************************************
DNN::DNN(int nInputs, int nSamples) : FuncApprox(nInputs, nSamples)
{	
  int  ii, numNodes;
  char pString[1000];

  //**/ ----------------------------------------------------------------
  //**/ initialize some standard hyperparameters
  //**/ ----------------------------------------------------------------
  faID_      = PSUADE_RS_ANN;
  nLevels_   = 2;   /* 2 = 1 hidden layer */
  numNodes   = 20;  /* default number of nodes */
  optScheme_ = 0;   /* 0 - LBFGS, 1 - gradient descent */
  alpha_     = 0.1; /* learning rate : not used in LBFGS */
  maxIter_   = 1000;

  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*           Neural Network Model\n");
    printOutTS(PL_INFO,"* Default architecture:\n");
    printOutTS(PL_INFO,"* - 1 input layer\n");
    printOutTS(PL_INFO,"* - 1 output layer\n");
    printOutTS(PL_INFO,"* - 1 hidden layer\n");
    printOutTS(PL_INFO,"* Default number of nodes per hidden layer = 20\n");
    printOutTS(PL_INFO,"* Default activation function = ReLU\n");
    printOutTS(PL_INFO,"* Default weight initialization = He's method\n");
    printOutTS(PL_INFO,"* Default optimizer = LBFGS\n");
    printOutTS(PL_INFO,"* Default max iter  = %d\n", maxIter_);
    printEquals(PL_INFO, 0);
  }

  //**/ ----------------------------------------------------------------
  //**/ allow users to modify these parameters
  //**/ (Note: during CV, both rs_expert and interactive are turned off)
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    char pString[1000];
    snprintf(pString,100,"How many hidden layers? (1 - 4) ");
    nLevels_ = getInt(1,4,pString) + 1;
    snprintf(pString,100,"RS_DNN_nlevels = %d", nLevels_);
    psConfig_.putParameter(pString);

    snprintf(pString,100,"Number of nodes in each hidden layer? (3 - 100) ");
    numNodes = getInt(3,100,pString);
    snprintf(pString,100,"RS_DNN_nnodes = %d", numNodes);
    psConfig_.putParameter(pString);

    snprintf(pString,100,
      "Optimization scheme (0: LBFGS, 1: Gradient descent) ? (0 - 1) ");
    optScheme_ = getInt(0,1,pString);
    snprintf(pString,100,"RS_DNN_optscheme = %d", optScheme_);
    psConfig_.putParameter(pString);

    if (optScheme_ == 1)
    {
      snprintf(pString,100,
           "Learning rate (default = 0.1, suggested: [0.01,0.5]) ? ");
      alpha_ = getDouble(pString);
      snprintf(pString,100,"RS_DNN_alpha = %e", alpha_);
      psConfig_.putParameter(pString);
    }

    snprintf(pString,100,
      "Maximum iteration for optimization (100 - 10000) : ");
    maxIter_ = getInt(100,10000,pString);
    snprintf(pString,100,"RS_DNN_maxiter = %d", maxIter_);
    psConfig_.putParameter(pString);
  }

  //**/ or, these hyperparameters can be extracted from the config object
  else
  {
    char winput1[1000], winput2[1000];
    char *cString = psConfig_.getParameter("RS_DNN_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels_);
      if (nLevels_ < 1 || nLevels_ > 4)
      {
        printOutTS(PL_INFO,
             "DNN WARNING: Invalid nlevels %d - reset to 1\n",nLevels_);
        nLevels_ = 1;
      }
      printOutTS(PL_INFO,"DNN INFO: nLevels set to %d\n",nLevels_);
    }
    cString = psConfig_.getParameter("RS_DNN_nnodes");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &numNodes);
      if (numNodes < 3 || numNodes > 100)
      {
        printOutTS(PL_INFO,
          "DNN WARNING: Invalid nNodes/level %d - reset to 10\n",
          numNodes);
        numNodes = 10;
      }
      printOutTS(PL_INFO,"DNN INFO: numNodes set to %d\n",numNodes);
    }
    cString = psConfig_.getParameter("RS_DNN_optscheme");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &optScheme_);
      if (optScheme_ < 0 || optScheme_ > 1)
      {
        printOutTS(PL_INFO,
           "DNN WARNING: Invalid Opt scheme - reset to LBFGS\n");
        optScheme_ = 0;
      }
      if (optScheme_ == 0)
        printOutTS(PL_INFO,"DNN INFO: Opt scheme set to LBFGS\n");
      else
        printOutTS(PL_INFO,
            "DNN INFO: Opt scheme set to Gradient descent\n");
    }
    if (optScheme_ == 1)
    {
      cString = psConfig_.getParameter("RS_DNN_alpha");
      if (cString != NULL)
      {
        sscanf(cString, "%s %s %lg", winput1, winput2, &alpha_);
        if (alpha_ < 0.01 || alpha_ > 0.5)
        {
          printOutTS(PL_INFO,
            "DNN WARNING: Invalid Opt learning rate - reset to 0.1\n");
          alpha_ = 0.1;
        }
        printOutTS(PL_INFO,
            "DNN INFO: Opt learning rate set to %e\n",alpha_);
      }
    }
    cString = psConfig_.getParameter("RS_DNN_maxiter");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &maxIter_);
      if (maxIter_ < 100 || maxIter_ > 10000)
      {
        printOutTS(PL_INFO,
             "DNN WARNING: Invalid Opt max iter - reset to 1000\n");
        maxIter_ = 1000;
      }
      printOutTS(PL_INFO,"DNN INFO: Opt max iter set to %d\n",maxIter_);
    }
  }

  //**/ ----------------------------------------------------------------
  //*/ set other hyperparameters, which are not currently used
  //**/ ----------------------------------------------------------------
  alpha0_  = 0.2;   /* alpha0 in adaptive learning rate: alpha0/(1+epoch) */
  beta_    = 0.9;   /* beta in exponential weighted averages */
  beta1_   = 0.9;   /* beta1 in Adam (momentum) */
  beta2_   = 0.999; /* beta2 in Adam (RMSprop) */
  epsilon_ = 1e-8;  /* epsilon in Adam */
  regularizationOn_ = 0;

  //**/ ----------------------------------------------------------------
  //**/ allocate internal storage
  //**/ ----------------------------------------------------------------
  VecDropOutRatios_.setLength(nLevels_+1);
  VecActivationFcns_.setLength(nLevels_+1);;
  VecNumNodes_.setLength(nLevels_+1);
  MatWs_  = new psMatrix*[nLevels_+1];
  VecBs_  = new psVector*[nLevels_+1];
  MatZs_  = new psMatrix*[nLevels_+1];
  MatAs_  = new psMatrix*[nLevels_+1];
  MatdWs_ = new psMatrix*[nLevels_+1];
  VecdBs_ = new psVector*[nLevels_+1];
  MatdZs_ = new psMatrix*[nLevels_+1];
  for (ii = 0; ii <= nLevels_; ii++)
  {
    if (ii > 0) 
    {
      MatWs_[ii] = new psMatrix();
      VecBs_[ii] = new psVector();
    }
    else
    {
      MatWs_[ii] = NULL;
      VecBs_[ii] = NULL;
    }
    VecBs_[ii] = new psVector();
    MatZs_[ii] = new psMatrix();
    MatAs_[ii] = new psMatrix();
    MatdWs_[ii] = new psMatrix();
    VecdBs_[ii] = new psVector();
    MatdZs_[ii] = new psMatrix();
    VecActivationFcns_[ii] = 2; /* ReLU */
    VecNumNodes_[ii] = numNodes;
    VecDropOutRatios_[ii] = 0;  /* for now, no drop out */
  }
  VecActivationFcns_[0] = -1;       /* Input layer: no activation fcn */
  VecActivationFcns_[nLevels_] = 1; /* Output layer: tanh */
  VecNumNodes_[0] = nInputs_;       /* Input layer number of nodes */
  VecNumNodes_[nLevels_] = 1;       /* Output layer number of nodes = 1 */
  batchSize_ = nSamples_;           /* for now, use all */
}

//*************************************************************************
// destructor
//*************************************************************************
DNN::~DNN()
{
  for (int ii = 0; ii <= nLevels_; ii++)
  {
    if (MatWs_[ii]  != NULL) delete MatWs_[ii];
    if (VecBs_[ii]  != NULL) delete VecBs_[ii];
    if (MatZs_[ii]  != NULL) delete MatZs_[ii];
    if (MatAs_[ii]  != NULL) delete MatAs_[ii];
    if (MatdWs_[ii] != NULL) delete MatdWs_[ii];
    if (VecdBs_[ii] != NULL) delete VecdBs_[ii];
    if (MatdZs_[ii] != NULL) delete MatdZs_[ii];
  }
  if (MatWs_  != NULL) delete [] MatWs_;
  if (VecBs_  != NULL) delete [] VecBs_;
  if (MatZs_  != NULL) delete [] MatZs_;
  if (MatAs_  != NULL) delete [] MatAs_;
  if (MatdWs_ != NULL) delete [] MatdWs_;
  if (VecdBs_ != NULL) delete [] VecdBs_;
  if (MatdZs_ != NULL) delete [] MatdZs_;
}

//*************************************************************************
// initialization given training sample
//*************************************************************************
int DNN::initialize(double *XIn, double *YIn) 
{
  psVector vecXN, vecYN;
  //**/ normalize incoming inputs and outputs
  vecXN.setLength(nSamples_*nInputs_);
  initInputScaling(XIn, vecXN.getDVector(), 1);
  vecYN.setLength(nSamples_);
  initOutputScaling(YIn, vecYN.getDVector());
  //**/ call training function
  train(vecXN, vecYN);
  return 0;
}

// ************************************************************************
// interpolate on a nD grid
// ************************************************************************
int DNN::genNDGridData(double *X,double *Y,int *N,double **X2,double **Y2)
{
  int totPts, mm;
  psVector vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ create network only if training never performed
  //**/ ---------------------------------------------------------------
  initialize(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*N) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(N, X2);
  if ((*N) == 0) return 0;
  totPts = (*N);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  vecYOut.setLength(totPts);
  (*Y2) = vecYOut.takeDVector();
  (*N)  = totPts;
  for (mm = 0; mm < totPts; mm++)
    (*Y2)[mm] = evaluatePoint(&((*X2)[mm*nInputs_]));
  return 0;
}

// ************************************************************************
// interpolate on a 1D grid
// ************************************************************************
int DNN::gen1DGridData(double *X, double *Y, int ind1, double *settings, 
                       int *NN, double **XX, double **YY)
{
  int ii;
  psVector vecXT, vecXOut, vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ create network only if training never performed
  //**/ ---------------------------------------------------------------
  initialize(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*NN) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  double HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  (*NN) = nPtsPerDim_;
  vecXOut.setLength(nPtsPerDim_);
  vecYOut.setLength(nPtsPerDim_);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecXT[ii] = settings[ii]; 

  for (ii = 0; ii < nPtsPerDim_; ii++)
  {
    vecXT[ind1] = HX * ii + VecLBs_[ind1];
    (*XX)[ii] = vecXT[ind1];
    (*YY)[ii] = evaluatePoint(vecXT.getDVector());
  }
  return 0;
}

// ************************************************************************
// interpolate on a 2D grid
// ************************************************************************
int DNN::gen2DGridData(double *X, double *Y, int ind1, int ind2, 
                       double *settings, int *NN, double **XX, double **YY)
{
  int mm, nn, index;
  psVector vecHX, vecXT, vecXOut, vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ create network only if training never performed
  //**/ ---------------------------------------------------------------
  initialize(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*NN) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  (*NN) = totPts;
  vecXOut.setLength(totPts * 2);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
	    
  for (mm = 0; mm < nPtsPerDim_; mm++)
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      index = mm * nPtsPerDim_ + nn;
      vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
      vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
      (*XX)[index*2]   = vecXT[ind1];
      (*XX)[index*2+1] = vecXT[ind2];
      (*YY)[index] = evaluatePoint(vecXT.getDVector());
    }
  }
  return 0;
}

// ************************************************************************
// interpolate on a 3D grid
// ************************************************************************
int DNN::gen3DGridData(double *X, double *Y, int ind1, int ind2, int ind3,
                      double *settings, int *NN, double **XX, double **YY)
{
  int mm, nn, pp, index;
  psVector vecHX, vecXT, vecXOut, vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ create network only if training never performed
  //**/ ---------------------------------------------------------------
  initialize(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*NN) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  (*NN) = totPts;
  vecXOut.setLength(totPts * 3);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
	    
  for (mm = 0; mm < nPtsPerDim_; mm++)
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        index = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
        vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
        vecXT[ind3] = vecHX[2] * pp + VecLBs_[ind3];
        (*XX)[index*3]   = vecXT[ind1];
        (*XX)[index*3+1] = vecXT[ind2];
        (*XX)[index*3+2] = vecXT[ind3];
        (*YY)[index] = evaluatePoint(vecXT.getDVector());
      }
    }
  }
  return 0;
}

// ************************************************************************
// interpolate on a 4D grid
// ************************************************************************
int DNN::gen4DGridData(double *X, double *Y, int ind1, int ind2, int ind3,
                       int ind4, double *settings, int *NN, double **XX, 
                       double **YY)
{
  int mm, nn, pp, qq, index;
  psVector vecHX, vecXT, vecXOut, vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ create network only if training never performed
  //**/ ---------------------------------------------------------------
  initialize(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*NN) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  (*NN) = totPts;
  vecXOut.setLength(totPts * 4);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
	    
  for (mm = 0; mm < nPtsPerDim_; mm++)
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        for (qq = 0; qq < nPtsPerDim_; qq++)
        {
          index = mm*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ + 
                  nn*nPtsPerDim_*nPtsPerDim_ + pp * nPtsPerDim_ + qq;
          vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
          vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
          vecXT[ind3] = vecHX[2] * pp + VecLBs_[ind3];
          vecXT[ind4] = vecHX[3] * qq + VecLBs_[ind4];
          (*XX)[index*4]   = vecXT[ind1];
          (*XX)[index*4+1] = vecXT[ind2];
          (*XX)[index*4+2] = vecXT[ind3];
          (*XX)[index*4+3] = vecXT[ind4];
          (*YY)[index] = evaluatePoint(vecXT.getDVector());
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// evaluate at a given point  
// ************************************************************************
double DNN::evaluatePoint(double *X)
{
  psVector vecX;
  vecX.setLength(nInputs_);
  for (int ii = 0; ii < nInputs_; ii++)
    vecX[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  propagateForward(vecX);
  double *arrayY = MatAs_[nLevels_]->getMatrix1D();
  arrayY[0] = arrayY[0] * YStd_ + YMean_;
  return arrayY[0];
}

// ************************************************************************
// evaluate at given set of points 
// ************************************************************************
double DNN::evaluatePoint(int npts, double *X, double *Y)
{
  int ii, kk;
  psVector vecX;
  vecX.setLength(npts*nInputs_);
  for (kk = 0; kk < npts; kk++)
    for (ii = 0; ii < nInputs_; ii++)
      vecX[kk*nInputs_+ii] = 
           (X[kk*nInputs_+ii] - VecXMeans_[ii]) / VecXStds_[ii];
  propagateForward(vecX);
  double *Ys = MatAs_[nLevels_]->getMatrix1D();
  for (ii = 0; ii < npts; ii++) Y[ii] = Ys[ii] * YStd_ + YMean_;
  return 0.0;
}

// ************************************************************************
// evaluate at new point with error  
// ************************************************************************
double DNN::evaluatePointFuzzy(double *X, double &std)
{
  double Y = evaluatePoint(X);
  std = 0.0;
  return Y;
}

//*************************************************************************
// evaluate at new set of points with errors 
//*************************************************************************
double DNN::evaluatePointFuzzy(int npts,double *X, double *Y, double *Ystd)
{
  evaluatePoint(npts, X, Y);
  if (Ystd != NULL)
    for (int ii = 0; ii < npts; ii++) Ystd[ii] = 0;
  return 0.0;
}

//*************************************************************************
// train NN  
//*************************************************************************
int DNN::train(psVector vecXIn, psVector vecYIn)
{
  if (optScheme_ == 1) return optimizeGradDescent(vecXIn,vecYIn);
  else                 return optimizeLBFGS(vecXIn,vecYIn);
}

//*************************************************************************
// run gradient descent algorithm   
//*************************************************************************
int DNN::optimizeGradDescent(psVector vecXIn, psVector vecYIn)
{
  int    ss, ll, iter=0, nSamp;
  double cost, *YT, dZero=0, dOne=1;
  char   matOp[100];
  psMatrix matdW;

  //**/ ----------------------------------------------------------------
  //**/ initialize weights randomly N(0, sigma^2)
  //**/ W[1] has dimension nNodes * nInputs
  //**/ B[1] has dimension nNodes * 1
  //**/ W[2] has dimension nNodes * nNodes
  //**/ B[2] has dimension nNodes * 1
  //**/ ...
  //**/ W[nLevels] has dimension 1 * nNodes
  //**/ B[nLevels] has dimension 1
  //**/ ----------------------------------------------------------------
  nSamp = vecXIn.length() / nInputs_;
  MatWs_[1]->setDim(VecNumNodes_[1], nInputs_);
  VecBs_[1]->setLength(VecNumNodes_[1]);
  for (ll = 1; ll <= nLevels_; ll++)
  {
    MatWs_[ll]->genRandom(VecNumNodes_[ll],VecNumNodes_[ll-1],dZero,dOne);
    VecBs_[ll]->genRandom(VecNumNodes_[ll],dZero, dOne);
  } 
  strcpy(matOp, "add");

  //**/ ----------------------------------------------------------------
  //**/ loop until convergence
  //**/ ----------------------------------------------------------------
  double maxW = -PSUADE_UNDEFINED, minW = PSUADE_UNDEFINED;
  double maxB = -PSUADE_UNDEFINED, minB = PSUADE_UNDEFINED;
  while (iter < maxIter_)
  {
    iter++;

    //**/ propagate from input layer to output layer
    propagateForward(vecXIn);

    //**/ after forward propagation, the prediction is at
    //**/ MatAs_[nLevels_] = YT, so compute the cost function
    YT = MatAs_[nLevels_]->getMatrix1D();
    cost = 0.0;
    for (ss = 0; ss < nSamp; ss++)
    {
      cost += pow(YT[ss] - vecYIn[ss], 2.0);
      //printf("%4d = %16.8e     %16.8e\n",ss+1,YT[ss],vecYIn[ss]);
    }
    cost *= 0.5;
    if (outputLevel_ > 0 && (iter % 100 == 0))
      printf("Iteration = %5d : cost = %e\n",iter,cost);

    //**/ propagate backward the gradients with respect to the 
    //**/ parameters 
    propagateBackward(vecYIn);

    //**/ update weight 
    for (ll = 1; ll <= nLevels_; ll++)
    {
      matdW = *(MatdWs_[ll]);
      matdW.scale(-alpha_);
      MatWs_[ll]->elementOp(matOp, matdW); 
      VecBs_[ll]->axpy(-alpha_, *(VecdBs_[ll])); 
    }
    for (ll = 1; ll <= nLevels_; ll++)
    {
      for (int jj1 = 0; jj1 < MatdWs_[ll]->nrows(); jj1++)
      {
        for (int jj2 = 0; jj2 < MatdWs_[ll]->ncols(); jj2++)
        {
          if (MatWs_[ll]->getEntry(jj1,jj2) > maxW)
            maxW = MatWs_[ll]->getEntry(jj1, jj2);
          if (MatWs_[ll]->getEntry(jj1,jj2) < minW)
            minW = MatWs_[ll]->getEntry(jj1, jj2);
        }
      }
      for (int jj3 = 0; jj3 < VecBs_[ll]->length(); jj3++)
      {
        if ((*VecBs_[ll])[jj3] > maxB)
          maxB = (*VecBs_[ll])[jj3];
        if ((*VecBs_[ll])[jj3] < minB)
          minB = (*VecBs_[ll])[jj3];
      }
    }
  }
  printf("W min and max = %e %e\n", minW, maxW);
  printf("B min and max = %e %e\n", minB, maxB);

  //**/ ----------------------------------------------------------------
  //**/Notes: regularization methods:
  //**/(1) prevent overfitting: at each iteration drop 50% of nodes
  //**/(2) early stopping (at some iteration, testing on test set until
  //**/    the loss for test data set bottoms out
  //**/ (a/b)eta ~ adaptive (momentum, Adagrad, Adadelta, Adam, RMSProp)
  //**/For noisy data: use mini-batch (use average gradient)
  //**/ minibatch size 10-100
  //**/ initialize weights randomly N(0, sigma^2)
  //**/ ----------------------------------------------------------------
  return 0;
}

//*************************************************************************
// run LBFGS optimization
//*************************************************************************
int DNN::optimizeLBFGS(psVector vecXIn, psVector vecYIn)
{
#ifdef HAVE_LBFGS
  int     ii, jj, kk, ll, nn, ss, length, its, nLHS=3, minIndex;
  integer nInps, iprint=0, itask, *task=&itask, lsave[4], isave[44];
  integer *iwork, nCorr=5, *nbds, csave[60], maxIter=2000;
  double  factr, pgtol, dsave[29], normGrad, FValue, *YT, dZero=0, dOne=1;
  double  lb, ub, ddata;
#endif

#ifndef HAVE_LBFGS
  printf("DNN ERROR: DNN is not available because LBFGS is not installed.\n");
  exit(1);
#endif

  //**/ ----------------------------------------------------------------
  //**/ count the number of parameters, generate initial guess
  //**/ ----------------------------------------------------------------
#ifdef HAVE_LBFGS
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0) 
    printf("NN LBFGS optimization begins ...\n");
  MatWs_[1]->setDim(VecNumNodes_[1], nInputs_);
  VecBs_[1]->setLength(VecNumNodes_[1]);
  for (ll = 1; ll <= nLevels_; ll++)
  {
    ddata = sqrt(2.0/VecNumNodes_[ll-1]);
    lb = - ddata;
    ub =   ddata;
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      if (ll < nLevels_)
        printf("NN hidden layer %d weights bounds = [%12.4e %12.4e]\n",
               ll,lb,ub);
      else
        printf("NN output layer %d weights bounds = [%12.4e %12.4e]\n",
               ll,lb,ub);
    }
    MatWs_[ll]->genRandom(VecNumNodes_[ll],VecNumNodes_[ll-1],lb,ub);
    VecBs_[ll]->genRandom(VecNumNodes_[ll], -dOne, dOne);
  } 
  length = 0;
  for (ii = 1; ii <= nLevels_; ii++)
  {
    length += MatWs_[ii]->nrows() * MatWs_[ii]->ncols();
    length += VecBs_[ii]->length();
  }
  nInps = length;

  //**/ ----------------------------------------------------------------
  //**/ create a LHS sample of parameters
  //**/ the bounds are presribed using He's method
  //**/ ----------------------------------------------------------------
  psVector  vecLBs, vecUBs, vecXS, vecYS;
  psIVector vecIS;
  //**/ prevent LHSampling to spit out messages
  //printf("NN: use LHS to create multiple initial weights.\n");
  Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  vecLBs.setLength(nInps);
  vecUBs.setLength(nInps);
  length = 0;
  for (ii = 1; ii <= nLevels_; ii++)
  {
    kk = MatWs_[ii]->nrows() * MatWs_[ii]->ncols();
    ddata = sqrt(2.0 / MatWs_[ii]->ncols());
    lb = - ddata;
    ub =   ddata;
    for (jj = length; jj < length+kk; jj++) 
    {
      vecLBs[jj] = lb; 
      vecUBs[jj] = ub; 
    }
    length += kk;
    kk = VecBs_[ii]->length();
    lb = - dOne;
    ub =   dOne;
    for (jj = length; jj < length+kk; jj++) 
    {
      vecLBs[jj] = lb; 
      vecUBs[jj] = ub; 
    }
    length += kk;
  }
  sampler->setInputBounds(nInps,vecLBs.getDVector(),vecUBs.getDVector());
  sampler->setOutputParams(1);
  sampler->setSamplingParams(nLHS, 1, 0);
  sampler->initialize(0);
  vecIS.setLength(nLHS);
  vecXS.setLength(nLHS*nInps);
  vecYS.setLength(nLHS);
  sampler->getSamples(nLHS,(int)nInps,1,vecXS.getDVector(),
                      vecYS.getDVector(), vecIS.getIVector());
  delete sampler;

  //**/ ----------------------------------------------------------------
  //**/ set up for optimization
  //**/ ----------------------------------------------------------------
  double   minFval=PSUADE_UNDEFINED;
  psVector vecPVals, vecGrads, vecBestP, vecW;
  vecPVals.setLength(nInps);
  vecGrads.setLength(nInps);

  //**/ ----------------------------------------------------------------
  //**/ 1e12 for low accuracy, 1e7 for moderate, 1e1 for high
  //**/ ----------------------------------------------------------------
  factr = 1e1;
  pgtol = 1e-8;
  kk = (2 * nCorr + 5) * nInps + 11 * nCorr * nCorr + 8 * nCorr;
  vecW.setLength(kk);
  iwork = new integer[3*nInps];
  nbds = new integer[nInps];
  for (ii = 0; ii < nInps; ii++) nbds[ii] = 2;

  //**/ ----------------------------------------------------------------
  //**/ perform nLHS optimizations
  //**/ ----------------------------------------------------------------
  for (nn = 1; nn < nLHS; nn++)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
      printf("LBFGS #%d begins ... (variable size = %ld)\n",nn,nInps);
    //**/ load initial guess (from vecXS)
    //**/ this segment of code is actually not needed
    length = 0;
    for (ii = 1; ii <= nLevels_; ii++)
    {
      for (jj = 0; jj < MatWs_[ii]->nrows(); jj++)
      {
        for (kk = 0; kk < MatWs_[ii]->ncols(); kk++)
        {
          MatWs_[ii]->setEntry(jj,kk,vecXS[nInps*nn+length]); 
          length++;
        }
      }
      for (jj = 0; jj < VecBs_[ii]->length(); jj++)
      {
        (*VecBs_[ii])[jj] = vecXS[nInps*nn+length]; 
        length++;
      }
    }
    for (ii = 0; ii < nInps; ii++) vecPVals[ii] = vecXS[nInps*nn+ii];
 
    //**/ --------------------------------------------------------------
    //**/ perform one optimization
    //**/ --------------------------------------------------------------
    its   = 0;
    *task = (integer) START;
    while (its < maxIter)
    {
      its++;
      //**/ calll LBFGS to advance parameters
      setulb(&nInps,&nCorr,vecPVals.getDVector(),vecLBs.getDVector(),
             vecUBs.getDVector(),nbds,&FValue,vecGrads.getDVector(),
             &factr,&pgtol,vecW.getDVector(),iwork,task,&iprint,
             csave,lsave,isave,dsave);

      //**/ depending on the code returned from LBFGS, act accordingly
      if (isave[33] >= maxIter)
      {
        *task = STOP_ITER;
        if (psConfig_.InteractiveIsOn()) 
        {
          printf("INFO: PSUADE issues a stop (max iterations reached)\n");
          printf("INFO: current best FValue = %e\n", FValue);
        }
      }
      else if (IS_FG(*task))
      {
        //**/ request for evaluation given parameter values, so
        //**/ (1) load parameters into local parameter stores
        length = 0;
        for (ii = 1; ii <= nLevels_; ii++)
        {
          for (jj = 0; jj < MatWs_[ii]->nrows(); jj++)
            for (kk = 0; kk < MatWs_[ii]->ncols(); kk++)
              MatWs_[ii]->setEntry(jj, kk, vecPVals[length++]); 
          for (jj = 0; jj < VecBs_[ii]->length(); jj++)
            (*VecBs_[ii])[jj] = vecPVals[length++]; 
        }

        //**/ (2)propagate forward to get function value
        propagateForward(vecXIn);
        YT = MatAs_[nLevels_]->getMatrix1D();
        FValue = 0.0;
        for (ss = 0; ss < nSamples_; ss++)
          FValue += pow(YT[ss] - vecYIn[ss],2.0);

        //**/ (3) compute gradients
        propagateBackward(vecYIn);
        length = 0;
        for (ii = 1; ii <= nLevels_; ii++)
        {
          for (jj = 0; jj < MatWs_[ii]->nrows(); jj++)
            for (kk = 0; kk < MatWs_[ii]->ncols(); kk++)
              vecGrads[length++] = MatdWs_[ii]->getEntry(jj,kk);
          for (jj = 0; jj < VecBs_[ii]->length(); jj++)
            vecGrads[length++] = (*VecdBs_[ii])[jj];
        }

        //**/ convergence analysis
        normGrad = 0.0;
        for (kk = 0; kk < nInps; kk++)
          normGrad += pow(vecGrads[kk], 2.0);
        normGrad = sqrt(normGrad);
        if (psConfig_.InteractiveIsOn() && outputLevel_ > 2 && (its % 100 == 0))
          printf("Iteration %5d: FValue = %16.8e (|grad| = %e)\n",its,
                 FValue, normGrad);
      }
      else if (*task == NEW_X)
      {
        if (isave[33] >= maxIter)
        {
          *task = STOP_ITER;
          if (psConfig_.InteractiveIsOn())
          {
            printf("INFO: PSUADE issues a stop (max iterations reached)\n");
            printf("INFO: current best FValue = %e\n", FValue);
          }
        }
      }
      else
      { 
        //printf("INFO: LBFGS issues a stop (its = %d, code = %s)\n",its,
        //       (char *) task);
        if (psConfig_.InteractiveIsOn())
          printf("INFO: LBFGS issues a stop (its = %d, code = %d)\n",its,
                 (int) itask);
        break;
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
      printf("LBFGS #%d: best fval = %e\n",nn,FValue);
    if (FValue < minFval)
    {
      minFval = FValue;
      minIndex = nn;
      vecBestP = vecPVals;
    }
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
    printf("NN LBFGS ends - iteration %d has minimum fvalue\n",minIndex);

  //**/ ----------------------------------------------------------------
  //**/ when all is done, load the best parameters 
  //**/ ----------------------------------------------------------------
  length = 0;
  for (ii = 1; ii <= nLevels_; ii++)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      printf("W's for hidden layer %d\n",ii);
    for (jj = 0; jj < MatWs_[ii]->nrows(); jj++)
    {
      for (kk = 0; kk < MatWs_[ii]->ncols(); kk++)
      {
        if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
          printf("W %6d = %12.4e\n",jj*MatWs_[ii]->ncols()+kk+1,
                 vecBestP[length]);
        MatWs_[ii]->setEntry(jj, kk, vecBestP[length++]); 
      }
    }
    for (jj = 0; jj < VecBs_[ii]->length(); jj++)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
        printf("B %6d = %12.4e\n", jj+1, vecBestP[length]);
      (*VecBs_[ii])[jj] = vecBestP[length++]; 
    }
  }
#endif
  return 0;
}

//*************************************************************************
// propagate input vector through the network 
//*************************************************************************
int DNN::propagateForward(psVector vecXIn)
{
  int  ll, nSamp;
  char activationStr[100];

  //**/ ----------------------------------------------------------------
  //**/ load incoming inputs into the input layer
  //**/ ----------------------------------------------------------------
  nSamp = vecXIn.length() / nInputs_;
  MatAs_[0]->load(nInputs_, nSamp, vecXIn.getDVector());
  
  //**/ ----------------------------------------------------------------
  //**/ traverse all levels
  //**/ ----------------------------------------------------------------
  for (ll = 1; ll <= nLevels_; ll++)
  {
    //**/ --------------------------------------------------------------
    //**/ Z[1] = W[1] * A[0] + B[1]
    //**/ W[1] has dimension nNodes * nInputs
    //**/ B[1] has dimension nNodes * 1
    //**/ A[0] has dimension nInputs * nSamples
    //**/ Z[1] has dimension nNodes * nSamples
    //**/ Z[2] = W[2] * A[1] + B[2]
    //**/ W[2] has dimension nNodes * nNodes
    //**/ B[2] has dimension nNodes * 1
    //**/ A[1] has dimension nNodes * nSamples
    //**/ Z[2] has dimension nNodes * nSamples
    //**/ ...
    //**/ Z[nLevels] = W[nLevels] * A[nLevels-1] + B[nLevels]
    //**/ W[nLevels] has dimension 1 * nNodes
    //**/ B[nLevels] has dimension 1
    //**/ A[nLevels-1] has dimension nNodes * nSamples
    //**/ Z[nLevels] has dimension 1 * nSamples
    //**/ A[nLevels] has dimension 1 * nSamples
    //**/ --------------------------------------------------------------
    MatWs_[ll]->matmult(*(MatAs_[ll-1]), *(MatZs_[ll]));
    MatZs_[ll]->addVec2AllCols(*(VecBs_[ll]));

    //**/ --------------------------------------------------------------
    //**/ A[1] = g(Z[1])
    //**/ A[2] = g(Z[2])
    //**/ ...
    //**/ A[nLevels] = Z[nLevels]
    //**/ --------------------------------------------------------------
    if (VecActivationFcns_[ll] == 0)
       strcpy(activationStr, "sigmoid");
    else if (VecActivationFcns_[ll] == 1)
       strcpy(activationStr, "tanh");
    else if (VecActivationFcns_[ll] == 2)
       strcpy(activationStr, "relu");
    *(MatAs_[ll]) = *(MatZs_[ll]); 
    if (ll < nLevels_) MatAs_[ll]->elementOp(activationStr);
  }
  return 0;
}

//*************************************************************************
// propagate backward derivatives
//*************************************************************************
int DNN::propagateBackward(psVector vecYIn) 
{
  int    ii, ll, iZero=0, nSamp;
  double ddata;
  char   activationStr[100], elementMultiply[100];
  psMatrix matAT, matZ, matGZ, matWT;

  //**/ ----------------------------------------------------------------
  //**/ at level L=nLevels_ (output level), activation g(A[L]) = A[L]
  //**/ loss function is J(A) = 1/2 (A[L] - YIn)^2
  //**/ dJ(A)/dA[L] = (A[L] - YIn)
  //**/ dJ(L)/dZ[L] = dJ(L)/dA[L] * dA[L]/dZ[L]
  //**/    = (A[L] - YIn) * 1  (since g'(A[L]) = 1 
  //**/ ----------------------------------------------------------------
  nSamp = vecYIn.length();
  *MatdZs_[nLevels_] = *MatAs_[nLevels_];
  double *dZL = MatdZs_[nLevels_]->getMatrix1D();
  for (ii = 0; ii < nSamp; ii++) dZL[ii] -= vecYIn[ii];
  strcpy(elementMultiply, "multiply");

  for (ll = nLevels_; ll > 0; ll--)
  {
    //**/ --------------------------------------------------------------
    //**/ dZ[ll] = W[ll+1]^T * dZ[ll+1] * g'[ll](Z[ll]) (* - element product)
    //**/ W[ll]  has dimension nNodes[ll] * nNodes[ll-1]
    //**/ dZ[ll] has dimension nNodes[ll] * nSamples
    //**/ --------------------------------------------------------------
    if (ll < nLevels_)
    {
      matWT = *(MatWs_[ll+1]);
      matWT.transpose();
      matWT.matmult(*(MatdZs_[ll+1]), *(MatdZs_[ll]));
      if (VecActivationFcns_[ll] == 0)
        strcpy(activationStr, "sigmoid_prime");
      else if (VecActivationFcns_[ll] == 1)
        strcpy(activationStr, "tanh_prime");
      else if (VecActivationFcns_[ll] == 2)
        strcpy(activationStr, "relu_prime");
      else
      {
        printf("NN propagateBackward ERROR: wrong activation code %d\n",
               VecActivationFcns_[ll]);
        exit(1);
      }
      matGZ = *(MatZs_[ll]);
      matGZ.elementOp(activationStr);
      MatdZs_[ll]->elementOp(elementMultiply, matGZ);
    }

    //**/ --------------------------------------------------------------
    //**/ dW[ll] = dZ[ll] * A[ll-1]^T / nSamples
    //**/ dZ[ll]  has dimension nNodes[ll] * nSamples
    //**/ A[ll-1] has dimension nNodes[ll-1] * nSamples
    //**/ dW[ll]  has dimension nNodes[ll] * nNodes[ll-1]
    //**/ --------------------------------------------------------------
    matAT = *(MatAs_[ll-1]);
    matAT.transpose();
    MatdZs_[ll]->matmult(matAT, *(MatdWs_[ll]));
    ddata = 1.0 / (double) nSamp;
    MatdWs_[ll]->scale(ddata);

    //**/ --------------------------------------------------------------
    //**/ dB[ll] = np.sum(dZ[ll],axis=1) / nSamples
    //**/ dB[ll] has dimension nNodes[ll]
    //**/ --------------------------------------------------------------
    MatdZs_[ll]->rowsum(*(VecdBs_[ll]));
    VecdBs_[ll]->scale(ddata);
  }
  return 0;
}

//*************************************************************************
// print diagnostics details 
//*************************************************************************
void DNN::showDetails()
{
  int ll;
  for (ll = 1; ll < nLevels_; ll++)
  {
    printf("NN Hidden Layer %d: \n", ll);
    printf("  Activation function = ");
    if      (VecActivationFcns_[ll] == 0) printf("Sigmoid\n");
    else if (VecActivationFcns_[ll] == 1) printf("Tanh\n");
    else if (VecActivationFcns_[ll] == 2) printf("ReLU\n");
    printf("  Number of nodes     = %d\n", VecNumNodes_[ll]);
  }
}

