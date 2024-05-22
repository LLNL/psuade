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
// Functions for the class FuncApprox  
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "Psuade.h"
#include "FuncApprox.h"
#include "Mars.h"
#include "GP1.h"
#include "GP3.h"
#include "MGP3.h"
#include "MDNN.h"
#include "TBGP.h"
#include "MTGP.h"
#include "SVM.h"
#include "SelectiveRegression.h"
#include "UserRegression.h"
#include "LegendreRegression.h"
#include "Regression.h"
#include "DNN.h"
#include "PWLinear.h"
#include "MarsBagg.h"
#include "SumOfTrees.h"
#include "SGRegression.h"
#include "GradLegendreRegression.h"
#include "Kriging.h"
#include "Splines.h"
#include "KNN.h"
#include "RBF.h"
#include "RBFBagg.h"
#include "MRBF.h"
#include "Acosso.h"
#include "BSSAnova.h"
#include "PsuadeRegression.h"
#include "HomLegendreRegression.h"
#include "HGP3.h"
#include "HKriging.h"
#include "PLS.h"
#include "MMars.h"
#include "HybridGP.h"
#include "QGP.h"
#include "PadeRegression.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PDFBase.h"
#include "pData.h"
#include "PsuadeData.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
FuncApprox::FuncApprox()
{
  outputLevel_ = 0;
  nPtsPerDim_  = 10;
  YMean_ = 0.0;
  YStd_ = 1.0;
}

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
FuncApprox::FuncApprox(int nInputs, int nSamples)
{
  if (nSamples <= 0 || nInputs <= 0) 
  {
    printf("FuncApprox::FuncApprox ERROR - invalid inputs. \n");
    printf("            nSamples = %d\n", nSamples);
    printf("            nInputs  = %d\n", nInputs);
    exit(-1);
  }
  outputLevel_ = 0;
  nSamples_    = nSamples;
  nInputs_     = nInputs;
  nPtsPerDim_  = 10;
  VecLBs_.setLength(nInputs_);
  VecUBs_.setLength(nInputs_);
  VecWghts_.setLength(nSamples_);
  for (int jj = 0 ; jj < nSamples_; jj++) VecWghts_[jj] = 1.0;
  VecXMeans_.setLength(nInputs_);
  VecXStds_.setLength(nInputs_);
  for (int ii = 0; ii < nInputs_; ii++) VecXStds_[ii] = 1.0;
  YMean_ = 0.0;
  YStd_ = 1.0;
}

// ************************************************************************
// Copy constructor by Bill Oliver 
// ------------------------------------------------------------------------
FuncApprox::FuncApprox(const FuncApprox & fa)
{
  outputLevel_ = fa.outputLevel_;
  nSamples_ = fa.nSamples_;
  nInputs_ = fa.nInputs_;
  nPtsPerDim_ = fa.nPtsPerDim_;
  faID_ = fa.faID_;
  VecLBs_ = fa.VecLBs_;
  VecUBs_ = fa.VecUBs_;
  VecWghts_ = fa.VecWghts_;
  VecXMeans_ = fa.VecXMeans_;
  VecXStds_ = fa.VecXStds_;
  YMean_ = fa.YMean_;
  YStd_ = fa.YStd_;
}

// ************************************************************************
// operator= by Bill Oliver 
// ------------------------------------------------------------------------
FuncApprox & FuncApprox::operator=(const FuncApprox & fa)
{
  if (this == &fa) return *this;
  
  outputLevel_ = fa.outputLevel_;
  nSamples_ = fa.nSamples_;
  nInputs_ = fa.nInputs_;
  nPtsPerDim_ = fa.nPtsPerDim_;
  faID_ = fa.faID_;
  VecLBs_ = fa.VecLBs_;
  VecUBs_ = fa.VecUBs_;
  VecWghts_ = fa.VecWghts_;
  VecXMeans_ = fa.VecXMeans_;
  VecXStds_ = fa.VecXStds_;
  YMean_ = fa.YMean_;
  YStd_ = fa.YStd_;
  return *this;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
FuncApprox::~FuncApprox()
{
}

// ************************************************************************
// get function approximator identifier
// ------------------------------------------------------------------------
int FuncApprox::getID()
{
  return faID_;
}

// ************************************************************************
// Set print level
// ------------------------------------------------------------------------
int FuncApprox::setOutputLevel(int level)
{
  if (level >= -1) outputLevel_ = level;
  return 0;
}

// ************************************************************************
// Set input bounds for object class FuncApprox
// ------------------------------------------------------------------------
int FuncApprox::setBounds(double *lower, double *upper)
{
  for (int ii = 0 ; ii < nInputs_; ii++) 
  {
    VecLBs_[ii] = lower[ii];
    VecUBs_[ii] = upper[ii];
  }
  return 0;
}

// ************************************************************************
// Reset sample size 
// ------------------------------------------------------------------------
int FuncApprox::setSampleSize(int newSize)
{
  nSamples_ = newSize;
  VecWghts_.setLength(nSamples_);
  for (int jj = 0 ; jj < nSamples_; jj++) VecWghts_[jj] = 1.0;
  return 0;
}

// ************************************************************************
// load output weights
// ------------------------------------------------------------------------
int FuncApprox::loadWeights(int n, double *wgts)
{
  if (n != nSamples_)
  {
    printf("FuncApprox::loadWeights ERROR : invalid length %d.\n",n);
    exit(1);
  }
  VecWghts_.clean();
  for (int ii = 0 ; ii < n; ii++) 
  {
    if (wgts[ii] < 0.0)
    {
      printf("FuncApprox::loadWeights WARNING : weight < 0 - not used.\n");
      return 0;
    }
  }
  VecWghts_.setLength(nSamples_);
  for (int jj = 0 ; jj < n; jj++) VecWghts_[jj] = wgts[jj];
  return 0;
}

// ************************************************************************
// set number of points to generate in each dimension
// ------------------------------------------------------------------------
void FuncApprox::setNPtsPerDim(int npoints)
{
  if (npoints > 0) nPtsPerDim_ = npoints;
}

// ************************************************************************
// get number of points to generate in each dimension
// ------------------------------------------------------------------------
int FuncApprox::getNPtsPerDim()
{
  return nPtsPerDim_;
}

// ************************************************************************
// generate N dimensional data
// ------------------------------------------------------------------------
int FuncApprox::genNDGridData(double*,double*,int*,double**,double**) 
{
  return -1;
}

// ************************************************************************
// generate 1-dimensional data
// ------------------------------------------------------------------------
int FuncApprox::gen1DGridData(double*,double*,int,double*,int*,double**,
                              double**)
{
  return -1;
}

// ************************************************************************
// generate 2 dimensional surface data
// ------------------------------------------------------------------------
int FuncApprox::gen2DGridData(double*,double*,int,int,double*,int*,
                              double**,double**) 
{
  return -1;
}

// ************************************************************************
// generate 3 dimensional surface data
// ------------------------------------------------------------------------
int FuncApprox::gen3DGridData(double*,double*,int,int,int,double*,int*, 
                              double**,double**) 
{
  return -1;
}

// ************************************************************************
// generate 4 dimensional surface data
// ------------------------------------------------------------------------
int FuncApprox::gen4DGridData(double*,double*,int,int,int,int,double*, 
                              int*,double**,double**) 
{
  return -1;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double FuncApprox::evaluatePointFuzzy(double *, double &std)
{
  std = 0.0;
  return 0.0;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double FuncApprox::evaluatePointFuzzy(int npts, double *, double *Y,
                                      double *Ystd)
{
  for (int ii = 0; ii < npts; ii++)
  {
    Y[ii]  = 0.0;
    Ystd[ii]  = 0.0;
  }
  return 0.0;
}

// ************************************************************************
// Set parameters
// ------------------------------------------------------------------------
double FuncApprox::setParams(int, char **)
{
  return -1.0;
}

// ************************************************************************
// initialize scaling for inputs
// ------------------------------------------------------------------------
int FuncApprox::initInputScaling(double *XIn, double *XOut, int flag)
{
  int    ii, jj, kk;
  double ddata;
  char   pString[500], response[100];
                                                                                
  VecXMeans_.setLength(nInputs_);
  VecXStds_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    VecXMeans_[ii] = 0.0;
    VecXStds_[ii]  = 1.0;
  }
  for (ii = 0; ii < nInputs_*nSamples_; ii++) XOut[ii] = XIn[ii];

  response[0] = 'n';
  if (flag != 1 && psConfig_.RSExpertModeIsOn())
  {
    snprintf(pString,100,"Scale the sample matrix ? (y or n) ");
    getString(pString, response);
  }
  if (flag == 1 || response[0] == 'y')
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = 0.0;
      for (jj = 0; jj < nSamples_; jj++) ddata += XIn[jj*nInputs_+ii];
      VecXMeans_[ii] = ddata / (double) nSamples_;
      ddata = 0.0;
      for (jj = 0; jj < nSamples_; jj++)
        ddata += pow(XIn[jj*nInputs_+ii] - VecXMeans_[ii], 2.0);
      VecXStds_[ii] = sqrt(ddata / (double) (nSamples_ - 1));
      if (VecXStds_[ii] == 0.0) 
      {
        printf("FuncApprox ERROR: input %d has 0 variance.\n", ii+1);
        VecXStds_[ii] = 1.0;
        for (jj = 0; jj < nSamples_; jj++)
        {
          printf("Sample %d: \n   Inputs = ",jj+1);
          for (kk = 0; kk < nInputs_; kk++)
            printf("%12.4e ",XIn[jj*nInputs_+kk]);
          printf("\n");
        }
      }
      for (jj = 0; jj < nSamples_; jj++)
        XOut[jj*nInputs_+ii] = (XIn[jj*nInputs_+ii]-VecXMeans_[ii])/
                                VecXStds_[ii];
      if (psConfig_.InteractiveIsOn() && 
          (outputLevel_ > 3 && psConfig_.RSExpertModeIsOn()))
        printf("Input %d scaling info : mean, std = %e %e\n",ii+1,
                VecXMeans_[ii], VecXStds_[ii]);
    }
  }
  return 0;
}

// ************************************************************************
// initialize scaling for output
// ------------------------------------------------------------------------
int FuncApprox::initOutputScaling(double *YIn, double *YOut)
{
  int    ii;
  double ddata;
                                                                                
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) ddata += YIn[ii];
  YMean_ = ddata / (double) nSamples_;
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++)
     ddata += pow(YIn[ii] - YMean_, 2.0);
  YStd_ = sqrt(ddata / (double) (nSamples_ - 1));
  if (YStd_ == 0.0) YStd_ = 1.0;
  for (ii = 0; ii < nSamples_; ii++)
    YOut[ii] = (YIn[ii] - YMean_) / YStd_;
  return 0;
}

// ************************************************************************
// generate m-dimensional grid data
// ------------------------------------------------------------------------
int FuncApprox::genNDGrid(int *nPts, double **XOut) 
{
  int    ii, mm, totPts;
  psVector vecHX, vecXT, vecXOut;

  //**/ ---------------------------------------------------------------
  //**/ nInputs > 21 not supported
  //**/ ---------------------------------------------------------------
  if (nInputs_ > 21)
  {
    printf("FuncApprox genNDGrid INFO: nInputs > 21 not supported.\n");
    (*nPts) = 0;
    (*XOut) = NULL;
    return 0;
  }

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  if (nInputs_ == 21 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 20 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 19 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 18 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 17 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 16 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 15 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 14 && nPtsPerDim_ >    2) nPtsPerDim_ =  2;
  if (nInputs_ == 13 && nPtsPerDim_ >    3) nPtsPerDim_ =  3;
  if (nInputs_ == 12 && nPtsPerDim_ >    3) nPtsPerDim_ =  3;
  if (nInputs_ == 11 && nPtsPerDim_ >    4) nPtsPerDim_ =  4;
  if (nInputs_ == 10 && nPtsPerDim_ >    5) nPtsPerDim_ =  5;
  if (nInputs_ ==  9 && nPtsPerDim_ >    6) nPtsPerDim_ =  6;
  if (nInputs_ ==  8 && nPtsPerDim_ >    7) nPtsPerDim_ =  7;
  if (nInputs_ ==  7 && nPtsPerDim_ >   10) nPtsPerDim_ = 10;
  if (nInputs_ ==  6 && nPtsPerDim_ >   14) nPtsPerDim_ = 14;
  if (nInputs_ ==  5 && nPtsPerDim_ >   24) nPtsPerDim_ = 24;
  if (nInputs_ ==  4 && nPtsPerDim_ >   50) nPtsPerDim_ = 50;
  if (nInputs_ ==  3 && nPtsPerDim_ >  200) nPtsPerDim_ = 200;
  if (nInputs_ ==  2 && nPtsPerDim_ > 3000) nPtsPerDim_ = 3000;
  if (nInputs_ ==  1 && nPtsPerDim_ > 8000000) 
    nPtsPerDim_ = 8000000;
  totPts = nPtsPerDim_;
  for (ii = 1; ii < nInputs_; ii++) totPts = totPts * nPtsPerDim_;
  vecHX.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    vecHX[ii] = (VecUBs_[ii] - VecLBs_[ii])/(double) (nPtsPerDim_ - 1);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  vecXOut.setLength(nInputs_ * totPts);
  vecXT = VecLBs_;

  for (mm = 0; mm < totPts; mm++)
  {
    for (ii = 0; ii < nInputs_; ii++ ) 
      vecXOut[mm*nInputs_+ii] = vecXT[ii];
    for (ii = 0; ii < nInputs_; ii++ )
    {
      vecXT[ii] += vecHX[ii];
      if (vecXT[ii] < VecUBs_[ii] ||
          PABS(vecXT[ii] - VecUBs_[ii]) < 1.0E-7) break;
      else vecXT[ii] = VecLBs_[ii];
    }
  }
  (*XOut) = vecXOut.takeDVector();
  (*nPts) = totPts;
  return 0;
}

// ************************************************************************
// ************************************************************************
// ************************************************************************
// friend function (print current function approximator)
// ------------------------------------------------------------------------
extern "C" 
int getFAType(char *pString)
{
  int faType;

  faType = getInt(0, PSUADE_NUM_RS-1, pString);
#ifndef HAVE_MARS
  if (faType == PSUADE_RS_MARS)  faType = -1;
  if (faType == PSUADE_RS_MARSB) faType = -1;
  if (faType == PSUADE_RS_MMARS) faType = -1;
#endif
#ifndef HAVE_TPROS
  //**/ if TPROS is not available, use resident GP
  if (faType == PSUADE_RS_GP1) faType = PSUADE_RS_GP3;
#endif
#ifndef HAVE_SVM
  if (faType == PSUADE_RS_SVM) faType = -1;
#endif
#ifndef HAVE_TGP
  if (faType == PSUADE_RS_TGP) faType = -1;
#endif
  return faType;
}

// ************************************************************************
// friend function (print current function approximator)
// ------------------------------------------------------------------------
extern "C" 
void printThisFA(int faType)
{
  switch( faType )
  {
#ifdef HAVE_MARS
    case PSUADE_RS_MARS: 
         printf("MARS model\n"); 
         break;
#else
    case PSUADE_RS_MARS: 
         printf("MARS model (not installed)\n"); 
         break;
#endif
    case PSUADE_RS_REGR1: 
         printf("Linear regression model\n"); 
         break;
    case PSUADE_RS_REGR2: 
         printf("Quadratic regression model\n"); 
         break;
    case PSUADE_RS_REGR3: 
         printf("Cubic regression model\n"); 
         break;
    case PSUADE_RS_REGR4: 
         printf("Quartic regression model\n"); 
         break;
    case PSUADE_RS_ANN: 
         printf("Artificial neural network model\n"); 
         break;
    case PSUADE_RS_REGRS: 
         printf("User-defined regression model\n"); 
         break;
#ifdef HAVE_TPROS
    case PSUADE_RS_GP1: 
         printf("Gaussian Process (MacKay) model\n"); 
         break;
#else
    case PSUADE_RS_GP1: 
         printf("Gaussian Process (MacKay) model (not installed)\n");
         break;
#endif
    case PSUADE_RS_GP3: 
         printf("Gaussian Process (Tong) model\n"); 
         break;
#ifdef HAVE_SVM
    case PSUADE_RS_SVM: 
         printf("SVM-light (Joachims) model\n"); 
         break;
#else
    case PSUADE_RS_SVM: 
         printf("SVM-light (Joachims) model (not installed)\n");
         break;
#endif
    case PSUADE_RS_REGRGL: 
         printf("Derivative-based Legendre Polynomial Regression\n"); 
         break;
#ifdef HAVE_TGP
    case PSUADE_RS_TGP: 
         printf("Tree-based Gaussian Process\n"); 
         break;
#else
    case PSUADE_RS_TGP: 
         printf("Tree-based Gaussian Process (not installed)\n");
         break;
#endif
#ifdef HAVE_MARS
    case PSUADE_RS_MARSB: 
         printf("MARS with bagging model\n");
         break;
#else
    case PSUADE_RS_MARSB: 
         printf("MARS with bagging model (not installed)\n");
         break;
#endif
    case PSUADE_RS_SOTS: 
         printf("Sum-of-trees model\n"); 
         break;
    case PSUADE_RS_REGRL: 
         printf("Legendre Polynomial Regression\n"); 
         break;
    case PSUADE_RS_REGRU: 
         printf("User-defined (nonpolynomial) Regression\n"); 
         break;
    case PSUADE_RS_REGSG: 
         printf("Sparse Grid Polynomial Regression\n"); 
         break;
    case PSUADE_RS_KR: 
         printf("Kriging\n"); 
         break;
    case PSUADE_RS_SPLINES: 
         printf("Splines on regular grid (1D, 2D, or 3D only)\n");
         break;
    case PSUADE_RS_KNN: 
         printf("K-nearest neighbor\n"); 
         break;
    case PSUADE_RS_RBF: 
         printf("Radial Basis Function\n"); 
         break;
    case PSUADE_RS_RBFB: 
         printf("Radial Basis Function with bagging\n"); 
         break;
    case PSUADE_RS_MRBF: 
         printf("Multi-Radial Basis Function\n"); 
         break;
    case PSUADE_RS_MGP3: 
         printf("Multi-Gaussian Process (Tong)\n"); 
         break;
    case PSUADE_RS_MDNN: 
         printf("Multi-Domain Neural Network (Tong)\n"); 
         break;
    case PSUADE_RS_PADE: 
         printf("Pade Regression\n"); 
         break;
    case PSUADE_RS_HLEG: 
         printf("Homogeneous Legendre Regression\n"); 
         break;
    case PSUADE_RS_HGP3: 
         printf("Homogeneous Gaussian Process\n"); 
         break;
    case PSUADE_RS_HYGP: 
         printf("Hybrid Homogeneous Gaussian Process\n"); 
         break;
    case PSUADE_RS_QGP: 
         printf("Quantile Homogeneous Gaussian Process\n"); 
#ifdef HAVE_MARS
    case PSUADE_RS_MMARS: 
         printf("Multiple MARS model\n");
         break;
#else
    case PSUADE_RS_MMARS: 
         printf("Multiple MARS model (MARS not installed)\n");
         break;
#endif
#ifdef HAVE_TGP
    case PSUADE_RS_MTGP: 
         printf("Multiple Tree-based Gaussian Process\n"); 
         break;
#else
    case PSUADE_RS_MTGP: 
         printf("Multiple Tree-based Gaussian Process (not installed)\n");
         break;
#endif
  }
}

// ************************************************************************
// friend function (print function approximator information)
// ------------------------------------------------------------------------
extern "C" 
int writeFAInfo(int level)
{
  printDashes(PL_INFO, 0);
  printf("Available response surface tools: \n");
  printDashes(PL_INFO, 0);
  if (level > 3)
  {
   printf("Expert advices: \n");
#ifdef HAVE_MARS
   printf(" MARS - may have accuracy problem near domain boundary. Use\n");
   printf("   this option if sample size is sufficiently large (>100).\n");
#endif
   printf(" LINEAR, QUADRATIC, CUBIC, QUARTIC - good for small sample\n");
   printf("   sizes; and when the function is sufficiently smooth. For\n");
   printf("   higher than fourth order, use LEGENDRE (option 15) with\n");
   printf("   response surface expert mode turned on to select order.\n");
   printf(" ANN - artificial neural network.\n");
   printf(" SELECTIVE POLYNOMIAL REGRESSION - if you desire to include\n");
   printf("   some higher order regression terms (provided you know\n");
   printf("   which ones.)\n");

   printf(" GAUSSIAN PROCESS - may  encounter non-definite covariance\n");
   printf("   matrix problem. If no such problem occurs, it is very\n");
   printf("   good for small samples (a few to a few tens). GP is\n");
   printf("   relatively slow, so for sample sizes of more than a few\n");
   printf("   hundred, be patient. For nonsmooth functions, try TGP.\n");
#ifdef HAVE_SVM
   printf(" SVM - provides 3 options: turn on rs_expert to select.\n");
   printf("   Also, use svmfind to search for best settings.\n");
#endif
   printf(" BOOTSTRAPPED MARS - more expensive than MARS but provide\n");
   printf("   prediction uncertainty information (useful for, for.\n");
   printf("   example, adaptive sample refinement.)\n");
   printf(" SUM-OF-TREES REGRESSION - usually gives non-smooth response\n");
   printf("   surfaces. It is provided here for completeness, but is\n");
   printf("   not generally recommended (except for very large sample).\n");
   printf(" SPARSE GRID REGRESSION - has to use sparse grid designs. \n");
   printf("   Also, you cannot use CV on sparse grid regression. Use\n");
   printf("   a holdout set (rstest) to validate your response surface.\n");
   printf(" KRIGING - This is another form of the Gaussian process which\n");
   printf("   does not have as many hyperparameters. This method is good\n");
   printf("   for up to about 2000 sample points; otherise is may be\n");
   printf("   computationally too expensive.\n");
   printf(" SPLINES - currently only supports 1D, 2D, or 3D. This method\n");
   printf("   works only with full factorial designs. Also, you cannot\n");
   printf("   use cross validation with this method. Instead, use a\n");
   printf("   holdout set (rstest) to validate your response surface.\n");
   printf(" K-NEAREST NEIGHBOR - for large data set when data points are\n");
   printf("   relatively close to one another, this may work well.\n");
   printf(" RBF  - for small to medium data set (too expensive otherwise).\n");
   printf(" MRBF - for larger data set (> 2000).\n");
   printf(" MGP3 - (Gaussian process) for larger data set (> 2000).\n");
   printf(" MDNN - (Neural Network process) for larger data set (> 2000).\n");
   printf(" MTGP - (multiple TGP) for larger data set (> 2000).\n");
   printf(" MMARS - (multiple MARS) for even larger data set (> 20000).\n");
   printf(" PADE - Pade regression.\n");
   printf(" HLR - Legendre polynomial for homogeneous (isotropic) inputs\n");
   printf(" HGP - GP for isotropic inputs (optional: last input quantile)\n");
   printf(" HKR - Kriging for isotropic inputs (last input quantile)\n");
   printf(" HYGP - Legendre-GP for isotropic inputs (with quantile input)\n");
   printf(" QGP - Quantile GP for isotropic/anisotropic inputs with\n");
   printf("     isotropic case akin to HGP but different implementation.\n");
  }
#ifdef HAVE_MARS
  printDashes(PL_INFO, 0);
  printf("0.  MARS (Friedman's multivariate splines method)\n");
#endif
  if (level > 3)
    printf("==> Linear/nonlinear polynomial regression methods:\n");
  printf("1.  Linear regression \n");
  printf("2.  Quadratic regression \n");
  printf("3.  Cubic regression \n");
  printf("4.  Quartic regression (may be unstable)\n");
  printf("5.  Selective polynomial regression (user selects terms to use)\n");
  printf("6.  Derivative-based Legendre polynomial regression\n");
  if (level > 3)
   printf("    - need 2 samples: the second one with nInputs derivatives.\n");
  printf("7.  Legendre polynomial regression\n");
  printf("8.  User-defined regression (user provides basis functions)\n");
  if (level > 3) printf("==> Other response surface methods: \n");
#ifdef HAVE_TPROS
  printf("9.  Gaussian process (MacKay's implementation)\n");
#endif
  printf("10. Gaussian process (Tong's implementation)\n");
  printf("11. Kriging\n"); 
  printf("12. Radial Basis Function\n");
  printf("13. Sum-of-trees model\n");
  printf("14. K nearest neighbors \n");
  if (level > 3)
    printf("    - needs large samples to improve prediction accuracy\n"); 
  printf("15. Artificial neural network\n");
#ifdef HAVE_TGP
  printf("16. Tree-based Gaussian Process (Gramacy and Lee)\n");
#endif
#ifdef HAVE_SVM
  printf("17. SVM-light (Joachims)\n");
#endif
  printf("18. Sparse Grid polynomial regression\n"); 
  if (level > 3)
    printf("    - Works only with sparse grid sampling design.\n");
  printf("19. Splines on regular grid (1D, 2D, or 3D only)\n");
  if (level > 3)
    printf("    - Works only with factorial designs in 1-3D.\n");
  printf("20. Acosso (by Storlie, LANL. Need R to run)\n");
  printf("21. BSSAnova (by Storlie, LANL. Need R to run)\n");
  printf("22. Partial Least Squares Linear Regression (PLS)\n");
  if (level > 3)
  {
    printf("    - Linear regression but geared toward the most\n");
    printf("      eigenvectors.\n");
    printf("==> Methods based on bootstrapped aggregation:\n");
  }
#ifdef HAVE_MARS
  printf("23. MARS with bootstrap aggregating (bagging)\n");
#endif
  printf("24. Radial Basis Function with bagging\n");
  if (level > 3)
  {
    printf("==> Domain (input space) decomposition-based methods:\n");
    printf(" - Idea: Divide input space into subdomains to reduce cost\n");
    printf("         since each domain has a small sample size.\n");
    printf(" - Binary partitioning along most sensitive dimensioins\n");
  }
  printf("25. Multi-Radial Basis Function (for large samples)\n");
  printf("26. Multi-Gaussian process (Tong, for large samples)\n");
  printf("27. Multi-MARS (for large samples)\n");
  printf("28. Multi-Treed Gaussian process (for large samples)\n");
  if (level > 3)
    printf("==> Homogeneous and quantile-based methods:\n");
  printf("29. Homogeneous Legendre regression (HLR)\n");
  if (level > 3)
    printf("    This method is for homogeneous inputs (same bounds).\n");
  printf("30. Homogeneous GP (HGP)\n");
  if (level > 3)
  {
    printf("  This method is for either:\n");
    printf("    a. All inputs are homogeneous (and same bounds too).\n");
    printf("    b. Only the last input is nonhomogeneous (quantile).\n");
    printf("       Different sample points can have different quantiles.\n");
  }
  printf("31. Homogeneous Kriging (all homogeneous inputs: same bounds)\n");
  printf("32. Hybrid Homogeneous GP (HyHGP)\n");
  if (level > 3)
  {
    printf("  This method is for either:\n");
    printf("    a. All inputs are homogeneous (and same bounds too), or\n");
    printf("    b. Only the last input is nonhomogeneous (quantile).\n");
    printf("       - All N sample points must have same Q quantiles.\n");
    printf("       - For CV, must use nGroups be factors of N/Q, and\n");
    printf("         must not use randomization. Otherwise, CV fails.\n");
  }
  printf("33. Quantile GP (can be anisotropic)\n");
  if (level > 3)
  {
    printf("  This method requires:\n");
    printf("    - Quantile variable as the last input.\n");
    printf("    - All N sample points have same Q quantiles.\n");
    printf("    - There must be a quantile input value of 0.5\n");
    printf("    - For CV, must use nGroups be factors of N/Q, and\n");
    printf("      must not use randomization. Otherwise, CV fails.\n");
    printf("    Note: Similar to HGP(b) but different implementation.\n");
  }
  printf("34. User-modified general nonlinear function\n");
  printf("    (Need to modify the PsuadeRegression.cpp file)\n");
  printf("35. Multi-domain neural network (for large samples)\n");
  printf("36. Pade regression F(X) ~ P(X) / Q(X)\n");
  return PSUADE_NUM_RS;
}

// ************************************************************************
// friend function (create a function approximator from a few parameters)
// ------------------------------------------------------------------------
extern "C" 
FuncApprox *genFA(int faType, int nInputs, int outLevel, int nSamples)
{
  int        rsType, nsize;
  FuncApprox *faPtr=NULL;
  char       *params[1], winput[10000], *strPtr, equal[100];

  //**/ -------------------------------------------------
  //**/ get response surface type
  //**/ -------------------------------------------------
  if (faType >= 0) rsType = faType;
  else
  {
    rsType = -1;
    while (rsType < 0 || rsType >= PSUADE_NUM_RS)
    {
      writeFAInfo(outLevel);
      snprintf(winput,100,"Please enter your choice ? ");
      rsType = getInt(0, PSUADE_NUM_RS, winput);
    }
  }

  //**/ -------------------------------------------------
  //**/ instantiate response surface
  //**/ -------------------------------------------------
  if      (rsType == PSUADE_RS_MARS) faPtr = new Mars(nInputs, nSamples);
  else if (rsType == PSUADE_RS_ANN)  faPtr = new DNN(nInputs, nSamples);
  else if (rsType == PSUADE_RS_REGRS)
          faPtr = new SelectiveRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_GP1) faPtr = new GP1(nInputs, nSamples);
  else if (rsType == PSUADE_RS_GP3) 
  {
#if 0
    // disable automatic switching to MGP3 for large sample size 
    faPtr = NULL;
    //**/ if override, do not switch to MGP3
    strPtr = psConfig_.getParameter("RS_no_multi_domain");
    if (strPtr != NULL) faPtr = new RBF(nInputs, nSamples);
    //**/ if users have modified group sample size, use it to switch
    //**/ to MGP3 is sample size too large
    if (faPtr == NULL)
    {
      nsize = 2000; 
      strPtr = psConfig_.getParameter("RS_MGP_max_sam_per_group");
      if (strPtr != NULL)
      {
        sscanf(strPtr, "%s %s %d", winput, equal, &nsize);
        if (nsize < 100) nsize = 1000;
      }
      if (nSamples > nsize)
      {
        printf("nSamples > %d ==> switch to MGP3\n", nsize);
        printf("NOTE: to remain in GP3, set RS_no_multi_domain\n");
        faPtr = new MGP3(nInputs, nSamples);
      }
      else faPtr = new GP3(nInputs, nSamples);
    }
#endif
    faPtr = new GP3(nInputs, nSamples);
  }
  else if (rsType == PSUADE_RS_SVM) faPtr = new SVM(nInputs, nSamples);
  else if (rsType == PSUADE_RS_REGRGL)
          faPtr = new GradLegendreRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_TGP)   faPtr = new TGP(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MARSB) faPtr = new MarsBagg(nInputs, nSamples);
  else if (rsType == PSUADE_RS_SOTS) faPtr = new SumOfTrees(nInputs,nSamples);
  else if (rsType == PSUADE_RS_REGRL)
          faPtr = new LegendreRegression(nInputs,nSamples);
  else if (rsType == PSUADE_RS_REGRU)
          faPtr = new UserRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_REGSG)
          faPtr = new SparseGridRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_KR)
          faPtr = new Kriging(nInputs, nSamples);
  else if (rsType == PSUADE_RS_SPLINES)
  {
    if (nInputs > 3)
    {
      printf("genFA ERROR: Splines does not support nInputs > 3.\n");
      exit(1);
    }
    faPtr = new Splines(nInputs, nSamples);
  }
  else if (rsType == PSUADE_RS_KNN) faPtr = new KNN(nInputs, nSamples);
  else if (rsType == PSUADE_RS_RBF)
  {
    faPtr = NULL;
    //**/ if override, do not switch to MRBF
    strPtr = psConfig_.getParameter("RS_no_multi_domain");
    if (strPtr != NULL) faPtr = new RBF(nInputs, nSamples);
    //**/ if users have modified group sample size, use it to switch
    //**/ to MRBF is sample size too large
    if (faPtr == NULL)
    {
      nsize = 5000; 
      strPtr = psConfig_.getParameter("RS_MRBF_max_sam_per_group");
      if (strPtr != NULL)
      {
        sscanf(strPtr, "%s %s %d", winput, equal, &nsize);
        if (nsize < 100) nsize = 2000;
      }
      if (nSamples > nsize)
      {
        printf("nSamples > %d ==> switch to MRBF\n", nsize);
        printf("NOTE: to remain in RBF, set RS_no_multi_domain\n");
        faPtr = new MRBF(nInputs, nSamples);
      }
      else faPtr = new RBF(nInputs, nSamples);
    }
  }
  else if (rsType == PSUADE_RS_ACOSSO)
       faPtr = new Acosso(nInputs, nSamples);
  else if (rsType == PSUADE_RS_BSSANOVA)
       faPtr = new BSSAnova(nInputs, nSamples);
  else if (rsType == PSUADE_RS_RBFB)
       faPtr = new RBFBagg(nInputs, nSamples);
  else if (rsType == PSUADE_RS_PLS)
       faPtr = new PLS(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MRBF)
       faPtr = new MRBF(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MGP3)
       faPtr = new MGP3(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MDNN)
       faPtr = new MDNN(nInputs, nSamples);
  else if (rsType == PSUADE_RS_PADE)
       faPtr = new PadeRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MMARS)
       faPtr = new MMars(nInputs, nSamples);
  else if (faType == PSUADE_RS_LOCAL)
       faPtr = new PsuadeRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_HLEG)
       faPtr = new HomLegendreRegression(nInputs, nSamples);
  else if (rsType == PSUADE_RS_HGP3)
       faPtr = new HGP3(nInputs, nSamples);
  else if (rsType == PSUADE_RS_MTGP)
       faPtr = new MTGP(nInputs, nSamples);
  else if (rsType == PSUADE_RS_HKR)
       faPtr = new HKriging(nInputs, nSamples);
  else if (rsType == PSUADE_RS_HYGP)
       faPtr = new HybridGP(nInputs, nSamples);
  else if (rsType == PSUADE_RS_QGP)
       faPtr = new QGP(nInputs, nSamples);
  else
  {
    //printf("INFO: rstype has been set to default = regression.\n");
    faPtr = new Regression(nInputs, nSamples);
    params[0] = (char *) &rsType;
    faPtr->setParams(1, params);
  }
  return faPtr;
}

// ************************************************************************
// friend function (create a function approximator from a data file)
//**/ (flag == 0 ==> use the type from psuadeIO) 
//**/ (flag == 1 ==> ask user which RS to use)
//**/ (flag == 2 ==> use RS type from psuadeIO and create the model)
//**/ (flag == 3 ==> ask user for RS type and create model - initialize)
//**/ outputID fetched from psuadeIO
//**/ no data transformation due to PDF
// ------------------------------------------------------------------------
extern "C" 
FuncApprox *genFAInteractive(PsuadeData *psuadeIO, int flag)
{
  int        faType, nInputs, nSamples, nOutputs, wgtID, ii, nPtsPerDim;
  int        totPts, outputID, printLevel, nsize;
  double     *wghts, *Y;
  FuncApprox *faPtr;
  char       *params[3], winput[5001], equal[100], *strPtr;
  pData      pPtr, pInputs, pOutputs, pStates, pLower, pUpper;

  //**/ -------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------
  if (psuadeIO == NULL)
  {
    printf("ERROR: PsuadeData does not exist.\n");
    return NULL;
  }
  psuadeIO->getParameter("ana_diagnostics", pPtr);
  printLevel = pPtr.intData_;

  //**/ -------------------------------------------------
  //**/ fetch or ask for function approximation type
  //**/ -------------------------------------------------
  if ((flag & 1) == 0)
  {
    assert(psuadeIO->getParameter("ana_rstype", pPtr) == 0);
    faType = pPtr.intData_;
    if (faType < 0 || faType >= PSUADE_NUM_RS)
    {
      printf("genFAInteractive : faType (%d) not valid.\n",
             faType);
      exit(1);
    }
  }
  else
  {
    faType = -1;
    while (faType < 0 || faType >= PSUADE_NUM_RS)
    {
      writeFAInfo(printLevel);
      snprintf(winput,100,"Please enter your choice ? ");
      faType = getInt(0, PSUADE_NUM_RS-1, winput);
    }
  }

  //**/ -------------------------------------------------
  //**/ fetch the other parameters
  //**/ -------------------------------------------------
  psuadeIO->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  if (nSamples > psConfig_.RSMaxPts_)
  {
    printf("PSUADE WARNING: For nSamples > %d,\n", 
           psConfig_.RSMaxPts_);
    printf("  It can be very expensive to create ");
    printf("response surfaces.\n");
    printf("  Use the rsmax <num> to override.\n");
    exit(1);
  }
  psuadeIO->getParameter("ana_outputid", pPtr);
  outputID = pPtr.intData_;
  psuadeIO->getParameter("ana_regressionwgtid", pPtr);
  wgtID = pPtr.intData_;
  psuadeIO->getParameter("output_sample", pOutputs);
  psuadeIO->getParameter("input_sample", pInputs);
  psuadeIO->getParameter("output_states", pStates);
  psuadeIO->getParameter("input_lbounds", pLower);
  psuadeIO->getParameter("input_ubounds", pUpper);

  //**/ -------------------------------------------------
  //**/ instantiate function approximation object
  //**/ -------------------------------------------------
  if (faType == PSUADE_RS_MARS) 
       faPtr = new Mars(nInputs, nSamples);
  else if (faType == PSUADE_RS_ANN)
       faPtr = new DNN(nInputs, nSamples);
  else if (faType == PSUADE_RS_REGRS)
       faPtr = new SelectiveRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_GP1)  
       faPtr = new GP1(nInputs, nSamples);
  else if (faType == PSUADE_RS_GP3)
  {
#if 0
    //**/ March 2023: let's not do this automatically. Let users
    //**/             decide
    faPtr = NULL;
    //**/ if override, do not switch to MGP3
    strPtr = psConfig_.getParameter("RS_no_multi_domain");
    if (strPtr != NULL) faPtr = new RBF(nInputs, nSamples);
    //**/ if users have modified group sample size, use it 
    //**/ to switch to MGP3 is sample size too large
    if (faPtr == NULL)
    {
      nsize = 2000; 
      strPtr = psConfig_.getParameter("RS_MGP_max_sam_per_group");
      if (strPtr != NULL)
      {
        sscanf(strPtr, "%s %s %d", winput, equal, &nsize);
        if (nsize < 100) nsize = 1000;
      }
      if (nSamples > nsize)
      {
        printf("nSamples > %d ==> switch to MGP3\n", nsize);
        faPtr = new MGP3(nInputs, nSamples);
      }
      else faPtr = new GP3(nInputs, nSamples);
    }
#endif
    faPtr = new GP3(nInputs, nSamples);
  }
  else if (faType == PSUADE_RS_SVM)
          faPtr = new SVM(nInputs, nSamples);
  else if (faType == PSUADE_RS_REGRGL)
          faPtr = new GradLegendreRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_TGP)
          faPtr = new TGP(nInputs, nSamples);
  else if (faType == PSUADE_RS_MARSB)
          faPtr = new MarsBagg(nInputs, nSamples);
  else if (faType == PSUADE_RS_SOTS)
          faPtr = new SumOfTrees(nInputs,nSamples);
  else if (faType == PSUADE_RS_REGRL)
  {
    faPtr = new LegendreRegression(nInputs,nSamples);
    psuadeIO->getParameter("ana_poly_order", pPtr);
    ii = pPtr.intData_;
    if (ii > 0)
    {
      params[0] = (char *) &ii;
      faPtr->setParams(1, params);
    }
  }
  else if (faType == PSUADE_RS_REGRU)
          faPtr = new UserRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_REGSG)
          faPtr = new SparseGridRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_KR)
          faPtr = new Kriging(nInputs, nSamples);
  else if (faType == PSUADE_RS_SPLINES)
  {
    if (nInputs > 3)
    {
      printf("genFAInteractive ERROR: Splines does not support nInputs > 3.\n");
      exit(1);
    }
    faPtr = new Splines(nInputs, nSamples);
  }
  else if (faType == PSUADE_RS_KNN) faPtr = new KNN(nInputs, nSamples);
  else if (faType == PSUADE_RS_RBF)
  {
    faPtr = NULL;
    //**/ if override, do not switch to MRBF
    strPtr = psConfig_.getParameter("RS_no_multi_domain");
    if (strPtr != NULL) faPtr = new RBF(nInputs, nSamples);
    //**/ if users have modified group sample size, use it to switch
    //**/ to MRBF is sample size too large
    if (faPtr == NULL)
    {
      nsize = 5000; 
      strPtr = psConfig_.getParameter("RS_MRBF_max_sam_per_group");
      if (strPtr != NULL)
      {
        sscanf(strPtr, "%s %s %d", winput, equal, &nsize);
        if (nsize < 100) nsize = 2000;
      }
      if (nSamples > nsize)
      {
        printf("nSamples > %d ==> switch to MRBF\n", nsize);
        faPtr = new MRBF(nInputs, nSamples);
      }
      else faPtr = new RBF(nInputs, nSamples);
    }
  }
  else if (faType == PSUADE_RS_ACOSSO)
       faPtr = new Acosso(nInputs, nSamples);
  else if (faType == PSUADE_RS_BSSANOVA)
       faPtr = new BSSAnova(nInputs, nSamples);
  else if (faType == PSUADE_RS_RBFB)
       faPtr = new RBFBagg(nInputs, nSamples);
  else if (faType == PSUADE_RS_PLS)
       faPtr = new PLS(nInputs, nSamples);
  else if (faType == PSUADE_RS_MRBF)
       faPtr = new MRBF(nInputs, nSamples);
  else if (faType == PSUADE_RS_MGP3)
       faPtr = new MGP3(nInputs, nSamples);
  else if (faType == PSUADE_RS_MDNN)
       faPtr = new MDNN(nInputs, nSamples);
  else if (faType == PSUADE_RS_PADE)
       faPtr = new PadeRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_MMARS)
       faPtr = new MMars(nInputs, nSamples);
  else if (faType == PSUADE_RS_LOCAL)
       faPtr = new PsuadeRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_HLEG)
       faPtr = new HomLegendreRegression(nInputs, nSamples);
  else if (faType == PSUADE_RS_HGP3)
       faPtr = new HGP3(nInputs, nSamples);
  else if (faType == PSUADE_RS_HKR)
       faPtr = new HKriging(nInputs, nSamples);
  else if (faType == PSUADE_RS_HYGP)
       faPtr = new HybridGP(nInputs, nSamples);
  else if (faType == PSUADE_RS_QGP)
       faPtr = new QGP(nInputs, nSamples);
  else
  {
    faPtr = new Regression(nInputs, nSamples);
    params[0] = (char *) &faType;
    faPtr->setParams(1, params);
  }
  faPtr->setBounds(pLower.dbleArray_, pUpper.dbleArray_);
  faPtr->setOutputLevel(printLevel);

  //**/ -------------------------------------------------
  //**/ set number of points per dimension
  //**/ -------------------------------------------------
  nPtsPerDim = 256;
  totPts = 1000001;
  while (totPts > 1000000)
  {
    nPtsPerDim = nPtsPerDim / 2;
    totPts = 1;
    for (ii = 0; ii < nInputs; ii++)
    {
      totPts *= nPtsPerDim;
      if (totPts > 1000000) break;
    }
  }
  faPtr->setNPtsPerDim(nPtsPerDim);

  //**/ -------------------------------------------------
  //**/ load in sample weights, if any
  //**/ -------------------------------------------------
  psVector vecWgts;
  if (wgtID >= 0 && wgtID < nOutputs)
  {
    vecWgts.setLength(nSamples);
    for (ii = 0; ii < nSamples; ii++)
      vecWgts[ii] = pOutputs.dbleArray_[nOutputs*ii+wgtID];
    faPtr->loadWeights(nSamples, vecWgts.getDVector());
  }

  //**/ -------------------------------------------------
  //**/ load data
  //**/ -------------------------------------------------
  if (flag & 2)
  {
    psVector vecYT;
    vecYT.setLength(nSamples);
    for (ii = 0; ii < nSamples; ii++)
       vecYT[ii] = pOutputs.dbleArray_[nOutputs*ii+outputID];
    faPtr->initialize(pInputs.dbleArray_, vecYT.getDVector());
  }
  return faPtr;
}

// ************************************************************************
// friend function (create a function approximator given a file name)
// perform PDF transformation
// check invalid sample points
// RS type from file 
// ------------------------------------------------------------------------
extern "C" 
FuncApprox *genFAFromFile(char *fname, int outputID)
{
  int        ii, nInputs, nOutputs, nSamples, status, *sampleStates;
  double     *sampleInputs, *sampleOutputs;
  PsuadeData *psuadeIO = new PsuadeData();
  FuncApprox *faPtr;
  pData      pPtr, pInpData, pOutData, pStates;

  //**/ ----------------------------------------------------------------
  //**/ read the model data
  //**/ ----------------------------------------------------------------
  for (ii = strlen(fname)-1; ii >= 0; ii--) if (fname[ii] == '/') break;
  status = psuadeIO->readPsuadeFile(fname);
  if (status != 0)
  {
    delete psuadeIO;
    return NULL;
  }

  //**/ ----------------------------------------------------------------
  //**/ retrieve the model parameters
  //**/ ----------------------------------------------------------------
  psuadeIO->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  psuadeIO->getParameter("input_sample", pInpData);
  sampleInputs = pInpData.dbleArray_;
  psuadeIO->getParameter("output_sample", pOutData);
  sampleOutputs = pOutData.dbleArray_;
  psuadeIO->getParameter("output_states", pStates);
  sampleStates = pStates.intArray_;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < nSamples; ii++)
  {
    if (sampleStates[ii] != 1 || 
        sampleOutputs[nOutputs*ii+outputID] == PSUADE_UNDEFINED)
    {
      printf("FuncApprox::genRSModel ERROR - invalid output.\n");
      printf("  Advice: check your sample data file to see if there is\n");
      printf("          any sample output having the value 9.999e+34,\n");
      printf("          any sample status flag not equal to 1.\n");
      exit(1);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ create a RS model (need to transform data first)
  //**/ ----------------------------------------------------------------
  PDFTransform(psuadeIO, nSamples, nInputs, sampleInputs);
  psuadeIO->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                               sampleInputs,NULL,NULL,NULL,NULL,NULL);
  psuadeIO->getParameter("ana_outputid", pPtr);
  ii = pPtr.intData_;
  psuadeIO->updateAnalysisSection(-1,-1,-1,-1,outputID,-1);
  faPtr = genFAInteractive(psuadeIO, 2);
  psuadeIO->updateAnalysisSection(-1,-1,-1,-1,ii,-1);
  delete psuadeIO;
  return faPtr;
}

