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
// Functions for the class GradLegendreRegression
// AUTHOR : CHARLES TONG
// DATE   : 2013
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "GradLegendreRegression.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PsuadeData.h"
#include "pData.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
GradLegendreRegression::GradLegendreRegression(int nInputs,int nSamples):
                                               FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_REGRGL;
  pOrder_ = -1;
  numPerms_ = 0;
  //**/ need to be normalized (1/4/13)
  //**/if (psConfig_.RSExpertModeIsOn())
  //**/{
  //**/   printf("Normalize the input parameters to [-1, 1]? (y - yes) ");
  //**/   fgets(line, 100, stdin);
  //**/   if (line[0] == 'y') normalizeFlag_ = 1;
  //**/}
  normalizeFlag_ = 1;
  nOutputs_ = 1;
  printAsterisks(PL_INFO, 0);
  printf("* GradLegendreRegression constructor\n");
  printDashes(PL_INFO, 0);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
GradLegendreRegression::~GradLegendreRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int GradLegendreRegression::initialize(double *X, double *Y)
{
  int  ii, status;
  char pString[101];
  FILE *fp;

  //**/ ---------------------------------------------------------------
  //**/ if derivatives have not been loaded, ask for a file
  //**/ ---------------------------------------------------------------
  if (VecDerivSample_.length() != nSamples_*nInputs_)
  {
    printf("Users need to provide a sample file (in PSUADE format) ");
    printf("that has\n");
    printf("derivative information (where the sample outputs ");
    printf("are derivatives\n");
    printf("of Y with respect to each input).\n");
    printf("NOTE: The number of inputs/outputs in this file ");
    printf("should match that of\n");
    printf("      the sample used for creating response surface.\n");
    printf(" ==>  The number of outputs in this file should be %d\n",
           nInputs_);
    printf(" ==>  The sample size of this file should be %d\n",
           nSamples_);
    snprintf(pString,100,"Enter name of derivative file (PSUADE format): ");
    getString(pString, gradFile_);
    ii = strlen(gradFile_);
    gradFile_[ii-1] = '\0';
    fp = fopen(gradFile_, "r");
    if (fp == NULL)
    {
      printf("GradLegendreRegr ERROR: cannot open derivative file (%s)\n",
             gradFile_);
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ launch regression
  //**/ ---------------------------------------------------------------
  status = analyze(X, Y);
  if (status != 0)
  {
    printf("GradLegendreRegression::initialize - ERROR detected.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int GradLegendreRegression::genNDGridData(double *X, double *Y, int *NOut,
                                          double **XOut, double **YOut)
{
  int totPts, ss, status;
  //**/ ---------------------------------------------------------------
  //**/ launch regression
  //**/ ---------------------------------------------------------------
  status = analyze(X, Y);
  if (status != 0)
  {
    printf("GradLegendreRegression::genNDGridData - ERROR detected.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ ---------------------------------------------------------------
  //**/ return if there is no request to create lattice points
  //**/ ---------------------------------------------------------------
  if ((*NOut) == -999) return 0;

  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();

  for (ss = 0; ss < totPts; ss++)
    (*YOut)[ss] = evaluatePoint(&((*XOut)[ss*nInputs_]));
  return 0;
}

// ************************************************************************
// Generate 1D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int GradLegendreRegression::gen1DGridData(double *X, double *Y, int ind1,
                                          double *settings, int *NOut, 
                                          double **XOut, double **YOut)
{
  int    totPts, mm, nn;
  double HX;
  psVector vecXT;

  //**/ ---------------------------------------------------------------
  //**/ create response surface
  //**/ ---------------------------------------------------------------
  (*NOut) = -999;
  genNDGridData(X, Y, NOut, NULL, NULL);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    vecXT[ind1] = HX * mm + VecLBs_[ind1];
    (*XOut)[mm] = vecXT[ind1];
    (*YOut)[mm] = evaluatePoint(vecXT.getDVector());
  }

  //**/ ---------------------------------------------------------------
  //**/ return 
  //**/ ---------------------------------------------------------------
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int GradLegendreRegression::gen2DGridData(double *XIn,double *YIn,int ind1,
                                     int ind2, double *settings, int *NOut, 
                                     double **XOut, double **YOut)
{
  int totPts, mm, nn, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ create response surface
  //**/ ---------------------------------------------------------------
  (*NOut) = -999;
  genNDGridData(XIn, YIn, NOut, NULL, NULL);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts * 2);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      index = mm * nPtsPerDim_ + nn;
      vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
      vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
      (*XOut)[index*2]   = vecXT[ind1];
      (*XOut)[index*2+1] = vecXT[ind2];
      (*YOut)[index] = evaluatePoint(vecXT.getDVector());
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ return 
  //**/ ---------------------------------------------------------------
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int GradLegendreRegression::gen3DGridData(double *XIn,double *YIn,int ind1,
                                   int ind2, int ind3, double *settings, 
                                   int *NOut, double **XOut, double **YOut)
{
  int totPts, mm, nn, pp, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ create response surface
  //**/ ---------------------------------------------------------------
  (*NOut) = -999;
  genNDGridData(XIn, YIn, NOut, NULL, NULL);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts * 3);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;
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
        (*XOut)[index*3]   = vecXT[ind1];
        (*XOut)[index*3+1] = vecXT[ind2];
        (*XOut)[index*3+2] = vecXT[ind3];
        (*YOut)[index] = evaluatePoint(vecXT.getDVector());
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ return 
  //**/ ---------------------------------------------------------------
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int GradLegendreRegression::gen4DGridData(double *XIn,double *YIn,int ind1, 
                                          int ind2, int ind3, int ind4, 
                                          double *settings, int *NOut, 
                                          double **XOut, double **YOut)
{
  int totPts, mm, nn, pp, qq, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ create response surface
  //**/ ---------------------------------------------------------------
  (*NOut) = -999;
  genNDGridData(XIn, YIn, NOut, NULL, NULL);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts * 4);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;
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
          (*XOut)[index*4]   = vecXT[ind1];
          (*XOut)[index*4+1] = vecXT[ind2];
          (*XOut)[index*4+2] = vecXT[ind3];
          (*XOut)[index*4+3] = vecXT[ind4];
          (*YOut)[index] = evaluatePoint(vecXT.getDVector());
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ return 
  //**/ ---------------------------------------------------------------
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double GradLegendreRegression::evaluatePoint(double *X)
{
  int    ii, nn;
  double Y, multiplier, normalX;
  psMatrix MatLTable;

  //**/ check to make sure it has been initialized
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("GradLegendre ERROR: initialize has not been called.\n");
    exit(1);
  }

  //**/ allocate for computing Legendre table 
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  //**/ evaluate Legendre polynomial
  Y = 0.0;
  for (nn = 0; nn < numPerms_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        EvalLegendrePolynomials(X[ii], LTable[ii]);
      }
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
        normalX = normalX * 2.0 - 1.0;
        EvalLegendrePolynomials(normalX, LTable[ii]);
      }
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    Y += VecRegCoeffs_[nn]* multiplier;
  }
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double GradLegendreRegression::evaluatePoint(int npts, double *X, double *Y)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double GradLegendreRegression::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, nn;
  double   Y, multiplier, normalX, stdev, dtmp;
  psVector VecXs;
  psMatrix MatLTable;

  //**/ check to make sure it has been initialized
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("GradLegendre ERROR: initialize has not been called.\n");
    exit(1);
  }

  //**/ allocate for computing Legendre table 
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  //**/ evaluate Legendre polynomial
  VecXs.setLength(numPerms_);
  Y = 0.0;
  for (nn = 0; nn < numPerms_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        EvalLegendrePolynomials(X[ii], LTable[ii]);
      }
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
        normalX = normalX * 2.0 - 1.0;
        EvalLegendrePolynomials(normalX, LTable[ii]);
      }
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    Y += VecRegCoeffs_[nn]* multiplier;
    VecXs[nn] = multiplier;
  }

  //**/ compute standard deviation
  stdev = 0.0;
  for (ii = 0; ii < numPerms_; ii++)
  {
    dtmp = 0.0;
    for (nn = 0; nn < numPerms_; nn++)
      dtmp += MatInvCov_.getEntry(ii,nn) * VecXs[nn];
    stdev += dtmp * VecXs[ii];
  }
  std = sqrt(stdev);

  //**/ clean up
  for (ii = 0; ii < nInputs_; ii++) delete [] LTable[ii];
  delete [] LTable;
  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double GradLegendreRegression::evaluatePointFuzzy(int npts, double *X,
                                                  double *Y, double *Ystd)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&X[kk*nInputs_], Ystd[kk]);
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int GradLegendreRegression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int GradLegendreRegression::analyze(psVector VecX, psVector VecY)
{
  int    N, M, ii, mm, nn, wlen, info, NRevised, status, nInps;
  double SSreg, SStotal, R2, var, esum, ymax;
  char   pString[101];
  FILE   *fp;
  pData    pPtr;
  psMatrix eigMatT, MatXX, MatA, MatU, MatV;
  psVector eigVals, VecYY, VecS, VecW, VecB;
  PsuadeData *ioPtr;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs_ <= 0 || nSamples_ <= 0)
  {
    printf("GradLegendreRegression::analyze ERROR - invalid arguments.\n");
    exit(1);
  } 
  GenPermutations();
   
  //**/ ---------------------------------------------------------------
  //**/ call regression analysis
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    printf("*                GradLegendre Regression Analysis\n");
    printf("* R-square gives a measure of the goodness of the model.\n");
    printf("* R-square should be close to 1 if it is a good model.\n");
    printf("* Turn on rs_expert mode to output regression matrix.\n");
    printDashes(PL_INFO, 0);
    printf("* Each input will be scaled to [-1, 1].\n");
    printf("* With this, statistics such as mean, variances, and\n");
    printf("* conditional variances are readily available.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ read derivative file 
  //**/ ---------------------------------------------------------------
  if (VecDerivSample_.length() == 0)
  {
    ioPtr = new PsuadeData();
    status = ioPtr->readPsuadeFile(gradFile_);
    if (status != 0)
    {
      printf("GradLegendreReg ERROR: problem reading derivative file.\n");
      exit(1);
    }
    ioPtr->getParameter("input_ninputs", pPtr);
    int nInps = pPtr.intData_;
    if (nInps != nInputs_)
    {
      printf("GradLegendreReg ERROR: nInput mismatch in derivative file.\n");
      exit(1);
    }
    ioPtr->getParameter("output_noutputs", pPtr);
    if (pPtr.intData_ != nInputs_)
    {
      printf("GradLegendreReg ERROR: invalid nOutputs in derivative file.\n");
      printf("                       Should be equal to %d\n", nInputs_);
      exit(1);
    }
    ioPtr->getParameter("method_nsamples", pPtr);
    if (pPtr.intData_ != nSamples_)
    {
      printf("GradLegendreReg ERROR: invalid sample size in derivative file\n");
      printf("                        Should be equal to %d\n", nSamples_);
      exit(1);
    }
    ioPtr->getParameter("output_noutputs", pPtr);
    nn = pPtr.intData_;
    if (nn != nInputs_)
    {
      printf("GradLegendreReg ERROR - derivative file should have %d outputs\n",
             nInputs_);
      printf("                               You only have %d outputs.\n",nn);
      exit(1);
    }
    ioPtr->getParameter("output_sample", pPtr);
    delete ioPtr;
  }

  //**/ ==============================================================
  //**/ VecYY should have the output plus the derivatives
  //**/ ==============================================================
  double *derivatives;
  if (VecDerivSample_.length() == 0) 
       derivatives = pPtr.dbleArray_;
  else derivatives = VecDerivSample_.getDVector();
  M = nSamples_ * (nInputs_ + 1);
  VecYY.setLength(M);
  for (mm = 0; mm < nSamples_; mm++)
  {
    VecYY[mm*(nInputs_+1)] = VecY[mm];
    for (nn = 0; nn < nInputs_; nn++)
      VecYY[mm*(nInputs_+1)+nn+1] = derivatives[mm*nInputs_+nn];
  }

  //**/ ==============================================================
  //**/ load the A matrix
  //**/ ==============================================================
  N = loadXMatrix(VecX, MatXX);
  psVector VecA;
  VecA.setLength(M*N);
  double *arrayXX = MatXX.getMatrix1D();
  for (mm = 0; mm < M; mm++)
    for (nn = 0; nn < N; nn++) VecA[mm+nn*M] = arrayXX[mm+nn*M];
  MatA.load(M, N, VecA.getDVector());

  //**/ ==============================================================
  //**/ diagnostics 
  //**/ ==============================================================
  if (psConfig_.RSExpertModeIsOn() && outputLevel_ > 0)
  {
    fp = fopen("Grad_Legendre_regression_matrix.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n",
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "%% the sample matrix where svd is computed\n");
    fprintf(fp, "%% the last column is the right hand side\n");
    fprintf(fp, "%% B is the vector of coefficients\n");
    fprintf(fp, "AA = [\n");
    for (mm = 0; mm < M; mm++)
    {
      for (nn = 0; nn < N; nn++) 
        fprintf(fp, "%16.6e ", VecA[mm+nn*M]);
      fprintf(fp, "%16.6e \n",VecYY[mm]);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "A = AA(:,1:%d);\n", N);
    fprintf(fp, "B = AA(:,%d);\n", N+1);
    fclose(fp);
    printf("Regression matrix written to Grad_Legendre_regression_matrix.m\n");
  }

  //**/ ==============================================================
  //**/ perform SVD 
  //**/ ==============================================================
  if (outputLevel_ > 3) printf("Running SVD ...\n");
  info = MatA.computeSVD(MatU, VecS, MatV);
  if (outputLevel_ > 3)
    printf("SVD completed: status = %d (should be 0).\n",info);

  if (info != 0)
  {
    printf("* GradLegendreRegression Info: dgesvd returns a nonzero (%d).\n",
           info);
    printf("* GradLegendreRegression terminates further processing.\n");
    printf("* Consult PSUADE developers for advice.\n");
    return -1;
  }

  //**/ ==============================================================
  //**/ eliminate the noise components 
  //**/ ==============================================================
  if (VecS[0] == 0.0) NRevised = 0;
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++)
      if (VecS[nn-1] > 0 && VecS[nn]/VecS[nn-1] < 1.0e-8) break;
    NRevised = nn;
  }
  if (NRevised < N)
  {
    printf("GradLegendreRegression WARNING: True matrix rank = %d (N=%d)\n",
           NRevised,N);
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    {
      printf("INFO: This can be due to the quality of the sample.\n");
      for (nn = 0; nn < N; nn++)
        printf("Singular value %5d = %e\n",nn+1,VecS[nn]);
    }
  }
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("GradLegendreReg: Singular values for the Vandermonde matrix\n");
    printf(" - The VERY small ones may cause poor numerical accuracy\n");
    printf(" - but not keeping them may ruin the approximation power\n");
    printf(" - So, select them judiciously\n");
    for (nn = 0; nn < N; nn++)
      printf("Singular value %5d = %e\n", nn+1, VecS[nn]);
    snprintf(pString,100,"How many to keep (1 - %d) ? ", N);
    NRevised = getInt(1,N,pString);
    for (nn = NRevised; nn < N; nn++) VecS[nn] = 0.0;
  }
  else
  {
    for (nn = NRevised; nn < N; nn++) VecS[nn] = 0.0;
    if (NRevised < N)
      printf("GradLegendre INFO: Number of singular values removed = %d\n",
             N-NRevised);
  }

  //**/ ==============================================================
  //**/ compute B
  //**/ ==============================================================
  VecW.setLength(M+N);
  double *UU = MatU.getMatrix1D();
  for (mm = 0; mm < NRevised; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++) VecW[mm] += UU[mm*M+nn] * VecYY[nn];
  }
  for (nn = 0; nn < NRevised; nn++) VecW[nn] /= VecS[nn];
  for (nn = NRevised; nn < N; nn++) VecW[nn] = 0.0;
  VecB.setLength(N);
  double *VV = MatV.getMatrix1D();
  for (mm = 0; mm < N; mm++)
  {
    VecB[mm] = 0.0;
    for (nn = 0; nn < NRevised; nn++) VecB[mm] += VV[nn+mm*N] * VecW[nn];
  }

  //**/ ==============================================================
  //**/ store eigenvectors VV and eigenvalues SS^2
  //**/ ==============================================================
  eigMatT.load(N, N, VV);
  eigVals.load(N, VecS.getDVector());
  for (nn = 0; nn < N; nn++) eigVals[nn] = pow(eigVals[nn], 2.0);

  //**/ ==============================================================
  //**/ compute residual
  //**/ ==============================================================
  fp = NULL;
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    fp = fopen("grad_legendre_regression_err_file.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n",
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "%% This file contains errors of each data point.\n");
  }

  esum = ymax = 0.0;
  for (mm = 0; mm < M; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++)
      VecW[mm] = VecW[mm] + arrayXX[mm+nn*M] * VecB[nn];
    VecW[mm] = VecW[mm] - VecYY[mm];
    esum = esum + VecW[mm] * VecW[mm];
    if (fp != NULL) 
      fprintf(fp, "%6d %24.16e\n",mm+1,VecW[mm]);
    if (PABS(VecYY[mm]) > ymax) ymax = PABS(VecYY[mm]);
  }
  esum /= (double) M;
  esum = sqrt(esum);
  printf("* GradLegendreRegression: LS mean error = %10.3e (max=%10.3e)\n",
         esum, ymax); 

  if (fp != NULL)
  {
    fclose(fp);
    printf("FILE grad_legendre_regression_err_file contains data errors\n");
  }

  //**/ ==============================================================
  //**/ form compute SS statistics 
  //**/ ==============================================================
  status = computeSS(MatXX, VecYY, VecB, SSreg, SStotal);
  
  //**/ normal case
  if (status == 0)
  {
    if (SStotal == 0) R2 = 1.0;
    else              R2  = SSreg / SStotal;
    if (M > N) var = (SStotal - SSreg) / (double) (M - N);
    else       var = 0.0;
    if (var < 0)
    {
      if (PABS(var) > 1.0e-12)
      {
        printf("GradLegendreRegression WARNING: variance %e < 0.\n",var);
        printf("                       Temporary absolutize var.\n");
        var = PABS(var);
      }
      else var = 0;
    }
  }
  else
  //**/ in case SSreg > SStotal
  {
    if (SStotal == 0) R2 = 1.0;
    else              R2  = 1.0 - SSreg / SStotal;
    if (M > N) var = SSreg / (double) (M - N);
    if (var < 0)
    {
      if (PABS(var) > 1.0e-12)
           printf("GradLegendreRegression WARNING: var %e < 0.\n", var);
      else var = 0;
    }
  }
  VecRegCoeffs_.load(VecB);

  //**/ ==============================================================
  //**/ find variance of each coefficient 
  //**/ ==============================================================
  computeCoeffVariance(eigMatT, eigVals, var);
  psVector VecBstd;
  VecBstd.setLength(N);
  for (ii = 0; ii < N; ii++)
    VecBstd[ii] = sqrt(MatInvCov_.getEntry(ii,ii));

  //**/ ==============================================================
  //**/ print out regression coefficients 
  //**/ ==============================================================
  if (outputLevel_ >= 0)
  {
    if (outputLevel_ > 0) printRC(VecB, VecBstd, MatXX, VecYY);
    printf("* Regression R-square = %10.3e\n", R2);
    if (M-N-1 > 0)
      printf("* adjusted   R-square = %10.3e\n",
             1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
  }
  printAsterisks(PL_INFO, 0);

  //**/ ==============================================================
  //**/ create function file 
  //**/ ==============================================================
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"/* ***********************************************/\n");
    fprintf(fp,"/* GradLegendre regression interpolator from PSUADE. */\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   3. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"int interpolate(int,double *,double *,double *);\n");
    fprintf(fp,"main(int argc, char **argv) {\n");
    fprintf(fp,"  int    i, iOne=1, nInps;\n");
    fprintf(fp,"  double X[%d], Y, S;\n",nInputs_);
    fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
    fprintf(fp,"  if (argc < 3) {\n");
    fprintf(fp,"     printf(\"ERROR: not enough argument.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
    fprintf(fp,"  if (fIn == NULL) {\n");
    fprintf(fp,"     printf(\"ERROR: cannot open input file.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fscanf(fIn, \"%%d\", &nInps);\n");
    fprintf(fp,"  if (nInps != %d) {\n", nInputs_);
    fprintf(fp,"    printf(\"ERROR - wrong nInputs.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  for (i=0; i<%d; i++) fscanf(fIn, \"%%lg\", &X[i]);\n",
            nInputs_);
    fprintf(fp,"  fclose(fIn);\n");
    fprintf(fp,"  interpolate(iOne, X, &Y, &S);\n");
    fprintf(fp,"  printf(\"Y = %%e\\n\", Y);\n");
    fprintf(fp,"  printf(\"S = %%e\\n\", S);\n");
    fprintf(fp,"  fOut = fopen(argv[2], \"w\");\n");
    fprintf(fp,"  if (fOut == NULL) {\n");
    fprintf(fp,"     printf(\"ERROR: cannot open output file.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fprintf(fOut,\" %%e\\n\", Y);\n");
    fprintf(fp,"  fclose(fOut);\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"/* GradLegendre regression interpolation function*/\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"static int\n");
    fprintf(fp,"pcePerms[%d][%d] = \n", numPerms_, nInputs_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numPerms_; mm++)
    {
      fprintf(fp,"  {");
      for (ii = 0; ii < nInputs_-1; ii++)
        fprintf(fp," %d,", MatPCEPerms_.getEntry(mm,ii));
      fprintf(fp," %d },\n", MatPCEPerms_.getEntry(mm,nInputs_-1));
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"invCovMat[%d][%d] = \n", numPerms_, numPerms_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numPerms_; mm++)
    {
       fprintf(fp,"  {");
       for (ii = 0; ii < numPerms_-1; ii++)
          fprintf(fp," %24.16e,", MatInvCov_.getEntry(mm,ii));
       fprintf(fp," %24.16e },\n", MatInvCov_.getEntry(mm,numPerms_-1));
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"regCoefs[%d] = \n", numPerms_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numPerms_; mm++)
      fprintf(fp," %24.16e,", VecRegCoeffs_[mm]);
    fprintf(fp,"};\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
    fprintf(fp,"  int    ii, kk, ss, nn;\n");
    fprintf(fp,"  double *x, y, **LTable, normX, mult;\n");
    fprintf(fp,"  double mean, std, *x2, dtmp;\n");
    fprintf(fp,"  LTable = (double **) malloc(%d * sizeof(double*));\n",
               nInputs_);
    fprintf(fp,"  for (ii = 0; ii < %d; ii++)\n", nInputs_);
    fprintf(fp,"    LTable[ii] = (double *) malloc((%d+1)*sizeof(double));\n",
            pOrder_);
    fprintf(fp,"  x2 = (double *) malloc(%d * sizeof(double));\n",numPerms_);
    fprintf(fp,"  for (ss = 0; ss < npts; ss++) {\n");
    fprintf(fp,"    x = &X[ss * %d];\n", nInputs_);
    fprintf(fp,"    y = 0.0;\n");
    fprintf(fp,"    for (nn = 0; nn < %d; nn++) {\n", numPerms_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        fprintf(fp,"      normX = X[%d] - %24.16e;\n",ii, VecXMeans_[ii]);
        fprintf(fp,"      normX /= %24.16e;\n", VecXStds_[ii]);
        fprintf(fp,"      EvalLegendrePolynomials(normX,LTable[%d]);\n",ii);
      }
      else
      {
        fprintf(fp,"      normX = X[%d] - %24.16e;\n",
                ii, VecLBs_[ii]);
        fprintf(fp,"      normX /= (%24.16e - %24.16e);\n",
                VecUBs_[ii], VecLBs_[ii]);
        fprintf(fp,"      normX = normX * 2.0 - 1.0;\n");
        fprintf(fp,"      EvalLegendrePolynomials(normX,LTable[%d]);\n",
                ii);
      }
    }
    fprintf(fp,"      mult = 1.0;\n");
    for (ii = 0; ii < nInputs_; ii++)
    fprintf(fp,"      mult *= LTable[%d][pcePerms[nn][%d]];\n",ii,ii);
    fprintf(fp,"      y += regCoefs[nn] * mult;\n");
    fprintf(fp,"      x2[nn] = mult;\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    Y[ss] = y * %e + %e;\n", YStd_, YMean_);
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for (ii = 0; ii < %d; ii++) {\n",numPerms_);
    fprintf(fp,"      dtmp = 0.0;\n");
    fprintf(fp,"      for (kk = 0; kk < %d; kk++)\n",numPerms_);
    fprintf(fp,"        dtmp += invCovMat[ii][kk] * x2[kk];\n");
    fprintf(fp,"      std += dtmp * x2[ii];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    std = sqrt(std);\n");
    fprintf(fp,"    S[ss] = std;\n");
    fprintf(fp,"  }\n");
    for (ii = 0; ii < nInputs_; ii++)
       fprintf(fp,"  free(LTable[%d]);\n", ii);
    fprintf(fp,"  free(LTable);\n");
    fprintf(fp,"  free(x2);\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"int EvalLegendrePolynomials(double X, double *LTable) {\n");
    fprintf(fp,"  int    ii;\n");
    fprintf(fp,"  LTable[0] = 1.0;\n");
    fprintf(fp,"  if (%d >= 1) {\n", pOrder_);
    fprintf(fp,"     LTable[1] = X;\n");
    fprintf(fp,"     for (ii = 2; ii <= %d; ii++)\n", pOrder_);
    fprintf(fp,"        LTable[ii] = ((2 * ii - 1) * X * LTable[ii-1] -\n");
    fprintf(fp,"                      (ii - 1) * LTable[ii-2]) / ii;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* ==============================================*/\n");
    fclose(fp);
    printf("FILE psuade_rs.info contains information about\n");
    printf("     the Legendre polynomial.\n");
  }
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# GradLegendre Regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"pcePerms = [\n");
    for (mm = 0; mm < numPerms_; mm++)
    {
      fprintf(fp," [ %d", MatPCEPerms_.getEntry(mm,0));
      for (ii = 1; ii < nInputs_; ii++)
        fprintf(fp,", %d", MatPCEPerms_.getEntry(mm,ii));
      fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"invCovMat = [\n");
    for (mm = 0; mm < numPerms_; mm++)
    {
      fprintf(fp," [ %24.16e", MatInvCov_.getEntry(mm,0));
      for (nn = 1; nn < numPerms_; nn++)
        fprintf(fp,", %24.16e", MatInvCov_.getEntry(mm,nn));
      fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"regCoefs = [\n");
    for (mm = 0; mm < numPerms_-1; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp," %24.16e ]\n", VecRegCoeffs_[numPerms_-1]);
    fprintf(fp,"###################################################\n");
    fprintf(fp,"def EvalLegendrePolynomials(X) :\n");
    fprintf(fp,"  LTable = %d * [0.0]\n", pOrder_+1);
    fprintf(fp,"  LTable[0] = 1.0;\n");
    fprintf(fp,"  if (%d >= 1) :\n", pOrder_);
    fprintf(fp,"    LTable[1] = X;\n");
    fprintf(fp,"    for ii in range(%d) : \n", pOrder_-1);
    fprintf(fp,"      jj = ii + 2\n");
    fprintf(fp,"      LTable[jj] = ((2 * jj - 1) * X * LTable[jj-1] -\n");
    fprintf(fp,"                    (jj - 1) * LTable[jj-2]) / jj;\n");
    fprintf(fp,"  return LTable\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# Regression interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(X): \n");
    fprintf(fp,"  nSamp = int(len(X) / %d + 1.0e-8)\n",nInputs_);
    fprintf(fp,"  Xt = %d * [0.0]\n", nInputs_);
    fprintf(fp,"  Xs = %d * [0.0]\n", numPerms_);
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  for ss in range(nSamp): \n");
    fprintf(fp,"    for ii in range(%d) : \n", nInputs_);
    fprintf(fp,"      Xt[ii] = X[ss*%d+ii]\n",nInputs_);
    fprintf(fp,"    Y = 0.0\n");
    fprintf(fp,"    for nn in range(%d): \n", numPerms_);
    fprintf(fp,"      mult = 1.0\n");
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        fprintf(fp,"      x2 = Xt[%d] - %24.16e\n",ii, VecXMeans_[ii]);
        fprintf(fp,"      x2 = x2 / %24.16e\n", VecXStds_[ii]);
        fprintf(fp,"      LTable = EvalLegendrePolynomials(x2)\n");
      }
      else
      {
        fprintf(fp,"      x2 = Xt[%d] - %24.16e\n",ii,VecLBs_[ii]);
        fprintf(fp,"      x2 = x2 / (%24.16e - %24.16e)\n",
                VecUBs_[ii], VecLBs_[ii]);
        fprintf(fp,"      x2 = x2 * 2.0 - 1.0\n");
        fprintf(fp,"      LTable = EvalLegendrePolynomials(x2)\n");
      }
      fprintf(fp,"      mult *= LTable[pcePerms[nn][%d]]\n",ii);
    }
    fprintf(fp,"      Xs[nn] = mult\n");
    fprintf(fp,"      Y = Y + regCoefs[nn] * mult\n");
    fprintf(fp,"    Ys[2*ss] = Y * %e + %e\n",YStd_, YMean_);
    fprintf(fp,"    std = 0.0\n");
    fprintf(fp,"    for jj in range(%d): \n", numPerms_);
    fprintf(fp,"      dtmp = 0.0\n");
    fprintf(fp,"      for kk in range(%d): \n", numPerms_);
    fprintf(fp,"        dtmp = dtmp + invCovMat[jj][kk] * Xs[kk]\n");
    fprintf(fp,"      std = std + dtmp * Xs[jj]\n");
    fprintf(fp,"    Ys[ss*2+1] = math.sqrt(std)\n");
    fprintf(fp,"  return Ys\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# main program\n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"infileName  = sys.argv[1]\n");
    fprintf(fp,"outfileName = sys.argv[2]\n");
    fprintf(fp,"inputs = getInputData(infileName)\n");
    fprintf(fp,"outputs = interpolate(inputs)\n");
    fprintf(fp,"genOutputFile(outfileName, outputs)\n");
    fprintf(fp,"###################################################\n");
    printf("FILE psuade_rs.py contains the final Legendre polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int GradLegendreRegression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int    M, N=0, ss, ii, kk, nn;
  double multiplier, normalX, range;
  psVector VecXX;
  psMatrix MatLTable, MatDTable;

  //**/ ==============================================================
  //**/ lower and upper bounds needed for derivatives
  //**/ ==============================================================
  for (ii = 0; ii < nInputs_; ii++)
  {
    multiplier = VecUBs_[ii] - VecLBs_[ii];
    if (multiplier == 0.0)
    {
      printf("GradLegendreRegression ERROR: inputs not normalized.\n");
      exit(1);
    }
  }

  //**/ ==============================================================
  //**/ allocating memory for tables
  //**/ ==============================================================
  M = nSamples_ * (nInputs_ + 1);
  N = numPerms_;
  VecXX.setLength(M*N);
  MatLTable.setFormat(PS_MAT2D);
  MatDTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  MatDTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();
  double **DTable = MatDTable.getMatrix2D();

  //**/ ==============================================================
  //**/ filling in the matrix
  //**/ ==============================================================
  for (ss = 0; ss < nSamples_; ss++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0) normalX = VecX[ss*nInputs_+ii]; 
      else
      {
        normalX = VecX[ss*nInputs_+ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
        normalX = normalX * 2.0 - 1.0;
      }
      EvalLegendrePolynomials(normalX, LTable[ii]);
      EvalLegendrePolynomialsDerivative(normalX, DTable[ii]);
      //**/ why did I insert these lines?? (7/2014)
      if (normalizeFlag_ == 1)
      {
        range = 2.0 / (VecUBs_[ii] - VecLBs_[ii]);
        for (kk = 0; kk < pOrder_+1; kk++) DTable[ii][kk] *= range;
      }
    }
    for (nn = 0; nn < numPerms_; nn++)
    {
      multiplier = 1.0;
      for (ii = 0; ii < nInputs_; ii++)
        multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
      VecXX[M*nn+ss*(nInputs_+1)] = multiplier;
    }
    for (kk = 0; kk < nInputs_; kk++)
    {
      for (nn = 0; nn < numPerms_; nn++)
      {
        multiplier = 1.0;
        for (ii = 0; ii < nInputs_; ii++)
        {
          if (kk == ii) 
               multiplier *= DTable[ii][MatPCEPerms_.getEntry(nn,ii)];
          else multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
        }
        VecXX[M*nn+ss*(nInputs_+1)+kk+1] = multiplier;
      }
    }
  }
  MatXX.setFormat(PS_MAT1D);
  MatXX.load(M, N, VecXX.getDVector());
  return N;
}

// *************************************************************************
// compute SS (sum of squares) statistics
//**/ SSred = (Y - Xb)' W (Y - Xb)
//**/       = Y' W Y - 2 b 'X W 'Y + b' X' W X b
//**/       = Y' W Y - 2 b' X' W Y + b' X' W Y  (since X'WXb=X'WY)
//**/       = Y' W Y - b' X' W Y = (Y - Xb)' W Y
//**/ SStot = Y' W Y - N * (mean (W^(1/2) Y))^2
// -------------------------------------------------------------------------
int GradLegendreRegression::computeSS(psMatrix MatXX, psVector VecY,
                                      psVector VecB, double &SSreg, 
                                      double &SStotal)
{
  int    nn, mm, M, N;
  double ymean, SSresid, SSresidCheck, ddata, rdata;
                                                                               
  N = VecB.length();
  double *arrayXX = MatXX.getMatrix1D();
  M = nSamples_ * (nInputs_ + 1);
  SSresid = SSresidCheck = SStotal = ymean = SSreg = 0.0;
  for (mm = 0; mm < M; mm++) ymean += VecY[mm];
  ymean /= (double) M;
  for (mm = 0; mm < M; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) ddata += (arrayXX[mm+nn*M] * VecB[nn]);
    rdata = VecY[mm] - ddata;
    SSresid += VecY[mm] * rdata;
    SSresidCheck += rdata * rdata;
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < M; mm++)
    SStotal += (VecY[mm] - ymean) * (VecY[mm] - ymean);
  printf("* GradLegendreRegression: SStot = %e\n", SStotal);
  printf("* GradLegendreRegression: SSreg = %e\n", SSreg);
  printf("* GradLegendreRegression: SSres = %e\n", SSresid);
  printf("* GradLegendreRegression: SSres = %e (true)\n", SSresidCheck);
  SSresid = SSresidCheck;
  if (SStotal < SSreg)
  {
    SSreg = SSresid;
    return 1;
  }
  return 0;
}

// *************************************************************************
// compute coefficient variance ((diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int GradLegendreRegression::computeCoeffVariance(psMatrix &eigMatT,
                               psVector &eigVals, double var)
{
  int      ii, jj, nRows;
  double   invEig, dtmp;
  psMatrix tMat;

  nRows = eigMatT.nrows();
  tMat.setDim(nRows, nRows);

  //**/ compute var * V S^{-2}
  for (ii = 0; ii < nRows; ii++)
  {
    invEig = eigVals[ii];
    if (invEig != 0.0) invEig = 1.0 / invEig;
    for (jj = 0; jj < nRows; jj++)
    {
      dtmp = invEig * eigMatT.getEntry(ii,jj) * var;
      tMat.setEntry(jj, ii, dtmp);
    }
  }
  //**/ compute var V * S^{-2} * V^T 
  tMat.matmult(eigMatT, MatInvCov_);
  return 0;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int GradLegendreRegression::printRC(psVector VecB, psVector VecBstd,
                                    psMatrix, psVector VecY)
{
  int    ii, jj, kk, maxTerms, flag;
  double coef, ddata, variance;
  FILE   *fp;

  maxTerms = 0;
  for (ii = 0; ii < numPerms_; ii++) 
  {
    kk = MatPCEPerms_.getEntry(ii, 0); 
    if (kk > maxTerms) maxTerms = kk;
  }

  printEquals(PL_INFO, 0);
  if (normalizeFlag_ == 1)
    printf("* Note: coefficients below are for normalized input ranges.\n");
  printDashes(PL_INFO, 0);
  printf("*      ");
  for (ii = 0; ii < nInputs_; ii++) printf("     ");
  printf("               coefficient  std. error  t-value\n");

  //**/ --------- print out the linear coefficients --------------
  for (ii = 0; ii < numPerms_; ii++)
  {
    if (PABS(VecBstd[ii]) < 1.0e-15) coef = 0.0;
    else                             coef = VecB[ii] / VecBstd[ii]; 
//**/ if (PABS(coef) > 1.0)
    {
      printf("* Input orders: ");
      for (jj = 0; jj < nInputs_; jj++)
        printf(" %4d ", MatPCEPerms_.getEntry(ii,jj));
      printf("= %11.3e %11.3e %11.3e\n", VecB[ii], VecBstd[ii], coef);
    }
  }
  //**/ --------- print out partial variances --------------
  flag = 1;
  for (ii = 0; ii < nInputs_; ii++)
    if (VecUBs_[ii] != 1.0) flag = 0;
  for (ii = 0; ii < nInputs_; ii++)
    if (VecLBs_[ii] != -1.0) flag = 0;
  if (normalizeFlag_ == 1 || flag == 1)
  {
    printDashes(PL_INFO, 0);
    printf("* Mean     = %12.4e\n", VecB[0]);
    coef = 0.0;
    for (jj = 1; jj < numPerms_; jj++) 
    {
      ddata = VecB[jj];
      for (kk = 0; kk < nInputs_; kk++)
        ddata /= sqrt(1.0+MatPCEPerms_.getEntry(jj,kk)*2); 
      coef = coef + ddata * ddata;
    }
    printf("* Variance = %12.4e\n", coef);
    variance = coef;
    fp = fopen("matlabgradlegendre.m", "w");
    fwritePlotCLF(fp);
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nInputs_; ii++)
    {
      coef = 0.0;
      for (jj = 1; jj < numPerms_; jj++)
      {
        flag = 1;
        for (kk = 0; kk < nInputs_; kk++)
          if (kk != ii && MatPCEPerms_.getEntry(jj,kk) != 0) flag = 0;
        if (flag == 1)
        {
          ddata = VecB[jj];
          for (kk = 0; kk < nInputs_; kk++)
            ddata /= sqrt(1.0+MatPCEPerms_.getEntry(jj,kk)*2); 
          coef = coef + ddata * ddata;
        }
      }
      fprintf(fp, "%e\n", coef/variance);
      printf("* Conditional variance %4d = %12.4e\n", ii+1, coef);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "bar(A, 0.8);\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Legendre VCE Rankings");
    fwritePlotXLabel(fp, "Input parameters");
    fwritePlotYLabel(fp, "Rank Metric");
    fclose(fp);
    printf("Legendre VCE ranking is now in matlabgradlegendre.m.\n");
  }
  printEquals(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// generate all combinations of a multivariate Legendre expansion
// This code is a direct translation from Burkardt's matlab code)
// -------------------------------------------------------------------------
int GradLegendreRegression::GenPermutations()
{
  int  ii, kk, orderTmp, rvTmp, M;
  char pString[101];

  //**/ search for maximum order
  M = nSamples_ * (nInputs_ + 1);
  if (pOrder_ <= 0)
  {
    numPerms_ = 0;
    pOrder_ = 0;
    while (numPerms_ < M)
    { 
      pOrder_++;
      numPerms_ = 1;
      if (nInputs_ < pOrder_)
      {
        for (ii = nInputs_+pOrder_; ii > nInputs_; ii--)
          numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
      }
      else
      {
        for (ii = nInputs_+pOrder_; ii > pOrder_; ii--)
          numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
      }
    }
    if (numPerms_ > M) pOrder_--;
    printf("* Legendre polynomial maximum order = %d\n", pOrder_);
    snprintf(pString,100,"Desired order (>=1 and <= %d) ? ", pOrder_);
    pOrder_ = getInt(1, pOrder_, pString);
  }
  numPerms_ = 1;
  if (nInputs_ < pOrder_)
  {
    for (ii = nInputs_+pOrder_; ii > nInputs_; ii--)
      numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
  }
  else
  {
    for (ii = nInputs_+pOrder_; ii > pOrder_; ii--)
      numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
  }
  printf("* GradLegendreRegression: order of polynomials   = %d\n", pOrder_);
  printf("* GradLegendreRegression: number of permutations = %d\n",numPerms_);
   
  //**/ construct the permutations
  MatPCEPerms_.setFormat(PS_MAT2D);
  MatPCEPerms_.setDim(numPerms_, nInputs_);

  numPerms_ = 0;
  for (kk = 0; kk <= pOrder_; kk++)
  {
    orderTmp = kk;
    rvTmp = 0;
    MatPCEPerms_.setEntry(numPerms_, 0, orderTmp);
    for (ii = 1; ii < nInputs_; ii++) 
      MatPCEPerms_.setEntry(numPerms_, ii, 0);
    while (MatPCEPerms_.getEntry(numPerms_, nInputs_-1) != kk)
    {
      numPerms_++;
      for (ii = 0; ii < nInputs_; ii++)
        MatPCEPerms_.setEntry(numPerms_, ii, 
                              MatPCEPerms_.getEntry(numPerms_-1, ii)); 
      if (orderTmp > 1) rvTmp = 1;
      else              rvTmp++;
      MatPCEPerms_.setEntry(numPerms_, rvTmp-1, 0);
      orderTmp = MatPCEPerms_.getEntry(numPerms_-1, rvTmp-1);
      MatPCEPerms_.setEntry(numPerms_, 0, orderTmp-1);
      MatPCEPerms_.setEntry(numPerms_, rvTmp, 
                        MatPCEPerms_.getEntry(numPerms_-1, rvTmp)+1); 
    }
    numPerms_++;
  }
  return 0;
}

// *************************************************************************
// Purpose: evaluate 1D Legendre polynomials (semi-normalized)
// -------------------------------------------------------------------------
int GradLegendreRegression::EvalLegendrePolynomials(double X, double *LTable)
{
  int    ii;
  LTable[0] = 1.0;
  if (pOrder_ >= 1)
  {
    LTable[1] = X;
    for (ii = 2; ii <= pOrder_; ii++)
      LTable[ii] = ((2 * ii - 1) * X * LTable[ii-1] -
                    (ii - 1) * LTable[ii-2]) / ii;
  }
  //**/ do not normalize (the Legendre form is harder to recognize)
  //**/ should use this instead of sqrt(0.5+2*ii) since
  //**/ need to compute 1/2 int phi_i^2 dx
  //**/for (ii = 0; ii <= pOrder_; ii++) LTable[ii] *= sqrt(1.0+2.0*ii);
  return 0;
}

// *************************************************************************
// Purpose: evaluate 1D Legendre polynomials derivative (semi-normalized)
// -------------------------------------------------------------------------
int GradLegendreRegression::EvalLegendrePolynomialsDerivative(double X, 
                                                              double *DTable)
{
  int    ii;
  DTable[0] = 0.0;
  if (pOrder_ >= 1) DTable[1] = 1.0;
  if (pOrder_ >= 2)
  {
    DTable[2] = 3.0 * X;
    for (ii = 3; ii <= pOrder_; ii++)
      DTable[ii] = ((2 * ii - 1) * X * DTable[ii-1] -
                    ii * DTable[ii-2]) / (ii - 1.0);
  }
  //**/ do not normalize (the Legendre form is harder to recognize)
  //**/ should use this instead of sqrt(0.5+2*ii) since
  //**/ need to compute 1/2 int phi_i^2 dx
  //**/for (ii = 0; ii <= pOrder_; ii++) DTable[ii] *= sqrt(1.0+2.0*ii);
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double GradLegendreRegression::setParams(int targc, char **targv)
{
  if (targc == 1)
  {
    pOrder_ = *(int *) targv[0];
    if (pOrder_ <= 0)
    {
      printf("LegendreRegression setParams: pOrder not valid.\n");
      exit(1);
    }
    printf("LegendreRegression setParams: pOrder set to %d.\n",pOrder_);
  }
  else if (targc == 2 && !strcmp(targv[0], "nOutputs"))
  {
    nOutputs_ = *(int *) targv[1];
    if (nOutputs_ <= 0)
    {
      printf("LegendreRegression setParams: nOutputs not valid.\n");
      exit(1);
    }
  }
  else if (targc == 2 && !strcmp(targv[0], "deriv_sample"))
  {
    int    length = *(int *) targv[1];
    double *derivSample = (double *) targv[2];
    if (length <= 0)
    {
      printf("LegendreRegression setParams ERROR: \n");
      printf("                   invalid derivative sample.\n");
      return -1;
    }
    VecDerivSample_.load(length, derivSample);
  }
  else
  {
    printf("LegendreRegression setParams ERROR: keyword not recognized.\n");
    exit(1);
  }
  return 0.0;
}

