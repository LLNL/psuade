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
// Functions for the class SelectiveRegression
// AUTHOR : CHARLES TONG
// DATE   : 2005
//**/ add uncertainty with covariance and code gen : 3/2014
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "SelectiveRegression.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

extern "C" {
   void dgesvd_(char *, char *, int *, int *, double *, int *, double *,
               double *, int *, double *, int *, double *, int *, int *);
}

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
SelectiveRegression::SelectiveRegression(int nInputs,int nSamples):
                                         FuncApprox(nInputs,nSamples)
{
  int  ii, jj, kk, termCheck, length, maxLeng;
  char inStr[300];
  FILE *fp = fopen("selective_regression_file", "r");

  faID_ = PSUADE_RS_REGRU;
  if (fp == NULL)
  {
    printf("SelectiveRegr ERROR: selective_regression_file not found.\n");
    printf("This file is needed to identify selected terms.\n");
    printf("This file should have the following format: \n");
    printf("line   1: PSUADE_BEGIN\n");
    printf("line   2: the number of terms.\n");
    printf("line   3: 1  1 0 (meaning constant term)\n");
    printf("line   4: 2  number of inputs, input list (1-based).\n");
    printf("line   5: 3  number of inputs, input list (1-based).\n");
    printf(".........\n");
    printf("lastline: PSUADE_END\n");
    exit(1);
  }
  else
  {
    printf("SelectiveRegr INFO: using selective_regression_file.\n");
  }
  fscanf(fp, "%s", inStr);
  if (strcmp(inStr, "PSUADE_BEGIN"))
  {
    printf("SelectiveRegr ERROR: keyword PSUADE_BEGIN not found\n");
    printf("Correct format in selective_regression_file:\n");
    printf("line   1: PSUADE_BEGIN\n");
    printf("line   2: the number of terms.\n");
    printf("line   3: 1  1 0 (meaning constant term)\n");
    printf("line   4: 2  number of inputs, input list (1-based).\n");
    printf("line   5: 3  number of inputs, input list (1-based).\n");
    printf(".........\n");
    printf("lastline: PSUADE_END\n");
    fclose(fp);
    exit(1);
  }
  fscanf(fp, "%d", &numTerms_);
  if (numTerms_ >= nSamples)
  {
    printf("SelectiveRegr ERROR: no. of terms should be < nSamples.\n");
    printf("              nTerms   = %d\n", numTerms_);
    printf("              nSamples = %d\n", nSamples);
    printf("Correct format in selective_regression_file:\n");
    printf("line   1: PSUADE_BEGIN\n");
    printf("line   2: the number of terms.\n");
    printf("line   3: 1  1 0 (meaning constant term)\n");
    printf("line   4: 2  number of inputs, input list (1-based).\n");
    printf("line   5: 3  number of inputs, input list (1-based).\n");
    printf(".........\n");
    printf("lastline: PSUADE_END\n");
    exit(1);
  }

  //**/ find maximum length 
  maxLeng = -10000;
  for (ii = 0; ii < numTerms_; ii++)
  {
    fscanf(fp, "%d %d", &termCheck, &length);
    if (length > maxLeng) maxLeng = length;
    if (termCheck != (ii+1))
    {
      printf("SelectiveRegr ERROR: %d-th term has index %d.\n",ii+1,
             termCheck);
      printf("              Check line %d in selective_regression_file\n",
             ii+3);
      exit(1);
    }
    if (length < 1)
    {
      printf("SelectiveRegr ERROR: each term should have >1 element.\n");
      printf("              Check line %d in selective_regression_file\n",
             ii+3);
      exit(1);
    }
    for (jj = 0; jj < length; jj++) 
    {
      fscanf(fp, "%d", &kk);
      if (kk == 0 && length != 1)
      {
        printf("SelectiveRegr ERROR: constant term should have leng=1\n");
        printf("              Check line %d in selective_regression_file\n",
               ii+3);
        exit(1);
      }
      if (kk < 0)
      {
        printf("SelectiveRegression ERROR : input number should be > 0\n");
        exit(1);
      }
      if (kk > nInputs)
      {
        printf("SelectiveRegression ERROR : input number should be <= %d\n",
               nInputs);
        exit(1);
      }
    }
  }

  //**/ instantiate coefficient term matrix and read file
  MatCoefTerms_.setFormat(PS_MAT2D);
  MatCoefTerms_.setDim(numTerms_, maxLeng+1);

  fclose(fp);
  fp = fopen("selective_regression_file", "r");
  fscanf(fp, "%s", inStr);
  fscanf(fp, "%d", &numTerms_);

  for (ii = 0; ii < numTerms_; ii++)
  {
    fscanf(fp, "%d %d", &termCheck, &length);
    MatCoefTerms_.setEntry(ii, 0, length);
    for (jj = 0; jj < length; jj++) 
    {
      fscanf(fp, "%d", &kk);
      MatCoefTerms_.setEntry(ii, jj+1, kk-1);
    }
  }
  fscanf(fp, "%s", inStr);
  fclose(fp);
  if (strcmp(inStr, "PSUADE_END"))
  {
    printf("SelectiveRegr ERROR: wrong format in selective_regression_file\n");
    printf("The file should end with PSUADE_END\n");
    printf("Correct format in selective_regression_file:\n");
    printf("line   1: PSUADE_BEGIN\n");
    printf("line   2: the number of terms.\n");
    printf("line   3: 1  1 0 (meaning constant term)\n");
    printf("line   4: 2  number of inputs, input list (1-based).\n");
    printf("line   5: 3  number of inputs, input list (1-based).\n");
    printf(".........\n");
    printf("lastline: PSUADE_END\n");
    exit(1);
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
SelectiveRegression::~SelectiveRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int SelectiveRegression::initialize(double *X, double *Y)
{
  int      ii, status;
  psVector VecX2;

  //**/ ===============================================================
  //**/ error checking
  //**/ ===============================================================
  if (nSamples_ <= nInputs_)
  {
    printf("SelectiveRegression::initialize INFO- not enough points.\n");
    printf("                nSamples should be larger than nInputs.\n");
    return -1;
  }
   
  //**/ ===============================================================
  //**/ launch different order regression (up to a maximum of 4)
  //**/ ===============================================================
  printEquals(PL_INFO, 0);
  printf("* Selective Regression Analysis\n");
  printEquals(PL_INFO, 0);
  VecX2.setLength(nSamples_ * nInputs_);
  if (psConfig_.MasterModeIsOn())
  {
    printf("* SelectiveRegression INFO: scaling turned off.\n");
    printf("*                  To turn scaling on in master mode,\n");
    printf("*                  have rs_expert mode on also.\n");
    initInputScaling(X, VecX2.getDVector(), 0);
  }
  else initInputScaling(X, VecX2.getDVector(), 1);

  //**/ ===============================================================
  //**/ analyze regression
  //**/ ===============================================================
  status = analyze(VecX2.getDVector(),Y);

  if (status != 0)
  {
    printf("SelectiveRegression::initialize - ERROR detected.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int SelectiveRegression::genNDGridData(double *X, double *Y, int *NN,
                                       double **XX, double **YY)
{
  int totPts, ss;
  psVector vecYOut;

  //**/ ===============================================================
  //**/ initialization
  //**/ ===============================================================
  if (initialize(X,Y) != 0)
  {
    printf("SelectiveRegression::genNDGridData - ERROR detected.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ return if there is no request to create lattice points
  //**/ ===============================================================
  if ((*NN) == -999) return 0;

  //**/ ===============================================================
  //**/ generating regular grid data
  //**/ ===============================================================
  genNDGrid(NN, XX);
  if ((*NN) == 0) return 0;
  totPts = (*NN);

  //**/ ===============================================================
  //**/ allocate storage for the data points and generate them
  //**/ ===============================================================
  vecYOut.setLength(totPts);
  (*YY) = vecYOut.takeDVector();
  for (ss = 0; ss < totPts; ss++)
    (*YY)[ss] = evaluatePoint(&((*XX)[ss*nInputs_]));
  return 0;
}

// ************************************************************************
// Generate 1D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int SelectiveRegression::gen1DGridData(double *X, double *Y, int ind1,
                                       double *settings, int *NN, 
                                       double **XX, double **YY)
{
  int    totPts, mm, nn;
  double HX;
  psVector vecXOut, vecYOut, vecXT;

  //**/ ===============================================================
  //**/ initialization
  //**/ ===============================================================
  if (initialize(X,Y) != 0)
  {
    printf("SelectiveRegression::gen1DGridData - ERROR detected.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ set up for generating regular grid data
  //**/ ===============================================================
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ===============================================================
  //**/ allocate storage for and then generate the data points
  //**/ ===============================================================
  (*NN) = totPts;
  vecXOut.setLength(totPts);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    vecXT[ind1] = HX * mm + VecLBs_[ind1];
    (*XX)[mm] = vecXT[ind1];
    (*YY)[mm] = evaluatePoint(vecXT.getDVector());
  }
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int SelectiveRegression::gen2DGridData(double *X, double *Y, int ind1,
                                       int ind2, double *settings, int *NN, 
                                       double **XX, double **YY)
{
  int totPts, mm, nn, index;
  psVector vecXOut, vecYOut, vecXT, vecHX;

  //**/ ===============================================================
  //**/ initialization
  //**/ ===============================================================
  if (initialize(X,Y) != 0)
  {
    printf("SelectiveRegression::gen2DGridData - ERROR detected.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ set up for generating regular grid data
  //**/ ===============================================================
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ===============================================================
  //**/ allocate storage for and then generate the data points
  //**/ ===============================================================
  (*NN) = totPts;
  vecXOut.setLength(totPts*2);
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
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int SelectiveRegression::gen3DGridData(double *X, double *Y, int ind1,
                                  int ind2, int ind3, double *settings, 
                                  int *NN, double **XX, double **YY)
{
  int totPts, mm, nn, pp, index;
  psVector vecXOut, vecYOut, vecXT, vecHX;

  //**/ ===============================================================
  //**/ initialization
  //**/ ===============================================================
  if (initialize(X,Y) != 0)
  {
    printf("SelectiveRegression::gen3DGridData - ERROR detected.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ set up for generating regular grid data
  //**/ ===============================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ ===============================================================
  //**/ allocate storage for and then generate the data points
  //**/ ===============================================================
  (*NN) = totPts;
  vecXOut.setLength(totPts*3);
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
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int SelectiveRegression::gen4DGridData(double *X, double *Y, int ind1, 
                            int ind2, int ind3, int ind4, double *settings, 
                            int *NN, double **XX, double **YY)
{
  int totPts, mm, nn, pp, qq, index;
  psVector vecXOut, vecYOut, vecXT, vecHX;

  //**/ ===============================================================
  //**/ initialization
  //**/ ===============================================================
  if (initialize(X,Y) != 0)
  {
    printf("SelectiveRegression::gen4DGridData - ERROR detected.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ set up for generating regular grid data
  //**/ ===============================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ ===============================================================
  //**/ allocate storage for and then generate the data points
  //**/ ===============================================================
  (*NN) = totPts;
  vecXOut.setLength(totPts*4);
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
// Evaluate a given point
// ------------------------------------------------------------------------
double SelectiveRegression::evaluatePoint(double *X)
{
  int    ii, jj, kk;
  double Y, multiplier, x;

  //**/ ===============================================================
  //**/ check to make sure initialization has been called
  //**/ ===============================================================
  if (invCovMat_.nrows() <= 0)
  {
    printf("SelectiveRegression ERROR: initialization has not been done.\n");
    return 0.0;
  }

  //**/ ===============================================================
  //**/ compute from all terms
  //**/ ===============================================================
  Y = 0.0;
  for (ii = 0; ii < numTerms_; ii++)
  {
    multiplier = 1.0;
    for (jj = 0; jj < MatCoefTerms_.getEntry(ii,0); jj++)
    {
      kk = MatCoefTerms_.getEntry(ii,jj+1); 
      if (kk < 0) x = 1.0;
      else        x = (X[kk] - VecXMeans_[kk]) / VecXStds_[kk];
      multiplier *= x;
    }
    Y += VecRegCoeffs_[ii] * multiplier;
  }
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double SelectiveRegression::evaluatePoint(int npts, double *X, double *Y)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double SelectiveRegression::evaluatePointFuzzy(double *X, double &std)
{
  int    ii, jj, kk;
  double multiplier, *Xs, Y, sd, dtmp;
  psVector vecXs;

  //**/ ===============================================================
  //**/ check to make sure initialization has been called
  //**/ ===============================================================
  if (invCovMat_.nrows() <= 0)
  {
    printf("SelectiveRegression ERROR: initialization has not been done.\n");
    return 0.0;
  }

  //**/ ===============================================================
  //**/ compute from all terms
  //**/ ===============================================================
  vecXs.setLength(numTerms_);
  Y = 0.0;
  for (ii = 0; ii < numTerms_; ii++)
  {
    multiplier = 1.0;
    for (jj = 0; jj < MatCoefTerms_.getEntry(ii,0); jj++)
    {
      kk = MatCoefTerms_.getEntry(ii,jj+1); 
      if (kk < 0) dtmp = 1.0;
      else        dtmp = (X[kk] - VecXMeans_[kk]) / VecXStds_[kk];
      multiplier *= dtmp;
    }
    Y += VecRegCoeffs_[ii] * multiplier;
    vecXs[ii] = multiplier;
  }

  //**/ ===============================================================
  //**/ compute standard deviations
  //**/ ===============================================================
  sd = 0.0;
  for (ii = 0; ii < numTerms_; ii++)
  {
    dtmp = 0.0;
    for (jj = 0; jj < numTerms_; jj++)
      dtmp += invCovMat_.getEntry(ii,jj) * vecXs[jj];
    sd += dtmp * vecXs[ii];
  }
  std = sqrt(sd) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double SelectiveRegression::evaluatePointFuzzy(int npts, double *X, 
                                               double *Y, double *Ystd)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&X[kk*nInputs_], Ystd[kk]);
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int SelectiveRegression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int SelectiveRegression::analyze(psVector VecX, psVector VecY)
{
  int    N, M, mm, nn, wlen, info, NRevised;
  double *arrayXX, SSresid, SStotal, R2, var, esum, ymax, *UU, *VV;
  char   pString[101];
  FILE   *fp;
  psMatrix eigMatT, MatA, MatXX;
  psVector eigVals;

  //**/ ===============================================================
  //**/ error checking
  //**/ ===============================================================
  if (nSamples_ <= nInputs_) 
  {
    printf("SelectiveRegression::analyze ERROR - sample size too small.\n");
    return -1;
  }
   
  //**/ ===============================================================
  //**/ display banner
  //**/ ===============================================================
  if (outputLevel_ >= 1)
  {
    printAsterisks(PL_INFO, 0);
    printf("*          Selective Regression Analysis\n");
    printf("* R-square gives a measure of the goodness of the model.\n");
    printf("* R-square should be close to 1 if it is a good model.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ===============================================================
  //**/ form regression matrix
  //**/ ===============================================================
  N = loadXMatrix(VecX, MatXX);
  M = nSamples_;

  //**/ ===============================================================
  //**/ load A matrix
  //**/ ===============================================================
  psVector VecA;
  VecA.setLength(M*N);
  arrayXX = MatXX.getMatrix1D();
  for (mm = 0; mm < M; mm++)
    for (nn = 0; nn < N; nn++)
      VecA[mm+nn*M] = sqrt(VecWghts_[mm]) * arrayXX[mm+nn*M];
  MatA.load(M, N, VecA.getDVector());

  //**/ ===============================================================
  //**/ perform SVD
  //**/ ===============================================================
  psMatrix MatU, MatV;
  psVector VecS;
  if (outputLevel_ > 3) printf("Running SVD ...\n");
  info = MatA.computeSVD(MatU, VecS, MatV);
  if (outputLevel_ > 3) 
    printf("SVD completed: status = %d (should be 0).\n",info);

  if (info != 0)
  {
    printf("* SelectiveRegression Info: dgesvd returns a nonzero (%d).\n",
           info);
    printf("* SelectiveRegression terminates further processing.\n");
    return -1;
  }

  //**/ ===============================================================
  //**/ eliminate the noise components 
  //**/ ===============================================================
  mm = 0;
  for (nn = 0; nn < N; nn++) if (VecS[nn] < 0) mm++;
  if (mm > 0)
  {
    printf("* SelectiveRegression WARNING: some of the singular values\n");
    printf("*            are < 0. May spell trouble but will\n");
    printf("*            proceed anyway (%d).\n", mm);
  }
  if (VecS[0] == 0.0) NRevised = 0;
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++)
      if (VecS[nn-1] > 0 && VecS[nn]/VecS[nn-1] < 1.0e-8) NRevised--;
  }
  if (NRevised < N)
  {
    printf("* SelectiveRegression ERROR: sample rank = %d (need %d)\n",
           NRevised, N);
    return -1;
  }
  if (psConfig_.MasterModeIsOn())
  {
    printf("SelectiveRegression: matrix singular values \n");
    printf("The VERY small ones may cause poor numerical accuracy,\n");
    printf("but not keeping them may ruin the approximation power.\n");
    printf("So, select them judiciously.\n");
    for (nn = 0; nn < N; nn++)
      printf("Singular value %5d = %e\n", nn+1, VecS[nn]);
    snprintf(pString,100,"How many to keep (1 - %d, 0 - all) ? ", N);
    NRevised = getInt(0,N,pString);
    if (NRevised == 0) NRevised = N;
    for (nn = NRevised; nn < N; nn++) VecS[nn] = 0.0;
  }
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++)
    {
      if (VecS[nn-1] > 0.0 && VecS[nn]/VecS[nn-1] < 1.0e-8)
      {
        VecS[nn] = 0.0;
        NRevised--;
      }
    }
  }
  if (NRevised < N)
    printf("Number of singular values deleted = %d\n",N-NRevised);

  //**/ ===============================================================
  //**/ compute B 
  //**/ ===============================================================
  psVector VecW, VecB;
  VecW.setLength(M+N);
  UU = MatU.getMatrix1D();
  for (mm = 0; mm < NRevised; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++)
      VecW[mm] += UU[mm*M+nn] * sqrt(VecWghts_[nn]) * VecY[nn];
  }
  for (nn = 0; nn < NRevised; nn++) VecW[nn] /= VecS[nn];
  for (nn = NRevised; nn < N; nn++) VecW[nn] = 0.0;
  VecB.setLength(N);
  VV = MatV.getMatrix1D();
  for (mm = 0; mm < N; mm++)
  {
    VecB[mm] = 0.0;
    for (nn = 0; nn < NRevised; nn++) VecB[mm] += VV[nn+mm*N] * VecW[nn];
  }

  //**/ ===============================================================
  //**/ store eigenvectors and eigenvalues
  //**/ ===============================================================
  eigMatT.load(N, N, VV);
  eigVals.load(VecS);
  for (nn = 0; nn < N; nn++) eigVals[nn] = pow(eigVals[nn], 2.0);

  //**/ ===============================================================
  //**/ compute residual
  //**/ ===============================================================
  if (psConfig_.MasterModeIsOn())
  {
    fp = fopen("selective_regression_error.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n",
              __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "%% This file contains errors of each data point.\n");
  }
  else fp = NULL;

  esum = ymax = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++)
      VecW[mm] = VecW[mm] + arrayXX[mm+nn*nSamples_] * VecB[nn];
    VecW[mm] = VecW[mm] - VecY[mm];
    esum = esum + VecW[mm] * VecW[mm] * VecWghts_[mm];
    if (fp != NULL) 
      fprintf(fp, "%6d %24.16e\n",mm+1,VecW[mm]*sqrt(VecWghts_[mm]));
    if (PABS(VecY[mm]) > ymax) ymax = PABS(VecY[mm]);
  }
  esum /= (double) nSamples_;
  esum = sqrt(esum);
  printf("* SelectiveRegression:: LS mean error = %11.4e (max=%11.4e)\n",
         esum, ymax); 

  if (fp != NULL)
  {
    fclose(fp);
    printf("FILE selective_regression_error.m contains data errors.\n");
  }

  //**/ ===============================================================
  //**/ form compute SS statistics 
  //**/ ===============================================================
  computeSS(MatXX, VecY, VecB, SSresid, SStotal);
  if (SStotal == 0) R2 = 1.0;
  else              R2  = 1.0 - SSresid / SStotal;
  if (nSamples_ > N) var = SSresid / (double) (nSamples_ - N);
  else               var = 0.0;
  if (var < 0)
  { 
    if (PABS(var) > 1.0e-12)
    {
      printf("SelectiveRegression WARNING: variance < 0.\n");
      printf("         Temporarily absolutize var (may have problems).\n");
      var = PABS(var); 
    }
    else var = 0;
  }
  VecRegCoeffs_.load(VecB);

  //**/ ===============================================================
  //**/ find variance of each coefficient 
  //**/ ===============================================================
  computeCoeffVariance(eigMatT, eigVals, var);
  psVector VecBstd;
  VecBstd.setLength(N);
  for (mm = 0; mm < N; mm++)
    VecBstd[mm] = sqrt(invCovMat_.getEntry(mm,mm));

  //**/ ===============================================================
  //**/ print out regression coefficients 
  //**/ ===============================================================
  if (outputLevel_ >= 0)
  {
    if (outputLevel_ > 0) printRC(VecB, VecBstd, MatXX, VecY);
    printf("* SelectiveRegression model R-square = %12.4e\n",R2);
    printf("* adjusted   R-square = %12.4e\n",
           1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
    if (outputLevel_ > 1) printSRC(VecX, VecB, SStotal);
  }

  //**/ ===============================================================
  //**/ create standalone interpolator
  //**/ ===============================================================
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"/* ************************************************/\n");
    fprintf(fp,"/* Selective regression interpolator from PSUADE. */\n");
    fprintf(fp,"/* ===============================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   3. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ===============================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
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
    fprintf(fp,"/* ===============================================*/\n");
    fprintf(fp,"/* Selective regression interpolation function    */\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ===============================================*/\n");
    fprintf(fp,"static int\n");
    fprintf(fp,"CoefTerms[%d][%d] = \n", numTerms_, nInputs_+1);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numTerms_; mm++)
    {
      fprintf(fp,"  { %d", MatCoefTerms_.getEntry(mm,0));
      for (nn = 0; nn < MatCoefTerms_.getEntry(mm,0); nn++)
        fprintf(fp," , %d", MatCoefTerms_.getEntry(mm,nn+1));
      for (nn = MatCoefTerms_.getEntry(mm,0); nn < nInputs_; nn++)
        fprintf(fp," , 0");
      fprintf(fp," },\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"XStat[%d][2] = \n", nInputs_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < nInputs_; mm++)
      fprintf(fp,"  { %24.16e , %24.16e},\n",VecXMeans_[mm],VecXStds_[mm]);
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"invCovMat[%d][%d] = \n", numTerms_, numTerms_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numTerms_; mm++)
    {
       fprintf(fp,"  {");
       for (nn = 0; nn < numTerms_-1; nn++)
          fprintf(fp," %24.16e,", invCovMat_.getEntry(mm,nn));
       fprintf(fp," %24.16e },\n", invCovMat_.getEntry(mm,numTerms_-1));
    }
    fprintf(fp," };\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"regCoefs[%d] = \n", numTerms_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < numTerms_; mm++)
      fprintf(fp," %24.16e,", VecRegCoeffs_[mm]);
    fprintf(fp,"};\n");
    fprintf(fp,"/* ===============================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
    fprintf(fp,"  int    ii,jj,ll,mm,nterms=%d,nInps=%d;\n",
            numTerms_,nInputs_);
    fprintf(fp,"  double y, mult, *xt, *x, std, dtmp;\n");
    fprintf(fp,"  xt = (double *) malloc(%d * sizeof(double));\n",numTerms_);
    fprintf(fp,"  for (ii = 0; ii < npts; ii++) {\n");
    fprintf(fp,"    x = &(X[ii* nInps]);\n");
    fprintf(fp,"    y = 0.0;\n");
    fprintf(fp,"    for (jj = 0; jj < nterms; jj++) {\n");
    fprintf(fp,"      mult = 1;\n");
    fprintf(fp,"      for (ll = 0; ll < CoefTerms[jj][0]; ll++){\n");
    fprintf(fp,"        mm = CoefTerms[jj][ll+1];\n");
    fprintf(fp,"        if (mm >= 0)\n");
    fprintf(fp,"          mult=mult*(x[mm]-XStat[mm][0])/XStat[mm][1];\n");
    fprintf(fp,"      }\n");
    fprintf(fp,"      y += regCoefs[jj] * mult;\n");
    fprintf(fp,"      xt[jj] = mult;\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    Y[ii] = y * %e + %e;\n", YStd_, YMean_);
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for (jj = 0; jj < %d; jj++) {\n",numTerms_);
    fprintf(fp,"      dtmp = 0.0;\n");
    fprintf(fp,"      for (ll = 0; ll < %d; ll++)\n",numTerms_);
    fprintf(fp,"        dtmp += invCovMat[jj][ll] * xt[ll];\n");
    fprintf(fp,"      std += dtmp * xt[jj];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    std = sqrt(std);\n");
    fprintf(fp,"    S[ii] = std;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  free(xt);\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* ===============================================*/\n");
    fclose(fp);
    printf("FILE psuade_rs.info contains the polynomial functional\n");
    printf("     form.\n");
  }
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# Selective regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"CoefTerms = [\n");
    for (mm = 0; mm < numTerms_; mm++)
    {
       fprintf(fp," [ %d", MatCoefTerms_.getEntry(mm,0));
       for (nn = 0; nn < MatCoefTerms_.getEntry(mm,0); nn++)
          fprintf(fp,", %d", MatCoefTerms_.getEntry(mm,nn+1));
       for (nn = MatCoefTerms_.getEntry(mm,0); nn < nInputs_; nn++)
          fprintf(fp,", 0");
       fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"XStat = [\n");
    for (mm = 0; mm < nInputs_; mm++)
       fprintf(fp," [ %24.16e, %24.16e ],\n",VecXMeans_[mm],VecXStds_[mm]);
    fprintf(fp,"]\n");
    fprintf(fp,"invCovMat = [\n");
    for (mm = 0; mm < numTerms_; mm++)
    {
      fprintf(fp," [ %24.16e", invCovMat_.getEntry(mm,0));
      for (nn = 1; nn < numTerms_; nn++)
        fprintf(fp,", %24.16e", invCovMat_.getEntry(mm,nn));
      fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"regCoefs = [\n");
    for (mm = 0; mm < numTerms_-1; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp," %24.16e ]\n", VecRegCoeffs_[numTerms_-1]);
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# Regression interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(X): \n");
    fprintf(fp,"  nSamp = int(len(X) / %d)\n",nInputs_);
    fprintf(fp,"  Xs = %d * [0.0]\n", numTerms_);
    fprintf(fp,"  Xt = %d * [0.0]\n", nInputs_);
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  for ss in range(nSamp) : \n");
    fprintf(fp,"    for ii in range(%d) : \n", nInputs_);
    fprintf(fp,"      Xt[ii] = X[ss*%d+ii]\n",nInputs_);
    fprintf(fp,"    Y = 0.0\n");
    fprintf(fp,"    for jj in range(%d) : \n", numTerms_);
    fprintf(fp,"      mult = 1;\n");
    fprintf(fp,"      for ll in range(CoefTerms[jj][0]) : \n");
    fprintf(fp,"        mm = CoefTerms[jj][ll+1]\n");
    fprintf(fp,"        if (mm >= 0) :\n");
    fprintf(fp,"          mult=mult*(Xt[mm]-XStat[mm][0])/XStat[mm][1]\n");
    fprintf(fp,"      Y += regCoefs[jj] * mult\n");
    fprintf(fp,"      Xs[jj] = mult\n");
    fprintf(fp,"    Ys[2*ss] = Y * %e + %e;\n", YStd_, YMean_);
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for jj in range(%d): \n", numTerms_);
    fprintf(fp,"      dtmp = 0.0\n");
    fprintf(fp,"      for kk in range(%d): \n", numTerms_);
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
    printf("FILE psuade_rs.py contains the final selective polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int SelectiveRegression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int      M, N, mm, nn, kk;
  double   multiplier;
  psVector VecXX;

  M = nSamples_;
  N = numTerms_;
  VecXX.setLength(M * N);
  for (mm = 0; mm < M; mm++)
  {
    for (nn = 0; nn < N; nn++)
    {
      multiplier = 1.0;
      for (kk = 0; kk < MatCoefTerms_.getEntry(nn,0); kk++)
        if (MatCoefTerms_.getEntry(nn,kk+1) >= 0)
          multiplier *= VecX[mm*nInputs_+MatCoefTerms_.getEntry(nn,kk+1)];
      VecXX[mm+nn*M] = multiplier;
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
int SelectiveRegression::computeSS(psMatrix MatXX, psVector VecY,
                            psVector VecB, double &SSresid, double &SStotal)
{
  int    nn, mm, N;
  double ymean, SSreg, SSresidCheck, ddata, rdata, *arrayXX;
                                                                                
  N = VecB.length();
  arrayXX = MatXX.getMatrix1D();

  SSresid = SStotal = ymean = SSreg = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
     ymean += (sqrt(VecWghts_[mm]) * VecY[mm]);
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) 
      ddata += (arrayXX[mm+nn*nSamples_] * VecB[nn]);
    rdata = VecY[mm] - ddata;
    SSresid += rdata * VecY[mm] * VecWghts_[mm];
    SSresidCheck += rdata * rdata * VecWghts_[mm];
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += VecWghts_[mm] * (VecY[mm] - ymean) * (VecY[mm] - ymean);
  if (outputLevel_ > 0)
  {
    printf("* SelectiveRegression: SStot  = %24.16e\n", SStotal);
    printf("* SelectiveRegression: SSreg  = %24.16e\n", SSreg);
    printf("* SelectiveRegression: SSres  = %24.16e\n", SSresid);
    printf("* SelectiveRegression: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (outputLevel_ > 0 && nSamples_ != N)
  {
    printf("* SelectiveRegression: eps(Y) = %24.16e\n",
           SSresidCheck/(nSamples_-N));
  }
  return 0;
}

// *************************************************************************
// compute coefficient variance ((diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int SelectiveRegression::computeCoeffVariance(psMatrix &eigMatT,
                                psVector &eigVals, double var)
{
  int      ii, jj, nRows;
  double   invEig, dtmp;
  psMatrix tMat;

  nRows = eigMatT.nrows();
  tMat.setDim(nRows, nRows);

  //**/ compute var * V * S^{-2}
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

  //**/ compute (var * V * S^{-2}) * V^T
  tMat.matmult(eigMatT, invCovMat_);
  return 0;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int SelectiveRegression::printRC(psVector VecB, psVector VecBstd, 
                                 psMatrix MatXX, psVector VecY)
{
  int    ii, jj, maxTerms, N;
  double coef, *arrayXX;
  char   fname[200];
  FILE   *fp;

  maxTerms = 0;
  for (jj = 0; jj < numTerms_; jj++) 
    if (MatCoefTerms_.getEntry(jj,0) > maxTerms) 
      maxTerms = MatCoefTerms_.getEntry(jj,0);

  printf("*----------------------------------------------------------*\n");
  printf("*      ");
  for (jj = 0; jj < maxTerms; jj++) printf("    ");
  printf("   coefficient   std. error   t-value\n");

  //**/ =================================================================
  //**/ print out the linear coefficients 
  //**/ =================================================================
  N = VecB.length();
  arrayXX = MatXX.getMatrix1D();
  for (ii = 0; ii < numTerms_; ii++)
  {
    if (PABS(VecBstd[ii]) < 1.0e-15) coef = 0.0;
    else                             coef = VecB[ii] / VecBstd[ii]; 
    if (PABS(coef) > 0.0)
    {
      printf("* Input");
      for (jj = 0; jj < MatCoefTerms_.getEntry(ii,0); jj++)
        printf(" %2d ", MatCoefTerms_.getEntry(ii,jj+1)+1);
      for (jj = MatCoefTerms_.getEntry(ii,0); jj < maxTerms; jj++) 
        printf("    ");
      printf("= %12.4e %12.4e %12.4e\n", VecB[ii], VecBstd[ii], coef);
    }
  }
  strcpy(fname, "dataVariance");
  printDashes(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ print to file
  //**/ ---------------------------------------------------------------
  if (psConfig_.MasterModeIsOn())
  {
    fp = fopen(fname, "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n", 
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "data number     data         standard error\n");
    for (ii = 0; ii < nSamples_; ii++)
    {
      coef = 0.0;
      for (jj = 0; jj < N; jj++) 
         coef += arrayXX[jj*nSamples_+ii] * VecBstd[jj];
      if (coef < 0)
        fprintf(fp,"%7d         %12.4e %12.4e\n",ii+1,VecY[ii],-sqrt(-coef));
      else
        fprintf(fp,"%7d         %12.4e %12.4e\n",ii+1,VecY[ii],sqrt(coef));
    }
    fclose(fp);
  }
  return 0;
}

// *************************************************************************
// print standardized regression coefficients
// -------------------------------------------------------------------------
int SelectiveRegression::printSRC(psVector VecX,psVector VecB,double SStotal)
{
  int    nn, mm, ii, index, length, maxTerms;
  double denom, xmean, coef, Bmax, coef1;
  psVector VecB2;

  printEquals(PL_INFO, 0);
  printf("* Standardized Regression Coefficients (SRC)\n");
  printf("* When R-square is acceptable (order assumption holds), the*\n");
  printf("* absolute values of SRCs provide variable importance.     *\n"); 
  printDashes(PL_INFO, 0);
  printf("* based on nSamples = %d\n", nSamples_);

  maxTerms = 0;
  for (ii = 0; ii < numTerms_; ii++) 
    if (MatCoefTerms_.getEntry(ii,0) > maxTerms) 
      maxTerms = MatCoefTerms_.getEntry(ii,0);
  VecB2.setLength(nSamples_);
  denom = sqrt(SStotal / (double) (nSamples_ - 1));
  Bmax  = 0.0;
  for (nn = 0; nn < numTerms_; nn++)
  {
    coef = 1.0;
    length = MatCoefTerms_.getEntry(nn,0);
    for (ii = 0; ii < length; ii++)
    {
      xmean = 1.0;
      coef1 = 0.0;
      index = MatCoefTerms_.getEntry(nn,ii+1);
      if (index >= 0)
      {
        for (mm = 0; mm < nSamples_; mm++) 
          xmean += VecX[mm*nInputs_+index];
        xmean /= (double) nSamples_;
        for (mm = 0; mm < nSamples_; mm++)
          coef1 += (VecX[mm*nInputs_+index]-xmean)*
                   (VecX[mm*nInputs_+index]-xmean);
        coef1 = sqrt(coef1 / (double) (nSamples_ - 1));
        coef *= coef1;
      }
    }
    VecB2[nn] = VecB[nn] * coef / denom;
    if (PABS(VecB2[nn]) > Bmax) Bmax = PABS(VecB2[nn]);
  }
  for (nn = 0; nn < numTerms_; nn++)
  {
    if (PABS(VecB2[nn]) > 1.0e-12 * Bmax)
    {
      printf("* Input");
      for (ii = 0; ii < MatCoefTerms_.getEntry(nn,0); ii++)
        printf(" %2d ",MatCoefTerms_.getEntry(nn,ii+1)+1);
      for (ii = MatCoefTerms_.getEntry(nn,0); ii < maxTerms; ii++) 
        printf("    ");
      printf("= %12.4e\n",VecB2[nn]);
    }
  }
  printAsterisks(PL_INFO, 0);
  return 0;
}

