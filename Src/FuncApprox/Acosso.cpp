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
// Functions for the class Acosso
// Reference: This class uses Curt Storlie's (LANL) Acosso method (in R)
// DATE   : 2015
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "Acosso.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class Acosso
// ------------------------------------------------------------------------
Acosso::Acosso(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int  errFlag;
  char lineIn[10001], winput[1000];
  FILE *fp;

  faID_ = PSUADE_RS_ACOSSO;

  //**/ test to see if R with quadprog can be called
  fp = fopen("RTest","w");
  if (fp == NULL)
  {
    printf("Acosso constructor ERROR: cannot open file.\n");
    printf("       Check access/permission to directory\n");
    exit(1);
  }
  fprintf(fp,"library(\"quadprog\")\n");
  fprintf(fp,"quit()\n");
  fprintf(fp,"n\n");
  fclose(fp);
  system("R CMD BATCH RTest");
  fp = fopen("RTest.Rout","r");
  if (fp == NULL)
  {
    strcpy(Rpath_, "NONE");
    printf("Acosso constructor INFO: R not found.\n");
    unlink("RTest");
    exit(1);
  }
  else
  {
    errFlag = 0;
    while (feof(fp) == 0)
    {
      fgets(lineIn, 10000, fp);
      sscanf(lineIn, "%s", winput);
      if (!strcmp(winput, "Error")) errFlag = 1;
    }
    fclose(fp);
    unlink("RTest.Rout");
    if (errFlag == 1)
    {
      printf("Acosso constructor ERROR: R quadprog package not found.\n");
      printf("  INFO: to obtain quadprog, launch R and use\n");
      printf("          install.packages(\"quadprog\")\n");
      unlink("RTest");
      exit(1);
    }
    strcpy(Rpath_, "R");
  }
  unlink("RTest");
  initialized_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
Acosso::~Acosso()
{
  vecSamInputs_.clean();
  vecSamOutputs_.clean();
  FILE *fp = fopen("acosso.R", "r");
  if (fp != NULL) 
  {
    fclose(fp);
    unlink("acosso.R");
  }
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int Acosso::initialize(double *XIn, double *YIn)
{
  int ii, jj;

  if (initialized_ == 0)
  {
    genAcosso();
    vecSamInputs_.setLength(nSamples_ * nInputs_);
    vecSamOutputs_.setLength(nSamples_);
  }
  for (ii = 0; ii < nSamples_; ii++)
  {
    for (jj = 0; jj < nInputs_; jj++)
      vecSamInputs_[ii*nInputs_+jj] = XIn[ii*nInputs_+jj];
    vecSamOutputs_[ii] = YIn[ii];
  }
  initialized_ = 1;
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int Acosso::genNDGridData(double *XIn, double *YIn, int *N, double **X2, 
                          double **Y2)
{
  int    totPts, ii, jj;
  double *XX;
  psVector  vecYY;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ initialize 
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || X2 == NULL || Y2 == NULL) return 0;
  
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(N, &XX);
  if ((*N) == 0) return 0;
  totPts = (*N);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Acosso interpolation begins....\n");
  vecIndices.setLength(nInputs_);
  vecYY.setLength(totPts);
  for (ii = 0; ii < nInputs_; ii++) vecIndices[ii] = ii;
  runAcosso(totPts, XX, vecYY.getDVector(), nInputs_, 
            vecIndices.getIVector());
  if (outputLevel_ >= 1) printf("Acosso interpolation completed.\n");
  (*X2) = XX;
  (*Y2) = vecYY.takeDVector();
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int Acosso::gen1DGridData(double *XIn, double *YIn, int ind1,
                          double *settings, int *N, double **X2, double **Y2)
{
  int    totPts, ii, kk, iOne=1;
  double *XX, *YY, HX;
  psVector vecXX, vecYY, vecXT;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || X2 == NULL || Y2 == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXX.setLength(totPts);
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    vecXX[ii] = HX * ii + VecLBs_[ind1];
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Acosso interpolation begins....\n");
  vecYY.setLength(totPts);
  vecIndices.setLength(1);
  vecIndices[0] = ind1;
  runAcosso(totPts, vecXT.getDVector(), vecYY.getDVector(), iOne, 
            vecIndices.getIVector());
  if (outputLevel_ >= 1) printf("Acosso interpolation completed.\n");
  (*N) = totPts;
  (*X2) = vecXX.takeDVector();
  (*Y2) = vecYY.takeDVector();
  return 0;
}

// ************************************************************************
// Generate 2D results 
// ------------------------------------------------------------------------
int Acosso::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                          double *settings, int *N, double **X2, double **Y2)
{
  int totPts, ii, jj, kk, index, iTwo=2;
  psVector vecXX, vecYY, vecXT, vecHX;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || X2 == NULL || Y2 == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXX.setLength(2*totPts);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        vecXT[index*nInputs_+kk] = settings[kk]; 
      vecXX[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      vecXX[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
      vecXT[index*nInputs_+ind1] = vecXX[index*2];
      vecXT[index*nInputs_+ind2] = vecXX[index*2+1];
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Acosso interpolation begins....\n");
  vecYY.setLength(totPts);
  vecIndices.setLength(2);
  vecIndices[0] = ind1;
  vecIndices[1] = ind2;
  runAcosso(totPts, vecXT.getDVector(), vecYY.getDVector(), iTwo, 
            vecIndices.getIVector());
  if (outputLevel_ >= 1) printf("Acosso interpolation completed.\n");
  (*N)  = totPts;
  (*X2) = vecXX.takeDVector();
  (*Y2) = vecYY.takeDVector();
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int Acosso::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                          int ind3, double *settings, int *N, double **X2, 
                          double **Y2)
{
  int    totPts, ii, jj, kk, ll, index, iThree=3;
  psVector vecHX, vecXX, vecXT, vecYY;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || X2 == NULL || Y2 == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXX.setLength(totPts * 3);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (kk = 0; kk < nInputs_; kk++) 
          vecXX[index*nInputs_+kk] = settings[kk]; 
        vecXX[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        vecXX[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        vecXX[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
        vecXT[index*nInputs_+ind1] = vecXX[index*3];
        vecXT[index*nInputs_+ind2] = vecXX[index*3+1];
        vecXT[index*nInputs_+ind3] = vecXX[index*3+2];
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Acosso interpolation begins....\n");
  vecYY.setLength(totPts);
  vecIndices.setLength(3);
  vecIndices[0] = ind1;
  vecIndices[1] = ind2;
  vecIndices[2] = ind3;
  runAcosso(totPts, vecXT.getDVector(), vecYY.getDVector(), iThree, 
            vecIndices.getIVector());
  if (outputLevel_ >= 1) printf("Acosso interpolation completed.\n");
  (*N)  = totPts;
  (*X2) = vecXX.takeDVector();
  (*Y2) = vecYY.takeDVector();
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int Acosso::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                          int ind3, int ind4, double *settings, int *N, 
                          double **X2, double **Y2)
{
  int    totPts, ii, jj, kk, ll, mm, index, iFour=4;
  psVector vecHX, vecXX, vecXT, vecYY;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || X2 == NULL || Y2 == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXX.setLength(totPts*4);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        for (mm = 0; mm < nPtsPerDim_; mm++) 
        {
          index = ii*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          for (kk = 0; kk < nInputs_; kk++) 
            vecXT[index*nInputs_+kk] = settings[kk]; 
          vecXX[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          vecXX[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          vecXX[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          vecXX[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
          vecXT[index*nInputs_+ind1] = vecXX[index*4];
          vecXT[index*nInputs_+ind2] = vecXX[index*4+1];
          vecXT[index*nInputs_+ind3] = vecXX[index*4+2];
          vecXT[index*nInputs_+ind4] = vecXX[index*4+3];
        }
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Acosso interpolation begins....\n");
  vecYY.setLength(totPts);
  vecIndices.setLength(4);
  vecIndices[0] = ind1;
  vecIndices[1] = ind2;
  vecIndices[2] = ind3;
  vecIndices[3] = ind4;
  runAcosso(totPts, vecXT.getDVector(), vecYY.getDVector(), iFour, 
            vecIndices.getIVector());
  if (outputLevel_ >= 1) printf("Acosso interpolation completed.\n");
  (*N)  = totPts;
  (*X2) = vecXX.takeDVector();
  (*Y2) = vecYY.takeDVector();
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double Acosso::evaluatePoint(double *X)
{
  int    iOne=1, iZero=0;
  double Y=0.0;
  runAcosso(iOne, X, &Y, iZero, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double Acosso::evaluatePoint(int npts, double *X, double *Y)
{
  int iZero=0;
  runAcosso(npts, X, Y, iZero, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double Acosso::evaluatePointFuzzy(double *X, double &std)
{
  double Y = evaluatePoint(X);
  std = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double Acosso::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystd)
{
   evaluatePoint(npts, X, Y);
   for (int ii = 0; ii < npts; ii++) Ystd[ii] = 0.0;
   return 0.0;
}

// ************************************************************************
// run Storlie's Acosso code
// ------------------------------------------------------------------------
int Acosso::runAcosso(int nPoints, double *XN, double *YN, int nInps, 
                      int *indices)
{
  int  ii, ss, count;
  char pString[5000];
  FILE *fp;

  //**/ open a file to be run with R
  fp = fopen("psuade_acosso.R", "w");
  if (fp == NULL)
  {
    printf("Acosso ERROR: cannot open file.\n");
    exit(1);
  }
  fprintf(fp,"source(\"acosso.R\")\n");
  fprintf(fp,"set.seed(22)\n");
  fprintf(fp,"n <- %d\n", nSamples_);
  fprintf(fp,"p <- %d\n", nInputs_);
  fprintf(fp,"X = matrix(\n");
  fprintf(fp,"c(");
  //**/ write the sample data 
  count = 0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (ss = 0; ss < nSamples_; ss++)
    {
      if (ss == nSamples_-1 && ii == nInputs_-1)
        fprintf(fp,"%e),\n", vecSamInputs_[ss*nInputs_+ii]);
      else
      {
        fprintf(fp,"%e, ", vecSamInputs_[ss*nInputs_+ii]);
        count++;
        if (count >= 10)
        {
          count = 0;
          fprintf(fp, "\n");
        } 
      } 
    } 
  } 
  fprintf(fp,"nrow=%d,", nSamples_);
  fprintf(fp,"ncol=%d)\n", nInputs_);
  fprintf(fp,"Y = c(\n");
  count = 0;
  for (ss = 0; ss < nSamples_; ss++)
  {
    if (ss == nSamples_-1)
      fprintf(fp,"%e)\n", vecSamOutputs_[ss]);
    else
    {
      fprintf(fp,"%e, ", vecSamOutputs_[ss]);
      count++;
      if (count >= 10)
      {
        count = 0;
        fprintf(fp, "\n");
      } 
    } 
  } 
  //**/ write the prediction sample
  int    i1, i2, i3, i4, n1, n2, n3, n4;
  double hx;
  if (nInps == 1)
  {
    i1 = indices[0];
    n1 = nPtsPerDim_;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (ii != i1)
      {
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",XN[ii],n1,n1); 
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", VecLBs_[i1]);
        fprintf(fp,"V = matrix(v,%d,1)\n",n1);
      }
      if (ii == 0)
      {
        fprintf(fp,"XN <- V\n");
      }
      else
      {
        fprintf(fp,"XN = cbind(XN, V)\n");
      }
    }
  }
  else if (nInps == 2)
  {
    i1 = indices[0];
    i2 = indices[1];
    n1 = nPtsPerDim_;
    n2 = nPtsPerDim_ * nPtsPerDim_;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (ii != i1 && ii != i2)
      {
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",XN[ii],n2,n2);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i1]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i2)
      {
        hx = (VecUBs_[i2] - VecLBs_[i2]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", VecLBs_[i2]);
        fprintf(fp,"V = matrix(v,%d,1)\n",n1);
        fprintf(fp,"v2 <- V \n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   V = rbind(V, v2)\n");
        fprintf(fp,"}\n");
      }
      if (ii == 0) fprintf(fp,"XN <- V\n");
      else         fprintf(fp,"XN = cbind(XN, V)\n");
    }
  }
  else if (nInps == 3)
  {
    i1 = indices[0];
    i2 = indices[1];
    i3 = indices[2];
    n1 = nPtsPerDim_;
    n2 = nPtsPerDim_ * nPtsPerDim_;
    n3 = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (ii != i1 && ii != i2 && ii != i3)
      {
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",XN[ii],n3,n3);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i1]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n2,n2);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n2,n2);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i2)
      {
        hx = (VecUBs_[i2] - VecLBs_[i2]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i2]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
        fprintf(fp,"VV <- V\n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   V  = rbind(V, VV)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i3)
      {
        hx = (VecUBs_[i3] - VecLBs_[i3]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", VecLBs_[i3]);
        fprintf(fp,"V = matrix(v,%d,1)\n",n1);
        fprintf(fp,"v2 <- V \n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n2-1);
        fprintf(fp,"   V = rbind(V, v2)\n");
        fprintf(fp,"}\n");
      }
      if (ii == 0)
      {
        fprintf(fp,"XN <- V\n");
      }
      else
      {
        fprintf(fp,"XN = cbind(XN, V)\n");
      }
    }
  }
  else if (nInps == 4)
  {
    i1 = indices[0];
    i2 = indices[1];
    i3 = indices[2];
    i4 = indices[3];
    n1 = nPtsPerDim_;
    n2 = nPtsPerDim_ * nPtsPerDim_;
    n3 = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
    n4 = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (ii != i1 && ii != i2 && ii != i3 && ii != i4)
      {
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",XN[ii],n4,n4);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i1]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n3,n3);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n3,n3);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i2)
      {
        hx = (VecUBs_[i2] - VecLBs_[i2]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i2]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n2,n2);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n2,n2);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
        fprintf(fp,"VV <- V\n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   V  = rbind(V, VV)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i3)
      {
        hx = (VecUBs_[i3] - VecLBs_[i3]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", VecLBs_[i3]);
        fprintf(fp,"V = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
        fprintf(fp,"   hx <- hx + %e \n", hx);
        fprintf(fp,"   v1 = matrix(rep(hx,%d),%d,1)\n",n1,n1);
        fprintf(fp,"   V  = rbind(V, v1)\n");
        fprintf(fp,"}\n");
        fprintf(fp,"VV <- V\n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n2-1);
        fprintf(fp,"   V  = rbind(V, VV)\n");
        fprintf(fp,"}\n");
      }
      else if (ii == i4)
      {
        hx = (VecUBs_[i4] - VecLBs_[i4]) / (n1 - 1); 
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", VecLBs_[i4]);
        fprintf(fp,"V = matrix(v,%d,1)\n",n1);
        fprintf(fp,"v2 <- V \n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n3-1);
        fprintf(fp,"   V = rbind(V, v2)\n");
        fprintf(fp,"}\n");
      }
      if (ii == 0)
      {
        fprintf(fp,"XN <- V\n");
      }
      else
      {
        fprintf(fp,"XN = cbind(XN, V)\n");
      }
    }
  }
  else
  {
    fprintf(fp,"XN = matrix(\n");
    fprintf(fp,"c(");
    count = 0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nPoints; ss++)
      {
        if (ss == nPoints-1 && ii == nInputs_-1)
          fprintf(fp,"%e),\n", XN[ss*nInputs_+ii]);
        else
        {
          fprintf(fp,"%e, ", XN[ss*nInputs_+ii]);
          count++;
          if (count >= 10)
          {
            count = 0;
            fprintf(fp, "\n");
          } 
        }
      }
    }
    fprintf(fp,"nrow=%d,", nPoints);
    fprintf(fp,"ncol=%d)\n", nInputs_);
  } 
  fprintf(fp,"acosso.fit <- acosso(X, Y, order=2, wt.pow=1, cv='gcv')\n");
  fprintf(fp,"acosso.pred <- predict.acosso(XN, acosso.fit)\n");
  fprintf(fp,"write(acosso.pred,\"psuade_acosso_data\")\n");
  fprintf(fp,"quit()\n");
  fclose(fp);
  //**/ run R script
  if (!strcmp(Rpath_, "NONE"))
  {
    printf("Acosso ERROR: R not found.\n");
    printf("Please enter the full path of R (loaded with quadprog) : ");
    scanf("%s", Rpath_);
    fgets(pString, 500, stdin);
    fp = fopen(Rpath_,"r");
    if (fp == NULL)
    {
      printf("Acosso ERROR: R still not found.\n");
      exit(1);
    }
  }
  snprintf(pString,1000,"%s CMD BATCH psuade_acosso.R", Rpath_);
  system(pString);
  //**/ read output
  fp = fopen("psuade_acosso_data", "r");
  if (fp == NULL)
  {
    printf("Acosso ERROR: cannot open output file.\n");
    printf("For diagnosis, run the following and track errors:\n");
    printf("> %s CMD BATCH psuade_acosso.R\n", Rpath_);
    exit(1);
  }
  for (ss = 0; ss < nPoints; ss++) fscanf(fp, "%lg", &YN[ss]);
  fclose(fp);
  unlink("psuade_acosso_data");
  unlink("psuade_acosso.R");
  unlink("psuade_acosso.Rout");
  return 0;
}

// ************************************************************************
// generate Acosso code
// ------------------------------------------------------------------------
int Acosso::genAcosso()
{
  int  ii, ss, count;
  char pString[5000];
  FILE *fp;

  fp = fopen("acosso.R", "w");
  if (fp == NULL)
  {
    printf("Acosso genAcosso ERROR: cannot open file.\n");
    exit(1);
  }
  fprintf(fp,"library(\"MASS\")\n");
  fprintf(fp,"library(\"quadprog\")\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"############### ACOSSO Function ##########################\n");
  fprintf(fp,"# CODE SOURCE: CURT STORLIE (LANL) #######################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"acosso <- function(X,y,order=2,wt.pow=1,cv='bic',w,lambda.0,\n");
  fprintf(fp,"          gcv.pen=1.01,categorical=\"auto\",min.distinct=2){\n");
  fprintf(fp,"\n");
  fprintf(fp,"########## INPUTS ########################################\n");
  fprintf(fp,"## X        - a matrix of predictors\n");
  fprintf(fp,"## y        - a vector of responses\n");
  fprintf(fp,"## order    - the order of interactions to consider (1 or 2)\n");
  fprintf(fp,"## wt.pow   - the weights used in the adaptive penalty are ||P^j||^{-wt.pow} \n");
  fprintf(fp,"##            wt.pow=0 is the COSSO\n");
  fprintf(fp,"## cv       - the method used to select the tuning parameter M from the ACOSSO\n");
  fprintf(fp,"##            paper. (the tuning paramter is actually called K in this code).\n");
  fprintf(fp,"##            Options are '5cv', 'gcv', and 'bic', or a numeric value to use \n");
  fprintf(fp,"##            for M\n");
  fprintf(fp,"## w        - optional vector to use for w (only used if 'cv' is numeric)\n");
  fprintf(fp,"## lambda.0 - optional value to use for lambda.0(only used if 'cv' is numeric)\n");
  fprintf(fp,"## categorical - vector containg the columns to be treated as categorical\n");
  fprintf(fp,"## min.distinct - minimum number of distinct values to treat a predictor as\n");
  fprintf(fp,"##                continuous (instead of categorical).  Only used if\n");
  fprintf(fp,"##                categorical=\"auto\".\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"########## OUTPUTS #######################################\n");
  fprintf(fp,"## c.hat   - coefficients of the Kernel representation f(x)=mu+sum K(x_i,x)*c\n");
  fprintf(fp,"## mu.hat  - estimated constant in above representation\n");
  fprintf(fp,"## y.hat   - vector of the predicted y's\n");
  fprintf(fp,"## res     - vector of the residuals\n");
  fprintf(fp,"## dfmod   - approximate degrees of freedom of the fit\n");
  fprintf(fp,"## w       - adaptive weights used in the estimation\n");
  fprintf(fp,"## gcv     - gcv score\n");
  fprintf(fp,"## bic     - bic score\n");
  fprintf(fp,"## theta   - estimated theta vector\n");
  fprintf(fp,"## Rsq     - R^2 = 1-SSE/SSTOT\n");
  fprintf(fp,"## Gram    - Gram matrix.  The (i,j)th element is K(x_i,x_j)\n");
  fprintf(fp,"## y.mat   - Matrix of fitted y's for each functional component \n");
  fprintf(fp,"##           (i.e. y.hat = mu.hat + sum(y.mat))\n");
  fprintf(fp,"## X       - the original matrix of predictors\n");
  fprintf(fp,"## y	   - the original vector of inputs\n");
  fprintf(fp,"## rescale - Did the data need to be rescaled to (0,1).  Used for prediction.\n");
  fprintf(fp,"## order   - the order of interactions specified\n");
  fprintf(fp,"## K	   - the value of K chosen by cv\n");
  fprintf(fp,"## lambda.0- the value used for lambda.0\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"if(!is.numeric(cv)){\n");
  fprintf(fp,"  ans.cv <- venus.cv(X, y, order=order, cv=cv, wt.pow=wt.pow, w='L2', \n");
  fprintf(fp,"             f.est='trad', seed=110, K.lim='data', nK=5, gcv.pen=gcv.pen, \n");
  fprintf(fp,"             rel.K=5E-2, nlambda=5, lambda.lim=c(1E-10, 1E3), rel.lambda=5E-2,\n");
  fprintf(fp,"             rel.theta=1E-2, maxit=2, init.cv='gcv', cat.pos=categorical, \n");
  fprintf(fp,"             min.distinct=min.distinct)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  lambda.0 <- ans.cv$lambda.0\n");
  fprintf(fp,"  K <- ans.cv$K\n");
  fprintf(fp,"  fit.acosso <- venus(X, y, order=order, K=ans.cv$K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"             maxit=5, rel.tol=1E-2, cv='gcv', lambda.0=ans.cv$lambda.0, \n");
  fprintf(fp,"             w=ans.cv$w, seed=110, cat.pos=ans.cv$cat.pos, \n");
  fprintf(fp,"             min.distinct=min.distinct)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"else{  ## use a prespecified numerical value for K\n");
  fprintf(fp,"  K <- cv\n");
  fprintf(fp,"  if(missing(w)||missing(lambda.0)){\n");
  fprintf(fp,"    ans.cv <- venus.cv(X, y, order=order, cv='gcv', wt.pow=wt.pow, w='L2', \n");
  fprintf(fp,"             f.est='trad', seed=110, K.lim='data', nK=5, gcv.pen=gcv.pen, \n");
  fprintf(fp,"             rel.K=5E-2, nlambda=5, lambda.lim=c(1E-10, 1E3), \n");
  fprintf(fp,"             rel.lambda=5E-2, rel.theta=1E-2, maxit=2, init.cv='gcv', \n");
  fprintf(fp,"             cat.pos=categorical, min.distinct=min.distinct)\n");
  fprintf(fp,"    w <- ans.cv$w\n");
  fprintf(fp,"    lambda.0 <- ans.cv$lambda.0\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fit.acosso <- venus(X, y, order=order, K=K, gcv.pen=gcv.pen, maxit=5, \n");
  fprintf(fp,"             rel.tol=1E-2, cv='gcv', lambda.0=lambda.0, w=w, seed=110, \n");
  fprintf(fp,"             cat.pos=ans.cv$cat.pos, min.distinct=min.distinct)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"fit.acosso$order <- order\n");
  fprintf(fp,"fit.acosso$K <- K\n");
  fprintf(fp,"fit.acosso$lambda.0 <- lambda.0\n");
  fprintf(fp,"\n");
  fprintf(fp,"return(fit.acosso)\n");
  fprintf(fp,"\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"############### ACOSSO Prediction ########################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"predict.acosso <- function(X.new, obj){\n");
  fprintf(fp,"\n");
  fprintf(fp,"###################### INPUTS #############################\n");
  fprintf(fp,"## X.new - a matrix of new values for the predictors\n");
  fprintf(fp,"## obj - a fitted acosso object\n");
  fprintf(fp,"###########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"###################### OUTPUT #############################\n");
  fprintf(fp,"## a vector of the predicted y's\n");
  fprintf(fp,"###########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(predict.venus(X.new, obj, order=obj$order))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"###########################################################\n");
  fprintf(fp,"###########################################################\n");
  fprintf(fp,"##### Other Functions Used in the Creation of ACOSSO ######\n");
  fprintf(fp,"###########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"index <- function(m,n){\n");
  fprintf(fp,"  if(m<=n) return(m:n)\n");
  fprintf(fp,"  else return(numeric(0))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"which.equal <- function(x, y){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- length(x)\n");
  fprintf(fp,"  ans <- rep(0,n)\n");
  fprintf(fp,"  for(i in 1:n){\n");
  fprintf(fp,"    ans[i] <- any(approx.equal(y,x[i]))\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(as.logical(ans))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"approx.equal <- function(x, y, tol=1E-9){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(abs(x - y) < tol)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"## Generates a matrix whose columns are random samples from 1:n\n");
  fprintf(fp,"get.obs.ind <- function(n, nfolds=5, seed=220){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  replace.seed <- T\n");
  fprintf(fp,"  if(missing(seed))\n");
  fprintf(fp,"    replace.seed <- F\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(replace.seed){\n");
  fprintf(fp,"   ## set seed to specified value\n");
  fprintf(fp,"    if(!any(ls(name='.GlobalEnv', all.names=T)=='.Random.seed')){\n");
  fprintf(fp,"      set.seed(1)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    save.seed <- .Random.seed\n");
  fprintf(fp,"    set.seed(seed)  \n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  perm <- sample(1:n, n)\n");
  fprintf(fp,"  n.cv <- rep(floor(n/nfolds),nfolds)\n");
  fprintf(fp,"  rem <- n - n.cv[1]*nfolds\n");
  fprintf(fp,"  n.cv[index(1,rem)] <- n.cv[index(1,rem)]+1\n");
  fprintf(fp,"  obs.ind <- list()\n");
  fprintf(fp,"  ind2 <- 0\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  for(i in 1:nfolds){\n");
  fprintf(fp,"    ind1 <- ind2+1\n");
  fprintf(fp,"    ind2 <- ind2+n.cv[i]\n");
  fprintf(fp,"    obs.ind[[i]] <- perm[ind1:ind2]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  if(replace.seed){\n");
  fprintf(fp,"   ## restore random seed to previous value\n");
  fprintf(fp,"    .Random.seed <<- save.seed\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(obs.ind)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"ginv2 <- function(X, eps=1E-12){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  eig.X <- eigen(X, symmetric=T)\n");
  fprintf(fp,"  P <- eig.X[[2]]\n");
  fprintf(fp,"  lambda <- eig.X[[1]]\n");
  fprintf(fp,"  ind <- lambda>eps\n");
  fprintf(fp,"  lambda[ind] <- 1/lambda[ind]\n");
  fprintf(fp,"  lambda[!ind] <- 0\n");
  fprintf(fp,"  ans <- P%%*%%diag(lambda,nrow=length(lambda))%%*%%t(P)\n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"sym.sqrt <- function(X){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  eig.X <- eigen(X)\n");
  fprintf(fp,"  P <- eig.X[[2]]\n");
  fprintf(fp,"  lambda <- eig.X[[1]]\n");
  fprintf(fp,"  lambda <- sqrt(lambda)\n");
  fprintf(fp,"  ans <- P%%*%%diag(lambda)%%*%%t(P)\n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"sym.sqrt.inv <- function(X, eps=1E-12){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  eig.X <- eigen(X)\n");
  fprintf(fp,"  P <- eig.X[[2]]\n");
  fprintf(fp,"  lambda <- eig.X[[1]]\n");
  fprintf(fp,"  ind <- lambda>eps\n");
  fprintf(fp,"  lambda[ind] <- 1/sqrt(lambda[ind])\n");
  fprintf(fp,"  lambda[!ind] <- 0\n");
  fprintf(fp,"  ans <- P%%*%%diag(lambda)%%*%%t(P)\n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"################################\n");
  fprintf(fp,"##### Sobolev RK function ######\n");
  fprintf(fp,"################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"k1 <- function(t){\n");
  fprintf(fp,"  return(t-.5)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"k2 <- function(t){\n");
  fprintf(fp,"  return( (k1(t)^2-1/12)/2 )\n");
  fprintf(fp,"}\n");
  fprintf(fp,"k4 <- function(t){\n");
  fprintf(fp,"  return( (k1(t)^4-k1(t)^2/2+7/240)/24 )\n");
  fprintf(fp,"}\n");
  fprintf(fp,"K.sob <- function(s,t){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ans <- k1(s)*k1(t) + k2(s)*k2(t) - k4(abs(s-t))\n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"K.sob <- function(s,t){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ans <- k1(s)*k1(t) + k2(s)*k2(t) - k4(abs(s-t))\n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"################################\n");
  fprintf(fp,"##### Sobolev RK function ######\n");
  fprintf(fp,"################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"K.cat <- function(s,t,G){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ans <- (G-1)/G*(s==t) - 1/G*(s!=t) \n");
  fprintf(fp,"  return(ans)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"###### Get MC integral for d2y\n");
  fprintf(fp,"\n");
  fprintf(fp,"est.d2y <- function(y, x){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- length(y)\n");
  fprintf(fp,"  ord.x <- order(x)\n");
  fprintf(fp,"  x <- x[ord.x]\n");
  fprintf(fp,"  y <- y[ord.x]\n");
  fprintf(fp,"  dy <- (y[-1]-y[-n])/(x[-1]-x[-n])\n");
  fprintf(fp,"  d2y <- (dy[-1]-dy[-(n-1)])/((x[-c(1,2)]-x[-c(n-1,n)])/2)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#           (((x[-c(1,n)]-x[-c(n-1,n)])+(x[-c(1,2)]-x[-c(1,n)]))/2)\n");
  fprintf(fp,"  return(d2y)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"### Create cross reference for kth component & ij th interaction ##\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.int2ind <- function(p){\n");
  fprintf(fp,"\n");
  fprintf(fp," ## create int2ind and ind2int to be able to switch back and forth between\n");
  fprintf(fp," ##  i,j th interaction and kth component\n");
  fprintf(fp,"  P <- p + choose(p,2)\n");
  fprintf(fp,"  int2ind <- matrix(0,p,p)\n");
  fprintf(fp,"  ind2int <- matrix(NA,P,2)\n");
  fprintf(fp,"  diag(int2ind) <- index(1,p)\n");
  fprintf(fp,"  ind2int[index(1,p),] <- cbind(index(1,p), index(1,p))\n");
  fprintf(fp,"  next.ind <- p+1\n");
  fprintf(fp,"  for(i in index(1,p-1)){\n");
  fprintf(fp,"    for(j in index(i+1,p)){\n");
  fprintf(fp,"      int2ind[i,j] <- int2ind[j,i] <- next.ind\n");
  fprintf(fp,"      ind2int[next.ind,] <- c(i,j)\n");
  fprintf(fp,"      next.ind <- next.ind+1\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(list(int2ind=int2ind, ind2int=ind2int))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"####### Creates p Gram matrices, 1 for each predictor  ###\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.gram <- function(X1, X2, order, cat.pos){\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Calculates K(X1[i,j]\n");
  fprintf(fp,"  n1 <- nrow(X1)\n");
  fprintf(fp,"  n2 <- nrow(X2)\n");
  fprintf(fp,"  p <- ncol(X1)\n");
  fprintf(fp,"  gram <- list()\n");
  fprintf(fp,"  if(length(cat.pos)>0)\n");
  fprintf(fp,"    cont.pos <- (1:p)[-cat.pos]\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    cont.pos <- (1:p)\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  for(i in cont.pos){\n");
  fprintf(fp,"    x1 <- rep(X1[,i], times=n2)\n");
  fprintf(fp,"    x2 <- rep(X2[,i], each=n1)\n");
  fprintf(fp,"    ans <- K.sob(x1,x2)\n");
  fprintf(fp,"    gram[[i]] <- matrix(ans, n1, n2)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  for(i in cat.pos){\n");
  fprintf(fp,"    x1 <- rep(X1[,i], times=n2)\n");
  fprintf(fp,"    x2 <- rep(X2[,i], each=n1)\n");
  fprintf(fp,"    G <- length(unique(x1))\n");
  fprintf(fp,"    ans <- K.cat(x1,x2,G)\n");
  fprintf(fp,"    gram[[i]] <- matrix(ans, n1, n2)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  if(order==2){\n");
  fprintf(fp,"    next.ind <- p+1\n");
  fprintf(fp,"    for(i in index(1,p-1)){\n");
  fprintf(fp,"      for(j in index(i+1,p)){\n");
  fprintf(fp,"        gram[[next.ind]] <- gram[[i]]*gram[[j]]\n");
  fprintf(fp,"        next.ind <- next.ind+1\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(gram)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"## Creates a single Gram matrix for Prediction of new obs #\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.gram.predict <- function(X1, X2, order, theta, w, cat.pos){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n1 <- nrow(X1)\n");
  fprintf(fp,"  n2 <- nrow(X2)\n");
  fprintf(fp,"  p <- ncol(X1)\n");
  fprintf(fp,"  gram.mat <- matrix(0, n1, n2)\n");
  fprintf(fp,"  gram <- list()\n");
  fprintf(fp,"  if(length(cat.pos)>0)\n");
  fprintf(fp,"    cont.pos <- (1:p)[-cat.pos]\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    cont.pos <- (1:p)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  for(i in cont.pos){\n");
  fprintf(fp,"    x1 <- rep(X1[,i], times=n2)\n");
  fprintf(fp,"    x2 <- rep(X2[,i], each=n1)\n");
  fprintf(fp,"    ans <- K.sob(x1,x2)\n");
  fprintf(fp,"    gram[[i]] <- matrix(ans, n1, n2)\n");
  fprintf(fp,"    if(theta[i] > 1E-6)\n");
  fprintf(fp,"      gram.mat <- gram.mat + theta[i]*w[i]^2*gram[[i]]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  for(i in cat.pos){\n");
  fprintf(fp,"    x1 <- rep(X1[,i], times=n2)\n");
  fprintf(fp,"    x2 <- rep(X2[,i], each=n1)\n");
  fprintf(fp,"    G <- length(unique(x1))\n");
  fprintf(fp,"    ans <- K.cat(x1,x2,G)\n");
  fprintf(fp,"    gram[[i]] <- matrix(ans, n1, n2)\n");
  fprintf(fp,"    if(theta[i] > 1E-6)\n");
  fprintf(fp,"      gram.mat <- gram.mat + theta[i]*w[i]^2*gram[[i]]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  if(order==2){\n");
  fprintf(fp,"    next.ind <- p+1\n");
  fprintf(fp,"    for(i in index(1,p-1)){\n");
  fprintf(fp,"      for(j in index(i+1,p)){\n");
  fprintf(fp,"        if(theta[next.ind] > 1E-6)\n");
  fprintf(fp,"          gram.mat <-gram.mat+theta[next.ind]*w[next.ind]^2*gram[[i]]*gram[[j]]\n");
  fprintf(fp,"        next.ind <- next.ind+1\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(gram.mat)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"###### Not using since efficiency gain was minimal #######\n");
  fprintf(fp,"#get.gram.predict.C <- function(X1, X2, order, theta, w){\n");
  fprintf(fp,"#  n1 <- nrow(X1)\n");
  fprintf(fp,"#  n2 <- nrow(X2)\n");
  fprintf(fp,"#  p <- ncol(X1)\n");
  fprintf(fp,"#  P <- length(theta)\n");
  fprintf(fp,"#  gram.mat <- matrix(0, n1, n2)  \n");
  fprintf(fp,"#  gram.mat <- .C(\"R_get_gram_predict\", as.double(as.matrix(X1)), as.integer(n1),  as.integer(p), as.double(as.matrix(X2)), as.integer(n2), as.integer(order), as.double(theta), as.integer(P), as.double(w), as.double(gram.mat))[[10]]\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#  return(matrix(gram.mat, ncol=n2))\n");
  fprintf(fp,"#}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"# Creates a single Gram matrix for Prediction of a bootstrap sample #\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.gram.boot <- function(X.perm, gram.list, order, theta, w){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n1 <- nrow(X.perm)\n");
  fprintf(fp,"  n2 <- nrow(gram.list[[1]])\n");
  fprintf(fp,"  p <- ncol(X.perm)\n");
  fprintf(fp,"  gram.mat <- matrix(0, n1, n2)\n");
  fprintf(fp,"  gram.new.list <- list()\n");
  fprintf(fp,"  ind2int <- get.int2ind(p)$ind2int\n");
  fprintf(fp,"  imp.vars <- sort(unique(as.vector(ind2int[theta>1E-6,])))\n");
  fprintf(fp,"\n");
  fprintf(fp,"  for(i in imp.vars){\n");
  fprintf(fp,"#    gram.new.list[[i]] <- gram.list[[i]][X.perm[,i],]\n");
  fprintf(fp,"#    if(theta[i] > 1E-6)\n");
  fprintf(fp,"#      gram.mat <- gram.mat + theta[i]*w[i]^2*gram.new.list[[i]]\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(theta[i] > 1E-6)\n");
  fprintf(fp,"      gram.mat <- gram.mat + theta[i]*w[i]^2*gram.list[[i]][X.perm[,i],]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  if(order==2){\n");
  fprintf(fp,"    ind <-  which(theta>1E-6 & c(rep(F,p), rep(T,choose(p,2))))\n");
  fprintf(fp,"    for(i in ind){\n");
  fprintf(fp,"      int1 <- ind2int[i,1]\n");
  fprintf(fp,"      int2 <- ind2int[i,2]\n");
  fprintf(fp,"#      gram.mat <- gram.mat + theta[i]*w[i]^2*gram.new.list[[int1]]*\n");
  fprintf(fp,"#                             gram.new.list[[int2]]\n");
  fprintf(fp,"\n");
  fprintf(fp,"      gram.mat <- gram.mat + theta[i]*w[i]^2*gram.list[[int1]][X.perm[,int1],]*\n");
  fprintf(fp,"                             gram.list[[int2]][X.perm[,int2],]\n");
  fprintf(fp,"\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(gram.mat)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"## Fits a penalized spline for given theta's #############\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"pen.spline <- function(X, y, order=1, lambda.0=1, theta, gcv.pen=1.01,\n");
  fprintf(fp,"                       Gram, cv='gcv', seed=220){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  if(missing(theta))\n");
  fprintf(fp,"    theta <- 1\n");
  fprintf(fp,"  if(length(theta)==1)\n");
  fprintf(fp,"    theta <- rep(theta,P)\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  rescale <- rep(F, p)\n");
  fprintf(fp,"  for(i in 1:p){\n");
  fprintf(fp,"    if(any( X[,i]<0 | X[,i]>1)){\n");
  fprintf(fp,"      X[,i] <- (X[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))\n");
  fprintf(fp,"      rescale[i] <- T\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  if(missing(Gram)){\n");
  fprintf(fp,"   ## Get Gram Matrix\n");
  fprintf(fp,"    Gram <- get.gram(X, X, order=order, cat.pos=numeric(0))\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ############### Use 5 fold CV ##################\n");
  fprintf(fp,"  if(cv=='5cv'){\n");
  fprintf(fp,"    y.hat <- numeric(n)\n");
  fprintf(fp,"    obs.ind <- get.obs.ind(n, nfolds=5, seed=seed)\n");
  fprintf(fp,"    for(i in 1:5){\n");
  fprintf(fp,"      X.i <- as.matrix(X[-obs.ind[[i]],])\n");
  fprintf(fp,"      y.i <- y[-obs.ind[[i]]]\n");
  fprintf(fp,"      Gram.i <- list()\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        Gram.i[[j]] <- (Gram[[j]])[-obs.ind[[i]],-obs.ind[[i]]]\n");
  fprintf(fp,"      fit.i <- pen.spline(X.i, y.i, order=order, lambda.0=lambda.0,theta=theta,\n");
  fprintf(fp,"                          cv='gcv', gcv.pen=gcv.pen, Gram=Gram.i)\n");
  fprintf(fp,"                          \n");
  fprintf(fp,"      K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        K.theta <- K.theta + fit.i$theta[j]*Gram[[j]]\n");
  fprintf(fp,"      y.hat[obs.ind[[i]]]<- K.theta[obs.ind[[i]],-obs.ind[[i]]]%%*%%fit.i$c.hat +\n");
  fprintf(fp,"                             fit.i$mu.hat \n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    fit.spline <- pen.spline(X, y, order=order, lambda.0=lambda.0, theta=theta,\n");
  fprintf(fp,"                             cv='gcv', gcv.pen=gcv.pen, Gram=Gram)\n");
  fprintf(fp,"    gcv <- sum((y-y.hat)^2)\n");
  fprintf(fp,"    mu.hat <- fit.spline$mu.hat\n");
  fprintf(fp,"    c.hat <- fit.spline$c.hat\n");
  fprintf(fp,"    y.hat <- fit.spline$y.hat\n");
  fprintf(fp,"    y.mat <- fit.spline$y.mat\n");
  fprintf(fp,"    res <- fit.spline$res\n");
  fprintf(fp,"    df <- fit.spline$df \n");
  fprintf(fp,"    Rsq <- fit.spline$Rsq\n");
  fprintf(fp,"    bic <- fit.spline$bic\n");
  fprintf(fp,"    norm <- fit.spline$norm\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp," ############### Use GCV ##################\n");
  fprintf(fp,"  else{  ## Use gcv\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"    K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"    for(i in 1:P){\n");
  fprintf(fp,"      K.theta <- K.theta + theta[i]*Gram[[i]]\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    K.inv <- solve(K.theta + lambda.0*diag(n))\n");
  fprintf(fp,"    J <- rep(1,n)\n");
  fprintf(fp,"    alpha <- sum(K.inv)^(-1)*t(J)%%*%%K.inv\n");
  fprintf(fp,"    mu.hat <- as.numeric(alpha%%*%%y)\n");
  fprintf(fp,"    c.hat <- K.inv%%*%%(y-J*mu.hat)\n");
  fprintf(fp,"    H <- K.theta%%*%%K.inv%%*%%(diag(n)-J%%*%%alpha)+J%%*%%alpha\n");
  fprintf(fp,"\n");
  fprintf(fp,"    y.hat <- mu.hat + K.theta%%*%%c.hat\n");
  fprintf(fp,"    res <- y-y.hat\n");
  fprintf(fp,"    SSE <- sum(res^2)\n");
  fprintf(fp,"    Rsq <- 1-SSE/sum((y-mean(y))^2)\n");
  fprintf(fp,"    df <- sum(diag(H)) \n");
  fprintf(fp,"    if(gcv.pen*df >= n)\n");
  fprintf(fp,"      gcv <- Inf\n");
  fprintf(fp,"    else\n");
  fprintf(fp,"      gcv <- SSE/(1-gcv.pen*df/n)^2\n");
  fprintf(fp,"    bic <- n*log(SSE/n) + df*log(n)\n");
  fprintf(fp,"    norm <- numeric(P)\n");
  fprintf(fp,"    for(i in 1:P){\n");
  fprintf(fp,"      norm[i] <- t(c.hat)%%*%%(theta[[i]]^2*Gram[[i]])%%*%%c.hat\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    y.mat <- matrix(0,n,P)\n");
  fprintf(fp,"    for(i in 1:P){\n");
  fprintf(fp,"      y.mat[,i] <- theta[[i]]*Gram[[i]]%%*%%c.hat\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(mu.hat=mu.hat, c.hat=c.hat, y.hat=y.hat, res=res, dfmod=df, \n");
  fprintf(fp,"              gcv=gcv, bic=bic, Rsq=Rsq, norm=norm, y.mat=y.mat, X=X, y=y,\n");
  fprintf(fp,"              theta=theta, Gram=Gram, w=rep(1,P), rescale=rescale))\n");
  fprintf(fp,"\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"# used by VENUS: solves for beta.hat and c.hat for fixed theta's #\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.H.c <- function(Gram, y, theta, lambda.0, w, gcv.pen){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- length(y)\n");
  fprintf(fp,"  P <- length(theta)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    K.theta <- K.theta + theta[i]*w[i]^2*Gram[[i]]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  K.inv <- solve(K.theta + lambda.0*diag(n))\n");
  fprintf(fp,"  J <- rep(1,n)\n");
  fprintf(fp,"  alpha <- sum(K.inv)^(-1)*t(J)%%*%%K.inv\n");
  fprintf(fp,"  mu.hat <- as.numeric(alpha%%*%%y)\n");
  fprintf(fp,"  c.hat <- K.inv%%*%%(y-J*mu.hat)\n");
  fprintf(fp,"  H <- K.theta%%*%%K.inv%%*%%(diag(n)-J%%*%%alpha)+J%%*%%alpha\n");
  fprintf(fp,"\n");
  fprintf(fp,"  df <- sum(diag(H))\n");
  fprintf(fp,"  y.hat <- mu.hat + K.theta%%*%%c.hat\n");
  fprintf(fp,"  res <- y-y.hat\n");
  fprintf(fp,"  SSE <- sum(res^2)\n");
  fprintf(fp,"  if(gcv.pen*df >= n)\n");
  fprintf(fp,"    gcv <- Inf\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    gcv <- SSE/(1-gcv.pen*df/n)^2\n");
  fprintf(fp,"  bic <- n*log(SSE/n) + df*log(n)\n");
  fprintf(fp,"  Rsq <- 1-SSE/sum((y-mean(y))^2)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(mu.hat=mu.hat, H=H, df=df, y.hat=y.hat, gcv=gcv, Rsq=Rsq,\n");
  fprintf(fp,"              c.hat=c.hat, bic=bic))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"##### used by VENUS: solves for theta for fixed c's ######\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.theta.hat <- function(Gram, mu.hat, c.hat, y, lambda.0, theta.0.ind, K, w){\n");
  fprintf(fp," \n");
  fprintf(fp,"#theta.0.ind <- numeric(0)\n");
  fprintf(fp,"\n");
  fprintf(fp,"      n <- length(y)\n");
  fprintf(fp,"      P <- length(Gram)\n");
  fprintf(fp,"      if(length(theta.0.ind)==0)\n");
  fprintf(fp,"        keep.col <- 1:P\n");
  fprintf(fp,"      else\n");
  fprintf(fp,"        keep.col <- (1:P)[-theta.0.ind]\n");
  fprintf(fp,"      G <- matrix(NA,n,P)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#n..<<-n\n");
  fprintf(fp,"#Gram..<<-Gram\n");
  fprintf(fp,"#c.hat..<<-c.hat\n");
  fprintf(fp,"#keep.col..<<-keep.col\n");
  fprintf(fp,"\n");
  fprintf(fp,"      for(i in 1:P){\n");
  fprintf(fp,"        G[,i] <- w[i]^2*Gram[[i]]%%*%%c.hat\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      G.red <- G[,keep.col]\n");
  fprintf(fp,"      P.red <- length(keep.col)\n");
  fprintf(fp,"      D <- t(G.red)%%*%%G.red\n");
  fprintf(fp,"      d <- t( (t(y) - mu.hat - .5*lambda.0*t(c.hat))%%*%%G.red )\n");
  fprintf(fp,"      A <- t(rbind(diag(1,P.red), rep(-1,P.red)))\n");
  fprintf(fp,"      b.0 <- c(rep(0,P.red), -K)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#G.red..<<-G.red\n");
  fprintf(fp,"#P.red..<<-P.red\n");
  fprintf(fp,"#A2..<<-A\n");
  fprintf(fp,"#D2..<<-D\n");
  fprintf(fp,"#d2..<<-d\n");
  fprintf(fp,"#b2.0..<<-b.0\n");
  fprintf(fp,"\n");
  fprintf(fp,"      #opt.ans <- solve.QP2(D, d, A, b.0, meq=0)\n");
  fprintf(fp,"      #add.to.diag <- sum(diag(D)/ncol(D))*1E-10\n");
  fprintf(fp,"      #while(opt.ans$solution[1] == -2){\n");
  fprintf(fp,"      #  add.to.diag <- add.to.diag*10\n");
  fprintf(fp,"      #  opt.ans <- solve.QP2(D+diag(add.to.diag,P.red), d, A, b.0, meq=0)\n");
  fprintf(fp,"      #}\n");
  fprintf(fp,"\n");
  fprintf(fp,"      opt.ans <- try(solve.QP(D, d, A, b.0, meq=0), silent=T)\n");
  fprintf(fp,"      add.to.diag <- sum(diag(D)/ncol(D))*1E-10\n");
  fprintf(fp,"      while(is.character(opt.ans)){\n");
  fprintf(fp,"        add.to.diag <- add.to.diag*10\n");
  fprintf(fp,"        opt.ans <- try(solve.QP(D+diag(add.to.diag,P.red), d, A, b.0, meq=0), silent=T)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"\n");
  fprintf(fp,"      theta.new <- opt.ans$solution\n");
  fprintf(fp,"      theta <- rep(0,P)\n");
  fprintf(fp,"      theta[keep.col] <- theta.new\n");
  fprintf(fp,"      return(theta)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"## VENUS for a fixed smoothing param M   #################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"venus <- function(X, y, K, order=1, gcv.pen=1.01,lambda.0, theta.0, rel.tol=1E-2,\n");
  fprintf(fp,"                  maxit=2, Gram, cv='gcv', w, seed=220, \n");
  fprintf(fp,"                  alpha=.05, nvar=ncol(X), nfit=20, cat.pos=\"auto\",\n");
  fprintf(fp,"                  min.distinct=7){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)\n");
  fprintf(fp,"  if(missing(lambda.0))\n");
  fprintf(fp,"    lambda.0 <- .01\n");
  fprintf(fp,"  if(missing(theta.0))\n");
  fprintf(fp,"    theta.0 <- rep(1,P)\n");
  fprintf(fp,"  if(missing(w))\n");
  fprintf(fp,"    w <- rep(1,P)\n");
  fprintf(fp,"  if(length(w)==1)\n");
  fprintf(fp,"    w <- rep(w,P)\n");
  fprintf(fp,"  theta.hat <- theta.0\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ## Identify categorical vars \n");
  fprintf(fp,"  if(length(cat.pos)>0 && cat.pos[1]=='auto'){\n");
  fprintf(fp,"   # scan for variables with min.distinct or less distinct values. \n");
  fprintf(fp,"    cat.pos <- numeric(0)\n");
  fprintf(fp,"    for(i in 1:p){\n");
  fprintf(fp,"      unique.i <- unique(X[,i])\n");
  fprintf(fp,"      if(length(unique.i)<=min.distinct || is.character(X[,i])){\n");
  fprintf(fp,"        cat.pos <- c(cat.pos, i)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(length(cat.pos)>0)\n");
  fprintf(fp,"    cont.pos <- (1:p)[-cat.pos]\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    cont.pos <- (1:p)\n");
  fprintf(fp,"  ## shift and rescale continuous x's to [0,1]\n");
  fprintf(fp,"  rescale <- rep(F, p)\n");
  fprintf(fp,"  X.orig <- X\n");
  fprintf(fp,"  for(i in cont.pos){\n");
  fprintf(fp,"    if(any( X[,i]<0 | X[,i]>1)){\n");
  fprintf(fp,"      X[,i] <- (X[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))*.9 + .05\n");
  fprintf(fp,"      rescale[i] <- T\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  if(missing(Gram)){\n");
  fprintf(fp,"    Gram <- get.gram(X, X, order=order, cat.pos=cat.pos)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ############### Use 5 fold CV ##################\n");
  fprintf(fp,"  if(cv=='5cv'){\n");
  fprintf(fp,"    y.hat <- numeric(n)\n");
  fprintf(fp,"    obs.ind <- get.obs.ind(n, nfolds=5, seed=seed)\n");
  fprintf(fp,"    for(i in 1:5){\n");
  fprintf(fp,"      X.i <- as.matrix(X[-obs.ind[[i]],])\n");
  fprintf(fp,"      y.i <- y[-obs.ind[[i]]]\n");
  fprintf(fp,"      Gram.i <- list()\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        Gram.i[[j]] <- (Gram[[j]])[-obs.ind[[i]],-obs.ind[[i]]]\n");
  fprintf(fp,"      fit.i<-venus(X=X.i, y=y.i, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                   lambda.0=.8*lambda.0, theta.0=theta.0, rel.tol=rel.tol,\n");
  fprintf(fp,"                   maxit=maxit, Gram=Gram.i, cv='gcv', w=w,\n");
  fprintf(fp,"                   min.distinct=min.distinct)\n");
  fprintf(fp,"      K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        K.theta <- K.theta + fit.i$theta[j]*w[j]^2*Gram[[j]]\n");
  fprintf(fp,"\n");
  fprintf(fp,"      y.hat[obs.ind[[i]]]<- K.theta[obs.ind[[i]],-obs.ind[[i]]]%%*%%fit.i$c.hat +\n");
  fprintf(fp,"                             fit.i$mu.hat \n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    fit.venus <- venus(X=X, y=y, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                       lambda.0=lambda.0, theta.0=theta.0, rel.tol=rel.tol, \n");
  fprintf(fp,"                       maxit=maxit, Gram=Gram, cv='gcv', w=w,\n");
  fprintf(fp,"                       min.distinct=min.distinct)\n");
  fprintf(fp,"    fit.venus$Gram <- NULL\n");
  fprintf(fp,"    gcv <- sum((y-y.hat)^2)\n");
  fprintf(fp,"    mu.hat <- fit.venus$mu.hat\n");
  fprintf(fp,"    c.hat <- fit.venus$c.hat\n");
  fprintf(fp,"    y.hat <- fit.venus$y.hat\n");
  fprintf(fp,"    res <- fit.venus$res\n");
  fprintf(fp,"    df <- fit.venus$df \n");
  fprintf(fp,"    theta.hat <- fit.venus$theta\n");
  fprintf(fp,"    Rsq <- fit.venus$Rsq\n");
  fprintf(fp,"    bic <- fit.venus$bic\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"    \n");
  fprintf(fp," ############### Use 10 fold CV ##################\n");
  fprintf(fp,"  else if(cv=='10cv'){\n");
  fprintf(fp,"    y.hat <- numeric(n)\n");
  fprintf(fp,"    obs.ind <- get.obs.ind(n, nfolds=10, seed=seed)\n");
  fprintf(fp,"    for(i in 1:10){\n");
  fprintf(fp,"      X.i <- as.matrix(X[-obs.ind[[i]],])\n");
  fprintf(fp,"      y.i <- y[-obs.ind[[i]]]\n");
  fprintf(fp,"      Gram.i <- list()\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        Gram.i[[j]] <- (Gram[[j]])[-obs.ind[[i]],-obs.ind[[i]]]\n");
  fprintf(fp,"      fit.i<-venus(X=X.i, y=y.i, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                   lambda.0=.9*lambda.0, theta.0=theta.0, rel.tol=rel.tol,\n");
  fprintf(fp,"                   maxit=maxit, Gram=Gram.i, cv='gcv', w=w,\n");
  fprintf(fp,"                   min.distinct=min.distinct)\n");
  fprintf(fp,"      K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        K.theta <- K.theta + fit.i$theta[j]*w[j]^2*Gram[[j]]\n");
  fprintf(fp,"\n");
  fprintf(fp,"      y.hat[obs.ind[[i]]]<- K.theta[obs.ind[[i]],-obs.ind[[i]]]%%*%%fit.i$c.hat +\n");
  fprintf(fp,"                             fit.i$mu.hat \n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    fit.venus <- venus(X=X, y=y, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                       lambda.0=lambda.0, theta.0=theta.0, rel.tol=rel.tol, \n");
  fprintf(fp,"                       maxit=maxit, Gram=Gram, cv='gcv', w=w,\n");
  fprintf(fp,"                       min.distinct=min.distinct)\n");
  fprintf(fp,"    fit.venus$Gram <- NULL\n");
  fprintf(fp,"    gcv <- sum((y-y.hat)^2)\n");
  fprintf(fp,"    mu.hat <- fit.venus$mu.hat\n");
  fprintf(fp,"    c.hat <- fit.venus$c.hat\n");
  fprintf(fp,"    y.hat <- fit.venus$y.hat\n");
  fprintf(fp,"    res <- fit.venus$res\n");
  fprintf(fp,"    df <- fit.venus$df \n");
  fprintf(fp,"    theta.hat <- fit.venus$theta\n");
  fprintf(fp,"    Rsq <- fit.venus$Rsq\n");
  fprintf(fp,"    bic <- fit.venus$bic\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"    \n");
  fprintf(fp," ############### Use type I Error Rate ##################\n");
  fprintf(fp,"  else if(cv=='var'){\n");
  fprintf(fp,"    for(i in 1:nfit){\n");
  fprintf(fp,"      X.i <- as.matrix(X[-obs.ind[[i]],])\n");
  fprintf(fp,"      y.i <- y[-obs.ind[[i]]]\n");
  fprintf(fp,"      Gram.i <- list()\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        Gram.i[[j]] <- (Gram[[j]])[-obs.ind[[i]],-obs.ind[[i]]]\n");
  fprintf(fp,"      fit.i<-venus(X=X.i, y=y.i, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                   lambda.0=lambda.0, theta.0=theta.0, rel.tol=rel.tol,\n");
  fprintf(fp,"                   maxit=maxit, Gram=Gram.i, cv='gcv', w=w,\n");
  fprintf(fp,"                   min.distinct=min.distinct)\n");
  fprintf(fp,"      K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"      for(j in 1:P)\n");
  fprintf(fp,"        K.theta <- K.theta + fit.i$theta[j]*w[j]^2*Gram[[j]]\n");
  fprintf(fp,"\n");
  fprintf(fp,"      y.hat[obs.ind[[i]]]<- K.theta[obs.ind[[i]],-obs.ind[[i]]]%%*%%fit.i$c.hat +\n");
  fprintf(fp,"                             fit.i$mu.hat \n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    fit.venus <- venus(X=X, y=y, order=order, K=K, gcv.pen=gcv.pen, \n");
  fprintf(fp,"                       lambda.0=lambda.0, theta.0=theta.0, rel.tol=rel.tol, \n");
  fprintf(fp,"                       maxit=maxit, Gram=Gram, cv='gcv', w=w,\n");
  fprintf(fp,"                       min.distinct=min.distinct)\n");
  fprintf(fp,"    fit.venus$Gram <- NULL\n");
  fprintf(fp,"    gcv <- sum((y-y.hat)^2)\n");
  fprintf(fp,"    mu.hat <- fit.venus$mu.hat\n");
  fprintf(fp,"    c.hat <- fit.venus$c.hat\n");
  fprintf(fp,"    y.hat <- fit.venus$y.hat\n");
  fprintf(fp,"    res <- fit.venus$res\n");
  fprintf(fp,"    df <- fit.venus$df \n");
  fprintf(fp,"    theta.hat <- fit.venus$theta\n");
  fprintf(fp,"    Rsq <- fit.venus$Rsq\n");
  fprintf(fp,"    bic <- fit.venus$bic\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"    \n");
  fprintf(fp," ############### Use gcv ##################\n");
  fprintf(fp,"  else{\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## Solve for c.hat when theta=theta.0\n");
  fprintf(fp,"    H.c <- get.H.c(Gram, y, theta.0, lambda.0, w, gcv.pen)\n");
  fprintf(fp,"    mu.hat <- H.c$mu.hat\n");
  fprintf(fp,"    c.hat <- H.c$c.hat\n");
  fprintf(fp,"    df <- H.c$df\n");
  fprintf(fp,"    y.hat <- H.c$y.hat\n");
  fprintf(fp,"    res <- H.c$res\n");
  fprintf(fp,"    gcv <- H.c$gcv\n");
  fprintf(fp,"    bic <- H.c$bic\n");
  fprintf(fp,"    Rsq <- H.c$Rsq\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## Iterative solving for theta, then c ...\n");
  fprintf(fp,"    iter <- 1\n");
  fprintf(fp,"\n");
  fprintf(fp,"    theta.0.ind <- which(theta.hat < 1E-9 | w==0)\n");
  fprintf(fp,"\n");
  fprintf(fp,"    repeat{\n");
  fprintf(fp,"#cat(\"\\niter =\", iter)\n");
  fprintf(fp,"      if(iter > maxit){\n");
  fprintf(fp,"#warning(paste(\"Maximum iterations\", maxit, \"reached\"))\n");
  fprintf(fp,"        break\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      iter <- iter + 1\n");
  fprintf(fp,"\n");
  fprintf(fp,"     ## First solve for theta for a fixed c\n");
  fprintf(fp,"      if(K==0){\n");
  fprintf(fp,"        theta.new <- rep(0,P)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      else{\n");
  fprintf(fp,"        theta.new <- get.theta.hat(Gram, mu.hat, c.hat, y, lambda.0, \n");
  fprintf(fp,"                                   theta.0.ind, K, w)\n");
  fprintf(fp,"        if(any(theta.new == -1)||any(is.na(theta.new))){### prob with solution\n");
  fprintf(fp,"          theta.new <- theta.hat\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"\n");
  fprintf(fp,"      theta.0.ind <- which(theta.new <= 1E-9 | w==0)\n");
  fprintf(fp,"      theta.new[theta.0.ind] <- 0\n");
  fprintf(fp,"\n");
  fprintf(fp,"#print(theta.new)\n");
  fprintf(fp,"\n");
  fprintf(fp,"     ## Now solve for c.hat for a fixed theta\n");
  fprintf(fp,"      H.c <- get.H.c(Gram, y, theta.new, lambda.0, w, gcv.pen)\n");
  fprintf(fp,"      mu.hat <- H.c$mu.hat\n");
  fprintf(fp,"      c.hat <- H.c$c.hat\n");
  fprintf(fp,"      H <- H.c$H\n");
  fprintf(fp,"      df <- H.c$df\n");
  fprintf(fp,"      y.hat <- H.c$y.hat\n");
  fprintf(fp,"           \n");
  fprintf(fp,"\n");
  fprintf(fp,"      res <- H.c$res\n");
  fprintf(fp,"      gcv <- H.c$gcv\n");
  fprintf(fp,"      bic <- H.c$bic\n");
  fprintf(fp,"      Rsq <- H.c$Rsq\n");
  fprintf(fp,"\n");
  fprintf(fp,"     ## Now check for convergence\n");
  fprintf(fp,"      divisor <- theta.new\n");
  fprintf(fp,"      divisor[theta.new < 1E-6] <- 1\n");
  fprintf(fp,"      rel.norm <- sqrt(sum(((theta.hat-theta.new)/divisor)^2))\n");
  fprintf(fp,"\n");
  fprintf(fp,"      if(rel.norm < rel.tol){\n");
  fprintf(fp,"        theta.hat <- theta.new\n");
  fprintf(fp,"        break\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      theta.hat <- theta.new\n");
  fprintf(fp,"    }  \n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  y.mat <- matrix(0,n,P)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    y.mat[,i] <- theta.hat[i]*w[i]^2*Gram[[i]]%%*%%c.hat\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(list(c.hat=c.hat, mu.hat=mu.hat, y.hat=y.hat, res=res, dfmod=df,\n");
  fprintf(fp,"              w=w, gcv=gcv, bic=bic, theta=theta.hat, Rsq=Rsq, Gram=Gram, \n");
  fprintf(fp,"              y.mat=y.mat, X=X, X.orig=X.orig, y=y, rescale=rescale,\n");
  fprintf(fp,"              cat.pos=cat.pos))\n");
  fprintf(fp,"\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"#####Predict new obs for Venus ###########################\n");
  fprintf(fp,"##########################################################\n");
   fprintf(fp,"\n");
  fprintf(fp,"predict.venus <- function(X.new, obj, order=1){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  theta <- obj$theta\n");
  fprintf(fp,"  w <- obj$w\n");
  fprintf(fp,"  c.hat <- obj$c.hat\n");
  fprintf(fp,"  mu.hat <- obj$mu.hat\n");
  fprintf(fp,"  X <- obj$X\n");
  fprintf(fp,"  X.orig <- obj$X.orig\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  rescale <- obj$rescale\n");
  fprintf(fp,"  cat.pos <- obj$cat.pos\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  for(i in (1:p)[rescale])\n");
  fprintf(fp,"    X.new[,i] <- (X.new[,i]-min(X.orig[,i]))/(max(X.orig[,i])-min(X.orig[,i]))*.9 + .05\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix & predict y\n");
  fprintf(fp,"  Gram.mat <- get.gram.predict(X.new, X, order, theta, w, cat.pos=cat.pos)\n");
  fprintf(fp,"  y.hat <- as.vector(Gram.mat%%*%%c.hat + mu.hat)\n");
  fprintf(fp,"  return(y.hat)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"###### Predict bootstrap sample of obs for Venus #########\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"predict.acosso.boot <- function(X.perm, obj){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  gram.list <- obj$Gram\n");
  fprintf(fp,"  order <- obj$order\n");
  fprintf(fp,"  theta <- obj$theta\n");
  fprintf(fp,"  w <- obj$w\n");
  fprintf(fp,"  c.hat <- obj$c.hat\n");
  fprintf(fp,"  mu.hat <- obj$mu.hat\n");
  fprintf(fp,"  cat.pos <- obj$cat.pos\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix & predict y\n");
  fprintf(fp,"  Gram.mat <- get.gram.boot(X.perm, gram.list, order, theta, w)\n");
  fprintf(fp,"  y.hat <- as.vector(Gram.mat%%*%%c.hat + mu.hat)\n");
  fprintf(fp,"  return(y.hat)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"##### Predict component curves for Venus #################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"predict.venus.components <- function(X.new, obj, order=1){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n.new <- nrow(X.new)\n");
  fprintf(fp,"  theta <- obj$theta\n");
  fprintf(fp,"  w <- obj$w\n");
  fprintf(fp,"  c.hat <- obj$c.hat\n");
  fprintf(fp,"  mu.hat <- obj$mu.hat\n");
  fprintf(fp,"  X <- obj$X\n");
  fprintf(fp,"  n <- nrow(obj$X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  rescale <- obj$rescale\n");
  fprintf(fp,"  cat.pos <- obj$cat.pos\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  for(i in (1:p)[rescale])\n");
  fprintf(fp,"    X.new[,i] <- (X.new[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  Gram <- get.gram(X.new, X, order, cat.pos=cat.pos)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  y.mat <- matrix(0,n.new,P)\n");
  fprintf(fp,"  for(j in 1:P){\n");
  fprintf(fp,"    y.mat[,j] <- theta[j]*w[j]^2*Gram[[j]]%%*%%c.hat\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  y.hat <- rowSums(y.mat) + mu.hat\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(y.hat=y.hat, y.mat=y.mat))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"############ GET NORMS FOR A VENUS OBJECT  ###############\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"get.venus.norms <- function(obj){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  y.mat <- obj$y.mat\n");
  fprintf(fp,"  P <- ncol(y.mat)\n");
  fprintf(fp,"  c.hat <- obj$c.hat\n");
  fprintf(fp,"  Gram <- obj$Gram\n");
  fprintf(fp,"  theta <- obj$theta\n");
  fprintf(fp,"  w <- obj$w\n");
  fprintf(fp,"\n");
  fprintf(fp,"  L2.norm <- numeric(P)\n");
  fprintf(fp,"  H2.norm <- numeric(P)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    H2.norm[i] <- sqrt(theta[i]^2*w[i]^4*t(c.hat)%%*%%(Gram[[i]])%%*%%c.hat)\n");
  fprintf(fp,"    L2.norm[i] <- sqrt(mean((y.mat[,i])^2))\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(L2.norm=L2.norm, H2.norm=H2.norm))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"######### ESTIMATE w VIA FULL SMOOTHING SPLINE ###########\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"           \n");
  fprintf(fp,"get.w <- function(X, y, order, lambda.lim, nlambda=3, gcv.pen, wt.pow,\n");
  fprintf(fp,"                  rel.tol, df.lim, w='L2', w.0=1, w.lim=c(1E-10, 1E10), Gram,\n");
  fprintf(fp,"                  cat.pos){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)  \n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(missing(df.lim)){\n");
  fprintf(fp,"    df.lim <- c(0,.5*n)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(w[1]!='L2' && w[1]!='RKHS'){\n");
  fprintf(fp,"    if(length(w)==1)\n");
  fprintf(fp,"      w <- rep(w,P)\n");
  fprintf(fp,"    w.0 <- w\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  else if(length(w.0)==1){\n");
  fprintf(fp,"    w.0 <- rep(w.0,P)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  for(i in 1:p){\n");
  fprintf(fp,"    if(any( X[,i]<0 | X[,i]>1)){\n");
  fprintf(fp,"      X[,i] <- (X[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  if(missing(Gram)){\n");
  fprintf(fp,"    Gram <- get.gram(X, X, order=order, cat.pos=cat.pos)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    K.theta <- K.theta + w.0[i]^2*Gram[[i]]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  lambda.min <- lambda.lim[1]\n");
  fprintf(fp,"  lambda.max <- lambda.lim[2]\n");
  fprintf(fp,"  log.lambda.best <- .01\n");
  fprintf(fp,"  LAMBDAMIN <- log(lambda.min)\n");
  fprintf(fp,"  LAMBDAMAX <- log(lambda.max)\n");
  fprintf(fp,"  log.lambda.min <- LAMBDAMIN\n");
  fprintf(fp,"  log.lambda.max <- LAMBDAMAX\n");
  fprintf(fp,"  inc <- (log.lambda.max - log.lambda.min)/(nlambda-1)\n");
  fprintf(fp,"  log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc)\n");
  fprintf(fp,"  log.lambda.old <- log.lambda.vec\n");
  fprintf(fp,"  gcv.best <- Inf\n");
  fprintf(fp,"\n");
  fprintf(fp,"  repeat{\n");
  fprintf(fp,"    nlambda.now <- length(log.lambda.vec)\n");
  fprintf(fp,"    df.vec <- rep(0, length(log.lambda.vec))\n");
  fprintf(fp,"    for(j in 1:nlambda.now){\n");
  fprintf(fp,"      llambda <- log.lambda.vec[j]\n");
  fprintf(fp,"      lambda <- exp(llambda)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#K.theta..<<-K.theta\n");
  fprintf(fp,"#lambda..<<-lambda\n");
  fprintf(fp,"\n");
  fprintf(fp,"      K.inv <- solve(K.theta + lambda*diag(n), tol=1E-25)\n");
  fprintf(fp,"      J <- rep(1,n)\n");
  fprintf(fp,"      alpha <- sum(K.inv)^(-1)*t(J)%%*%%K.inv\n");
  fprintf(fp,"      mu.hat <- as.numeric(alpha%%*%%y)\n");
  fprintf(fp,"      c.hat <- K.inv%%*%%(y-J*mu.hat)\n");
  fprintf(fp,"      H <- K.theta%%*%%K.inv%%*%%(diag(n)-J%%*%%alpha)+J%%*%%alpha\n");
  fprintf(fp,"\n");
  fprintf(fp,"      df <- sum(diag(H))\n");
  fprintf(fp,"      df.vec[j] <- df\n");
  fprintf(fp,"      y.hat <- mu.hat + K.theta%%*%%c.hat\n");
  fprintf(fp,"      res <- y-y.hat\n");
  fprintf(fp,"      SSE <- sum(res^2)\n");
  fprintf(fp,"      if(gcv.pen*df >= n)\n");
  fprintf(fp,"        gcv <- Inf\n");
  fprintf(fp,"      else\n");
  fprintf(fp,"        gcv <- SSE/(1-gcv.pen*df/n)^2\n");
  fprintf(fp,"\n");
  fprintf(fp,"      if(gcv < gcv.best && df>=df.lim[1] && df<=df.lim[2]){\n");
  fprintf(fp,"        gcv.best <- gcv     \n");
  fprintf(fp,"        c.best <- c.hat\n");
  fprintf(fp,"        df.best <- df\n");
  fprintf(fp,"        log.lambda.best <- llambda\n");
  fprintf(fp,"        lambda.best <- exp(log.lambda.best)\n");
  fprintf(fp,"        mu.best <- mu.hat\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(gcv.best==Inf){\n");
  fprintf(fp,"      df.err <- (df.vec-df.lim[1])^2+(df.vec-df.lim[2])^2\n");
  fprintf(fp,"      ind.j <- order(df.err)[1]\n");
  fprintf(fp,"      df.best <- df.vec[ind.j]\n");
  fprintf(fp,"      log.lambda.best <- log.lambda.vec[ind.j]\n");
  fprintf(fp,"      lambda.best <- exp(log.lambda.best)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    diff <- exp(log.lambda.best+inc) - lambda.best\n");
  fprintf(fp,"    if(diff/lambda.best <= rel.tol)\n");
  fprintf(fp,"      break\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## create log.lambda.vec for next pass\n");
  fprintf(fp,"    log.lambda.min <- log.lambda.best - floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.min <- max(LAMBDAMIN, log.lambda.min)  \n");
  fprintf(fp,"    log.lambda.max <- log.lambda.best + floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.max <- min(LAMBDAMAX, log.lambda.max)  \n");
  fprintf(fp,"    inc <- inc/2\n");
  fprintf(fp,"    log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc) \n");
  fprintf(fp,"    ind <- which.equal(log.lambda.vec, log.lambda.old)\n");
  fprintf(fp,"    log.lambda.vec <- log.lambda.vec[!ind]\n");
  fprintf(fp,"    log.lambda.old <- c(log.lambda.old, log.lambda.vec)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Now get weights w\n");
  fprintf(fp,"  L2.norm <- numeric(P)\n");
  fprintf(fp,"  H2.norm <- numeric(P)\n");
  fprintf(fp,"  y.mat <- matrix(0,n,P)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    y.mat[,i] <- w.0[i]^2*Gram[[i]]%%*%%c.hat\n");
  fprintf(fp,"    H2.norm[i] <- sqrt(t(c.best)%%*%%(w.0[i]^4*Gram[[i]])%%*%%c.best)\n");
  fprintf(fp,"    L2.norm[i] <- sqrt(mean((y.mat[,i])^2))\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"#L2.norm..<<-L2.norm\n");
  fprintf(fp,"#H2.norm..<<-H2.norm\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(w[1]=='L2'){\n");
  fprintf(fp,"    w <- L2.norm^wt.pow\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  else if(w[1]=='RKHS'){\n");
  fprintf(fp,"    w <- H2.norm^wt.pow\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(lambda=lambda.best, gcv=gcv.best, df=df.best, w=w, \n");
  fprintf(fp,"              Gram=Gram, y.mat=y.mat, X=X, y=y, c.hat=c.best, mu.hat=mu.best))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"######### GET LAMBDA_0 VIA FULL SMOOTHING SPLINE #########\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"           \n");
  fprintf(fp,"get.lambda0 <- function(X, y, order, lambda.lim, nlambda, gcv.pen, rel.tol, \n");
  fprintf(fp,"                        w, Gram, df.lim, cat.pos){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)  \n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(missing(df.lim)){\n");
  fprintf(fp,"    df.lim <- c(0,.5*n)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(length(w)==1)\n");
  fprintf(fp,"    w <- rep(w,P)\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  for(i in 1:p){\n");
  fprintf(fp,"    if(any( X[,i]<0 | X[,i]>1)){\n");
  fprintf(fp,"      X[,i] <- (X[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  if(missing(Gram)){\n");
  fprintf(fp,"    Gram <- get.gram(X, X, order=order, cat.pos=cat.pos)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  K.theta <- matrix(0,n,n)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    K.theta <- K.theta + w[i]^2*Gram[[i]]\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  lambda.min <- lambda.lim[1]\n");
  fprintf(fp,"  lambda.max <- lambda.lim[2]\n");
  fprintf(fp,"  log.lambda.best <- .01\n");
  fprintf(fp,"  LAMBDAMIN <- log(lambda.min)\n");
  fprintf(fp,"  LAMBDAMAX <- log(lambda.max)\n");
  fprintf(fp,"  log.lambda.min <- LAMBDAMIN\n");
  fprintf(fp,"  log.lambda.max <- LAMBDAMAX\n");
  fprintf(fp,"  inc <- (log.lambda.max - log.lambda.min)/(nlambda-1)\n");
  fprintf(fp,"  log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc)\n");
  fprintf(fp,"  log.lambda.old <- log.lambda.vec\n");
  fprintf(fp,"  gcv.best <- Inf\n");
  fprintf(fp,"\n");
  fprintf(fp,"  repeat{\n");
  fprintf(fp,"    nlambda.now <- length(log.lambda.vec)\n");
  fprintf(fp,"    df.vec <- rep(0, length(log.lambda.vec))\n");
  fprintf(fp,"    for(j in 1:nlambda.now){\n");
  fprintf(fp,"      llambda <- log.lambda.vec[j]\n");
  fprintf(fp,"      lambda <- exp(llambda)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#K.theta..<<-K.theta\n");
  fprintf(fp,"#lambda..<<-lambda\n");
  fprintf(fp,"\n");
  fprintf(fp,"      K.inv <- solve(K.theta + lambda*diag(n), tol=1E-25)\n");
  fprintf(fp,"      J <- rep(1,n)\n");
  fprintf(fp,"      alpha <- sum(K.inv)^(-1)*t(J)%%*%%K.inv\n");
  fprintf(fp,"      mu.hat <- as.numeric(alpha%%*%%y)\n");
  fprintf(fp,"      c.hat <- K.inv%%*%%(y-J*mu.hat)\n");
  fprintf(fp,"      H <- K.theta%%*%%K.inv%%*%%(diag(n)-J%%*%%alpha)+J%%*%%alpha\n");
  fprintf(fp,"\n");
  fprintf(fp,"      df <- sum(diag(H))\n");
  fprintf(fp,"      df.vec[j] <- df\n");
  fprintf(fp,"      y.hat <- mu.hat + K.theta%%*%%c.hat\n");
  fprintf(fp,"      res <- y-y.hat\n");
  fprintf(fp,"      SSE <- sum(res^2)\n");
  fprintf(fp,"      if(gcv.pen*df >= n)\n");
  fprintf(fp,"        gcv <- Inf\n");
  fprintf(fp,"      else\n");
  fprintf(fp,"        gcv <- SSE/(1-gcv.pen*df/n)^2\n");
  fprintf(fp,"\n");
  fprintf(fp,"      if(gcv < gcv.best && df>=df.lim[1] && df<=df.lim[2]){\n");
  fprintf(fp,"        gcv.best <- gcv     \n");
  fprintf(fp,"        c.best <- c.hat\n");
  fprintf(fp,"        log.lambda.best <- llambda\n");
  fprintf(fp,"        lambda.best <- exp(log.lambda.best)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(gcv.best==Inf){\n");
  fprintf(fp,"      df.err <- (df.vec-df.lim[1])^2+(df.vec-df.lim[2])^2\n");
  fprintf(fp,"      ind.j <- order(df.err)[1]\n");
  fprintf(fp,"      log.lambda.best <- log.lambda.vec[ind.j]\n");
  fprintf(fp,"      lambda.best <- exp(log.lambda.best)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    diff <- exp(log.lambda.best+inc) - lambda.best\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(diff/lambda.best <= rel.tol)\n");
  fprintf(fp,"      break\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## create log.lambda.vec for next pass\n");
  fprintf(fp,"    log.lambda.min <- log.lambda.best - floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.min <- max(LAMBDAMIN, log.lambda.min)  \n");
  fprintf(fp,"    log.lambda.max <- log.lambda.best + floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.max <- min(LAMBDAMAX, log.lambda.max)  \n");
  fprintf(fp,"    inc <- inc/2\n");
  fprintf(fp,"    log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc) \n");
  fprintf(fp,"    ind <- which.equal(log.lambda.vec, log.lambda.old)\n");
  fprintf(fp,"    log.lambda.vec <- log.lambda.vec[!ind]\n");
  fprintf(fp,"    log.lambda.old <- c(log.lambda.old, log.lambda.vec)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Now get norm and y.mat\n");
  fprintf(fp,"  norm <- numeric(P)\n");
  fprintf(fp,"  y.mat <- matrix(0,n,P)\n");
  fprintf(fp,"  for(i in 1:P){\n");
  fprintf(fp,"    y.mat[,i] <- w[i]^2*Gram[[i]]%%*%%c.hat\n");
  fprintf(fp,"    norm[i] <- t(c.hat)%%*%%(w[i]^4*Gram[[i]])%%*%%c.hat\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return(list(lambda=lambda.best, gcv=gcv.best, w=w, Gram=Gram, \n");
  fprintf(fp,"              y.mat=y.mat, X=X, y=y, norm=norm))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"######## GET INITIAL WEIGHTS VIA FULL SMOOTHING SPLINE ###\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"           \n");
  fprintf(fp,"get.lambda.w <- function(X, y, order, lambda.lim, nlambda, w='L2', \n");
  fprintf(fp,"                         gcv.pen, rel.tol, wt.pow, w.0, w.lim=c(1E-10, 1E10),\n");
  fprintf(fp,"                         f.est='trad', rel.K, rel.lambda, rel.theta, maxit, \n");
  fprintf(fp,"                         nK, seed, cv, init.cv='gcv', cat.pos, min.distinct){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(f.est=='trad'){\n");
  fprintf(fp,"    ans1 <- get.w(X=X, y=y, order=order, lambda.lim=lambda.lim, \n");
  fprintf(fp,"                  nlambda=nlambda, gcv.pen=gcv.pen, wt.pow=wt.pow, \n");
  fprintf(fp,"                  rel.tol=rel.tol, w=w, w.lim=w.lim, cat.pos=cat.pos)\n");
  fprintf(fp,"    ans1$Gram <- NULL\n");
  fprintf(fp,"    w <- ans1$w\n");
  fprintf(fp,"\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  else{ #f.est = 'cosso'\n");
  fprintf(fp,"    ans1 <- venus.cv(X,y, order=order, w=1, wt.pow=0, seed=seed, K.lim='data',\n");
  fprintf(fp,"                     nK=nK, gcv.pen=gcv.pen, rel.K=rel.K, nlambda=nlambda, \n");
  fprintf(fp,"                     lambda.lim=lambda.lim, rel.lambda=rel.lambda, \n");
  fprintf(fp,"                     rel.theta=rel.theta, maxit=maxit, cv=init.cv,\n");
  fprintf(fp,"                     cat.pos=cat.pos, min.distinct=min.distinct)\n");
  fprintf(fp,"    ans1$Gram <- NULL\n");
  fprintf(fp,"    fit.venus <- venus(X, y, order=order, K=ans1$K, gcv.pen=gcv.pen, maxit=5, \n");
  fprintf(fp,"                       rel.tol=1E-2, cv='gcv', lambda.0=ans1$lambda.0, \n");
  fprintf(fp,"                       w=ans1$w, cat.pos=cat.pos, min.distinct=min.distinct)\n");
  fprintf(fp,"    ans.norm <- get.venus.norms(fit.venus)\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(w[1]=='L2'){\n");
  fprintf(fp,"      w <- ans.norm$L2.norm^wt.pow\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else if(w[1]=='RKHS'){\n");
  fprintf(fp,"      w <- ans.norm$H2.norm^wt.pow\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ans2 <- get.lambda0(X=X, y=y, order=order, lambda.lim=lambda.lim, \n");
  fprintf(fp,"                      nlambda=nlambda, gcv.pen=gcv.pen, rel.tol=rel.tol, w=w,\n");
  fprintf(fp,"                      cat.pos=cat.pos)\n");
  fprintf(fp,"  lambda <- ans2$lambda\n");
  fprintf(fp,"  norm <- ans2$norm\n");
  fprintf(fp,"  y.mat <- ans2$y.mat\n");
  fprintf(fp,"  w <- ans2$w\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(lambda=lambda,norm=norm, X=X, y=y, Gram=ans2$Gram, y.mat=y.mat, \n");
  fprintf(fp,"              w=w))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"################### CROSS VALIDATE ON K ##################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"venus.cv <- function(X, y, order=1, w='L2', K.lim='data', nK=5, gcv.pen, \n");
  fprintf(fp,"           rel.K=1E-2, nlambda=10, lambda.lim=c(1E-10, 1E2), rel.lambda=1E-2, \n");
  fprintf(fp,"           theta.0, rel.theta=1E-2, maxit=2, cv='gcv', wt.pow=1, seed=220, \n");
  fprintf(fp,"           w.lim=c(1E-12, 1E10), lambda.0='est', Gram, f.est='trad', \n");
  fprintf(fp,"           init.cv='gcv', cat.pos=\"auto\", min.distinct=7){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)  \n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(missing(theta.0))\n");
  fprintf(fp,"    theta.0 <- rep(1,P)\n");
  fprintf(fp,"  theta.00 <- theta.0\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ## Identify categorical vars \n");
  fprintf(fp,"  if(length(cat.pos)>0 && cat.pos[1]=='auto'){\n");
  fprintf(fp,"   # scan for variables with 5 or less distinct values. \n");
  fprintf(fp,"    cat.pos <- numeric(0)\n");
  fprintf(fp,"    for(i in 1:p){\n");
  fprintf(fp,"      unique.i <- unique(X[,i])\n");
  fprintf(fp,"      if(length(unique.i)<=min.distinct || is.character(X[,i])){\n");
  fprintf(fp,"        cat.pos <- c(cat.pos, i)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  \n");
  fprintf(fp," ## First get lambda.0 \n");
  fprintf(fp,"  if(lambda.0=='est' || missing(Gram)){\n");
  fprintf(fp,"    ans.lambda0 <- get.lambda.w(X=X, y=y, order=order, lambda.lim=lambda.lim, \n");
  fprintf(fp,"                  nlambda=nlambda, w=w, gcv.pen=gcv.pen, rel.tol=rel.lambda, \n");
  fprintf(fp,"                  wt.pow=wt.pow, w.0=1, w.lim=w.lim, f.est=f.est, rel.K=rel.K,\n");
  fprintf(fp,"                  rel.lambda=rel.lambda, rel.theta=rel.theta, maxit=maxit,\n");
  fprintf(fp,"                  nK=nK, seed=seed, cv=cv, init.cv=init.cv, cat.pos=cat.pos,\n");
  fprintf(fp,"                  min.distinct=min.distinct)\n");
  fprintf(fp,"    w <- ans.lambda0$w\n");
  fprintf(fp,"\n");
  fprintf(fp,"### Changed this for numerical stability\n");
  fprintf(fp,"#    lambda.0 <- ans.lambda0$lambda*1E-3\n");
  fprintf(fp,"    lambda.0 <- ans.lambda0$lambda*1E0\n");
  fprintf(fp,"#######################################\n");
  fprintf(fp,"    Gram <- ans.lambda0$Gram\n");
  fprintf(fp,"    ans.lambda0$Gram <- NULL\n");
  fprintf(fp,"    norm <- ans.lambda0$norm\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(K.lim[1]=='data'){\n");
  fprintf(fp,"    K.lim <- c(0, 10*P)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Now conduct grid search on K\n");
  fprintf(fp,"  K.min <- K.lim[1]\n");
  fprintf(fp,"  K.max <- K.lim[2]\n");
  fprintf(fp,"  KMIN <- log10(K.min+1)\n");
  fprintf(fp,"  KMAX <- log10(K.max+1)\n");
  fprintf(fp,"  log.K.min <- KMIN\n");
  fprintf(fp,"  log.K.max <- KMAX\n");
  fprintf(fp,"\n");
  fprintf(fp,"  inc <- (log.K.max - log.K.min)/(nK-1)\n");
  fprintf(fp,"  log.K.vec <- seq(log.K.min, log.K.max, inc)\n");
  fprintf(fp,"  log.K.old <- log.K.vec\n");
  fprintf(fp,"  pen.best <- Inf\n");
  fprintf(fp,"\n");
  fprintf(fp,"  repeat{\n");
  fprintf(fp,"    nK.now <- length(log.K.vec)\n");
  fprintf(fp,"    theta.0 <- theta.00\n");
  fprintf(fp,"    for(j in 1:nK.now){\n");
  fprintf(fp,"      K <- 10^(log.K.vec[j])-1\n");
  fprintf(fp,"      fit.venus <- venus(X=X,y=y,order=order, K=K,gcv.pen=gcv.pen, \n");
  fprintf(fp,"       lambda.0=lambda.0, theta.0=theta.0,rel.tol=rel.theta, \n");
  fprintf(fp,"       maxit=maxit, Gram=Gram, cv=cv, w=w, seed=seed,\n");
  fprintf(fp,"       cat.pos=cat.pos, min.distinct=min.distinct)\n");
  fprintf(fp,"      fit.venus$Gram <- NULL\n");
  fprintf(fp,"      if(cv=='bic')\n");
  fprintf(fp,"        pen <- fit.venus$bic\n");
  fprintf(fp,"      else\n");
  fprintf(fp,"        pen <- fit.venus$gcv\n");
  fprintf(fp,"\n");
  fprintf(fp,"#theta.0 <- fit.venus$theta\n");
  fprintf(fp,"\n");
  fprintf(fp,"      if(pen < pen.best){\n");
  fprintf(fp,"        pen.best <- pen     \n");
  fprintf(fp,"        K.best <- K\n");
  fprintf(fp,"        log.K.best <- log.K.vec[j]\n");
  fprintf(fp,"        theta.best <- fit.venus$theta\n");
  fprintf(fp,"\n");
  fprintf(fp,"#theta.00 <- theta.0\n");
  fprintf(fp,"\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    diff <- 10^(log.K.best+inc)-1 - K.best\n");
  fprintf(fp,"    if(diff/(K.best+1E-6) <= rel.K)\n");
  fprintf(fp,"      break\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## create K.vec for next pass\n");
  fprintf(fp,"    log.K.min <- log.K.best - floor(nK/2)/2*inc\n");
  fprintf(fp,"    log.K.min <- max(KMIN, log.K.min)  \n");
  fprintf(fp,"    log.K.max <- log.K.best + floor(nK/2)/2*inc\n");
  fprintf(fp,"    log.K.max <- min(KMAX, log.K.max)  \n");
  fprintf(fp,"    inc <- inc/2\n");
  fprintf(fp,"    log.K.vec <- seq(log.K.min, log.K.max, inc) \n");
  fprintf(fp,"    ind <- which.equal(log.K.vec, log.K.old)\n");
  fprintf(fp,"    log.K.vec <- log.K.vec[!ind]\n");
  fprintf(fp,"    log.K.old <- c(log.K.old, log.K.vec)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(K=K.best, lambda.0=lambda.0, theta=theta.best, Gram=Gram,\n");
  fprintf(fp,"              pen=pen.best, w=w, cat.pos=cat.pos))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"################# CROSS VALIDATE ON K and wt.pow #########\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"venus.cv2 <- function(X, y, order=1, w='L2', K.lim='data', nK=5, gcv.pen, \n");
  fprintf(fp,"           rel.K=1E-2, nlambda=10, lambda.lim=c(1E-10, 1E2), rel.lambda=1E-2, \n");
  fprintf(fp,"           theta.0, rel.theta=1E-2, maxit=2, cv='gcv', wt.pow.vec=c(.5,1,2), \n");
  fprintf(fp,"           seed=220, w.lim=c(1E-12, 1E10), lambda.0='est', Gram, \n");
  fprintf(fp,"           f.est='trad', init.cv='gcv'){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  pen.best <- Inf\n");
  fprintf(fp,"  for(wt.pow in wt.pow.vec){\n");
  fprintf(fp,"    if(wt.pow==0){\n");
  fprintf(fp,"      f.est.now <- 'trad'\n");
  fprintf(fp,"      init.cv.now <- 'gcv'\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else{\n");
  fprintf(fp,"      f.est.now <- f.est\n");
  fprintf(fp,"      init.cv.now <- init.cv\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"   \n");
  fprintf(fp,"    ans.now <- venus.cv(X=X, y=y, order=order, w=w, K.lim=K.lim, nK=nK, \n");
  fprintf(fp,"                gcv.pen=gcv.pen, rel.K=rel.K, nlambda=nlambda, \n");
  fprintf(fp,"                lambda.lim=lambda.lim, rel.lambda=rel.lambda, theta.0=theta.0,\n");
  fprintf(fp,"                rel.theta=rel.theta, maxit=maxit, cv=cv, wt.pow=wt.pow, \n");
  fprintf(fp,"                seed=seed, w.lim=w.lim, lambda.0=lambda.0, Gram=Gram, \n");
  fprintf(fp,"                f.est=f.est.now, init.cv=init.cv.now)\n");
  fprintf(fp,"\n");
  fprintf(fp,"cat(\"\n wt.pow =\",wt.pow, \"     pen =\",ans.now$pen) \n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(ans.now$pen < pen.best){\n");
  fprintf(fp,"      ans.best <- ans.now\n");
  fprintf(fp,"      pen.best <- ans.now$pen\n");
  fprintf(fp,"      wt.pow.best <- wt.pow\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  print(wt.pow.best) \n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(ans.best)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"########### CROSS VALIDATE FOR TRAD SMOOTHING SPLINE #####\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"pen.spline.cv <- function(X, y, order=1, gcv.pen=1.01, nlambda=5,\n");
  fprintf(fp,"                          lambda.lim=c(1E-8,1E4), theta, df.lim,\n");
  fprintf(fp,"                          rel.lambda=1E-2, cv='gcv', seed=220){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  p <- ncol(X)\n");
  fprintf(fp,"  if(order==1)\n");
  fprintf(fp,"    P <- p\n");
  fprintf(fp,"  else\n");
  fprintf(fp,"    P <- p + choose(p,2)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(missing(df.lim)){\n");
  fprintf(fp,"    df.lim <- c(0,.75*n)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(missing(theta))\n");
  fprintf(fp,"    theta <- rep(1,P)\n");
  fprintf(fp,"\n");
  fprintf(fp," ## shift and rescale x's to [0,1]\n");
  fprintf(fp,"  for(i in 1:p){\n");
  fprintf(fp,"    if(any( X[,i]<0 | X[,i]>1)){\n");
  fprintf(fp,"      X[,i] <- (X[,i]-min(X[,i]))/(max(X[,i])-min(X[,i]))\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Get Gram Matrix\n");
  fprintf(fp,"  Gram <- get.gram(X, X, order=order, cat.pos=numeric(0))\n");
  fprintf(fp,"\n");
  fprintf(fp," ## Now conduct grid search on lambda\n");
  fprintf(fp,"  lambda.min <- lambda.lim[1]\n");
  fprintf(fp,"  lambda.max <- lambda.lim[2]\n");
  fprintf(fp,"  lambdaMIN <- log10(lambda.min+1)\n");
  fprintf(fp,"  lambdaMAX <- log10(lambda.max+1)\n");
  fprintf(fp,"  log.lambda.min <- lambdaMIN\n");
  fprintf(fp,"  log.lambda.max <- lambdaMAX\n");
  fprintf(fp,"\n");
  fprintf(fp,"  inc <- (log.lambda.max - log.lambda.min)/(nlambda-1)\n");
  fprintf(fp,"  log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc)\n");
  fprintf(fp,"  log.lambda.old <- log.lambda.vec\n");
  fprintf(fp,"  pen.best <- Inf\n");
  fprintf(fp,"\n");
  fprintf(fp,"  repeat{\n");
  fprintf(fp,"    nlambda.now <- length(log.lambda.vec)\n");
  fprintf(fp,"    df.vec <- rep(0, length(log.lambda.vec))\n");
  fprintf(fp,"    for(j in 1:nlambda.now){\n");
  fprintf(fp,"      lambda <- 10^(log.lambda.vec[j])-1\n");
  fprintf(fp,"\n");
  fprintf(fp,"      fit.spline<- pen.spline(X, y, order=order, lambda.0=lambda, theta=theta,\n");
  fprintf(fp,"                              seed=seed, gcv.pen=gcv.pen, Gram=Gram, cv=cv)\n");
  fprintf(fp,"      if(cv=='bic')\n");
  fprintf(fp,"        pen <- fit.spline$bic\n");
  fprintf(fp,"      else\n");
  fprintf(fp,"        pen <- fit.spline$gcv\n");
  fprintf(fp,"\n");
  fprintf(fp,"      df.vec[j] <- df <- fit.spline$df\n");
  fprintf(fp,"\n");
  fprintf(fp,"      if(pen < pen.best && df>=df.lim[1] && df<=df.lim[2]){\n");
  fprintf(fp,"        pen.best <- pen     \n");
  fprintf(fp,"        lambda.best <- lambda\n");
  fprintf(fp,"        log.lambda.best <- log.lambda.vec[j]\n");
  fprintf(fp,"        theta.best <- fit.spline$theta\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    if(pen.best==Inf){\n");
  fprintf(fp,"      df.err <- (df.vec-df.lim[1])^2+(df.vec-df.lim[2])^2\n");
  fprintf(fp,"      ind.j <- order(df.err)[1]\n");
  fprintf(fp,"      log.lambda.best <- log.lambda.vec[ind.j]\n");
  fprintf(fp,"      lambda.best <- exp(log.lambda.best)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    diff <- 10^(log.lambda.best+inc)-1 - lambda.best\n");
  fprintf(fp,"    if(diff/(lambda.best+1E-6) <= rel.lambda)\n");
  fprintf(fp,"      break\n");
  fprintf(fp,"\n");
  fprintf(fp,"   ## create lambda.vec for next pass\n");
  fprintf(fp,"    log.lambda.min <- log.lambda.best - floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.min <- max(lambdaMIN, log.lambda.min)  \n");
  fprintf(fp,"    log.lambda.max <- log.lambda.best + floor(nlambda/2)/2*inc\n");
  fprintf(fp,"    log.lambda.max <- min(lambdaMAX, log.lambda.max)  \n");
  fprintf(fp,"    inc <- inc/2\n");
  fprintf(fp,"    log.lambda.vec <- seq(log.lambda.min, log.lambda.max, inc) \n");
  fprintf(fp,"    ind <- which.equal(log.lambda.vec, log.lambda.old)\n");
  fprintf(fp,"    log.lambda.vec <- log.lambda.vec[!ind]\n");
  fprintf(fp,"    log.lambda.old <- c(log.lambda.old, log.lambda.vec)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(lambda=lambda.best, Gram=Gram, pen=pen.best))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"###### REWRITE QP.solve to change error to warning #######\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"solve.QP2 <- function (Dmat, dvec, Amat, bvec, meq = 0, factorized = FALSE)\n");
  fprintf(fp,"{\n");
  fprintf(fp,"    n <- nrow(Dmat)\n");
  fprintf(fp,"    q <- ncol(Amat)\n");
  fprintf(fp,"    if (missing(bvec))\n");
  fprintf(fp,"        bvec <- rep(0, q)\n");
  fprintf(fp,"    if (n != ncol(Dmat))\n");
  fprintf(fp,"        stop(\"Dmat is not symmetric!\")\n");
  fprintf(fp,"    if (n != length(dvec))\n");
  fprintf(fp,"        stop(\"Dmat and dvec are incompatible!\")\n");
  fprintf(fp,"    if (n != nrow(Amat))\n");
  fprintf(fp,"        stop(\"Amat and dvec are incompatible!\")\n");
  fprintf(fp,"    if (q != length(bvec))\n");
  fprintf(fp,"        stop(\"Amat and bvec are incompatible!\")\n");
  fprintf(fp,"    if ((meq > q) || (meq < 0))\n");
  fprintf(fp,"        stop(\"Value of meq is invalid!\")\n");
  fprintf(fp,"    iact <- rep(0, q)\n");
  fprintf(fp,"    nact <- 0\n");
  fprintf(fp,"    r <- min(n, q)\n");
  fprintf(fp,"    sol <- rep(0, n)\n");
  fprintf(fp,"    crval <- 0\n");
  fprintf(fp,"    work <- rep(0, 2 * n + r * (r + 5)/2 + 2 * q + 1)\n");
  fprintf(fp,"    iter <- rep(0, 2)\n");
  fprintf(fp,"    res1 <- .Fortran(\"qpgen2\", as.double(Dmat), dvec = as.double(dvec),\n");
  fprintf(fp,"        as.integer(n), as.integer(n), sol = as.double(sol), \n");
  fprintf(fp,"        crval = as.double(crval),\n");
  fprintf(fp,"        as.double(Amat), as.double(bvec), as.integer(n), as.integer(q),\n");
  fprintf(fp,"        as.integer(meq), iact = as.integer(iact), nact = as.integer(nact),\n");
  fprintf(fp,"        iter = as.integer(iter), work = as.double(work), \n");
  fprintf(fp,"        ierr = as.integer(factorized),\n");
  fprintf(fp,"        PACKAGE = \"quadprog\")\n");
  fprintf(fp,"    if (res1$ierr == 1){\n");
  fprintf(fp,"        #warning(\"constraints are inconsistent, no solution!\")\n");
  fprintf(fp,"        res1$sol <- rep(-1,n)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else if (res1$ierr == 2){\n");
  fprintf(fp,"        #warning(\"matrix D in quadratic function is not positive definite!\")\n");
  fprintf(fp,"        res1$sol <- rep(-2,n)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    list(solution = res1$sol, value = res1$crval, \n");
  fprintf(fp,"        unconstrainted.solution = res1$dvec, iterations = res1$iter, \n");
  fprintf(fp,"        iact = res1$iact[1:res1$nact])\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"svd2 <- function (x, nu = min(n, p), nv = min(n, p), LINPACK = FALSE)\n");
  fprintf(fp,"{\n");
  fprintf(fp,"    x <- as.matrix(x)\n");
  fprintf(fp,"    if (any(!is.finite(x)))\n");
  fprintf(fp,"        stop(\"infinite or missing values in 'x'\")\n");
  fprintf(fp,"    dx <- dim(x)\n");
  fprintf(fp,"    n <- dx[1]\n");
  fprintf(fp,"    p <- dx[2]\n");
  fprintf(fp,"    if (!n || !p)\n");
  fprintf(fp,"        stop(\"0 extent dimensions\")\n");
  fprintf(fp,"    if (is.complex(x)) {\n");
  fprintf(fp,"       res <- La.svd2(x, nu, nv)\n");
  fprintf(fp,"       return(list(d = res$d, u = if (nu) res$u, v = if (nv) Conj(t(res$vt))))\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    if (!LINPACK) {\n");
  fprintf(fp,"        res <- La.svd2(x, nu, nv, method = \"dgesvd\")\n");
  fprintf(fp,"        return(list(d = res$d, u = if (nu) res$u, v = if (nv) t(res$vt)))\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    if (!is.numeric(x))\n");
  fprintf(fp,"        stop(\"argument to 'svd' must be numeric\")\n");
  fprintf(fp,"    if (nu == 0) {\n");
  fprintf(fp,"        job <- 0\n");
  fprintf(fp,"        u <- double(0)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else if (nu == n) {\n");
  fprintf(fp,"        job <- 10\n");
  fprintf(fp,"        u <- matrix(0, n, n)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else if (nu == p) {\n");
  fprintf(fp,"        job <- 20\n");
  fprintf(fp,"        u <- matrix(0, n, p)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else stop(\"'nu' must be 0, nrow(x) or ncol(x)\")\n");
  fprintf(fp,"    job <- job + if (nv == 0)\n");
  fprintf(fp,"        0\n");
  fprintf(fp,"    else if (nv == p || nv == n)\n");
  fprintf(fp,"        1\n");
  fprintf(fp,"    else stop(\"'nv' must be 0 or ncol(x)\")\n");
  fprintf(fp,"    v <- if (job == 0)\n");
  fprintf(fp,"        double(0)\n");
  fprintf(fp,"    else matrix(0, p, p)\n");
  fprintf(fp,"    mn <- min(n, p)\n");
  fprintf(fp,"    mm <- min(n + 1, p)\n");
  fprintf(fp,"    z <- .Fortran(\"dsvdc\", as.double(x), n, n, p, d = double(mm),\n");
  fprintf(fp,"        double(p), u = u, n, v = v, p, double(n), as.integer(job),\n");
  fprintf(fp,"        info = integer(1), DUP = FALSE, PACKAGE = \"base\")[c(\"d\",\n");
  fprintf(fp,"        \"u\", \"v\", \"info\")]\n");
  fprintf(fp,"    if (z$info){\n");
  fprintf(fp,"        stop(gettextf(\"error %%d in 'dsvdc'\", z$info), domain = NA)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    z$d <- z$d[1:mn]\n");
  fprintf(fp,"    if (nv && nv < p)\n");
  fprintf(fp,"        z$v <- z$v[, 1:nv, drop = FALSE]\n");
  fprintf(fp,"    z[c(\"d\", if (nu) \"u\", if (nv) \"v\")]\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"La.svd2 <- function (x, nu = min(n, p), nv = min(n, p), method = c(\"dgesdd\",\n");
  fprintf(fp,"    \"dgesvd\"))\n");
  fprintf(fp,"{\n");
  fprintf(fp,"    if (!is.numeric(x) && !is.complex(x))\n");
  fprintf(fp,"        stop(\"argument to 'La.svd' must be numeric or complex\")\n");
  fprintf(fp,"    if (any(!is.finite(x)))\n");
  fprintf(fp,"        stop(\"infinite or missing values in 'x'\")\n");
  fprintf(fp,"    method <- match.arg(method)\n");
  fprintf(fp,"    x <- as.matrix(x)\n");
  fprintf(fp,"    if (is.numeric(x))\n");
  fprintf(fp,"        storage.mode(x) <- \"double\"\n");
  fprintf(fp,"    n <- nrow(x)\n");
  fprintf(fp,"    p <- ncol(x)\n");
  fprintf(fp,"    if (!n || !p)\n");
  fprintf(fp,"        stop(\"0 extent dimensions\")\n");
  fprintf(fp,"    if (is.complex(x) || method == \"dgesvd\") {\n");
  fprintf(fp,"        if (nu == 0) {\n");
  fprintf(fp,"            jobu <- \"N\"\n");
  fprintf(fp,"            u <- matrix(0, 1, 1)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else if (nu == n) {\n");
  fprintf(fp,"            jobu <- ifelse(n > p, \"A\", \"S\")\n");
  fprintf(fp,"            u <- matrix(0, n, n)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else if (nu == p) {\n");
  fprintf(fp,"            jobu <- ifelse(n > p, \"S\", \"A\")\n");
  fprintf(fp,"            u <- matrix(0, n, p)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else stop(\"'nu' must be 0, nrow(x) or ncol(x)\")\n");
  fprintf(fp,"        if (nv == 0) {\n");
  fprintf(fp,"            jobv <- \"N\"\n");
  fprintf(fp,"            v <- matrix(0, 1, 1)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else if (nv == n) {\n");
  fprintf(fp,"            jobv <- ifelse(n > p, \"A\", \"S\")\n");
  fprintf(fp,"            v <- matrix(0, min(n, p), p)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else if (nv == p) {\n");
  fprintf(fp,"            jobv <- ifelse(n > p, \"S\", \"A\")\n");
  fprintf(fp,"            v <- matrix(0, p, p)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else stop(\"'nv' must be 0, nrow(x) or ncol(x)\")\n");
  fprintf(fp,"        if (is.complex(x)) {\n");
  fprintf(fp,"            u[] <- as.complex(u)\n");
  fprintf(fp,"            v[] <- as.complex(v)\n");
  fprintf(fp,"            res <- .Call(\"La_svd_cmplx\", jobu, jobv, x, double(min(n,\n");
  fprintf(fp,"                p)), u, v, PACKAGE = \"base\")\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else {\n");
  fprintf(fp,"            res <- .Call(\"La_svd\", jobu, jobv, x, double(min(n,\n");
  fprintf(fp,"                p)), u, v, method, PACKAGE = \"base\")\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        return(res[c(\"d\", if (nu) \"u\", if (nv) \"vt\")])\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else {\n");
  fprintf(fp,"        if (nu > 0 || nv > 0) {\n");
  fprintf(fp,"            np <- min(n, p)\n");
  fprintf(fp,"            if (nu <= np && nv <= np) {\n");
  fprintf(fp,"                jobu <- \"S\"\n");
  fprintf(fp,"                u <- matrix(0, n, np)\n");
  fprintf(fp,"                v <- matrix(0, np, p)\n");
  fprintf(fp,"            }\n");
  fprintf(fp,"            else {\n");
  fprintf(fp,"                jobu <- \"A\"\n");
  fprintf(fp,"                u <- matrix(0, n, n)\n");
  fprintf(fp,"                v <- matrix(0, p, p)\n");
  fprintf(fp,"            }\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        else {\n");
  fprintf(fp,"            jobu <- \"N\"\n");
  fprintf(fp,"            u <- matrix(0, 1, 1)\n");
  fprintf(fp,"            v <- matrix(0, 1, 1)\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"        jobv <- \"\"\n");
  fprintf(fp,"        res <- .Call(\"La_svd\", jobu, jobv, x, double(min(n, p)),\n");
  fprintf(fp,"            u, v, method, PACKAGE = \"base\")\n");
  fprintf(fp,"        res <- res[c(\"d\", if (nu) \"u\", if (nv) \"vt\")]\n");
  fprintf(fp,"        if (nu)\n");
  fprintf(fp,"            res$u <- res$u[, 1:min(n, nu), drop = FALSE]\n");
  fprintf(fp,"        if (nv)\n");
  fprintf(fp,"            res$vt <- res$vt[1:min(p, nv), , drop = FALSE]\n");
  fprintf(fp,"        return(res)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fclose(fp);
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double Acosso::setParams(int targc, char **targv)
{
  int  errFlag;
  char pString[1001], lineIn[10001], winput[1000];
  FILE *fp;
  if (targc > 1 && !strcmp(targv[0], "setRpath") && targv[1] != NULL)
  {
    strcpy(Rpath_, targv[1]);
    fp = fopen("RTest","r");
    if (fp == NULL)
    {
      printf("Acosso setParams ERROR: cannot write to test file.\n");
      exit(1);
    }
    fprintf(fp,"library(\"quadprog\")\n");
    fprintf(fp,"quit()\n");
    fclose(fp);
    snprintf(pString,1000,"%s CMD BATCH RTest", Rpath_);
    system(pString);
    fp = fopen("RTest.Rout","r");
    if (fp == NULL)
    {
      printf("Acosso setParams ERROR: R not found in setRpath.\n");
      strcpy(Rpath_, "NONE");
      unlink("RTest");
      exit(1);
    }
    else
    {
      errFlag = 0;
      while (feof(fp) == 0)
      {
        fgets(lineIn, 10000, fp);
        sscanf(lineIn, "%s", winput);
        if (!strcmp(winput, "Error")) errFlag = 1;
      }
      fclose(fp);
      unlink("RTest.Rout");
      if (errFlag == 1)
      {
        printf("Acosso constructor ERROR: R quadprog package not found.\n");
        printf("  INFO: to obtain quadprog, launch R and use\n");
        printf("          install.packages(\"quadprog\")\n");
        unlink("RTest");
        exit(1);
      }
    }
  }
  return 0.0;
}

