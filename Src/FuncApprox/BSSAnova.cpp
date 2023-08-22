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
// Functions for the class BSSAnova
// Reference: This class uses Curt Storlie's (LANL) BSSAnova method (in R)
// DATE   : 2015
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "BSSAnova.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class BSSAnova
// ------------------------------------------------------------------------
BSSAnova::BSSAnova(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  char lineIn[1001];
  faID_ = PSUADE_RS_ACOSSO;
  FILE *fp = fopen("RTest","w");
  if (fp == NULL)
  {
    printf("BSSAnova constructor ERROR: cannot open file.\n");
    exit(1);
  }
  fprintf(fp,"quit()\n");
  fprintf(fp,"n\n");
  fclose(fp);
  system("R CMD BATCH RTest");
  fp = fopen("RTest.Rout","r");
  if (fp == NULL)
  {
    strcpy(Rpath_, "NONE");
    printf("BSSAnova constructor INFO: R not found.\n");
  }
  else
  {
    unlink("RTest.Rout");
    strcpy(Rpath_, "R");
    printf("R found at : \n");
    system("which R");
  }
  unlink("RTest");
  initialized_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
BSSAnova::~BSSAnova()
{
  FILE *fp = fopen("bssanova.R", "r");
  if (fp != NULL) unlink("bssanova.R");
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int BSSAnova::initialize(double *XIn, double *YIn)
{
  int ii;

  if (initialized_ == 0)
  {
    genBSSAnova();
    vecSamInps_.setLength(nSamples_*nInputs_);
    vecSamOuts_.setLength(nSamples_);
  }
  for (ii = 0; ii < nSamples_; ii++) vecSamInps_[ii] = YIn[ii];
  initInputScaling(XIn, vecSamInps_.getDVector(), 1);
  initialized_ = 1;
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int BSSAnova::genNDGridData(double *XIn, double *YIn, int *N,double **XOut, 
                          double **YOut)
{
  int    totPts, ii, jj, *indices;
  double *XX;
  psVector vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize 
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(N, &XX);
  if ((*N) == 0) return 0;
  totPts = (*N);
  (*XOut) = XX;

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("BSSAnova interpolation begins....\n");
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  psIVector vecInds;
  vecInds.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecInds[ii] = ii;
  runBSSAnova(totPts, XX, *YOut, nInputs_, vecInds.getIVector());
  if (outputLevel_ >= 1) printf("BSSAnova interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int BSSAnova::gen1DGridData(double *XIn, double *YIn, int ind1,
                   double *settings, int *N, double **XOut, double **YOut)
{
  int    totPts, ii, kk, *indices, iOne=1;
  double HX;
  psVector vecXT, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
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
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("BSSAnova interpolation begins....\n");
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  psIVector vecInds;
  vecInds.setLength(1);
  vecInds[0] = ind1;
  runBSSAnova(totPts,vecXT.getDVector(), *YOut,iOne,vecInds.getIVector());
  if (outputLevel_ >= 1) printf("BSSAnova interpolation completed.\n");
  (*N) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results 
// ------------------------------------------------------------------------
int BSSAnova::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                   double *settings, int *N, double **XOut, double **YOut)
{
  int    totPts, ii, jj, kk, index, *indices, iTwo=2;
  psVector vecXT, vecHX, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        vecXT[index*nInputs_+kk] = settings[kk]; 
      vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("BSSAnova interpolation begins....\n");
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  psIVector vecInds;
  vecInds[0] = ind1;
  vecInds[1] = ind2;
  runBSSAnova(totPts,vecXT.getDVector(),*YOut,iTwo,vecInds.getIVector());
  if (outputLevel_ >= 1) printf("BSSAnova interpolation completed.\n");
  (*N)  = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int BSSAnova::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                       int ind3, double *settings, int *N, double **XOut, 
                       double **YOut)
{
  int totPts, ii, jj, kk, ll, index, *indices, iThree=3;
  psVector vecHX, vecXT, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();

  vecXT.setLength(totPts*nInputs_);
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
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("BSSAnova interpolation begins....\n");
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  psIVector vecInds;
  vecInds.setLength(3);
  vecInds[0] = ind1;
  vecInds[1] = ind2;
  vecInds[2] = ind3;
  runBSSAnova(totPts,vecXT.getDVector(),*YOut,iThree, vecInds.getIVector());
  if (outputLevel_ >= 1) printf("BSSAnova interpolation completed.\n");
  (*N)  = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int BSSAnova::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                          int ind3, int ind4, double *settings, int *N, 
                          double **XOut, double **YOut)
{
  int totPts, ii, jj, kk, ll, mm, index, *indices, iFour=4;
  psVector vecHX, vecXT, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*N) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();

  vecXT.setLength(totPts*nInputs_);
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
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (outputLevel_ >= 1) printf("BSSAnova interpolation begins....\n");
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  psIVector vecInds;
  vecInds[0] = ind1;
  vecInds[1] = ind2;
  vecInds[2] = ind3;
  vecInds[3] = ind4;
  runBSSAnova(totPts,vecXT.getDVector(),*YOut,iFour,vecInds.getIVector());
  if (outputLevel_ >= 1) printf("BSSAnova interpolation completed.\n");
  (*N)  = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double BSSAnova::evaluatePoint(double *X)
{
  int    iOne=1, iZero=0;
  double Y=0.0;
  runBSSAnova(iOne, X, &Y, iZero, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double BSSAnova::evaluatePoint(int npts, double *X, double *Y)
{
  int iZero=0;
  runBSSAnova(npts, X, Y, iZero, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double BSSAnova::evaluatePointFuzzy(double *X, double &std)
{
  double Y = evaluatePoint(X);
  std = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double BSSAnova::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ii = 0; ii < npts; ii++) Ystd[ii] = 0.0;
  return 0.0;
}

// ************************************************************************
// run Storlie's BSSAnova code
// ------------------------------------------------------------------------
int BSSAnova::runBSSAnova(int nPoints, double *XN, double *YN, int nInps, 
                          int *indices)
{
  int  ii, ss, count;
  char pString[1001];
  FILE *fp;

  //**/ open a file to be run with R
  fp = fopen("psuade_bssanova.R", "w");
  if (fp == NULL)
  {
    printf("BSSAnova ERROR: cannot open file.\n");
    exit(1);
  }
  fprintf(fp,"source(\"bssanova.R\")\n");
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
          fprintf(fp,"%e),\n", vecSamInps_[ss*nInputs_+ii]);
      else
      {
        fprintf(fp,"%e, ", vecSamInps_[ss*nInputs_+ii]);
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
      fprintf(fp,"%e)\n", vecSamOuts_[ss]);
    else
    {
      fprintf(fp,"%e, ", vecSamOuts_[ss]);
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
  double hx, ddata;
  if (nInps == 1)
  {
    i1 = indices[0];
    n1 = nPtsPerDim_;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (ii != i1)
      {
        ddata = (XN[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",ddata,n1,n1); 
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        hx = hx / VecXStds_[i1];
        ddata = (VecLBs_[i1] - VecXMeans_[i1]) / VecXStds_[i1];
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", ddata);
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
        ddata = (XN[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",ddata,n2,n2);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        hx = hx / VecXStds_[i1];
        ddata = (VecLBs_[i1] - VecXMeans_[i1]) / VecXStds_[i1];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i2];
        ddata = (VecLBs_[i2] - VecXMeans_[i2]) / VecXStds_[i2];
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", ddata);
        fprintf(fp,"V = matrix(v,%d,1)\n",n1);
        fprintf(fp,"v2 <- V \n");
        fprintf(fp,"for ( nt in 1:%d ) {\n", n1-1);
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
        ddata = (XN[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",ddata,n3,n3);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        hx = hx / VecXStds_[i1];
        ddata = (VecLBs_[i1] - VecXMeans_[i1]) / VecXStds_[i1];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i2];
        ddata = (VecLBs_[i2] - VecXMeans_[i2]) / VecXStds_[i2];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i3];
        ddata = (VecLBs_[i3] - VecXMeans_[i3]) / VecXStds_[i3];
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", ddata);
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
        ddata = (XN[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        fprintf(fp,"V = matrix(rep(%e,%d),%d,1)\n",ddata,n4,n4);
      }
      else if (ii == i1)
      {
        hx = (VecUBs_[i1] - VecLBs_[i1]) / (n1 - 1); 
        hx = hx / VecXStds_[i1];
        ddata = (VecLBs_[i1] - VecXMeans_[i1]) / VecXStds_[i1];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i2];
        ddata = (VecLBs_[i2] - VecXMeans_[i2]) / VecXStds_[i2];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i3];
        ddata = (VecLBs_[i3] - VecXMeans_[i3]) / VecXStds_[i3];
        fprintf(fp,"hx <- %e\n", ddata);
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
        hx = hx / VecXStds_[i4];
        ddata = (VecLBs_[i4] - VecXMeans_[i4]) / VecXStds_[i4];
        fprintf(fp,"hx <- %e\n", hx);
        fprintf(fp,"v = 0 : %d\n", n1 - 1);
        fprintf(fp,"v <- v * hx + %e\n", ddata);
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
        {
          ddata = (XN[ss*nInputs_+ii] - VecXMeans_[ii]) / VecXStds_[ii];
          fprintf(fp,"%e),\n", ddata);
        }
        else
        {
          ddata = (XN[ss*nInputs_+ii] - VecXMeans_[ii]) / VecXStds_[ii];
          fprintf(fp,"%e, ", ddata);
          count++;
          if (count >= 10)
          {
            count = 0;
            fprintf(fp, "\n");
          } 
        }
      }
    }
    if (nPoints == 1)
    {
      fprintf(fp,"nrow=%d,", nPoints+1);
      fprintf(fp,"ncol=%d)\n", nInputs_);
      fprintf(fp,"XX = rbind(XN,XN);\n");
      fprintf(fp,"XN <- XX\n");
    } 
    else
    {
      fprintf(fp,"nrow=%d,", nPoints);
      fprintf(fp,"ncol=%d)\n", nInputs_);
    }
  } 
  fprintf(fp,"bssanova.fit <- bssanova(X,Y,BTE=c(1000,2000,1),n.terms=25,\n");
  fprintf(fp,"                         lambda=2, int.order=2)\n");
  fprintf(fp,"bssanova.pred <- predict.bssanova(XN, bssanova.fit, nreal=1000)\n");
  fprintf(fp,"bssanova.yhat <- bssanova.pred$yhat\n");
  fprintf(fp,"write(bssanova.yhat,\"psuade_bssanova_data\")\n");
  fprintf(fp,"quit()\n");
  fprintf(fp,"n\n");
  fclose(fp);
  //**/ run R script
  if (!strcmp(Rpath_, "NONE"))
  {
    printf("BSSAnvoa ERROR: R not found.\n");
    printf("Please enter the full path of R (loaded with quadprog) : ");
    scanf("%s", Rpath_);
    fgets(pString, 500, stdin);
    fp = fopen(Rpath_,"r");
    if (fp == NULL)
    {
      printf("BSSAnvoa ERROR: R still not found.\n");
      exit(1);
    }
  }
  snprintf(pString,1000,"%s CMD BATCH psuade_bssanova.R", Rpath_);
  system(pString);
  //**/ read output
  fp = fopen("psuade_bssanova_data", "r");
  if (fp == NULL)
  {
    printf("BSSAnova ERROR: cannot open output file.\n");
    printf("For diagnosis, run the following and track errors:\n");
    printf("> %s CMD BATCH psuade_bssanova.R\n", Rpath_);
    exit(1);
  }
  for (ss = 0; ss < nPoints; ss++) fscanf(fp, "%lg", &YN[ss]);
  fclose(fp);
  unlink("psuade_bssanova_data");
  unlink("psuade_bssanova.R");
  unlink("psuade_bssanova.Rout");
  return 0;
}

// ************************************************************************
// generate BSSAnova code
// ------------------------------------------------------------------------
int BSSAnova::genBSSAnova()
{
  int  ii, ss, count;
  FILE *fp;

  fp = fopen("bssanova.R", "w");
  if (fp == NULL)
  {
    printf("BSSAnova genBSSAnova ERROR: cannot open file.\n");
    exit(1);
  }
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"############### BSSANOVA Function ########################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"bssanova <- function(X,y,BTE=c(200,1000,1),cat.pos=\"auto\",min.distinct=2,\n");
  fprintf(fp,"             n.terms=25, lambda=2, int.order=2, l2.min=0, priorprob=.5){\n");
  fprintf(fp,"\n");
  fprintf(fp,"########## INPUTS #########################################################\n");
  fprintf(fp,"## X         - a matrix of predictors.\n");
  fprintf(fp,"## y         - a vector of responses.\n");
  fprintf(fp,"## BTE       - 3-vector specifying number of iterations for\n");
  fprintf(fp,"##             (i) burn-in (toss out this many from the beginning),\n");
  fprintf(fp,"##             (ii) total number of MCMC iterations, and\n");
  fprintf(fp,"##             (iii) how often to record (keep every E samples).\n");
  fprintf(fp,"## cat.pos   - vector containg the columns to be treated as categorical\n");
  fprintf(fp,"## min.distinct - minimum number of distinct values to treat a predictor as\n");
  fprintf(fp,"##                continuous (instead of categorical).  Only used if\n");
  fprintf(fp,"##                categorical=\"auto\".\n");
  fprintf(fp,"## n.terms   - No. of eigenvalues to use to approximate BSSANOVA covariance\n");
  fprintf(fp,"##             More the merrier, but slows down computation with too many.\n");
  fprintf(fp,"## lambda    - Half-Cauchy hyperparameter\n");
  fprintf(fp,"## int.order - the order of interactions to consider (1 or 2).\n");
  fprintf(fp,"## l2.min    - minimum proportion of variance by a functional component to be\n");
  fprintf(fp,"##             included in the variable order ranking (set to 0 to include all)\n");
  fprintf(fp,"## prior.prob- the prior probability that a given component is uninformative\n");
  fprintf(fp,"###########################################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"########## OUTPUTS ########################################################\n");
  fprintf(fp,"## bss.model - a list with the elements...\n");
  fprintf(fp,"##             * fittedvalues,fittedsds are the posterior means and sds \n");
  fprintf(fp,"##                       of f at the data points:\n");
  fprintf(fp,"##             * inprob is the posterior inclusion probability\n");
  fprintf(fp,"##             * l2 is the posterior distribution of the l2 norm of each component\n");
  fprintf(fp,"##             * curves and curvessd are the posterior means and sds of the\n");
  fprintf(fp,"##               individual components f_{ij}\n");
  fprintf(fp,"##             * r is the posterior distribution of the variance r\n");
  fprintf(fp,"##             * predmn is the posterior predicive mean of f(xnew) \n");
  fprintf(fp,"##             * term1 and term2 index the compnents f_{ij}.  For example, if\n");
  fprintf(fp,"##               term1[j]=3 and term2[j]=4 then curves[,j] is the posterior \n");
  fprintf(fp,"##               mean of f_{3,4}.  Terms with terms1[j]=terms2[j] \n");
  fprintf(fp,"##               are main effects\n");
  fprintf(fp,"##             * dev is the posterior samples of the deviance\n");
  fprintf(fp,"##             * int is the posterior samples of the intercept\n");
  fprintf(fp,"##             * sigma is the posterior samples of the error sd\n");
  fprintf(fp,"## yhat      - predicted values at the rows of the X matrix\n");
  fprintf(fp,"## Rsq       - R^2 of the posterior mean function fit\n");
  fprintf(fp,"## order     - Variable order importance according to total variance\n");
  fprintf(fp,"##             (assuming uniform distribution for inputs)\n");
  fprintf(fp,"###########################################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"  X <- as.matrix(X)\n");
  fprintf(fp,"  n <- nrow(X)\n");
  fprintf(fp,"  nx <- ncol(X)\n");
  fprintf(fp,"  if(length(dimnames(X)[[2]]) == 0){\n");
  fprintf(fp,"    dimnames(X)[[2]] <- list()\n");
  fprintf(fp,"    dimnames(X)[[2]] <- paste(\"x\", 1:nx, sep=\"\")\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  runs<-BTE[2]       #number of MCMC samples\n");
  fprintf(fp,"  burn<-BTE[1]	     #Toss out these many\n");
  fprintf(fp,"  update<-25         #How often to display the current iteration?\n");
  fprintf(fp,"  n.terms<-n.terms   #Number of eigenvalues to keep\n");
   fprintf(fp,"  lambda<-lambda     #Half-Cauchy hyperparameter			\n");
  fprintf(fp,"\n");
  fprintf(fp,"  interactions <- (int.order>1) #Include interactions?\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ## Identify categorical vars \n");
  fprintf(fp,"  if(cat.pos[1]=='auto'){\n");
  fprintf(fp,"   # scan for variables with min.distinct or less distinct values. \n");
  fprintf(fp,"    cat.pos <- numeric(0)\n");
  fprintf(fp,"    for(i in index(1,nx)){\n");
  fprintf(fp,"      unique.i <- unique(X[,i])\n");
  fprintf(fp,"      if(length(unique.i)<=min.distinct || is.character(X[,i])){\n");
  fprintf(fp,"        cat.pos <- c(cat.pos, i)\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  categorical <- rep(FALSE, nx)\n");
  fprintf(fp,"  categorical[cat.pos] <- TRUE\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  bss.model<-BSSANOVA(y,X,categorical=categorical,lambda=lambda,\n");
  fprintf(fp,"          nterms=n.terms,twoway=interactions,runs=runs,burn=burn,\n");
  fprintf(fp,"          update=update,priorprob=priorprob)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  res <- y-bss.model$fittedvalues\n");
  fprintf(fp,"  yhat <- bss.model$fittedvalues\n");
  fprintf(fp,"  SSE <- sum(res^2)\n");
  fprintf(fp,"  SSTot <- sum((y-mean(y))^2)\n");
  fprintf(fp,"  SSReg <- SSTot - SSE\n");
  fprintf(fp,"  Rsq <- SSReg/SSTot\n");
  fprintf(fp,"\n");
  fprintf(fp,"  l2.t <- numeric(nx)\n");
  fprintf(fp,"  l2 <- colMeans(bss.model$l2)\n");
  fprintf(fp,"  for(j in 1:nx)\n");
  fprintf(fp,"    l2.t[j] <- sum(l2[bss.model$term1==j | bss.model$term2==j])/sum(l2)\n");
  fprintf(fp,"  var.ord <- order(-l2.t)\n");
  fprintf(fp,"  num.gt.min <- sum(l2.t>l2.min)\n");
  fprintf(fp,"  var.ord <- var.ord[1:num.gt.min]\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  return(list(bss.model=bss.model, yhat=yhat, Rsq=Rsq, order=var.ord))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"############### BSSANOVA Prediction ######################\n");
  fprintf(fp,"##########################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"predict.bssanova <- function(X.new, object, nreal=NA){\n");
  fprintf(fp,"\n");
  fprintf(fp,"###################### INPUTS ################################\n");
  fprintf(fp,"## X.new - a matrix of new values for the predictors\n");
  fprintf(fp,"## obj - a fitted acosso object\n");
  fprintf(fp,"## nreal - how many posterior realizations to obtain (defaults to same as fit object)\n");
  fprintf(fp,"##############################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"###################### OUTPUT #################################\n");
  fprintf(fp,"## yhat - a nrow(X.new) vector of the posterior mean of predicted y's\n");
  fprintf(fp,"## yreal - a (nreal) x (nrow(X.new)) matrix:\n");
  fprintf(fp,"##          each row gives X predictions for a given posterior realization \n");
  fprintf(fp,"## curves - a (# functional components) x (nreal) x (nrow(X.new)) array:\n");
  fprintf(fp,"##           e.g., curves[j,r,] provides predictions for j-th functional \n");
  fprintf(fp,"##	     component for the r-th posterior realization\n");
  fprintf(fp,"###############################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"  X.new <- as.matrix(X.new)\n");
  fprintf(fp,"  nx <- ncol(X.new)\n");
  fprintf(fp,"  if(length(dimnames(X.new)[[2]]) == 0){\n");
  fprintf(fp,"    dimnames(X.new)[[2]] <- list()\n");
  fprintf(fp,"    dimnames(X.new)[[2]] <- paste(\"x\", 1:nx, sep=\"\")\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  bss.model <- object$bss.model\n");
  fprintf(fp,"\n");
  fprintf(fp,"  pred.bss <- BSSANOVA.predict(bss.model, X.new, runs=bss.model$burn+nreal)\n");
  fprintf(fp,"  yhat <- pred.bss$pred.mn\n");
  fprintf(fp,"  yreal <- pred.bss$y.pred\n");
  fprintf(fp,"\n");
  fprintf(fp,"  return(list(yhat=yhat, yreal=yreal, curves=pred.bss$curves))\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"predict.bssanova.at.mean <- function(X.new, object){\n");
  fprintf(fp,"\n");
  fprintf(fp,"  X.new <- as.matrix(X.new)\n");
  fprintf(fp,"  nx <- ncol(X.new)\n");
  fprintf(fp,"  if(length(dimnames(X.new)[[2]]) == 0){\n");
  fprintf(fp,"    dimnames(X.new)[[2]] <- list()\n");
  fprintf(fp,"    dimnames(X.new)[[2]] <- paste(\"x\", 1:nx, sep=\"\")\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  bss.model <- object$bss.model\n");
  fprintf(fp,"\n");
  fprintf(fp,"  pred.bss <- BSSANOVA.predict.at.mean(bss.model, X.new)\n");
  fprintf(fp,"  return(pred.bss)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"########################################################################\n");
  fprintf(fp,"#################### REST OF BOSCO CODE GOES HERE ######################\n");
  fprintf(fp,"########################################################################\n");
  fprintf(fp,"\n");
  fprintf(fp,"BSSANOVA<-function(y,x,categorical=NA,\n");
  fprintf(fp,"    runs=10000,burn=2000,update=10,\n");
  fprintf(fp,"    ae=.01,be=.01,priorprob=0.5,nterms=25,lambda=2,\n");
  fprintf(fp,"    main=T,twoway=F,const=10,init.sigma=NA){\n");
  fprintf(fp,"\n");
  fprintf(fp,"#########  Definitions of the input    ################:\n");
  fprintf(fp,"#  y is the n*1 vector of data\n");
  fprintf(fp,"#  x is the n*p design matrix\n");
  fprintf(fp,"#  categorical is the p-vector indicating (TRUE/FALSE) which\n");
  fprintf(fp,"#       columns of x are categorical variables\n");
  fprintf(fp,"#  runs is the number of MCMC samples to generate\n");
  fprintf(fp,"#  burn is the number of samples to discard as burnin\n");
  fprintf(fp,"#  update is the number of interations between displays\n");
  fprintf(fp,"#  error variance sigma^2~Gamma(ae,be)\n");
  fprintf(fp,"#  priorprob is the prior includion probability for \n");
  fprintf(fp,"#       each functional component\n");
  fprintf(fp,"#  nterms is the number of eigenvector to retain\n");
  fprintf(fp,"#  lambda is the hyperparameters in the half-Cauchy prior\n");
  fprintf(fp,"#  main indicates whether to include main effects\n");
  fprintf(fp,"#  twoway incicates whether to include interactions\n");
  fprintf(fp,"#  const is the relative variance for the polynomial trend\n");
  fprintf(fp,"#  init.sigma is the initial value for the error sd\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set some sample size parameters \n");
  fprintf(fp,"  n<-length(y)\n");
  fprintf(fp,"  ncurves<-0\n");
  fprintf(fp,"  p<-ncol(x)\n");
  fprintf(fp,"  if(is.na(mean(categorical))){categorical<-rep(F,p)}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(main){ncurves<-ncurves+p}\n");
  fprintf(fp,"  if(twoway){ncurves<-ncurves+p*(p-1)/2}\n");
  fprintf(fp,"  term1<-term2<-rep(0,ncurves)\n");
  fprintf(fp,"  if(p>0){o1<-order(x[,1])}\n");
  fprintf(fp,"  if(p>1){o2<-order(x[,2])}\n");
  fprintf(fp,"  if(p>2){o3<-order(x[,3])}\n");
  fprintf(fp,"  if(p>3){o4<-order(x[,4])}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #Set up the covariance matrices for the main effects\n");
  fprintf(fp,"  B<-array(0,c(ncurves,n,nterms))\n");
  fprintf(fp,"  Gamma<-array(0,c(ncurves,nterms,nterms))\n");
  fprintf(fp,"  D<-matrix(0,nterms,ncurves)\n");
  fprintf(fp,"  GammaB<-array(0,c(ncurves,nterms,n))\n");
  fprintf(fp,"  count<-1\n");
  fprintf(fp,"\n");
  fprintf(fp,"  theta.basis<-make.eig.functions(const=const,npts=500,nterms=nterms,nknots=100)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set up the covariances for the main effects\n");
  fprintf(fp,"  if(main){for(j in 1:p){\n");
  fprintf(fp,"     term1[count]<-term2[count]<-j\n");
  fprintf(fp,"     if(!categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cont(x[,j],theta.basis)\n");
  fprintf(fp,"        eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"        Gamma[count,,]<-eig$vectors\n");
  fprintf(fp,"        D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"        GammaB[count,,]<-t(Gamma[count,,])%%*%%t(B[count,,])\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"     if(categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cat(x[,j],ncats=max(x[,j]),nterms=nterms)\n");
  fprintf(fp,"        eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"        Gamma[count,,]<-eig$vectors\n");
  fprintf(fp,"        D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"        GammaB[count,,]<-t(Gamma[count,,])%%*%%t(B[count,,])\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set up the covariances for the two-way interactions\n");
  fprintf(fp,"  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){\n");
  fprintf(fp,"     term1[count]<-j1;term2[count]<-j2\n");
  fprintf(fp,"     term<-1\n");
  fprintf(fp,"     for(k1 in 1:round(sqrt(nterms))){\n");
  fprintf(fp,"         for(k2 in 1:round(sqrt(nterms)+1)){\n");
  fprintf(fp,"            if(term<=nterms){\n");
  fprintf(fp,"               B[count,,term]<-B[j1,,k1]*B[j2,,k2]\n");
  fprintf(fp,"               term<-term+1\n");
  fprintf(fp,"             }\n");
  fprintf(fp,"          }\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"     eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"     Gamma[count,,]<-eig$vectors\n");
  fprintf(fp,"     D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"     GammaB[count,,]<-t(Gamma[count,,])%%*%%t(B[count,,])\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  ########                 Initial values      ###################\n");
  fprintf(fp,"  int<-mean(y)\n");
  fprintf(fp,"  sige<-ifelse(is.na(init.sigma),0.5*sd(lm(y~x)$residuals),init.sigma)\n");
  fprintf(fp,"  taue<-1/sige^2\n");
  fprintf(fp,"  curves<-matrix(0,n,ncurves)\n");
  fprintf(fp,"  curfit<-int+sige*apply(curves,1,sum)\n");
  fprintf(fp,"  r<-rep(0,ncurves)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #keep track of the mean of the fitted values\n");
  fprintf(fp,"  afterburn<-0\n");
  fprintf(fp,"  sumfit<-sumfit2<-rep(0,n)\n");
  fprintf(fp,"  suminout<-rep(0,ncurves)\n");
  fprintf(fp,"  sumcurves<-sumcurves2<-matrix(0,n,ncurves)\n");
  fprintf(fp,"  keepr<-keepl2<-matrix(0,runs,ncurves)\n");
  fprintf(fp,"  dev<-keepsige<-keepint<-rep(0,runs)\n");
  fprintf(fp,"  theta<-array(0,c(ncurves,runs,nterms))\n");
  fprintf(fp,"\n");
  fprintf(fp,"  npts<-50\n");
  fprintf(fp,"  if(priorprob==1){mxgx<-grid<-c(seq(0.0001,2,length=npts/2),seq(2,1000,length=npts/2))}\n");
  fprintf(fp,"  if(priorprob<1){mxgx<-grid<-c(seq(0,2,length=npts/2),seq(2,1000,length=npts/2))} \n");
  fprintf(fp,"\n");
  fprintf(fp,"  ########             Start the sampler       ###################\n");
  fprintf(fp,"  countiter<-0\n");
  fprintf(fp,"  start<-proc.time()[3]\n");
  fprintf(fp,"  sdy<-sd(y)\n");
  fprintf(fp,"  acc<-att<-0;mh<-sdy/10;\n");
  fprintf(fp,"  for(i in 1:runs){\n");
  fprintf(fp,"   sumcurv<-apply(curves,1,sum)\n");
  fprintf(fp,"\n");
  fprintf(fp,"   #new sige\n");
  fprintf(fp,"   if(F){cansige<-rnorm(1,sige,mh)\n");
  fprintf(fp,"   if(cansige>0 & cansige<2*sdy){\n");
  fprintf(fp,"      att<-att+1  \n");
  fprintf(fp,"      R<-sum(dnorm(y,int+cansige*sumcurv,cansige,log=T))-\n");
  fprintf(fp,"         sum(dnorm(y,int+sige*sumcurv,sige,log=T))\n");
  fprintf(fp,"      if(runif(1)<exp(R)){sige<-cansige;acc<-acc+1}\n");
  fprintf(fp,"      taue<-1/sige^2\n");
  fprintf(fp,"   }}\n");
  fprintf(fp,"\n");
  fprintf(fp,"   #new taue\n");
  fprintf(fp,"    cantaue<-rnorm(1,taue,0.05*sd(y))\n");
  fprintf(fp,"    if(cantaue>0){\n");
  fprintf(fp,"      cansige<-1/sqrt(cantaue)\n");
  fprintf(fp,"      MHrate<-sum(dnorm(y,int+cansige*apply(curves,1,sum),cansige,log=T))\n");
  fprintf(fp,"      MHrate<-MHrate-sum(dnorm(y,int+sige*apply(curves,1,sum),sige,log=T))\n");
  fprintf(fp,"      MHrate<-MHrate+dgamma(cantaue,ae,rate=be,log=T)-dgamma(taue,ae,rate=be,log=T) \n");
  fprintf(fp,"      if(runif(1)<exp(MHrate)){taue<-cantaue;sige<-cansige}\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"   if(i<burn & att>50){\n");
  fprintf(fp,"     mh<-mh*ifelse(acc/att<0.2,0.5,1)*ifelse(acc/att>.6,1.5,1)\n");
  fprintf(fp,"     acc<-att<-0\n");
  fprintf(fp,"   }\n");
  fprintf(fp,"\n");
  fprintf(fp,"   #new intercept\n");
  fprintf(fp,"   int<-rnorm(1,mean(y-sige*sumcurv),sige/sqrt(n))\n");
  fprintf(fp," \n");
  fprintf(fp,"   #new curves\n");
  fprintf(fp,"   for(j in 1:ncurves){\n");
  fprintf(fp,"      #first draw the sd:\n");
  fprintf(fp,"\n");
  fprintf(fp,"      ncp<-min(c(median(keepr[1:i,j]),10))\n");
  fprintf(fp,"      if(i==1){npc<-5}\n");
  fprintf(fp,"      ncp<-min(c(median(keepr[,j]),2))\n");
  fprintf(fp,"      mxgx<-grid<-c(0,qt(seq(0.01,0.99,length=npts-1),1,ncp))\n");
  fprintf(fp,"      mxgx<-grid<-grid[grid>=0]\n");
  fprintf(fp,"      rrr<-y-int-sige*apply(curves[,-j],1,sum)\n");
  fprintf(fp,"      z<-GammaB[j,,]%%*%%rrr\n");
  fprintf(fp,"      for(jjj in 1:length(grid)){\n");
  fprintf(fp,"         mxgx[jjj]<-g(grid[jjj],z,sige,D[,j],priorprob=priorprob,lambda=lambda)-\n");
  fprintf(fp,"                    log(dcan(grid[jjj],ncp,priorprob=priorprob))\n");
  fprintf(fp,"      } \n");
  fprintf(fp,"\n");
  fprintf(fp,"      big<-max(mxgx)\n");
  fprintf(fp,"      highpt<-1.05*max(exp(mxgx-big))\n");
  fprintf(fp,"      ratio<-0\n");
  fprintf(fp,"      if(highpt==0){r[j]<-10;ratio<-1}\n");
  fprintf(fp,"      while(ratio<1){\n");
  fprintf(fp,"        r[j]<-rcan(1,ncp,priorprob)\n");
  fprintf(fp,"        lll<-g(r[j],z,sige,D[,j],priorprob=priorprob,lambda=lambda)-\n");
  fprintf(fp,"             log(dcan(r[j],ncp,priorprob=priorprob))\n");
  fprintf(fp,"        ratio<-exp(lll-big)/highpt/runif(1)\n");
  fprintf(fp,"        if(is.na(ratio)){ratio<-0}\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"\n");
  fprintf(fp,"      #then draw the curve\n");
  fprintf(fp,"      if(r[j]==0){curves[,j]<-0}\n");
  fprintf(fp,"      if(r[j]>0){\n");
  fprintf(fp,"        var<-r[j]/(r[j]*D[,j]+1)\n");
  fprintf(fp,"        theta[j,i,]<-sqrt(var)*rnorm(nterms)\n");
  fprintf(fp,"        theta[j,i,]<-Gamma[j,,]%%*%%(theta[j,i,]+var*z/sige)\n");
  fprintf(fp,"        curves[,j]<-B[j,,]%%*%%theta[j,i,]\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"   }\n");
  fprintf(fp,"\n");
  fprintf(fp,"   #Record results:\n");
  fprintf(fp,"   keepr[i,]<-r  \n");
  fprintf(fp,"   keepl2[i,]<-apply(curves^2,2,mean)\n");
  fprintf(fp,"   fit<-int+sige*apply(curves,1,sum)\n");
  fprintf(fp,"   dev[i]<- -2*sum(dnorm(y,fit,sige,log=T))\n");
  fprintf(fp,"   keepsige[i]<-sige\n");
  fprintf(fp,"   keepint[i]<-int\n");
  fprintf(fp,"\n");
  fprintf(fp,"   if(i>burn){\n");
  fprintf(fp,"      afterburn<-afterburn+1\n");
  fprintf(fp,"      sumfit<-sumfit+fit\n");
  fprintf(fp,"      sumfit2<-sumfit2+fit^2\n");
  fprintf(fp,"      suminout<-suminout+ifelse(r>0,1,0)\n");
  fprintf(fp,"      sumcurves<-sumcurves+sige*curves\n");
  fprintf(fp,"      sumcurves2<-sumcurves2+(sige*curves)^2\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"\n");
  fprintf(fp,"    #display current value of the chain\n");
  fprintf(fp,"#   if(i%%%%update==0){ \n");
  fprintf(fp,"#    par(mfrow=c(2,2))\n");
  fprintf(fp,"#    if(p>0){plot(x[o1,1],y[o1],main=i,col=gray(0.5));\n");
  fprintf(fp,"#      lines(x[o1,1],int+sige*curves[o1,1],col=4)}    \n");
  fprintf(fp,"#    if(p>1){plot(x[o2,2],y[o2],main=i,col=gray(0.5));\n");
  fprintf(fp,"#      lines(x[o2,2],int+sige*curves[o2,2],col=4)}    \n");
  fprintf(fp,"#    plot(keepsige[1:i],type=\"l\")\n");
  fprintf(fp,"#    plot(fit,y,main=i,col=gray(0.5),xlim=range(y))\n");
  fprintf(fp,"#    abline(0,1)\n");
  fprintf(fp,"\n");
  fprintf(fp,"#   }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  stop<-proc.time()[3]\n");
  fprintf(fp,"  print(paste(\"Sampling took\",round(stop-start),\"seconds\"))\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #Calculate posterior means:\n");
  fprintf(fp,"  fitmn<-sumfit/afterburn\n");
  fprintf(fp,"  fitsd<-sqrt(sumfit2/afterburn-fitmn^2)\n");
  fprintf(fp,"  curves<-sumcurves/afterburn\n");
  fprintf(fp,"  curvessd<-sqrt(sumcurves2/afterburn-curves^2)\n");
  fprintf(fp,"  probin<-suminout/afterburn\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");
  fprintf(fp,"#########  Definitions of the output    ################:\n");
  fprintf(fp,"# fittedvalues,fittedsds are the posterior means and sds \n");
  fprintf(fp,"#                        of f at the data points:\n");
  fprintf(fp,"# inprob is the posterior inclusion probability\n");
  fprintf(fp,"# l2 is the posterior distribution of the l2 norm of each component\n");
  fprintf(fp,"# curves and curvessd are the posterior means and sds of the\n");
  fprintf(fp,"#                     individual components f_{ij}\n");
  fprintf(fp,"# r is the posterior distribution of the variance r\n");
  fprintf(fp,"# predmn is the posterior predicive mean of f(xnew) \n");
  fprintf(fp,"# term1 and term2 index the compnents f_{ij}.  For example, if\n");
  fprintf(fp,"#     term1[j]=3 and term2[j]=4 then curves[,j] is the posterior \n");
  fprintf(fp,"#     mean of f_{3,4}.  Terms with terms1[j]=terms2[j] \n");
  fprintf(fp,"#     are main effects\n");
  fprintf(fp,"# dev is the posterior samples of the deviance\n");
  fprintf(fp,"# int is the posterior samples of the intercept\n");
  fprintf(fp,"# sigma is the posterior samples of the error sd\n");
  fprintf(fp,"par(mfrow=c(1,1))\n");
  fprintf(fp,"list(fittedvalues=fitmn,fittedsds=fitsd,inprob=probin,l2=keepl2[burn:runs,],\n");
  fprintf(fp,"curves=curves,curvessd=curvessd,r=keepr[burn:runs,],\n");
  fprintf(fp,"term1=term1,term2=term2,dev=dev,int=keepint,sigma=keepsige,\n");
  fprintf(fp,"x=x,categorical=categorical,nterms=nterms,main=main,\n");
  fprintf(fp,"twoway=twoway,const=const,theta=theta,burn=burn,runs=runs,\n");
  fprintf(fp,"theta.basis=theta.basis)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"BSSANOVA.predict<-function(fit,newx,runs=NA){\n");
  fprintf(fp,"  x<-fit$x\n");
  fprintf(fp,"  xnew<-newx\n");
  fprintf(fp,"  main <- fit$main\n");
  fprintf(fp,"  twoway <- fit$twoway\n");
  fprintf(fp,"  categorical <- fit$categorical\n");
  fprintf(fp,"\n");
  fprintf(fp,"x..<<-x\n");
  fprintf(fp,"xnew..<<-xnew\n");
  fprintf(fp,"\n");
  fprintf(fp,"  all.x<-rbind(x,xnew)\n");
  fprintf(fp,"  if(is.na(runs) || runs>length(fit$int))\n");
  fprintf(fp,"    runs<-length(fit$int)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set some sample size parameters \n");
  fprintf(fp,"  n<-nrow(x)\n");
  fprintf(fp,"  nnew<-nrow(newx)\n");
  fprintf(fp,"  nterms<-fit$nterms\n");
  fprintf(fp,"  ncurves<-0\n");
  fprintf(fp,"  p<-ncol(x)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(main){ncurves<-ncurves+p}\n");
  fprintf(fp,"  if(fit$twoway){ncurves<-ncurves+p*(p-1)/2}\n");
  fprintf(fp,"  term1<-term2<-rep(0,ncurves)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #Set up the covariance matrices for the main effects\n");
  fprintf(fp,"#  B2<-array(0,c(ncurves,nnew,nterms))\n");
  fprintf(fp,"  B<-array(0,c(ncurves,nnew,nterms))  \n");
  fprintf(fp,"  \n");
  fprintf(fp,"  ###########################\n");
  fprintf(fp,"  ## These are new lines ####\n");
  fprintf(fp,"#  Gamma<-array(0,c(ncurves,nterms,nterms))\n");
  fprintf(fp,"#  D<-matrix(0,nterms,ncurves)\n");
  fprintf(fp,"#  GammaB<-array(0,c(ncurves,nterms,nnew))\n");
  fprintf(fp,"  ############################\n");
  fprintf(fp,"  ############################\n");
  fprintf(fp,"  count<-1\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  #set up the covariances for the main effects\n");
  fprintf(fp,"  if(main){for(j in 1:p){\n");
  fprintf(fp,"     term1[count]<-term2[count]<-j\n");
  fprintf(fp,"     if(!categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cont(xnew[,j],fit$theta.basis)\n");
  fprintf(fp,"#        eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"#        Gamma.count<-eig$vectors\n");
  fprintf(fp,"#        D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"#        GammaB[count,,]<-t(Gamma.count)%%*%%t(B[count,,])\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"     if(categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cat(xnew[,j],ncats=max(xnew[,j]),nterms=nterms)\n");
  fprintf(fp,"#        eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"#        Gamma.count<-eig$vectors\n");
  fprintf(fp,"#        D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"#        GammaB[count,,]<-t(Gamma.count)%%*%%t(B[count,,])\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set up the covariances for the two-way interactions\n");
  fprintf(fp,"  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){\n");
  fprintf(fp,"     term1[count]<-j1;term2[count]<-j2\n");
  fprintf(fp,"     term<-1\n");
  fprintf(fp,"     for(k1 in 1:round(sqrt(nterms))){\n");
  fprintf(fp,"         for(k2 in 1:round(sqrt(nterms)+1)){\n");
  fprintf(fp,"            if(term<=nterms){\n");
  fprintf(fp,"               B[count,,term]<-B[j1,,k1]*B[j2,,k2]\n");
  fprintf(fp,"               term<-term+1\n");
  fprintf(fp,"             }\n");
  fprintf(fp,"          }\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"#     eig<-eigen(t(B[count,,])%%*%%(B[count,,]))\n");
  fprintf(fp,"#     Gamma.count<-eig$vectors\n");
  fprintf(fp,"#     D[,count]<-abs(eig$values)   \n");
  fprintf(fp,"#     GammaB[count,,]<-t(Gamma.count)%%*%%t(B[count,,])\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  curves<-array(0,c(ncurves,runs,nnew))\n");
  fprintf(fp,"  y.pred<-matrix(0,runs,nnew)\n");
  fprintf(fp,"  ########             Start the sampler       ###################\n");
  fprintf(fp,"  countiter<-0\n");
  fprintf(fp,"  start<-proc.time()[3]\n");
  fprintf(fp,"  for(i in 1:runs){\n");
  fprintf(fp,"      for(j in 1:ncurves){\n");
  fprintf(fp,"#        curves[j,i,]<-fit$sigma[i]*B2[j,,]%%*%%fit$theta[j,i,]\n");
  fprintf(fp,"        curves[j,i,]<-fit$sigma[i]*B[j,,]%%*%%fit$theta[j,i,]\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      y.pred[i,]<-fit$int[i]+apply(curves[,i,],2,sum)\n");
  fprintf(fp,"      #display current value of the chain\n");
  fprintf(fp,"#     if(i%%%%10==0)\n");
  fprintf(fp,"#       cat(\"\\niteration =\", i,\"out of\",runs)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"\n");
  fprintf(fp,"  pred.mn<-apply(y.pred[(fit$burn+1):fit$runs,],2,mean)\n");
  fprintf(fp,"list(pred.mn=pred.mn, y.pred=y.pred[(fit$burn+1):fit$runs,],\n");
  fprintf(fp,"     int=fit$int[(fit$burn+1):fit$runs], curves=curves[,(fit$burn+1):fit$runs,],\n");
  fprintf(fp,"     term1=fit$term1,term2=fit$term2)}\n");
  fprintf(fp,"\n");
  fprintf(fp,"BSSANOVA.predict.at.mean<-function(fit,newx){\n");
  fprintf(fp,"  x<-fit$x\n");
  fprintf(fp,"  xnew<-newx\n");
  fprintf(fp,"  main <- fit$main\n");
  fprintf(fp,"  twoway <- fit$twoway\n");
  fprintf(fp,"  categorical <- fit$categorical\n");
  fprintf(fp,"  all.x<-rbind(x,xnew)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set some sample size parameters \n");
  fprintf(fp,"  n<-nrow(x)\n");
  fprintf(fp,"  nnew<-nrow(newx)\n");
  fprintf(fp,"  nterms<-fit$nterms\n");
  fprintf(fp,"  ncurves<-0\n");
  fprintf(fp,"  p<-ncol(x)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  if(main){ncurves<-ncurves+p}\n");
  fprintf(fp,"  if(fit$twoway){ncurves<-ncurves+p*(p-1)/2}\n");
  fprintf(fp,"  term1<-term2<-rep(0,ncurves)\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #Set up the covariance matrices for the main effects\n");
  fprintf(fp,"  B<-array(0,c(ncurves,nnew,nterms))  \n");
  fprintf(fp,"  \n");
  fprintf(fp,"  count<-1\n");
  fprintf(fp,"  \n");
  fprintf(fp,"  #set up the covariances for the main effects\n");
  fprintf(fp,"  if(main){for(j in 1:p){\n");
  fprintf(fp,"     term1[count]<-term2[count]<-j\n");
  fprintf(fp,"     if(!categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cont(xnew[,j],fit$theta.basis)\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"     if(categorical[j]){\n");
  fprintf(fp,"        B[count,,]<-makeB.cat(xnew[,j],ncats=max(xnew[,j]),nterms=nterms)\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  #set up the covariances for the two-way interactions\n");
  fprintf(fp,"  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){\n");
  fprintf(fp,"     term1[count]<-j1;term2[count]<-j2\n");
  fprintf(fp,"     term<-1\n");
  fprintf(fp,"     for(k1 in 1:round(sqrt(nterms))){\n");
  fprintf(fp,"         for(k2 in 1:round(sqrt(nterms)+1)){\n");
  fprintf(fp,"            if(term<=nterms){\n");
  fprintf(fp,"               B[count,,term]<-B[j1,,k1]*B[j2,,k2]\n");
  fprintf(fp,"               term<-term+1\n");
  fprintf(fp,"             }\n");
  fprintf(fp,"          }\n");
  fprintf(fp,"     }\n");
  fprintf(fp,"     count<-count+1\n");
  fprintf(fp,"  }}}\n");
  fprintf(fp,"\n");
  fprintf(fp,"  curves<-array(0,c(ncurves,nnew))\n");
  fprintf(fp,"  countiter<-0\n");
  fprintf(fp,"  start<-proc.time()[3]\n");
  fprintf(fp,"  for(j in 1:ncurves){\n");
  fprintf(fp,"    curves[j,]<-mean(fit$sigma)*B[j,,]%%*%%apply(fit$theta[j,,],2,mean)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  y.pred<-mean(fit$int)+apply(curves,2,sum)\n");
  fprintf(fp,"  return(y.pred)}\n");
  fprintf(fp,"\n");
  fprintf(fp,"priorr<-function(r,priorprob=0.5,lambda=2){\n");
  fprintf(fp,"  ifelse(r==0,1-priorprob,priorprob*2*dt(sqrt(r)/lambda,1)/lambda)\n");
  fprintf(fp,"}\n");
  fprintf(fp,"\n");
  fprintf(fp,"rcan<-function(n,ncp,priorprob=0.5){\n");
  fprintf(fp,"  if(priorprob==1){rrr<-abs(rt(n,1,ncp=ncp))}\n");
  fprintf(fp,"  if(priorprob<1){rrr<-ifelse(runif(n,0,1)<0.975,abs(rt(n,1,ncp=ncp)),0)}\n");
  fprintf(fp,"rrr}\n");
  fprintf(fp,"\n");
  fprintf(fp,"dcan<-function(r,ncp,priorprob=0.5){\n");
  fprintf(fp,"  if(priorprob==1){2*dt(r,1,ncp=ncp)}\n");
  fprintf(fp,"  if(priorprob<1){rrr<-ifelse(r==0,1-0.975,0.975*2*dt(r,1,ncp=ncp))}\n");
  fprintf(fp,"rrr}\n");
  fprintf(fp,"\n");
  fprintf(fp,"g<-function(r,z,sige,d,priorprob=.5,lambda=2){\n");
  fprintf(fp,"   lll<- -sum(log((r*d+1)))+sum(r*z*z/(r*d+1))/sige^2\n");
  fprintf(fp,"0.5*lll+log(priorr(r,priorprob,lambda=lambda))}\n");
  fprintf(fp,"\n");
  fprintf(fp,"#Define Bernoulli polynomials\n");
  fprintf(fp,"B0<-function(x){1+0*x}\n");
  fprintf(fp,"B1<-function(x){x-.5}\n");
  fprintf(fp,"B2<-function(x){x^2-x+1/6}\n");
  fprintf(fp,"B3<-function(x){x^3-1.5*x^2+.5*x}\n");
  fprintf(fp,"B4<-function(x){x^4-2*x^3+x^2-1/30}\n");
  fprintf(fp,"B5<-function(x){x^5-2.5*x^4+1.667*x^3-x/6}\n");
  fprintf(fp,"B6<-function(x){x^6-3*x^5+2.5*x^4-.5*x^2+1/42}\n");
  fprintf(fp,"\n");
  fprintf(fp,"makeB.cat<-function(x.pred,ncats,nterms){\n");
  fprintf(fp,"  xxx<-1:ncats\n");
  fprintf(fp,"  sss<-matrix(xxx,ncats,ncats,byrow=T)\n");
  fprintf(fp,"  ttt<-matrix(xxx,ncats,ncats,byrow=F)\n");
  fprintf(fp,"  equals<-ifelse(sss==ttt,1,0)\n");
  fprintf(fp,"  COV<-(ncats-1)*equals/ncats -(1-equals)/ncats\n");
  fprintf(fp,"  COV<-COV/mean(diag(COV))\n");
  fprintf(fp,"  eig<-eigen(COV);\n");
  fprintf(fp,"  Gamma<-eig$vectors%%*%%diag(sqrt(abs(eig$values)))\n");
  fprintf(fp,"  B<-matrix(0,length(x.pred),nterms)\n");
  fprintf(fp,"  for(j in 1:(ncats-1)){\n");
  fprintf(fp,"     dat<-list(y=Gamma[,j],x=xxx)\n");
  fprintf(fp,"     pred.dat<-list(x=x.pred)\n");
  fprintf(fp,"     fit<-lm(y~as.factor(x),data=dat)\n");
  fprintf(fp,"     B[,j]<-predict(fit,newdata=pred.dat)\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"B}      \n");
  fprintf(fp," \n");
  fprintf(fp,"make.basis.bosco<-function(x,n.knots){\n");
  fprintf(fp,"   B<-matrix(1,length(x),n.knots)\n");
  fprintf(fp,"   knots<-seq(0,1,length=n.knots-2)[-(n.knots-2)]\n");
  fprintf(fp,"   B[,2]<-x\n");
  fprintf(fp,"   B[,3]<-x*x\n");
  fprintf(fp,"   for(j in 1:length(knots)){\n");
  fprintf(fp,"      B[,j+3]<-ifelse(x>knots[j],(x-knots[j])^3,0)\n");
  fprintf(fp,"   }\n");
  fprintf(fp,"B}\n");
  fprintf(fp,"\n");
  fprintf(fp,"make.eig.functions<-function(const=10,npts=1000,nterms=25,nknots){\n");
  fprintf(fp,"  xxx<-seq(0,1,length=npts)\n");
  fprintf(fp,"  sss<-matrix(xxx,length(xxx),length(xxx),byrow=T)\n");
  fprintf(fp,"  ttt<-matrix(xxx,length(xxx),length(xxx),byrow=F)\n");
  fprintf(fp,"  diff<-as.matrix(dist(xxx,diag=T,upper=T))\n");
  fprintf(fp,"  COV<-const*(B1(sss)*B1(ttt)+B2(sss)*B2(ttt)/4)-B4(diff)/24\n");
  fprintf(fp,"  COV<-COV/mean(diag(COV[1:npts,1:npts]))\n");
  fprintf(fp,"  eig<-eigen(COV);\n");
  fprintf(fp,"  Gamma<-eig$vectors[,1:nterms]%%*%%\n");
  fprintf(fp,"            diag(sqrt(abs(eig$values[1:nterms])))\n");
  fprintf(fp,"  B<-make.basis.bosco(xxx,nknots)\n");
  fprintf(fp,"  theta<-matrix(0,nknots,nterms)\n");
  fprintf(fp,"  for(j in 1:nterms){\n");
  fprintf(fp,"     theta[,j]<-lm(Gamma[,j]~B-1)$coef\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"theta}  \n");
  fprintf(fp,"\n");
  fprintf(fp,"makeB.cont<-function(x.pred,theta){\n");
  fprintf(fp,"  X<-make.basis.bosco(x.pred,nrow(theta))\n");
  fprintf(fp,"X%%*%%theta}      \n");
  fprintf(fp,"\n");
  fclose(fp);
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double BSSAnova::setParams(int targc, char **targv)
{
  char pString[1001];
  FILE *fp;
  if (targc > 1 && !strcmp(targv[0], "setRpath") && targv[1] != NULL)
  {
    strcpy(Rpath_, targv[1]);
    fp = fopen("RTest","r");
    if (fp == NULL)
    {
      printf("BSSAnova setParams ERROR: cannot write to test file.\n");
      exit(1);
    }
    fprintf(fp,"quit()\n");
    fclose(fp);
    snprintf(pString,1000,"%s CMD BATCH RTest", Rpath_);
    system(pString);
    fp = fopen("RTest.Rout","r");
    if (fp == NULL)
    {
      printf("BSSAnova setParams ERROR: R not found in setRpath.\n");
      strcpy(Rpath_, "NONE");
    }
    else
    {
      fclose(fp);
      unlink("Rtest.Rout");
    }
    unlink("Rtest");
  }
  return 0.0;
}

