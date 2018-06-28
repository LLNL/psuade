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
// Functions for the class PsuadeRegression
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PsuadeRegression.h"
#include "sysdef.h"
#include "PrintingTS.h"

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
PsuadeRegression::PsuadeRegression(int nInputs,int nSamples):
                                   FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_LOCAL;

  // display banner and additonal information
  printAsterisks(PL_INFO, 0);
  printf("*                Psuade Regression Analysis\n");
  printf("* To use this regression, a user needs to go into the\n");
  printf("* PsuadeRegression.cpp file and insert their function\n");
  printf("* (to replace userFunction).\n");
  printEquals(PL_INFO, 0);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PsuadeRegression::~PsuadeRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int PsuadeRegression::initialize(double *, double *)
{
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int PsuadeRegression::genNDGridData(double *X, double *Y, int *N2,
                                    double **X2, double **Y2)
{
  int totPts, ss;

  if ((*N2) <= 0)
  {
    printf("PsuadeRegression::genNDGridData - ERROR detected (N <= 0).\n");
    (*N2) = 0;
    return -1;
  }

  if ((*N2) == -999) return 0;

  genNDGrid(N2, X2);
  if ((*N2) <= 0) return 0;
  totPts = (*N2);

  (*Y2) = new double[totPts];
  checkAllocate(*Y2, "Y2 in PsuadeRegression::genNDGridData");
  for (ss = 0; ss < totPts; ss++)
    (*Y2)[ss] = evaluatePoint(&((*X2)[ss*nInputs_]));

  return 0;
}

// ************************************************************************
// Generate 1D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int PsuadeRegression::gen1DGridData(double *X, double *Y, int ind1,
                                    double *settings, int *NN, 
                                    double **XX, double **YY)
{
  int    totPts, mm, nn;
  double HX, *Xloc;

  if (NN <= 0)
  {
    printf("PsuadeRegression::gen1DGridData - ERROR detected (N <= 0).\n");
    (*NN) = 0;
    return -1;
  }

  totPts = nPtsPerDim_;
  HX = (upperBounds_[ind1] - lowerBounds_[ind1]) / (nPtsPerDim_ - 1); 

  (*NN) = totPts;
  (*XX) = new double[totPts];
  (*YY) = new double[totPts];
  Xloc  = new double[nInputs_];
  checkAllocate(Xloc, "Xloc in PsuadeRegression::gen1DGridData");
  for (nn = 0; nn < nInputs_; nn++) Xloc[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    Xloc[ind1] = HX * mm + lowerBounds_[ind1];
    (*XX)[mm] = Xloc[ind1];
    (*YY)[mm] = evaluatePoint(Xloc);
  }

  delete [] Xloc;
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int PsuadeRegression::gen2DGridData(double *X, double *Y, int ind1,
                                    int ind2, double *settings, int *NN, 
                                    double **XX, double **YY)
{
  int    totPts, mm, nn, index;
  double *HX, *Xloc;

  if (NN <= 0)
  {
    printf("PsuadeRegression::gen2DGridData - ERROR detected (N <= 0).\n");
    (*NN) = 0;
    return -1;
  }

  totPts = nPtsPerDim_ * nPtsPerDim_;
  HX    = new double[2];
  HX[0] = (upperBounds_[ind1] - lowerBounds_[ind1]) / (nPtsPerDim_ - 1); 
  HX[1] = (upperBounds_[ind2] - lowerBounds_[ind2]) / (nPtsPerDim_ - 1); 

  (*NN) = totPts;
  (*XX) = new double[totPts * 2];
  (*YY) = new double[totPts];
  Xloc  = new double[nInputs_];
  checkAllocate(Xloc, "Xloc in PsuadeRegression::gen2DGridData");
  for (nn = 0; nn < nInputs_; nn++) Xloc[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      index = mm * nPtsPerDim_ + nn;
      Xloc[ind1] = HX[0] * mm + lowerBounds_[ind1];
      Xloc[ind2] = HX[1] * nn + lowerBounds_[ind2];
      (*XX)[index*2]   = Xloc[ind1];
      (*XX)[index*2+1] = Xloc[ind2];
      (*YY)[index] = evaluatePoint(Xloc);
    }
  }

  delete [] Xloc;
  delete [] HX;
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PsuadeRegression::gen3DGridData(double *X, double *Y, int ind1,
                                    int ind2, int ind3, double *settings, 
                                    int *NN, double **XX, double **YY)
{
  int    totPts, mm, nn, pp, index;
  double *HX, *Xloc;

  if (NN <= 0)
  {
    printf("PsuadeRegression::gen3DGridData - ERROR detected (N <= 0).\n");
    (*NN) = 0;
    return -1;
  }

  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  HX    = new double[3];
  HX[0] = (upperBounds_[ind1] - lowerBounds_[ind1]) / (nPtsPerDim_ - 1); 
  HX[1] = (upperBounds_[ind2] - lowerBounds_[ind2]) / (nPtsPerDim_ - 1); 
  HX[2] = (upperBounds_[ind3] - lowerBounds_[ind3]) / (nPtsPerDim_ - 1); 

  (*NN) = totPts;
  (*XX) = new double[totPts * 3];
  (*YY) = new double[totPts];
  Xloc  = new double[nInputs_];
  checkAllocate(Xloc, "Xloc in PsuadeRegression::gen3DGridData");
  for (nn = 0; nn < nInputs_; nn++) Xloc[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        index = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        Xloc[ind1] = HX[0] * mm + lowerBounds_[ind1];
        Xloc[ind2] = HX[1] * nn + lowerBounds_[ind2];
        Xloc[ind3] = HX[2] * pp + lowerBounds_[ind3];
        (*XX)[index*3]   = Xloc[ind1];
        (*XX)[index*3+1] = Xloc[ind2];
        (*XX)[index*3+2] = Xloc[ind3];
        (*YY)[index] = evaluatePoint(Xloc);
      }
    }
  }

  delete [] Xloc;
  delete [] HX;
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PsuadeRegression::gen4DGridData(double *X, double *Y, int ind1, int ind2,
                                  int ind3, int ind4, double *settings, 
                                  int *NN, double **XX, double **YY)
{
  int    totPts, mm, nn, pp, qq, index;
  double *HX, *Xloc;

  if (NN <= 0)
  {
    printf("PsuadeRegression::gen4DGridData - ERROR detected (N <= 0).\n");
    (*NN) = 0;
    return -1;
  }

  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  HX    = new double[4];
  HX[0] = (upperBounds_[ind1] - lowerBounds_[ind1]) / (nPtsPerDim_ - 1); 
  HX[1] = (upperBounds_[ind2] - lowerBounds_[ind2]) / (nPtsPerDim_ - 1); 
  HX[2] = (upperBounds_[ind3] - lowerBounds_[ind3]) / (nPtsPerDim_ - 1); 
  HX[3] = (upperBounds_[ind4] - lowerBounds_[ind4]) / (nPtsPerDim_ - 1); 

  (*NN) = totPts;
  (*XX) = new double[totPts * 4];
  (*YY) = new double[totPts];
  Xloc  = new double[nInputs_];
  checkAllocate(Xloc, "Xloc in PsuadeRegression::gen4DGridData");
  for (nn = 0; nn < nInputs_; nn++) Xloc[nn] = settings[nn]; 
    
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
          Xloc[ind1] = HX[0] * mm + lowerBounds_[ind1];
          Xloc[ind2] = HX[1] * nn + lowerBounds_[ind2];
          Xloc[ind3] = HX[2] * pp + lowerBounds_[ind3];
          Xloc[ind4] = HX[3] * qq + lowerBounds_[ind4];
          (*XX)[index*4]   = Xloc[ind1];
          (*XX)[index*4+1] = Xloc[ind2];
          (*XX)[index*4+2] = Xloc[ind3];
          (*XX)[index*4+3] = Xloc[ind4];
          (*YY)[index] = evaluatePoint(Xloc);
        }
      }
    }
  }

  delete [] Xloc;
  delete [] HX;
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double PsuadeRegression::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y;
  userFunction(nInputs_, X, iOne, &Y);
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double PsuadeRegression::evaluatePoint(int npts, double *X, double *Y)
{
  int ii, iOne=1;
  for (ii = 0 ; ii < npts; ii++)
     userFunction(nInputs_, &X[nInputs_*ii], iOne, &Y[ii]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double PsuadeRegression::evaluatePointFuzzy(double *X, double &stdev)
{
  int    iOne=1;
  double Y;
  evaluatePoint(iOne, X, &Y);
  stdev = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double PsuadeRegression::evaluatePointFuzzy(int npts, double *X, double *Y,
                                            double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ii = 0 ; ii < npts; ii++) Ystd[ii] = 0;
  return 0.0;
}

// ************************************************************************
// User function 
// ------------------------------------------------------------------------
int PsuadeRegression::userFunction(int nInputs, double *inputs, 
                                   int nOutputs, double *outputs)
{
  if (nInputs != 10)
  {
    printf("PsuadeRegression user function ERROR: nInputs != 10\n");
    exit(1);
  }
  if (nOutputs != 1)
  {
    printf("PsuadeRegression user function ERROR: nOutputs != 1\n");
    exit(1);
  }
  double MEA = inputs[0];
  double TEM = inputs[1];
  double CO2 = inputs[2];
  double A   = inputs[3];
  double B   = inputs[4];
  double C   = inputs[5];
  double D   = inputs[6];
  double E   = inputs[7];
  double F   = inputs[8];
  double G   = inputs[9];
  double muH2O, dtemp;

  muH2O = 1.002*pow(10.0, 1.3272*(293.15-TEM-0.001053*(TEM-293.15)*
           (TEM-293.15))/(TEM-168.15));
  dtemp = (A * MEA + B) * TEM + C * MEA + D;
  outputs[0] = muH2O*exp(dtemp*(CO2*(E*MEA+F*TEM+G)+1)*MEA/(TEM*TEM));
  return 0;
}

