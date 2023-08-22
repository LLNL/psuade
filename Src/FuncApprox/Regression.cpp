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
// Functions for the class Regression
// (Up to fourth order only. Maybe not be stable for higher order)
// AUTHOR : CHARLES TONG
// DATE   : 2005
//**/ *********************************************************************
//**/ add code gen and fuzzy with covariance (3/2014)
//**/ change prediction variable method (6/2017)
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sysdef.h"
#include "Psuade.h"
#include "Regression.h"
#include "PsuadeUtil.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
Regression::Regression(int nInputs,int nSamples):FuncApprox(nInputs,nSamples)
{
  faID_  = PSUADE_RS_REGR4;
  order_ = 4;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
Regression::~Regression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int Regression::initialize(double *X, double *Y)
{
  int status;
 
  //**/ =================================================================
  //**/ launch different order regression (up to a maximum of 4)
  //**/ =================================================================
  status = analyze(X, Y);
  if (status != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int Regression::genNDGridData(double *X, double *Y, int *NN, double **XX, 
                              double **YY)
{
  int mm, totPts;

  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ return if there is no request to create lattice points
  //**/ =================================================================
  if ((*NN) == -999) return 0;

  //**/ =================================================================
  //**/ generating regular grid data
  //**/ =================================================================
  genNDGrid(NN, XX);
  if ((*NN) == 0) return 0;
  totPts = (*NN);

  //**/ =================================================================
  //**/ allocate storage for the data points and generate them
  //**/ =================================================================
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  for (mm = 0; mm < totPts; mm++)
    (*YY)[mm] = evaluatePoint(&((*XX)[mm*nInputs_]));
  return 0;
}

// ************************************************************************
// Generate 1D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int Regression::gen1DGridData(double *X, double *Y, int ind1,
                              double *settings, int *NN, 
                              double **XX, double **YY)
{
  int      totPts, mm, nn;
  double   HX;
  psVector VecXLocal;

  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts);
  VecYOut.setLength(totPts);
  (*XX) = VecXOut.takeDVector();
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  VecXLocal.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXLocal[nn] = settings[nn]; 
   
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    VecXLocal[ind1] = HX * mm + VecLBs_[ind1];
    (*XX)[mm] = VecXLocal[ind1];
    (*YY)[mm] = evaluatePoint(VecXLocal.getDVector());
  }
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int Regression::gen2DGridData(double *X, double *Y, int ind1,
                              int ind2, double *settings, int *NN, 
                              double **XX, double **YY)
{
  int      totPts, mm, nn, ind;
  psVector VecHX, VecXLocal;

  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(2);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*2);
  VecYOut.setLength(totPts);
  (*XX) = VecXOut.takeDVector();
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  VecXLocal.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXLocal[nn] = settings[nn]; 
   
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      ind = mm * nPtsPerDim_ + nn;
      VecXLocal[ind1] = VecHX[0] * mm + VecLBs_[ind1];
      VecXLocal[ind2] = VecHX[1] * nn + VecLBs_[ind2];
      (*XX)[ind*2]   = VecXLocal[ind1];
      (*XX)[ind*2+1] = VecXLocal[ind2];
      (*YY)[ind] = evaluatePoint(VecXLocal.getDVector());
    }
  }
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int Regression::gen3DGridData(double *X, double *Y, int ind1,
                              int ind2, int ind3, double *settings, 
                              int *NN, double **XX, double **YY)
{
  int      totPts, mm, nn, pp, ind;
  psVector VecHX, VecXLocal;

  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(3);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*3);
  VecYOut.setLength(totPts);
  (*XX) = VecXOut.takeDVector();
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  VecXLocal.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXLocal[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        ind = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        VecXLocal[ind1] = VecHX[0] * mm + VecLBs_[ind1];
        VecXLocal[ind2] = VecHX[1] * nn + VecLBs_[ind2];
        VecXLocal[ind3] = VecHX[2] * pp + VecLBs_[ind3];
        (*XX)[ind*3]   = VecXLocal[ind1];
        (*XX)[ind*3+1] = VecXLocal[ind2];
        (*XX)[ind*3+2] = VecXLocal[ind3];
        (*YY)[ind] = evaluatePoint(VecXLocal.getDVector());
      }
    }
  }
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int Regression::gen4DGridData(double *X, double *Y, int ind1, int ind2,
                              int ind3, int ind4, double *settings, 
                              int *NN, double **XX, double **YY)
{
  int      totPts, mm, nn, pp, qq, ind;
  psVector VecHX, VecXLocal;

  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("Regression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }
 
  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(4);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  VecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*4);
  VecYOut.setLength(totPts);
  (*XX) = VecXOut.takeDVector();
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  VecXLocal.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXLocal[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        for (qq = 0; qq < nPtsPerDim_; qq++)
        {
          ind = mm*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                nn*nPtsPerDim_*nPtsPerDim_ + pp*nPtsPerDim_ + qq;
          VecXLocal[ind1] = VecHX[0] * mm + VecLBs_[ind1];
          VecXLocal[ind2] = VecHX[1] * nn + VecLBs_[ind2];
          VecXLocal[ind3] = VecHX[2] * pp + VecLBs_[ind3];
          VecXLocal[ind4] = VecHX[3] * qq + VecLBs_[ind4];
          (*XX)[ind*4]   = VecXLocal[ind1];
          (*XX)[ind*4+1] = VecXLocal[ind2];
          (*XX)[ind*4+2] = VecXLocal[ind3];
          (*XX)[ind*4+3] = VecXLocal[ind4];
          (*YY)[ind] = evaluatePoint(VecXLocal.getDVector());
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double Regression::evaluatePoint(double *X)
{
  int    mm, nn, pp, qq, offset;
  double Xdata, Xdata2, Xdata3, Xdata4, Y;

  //**/ =================================================================
  //**/ check to make sure initialization has been called
  //**/ =================================================================
  if (invCovMat_.nrows() <= 0)
  {
    printf("Regression ERROR: initialization has not been done.\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ traverse all orders
  //**/ =================================================================
  Y = VecRegCoeffs_[0];
  offset = 1;
  if (order_ >= 1)
  {
    for (mm = 0; mm < nInputs_; mm++)
    {
       Xdata = (X[mm] - VecXMeans_[mm]) / VecXStds_[mm]; 
       Y += VecRegCoeffs_[mm+1] * Xdata;
       offset++;
    }
  }
  if (order_ >= 2)
  {
    for (mm = 0; mm < nInputs_; mm++)
    {
      Xdata = (X[mm] - VecXMeans_[mm]) / VecXStds_[mm]; 
      for (nn = mm; nn < nInputs_; nn++)
      {
        Xdata2 = (X[nn] - VecXMeans_[nn]) / VecXStds_[nn]; 
        Y += (VecRegCoeffs_[offset++] * Xdata * Xdata2);
      }
    }
  }
  if (order_ >= 3)
  {
    for (mm = 0; mm < nInputs_; mm++)
    {
      Xdata = (X[mm] - VecXMeans_[mm]) / VecXStds_[mm]; 
      for (nn = mm; nn < nInputs_; nn++)
      {
        Xdata2 = (X[nn] - VecXMeans_[nn]) / VecXStds_[nn]; 
        for (pp = nn; pp < nInputs_; pp++)
        {
           Xdata3 = (X[pp] - VecXMeans_[pp]) / VecXStds_[pp]; 
           Y += VecRegCoeffs_[offset++] * Xdata * Xdata2 * Xdata3;
        }
      }
    }
  }
  if (order_ >= 4)
  {
    for (mm = 0; mm < nInputs_; mm++)
    {
      Xdata = (X[mm] - VecXMeans_[mm]) / VecXStds_[mm]; 
      for (nn = mm; nn < nInputs_; nn++)
      {
        Xdata2 = (X[nn] - VecXMeans_[nn]) / VecXStds_[nn]; 
        for (pp = nn; pp < nInputs_; pp++)
        {
          Xdata3 = (X[pp] - VecXMeans_[pp]) / VecXStds_[pp]; 
          for (qq = pp; qq < nInputs_; qq++)
          {
            Xdata4 = (X[qq] - VecXMeans_[qq]) / VecXStds_[qq]; 
            Y += VecRegCoeffs_[offset++] * Xdata * Xdata2 * Xdata3 * Xdata4;
          }
        }
      }
    }
  }
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double Regression::evaluatePoint(int npts, double *X, double *Y)
{
  //**/ =================================================================
  //**/ check to make sure initialization has been called
  //**/ =================================================================
  if (invCovMat_.nrows() <= 0)
  {
    printf("Regression ERROR: initialization has not been done.\n");
    exit(1);
  }
  //**/ =================================================================
  //**/ evaluate all points one at a time
  //**/ =================================================================
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double Regression::evaluatePointFuzzy(double *X, double &std)
{
  int    mm, nn, pp, qq, offset;
  double accum, *Xs, dtmp;
  double Xdata, Xdata2, Xdata3;
  psVector VecXs;

  //**/ =================================================================
  //**/ check to make sure initialization has been called
  //**/ =================================================================
  if (invCovMat_.nrows() <= 0)
  {
    printf("Regression ERROR: initialization has not been done.\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ begin processing: first scale X 
  //**/ =================================================================
  mm = invCovMat_.nrows();
  VecXs.setLength(mm);
  VecXs[0] = 1.0;
  for (mm = 0; mm < nInputs_; mm++)
    VecXs[mm+1] = (X[mm] - VecXMeans_[mm]) / VecXStds_[mm]; 

  //**/ =================================================================
  //**/ evaluate for all orders
  //**/ =================================================================
  accum = VecRegCoeffs_[0];
  offset = 1;
  if (order_ >= 1)
  {
    for (mm = 1; mm <= nInputs_; mm++) 
    {
      accum += VecRegCoeffs_[mm] * VecXs[mm];
      offset++;
    }
    if (order_ >= 2)
    {
      for (mm = 1; mm <= nInputs_; mm++)
      {
        Xdata = VecXs[mm];
        for (nn = mm; nn <= nInputs_; nn++)
        {
          accum += (VecRegCoeffs_[offset] * Xdata * VecXs[nn]);
          VecXs[offset] = Xdata * VecXs[nn];
          offset++;
        }
      }
    }
    if (order_ >= 3)
    {
      for (mm = 1; mm <= nInputs_; mm++)
      {
        Xdata = VecXs[mm];
        for (nn = mm; nn <= nInputs_; nn++)
        {
          Xdata2 = VecXs[nn];
          for (pp = nn; pp <= nInputs_; pp++)
          {
            accum += (VecRegCoeffs_[offset] * Xdata * Xdata2 * VecXs[pp]);
            VecXs[offset] = Xdata * Xdata2 * VecXs[pp];
            offset++;
          }
        }
      }
    }
    if (order_ >= 4)
    {
      for (mm = 1; mm <= nInputs_; mm++)
      {
        Xdata = VecXs[mm];
        for (nn = mm; nn <= nInputs_; nn++)
        {
          Xdata2 = VecXs[nn];
          for (pp = nn; pp <= nInputs_; pp++)
          {
            Xdata3 = VecXs[pp];
            for (qq = pp; qq <= nInputs_; qq++)
            {
              accum += (VecRegCoeffs_[offset]*Xdata*Xdata2*Xdata3*
                        VecXs[qq]);
              VecXs[offset] = Xdata * Xdata2 * Xdata3 * VecXs[qq];
              offset++;
            }
          }
        }
      }
    }
  }
  accum = accum * YStd_ + YMean_;

  //**/ =================================================================
  //**/ compute standard deviation = sqrt(x' * sig^2 inv(X'X) x)
  //**/ =================================================================
  std = 0.0;
  for (mm = 0; mm < offset; mm++)
  {
    dtmp = 0.0;
    for (nn = 0; nn < offset; nn++)
      dtmp += invCovMat_.getEntry(mm,nn) * VecXs[nn];
    std += dtmp * VecXs[mm];
  }
  std = sqrt(std) * YStd_;
  return accum;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double Regression::evaluatePointFuzzy(int npts, double *X, double *Y,
                                      double *Ystd)
{
  //**/ evaluate one point at a time
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&(X[kk*nInputs_]), Ystd[kk]);
  return 0.0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double Regression::setParams(int targc, char **targv)
{
  order_ = *(int *) targv[0];
  if (order_ <= 0 || order_ > 4) order_ = 0;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 4)
    printf("Regression setParam polynomial order = %d\n",order_);
  switch (order_)
  {
    case 0: faID_ = 0;               break;
    case 1: faID_ = PSUADE_RS_REGR1; break;
    case 2: faID_ = PSUADE_RS_REGR2; break;
    case 3: faID_ = PSUADE_RS_REGR3; break;
    case 4: faID_ = PSUADE_RS_REGR4; break;
  }
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int Regression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int Regression::analyze(psVector VecXin, psVector VecY)
{
  int    M, N, ii, mm, nn, nn2, nn3, nn4, info, ind, last, NRevised;
  double SSresid, SStotal, R2, var, *arrayX;
  double esum, ymax, *tmpArray, *UU, *VV, *arrayXX;
  char   pString[101], response[1000];
  FILE   *fp;
  psMatrix eigMatT, MatXX, MatA;
  psVector eigVals, tmpVec;

  //**/ =================================================================
  //**/ preliminary error checking
  //**/ =================================================================
  if (nInputs_ <= 0 || nSamples_ <= 0)
  {
    printf("Regression ERROR: nInputs or nSamples <= 0.\n");
    exit( 1 );
  } 
   
  //**/ =================================================================
  //**/ check for the order (needs order+1 distinct points in each input)
  //**/ =================================================================
  tmpVec.setLength(nSamples_);
  tmpArray = tmpVec.getDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (mm = 0; mm < nSamples_; mm++)
      tmpArray[mm] = VecXin[mm*nInputs_+ii];
    sortDbleList(nSamples_, tmpArray);
    last = 1;
    for (mm = 1; mm < nSamples_; mm++)
    {
      if (tmpArray[mm] != tmpArray[last-1])
      {
        tmpArray[last] = tmpArray[mm];
        last++;
      }
    }
    if (order_ >= last)
    {
      order_ = last - 1;
      printf("* Regression: order reduced to %d - not enough levels.\n",
             order_);
    }
  }

  //**/ =================================================================
  //**/ print linear, quadratic, cubic, or quartic regression analysis
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    if (order_ == 0)
      printf("*               Constant Regression Analysis\n");
    else if (order_ == 1)
      printf("*               Linear Regression Analysis\n");
    else if (order_ == 2)
      printf("*             Quadratic Regression Analysis\n");
    else if (order_ == 3)
      printf("*               cubic Regression Analysis\n");
    else if (order_ == 4)
      printf("*               quartic Regression Analysis\n");
    printf("* R-squared gives a measure of the goodness of the model.\n");
    printf("* R-squared should be close to 1 if it is a good model.\n");
    printf("* TURN ON rs_expert mode to output regression matrix.\n");
    printf("* TURN ON rs_expert mode to output regression function.\n");
    //**/printf("* SET print level to 5 to output regression error splot.\n");
    printf("* SET print level to 4 to output data standard deviations.\n");
    printDashes(PL_INFO, 0);
    printf("* Suggestion: if your parameter ranges are too high, ");
    printf("SCALE them first\n");
    printf("*             using 'irerange' command in PSUADE ");
    printf("command line mode.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ =================================================================
  //**/ optional scaling of the sample matrix (Xin ==> X/VecX)
  //**/ =================================================================
  psVector VecX;
  VecX.setLength(nSamples_ * nInputs_);
  if (psConfig_.MasterModeIsOn())
  {
    printf("* Regression INFO: scaling turned off in master mode.\n");
    printf("*                  To turn scaling on in master mode,\n");
    printf("*                  have rs_expert mode on also.\n");
    initInputScaling(VecXin.getDVector(), VecX.getDVector(), 0);
  }
  else initInputScaling(VecXin.getDVector(), VecX.getDVector(), 1);

  //**/ =================================================================
  //**/ further modify the polynomial order based on number of terms
  //**/ =================================================================
  N = loadXMatrix(VecX, MatXX); 
  if (N == 0) return -1;
  if (N > nSamples_ && order_ >= 0)
  {
    printf("* Regression ERROR: sample too small for order %d.\n",order_);
    printf("                    Try lower order or larger sample.\n");
    return -1;
  }
  if (N > nSamples_)
  {
    printf("* Regression: sample size too small (%d > %d).\n",N,nSamples_);
    return -1;
  }
  M = nSamples_;

  //**/ =================================================================
  //**/ fill the A matrix
  //**/ =================================================================
  psVector VecA;
  VecA.setLength(M*N);
  arrayXX = MatXX.getMatrix1D();
  for (mm = 0; mm < M; mm++) 
    for (nn = 0; nn < N; nn++) 
      VecA[mm+nn*M] = sqrt(VecWghts_[mm]) * arrayXX[mm+nn*M];
  MatA.load(M, N, VecA.getDVector());

  //**/ =================================================================
  //**/ diagnostics 
  //**/ =================================================================
  if (psConfig_.MasterModeIsOn())
  {
    printf("You have the option to store the regression matrix (that\n");
    printf("is, the matrix A in Ax=b) in a matlab file for inspection.\n"); 
    snprintf(pString,100,"Store regression matrix? (y or n) ");
    getString(pString, response);
    if (response[0] == 'y')
    {
      fp = fopen("regression_matrix.m", "w");
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
        fprintf(fp, "%16.6e \n",VecY[mm]);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "A = AA(:,1:%d);\n", N);
      fprintf(fp, "Y = AA(:,%d);\n", N+1);
      fprintf(fp, "B = A \\ Y;\n");
      fclose(fp);
      printf("Regression matrix now in regression_matrix.m\n");
    }
  }

  //**/ =================================================================
  //**/ perform SVD on A
  //**/ =================================================================
  psMatrix MatU, MatV;
  psVector VecS;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("Running SVD ...\n"); 
  info = MatA.computeSVD(MatU, VecS, MatV);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("SVD completed: status = %d (should be 0).\n",info); 

  if (info != 0)
  {
    printf("* Regression ERROR: dgesvd returns a nonzero (%d).\n",info);
    printf("* Regression terminates further processing.\n");
    printf("* To diagnose problem, re-run with rs_expert on.\n");
    return -1;
  }

  //**/ =================================================================
  //**/ eliminate the noise components in S by zeroing small singular
  //**/ values
  //**/ =================================================================
  mm = 0;
  for (nn = 0; nn < N; nn++) if (VecS[nn] < 0) mm++;
  if (mm > 0)
  {
    printf("* Regression WARNING: some of the singular values\n");
    printf("*            are < 0. May spell trouble but will.\n");
    printf("*            proceed anyway (%d).\n",mm);
  }
  if (VecS[0] == 0.0) NRevised = 0;
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++) 
      if (VecS[nn-1] > 0 && VecS[nn]/VecS[nn-1] < 1.0e-8 && 
          VecS[nn] < 1e-12) break;
    if (nn < N) NRevised = nn - 1;
  }
  if (NRevised < N)
  {
    printf("* Regression WARNING: true rank of sample = %d (need >= %d)\n",
           NRevised, N);
    for (nn = 0; nn < N; nn++) 
      printf("Singular value %5d = %e\n",nn+1,VecS[nn]);
    printf("Cut off at n when S[n]/S[n-1] < 1e-8\n");
    printf("* This can be due to the quality of the sample.\n");
  }
  if (psConfig_.MasterModeIsOn())
  {
    printf("Regression: matrix singular values\n");
    printf("The VERY small ones may cause poor numerical accuracy, ");
    printf("but not keeping\n");
    printf("them may ruin the approximation power. So select ");
    printf("them judiously.\n");
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
      if (VecS[nn-1] > 0 && VecS[nn]/VecS[nn-1] < 1.0e-8)
      {
        VecS[nn] = 0.0;
        NRevised--;
      }
    }
    if (NRevised != N) 
      printf("Regression INFO: %d singular values have been truncated.\n",
             N-NRevised);
  }

  //**/ =================================================================
  //**/ coefficients B = V S^{-1} U^T * sq(W) Y
  //**/ =================================================================
  psVector VecW, VecB;
  VecW.setLength(M+N);
  UU = MatU.getMatrix1D();
  for (mm = 0; mm < NRevised; mm++) 
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++) 
      VecW[mm] += UU[nn+mm*M] * sqrt(VecWghts_[nn]) * VecY[nn]; 
  }
  for (nn = 0; nn < NRevised; nn++) VecW[nn] /= VecS[nn];
  for (nn = NRevised; nn < N; nn++) VecW[nn] = 0.0;
  VecB.setLength(N);
  VV = MatV.getMatrix1D();
  for (mm = 0; mm < N; mm++) 
  {
    VecB[mm] = 0.0;
    for (nn = 0; nn < N; nn++) VecB[mm] += VV[nn+mm*N] * VecW[nn]; 
  }

  //**/ =================================================================
  //**/ store eigenvectors VV and eigenvalues SS^2
  //**/ =================================================================
  eigMatT.load(N, N, VV);
  eigVals.load(N, VecS.getDVector());
  for (nn = 0; nn < N; nn++) eigVals[nn] = pow(eigVals[nn], 2.0);

  //**/ =================================================================
  //**/ compute residual and generate training statistics (error)
  //**/ =================================================================
  fp = NULL;
  if (psConfig_.MasterModeIsOn())
  {
    fp = fopen("regression_err_splots.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n",
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "%% This file contains scatter plots of\n");
    fprintf(fp, "%% errors with respect to each input.\n");
    fprintf(fp, "%% This is useful to see which input fits the worst.\n");
    fprintf(fp,"X = [\n");
    for (mm = 0; mm < nSamples_; mm++)
    {
      for (nn = 0; nn < nInputs_; nn++) 
        fprintf(fp,"%16.8e ", VecX[mm*nInputs_+nn]);
      fprintf(fp,"\n");
    }
    fprintf(fp,"];\n");
    fprintf(fp,"R = [\n");
  }

  esum = ymax = 0.0;
  for (mm = 0; mm < M; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++) VecW[mm] += arrayXX[mm+nn*M] * VecB[nn];
    VecW[mm] -= VecY[mm];
    esum = esum + VecW[mm] * VecW[mm] * VecWghts_[mm];
    if (fp != NULL) 
      fprintf(fp, "%6d %24.16e\n", mm+1, VecW[mm]*sqrt(VecWghts_[mm]));
    if (PABS(VecY[mm]) > ymax) ymax = PABS(VecY[mm]);
  }
  esum /= (double) nSamples_;
  esum = sqrt(esum);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("* Regression: interpolation rms error = %11.4e (Ymax=%9.2e)\n",
           esum, ymax); 
  if (fp != NULL) 
  {
    fprintf(fp,"];\n");
    fprintf(fp,"for ii = 1 : size(X,2)\n");
    fprintf(fp,"   plot(X(:,ii), R(:,2), 'x')\n");
    fprintf(fp,"   title('Residual Plot')\n");
    fprintf(fp,"   disp(['Residual versus input ' int2str(ii)])\n");
    fprintf(fp,"   disp('Press Enter to continue')\n");
    fprintf(fp,"   pause\n");
    fprintf(fp,"end\n");
    fclose(fp);
    printf("FILE regression_err_splots.m contains fitting errors.\n");
    printf("     (resubstitution test) with respect to each input.\n");
  }

  //**/ =================================================================
  //**/ compute variance and R2 
  //**/ =================================================================
  computeSS(MatXX, VecY, VecB, SSresid, SStotal);
  R2 = 1.0;
  if (SStotal != 0.0) R2  = 1.0 - SSresid / SStotal;
  if (nSamples_ > N) var = SSresid / (double) (nSamples_ - N);
  else               var = 0.0;
  if (var < 0)
  { 
    if (PABS(var) > 1.0e-12)
    {
      printf("Regression WARNING: variance < 0.\n");
      printf("           Temporarily absolutize var (may have problems).\n");
      var = PABS(var);
    }
    else var = 0;
  }
  VecRegCoeffs_.load(VecB);

  //**/ =================================================================
  //**/ find standard deviation of each coefficient 
  //**/ =================================================================
  computeCoeffVariance(eigMatT, eigVals, var);
  psVector VecBstd;
  VecBstd.setLength(N);
  for (ii = 0; ii < N; ii++)
    VecBstd[ii] = sqrt(invCovMat_.getEntry(ii,ii));

  //**/ =================================================================
  //**/ print out regression coefficients 
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
  {
    if (outputLevel_ > 0) printRC(VecB, VecBstd, MatXX, VecY);
    printf("* Regression R-squared = %10.3e (SSresid,SStotal=%8.1e,%8.1e)\n",
           R2, SSresid, SStotal);
    if ((M - N - 1) > 0)
      printf("* adjusted   R-squared = %10.3e\n",
             1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
    if (R2 < 0)
      printf("* NOTE: negative R2 ==> residual > total sum of square\n");
    if (outputLevel_ > 1) printSRC(VecX, VecB, SStotal);
  }

  //**/ =================================================================
  //**/ generate standalone response surface function
  //**/ =================================================================
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/* Regression interpolator from PSUADE.*/\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   3. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
    fprintf(fp,"main(int argc, char **argv) {\n");
    fprintf(fp,"  int    i, iOne=1, nInps;\n");
    fprintf(fp,"  double X[%d], Y, S;\n",nInputs_);
    fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
    fprintf(fp,"  if (argc < 3) {\n");
    fprintf(fp,"    printf(\"ERROR: not enough argument.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
    fprintf(fp,"  if (fIn == NULL) {\n");
    fprintf(fp,"    printf(\"ERROR: cannot open input file.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
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
    fprintf(fp,"}\n\n");
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/*  Regression interpolation function  */\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"regCoefs[%d] = \n", N);
    fprintf(fp,"{\n");
    for (mm = 0; mm < N; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp,"};\n");
    fprintf(fp,"static double invCovMat[%d][%d] = \n", N, N);
    fprintf(fp,"{\n");
    for (mm = 0; mm < N; mm++)
    {
      fprintf(fp," { %24.16e", invCovMat_.getEntry(mm,0));
      for (nn = 1; nn < N; nn++)
        fprintf(fp,", %24.16e", invCovMat_.getEntry(mm,nn));
      fprintf(fp," },\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static int N=%d;\n",N);
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
    fprintf(fp,"  int    ii, jj, kk, nInps=%d;\n",nInputs_);
    fprintf(fp,"  double y, *x, *x2, std, dtmp;\n");
    fprintf(fp,"  x2 = (double *) malloc(%d * sizeof(double));\n",
            N);
    fprintf(fp,"  for (ii = 0; ii < npts; ii++) {\n");
    fprintf(fp,"    x = &X[ii * %d];\n", nInputs_);
    fprintf(fp,"    y = regCoefs[0];\n");
    fprintf(fp,"    x2[0] = 1.0;\n");
    for (nn = 0; nn < nInputs_; nn++)
    {
      if (VecXMeans_[nn] != 0.0 || VecXStds_[nn] != 1.0)
        fprintf(fp,"    x2[%d] = (x[%d] - %e) / %e;\n", nn+1, nn,
                VecXMeans_[nn], VecXStds_[nn]);
      else
        fprintf(fp,"    x2[%d] = x[%d];\n", nn+1, nn);
    }
    for (nn = 1; nn <= nInputs_; nn++)
    {
      fprintf(fp,"    y += regCoefs[%d] * x2[%d];\n", nn, nn);
    }
    if (order_ >= 2)
    {
      ind = nInputs_ + 1;
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          fprintf(fp, "    y += regCoefs[%d] * x2[%d] * x2[%d];\n",ind,
                  nn,nn2);
          fprintf(fp, "    x2[%d] = x2[%d] * x2[%d];\n",ind, nn,nn2);
          ind++;
        }
      }
    }
    if (order_ >= 3)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            fprintf(fp, "    y += regCoefs[%d] * x2[%d] * ",ind,nn);
            fprintf(fp, "x2[%d] * x2[%d];\n", nn2, nn3);
            fprintf(fp, "    x2[%d]=x2[%d]*x2[%d]*x2[%d];\n",ind,nn,nn2,nn3);
            ind++;
          }
        }
      }
    }
    if (order_ >= 4)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            for (nn4 = nn3; nn4 <= nInputs_; nn4++)
            {
              fprintf(fp, "    y += regCoefs[%d] * x2[%d] * ",ind,nn);
              fprintf(fp, "x2[%d] * x2[%d] * x2[%d];\n",nn2,nn3,nn4);
              fprintf(fp, "    x2[%d]=x2[%d]*x2[%d]*x2[%d]*x2[%d];\n",
                      ind,nn,nn2,nn3,nn4);
              ind++;
            }
          }
        }
      }
    }
    fprintf(fp,"    Y[ii] = y * %e + %e;\n",YStd_, YMean_);
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for (jj = 0; jj < N; jj++) {\n");
    fprintf(fp,"      dtmp = 0.0;\n");
    fprintf(fp,"      for (kk = 0; kk < N; kk++)\n");
    fprintf(fp,"        dtmp += invCovMat[jj][kk] * x2[kk];\n");
    fprintf(fp,"      std += dtmp * x2[jj];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    std = sqrt(std);\n");
    fprintf(fp,"    S[ii] = std;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  free(x2);\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n\n");
    fprintf(fp,"/* ==============================================*/\n");
    printf("FILE psuade_rs.info contains the final polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }

  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# Regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"regCoefs = [\n");
    for (mm = 0; mm < N-1; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp," %24.16e ]\n", VecRegCoeffs_[N-1]);
    fprintf(fp,"invCovMat = [\n");
    for (mm = 0; mm < N; mm++)
    {
      fprintf(fp," [ %24.16e", invCovMat_.getEntry(mm,0));
      for (nn = 1; nn < N; nn++)
        fprintf(fp,", %24.16e", invCovMat_.getEntry(mm,nn));
      fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# Regression interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(X): \n");
    fprintf(fp,"  nSamp = int(len(X) / %d)\n",nInputs_);
    fprintf(fp,"  Xt = %d * [0.0]\n", nInputs_);
    fprintf(fp,"  X2 = %d * [0.0]\n", N);
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  for ss in range(nSamp) : \n");
    fprintf(fp,"    for ii in range(%d) : \n", nInputs_);
    fprintf(fp,"      Xt[ii] = X[ss*%d+ii]\n",nInputs_);
    fprintf(fp,"    X2[0] = 1.0;\n");
    for (nn = 0; nn < nInputs_; nn++)
    {
      if (VecXMeans_[nn] != 0.0 || VecXStds_[nn] != 1.0)
           fprintf(fp,"    X2[%d] = (Xt[%d] - %e) / %e;\n", 
                   nn+1, nn, VecXMeans_[nn], VecXStds_[nn]);
      else fprintf(fp,"    X2[%d] = Xt[%d];\n", nn+1, nn);
    }
    fprintf(fp,"    Y = regCoefs[0]\n");
    if (order_ >= 1)
    {
      for (nn = 1; nn <= nInputs_; nn++)
        fprintf(fp,"    Y = Y + regCoefs[%d] * X2[%d];\n", nn, nn);
    }
    if (order_ >= 2)
    {
      ind = nInputs_ + 1;
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <=nInputs_; nn2++)
        {
          fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]\n",ind,nn,nn2);
          fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]\n",ind,nn,nn2);
          ind++;
        }
      }
    }
    if (order_ >= 3)
    {
      for (nn = 1; nn <=nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <=nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]*X2[%d]\n",
                    ind,nn,nn2,nn3);
            fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]*X2[%d]\n",ind,nn,nn2,nn3);
            ind++;
          }
        }
      }
    }
    if (order_ >= 4)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            for (nn4 = nn3; nn4 <= nInputs_; nn4++)
            {
              fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]",ind,nn,nn2);
              fprintf(fp, "*X2[%d]*X2[%d]\n", nn3, nn4);
              fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]*X2[%d]*X2[%d]\n",
                      ind,nn,nn2,nn3,nn4);
              ind++;
            }
          }
        }
      }
    }
    fprintf(fp,"    Ys[ss*2] = Y * %e + %e\n",YStd_, YMean_);
    fprintf(fp,"    std = 0.0\n");
    fprintf(fp,"    for jj in range(%d): \n", N);
    fprintf(fp,"      dtmp = 0.0\n");
    fprintf(fp,"      for kk in range(%d): \n", N);
    fprintf(fp,"        dtmp = dtmp + invCovMat[jj][kk] * X2[kk]\n");
    fprintf(fp,"      std = std + dtmp * X2[jj]\n");
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
    printf("FILE psuade_rs.py contains the final polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int Regression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int      M, N=0, mm, nn, nn2, nn3, nn4, ind, N2;
  psVector VecXX;

  if (order_ > 4) return 0;

  M = nSamples_;
  if (order_ >= 0) N = 1;
  if (order_ >= 1)
  {
    N2 = N;
    N += nInputs_;
    if (nSamples_ < N)
    {
      N = N2;
      order_ = 0;
      printf("Regression INFO: order reduced to 0.\n");
    }
  }
  if (order_ >= 2)
  {
    N2 = N;
    N += nInputs_ * (nInputs_ + 1) / 2;
    if (nSamples_ < N)
    {
       N = N2;
       order_ = 1;
       printf("Regression INFO: order reduced to 1.\n");
    }
  }
  if (order_ >= 3)
  {
    N2 = N;
    for (nn = 0; nn < nInputs_; nn++)
      for (nn2 = nn; nn2 < nInputs_; nn2++)
        for (nn3 = nn2; nn3 < nInputs_; nn3++) N++;
    if (nSamples_ < N)
    {
      N = N2;
      order_ = 2;
      printf("Regression INFO: order reduced to 2.\n");
    }
  }
  if (order_ >= 4)
  {
    N2 = N;
    for (nn = 0; nn < nInputs_; nn++)
      for (nn2 = nn; nn2 < nInputs_; nn2++)
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
          for (nn4 = nn3; nn4 < nInputs_; nn4++) N++;
    if (nSamples_ < N)
    {
      N = N2;
      order_ = 3;
      printf("Regression INFO: order reduced to 3.\n");
    }
  }
  if (N > M) return N;
  VecXX.setLength(M*N);
  if (order_ >= 0)
  {
    for (mm = 0; mm < M; mm++) VecXX[mm] = 1.0;
  }
  if (order_ >= 1)
  {
    for (mm = 0; mm < M; mm++)
    {
      VecXX[mm] = 1.0;
      for (nn = 0; nn < nInputs_; nn++)
        VecXX[M*(nn+1)+mm] = VecX[mm*nInputs_+nn];
    }
  }
  if (order_ >= 2)
  {
    ind = nInputs_ + 1;
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        for (mm = 0; mm < M; mm++)
          VecXX[M*ind+mm] = VecX[mm*nInputs_+nn] * VecX[mm*nInputs_+nn2];
        ind++;
      }
    }
  }
  if (order_ >= 3)
  {
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (mm = 0; mm < M; mm++)
            VecXX[M*ind+mm] = VecX[mm*nInputs_+nn] * VecX[mm*nInputs_+nn2] * 
                              VecX[mm*nInputs_+nn3];
          ind++;
        }
      }
    }
  }
  if (order_ >= 4)
  {
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
             for (mm = 0; mm < M; mm++)
                VecXX[M*ind+mm] = VecX[mm*nInputs_+nn]*VecX[mm*nInputs_+nn2]* 
                               VecX[mm*nInputs_+nn3] * VecX[mm*nInputs_+nn4];
             ind++;
          }
        }
      }
    }
  }
  MatXX.setFormat(PS_MAT1D);
  MatXX.load(M, N, VecXX.getDVector()); 
  return N;
}

// *************************************************************************
// compute SS (sum of squares) statistics
// -------------------------------------------------------------------------
int Regression::computeSS(psMatrix MatXX, psVector VecY,
                          psVector VecB, double &SSresid, double &SStotal)
{
  int    nn, mm, N;
  double rdata, ymean, SSreg, ddata, SSresidCheck, *arrayXX;

  N = VecB.length();
  arrayXX = MatXX.getMatrix1D();
  SSresid = SSresidCheck = SStotal = SSreg = ymean = 0.0;
  for (mm = 0; mm < nSamples_; mm++) ymean += sqrt(VecWghts_[mm]) * VecY[mm];
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) ddata += (arrayXX[mm+nn*nSamples_]*VecB[nn]);
    rdata = VecY[mm] - ddata;
    SSresidCheck += rdata * rdata * VecWghts_[mm];
    SSresid += rdata * VecY[mm] * VecWghts_[mm];
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += VecWghts_[mm] * (VecY[mm] - ymean) * (VecY[mm] - ymean);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0)
  {
    printf("* Regression: SStot  = %24.16e\n", SStotal);
    printf("* Regression: SSreg  = %24.16e\n", SSreg);
    printf("* Regression: SSres  = %24.16e\n", SSresid);
    printf("* Regression: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0 && nSamples_ != N)
  {
    printf("* Regression: eps(Y) = %24.16e\n",
           SSresidCheck/(nSamples_-N));
  }

  //**/ ****************************************************
  //**/ Old version based on the following formulas (but works)
  //**/ SSred = (Y - Xb)' W (Y - Xb) 
  //**/       = Y' W Y - 2 b 'X W 'Y + b' X' W X b
  //**/       = Y' W Y - 2 b' X' W Y + b' X' W Y  (since X'WXb=X'WY) 
  //**/       = Y' W Y - b' X' W Y = (Y - Xb)' W Y
  //**/ SStot = Y' W Y - N * (mean (W^(1/2) Y))^2
  //**/ ===================================================
  //**/SSresid = SStotal = ymean = 0.0;
  //**/R = new double[nSamples_];
  //**/for (mm = 0; mm < nSamples_; mm++)
  //**/{
  //**/   R[mm] = Y[mm];
  //**/   for (nn = 0; nn < N; nn++) R[mm] -= (XX[mm+nn*nSamples_] * B[nn]);
  //**/   SSresid += R[mm] * Y[mm] * VecWghts_[mm];
  //**/   ymean += (sqrt(VecWghts_[mm]) * Y[mm]);
  //**/}
  //**/ymean /= (double) nSamples_;
  //**/SStotal = - ymean * ymean * (double) nSamples_;
  //**/for (mm = 0; mm < nSamples_; mm++)
  //**/   SStotal += VecWghts_[mm] * Y[mm] * Y[mm];
  //**/ ****************************************************
  return 0;
}

// *************************************************************************
// compute coefficient variances (diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int Regression::computeCoeffVariance(psMatrix &eigMatT, psVector &eigVals,
                                     double var)
{
  int      ii, jj, nRows;
  double   invEig, dtmp;
  psMatrix tMat;

  nRows = eigMatT.nrows();
  tMat.setDim(nRows, nRows);

  //**/ =================================================================
  //**/ compute sigma^2 * V * D^{-1} 
  //**/ =================================================================
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
  //**/ =================================================================
  //**/ compute (sigma^2 * V * D^{-1}) V^T 
  //**/ =================================================================
  tMat.matmult(eigMatT, invCovMat_);
  return 0;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int Regression::printRC(psVector VecB, psVector VecBstd, psMatrix MatXX,
                        psVector VecY)
{
  int    nn1, ii, ind, nn2, nn3, nn4, N;
  double coef, Bmax, *arrayXX;
  char   fname[200];
  FILE   *fp;

  if (order_ < 0 || order_ > 4) return 0;
  printEquals(PL_INFO, 0);
  printf("*** NOTE: These coefficients may not be true coefficients due\n");
  printf("***       to sample matrix scaling (i.e., they may be scaled).\n");
  printf("***       To turn off scaling, turn on 'master' mode.\n");
  if (psConfig_.MasterModeIsOn())
    printf("***       NOTE: master mode is ON (==> no scaling).\n");
  else
    printf("***       NOTE: master mode is OFF (==> scaling is ON)\n");
  printDashes(PL_INFO, 0);
  printf("*            ");
  for (ii = 1; ii < order_; ii++) printf("    ");
  printf("  coefficient   std. error   t-value\n");
  printDashes(PL_INFO, 0);
  //**/ =================================================================
  //**/ ------ print out the scaled linear coefficients -----------
  //**/ =================================================================
  arrayXX = MatXX.getMatrix1D();
  N = VecB.length();
  if (PABS(VecBstd[0]) < 1.0e-15) coef = 0.0;
  else                            coef = VecB[0] / VecBstd[0]; 
  printf("* Constant  ");
  for (ii = 1; ii < order_; ii++) printf("    ");
  printf("= %12.4e %12.4e %12.4e\n", VecB[0], VecBstd[0], coef);
  if (order_ >= 1)
  {
    for (nn1 = 1; nn1 <= nInputs_; nn1++)
    {
      if (PABS(VecBstd[nn1]) < 1.0e-15) coef = 0.0;
      else                              coef = VecB[nn1] / VecBstd[nn1]; 
      //**/ if (PABS(coef) > 1.0)
      {
        printf("* Input %3d ", nn1);
        for (ii = 1; ii < order_; ii++) printf("    ");
        printf("= %12.4e %12.4e %12.4e\n",VecB[nn1],VecBstd[nn1], coef);
      }
      strcpy(fname, "dataVariance1");
    }
  }
  if (order_ >= 2)
  {
    //**/ ===============================================================
    //**/ --------- print out the quadratic coefficients --------------
    //**/ ===============================================================
    Bmax = 0.0;
    for (nn1 = nInputs_+1; nn1 < N; nn1++)
      if (PABS(VecB[nn1]) > Bmax) Bmax = PABS(VecB[nn1]); 
    ind = nInputs_ + 1;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        if (PABS(VecB[ind]) > 1.0e-6 * Bmax) 
        {
          if (PABS(VecBstd[ind]) < 1.0e-15) coef = 0.0;
          else coef = VecB[ind] / VecBstd[ind]; 
          //**/ if (PABS(coef) > 1.0)
          {
            printf("* Input %3d %3d ", nn1+1, nn2+1);
            for (ii = 2; ii < order_; ii++) printf("    ");
            printf("= %12.4e %12.4e %12.4e\n",VecB[ind],VecBstd[ind],coef); 
          }
        }
        ind++;
      }
    }
    strcpy(fname, "dataVariance2");
  }
  if (order_ >= 3)
  {
    //**/ ===============================================================
    //**/ ---------------- print the cubic coefficients -----------------
    //**/ ===============================================================
    Bmax = 0.0;
    for (nn1 = ind; nn1 < N; nn1++)
      if (PABS(VecB[nn1]) > Bmax) Bmax = PABS(VecB[nn1]); 
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          if (PABS(VecB[ind]) > 1.0e-6 * Bmax) 
          {
            if (PABS(VecBstd[ind]) < 1.0e-15) coef = 0.0;
            else coef = VecB[ind] / VecBstd[ind]; 
            //**/ if (PABS(coef) > 1.0)
            {
              printf("* Input %3d %3d %3d ", nn1+1, nn2+1, nn3+1);
              for (ii = 3; ii < order_; ii++) printf("    ");
              printf("= %12.4e %12.4e %12.4e\n",VecB[ind],VecBstd[ind],
                     coef);
            }
          }
          ind++;
        }
      }
    }
    strcpy(fname, "dataVariance3");
  }
  if (order_ >= 4)
  {
    //**/ ===============================================================
    //**/ ---------- print the quartic coefficients -------------------
    //**/ ===============================================================
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            if (PABS(VecB[ind]) > 1.0e-6 * Bmax) 
            {
              if (PABS(VecBstd[ind]) < 1.0e-15) coef = 0.0;
              else coef = VecB[ind] / VecBstd[ind]; 
              //**/ if (PABS(coef) > 1.0)
              {
                printf("* Input %3d %3d %3d %3d ",nn1+1,nn2+1,nn3+1,
                       nn4+1);
                for (ii = 4; ii < order_; ii++) printf("    ");
                printf("= %12.4e %12.4e %12.4e\n",VecB[ind],VecBstd[ind],
                       coef); 
              }
            }
            ind++;
          }
        }
      }
    }
    strcpy(fname, "dataVariance4");
  }
  printDashes(PL_INFO, 0);

  //**/ =================================================================
  //**/ print to file
  //**/ =================================================================
  if (order_ >= 0 && order_ <= 4 && psConfig_.MasterModeIsOn())
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
      for (nn1 = 0; nn1 < N; nn1++) 
        coef += PABS(arrayXX[ii+nn1*nSamples_]*VecBstd[nn1]);
      fprintf(fp,"%7d         %12.4e %12.4e\n",ii+1,VecY[ii],sqrt(coef));
    }
    fclose(fp);
    printf("FILE %s contains output data standard errors.\n",fname);
  }
  return 0;
}

// *************************************************************************
// print standardized regression coefficients
// -------------------------------------------------------------------------
int Regression::printSRC(psVector VecX, psVector VecB, double SStotal)
{
  int    nn, mm, ind, ii, nn2, itemp, nn3, nn4;
  double denom, xmean, coef, Bmax, coef1, coef2, xmean1, xmean2;
  double xmean3, coef3, xmean4, coef4;
  psVector  VecB2;
  psIVector VecIndices;

  if (order_ < 0 || order_ > 4) return 0;

  printAsterisks(PL_INFO, 0);
  printf("*       Standardized Regression Coefficients (SRC)\n");
  printf("* When R-square is acceptable (order assumption holds), the\n");
  printf("* absolute values of SRCs provide variable importance.\n"); 
  printEquals(PL_INFO, 0);
  printf("* based on nSamples = %d\n", nSamples_);

  VecB2.setLength(nSamples_);
  if (order_ >= 1)
  {
    VecIndices.setLength(nInputs_);
    denom = sqrt(SStotal / (double) (nSamples_ - 1));
    Bmax = 0.0;
    for (nn = 0; nn < nInputs_; nn++)
    {
      xmean = 0.0;
      for (mm = 0; mm < nSamples_; mm++) 
        xmean += VecX[mm*nInputs_+nn] * sqrt(VecWghts_[mm]);
      xmean /= (double) nSamples_;
      coef = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
      {
        coef += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean) * 
                (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean);
      }
      coef = sqrt(coef / (double) (nSamples_ - 1)) / denom;
      printf("* Input %3d ", nn+1);
      for (ii = 1; ii < order_; ii++) printf("    ");
      if ((VecB[nn+1]*coef) > Bmax) Bmax = VecB[nn+1]*coef;
      printf("= %12.4e\n", VecB[nn+1]*coef);
      VecB2[nn] = PABS(VecB[nn+1] * coef);
    }
    printDashes(PL_INFO, 0);
    printf("*    ordered list of SRCs\n");
    printDashes(PL_INFO, 0);
    for (nn = 0; nn < nInputs_; nn++) VecIndices[nn] = nn;
    sortDbleList2a(nInputs_,VecB2.getDVector(),VecIndices.getIVector()); 
    for (nn = nInputs_-1; nn >= 0; nn--)
    {
      printf("* Input %3d ", VecIndices[nn]+1);
      for (ii = 1; ii < order_; ii++) printf("    ");
      printf("= %12.4e\n", VecB2[nn]);
    }
  }
  if (order_ >= 2)
  {
    ind = nInputs_ + 1;
    for (nn = 0; nn < nInputs_; nn++)
    {
      xmean1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++) 
        xmean1 += VecX[mm*nInputs_+nn] * sqrt(VecWghts_[mm]);
      xmean1 /= (double) nSamples_;
      coef1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        coef1 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1) * 
                 (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1);
      coef1 = sqrt(coef1 / (double) (nSamples_ - 1));
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        xmean2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          xmean2 += VecX[mm*nInputs_+nn2] * sqrt(VecWghts_[mm]);
        xmean2 /= (double) nSamples_;
        coef2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          coef2 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2) * 
                   (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2);
        coef2 = sqrt(coef2 / (double) (nSamples_ - 1));
        VecB2[ind] = VecB[ind] * coef1 * coef2 / denom;
        if (PABS(VecB2[ind]) > Bmax) Bmax = PABS(VecB2[ind]);
        ind++;
      }
    }
    ind = nInputs_ + 1;
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        if (PABS(VecB2[ind]) > 1.0e-3 * Bmax)
        {
          printf("* Input %3d,%3d ", nn+1, nn2+1);
          for (ii = 2; ii < order_; ii++) printf("    ");
          printf("= %12.4e\n", VecB2[ind]);
        }
        ind++;
      }
    }
  }
  if (order_ >= 3)
  {
    itemp = ind;
    for (nn = 0; nn < nInputs_; nn++)
    {
      xmean1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        xmean1 += VecX[mm*nInputs_+nn] * sqrt(VecWghts_[mm]);
      xmean1 /= (double) nSamples_;
      coef1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        coef1 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1) * 
                 (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1);
      coef1 = sqrt(coef1 / (double) (nSamples_ - 1));
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        xmean2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          xmean2 += VecX[mm*nInputs_+nn2] * sqrt(VecWghts_[mm]);
        xmean2 /= (double) nSamples_;
        coef2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          coef2 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2) * 
                   (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2);
        coef2 = sqrt(coef2 / (double) (nSamples_ - 1));
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          xmean3 = 0.0;
          for (mm = 0; mm < nSamples_; mm++)
            xmean3 += VecX[mm*nInputs_+nn3] * sqrt(VecWghts_[mm]);
          xmean3 /= (double) nSamples_;
          coef3 = 0.0;
          for (mm = 0; mm < nSamples_; mm++)
            coef3 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn3] - xmean3) * 
                     (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn3] - xmean3);
          coef3 = sqrt(coef3 / (double) (nSamples_ - 1));
          VecB2[ind] = VecB[ind] * coef1 * coef2 *coef3 / denom;
          if (PABS(VecB2[ind]) > Bmax) Bmax = PABS(VecB2[ind]);
          ind++;
        }
      }
    }
    ind = itemp;
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          if (PABS(VecB2[ind]) > 1.0e-3 * Bmax)
          {
            printf("* Input %3d,%3d,%3d ", nn+1, nn2+1, nn3+1);
            for (ii = 3; ii < order_; ii++) printf("    ");
            printf("= %12.4e\n", VecB2[ind]);
          }
          ind++;
        }
      }
    }
  }
  if (order_ >= 4)
  {
    itemp = ind;
    for (nn = 0; nn < nInputs_; nn++)
    {
      xmean1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        xmean1 += VecX[mm*nInputs_+nn] * sqrt(VecWghts_[mm]);
      xmean1 /= (double) nSamples_;
      coef1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        coef1 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1) * 
                 (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn] - xmean1);
      coef1 = sqrt(coef1 / (double) (nSamples_ - 1));
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        xmean2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          xmean2 += VecX[mm*nInputs_+nn2] * sqrt(VecWghts_[mm]);
        xmean2 /= (double) nSamples_;
        coef2 = 0.0;
        for (mm = 0; mm < nSamples_; mm++)
          coef2 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2) * 
                   (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn2] - xmean2);
        coef2 = sqrt(coef2 / (double) (nSamples_ - 1));
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          xmean3 = 0.0;
          for (mm = 0; mm < nSamples_; mm++)
            xmean3 += VecX[mm*nInputs_+nn3] * sqrt(VecWghts_[mm]);
          xmean3 /= (double) nSamples_;
          coef3 = 0.0;
          for (mm = 0; mm < nSamples_; mm++)
            coef3 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn3] - xmean3) * 
                     (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn3] - xmean3);
          coef3 = sqrt(coef3 / (double) (nSamples_ - 1));
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            xmean4 = 0.0;
            for (mm = 0; mm < nSamples_; mm++)
              xmean3 += VecX[mm*nInputs_+nn4] * sqrt(VecWghts_[mm]);
            xmean4 /= (double) nSamples_;
            coef4 = 0.0;
            for (mm = 0; mm < nSamples_; mm++)
              coef4 += (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn4] - xmean4) * 
                       (sqrt(VecWghts_[mm])*VecX[mm*nInputs_+nn4] - xmean4);
            coef4 = sqrt(coef4 / (double) (nSamples_ - 1));
            VecB2[ind] = VecB[ind] * coef1 * coef2 *coef3 *coef4 / denom;
            if (PABS(VecB2[ind]) > Bmax) Bmax = PABS(VecB2[ind]);
            ind++;
          }
        }
      }
    }
    ind = itemp;
    for (nn = 0; nn < nInputs_; nn++)
    {
      for (nn2 = nn; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            if (PABS(VecB2[ind]) > 1.0e-3 * Bmax)
            {
              printf("* Input %3d,%3d,%3d,%3d ",nn+1,nn2+1,nn3+1,nn4+1);
              for (ii = 4; ii < order_; ii++) printf("    ");
              printf("= %12.4e\n",VecB2[ind]);
            }
            ind++;
          }
        }
      }
    }
  }
  printAsterisks(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// print coefficients 
// -------------------------------------------------------------------------
int Regression::printCoefs(psVector VecB)
{
  int    ii, jj, N, nn1, nn2, nn3, nn4, **indTable, ptr2, ptr3, ptr4, cnt;
  int    mm1, mm2, kk, *indices;
  double Bmax, ddata;
  psIMatrix VecIndTable;

  //**/ =================================================================
  //**/ create index table
  //**/ =================================================================
  N = VecB.length();
  VecIndTable.setFormat(PS_MAT2D);
  VecIndTable.setDim(N, nInputs_);
  indTable = VecIndTable.getIMatrix2D();
  for (ii = 0; ii < nInputs_; ii++) 
  {
    for (jj = 0; jj < nInputs_; jj++) indTable[ii+1][jj] = 0; 
    indTable[ii+1][ii] = 1; 
  }
  ptr2 = ptr3 = nInputs_ + 1;
  if (order_ >= 2)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (jj = 0; jj < nInputs_; jj++) indTable[ptr3][jj] = 0; 
        indTable[ptr3][nn1]++; 
        indTable[ptr3][nn2]++; 
        ptr3++;
      }
    }
  }
  ptr4 = ptr3;
  if (order_ >= 3)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (jj = 0; jj < nInputs_; jj++) indTable[ptr4][jj] = 0; 
          indTable[ptr4][nn1]++; 
          indTable[ptr4][nn2]++; 
          indTable[ptr4][nn3]++; 
          ptr4++;
        }
      }
    }
  }
  cnt = ptr4;
  if (order_ >= 4)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            for (jj = 0; jj < nInputs_; jj++) indTable[cnt][jj] = 0; 
            indTable[cnt][nn1]++; 
            indTable[cnt][nn2]++; 
            indTable[cnt][nn3]++; 
            indTable[cnt][nn4]++; 
            cnt++;
          }
        }
      }
    }
  }
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < nInputs_; jj++)
      printf("%d ", indTable[ii][jj]);
    printf("\n");
  }
    
  printEquals(PL_INFO, 0);
  printf("*** Note: these coefficients are true coefficients.\n");
  printDashes(PL_INFO, 0);
  printf("*            ");
  for (ii = 1; ii < order_; ii++) printf("    ");
  printf("  coefficient\n");
  printDashes(PL_INFO, 0);

  //**/ =================================================================
  //**/ --------- print out the linear coefficients --------------
  //**/ =================================================================
  psVector VecTrueCoefs;
  VecTrueCoefs.setLength(N);
  for (ii = 0; ii < N; ii++) VecTrueCoefs[ii] = 0.0;
  VecTrueCoefs[0] = VecB[0];
  psIVector VecIndices;
  VecIndices.setLength(nInputs_);
  if (order_ >= 1)
  {
    for (nn1 = 1; nn1 <= nInputs_; nn1++)
    {
      VecTrueCoefs[nn1] = VecB[nn1] / VecXStds_[nn1-1];
      VecTrueCoefs[0]  -= VecB[nn1] * VecXMeans_[nn1-1] / VecXStds_[nn1-1];
    }
  }
  if (order_ >= 2)
  {
    cnt = ptr2;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        VecTrueCoefs[cnt] = VecB[cnt] / (VecXStds_[nn1] * VecXStds_[nn2]);
        VecTrueCoefs[nn1+1] -= VecB[cnt]*VecXMeans_[nn2]/
                            (VecXStds_[nn1]*VecXStds_[nn2]);
        VecTrueCoefs[nn2+1] -= VecB[cnt]*VecXMeans_[nn1]/
                            (VecXStds_[nn1]*VecXStds_[nn2]);
        VecTrueCoefs[0] += VecB[cnt]*(VecXMeans_[nn1]*VecXMeans_[nn2])/
                               (VecXStds_[nn1]*VecXStds_[nn2]);
        cnt++;
      }
    }
  }
  if (order_ >= 3)
  {
    cnt = ptr3;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          ddata = VecB[cnt];
          ddata /= VecXStds_[nn1];
          ddata /= VecXStds_[nn2];
          ddata /= VecXStds_[nn3];
          VecTrueCoefs[cnt] = ddata;

          //**/ (0)
          ddata = VecXMeans_[nn1] / VecXStds_[nn1];
          ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
          ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
          VecTrueCoefs[0] -= VecB[cnt] * ddata;

          //**/ (1)
          ddata = 1.0 / VecXStds_[nn1];
          ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
          ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
          VecTrueCoefs[nn1+1] += VecB[cnt] * ddata;

          //**/ (2)
          ddata = 1.0 / VecXStds_[nn2];
          ddata *= VecXMeans_[nn1] / VecXStds_[nn1];
          ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
          VecTrueCoefs[nn2+1] += VecB[cnt] * ddata;

          //**/ (3)
          ddata = 1.0 / VecXStds_[nn3];
          ddata *= VecXMeans_[nn1] / VecXStds_[nn1];
          ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
          VecTrueCoefs[nn3+1] += VecB[cnt] * ddata;

          //**/ doubles
          ii = ptr2;
          for (mm1 = 0; mm1 < nInputs_; mm1++) VecIndices[mm1] = 0;
          VecIndices[nn1]++;
          VecIndices[nn2]++;
          VecIndices[nn3]++;
          
          for (mm1 = 0; mm1 < nInputs_; mm1++)
          {
            for (mm2 = mm1; mm2 < nInputs_; mm2++)
            {
              ddata = 1.0;
              if (VecIndices[nn1] >= indTable[ii][nn1] &&
                  VecIndices[nn2] >= indTable[ii][nn2])
              {
                ddata = 1.0 / VecXStds_[nn1];
                ddata /= VecXStds_[nn2];
                ddata  *= VecXMeans_[nn3] / VecXStds_[nn3];
                VecTrueCoefs[ii] -= VecB[cnt] * ddata;
                printf("(a) ii = %d\n",ii+1);
              }
              else if (VecIndices[nn1] >= indTable[ii][nn1] &&
                       VecIndices[nn3] >= indTable[ii][nn3])
              {
                ddata = 1.0 / VecXStds_[nn1];
                ddata /= VecXStds_[nn3];
                ddata  *= VecXMeans_[nn2] / VecXStds_[nn2];
                VecTrueCoefs[ii] -= VecB[cnt] * ddata;
                printf("(b) ii = %d\n",ii+1);
              }
              else if (VecIndices[nn2] >= indTable[ii][nn2] &&
                       VecIndices[nn3] >= indTable[ii][nn3])
              {
                ddata = 1.0 / VecXStds_[nn2];
                ddata /= VecXStds_[nn3];
                ddata  *= VecXMeans_[nn1] / VecXStds_[nn1];
                VecTrueCoefs[ii] -= VecB[cnt] * ddata;
                printf("(c) ii = %d\n",ii+1);
              }
              ii++;
            }
          }
          cnt++;
        }
      }
    }
  }
#if 0
  if (order_ >= 4)
  {
    cnt = ptr4;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            ddata = VecB[cnt];
            ddata /= VecXStds_[nn1];
            ddata /= VecXStds_[nn2];
            ddata /= VecXStds_[nn3];
            ddata /= VecXStds_[nn4];
            VecTrueCoefs[cnt] = ddata;

            //**/ (0)
            ddata = VecXMeans_[nn1] / VecXStds_[nn1];
            ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
            ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
            ddata *= VecXMeans_[nn4] / VecXStds_[nn4];
            VecTrueCoefs[0] += VecB[cnt] * ddata;

            //**/ (1)
            if (indTable[cnt][0] = 2)
            {
              ddata = 1.0 / VecXStds_[nn1];
              ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
              ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
              ddata *= VecXMeans_[nn4] / VecXStds_[nn4];
              VecTrueCoefs[nn1+1] -= VecB[cnt] * ddata;
            }

            //**/ (2)
            if (indTable[cnt][1] = 1)
            {
              ddata = 1.0 / VecXStds_[nn2];
              ddata *= VecXMeans_[nn1] / VecXStds_[nn1];
              ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
              ddata *= VecXMeans_[nn4] / VecXStds_[nn4];
              VecTrueCoefs[nn2+1] -= VecB[cnt] * ddata;
            }

            //**/ (3)
            if (indTable[cnt][2] == 1)
            {
              ddata = 1.0 / VecXStds_[nn3];
              ddata *= VecXMeans_[nn1] / VecXStds_[nn1];
              ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
              ddata *= VecXMeans_[nn4] / VecXStds_[nn4];
              VecTrueCoefs[nn3+1] -= VecB[cnt] * ddata;
            }

            //**/ (4)
            if (indTable[cnt][3] == 1)
            {
              ddata = 1.0 / VecXStds_[nn4];
              ddata *= VecXMeans_[nn1] / VecXStds_[nn1];
              ddata *= VecXMeans_[nn2] / VecXStds_[nn2];
              ddata *= VecXMeans_[nn3] / VecXStds_[nn3];
              VecTrueCoefs[nn4+1] -= VecB[cnt] * ddata;
            }

            //**/ (1,2)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn1 != nn2) && (indTable[ii][nn1] >= 1 && 
                indTable[ii][nn2] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn3] / VecXStds_[nn3];
              ddata  *= VecXMeans_[nn4] / VecXStds_[nn4];
              ddata  /= VecXStds_[nn1];
              ddata  /= VecXStds_[nn2];
              VecTrueCoefs[ii] += B[cnt] * ddata;
            }

            //**/ (1,3)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn1 != nn3) && (indTable[ii][nn1] == 1 && 
                indTable[ii][nn3] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn2] / VecXStds_[nn2];
              ddata  *= VecXMeans_[nn4] / VecXStds_[nn4];
              ddata  /= VecXStds_[nn1];
              ddata  /= VecXStds_[nn3];
              VecTrueCoefs[ii] += VecB[cnt] * ddata;
            }

            //**/ (1,4)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn1 != nn4) && (indTable[ii][nn1] >= 1 && 
                indTable[ii][nn4] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn2] / VecXStds_[nn2];
              ddata  *= VecXMeans_[nn3] / VecXStds_[nn3];
              ddata  /= VecXStds_[nn1];
              ddata  /= VecXStds_[nn4];
              VecTrueCoefs[ii] += VecB[cnt] * ddata;
            }

            //**/ (2,3)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn2 != nn3) && (indTable[ii][nn2] == 1 && 
                indTable[ii][nn3] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn1] / VecXStds_[nn1];
              ddata  *= VecXMeans_[nn4] / VecXStds_[nn4];
              ddata  /= VecXStds_[nn2];
              ddata  /= VecXStds_[nn3];
              VecTrueCoefs[ii] += VecB[cnt] * ddata;
            }

            //**/ (2,4)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn2 != nn4) && (indTable[ii][nn2] == 1 && 
                indTable[ii][nn4] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn1] / VecXStds_[nn1];
              ddata  *= VecXMeans_[nn3] / VecXStds_[nn3];
              ddata  /= VecXStds_[nn2];
              ddata  /= VecXStds_[nn4];
              VecTrueCoefs[ii] += VecB[cnt] * ddata;
            }

            //**/ (3,4)
            ii = ptr2;
            while (ii < ptr3)
            {
              if ((nn3 != nn4) && (indTable[ii][nn3] == 1 && 
                indTable[ii][nn4] == 1))
              break;
              ii++;
            }
            if (ii < ptr3)
            {
              ddata  = VecXMeans_[nn1] / VecXStds_[nn1];
              ddata  *= VecXMeans_[nn2] / VecXStds_[nn2];
              ddata  /= VecXStds_[nn3];
              ddata  /= VecXStds_[nn4];
              VecTrueCoefs[ii] += VecB[cnt] * ddata;
            }

            //**/ (1,2,3)
            ii = ptr3;
            while (ii < N)
            {
              if (indTable[ii][nn1] >= 1 && 
                  indTable[ii][nn2] >= 1 && 
                  indTable[ii][nn3] >= 1) 
                break;
              ii++;
            }
            ddata  = VecXMeans_[nn4] / VecXStds_[nn4];
            ddata  /= VecXStds_[nn1];
            ddata  /= VecXStds_[nn2];
            ddata  /= VecXStds_[nn3];
            VecTrueCoefs[ii] -= VecB[cnt] * ddata;

            //**/ (1,2,4)
            ii = ptr3;
            while (ii < N)
            {
              if (indTable[ii][nn1] >= 1 && 
                  indTable[ii][nn2] >= 1 && 
                  indTable[ii][nn4] >= 1) 
                break;
              ii++;
            }
            ddata  = VecXMeans_[nn3] / VecXStds_[nn3];
            ddata  /= VecXStds_[nn1];
            ddata  /= VecXStds_[nn2];
            ddata  /= VecXStds_[nn4];
            VecTrueCoefs[ii] -= VecB[cnt] * ddata;

            //**/ (1,3,4)
            ii = ptr3;
            while (ii < N)
            {
              if (indTable[ii][nn1] >= 1 && 
                  indTable[ii][nn3] >= 1 && 
                  indTable[ii][nn4] >= 1) 
                break;
              ii++;
            }
            ddata  = VecXMeans_[nn2] / VecXStds_[nn2];
            ddata  /= VecXStds_[nn1];
            ddata  /= VecXStds_[nn3];
            ddata  /= VecXStds_[nn4];
            VecTrueCoefs[ii] -= VecB[cnt] * ddata;

            //**/ (2,3,4)
            ii = ptr3;
            while (ii < N)
            {
              if (indTable[ii][nn2] >= 1 && 
                  indTable[ii][nn3] >= 1 && 
                  indTable[ii][nn4] >= 1) 
                break;
              ii++;
            }
            ddata  = VecXMeans_[nn1] / VecXStds_[nn1];
            ddata  /= VecXStds_[nn2];
            ddata  /= VecXStds_[nn3];
            ddata  /= VecXStds_[nn4];
            VecTrueCoefs[ii] -= VecB[cnt] * ddata;
            cnt++;
          }
        }
      }
    }
  }
#endif
  //**/ =================================================================
  //**/ zero out the very small entries
  //**/ =================================================================
  Bmax = VecTrueCoefs[0];
  for (nn1 = 1; nn1 < N; nn1++)
    if (PABS(VecTrueCoefs[nn1]) > Bmax) 
      Bmax = PABS(VecTrueCoefs[nn1]);
  if (Bmax == 0) Bmax = 1;
  for (nn1 = 0; nn1 < N; nn1++)
    if (PABS(VecTrueCoefs[nn1]/Bmax) < 1.0e-8)
      VecTrueCoefs[nn1] = 0;

  printf("* Constant  ");
  for (ii = 1; ii < order_; ii++) printf("    ");
  printf("= %16.8e \n", VecTrueCoefs[0]);
  if (order_ >= 1)
  {
    for (nn1 = 1; nn1 <= nInputs_; nn1++)
    {
      if (VecTrueCoefs[nn1] != 0)
      {
        printf("* Input %3d ", nn1);
        for (ii = 1; ii < order_; ii++) printf("    ");
        printf("= %16.8e \n", VecTrueCoefs[nn1]);
      }
    }
  }
  if (order_ >= 2)
  {
    cnt = nInputs_ + 1;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        if (VecTrueCoefs[cnt] != 0)
        {
          printf("* Input %3d %3d ", nn1+1, nn2+1);
          for (ii = 2; ii < order_; ii++) printf("    ");
            printf("= %16.8e \n",VecTrueCoefs[cnt]);
        }
        cnt++;
      }
    }
  }
  if (order_ >= 3)
  {
    cnt = ptr3;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          if (VecTrueCoefs[cnt] != 0)
          {
            printf("* Input %3d %3d %3d ", nn1+1, nn2+1, nn3+1);
            for (ii = 3; ii < order_; ii++) printf("    ");
              printf("= %16.8e \n",VecTrueCoefs[cnt]);
          }
          cnt++;
        }
      }
    }
  }
  if (order_ >= 4)
  {
    cnt = ptr4;
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = nn1; nn2 < nInputs_; nn2++)
      {
        for (nn3 = nn2; nn3 < nInputs_; nn3++)
        {
          for (nn4 = nn3; nn4 < nInputs_; nn4++)
          {
            if (VecTrueCoefs[cnt] != 0)
            {
              printf("* Input %3d %3d %3d %3d ",nn1+1,nn2+1,
                     nn3+1, nn4+1);
              for (ii = 4; ii < order_; ii++) printf("    ");
                printf("= %16.8e \n",VecTrueCoefs[cnt]);
            }
            cnt++;
          }
        }
      }
    }
  }
  printDashes(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// generate C and Python codes
// -------------------------------------------------------------------------
void Regression::genRSCode()
{
  int  nn, nn2, nn3, nn4, mm, ind, N;
  FILE *fp = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/* Regression interpolator from PSUADE.*/\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   3. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
    fprintf(fp,"main(int argc, char **argv) {\n");
    fprintf(fp,"  int    i, iOne=1, nInps;\n");
    fprintf(fp,"  double X[%d], Y, S;\n",nInputs_);
    fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
    fprintf(fp,"  if (argc < 3) {\n");
    fprintf(fp,"    printf(\"ERROR: not enough argument.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
    fprintf(fp,"  if (fIn == NULL) {\n");
    fprintf(fp,"    printf(\"ERROR: cannot open input file.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
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
    fprintf(fp,"}\n\n");
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/*  Regression interpolation function  */\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"static double\n");
    N = VecRegCoeffs_.length();
    fprintf(fp,"regCoefs[%d] = \n", N);
    fprintf(fp,"{\n");
    for (mm = 0; mm < N; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp,"};\n");
    fprintf(fp,"static double invCovMat[%d][%d] = \n", N, N);
    fprintf(fp,"{\n");
    for (mm = 0; mm < N; mm++)
    {
      fprintf(fp," { %24.16e", invCovMat_.getEntry(mm,0));
      for (nn = 1; nn < N; nn++)
        fprintf(fp,", %24.16e", invCovMat_.getEntry(mm,nn));
      fprintf(fp," },\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static int N=%d;\n",N);
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
    fprintf(fp,"  int    ii, jj, kk, nInps=%d;\n",nInputs_);
    fprintf(fp,"  double y, *x, *x2, std, dtmp;\n");
    fprintf(fp,"  x2 = (double *) malloc(%d * sizeof(double));\n",
            N);
    fprintf(fp,"  for (ii = 0; ii < npts; ii++) {\n");
    fprintf(fp,"    x = &X[ii * %d];\n", nInputs_);
    fprintf(fp,"    y = regCoefs[0];\n");
    fprintf(fp,"    x2[0] = 1.0;\n");
    for (nn = 0; nn < nInputs_; nn++)
    {
      if (VecXMeans_[nn] != 0.0 || VecXStds_[nn] != 1.0)
        fprintf(fp,"    x2[%d] = (x[%d] - %e) / %e;\n", nn+1, nn,
                VecXMeans_[nn], VecXStds_[nn]);
      else
        fprintf(fp,"    x2[%d] = x[%d];\n", nn+1, nn);
    }
    for (nn = 1; nn <= nInputs_; nn++)
    {
      fprintf(fp,"    y += regCoefs[%d] * x2[%d];\n", nn, nn);
    }
    if (order_ >= 2)
    {
      ind = nInputs_ + 1;
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          fprintf(fp, "    y += regCoefs[%d] * x2[%d] * x2[%d];\n",ind,
                  nn,nn2);
          fprintf(fp, "    x2[%d] = x2[%d] * x2[%d];\n",ind, nn,nn2);
          ind++;
        }
      }
    }
    if (order_ >= 3)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            fprintf(fp, "    y += regCoefs[%d] * x2[%d] * ",ind,nn);
            fprintf(fp, "x2[%d] * x2[%d];\n", nn2, nn3);
            fprintf(fp, "    x2[%d]=x2[%d]*x2[%d]*x2[%d];\n",ind,nn,nn2,nn3);
            ind++;
          }
        }
      }
    }
    if (order_ >= 4)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            for (nn4 = nn3; nn4 <= nInputs_; nn4++)
            {
              fprintf(fp, "    y += regCoefs[%d] * x2[%d] * ",ind,nn);
              fprintf(fp, "x2[%d] * x2[%d] * x2[%d];\n",nn2,nn3,nn4);
              fprintf(fp, "    x2[%d]=x2[%d]*x2[%d]*x2[%d]*x2[%d];\n",
                      ind,nn,nn2,nn3,nn4);
              ind++;
            }
          }
        }
      }
    }
    fprintf(fp,"    Y[ii] = y * %e + %e;\n",YStd_, YMean_);
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for (jj = 0; jj < N; jj++) {\n");
    fprintf(fp,"      dtmp = 0.0;\n");
    fprintf(fp,"      for (kk = 0; kk < N; kk++)\n");
    fprintf(fp,"        dtmp += invCovMat[jj][kk] * x2[kk];\n");
    fprintf(fp,"      std += dtmp * x2[jj];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    std = sqrt(std);\n");
    fprintf(fp,"    S[ii] = std;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  free(x2);\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n\n");
    fprintf(fp,"/* ==============================================*/\n");
    printf("FILE psuade_rs.info contains the final polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
  fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# Regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"regCoefs = [\n");
    for (mm = 0; mm < N-1; mm++)
      fprintf(fp," %24.16e,\n", VecRegCoeffs_[mm]);
    fprintf(fp," %24.16e ]\n", VecRegCoeffs_[N-1]);
    fprintf(fp,"invCovMat = [\n");
    for (mm = 0; mm < N; mm++)
    {
      fprintf(fp," [ %24.16e", invCovMat_.getEntry(mm,0));
      for (nn = 1; nn < N; nn++)
        fprintf(fp,", %24.16e", invCovMat_.getEntry(mm,nn));
      fprintf(fp," ],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# Regression interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(X): \n");
    fprintf(fp,"  nSamp = int(len(X) / %d)\n",nInputs_);
    fprintf(fp,"  Xt = %d * [0.0]\n", nInputs_);
    fprintf(fp,"  X2 = %d * [0.0]\n", N);
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  for ss in range(nSamp) : \n");
    fprintf(fp,"    for ii in range(%d) : \n", nInputs_);
    fprintf(fp,"      Xt[ii] = X[ss*%d+ii]\n",nInputs_);
    fprintf(fp,"    X2[0] = 1.0;\n");
    for (nn = 0; nn < nInputs_; nn++)
    {
      if (VecXMeans_[nn] != 0.0 || VecXStds_[nn] != 1.0)
           fprintf(fp,"    X2[%d] = (Xt[%d] - %e) / %e;\n", 
                   nn+1, nn, VecXMeans_[nn], VecXStds_[nn]);
      else fprintf(fp,"    X2[%d] = Xt[%d];\n", nn+1, nn);
    }
    fprintf(fp,"    Y = regCoefs[0]\n");
    if (order_ >= 1)
    {
      for (nn = 1; nn <= nInputs_; nn++)
        fprintf(fp,"    Y = Y + regCoefs[%d] * X2[%d];\n", nn, nn);
    }
    if (order_ >= 2)
    {
      ind = nInputs_ + 1;
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <=nInputs_; nn2++)
        {
          fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]\n",ind,nn,nn2);
          fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]\n",ind,nn,nn2);
          ind++;
        }
      }
    }
    if (order_ >= 3)
    {
      for (nn = 1; nn <=nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <=nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]*X2[%d]\n",
                    ind,nn,nn2,nn3);
            fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]*X2[%d]\n",ind,nn,nn2,nn3);
            ind++;
          }
        }
      }
    }
    if (order_ >= 4)
    {
      for (nn = 1; nn <= nInputs_; nn++)
      {
        for (nn2 = nn; nn2 <= nInputs_; nn2++)
        {
          for (nn3 = nn2; nn3 <= nInputs_; nn3++)
          {
            for (nn4 = nn3; nn4 <= nInputs_; nn4++)
            {
              fprintf(fp, "    Y=Y+regCoefs[%d]*X2[%d]*X2[%d]",ind,nn,nn2);
              fprintf(fp, "*X2[%d]*X2[%d]\n", nn3, nn4);
              fprintf(fp, "    X2[%d]=X2[%d]*X2[%d]*X2[%d]*X2[%d]\n",
                      ind,nn,nn2,nn3,nn4);
              ind++;
            }
          }
        }
      }
    }
    fprintf(fp,"    Ys[ss*2] = Y * %e + %e\n",YStd_, YMean_);
    fprintf(fp,"    std = 0.0\n");
    fprintf(fp,"    for jj in range(%d): \n", N);
    fprintf(fp,"      dtmp = 0.0\n");
    fprintf(fp,"      for kk in range(%d): \n", N);
    fprintf(fp,"        dtmp = dtmp + invCovMat[jj][kk] * X2[kk]\n");
    fprintf(fp,"      std = std + dtmp * X2[jj]\n");
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
    printf("FILE psuade_rs.py contains the final polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
}

