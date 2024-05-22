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
// Functions for the class HomLegendreRegression
// AUTHOR : CHARLES TONG
// DATE   : 2018
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "HomLegendreRegression.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
HomLegendreRegression::HomLegendreRegression(int nInputs,int nSamples):
                       FuncApprox(nInputs,nSamples)
{
  //**/ =================================================================
  //**/ default parameters
  //**/ =================================================================
  faID_ = PSUADE_RS_HLEG;
  pOrder_ = 1;
  numPerms_ = 0;

  //**/ =================================================================
  //**/ print header
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*       Homogeneous Legendre Regression Analysis\n");
    printf("* R-square gives a measure of the goodness of the model.\n");
    printf("* R-square should be close to 1 if it is a good model.\n");
    printf("* Turn on master mode to spit out regression matrix stuff.\n");
    printDashes(PL_INFO, 0);
  }

  //**/ =================================================================
  //**/ set options through expert mode or config file
  //**/ =================================================================
  int  ii;
  char pString[500], *cString, winput[500], equal[100];
  if (psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100,"HomLegendre: desired polynomial order (1 - 5): ");
    pOrder_ = getInt(1, 5, pString);
    snprintf(pString,100,"RS_HomLegendre_order = %d", pOrder_);
    psConfig_.putParameter(pString);
  }
  else
  {
    cString = psConfig_.getParameter("RS_HomLegendre_order");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput, equal, &ii);
      if (ii < 0)
      {
        printf("HomLegendre INFO: Invalid polynomial order %d.\n",ii);
        printf("                  Polynomial order unchanged at %d.\n",
               pOrder_);
      }
      else
      {
        pOrder_ = ii;
        printf("HomLegendre INFO: polynomial order set to %d (config)\n",
               pOrder_);
      }
    }
    else
      printf("HomLegendre INFO: default polynomial = %d\n",pOrder_);

  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
HomLegendreRegression::~HomLegendreRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int HomLegendreRegression::initialize(double *X, double *Y)
{
  return analyze(X, Y);
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int HomLegendreRegression::genNDGridData(double *XIn,double *YIn,int *NOut,
                                         double **XOut, double **YOut)
{
  int      totPts, ss, status;
  psVector VecYOut;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("HomLegendre::genNDGridData - ERROR detected.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ return if there is no request to create lattice points
  //**/ =================================================================
  if ((*NOut) == -999) return 0;

  //**/ =================================================================
  //**/ generating regular grid data
  //**/ =================================================================
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  totPts = (*NOut);

  //**/ =================================================================
  //**/ allocate storage for the data points and generate them
  //**/ =================================================================
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  for (ss = 0; ss < totPts; ss++)
    (*YOut)[ss] = evaluatePoint(&((*XOut)[ss*nInputs_]));

  return 0;
}

// ************************************************************************
// Generate 1D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int HomLegendreRegression::gen1DGridData(double *XIn,double *YIn,int ind1,
                                         double *settings, int *NOut, 
                                         double **XOut, double **YOut)
{
  int      totPts, mm, nn;
  double   HX;
  psVector VecXT, VecXOut, VecYOut;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("HomLegendre::gen1DGridData - ERROR detected.\n");
    (*NOut) = 0;
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
  VecXOut.setLength(totPts);
  VecYOut.setLength(totPts);
  (*NOut) = totPts;
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    VecXT[ind1] = HX * mm + VecLBs_[ind1];
    (*XOut)[mm] = VecXT[ind1];
    (*YOut)[mm] = evaluatePoint(VecXT.getDVector());
  }
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (set all other inputs to nominal values)
// ------------------------------------------------------------------------
int HomLegendreRegression::gen2DGridData(double *XIn,double *YIn, int ind1,
                                    int ind2,double *settings,int *NOut, 
                                    double **XOut, double **YOut)
{
  int      totPts, mm, nn, index;
  psVector VecHX, VecXT, VecXOut, VecYOut;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("HomLegendre::gen2DGridData - ERROR detected.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(2);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  VecXOut.setLength(totPts * 2);
  VecYOut.setLength(totPts);
  (*NOut) = totPts;
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      index = mm * nPtsPerDim_ + nn;
      VecXT[ind1] = VecHX[0] * mm + VecLBs_[ind1];
      VecXT[ind2] = VecHX[1] * nn + VecLBs_[ind2];
      (*XOut)[index*2]   = VecXT[ind1];
      (*XOut)[index*2+1] = VecXT[ind2];
      (*YOut)[index] = evaluatePoint(VecXT.getDVector());
    }
  }
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int HomLegendreRegression::gen3DGridData(double *XIn,double *YIn, int ind1,
                                     int ind2, int ind3, double *settings, 
                                     int *NOut,double **XOut,double **YOut)
{
  int      totPts, mm, nn, pp, index;
  psVector VecHX, VecXT, VecXOut, VecYOut;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("HomLegendre::gen3DGridData - ERROR detected.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(3);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  VecXOut.setLength(totPts * 3);
  VecYOut.setLength(totPts);
  (*NOut) = totPts;
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        index = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        VecXT[ind1] = VecHX[0] * mm + VecLBs_[ind1];
        VecXT[ind2] = VecHX[1] * nn + VecLBs_[ind2];
        VecXT[ind3] = VecHX[2] * pp + VecLBs_[ind3];
        (*XOut)[index*3]   = VecXT[ind1];
        (*XOut)[index*3+1] = VecXT[ind2];
        (*XOut)[index*3+2] = VecXT[ind3];
        (*YOut)[index] = evaluatePoint(VecXT.getDVector());
      }
    }
  }
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int HomLegendreRegression::gen4DGridData(double *XIn,double *YIn,int ind1, 
                                         int ind2, int ind3, int ind4, 
                                         double *settings, int *NOut, 
                                         double **XOut, double **YOut)
{
  int      totPts, mm, nn, pp, qq, index;
  psVector VecHX, VecXT, VecXOut, VecYOut;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("HomLegendre::gen4DGridData - ERROR detected.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(4);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  VecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  VecXOut.setLength(totPts * 4);
  VecYOut.setLength(totPts);
  (*NOut) = totPts;
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) VecXT[nn] = settings[nn]; 
    
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
          VecXT[ind1] = VecHX[0] * mm + VecLBs_[ind1];
          VecXT[ind2] = VecHX[1] * nn + VecLBs_[ind2];
          VecXT[ind3] = VecHX[2] * pp + VecLBs_[ind3];
          VecXT[ind4] = VecHX[3] * qq + VecLBs_[ind4];
          (*XOut)[index*4]   = VecXT[ind1];
          (*XOut)[index*4+1] = VecXT[ind2];
          (*XOut)[index*4+2] = VecXT[ind3];
          (*XOut)[index*4+3] = VecXT[ind4];
          (*YOut)[index] = evaluatePoint(VecXT.getDVector());
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePoint(double *X)
{
  int    ii, nCols, iZero=0;
  double Y, normalX;
  psMatrix MatLTable;
  psVector VecXX;

  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("HomLegendre ERROR: initialize has not been called\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  nCols = VecOrderOffset_[pOrder_+1];
  for (ii = 0; ii < nInputs_; ii++)
  {
    normalX = X[ii] - VecLBs_[ii];
    normalX /= (VecUBs_[ii] - VecLBs_[ii]);
    normalX = normalX * 2.0 - 1.0;
    EvalLegendrePolynomials(normalX, LTable[ii]);
  }
  VecXX.setLength(nCols);
  ComputeMatrixRow(MatLTable, VecXX, iZero); 

  //**/ =================================================================
  //**/ evaluate Legendre polynomial 
  //**/ =================================================================
  Y = 0.0;
  for (ii = 0; ii < nCols; ii++) Y += VecRegCoeffs_[ii] * VecXX[ii];
  Y = Y * YStd_ + YMean_;
  return Y;
}

// ************************************************************************
// Evaluate a given point given coefficients
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePoint(double *X, double *coefs)
{
  int    ii, nCols, iZero=0;
  double Y, normalX;
  psMatrix MatLTable;
  psVector VecXX;

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  if (VecOrderOffset_.length() != (pOrder_+2)) GenPermutations();

  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  nCols = VecOrderOffset_[pOrder_+1];
  for (ii = 0; ii < nInputs_; ii++)
  {
    normalX = X[ii] - VecLBs_[ii];
    normalX /= (VecUBs_[ii] - VecLBs_[ii]);
    normalX = normalX * 2.0 - 1.0;
    EvalLegendrePolynomials(normalX, LTable[ii]);
  }
  VecXX.setLength(nCols);
  ComputeMatrixRow(MatLTable, VecXX, iZero); 

  //**/ =================================================================
  //**/ evaluate Legendre polynomial 
  //**/ =================================================================
  Y = 0.0;
  for (ii = 0; ii < nCols; ii++) Y += coefs[ii] * VecXX[ii];
  Y = Y * YStd_ + YMean_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePoint(int npts, double *X, double *Y)
{
  int kk;
#pragma omp parallel shared(X) private(kk)
{
#pragma omp for
  for (kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
}
  return 0.0;
}

// ************************************************************************
// Evaluate a number of points given the coefficients
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePoint(int npts, double *X, double *Y,
                                            double *coefs)
{
  int kk;
#pragma omp parallel shared(X) private(kk)
{
#pragma omp for
  for (kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_], coefs);
}
  return 0.0;
}

// ************************************************************************
// Evaluate derivative given a point and coefficients
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePointDerivative(double *X, 
                                      double *coefs, int pp)
{
  int    ii, nCols, iZero=0;
  double Y, normalX;
  psMatrix MatLTable;
  psVector VecXX;

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  nCols = VecOrderOffset_[pOrder_+1];
  for (ii = 0; ii < nInputs_; ii++)
  {
    normalX = X[ii] - VecLBs_[ii];
    normalX /= (VecUBs_[ii] - VecLBs_[ii]);
    normalX = normalX * 2.0 - 1.0;
    EvalLegendrePolynomials(normalX, LTable[ii]);
  }
  VecXX.setLength(nCols);
  ComputeMatrixRow(MatLTable, VecXX, iZero); 
  return VecXX[pp]*YStd_;
  return 0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, nn, nCols, iZero=0;
  double   Y, stdev, dtmp, normalX;
  psVector VecXX;
  psMatrix MatLTable;

  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("HomLegendre ERROR: initialize has not been called.\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();
  nCols = VecOrderOffset_[pOrder_+1];

  //**/ =================================================================
  //**/ evaluate Legendre polynomial 
  //**/ =================================================================
  for (ii = 0; ii < nInputs_; ii++)
  {
    normalX = X[ii] - VecLBs_[ii];
    normalX /= (VecUBs_[ii] - VecLBs_[ii]);
    normalX = normalX * 2.0 - 1.0;
    EvalLegendrePolynomials(normalX, LTable[ii]);
  }
  VecXX.setLength(nCols);
  ComputeMatrixRow(MatLTable, VecXX, iZero); 
  Y = 0.0;
  for (ii = 0; ii < nCols; ii++) Y += VecRegCoeffs_[ii] * VecXX[ii];
  Y = Y * YStd_ + YMean_;

  //**/ =================================================================
  //**/ compute standard deviation 
  //**/ =================================================================
  stdev = 0.0;
  for (ii = 0; ii < nCols; ii++)
  {
    dtmp = 0.0;
    for (nn = 0; nn < nCols; nn++)
      dtmp += MatInvCov_.getEntry(ii,nn) * VecXX[nn];
    stdev += dtmp * VecXX[ii];
  }
  std = sqrt(stdev) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double HomLegendreRegression::evaluatePointFuzzy(int npts,double *X,
                                                 double *Y, double *Ystd)
{
  int kk;
#pragma omp parallel private(kk)
{
#pragma omp for
  for (kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&X[kk*nInputs_], Ystd[kk]);
}
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int HomLegendreRegression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int HomLegendreRegression::analyze(psVector VecX, psVector VecY)
{
  int    N, M, ii, mm, nn, info;
  double SSresid, SStotal, R2, var, esum, ymax;
  char   pString[101], response[1000];
  FILE   *fp;
  psMatrix eigMatT, MatXX;
  psVector eigVals;

  //**/ =================================================================
  //**/ set up permutation 
  //**/ =================================================================
  GenPermutations();
   
  //**/ =================================================================
  //**/ generate regression matrix
  //**/ =================================================================
  loadXMatrix(VecX, MatXX);
  M = MatXX.nrows();
  N = MatXX.ncols();
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HomLegendre INFO: regression matrix size = %d %d\n",M,N);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("Running SVD ...\n");
  if (M < N)
  {
    printf("HomLegendre::analyze ERROR - sample size too small.\n");
    return -1;
  }

  //**/ =================================================================
  //**/ run SVD and check errors
  //**/ =================================================================
  psMatrix Umat, Vmat;
  psVector Svec;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("Running SVD ...\n");
  info = MatXX.computeSVD(Umat, Svec, Vmat);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("SVD completed: status = %d (should be 0).\n",info);

  if (info != 0)
  {
    printf("* HomLegendre Info: dgesvd returns a nonzero (%d).\n",
           info);
    printf("* HomLegendre terminates further processing.\n");
    return -1;
  }

  //**/ =================================================================
  //**/ eliminate the noise components 
  //**/ =================================================================
  double *SS = Svec.getDVector();
  mm = 0;
  for (nn = 0; nn < N; nn++) if (SS[nn] <= 0) mm++;
  if (mm > 0)
  {
    printf("* HomLegendre WARNING: some of the singular values\n"); 
    printf("*            are <= 0. May spell trouble but will\n");
    printf("*            proceed anyway (%d).\n",mm);
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    printf("* HomLegendre: matrix singular values \n");
    printf("* The VERY small ones may cause poor numerical accuracy,\n");
    printf("* but not keeping them may ruin the approximation power.\n");
    printf("* So, select them judiciously.\n");
    for (nn = 0; nn < N; nn++)
      printf("* Singular value %5d = %e\n", nn+1, SS[nn]);
  }

  //**/ =================================================================
  //**/ coefficients B = V S^{-1} U^T * sq(W) Y
  //**/ =================================================================
  psVector VecW, VecB;
  VecW.setLength(M+N);
  double *UU = Umat.getMatrix1D();
  for (mm = 0; mm < N; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++)
      VecW[mm] += UU[nn+mm*M] * VecY[nn];
  }
  for (nn = 0; nn < N; nn++) 
  {
    if (SS[nn] > 1e-12) VecW[nn] /= SS[nn];
    else                VecW[nn] = 0;
  }
  VecB.setLength(N);
  double *VV = Vmat.getMatrix1D();
  for (mm = 0; mm < N; mm++)
  {
    VecB[mm] = 0.0;
    for (nn = 0; nn < N; nn++) VecB[mm] += VV[nn+mm*N] * VecW[nn];
  }

  //**/ =================================================================
  //**/ store eigenvectors VV and eigenvalues SS^2
  //**/ =================================================================
  eigMatT.load(N, N, VV);
  eigVals.load(N, SS);
  for (nn = 0; nn < N; nn++) eigVals[nn] = pow(eigVals[nn], 2.0);

  //**/ =================================================================
  //**/ compute residual
  //**/ =================================================================
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    fp = fopen("homlegendre_regression_error.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n", 
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp,"disp('This file contains errors of each data point')\n");
    printf("homlegendre_regression_error.m is to be created to show\n");
    printf("interpolation errors of each data point.\n");
    fprintf(fp, "A = [\n"); 
  }
  else fp = NULL;

  esum = ymax = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++)
      VecW[mm] = VecW[mm] + MatXX.getEntry(mm, nn) * VecB[nn];
    VecW[mm] = VecW[mm] - VecY[mm];
    esum = esum + VecW[mm] * VecW[mm];
    if (fp != NULL) 
      fprintf(fp, "%6d %24.16e\n",mm+1,VecW[mm]);
    if (PABS(VecY[mm]) > ymax) ymax = PABS(VecY[mm]);
  }
  esum /= (double) nSamples_;
  esum = sqrt(esum);
  if (psConfig_.InteractiveIsOn())
    printf("* HLegendre:: interpolation rms error = %10.3e (ymax=%10.3e)\n",
           esum, ymax); 

  if (fp != NULL)
  {
    if (fp != NULL) fprintf(fp, "];\n"); 
    fclose(fp);
    printf("homlegendre_regression_error.m file contains data errors.\n");
  }

  //**/ =================================================================
  //**/ compute variance and R2 
  //**/ =================================================================
  computeSS(MatXX, VecY, VecB, SSresid, SStotal);
  if (SStotal == 0) R2 = 1.0;
  else              R2 = 1.0 - SSresid / SStotal;
  if (nSamples_ > N) var = SSresid / (double) (nSamples_ - N);
  else               var = 0.0;
  if (var < 0)
  { 
    if (PABS(var) > 1.0e-12)
    {
      printf("HomLegendre WARNING: variance < 0.\n");
      printf("    Temporarily absolutize var (may have problems).\n");
      var = PABS(var);
    }
    else var = 0.0;
  }
  VecRegCoeffs_.load(VecB);

  //**/ =================================================================
  //**/ find variance of each coefficient 
  //**/ =================================================================
  computeCoeffVariance(eigMatT, eigVals, var);
  psVector VecBstd;
  VecBstd.setLength(N);
  for (ii = 0; ii < N; ii++)
    VecBstd[ii] = sqrt(MatInvCov_.getEntry(ii,ii));

  //**/ =================================================================
  //**/ this is for diagnostics
  //**/ =================================================================
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("You have the option to store the regression matrix (that\n");
    printf("is, the matrix A in Ax=b) in a matlab file for inspection.\n");
    snprintf(pString,100,"Store regression matrix? (y or n) ");
    getString(pString, response);
    if (response[0] == 'y')
    {
      fp = fopen("homlegendre_regression_matrix.m", "w");
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
          fprintf(fp, "%16.6e ", MatXX.getEntry(mm, nn));
        fprintf(fp, "%16.6e \n", VecY[mm]);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "A = AA(:,1:%d);\n", N);
      fprintf(fp, "Y = AA(:,%d);\n", N+1);
      fprintf(fp, "B = A\\Y;\n");
      fprintf(fp, "S = [\n");
      for (mm = 0; mm < Svec.length(); mm++)
        fprintf(fp, "%16.6e \n", Svec[mm]);
      fprintf(fp, "];\n");
      fprintf(fp, "VT = [\n");
      for (mm = 0; mm < Vmat.nrows(); mm++)
      {
        for (nn = 0; nn < Vmat.ncols(); nn++) 
          fprintf(fp, "%16.6e ", Vmat.getEntry(mm,nn));
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "U = [\n");
      for (mm = 0; mm < Umat.nrows(); mm++)
      {
        for (nn = 0; nn < Umat.ncols(); nn++) 
          fprintf(fp, "%16.6e ", Umat.getEntry(mm,nn));
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "var = %16.6e;\n", var);
      fprintf(fp, "norm(norm(A-U*S*VT));\n");
      fprintf(fp, "disp('Check U * S * VT against A')\n");
      fclose(fp);
      printf("Regression matrix now in homlegendre_regression_matrix.m\n");
    }
  }

  //**/ =================================================================
  //**/ print out regression coefficients 
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0) 
  {
    if (outputLevel_ > 3) 
    {
      printf("Inverse of covariance matrix of the coefficients:\n");
      MatInvCov_.print();
    }
    if (outputLevel_ > 0) printRC(VecB, VecBstd, MatXX, VecY);
    printf("* Regression R-square = %10.3e\n", R2);
    if (SStotal < 1e-14) 
      printf("* (This may not be accurate as SStotal is very small.\n");

    if (M-N-1 > 0)
      printf("* adjusted   R-square = %10.3e\n",
             1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
    printAsterisks(PL_INFO, 0);
  }
  return 0;
}

// *************************************************************************
// load the MatXX matrix 
// -------------------------------------------------------------------------
int HomLegendreRegression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int ss, ii, errFlagL=0, errFlagU=0, iZero=0;

  //**/ check that the lower and upper bounds exist and are consistent
  for (ii = 1; ii < nInputs_; ii++)
  {
    if (VecLBs_[ii] != VecLBs_[0]) errFlagL++;
    if (VecUBs_[ii] != VecUBs_[0]) errFlagU++;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (errFlagL != 0)
      printf("HomLegendre: inconsistent lower bounds (%d violations)\n",
             errFlagL);
    if (errFlagU != 0)
      printf("HomLegendre: inconsistent upper bounds (%d violations)\n",
             errFlagU);
  }

  //**/ create MatXX - the Legendre X matrix
  double normalX;
  psMatrix MatLTable;
  psVector VecXX;

  MatXX.setFormat(PS_MAT1D);
  MatXX.setDim(nSamples_, VecOrderOffset_[pOrder_+1]);
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();
  VecXX.setLength(MatXX.ncols());

  for (ss = 0; ss < nSamples_; ss++)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
      printf("HomLegendre loadX: computing row %d of %d\n",ss+1,nSamples_); 
    for (ii = 0; ii < nInputs_; ii++)
    {
      normalX = VecX[ss*nInputs_+ii] - VecLBs_[ii];
      normalX /= (VecUBs_[ii] - VecLBs_[ii]);
      normalX = normalX * 2.0 - 1.0;
      EvalLegendrePolynomials(normalX, LTable[ii]);
    }
    ComputeMatrixRow(MatLTable, VecXX, iZero); 
    for (ii = 0; ii < VecXX.length(); ii++) 
      MatXX.setEntry(ss, ii, VecXX[ii]);
  }
  return 0;
}

// *************************************************************************
// compute SS (sum of squares) statistics
//**/ SSred = (Y - Xb)' W (Y - Xb)
//**/       = Y' W Y - 2 b 'X W 'Y + b' X' W X b
//**/       = Y' W Y - 2 b' X' W Y + b' X' W Y  (since X'WXb=X'WY)
//**/       = Y' W Y - b' X' W Y = (Y - Xb)' W Y
//**/ SStot = Y' W Y - N * (mean (W^(1/2) Y))^2
// -------------------------------------------------------------------------
int HomLegendreRegression::computeSS(psMatrix MatXX, psVector VecY,
                            psVector VecB, double &SSresid, double &SStotal)
{
  int    N, nn, mm, M;
  double ymean, SSresidCheck, SSreg, ddata, rdata;
                                                                                
  M = MatXX.nrows();
  N = VecB.length();
  double *B = VecB.getDVector();
  double *Y = VecY.getDVector();
  double *arrayXX = MatXX.getMatrix1D();

  SSresid = SSresidCheck = SStotal = ymean = SSreg = 0.0;
  for (mm = 0; mm < nSamples_; mm++) ymean += Y[mm];
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) ddata += (arrayXX[mm+nn*M] * B[nn]);
    rdata = Y[mm] - ddata;
    SSresid += rdata * Y[mm];
    SSresidCheck += rdata * rdata;
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += (Y[mm] - ymean) * (Y[mm] - ymean);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
  {
    printf("* HomLegendre: SStot  = %24.16e\n", SStotal);
    printf("* HomLegendre: SSreg  = %24.16e\n", SSreg);
    printf("* HomLegendre: SSres  = %24.16e\n", SSresid);
    printf("* HomLegendre: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0 && nSamples_ != N)
  {
    printf("* HomLegendre: eps(Y) = %24.16e\n", 
           SSresidCheck/(nSamples_-N));
  }
  return 0;
}

// *************************************************************************
// compute coefficient variance ((diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int HomLegendreRegression::computeCoeffVariance(psMatrix &eigMatT, 
                                  psVector &eigVals, double var)
{
  int      ii, jj, nRows;
  double   invEig, dtmp;
  psMatrix tMat;

  nRows = eigMatT.nrows();
  tMat.setDim(nRows, nRows);

  //**/ =================================================================
  //**/ compute var * V * S^{-2}
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
  //**/ compute (var * V * S^{-2}) * V^T 
  //**/ =================================================================
  tMat.matmult(eigMatT, MatInvCov_);
  return 0;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int HomLegendreRegression::printRC(psVector VecB, psVector VecBstd, 
                                   psMatrix MatXX, psVector VecY)
{
  double coef;

  printEquals(PL_INFO, 0);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 2)
  {
    printf("* Note: the coefficients below have been normalized\n");
    printDashes(PL_INFO, 0);
    printf("*          Coefficient std. error  t-value\n");
  }

  //**/ =================================================================
  //**/ --------- print out the linear coefficients --------------
  //**/ =================================================================
  for (int ii = 0; ii < MatXX.ncols(); ii++)
  {
    if (PABS(VecBstd[ii]) < 1.0e-15) coef = 0.0;
    else                             coef = VecB[ii] / VecBstd[ii]; 
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2)
    {
      printf("Term %3d = %11.3e      %11.3e      %11.3e\n", 
             ii+1, VecB[ii], VecBstd[ii], coef);
    }
  }
  printEquals(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// generate a mapping from permutation number to actual column number of
// the compressed matrix
// -------------------------------------------------------------------------
int HomLegendreRegression::GenPermutations()
{
  int  ii;
  char pString[101];

  //**/ =================================================================
  //**/ ask for polynomial order and then number of terms
  //**/ =================================================================
  if (pOrder_ < 0)
  {
    snprintf(pString,100,"Desired order (>=1 and <= 5) ? ");
    pOrder_ = getInt(1, 5, pString);
  }
  numPerms_ = 1;
  if (nInputs_ > pOrder_)
  {
    for (ii = nInputs_+pOrder_; ii > nInputs_; ii--)
      numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
  }
  else
  {
    for (ii = nInputs_+pOrder_; ii > pOrder_; ii--)
      numPerms_ = numPerms_ * ii / (nInputs_+pOrder_-ii+1);
  }
  if (psConfig_.InteractiveIsOn())
  {
    printf("* HomLegendre: order of polynomials   = %d\n", pOrder_);
    printf("* HomLegendre: number of permutations = %d\n",numPerms_);
  }
  
  //**/ =================================================================
  //**/ construct possible combination of orders up to 6th order
  //**/ =================================================================
  int maxEntries = 19;
  VecOrderOffset_.setLength(pOrder_+2);

  MatPermTable_.setDim(maxEntries, pOrder_);
  VecOrderOffset_[0] = 0;
  VecOrderOffset_[1] = 1;
  MatPermTable_.setEntry(1,0,1); /* 1 0 ... */
  VecOrderOffset_[2] = 2;
  if (pOrder_ >= 2)
  {
    MatPermTable_.setEntry(2,0,2); /* 2 0 ... */
    MatPermTable_.setEntry(3,0,1); /* 1 1 ... */
    MatPermTable_.setEntry(3,1,1);
    VecOrderOffset_[3] = 4;
  }
  if (pOrder_ >= 3)
  {
    MatPermTable_.setEntry(4,0,3); /* 3 0 ... */
    MatPermTable_.setEntry(5,0,2); /* 2 1 ... */
    MatPermTable_.setEntry(5,1,1); /* 2 1 ... */
    MatPermTable_.setEntry(6,0,1); /* 1 1 1.. */
    MatPermTable_.setEntry(6,1,1);
    MatPermTable_.setEntry(6,2,1);
    VecOrderOffset_[4] = 7;
  }
  if (pOrder_ >= 4)
  { 
    MatPermTable_.setEntry(7,0,4); /* 4 0 ... */
    MatPermTable_.setEntry(8,0,3); /* 3 1 ... */
    MatPermTable_.setEntry(8,1,1);
    MatPermTable_.setEntry(9,0,2); /* 2 2 ... */
    MatPermTable_.setEntry(9,1,2);
    MatPermTable_.setEntry(10,0,2); /* 2 1 1 ... */
    MatPermTable_.setEntry(10,1,1);
    MatPermTable_.setEntry(10,2,1);
    MatPermTable_.setEntry(11,0,1); /* 1 1 1 1 ... */
    MatPermTable_.setEntry(11,1,1);
    MatPermTable_.setEntry(11,2,1);
    MatPermTable_.setEntry(11,3,1);
    VecOrderOffset_[5] = 12;
  }
  if (pOrder_ >= 5)
  {
    MatPermTable_.setEntry(12,0,5); /* 5 0 ... */
    MatPermTable_.setEntry(13,0,4); /* 4 1 ... */
    MatPermTable_.setEntry(13,1,1);
    MatPermTable_.setEntry(14,0,3); /* 3 2 ... */
    MatPermTable_.setEntry(14,1,2);
    MatPermTable_.setEntry(15,0,3); /* 3 1 1 ... */
    MatPermTable_.setEntry(15,1,1);
    MatPermTable_.setEntry(15,2,1);
    MatPermTable_.setEntry(16,0,2); /* 2 2 1 ... */
    MatPermTable_.setEntry(16,1,2);
    MatPermTable_.setEntry(16,2,1);
    MatPermTable_.setEntry(17,0,2); /* 2 1 1 1 ... */
    MatPermTable_.setEntry(17,1,1);
    MatPermTable_.setEntry(17,2,1);
    MatPermTable_.setEntry(17,3,1);
    MatPermTable_.setEntry(18,0,1); /* 1 1 1 1 1 ... */
    MatPermTable_.setEntry(18,1,1);
    MatPermTable_.setEntry(18,2,1);
    MatPermTable_.setEntry(18,3,1);
    MatPermTable_.setEntry(18,4,1);
    VecOrderOffset_[6] = 19;
  }
  return 0;
}

// *************************************************************************
// compare state
// -------------------------------------------------------------------------
int HomLegendreRegression::SearchPermutation(psIVector v1)
{
  int ii, jj;
  for (ii = 0; ii < MatPermTable_.nrows(); ii++)
  {
    for (jj = 0; jj < MatPermTable_.ncols(); jj++)
      if (v1[jj] != MatPermTable_.getEntry(ii,jj)) break;
    if (jj == MatPermTable_.ncols()) return ii;
  }
  return -1;
}

// *************************************************************************
// Purpose: evaluate 1D Legendre polynomials (normalized)
// -------------------------------------------------------------------------
int HomLegendreRegression::EvalLegendrePolynomials(double X, double *LTable)
{
  int ii;
  LTable[0] = 1.0;
  if (pOrder_ >= 1)
  {
    LTable[1] = X;
    for (ii = 2; ii <= pOrder_; ii++)
      LTable[ii] = ((2 * ii - 1) * X * LTable[ii-1] -
                    (ii - 1) * LTable[ii-2]) / ii;
  }
  //**/ normalize
  //**/ do not normalize (the Legendre form is harder to recognize)
  //**/ should use this instead of sqrt(0.5+2*ii) since
  //**/ need to compute 1/2 int phi_i^2 dx
  //**/ for (ii = 0; ii <= pOrder_; ii++) LTable[ii] *= sqrt(1.0+2.0*ii);
  return 0;
}

// *************************************************************************
// ComputeMatrixRow: compute matrix row 
// -------------------------------------------------------------------------
int HomLegendreRegression::ComputeMatrixRow(psMatrix MatLTable,
                                            psVector &VecRow, int printFlag)
{
  int    nn1, nn2, nn3, nn4, nn5, index, iZero=0;
  double **LTable, dprod, dZero=0.0;
  psIVector VecCmp, VecCnts;

  VecCmp.setLength(nInputs_);
  LTable = MatLTable.getMatrix2D();
  VecRow.setConstant(dZero);
  VecCnts.setLength(nInputs_);
  VecCnts.setConstant(iZero);
  VecRow[0] = 1.0;
  VecCnts[0] = 1;
  if (pOrder_ >= 1)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++) 
    {
      VecRow[1] += LTable[nn1][1];
      VecCnts[1]++;
    }
  }
  if (pOrder_ >= 2)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = 0; nn2 < nInputs_; nn2++)
      {
        VecCmp.setConstant(dZero);
        if (nn1 == nn2)
        {
          VecCmp[0] = 2;
          dprod = LTable[nn1][2];
        }
        else
        {
          VecCmp[0] = VecCmp[1] = 1;
          dprod = LTable[nn1][1] * LTable[nn2][1];
        }
        index = SearchPermutation(VecCmp);   
        VecRow[index] += dprod;
        VecCnts[index]++;
      }    
    }
  }
  if (pOrder_ >= 3)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = 0; nn2 < nInputs_; nn2++)
      {
        for (nn3 = 0; nn3 < nInputs_; nn3++)
        {
          VecCmp.setConstant(dZero);
          if (nn1 == nn2 && nn1 == nn3)
          {
            VecCmp[0] = 3;
            dprod = LTable[nn1][3];
          }
          else if (nn1 == nn2)
          {
            VecCmp[0] = 2;
            VecCmp[1] = 1;
            dprod = LTable[nn1][2] * LTable[nn3][1];
          }
          else if (nn1 == nn3)
          {
            VecCmp[0] = 2;
            VecCmp[1] = 1;
            dprod = LTable[nn1][2] * LTable[nn2][1];
          }
          else if (nn2 == nn3)
          {
            VecCmp[0] = 2;
            VecCmp[1] = 1;
            dprod = LTable[nn1][1] * LTable[nn2][2];
          }
          else
          {
            VecCmp[0] = 1;
            VecCmp[1] = 1;
            VecCmp[2] = 1;
            dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][1];
          }
          index = SearchPermutation(VecCmp);   
          VecRow[index] += dprod;
          VecCnts[index]++;
        }
      }
    }
  }
  if (pOrder_ >= 4)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = 0; nn2 < nInputs_; nn2++)
      {
        for (nn3 = 0; nn3 < nInputs_; nn3++)
        {
          for (nn4 = 0; nn4 < nInputs_; nn4++)
          {
            VecCmp.setConstant(dZero);
            if (nn1 == nn2 && nn1 == nn3 && nn1 == nn4)
            {
              VecCmp[0] = 4;
              dprod = LTable[nn1][4];
            }
            else if (nn1 == nn2 && nn1 == nn3)
            {
              VecCmp[0] = 3;
              VecCmp[1] = 1;
              dprod = LTable[nn1][3] * LTable[nn4][1];
            }
            else if (nn1 == nn2 && nn1 == nn4)
            {
              VecCmp[0] = 3;
              VecCmp[1] = 1;
              dprod = LTable[nn1][3] * LTable[nn3][1];
            }
            else if (nn1 == nn3 && nn1 == nn4)
            {
              VecCmp[0] = 3;
              VecCmp[1] = 1;
              dprod = LTable[nn1][3] * LTable[nn2][1];
            }
            else if (nn2 == nn3 && nn2 == nn4)
            {
              VecCmp[0] = 3;
              VecCmp[1] = 1;
              dprod = LTable[nn2][3] * LTable[nn1][1];
            }
            else if (nn1 == nn2 && nn3 == nn4)
            {
              VecCmp[0] = 2;
              VecCmp[1] = 2;
              dprod = LTable[nn1][2] * LTable[nn3][2];
            }
            else if (nn1 == nn3 && nn2 == nn4) 
            {
              VecCmp[0] = 2;
              VecCmp[1] = 2;
              dprod = LTable[nn1][2] * LTable[nn2][2];
            }
            else if (nn1 == nn4 && nn2 == nn3)
            {
              VecCmp[0] = 2;
              VecCmp[1] = 2;
              dprod = LTable[nn1][2] * LTable[nn2][2];
            }
            else if (nn1 == nn2)
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn1][2] * LTable[nn3][1] * LTable[nn4][1];
            }
            else if (nn1 == nn3)
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn1][2] * LTable[nn2][1] * LTable[nn4][1];
            }
            else if (nn1 == nn4) 
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn1][2] * LTable[nn2][1] * LTable[nn3][1];
            }
            else if (nn2 == nn3) 
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn2][2] * LTable[nn1][1] * LTable[nn4][1];
            }
            else if (nn2 == nn4) 
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn2][2] * LTable[nn1][1] * LTable[nn3][1];
            }
            else if (nn3 == nn4) 
            {
              VecCmp[0] = 2;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              dprod = LTable[nn3][2] * LTable[nn1][1] * LTable[nn2][1];
            }
            else
            {
              VecCmp[0] = 1;
              VecCmp[1] = 1;
              VecCmp[2] = 1;
              VecCmp[3] = 1;
              dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][1] *
                      LTable[nn4][1];
            }
            index = SearchPermutation(VecCmp);   
            VecRow[index] += dprod;
            VecCnts[index]++;
          }
        }
      }
    }
  }
  if (pOrder_ >= 5)
  {
    for (nn1 = 0; nn1 < nInputs_; nn1++)
    {
      for (nn2 = 0; nn2 < nInputs_; nn2++)
      {
        for (nn3 = 0; nn3 < nInputs_; nn3++)
        {
          for (nn4 = 0; nn4 < nInputs_; nn4++)
          {
            for (nn5 = 0; nn5 < nInputs_; nn5++)
            {
              VecCmp.setConstant(dZero);
              if (nn1 == nn2 && nn1 == nn3 && nn1 == nn4 && nn1 == nn5)
              {
                VecCmp[0] = 5;
                dprod = LTable[nn1][5];
              }
              else if (nn1 == nn2 && nn1 == nn3 && nn1 == nn4)
              {
                VecCmp[0] = 4;
                VecCmp[1] = 1;
                dprod = LTable[nn1][4] * LTable[nn5][1];
              }
              else if (nn1 == nn2 && nn1 == nn3 && nn1 == nn5)
              {
                VecCmp[0] = 4;
                VecCmp[1] = 1;
                dprod = LTable[nn1][4] * LTable[nn4][1];
              }
              else if (nn1 == nn2 && nn1 == nn4 && nn1 == nn5)
              {
                VecCmp[0] = 4;
                VecCmp[1] = 1;
                dprod = LTable[nn1][4] * LTable[nn3][1];
              }
              else if (nn1 == nn3 && nn1 == nn4 && nn1 == nn5)
              {
                VecCmp[0] = 4;
                VecCmp[1] = 1;
                dprod = LTable[nn1][4] * LTable[nn2][1];
              }
              else if (nn2 == nn3 && nn2 == nn4 && nn2 == nn5)
              {
                VecCmp[0] = 4;
                VecCmp[1] = 1;
                dprod = LTable[nn2][4] * LTable[nn1][1];
              }
              else if (nn1 == nn2 && nn2 == nn3 && nn4 == nn5)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn4][2];
              }
              else if (nn1 == nn2 && nn2 == nn4 && nn3 == nn5)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn3][2];
              }
              else if (nn1 == nn2 && nn2 == nn5 && nn3 == nn4)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn3][2];
              }
              else if (nn1 == nn3 && nn1 == nn4 && nn2 == nn5)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn2][2];
              }
              else if (nn1 == nn3 && nn1 == nn5 && nn2 == nn4)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn2][2];
              }
              else if (nn1 == nn4 && nn1 == nn5 && nn2 == nn3)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][3] * LTable[nn2][2];
              }
              else if (nn2 == nn3 && nn2 == nn4 && nn1 == nn5)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][2] * LTable[nn2][3];
              }
              else if (nn2 == nn3 && nn2 == nn5 && nn1 == nn4)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][2] * LTable[nn2][3];
              }
              else if (nn2 == nn4 && nn2 == nn5 && nn1 == nn3)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][2] * LTable[nn2][3];
              }
              else if (nn3 == nn4 && nn3 == nn5 && nn1 == nn2)
              {
                VecCmp[0] = 3;
                VecCmp[1] = 2;
                dprod = LTable[nn1][2] * LTable[nn3][3];
              }
              else if (nn1 == nn2 && nn3 == nn4)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn3][2] * LTable[nn5][1];
              }
              else if (nn1 == nn2 && nn3 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn3][2] * LTable[nn4][1];
              }
              else if (nn1 == nn2 && nn4 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn4][2] * LTable[nn3][1];
              }
              else if (nn1 == nn3 && nn2 == nn4)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn5][1];
              }
              else if (nn1 == nn3 && nn2 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn4][1];
              }
              else if (nn1 == nn3 && nn4 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn4][2] * LTable[nn2][1];
              }
              else if (nn1 == nn4 && nn2 == nn3)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn5][1];
              }
              else if (nn1 == nn4 && nn2 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn3][1];
              }
              else if (nn1 == nn4 && nn3 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn3][2] * LTable[nn2][1];
              }
              else if (nn1 == nn5 && nn2 == nn3)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn4][1];
              }
              else if (nn1 == nn5 && nn2 == nn4)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][2] * LTable[nn3][1];
              }
              else if (nn1 == nn5 && nn3 == nn4)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][2] * LTable[nn3][2] * LTable[nn2][1];
              }
              else if (nn2 == nn3 && nn4 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn4][2];
              }
              else if (nn2 == nn4 && nn3 == nn5)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn3][2];
              }
              else if (nn2 == nn5 && nn3 == nn4)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 2;
                VecCmp[2] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn3][2];
              }
              else if (nn1 == nn2)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][2] * LTable[nn3][1] * LTable[nn4][1] *
                        LTable[nn5][1];
              }
              else if (nn1 == nn3)
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][1] * LTable[nn4][1] *
                        LTable[nn5][1];
              }
              else if (nn1 == nn4) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][1] * LTable[nn3][1] *
                        LTable[nn5][1];
              }
              else if (nn1 == nn5) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][2] * LTable[nn2][1] * LTable[nn3][1] *
                        LTable[nn4][1];
              }
              else if (nn2 == nn3) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn4][1] *
                        LTable[nn5][1];
              }
              else if (nn2 == nn4) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn3][1] *
                        LTable[nn5][1];
              }
              else if (nn2 == nn5) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][2] * LTable[nn3][1] *
                        LTable[nn4][1];
              }
              else if (nn3 == nn4) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][2] *
                        LTable[nn5][1];
              }
              else if (nn3 == nn5) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][2] *
                        LTable[nn4][1];
              }
              else if (nn4 == nn5) 
              {
                VecCmp[0] = 2;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][1] *
                        LTable[nn4][2];
              }
              else
              {
                VecCmp[0] = 1;
                VecCmp[1] = 1;
                VecCmp[2] = 1;
                VecCmp[3] = 1;
                VecCmp[4] = 1;
                dprod = LTable[nn1][1] * LTable[nn2][1] * LTable[nn3][1] *
                        LTable[nn4][1] * LTable[nn5][1];
              }
              index = SearchPermutation(VecCmp);   
              VecRow[index] += dprod;
              VecCnts[index]++;
            }
          }
        }
      }
    }
  }
  if (psConfig_.InteractiveIsOn() && printFlag == 1)
  {
    for (nn1 = 0; nn1 < VecRow.length(); nn1++)
      printf("HLR Coefficent %2d has %7d occurrences\n",nn1,VecCnts[nn1]);
  }
  //**/ this is used if we want to scale the matrix
  //**/ but this may not help in reducing prediction 
  //**/ uncertainty (Aug 21, 2019)
  //**/for (nn1 = 0; nn1 < VecRow.length(); nn1++)
  //**/  VecRow[nn1] = VecRow[nn1] / VecCnts[nn1];
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double HomLegendreRegression::setParams(int targc, char **targv)
{
  pOrder_ = *(int *) targv[0];
  if (pOrder_ <= 0 || pOrder_ > 5)
  {
    pOrder_ = -1;
    printf("HomLegendre setParams ERROR: pOrder not valid.\n");
    exit(1);
  }
  else 
  {
    if (psConfig_.InteractiveIsOn())
      printf("HomLegendre setParams: pOrder set to %d.\n",pOrder_);
  }
  return 0.0;
}

// ************************************************************************
// get parameters
// ------------------------------------------------------------------------
int HomLegendreRegression::getRegressionCoefficients(psVector &VecCoefs,
                                                     psVector &VecStds)
{
  int    ii;
  double dtmp;
  VecCoefs = VecRegCoeffs_;
  VecStds.setLength(VecRegCoeffs_.length());
  for (ii = 0; ii < VecRegCoeffs_.length(); ii++)
  {
    dtmp = MatInvCov_.getEntry(ii,ii);
    if (dtmp > 0) VecStds[ii] = sqrt(dtmp);
  }
  return 0;
}

