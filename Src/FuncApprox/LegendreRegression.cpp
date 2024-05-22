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
// Functions for the class LegendreRegression
// AUTHOR : CHARLES TONG
// DATE   : 2010
//**/ add code gen function and use covariance matrix (3/2014)
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "LegendreRegression.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
LegendreRegression::LegendreRegression(int nInputs,int nSamples):
                                       FuncApprox(nInputs,nSamples)
{
  int  ii;
  char line[1001], *cString, winput1[500], winput2[500], pString[101];

  //**/ =================================================================
  //**/ default parameters
  //**/ =================================================================
  faID_ = PSUADE_RS_REGRL;
  pOrder_ = 2;
  numPerms_ = 0;
  normalizeFlag_ = 0;

  //**/ =================================================================
  //**/ print header
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*                Legendre Regression Analysis\n");
    printf("* R-square gives a measure of the goodness of the model.\n");
    printf("* R-square should be close to 1 if it is a good model.\n");
    printf("* Turn on master mode to output regression matrix.\n");
    printf("* Turn on master mode to output regression error splot.\n");
    printDashes(PL_INFO, 0);
    printf("* Turn on rs_expert mode to scale the inputs to [-1, 1].\n");
    printf("* With this, statistics such as mean, variances, and\n");
    printf("* conditional variances are readily available.\n");
    printf("* Otherwise, default is: scale the inputs to [0,1].\n");
    printEquals(PL_INFO, 0);
  }

  //**/ =================================================================
  //**/ set options through expert mode or config file
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn())
  {
    numPerms_ = 0;
    pOrder_ = 0;
    while (numPerms_ < nSamples_)
    { 
      pOrder_++;
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
    }
    if (numPerms_ > nSamples_) pOrder_--;

    printf("* Legendre polynomial maximum order = %d\n", pOrder_);
    snprintf(pString,100,"Desired order (>=1 and <= %d) ? ", pOrder_);
    pOrder_ = getInt(1, pOrder_, pString);
    printf("Normalize the input parameters to [-1, 1]? (y - yes) ");
    fgets(line, 100, stdin);
    if (line[0] == 'y') normalizeFlag_ = 1;
    snprintf(pString,100,"RS_Legendre_order = %d", pOrder_);
    psConfig_.putParameter(pString);
    if (normalizeFlag_ == 1)
    {
      snprintf(pString,100,"normalize_inputs");
      psConfig_.getParameter(pString);
    }
  }
  else
  {
    cString = psConfig_.getParameter("normalize_inputs");
    if (cString != NULL) normalizeFlag_ = 1;
    cString = psConfig_.getParameter("RS_Legendre_order");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &ii);
      if (ii < 0)
      {
        if (psConfig_.InteractiveIsOn())
        {
          printf("Legendre INFO: polynomial order %d not valid.\n",ii);
          printf("               polynomial order unchanged at %d.\n",
                 pOrder_);
        }
      }
      else
      {
        pOrder_ = ii;
        printf("Legendre INFO: polynomial order set to %d (config)\n",
               pOrder_);
      }
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
LegendreRegression::~LegendreRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int LegendreRegression::initialize(double *X, double *Y)
{
  int      status;
  psVector VecX2;

  //**/ =================================================================
  //**/ If Legendre normalization is not requested, normalize to [0,1]
  //**/ =================================================================
  if (normalizeFlag_ == 0)
  {
    VecX2.setLength(nInputs_*nSamples_);
    initInputScaling(X, VecX2.getDVector(), 1);
    status = analyze(VecX2.getDVector(), Y);
  }
  else status = analyze(X, Y);
  return status; 
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int LegendreRegression::genNDGridData(double *XIn, double *YIn, int *NOut,
                                      double **XOut, double **YOut)
{
  int totPts, ss, status;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("LegendreRegression::genNDGridData - ERROR detected.\n");
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
int LegendreRegression::gen1DGridData(double *XIn, double *YIn, int ind1,
                                      double *settings, int *NOut, 
                                      double **XOut, double **YOut)
{
  int      totPts, mm, nn;
  double   HX;
  psVector VecXT;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("LegendreRegression::gen1DGridData - ERROR detected.\n");
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
  psVector VecXOut, VecYOut;
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
int LegendreRegression::gen2DGridData(double *XIn, double *YIn, int ind1,
                                      int ind2,double *settings,int *NOut, 
                                      double **XOut, double **YOut)
{
  int      totPts, mm, nn, index;
  psVector VecHX, VecXT;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("LegendreRegression::gen2DGridData - ERROR detected.\n");
    (*NOut) = 0;
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
int LegendreRegression::gen3DGridData(double *XIn, double *YIn, int ind1,
                                     int ind2, int ind3, double *settings, 
                                     int *NOut,double **XOut,double **YOut)
{
  int      totPts, mm, nn, pp, index;
  psVector VecHX, VecXT;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("LegendreRegression::gen3DGridData - ERROR detected.\n");
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
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*3);
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
int LegendreRegression::gen4DGridData(double *XIn, double *YIn, int ind1, 
                                      int ind2, int ind3, int ind4, 
                                      double *settings, int *NOut, 
                                      double **XOut, double **YOut)
{
  int      totPts, mm, nn, pp, qq, index;
  psVector VecHX, VecXT;

  //**/ =================================================================
  //**/ initialize 
  //**/ =================================================================
  if (initialize(XIn, YIn) != 0)
  {
    printf("LegendreRegression::gen4DGridData - ERROR detected.\n");
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
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*4);
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
double LegendreRegression::evaluatePoint(double *X)
{
  int    ii, nn;
  double Y, multiplier, **LTable, normalX;
  psMatrix MatLTable;

  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("LegendreRegression ERROR: initialize has not been called.\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  LTable = MatLTable.getMatrix2D();

  //**/ =================================================================
  //**/ evaluate Legendre polynomial 
  //**/ =================================================================
  Y = 0.0;
  for (nn = 0; nn < numPerms_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        EvalLegendrePolynomials(normalX, LTable[ii]);
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
    Y += VecRegCoeffs_[nn] * multiplier;
  }
  Y = Y * YStd_ + YMean_;

  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double LegendreRegression::evaluatePoint(int npts, double *X, double *Y)
{
  int kk;
  for (kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double LegendreRegression::evaluatePointFuzzy(double *X, double &std)
{
  int      iOne=1, ii, nn;
  double   Y, stdev, dtmp, multiplier, **LTable, normalX, *Xs;
  psVector VecXs;
  psMatrix MatLTable;

  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
     printf("LegendreRegression ERROR: initialize has not been called.\n");
     exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing Legendre table 
  //**/ =================================================================
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  LTable = MatLTable.getMatrix2D();

  //**/ =================================================================
  //**/ evaluate Legendre polynomial 
  //**/ =================================================================
  VecXs.setLength(numPerms_);
  Y = 0.0;
  for (nn = 0; nn < numPerms_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
        EvalLegendrePolynomials(normalX, LTable[ii]);
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
    Y += VecRegCoeffs_[nn] * multiplier;
    VecXs[nn] = multiplier;
  }
  Y = Y * YStd_ + YMean_;

  //**/ =================================================================
  //**/ compute standard deviation 
  //**/ =================================================================
  stdev = 0.0;
  for (ii = 0; ii < numPerms_; ii++)
  {
    dtmp = 0.0;
    for (nn = 0; nn < numPerms_; nn++)
      dtmp += MatInvCov_.getEntry(ii,nn) * VecXs[nn];
    stdev += dtmp * VecXs[ii];
  }
  std = sqrt(stdev) * YStd_;

  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double LegendreRegression::evaluatePointFuzzy(int npts,double *X,double *Y,
                                              double *Ystd)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&X[kk*nInputs_], Ystd[kk]);
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int LegendreRegression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int LegendreRegression::analyze(psVector VecX, psVector VecY)
{
  int    N, M, ii, mm, nn, info, NRevised;
  double *X, *Y, SSresid, SStotal, R2, var;
  double esum, ymax, *arrayA, *arrayXX, *SS, *UU, *VV;
  char   pString[101], response[1000];
  FILE   *fp;
  psMatrix eigMatT, MatXX;
  psVector eigVals;

  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (nSamples_ <= nInputs_)
  {
    printf("LegendreRegression::analyze ERROR - sample size too small.\n");
    return -1;
  }
  X = VecX.getDVector();
  Y = VecY.getDVector();

  //**/ =================================================================
  //**/ set up permutation 
  //**/ =================================================================
  GenPermutations();
   
  //**/ =================================================================
  //**/ generate regression matrix
  //**/ =================================================================
  N = loadXMatrix(VecX, MatXX);
  M = nSamples_;

  //**/ =================================================================
  //**/ set up for SVD
  //**/ =================================================================
  psMatrix MatA;
  psVector VecA;
  VecA.setLength(M*N);
  arrayA = VecA.getDVector();
  arrayXX = MatXX.getMatrix1D();
  for (mm = 0; mm < M; mm++)
    for (nn = 0; nn < N; nn++)
      arrayA[mm+nn*M] = sqrt(VecWghts_[mm]) * arrayXX[mm+nn*M];
  MatA.load(M, N, arrayA);

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
      fp = fopen("legendre_regression_matrix.m", "w");
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
          fprintf(fp, "%16.6e ", arrayA[mm+nn*M]);
        fprintf(fp, "%16.6e \n",Y[mm]);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "A = AA(:,1:%d);\n", N);
      fprintf(fp, "Y = AA(:,%d);\n", N+1);
      fprintf(fp, "B = A\\Y;\n");
      fclose(fp);
      printf("Regression matrix written to legendre_regression_matrix.m\n");
    }
  }

  //**/ =================================================================
  //**/ run SVD and check errors
  //**/ =================================================================
  psMatrix Umat, Vmat;
  psVector Svec;
  if (outputLevel_ > 3 && psConfig_.InteractiveIsOn())
    printf("Running SVD ...\n");
  info = MatA.computeSVD(Umat, Svec, Vmat);
  if (outputLevel_ > 3 && psConfig_.InteractiveIsOn())
    printf("SVD completed: status = %d (should be 0).\n",info);

  if (info != 0 && psConfig_.InteractiveIsOn())
  {
    printf("* LegendreRegression Info: dgesvd returns a nonzero (%d).\n",
           info);
    printf("* LegendreRegression terminates further processing.\n");
    return -1;
  }

  //**/ =================================================================
  //**/ eliminate the noise components 
  //**/ =================================================================
  SS = Svec.getDVector();
  mm = 0;
  for (nn = 0; nn < N; nn++) if (SS[nn] < 0) mm++;
  if (mm > 0 && psConfig_.InteractiveIsOn())
  {
    printf("LegendreRegression WARNING: Some singular values are < 0.\n"); 
    printf("                   May spell trouble but will proceed.\n");
    for (nn = 0; nn < N; nn++) if (SS[nn] < 0) SS[nn] = 0;
  }
  if (SS[0] == 0.0) NRevised = 0;
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++)
      if (SS[nn-1] > 0 && SS[nn]/SS[nn-1] < 1.0e-8) break;
    NRevised = nn;
  }
  if (NRevised < N && psConfig_.InteractiveIsOn() && outputLevel_ > 1)
  { 
    printf("LegendreRegression WARNING: True matrix rank = %d (N=%d)\n",
           NRevised,N);
    printf("INFO: This can be due to the quality of the sample.\n");
    for (nn = 0; nn < N; nn++) 
      printf("Singular value %5d = %e\n",nn+1,SS[nn]);
  }
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("LegendreRegression: For the matrix singular values,\n");
    printf(" - The VERY small ones may cause poor numerical accuracy\n");
    printf(" - but not keeping them may ruin the approximation power\n");
    printf(" - So, select them judiciously.\n");
    for (nn = 0; nn < N; nn++)
      printf("* Singular value %5d = %e\n", nn+1, SS[nn]);
    snprintf(pString,100,"How many to keep (1 - %d, 0 - all) ? ", N);
    NRevised = getInt(0,N,pString);
    if (NRevised == 0) NRevised = N;
    for (nn = NRevised; nn < N; nn++) SS[nn] = 0.0;
  }
  else
  {
    for (nn = NRevised; nn < N; nn++) SS[nn] = 0.0;
    if (NRevised < N && psConfig_.InteractiveIsOn())
      printf("LegendreRegression INFO: %d singular values removed.\n",
             N-NRevised);
  }

  //**/ =================================================================
  //**/ coefficients B = V S^{-1} U^T * sq(W) Y
  //**/ =================================================================
  psVector VecW, VecB;
  VecW.setLength(M+N);
  UU = Umat.getMatrix1D();
  for (mm = 0; mm < NRevised; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++)
      VecW[mm] += UU[nn+mm*M] * sqrt(VecWghts_[nn]) * Y[nn];
  }
  for (nn = 0; nn < NRevised; nn++) VecW[nn] /= SS[nn];
  for (nn = NRevised; nn < N; nn++) VecW[nn] = 0.0;
  VecB.setLength(N);
  VV = Vmat.getMatrix1D();
  for (mm = 0; mm < N; mm++)
  {
    VecB[mm] = 0.0;
    for (nn = 0; nn < NRevised; nn++) VecB[mm] += VV[nn+mm*N] * VecW[nn];
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
    fp = fopen("legendre_regression_error.m", "w");
    if(fp == NULL)
    {
      printf("fopen returned NULL in file %s line %d, exiting\n", 
             __FILE__, __LINE__);
      exit(1);
    }
    fprintf(fp, "disp('This file contains errors of each data point')\n");
    printf("legendre_regression_error.m file is to be created to\n");
    printf("show interpolation errors of each data point.\n");
    fprintf(fp, "A = [\n");
  }
  else fp = NULL;

  esum = ymax = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
  {
    VecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++)
      VecW[mm] = VecW[mm] + arrayXX[mm+nn*nSamples_] * VecB[nn];
    VecW[mm] = VecW[mm] - Y[mm];
    esum = esum + VecW[mm] * VecW[mm] * VecWghts_[mm];
    if (fp != NULL) 
      fprintf(fp, "%6d %24.16e\n",mm+1,VecW[mm]*sqrt(VecWghts_[mm]));
    if (PABS(Y[mm]) > ymax) ymax = PABS(Y[mm]);
  }
  esum /= (double) nSamples_;
  esum = sqrt(esum);
  if (psConfig_.InteractiveIsOn())
    printf("* LegendreR:: interpolation rms error = %10.3e (max=%10.3e)\n",
           esum, ymax); 

  if (fp != NULL)
  {
    fprintf(fp, "];\n");
    fclose(fp);
    if (psConfig_.InteractiveIsOn())
      printf("Now legendre_regression_error.m file contains data errors.\n");
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
    if (PABS(var) > 1.0e-12 && psConfig_.InteractiveIsOn())
    {
      printf("LegendreRegression WARNING: variance < 0.\n");
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
  //**/ print out regression coefficients 
  //**/ =================================================================
  if (outputLevel_ >= 0 && psConfig_.InteractiveIsOn())
  {
    if (outputLevel_ > 0) printRC();
    printf("* Regression R-square = %10.3e\n", R2);
    if (M-N-1 > 0)
      printf("* adjusted   R-square = %10.3e\n",
             1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
    if (outputLevel_ > 1) printSRC(VecX, VecB, SStotal);
    printAsterisks(PL_INFO, 0);
  }
  if (psConfig_.RSCodeGenIsOn()) genRSCode();
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int LegendreRegression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int    M, N, ss, ii, nn;
  double multiplier, normalX;
  psVector VecXX;
  psMatrix MatLTable;

  if (normalizeFlag_ == 1)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      multiplier = VecUBs_[ii] - VecLBs_[ii];
      if (multiplier == 0.0)
      {
        normalizeFlag_ = 0;
        printf("Legendre INFO: inputs not normalized - bounds not set.\n");
        break;
      }
    }
  }
  M = nSamples_;
  N = numPerms_;
  VecXX.setLength(M*N);

  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

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
    }
    for (nn = 0; nn < numPerms_; nn++)
    {
      multiplier = 1.0;
      for (ii = 0; ii < nInputs_; ii++)
        multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
      VecXX[nSamples_*nn+ss] = multiplier;
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
int LegendreRegression::computeSS(psMatrix MatXX, psVector VecY,
                            psVector VecB, double &SSresid, double &SStotal)
{
  int    N, nn, mm;
  double ymean, SSresidCheck, SSreg, ddata, rdata;
                                                                                
  N = VecB.length();
  double *VB = VecB.getDVector();
  double *VY = VecY.getDVector();
  double *VX = MatXX.getMatrix1D();

  SSresid = SSresidCheck = SStotal = ymean = SSreg = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
    ymean += (sqrt(VecWghts_[mm]) * VY[mm]);
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) ddata += (VX[mm+nn*nSamples_] * VB[nn]);
    rdata = VY[mm] - ddata;
    SSresid += rdata * VY[mm] * VecWghts_[mm];
    SSresidCheck += rdata * rdata * VecWghts_[mm];
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += VecWghts_[mm] * (VY[mm] - ymean) * (VY[mm] - ymean);
  if (outputLevel_ > 0 && psConfig_.InteractiveIsOn())
  {
    printf("* LegendreRegression: SStot  = %24.16e\n", SStotal);
    printf("* LegendreRegression: SSreg  = %24.16e\n", SSreg);
    printf("* LegendreRegression: SSres  = %24.16e\n", SSresid);
    printf("* LegendreRegression: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (outputLevel_ > 0 && nSamples_ != N && psConfig_.InteractiveIsOn())
  {
    printf("* LegendreRegression: eps(Y) = %24.16e\n", 
           SSresidCheck/(nSamples_-N));
  }
  return 0;
}

// *************************************************************************
// compute coefficient variance ((diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int LegendreRegression::computeCoeffVariance(psMatrix &eigMatT, 
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
  //**/ compute (var * V * Sigma^{-2}) * V^T 
  //**/ =================================================================
  tMat.matmult(eigMatT, MatInvCov_);
  return 0;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int LegendreRegression::printRC()
{
  int    ii, jj, kk, maxTerms, flag;
  double coef, ddata, variance;
  FILE   *fp;

  maxTerms = 0;
  for (ii = 0; ii < numPerms_; ii++) 
    if (MatPCEPerms_.getEntry(ii,0) > maxTerms) 
      maxTerms = MatPCEPerms_.getEntry(ii,0);

  printEquals(PL_INFO, 0);
  if (normalizeFlag_ == 1)
    printf("* Note: the coefficients below have been normalized\n");
  printDashes(PL_INFO, 0);
  printf("*      ");
  for (ii = 0; ii < nInputs_; ii++) printf("     ");
  printf("               coefficient  std. error  t-value\n");

  //**/ =================================================================
  //**/ --------- print out the linear coefficients --------------
  //**/ =================================================================
  for (ii = 0; ii < numPerms_; ii++)
  {
    ddata = sqrt(MatInvCov_.getEntry(ii,ii));
    if (PABS(ddata) < 1.0e-15) coef = 0.0;
    else                       coef = VecRegCoeffs_[ii] / ddata;
//**/ if (PABS(coef) > 1.0)
    {
       printf("* Input orders: ");
       for (jj = 0; jj < nInputs_; jj++)
         printf(" %4d ", MatPCEPerms_.getEntry(ii,jj));
       printf("= %11.3e %11.3e %11.3e\n", VecRegCoeffs_[ii], ddata, coef);
    }
  }
  //**/ =================================================================
  //**/ --------- print out partial variances --------------
  //**/ =================================================================
  flag = 1;
  for (ii = 0; ii < nInputs_; ii++) if (VecUBs_[ii] != 1.0) flag = 0;
  for (ii = 0; ii < nInputs_; ii++) if (VecLBs_[ii] != -1.0) flag = 0;
  if (normalizeFlag_ == 1 || flag == 1)
  {
    printDashes(PL_INFO, 0);
    printf("* Mean     = %12.4e\n", VecRegCoeffs_[0]);
    coef = 0.0;
    for (jj = 1; jj < numPerms_; jj++) 
    {
      ddata = VecRegCoeffs_[jj];
      for (kk = 0; kk < nInputs_; kk++)
        ddata /= sqrt(1.0+MatPCEPerms_.getEntry(jj,kk)*2); 
      coef = coef + ddata * ddata;
    }
    printf("* Variance = %12.4e\n", coef);
    variance = coef;
    fp = fopen("matlablegendre.m", "w");
    fwriteHold(fp,0);
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
          ddata = VecRegCoeffs_[jj];
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
    printf("Legendre VCE ranking is now in matlablegendre.m.\n");
  }
  printEquals(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// print standardized regression coefficients
// -------------------------------------------------------------------------
int LegendreRegression::printSRC(psVector VecX,psVector VecB,double SStotal)
{
  int      nn, mm, ii;
  double   denom, xmean, coef, Bmax, coef1;
  psVector VecB2;

  printEquals(PL_INFO, 0);
  printf("* Standardized Regression Coefficients (SRC)\n");
  printf("* When R-square is acceptable (order assumption holds), the\n");
  printf("* absolute values of SRCs provide variable importance.\n"); 
  printDashes(PL_INFO, 0);
  printf("* based on nSamples = %d\n", nSamples_);

  VecB2.setLength(nSamples_);
  denom = sqrt(SStotal / (double) (nSamples_ - 1));
  Bmax  = 0.0;
  for (nn = 0; nn < numPerms_; nn++)
  {
    coef = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      xmean = 0.0;
      for (mm = 0; mm < nSamples_; mm++) xmean += VecX[mm*nInputs_+ii];
      xmean /= (double) nSamples_;
      coef1 = 0.0;
      for (mm = 0; mm < nSamples_; mm++)
        coef1 += (VecX[mm*nInputs_+ii]-xmean)*(VecX[mm*nInputs_+ii]-xmean);
      coef1 = sqrt(coef1 / (double) (nSamples_ - 1));
      coef *= coef1;
    }
    VecB2[nn] = VecB[nn] * coef / denom;
    if (PABS(VecB2[nn]) > Bmax) Bmax = PABS(VecB2[nn]);
  }
  //**/ =================================================================
  //**/ importance measure 
  //**/ =================================================================
  for (nn = 0; nn < numPerms_; nn++)
  {
    if (PABS(VecB2[nn]) > 1.0e-12 * Bmax)
    {
      printf("* Input orders: ");
      for (ii = 0; ii < nInputs_; ii++)
        printf(" %2d ",MatPCEPerms_.getEntry(nn,ii));
      printf("= %12.4e \n", VecB2[nn]);
    }
  }
  printAsterisks(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// generate all combinations of a multivariate Legendre expansion
// This code is a direct translation from Burkardt's matlab code)
// -------------------------------------------------------------------------
int LegendreRegression::GenPermutations()
{
  int  ii, jj, kk, orderTmp, rvTmp;

  //**/ construct permutations
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
  if (psConfig_.InteractiveIsOn())
  {
    printf("* LegendreRegression: order of polynomials   = %d\n", pOrder_);
    printf("* LegendreRegression: number of permutations = %d\n",numPerms_);
  }
  
  //**/ =================================================================
  //**/ construct the permutations
  //**/ =================================================================
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
    while (MatPCEPerms_.getEntry(numPerms_,nInputs_-1) != kk)
    {
      numPerms_++;
      for (ii = 0; ii < nInputs_; ii++)
      {
        jj = MatPCEPerms_.getEntry(numPerms_-1, ii);
        MatPCEPerms_.setEntry(numPerms_, ii, jj);
      }
      if (orderTmp > 1) rvTmp = 1;
      else              rvTmp++;
      MatPCEPerms_.setEntry(numPerms_, rvTmp-1, 0);
      orderTmp = MatPCEPerms_.getEntry(numPerms_-1, rvTmp-1);
      MatPCEPerms_.setEntry(numPerms_, 0, orderTmp-1);
      jj = MatPCEPerms_.getEntry(numPerms_-1, rvTmp);
      MatPCEPerms_.setEntry(numPerms_, rvTmp, jj+1);
    }
    numPerms_++;
  }
  return 0;
}

// *************************************************************************
// Purpose: evaluate 1D Legendre polynomials (normalized)
// -------------------------------------------------------------------------
int LegendreRegression::EvalLegendrePolynomials(double X, double *LTable)
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
  //**/ normalize
  //**/ do not normalize (the Legendre form is harder to recognize)
  //**/ should use this instead of sqrt(0.5+2*ii) since
  //**/ need to compute 1/2 int phi_i^2 dx
  //**/ for (ii = 0; ii <= pOrder_; ii++) LTable[ii] *= sqrt(1.0+2.0*ii);
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double LegendreRegression::setParams(int targc, char **targv)
{
  pOrder_ = *(int *) targv[0];
  if (pOrder_ <= 0)
  {
    pOrder_ = -1;
    if (psConfig_.InteractiveIsOn())
      printf("LegendreRegression setParams: pOrder not valid.\n");
  }
  else 
  {
    if (psConfig_.InteractiveIsOn())
      printf("LegendreRegression setParams: pOrder set to %d.\n", 
             pOrder_);
  }
  return 0.0;
}

// ************************************************************************
// generate C and Python codes
// ------------------------------------------------------------------------
void LegendreRegression::genRSCode()
{
  int  ii, mm, nn;
  FILE *fp = fopen("psuade_rs.info", "w");

  if (fp != NULL)
  {
    fprintf(fp,"/* ***********************************************/\n");
    fprintf(fp,"/* Legendre regression interpolator from PSUADE. */\n");
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
    fprintf(fp,"/* Legendre regression interpolation function    */\n");
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
    fprintf(fp,"  double std, *x2, dtmp;\n");
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
  fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# Legendre Regression interpolation\n");
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
}

