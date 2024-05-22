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
// Functions for the class CSRegression (not done)
// AUTHOR : CHARLES TONG
// DATE   : 2016
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sysdef.h"
#include "Psuade.h"
#include "CSRegression.h"
#include "PDFManager.h"
#include "PsuadeUtil.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

extern "C"
{
   void dgels_(char *, int *, int *, int *, double *, int *,
               double *, int *, double *, int *, int *);
   void dgetrf_(int *, int *, double *, int *, int *, int *);
   void dgetri_(int *, double *, int *, int *, double*, int *, int *);
}

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
CSRegression::CSRegression(int nInputs,int nSamples):
                    FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_CSREG;
  numTerms_  = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
CSRegression::~CSRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int CSRegression::initialize(double *X, double *Y)
{
  //**/ ---------------------------------------------------------------
  //**/ launch regression 
  //**/ ---------------------------------------------------------------
  int status = analyze(X, Y);
  if (status != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int CSRegression::genNDGridData(double *XIn, double *YIn, int *NOut, 
                                double **XOut, double **YOut)
{
  int mm, totPts;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  if (initialize(XIn,YIn) != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
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
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;
  for (mm = 0; mm < totPts; mm++)
    (*YOut)[mm] = evaluatePoint(&((*XOut)[mm*nInputs_]));
  return 0;
}

// ************************************************************************
// Generate 1D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int CSRegression::gen1DGridData(double *XIn, double *YIn, int ind1,
                                double *settings, int *NOut, 
                                double **XOut, double **YOut)
{
  int    totPts, mm, nn;
  double HX;
  psVector vecXT;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  if (initialize(XIn,YIn) != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
    (*NOut) = 0;
    return -1;
  }

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for and then generate the data points
  //**/ ---------------------------------------------------------------
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    vecXT[ind1] = HX * mm + VecLBs_[ind1];
    (*XOut)[mm] = vecXT[ind1];
    (*YOut)[mm] = evaluatePoint(vecXT.getDVector());
  }
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int CSRegression::gen2DGridData(double *XIn, double *YIn, int ind1,
                                int ind2, double *settings, int *NOut, 
                                double **XOut, double **YOut)
{
  int totPts, mm, nn, ind;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  if (initialize(XIn,YIn) != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
    (*NOut) = 0;
    return -1;
  }

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
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      ind = mm * nPtsPerDim_ + nn;
      vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
      vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
      (*XOut)[ind*2]   = vecXT[ind1];
      (*XOut)[ind*2+1] = vecXT[ind2];
      (*YOut)[ind] = evaluatePoint(vecXT.getDVector());
    }
  }
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int CSRegression::gen3DGridData(double *XIn, double *YIn, int ind1,
                                int ind2, int ind3, double *settings, 
                                int *NOut, double **XOut, double **YOut)
{
  int totPts, mm, nn, pp, ind;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  if (initialize(XIn,YIn) != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
    (*NOut) = 0;
    return -1;
  }

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
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  vecXT.setLength(nInputs_);
  for (nn = 0; nn < nInputs_; nn++) vecXT[nn] = settings[nn]; 
    
  for (mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (pp = 0; pp < nPtsPerDim_; pp++)
      {
        ind = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
        vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
        vecXT[ind3] = vecHX[2] * pp + VecLBs_[ind3];
        (*XOut)[ind*3]   = vecXT[ind1];
        (*XOut)[ind*3+1] = vecXT[ind2];
        (*XOut)[ind*3+2] = vecXT[ind3];
        (*YOut)[ind] = evaluatePoint(vecXT.getDVector());
      }
    }
  }
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int CSRegression::gen4DGridData(double *XIn, double *YIn,int ind1,int ind2,
                                int ind3, int ind4, double *settings, 
                                int *NOut, double **XOut, double **YOut)
{
  int totPts, mm, nn, pp, qq, ind;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  if (initialize(XIn,YIn) != 0)
  {
    printf("CSRegression: ERROR detected in regression analysis.\n");
    (*NOut) = 0;
    return -1;
  }
 
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
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
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
          ind = mm*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                nn*nPtsPerDim_*nPtsPerDim_ + pp*nPtsPerDim_ + qq;
          vecXT[ind1] = vecHX[0] * mm + VecLBs_[ind1];
          vecXT[ind2] = vecHX[1] * nn + VecLBs_[ind2];
          vecXT[ind3] = vecHX[2] * pp + VecLBs_[ind3];
          vecXT[ind4] = vecHX[3] * qq + VecLBs_[ind4];
          (*XOut)[ind*4]   = vecXT[ind1];
          (*XOut)[ind*4+1] = vecXT[ind2];
          (*XOut)[ind*4+2] = vecXT[ind3];
          (*XOut)[ind*4+3] = vecXT[ind4];
          (*YOut)[ind] = evaluatePoint(vecXT.getDVector());
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double CSRegression::evaluatePoint(double *X)
{
  int    mm, nn;
  double Xdata, Xdata2, Y;

  if (VecRegCoeffs_.length() == 0)
  {
    printf("CSRegression ERROR: need to call initialize first.\n");
    return 0.0;
  }
  Y = VecRegCoeffs_[0];
  for (mm = 1; mm < numTerms_; mm++)
  {
    Xdata = 0.0;
    for (nn = 0; nn < nInputs_; nn++)
    {
      Xdata2 = (X[nn] - VecXMeans_[nn]) / VecXStds_[nn]; 
      Xdata += pow(Xdata2, 1.0*MatTermOrders_.getEntry(mm,nn));
    }
    Y += VecRegCoeffs_[mm] * Xdata;
  }
  Y = Y * YStd_ + YMean_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double CSRegression::evaluatePoint(int npts, double *X, double *Y)
{
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePoint(&X[kk*nInputs_]);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point and also its standard deviation
// ------------------------------------------------------------------------
double CSRegression::evaluatePointFuzzy(double *X, double &std)
{
  int    mm, nn;
  double YOut, Xdata, Xdata2, stdev, ddata;
  psVector vecXs;

  //**/ evaluate mean 
  vecXs.setLength(nInputs_+1);
  vecXs[0] = 1;
  for (nn = 0; nn < nInputs_; nn++)
    vecXs[nn] = (X[nn] - VecXMeans_[nn]) / VecXStds_[nn]; 

  YOut = VecRegCoeffs_[0];
  for (mm = 1; mm < numTerms_; mm++)
  {
    Xdata = 0.0;
    for (nn = 0; nn < nInputs_; nn++)
      Xdata += pow(vecXs[nn], 1.0*MatTermOrders_.getEntry(mm,nn));
    YOut += VecRegCoeffs_[mm] * Xdata;
  }
  YOut = YOut * YStd_ + YMean_;

  //**/ evaluate standard deviation
  stdev = 0.0;
  for (mm = 0; mm < numTerms_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < numTerms_; nn++)
      ddata += invCovMat_.getEntry(mm,nn) * vecXs[nn];
    stdev += ddata * vecXs[mm];
  }
  stdev = sqrt(stdev) * YStd_;
  return YOut;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double CSRegression::evaluatePointFuzzy(int npts, double *X, double *Y,
                                        double *Ystd)
{
  //**/ if no coefficient, flag error
  if (VecRegCoeffs_.length() == 0)
  {
    printf("CSRegression ERROR: initialize has not been called.\n");
    exit(1);
  }
  //**/ evaluate one point at a time
  for (int kk = 0; kk < npts; kk++)
    Y[kk] = evaluatePointFuzzy(&(X[kk*nInputs_]), Ystd[kk]);
  return 0.0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int CSRegression::analyze(double *Xin, double *Y)
{
  psVector VecX, VecY;
  VecX.load(nSamples_*nInputs_, Xin);
  VecY.load(nSamples_, Y);
  return analyze(VecX, VecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int CSRegression::analyze(psVector VecXin, psVector VecYin)
{
  int    ii, mm, nn, status;
  double SSresid, SStotal, R2, var, enorm, ymax, dtemp, *arrayXX;
  psVector vecXT;
  psMatrix matXX, matXTX;

  //**/ =================================================================
  //**/ preliminary error checking
  //**/ =================================================================
  if (nInputs_ <= 0 || nSamples_ <= 0)
  {
    printf("CSRegression ERROR: consult PSUADE developers.\n");
    exit( 1 );
  } 
   
  //**/ =================================================================
  //**/ print linear, quadratic, cubic, or quartic regression analysis
  //**/ =================================================================
  if (outputLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    printf("*         Compressed Sensing Regression Analysis\n");
    printf("* R-squared gives a measure of the goodness of the model.\n");
    printf("* R-squared should be close to 1 if it is a good model.\n");
    printDashes(PL_INFO, 0);
  }

  //**/ =================================================================
  //**/ optional scaling of the sample matrix
  //**/ =================================================================
  vecXT.setLength(nSamples_*nInputs_);
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("* CSRegression INFO: scaling turned off.\n");
    printf("*              To turn on scaling, use rs_expert mode.\n");
    initInputScaling(VecXin.getDVector(), vecXT.getDVector(), 0);
  }
  else initInputScaling(VecXin.getDVector(), vecXT.getDVector(), 1);

  //**/ =================================================================
  //**/ further modify the polynomial order based on number of terms
  //**/ =================================================================
  status = optimize(vecXT, VecYin);
  if (status < 0)
  {
    printf("* CSRegression ERROR: error in compression.\n");
    exit(1);
  }   

  //**/ =================================================================
  //**/ load matrix based on orders
  //**/ =================================================================
  loadXMatrix(vecXT, matXX);

  //**/ =================================================================
  //**/ perform SVD 
  //**/ =================================================================
  arrayXX = matXX.getMatrix1D();
  enorm = ymax = 0.0;
  for (mm = 0; mm < nSamples_; mm++)
  {
    dtemp = 0.0;
    for (nn = 0; nn < numTerms_; nn++) 
      dtemp += arrayXX[mm+nn*nSamples_] * VecRegCoeffs_[nn];
    dtemp -= VecYin[mm];
    enorm = enorm + dtemp * dtemp;
    if (PABS(VecYin[mm]) > ymax) ymax = PABS(VecYin[mm]);
  }
  enorm /= (double) nSamples_;
  enorm = sqrt(enorm);
  if (outputLevel_ > 1)
    printf("* CSRegression: interpolation RMS error = %11.4e (Ymax=%9.2e)\n",
           enorm, ymax); 

  //**/ =================================================================
  //**/ form compute SS statistics 
  //**/ =================================================================
  computeSS(matXX, VecYin, VecRegCoeffs_, SSresid, SStotal);
  R2 = 1.0;
  if (SStotal != 0.0) R2  = 1.0 - SSresid / SStotal;
  if (nSamples_ > numTerms_) 
       var = SSresid / (double) (nSamples_ - numTerms_);
  else var = 0.0;
  if (var < 0)
  { 
    if (PABS(var) > 1.0e-12)
         printf("CSRegression WARNING: var < 0.\n");
    else var = 0;
  }

  //**/ =================================================================
  //**/ find variance of each coefficient 
  //**/ =================================================================
  VecRegStdevs_.setLength(numTerms_);
  computeXTX(matXX, matXTX);
  computeCoeffVariance(matXTX, var, VecRegStdevs_);

  //**/ =================================================================
  //**/ print out regression coefficients 
  //**/ =================================================================
  if (outputLevel_ >= 0)
  {
    if (outputLevel_ > 0) printRC();
    printf("* CSRegression R-square = %12.4e ", R2);
    printf("(SSresid,SStotal=%10.2e,%10.2e)\n", SSresid, SStotal);
    if ((nSamples_ - numTerms_ - 1) > 0)
      printf("* adjusted   R-square = %12.4e\n",
             1.0 - (1.0-R2)*((nSamples_-1)/(nSamples_-numTerms_-1)));
  }
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int CSRegression::loadXMatrix(psVector VecX, psMatrix &MatXX)
{
  int    mm, nn, ii;
  double dtemp, dprod;
  psVector vecXX;

  vecXX.setLength(nSamples_*numTerms_);
  for (mm = 0; mm < nSamples_; mm++) vecXX[mm] = 1.0;
  for (nn = 1; nn < numTerms_; nn++)
  {
    for (mm = 0; mm < nSamples_; mm++)
    {
      dprod = 1.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        dtemp = VecX[mm*nInputs_+ii];
        dprod *= pow(dtemp, 1.0*MatTermOrders_.getEntry(nn,ii));
      }
      vecXX[nSamples_*nn+mm] = dprod;
    }
  }
  MatXX.setFormat(PS_MAT1D);
  MatXX.load(nSamples_, numTerms_, vecXX.getDVector());
  return 0;
}

// *************************************************************************
// form X^T X 
// -------------------------------------------------------------------------
int CSRegression::computeXTX(psMatrix matX, psMatrix &MatXTX)
{
  double   coef, *arrayX;
  psVector vecXX;

  arrayX = matX.getMatrix1D();
  vecXX.setLength(numTerms_*numTerms_);
  for (int nn = 0; nn < numTerms_; nn++)
  {
    for (int nn2 = 0; nn2 < numTerms_; nn2++)
    {
      coef = 0.0;
      for (int mm = 0; mm < nSamples_; mm++)
        coef += arrayX[nn*nSamples_+mm]*VecWghts_[mm]*
                arrayX[nn2*nSamples_+mm];
      vecXX[nn*numTerms_+nn2] = coef;
    }
  }
  MatXTX.setFormat(PS_MAT1D);
  MatXTX.load(numTerms_, numTerms_, vecXX.getDVector());
  return 0;
}

// *************************************************************************
// compute SS (sum of squares) statistics
// -------------------------------------------------------------------------
int CSRegression::computeSS(psMatrix MatXX, psVector VecYin, psVector VecB,
                            double &SSresid, double &SStotal)
{
  int    nn, mm;
  double rdata, ymean, SSreg, ddata, SSresidCheck, *arrayXX;

  arrayXX = MatXX.getMatrix1D();
  SSresid = SSresidCheck = SStotal = SSreg = ymean = 0.0;
  for (mm = 0; mm < nSamples_; mm++) 
    ymean += sqrt(VecWghts_[mm]) * VecYin[mm];
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < numTerms_; nn++) 
      ddata += (arrayXX[mm+nn*nSamples_] * VecB[nn]);
    rdata = VecYin[mm] - ddata;
    SSresidCheck += rdata * rdata * VecWghts_[mm];
    SSresid += rdata * VecYin[mm] * VecWghts_[mm];
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += VecWghts_[mm] * (VecYin[mm] - ymean) * (VecYin[mm] - ymean);
  if (outputLevel_ > 0)
  {
    printf("* CSRegression: SStot  = %24.16e\n", SStotal);
    printf("* CSRegression: SSreg  = %24.16e\n", SSreg);
    printf("* CSRegression: SSres  = %24.16e\n", SSresid);
    printf("* CSRegression: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (outputLevel_ > 0 && nSamples_ != numTerms_)
  {
    printf("* CSRegression: eps(Y) = %24.16e\n",
           SSresidCheck/(nSamples_-numTerms_));
  }
  return 0;
}

// *************************************************************************
// compute coefficient variances (diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int CSRegression::computeCoeffVariance(psMatrix MatXTX, double var,
                                       psVector VecB)
{
  int    nn, nn2, lwork, info;
  double ddata, ddata2, *arrayXTX;
  psVector  vecW, vecInvA;
  psIVector vecIPiv;

  //**/ compute inverse of MatXTX
  arrayXTX = MatXTX.getMatrix1D();
  vecIPiv.setLength(numTerms_+1);
  lwork = 2 * numTerms_ * numTerms_;
  vecInvA.setLength(lwork);
  for (nn = 0; nn < numTerms_*numTerms_; nn++) 
    vecInvA[nn] = arrayXTX[nn];
  dgetrf_(&numTerms_, &numTerms_, vecInvA.getDVector(), &numTerms_, 
          vecIPiv.getIVector(), &info);
  if (info != 0)
    printf("CSRegression WARNING: dgels returns error %d.\n",info);

  vecW.setLength(lwork);
  dgetri_(&numTerms_,vecInvA.getDVector(),&numTerms_,vecIPiv.getIVector(), 
          vecW.getDVector(), &lwork, &info);
  invCovMat_.setDim(numTerms_,numTerms_);
  for (nn = 0; nn < numTerms_; nn++)
  {
    for (nn2 = 0; nn2 < numTerms_; nn2++)
    {
      ddata = vecInvA[nn*numTerms_+nn2] * var;
      invCovMat_.setEntry(nn,nn2,ddata);
    }
  }
  //**/ symmetrizing the scaled inverse covariance matrix
  for (nn = 0; nn < numTerms_; nn++)
  {
    for (nn2 = 0; nn2 < nn; nn2++)
    {
      ddata  = invCovMat_.getEntry(nn,nn2);
      ddata2 = invCovMat_.getEntry(nn2,nn);
      ddata  = 0.5 * (ddata + ddata2);
      invCovMat_.setEntry(nn,nn2,ddata);
      invCovMat_.setEntry(nn2,nn,ddata);
    }
  }
  return info;
}

// *************************************************************************
// print statistics
// -------------------------------------------------------------------------
int CSRegression::printRC()
{
  int mm, ii;
  double bstd;
  printEquals(PL_INFO, 0);
  printf("*** Note: these coefficients may not be true coefficients due\n");
  printf("***       to sample matrix scaling (i.e., they may be scaled).\n");
  printDashes(PL_INFO, 0);
  printf("* ");
  for (ii = 0; ii < nInputs_; ii++) printf("   ");
  printf("  coefficient   std. error   t-value\n");
  printDashes(PL_INFO, 0);
  //**/ ------ print out the scaled linear coefficients -----------
  for (mm = 0; mm < numTerms_; mm++)
  {
    for (ii = 0; ii < nInputs_; ii++)
      printf(" %d ", MatTermOrders_.getEntry(mm,ii));
    bstd = invCovMat_.getEntry(mm, mm);
    printf("= %12.4e %12.4e %12.4e\n", VecRegCoeffs_[mm], bstd, 
           VecRegCoeffs_[mm]/bstd);
  }
  printDashes(PL_INFO, 0);
  return 0;
}

// *************************************************************************
// optimize 
// -------------------------------------------------------------------------
int CSRegression::optimize(psVector VecX, psVector VecY)
{
  printf("* CSRegression ERROR: CSRegression not implemented yet.\n");
  return -1;
}

