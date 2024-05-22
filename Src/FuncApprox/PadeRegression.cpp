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
// Functions for the class PadeRegression
// AUTHOR : CHARLES TONG
// DATE   : 2023
//**/ *********************************************************************
//**/ What does this method do?
//**/ Given a function F(X) = multivariate polynomial of some degree, to
//**/ approximate F(X) by F(X) ~ P(X)/Q(X) where
//**/ P(X) is a multivariate polynomial of degree p, and
//**/ Q(X) is a multivariate polynomial of degree q
//**/ Main task: to estimate the coefficients of P(X) and Q(X) so that
//**/      |hat(F)(X) - F(X)| is minimized (hat(F)(X) - the approximant)  
//**/ This regression procedure is performed given a set of sample points
// ************************************************************************
//**/ The Pade method:
//**/ * The above equation can be rewritten as (let Y = F(X))
//**/          Y Q(X) ~ P(X)     
//**/ * for each sample point k
//**/       y_k Q(X_k) = P(X_k), or
//**/       P(X_k) - [Q(X_k) - 1] y_k = y_k  (assume q_0=1)
//**/ * Compiling equations for all sample points gives rise to a linear
//**/   system Y = X c where
//**/    X = [1 x_11 x_12 ... {np-th} (-y_1 x_11) (-y_1 x_12) ... {nq-th} 
//**/         1 x_21 x_22 ... {np-th} (-y_2 x_21) (-y_2 x_22) ... {nq-th} 
//**/         ...]
//**/    c = [p_0 p_1 ... q_1 q_2 ...]
//**/   where x_ij is the value of the j-th variable in the i-th sample pt
//**/   X is a N x (np+nq+1) matrix for sample size N
//**/   where np = number of terms in p-th multivariate polynomial
//**/ * To solve this linear system, we can use least squares method
//**/     X' Y = X'X c ==> c = (X'X)^{-1} X' Y
//**/ * However, X is nearly singular ==> need regularization
//**/ * Solve using total least squares method (Reference: Estimating the 
//**/   Nonparametric Regression by Using Pade approximation based on
//**/   Total Least Squares by S. Ahmed, D. Aydin and E. Yilmaz)
//**/   a. Form an augmented matrix [X,Y] ==> N x (np+nq+2)  
//**/   b. Compute SVD of the augmented matrix U S V' where U is a NxN
//**/      matrix, V is a (np+nq+2)x)(np+nq+2) matrix, and S is Nx(np+nq+2)
//**/      If sv(np+nq+1) > sv(np+nq+2), then [X,Y] = U S* V is a least
//**/      squares solution where S* = S with S(np+nq+2)=0. With this
//**/      c = -1/v(m,m) [V(1,m+1) ... V(m,m+1)]' where m=np+nq+2 
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PadeRegression.h"
#include "PsuadeUtil.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
PadeRegression::PadeRegression(int nInputs,int nSamples):
                FuncApprox(nInputs,nSamples)
{
  faID_  = PSUADE_RS_PADE;
  normalizeFlag_ = 0;
  pOrder_ = 7;
  qOrder_ = 7;
  maxOrder_ = 10;

  //**/ =================================================================
  //**/ print header
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*               Pade Regression Analysis\n");
    printDashes(PL_INFO, 0);
    printf("* Pade approximation uses F(X) ~ P(X) / Q(X) where ");
    printf("both P(X) and Q(X)\n");
    printf("* are polynomials with different orders. Do ");
    printf("experiment with different\n");
    printEquals(PL_INFO, 0);
    printf("* orders to see how good the fits are.\n");
    printf("* Numerator   polynomial order = %d\n",pOrder_);
    printf("* Denominator polynomial order = %d\n",qOrder_);
    printf("* R-squared gives a measure of the goodness of the model.\n");
    printf("* R-squared should be close to 1 if it is a good model.\n");
    printEquals(PL_INFO, 0);
  }

  char pString[101];
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    snprintf(pString,100,"Desired order for P(x) (>=1 and <= 10) ? ");
    pOrder_ = getInt(1, 10, pString);
    snprintf(pString,100,"Desired order for Q(x) (>=1 and <= 10) ? ");
    qOrder_ = getInt(1, 10, pString);
    snprintf(pString,100,"RS_PADE_porder = %d", pOrder_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RS_PADE_qorder = %d", qOrder_);
    psConfig_.putParameter(pString);
  }
  else
  {
    //**/ =======================================================
    // read from configure file, if any 
    //**/ =======================================================
    int  order;
    char winput[500], winput2[500], *strPtr;
    strPtr = psConfig_.getParameter("RS_PADE_porder");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, winput2, &order);
      if (order < 1 || order > 10)
      {
        printf("PADE INFO: pOrder from config not valid.\n");
        printf("           pOrder kept at %d.\n", pOrder_);
      }
      else
      {
        pOrder_ = order;
        printf("PADE INFO: pOrder from config = %d.\n",pOrder_);
      }
    }
    strPtr = psConfig_.getParameter("RS_PADE_qorder");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, winput2, &order);
      if (order < 1 || order > 10)
      {
        printf("PADE INFO: qOrder from config not valid.\n");
        printf("           qOrder kept at %d.\n", qOrder_);
      }
      else
      {
        qOrder_ = order;
        printf("PADE INFO: pOrder from config = %d.\n",qOrder_);
      }
    }
  }
  maxOrder_ = pOrder_;
  if (qOrder_ > maxOrder_) maxOrder_ = qOrder_;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PadeRegression::~PadeRegression()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int PadeRegression::initialize(double *X, double *Y)
{
  int status;
 
  //**/ =================================================================
  //**/ launch different order regression (up to a maximum of 4)
  //**/ =================================================================
  status = analyze(X, Y);
  if (status != 0)
  {
    printf("PadeRegression: ERROR detected in regression analysis.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// Generate lattice data based on the input set
// ------------------------------------------------------------------------
int PadeRegression::genNDGridData(double *X, double *Y, int *NN, 
                                  double **XX, double **YY)
{
  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("PadeRegression: ERROR detected in regression analysis.\n");
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
  int totPts = (*NN);

  //**/ =================================================================
  //**/ allocate storage for the data points and generate them
  //**/ =================================================================
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YY) = VecYOut.takeDVector();
  (*NN) = totPts;
  for (int mm = 0; mm < totPts; mm++)
    (*YY)[mm] = evaluatePoint(&((*XX)[mm*nInputs_]));
  return 0;
}

// ************************************************************************
// Generate 1D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PadeRegression::gen1DGridData(double *X, double *Y, int ind1,
                                  double *settings, int *NN, 
                                  double **XX, double **YY)
{
  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("PadeRegression: ERROR detected in gen1DGridData.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  int    totPts = nPtsPerDim_;
  double HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector vecXLocal, vecXOut, vecYOut;
  vecXOut.setLength(totPts);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  (*NN) = totPts;
  vecXLocal.setLength(nInputs_);
  for (int nn = 0; nn < nInputs_; nn++) vecXLocal[nn] = settings[nn]; 
   
  for (int mm = 0; mm < nPtsPerDim_; mm++) 
  {
    vecXLocal[ind1] = HX * mm + VecLBs_[ind1];
    (*XX)[mm] = vecXLocal[ind1];
    (*YY)[mm] = evaluatePoint(vecXLocal.getDVector());
  }
  return 0;
}

// ************************************************************************
// Generate 2D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PadeRegression::gen2DGridData(double *X, double *Y, int ind1,
                                  int ind2, double *settings, int *NN, 
                                  double **XX, double **YY)
{
  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("PadeRegression: ERROR detected in gen2DGridData.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector vecXOut, vecYOut, vecXLocal;
  vecXOut.setLength(totPts*2);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  (*NN) = totPts;
  vecXLocal.setLength(nInputs_);
  for (int ii = 0; ii < nInputs_; ii++) vecXLocal[ii] = settings[ii]; 
   
  int ind;
  for (int mm = 0; mm < nPtsPerDim_; mm++)
  {
    for (int nn = 0; nn < nPtsPerDim_; nn++)
    {
      ind = mm * nPtsPerDim_ + nn;
      vecXLocal[ind1] = vecHX[0] * mm + VecLBs_[ind1];
      vecXLocal[ind2] = vecHX[1] * nn + VecLBs_[ind2];
      (*XX)[ind*2]   = vecXLocal[ind1];
      (*XX)[ind*2+1] = vecXLocal[ind2];
      (*YY)[ind] = evaluatePoint(vecXLocal.getDVector());
    }
  }
  return 0;
}

// ************************************************************************
// Generate 3D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PadeRegression::gen3DGridData(double *X, double *Y, int ind1,
                                  int ind2, int ind3, double *settings, 
                                  int *NN, double **XX, double **YY)
{
  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("PadeRegression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }

  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector vecXOut, vecYOut, vecXLocal;
  vecXOut.setLength(totPts*3);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  (*NN) = totPts;
  vecXLocal.setLength(nInputs_);
  for (int ii = 0; ii < nInputs_; ii++) vecXLocal[ii] = settings[ii]; 
    
  int ind;
  for (int mm = 0; mm < nPtsPerDim_; mm++) 
  {
    for (int nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (int pp = 0; pp < nPtsPerDim_; pp++)
      {
        ind = mm * nPtsPerDim_ * nPtsPerDim_ + nn * nPtsPerDim_ + pp;
        vecXLocal[ind1] = vecHX[0] * mm + VecLBs_[ind1];
        vecXLocal[ind2] = vecHX[1] * nn + VecLBs_[ind2];
        vecXLocal[ind3] = vecHX[2] * pp + VecLBs_[ind3];
        (*XX)[ind*3]   = vecXLocal[ind1];
        (*XX)[ind*3+1] = vecXLocal[ind2];
        (*XX)[ind*3+2] = vecXLocal[ind3];
        (*YY)[ind] = evaluatePoint(vecXLocal.getDVector());
      }
    }
  }
  return 0;
}

// ************************************************************************
// Generate 4D mesh results (setting others to some nominal values) 
// ------------------------------------------------------------------------
int PadeRegression::gen4DGridData(double *X, double *Y, int ind1, int ind2,
                                  int ind3, int ind4, double *settings, 
                                  int *NN, double **XX, double **YY)
{
  //**/ =================================================================
  //**/ initialization
  //**/ =================================================================
  if (initialize(X,Y) != 0)
  {
    printf("PadeRegression: ERROR detected in regression analysis.\n");
    (*NN) = 0;
    return -1;
  }
 
  //**/ =================================================================
  //**/ set up for generating regular grid data
  //**/ =================================================================
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ =================================================================
  //**/ allocate storage for and then generate the data points
  //**/ =================================================================
  psVector vecXOut, vecYOut, vecXLocal;
  vecXOut.setLength(totPts*4);
  vecYOut.setLength(totPts);
  (*XX) = vecXOut.takeDVector();
  (*YY) = vecYOut.takeDVector();
  (*NN) = totPts;
  vecXLocal.setLength(nInputs_);
  for (int ii = 0; ii < nInputs_; ii++) vecXLocal[ii] = settings[ii]; 
    
  int ind;
  for (int mm = 0; mm < nPtsPerDim_; mm++)
  {
    for (int nn = 0; nn < nPtsPerDim_; nn++)
    {
      for (int pp = 0; pp < nPtsPerDim_; pp++)
      {
        for (int qq = 0; qq < nPtsPerDim_; qq++)
        {
          ind = mm*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                nn*nPtsPerDim_*nPtsPerDim_ + pp*nPtsPerDim_ + qq;
          vecXLocal[ind1] = vecHX[0] * mm + VecLBs_[ind1];
          vecXLocal[ind2] = vecHX[1] * nn + VecLBs_[ind2];
          vecXLocal[ind3] = vecHX[2] * pp + VecLBs_[ind3];
          vecXLocal[ind4] = vecHX[3] * qq + VecLBs_[ind4];
          (*XX)[ind*4]   = vecXLocal[ind1];
          (*XX)[ind*4+1] = vecXLocal[ind2];
          (*XX)[ind*4+2] = vecXLocal[ind3];
          (*XX)[ind*4+3] = vecXLocal[ind4];
          (*YY)[ind] = evaluatePoint(vecXLocal.getDVector());
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// Evaluate a given point
// ------------------------------------------------------------------------
double PadeRegression::evaluatePoint(double *X)
{
  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
    printf("LegendreRegression ERROR: initialize has not been called.\n");
    exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing polynomial index table
  //**/ Note: Only table for P(x) is needed since Q(x) has lower order so
  //**/       just call it LTable
  //**/ =================================================================
  psMatrix MatLTable;
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, maxOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  //**/ =================================================================
  //**/ evaluate numerator polynomial
  //**/ =================================================================
  int    ii, nn;                      
  double numerator=0.0, normalX, multiplier;
  for (nn = 0; nn < numPermsP_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
      }
      evalLegendrePolynomials(normalX, LTable[ii]);
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    numerator += VecRegCoeffs_[nn] * multiplier;
  }

  //**/ =================================================================
  //**/ evaluate denominator polynomial
  //**/ (Note: the denominator = 1 + .... (i.e., assume q_0=1)
  //**/ =================================================================
  double denominator = 1.0;
  for (nn = 1; nn < numPermsQ_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
      }
      evalLegendrePolynomials(normalX, LTable[ii]);
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    denominator += VecRegCoeffs_[numPermsP_+nn-1] * multiplier;
  }
  double Y;
  if (denominator == 0)
  {
    printf("PadeRegression evaluatePoint ERROR: denominator = 0.\n"); 
    printf("                             Set output = 0.\n"); 
    Y = 0;
  }
  else Y = numerator / denominator * YStd_ + YMean_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double PadeRegression::evaluatePoint(int npts, double *X, double *Y)
{
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
double PadeRegression::evaluatePointFuzzy(double *X, double &std)
{
  //**/ =================================================================
  //**/ error checking
  //**/ =================================================================
  if (VecRegCoeffs_.length() <= 0)
  {
     printf("PadeRegression ERROR: initialize has not been called.\n");
     exit(1);
  }

  //**/ =================================================================
  //**/ allocate for computing polynomial index table
  //**/ Note: the reason this does not just call evaluatePoint to get Y
  //**/       is that we need vecPs and vecQs here to compute standard
  //**/       deviation
  //**/ Note: Both P(x) and Q(x) can share the same table but the table
  //**/       should have been and have been loaded with polynomial order
  //**/       the max of that of P(x) and Q(x)
  //**/ =================================================================
  psMatrix MatLTable;
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, maxOrder_+1);
  double **LTable = MatLTable.getMatrix2D();

  //**/ =================================================================
  //**/ evaluate numerator polynomial
  //**/ =================================================================
  psVector vecXs;
  vecXs.setLength(numPermsP_+numPermsQ_-1);
  int ii, nn;
  double numerator = 0.0, multiplier, normalX;
  for (nn = 0; nn < numPermsP_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
      {
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
      }
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
      }
      evalLegendrePolynomials(normalX, LTable[ii]);
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    numerator += VecRegCoeffs_[nn] * multiplier;
    vecXs[nn] = multiplier;
  }

  //**/ =================================================================
  //**/ evaluate denominator polynomial
  //**/ (Note: the denominator = 1 + .... (i.e., assume q_0=1)
  //**/ =================================================================
  double denominator = 1.0;
  for (nn = 1; nn < numPermsQ_; nn++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0)
        normalX = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
      else
      {
        normalX = X[ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
      }
      evalLegendrePolynomials(normalX, LTable[ii]);
    }
    multiplier = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
    denominator += VecRegCoeffs_[numPermsP_+nn-1] * multiplier;
    vecXs[numPermsP_+nn-1] = multiplier; /* Need to multiply by -y? */
  }
  double Y;
  if (denominator == 0)
  {
    printf("PadeRegression ERROR: Denominator = 0 in evaluatePoint.\n"); 
    printf("                      Set output = 0.\n"); 
    Y = 0;
  }
  else Y = numerator / denominator * YStd_ + YMean_;

#if 1
  //**/ =================================================================
  //**/ compute standard deviation 
  //**/ =================================================================
  int numPerms = numPermsP_ + numPermsQ_ - 1;
  double stdev = 0.0;
  double dtmp;
  //**/ Should this be done?
  for (ii = numPermsP_; ii < numPerms; ii++)
    vecXs[ii] = -vecXs[ii] * Y;
  for (ii = 0; ii < numPerms; ii++)
  {
    dtmp = 0.0;
    for (nn = 0; nn < numPerms; nn++)
      dtmp += MatInvCov_.getEntry(ii,nn) * vecXs[nn];
    stdev += dtmp * vecXs[ii];
  }
  if (stdev < 0) stdev = 0;
  std = sqrt(stdev) * YStd_;
#else
  std = 0;
#endif
  return Y;
}

// ************************************************************************
// Evaluate a number of points and also their standard deviations
// ------------------------------------------------------------------------
double PadeRegression::evaluatePointFuzzy(int npts, double *X, double *Y,
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
double PadeRegression::setParams(int targc, char **targv)
{
  pOrder_ = *(int *) targv[0];
  qOrder_ = *(int *) targv[0];
  if (pOrder_ <= 0 || pOrder_ > 6) pOrder_ = 6;
  if (qOrder_ <= 0 || qOrder_ > 5) qOrder_ = 5;
  maxOrder_ = pOrder_;
  if (qOrder_ > maxOrder_) maxOrder_ = qOrder_;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 4)
  {
    printf("PadeRegression setParam: \n");
    printf("    Numerator   polynomial order = %d\n",pOrder_);
    printf("    Denominator polynomial order = %d\n",qOrder_);
  }
  return 0;
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int PadeRegression::analyze(double *Xin, double *Y)
{
  psVector vecX, vecY;
  vecX.load(nSamples_*nInputs_, Xin);
  vecY.load(nSamples_, Y);
  return analyze(vecX, vecY);
}

// ************************************************************************
// perform regression analysis
// ------------------------------------------------------------------------
int PadeRegression::analyze(psVector vecXin, psVector vecY)
{
  //**/ =================================================================
  //**/ preliminary error checking
  //**/ =================================================================
  if (nInputs_ <= 0 || nSamples_ <= 0)
  {
    printf("PadeRegression ERROR: nInputs or nSamples <= 0.\n");
    exit( 1 );
  } 
   
  //**/ =================================================================
  //**/ check for the order (needs order+1 distinct points in each input)
  //**/ =================================================================
  int ii, mm, nn, last;
  psVector vecTmp;
  vecTmp.setLength(nSamples_);
  double *tmpArray = vecTmp.getDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (mm = 0; mm < nSamples_; mm++)
      tmpArray[mm] = vecXin[mm*nInputs_+ii];
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
    if (last <= pOrder_ || last <= qOrder_)
    {
      pOrder_ = last - 1;
      printf("PadeRegression ERROR: Not enough distinct values in ");
      printf("input %d to support\n",ii+1); 
      printf("               polynomial orders (%d,%d).\n",pOrder_,qOrder_);
      exit(1);
    }
  }

  //**/ =================================================================
  //**/ set up permutation ==> MatPCEPerms_  
  //**/ =================================================================
  genPermutations();

  //**/ =================================================================
  //**/ optional scaling of the sample matrix (vecXin ==> vecX)
  //**/ (default is : do scaling)
  //**/ =================================================================
  psVector vecX;
  vecX.setLength(nSamples_ * nInputs_);
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("PadeRegression INFO: Scaling turned off in master mode.\n");
    printf("               To turn scaling on in master mode, ");
    printf("turn on rs_expert\n");
    printf("               mode first.\n");
    initInputScaling(vecXin.getDVector(), vecX.getDVector(), 0);
  }
  else initInputScaling(vecXin.getDVector(), vecX.getDVector(), 1);

  //**/ =================================================================
  //**/ make sure sample size is large enough to support polynomial order
  //**/ =================================================================
  psMatrix matXX;
  int N = loadXMatrix(vecX, vecY, matXX); 
  if (N == 0) return -1;
  if (N > nSamples_)
  {
    printf("PadeRegression ERROR: Sample too small for polynomial order.\n");
    return -1;
  }
  int M = nSamples_;

  //**/ =================================================================
  //**/ fill the A matrix (M by N matrix)
  //**/ =================================================================
  psVector VecA;
  VecA.setLength(M*N);
  double *arrayXX = matXX.getMatrix1D();
  for (mm = 0; mm < M; mm++) 
    for (nn = 0; nn < N; nn++) 
      VecA[mm+nn*M] = arrayXX[mm+nn*M];
  psMatrix matA;
  matA.load(M, N, VecA.getDVector());

  //**/ =================================================================
  //**/ diagnostics
  //**/ =================================================================
  FILE *fp=NULL;
  char pString[101], response[1000];
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("You have the option to store the regression matrix (that\n");
    printf("is, the matrix A in Ax=b) in a matlab file for inspection.\n");
    snprintf(pString,100,"Store regression matrix? (y or n) ");
    getString(pString, response);
    if (response[0] == 'y')
    {
      fp = fopen("pade_matrix.m", "w");
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
        fprintf(fp, "%16.6e \n",vecY[mm]);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "A = AA(:,1:%d);\n", N);
      fprintf(fp, "Y = AA(:,%d);\n", N+1);
      fprintf(fp, "B = A \\ Y;\n");
      fclose(fp);
      printf("Pade Regression matrix is now in pade_matrix.m\n");
    }
  }

  //**/ =================================================================
  //**/ perform SVD on A
  //**/ =================================================================
  psMatrix matU, matV;
  psVector vecS;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("Running SVD ...\n"); 
  int info = matA.computeSVD(matU, vecS, matV);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("SVD completed: status = %d (should be 0).\n",info); 

  if (info != 0)
  {
    printf("PadeRegression ERROR: dgesvd returns a nonzero (%d).\n",info);
    printf("PadeRegression terminates further processing.\n");
    printf("To diagnose problem, re-run with rs_expert on.\n");
    return -1;
  }

  //**/ =================================================================
  //**/ eliminate the noise components in S by zeroing small singular
  //**/ values
  //**/ =================================================================
  mm = 0;
  for (nn = 0; nn < N; nn++) if (vecS[nn] < 0) mm++;
  if (mm > 0)
  {
    printf("PadeRegression WARNING: Some singular values are < 0.\n");
    printf("               May spell trouble but will proceed anyway.\n");
    for (nn = 0; nn < N; nn++) if (vecS[nn] < 0) vecS[nn] = 0;
  }
  int NRevised;
  if (vecS[0] == 0.0) NRevised = 0;
  else
  {
    NRevised = N;
    for (nn = 1; nn < N; nn++) 
      if (vecS[nn-1] > 0 && vecS[nn]/vecS[nn-1] < 1.0e-8) break;
    NRevised = nn;
  }
  if (NRevised < N)
  {
    printf("PadeRegression WARNING: True matrix rank = %d (N=%d)\n",
           NRevised, N);
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    {
      printf("INFO: This can be due to the quality of the sample.\n");
      for (nn = 0; nn < N; nn++) 
        printf("Singular value %5d = %e\n",nn+1,vecS[nn]);
    }
  }
  if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("PadeRegression: For matrix singular values\n");
    printf(" - The VERY small ones may cause poor numerical accuracy\n");
    printf(" - But not keeping them may ruin the approximation power\n");
    printf(" - So select them judiously\n");
    for (nn = 0; nn < N; nn++) 
      printf("Singular value %5d = %e\n", nn+1, vecS[nn]);
    snprintf(pString,100,"How many to keep (1 - %d, 0 - all) ? ", N); 
    NRevised = getInt(0,N,pString);
    if (NRevised == 0) NRevised = N;
    for (nn = NRevised; nn < N; nn++) vecS[nn] = 0.0;
  }
  else
  {
    for (nn = NRevised; nn < N; nn++) vecS[nn] = 0.0;
    if (NRevised != N) 
      printf("PadeRegression INFO: %d singular values have been removed.\n",
             N-NRevised);
  }

  //**/ =================================================================
  //**/ coefficients B = V S^{-1} U^T * sq(W) Y
  //**/ =================================================================
  psVector vecW, vecB;
  vecW.setLength(M+N);
  double *UU = matU.getMatrix1D();
  for (mm = 0; mm < NRevised; mm++) 
  {
    vecW[mm] = 0.0;
    for (nn = 0; nn < M; nn++) 
      vecW[mm] += UU[nn+mm*M] * vecY[nn]; 
  }
  for (nn = 0; nn < NRevised; nn++) vecW[nn] /= vecS[nn];
  for (nn = NRevised; nn < N; nn++) vecW[nn] = 0.0;
  vecB.setLength(N);
  double *VV = matV.getMatrix1D();
  for (mm = 0; mm < N; mm++) 
  {
    vecB[mm] = 0.0;
    for (nn = 0; nn < N; nn++) vecB[mm] += VV[nn+mm*N] * vecW[nn]; 
  }

  //**/ =================================================================
  //**/ store eigenvectors VV and eigenvalues SS^2
  //**/ =================================================================
  psMatrix matEigT;;
  psVector vecEigvs;
  matEigT.load(N, N, VV);
  vecEigvs.load(N, vecS.getDVector());
  for (nn = 0; nn < N; nn++) vecEigvs[nn] = pow(vecEigvs[nn], 2.0);

  double esum=0, ymax=0;
  for (mm = 0; mm < M; mm++)
  {
    vecW[mm] = 0.0;
    for (nn = 0; nn < N; nn++) vecW[mm] += arrayXX[mm+nn*M] * vecB[nn];
    vecW[mm] -= vecY[mm];
    esum = esum + vecW[mm] * vecW[mm];
    if (PABS(vecY[mm]) > ymax) ymax = PABS(vecY[mm]);
  }
  esum /= (double) nSamples_;
  esum = sqrt(esum);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("PadeRegression: Interpolation rms error = %11.4e (Ymax=%9.2e)\n",
           esum, ymax); 

  //**/ =================================================================
  //**/ compute variance and R2 
  //**/ =================================================================
  double SSresid, SStotal, R2, var;
  computeSS(matXX, vecY, vecB, SSresid, SStotal);
  R2 = 1.0;
  if (SStotal != 0.0) R2  = 1.0 - SSresid / SStotal;
  if (nSamples_ > N) var = SSresid / (double) (nSamples_ - N);
  else               var = 0.0;
  if (var < 0)
  { 
    if (PABS(var) > 1.0e-12)
    {
      printf("PadeRegression WARNING: variance < 0.\n");
      printf("    Temporarily absolutize var (may have problems).\n");
      var = PABS(var);
    }
    else var = 0;
  }
  VecRegCoeffs_.load(vecB);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0)
  {
    printf("PadeRegression Coefficients: \n");
    for (ii = 0; ii < VecRegCoeffs_.length(); ii++)
      printf(" %5d   = %16.8e\n", ii+1, VecRegCoeffs_[ii]);
  }

  //**/ =================================================================
  //**/ find standard deviation of each coefficient 
  //**/ =================================================================
  computeCoeffVariance(matEigT, vecEigvs, var);
  psVector vecBstd;
  vecBstd.setLength(N);
  for (ii = 0; ii < N; ii++)
    vecBstd[ii] = sqrt(MatInvCov_.getEntry(ii,ii));

  //**/ =================================================================
  //**/ print out regression coefficients 
  //**/ =================================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
  {
    printf("PadeRegression R-squared = %10.3e (SSresid,SStotal=%8.1e,%8.1e)\n",
           R2, SSresid, SStotal);
    if ((M - N - 1) > 0)
      printf("* adjusted   R-squared = %10.3e\n",
             1.0 - (1.0 - R2) * ((M - 1) / (M - N - 1)));
    if (R2 < 0)
      printf("NOTE: Negative R2 ==> residual > total sum of square\n");
  }
  if (psConfig_.RSCodeGenIsOn()) genRSCode();
  return 0;
}

// *************************************************************************
// generate all combinations of a multivariate Legendre expansion
// This code is a direct translation from Burkardt's matlab code)
// -------------------------------------------------------------------------
int PadeRegression::genPermutations()
{
  //**/ construct permutations
  int  ii;
  numPermsP_ = 1;
  if (nInputs_ < pOrder_)
  {
    for (ii = nInputs_+pOrder_; ii > nInputs_; ii--)
      numPermsP_ = numPermsP_ * ii / (nInputs_+pOrder_-ii+1);
  }
  else
  {
    for (ii = nInputs_+pOrder_; ii > pOrder_; ii--)
      numPermsP_ = numPermsP_ * ii / (nInputs_+pOrder_-ii+1);
  }
  numPermsQ_ = 1;
  if (nInputs_ < qOrder_)
  {
    for (ii = nInputs_+qOrder_; ii > nInputs_; ii--)
      numPermsQ_ = numPermsQ_ * ii / (nInputs_+qOrder_-ii+1);
  }
  else
  {
    for (ii = nInputs_+qOrder_; ii > qOrder_; ii--)
      numPermsQ_ = numPermsQ_ * ii / (nInputs_+qOrder_-ii+1);
  }
  if (psConfig_.InteractiveIsOn())
  {
    printf("PadeRegression INFO: pOrder and pTerms = %d %d\n",
           pOrder_, numPermsP_);
    printf("                     qOrder and qTerms = %d %d\n",
           qOrder_, numPermsQ_);
    printf("NOTE: Total number of regression coefficients = %d\n",
           numPermsP_+numPermsQ_-1);
    printf("      (The constant term of Q(x) is assume to be 1)\n");
  }
  if (numPermsP_ + numPermsQ_ - 1 > nSamples_)
  {
    printf("PadeRegression ERROR: Insufficient sample size.\n");
    printf("  For p = %d and q = %d\n",pOrder_,qOrder_);
    printf("  Sample size must be >= %d\n",numPermsP_+numPermsQ_-1);
    exit(1);
  }
 
  //**/ =================================================================
  //**/ construct the permutations
  //**/ =================================================================
  MatPCEPerms_.setFormat(PS_MAT2D);
  int numPerms = numPermsP_, jj, kk, orderTmp, rvTmp;
  if (numPermsQ_ > numPermsP_) numPerms = numPermsQ_;
  MatPCEPerms_.setDim(numPerms, nInputs_);

  numPerms = 0;
  for (kk = 0; kk <= maxOrder_; kk++)
  {
    orderTmp = kk;
    rvTmp = 0;
    MatPCEPerms_.setEntry(numPerms, 0, orderTmp);
    for (ii = 1; ii < nInputs_; ii++) 
      MatPCEPerms_.setEntry(numPerms, ii, 0);
    while (MatPCEPerms_.getEntry(numPerms,nInputs_-1) != kk)
    {
      numPerms++;
      for (ii = 0; ii < nInputs_; ii++)
      {
        jj = MatPCEPerms_.getEntry(numPerms-1, ii);
        MatPCEPerms_.setEntry(numPerms, ii, jj);
      }
      if (orderTmp > 1) rvTmp = 1;
      else              rvTmp++;
      MatPCEPerms_.setEntry(numPerms, rvTmp-1, 0);
      orderTmp = MatPCEPerms_.getEntry(numPerms-1, rvTmp-1);
      MatPCEPerms_.setEntry(numPerms, 0, orderTmp-1);
      jj = MatPCEPerms_.getEntry(numPerms-1, rvTmp);
      MatPCEPerms_.setEntry(numPerms, rvTmp, jj+1);
    }
    numPerms++;
  }
  return 0;
}

// *************************************************************************
// load the X matrix
// -------------------------------------------------------------------------
int PadeRegression::loadXMatrix(psVector vecX,psVector vecY,psMatrix &matXX)
{
  int M = nSamples_;
  int N = numPermsP_ + numPermsQ_ - 1;
  psVector vecXX;
  vecXX.setLength(M*N);
  psMatrix MatLTable;
  MatLTable.setFormat(PS_MAT2D);
  MatLTable.setDim(nInputs_, pOrder_+1);
  double **LTable = MatLTable.getMatrix2D();
  int ii, ss, nn, orderT;
  double normalX, multiplier;

  //**/ X = [1 x_11 x_12 ... {np-th} (-y_1 x_11) (-y_1 x_12) ... {nq-th} 
  //**/      1 x_21 x_22 ... {np-th} (-y_2 x_21) (-y_2 x_22) ... {nq-th} 
  //**/      ...]
  //**/ for each row
  for (ss = 0; ss < nSamples_; ss++)
  {
    //**/ create input values at different polynomial orders
    //**/ and put into PTable (PTable[ii] has input ii+1)
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (normalizeFlag_ == 0) normalX = vecX[ss*nInputs_+ii];
      else
      {
        normalX = vecX[ss*nInputs_+ii] - VecLBs_[ii];
        normalX /= (VecUBs_[ii] - VecLBs_[ii]);
        normalX = normalX * 2.0 - 1.0;
      }
      evalLegendrePolynomials(normalX, LTable[ii]);
    }
    //**/ load the A matrix (vecXX) with p(X) polynomial values
    for (nn = 0; nn < numPermsP_; nn++)
    {
      multiplier = 1.0;
      for (ii = 0; ii < nInputs_; ii++)
        multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
      vecXX[nSamples_*nn+ss] = multiplier;
    }
    //**/ load the A matrix (vecXX) with q(X) polynomial values
    //**/ (note: the constant term is ignored)
    for (nn = 1; nn < numPermsQ_; nn++)
    {
      multiplier = 1.0;
      for (ii = 0; ii < nInputs_; ii++)
        multiplier *= LTable[ii][MatPCEPerms_.getEntry(nn,ii)];
      vecXX[nSamples_*(numPermsP_+nn-1)+ss] = -multiplier * vecY[ss];
    }
  }
  matXX.setFormat(PS_MAT1D);
  matXX.load(M, N, vecXX.getDVector());
  return N;
}

// *************************************************************************
// compute SS (sum of squares) statistics
// -------------------------------------------------------------------------
int PadeRegression::computeSS(psMatrix matXX, psVector vecY, psVector vecB, 
                              double &SSresid, double &SStotal)
{
  int    nn, mm, N;
  double rdata, ymean, SSreg, ddata, SSresidCheck, *arrayXX;

  N = vecB.length();
  arrayXX = matXX.getMatrix1D();
  SSresid = SSresidCheck = SStotal = SSreg = ymean = 0.0;
  for (mm = 0; mm < nSamples_; mm++) ymean += vecY[mm];
  ymean /= (double) nSamples_;
  for (mm = 0; mm < nSamples_; mm++)
  {
    ddata = 0.0;
    for (nn = 0; nn < N; nn++) ddata += (arrayXX[mm+nn*nSamples_]*vecB[nn]);
    rdata = vecY[mm] - ddata;
    SSresidCheck += rdata * rdata;
    SSresid += rdata * vecY[mm];
    SSreg += (ddata - ymean) * (ddata - ymean);
  }
  for (mm = 0; mm < nSamples_; mm++)
    SStotal += (vecY[mm] - ymean) * (vecY[mm] - ymean);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0)
  {
    printf("PadeRegression: SStot  = %24.16e\n", SStotal);
    printf("PadeRegression: SSreg  = %24.16e\n", SSreg);
    printf("PadeRegression: SSres  = %24.16e\n", SSresid);
    printf("PadeRegression: SSres  = %24.16e (true)\n", SSresidCheck);
  }
  SSresid = SSresidCheck;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0 && nSamples_ != N)
  {
    printf("* Regression: eps(Y) = %24.16e\n",
           SSresidCheck/(nSamples_-N));
  }

  //**/ ***********************************************************
  //**/ Old version based on the following formulas (but works)
  //**/ SSred = (Y - Xb)' W (Y - Xb) 
  //**/       = Y' W Y - 2 b 'X W 'Y + b' X' W X b
  //**/       = Y' W Y - 2 b' X' W Y + b' X' W Y  (since X'WXb=X'WY) 
  //**/       = Y' W Y - b' X' W Y = (Y - Xb)' W Y
  //**/ SStot = Y' W Y - N * (mean (W^(1/2) Y))^2
  //**/ ===========================================================
  //**/SSresid = SStotal = ymean = 0.0;
  //**/R = new double[nSamples_];
  //**/for (mm = 0; mm < nSamples_; mm++)
  //**/{
  //**/   R[mm] = Y[mm];
  //**/   for (nn = 0; nn < N; nn++) R[mm] -= (XX[mm+nn*nSamples_] * B[nn]);
  //**/   SSresid += R[mm] * Y[mm] * VecWghts_[mm];
  //**/   ymean += Y[mm);
  //**/}
  //**/ymean /= (double) nSamples_;
  //**/SStotal = - ymean * ymean * (double) nSamples_;
  //**/for (mm = 0; mm < nSamples_; mm++)
  //**/   SStotal += Y[mm] * Y[mm];
  //**/ ***********************************************************
  return 0;
}

// *************************************************************************
// compute coefficient variances (diagonal of sigma^2 (X' X)^(-1))
// -------------------------------------------------------------------------
int PadeRegression::computeCoeffVariance(psMatrix &matEigT,psVector 
                                         &vecEigvs, double var)
{

  //**/ =================================================================
  //**/ compute sigma^2 * V * D^{-1} 
  //**/ =================================================================
  int nRows = matEigT.nrows();
  psMatrix tMat;
  tMat.setDim(nRows, nRows);
  int    ii, jj;
  double invEig, dtmp;
  for (ii = 0; ii < nRows; ii++)
  {
    invEig = vecEigvs[ii];
    if (invEig != 0.0) invEig = 1.0 / invEig;
    for (jj = 0; jj < nRows; jj++)
    {
      dtmp = invEig * matEigT.getEntry(ii,jj) * var;
      tMat.setEntry(jj, ii, dtmp);
    }
  }
  //**/ =================================================================
  //**/ compute (sigma^2 * V * D^{-1}) V^T 
  //**/ =================================================================
  tMat.matmult(matEigT, MatInvCov_);
  return 0;
}

// *************************************************************************
// Purpose: evaluate 1D Legendre polynomials (normalized)
// -------------------------------------------------------------------------
int PadeRegression::evalLegendrePolynomials(double X, double *LTable)
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

// ************************************************************************
// generate C and Python codes
// ------------------------------------------------------------------------
void PadeRegression::genRSCode()
{
  int  ii, mm, nn;
  FILE *fp = fopen("psuade_rs.info", "w");

  if (fp != NULL)
  {
    fprintf(fp,"/* ***********************************************/\n");
    fprintf(fp,"/* Pade regression interpolator from PSUADE. */\n");
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
    fprintf(fp,"/* Pade regression interpolation function    */\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"static int\n"); 
    int numPerms = MatPCEPerms_.nrows();
    fprintf(fp,"pcePerms[%d][%d] = \n", numPerms, nInputs_);
    fprintf(fp,"{\n"); 
    for (mm = 0; mm < numPerms; mm++)
    {
       fprintf(fp,"  {"); 
       for (ii = 0; ii < nInputs_-1; ii++)
          fprintf(fp," %d,", MatPCEPerms_.getEntry(mm,ii)); 
       fprintf(fp," %d },\n", MatPCEPerms_.getEntry(mm,nInputs_-1)); 
    }
    fprintf(fp,"};\n"); 
    fprintf(fp,"static double\n"); 
    fprintf(fp,"invCovMat[%d][%d] = \n", numPerms, numPerms);
    fprintf(fp,"{\n"); 
    for (mm = 0; mm < numPerms; mm++)
    {
       fprintf(fp,"  {"); 
       for (ii = 0; ii < numPerms-1; ii++)
          fprintf(fp," %24.16e,", MatInvCov_.getEntry(mm,ii)); 
       fprintf(fp," %24.16e },\n", MatInvCov_.getEntry(mm,numPerms-1)); 
    }
    fprintf(fp,"};\n"); 
    fprintf(fp,"static double\n"); 
    fprintf(fp,"regCoefs[%d] = \n", numPerms);
    fprintf(fp,"{\n"); 
    for (mm = 0; mm < numPerms; mm++)
      fprintf(fp," %24.16e,", VecRegCoeffs_[mm]);
    fprintf(fp,"};\n"); 
    fprintf(fp,"/* ==============================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
    fprintf(fp,"  int    ii, kk, ss, nn;\n");
    fprintf(fp,"  double *x, y, **LTable, normX, mult;\n");
    fprintf(fp,"  double std, *x2, dtmp;\n");
    fprintf(fp,"  int numP = %d;\n",numPermsP_);
    fprintf(fp,"  LTable = (double **) malloc(%d * sizeof(double*));\n", 
               nInputs_);
    fprintf(fp,"  for (ii = 0; ii < %d; ii++)\n", nInputs_);
    fprintf(fp,"    LTable[ii] = (double *) malloc((%d+1)*sizeof(double));\n",
            pOrder_);
    fprintf(fp,"  x2 = (double *) malloc(%d * sizeof(double));\n",numPerms);
    fprintf(fp,"  for (ss = 0; ss < npts; ss++) {\n");
    fprintf(fp,"    x = &X[ss * %d];\n", nInputs_);
    fprintf(fp,"    numerator = 0.0;\n");
    fprintf(fp,"    for (nn = 0; nn < %d; nn++) {\n", numPerms);
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
    fprintf(fp,"      numerator += regCoefs[nn] * mult;\n");
    fprintf(fp,"      x2[nn] = mult;\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    denom = 1.0;\n");
    fprintf(fp,"    for (nn = 1; nn < %d; nn++) {\n", numPerms);
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
    fprintf(fp,"      denom += regCoefs[numP+nn-1] * mult;\n");
    fprintf(fp,"      x2[numP+nn-1] = mult;\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    if (denom == 0) Y[ss] = 0;\n");
    fprintf(fp,"    else Y[ss] = y * %e + %e;\n", YStd_, YMean_);
    fprintf(fp,"    for (ii = %d; ii < %d; ii++)\n",numPermsP_,numPerms);
    fprintf(fp,"      x2[ii] = - x2[ii] * y;\n");
    fprintf(fp,"    std = 0.0;\n");
    fprintf(fp,"    for (ii = 0; ii < %d; ii++) {\n",numPerms);
    fprintf(fp,"      dtmp = 0.0;\n");
    fprintf(fp,"      for (kk = 0; kk < %d; kk++)\n",numPerms);
    fprintf(fp,"        dtmp += invCovMat[ii][kk] * x2[kk];\n");
    fprintf(fp,"      std += dtmp * x2[ii];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    if (std >= 0) std = sqrt(std);\n");
    fprintf(fp,"    else          std = 0;\n");
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
    int order = pOrder_;
    if (qOrder_ > order) order = qOrder_;
    fprintf(fp,"  if (%d >= 1) {\n", order);
    fprintf(fp,"     LTable[1] = X;\n");
    fprintf(fp,"     for (ii = 2; ii <= %d; ii++)\n", order);
    fprintf(fp,"        LTable[ii] = ((2 * ii - 1) * X * LTable[ii-1] -\n");
    fprintf(fp,"                      (ii - 1) * LTable[ii-2]) / ii;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* ==============================================*/\n");
    fclose(fp);
    printf("FILE psuade_rs.info contains information about the ");
    printf("Pade interpolator.\n");
  }
}

