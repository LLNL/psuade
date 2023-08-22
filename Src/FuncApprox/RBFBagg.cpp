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
// Functions for the class RBFBagg
// AUTHOR : CHARLES TONG
// DATE   : 2015
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "RBFBagg.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class RBFBagg
// ------------------------------------------------------------------------
RBFBagg::RBFBagg(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int  ii;
  char pString[101];

  //**/ ==========================================================
  //**/ set internal parameters
  //**/ ==========================================================
  //**/ set identifier
  faID_ = PSUADE_RS_RBFB;

  //**/ default number of RBF instantiation
  numRBFs_ = 25;

  //**/ control the sample quality of each instantiation 
  usageIndex_ = 4;

  //**/ display configuration
  if (outputLevel_ > 1)
  {
    //**/ =======================================================
    //**/ set up mode and thetas
    //**/ =======================================================
    printAsterisks(PL_INFO, 0);
    printf("*                RBFBagg Analysis\n");
    printDashes(PL_INFO, 0);
    printf("Number of instantiations   = %d\n",numRBFs_);
    printEquals(PL_INFO, 0);
  }
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100, 
             "How many instantiation of RBF (10-5000, default=100) ? ");
    numRBFs_ = getInt(10, 5000, pString);
  }

  //**/ set number of instantiation 
  psConfig_.RSExpertModeSaveAndReset();
  psConfig_.InteractiveSaveAndReset();
  rbfObjs_ = new RBF*[numRBFs_];
  PsuadeConfig tmpConfig = psConfig_;
  psConfig_.reset();
  for (ii = 0; ii < numRBFs_; ii++) 
    rbfObjs_[ii] = new RBF(nInputs_, nSamples_);
  psConfig_ = tmpConfig;
  psConfig_.RSExpertModeRestore();
  psConfig_.InteractiveRestore();
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
RBFBagg::~RBFBagg()
{
  int ii;

  if (rbfObjs_ != NULL) 
  {
    for (ii = 0; ii < numRBFs_; ii++) delete rbfObjs_[ii];
    delete [] rbfObjs_;
  }
}

// ************************************************************************
// Set lower and upper bounds 
// ------------------------------------------------------------------------
int RBFBagg::setBounds( double *lower, double *upper )
{
  for (int ii=0 ; ii<numRBFs_; ii++) rbfObjs_[ii]->setBounds(lower, upper);
  return 0;
}

// ************************************************************************
// set number of points to generate in each dimension
// ------------------------------------------------------------------------
void RBFBagg::setNPtsPerDim(int npoints)
{
  nPtsPerDim_ = npoints;
  for (int ii=0 ; ii<numRBFs_; ii++) rbfObjs_[ii]->setNPtsPerDim(npoints);
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int RBFBagg::initialize(double *XX, double *Y)
{
  int  ii, ss, jj, index;
  FILE *fp;
  psVector  vecXB, vecYB;
  psIVector vecCnts;

  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  vecCnts.setLength(nSamples_);
  psConfig_.RSExpertModeSaveAndReset();
  for (ii = 0; ii < numRBFs_; ii++)
  {
    if (outputLevel_ >= 2)
      printf("RBFBagg::initialize : creating RBF #%d (of %d)\n",
             ii+1, numRBFs_);
    if (MatDataSetX_.nrows() == 0)
    {
      for (ss = 0; ss < nSamples_; ss++) vecCnts[ss] = usageIndex_ * 2;
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = -1;
        while (index == -1)
        {
          index = PSUADE_rand() % nSamples_;
          if (vecCnts[index] > 0 && (vecCnts[index] % usageIndex_) == 0)
          {
            vecCnts[index]--;
          }
          else if (vecCnts[index] > 0 && (vecCnts[index] % usageIndex_) != 0)
          {
            vecCnts[index]--;
            index = -1;
          }
          else if (vecCnts[index] <= 0) index = -1;
        }
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XX[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
      index = 0;
      for (ss = 0; ss < nSamples_; ss++) 
        if (vecCnts[ss] < usageIndex_*2) index++;
      if (outputLevel_ >= 2)
        printf("     Number of sample points used = %d (out of %d)\n",
               index,nSamples_);
    }
    else
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
        vecYB[ss] = MatDataSetY_.getEntry(ii,ss); 
      }
    }
    rbfObjs_[ii]->setOutputLevel(0);
    jj = 0;
    for (ss = 0; ss < nSamples_; ss++)
      if (vecYB[ss] >= PSUADE_UNDEFINED) jj++;
    if (jj > 0)
    {
      printf("RBFBagg ERROR: some of the sample outputs are undefined.\n");
      exit(1);
    }
    rbfObjs_[ii]->initialize(vecXB.getDVector(), vecYB.getDVector());
    fp = fopen("psuade_print", "r");
    if (fp != NULL)
    {
      printf("RBFBagg: set print level to 2\n");
      outputLevel_ = 2;
      fclose(fp);
    }
  }
  psConfig_.RSExpertModeRestore();
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBFBagg::genNDGridData(double *XX, double *Y, int *N, double **XX2, 
                           double **Y2)
{
  int    totPts, ii, ss, jj, index;
  double *XXt, *Yt;
  FILE   *fp;
  psVector  vecXB, vecYB;
  psIVector vecCnts;

  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  if ((*N) == -999)
  {
    vecCnts.setLength(nSamples_);
    psConfig_.RSExpertModeSaveAndReset();
    for (ii = 0; ii < numRBFs_; ii++)
    {
      if (outputLevel_ >= 2)
        printf("RBFBagg::genNDGridData : creating RBF_ #%d (of %d)\n",
               ii+1, numRBFs_);
      if (MatDataSetX_.nrows() == 0)
      {
        for (ss = 0; ss < nSamples_; ss++) vecCnts[ss] = usageIndex_ * 2;
        for (ss = 0; ss < nSamples_; ss++)
        {
          index = -1;
          while (index == -1)
          {
            index = PSUADE_rand() % nSamples_;
            if (vecCnts[index] > 0 && (vecCnts[index] % usageIndex_) == 0)
            {
              vecCnts[index]--;
            }
            else if (vecCnts[index] > 0 && 
                     (vecCnts[index] % usageIndex_) != 0)
            {
              vecCnts[index]--;
              index = -1;
            }
            else if (vecCnts[index] <= 0) index = -1;
          }
          for (jj = 0; jj < nInputs_; jj++)
            vecXB[ss*nInputs_+jj] = XX[index*nInputs_+jj]; 
          vecYB[ss] = Y[index]; 
        }
        index = 0;
        for (ss = 0; ss < nSamples_; ss++) 
          if (vecCnts[ss] < usageIndex_*2) index++;
        if (outputLevel_ >= 2)
          printf("     Number of sample points used = %d (out of %d)\n",
                 index,nSamples_);
      }
      else
      {
        for (ss = 0; ss < nSamples_; ss++)
        {
          for (jj = 0; jj < nInputs_; jj++)
            vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
          vecYB[ss] = MatDataSetY_.getEntry(ii, ss); 
        }
      }
      rbfObjs_[ii]->setOutputLevel(0);
      rbfObjs_[ii]->genNDGridData(vecXB.getDVector(), vecYB.getDVector(), 
                                  N, NULL, NULL);
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        printf("RBFBagg: set print level to 2\n");
        outputLevel_ = 2;
        fclose(fp);
      }
    }
    psConfig_.RSExpertModeRestore();
  }
  else
  {
    psConfig_.RSExpertModeSaveAndReset();
    //**/ set up for generating regular grid data
    totPts = nPtsPerDim_;
    for (ii = 1; ii < nInputs_; ii++) totPts = totPts * nPtsPerDim_;
    psVector vecXOut, vecYOut;
    vecXOut.setLength(nInputs_ * totPts);
    vecYOut.setLength(totPts);
    (*XX2) = vecXOut.takeDVector();
    (*Y2)  = vecYOut.takeDVector();
    for (ii = 0; ii < totPts; ii++) (*Y2)[ii] = 0.0;

    //**/ generate response surface and interpolate
    for (ii = 0; ii < numRBFs_; ii++)
    {
      //**/ generate bootstrap sample 
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XX[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
      rbfObjs_[ii]->genNDGridData(vecXB.getDVector(), vecYB.getDVector(), 
                                  N, &XXt, &Yt);

      //**/ compute mean
      for (ss = 0; ss < totPts; ss++) (*Y2)[ss] += Yt[ss];

      //**/ store inputs
      if (ii == numRBFs_-1)
         for (ss = 0; ss < totPts*nInputs_; ss++) (*XX2)[ss] = XXt[ss];
      delete [] XXt;
      delete [] Yt;
    }
    (*N) = totPts;

    for (ss = 0; ss < totPts; ss++) (*Y2)[ss] /= (double) numRBFs_;
    psConfig_.RSExpertModeRestore();
  }
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBFBagg::gen1DGridData(double *X, double *Y, int ind1, 
                            double *settings, int *N, double **XX, 
                            double **YY)
{
  int    totPts, ii, ss, jj, index;
  double *XXt, *Yt;
  psVector  vecXB, vecYB, vecXOut, vecYOut;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_;
  (*N)   = totPts;
  vecXOut.setLength(totPts);
  vecYOut.setLength(totPts);
  (*XX)  = vecXOut.takeDVector();
  (*YY)  = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YY)[ii] = 0.0;

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numRBFs_; ii++)
  {
    if (MatDataSetX_.nrows() == 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = X[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
    }
    else
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
        vecYB[ss] = MatDataSetY_.getEntry(ii,ss); 
      }
    }
    rbfObjs_[ii]->gen1DGridData(vecXB.getDVector(), vecYB.getDVector(), 
                                ind1, settings, N, &XXt, &Yt);

    for (ss = 0; ss < totPts; ss++) (*YY)[ss] += Yt[ss];
    if (ii == numRBFs_-1)
      for (ss = 0; ss < totPts; ss++) (*XX)[ss] = XXt[ss];
    delete [] XXt;
    delete [] Yt;
  }
  for (ss = 0; ss < totPts; ss++) (*YY)[ss] /= (double) numRBFs_;
  psConfig_.RSExpertModeRestore();
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBFBagg::gen2DGridData(double *X, double *Y, int ind1, int ind2, 
                           double *settings, int *N, double **XX, 
                           double **YY)
{
  int    totPts, ii, ss, jj, index;
  double *XXt, *Yt;
  psVector vecXB, vecYB, vecXOut, vecYOut;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_;
  (*N)   = totPts;
  vecXOut.setLength(totPts*2);
  (*XX)  = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YY)  = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YY)[ii] = 0.0;

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numRBFs_; ii++)
  {
    if (outputLevel_ >= 1)
      printf("RBFBagg::gen2DGridData : creating RBF #%d (of %d)\n",
             ii+1, numRBFs_);
    if (MatDataSetX_.nrows() == 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = X[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
    }
    else
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
        vecYB[ss] = MatDataSetY_.getEntry(ii,ss); 
      }
    }
    rbfObjs_[ii]->gen2DGridData(vecXB.getDVector(), vecYB.getDVector(), 
                       ind1, ind2, settings, N, &XXt, &Yt);

    //**/ collect outputs
    for (ss = 0; ss < totPts; ss++) (*YY)[ss] += Yt[ss];

    //**/ collect inputs
    if (ii == numRBFs_-1)
      for (ss = 0; ss < 2*totPts; ss++) (*XX)[ss] = XXt[ss];
    delete [] XXt;
    delete [] Yt;
  }
  for (ss = 0; ss < totPts; ss++) (*YY)[ss] /= (double) numRBFs_;
  psConfig_.RSExpertModeRestore();
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int RBFBagg::gen3DGridData(double *X, double *Y, int ind1, int ind2, 
                           int ind3, double *settings, int *N, double **XX, 
                           double **YY)
{
  int    totPts, ii, ss, jj, index;
  double *XXt, *Yt;
  psVector vecXB, vecYB, vecXOut, vecYOut;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*N)   = totPts;
  vecXOut.setLength(totPts*3);
  (*XX)  = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YY)  = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YY)[ii] = 0.0;

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numRBFs_; ii++)
  {
    if (MatDataSetX_.nrows() == 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = X[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
    }
    else
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
        vecYB[ss] = MatDataSetY_.getEntry(ii,ss); 
      }
    }
    rbfObjs_[ii]->gen3DGridData(vecXB.getDVector(), vecYB.getDVector(), 
                         ind1, ind2, ind3, settings, N, &XXt, &Yt);

    //**/ collect outputs
    for (ss = 0; ss < totPts; ss++) (*YY)[ss] += Yt[ss];

    //**/ collect inputs
    if (ii == numRBFs_-1)
      for (ss = 0; ss < 3*totPts; ss++) (*XX)[ss] = XXt[ss];
    delete [] XXt;
    delete [] Yt;
  }

  //**/ process outputs based on the mode
  for (ss = 0; ss < totPts; ss++) (*YY)[ss] /= (double) numRBFs_;

  psConfig_.RSExpertModeRestore();
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int RBFBagg::gen4DGridData(double *X, double *Y, int ind1, int ind2, 
                            int ind3, int ind4, double *settings, int *N, 
                            double **XX, double **YY)
{
  int    totPts, ii, ss, jj, index;
  double *XXt, *Yt;
  psVector vecXB, vecYB, vecXOut, vecYOut;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*N)   = totPts;
  vecXOut.setLength(totPts*4);
  (*XX)  = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YY)  = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YY)[ii] = 0.0;

  for (ii = 0; ii < numRBFs_; ii++)
  {
    if (MatDataSetX_.nrows() == 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = X[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
    }
    else
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = MatDataSetX_.getEntry(ii,ss*nInputs_+jj); 
        vecYB[ss] = MatDataSetY_.getEntry(ii,ss); 
      }
    }
    rbfObjs_[ii]->gen4DGridData(vecXB.getDVector(), vecYB.getDVector(), 
                       ind1, ind2, ind3, ind4, settings, N, &XXt, &Yt);

    for (ss = 0; ss < totPts; ss++) (*YY)[ss] += Yt[ss];

    //**/ collect inputs
    if (ii == numRBFs_-1)
      for (ss = 0; ss < 4*totPts; ss++) (*XX)[ss] = XXt[ss];
    delete [] XXt;
    delete [] Yt;
  }

  //**/ process outputs 
  for (ss = 0; ss < totPts; ss++) (*YY)[ss] /= (double) numRBFs_;
  psConfig_.RSExpertModeRestore();
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double RBFBagg::evaluatePoint(double *X)
{
  int    ii;
  double Yt, Y=0.0;

  //**/ interpolate
  for (ii = 0; ii < numRBFs_; ii++) 
  {
    Yt = rbfObjs_[ii]->evaluatePoint(X);
    Y += Yt;
  }
  Y /= (double) numRBFs_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double RBFBagg::evaluatePoint(int npts, double *X, double *Y)
{
  int    in, ii;
  double YY, Yt;

  //**/ interpolate
  for (in = 0; in < npts; in++) 
  {
    YY = 0.0;
    for (ii = 0; ii < numRBFs_; ii++) 
    {
      Yt = rbfObjs_[ii]->evaluatePoint(&(X[in*nInputs_]));
      YY += Yt;
    }
    Y[in] = YY / (double) numRBFs_;
  }
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double RBFBagg::evaluatePointFuzzy(double *X, double &std)
{
  int    ii;
  double Ymean=0.0;
  psVector vecYT;

  vecYT.setLength(numRBFs_);
  for (ii = 0; ii < numRBFs_; ii++) 
  {
    vecYT[ii] = rbfObjs_[ii]->evaluatePoint(X);
    Ymean += vecYT[ii];
  }
  Ymean /= (double) numRBFs_;
  std = 0.0;
  for (ii = 0; ii < numRBFs_; ii++) 
    std += (vecYT[ii] - Ymean) * (vecYT[ii] - Ymean);
  std = sqrt(std / (double) numRBFs_);
  return Ymean;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double RBFBagg::evaluatePointFuzzy(int npts, double *X, double *Y,
                                   double *Ystd)
{
  int in, ii;
  psVector vecYT;

  vecYT.setLength(numRBFs_);
  for (in = 0; in < npts; in++) 
  {
    for (ii = 0; ii < numRBFs_; ii++) 
      vecYT[ii] = rbfObjs_[ii]->evaluatePoint(&(X[in*nInputs_]));
    Y[in] = 0.0;
    for (ii = 0; ii < numRBFs_; ii++) Y[in] += vecYT[ii];
    Y[in] /= (double) numRBFs_;
    Ystd[in] = 0.0;
    for (ii = 0; ii < numRBFs_; ii++) 
      Ystd[in] += (vecYT[ii] - Y[in]) * (vecYT[ii] - Y[in]);
    Ystd[in] = sqrt(Ystd[in] / (double) numRBFs_);
  }
  return 0.0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double RBFBagg::setParams(int targc, char **targv)
{
  int  ii, itmp, leng;
  char cString[500], *argv[3];

  if (targc == 2 && !strcmp(targv[0], "num_rbf"))
  {
    if (rbfObjs_ != NULL) 
    {
      for (ii = 0; ii < numRBFs_; ii++) delete rbfObjs_[ii];
      delete [] rbfObjs_;
    }
    numRBFs_ = *(int *) targv[1];
    if (numRBFs_ < 2) numRBFs_ = 2;
    printf("RBF with bagging: no. of RBF set to = %d.\n", numRBFs_);
    psConfig_.RSExpertModeSaveAndReset();
    rbfObjs_ = new RBF*[numRBFs_];
    for (ii = 0; ii < numRBFs_; ii++) 
      rbfObjs_[ii] = new RBF(nInputs_, nSamples_);
    psConfig_.RSExpertModeRestore();
  }
  else if (targc == 5 && !strcmp(targv[0], "rbf_sample"))
  {
    itmp = *(int *) targv[1];
    if (itmp < 0 || itmp >= numRBFs_)
    {
      printf("RBFBagg ERROR: in loading sample - invalid index.\n");
      exit(1);
    }
    leng = *(int *) targv[2];
    if (leng != nSamples_)
    {
      printf("RBFBagg ERROR: in loading sample - nSamples mismatch.\n");
      exit(1);
    }
    double *Xdata = (double *) targv[3];
    double *Ydata = (double *) targv[4];
    if (MatDataSetX_.nrows() == 0)
    {
      MatDataSetX_.setFormat(PS_MAT2D);
      MatDataSetX_.setDim(numRBFs_, nSamples_*nInputs_);
      MatDataSetY_.setFormat(PS_MAT2D);
      MatDataSetY_.setDim(numRBFs_, nSamples_);
    }
    for (ii = 0; ii < leng*nInputs_; ii++) 
      MatDataSetX_.setEntry(itmp, ii, Xdata[ii]);
    for (ii = 0; ii < leng; ii++) 
      MatDataSetY_.setEntry(itmp, ii, Ydata[ii]);
  }
  else
  {
    printf("RBFBagg setParams ERROR: invalid command %s.\n", targv[0]);
  }
  return 0.0;
}

