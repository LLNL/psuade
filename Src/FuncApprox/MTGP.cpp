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
// Functions for the class MTGP
// AUTHOR : CHARLES TONG
// DATE   : 2018
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "TBGP.h"
#include "MTGP.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "Sampling.h"
#include "MainEffectAnalyzer.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class MTGP
// ------------------------------------------------------------------------
MTGP::MTGP(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    ii, idata;
  double ddata;
  char   *strPtr, winput[1000], equal[100], pString[101];
  faID_ = PSUADE_RS_MTGP;
  nPartitions_ = 0;

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  boxes_ = NULL;
  partSize_ = 300;

  //**/ =======================================================
  // display banner and additional information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*   Multi-Treed Gaussian Process (MTGP) Analysis\n");
    printOutTS(PL_INFO,"* Set printlevel to 1-4 to see details.\n");
    printOutTS(PL_INFO,"* Turn on rs_expert mode to make changes.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ =======================================================
  // vecXd contains the amount of overlaps between partitions
  //**/ =======================================================
  vecXd_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = 0.05;
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("You can improve smoothness across partitions by allowing\n");
    printf("overlaps. The recommended overlap is 0.1 (or 10%%).\n");
    snprintf(pString,100,"Enter the degree of overlap (0 - 0.4) : ");
    ddata = getDouble(pString);
    if (ddata < 0 || ddata > 0.4)
    {
      ddata = 0.05;
      printf("ERROR: Degree of overlap should be > 0 and <= 0.4.\n");
      printf("INFO:  Degree of overlap set to default = 0.05\n");
    }
    snprintf(pString,100,"RS_MTGP_overlap = %e",ddata);
    psConfig_.putParameter(pString);
    for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    printf("You can decide the sample size of each partition.\n");
    printf("Larger sample size per partition will take more setup time.\n");
    printf("The default is 300 (will have more if there is overlap).\n");
    snprintf(pString,100,"Enter the partition sample size (300 - 1000) : ");
    partSize_ = getInt(100, 3000, pString);
    snprintf(pString,100,"RS_MTGP_max_sam_per_group = %d",partSize_);
    psConfig_.putParameter(pString);
  }
  else
  {
    //**/ =======================================================
    // user-adjustable parameters
    //**/ =======================================================
    strPtr = psConfig_.getParameter("RS_MTGP_max_sam_per_group");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, equal, &partSize_);
      if (partSize_ < 100) 
      {
        printf("MTGP INFO: config parameter setting not done.\n");
        printf("     max_samples_per_group %d too small.\n",partSize_);
        printf("     max_samples_per_group should be >= 100.\n");
        partSize_ = 100;
      }
      printf("MTGP maximum number of samples per subdomain = %d\n",
             partSize_); 
    }
    strPtr = psConfig_.getParameter("RS_MTGP_overlap");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &ddata);
      printf("MTGP subdomain degree of overlap = %e\n",ddata); 
      for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    }
  }

  nPartitions_ = (nSamples + partSize_/2) / partSize_;
  ddata = log(1.0*nPartitions_) / log(2.0);
  idata = (int) ddata;
  //if (idata > nInputs_) idata = nInputs_;
  nPartitions_ = 1 << idata;
  printf("MTGP: number of partitions = %d (psize=%d)\n", nPartitions_,
         partSize_);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MTGP::~MTGP()
{
  if (boxes_ != NULL)
  {
    for (int ii = 0; ii < nPartitions_; ii++)
      if (boxes_[ii]->rsPtr_   != NULL) delete boxes_[ii]->rsPtr_;
    delete [] boxes_;
  }
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int MTGP::initialize(double *XIn, double *YIn)
{
  int    ii, jj, ss, incr, nSubs, index, samCnt;
  double diff, var, ddata;
  psVector  vecVces, vecXX, vecYY;
  psIVector vecI;

  //**/ ----------------------------------------------------------------
  //**/ clean up
  //**/ ----------------------------------------------------------------
  if (boxes_ != NULL)
  {
    for (ii = 0; ii < nPartitions_; ii++)
      if (boxes_[ii]->rsPtr_   != NULL) delete boxes_[ii]->rsPtr_;
    delete [] boxes_;
  }
  boxes_ = NULL;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printOutTS(PL_ERROR,
         "MTGP initialize ERROR: sample bounds not set yet.\n");
    return -1;
  }

  //**/ ----------------------------------------------------------------
  //**/ normalize sample values
  //**/ ----------------------------------------------------------------
  XData_.setLength(nSamples_*nInputs_);
  for (ii = 0; ii < nSamples_*nInputs_; ii++) XData_[ii] = XIn[ii];
  YData_.setLength(nSamples_);
  for (ii = 0; ii < nSamples_; ii++) YData_[ii] = YIn[ii];
   
  //**/ ----------------------------------------------------------------
  //**/ divide up into sub-regions
  //**/ ----------------------------------------------------------------
  //**/ first use main effect analysis to guide which input to divide
  if (outputLevel_ > 1)
  {
    printf("Step 1: use main effect analysis to guide partitioning.\n");
  }
  MainEffectAnalyzer *me = new MainEffectAnalyzer();
  psConfig_.AnaExpertModeSaveAndReset();
  psConfig_.RSExpertModeSaveAndReset();
  turnPrintTSOff();
  vecVces.setLength(nInputs_);
  me->computeVCECrude(nInputs_,nSamples_,XIn,YIn,VecLBs_.getDVector(),
                      VecUBs_.getDVector(), var, vecVces.getDVector());
  delete me;
  psConfig_.RSExpertModeRestore();
  psConfig_.AnaExpertModeRestore();
  vecI.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecI[ii] = ii;
  ddata = 0;
  for (ii = 0; ii < nInputs_; ii++) ddata += vecVces[ii];
  for (ii = 0; ii < nInputs_; ii++) vecVces[ii] /= ddata;
  sortDbleList2a(nInputs_, vecVces.getDVector(), vecI.getIVector());
  //if (outputLevel_ > 1)
  //{
  //  for (ii = 0; ii < nInputs_; ii++)
  //    printf("  VCE %d = %e\n", vecI[ii], vecVces[ii]);
  //}

  //**/ actual division into boxes
  nSubs = int(log(1.0*nPartitions_ + 1.0e-9) / log(2.0));
  boxes_ = new MTGP_Box*[nPartitions_];
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii] = new MTGP_Box();
    boxes_[ii]->vecLBs_.setLength(nInputs_);
    boxes_[ii]->vecUBs_.setLength(nInputs_);
    boxes_[ii]->rsPtr_ = NULL;
    for (jj = 0; jj < nInputs_; jj++)
    {
      boxes_[ii]->vecLBs_[jj] = VecLBs_[jj];
      boxes_[ii]->vecUBs_[jj] = VecUBs_[jj];
    }
  }
  for (ii = 0; ii < nSubs; ii++)
  {
    index = vecI[nInputs_-1];
    if (outputLevel_ > 1)
    {
      printf("Partitioning step #%d: \n", ii+1);
      for (jj = 0; jj < nInputs_; jj++)
        printf("  VCE %3d = %e\n", vecI[jj]+1, vecVces[jj]);
      printf("Selected input for binary partitioning %d = %d\n",ii+1, 
             index+1);
    }
    incr = 1 << (nSubs - ii - 1);
    for (jj = 0; jj < nPartitions_; jj++)
    {
      if (((jj / incr) % 2) == 0)
      {
        boxes_[jj]->vecUBs_[index] = 0.5 *
          (boxes_[jj]->vecUBs_[index] + boxes_[jj]->vecLBs_[index]);
      }
      else
      {
        boxes_[jj]->vecLBs_[index] = 0.5 *
          (boxes_[jj]->vecUBs_[index] + boxes_[jj]->vecLBs_[index]);
      }
    }
    vecVces[nInputs_-1] *= 0.25;
    sortDbleList2a(nInputs_, vecVces.getDVector(), vecI.getIVector());
  }
  if (outputLevel_ > 3)
  {
    printf("Paritioning information:\n");
    for (jj = 0; jj < nPartitions_; jj++)
    {
      printf("Partition %d:\n", jj);
        for (ii = 0; ii < nInputs_; ii++)
          printf("Input %2d = %12.4e %12.4e\n",ii+1,
            boxes_[jj]->vecLBs_[ii], boxes_[jj]->vecUBs_[ii]);
    }
  }
  //**/ check coverage
  double dcheck1=0, dcheck2=0;
  for (jj = 0; jj < nPartitions_; jj++)
  {
    ddata = 1.0;
    for (ii = 0; ii < nInputs_; ii++)
      ddata *= (boxes_[jj]->vecUBs_[ii]-boxes_[jj]->vecLBs_[ii]);
    dcheck2 += ddata;
  }
  dcheck1 = 1;
  for (ii = 0; ii < nInputs_; ii++)
    dcheck1 *= (VecUBs_[ii] - VecLBs_[ii]);
  if (outputLevel_ > 2)
  {
    //printf("MTGP: Partition coverage check: %e (orig) ?= %e (sum)\n", 
    //       dcheck1, dcheck2);
    if (dcheck2 >= dcheck1) printf("MTGP: Partition Coverage - good\n");
    else
      printf("MTGP: potential problem with partition coverage.\n");
  }

  //**/ put sample points into boxes and initialize
  if (outputLevel_ > 1) printf("MTGP training begins....\n");
  int total=0;
  vecXX.setLength(nSamples_*nInputs_);
  vecYY.setLength(nSamples_);
  psConfig_.RSExpertModeSaveAndReset();
  for (ii = 0; ii < nPartitions_; ii++)
  {
    double *lbs = boxes_[ii]->vecLBs_.getDVector();
    double *ubs = boxes_[ii]->vecUBs_.getDVector();
    samCnt = 0;
    for (ss = 0; ss < nSamples_; ss++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        diff = vecXd_[jj] * (ubs[jj] - lbs[jj]);
        ddata = XData_[ss*nInputs_+jj];
        if (ddata < lbs[jj]-diff || ddata > ubs[jj]+diff) break;
      }
      if (jj == nInputs_)
      {
        for (jj = 0; jj < nInputs_; jj++)
           vecXX[samCnt*nInputs_+jj] = XData_[ss*nInputs_+jj];
        vecYY[samCnt] = YData_[ss];
        samCnt++;
      }
    }
    if (outputLevel_ > 0)
    {
      printf("Partition %d has %d sample points (ovlap=%e).\n",ii+1,
             samCnt, vecXd_[0]);
    }
    if (samCnt == 0)
    {
      printf("MTGP INFO: some partition has no sample points.\n");
      boxes_[ii]->rsPtr_ = NULL;
    }
    else
    {
      boxes_[ii]->rsPtr_ = new TGP(nInputs_, samCnt);
      boxes_[ii]->rsPtr_->setOutputLevel(0);
      boxes_[ii]->rsPtr_->setBounds(boxes_[ii]->vecLBs_.getDVector(),
                                    boxes_[ii]->vecUBs_.getDVector());
      boxes_[ii]->rsPtr_->initialize(vecXX.getDVector(), 
                                     vecYY.getDVector());
    }
    boxes_[ii]->nSamples_ = samCnt;
    total += samCnt;
  }
  psConfig_.RSExpertModeRestore();
  if (outputLevel_ >= 0)
  {
    printf("Sample size = %d\n",nSamples_);
    printf("Total sample sizes from all partitions = %d\n",total);
    printf("INFO: Total from all partitions may be larger than original\n");
    printf("      sample due to overlap. If the total is too large so\n");
    printf("      that it is close to the original size, partitioning\n");
    printf("      is not worthwhile -> you may want to reduce overlap.\n");
  }
  turnPrintTSOn();

  if (outputLevel_ > 1) printf("MTGP training completed.\n");
  if (psConfig_.RSCodeGenIsOn())
    printf("MTGP INFO: response surface stand-alone code not available.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MTGP::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
                        double **YOut)
{
  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  int totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("MTGP interpolation begins....\n");
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);
  if (outputLevel_ >= 1) printf("MTGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int MTGP::gen1DGridData(double *XIn, double *YIn, int ind1,
                        double *settings, int *NOut, double **XOut, 
                        double **YOut)
{
  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  int totPts = nPtsPerDim_;
  double HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector vecXOut, vecXT;
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (int ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (int kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
   
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("MTGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (outputLevel_ >= 1) printf("MTGP interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int MTGP::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                        double *settings, int *NOut, double **XOut, 
                        double **YOut)
{
  int ii, jj, kk, index;
  psVector vecXT, vecHX, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(2*totPts);
  (*XOut) = vecXOut.takeDVector();
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
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("MTGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (outputLevel_ >= 1) printf("MTGP interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MTGP::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
                        double *settings, int *NOut, double **XOut, 
                        double **YOut)
{
  int ii, jj, kk, ll, index;
  psVector vecXT, vecHX, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(3*totPts);
  (*XOut) = vecXOut.takeDVector();
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
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("MTGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (outputLevel_ >= 1) printf("MTGP interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MTGP::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
                        int ind4,double *settings,int *NOut,double **XOut, 
                        double **YOut)
{
  int totPts, ii, jj, kk, ll, mm, index;
  psVector vecXT, vecHX, vecXOut, vecYOut;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
 
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_-1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  vecXT.setLength(totPts*nInputs_);
  vecXOut.setLength(4*totPts);
  (*XOut) = vecXOut.takeDVector();
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
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("MTGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (outputLevel_ >= 1) printf("MTGP interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MTGP::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double MTGP::evaluatePoint(int npts, double *X, double *Y)
{
  interpolate(npts, X, Y, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double MTGP::evaluatePointFuzzy(double *X, double &std)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, &std);
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double MTGP::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystds)
{
  interpolate(npts, X, Y, Ystds);
  return 0.0;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int MTGP::interpolate(int npts, double *X, double *Y, double *Ystds)
{
  int    ii, pp, ss, count, highFlag, hasStds=0;
  double Yt, ddata, diff, *lbs, *ubs;
  psVector  vecXT, vecYT, vecST;
  psIVector vecCnts, vecInds;

  //**/ initialize
  if (Ystds != NULL) hasStds = 1;
  vecXT.setLength(npts*nInputs_);
  vecYT.setLength(npts*nInputs_);
  vecST.setLength(npts*nInputs_);
  vecCnts.setLength(npts*nInputs_);
  vecInds.setLength(npts*nInputs_);
  for (ss = 0; ss < npts; ss++) Y[ss] = 0.0;
  if (Ystds != NULL) for (ss = 0; ss < npts; ss++) Ystds[ss] = 0.0;

  //**/ traverse all partitions for evaluation
  for (pp = 0; pp < nPartitions_; pp++)
  {
    lbs = boxes_[pp]->vecLBs_.getDVector();
    ubs = boxes_[pp]->vecUBs_.getDVector();
    count = 0;
    //**/ gather all points that are in this partition
    for (ss = 0; ss < npts; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        if (ubs[ii] == VecUBs_[ii]) highFlag = 1;
        else                        highFlag = 0;
        ddata = X[ss*nInputs_+ii];
        diff = 0.05 * (ubs[ii] - lbs[ii]);
        if (highFlag == 0)
        {
          if (ddata < (lbs[ii]-diff) || ddata >= (ubs[ii]+diff)) break;
        }
        else
        {
          if (ddata < (lbs[ii]-diff) || ddata > (ubs[ii]+diff)) break;
        }
      }
      //**/ if inside the boundaries of the present partition
      if (ii == nInputs_)
      {
        for (ii = 0; ii < nInputs_; ii++) 
          vecXT[count*nInputs_+ii] = X[ss*nInputs_+ii];
        vecCnts[ss] = vecCnts[ss] + 1;
        vecInds[count] = ss;
        count++;
      }
    }
    //**/ evaluate all points in this partition
    if (count > 0)
    {
      if (hasStds) 
        boxes_[pp]->rsPtr_->evaluatePointFuzzy(count,vecXT.getDVector(),
                                 vecYT.getDVector(), vecST.getDVector());
      else
        boxes_[pp]->rsPtr_->evaluatePoint(count,vecXT.getDVector(),
                                          vecYT.getDVector());
    }
    //**/ put the output back to original location
    for (ss = 0; ss < count; ss++) Y[vecInds[ss]] += vecYT[ss];
    if (Ystds != NULL)
      for (ss = 0; ss < count; ss++) 
        Ystds[vecInds[ss]] += vecST[ss] * vecST[ss];
  }
  //**/ final step: taking average for points in overlapped regions
  for (ss = 0; ss < npts; ss++) 
  {
    if (vecCnts[ss] == 0) 
      printf("MTGP interpolate ERROR: point not in any partition.\n");
    else
    {
      Y[ss] /= (double) vecCnts[ss];
      if (Ystds != NULL)
        Ystds[ss] = sqrt(Ystds[ss]/(double) vecCnts[ss]);
    }
  }
  return 0;
}

