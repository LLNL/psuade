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
// Functions for the class MMars (for large data set)
// AUTHOR : CHARLES TONG
// DATE   : 2017
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "MMars.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "MainEffectAnalyzer.h"

// ************************************************************************
// Constructor for object class Multi-domain Mars
// ------------------------------------------------------------------------
MMars::MMars(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    idata, ii;
  double ddata;
  char   pString[501], *strPtr, equal[100], winput[5000];

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  boxes_ = NULL;
  partSize_ = 6000;

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*   Multi-Multivariate Regression (MMARS) Analysis\n");
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
    for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    printf("You can decide the sample size of each partition.\n");
    printf("Larger sample size per partition will take more setup time.\n");
    printf("The default is 1000 (will have more if there is overlap).\n");
    snprintf(pString,100,"Enter the partition sample size (1000 - 10000) : ");
    partSize_ = getInt(200, 20000, pString);
    snprintf(pString,100,"RS_MMARS_max_sam_per_group = %d", partSize_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RS_MMARS_overlap = %e", ddata);
    psConfig_.putParameter(pString);
  }

  //**/ =======================================================
  //**/ user adjustable parameters
  //**/ =======================================================
  else
  {
    strPtr = psConfig_.getParameter("RS_MMARS_max_sam_per_group");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, equal, &idata);
      if (idata >= 100) partSize_ = idata;
      else 
      {
        printf("MMars INFO: config parameter setting not done.\n");
        printf("            max_samples_per_group %d too small.\n",idata);
        printf("            max_samples_per_group should be >= 1000.\n");
      }
    }
    strPtr = psConfig_.getParameter("RS_MMARS_overlap");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &ddata);
      if (ddata >= 0) 
        for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    }
  }
  nPartitions_ = (nSamples + partSize_/2) / partSize_;
  ddata = log(1.0*nPartitions_) / log(2.0);
  idata = (int) ddata;
  //if (idata > nInputs_) idata = nInputs_;
  nPartitions_ = 1 << idata;
  if (psConfig_.InteractiveIsOn())
    printf("MMars: number of partitions = %d\n", nPartitions_);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MMars::~MMars()
{
  if (boxes_ != NULL)
  {
    for (int ii = 0; ii < nPartitions_; ii++) 
    {
      if (boxes_[ii] != NULL)
        if (boxes_[ii]->marsPtr_ != NULL) delete boxes_[ii]->marsPtr_;
    }
    delete [] boxes_;
  }
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int MMars::initialize(double *X, double *Y)
{
  int    ii, jj, ss, nSubs, index, incr, samCnt;
  double range, var=0,ddata, diff;
  char   pString[10000], winput[10000];
  psVector  vecVces, vecXT, vecYT;
  psIVector vecInds;

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  if (boxes_ != NULL)
  {
    for (ii = 0; ii < nPartitions_; ii++) 
    {
      if (boxes_[ii] != NULL)
        if (boxes_[ii]->marsPtr_ != NULL) delete boxes_[ii]->marsPtr_;
    }
    delete [] boxes_;
  }
  boxes_ = NULL;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printOutTS(PL_ERROR,
         "MMars initialize ERROR: sample bounds not set yet.\n");
    return -1;
  }

  //**/ ---------------------------------------------------------------
  //**/ divide up into sub-regions based on parameter sensitivities
  //**/ ---------------------------------------------------------------
  MainEffectAnalyzer *me = new MainEffectAnalyzer();
  psConfig_.AnaExpertModeSaveAndReset();
  turnPrintTSOff();
  vecVces.setLength(nInputs_);
  me->computeVCECrude(nInputs_,nSamples_,X,Y,VecLBs_.getDVector(),
                      VecUBs_.getDVector(),var,vecVces.getDVector());
  delete me;
  psConfig_.AnaExpertModeRestore();
  vecInds.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecInds[ii] = ii;
  ddata = 0;
  for (ii = 0; ii < nInputs_; ii++) ddata += vecVces[ii];
  for (ii = 0; ii < nInputs_; ii++) vecVces[ii] /= ddata;
  sortDbleList2a(nInputs_, vecVces.getDVector(), vecInds.getIVector());
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
  {
    for (ii = 0; ii < nInputs_; ii++) 
      printf("VCE %4d = %12.4e\n", vecInds[ii], vecVces[ii]);
  }

  //**/ special parameter settings
  int nBasis, maxVarPerBasis, normalizeY, localRSExpert=0;
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"MMars: Current number of basis functions = 100\n");
    nBasis = nSamples_;
    if (nSamples_ > 10)
    {
      snprintf(pString,100,"Enter the number of basis functions (>=10, <= %d): ",
               nSamples_);
      nBasis = getInt(10, nSamples_, pString);
    }
    maxVarPerBasis = 8;
    if (nInputs_ < maxVarPerBasis) maxVarPerBasis = nInputs_;
    printOutTS(PL_INFO,
         "MMars: Current degree of interactions    = %d\n",maxVarPerBasis);
    snprintf(pString,100,"Enter the degree of interactions (<=%d) : ",nInputs_);
    maxVarPerBasis = getInt(1, nInputs_, pString);
    snprintf(pString,100,"Mars: normalize output? (y or n) ");
    getString(pString, winput);
    normalizeY = 0;
    if (winput[0] == 'y') normalizeY = 1;
    snprintf(pString,100,"MARS_num_basis = %d", nBasis);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"MARS_interaction = %d", maxVarPerBasis);
    psConfig_.putParameter(pString);
    if (normalizeY == 1)
    {
      snprintf(pString,100,"normalize_outputs");
      psConfig_.putParameter(pString);
    }
    localRSExpert = 1;
    psConfig_.RSExpertModeSaveAndReset();
  }

  //**/ initialize boxes
  nSubs = int(log(1.0*nPartitions_ + 1.0e-9) / log(2.0));
  boxes_ = new MMars_Box*[nPartitions_];
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii] = new MMars_Box();
    boxes_[ii]->vecLBs_.setLength(nInputs_);
    boxes_[ii]->vecUBs_.setLength(nInputs_);
    boxes_[ii]->marsPtr_ = NULL;
    for (jj = 0; jj < nInputs_; jj++)
    {
      boxes_[ii]->vecLBs_[jj] = VecLBs_[jj];
      boxes_[ii]->vecUBs_[jj] = VecUBs_[jj];
    }
  }

  //**/ subdivide into boxes based on parameter sensitivities
  for (ii = 0; ii < nSubs; ii++)
  {
    index = vecInds[nInputs_-1]; 
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 0) 
    {
      printf("Selected input for partition %4dd = %4dd, VCE = %12.4e\n",
             ii+1, index+1, vecVces[nInputs_-1]);
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
    sortDbleList2a(nInputs_, vecVces.getDVector(), vecInds.getIVector());
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3)
  {
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 2)
  {
    //printf("MMars: Partition coverage check: %e (orig) ?= %e (sum)\n", 
    //       dcheck1, dcheck2);
    if (dcheck2 >= dcheck1) printf("MMars: Partition Coverage - good\n");
    else
      printf("MMars: potential problem with partition coverage.\n");
  }

  //**/ initialize Mars for each non-empty box
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("MMars training begins....\n");
  int total=0;
  double *lbs, *ubs;
  vecXT.setLength(nSamples_*nInputs_);
  vecYT.setLength(nSamples_);
  for (ii = 0; ii < nPartitions_; ii++)
  {
    lbs = boxes_[ii]->vecLBs_.getDVector();
    ubs = boxes_[ii]->vecUBs_.getDVector();
    samCnt = 0;
    for (ss = 0; ss < nSamples_; ss++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        diff = vecXd_[jj] * (ubs[jj] - lbs[jj]);
        ddata = X[ss*nInputs_+jj];
        if (ddata < lbs[jj]-diff || ddata > ubs[jj]+diff) break;
      } 
      if (jj == nInputs_)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXT[samCnt*nInputs_+jj] = X[ss*nInputs_+jj];
        vecYT[samCnt] = Y[ss];
        samCnt++;
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
      printf("Partition %d has %d sample points.\n",ii+1,samCnt);
    if (samCnt == 0)
    {
      printf("MMars INFO: some partition has no sample points.\n");
      boxes_[ii]->marsPtr_ = NULL;
    }
    else
    {
      boxes_[ii]->marsPtr_ = new Mars(nInputs_, samCnt);
      boxes_[ii]->marsPtr_->setBounds(boxes_[ii]->vecLBs_.getDVector(),
                                      boxes_[ii]->vecUBs_.getDVector());
      boxes_[ii]->marsPtr_->initialize(vecXT.getDVector(),
                                       vecYT.getDVector());
    }
    boxes_[ii]->nSamples_ = samCnt;
    total += samCnt;
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 0)
  {
    printf("Original sample size = %d\n",nSamples_);
    printf("Total sample sizes from all partitions = %d\n",total);
    printf("INFO: Total from all partitions may be larger than original\n");
    printf("      sample due to overlap. If the total is too large so\n");
    printf("      that it is close to the original size, partitioning\n");
    printf("      is not worthwhile -> you may want to reduce overlap.\n");
  }
  if (localRSExpert == 1)
  {
    // March 2019: These may not be compatible with CV if removed
    //psConfig_.removeParameter("MARS_num_basis");
    //psConfig_.removeParameter("MARS_interaction");
    //psConfig_.removeParameter("normalize_outputs");
    psConfig_.RSExpertModeRestore();
  }
  turnPrintTSOn();

  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
     printf("MMars training completed.\n");
  if (psConfig_.RSCodeGenIsOn())
    printf("MMars INFO: response surface stand-alone code not available.\n");

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MMars::genNDGridData(double *XIn,double *YIn,int *NOut,double **XOut,
                         double **YOut)
{
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ if requested not to create mesh, just return
  //**/ ---------------------------------------------------------------
  if ((*NOut) == -999) return 0;
 
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  int totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MMars::gen1DGridData(double *XIn,double *YIn,int ind1,double *settings, 
                         int *NOut, double **XOut, double **YOut)
{
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_;
  double HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  int ii, ss;
  psVector vecXT;
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) vecXT[ss*nInputs_+ii] = settings[ii]; 
   
  //**/ generate the data points 
  for (ss = 0; ss < totPts; ss++) 
  {
    vecXT[ss*nInputs_+ind1]  = HX * ss + VecLBs_[ind1];
    (*XOut)[ss] = HX * ss + VecLBs_[ind1];
    (*YOut)[ss] = 0.0;
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MMars::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                         double *settings, int *NOut, double **XOut, 
                         double **YOut)
{
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  int ii, ss, jj, index;
  psVector vecXT;
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      index = ii * nPtsPerDim_ + jj;
      vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MMars::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2,
                         int ind3, double *settings, int *NOut, 
                         double **XOut, double **YOut)
{
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  int ii, ss, jj, ll, index;
  psVector vecXT;
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii];

  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3]  = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MMars::gen4DGridData(double *XIn,double *YIn, int ind1, int ind2, 
                         int ind3, int ind4, double *settings, int *NOut, 
                         double **XOut, double **YOut)
{
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  int ii, ss, jj, ll, mm, index;
  psVector vecXT;
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        for (mm = 0; mm < nPtsPerDim_; mm++)
        {
          index = ii*nPtsPerDim_*nPtsPerDim_ * nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
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

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Evaluate a point
// ------------------------------------------------------------------------
double MMars::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y;
  evaluatePoint(1, X, &Y);
  return Y;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MMars::evaluatePoint(int nPts, double *X, double *Y)
{
  int    ss, pp, ii, nSamp, count, highFlag;
  double diff, Yt, ddata, *lbs, *ubs;

  for (ss = 0; ss < nPts; ss++) 
  {
    count = 0;
    //**/ look for a partition
    Yt = 0.0;
    for (pp = 0; pp < nPartitions_; pp++)
    {
      lbs = boxes_[pp]->vecLBs_.getDVector();
      ubs = boxes_[pp]->vecUBs_.getDVector();
      for (ii = 0; ii < nInputs_; ii++)
      {
        if (ubs[ii] == VecUBs_[ii]) highFlag = 1;
        else                        highFlag = 0;
        ddata = X[ss*nInputs_+ii];
        diff = vecXd_[ii] * (ubs[ii] - lbs[ii]);
        if (highFlag == 0)
        {
          if (ddata < (lbs[ii]-diff) || ddata >= (ubs[ii]+diff)) break;
        }
        else 
        {
          if (ddata < (lbs[ii]-diff) || ddata > (ubs[ii]+diff)) break;
        }
      } 
      if (ii == nInputs_ && boxes_[pp]->nSamples_ > 0)
      {
        if (boxes_[pp]->marsPtr_ != NULL)
        {
          Yt += boxes_[pp]->marsPtr_->evaluatePoint(&X[ss*nInputs_]);
          count++;
        }
      } 
    }
    if (count == 0)
    {
      printf("MMars evaluate WARNING: sample point not in any partition.\n");
      printf("   INFO: Evaluation will be performed by taking averarge\n");
      printf("         from all partitions (may be inaccurate).\n");
      Yt = 0;
      for (pp = 0; pp < nPartitions_; pp++)
      {
        if (boxes_[pp]->marsPtr_ != NULL)
        {
          Yt += boxes_[pp]->marsPtr_->evaluatePoint(&X[ss*nInputs_]);
          count++;
        }
      }
    }
    Y[ss] = Yt / (double) count;
  }
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double MMars::evaluatePointFuzzy(double *X, double &std)
{
  int    iOne=1;
  double Y=0.0;
  evaluatePoint(iOne, X, &Y);
  std = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double MMars::evaluatePointFuzzy(int npts,double *X,double *Y,double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ss = 0; ss < npts; ss++) Ystd[ss] = 0.0;
  return 0.0;
}

