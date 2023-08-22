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
// Functions for the class RBF
// AUTHOR : CHARLES TONG
// DATE   : 2017
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "MRBF.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "MainEffectAnalyzer.h"
#include "PrintingTS.h"

extern "C" {
#if 0
   void dgetrf_(int *, int *, double *, int *, int *, int *);
   void dgetrs_(char *,int *,int*,double*,int*,int*,double*,int*,int*);
#endif
   void dgesvd_(char *, char *, int *, int *, double *, int *, double *,
               double *, int *, double *, int *, double *, int *, int *);
}

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class RBF
// ------------------------------------------------------------------------
MRBF::MRBF(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    idata, ii;
  double ddata;
  char   pString[101], *strPtr, equal[100], winput[5000];

  //**/ =======================================================
  //**/ set identifier
  //**/ 0:  multiquadratic, 1: inverse multiquadratic, 
  //**/ 2:  Gaussian, 3: thin plate spline
  //**/ =======================================================
  type_ = 0;

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  svdThresh_ = 1e-15;
  gaussScale_ = 1;
  boxes_ = NULL;
  partSize_ = 500;

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*   Multiple Radial Basis Function (RBF) Analysis\n");
    printOutTS(PL_INFO,"* Set printlevel to 1-4 to see RBF details.\n");
    printOutTS(PL_INFO,"* Default kernel    = multi-quadratic \n");
    printOutTS(PL_INFO,
         "* Default threshold = 1.0e-15 (for SVD truncation)\n");
    printOutTS(PL_INFO,"* Turn on rs_expert mode to make changes.\n");
    printEquals(PL_INFO, 0);
  }
  
  //**/ =======================================================
  //**/ user adjustable parameters
  //**/ =======================================================
  vecXd_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = 0.05;
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("In the following you have the option to select the kernel. \n");
    printf("0. multi-quadratic\n");
    printf("1. inverse multi-quadratic\n");
    printf("2. Gaussian\n");
    printf("3. thin plate spline\n");
    snprintf(pString,100,"Enter your choice (0 - 3) : ");
    type_ = getInt(0, 3, pString);
    if (type_ == 2)
    {
      snprintf(pString,100,
         "Enter scaling factor for Gaussian kernel (default=1) : ");
      gaussScale_ = getDouble(pString);
    }
    printOutTS(PL_INFO,
         "The RBF matrix to be constructed may be near-singular.\n");
    printOutTS(PL_INFO,
         "Currently, singular values < max(svd)*1e-15 are truncated.\n");
    printOutTS(PL_INFO,
         "You have the option to change this threshold (1e-15).\n");
    printOutTS(PL_INFO,
         "NOTE: truncating singular values may lead to erroneous results.\n");
    snprintf(pString,100,"Enter new threshold for SVD (> 0 but << 1) : ");
    svdThresh_ = getDouble(pString);

    printf("You can improve smoothness across partitions by allowing\n");
    printf("overlaps. The recommended overlap is 0.1 (or 10%%).\n");
    snprintf(pString,100,"Enter the degree of overlap (0 - 0.4) : ");
    ddata = getDouble(pString);
    if (ddata < 0 || ddata > 0.4)
    {
      ddata = 0.1;
      printf("ERROR: Degree of overlap should be > 0 and <= 0.4.\n");
      printf("INFO:  Degree of overlap set to default = 0.05\n");
    }
    for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    printf("You can decide the sample size of each partition.\n");
    printf("Larger sample size per partition will take more setup time.\n");
    printf("The default is 500 (will have more if there is overlap).\n");
    snprintf(pString,100,"Enter the partition sample size (500 - 2000) : ");
    partSize_ = getInt(500, 20000, pString);
   
    snprintf(pString,100,"RBF_kernel = %d", type_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RBF_scale = %e", gaussScale_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RBF_thresh = %e", svdThresh_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"MRBF_max_samples_per_group = %d", partSize_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"MRBF_overlap = %e", vecXd_[0]);
    psConfig_.putParameter(pString);
  }
  else
  {
    strPtr = psConfig_.getParameter("RBF_kernel");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, equal, &type_);
      if (type_ < 0 || type_ > 3) type_ = 0;
    }

    strPtr = psConfig_.getParameter("RBF_scale");
    if (strPtr != NULL)
      sscanf(strPtr, "%s %s %lg", winput, equal, &gaussScale_);

    strPtr = psConfig_.getParameter("RBF_thresh");
    if (strPtr != NULL)
      sscanf(strPtr, "%s %s %lg", winput, equal, &svdThresh_);

    strPtr = psConfig_.getParameter("MRBF_max_samples_per_group");
    if (strPtr != NULL)
      sscanf(strPtr, "%s %s %d", winput, equal, &partSize_);

    strPtr = psConfig_.getParameter("MRBF_overlap");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &ddata);
      for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    }
  }

  //**/ =======================================================
  //**/ user adjustable parameters
  //**/ =======================================================
  nPartitions_ = (nSamples + partSize_/2) / partSize_;
  ddata = log(1.0*nPartitions_) / log(2.0);
  idata = (int) ddata;
  //if (idata > nInputs_) idata = nInputs_;
  nPartitions_ = 1 << idata;
  if (psConfig_.InteractiveIsOn())
    printf("MRBF: number of partitions = %d\n", nPartitions_);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MRBF::~MRBF()
{
  if (boxes_ != NULL) delete [] boxes_;
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int MRBF::initialize(double *X, double *Y)
{
  int    ii, jj, ss, nSubs, index, incr, samCnt;
  double range, var=0,ddata, diff;
  psVector  vecVces, vecXT;
  psIVector vecInds;

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  if (boxes_ != NULL) delete [] boxes_;
  boxes_ = NULL;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printOutTS(PL_ERROR,
         "MRBF initialize ERROR: sample bounds not set yet.\n");
    return -1;
  }

  //**/ ---------------------------------------------------------------
  //**/ divide up into sub-regions
  //**/ ---------------------------------------------------------------
  MainEffectAnalyzer *me = new MainEffectAnalyzer();
  turnPrintTSOff();
  vecVces.setLength(nInputs_);
  me->computeVCECrude(nInputs_,nSamples_,X,Y,VecLBs_.getDVector(),
                      VecUBs_.getDVector(), var, vecVces.getDVector());
  delete me;
  vecInds.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecInds[ii] = ii;
  sortDbleList2a(nInputs_, vecVces.getDVector(), vecInds.getIVector());
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    for (ii = 0; ii < nInputs_; ii++) 
      printf("VCE %d = %e\n", vecInds[ii], vecVces[ii]);
  }

  //**/ ---------------------------------------------------------------
  //**/ initialize boxes
  //**/ ---------------------------------------------------------------
  nSubs = int(log(1.0*nPartitions_ + 1.0e-9) / log(2.0));
  boxes_ = new MRBF_Box*[nPartitions_];
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii] = new MRBF_Box();
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
    index = vecInds[nInputs_-1]; 
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 0) 
    {
      for (jj = 0; jj < nInputs_; jj++)
        printf("vce %3d = %e\n", vecInds[jj]+1, vecVces[jj]);
      printf("Selected input for partition = %d\n", index+1);
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

  //**/ ---------------------------------------------------------------
  //**/ check coverage
  //**/ ---------------------------------------------------------------
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
    //printf("MRBF: Partition coverage check: %e (orig) ?= %e (sum)\n", 
    //       dcheck1, dcheck2);
    if (dcheck2 >= dcheck1) printf("MRBF: Partition Coverage - good\n");
    else
      printf("MRBF: potential problem with partition coverage.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ initialize for each partition
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("MRBF training begins....\n");
  int total=0;
  double *lbs, *ubs, *coefs;
  for (ii = 0; ii < nPartitions_; ii++)
  {
    lbs = boxes_[ii]->vecLBs_.getDVector();
    ubs = boxes_[ii]->vecUBs_.getDVector();
    vecXT.setLength(nSamples_*nInputs_);
    //**/ count the number of samples in this partition
    samCnt = 0;
    for (ss = 0; ss < nSamples_; ss++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        diff = vecXd_[jj] * (ubs[jj] - lbs[jj]);
        ddata = X[ss*nInputs_+jj];
        if (ddata < lbs[jj]-diff || ddata > ubs[jj]+diff) break;
      } 
      if (jj == nInputs_) samCnt++;
    }
    boxes_[ii]->nSamples_ = samCnt;
    //**/ allocate space to store the samples
    if (samCnt > 0)
    {
      boxes_[ii]->vecX_.setLength(samCnt*nInputs_);
      boxes_[ii]->vecY_.setLength(samCnt);
    }
    //**/ store the samples
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
          boxes_[ii]->vecX_[samCnt*nInputs_+jj] = X[ss*nInputs_+jj];
        boxes_[ii]->vecY_[samCnt] = Y[ss];
        samCnt++;
      }
    }
    total += samCnt;
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 0) 
      printf("Partition %d has %d sample points.\n",ii+1,samCnt);
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

  //**/ now set up RBF
  psConfig_.RSExpertModeSaveAndReset();

#pragma omp parallel private(ii)
{
#pragma omp for
  for (ii = 0; ii < nPartitions_; ii++)
  {
    if (boxes_[ii]->nSamples_ > 0)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
        printf("MRBF: processing partition %d (%d)\n",ii+1,nPartitions_);
      boxes_[ii]->rsPtr_ = new RBF(nInputs_, boxes_[ii]->nSamples_);
      boxes_[ii]->rsPtr_->setOutputLevel(0);
      boxes_[ii]->rsPtr_->setBounds(boxes_[ii]->vecLBs_.getDVector(),
                                    boxes_[ii]->vecUBs_.getDVector());
      boxes_[ii]->rsPtr_->initialize(boxes_[ii]->vecX_.getDVector(),
                                     boxes_[ii]->vecY_.getDVector());
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
        printf("MRBF: processing partition %d completed\n",ii+1);
    }
  }
}
  psConfig_.RSExpertModeRestore();
  turnPrintTSOn();

  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("MRBF training completed.\n");
  if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    printf("MRBF INFO: response surface stand-alone code not available.\n");

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MRBF::genNDGridData(double *XIn,double *YIn,int *NOut,double **XOut,
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
int MRBF::gen1DGridData(double *XIn,double *YIn,int ind1,double *settings, 
                        int *NOut, double **XOut, double **YOut)
{
  int ii, ss, ndim=1;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  double HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  int totPts = nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ---------------------------------------------------------------
  //**/ allocate local storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXT, VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
   
  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  for (ss = 0; ss < totPts; ss++) 
  {
    vecXT[ss*nInputs_+ind1]  = HX * ss + VecLBs_[ind1];
    (*XOut)[ss] = HX * ss + VecLBs_[ind1];
    (*YOut)[ss] = 0.0;
  }

  //**/ ---------------------------------------------------------------
  //**/ evaluate 
  //**/ ---------------------------------------------------------------
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MRBF::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                        double *settings, int *NOut, double **XOut, 
                        double **YOut)
{
  int ii, ss, jj, index, ndim=2;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  psVector vecHX;
  vecHX.setLength(ndim);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ---------------------------------------------------------------
  //**/ allocate local storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXT, VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
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

  //**/ ---------------------------------------------------------------
  //**/ evaluate 
  //**/ ---------------------------------------------------------------
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MRBF::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                        int ind3, double *settings, int *NOut, 
                        double **XOut, double **YOut)
{
  int    ii, ss, jj, ll, index, ndim=3;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  psVector vecHX;
  vecHX.setLength(ndim);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ---------------------------------------------------------------
  //**/ allocate local storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXT, VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii];

  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ evaluate 
  //**/ ---------------------------------------------------------------
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MRBF::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                        int ind3, int ind4, double *settings, int *NOut, 
                        double **XOut, double **YOut)
{
  int ii, ss, jj, ll, mm, index, ndim=4;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  psVector vecHX;
  vecHX.setLength(ndim);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_-1); 
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ---------------------------------------------------------------
  //**/ allocate local storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXT, VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
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
          vecXT[index*nInputs_+ind1] = vecHX[0]*ii+VecLBs_[ind1];
          vecXT[index*nInputs_+ind2] = vecHX[1]*jj+VecLBs_[ind2];
          vecXT[index*nInputs_+ind3] = vecHX[2]*ll+VecLBs_[ind3];
          vecXT[index*nInputs_+ind4] = vecHX[3]*mm+VecLBs_[ind4];
          (*XOut)[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ evaluate 
  //**/ ---------------------------------------------------------------
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Evaluate a point
// ------------------------------------------------------------------------
double MRBF::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y;
  evaluatePoint(1, X, &Y);
  return Y;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MRBF::evaluatePoint(int nPts, double *X, double *Y)
{
  int    ss, ii, pp, count, highFlag;
  double diff, Yt, ddata, *lbs, *ubs;
  psVector vecRanges;

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
        Yt += boxes_[pp]->rsPtr_->evaluatePoint(&X[ss*nInputs_]);
        count++;
      }
    }
    if (count == 0)
    {
      printf("MRBF evaluate ERROR: sample point outside range.\n");
      printf("INFO: this may happen during cross validation.\n"); 
      for (ii = 0; ii < nInputs_; ii++)
        printf("Input %d = %e (in [%e, %e]?)\n",ii+1,
           X[ss*nInputs_+ii],VecLBs_[ii],VecUBs_[ii]);
      printf("Sample prediction will be set to Ymean: may be wrong.\n");
      Yt = 0;
      count = 1;
    }
    Y[ss] = Yt / (double) count;
  }
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double MRBF::evaluatePointFuzzy(double *X, double &std)
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
double MRBF::evaluatePointFuzzy(int npts,double *X,double *Y,double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ss = 0; ss < npts; ss++) Ystd[ss] = 0.0;
  return 0.0;
}

