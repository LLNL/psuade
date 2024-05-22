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
// Functions for the class MGP3
// AUTHOR : CHARLES TONG
// DATE   : 2017
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "GP3.h"
#include "HGP3.h"
#include "MGP3.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "Sampling.h"
#include "MainEffectAnalyzer.h"
#ifdef PSUADE_OMP
#include <omp.h>
#endif

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class MGP3
// ------------------------------------------------------------------------
MGP3::MGP3(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    ii, idata;
  double ddata;
  char   *strPtr, winput[1000], equal[100], pString[101];
  faID_ = PSUADE_RS_MGP3;

  //**/ =======================================================
  //**/ clean up previously-generated files
  //**/ =======================================================
  FILE *fp = fopen(".psuade_gp3", "r");
  if (fp != NULL)
  {
    printf("MGP3 INFO: Found parameter file for GP3 = .psuade_gp3\n");
    printf("           To avoid confusion, this file is to be deleted.\n"); 
    fclose(fp);
    unlink(".psuade_gp3");
  }

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  nPartitions_ = 0;
  boxes_ = NULL;
  partSize_ = 500;
  useHGP_ = 0;

  //**/ =======================================================
  // display banner and additional information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*   Multi-Gaussian Process (MGP3) Analysis\n");
    printOutTS(PL_INFO,"* Set printlevel to 1-4 to see details.\n");
    printOutTS(PL_INFO,"* Turn on rs_expert mode to make changes.\n");
    printEquals(PL_INFO, 0);
    if (nInputs_ > 30)
    {
      printf("Since nInputs > 30, you may want to consider using HGP3.\n");
      snprintf(pString,100,"Use HGP3? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y') useHGP_ = 1; 
    }
  }

  //**/ =======================================================
  // vecXd contains the amount of overlaps between partitions
  //**/ =======================================================
  vecXd_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = 0.05;
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("You can improve smoothness across partitions by allowing ");
    printf("overlaps.\n");
    printf("The recommended overlap is 0.05 (or 5%%).\n");
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
    printf("The default is 500 (will have more if there is overlap).\n");
    snprintf(pString,100,"Enter the partition sample size (500 - 10000) : ");
    partSize_ = getInt(200, 20000, pString);
    snprintf(pString,100,"RS_MGP_max_sam_per_group = %d", partSize_);
    psConfig_.putParameter(pString);
  }
  else
  {
    //**/ =======================================================
    // user-adjustable parameters
    //**/ =======================================================
    strPtr = psConfig_.getParameter("RS_MGP_max_sam_per_group");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, equal, &partSize_);
      if (partSize_ < 200) 
      {
        printf("MGP3 INFO: config parameter setting not done.\n");
        printf("     max_samples_per_group %d too small.\n",partSize_);
        printf("     max_samples_per_group should be >= 200.\n");
        partSize_ = 200;
      }
    }
  }
  nPartitions_ = (nSamples + partSize_/2) / partSize_;
  ddata = log(1.0*nPartitions_) / log(2.0);
  idata = (int) ddata;
  //if (idata > nInputs_) idata = nInputs_;
  nPartitions_ = 1 << idata;
  if (psConfig_.InteractiveIsOn())
    printf("MGP3: Number of partitions = %d (psize=%d)\n", nPartitions_,
           partSize_);

  //**/ now get user GP parameters
  nStarts_ = 2;
  expPower_ = 2.0;
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    printOutTS(PL_INFO,"MGP3 default exponential degree = %e\n",expPower_);
    snprintf(pString,100,"Set the exponential degree to ? [1.5,2.0] ");
    expPower_ = 1;
    while (expPower_ < 1.5 || expPower_ > 2)
      expPower_ = getDouble(pString);
    snprintf(pString,100,"RS_MGP3_power = %e", expPower_);
    psConfig_.putParameter(pString);
    if (useHGP_ == 0)
    {
      printOutTS(PL_INFO,
           "MGP3 default optimization nstarts = %d\n",nStarts_);
      snprintf(pString,100,"Set the number of optimization starts to ? [1 - 20] ");
      nStarts_ = getInt(1, 20, pString);
      printf("MGP3: Optimization num_starts = %d\n", nStarts_);
      snprintf(pString,100,"RS_MGP3_nstarts = %d", nStarts_);
      psConfig_.putParameter(pString);
    }
  }
  else
  {
    char winput1[1000], winput2[1000];
    char *cString = psConfig_.getParameter("RS_MGP3_power");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %lg", winput1, winput2, &expPower_);
      if (expPower_ < 1.5 || expPower_ > 2)
      {
        printOutTS(PL_INFO,"MGP3 exponential (%e) invalid - reset to 2\n",
                   expPower_);
        expPower_ = 2.0;
      }
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,"MGP3 exponential degree set to %e\n",expPower_);
    }
    cString = psConfig_.getParameter("RS_MGP3_nstarts");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nStarts_);
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,
           "MGP3: Number of optimization starts = %d\n",nStarts_);
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MGP3::~MGP3()
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
int MGP3::initialize(double *XIn, double *YIn)
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
      if (boxes_[ii]->rsPtr_ != NULL) delete boxes_[ii]->rsPtr_;
    delete [] boxes_;
  }
  boxes_ = NULL;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printOutTS(PL_ERROR,
         "MGP3 initialize ERROR: sample bounds not set yet.\n");
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
  MainEffectAnalyzer *me = new MainEffectAnalyzer();
  psConfig_.AnaExpertModeSaveAndReset();
  psConfig_.RSExpertModeSaveAndReset();
  psConfig_.InteractiveSaveAndReset();
  turnPrintTSOff();
  vecVces.setLength(nInputs_);
  me->computeVCECrude(nInputs_,nSamples_,XIn,YIn,VecLBs_.getDVector(),
                      VecUBs_.getDVector(), var, vecVces.getDVector());
  delete me;
  psConfig_.RSExpertModeRestore();
  psConfig_.AnaExpertModeRestore();
  psConfig_.InteractiveRestore();
  vecI.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecI[ii] = ii;
  ddata = 0;
  for (ii = 0; ii < nInputs_; ii++) ddata += vecVces[ii];
  for (ii = 0; ii < nInputs_; ii++) vecVces[ii] /= ddata;
  sortDbleList2a(nInputs_, vecVces.getDVector(), vecI.getIVector());
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
  {
    printf("MGP3: Run sensitivity analysis to determine partitioning.\n");
    for (ii = 0; ii < nInputs_; ii++)
      printf("  VCE %d = %e\n", vecI[ii]+1, vecVces[ii]);
  }

  //**/ actual division into boxes
  nSubs = int(log(1.0*nPartitions_ + 1.0e-9) / log(2.0));
  boxes_ = new MGP3_Box*[nPartitions_];
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii] = new MGP3_Box();
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
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 0)
    {
      if (outputLevel_ > 3)
      {
        for (jj = 0; jj < nInputs_; jj++)
          printf("vce %3d = %e\n", vecI[jj]+1, vecVces[jj]);
      }
      printf("MGP3: Input selected for binary partitioning = %d\n",
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
    //printf("MGP3: Partition Coverage check: %e (orig) ?= %e (sum)\n",
    //       dcheck1,dcheck2);
    if (dcheck2 >= dcheck1) printf("MGP3: Partition Coverage - good\n");
    else                    
      printf("MGP3: potential problem with partition coverage.\n");
  }

  //**/ count number of samples in each box 
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("MGP3 training begins....\n");
  int total=0;
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii]->rsPtr_ = NULL;
    double *lbs = boxes_[ii]->vecLBs_.getDVector();
    double *ubs = boxes_[ii]->vecUBs_.getDVector();

    //**/ count the number of samples in this box
    samCnt = 0;
    for (ss = 0; ss < nSamples_; ss++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        diff = vecXd_[jj] * (ubs[jj] - lbs[jj]);
        ddata = XData_[ss*nInputs_+jj];
        if (ddata < lbs[jj]-diff || ddata > ubs[jj]+diff) break;
      }
      if (jj == nInputs_) samCnt++;
    }
    boxes_[ii]->nSamples_ = samCnt;

    //**/ allocate memory
    if (samCnt > 0) 
    {
      boxes_[ii]->vecX_.setLength(samCnt*nInputs_);
      boxes_[ii]->vecY_.setLength(samCnt);
    }

    //**/ fill the vecX and vecY vectors
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
          boxes_[ii]->vecX_[samCnt*nInputs_+jj] = XData_[ss*nInputs_+jj];
        boxes_[ii]->vecY_[samCnt] = YData_[ss];
        samCnt++;
      }
    }
    total += samCnt;
    if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
    {
      printf("MGP3: Partition %d has %d sample points (ovlap=%e).\n",ii+1,
             samCnt, vecXd_[0]);
    }
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
  {
    printf("MGP3: Original sample size = %d\n",nSamples_);
    printf("MGP3: Total sample sizes from all partitions = %d\n",total);
    printf("INFO: Total from all partitions may be larger than ");
    printf("original sample\n");
    printf("      due to overlap. If the total is too large so ");
    printf("that it is close\n");
    printf("      to the original size, partitioning is not ");
    printf("worthwhile ==>\n");
    printf("      You may want to reduce overlap.\n");
  }

  //**/ turn these feature off to prevent too many interactive questions
  psConfig_.AnaExpertModeSaveAndReset();
  psConfig_.RSExpertModeSaveAndReset();
  psConfig_.InteractiveSaveAndReset();

  //**/ instantiate response surface for each partition
  char *tArgv[2], pString[1000], pString2[1000];
  tArgv[0] = (char *) pString;
  for (ii = 0; ii < nPartitions_; ii++)
  {
    if (boxes_[ii]->nSamples_ > 0)
    {
      if (useHGP_)
      { 
        boxes_[ii]->rsPtr_ = new HGP3(nInputs_, boxes_[ii]->nSamples_);
        strcpy(pString, "no_checking");
        boxes_[ii]->rsPtr_->setParams(1, tArgv);
      }
      else
      {
        boxes_[ii]->rsPtr_ = new GP3(nInputs_, boxes_[ii]->nSamples_);
        strcpy(pString, "hyperparam_file");
        snprintf(pString2,100,"psuade_gp3.%d", ii+1);
        tArgv[1] = (char *) pString2;
        boxes_[ii]->rsPtr_->setParams(2, tArgv);
      }
      boxes_[ii]->rsPtr_->setOutputLevel(0);
      boxes_[ii]->rsPtr_->setBounds(boxes_[ii]->vecLBs_.getDVector(),
                                    boxes_[ii]->vecUBs_.getDVector());
    }
  }

  //**/ restore settings
  psConfig_.RSExpertModeRestore();
  psConfig_.AnaExpertModeRestore();
  psConfig_.InteractiveRestore();

#pragma omp parallel private(ii)
{
#pragma omp for
  for (ii = 0; ii < nPartitions_; ii++)
  {
    if (boxes_[ii]->nSamples_ > 0)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
#ifdef PSUADE_OMP
        printf("MGP3: Processing partition %d (%d,mypid=%d)\n",ii+1,
               nPartitions_,omp_get_thread_num());
#else
        printf("MGP3: Processing partition %d (%d)\n",ii+1,nPartitions_);
#endif
      boxes_[ii]->rsPtr_->initialize(boxes_[ii]->vecX_.getDVector(),
                                     boxes_[ii]->vecY_.getDVector());
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
        printf("MGP3: Processing partition %d completed\n",ii+1);
    }
  }
}
  turnPrintTSOn();

  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("MGP3 training completed.\n");
  if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    printf("MGP3 INFO: response surface stand-alone code not available.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MGP3::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation begins....\n");
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int MGP3::gen1DGridData(double *XIn, double *YIn, int ind1,
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
  psVector vecXT, vecXOut;
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int MGP3::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
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
  int totPts = nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  int index;
  psVector vecXOut, vecXT;
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (int ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (int jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (int kk = 0; kk < nInputs_; kk++) 
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
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MGP3::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  int index;
  psVector vecXOut, vecXT;
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (int ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (int jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (int ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (int kk = 0; kk < nInputs_; kk++) 
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
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MGP3::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
                        int ind4,double *settings,int *NOut,double **XOut, 
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
  int totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  psVector vecHX;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_-1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_-1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_-1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_-1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  int index;
  psVector vecXOut, vecXT;
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (int ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (int jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (int ll = 0; ll < nPtsPerDim_; ll++) 
      {
        for (int mm = 0; mm < nPtsPerDim_; mm++) 
        {
          index = ii*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          for (int kk = 0; kk < nInputs_; kk++) 
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
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MGP3 interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MGP3::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double MGP3::evaluatePoint(int npts, double *X, double *Y)
{
  interpolate(npts, X, Y, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double MGP3::evaluatePointFuzzy(double *X, double &std)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, &std);
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double MGP3::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystds)
{
  interpolate(npts, X, Y, Ystds);
  return 0.0;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int MGP3::interpolate(int npts, double *X, double *Y, double *Ystds)
{
  int    ii, pp, ss, count, highFlag, hasStds=0;
  double Yt, ddata, diff, *lbs, *ubs;

  if (Ystds != NULL) hasStds = 1;
  for (ss = 0; ss < npts; ss++)
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
      if (ii == nInputs_)
      {
        if (hasStds) 
          Yt += boxes_[pp]->rsPtr_->evaluatePointFuzzy(&X[ss*nInputs_],
                                                       Ystds[ss]);
        else
          Yt += boxes_[pp]->rsPtr_->evaluatePoint(&X[ss*nInputs_]);
        count++;
      }
    }
    if (count == 0)
    {
      printf("MGP3 evaluate ERROR: sample point outside range.\n");
      for (ii = 0; ii < nInputs_; ii++)
        printf("Input %d = %e (in [%e, %e]?)\n",ii+1,
               X[ss*nInputs_+ii],VecLBs_[ii],VecUBs_[ii]);
      return 0.0;
    }
    Y[ss] = Yt / (double) count;
  }
  return 0;
}

