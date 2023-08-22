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
// Functions for the class MDNN
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "MDNN.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "Sampling.h"
#include "MainEffectAnalyzer.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class MDNN
// ------------------------------------------------------------------------
MDNN::MDNN(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    ii;
  double ddata;
  char   winput[1000], equal[100], pString[1000];

  faID_ = PSUADE_RS_MDNN;

  //**/ =======================================================
  //**/ clean up previously-generated files
  //**/ =======================================================
  FILE *fp = fopen(".psuade_dnn", "r");
  if (fp != NULL)
  {
    printf("MDNN INFO: Found parameter file for DNN = .psuade_dnn\n");
    printf("           To avoid confusion, this file is to be deleted.\n"); 
    fclose(fp);
    unlink(".psuade_dnn");
  }

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  nPartitions_ = 0;
  boxes_ = NULL;
  partSize_ = 500;

  //**/ =======================================================
  // display banner and additional information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*   Multi-Domain Neural Network (NN) Analysis\n");
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
    printf("overlaps. The recommended overlap is 0.05 (or 5%%).\n");
    snprintf(pString,100,"Enter the degree of overlap (0 - 0.4) : ");
    ddata = getDouble(pString);
    if (ddata < 0 || ddata > 0.4)
    {
      ddata = 0.05;
      printf("ERROR: Degree of overlap should be > 0 and <= 0.4.\n");
      printf("INFO:  Degree of overlap set to default = 0.05\n");
    }
    for (ii = 0; ii < nInputs_; ii++) vecXd_[ii] = ddata;
    printf("You can decide the average sample size of each partition.\n");
    printf("Larger sample size per partition will take more setup time.\n");
    printf("The default is 1000 (will have more if there is overlap).\n");
    snprintf(pString,100,"Enter the partition sample size (1000 - 10000) : ");
    partSize_ = getInt(200, 20000, pString);
  }

  //**/ =======================================================
  // user-adjustable parameters
  //**/ =======================================================
  char *strPtr = psConfig_.getParameter("MDNN_max_samples_per_group");
  if (strPtr != NULL)
  {
    sscanf(strPtr, "%s %s %d", winput, equal, &partSize_);
    if (partSize_ < 200) 
    {
      printf("MDNN INFO: config parameter setting not done.\n");
      printf("     max_samples_per_group %d too small.\n",partSize_);
      printf("     max_samples_per_group should be >= 200.\n");
      partSize_ = 200;
    }
  }
  nPartitions_ = (nSamples + partSize_/2) / partSize_;
  ddata = log(1.0*nPartitions_) / log(2.0);
  int idata = (int) ddata;
  //if (idata > nInputs_) idata = nInputs_;
  nPartitions_ = 1 << idata;
  if (psConfig_.InteractiveIsOn())
    printf("MDNN: Number of partitions = %d (psize=%d)\n", 
           nPartitions_, partSize_);

  //**/ ----------------------------------------------------------------
  //**/ get DNN parameters
  //**/ ----------------------------------------------------------------
  nLevels_   = 1;
  numNodes_  = 50;
  optScheme_ = 0;
  maxIter_   = 10000;
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    snprintf(pString,100,"How many hidden layers? (1 - 4) ");
    nLevels_  = getInt(1,4,pString) + 1;
    snprintf(pString,100,"DNN_nlevels = %d", nLevels_);
    psConfig_.putParameter(pString);

    snprintf(pString,100,"Number of nodes in each hidden layer? (3 - 100) ");
    numNodes_ = getInt(3,100,pString);
    snprintf(pString,100,"DNN_nnodes = %d", numNodes_);
    psConfig_.putParameter(pString);

    snprintf(pString,100,
      "Optimization scheme (0: LBFGS, 1: Gradient descent) ? (0 - 1) ");
    optScheme_ = getInt(0,1,pString);
    snprintf(pString,100,"DNN_optscheme = %d", optScheme_);
    psConfig_.putParameter(pString);

    //snprintf(pString,100,
    //  "Maximum iteration for optimization (100 - 10000) : ");
    //maxIter_ = getInt(100,10000,pString);
  }
  else
  {
    char winput1[1000], winput2[1000];
    char *cString = psConfig_.getParameter("DNN_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels_);
      if (nLevels_ < 1 || nLevels_ > 4)
      {
        printOutTS(PL_INFO,"NN nlevels %d invalid - reset to 1\n",nLevels_);
        nLevels_ = 1;
      }
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,"NN nLevels set to %d\n",nLevels_);
    }
    cString = psConfig_.getParameter("DNN_nnodes");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &numNodes_);
      if (numNodes_ < 3 || numNodes_ > 100)
      {
        printOutTS(PL_INFO,"NN numNodes/level %d invalid - reset to 10\n",
                   numNodes_);
        numNodes_ = 10;
      }
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,"NN numNodes set to %d\n",numNodes_);
    }
    cString = psConfig_.getParameter("DNN_optscheme");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &optScheme_);
      if (optScheme_ < 0 || optScheme_ > 1)
      {
        printOutTS(PL_INFO,"NN Opt scheme invalid - reset to LBFGS\n");
        optScheme_ = 0;
      }
      if (psConfig_.InteractiveIsOn())
      {
        if (optScheme_ == 0)
          printOutTS(PL_INFO,"NN Opt scheme set to LBFGS\n");
        else
          printOutTS(PL_INFO,"NN Opt scheme set to Gradient descent\n");
      }
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MDNN::~MDNN()
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
int MDNN::initialize(double *XIn, double *YIn)
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
         "MDNN initialize ERROR: sample bounds not set yet.\n");
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
    for (ii = 0; ii < nInputs_; ii++)
      printf("VCE %d = %e\n", vecI[ii], vecVces[ii]);
  }

  //**/ actual division into boxes
  nSubs = int(log(1.0*nPartitions_ + 1.0e-9) / log(2.0));
  boxes_ = new MDNN_Box*[nPartitions_];
  for (ii = 0; ii < nPartitions_; ii++)
  {
    boxes_[ii] = new MDNN_Box();
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
          printf("MDNN: VCE %3d = %e\n", vecI[jj]+1, vecVces[jj]);
      }
      printf("* Selected input for binary partitioning %d = %d\n",ii+1, 
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
      printf("* Partition %d:\n", jj);
        for (ii = 0; ii < nInputs_; ii++)
          printf("  Input %2d = %12.4e %12.4e\n",ii+1,
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
    if (dcheck2 >= dcheck1) printf("MDNN: Partition Coverage - good\n");
    else                    
      printf("MDNN: potential problem with partition coverage.\n");
  }

  //**/ count number of samples in each box 
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("MDNN training begins....\n");
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
      printf("* Partition %d has %d sample points (ovlap=%e).\n",ii+1,
             samCnt, vecXd_[0]);
    }
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 0)
  {
    printf("Sample size = %d\n",nSamples_);
    printf("Total sample sizes from all partitions = %d\n",total);
    printf("INFO: Total from all partitions may be larger than original\n");
    printf("      sample due to overlap. If the total is too large so\n");
    printf("      that it is close to the original size, partitioning\n");
    printf("      is not worthwhile -> you may want to reduce overlap.\n");
  }

  //**/ now set up DNN for each partition
  psConfig_.AnaExpertModeSaveAndReset();
  psConfig_.RSExpertModeSaveAndReset();
  psConfig_.InteractiveSaveAndReset();
  for (ii = 0; ii < nPartitions_; ii++)
  {
    if (boxes_[ii]->nSamples_ > 0)
    {
      boxes_[ii]->rsPtr_ = new DNN(nInputs_, boxes_[ii]->nSamples_);
      boxes_[ii]->rsPtr_->setOutputLevel(0);
      boxes_[ii]->rsPtr_->setBounds(boxes_[ii]->vecLBs_.getDVector(),
                                    boxes_[ii]->vecUBs_.getDVector());
    }
  }
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
        printf("MDNN: processing partition %d (%d)\n",ii+1,nPartitions_);
      boxes_[ii]->rsPtr_->initialize(boxes_[ii]->vecX_.getDVector(),
                                     boxes_[ii]->vecY_.getDVector());
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
        printf("MDNN: processing partition %d completed\n",ii+1);
    }
  }
}
  turnPrintTSOn();

  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("MDNN training completed.\n");
  if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    printf("MDNN INFO: response surface stand-alone code not available.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MDNN::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
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
    printf("MDNN interpolation begins....\n");
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MDNN interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int MDNN::gen1DGridData(double *XIn, double *YIn, int ind1,
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
    printf("MDNN interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MDNN interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int MDNN::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
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
    printf("MDNN interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MDNN interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MDNN::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
    printf("MDNN interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MDNN interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MDNN::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
    printf("MDNN interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("MDNN interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MDNN::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double MDNN::evaluatePoint(int npts, double *X, double *Y)
{
  interpolate(npts, X, Y, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double MDNN::evaluatePointFuzzy(double *X, double &std)
{
  int    iOne=1;
  double Y=0.0;
  interpolate(iOne, X, &Y, &std);
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double MDNN::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystds)
{
  interpolate(npts, X, Y, Ystds);
  return 0.0;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int MDNN::interpolate(int npts, double *X, double *Y, double *Ystds)
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
      printf("MDNN evaluate ERROR: sample point outside range.\n");
      for (ii = 0; ii < nInputs_; ii++)
        printf("Input %d = %e (in [%e, %e]?)\n",ii+1,
               X[ss*nInputs_+ii],VecLBs_[ii],VecUBs_[ii]);
      return 0.0;
    }
    Y[ss] = Yt / (double) count;
  }
  return 0;
}

