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
// Functions for the User-specified Metis sampling class 
// (arbitrary domain - determined by UserMetisDriver) 
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FunctionInterface.h"
#include "UserMetisSampling.h"

#define PABS(x) ((x) > 0 ? x : -(x))

#ifdef HAVE_METIS

extern "C" 
{
void METIS_PartGraphRecursive(int *, int *, int *, int *, int *,
                              int *, int *, int *, int *, int *, int *);
}

#endif

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
UserMetisSampling::UserMetisSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_UMETIS;
  n1d_        = -1;
  nAggrs_     = 0;
  vecAggrLabels_ = NULL;
}

// ************************************************************************
// copy constructor added by Bill Oliver 
// ------------------------------------------------------------------------
UserMetisSampling::UserMetisSampling(const UserMetisSampling &ums):Sampling()
{
  n1d_ = ums.n1d_;
  nAggrs_ = ums.nAggrs_;
  vecAggrCnts_ = ums.vecAggrCnts_;
  nInputs_ = ums.nInputs_;
  if (ums.vecAggrLabels_ != NULL) 
  {
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = ums.vecAggrLabels_[ii];
  } 
  else 
  {
    initialize(0);
  }

  vecCellsOccupied_ = ums.vecCellsOccupied_;
  printLevel_ = ums.printLevel_;
  samplingID_ = ums.samplingID_;
  nSamples_ = ums.nSamples_;
  nOutputs_ = ums.nOutputs_;
  randomize_ = ums.randomize_;
  nReplications_ = ums.nReplications_;
  vecLBs_ = ums.vecLBs_; 
  vecUBs_ = ums.vecUBs_; 
  vecSamInps_ = ums.vecSamInps_;
  vecSamOuts_ = ums.vecSamOuts_;
  vecSamStas_ = ums.vecSamStas_;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
UserMetisSampling::~UserMetisSampling()
{
  if (vecAggrLabels_ != NULL)
  {
    for (int ii = 0; ii < nAggrs_; ii++) vecAggrLabels_[ii].clean();
    delete [] vecAggrLabels_;
  }
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int UserMetisSampling::initialize(int initLevel)
{
  int    inputID, ii, jj, kk, itmp, jtmp, nnz, sampleID, randFlag;
  int    options[10], index, haveExec, nAppFiles=5, newGraphN, graphN;
  double dtmp, constraints[2];
  char   **inputNames, **outputNames, **driverNames, command[500];
  char   pString[500];
  FILE   *fp;
  FunctionInterface *funcIO;
  psVector  vecLBs, vecUBs, vecRanges, vecOneSamInps, vecOneSamOuts;
  psIVector vecIncrs, vecGraphI, vecGraphJ, vecAggrMap;
                                                                                
  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("UserMetisSampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("UserMetisSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking for test file
  //**/ ----------------------------------------------------------------
  psVector vecCoeffs;
  haveExec = 1;
  fp = fopen("UserMetisDriver", "r");
  if (fp == NULL)
  {
    haveExec = 0;
    printf("UserMetisSampling missing UserMetisDriver: \n");
    snprintf(pString,100,
             "Is it a linear regression with simple bounds? (y or n) ");
    getString(pString, command);
    if (command[0] == 'y')
    {
      vecCoeffs.setLength(nInputs_+1);
      snprintf(pString,100,"Enter the constant coefficient : ");
      vecCoeffs[0] = getDouble(pString);
      for (ii = 1; ii <= nInputs_; ii++)
      {
        snprintf(pString,100,"Enter the coefficient for input %d : ",ii);
        vecCoeffs[ii] = getDouble(pString);
      }
      snprintf(pString,100,"Enter the lower bound : ");
      constraints[0] = getDouble(pString);
      snprintf(pString,100,"Enter the upper bound : ");
      constraints[1] = getDouble(pString);
    }
    else
    {
      printf("UserMetisSampling ERROR : missing program UserMetisDriver.\n");
      printf("      UserMetisDriver should be an executable taking\n");
      printf("      the first argument as input file and second argument\n");
      printf("      as output file. The input file has the first line\n");
      printf("      the number of inputs, followed by all the inputs.\n");
      printf("      The output file just contains all outputs.\n");
      exit(1);
    }
  }
  else fclose(fp);

  //**/ ----------------------------------------------------------------
  //**/ clean up
  //**/ ----------------------------------------------------------------
  if (vecAggrLabels_ != NULL)
  {
    for (ii = 0; ii < nAggrs_; ii++) vecAggrLabels_[ii].clean();
    delete [] vecAggrLabels_;
    vecAggrLabels_ = NULL;
  }

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  randFlag = 0;
  if (randomize_ & 1) randFlag = 1;
  if (nInputs_ > 12)
  {
    printf("UserMetisSampling ERROR : does not support nInputs > 12.\n");
    exit(1);
  }
  if (nInputs_ == 1 ) n1d_ = nSamples_;
  if (nInputs_ == 2 ) n1d_ = 512;
  if (nInputs_ == 3 ) n1d_ = 64;
  if (nInputs_ == 4 ) n1d_ = 23;
  if (nInputs_ == 5 ) n1d_ = 12;
  if (nInputs_ == 6 ) n1d_ = 8;
  if (nInputs_ == 7 ) n1d_ = 6;
  if (nInputs_ == 8 ) n1d_ = 5;
  if (nInputs_ == 9 ) n1d_ = 4;
  if (nInputs_ == 10) n1d_ = 3;
  if (nInputs_ >= 11) n1d_ = 3;

  vecRanges.setLength(nInputs_);
  vecLBs.setLength(nInputs_);
  vecUBs.setLength(nInputs_);
  if (printLevel_ > 1)
    printf("UserMetisSampling:: ranges expanded by 10 %% on each side.\n");
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = (vecUBs_[inputID] - vecLBs_[inputID])*0.1;
    vecLBs[inputID] = vecLBs_[inputID] - vecRanges[inputID];
    vecUBs[inputID] = vecUBs_[inputID] + vecRanges[inputID];
  }
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs[inputID] - vecLBs[inputID];

  //**/ ----------------------------------------------------------------
  //**/ set up the grid matrix
  //**/ ----------------------------------------------------------------
  vecIncrs.setLength(nInputs_+1);
  graphN   = 1;
  vecIncrs[0] = graphN;
  for (inputID = 1; inputID <= nInputs_; inputID++)
  {
    graphN *= n1d_;
    vecIncrs[inputID] = graphN;
  }
  if (nSamples_ > 2*graphN)
  {
    printf("UserMetisSampling ERROR : nSamples %d too large.\n",nSamples_);
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface 
  //**/ ----------------------------------------------------------------
  if (haveExec == 1)
  {
    funcIO = new FunctionInterface();
    inputNames = new char*[nInputs_];
    for (ii = 0; ii < nInputs_; ii++)
    {
      inputNames[ii] = new char[200];
      snprintf(inputNames[ii],100,"X%d", ii+1);
    }
    outputNames = new char*[nInputs_];
    for (ii = 0; ii < nOutputs_; ii++)
    {
      outputNames[ii] = new char[200];
      snprintf(outputNames[ii],100,"Y%d", ii+1);
    }
    driverNames = new char*[5];
    for (ii = 0; ii < 5; ii++)
    {
      driverNames[ii] = new char[200];
      strcpy(driverNames[ii], "NONE");
    }
    strcpy(driverNames[0], "UserMetisDriver");
    funcIO->loadInputData(nInputs_, inputNames);
    funcIO->loadOutputData(nOutputs_,outputNames);
    funcIO->loadFunctionData(nAppFiles, driverNames);
    for (ii = 0; ii < nInputs_; ii++) delete [] inputNames[ii];
    delete [] inputNames;
    for (ii = 0; ii < nOutputs_; ii++) delete [] outputNames[ii];
    delete [] outputNames;
    for (ii = 0; ii < 5; ii++) delete [] driverNames[ii];
    delete [] driverNames;
  }

  //**/ ----------------------------------------------------------------
  //**/ run samples
  //**/ ----------------------------------------------------------------
  vecCellsOccupied_.setLength(graphN);
  vecOneSamInps.setLength(nInputs_);
  vecOneSamOuts.setLength(nOutputs_);
  newGraphN  = 0;
  for (ii = 0; ii < graphN; ii++)
  {
    itmp = ii;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      dtmp = ((double) (jtmp + 0.5)) / (double) n1d_;
      vecOneSamInps[inputID] = dtmp * vecRanges[inputID] + vecLBs_[inputID];
    }
    if (ii % 100 == 0) 
      printf("UserMetisSampling::initialize - running %d out of %d.\n",
             ii+1, graphN);
    if (haveExec == 1)
      funcIO->evaluate(1,nInputs_,vecOneSamInps.getDVector(),nOutputs_,
                       vecOneSamOuts.getDVector(),0);
    else
    {
      for (kk = 0; kk < nOutputs_; kk++)
      {
        dtmp = vecCoeffs[0];
        for (jj = 0; jj < nOutputs_; jj++)
          dtmp += vecCoeffs[jj+1]*vecOneSamInps[jj];
        if (dtmp < constraints[0] || dtmp > constraints[1])
             vecOneSamOuts[kk] = 0.0;
        else vecOneSamOuts[kk] = 1.0;
      }
    }
    dtmp = 0.0;
    for (jj = 0; jj < nOutputs_; jj++) dtmp += vecOneSamOuts[jj];
    if (dtmp == 0.0) vecCellsOccupied_[ii] = newGraphN++;
    else             vecCellsOccupied_[ii] = 9999999;
  }
  if (haveExec == 1) delete funcIO;
 
  //**/ ----------------------------------------------------------------
  //**/ set up the grid graph
  //**/ ----------------------------------------------------------------
  vecGraphI.setLength(newGraphN+1);
  vecGraphJ.setLength(newGraphN*nInputs_*2);
  nnz = 0;
  vecGraphI[0] = nnz;
  newGraphN = 0;
  for (ii = 0; ii < graphN; ii++)
  {
    if (vecCellsOccupied_[ii] != 9999999)
    {
      itmp = ii;
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        jtmp = itmp % n1d_;
        itmp = itmp / n1d_;
        if (jtmp > 0)
        {
          jj = vecCellsOccupied_[ii-vecIncrs[inputID]];
          if (jj != 9999999) vecGraphJ[nnz++] = jj;
        }
        if (jtmp < n1d_-1)
        {
          jj = vecCellsOccupied_[ii+vecIncrs[inputID]];
          if (jj != 9999999) vecGraphJ[nnz++] = jj;
        }
      }
      newGraphN++;
      vecGraphI[newGraphN] = nnz;
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ call metis to perform partitioning
  //**/ ----------------------------------------------------------------
  options[0] = 0;
  vecAggrMap.setLength(newGraphN);
  if (printLevel_ > 1)
    printf("UserMetisSampling:: calling domain partitioner.\n");
#ifdef HAVE_METIS
  int wgtflag=0, numflag=0, edgeCut=0;
  METIS_PartGraphRecursive(&newGraphN,vecGraphI.getIVector(), 
        vecGraphJ.getIVector(), NULL, NULL, &wgtflag,&numflag,&nSamples_
        ,options,&edgeCut,vecAggrMap.getIVector());
#else
  printf("UserMetisSampling ERROR : METIS not installed.\n");
#endif
  if (printLevel_ > 1)
    printf("UserMetisSampling:: subdomains created.\n");

  //**/ ----------------------------------------------------------------
  //**/ store partition information
  //**/ ----------------------------------------------------------------
  nAggrs_   = nSamples_;
  if(nAggrs_ <= 0)
  {
    printf("nAggrs_ is <= 0 in file %s line %d\n", __FILE__, __LINE__);
    exit(1);
  }
  vecAggrCnts_.setLength(nAggrs_);
  for (ii = 0; ii < graphN; ii++)
  {
    if (vecCellsOccupied_[ii] != 9999999)
    {
      index = vecAggrMap[vecCellsOccupied_[ii]];
      vecAggrCnts_[index]++;  
    }
  }
   
  vecAggrLabels_ = new psIVector[nAggrs_];
  for (ii = 0; ii < nAggrs_; ii++)
  {
    vecAggrLabels_[ii].setLength(vecAggrCnts_[ii]);
    vecAggrCnts_[ii] = 0;
  }
  for (ii = 0; ii < graphN; ii++)
  {
    if (vecCellsOccupied_[ii] != 9999999)
    {
      index = vecAggrMap[vecCellsOccupied_[ii]];
      vecAggrLabels_[index][vecAggrCnts_[index]++] = ii;  
    }
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ map labels to data points
  //**/ ----------------------------------------------------------------
  allocSampleData();

  for (sampleID = 0; sampleID < nSamples_; sampleID++)
  {
    if (randFlag == 1)
         index = (int) (PSUADE_drand() * vecAggrCnts_[sampleID]);
    else index = vecAggrCnts_[sampleID] / 2;
    if (index == vecAggrCnts_[sampleID]) index--;
    index = vecAggrLabels_[sampleID][index];
    vecCellsOccupied_[index] = -(vecCellsOccupied_[index] + 1);
    itmp = index;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      if (randFlag == 1)
           dtmp = ((double) (jtmp + 0.999*PSUADE_drand())) / (double) n1d_;
      else dtmp = ((double) (jtmp + 0.5)) / (double) n1d_;
      vecSamInps_[sampleID*nInputs_+inputID] = dtmp * vecRanges[inputID] +
                                               vecLBs[inputID];
      if (vecSamInps_[sampleID*nInputs_+inputID] < vecLBs_[inputID])
        vecSamInps_[sampleID*nInputs_+inputID] = vecLBs_[inputID];
      if (vecSamInps_[sampleID*nInputs_+inputID] > vecUBs_[inputID])
        vecSamInps_[sampleID*nInputs_+inputID] = vecUBs_[inputID];
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("UserMetisSampling::initialize: nSamples = %d\n", nSamples_);
    printf("UserMetisSampling::initialize: nInputs  = %d\n", nInputs_);
    printf("UserMetisSampling::initialize: nOutputs = %d\n", nOutputs_);
    if (randFlag != 0)
         printf("UserMetisSampling::initialize: randomize on\n");
    else printf("UserMetisSampling::initialize: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    UserMetisSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int UserMetisSampling::refine(int,int,double, int, double *)
{
  printf("UserMetisSampling refine: not implemented yet.\n");
  return -1;
} 

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int UserMetisSampling::setParam(char *sparam)
{
  char winput[1001];
  FILE *fp;

  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "reset"))
  {
    fp = fopen("psuadeMetisInfo", "r");
    if (fp != NULL)
    {
      fclose(fp);
      unlink("psuadeMetisInfo");
    }
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
UserMetisSampling& UserMetisSampling::operator=(const UserMetisSampling & ums)
{
  if(this == &ums) return *this;

  int graphN = 1;
  n1d_ = ums.n1d_;
  nAggrs_ = ums.nAggrs_;
  if (ums.vecAggrLabels_ != NULL) 
  {
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = ums.vecAggrLabels_[ii];
  } 
  vecAggrCnts_ = ums.vecAggrCnts_; 
  nInputs_ = ums.nInputs_;
  vecCellsOccupied_ = ums.vecCellsOccupied_; 
  printLevel_ = ums.printLevel_;
  samplingID_ = ums.samplingID_;
  nSamples_ = ums.nSamples_;
  nOutputs_ = ums.nOutputs_;
  randomize_ = ums.randomize_;
  nReplications_ = ums.nReplications_;
  vecLBs_ = ums.vecLBs_; 
  vecUBs_ = ums.vecUBs_; 
  vecSamInps_ = ums.vecSamInps_;
  vecSamOuts_ = ums.vecSamOuts_;
  vecSamStas_ = ums.vecSamStas_;
  return (*this);
}

