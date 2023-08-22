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
// Functions for the GMetis sampling class (generalized Metis) 
// AUTHOR : CHARLES TONG
// DATE   : 2010
// ************************************************************************
#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <unistd.h>
//**/using namespace std;

#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FuncApprox.h"
#include "Mars.h"
#include "GMetisSampling.h"

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
GMetisSampling::GMetisSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_METIS;
  refineType_ = 0;
  refineSize_ = 1000000;
  n1d_        = -1;
  nAggrs_     = 0;
  vecAggrLabels_ = NULL;
  graphN_     = 0;
  changeInfoName_ = 0;
}

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
GMetisSampling::GMetisSampling(const GMetisSampling & gms) : Sampling()
{
  samplingID_ = gms.samplingID_;
  nSamples_ = gms.nSamples_;
  nInputs_ = gms.nInputs_;
  nOutputs_ = gms.nOutputs_;
  refineType_ = gms.refineType_;
  refineSize_ = gms.refineSize_;
  n1d_ = gms.n1d_;
  nAggrs_ = gms.nAggrs_;
  graphN_ = gms.graphN_;
  vecAggrCnts_ = gms.vecAggrCnts_;
  if (gms.vecAggrLabels_ != NULL)
  {
    vecAggrLabels_ = new psIVector[nAggrs_]; 
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = gms.vecAggrLabels_[ii];
  }
  printLevel_ = gms.printLevel_;
  randomize_ = gms.randomize_;
  nReplications_ = gms.nReplications_;
  vecLBs_ = gms.vecLBs_;
  vecUBs_ = gms.vecUBs_;
  vecSamInps_ = gms.vecSamInps_;
  vecSamOuts_ = gms.vecSamOuts_;
  vecSamStas_ = gms.vecSamStas_;
  vecGraphI_ = gms.vecGraphI_;
  vecGraphJ_ = gms.vecGraphJ_;
  vecCellsOccupied_ = gms.vecCellsOccupied_;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
GMetisSampling::~GMetisSampling()
{
  if (vecAggrLabels_ != NULL)
  {
    for (int ii = 0; ii < nAggrs_; ii++) vecAggrLabels_[ii].clean();
    delete [] vecAggrLabels_;
    vecAggrLabels_ = NULL;
  }
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int GMetisSampling::initialize(int initLevel)
{
  int    inputID, ii, jj, itmp, jtmp, nnz, sampleID;
  int    options[10], index, count, kk, saveFlag=1;
#ifdef HAVE_METIS
  int    wgtflag=0, numflag=0, edgeCut=0;
#endif
  double dtmp, expand=0.0;
  char   response[1001], pString[1001], filename[1001];
  FILE   *fp=NULL;
  psVector vecRanges, vecLBnds, vecUBnds;

  //**/ ===========================================================
  //**/ Error checking and clean up
  //**/ ===========================================================
  if( nSamples_ <= 0)
  {
    printf("nSamples_ in file %s line %d is <= 0\n",__FILE__,__LINE__);
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("GMetisSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ===========================================================
  //**/ Clean up
  //**/ ===========================================================
  vecSamInps_.clean();
  vecSamOuts_.clean();
  vecSamStas_.clean();
  vecAggrCnts_.clean();
  if (vecAggrLabels_ != NULL)
  {
    for (ii = 0; ii < nAggrs_; ii++) vecAggrLabels_[ii].clean();
    delete [] vecAggrLabels_;
    vecAggrLabels_ = NULL;
  }
  vecGraphI_.clean();
  vecGraphJ_.clean();
  vecCellsOccupied_.clean();
  nAggrs_ = 0;

  //**/ ===========================================================
  //**/ generate grid and the corresponding grid matrix
  //**/ ===========================================================
  if (nInputs_ > 21)
  {
    printf("GMetisSampling ERROR : nInputs > 21 currently not supported.\n");
     exit(1);
  }
  if (nInputs_ == 1 ) n1d_ = nSamples_*10;
  if (nInputs_ == 2 ) n1d_ = 1024;
  if (nInputs_ == 3 ) n1d_ = 100;
  if (nInputs_ == 4 ) n1d_ = 36;
  if (nInputs_ == 5 ) n1d_ = 16;
  if (nInputs_ == 6 ) n1d_ = 11;
  if (nInputs_ == 7 ) n1d_ = 8;
  if (nInputs_ == 8 ) n1d_ = 6;
  if (nInputs_ == 9 ) n1d_ = 5;
  if (nInputs_ == 10) n1d_ = 4;
  if (nInputs_ == 11) n1d_ = 3;
  if (nInputs_ == 12) n1d_ = 3;
  if (nInputs_ == 13) n1d_ = 3;
  if (nInputs_ >= 14) n1d_ = 2;

  psIVector vecIncrs;
  vecIncrs.setLength(nInputs_+1);
  graphN_ = 1;
  vecIncrs[0] = graphN_;
  for (inputID = 1; inputID <= nInputs_; inputID++)
  {
    graphN_ *= n1d_;
    vecIncrs[inputID] = graphN_;
  }
  if (nSamples_ > 2*graphN_)
  {
    printf("GMetisSampling ERROR : nSamples %d too large.\n",nSamples_);
    exit(1);
  }
  vecGraphI_.setLength(graphN_+1);
  vecGraphJ_.setLength(graphN_*nInputs_*2+1);
  nnz = 0;
  vecGraphI_[0] = nnz;
  for (ii = 0; ii < graphN_; ii++)
  {
    itmp = ii;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      if (jtmp > 0     ) vecGraphJ_[nnz++] = ii - vecIncrs[inputID];
      if (jtmp < n1d_-1) vecGraphJ_[nnz++] = ii + vecIncrs[inputID];
    }
    vecGraphI_[ii+1] = nnz;
  }
  vecCellsOccupied_.setLength(graphN_);

  //**/ ===========================================================
  //**/ read previously generated grid file
  //**/ ===========================================================
  if (changeInfoName_ == 0) fp = fopen("psuadeGMetisInfo", "r");
  else                      fp = fopen("psuadeGMetisInfo.tmp", "r");
  if (fp != NULL)
  {
    printf("INFO: psuadeGMetisInfo file found. Reading it in ...\n");
    fscanf(fp, "%d %d %d", &jj, &itmp, &jtmp);
    if (itmp != nSamples_ || jtmp != nInputs_)
    {
      fclose(fp);
      printf("GMetisSampling INFO: a partition file is found but\n");
      printf("      the data is not consistent with this setup\n");
      printf("      (The file name is psuadeGMetisInfo).\n");
      if (itmp != nSamples_)
        printf("      nSamples : %d != %d.\n", nSamples_, itmp);
      if (jtmp != nInputs_)
        printf("      nInputs  : %d != %d.\n", nInputs_, jtmp);
      printf("NOTE: THIS FILE IS NOT TO BE USED.\n");
    }
    else
    {
      nAggrs_ = jj;
      vecAggrCnts_.setLength(nAggrs_);
      vecAggrLabels_ = new psIVector[nAggrs_];
      for (ii = 0; ii < nAggrs_; ii++)
      {
        fscanf(fp, "%d", &count);
        if (printLevel_ > 4) 
          printf("GMetis read: aggr %8d, size = %d\n", ii+1, count);
        if (count > 0)
        {
          vecAggrCnts_[ii] = count;
          vecAggrLabels_[ii].setLength(count);
        }
        else vecAggrCnts_[ii] = count = 0;
        for (jj = 0; jj < count; jj++)
        {
          fscanf(fp, "%d", &kk);
          if (kk < 0 || kk >= graphN_)
          {
            printf("GMetis ERROR: psuadeGMetisInfo file has invalid info.\n");
            printf("              Aggregate number = %d (%d)\n",ii,nAggrs_);
            printf("              Invalid cell number = %d (%d)\n",kk,graphN_);
            exit(1);
          }
          if (kk < 0 || kk >= graphN_)
          {
            printf("GMetis ERROR: psuadeGMetisInfo file has invalid info.\n");
            printf("              Aggregate number = %d (%d)\n",ii,nAggrs_);
            printf("              Invalid cell number = %d (<%d)\n",kk,
                   graphN_);
            exit(1);
          }
          vecCellsOccupied_[kk] = ii;
          vecAggrLabels_[ii][jj] = kk;
        }
      }
      fscanf(fp, "%d", &count);
      for (ii = 0; ii < count; ii++)
      {
        fscanf(fp, "%d %d", &jj, &kk);
        if (jj < 0 || jj >= graphN_)
        {
          printf("GMetis ERROR: psuadeGMetisInfo file has invalid info\n");
          printf("              Aggregate number = %d (%d)\n",kk,nAggrs_);
          printf("              Invalid cell number = %d (%d)\n",jj,graphN_);
          exit(1);
        }
        if (kk < 0 || kk >= nAggrs_)
        {
          printf("GMetis ERROR: psuadeGMetisInfo file has invalid info\n");
          printf("              Invalid aggregate number = %d (%d)\n",kk,
                 nAggrs_);
          exit(1);
        }
        vecCellsOccupied_[jj] = - kk - 1;
        if (printLevel_ > 4) 
          printf("Sample %6d: cell occupied = %d\n",ii+1,jj+1);
      }
      fclose(fp);
      fp = NULL;
      saveFlag = 0;
    }
  }

  //**/ ===========================================================
  //**/ partition
  //**/ ===========================================================
  if (vecAggrCnts_.length() == 0)
  {
    options[0] = 0;
#ifdef HAVE_METIS
    if (nAggrs_ == 0)
    {
      if (psConfig_.SamExpertModeIsOn())
      {
        snprintf(pString,100,"GMetis: Enter number of partitions (2 - %d): ",
                 nSamples_);
        nAggrs_ = getInt(2, nSamples_, pString);
      }
      else
      {
        if (nSamples_ < 5) nAggrs_ = nSamples_;
        else               nAggrs_ = nSamples_;
        printf("GMetisSampling INFO: default number of partitions = %d.\n",
               nAggrs_);
      }
    }
    if (printLevel_ > 1)
      printf("GMetisSampling: creating %d partitions...\n", nAggrs_);
    METIS_PartGraphRecursive(&graphN_, vecGraphI_.getIVector(), 
                vecGraphJ_.getIVector(), NULL, NULL, &wgtflag, &numflag,
                &nAggrs_,options,&edgeCut,vecCellsOccupied_.getIVector());
#else
    printf("GMetisSampling ERROR : METIS not installed.\n");
    exit(1);
#endif
    if (printLevel_ > 1)
      printf("GMetisSampling:: %d subdomains created.\n", nAggrs_);

    vecAggrCnts_.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++) vecAggrCnts_[ii] = 0;
    for (ii = 0; ii < graphN_; ii++)
    {
      if (vecCellsOccupied_[ii] < 0 || vecCellsOccupied_[ii] >= nAggrs_) 
      {
        printf("GMetisSampling INTERNAL ERROR (1).\n");
        exit(1);
      }
      vecAggrCnts_[vecCellsOccupied_[ii]]++;  
    }
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (ii = 0; ii < nAggrs_; ii++)
    {
      if (vecAggrCnts_[ii] <= 0) 
      {
        printf("GMetisSampling INTERNAL ERROR (2).\n");
        exit(1);
      }
      vecAggrLabels_[ii].setLength(vecAggrCnts_[ii]);
      vecAggrCnts_[ii] = 0;
    }
    for (ii = 0; ii < graphN_; ii++)
    {
      index = vecCellsOccupied_[ii];
      if (index < 0 || index >= nAggrs_) 
      {
        printf("GMetisSampling INTERNAL ERROR (3).\n");
        exit(1);
      }
      vecAggrLabels_[index][vecAggrCnts_[index]++] = ii;  
    }
  }
  
  //**/ ===========================================================
  //**/ save newly generated partition file
  //**/ ===========================================================
  if (saveFlag == 1)
  {
    if (changeInfoName_ == 0) fp = fopen("psuadeGMetisInfo", "w");
    else                      fp = fopen("psuadeGMetisInfo.tmp", "w");
  }
  if (fp != NULL)
  {
    fprintf(fp, "%d %d %d\n", nAggrs_, nSamples_, nInputs_);
    for (ii = 0; ii < nAggrs_; ii++)
    {
      fprintf(fp, "%d\n", vecAggrCnts_[ii]);
      for (jj = 0; jj < vecAggrCnts_[ii]; jj++)
      {
        kk = vecAggrLabels_[ii][jj];
        fprintf(fp, "%d ", kk);
        if (jj != 0 && jj % 10 == 0) fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
    }
    jj = 0;
    for (ii = 0; ii < graphN_; ii++) if (vecCellsOccupied_[ii] < 0) jj++;
    fprintf(fp, "%d\n", jj);
    for (ii = 0; ii < graphN_; ii++)
      if (vecCellsOccupied_[ii] < 0) 
        fprintf(fp, "%d %d\n",ii,-(vecCellsOccupied_[ii]+1));
    fclose(fp);
    fp = NULL;
  }
  if (initLevel != 0) return 0;

  //**/ ===========================================================
  //**/ generate sample
  //**/ ===========================================================
  allocSampleData();
  vecRanges.setLength(nInputs_);
  vecLBnds.setLength(nInputs_);
  vecUBnds.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = (vecUBs_[inputID] - vecLBs_[inputID]) *
                         0.5 * expand;
    vecLBnds[inputID] = vecLBs_[inputID] - vecRanges[inputID];
    vecUBnds[inputID] = vecUBs_[inputID] + vecRanges[inputID];
  }
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID]  = vecUBnds[inputID] - vecLBnds[inputID];

  sampleID = 0;
  while (sampleID < nSamples_)
  {
    index = (int) (PSUADE_drand() * vecAggrCnts_[sampleID%nAggrs_]);
    if (index == vecAggrCnts_[sampleID%nAggrs_]) index--;
    index = vecAggrLabels_[sampleID%nAggrs_][index];
    vecCellsOccupied_[index] = -(vecCellsOccupied_[index] + 1);
    itmp = index;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      dtmp = ((double) (jtmp + 0.999*PSUADE_drand())) / (double) n1d_;
      vecSamInps_[sampleID*nInputs_+inputID] = dtmp * vecRanges[inputID] +
                                               vecLBnds[inputID];
    }
    sampleID++;
  }

  if (printLevel_ > 4)
  {
    printf("GMetisSampling::initialize: nSamples = %d\n", nSamples_);
    printf("GMetisSampling::initialize: nInputs  = %d\n", nInputs_);
    printf("GMetisSampling::initialize: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("GMetisSampling::initialize: randomize on\n");
    else printf("GMetisSampling::initialize: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    GMetisSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int GMetisSampling::refine(int nLevels, int randFlag, double threshold,
                           int nSamples, double *sampleErrors)
{
  int    inputID, ii, jj, count, itmp, jtmp, localN;
  int    maxN, rowInd, colInd, localNNZ, index, count0;
  int    options[10], count1;
#ifdef HAVE_METIS
  int    wgtflag=0, numflag=0, edgeCut=0, itwo=2;
#endif
  int    currNAggr, newCell, oldNumSamples, status, outputID, ss, maxNNZ;
  int    splitCount, splitSuccess, cellCnt=0;
  FILE   *fp;
  psVector  vecRanges;
  psIVector *vecTmpLabels;

  if (printLevel_ > 4) printf("GMetisSampling: refine.\n");
  //**/ ===========================================================
  //**/ error checking
  //**/ ===========================================================
  if (vecCellsOccupied_.length() == 0 || vecAggrLabels_ == NULL)
  {
    printf("GMetisSampling::refine ERROR - need to call initialize first.\n");
    exit(1);
  }
  if (printLevel_ > 0)
  {
    printf("GMetisSampling::refine(1): nSamples = %d\n", nSamples_);
    printf("GMetisSampling::refine(1): nInputs  = %d\n", nInputs_);
    printf("GMetisSampling::refine(1): nOutputs = %d\n", nOutputs_);
  }

  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
    if (vecRanges[inputID] <= 0.0)
    {
      printf("GMetisSampling::refine ERROR - lbound/ubound mismatch.\n");
      exit(1);
    }
  }

  //**/ ===========================================================
  //**/ put existing sample points into cells
  //**/ ===========================================================
  double dtmp;
  count = 0;
  for (ss = 0; ss < nSamples_; ss++)
  {
    itmp = 0;
    for (inputID = nInputs_-1; inputID >= 0; inputID--)
    {
      itmp = itmp * n1d_;
      dtmp = vecSamInps_[ss*nInputs_+inputID];
      dtmp = (dtmp - vecLBs_[inputID]) / vecRanges[inputID];
      if (dtmp == 1.0) jtmp = n1d_ - 1;
      else             jtmp = (int) (dtmp * n1d_);
      itmp += jtmp;
    }
    if (itmp >= graphN_)
    {
      printf("GMetis INTERNAL ERROR (4).\n");
      exit(1);
    }
    //**/ change cell aggr index to negative to indicate occupied
    if (vecCellsOccupied_[itmp] >= 0)
    {
      vecCellsOccupied_[itmp] = -(vecCellsOccupied_[itmp] + 1); 
      count++;
    }
  }
   
  //**/ ===========================================================
  //**/ expand the label array to accommodate more points
  //**/ ===========================================================
  psIVector vecTmpCnts;
  vecTmpCnts = vecAggrCnts_;
  vecTmpLabels = vecAggrLabels_;
  vecAggrCnts_.setLength(2*nAggrs_);
  vecAggrLabels_ = new psIVector[2*nAggrs_];
  for (ii = 0; ii < nAggrs_; ii++)
  {
    vecAggrCnts_[ii] = vecTmpCnts[ii];
    vecAggrLabels_[ii].setLength(vecAggrCnts_[ii]);
    for (jj = 0; jj < vecAggrCnts_[ii]; jj++) 
      vecAggrLabels_[ii][jj] = vecTmpLabels[ii][jj];
    vecTmpLabels[ii].clean();
  }
  delete [] vecTmpLabels;
  for (ii = nAggrs_; ii < 2*nAggrs_; ii++) 
  {
    vecAggrCnts_[ii] = 0;
    vecAggrLabels_[ii].clean();
  }

  //**/ ===========================================================
  //**/ preparation for refinement
  //**/ ===========================================================
  psIVector vecSubLabels;
  psIVector vecIA, vecJA, vecLabels, vecNode2Aggr, vecRefine;
  psVector  vecAggrErrs, vecDList1, vecDList2;
  maxN = 0;
  for (ss = 0; ss < nAggrs_; ss++)
    if (vecAggrCnts_[ss] > maxN) maxN = vecAggrCnts_[ss];
  maxNNZ = maxN * (2 * nInputs_ + 1);
  vecLabels.setLength(maxN);
  vecIA.setLength(maxN+1);
  vecJA.setLength(maxNNZ);

  if (refineType_ == 1 && sampleErrors == NULL)
  {
    printf("GMetisSampling::refine ERROR- error based but no error given.\n");
    exit(1);
  }
  if (refineType_ == 1 && sampleErrors != NULL)
  {
    printf("GMetisSampling::refine - maximum number of new points = %d.\n",
           refineSize_);
    //**/ ========================================================
    //**/ create a node to aggregate array
    //**/ ========================================================
    vecNode2Aggr.setLength(graphN_);
    for (ii = 0; ii < graphN_; ii++) vecNode2Aggr[ii] = -1;
    for (ii = 0; ii < nAggrs_; ii++)
    {
      localN = vecAggrCnts_[ii];
      vecSubLabels = vecAggrLabels_[ii];
      for (jj = 0; jj < localN; jj++)
      {
        rowInd = vecSubLabels[jj];
        if (rowInd < 0 || rowInd >= graphN_)
        {
          printf("GMetisSampling::refine ERROR (index out of bound.)\n");
          printf("               rowInd = %d (should be in [0,%d]\n",
          rowInd, graphN_-1);
          printf("               Aggregate = %d \n", ii);
          printf("               Please consult PSUADE developers.\n");
          exit(1);
        }
        vecNode2Aggr[rowInd] = ii;
      }
    }
    //**/ ========================================================
    //**/ check to make sure each node belongs to an aggregate
    //**/ ========================================================
    count = 0;
    for (ii = 0; ii < graphN_; ii++)
    {
      if (vecNode2Aggr[ii] == -1)
      {
        count++;
        printf("ERROR : node2Aggr %d = -1\n", ii);
      }
      if (count > 0)
      {
        printf("GMetisSampling::Internal ERROR - incorrect node2Aggr.\n");
        printf("      No. of wrong node2Aggr = %d (%d).\n",count,graphN_);
        printf("      Please consult PSUADE developers.\n");
        exit(1);
      }
    }

    //**/ ========================================================
    //**/ sort aggregate errors 
    //**/ ========================================================
    vecAggrErrs.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++) vecAggrErrs[ii] = sampleErrors[ii];
    vecDList1.setLength(nAggrs_);
    vecDList2.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++) vecDList1[ii] = PABS(vecAggrErrs[ii]);
    for (ii = 0; ii < nAggrs_; ii++) vecDList2[ii] = (double) ii;
    sortDbleList2(nAggrs_, vecDList1.getDVector(), vecDList2.getDVector());
    vecRefine.setLength(nAggrs_);
    for (ii = nAggrs_-1; ii >= 0; ii--) vecRefine[ii] = 0;
    cellCnt = 0;

    //**/ ========================================================
    //**/ tag the aggregate for refinement 
    //**/ ========================================================
    for (ii = nAggrs_-1; ii >= 0; ii--)
    {
      index = (int) vecDList2[ii];
      localN = vecAggrCnts_[index];
      dtmp = vecAggrErrs[index] * pow(1.0*localN,1.0/nInputs_);
      if (dtmp != 0.0) 
      {
        //printf("GMetisSampling::refine - chosen aggregate, error = %e\n",
        //        aggrErrs[index]);
        if (vecRefine[index] == 0)
        {
          printf("GMetis: %7d selected for refinement: error = %13.5e\n",
                 index+1, vecAggrErrs[index]);

          cellCnt++;
          vecRefine[index] = 1;
          if (cellCnt >= refineSize_) break;
        }
        if (cellCnt >= refineSize_) break;
      }
      if (cellCnt >= refineSize_) break;
    }
  }
  else
  {
    vecRefine.setLength(nAggrs_);
    for (ii = nAggrs_-1; ii >= 0; ii--) vecRefine[ii] = 1;
  }

  //**/ ===========================================================
  //**/ actual refinement
  //**/ ===========================================================
  options[0] = 0;
  currNAggr = nAggrs_;
  splitCount = splitSuccess = 0;
  for (ss = 0; ss < nAggrs_; ss++)
  {
    localN = vecAggrCnts_[ss];
    if (localN > maxN) 
    {
      printf("GMetisSampling INTERNAL ERROR (6)\n");
      exit(1);
    }
    if (vecRefine.length() == 0 || vecRefine[ss] == 1) splitCount++;
    if (localN == 1 && (vecRefine.length() == 0 || vecRefine[ss] == 1))
      printf("GMetisSampling::refine INFO- cannot split cell (too small).\n");
    if (localN > 1 && (vecRefine.length() == 0 || vecRefine[ss] == 1))
    {
      if (printLevel_ > 4)
        printf("GMetisSampling::refine - split sample %d\n",ss+1);
      splitSuccess++;
      localNNZ = 0;
      vecSubLabels = vecAggrLabels_[ss];
      vecIA[0] = localNNZ;
      for (ii = 0; ii < localN; ii++)
      {
        rowInd = vecSubLabels[ii];
        for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
        {
          colInd = vecGraphJ_[jj];
          status = binarySearchInt(colInd,vecSubLabels.getIVector(),localN);
          if (status >= 0) vecJA[localNNZ++] = status;
        }
        vecIA[ii+1] = localNNZ;
      }
      if (localNNZ > maxNNZ) 
      {
        printf("GMetisSampling INTERNAL ERROR (7)\n");
        exit(1);
      }

#ifdef HAVE_METIS
      METIS_PartGraphRecursive(&localN,vecIA.getIVector(),vecJA.getIVector(), 
                NULL, NULL,&wgtflag,&numflag,&itwo,options,&edgeCut,
                vecLabels.getIVector());
#else
      printf("GMetisSampling ERROR : METIS not installed.\n");
      exit(1);
#endif

      //**/ labels should be 0 or 1 (bisection)
      count0 = 0;
      for (ii = 0; ii < localN; ii++) if (vecLabels[ii] == 0) count0++;
      count1 = localN - count0;

      //**/ search all cells in the present aggregate to see if 
      //**/ any has been occupied (< 0)
      for (ii = 0; ii < localN; ii++) 
      {
        index = vecAggrLabels_[ss][ii];
        if (vecCellsOccupied_[index] < 0) break;
      }

      //**/ if a sample point does not exist in current aggregate,
      //**/ just put the new points in partition 0
      //**/ if a sample point is found and it is in partition 0, then
      //**/ the new point should be in partition 1
      newCell = 0;
      if (ii < localN && vecLabels[ii] == 0) newCell = 1;

      if (newCell == 0)
      {
        vecAggrCnts_[currNAggr] = count0;
        vecAggrLabels_[currNAggr].setLength(count0); 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == 0) 
          {
            vecAggrLabels_[currNAggr][count++] = vecAggrLabels_[ss][ii];
            if (vecCellsOccupied_[vecAggrLabels_[ss][ii]] < 0)
              vecCellsOccupied_[vecAggrLabels_[ss][ii]] = -(currNAggr+1);
            else
              vecCellsOccupied_[vecAggrLabels_[ss][ii]] = currNAggr;
          }
        }
        vecAggrCnts_[ss] = count1; 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == 1) 
            vecAggrLabels_[ss][count++] = vecAggrLabels_[ss][ii];
        }
      }
      else
      {
        vecAggrCnts_[currNAggr] = count1;
        vecAggrLabels_[currNAggr].setLength(count1); 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == 1) 
          {
            vecAggrLabels_[currNAggr][count++] = vecAggrLabels_[ss][ii];
            if (vecCellsOccupied_[vecAggrLabels_[ss][ii]] < 0)
              vecCellsOccupied_[vecAggrLabels_[ss][ii]] = -(currNAggr+1);
            else
              vecCellsOccupied_[vecAggrLabels_[ss][ii]] = currNAggr;
          }
        }
        vecAggrCnts_[ss] = count0; 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == 0) 
            vecAggrLabels_[ss][count++] = vecAggrLabels_[ss][ii];
        }
      }
      currNAggr++;
    }
  }
           
  if (printLevel_ > 4 && splitSuccess != splitCount)
    printf("GMetisSampling:: number of successful splits = %d (out of %d)\n",
           splitSuccess, splitCount);

  //**/ ===========================================================
  //**/ create refined sample
  //**/ ===========================================================
  psVector  vecSamInpsOld, vecSamOutsOld;
  psIVector vecSamStasOld;
  oldNumSamples = nSamples_;
  vecSamInpsOld = vecSamInps_;
  vecSamOutsOld = vecSamOuts_;
  vecSamStasOld = vecSamStas_;
  nSamples_ = nSamples_ + currNAggr - nAggrs_;
  vecSamInps_.setLength(nSamples_*nInputs_);
  vecSamOuts_.setLength(nSamples_*nOutputs_);
  vecSamStas_.setLength(nSamples_);

  for (ss = 0; ss < oldNumSamples; ss++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
      vecSamInps_[ss*nInputs_+inputID] = vecSamInpsOld[ss*nInputs_+inputID];
    for (outputID = 0; outputID < nOutputs_; outputID++)
      vecSamOuts_[ss*nOutputs_+outputID] =
                    vecSamOutsOld[ss*nOutputs_+outputID];
    vecSamStas_[ss] = vecSamStasOld[ss];
  }

  for (ss = nAggrs_; ss < currNAggr; ss++)
  {
    if (printLevel_ > 4)
      printf("Aggregate %d produces the following sample:\n", ss+1);
    index = (int) (PSUADE_drand() * vecAggrCnts_[ss]);
    if (index == vecAggrCnts_[ss]) index--;
    index = vecAggrLabels_[ss][index];
    if (printLevel_ > 4) printf("   Cell number = %d\n",index+1);
    count = 0;
    while (vecCellsOccupied_[index] < 0 && count < 10)
    {
      index = (int) (PSUADE_drand() * vecAggrCnts_[ss]);
      if (index == vecAggrCnts_[ss]) index--;
      index = vecAggrLabels_[ss][index];
      count++;
    }
    if (vecCellsOccupied_[index] < 0)
    {
      printf("GMetisSampling INFO ERROR (8).\n");
      exit(1);
    }
    vecCellsOccupied_[index] = -(vecCellsOccupied_[index] + 1);
    itmp = index;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      dtmp = ((double) (jtmp + 0.999*PSUADE_drand())) / (double) n1d_;
      vecSamInps_[oldNumSamples*nInputs_+inputID] = dtmp*vecRanges[inputID]+
                                                    vecLBs_[inputID];
      if (printLevel_ > 4)
        printf("  Input %3d = %16.8e\n", inputID+1, 
               vecSamInps_[oldNumSamples*nInputs_+inputID]);
    }
    oldNumSamples++;
  }
  count = currNAggr - nAggrs_;
  nAggrs_ = currNAggr;

  //**/ ===========================================================
  //**/ save GMetis information
  //**/ ===========================================================
  if (changeInfoName_ == 0) fp = fopen("psuadeGMetisInfo", "w");
  else                      fp = fopen("psuadeGMetisInfo.tmp", "w");
  if (fp != NULL)
  {
    fprintf(fp, "%d %d %d\n", nAggrs_, nSamples_, nInputs_);
    for (ii = 0; ii < nAggrs_; ii++)
    {
      fprintf(fp, "%d\n", vecAggrCnts_[ii]);
      for (jj = 0; jj < vecAggrCnts_[ii]; jj++)
      {
        fprintf(fp, "%d ", vecAggrLabels_[ii][jj]);
        if (jj % 10 == 0) fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
    }
    jj = 0;
    for (ii = 0; ii < graphN_; ii++) if (vecCellsOccupied_[ii] < 0) jj++;
    fprintf(fp, "%d\n", jj);
    for (ii = 0; ii < graphN_; ii++)
      if (vecCellsOccupied_[ii] < 0) 
         fprintf(fp, "%7d %7d\n",ii,-(vecCellsOccupied_[ii]+1));
    fclose(fp);
  }

  if (printLevel_ > 0)
  {
    printEquals(PL_INFO, 0);
    printf("GMetisSampling::refine: nAggrs   = %d\n", nAggrs_);
    printf("GMetisSampling::refine: nSamples = %d\n", nSamples_);
    printf("GMetisSampling::refine: nInputs  = %d\n", nInputs_);
    printf("GMetisSampling::refine: nOutputs = %d\n", nOutputs_);
    printEquals(PL_INFO, 0);
  }
  return count;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int GMetisSampling::setParam(char *sparam)
{
  int  ii, curVol, count;
  char winput[501];
  FILE *fp;

  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "reset"))
  {
    fp = fopen("psuadeGMetisInfo", "r");
    if (fp != NULL)
    {
      fclose(fp);
      unlink("psuadeGMetisInfo");
    }
    fp = fopen("psuadeGMetisInfo.tmp", "r");
    if (fp != NULL)
    {
      fclose(fp);
      unlink("psuadeGMetisInfo.tmp");
    }
    return 0;
  }
  else if (!strcmp(winput, "changeInfoName"))
  {
    changeInfoName_ = 1;
    return 0;
  }
  else if (!strcmp(winput, "setUniformRefinement"))
  {
    refineType_ = 0;
    return 0;
  }
  else if (!strcmp(winput, "setAdaptiveRefinementBasedOnErrors"))
  {
    refineType_ = 1;
    return 0;
  }
  else if (!strcmp(winput, "calVolumes"))
  {
    curVol = count = 0;
    for (ii = 0; ii < nAggrs_; ii++)
    {
      if (vecSamOuts_[ii] == 1)
      {
        curVol += vecAggrCnts_[ii];
        count++;
      }
    }
    return curVol;
  }
  else if (!strcmp(winput, "totalVolumes"))
  {
    curVol = 0;
    for (ii = 0; ii < nAggrs_; ii++) curVol += vecAggrCnts_[ii];
    return curVol;
  }
  else if (!strcmp(winput, "setRefineSize"))
  {
    sscanf(sparam, "%s %d", winput, &refineSize_);
    return 0;
  }
  printf("GMetisSampling ERROR:: setParam - invalid param.\n");
  return -1;
}

// ************************************************************************
// equal operator  Modified by Bill Oliver
// ------------------------------------------------------------------------
GMetisSampling& GMetisSampling::operator=(const GMetisSampling & gms)
{
  if (this == &gms) return *this;
  refineType_ = gms.refineType_;
  refineSize_ = gms.refineSize_;
  n1d_ = gms.n1d_;
  nAggrs_ = gms.nAggrs_;
  graphN_ = gms.graphN_;
  vecAggrCnts_ = gms.vecAggrCnts_;
  if (gms.vecAggrLabels_ != NULL)
  {
    vecAggrLabels_ = new psIVector[nAggrs_]; 
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = gms.vecAggrLabels_[ii]; 
  }
  printLevel_ = gms.printLevel_;
  samplingID_ = gms.samplingID_;
  nSamples_ = gms.nSamples_;
  nInputs_ = gms.nInputs_;
  nOutputs_ = gms.nOutputs_;
  randomize_ = gms.randomize_;
  nReplications_ = gms.nReplications_;
  vecLBs_ = gms.vecLBs_;
  vecUBs_ = gms.vecUBs_;
  vecSamInps_ = gms.vecSamInps_;
  vecSamOuts_ = gms.vecSamOuts_;
  vecSamStas_ = gms.vecSamStas_;
  vecGraphI_ = gms.vecGraphI_;
  vecGraphJ_ = gms.vecGraphJ_;
  vecCellsOccupied_ = gms.vecCellsOccupied_;
  return (*this);
}

