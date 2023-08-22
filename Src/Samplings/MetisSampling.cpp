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
// Functions for the METIS sampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <unistd.h>

#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FuncApprox.h"
#include "Mars.h"
#include "MetisSampling.h"

#define PABS(x) ((x) > 0 ? x : -(x))

#ifdef HAVE_METIS
extern "C" 
{
void METIS_PartGraphRecursive(int *, int *, int *, int *, int *,
                              int *, int *, int *, int *, int *, int *);
void METIS_PartGraphKway(int *, int *, int *, int *, int *,
                         int *, int *, int *, int *, int *, int *);
void METIS_PartGraphVKway(int *, int *, int *, int *, int *,
                         int *, int *, int *, int *, int *, int *);
}
#endif

int pruneSet_ = 0;
#define PS_METIS_METHOD 3

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
MetisSampling::MetisSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_METIS;
  refineType_ = 0;
  refineSize_ = 1000000;
  n1d_        = -1;
  nAggrs_     = 0;
  vecAggrLabels_  = NULL;
  changeInfoName_ = 0;
  RSPtr_ = NULL;
  randomize_ = 0;
  threshold_ = 1e35;
}

// ************************************************************************
// copy constructor added by Bill Oliver
// ------------------------------------------------------------------------
MetisSampling::MetisSampling(const MetisSampling & ms) : Sampling()
{
  nSamples_ = ms.nSamples_;
  nInputs_  = ms.nInputs_;
  nOutputs_ = ms.nOutputs_;
  refineType_ = ms.refineType_;
  refineSize_ = ms.refineSize_;
  n1d_ = ms.n1d_;
  nAggrs_ = ms.nAggrs_;
  graphN_ = ms.graphN_;
  if (ms.vecAggrLabels_ != NULL)
  {
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = ms.vecAggrLabels_[ii]; 
  }
  vecAggrCnts_ = ms.vecAggrCnts_;
  printLevel_ = ms.printLevel_;
  samplingID_ = ms.samplingID_;
  randomize_ = ms.randomize_;
  nReplications_ = ms.nReplications_;
  vecLBs_ = ms.vecLBs_;
  vecUBs_ = ms.vecUBs_;
  vecSamInps_ = ms.vecSamInps_;
  vecSamOuts_ = ms.vecSamOuts_;
  vecSamStas_ = ms.vecSamStas_;
  vecGraphI_ = ms.vecGraphI_;
  vecGraphJ_ = ms.vecGraphJ_;
  vecCellsOccupied_ = ms.vecCellsOccupied_;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MetisSampling::~MetisSampling()
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
int MetisSampling::initialize(int initLevel)
{
  int    inputID, ii, jj, kk, itmp, jtmp, nnz, sampleID;
  int    options[10], index, count, saveFlag=1;
#ifdef HAVE_METIS
  int    wgtflag=0, numflag=0, edgeCut=0;
#endif
  double dtmp, expand;
  char   response[1001], pString[1001], filename[1001];
  FILE   *fp;
  psIVector vecIncrs;
  psVector  vecRanges, vecLBs, vecUBs;
 
  //**/ ===========================================================
  //**/ Error checking and clean up
  //**/ ===========================================================
  if( nSamples_ <= 0)
  {
    printf("METISSam ERROR: nSamples_ in file %s line %d is <= 0\n",
           __FILE__, __LINE__);
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("METISSam INIT ERROR : input not set up (nInputs=0).\n");
    printf("INFO: error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }

  //**/ ===========================================================
  //**/ generate grid and the corresponding grid matrix
  //**/ ===========================================================
  if (nInputs_ > 23)
  {
    printf("METISSam ERROR: nInputs > 23 not supported.\n");
    printf("INFO: error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }
  if (nInputs_ == 23) n1d_ =  2;
  if (nInputs_ == 22) n1d_ =  2;
  if (nInputs_ == 21) n1d_ =  2;
  if (nInputs_ == 20) n1d_ =  2;
  if (nInputs_ == 19) n1d_ =  2;
  if (nInputs_ == 18) n1d_ =  2;
  if (nInputs_ == 17) n1d_ =  2;
  if (nInputs_ == 16) n1d_ =  2;
  if (nInputs_ == 15) n1d_ =  2;
  if (nInputs_ == 14) n1d_ =  2;
  if (nInputs_ == 13) n1d_ =  3;
  if (nInputs_ == 12) n1d_ =  3;
  if (nInputs_ == 11) n1d_ =  4;
  if (nInputs_ == 10) n1d_ =  5;
  if (nInputs_ ==  9) n1d_ =  6;
  if (nInputs_ ==  8) n1d_ =  7;
  if (nInputs_ ==  7) n1d_ = 10;
  if (nInputs_ ==  6) n1d_ = 14;
  if (nInputs_ ==  5) n1d_ = 24;
  if (nInputs_ ==  4) n1d_ = 48;
  if (nInputs_ ==  3) n1d_ = 200;
  if (nInputs_ ==  2) n1d_ = 3000;
  if (nInputs_ ==  1) n1d_ = 100000;

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
    printf("METISSam ERROR : nSamples = %d too large.\n",
           nSamples_);
    printf("METISSam INFO: error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
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
  //**/ read previously generated grid file, if any
  //**/ ===========================================================
  char fname[200];
  if (changeInfoName_ == 0) strcpy(fname, "psuadeMetisInfo");
  else                      strcpy(fname, "psuadeMetisInfo.tmp");
  printf("METISSam INIT: looking for %s file...\n", fname);
  fp = fopen(fname, "r");
  if (fp != NULL)
  {
    if (vecAggrLabels_ != NULL)
    {
      for (int ii = 0; ii < nAggrs_; ii++) vecAggrLabels_[ii].clean();
      delete [] vecAggrLabels_;
      vecAggrLabels_ = NULL;
    }
    printf("METISSam INIT: %s file found. Reading it ...\n", 
           fname);
    fscanf(fp, "%d %d %d", &jj, &itmp, &jtmp);
    if (itmp != nSamples_ || jtmp != nInputs_ || jj != nSamples_)
    {
      fclose(fp);
      printf("METISSam INIT: A partition file is found but ");
      printf("the info is inconsistent\n");
      printf("            with this setup (The partition ");
      printf("file name is %s).\n", fname);
      if (itmp != nSamples_ || jj != nSamples_)
        printf("          nSamples : %d != %d.\n",nSamples_,itmp);
      if (jtmp != nInputs_)
        printf("          nInputs  : %d != %d.\n", nInputs_, jtmp);
      snprintf(pString,100,"Overwrite %s? (y or n) ", fname);
      getString(pString, response);
      if (response[0] != 'y')
      {
        printf("METISSam INIT: TERMINATE.\n");
        exit(1);
      }
      else
      {
        unlink(fname);
        fp = NULL;
      }
    }
    if (fp != NULL)
    {
      if (itmp != nSamples_ || jtmp != nInputs_)
      {
        printf("METISSam INIT ERROR: partition file not valid.\n");
        printf("INFO: error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      if (printLevel_ > 0) 
      {
        printf("METISSam INIT: partition file has : \n");
        printf("      - %d subdomains and\n",jj);
        printf("      - %d sample points\n",itmp);
        printf("METISSam INIT: reconstructing the partitioning.\n");
        printf("      - Incoming nSamples : %d.\n", itmp);
        printf("      - Incoming nInputs  : %d.\n", jtmp);
      }
      nAggrs_ = jj;
      vecAggrCnts_.setLength(nAggrs_);
      vecAggrLabels_ = new psIVector[nAggrs_];
      for (ii = 0; ii < nAggrs_; ii++)
      {
        fscanf(fp, "%d", &count);
        if (printLevel_ > 4) 
          printf("METISSam INIT: aggr %8d, size = %d\n", 
                 ii+1, count);
        if (count > 0)
        {
          vecAggrCnts_[ii] = count;
          vecAggrLabels_[ii].setLength(count);
        }
        else vecAggrCnts_[ii] = count = 0;
        for (jj = 0; jj < count; jj++)
        {
          fscanf(fp, "%d", &kk);
          vecAggrLabels_[ii][jj] = kk;
          if (vecAggrLabels_[ii][jj] < 0 || vecAggrLabels_[ii][jj] >= graphN_)
          {
            printf("METISSam INIT ERROR: psuadeMetisInfo ");
            printf("file has invalid info.\n");
            printf("INFO: error occurs in file %s line %d\n",__FILE__, 
                   __LINE__);
            exit(1);
          }
          vecCellsOccupied_[vecAggrLabels_[ii][jj]] = ii;
        }
      }
      fscanf(fp, "%d", &count);
      for (ii = 0; ii < count; ii++)
      {
        fscanf(fp, "%d %d", &jj, &kk);
        if (jj < 0 || jj >= graphN_)
        {
          printf("METISSam INIT ERROR: psuadeMetisInfo file ");
          printf("has invalid info (2).\n");
          printf("INFO: error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        vecCellsOccupied_[jj] = - kk - 1;
        if (printLevel_ > 2)
        {
          printf("METISSam INIT: aggr %d and ",kk+1);
          printf("mesh cell %d is occupied\n", jj+1); 
        }
      }
      fclose(fp);
      saveFlag = 0;
    }
  }
  else
  {
    printf("METISSam INIT: %s file not found - create new one.\n",
           fname);
  }

  //**/ ===========================================================
  //**/ partition (if psuadeMetisInfo not available)
  //**/ ===========================================================
  if (vecAggrCnts_.length() == 0)
  {
    options[0] = 0;
    if (printLevel_ > 1)
      printf("METISSam INIT: calling domain partitioner.\n");

    //**/ if there are 2 inputs and nSamples is a square, generate
    //**/ regular mesh
    if (nInputs_==2 && pow((int)pow(nSamples_,0.5),2) == nSamples_)
    {
      printf("METIS: nInputs = 2, nSamples = n * n.\n");
      int m1d = (int) (sqrt(nSamples_ + 1e-15)); 
      int mstep = n1d_ / m1d;
      printf("METIS: nInputs = 2, nSamples = n * n, mstep = %d.\n",
             mstep);
      if (mstep * m1d < n1d_) mstep++;
      for (ii = 0; ii < n1d_; ii++)
      {
        itmp = (ii / mstep) * m1d;
        for (jj = 0; jj < n1d_; jj++)
          vecCellsOccupied_[ii*n1d_+jj] = itmp + (int) (jj / mstep);
      }
    }
    else
    {
#ifdef HAVE_METIS
      //**/ call partitioner - at the end vecCellsOccupied will have
      //**/ aggregate numbers for each cell
#if PS_METIS_METHOD == 1
      METIS_PartGraphRecursive(&graphN_, vecGraphI_.getIVector(), 
          vecGraphJ_.getIVector(), NULL, NULL, &wgtflag,&numflag,
          &nSamples_,options,&edgeCut,vecCellsOccupied_.getIVector());
#elif PS_METIS_METHOD == 2
      METIS_PartGraphKway(&graphN_, vecGraphI_.getIVector(), 
          vecGraphJ_.getIVector(), NULL, NULL, &wgtflag,&numflag,
          &nSamples_,options,&edgeCut,vecCellsOccupied_.getIVector());
#elif PS_METIS_METHOD == 3
      int volume = 1;
      METIS_PartGraphVKway(&graphN_, vecGraphI_.getIVector(), 
          vecGraphJ_.getIVector(), NULL, NULL, &wgtflag,&numflag,
          &nSamples_,options,&volume,vecCellsOccupied_.getIVector());
#endif
#else
      printf("METISSam INIT ERROR : METIS not installed.\n");
      printf("INFO: error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
#endif
    }
    if (printLevel_ > 1)
      printf("METISSam INIT: subdomains created.\n");

    //**/ count the number of cells in each aggregate
    nAggrs_ = nSamples_;
    vecAggrCnts_.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++) vecAggrCnts_[ii] = 0;
    for (ii = 0; ii < graphN_; ii++)
    {
      if (vecCellsOccupied_[ii] < 0 || vecCellsOccupied_[ii] >= nAggrs_)
      {
        printf("METISSam INIT ERROR.\n");
        printf("INFO: error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      vecAggrCnts_[vecCellsOccupied_[ii]]++;
    }  
    //**/ create labels: for each aggregate, register which cells
    //**  belong to it 
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (ii = 0; ii < nAggrs_; ii++)
    {
      if (vecAggrCnts_[ii] <= 0)
      {
        printf("METISSam INIT ERROR (2).\n");
        printf("INFO: error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
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
        printf("METISSam INIT ERROR (3).\n");
        printf("INFO: error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      vecAggrLabels_[index][vecAggrCnts_[index]] = ii;  
      vecAggrCnts_[index]++;
    }
  }

  //**/ ===========================================================
  //**/ save newly generated partition file 
  //**/ ===========================================================
  if (saveFlag == 1)
  {
    if (changeInfoName_ == 0) fp = fopen("psuadeMetisInfo", "w");
    else                      fp = fopen("psuadeMetisInfo.tmp", "w");
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
    }
  }
  if (initLevel != 0) return 0;

  //**/ ===========================================================
  //**/ provide option to cover boundary
  //**/ ===========================================================
  char *inStr, winput1[500], winput2[500];
  expand = 0.0;
  if (psConfig_.SamExpertModeIsOn())
  {
    printf("For the Metis sampling, you can divert more ");
    printf("data points to the surface\n");
    printf("of the parameter space by sampling from the ");
    printf("expanded parameter space\n");
    printf("folowed by projecting the sample points back ");
    printf("to the original parameter\n");
    printf("space. In order to do so, an expansion ratio ");
    printf("needs to be given. An\n");
    printf("expansion ratio of 0.0 means no expansion. ");
    printf("Normally, the expansion\n");
    printf("ratio should be no more than 0.1-0.2.\n");
    snprintf(pString,100,"Enter an expansion ratio (default=0): ");
    expand = -0.1;
    while (expand < 0.0) expand = getDouble(pString);
  }
  else
  {
    inStr = psConfig_.getParameter("METIS_expand_ratio");
    if (inStr != NULL)
    {
      sscanf(inStr, "%s %s %lg\n", winput1, winput2, &expand);
      if (winput2[0] != '=')
      {
        printf("METIS ERROR: in reading config info : %s\n", inStr);
        expand = 0.0;
      }
      else if (expand < 0.0 || expand >= 1.0)
      {
        printf("METIS ERROR: in reading config expand ratio: %e\n",
               expand);
        expand = 0.0;
      }
      else printf("METIS : expansion ratio set to %e\n", expand);
    }
  }

  //**/ ===========================================================
  //**/ generate sample
  //**/ ===========================================================
  allocSampleData();
  vecRanges.setLength(nInputs_);
  vecLBs.setLength(nInputs_);
  vecUBs.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = (vecUBs_[inputID]-vecLBs_[inputID])*0.5*expand;
    vecLBs[inputID] = vecLBs_[inputID] - vecRanges[inputID];
    vecUBs[inputID] = vecUBs_[inputID] + vecRanges[inputID];
  }
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs[inputID] - vecLBs[inputID];

  for (sampleID = 0; sampleID < nSamples_; sampleID++)
  {
    if ((randomize_ & 1) == 1)
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
      if ((randomize_ & 1) == 1)
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

  if (printLevel_ > 4)
  {
    printf("METISSam INIT: nSamples = %d\n", nSamples_);
    printf("METISSam INIT: nInputs  = %d\n", nInputs_);
    printf("METISSam INIT: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("METISSam INIT: randomize on\n");
    else printf("METISSam INIT: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
       printf("    METISSam input %3d = [%e %e]\n", inputID+1,
              vecLBs_[inputID], vecUBs_[inputID]);
  }
  return 0;
}

// ************************************************************************
// refine the sample space
//**/ Need to call initialize first to load the psuadeMetisInfo file
//**/ randFlag - if sampleErrors != NULL ==> randomly select cell for each
//**/            aggregate 
//**/            if sampleErrors == NULL ==> randomly select aggregate
// ------------------------------------------------------------------------
int MetisSampling::refine(int nLevels, int randFlag, double threshold,
                          int nSamples, double *sampleErrors)
{
  int    inputID, ii, jj, count, localN, count1, iOne=1;
  int    rowInd, colInd, localNNZ, index, count0, kk, options[10];
#ifdef HAVE_METIS
  int    wgtflag=0, numflag=0, edgeCut=0, itwo=2;
#endif
  int    currNAggr, newCell, oldNumSamples, mode;
  int    status, outputID, ss, marsCnt, aggr1, splitCount;
  int    splitSuccess, maxCellSize, minCellSize, avgCellSize, cellCnt=0;
  int    aggr2, rowInd2, colInd2, localN2, ii2, jj2, limit;
  double ddmax, ddmin, ddmean, thresh, ddsd, dtmp;
  char   pString[500];
  FILE   *fp;
  FuncApprox *faPtr;
  psIVector vecSubLabels, vecTags, vecTags2; 
  psIVector vecOldSamStas, vecSubLabels2;
  psVector  vecOldSamInps, vecOldSamOuts;
  psVector  vecDiffs, vecMarsIn, vecMarsOut;

  //**/ ===========================================================
  //**/ error checking
  //**/ ===========================================================
  if (nSamples != 0 && nSamples != nAggrs_)
  {
    printf("METISSam refine ERROR: nSamples != nAggregates\n");
    printf("     This may be due to wrong psuadeMetisInfo file.\n");
    printf("     Delete this file and re-do.\n");
    exit(1);
  }
  if (vecCellsOccupied_.length() == 0 || vecAggrLabels_ == NULL)
  {
    printf("METISSam refine ERROR: call initialize first\n");
    printf("                Consult PSUADE developer.\n");
    printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }
  if (printLevel_ > 0)
  {
    printDashes(PL_INFO, 0);
    printf("METISSam refine INFO: Sample information before refinement \n");
    printf("     nSamples = %d\n", nSamples_);
    printf("     nInputs  = %d\n", nInputs_);
    printf("     nOutputs = %d\n", nOutputs_);
    printDashes(PL_INFO, 0);
  }

  //**/ set lower and upper bounds (no expansion - expand = 0.0)
  double expand=0.0;
  psVector vecRanges, vecLBs, vecUBs;
  vecRanges.setLength(nInputs_);
  vecLBs.setLength(nInputs_);
  vecUBs.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = (vecUBs_[inputID] - vecLBs_[inputID])*expand;
    vecLBs[inputID] = vecLBs_[inputID] - vecRanges[inputID];
    vecUBs[inputID] = vecUBs_[inputID] + vecRanges[inputID];
  }
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    vecRanges[inputID] = vecUBs[inputID] - vecLBs[inputID];
    if (vecRanges[inputID] <= 0.0)
    {
      printf("METISSam refine ERROR: lbound/ubound mismatch.\n");
      printf("                Consult PSUADE developer.\n");
      printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
    }
  }

  //**/ ===========================================================
  //**/ if not already done, put existing sample points into cells
  //**/ (if vecCellsOccupied_[ii] < 0, it means cell ii contains
  //**/ one sample point already and it belong to aggregate
  //**/ -(vecCelsOccupied_[ii]+1) with 0-based)
  //**/ ===========================================================
  int numCellOccupied = 0, itmp, jtmp;
  for (ii = 0; ii < graphN_; ii++) 
  {
    if (vecCellsOccupied_[ii] < 0) 
    {
      if (printLevel_ > 3)
        printf("METISSam INFO: There is a sample in point aggr %d\n",
               -(vecCellsOccupied_[ii]+1)+1);
      numCellOccupied++;
    }
  }
  if (numCellOccupied == 0)
  {
    //**/ if no cell occupied, inject sample inputs into cells
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
      if (itmp < 0 || itmp >= graphN_)
      {
        printf("METISSam refine INTERNAL ERROR (1).\n");
        printf("         Consult PSUADE developer.\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      vecCellsOccupied_[itmp] = -(vecCellsOccupied_[itmp] + 1); 
      numCellOccupied++;
    }
  }
  if (numCellOccupied != nSamples_)
  {
    //**/ information stored in vecCellOccupied_ does not match
    //**/ with incoming sample information (vecSamInps_)
    printf("METISSam refine ERROR: Sample size mismatch\n");
    printf("     between psuadeMetisInfo and current data set.\n");
    printf("     Delete psuadeMetisInfo and re-do.\n");
    printf("     nSamples expected = %d\n",nSamples_);
    printf("     nSamples detected = %d\n",numCellOccupied);
    printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }
   
  //**/ ===========================================================
  //**/ make sure everything is set up okay
  //**/ ===========================================================
  psIVector vecChecks;
  vecChecks.setLength(nSamples_);
  for (ii = 0; ii < graphN_; ii++) 
  {
    if (vecCellsOccupied_[ii] < 0) 
    {
      jj = -(vecCellsOccupied_[ii]+1);
      vecChecks[jj] = 1;
    }
  }
  itmp = vecChecks.sum();
  if (itmp != nSamples_)
  {
    printf("METISSam refine ERROR: The sample for refinement ");
    printf("does not match the");
    printf("                information given in psuadeMetisInfo.\n");
    printf("                nSamples expected = %d\n", nSamples_);
    printf("                nSamples detected = %d\n", itmp);
    printf("NOTE: Maybe psuadeMetisInfo has been modified?\n");
    printf("INFO: error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }

  //**/ ===========================================================
  //**/ expand the label array to accommodate newly refined points
  //**/ (the maximum number of new points cannot be > 2*nAggrs as
  //**/ refinement ratio is 2)
  //**/ ===========================================================
  psIVector vecTmpCnts, *vecTmpLabels=NULL;
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

  //**/ ==========================================================
  //**/ create a node (or cell) to aggregate array
  //**/ so cell ii belongs to aggregate vecNode2Aggr[ii]
  //**/ So every cell in the entire mesh is mapped to an aggregate
  //**/ ==========================================================
  psIVector vecNode2Aggr;
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
        printf("METISSam refine ERROR: index out of bound.\n");
        printf("         index = %d (should be in [0,%d])\n",
               rowInd, graphN_-1);
        printf("         Aggregate = %d and local cell %d\n",ii,jj);
        printf("         Please consult PSUADE developers.\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      vecNode2Aggr[rowInd] = ii;
    }
  }

  //**/ ==========================================================
  //**/ check to make sure each node belongs to an aggregate
  //**/ ==========================================================
  for (ii = 0; ii < graphN_; ii++)
  {
    if (vecNode2Aggr[ii] == -1)
    {
      printf("METISSam refine ERROR: node2Aggr not correct.\n");
      printf("         Cell %d has no corresponding aggregate.\n",
             ii+1);
      printf("         Please consult PSUADE developers.\n");
      printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
    }
  }

  //**/ ===========================================================
  //**/ preparation for refinement: allocate localIA, localJA
  //**/ (these variables are used for aggregate splitting)
  //**/ ===========================================================
  int maxN = 0;
  psIVector vecLabels;
  psIVector vecLocalIA, vecLocalJA;
  for (ss = 0; ss < nAggrs_; ss++)
    if (vecAggrCnts_[ss] > maxN) maxN = vecAggrCnts_[ss];
  vecLabels.setLength(maxN);
  vecLocalIA.setLength(2*maxN+1);
  int maxNNZ = maxN * (2 * nInputs_ + 1);
  vecLocalJA.setLength(maxNNZ);

  //**/ ===========================================================
  //**/ refinement type error checking
  //**/ ===========================================================
  int useRS=0;
  psVector vecSamErrs;
  double *localSampErrors = sampleErrors;
  //**/ if refinement type is adaptive and no sample errors given,
  //**/ see if a response surface pointer has been given. If not,
  //**/ flag an error (the idea is that if no sample error is
  //**/ given, it can be estimated from the response surface)
  if (refineType_ == 1 && sampleErrors == NULL)
  {
    printf("METISSam refine INFO: adaptive refinement ");
    printf("requested but sample errors\n");
    printf("                      have not been given.\n");
    if (RSPtr_ == NULL)
    {
      printf("METISSam refine ERROR: No sample errors or RS given.\n");
      printf("         ==> cannot proceed. Consult PSUADE developers.\n");
      printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
    }
    else
    {
      useRS = 1;
      vecSamErrs.setLength(nAggrs_);
      localSampErrors = vecSamErrs.getDVector();
      if (printLevel_ > 0)
      {
        printf("METISSam refine INFO: Use RS evaluator ");
        printf("to compute errors.\n");
      }
    }
  }

  //**/ ===========================================================
  //**/ adaptive refinement based on given RS evaluator 
  //**/ This method evaluates prediction uncertainties at every
  //**/ cell in each aggregate (instead of a random cell in each
  //**/ aggregate and thus more time-consuming) and selects the
  //**/ cell in each aggregate that gives the maximum uncertainty
  //**/ as a candidate new point. At the end, vecVT[ss] will have 
  //**/ the prediction uncertainty for the cell with maximum
  //**/ prediction uncertainty in aggregate ss.
  //**/ ===========================================================
  psVector  vecXT, vecYT, vecVT;
  psIVector vecSelectedNodes, vecRefineFlags;
  //**/ Note: don't allocate vecRefineFlags yet 

  if (refineType_ == 1 && useRS == 1)
  {
    if (printLevel_ > 0)
    {
      printf("METISSam refine INFO: Use response surface analysis.\n");
      printf("                Maximum number of new points = %d.\n",
             refineSize_);
      printf("NOTE: Computed error to be scaled by the cell size.\n");
    }
    //**/ ========================================================
    //**/ uniform refinement and generate errors 
    //**/ (maxN has maximum number of cells among all aggregates)
    //**/ ========================================================
    double errVal;
    vecSelectedNodes.setLength(nAggrs_);
    vecRefineFlags.setLength(nAggrs_);
    vecXT.setLength(maxN*nInputs_);
    vecYT.setLength(maxN);
    vecVT.setLength(maxN);
    options[0] = 0;
    for (ss = 0; ss < nAggrs_; ss++)
    {
      if (printLevel_ > 3)
        printf("Processing aggregate %d (of %d)\n", ss+1, nAggrs_);
      localN = vecAggrCnts_[ss];
      if (localN > maxN) 
      {
        printf("METISSam refine ERROR (2): ");
        printf("              Consult PSUADE developers\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      if (localN == 1)
      {
        printf("METISSam refine ERROR: Cannot split ");
        printf("(aggregate has only 1 cell)\n");
        printf("                Please consult PSUADE developers.\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
      else
      {
        if (printLevel_ > 4)
        {
          printf("METISSam refine INFO: split aggregate %d\n",ss+1);
          printf("         Aggregate %d has %d cells before splitting.\n",
                 ss+1,localN);
        }
        //**/ put aggregate matrix into vecLocalIA and vecLocalJA
        localNNZ = 0;
        vecSubLabels = vecAggrLabels_[ss];
        vecLocalIA[0] = localNNZ;
        for (ii = 0; ii < localN; ii++)
        {
          rowInd = vecSubLabels[ii];
          for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
          {
            colInd = vecGraphJ_[jj];
            status = binarySearchInt(colInd,vecSubLabels.getIVector(),
                                     localN);
            if (status >= 0) vecLocalJA[localNNZ++] = status;
          }
          if (ii >= maxN)
          {
            printf("METISSam refine ERROR: Consult PSUADE developers\n");
            printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                   __LINE__);
            exit(1);
          }
          vecLocalIA[ii+1] = localNNZ;
        }
        if (localNNZ > maxNNZ) 
        {
          printf("METISSam refine ERROR: Consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
 
#ifdef HAVE_METIS
        //**/ split the aggregate into 2
#if PS_METIS_METHOD == 1
        METIS_PartGraphRecursive(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#elif PS_METIS_METHOD == 2
        METIS_PartGraphKway(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#elif PS_METIS_METHOD == 3
        //int volume = 0;
        //METIS_PartGraphVKway(&localN, vecLocalIA.getIVector(), 
        //   vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
        //   options,&volume,vecLabels.getIVector());
        //**/ Cannot use VKway because it performs partitioning randomly
        //**/ and so when it is to be done a second time later, there 
        //**/ will be conflicts
        METIS_PartGraphKway(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#endif
#else
        printf("METISSam refine ERROR: METIS not installed.\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
#endif
        //**/ vecLabels has either 0 or 1 for binary split
        count0 = 0;
        for (ii = 0; ii < localN; ii++) 
          if (vecLabels[ii] == 0) count0++;
        count1 = localN - count0;
 
        //**/ examine the current aggregate to see which cell is occupied
        //**/ (or which cell the current sample point is in)
        for (ii = 0; ii < localN; ii++) 
        {
          index = vecAggrLabels_[ss][ii];
          if (vecCellsOccupied_[index] < 0) break;
        }
        if (ii >= localN)
        {
          printf("METISSam refine ERROR: ");
          printf("Aggregate has no sample point.\n");
          printf("         Please consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }

        //**/ if occupied cell has label 0, new cell will be 
        //**/ selected from cells with label = 1
        newCell = 0;
        if (vecLabels[ii] == 0) newCell = 1;
 
        //**/ evaluate all cells in the 0-th or 1-st splitted aggregate 
        //**/ first put sample inputs of all cell centers in vecXT
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == newCell) 
          {
            itmp = vecAggrLabels_[ss][ii];
            for (jj = 0; jj < nInputs_; jj++)
            {
              jtmp = itmp % n1d_;
              itmp = itmp / n1d_;
              dtmp = ((double) (jtmp + 0.5)) / (double) n1d_;
              vecXT[count*nInputs_+jj] = dtmp * vecRanges[jj] +
                                         vecLBs[jj];
            }
            count++;
          }
        }
        if (newCell == 0 && count != count0)
        {
          printf("METISSam refine ERROR: ");
          printf("Consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        else if (newCell == 1 && count != count1)
        {
          printf("METISSam refine ERROR: ");
          printf("Consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        if (newCell == 0) count = count0;
        else              count = count1;

        //**/ evaluate every cell with response surface
        //**/ vecVT will have prediction uncertainties of all
        //**/ candidate cells 
        RSPtr_->evaluatePointFuzzy(count,vecXT.getDVector(),
                   vecYT.getDVector(), vecVT.getDVector());

        //**/ find which cell has the maximum prediction error
        errVal = - PSUADE_UNDEFINED;
        index = -1;
        for (ii = 0; ii < count; ii++)
        {
          if (vecVT[ii] > errVal) 
          {
            errVal = vecVT[ii];
            index = ii;
          }
        }

        //**/ identify the cell with maximum error
        //**/ if threshold has not been set
        if (threshold_ >= 1e34)
        {
          jtmp = 0;
          for (ii = 0; ii < localN; ii++)
          {
            if (vecLabels[ii] == newCell)
            {
              if (jtmp == index) break;
              jtmp++;
            }
          }
          printf("METIS refine INFO: aggr %4d cell %7d identified for ", 
                 ss+1, vecAggrLabels_[ss][ii]+1);
          printf("possible refinement\n");
          vecSelectedNodes[ss] = vecAggrLabels_[ss][ii];
          vecRefineFlags[ss] = 1;
        }
        else
        //**/ if threshold is set, look for the cell in the 
        //**/ aggregate that has neighbors on the opposite
        //**/ side of the threshold but above it and closest
        //**/ to it
        {
          int    bdryPt = 0, currCell, whichCell=-1;
          double curMin;
          psVector vecInds;
          vecInds.setLength(nInputs_);
          curMin = 1e35;
          for (ii = 0; ii < localN; ii++) 
          {
            //**/ only examine cells that are in new aggregate
            if (vecLabels[ii] == newCell) 
            {
              //**/ get cell number
              currCell = vecAggrLabels_[ss][ii];
              //**/ if the prediction mean is > threshold
              //**/ check if it is a boundary point
              if (vecYT[ii] > threshold_)
              {
                bdryPt = 0;
                //**/ extract coordinate indices for this cell
                //**/ and put them in vecInds
                itmp = currCell;
                for (jj = 0; jj < nInputs_; jj++)
                {
                  jtmp = itmp % n1d_;
                  vecInds[jj] = jtmp;
                  itmp = itmp / n1d_;
                }
                //**/ examine neighbors
                for (jj = 0; jj < nInputs_; jj++)
                {
                  //**/ examine left neighbor
                  dtmp  = pow(1.0 * n1d_, 1.0 * jj);
                  index = currCell - int(dtmp + 1e-15);
                  //**/ search to see if left neighbor is in this
                  //**/ aggregate. if not, do not check
                  for (kk = 0; kk < localN; kk++)
                    if (vecAggrLabels_[ss][kk] == index && 
                        vecYT[kk] <= threshold_)
                      break;
                  if (kk < localN) bdryPt++;
                  //**/ examine right neighbor
                  dtmp  = pow(1.0 * n1d_, 1.0 * jj);
                  index = currCell + int(dtmp + 1e-15);
                  for (kk = 0; kk < localN; kk++)
                    if (vecAggrLabels_[ss][kk] == index && 
                        vecYT[kk] <= threshold_)
                      break;
                  if (kk < localN) bdryPt++;
                }
                //**/ if bdryPt > 0 ==> boundary point
                //**/ if its mean is closer to threshold, keep it
                if (vecYT[currCell] < curMin)
                {
                  curMin = vecYT[currCell];
                  whichCell = ii;
                }
              }
            }
          }
          if (whichCell >= 0)
          {
            vecSelectedNodes[ss] = vecAggrLabels_[ss][whichCell];
            vecRefineFlags[ss] = 1;
            errVal = curMin;
          }
        }
       
        //**/ store information to be used later 
        if (vecRefineFlags[ss] == 1 && 
            vecCellsOccupied_[vecSelectedNodes[ss]] < 0)
        {
          printf("METISSam refine ERROR: cell %d ",
                 vecSelectedNodes[ss]+1);
          printf("has been occupied by\n");
          printf("aggregate %d (1-based).\n", 
                 -(vecCellsOccupied_[vecSelectedNodes[ss]]+1)+1);
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        localSampErrors[ss] = errVal;
      }
    }
    vecXT.clean();
    vecYT.clean();
    vecVT.clean();
    //**/ at this point vecSelectedNodes and localSampErrors have the
    //**/ information for next step
  }

  //**/ ===========================================================
  //**/ refinement based on sample prediction errors
  //**/ sort the errors and select the ones with the largest errors
  //**/ ===========================================================
  psVector vecSortList1, vecSortList2;
  if (refineType_ == 1 && localSampErrors != NULL)
  {
    printf("METISSam refine INFO: Use sample errors to ");
    printf("guide refinement.\n");
    printf("                Maximum number of new points = %d.\n",
           refineSize_);
    printf("NOTE: Errors are to be scaled by the cell size.\n");

    //**/ ========================================================
    //**/ sort aggregate errors 
    //**/ ========================================================
    vecSortList1.setLength(nAggrs_);
    vecSortList2.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++)
    {
      localN = vecAggrCnts_[ii];
      vecSortList1[ii] = PABS(localSampErrors[ii]);
      vecSortList1[ii] *= pow(1.0*localN,1.0/nInputs_);
    }
    for (ii = 0; ii < nAggrs_; ii++) vecSortList2[ii] = (double) ii;
    sortDbleList2(nAggrs_,vecSortList1.getDVector(),
                  vecSortList2.getDVector());
    cellCnt = 0;

    //**/ ========================================================
    //**/ tag the aggregate for refinement (set RefineFlag to 1)
    //**/ ========================================================
    vecRefineFlags.setLength(nAggrs_);
    for (ii = nAggrs_-1; ii >= 0; ii--)
    {
      index = (int) vecSortList2[ii];
      localN = vecAggrCnts_[index];
      if (localSampErrors[index] != 0.0) 
      {
        if (printLevel_ > 2)
        { 
          printf("METISSam refine INFO: aggregate #%d ", index+1);
          printf("selected for refinement\n");
          printf("                error = %13.5e (nCells = %d)\n",
                 localSampErrors[index], localN);
        }
        if (vecRefineFlags[index] == 0)
        {
          cellCnt++;
          vecRefineFlags[index] = 1;
          if (cellCnt >= refineSize_) break;
        }
        if (cellCnt >= refineSize_) break;
      }
      if (cellCnt >= refineSize_) break;
    }
    //**/ vecRefineFlags array is ready for actual refinement
  }

  //**/ ===========================================================
  //**/ refinement based on gradient with neighbors
  //**/ ===========================================================
  if (refineType_ == 2)
  {
    if (printLevel_ > 0)
    {
      printf("METISSam refine INFO: Based on gradient with neighbors.\n");
      printf("         Maximum allowable number of new points = %d.\n",
             refineSize_);
    }
    vecTags.setLength(nAggrs_);
    for (ii = 0; ii < nAggrs_; ii++) vecTags[ii] = 0;

    vecDiffs.setLength(nAggrs_);
    //**/ for each aggregate
    for (ss = 0; ss < nAggrs_; ss++)
    {
      vecDiffs[ss] = 0;
      localN = vecAggrCnts_[ss];
      vecSubLabels = vecAggrLabels_[ss];

      //**/ compute the mean of this aggregate and its neighbors
      //**/ (in order to reach its neighbors, it has to examine
      //**/  all cells internal to the aggregate)
      ddmean = vecSamOuts_[ss];
      count0 = 1;
      for (ii = 0; ii < localN; ii++)
      {
        rowInd = vecSubLabels[ii];
        for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
        {
          colInd = vecGraphJ_[jj];
          aggr1 = vecNode2Aggr[colInd];
          if (aggr1 != ss && vecTags[aggr1] == 0)
          {
            vecTags[aggr1] = 1;
            ddmean += vecSamOuts_[aggr1];
            count0++;
            localN2 = vecAggrCnts_[aggr1];
            vecSubLabels2 = vecAggrLabels_[aggr1];
            for (ii2 = 0; ii2 < localN2; ii2++)
            {
              rowInd2 = vecSubLabels2[ii2];
              for (jj2 = vecGraphI_[rowInd2];jj2 < vecGraphI_[rowInd2+1];
                   jj2++)
              {
                colInd2 = vecGraphJ_[jj2];
                aggr2 = vecNode2Aggr[colInd2];
                if (aggr2 != ss && vecTags[aggr2] == 0)
                {
                  vecTags[aggr2] = 1;
                  ddmean += vecSamOuts_[aggr2];
                  count0++;
                }
              }
            }
          }
        }
      }
      ddmean /= count0;
      //**/ compute standard deviations
      ddsd = 0.0;
      if (count0 > 1)
      {
        ddsd += (vecSamOuts_[ss] - ddmean) * (vecSamOuts_[ss] - ddmean);
        for (ii = 0; ii < localN; ii++)
        {
          rowInd = vecSubLabels[ii];
          for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
          {
            colInd = vecGraphJ_[jj];
            aggr1 = vecNode2Aggr[colInd];
            if (aggr1 != ss && vecTags[aggr1] == 1)
            {
              dtmp = vecSamOuts_[aggr1];
              ddsd += (dtmp - ddmean) * (dtmp - ddmean);
              vecTags[aggr1] = 0;
              localN2 = vecAggrCnts_[aggr1];
              vecSubLabels2 = vecAggrLabels_[aggr1];
              for (ii2 = 0; ii2 < localN2; ii2++)
              {
                rowInd2 = vecSubLabels2[ii2];
                for (jj2=vecGraphI_[rowInd2];jj2<vecGraphI_[rowInd2+1];jj2++)
                {
                  colInd2 = vecGraphJ_[jj2];
                  aggr2 = vecNode2Aggr[colInd2];
                  if (aggr2 != ss && vecTags[aggr2] == 1)
                  {
                    vecTags[aggr2] = 0;
                    dtmp = vecSamOuts_[aggr2];
                    ddsd += (dtmp - ddmean) * (dtmp - ddmean);
                  }
                }
              }
            }
          }
        }
      }
      if (count0 > 0) vecDiffs[ss] = sqrt(ddsd) / count0;
      //**/ at this point, vecDiffs has all aggregate gradients

      if (printLevel_ > 4 && nInputs_ == 2)
      {
        for (ii = 0; ii < nAggrs_; ii++) vecTags[ii] = 0;
        printf(" ..... Sample %6d inputs : %e %e = %e\n", ss+1, 
               vecSamInps_[ss*nInputs_],vecSamInps_[ss*nInputs_+1],
               vecSamOuts_[ss]);
        for (ii = 0; ii < localN; ii++)
        {
          rowInd = vecSubLabels[ii];
          for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
          {
            colInd = vecGraphJ_[jj];
            aggr1 = vecNode2Aggr[colInd];
            if (aggr1 != ss && vecTags[aggr1] == 0)
            {
              printf(" .....          neighbor : %e %e = %e\n", 
                 vecSamInps_[aggr1*nInputs_],vecSamInps_[aggr1*nInputs_+1],
                 vecSamOuts_[aggr1]);
              vecTags[aggr1] = 1;
            }
          }
        }
      }
    }

    double diffCnt = 0.0;
    for (ss = 0; ss < nAggrs_; ss++) diffCnt += vecDiffs[ss];
    maxCellSize = cellCnt = 0;
    avgCellSize = 0;
    minCellSize = 1000000000;
    //**/ diffCnt == 0 or nAggrs means all sample outputs are
    //**/ either all 0's or all 1's. If sample outputs are 
    //**/ either 0 or 1, then refinement is along interfaces
    //**/ (refer to a_refine_metis in PsuadeInterpreter.cpp)
    if (diffCnt == 0 || diffCnt == nAggrs_)
    { 
      //**/ tag all new aggregate to be refined
      for (ss = 0; ss < nAggrs_; ss++) 
      {
        vecRefineFlags[ss] = 0;
        localN = vecAggrCnts_[ss];
        if (localN > 1)
        {
          vecRefineFlags[ss] = 1;
          maxCellSize = (localN > maxCellSize) ? localN : maxCellSize;
          minCellSize = (localN < maxCellSize) ? localN : minCellSize;
          avgCellSize += localN;
          cellCnt++;
        }
      }
    }
    else
    {
      limit = 0;
      mode = 2;
      if (mode == 2)
      {
        //**/ scale errors with cell size
        for (ss = 0; ss < nAggrs_; ss++) 
          vecDiffs[ss] *= vecAggrCnts_[ss];
      }
#if 0
      //**/ not used any more (to be deleted later)
      //**/ but it is to give priority for cells that cannot be
      //**/ interpolated well by neighboring cells
      else if (mode == 3 || mode == 4)
      {
        if (mode == 4)
        {
          limit = nInputs_ + 1 + nInputs_ * (nInputs_ + 1) / 2;
        }
        vecMarsIn.setLength(nAggrs_*nInputs_);
        vecMarsOut.setLength(nAggrs_);
        vecTags2.setLength(nAggrs_);
        for (ss = 0; ss < nAggrs_; ss++)
        {
          for (ii = 0; ii < nAggrs_; ii++) vecTags2[ii] = 0;
          vecTags2[ss] = 1;
          marsCnt = 0;
          localN = vecAggrCnts_[ss];
          vecSubLabels = vecAggrLabels_[ss];
          for (ii = 0; ii < localN; ii++)
          {
            rowInd = vecSubLabels[ii];
            for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
            {
              colInd = vecGraphJ_[jj];
              aggr1 = vecNode2Aggr[colInd];
              if (vecTags2[aggr1] == 0)
              {
                for (kk = 0; kk < nInputs_; kk++)
                  vecMarsIn[marsCnt*nInputs_+kk] =
                         vecSamInps_[aggr1*nInputs_+kk];
                vecMarsOut[marsCnt++] = vecSamOuts_[aggr1];
                vecTags2[aggr1] = 1;
              }
            }
          }
          if (marsCnt < limit)
          {
            for (ii = 0; ii < localN; ii++)
            {
              rowInd = vecSubLabels[ii];
              for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
              {
                colInd = vecGraphJ_[jj];
                aggr1 = vecNode2Aggr[colInd];
                localN2 = vecAggrCnts_[aggr1];
                vecSubLabels2 = vecAggrLabels_[aggr1];
                for (ii2 = 0; ii2 < localN2; ii2++)
                {
                  rowInd2 = vecSubLabels2[ii2];
                  for (jj2=vecGraphI_[rowInd2];jj2<vecGraphI_[rowInd2+1];jj2++)
                  {
                    colInd2 = vecGraphJ_[jj2];
                    aggr2 = vecNode2Aggr[colInd2];
                  }
                  if (vecTags2[aggr2] == 0)
                  {
                    for (kk = 0; kk < nInputs_; kk++)
                      vecMarsIn[marsCnt*nInputs_+kk] = 
                                     vecSamInps_[aggr2*nInputs_+kk];
                    vecMarsOut[marsCnt++] = vecSamOuts_[aggr2];
                    vecTags2[aggr2] = 1;
                  }
                }
              }
            }
          }
          if (printLevel_ > 2)
            printf("METISSam INFO: refine aggr %d (%d): sample size = %d\n",
                   ss+1,nAggrs_,marsCnt);
          if (mode == 3)
               faPtr = genFA(0, nInputs_, iOne, marsCnt);
          else faPtr = genFA(2, nInputs_, iOne, marsCnt);
          faPtr->setBounds(vecLBs.getDVector(), vecUBs.getDVector());
          faPtr->setOutputLevel(1);
          faPtr->initialize(vecMarsIn.getDVector(), vecMarsOut.getDVector());
          double *dPtr = vecSamInps_.getDVector();
          vecDiffs[ss] = faPtr->evaluatePoint(&(dPtr[ss*nInputs_]));
          vecDiffs[ss] = PABS(vecDiffs[ss] - vecSamOuts_[ss]);
          delete faPtr;
        }
        dtmp = 0.0;
        for (ss = 0; ss < nAggrs_; ss++) dtmp += vecDiffs[ss];
        printf("Average interpolation error = %e\n", dtmp/nAggrs_);
      }
#endif

      thresh = 0.0;
      ddmax = -PSUADE_UNDEFINED;
      ddmin =  PSUADE_UNDEFINED;
      for (ii = 0; ii < nAggrs_; ii++)
      {
        if (vecDiffs[ii] > ddmax) ddmax = vecDiffs[ii];
        if (vecDiffs[ii] < ddmin) ddmin = vecDiffs[ii];
      }
      thresh = 0.0;
      if (psConfig_.SamExpertModeIsOn() && pruneSet_ == 0)
      {
        printf("METISSam refine INFO: \n");
        printf("  Maximum discrepancy between aggregates = %e\n",ddmax);
        printf("  Minimum discrepancy between aggregates = %e\n",ddmin);
        fp = fopen("arsmPDF.m","w");
        // add check for NULL by Bill Oliver
        if(fp != NULL)
        {
          fprintf(fp,"A = [\n");
          for (ii = 0; ii < nAggrs_; ii++)
            fprintf(fp,"%e\n",vecDiffs[ii]);
          fprintf(fp,"];\n");
          fprintf(fp,"hist(A,10)\n");
          fclose(fp);
        }
        printf("A PDF for error has been given to you in arsmPDF.m.\n");
        printf("Default threshold for pruning = 0.\n");
        snprintf(pString,100,"Enter thresh to prune refinement candidates : ");
        thresh = getDouble(pString);
        if (thresh < 0.0)
        {
          printf("WARNING: threshold < 0 not allowed. Reset to 0.\n");
          thresh = 0.0;
        }
        pruneSet_ = 1;
      }
      for (ii = 0; ii < nAggrs_; ii++)
        if (vecDiffs[ii] < thresh) vecDiffs[ii] = 0.0;

      vecSortList2.setLength(nAggrs_);
      for (ii = 0; ii < nAggrs_; ii++) vecSortList2[ii] = (double) ii;
      sortDbleList2(nAggrs_,vecDiffs.getDVector(),vecSortList2.getDVector());
      cellCnt = 0;
      ddmax = -PSUADE_UNDEFINED;
      vecRefineFlags.setLength(nAggrs_);
      for (ii = nAggrs_-1; ii >= 0; ii--)
      {
        jj = (int) vecSortList2[ii];
        localN = vecAggrCnts_[jj];
        if (localN > 1 && vecDiffs[ii] > 0.0)
        {
          vecRefineFlags[jj] = 1;
          maxCellSize = (localN > maxCellSize) ? localN : maxCellSize;
          minCellSize = (localN < maxCellSize) ? localN : minCellSize;
          avgCellSize += localN;
          if (vecDiffs[ii] > ddmax) ddmax = vecDiffs[ii];
          cellCnt++;
        }
        if (cellCnt >= refineSize_) break;
      }
      //printf("METISSam:: maximum neighbor discrepancy = %e\n",ddmax);
    }
    printf("METISSam refine INFO: Number of new sample points = %d.\n",
           cellCnt);
    if (printLevel_ > 4)
    {
      if (maxCellSize != 0)
        printf("METISSam refine INFO: maximum resolution = %d \n",
               maxCellSize);
      if (minCellSize != 1000000000)
        printf("METISSam refine INFO: minimum resolution = %d \n",
               minCellSize);
      if (cellCnt != 0)
        printf("METISSam refine INFO: average resolution = %e (%d)\n", 
               1.0*avgCellSize/cellCnt, cellCnt);
    }
  }

  //**/ ===========================================================
  //**/ if pick randomly from different aggregate
  //**/ (now vecRefineFlags are ready, and also vecSelectedNodes)
  //**/ ===========================================================
  if (randFlag == 1 && sampleErrors != NULL)
  {
    vecRefineFlags.setLength(nAggrs_);
    ss = 0;
    while (ss < refineSize_ && ss < nAggrs_)
    {
      ii = PSUADE_rand() % nAggrs_;
      if (vecRefineFlags[ii] != 1) 
      {
        vecRefineFlags[ii] = 1;
        ss++;
      }
    }
  }

  //**/ ===========================================================
  //**/ actual refinement 
  //**/ (now vecRefineFlags are ready, and also vecSelectedNodes)
  //**/ ===========================================================
  options[0] = 0;
  currNAggr = nAggrs_;
  splitCount = splitSuccess = 0;
  for (ss = 0; ss < nAggrs_; ss++)
  {
    localN = vecAggrCnts_[ss];
    if (localN > maxN) 
    {
      printf("METISSam refine ERROR: ");
      printf("Please consult PSUADE developers\n");
      printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
    }
    if (localN == 1)
    {
      printf("METISSam refine ERROR: cell too small to split.\n");
    }
    else if (vecRefineFlags.length() == 0 ||
             (vecRefineFlags.length() > 0 && vecRefineFlags[ss] == 1)) 
    {
      splitCount++;
      if (printLevel_ > 2)
        printf("METISSam refine INFO: Splitting sample %d\n",ss+1);
      splitSuccess++;
      localNNZ = 0;
      vecSubLabels = vecAggrLabels_[ss];
      vecLocalIA[0] = localNNZ;
      for (ii = 0; ii < localN; ii++)
      {
        rowInd = vecSubLabels[ii];
        for (jj = vecGraphI_[rowInd]; jj < vecGraphI_[rowInd+1]; jj++)
        {
          colInd = vecGraphJ_[jj];
          status = binarySearchInt(colInd,vecSubLabels.getIVector(),localN);
          if (status >= 0) vecLocalJA[localNNZ++] = status;
        }
        if (ii >= maxN)
        {
          printf("METISSam refine ERROR: ");
          printf("Please consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        vecLocalIA[ii+1] = localNNZ;
      }
      if (localNNZ > maxNNZ) 
      {
        printf("METISSam refine ERROR: ");
        printf("Please consult PSUADE developers\n");
        printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
               __LINE__);
        exit(1);
      }
 
#ifdef HAVE_METIS
#if PS_METIS_METHOD == 1
      METIS_PartGraphRecursive(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#elif PS_METIS_METHOD == 2
      METIS_PartGraphKway(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#elif PS_METIS_METHOD == 3
      //int volume = 0;
      //METIS_PartGraphKway(&localN, vecLocalIA.getIVector(), 
      //     vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
      //     options,&volume,vecLabels.getIVector());
      METIS_PartGraphKway(&localN, vecLocalIA.getIVector(), 
           vecLocalJA.getIVector(), NULL, NULL, &wgtflag,&numflag,&itwo,
           options,&edgeCut,vecLabels.getIVector());
#endif
#else
      printf("METISSam refine ERROR: METIS not installed.\n");
      printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
             __LINE__);
      exit(1);
#endif

      count0 = 0;
      for (ii = 0; ii < localN; ii++) if (vecLabels[ii] == 0) count0++;
      count1 = localN - count0;
 
      for (ii = 0; ii < localN; ii++) 
      {
        index = vecAggrLabels_[ss][ii];
        if (vecCellsOccupied_[index] < 0) break;
      }
      newCell = 0;
      if (vecLabels[ii] == 0) newCell = 1;
 
      //**/ if new cell is 0, then create the next aggregate with it
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
            vecCellsOccupied_[vecAggrLabels_[ss][ii]] = currNAggr;
          }
        }
        if (count != count0)
        {
          printf("METISSam refine ERROR: Please consult developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        vecAggrCnts_[ss] = count1; 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
          if (vecLabels[ii] == 1) 
            vecAggrLabels_[ss][count++] = vecAggrLabels_[ss][ii];
        if (count != count1)
        {
          printf("METISSam refine ERROR: Please consult developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
      }
      else
      //**/ if new cell is 1, then create the next aggregate with it
      {
        vecAggrCnts_[currNAggr] = count1;
        vecAggrLabels_[currNAggr].setLength(count1); 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
        {
          if (vecLabels[ii] == 1) 
          {
            vecAggrLabels_[currNAggr][count++] = vecAggrLabels_[ss][ii];
            vecCellsOccupied_[vecAggrLabels_[ss][ii]] = currNAggr;
          }
        }
        if (count != count1)
        {
          printf("METISSam refine ERROR: Please consult ");
          printf("PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
        vecAggrCnts_[ss] = count0; 
        count = 0;
        for (ii = 0; ii < localN; ii++) 
          if (vecLabels[ii] == 0) 
            vecAggrLabels_[ss][count++] = vecAggrLabels_[ss][ii];
        if (count != count0)
        {
          printf("METISSam refine ERROR: ");
          printf("Consult PSUADE developers\n");
          printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
                 __LINE__);
          exit(1);
        }
      }
      currNAggr++;
    }
  }
  if (printLevel_ > 1 && splitSuccess != splitCount)
    printf("METISSam refine INFO: no. of successful splits = %d (of %d)\n",
           splitSuccess, splitCount);

  //**/ ===========================================================
  //**/ create refined sample
  //**/ ===========================================================
  oldNumSamples = nSamples_;
  vecOldSamInps = vecSamInps_;
  vecOldSamOuts = vecSamOuts_;
  vecOldSamStas = vecSamStas_;
  nSamples_ = currNAggr;
  allocSampleData();
  for (ss = 0; ss < oldNumSamples; ss++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
      vecSamInps_[ss*nInputs_+inputID] = vecOldSamInps[ss*nInputs_+inputID];
    for (outputID = 0; outputID < nOutputs_; outputID++)
      vecSamOuts_[ss*nOutputs_+outputID] =
                 vecOldSamOuts[ss*nOutputs_+outputID];
    vecSamStas_[ss] = vecOldSamStas[ss];
  }
  int refineTrack=0;
  for (ss = oldNumSamples; ss < currNAggr; ss++)
  {
    if (vecSelectedNodes.length() > 0)
    {
      while (vecRefineFlags[refineTrack] == 0)
        refineTrack++;
      index = vecSelectedNodes[refineTrack];
      refineTrack++;
    }
    else
    {
      if ((randomize_ & 1) == 1)
           index = (int) (PSUADE_drand() * vecAggrCnts_[ss]);
      else index = vecAggrCnts_[ss] / 2;
      if (index == vecAggrCnts_[ss]) index--;
      index = vecAggrLabels_[ss][index];
    }
    if (printLevel_ > 2)
    {
      printf("METISSam refine: Creating new sample %d\n",ss+1);
      printf("                 cell selected = %d\n",index+1);
    }
    vecCellsOccupied_[index] = -(vecCellsOccupied_[index] + 1);
    itmp = index;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      jtmp = itmp % n1d_;
      itmp = itmp / n1d_;
      if ((randomize_ & 1) == 1)
           dtmp = ((double) (jtmp + 0.999*PSUADE_drand()))/(double) n1d_;
      else dtmp = ((double) (jtmp + 0.5)) / (double) n1d_;
      vecSamInps_[ss*nInputs_+inputID] = dtmp * vecRanges[inputID] +
                                         vecLBs[inputID];
      if (vecSamInps_[ss*nInputs_+inputID] < vecLBs_[inputID])
        vecSamInps_[ss*nInputs_+inputID] = vecLBs_[inputID];
      if (vecSamInps_[ss*nInputs_+inputID] > vecUBs_[inputID])
        vecSamInps_[ss*nInputs_+inputID] = vecUBs_[inputID];
    }
  }
  nAggrs_ = currNAggr;

  //**/ check goodness of assignment
  vecChecks.setLength(nSamples_);
  for (ii = 0; ii < graphN_; ii++) 
  {
    if (vecCellsOccupied_[ii] < 0) 
    {
      jj = -(vecCellsOccupied_[ii]+1);
      vecChecks[jj] = 1;
    }
  }
  itmp = vecChecks.sum();
  if (itmp != nAggrs_)
  {
    printf("METISSam refine ERROR: Fail final aggregate checking\n");
    printf("         nAggrs expected = %d\n",nAggrs_);
    printf("         nAggrs detected = %d\n",itmp);
    printf("INFO: Error occurs in file %s line %d\n",__FILE__, 
           __LINE__);
    exit(1);
  }
  else printf("METISSam refine INFO: Passed final checksum.\n");

  //**/ ===========================================================
  //**/ save Metis information
  //**/ ===========================================================
  if (changeInfoName_ == 0) fp = fopen("psuadeMetisInfo", "w");
  else                      fp = fopen("psuadeMetisInfo.tmp", "w");
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
    printDashes(PL_INFO, 0);
    printf("METISSam refine INFO: Sample information after refinement \n");
    printf("     nSamples = %d\n", nSamples_);
    printf("     nInputs  = %d\n", nInputs_);
    printf("     nOutputs = %d\n", nOutputs_);
    //if (randomize_ != 0)
    //     printf("METISSam refine: randomize on\n");
    //else printf("METISSam refine: randomize off\n");
    printDashes(PL_INFO, 0);
  }
  return 0;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int MetisSampling::setParam(char * sparam)
{
  int  ii, curVol, sID, localN, sID2, rowInd;
  char winput[501];
  FILE *fp;
  psIVector vecSubLabels, vecGridFlags;

  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "reset"))
  {
    if (changeInfoName_ == 0) fp = fopen("psuadeMetisInfo", "r");
    else                      fp = fopen("psuadeMetisInfo.tmp", "r");
    if (fp != NULL)
    {
      fclose(fp);
      if (changeInfoName_ == 0) unlink("psuadeMetisInfo");
      else                      unlink("psuadeMetisInfo.tmp");
    }
  }
  else if (!strcmp(winput, "changeInfoName"))
  {
    changeInfoName_ = 1;
  }
  else if (!strcmp(winput, "setUniformRefinement"))
  {
    refineType_ = 0;
  }
  else if (!strcmp(winput, "setAdaptiveRefinementBasedOnErrors"))
  {
    refineType_ = 1;
  }
  else if (!strcmp(winput, "setAdaptiveRefinementBasedOnOutputs"))
  {
    refineType_ = 2;
  }
  else if (!strcmp(winput, "calVolumes"))
  {
    curVol = 0;
    for (ii = 0; ii < nAggrs_; ii++)
      if (vecSamOuts_[ii] == 1) curVol += vecAggrCnts_[ii];
    return curVol;
  }
  else if (!strcmp(winput, "totalVolumes"))
  {
    curVol = 0;
    for (ii = 0; ii < nAggrs_; ii++) curVol += vecAggrCnts_[ii];
    return curVol;
  }
  else if (!strcmp(winput, "useThreshold"))
  {
    sscanf(sparam, "%s %lg", winput, &threshold_);
    return threshold_;
  }
  else if (!strcmp(winput, "setRefineSize"))
  {
    sscanf(sparam, "%s %d", winput, &refineSize_);
  }
  else if (!strcmp(winput, "genMeshPlot") && nInputs_ == 2)
  {
    vecGridFlags.setLength(graphN_);
    for (sID = 0; sID < nAggrs_; sID++)
    {
      localN = vecAggrCnts_[sID];
      vecSubLabels = vecAggrLabels_[sID];
      for (ii = 0; ii < localN; ii++)
      {
        rowInd = vecSubLabels[ii];
        //if (vecSamOuts_[sID] == 0) vecGridFlags[rowInd] = 0;
        //else                       vecGridFlags[rowInd] = 1;
        vecGridFlags[rowInd] = sID + 1;
      }
    }
    fp = fopen("metisMeshPlot.m", "w");
    if (fp != NULL)
    {
      fprintf(fp, "meshA = [ \n");
      for (sID = n1d_-1; sID >= 0; sID--)
      {
        for (sID2 = 0; sID2 < n1d_; sID2++)
          fprintf(fp, "%d ", vecGridFlags[sID*n1d_+sID2]);
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "x1 = [%e %e];\n",vecLBs_[0],vecUBs_[0]);
      fprintf(fp, "x2 = [%e %e];\n",vecLBs_[0],vecLBs_[1]);
      fprintf(fp, "meshA = meshA';\n");
      fprintf(fp, "imagesc(x1,x2,meshA)\n");
      fprintf(fp, "set(gca,'YDir','normal');\n");
      fclose(fp);
      printf("METISSam INFO: Partitioning plot is in metisMeshPlot.m.\n");
    }
    return 0;
  }
  return 0;
}

// ************************************************************************
// pass in a response surface pointer
// ------------------------------------------------------------------------
int MetisSampling::setParam(int argc, char **argv)
{
  if (argc == 2 && (!strcmp(argv[0], "setRSPTR")))
    RSPtr_ = (FuncApprox *) argv[1];;
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
MetisSampling& MetisSampling::operator=(const MetisSampling & ms)
{
  if (this == &ms) return *this;
  nSamples_ = ms.nSamples_;
  nInputs_ = ms.nInputs_;
  nOutputs_ = ms.nOutputs_;
  refineType_ = ms.refineType_;
  refineSize_ = ms.refineSize_;
  n1d_ = ms.n1d_;
  nAggrs_ = ms.nAggrs_;
  graphN_ = ms.graphN_;
  if (ms.vecAggrLabels_ != NULL)
  {
    vecAggrLabels_ = new psIVector[nAggrs_];
    for (int ii = 0; ii < nAggrs_; ii++)
      vecAggrLabels_[ii] = ms.vecAggrLabels_[ii]; 
  }
  vecAggrCnts_ = ms.vecAggrCnts_;
  printLevel_ = ms.printLevel_;
  samplingID_ = ms.samplingID_;
  randomize_ = ms.randomize_;
  nReplications_ = ms.nReplications_;
  vecLBs_ = ms.vecLBs_;
  vecUBs_ = ms.vecUBs_;
  vecSamInps_ = ms.vecSamInps_;
  vecSamOuts_ = ms.vecSamOuts_;
  vecSamStas_ = ms.vecSamStas_;
  vecGraphI_ = ms.vecGraphI_;
  vecGraphJ_ = ms.vecGraphJ_;
  vecCellsOccupied_ = ms.vecCellsOccupied_;
  return (*this);
}

