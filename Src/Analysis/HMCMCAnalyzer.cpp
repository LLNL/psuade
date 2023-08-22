// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free 
// Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class HMCMCAnalyzer
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONG
// DATE   : 2016
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
//**/using namespace std;

#include "HMCMCAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PDFManager.h"
#include "Psuade.h"
#include "Sampling.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
HMCMCAnalyzer::HMCMCAnalyzer(): nInputs_(0) 
{
  setName("HMCMC");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
HMCMCAnalyzer::~HMCMCAnalyzer()
{
}

// ************************************************************************
// perform MCMC analysis 
// ------------------------------------------------------------------------
double HMCMCAnalyzer::analyze()
{
  int    ii, jj, kk, status=0, nGroups, nSystems;
  int    **groupInfo, nParams;
  double mean, stdev, dOne=1.0, ddata;
  FILE   *fp=NULL;
  psVector  vecSysData;
  psIVector vecGroupSizes, vecGroupIDs;
  psMatrix  corMat;

  //**/ ---------------------------------------------------------------
  //**/ clean up 
  //**/ ---------------------------------------------------------------
  VecMeans_.clean();
  VecSigmas_.clean();
  VecMostLikelyInps_.clean();
  VecMostLikelyOuts_.clean();

  //**/ ---------------------------------------------------------------
  //**/ read group information
  //**/ ==> nGroups, nSystems, vecSysData, vecGroupSizes, groupInfo, 
  //**/     vecGroupIDs
  //**/ ---------------------------------------------------------------
  status = readUserSpec(&nGroups, &nSystems, vecSysData, vecGroupSizes, 
                        &groupInfo, vecGroupIDs);
  if (status < 0)
  {
    printf("ERROR in HMCMC - abort.\n");
    return -1.0;
  }

  //**/ ---------------------------------------------------------------
  //**/ initial set up 
  //**/ There are nGroups * 2 + 1 parameters:
  //**/    groups means, group std dev + overall mean
  //**/ ---------------------------------------------------------------
  nParams = nGroups * 2;
  VecMeans_.setLength(nParams);
  VecSigmas_.setLength(nParams);
  VecMostLikelyInps_.setLength(nParams);
  VecMostLikelyOuts_.setLength(1);

  //**/ ---------------------------------------------------------------
  //**/ set priors for aggregate hyperparameters
  //**/ group mean  discrepancies: normal N(0,sigma)
  //**/ group stdev discrepancies: Gamma(alpha,beta)
  //**/ overal mean              : Uniform
  //**/ ---------------------------------------------------------------
  psIVector vecInpPDFsH;
  psVector  vecInpMeansH, vecInpStdvsH;;
  vecInpPDFsH.setLength(3);
  vecInpMeansH.setLength(3);
  vecInpStdvsH.setLength(3);

  status = getHierarchicalPriors(nGroups,vecGroupSizes.getIVector(),
                  groupInfo, vecSysData.getDVector(), 
                  &mean, &stdev, vecInpPDFsH.getIVector(), 
                  vecInpMeansH.getDVector(),vecInpStdvsH.getDVector());
  //**/ distribution of group means  = N(0,sigma)
  //**/ distribution of group std    = L(alpha,beta)
  //**/ distribution of overall mean = U(a, b)

  //**/ ---------------------------------------------------------------
  //**/ generate a large sample for the actual hyperparameters
  //**/ ---------------------------------------------------------------
  int maxIterations=10000000;
  psIVector vecPPDFs;
  psVector  vecLBH, vecUBH, vecSamH, vecPMeans, vecPStdvs;;
  PDFManager *pdfmanH = new PDFManager();

  corMat.setDim(nParams,nParams); 
  for (ii = 0; ii < nParams; ii++) corMat.setEntry(ii,ii, dOne);

  vecPPDFs.setLength(nParams);
  vecPMeans.setLength(nParams);
  vecPStdvs.setLength(nParams);
  vecLBH.setLength(nParams); 
  vecUBH.setLength(nParams); 
  for (ii = 0; ii < nGroups; ii++)
  {
    vecPPDFs[ii]  = vecInpPDFsH[0];
    vecPMeans[ii] = vecInpMeansH[0];
    vecPStdvs[ii] = vecInpStdvsH[0];
    vecPPDFs[ii+nGroups]  = vecInpPDFsH[1];
    vecPMeans[ii+nGroups] = vecInpMeansH[1];
    vecPStdvs[ii+nGroups] = vecInpStdvsH[1];
  }
  pdfmanH->initialize(nParams,vecPPDFs.getIVector(),vecPMeans.getDVector(),
                      vecPStdvs.getDVector(), corMat, NULL, NULL);
  //**/ normal distribution
  for (ii = 0; ii < nGroups; ii++)
  {
    vecLBH[ii] = vecInpMeansH[0] - 3 * vecInpStdvsH[0];
    vecUBH[ii] = vecInpMeansH[0] + 3 * vecInpStdvsH[0];
    //**/ lognormal distribution
    vecLBH[ii+nGroups] = 1e-8;
    vecUBH[ii+nGroups] = 3 * vecInpStdvsH[1];
  }
  vecSamH.setLength(maxIterations*nParams);

  //**/ ---------------------------------------------------------------
  //  perform inference
  //**/ ---------------------------------------------------------------
  int    ss, hh, index, passCnt=0, loopCnt;
  double exponent, denom, maxLikelihood=0, dtemp;
  psVector vecInferStore, vecParams, vecXmax, vecGDelta, vecGSigma;

  vecInferStore.setLength(maxIterations);
  vecXmax.setLength(nParams);
  vecParams.setLength(maxIterations*nParams);
  vecGDelta.setLength(nGroups);
  vecGSigma.setLength(nGroups);

  printOutTS(PL_INFO, "Generating a large sample ... \n");
  pdfmanH->genSample(maxIterations, vecSamH, vecLBH, vecUBH);
  double cmax=-PSUADE_UNDEFINED, cmin=PSUADE_UNDEFINED;
  for (ii = 0; ii < nParams; ii++)
  {
    cmax=-PSUADE_UNDEFINED; cmin=PSUADE_UNDEFINED;
    for (hh = 0; hh < maxIterations; hh++)
    {
      if (vecSamH[hh*nParams+ii] > cmax) cmax = vecSamH[hh*nParams+ii];
      if (vecSamH[hh*nParams+ii] < cmin) cmin = vecSamH[hh*nParams+ii];
    }
    printf("Parameter %d max/min = %12.4e %12.4e\n",ii+1,cmax,cmin);
  }
  //**/ solution for the example
  //**/vecSamH[(maxIterations-1)*nParams] = -0.3;
  //**/vecSamH[(maxIterations-1)*nParams+1] = 0;
  //**/vecSamH[(maxIterations-1)*nParams+2] = 0.3;
  //**/vecSamH[(maxIterations-1)*nParams+3] = 0.1;
  //**/vecSamH[(maxIterations-1)*nParams+4] = 0.1;
  //**/vecSamH[(maxIterations-1)*nParams+5] = 0.1;
  printOutTS(PL_INFO, "HMCMC Inference begins ... \n");
  fflush(stdout);
  //**/ for each outer sample point, run inner iterations

  loopCnt = 0;
  for (hh = 0; hh < maxIterations; hh++)
  {
    //**/ fetch sample
    for (ii = 0; ii < nGroups; ii++) 
    {
      vecGDelta[ii] = vecSamH[hh*nParams+ii];
      vecGSigma[ii] = vecSamH[hh*nParams+ii+nGroups];
    }
    //if (hh % maxIterations == 0) printOutTS(PL_INFO, ".");
    //**/ compute likelihood
    exponent = 0.0;
    for (ii = 0; ii < nSystems; ii++)
    {
      index = vecGroupIDs[ii];
      ddata = vecSysData[ii] - mean - vecGDelta[index];
      ddata /= vecGSigma[index];
      exponent += 0.5 * pow(ddata, 2.0);
    }
    //**/ compute likelihood
    ddata = exp(-exponent);

    //**/ store the sample
    for (ii = 0; ii < nGroups; ii++) 
    {
      vecParams[hh*nParams+ii]   = vecGDelta[ii];
      vecParams[hh*nParams+ii+3] = vecGSigma[ii];
    }
    vecInferStore[hh] = ddata;

    if (ddata > maxLikelihood)
    {
      maxLikelihood = ddata;
      for (ii = 0; ii < nGroups; ii++) 
      {
        vecXmax[ii] = vecGDelta[ii];
        vecXmax[ii+3] = vecGSigma[ii];
      }
      printf("\nNew best point found: max likelihood = %e at\n", 
             maxLikelihood);
      for (ii = 0; ii < nGroups; ii++)
        printf("   Group %3d mean = %16.8e\n",ii+1, vecXmax[ii]+mean);
      for (ii = 0; ii < nGroups; ii++)
        printf("   Group %3d stdv = %16.8e\n",ii+1,vecXmax[nGroups+ii]);
    }
    if ((hh+1) % (maxIterations/100) == 0)
    {
      fp = fopen("psuade_stop","r");
      if (fp != NULL)
      {
        printf("psuade_stop file found ==> terminate.\n");
        fclose(fp);
        break;
      }
    }
    if (hh % (maxIterations/10) == 0 && (hh > 0))
    {
      denom = 0.0;
      for (ss = 0; ss < maxIterations; ss++) denom += vecInferStore[ss];
      passCnt = 0;
      printf("At iteration %d\n", hh);
      for (ii = 0; ii < nParams; ii++)
      {
        ddata = 0.0;
        for (ss = 0; ss < maxIterations; ss++) 
          ddata += vecParams[ss*nParams+ii]*vecInferStore[ss]/denom;
        dtemp = 0.0;
        for (ss = 0; ss < maxIterations; ss++) 
          dtemp += pow(vecParams[ss*nParams+ii]-ddata,2.0)*
                   vecInferStore[ss]/denom;
        VecMeans_[ii] = VecMeans_[ii] * loopCnt + ddata;
        loopCnt++;
        VecMeans_[ii] /= (double) loopCnt;
        if (ii < nGroups)
          printOutTS(PL_INFO,
             "HMCMC: Group %3d mean (mean,stdev) = %12.4e %12.4e\n",
             ii+1,VecMeans_[ii]+mean,sqrt(dtemp));
        else if (ii < 2*nGroups)
          printOutTS(PL_INFO,
             "HMCMC: Group %3d stdv (mean,stdev) = %12.4e %12.4e\n",
             ii+1-3,VecMeans_[ii],sqrt(dtemp));
        VecSigmas_[ii] = sqrt(dtemp);
        if (status == 1)
        {
          passCnt++;
          printf("MCMC input %3d converged.\n",ii+1);
        }
      }
      printf("Best found so far (max likelihood = %12.4e)\n",
             maxLikelihood);
      for (ii = 0; ii < nGroups; ii++)
        printf("   Group %3d mean = %16.8e\n",ii+1, vecXmax[ii]+mean);
      for (ii = 0; ii < nGroups; ii++)
        printf("   Group %3d stdv = %16.8e\n",ii+1,vecXmax[nGroups+ii]);
      if (passCnt == nParams) break;
    }
  }
  genMatlabFile(maxIterations, nParams, vecParams.getDVector(), 
                vecInferStore.getDVector());

  //**/ ---------------------------------------------------------------
  //**/ final processing 
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO,"HMCMC: Population mean = %e\n", mean);
  for (ii = 0; ii < nGroups; ii++)
  {
    printOutTS(PL_INFO,"HMCMC: Group %3d mean at likelihood peak = %e\n",
               ii+1, vecXmax[ii]+mean);
    VecMostLikelyInps_[ii] = vecXmax[ii];
  }
  for (ii = 0; ii < nGroups; ii++)
  {
    printOutTS(PL_INFO,"HMCMC: Group %3d stdv at likelihood peak = %e\n",
               ii+1, vecXmax[ii+nGroups]);
    VecMostLikelyInps_[ii] = vecXmax[ii];
  }
  denom = 0.0;
  for (ss = 0; ss < maxIterations; ss++) denom += vecInferStore[ss];
  VecMostLikelyOuts_[0] = maxLikelihood / denom;

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nGroups; ii++) delete [] groupInfo[ii];
  delete [] groupInfo;
  delete pdfmanH;
  return 0.0;
}

// ************************************************************************
// read specfile
// ------------------------------------------------------------------------
int HMCMCAnalyzer::readUserSpec(int *nGroups_in, int *nSystems_in, 
                       psVector &vecSysData, psIVector &vecGroupSizes,
                       int ***groupInfo_in, psIVector &vecGroupIDs)
{
  int    ii, jj, kk, leng, nGroups, nSystems, **groupInfo;
  double ddata;
  char   pString[1000], sysFile[2000];
  FILE   *fp;
  //**/ ---------------------------------------------------------------
  // read group information 
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printf("*    HMCMC performed single mixed effect analysis\n");
  printEquals(PL_INFO, 0);
  printf("Information needed:\n");
  printf(" - the number of groups\n");
  printf(" - 'systems' that belong to each group\n");
  printEquals(PL_INFO, 0);
  printf("The file that contains groups and systems information should \n");
  printf("have the following format:\n"); 
  printf("Line 1: <numGroups> <numSystems> (numSystems > numGroups)\n"); 
  printf("Next <numSystems> lines: \n");
  printf("<System number> <System value>   (number should be in order)\n");
  printf(" ... \n");
  printf("Next Line: 1 <nSys>   (<nSys> is no. of system in Group 1)\n"); 
  printf("<sysID> <sysID> ...   (System numbers for Group 1)\n");
  printf("Next Line: 2 <nSys>   (<nSys> is no. of system in Group 2)\n"); 
  printf("<sysID> <sysID> ...   (System numbers for Group 2)\n");
  printf(" ... \n");
  printf("Example: (3 groups, 9 systems)\n");
  printf("3 9\n");
  printf("1 0.29\n");
  printf("2 0.61\n");
  printf("3 0.89\n");
  printf("4 0.59\n");
  printf("5 0.30\n");
  printf("6 0.91\n");
  printf("7 0.60\n");
  printf("8 0.31\n");
  printf("9 0.90\n");
  printf("1 3   (group 1 has 3 systems)\n");
  printf("1 5 8 (group 1 has systems 1, 5, and 8)\n");
  printf("2 3   (group 2 has 3 systems)\n");
  printf("2 4 7 (group 2 has systems 2, 4, and 7)\n");
  printf("3 3   (group 3 has 3 systems)\n");
  printf("3 6 9 (group 3 has systems 3, 6, and 9)\n");
  snprintf(pString,100,"Enter the file name for group information: ");
  getString(pString, sysFile);
  leng = strlen(sysFile);
  sysFile[leng-1] = '\0';
  fp = fopen(sysFile, "r");
  if (fp == NULL)
  {
    printf("ERROR: file %s not found\n", sysFile);
    return -1;
  }
  fscanf(fp,"%d %d", &nGroups, &nSystems);
  if (nSystems <= 0 || nGroups <= 0)
  {
    printf("ERROR: nSystems and nGroups should be > 0\n"); 
    return -1;
  }
  if (nSystems <= nGroups)
  {
    printf("ERROR: nSystems %d should be > nGroups %d\n", 
           nSystems, nGroups);
    return -1;
  }
  vecSysData.setLength(nSystems);
  for (ii = 0; ii < nSystems; ii++)
  {
    fscanf(fp,"%d", &kk);
    if (kk != ii+1)
    {
      printf("ERROR: invalid system number %d detected.\n",kk);
      printf("       Expected number = %d.\n",ii+1);
      exit(1);
    }
    fscanf(fp,"%lg", &ddata);
    vecSysData[ii] = ddata;
    printf("System %5d has data = %e\n", ii+1, vecSysData[ii]);
  }
  groupInfo = new int*[nGroups];
  vecGroupSizes.setLength(nGroups);
  for (ii = 0; ii < nGroups; ii++)
  {
    fscanf(fp,"%d", &kk);
    if (kk != ii+1)
    {
      printf("ERROR: invalid group number %d detected.\n",kk);
      printf("       Expected number = %d.\n",ii+1);
      delete [] groupInfo;
      return -1;
    }
    fscanf(fp,"%d", &jj);
    vecGroupSizes[ii] = jj;
    if (vecGroupSizes[ii] <= 1)
    {
      printf("ERROR: group size should be > 1.\n");
      printf("       Size read = %d.\n",vecGroupSizes[ii]);
      delete [] groupInfo;
      return -1;
    }
    groupInfo[ii] = new int[vecGroupSizes[ii]];
    for (kk = 0; kk < vecGroupSizes[ii]; kk++)
    {
      fscanf(fp,"%d", &(groupInfo[ii][kk]));
      if (groupInfo[ii][kk] < 1 || groupInfo[ii][kk] > nSystems)
      {
        printf("ERROR: invalid group member %d.\n",groupInfo[ii][kk]);
        printf("       Expected: between 1 and %d\n",nSystems);
        for (jj = 0; jj < nGroups; jj++)
          if (groupInfo[ii] != NULL) delete [] groupInfo[ii];
        delete [] groupInfo;
        return -1;
      }
      groupInfo[ii][kk]--;
      printf("Group %4d has system %d\n", ii+1, groupInfo[ii][kk]+1);
    }
  }
  fclose(fp);
  kk = 0;
  for (ii = 0; ii < nGroups; ii++) kk += vecGroupSizes[ii];
  if (kk != nSystems)
  {
    printf("ERROR: sum of all group sizes is %d\n",kk);
    printf("       Expected sum = %d\n",nSystems);
    for (ii = 0; ii < nGroups; ii++) delete [] groupInfo[ii];
    delete [] groupInfo;
    exit(1);
  }
  vecGroupIDs.setLength(nSystems);
  for (ii = 0; ii < nSystems; ii++) vecGroupIDs[ii] = -1; 
  for (ii = 0; ii < nGroups; ii++) 
    for (kk = 0; kk < vecGroupSizes[ii]; kk++) 
      vecGroupIDs[groupInfo[ii][kk]] = ii;
  kk = 0;
  for (ii = 0; ii < nSystems; ii++) if (vecGroupIDs[ii] != -1) kk++;
  if (kk != nSystems)
  {
    printf("ERROR: %d systems have been left out (of %d).\n",
           kk, nSystems);
    for (ii = 0; ii < nGroups; ii++) delete [] groupInfo[ii];
    delete [] groupInfo;
    exit(1);
  }
  (*nGroups_in) = nGroups;
  (*nSystems_in) = nSystems; 
  (*groupInfo_in) = groupInfo;
  return 0;
}
 
// ************************************************************************
// get information 
// ------------------------------------------------------------------------
int HMCMCAnalyzer::getHierarchicalPriors(int nGroups, int *groupSizes,
                   int **groupInfo, double *sysData, double *mean,
                   double *stdev, int *inputPDFsH, double *inputMeansH, 
                   double *inputStdevH)
{
  int    ii, jj, nSystems, index;
  double pmean, pstdev, meanBias, sdBias;
  psVector vecGMeans, vecGStdvs;

  //**/ compute overal statistics: mean and standard deviation
  nSystems = 0;
  for (ii = 0; ii < nGroups; ii++) nSystems += groupSizes[ii];
  pmean = 0.0;
  for (ii = 0; ii < nSystems; ii++) pmean += sysData[ii];
  pmean /= (double) nSystems;
  (*mean) = pmean;
  pstdev = 0.0;
  for (ii = 0; ii < nSystems; ii++) 
    pstdev += pow(sysData[ii] - pmean, 2.0);
  pstdev = sqrt(pstdev/ (nSystems - 1));
  (*stdev) = pstdev;
  printAsterisks(PL_INFO,0);
  printf("Hierarchical prior information: \n");
  printEquals(PL_INFO,0);

  //**/ compute group means and standard deviations
  vecGMeans.setLength(nGroups);
  vecGStdvs.setLength(nGroups);
  for (ii = 0; ii < nGroups; ii++)
  {
    vecGMeans[ii] = 0;
    for (jj = 0; jj < groupSizes[ii]; jj++) 
    {
      index = groupInfo[ii][jj];  
      vecGMeans[ii] += sysData[index];
    }
    vecGMeans[ii] /= (double) groupSizes[ii];
    vecGStdvs[ii] = 0;
    for (jj = 0; jj < groupSizes[ii]; jj++) 
    {
      index = groupInfo[ii][jj];  
      vecGStdvs[ii] += pow(sysData[index] - vecGMeans[ii], 2.0);
    }
    vecGStdvs[ii] = sqrt(vecGStdvs[ii]/(double) (groupSizes[ii] - 1));
    printf("Group %2d mean and std dev = %12.4e %12.4e\n",ii+1,
           vecGMeans[ii],vecGStdvs[ii]);
  }

  //**/ now estimate hyper-prior parameters for group biases and
  //**/ group standard deviations
  //**/ parameter 1: alpha and beta  for mean bias 
  //**/ (1) compute mean of the group bias term
  printf("Hyperparameter 1, 2 : mean and std of mean bias between groups\n");
  inputPDFsH[0] = PSUADE_PDF_NORMAL;
  meanBias = 0.0;
  for (ii = 0; ii < nGroups; ii++) meanBias += (vecGMeans[ii]-pmean);
  meanBias /= (double) nGroups;
  inputMeansH[0] = meanBias;
  printf("Mean    group Bias = %12.4e\n", meanBias);
  sdBias = 0;
  for (ii = 0; ii < nGroups; ii++) 
    sdBias += pow(vecGMeans[ii]-pmean-meanBias,2.0);
  sdBias = sqrt(sdBias/(double) (nGroups-1));
  inputStdevH[0] = sdBias;
  printf("Std dev group Bias = %12.4e\n", sdBias);

  //**/ (2) compute the standard deviation of the group bias term
  printf("Hyperparameter 3, 4 : mean and std of group std devs\n");
  //inputPDFsH[1] = PSUADE_PDF_LOGNORMAL;
  inputPDFsH[1] = PSUADE_PDF_NORMAL;
  meanBias = 0.0;
  for (ii = 0; ii < nGroups; ii++) meanBias += vecGStdvs[ii];
  meanBias /= (double) nGroups;
  printf("Mean    group std devs = %12.4e\n",meanBias);
  sdBias = 0;
  for (ii = 0; ii < nGroups; ii++) 
    sdBias += pow(vecGStdvs[ii]-meanBias,2.0);
  if (sdBias == 0) sdBias = meanBias;
  else             sdBias = sqrt(sdBias/(double) (nGroups-1));
  printf("Std dev group std devs = %12.4e\n",sdBias);

  //inputStdevH[1] = sqrt(2.0 * log(meanBias/sdBias) + 1);
  //inputMeansH[1] = log(meanBias) - 0.5 * inputStdevH[1] * inputStdevH[1];
  //inputMeansH[1] = exp(meanBias + 0.5 * sdBias * sdBias);
  //inputStdevH[1] = exp(2.0 * meanBias + sdBias * sdBias) * 
  //                 (exp(sdBias*sdBias) - 1);
  //inputStdevH[1] = sqrt(inputStdevH[1]);
  inputMeansH[1] = meanBias;
  inputStdevH[1] = sdBias;

  printAsterisks(PL_INFO,0);
  printf("Hyper prior 1 (bias) = N(%12.4e, %12.4e)\n",inputMeansH[0],
         inputStdevH[0]);
  printf("Hyper prior 2 (stdv) = N(%12.4e, %12.4e)\n",inputMeansH[1],
         inputStdevH[1]);
  printAsterisks(PL_INFO,0);
  return 0;
}

// ************************************************************************
// check convergence 
// ------------------------------------------------------------------------
int HMCMCAnalyzer::checkConvergence(int num, double *values, int step)
{
  int    ii, jj, leng, part, nchains=5;
  double means[5], stds[5];
  double WStat, BStat, ddata, ddata2, thresh=1.02;

  //**/ compose means and stds
  leng = num / nchains * nchains;
  part = leng / nchains;
  for (ii = 0; ii < nchains; ii++)
  {
    ddata = 0.0;
    for (jj = 0; jj < part; jj++) ddata += values[(ii*part+jj)*step];
    ddata /= (double) part;
    ddata2 = 0.0;
    for (jj = 0; jj < part; jj++) 
      ddata2 += pow(values[(ii*part+jj)*step]-ddata,2.0);
    ddata2 /= (double) (part - 1.0);
    means[ii] = ddata;
    stds[ii] = sqrt(ddata2);
  } 

  //**/ compute PSRF
  WStat = 0.0;
  for (ii = 0; ii < nchains; ii++) WStat += stds[ii];
  WStat /= (double) nchains;
  if (WStat < 0) WStat = PSUADE_UNDEFINED;
  ddata = 0.0;
  for (ii = 0; ii < nchains; ii++) ddata += means[ii];
  ddata /= (double) nchains;
  BStat = 0.0;
  for (ii = 0; ii < nchains; ii++) 
    BStat += pow(means[ii]-ddata,2.0);
  printf("HMCMC check: std dev of chain means = %e (mean = %e)\n",
         sqrt(BStat/(nchains-1.0)), ddata);
  BStat = BStat / (nchains - 1.0) * part;
  ddata = (1 - 1.0/part) * WStat + BStat / part;
  ddata = ddata / WStat * (nchains + 1) / nchains - 
          (part - 1.0) / (double) (part * nchains); 
  if (ddata < 0) ddata2 = PSUADE_UNDEFINED;
  else           ddata2 = sqrt(ddata);
  printf("Convergence check: %e <? %e\n",ddata2,thresh);
  if (ddata2 < thresh) return 1;
  else                 return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
int HMCMCAnalyzer::genMatlabFile(int samSize, int nParams, double *sample,
                                 double *inferenceStore)
{
  int    ss, ii, jj, nbins=10, index, **bins;
  double denom, *lowers, *uppers, *ranges, ddata, *likelihoods;
  double **dbins;

  denom = 0.0;
  for (ss = 0; ss < samSize; ss++) denom += inferenceStore[ss];
  likelihoods = new double[samSize];
  for (ss = 0; ss < samSize; ss++) 
    likelihoods[ss] = inferenceStore[ss] / denom;
  lowers = new double[nParams];
  uppers = new double[nParams];
  ranges = new double[nParams];
  dbins = new double*[nbins];
  for (ii = 0; ii < nbins; ii++) 
  {
    dbins[ii] = new double[nParams];
    for (jj = 0; jj < nParams; jj++) dbins[ii][jj] = 0;
  }
  bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++) 
  {
    bins[ii] = new int[nParams];
    for (jj = 0; jj < nParams; jj++) bins[ii][jj] = 0;
  }
  for (ii = 0; ii < nParams; ii++) 
  {
    lowers[ii] = sample[ii];
    uppers[ii] = sample[ii];
    for (ss = 1; ss < samSize; ss++)
    {
      if (sample[ss*nParams+ii] < lowers[ii])
        lowers[ii] = sample[ss*nParams+ii];
      if (sample[ss*nParams+ii] > uppers[ii])
        uppers[ii] = sample[ss*nParams+ii];
    }
    ranges[ii] = uppers[ii] - lowers[ii];
  }
  for (ss = 0; ss < samSize; ss++)
  {
    for (ii = 0; ii < nParams; ii++) 
    {
      ddata = (sample[ss*nParams+ii] - lowers[ii]) / ranges[ii];
      index = (int) (ddata * nbins);
      if (index >= nbins) index = nbins - 1;
      dbins[index][ii] += likelihoods[ss];
    }
  }
  double Ymax;
  for (jj = 0; jj < nParams; jj++) 
  {
    Ymax = 0.0;
    for (ii = 0; ii < nbins; ii++)
      if (dbins[ii][jj] > Ymax) Ymax = dbins[ii][jj];
    for (ii = 0; ii < nbins; ii++)
      dbins[ii][jj] = dbins[ii][jj] / Ymax * samSize;
    for (ii = 0; ii < nbins; ii++)
      bins[ii][jj] = (int) dbins[ii][jj];
  }
  char cfname[1000], charString[1000];
  FILE *fp;
  strcpy(cfname, "matlabhmcmc.m");
  fp = fopen(cfname, "w");
  if (fp == NULL)
  {
    printOutTS(PL_ERROR, "ERROR: cannot open %s file.\n", cfname);
    return 0;
  }
  snprintf(charString,100,"This file shows posteriors plots");
  fwriteComment(fp, charString);
  fwritePlotCLF(fp);
  fprintf(fp, "L = [\n");
  for (ii = 0; ii < nParams; ii++) fprintf(fp, "%e ",lowers[ii]);
  fprintf(fp, "];\n");
  fprintf(fp, "U = [\n");
  for (ii = 0; ii < nParams; ii++) fprintf(fp, "%e ",uppers[ii]);
  fprintf(fp, "];\n");
  fprintf(fp, "X = zeros(%d,%d);\n", nParams, nbins);
  fprintf(fp, "D = zeros(%d,%d);\n", nParams, nbins);
  int kk, kk2, sumBins;
  for (kk = 0; kk < nParams; kk++)
  {
    for (kk2 = 0; kk2 < nParams; kk2++)
    {
      if (kk == kk2)
      {
        fprintf(fp, "X(%d,:) = [\n", kk+1);
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", ranges[kk]/nbins*(jj+0.5)+lowers[kk]);
        fprintf(fp, "];\n");
        sumBins = 0;
        for (jj = 0; jj < nbins; jj++) sumBins += bins[jj][kk];
        if (sumBins == 0) sumBins = 1;
        fprintf(fp, "D(%d,:) = [\n", kk+1);
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", (double) bins[jj][kk]/(double) sumBins);
        fprintf(fp, "];\n");
      }
    }
  }
  fprintf(fp,"nInps = %d;\n", nParams);
  fprintf(fp,"for ii = 1 : nInps\n");
  fprintf(fp,"   subplot(nInps,nInps,(ii-1)*nInps+ii)\n");
  fprintf(fp,"   n = length(D(ii,:));\n");
  fprintf(fp,"   DN = D(ii,:);\n");
  fprintf(fp,"   bar(X(ii,:), DN, 1.0);\n");
  fprintf(fp,"   xmin = min(X(ii,:));\n");
  fprintf(fp,"   xmax = max(X(ii,:));\n");
  fprintf(fp,"   xwid = xmax - xmin;\n");
  fprintf(fp,"   xmin = xmin - 0.5 * xwid / %d;\n", nbins);
  fprintf(fp,"   xmax = xmax + 0.5 * xwid / %d;\n", nbins);
  fprintf(fp,"   ymax = max(DN);\n");
  fprintf(fp,"   axis([xmin xmax 0 ymax])\n");
  fprintf(fp,"   set(gca,'linewidth',2)\n");
  fprintf(fp,"   set(gca,'fontweight','bold')\n");
  fprintf(fp,"   set(gca,'fontsize',12)\n");
  fprintf(fp,
     "   ylabel('Probabilities','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"   grid on\n");
  fprintf(fp,"   box on\n");
  fprintf(fp,"end;\n");
  fclose(fp);
  delete [] lowers;
  delete [] uppers;
  delete [] ranges;
  for (ii = 0; ii < nbins; ii++) delete [] dbins[ii];
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] dbins;
  delete [] bins;
  delete [] likelihoods;
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
HMCMCAnalyzer& HMCMCAnalyzer::operator=(const HMCMCAnalyzer &)
{
  printOutTS(PL_ERROR,
       "HMCMCAnalyzer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int HMCMCAnalyzer::get_nInputs()
{
  return nInputs_;
}
int HMCMCAnalyzer::get_nOutputs()
{
  return 1;
}
double *HMCMCAnalyzer::get_means()
{
  int    ii;
  double *retVal = NULL;
  if (VecMeans_.length() > 0)
  {
    retVal = new double[VecMeans_.length()];
    for (ii = 0; ii < VecMeans_.length(); ii++) 
      retVal[ii] = VecMeans_[ii];
  }
  return retVal;
}
double *HMCMCAnalyzer::get_mostLikelyInput()
{
  int    ii;
  double *retVal = NULL;
  if (VecMostLikelyInps_.length() > 0)
  {
    retVal = new double[VecMostLikelyInps_.length()];
    for (ii = 0; ii < VecMostLikelyInps_.length(); ii++) 
      retVal[ii] = VecMostLikelyInps_[ii];
  }
  return retVal;
}
double *HMCMCAnalyzer::get_mostLikelyOutput()
{
  int    ii;
  double *retVal = NULL;
  if (VecMostLikelyOuts_.length() > 0)
  {
    retVal = new double[1];
    retVal[0] = VecMostLikelyOuts_[0];
  }
  return retVal;
}

