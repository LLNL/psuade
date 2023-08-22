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
// Functions for the class ShapleyAnalyzer
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ------------------------------------------------------------------------
// Algorithm based on that described in
// 'A Simple Algorithm for global sensitivity analysis with Shapley Effect'
// by T. Goda. Three of the main ingredients of the algorithms are:
// (1) use Sobol' total sensitivity effect as the Shapley value function, 
// (2) use the Sobol' algorithm to compute the total sensitivity effect, 
// (3) a Monte-Carlo step to compute approximate Shapley effect (instead
//     of going through all permutations 
// phi(i) = sum_{u \in -{i}} [1/m C(m-1,|u|)^{-1} (tsi(u+i) - tsi(u))]
// where m = nInputs, and
//       1/m C(d-1,|u|)^{-1} = (m - 1 - |u|)! |u|! / m!
// Hence,
// phi(i) = sum_{u \in -{i}} [1/m C(m-1,|u|)^{-1} (tsi(u+i) - tsi(u))] =
// 1/m sum_{k=0}^{m-1} C(m-1,k)^-1 sum_{u \in -{i}, |u|=k} (tsi(u+i)-tsi(u))
// tsi(u) = 1/2 int_x int_y (F(x) - F(y_u,x_{-u}))^2 p(x) p(y) dx dy
// so
// phi(i) = sum sum 1/2 (F(x)-F(y_{S_i(k)+{i}},x_-{S_i(k)+{i}))^2 -
//                  1/2 (F(x)-F(y_{S_i(k)},x_-{S_i(k)))^2
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ShapleyAnalyzer.h"
#include "SobolAnalyzer.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "PDFManager.h"
#include "FuncApprox.h"
#include "PDFManager.h"
#include "ProbMatrix.h"
#ifdef PSUADE_OMP
#include <omp.h>
#endif

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ------------------------------------------------------------------------
// Both uniform and adaptive seem to be stable for entropy-based method
// although they give slightly different values
// ------------------------------------------------------------------------
//#define _ADAPTIVE_
// ------------------------------------------------------------------------

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
ShapleyAnalyzer::ShapleyAnalyzer() : Analyzer(), nInputs_(0)
{
  setName("SOBOL");
  sampleSize_ = 10000;
  costFunction_ = 1; /* 1: VCE-based, 2: entropy-based */
  MaxMapLength_=10000;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
ShapleyAnalyzer::~ShapleyAnalyzer()
{
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double ShapleyAnalyzer::analyze(aData &adata)
{
  printAsterisks(PL_INFO, 0);
  printf("*             Shapley Sensitivity Analysis\n"); 
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,
       "* Turn on analysis expert mode to choose different method.\n");
  printOutTS(PL_INFO,
       "* Turn on higher print level to see more information.\n");
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  nInputs_ = nInputs;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0) 
  {
    printOutTS(PL_ERROR, "ShapleyAnalyzer ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 

  //**/ ---------------------------------------------------------------
  //**/ check for valid samples
  //**/ ---------------------------------------------------------------
  int    errCnt = 0;
  int    outputID = adata.outputID_;
  double *yIn = adata.sampleOutputs_;
  for (int ss = 0; ss < nSamples; ss++)
    if (yIn[ss*nOutputs+outputID] > 0.99*PSUADE_UNDEFINED) errCnt++;
  if (errCnt > 0)
  {
    printf("ShapleyAnalyzer ERROR: Found %d invalid sample points,\n",
           errCnt);
    exit(1);
  }

  //**/ ---------------------------------------------------------------
  //**/ analyze using one of the 2 algorithms
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"You can select one of the 2 below : \n");
    printOutTS(PL_INFO,"1. Sobol'-based method\n");
    printOutTS(PL_INFO,"2. Entropy-based method\n");
    snprintf(pString,100,"Select 1 or 2 : ");
    costFunction_ = getInt(1, 2, pString);
  }
  if (costFunction_ == 1) return analyzeVCE(adata);
  else                    return analyzeEntropy(adata);
}
 
// ************************************************************************
// perform analysis using VCE as cost function
// ------------------------------------------------------------------------
double ShapleyAnalyzer::analyzeVCE(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;

  //**/ ---------------------------------------------------------------
  //**/ get sample size
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* Sobol'-based Shapley uses two large samples ");
    printOutTS(PL_INFO,"to compute sensitivity\n");
    printOutTS(PL_INFO,"* indices. The default sample size M is %d.\n",
               sampleSize_);
    printOutTS(PL_INFO,"* You may, however, select a different M.\n");
    printOutTS(PL_INFO,"* NOTE: large M may take long time, but ");
    printOutTS(PL_INFO,"gives more accurate results.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 10000-1000000) : ");
    sampleSize_ = getInt(10000, 1000000, pString);
  }

  //**/ ---------------------------------------------------------------
  //**/ create 2 random samples (vecXM1, vecXM2) 
  //**/ It is going to march from M1 to M2 in random input order
  //**/ ---------------------------------------------------------------
  srand48(15485863);
  psVector vecXM1, vecXM2;
  create2RandomSamples(adata, vecXM1, vecXM2);

  //**/ ---------------------------------------------------------------
  //**/ create a random integer matrix (matRandInt)
  //**/ (used to input order randomization)
  //**/ ---------------------------------------------------------------
  int largeNumSams = vecXM1.length() / nInputs;
  psIMatrix matRandInt;
  createRandomIntMatrix(largeNumSams, nInputs, matRandInt);

  //**/ ---------------------------------------------------------------
  //**/ create response surface 
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = createResponseSurface(adata);
  if (faPtr == NULL) return -1;

  //**/ ---------------------------------------------------------------
  //**/ evaluate vecXM1 using response surface ==> vecYM1 
  //**/ ---------------------------------------------------------------
  psVector vecYM1;
  vecYM1.setLength(largeNumSams);
  faPtr->evaluatePoint(largeNumSams,vecXM1.getDVector(),
                       vecYM1.getDVector());
     
  //**/ ---------------------------------------------------------------
  //**/ compute basic statistics
  //**/ ---------------------------------------------------------------
  int    ss, ii, count = 0;
  double *yIn = adata.sampleOutputs_;

  double dmean = 0.0;
  for (ss = 0; ss < largeNumSams; ss++) dmean += vecYM1[ss];
  dmean /= (double) largeNumSams;
  double dVar = 0.0;
  for (ss = 0; ss < largeNumSams; ss++) dVar += pow(vecYM1[ss]-dmean,2.0);
  dVar /= (double) (largeNumSams - 1);
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"Shapley: sample mean     = %10.3e\n",dmean);
    printOutTS(PL_INFO,"Shapley: sample variance = %10.3e\n",dVar);
  }
  if (dVar == 0)
  {
    printf("INFO: Total variance = 0 ==> no input sensitivities.\n");
    VecShapleys_.setLength(nInputs_);
    return 0;
  }

  //**/ ---------------------------------------------------------------
  //**/ perform Shapley analysis
  //**/ vecXMM holds the matrix for current single input modification
  //**/ ---------------------------------------------------------------
  int    ind;
  double ddata;
  VecShapleys_.setLength(nInputs_);
  VecShapleyStds_.setLength(nInputs_);
  psVector vecXMM, vecYMP, vecYMM;
  vecXMM = vecXM1; /* z = x */
  vecYMP = vecYM1; /* fz1 = fx */
  vecYMM.setLength(largeNumSams);

  for (ii = 0; ii < nInputs; ii++)
  {
    //**/ evolve sample (z(ind) = y(ind))
    for (ss = 0; ss < largeNumSams; ss++)
    {
      ind = matRandInt.getEntry(ss, ii);
      vecXMM[ss*nInputs+ind] = vecXM2[ss*nInputs+ind];
    }
    
    //**/ evaluate the newly-evolved sample (fz2 = F(z))
    faPtr->evaluatePoint(largeNumSams,vecXMM.getDVector(),
                         vecYMM.getDVector());

    //**/ elemental operations (fmarg = ((fx-fz1/2-fz2/2).*(fz1-fz2))')
    //**/ update Shapley values (phi1 = phi1 + fmarg*ind/n)
    for (ss = 0; ss < largeNumSams; ss++)
    {
      ind = matRandInt.getEntry(ss, ii);
      ddata = (vecYM1[ss]-0.5*(vecYMP[ss]+vecYMM[ss]))*
              (vecYMP[ss]-vecYMM[ss]); 
      VecShapleys_[ind] += (ddata / largeNumSams); 
      VecShapleyStds_[ind] += (ddata * ddata / largeNumSams); 
    }

    //**/ advance
    vecYMP = vecYMM;
  }

  printOutTS(PL_INFO, "Shapley Values (variance-based):\n");
  double totalChk=0;
  for (ii = 0; ii < nInputs; ii++)
  {
    VecShapleyStds_[ii] = (VecShapleyStds_[ii] - pow(VecShapleys_[ii],2.0))/
                          (largeNumSams - 1); 
    printOutTS(PL_INFO,
         "  Input %3d = %10.3e [%10.3e, %10.3e], Normalized = %10.3e\n",ii+1,
         VecShapleys_[ii],VecShapleys_[ii]-1.96*VecShapleyStds_[ii],
         VecShapleys_[ii]+1.96*VecShapleyStds_[ii],VecShapleys_[ii]/dVar);
    totalChk += VecShapleys_[ii];
  }
  printOutTS(PL_INFO,"Sum of Shapley values = %11.4e\n",totalChk);
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO, "Normalized Shapley Values (variance-based):\n");
  for (ii = 0; ii < nInputs_; ii++)
    printOutTS(PL_INFO,"  Input %3d = %10.3e\n",ii+1,
               VecShapleys_[ii]/dVar);
  printAsterisks(PL_INFO, 0);
  if (adata.printLevel_ > 2)
  { 
    //**/ perform Shapley based on MOAT modified gradients
    int kk;
    psVector vecShapleyMOAT;
    vecShapleyMOAT.setLength(nInputs);
    psVector vecLargeSample;
    vecLargeSample.setLength(largeNumSams*(nInputs+2)*nInputs);
    count = 0;
    for (ss = 0; ss < largeNumSams; ss++)
    {
      //**/ copy M1 times to vecLargeSample
      for (ii = 0; ii < nInputs; ii++)
        vecLargeSample[count*nInputs+ii] = vecXM1[ss*nInputs+ii];
      count++;
      //**/ evolve from M1 to M2
      for (kk = 0; kk < nInputs; kk++)
      {
        //**/ first make a copy of previous sample
        for (ii = 0; ii < nInputs; ii++)
          vecLargeSample[count*nInputs+ii] = 
            vecLargeSample[(count-1)*nInputs+ii]; 
        //**/ modify one entry based on kk
        vecLargeSample[count*nInputs+kk] = 
                                   vecXM2[ss*nInputs+kk];
        count++;
      }
      for (ii = 0; ii < nInputs; ii++)
        vecLargeSample[count*nInputs+ii] = vecXM2[ss*nInputs+ii];
      count++;
    }
 
    //**/ now we have a very large sample in vecLargeSample
    //**/ size = largeNumSams*(nInputs+2)
    //**/ evaluate the sample 
    psVector vecLargeY;
    vecLargeY.setLength(largeNumSams*(nInputs+2));
    count = largeNumSams * (nInputs + 2);
    faPtr->evaluatePoint(count,vecLargeSample.getDVector(),
                       vecLargeY.getDVector());

    psVector vecMeans, vecModMeans, vecStds;
    vecMeans.setLength(nInputs);
    vecModMeans.setLength(nInputs);
    vecStds.setLength(nInputs);
    MOATAnalyze(nInputs,count,vecLargeSample.getDVector(),
             vecLargeY.getDVector(),adata.iLowerB_,adata.iUpperB_,
             vecMeans.getDVector(),vecModMeans.getDVector(),
             vecStds.getDVector());
    for (ii = 0; ii < nInputs; ii++)
      printf("MOAT modified mean for input %4d = %e\n",ii+1,
             vecModMeans[ii]);
  }
  printAsterisks(PL_INFO, 0);
  delete faPtr;
  return 0.0;
}

// ************************************************************************
// perform analysis using entropy as cost function
// ------------------------------------------------------------------------
double ShapleyAnalyzer::analyzeEntropy(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;

  //**/ ---------------------------------------------------------------
  //**/ get sample size
  //**/ ---------------------------------------------------------------
  sampleSize_ = 1000;
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* Entropy-based Shapley uses a large sample ");
    printOutTS(PL_INFO,"to compute output entropy\n");
    printOutTS(PL_INFO,"* sensitivity indices. The default sample ");
    printOutTS(PL_INFO,"size M is %d.\n", sampleSize_);
    printOutTS(PL_INFO,"* You may, however, select a different M.\n");
    printOutTS(PL_INFO,"* NOTE: large M may take long time, but ");
    printOutTS(PL_INFO,"may give more accurate results.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 1000-1000000) : ");
    sampleSize_ = getInt(1000, 1000000, pString);
  }

  //**/ ---------------------------------------------------------------
  //**/ create response surface 
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = createResponseSurface(adata);
  if (faPtr == NULL) return -1;

  //**/ ---------------------------------------------------------------
  //**/ get input distribution information
  //**/ ---------------------------------------------------------------
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStdvs;
  int    *pdfFlags   = adata.inputPDFs_; 
  double *inputMeans = adata.inputMeans_;
  double *inputStdvs = adata.inputStdevs_;
  if (pdfFlags == NULL)
  {
    vecPdfFlags.setLength(nInputs_);
    vecInpMeans.setLength(nInputs_);
    vecInpStdvs.setLength(nInputs_);
    pdfFlags   = vecPdfFlags.getIVector(); 
    inputMeans = vecInpMeans.getDVector();
    inputStdvs = vecInpStdvs.getDVector();
  }

  //**/ ---------------------------------------------------------------
  //**/ first compute total entropy
  //**/ ---------------------------------------------------------------
  int      ii, ss, nSam=10000, iOne=1, nLevels=20, status;
  double   totalEntropy, dOne=1, Ymax, Ymin, Ywidth, ddata;
  psVector vecSam, vecL, vecU, vecY;
  psMatrix matCov;
  matCov.setDim(nInputs_, nInputs_);
  for (ii = 0; ii < nInputs_; ii++) matCov.setEntry(ii,ii,dOne);
  PDFManager *pdfman = new PDFManager();
  pdfman->initialize(nInputs_,pdfFlags,inputMeans,inputStdvs,matCov,
                     NULL,NULL);
  vecSam.setLength(nSam*nInputs_);
  vecL.load(nInputs_, adata.iLowerB_);
  vecU.load(nInputs_, adata.iUpperB_);
  pdfman->genSample(nSam, vecSam, vecL, vecU);
  delete pdfman;
  vecY.setLength(nSam);
  faPtr->evaluatePoint(nSam,vecSam.getDVector(),vecY.getDVector());
  ProbMatrix matProb;
  matProb.setDim(nSam, iOne);
  Ymax = -PSUADE_UNDEFINED;
  Ymin = +PSUADE_UNDEFINED;
  for (ss = 0; ss < nSam; ss++)
  {
    ddata = vecY[ss];
    if (ddata > Ymax) Ymax = ddata;
    if (ddata < Ymin) Ymin = ddata;
    matProb.setEntry(ss,0,ddata);
  }
  if (Ymax <= Ymin)
  {
    printf("Shapley INFO: Ymin (%e) = Ymax (%e)\n",Ymin,Ymax);
    printf("        ===> Assume zero total entropy w.r.t. ");
    printf("input variations.\n");
    printf("        ===> Assume zero Shapley values for all inputs.\n");
    printf("NOTE: This may be due to all insensitive inputs, ");
    printf("or this may be due to\n");
    printf("      poor quality of RS (inaccurate interpolations.)\n");
    VecShapleys_.setLength(nInputs_);
    return 0;
  }
  psVector vecYL, vecYU;
  vecYL.setLength(1);
  vecYU.setLength(1);
  vecYL[0] = Ymin;
  vecYU[0] = Ymax;
#ifdef _ADAPTIVE_
  //**/ bin widths vary to make all bins having same count
  status = matProb.binAdaptive(nLevels, vecYL, vecYU);
#else
  //**/ bin widths constant = 1/nLevels
  status = matProb.convert2Hist(nLevels, vecYL, vecYU);
#endif
  if (status == 1)
  {
    printf("Shapley ERROR: Unable to perform histogram conversion.\n");
    printf("Please consult developers.\n");
    exit(1);
  }
  totalEntropy = 0;
  for (ss = 0; ss < nLevels; ss++)
  {
    //**/ get total probability of the bin = P(Y_i) dY_i
    ddata = (double) matProb.getCount(ss) / nSam;
#ifdef _ADAPTIVE_
    //**/ compute bin probability P(Y_i) by dividing by bin width
    //**/ matProb(ss,1) has width for bin ss
    ddata /= matProb.getEntry(ss,0);
    //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
    if (ddata > 0) ddata = ddata * log(ddata);
    //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
    ddata *= matProb.getEntry(ss,0);
#else
    //**/ compute bin probability P(Y_i) by dividing by bin width
    ddata /= ((Ymax - Ymin) / nLevels);
    //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
    if (ddata > 0) ddata = ddata * log(ddata);
    //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
    ddata *= ((Ymax - Ymin) / nLevels);
#endif
    //**/ accumulate entropy
    totalEntropy -= ddata;
  }
  printf("Output entropy = %e\n", totalEntropy);

  //**/ ---------------------------------------------------------------
  //**/ declare all variables needed for processing
  //**/ ---------------------------------------------------------------
  PDFManager *pdfman1=NULL, *pdfman2=NULL;
  //**/ for subset selection
  psIVector vecIRand;
  psVector vecRand;
  //**/ for input and output bounds
  psVector vecL1, vecL2, vecU1, vecU2, vecYL1, vecYU1, vecYL2, vecYU2; 
  //**/ for input means and std dev of subsets
  psVector  vecIMean1, vecIMean2, vecIStdv1, vecIStdv2;
  psIVector vecIPDF1, vecIPDF2;
  psMatrix matCov1, matCov2;
  //**/ for storing subset sample and entire sample
  psVector vecSam1, vecSam2, vecLargeSam, vecLargeY;
  MatShapleyMap_.setDim(MaxMapLength_, nInputs_);
  MapLength_ = 0;
  VecShapleyTable_.setLength(MaxMapLength_);

  //**/ ---------------------------------------------------------------
  //**/ process each input 
  //**/ ---------------------------------------------------------------
  int    ii2, ss1, ss2, nInp1, nInp2, nSam1, nSam2, ind;
  double entropy1, entropy2, entropy;
  VecShapleys_.setLength(nInputs_);
#pragma omp parallel shared(ii,faPtr) \
    private(ss,ii2,vecRand,vecIRand,nInp1,nInp2,vecIPDF1,vecIMean1,\
        vecIStdv1,vecL1,vecU1,matCov1,vecIPDF2,vecIMean2,vecIStdv2,\
        vecL2,vecU2,matCov2,nSam1,nSam2,pdfman1,pdfman2,vecSam1,vecSam2,\
        ind,ss1,ss2,entropy1,entropy2,entropy,vecLargeSam,vecLargeY,\
        status,vecYL1,vecYL2,vecYU1,vecYU2,matProb,ddata,Ymin,Ymax)
{
#pragma omp for
  for (ii = 0; ii < nInputs_; ii++)
  {
#ifdef PSUADE_OMP
    printf("Processing input %d (thread=%d)",ii+1,omp_get_thread_num());
#else
    printf("Processing input %d ", ii+1);
#endif
    //**/ Let I be the set of all inputs. The algorithms goes like this:
    //**/ For each input:
    //**/ A. Compute mean entropy gain by taking the mean of the 
    //**/    following steps sampleSize_ times:  
    //**/ 1. Select a random subset of I without input ii and call it S
    //**/ 2. Form another subset to be union of S and ii (call it S+)
    //**/ 3. Compute entropy gain for S : H(S+) - H(S)
    //**/    a. Compute entropy for H(S+)
    //          - Create a sample for S+ (assume independent inputs)
    //**/       - Create a second sample from I\S+ of size N_2
    //**/       - For each sample point k for I\S+ compute conditional
    //**/         entropy H_k(S+|k-th sample point for I\S+)  
    //**/       - Compute mean(H(S+|I\S+)) = 1/N_2 sum_{k=1}^N_2 H_k(S+|k)
    //**/    c. - Create a sample from S
    //**/       - create a second sample from I\S of size N_2
    //**/       - For each sample point k for I\S compute conditional
    //**/         entropy H_k(S|k-th sample point for I\S)  
    //**/       - Compute mean(H(S|I\S)) = 1/N_2 sum_{k=1}^N_2 H_k(S|k)
    //**/    d. Compute difference dE = mean(S+|I\S+) - mean(S|I\S)
    //**/       (that is, entropy gain for the current random subset S
    //**/ B. Sum all entropy gains dE's in previous steps and take 
    //**/    average and this will be the entropy-based Shapley value
    vecIRand.setLength(nInputs_);
    vecRand.setLength(nInputs_);
    //**/ reset the lookup table
    MatShapleyMap_.setDim(MaxMapLength_, nInputs_);
    MapLength_ = 0;
    for (ss = 0; ss < sampleSize_; ss++)
    {
      //**/ generate a random subset S by using random numbers and 
      //**/ sorting 
      for (ii2 = 0; ii2 < nInputs_; ii2++) 
      {
        vecRand[ii2] = drand48(); 
        vecIRand[ii2] = ii2;
      }
      sortDbleList2a(nInputs_,vecRand.getDVector(),
                     vecIRand.getIVector());

      //**/ look up to see if this permutation has been analyzed
      //**/ before. If so, just return the value
      ddata = ShapleyLookup(vecIRand, ii);
      if (ddata != -9999)
      {
        VecShapleys_[ii] += PABS(ddata);
        continue;
      }

      //**/ search for the ii index (the position of the ii index 
      //**/ will be used as random subset size (size(S) = ii2)
      for (ii2 = 0; ii2 < nInputs_; ii2++) 
        if (vecIRand[ii2] == ii) break;

      //**/ set nInp1 = size(S+) (i.e. including input ii)
      //**/ set nInp2 = size(I\S+) 
      nInp1 = ii2 + 1;
      nInp2 = nInputs_ - nInp1;
      
      //**/ Construct a sample for S+ (sample size is somewhat
      //**/ arbitrary - just large enough for reasonable statistics)
      if      (nInp1 == 1) nSam1 = 200;
      else if (nInp1 == 2) nSam1 = 300;
      else if (nInp1 >= 3) nSam1 = 1000;
      //printf("   - Construct a sample: nInp1=%d, size=%d\n",nInp1,nSam1);
      pdfman1 = NULL;
      if (nInp1 > 0)
      {
        vecIPDF1.setLength(nInp1);
        vecIMean1.setLength(nInp1);
        vecIStdv1.setLength(nInp1);
        vecL1.setLength(nInp1);
        vecU1.setLength(nInp1);
        matCov1.setDim(nInp1, nInp1);
        for (ii2 = 0; ii2 < nInp1; ii2++)
        {
          ind = vecIRand[ii2];
          vecIPDF1[ii2] = pdfFlags[ind];
          vecIMean1[ii2] = inputMeans[ind];
          vecIStdv1[ii2] = inputStdvs[ind];
          vecL1[ii2] = adata.iLowerB_[ind];
          vecU1[ii2] = adata.iUpperB_[ind];
          matCov1.setEntry(ii2,ii2,dOne);
        }
        pdfman1 = new PDFManager();
        pdfman1->initialize(nInp1,vecIPDF1.getIVector(),
                  vecIMean1.getDVector(),vecIStdv1.getDVector(),
                  matCov1,NULL,NULL);
        vecSam1.setLength(nSam1*nInp1);
        pdfman1->genSample(nSam1, vecSam1, vecL1, vecU1);
        delete pdfman1;
      }

      //**/ construct a sample for I\S+ (sample size is also somewhat
      //**/ arbitrary - just large enough for reasonable statistics)
      //**/ If size(I\S+)=0, nSam2 = 1 ==> no need to compute average
      nSam2 = 1;
      if      (nInp2 == 1) nSam2 = 10;
      else if (nInp2 == 2) nSam2 = 20;
      else if (nInp2 >= 3) nSam2 = 10 * nInp2;
      //printf("   - Construct a sample: nInp2=%d, size=%d\n",nInp2,nSam2);
      pdfman2 = NULL;
      if (nInp2 > 0)
      {
        vecIPDF2.setLength(nInp2);
        vecIMean2.setLength(nInp2);
        vecIStdv2.setLength(nInp2);
        vecL2.setLength(nInp2);
        vecU2.setLength(nInp2);
        matCov2.setDim(nInp2,nInp2);
        for (ii2 = 0; ii2 < nInp2; ii2++)
        {
          ind = vecIRand[ii2+nInp1];
          vecIPDF2[ii2] = pdfFlags[ind];
          vecIMean2[ii2] = inputMeans[ind];
          vecIStdv2[ii2] = inputStdvs[ind];
          vecL2[ii2] = adata.iLowerB_[ind];
          vecU2[ii2] = adata.iUpperB_[ind];
          matCov2.setEntry(ii2,ii2,dOne);
        }
        pdfman2 = new PDFManager();
        pdfman2->initialize(nInp2,vecIPDF2.getIVector(),
                  vecIMean2.getDVector(),vecIStdv2.getDVector(),
                  matCov2,NULL,NULL);
        vecSam2.setLength(nSam2*nInp2);
        pdfman2->genSample(nSam2, vecSam2, vecL2, vecU2);
        delete pdfman2;
      }

      //**/ Merge the 2 samples into vecLargeSam (concatenation)
      //printf("   - construct large sample\n");
      vecLargeSam.setLength(nSam1*nSam2 * nInputs_);
      for (ss1 = 0; ss1 < nSam2; ss1++)
      {
        //**/ copy the whole first sample into a block
        for (ss2 = 0; ss2 < nSam1; ss2++)
        {
          for (ii2 = 0; ii2 < nInp1; ii2++)
          {
            ind = vecIRand[ii2];
            vecLargeSam[(ss1*nSam1+ss2)*nInputs_+ind] = 
               vecSam1[ss2*nInp1+ii2];
          }
          for (ii2 = 0; ii2 < nInp2; ii2++)
          {
            ind = vecIRand[ii2+nInp1];
            vecLargeSam[(ss1*nSam1+ss2)*nInputs_+ind] = 
               vecSam2[ss1*nInp2+ii2];
          }
        }
      }

      //**/ evaluation the large sample
      //printf("   - function evaluation\n");
      vecLargeY.setLength(nSam1*nSam2);
      faPtr->evaluatePoint(nSam1*nSam2,vecLargeSam.getDVector(),
                           vecLargeY.getDVector());
           
      //**/ for each of the sample point for I\S+, compute entropy
      //**/ then take the mean ==> entropy1
      //printf("   - compute entropy\n");
      entropy1 = 0;
      for (ss1 = 0; ss1 < nSam2; ss1++)
      {
        matProb.setDim(nSam1, iOne);
        //**/ search for lower and upper bounds of Y
        //**/ the reason to use tight bounds is that loose bound does
        //**/ not reflect the true delta Y due to discrete intervals
        //**/ because pdf(interval Y_i) ~ count/delta Y_i
        Ymax = -PSUADE_UNDEFINED;
        Ymin = +PSUADE_UNDEFINED;
        for (ss2 = ss1*nSam1; ss2 < (ss1+1)*nSam1; ss2++)
        {
          ddata = vecLargeY[ss2];
          if (ddata > Ymax) Ymax = ddata;
          if (ddata < Ymin) Ymin = ddata;
          ind = ss2 - ss1 * nSam1;
          matProb.setEntry(ind,0,ddata);
        }
        if (Ymin >= Ymax)
        {
          //printf("Shapley INFO: Ymin %e == Ymax %e (a)\n",Ymin,Ymax);
          //printf("   ==> Assume no entropy (no output variation).\n");
          //printf("NOTE: It may be due to insensitive inputs, or ");
          //printf("poor quality of RS (a).\n");
          //printf("#");
        }
        else
        //**/ binning
        {
          vecYL1.setLength(iOne);
          vecYU1.setLength(iOne);
          vecYL1[0] = Ymin;
          vecYU1[0] = Ymax;
#ifdef _ADAPTIVE_
          status = matProb.binAdaptive(nLevels, vecYL1, vecYU1);
#else
          status = matProb.convert2Hist(nLevels, vecYL1, vecYU1);
#endif
          if (status == 1)
          {
            printf("Shapley ERROR: Unable to perform histogram conversion (a).\n");
            printf("Please consult developers.\n");
            exit(1);
          }

          //**/ compute entropy based on binning data
          entropy = 0;
          for (ss2 = 0; ss2 < nLevels; ss2++)
          {
            //**/ get total probability of the bin = P(Y_i) dY_i
            ddata = (double) matProb.getCount(ss2) / nSam1;
#ifdef _ADAPTIVE_
            //**/ compute bin probability P(Y_i) by dividing by bin width
            //**/ matProb(ss,1) has width for bin ss
            if (matProb.getEntry(ss2,0) > 0)
              ddata /= matProb.getEntry(ss2,0);
            else
            {
              printf("Shapley ERROR: delta_Y(%d) = %e <= 0 (a)\n",ss2+1,
                     matProb.getEntry(ss2,0));
              exit(1);
            }
            //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
            if (ddata > 0) ddata = ddata * log(ddata);
            //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
            ddata *= matProb.getEntry(ss2,0);
#else
            //**/ compute bin probability P(Y_i) by dividing by bin width
            ddata /= ((Ymax - Ymin) / nLevels);
            //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
            if (ddata > 0) ddata = ddata * log(ddata);
            //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
            ddata *= ((Ymax - Ymin) / nLevels);
#endif
            //**/ accumulate entropy
            entropy -= ddata;
          }
          //**/ sum all entropies from all sample points in sample 2
          entropy1 += entropy;
        } /* Ymax > Ymin */
      }
      //**/ compute mean entropy
      entropy1 /= (double) nSam2;
      //printf("   - total entropy = %e\n", entropy1);

      //**/ nInp1 = size(S)    (note nInp1 may be 0)
      //**/ nInp2 = size(I\S)
      nInp1--;
      nInp2 = nInputs_ - nInp1;
      
      //**/ if nInp1 == 0, entropy H(S)=0 (no variation)
      //**/ Construct a sample for S (sample size is somewhat
      //**/ arbitrary - just large enough for reasonable statistics)
      entropy2 = 0;
      if (nInp1 > 0)
      {
        if      (nInp1 == 1) nSam1 = 200;
        else if (nInp1 == 2) nSam1 = 300;
        else if (nInp1 >= 3) nSam1 = 1000;
        vecIPDF1.setLength(nInp1);
        vecIMean1.setLength(nInp1);
        vecIStdv1.setLength(nInp1);
        vecL1.setLength(nInp1);
        vecU1.setLength(nInp1);
        matCov1.setDim(nInp1,nInp1);
        for (ii2 = 0; ii2 < nInp1; ii2++)
        {
          ind = vecIRand[ii2];
          vecIPDF1[ii2] = pdfFlags[ind];
          vecIMean1[ii2] = inputMeans[ind];
          vecIStdv1[ii2] = inputStdvs[ind];
          vecL1[ii2] = adata.iLowerB_[ind];
          vecU1[ii2] = adata.iUpperB_[ind];
          matCov1.setEntry(ii2,ii2,dOne);
        }
        pdfman1 = new PDFManager();
        pdfman1->initialize(nInp1,vecIPDF1.getIVector(),
                  vecIMean1.getDVector(),vecIStdv1.getDVector(),
                  matCov1,NULL,NULL);
        vecSam1.setLength(nSam1*nInp1);
        pdfman1->genSample(nSam1, vecSam1, vecL1, vecU1);
        delete pdfman1;

        //**/ nSam2 may be small but at least 1
        if      (nInp2 == 1) nSam2 = 200;
        else if (nInp2 == 2) nSam2 = 300;
        else if (nInp2 >= 3) nSam2 = 1000;
        vecIPDF2.setLength(nInp2);
        vecIMean2.setLength(nInp2);
        vecIStdv2.setLength(nInp2);
        vecL2.setLength(nInp2);
        vecU2.setLength(nInp2);
        matCov2.setDim(nInp2,nInp2);
        for (ii2 = 0; ii2 < nInp2; ii2++)
        {
          ind = vecIRand[ii2+nInp1];
          vecIPDF2[ii2] = pdfFlags[ind];
          vecIMean2[ii2] = inputMeans[ind];
          vecIStdv2[ii2] = inputStdvs[ind];
          vecL2[ii2] = adata.iLowerB_[ind];
          vecU2[ii2] = adata.iUpperB_[ind];
          matCov2.setEntry(ii2,ii2,dOne);
        }
        pdfman2 = new PDFManager();
        pdfman2->initialize(nInp2,vecIPDF2.getIVector(),
                  vecIMean2.getDVector(),vecIStdv2.getDVector(),
                  matCov2,NULL,NULL);
        vecSam2.setLength(nSam2*nInp2);
        pdfman2->genSample(nSam2, vecSam2, vecL2, vecU2);
        delete pdfman2;

        //**/ merge the 2 samples into vecLargeSam (if nInp1 > 0)
        vecLargeSam.setLength(nSam1*nSam2 * nInputs_);
        for (ss1 = 0; ss1 < nSam2; ss1++)
        {
          //**/ copy the whole first sample into a block
          for (ss2 = 0; ss2 < nSam1; ss2++)
          {
            for (ii2 = 0; ii2 < nInp1; ii2++)
            {
              ind = vecIRand[ii2];
              vecLargeSam[(ss1*nSam1+ss2)*nInputs_+ind] = 
                 vecSam1[ss2*nInp1+ii2];
            }
            for (ii2 = 0; ii2 < nInp2; ii2++)
            {
              ind = vecIRand[ii2+nInp1];
              vecLargeSam[(ss1*nSam1+ss2)*nInputs_+ind] = 
                 vecSam2[ss1*nInp2+ii2];
            }
          }
        }

        //**/ evaluation the large
        vecLargeY.setLength(nSam1*nSam2);
        faPtr->evaluatePoint(nSam1*nSam2,vecLargeSam.getDVector(),
                             vecLargeY.getDVector());
     
        //**/ compute mean of entropy
        for (ss1 = 0; ss1 < nSam2; ss1++)
        {
          matProb.setDim(nSam1, iOne);
          Ymax = -PSUADE_UNDEFINED;
          Ymin = +PSUADE_UNDEFINED;
          for (ss2 = ss1*nSam1; ss2 < (ss1+1)*nSam1; ss2++)
          {
            ddata = vecLargeY[ss2];
            if (ddata > Ymax) Ymax = ddata;
            if (ddata < Ymin) Ymin = ddata;
            ind = ss2 - ss1 * nSam1;
            matProb.setEntry(ind,0,ddata);
          }
          if (Ymin >= Ymax)
          {
            //printf("Shapley INFO: Ymin %e == Ymax %e (b)\n",Ymin,Ymax);
            //printf("   ==> Assume no entropy (no output variation).\n");
            //printf("NOTE: It may be due to insensitive inputs, or ");
            //printf("poor quality of RS (b).\n");
            //printf("#");
          }
          else
          //**/ binning
          {
            vecYL2.setLength(iOne);
            vecYU2.setLength(iOne);
            vecYL2[0] = Ymin;
            vecYU2[0] = Ymax;
#ifdef _ADAPTIVE_
            status = matProb.binAdaptive(nLevels, vecYL2, vecYU2);
#else
            status = matProb.convert2Hist(nLevels, vecYL2, vecYU2);
#endif
            if (status == 1)
            {
              printf("Shapley ERROR: Unable to perform histogram conversion (b).\n");
              printf("Please consult developers.\n");
              exit(1);
            }
            entropy = 0;
            for (ss2 = 0; ss2 < nLevels; ss2++)
            {
              //**/ get total probability of the bin = P(Y_i) dY_i
              ddata = (double) matProb.getCount(ss2) / nSam1;
#ifdef _ADAPTIVE_
              //**/ compute bin probability P(Y_i) by dividing by bin width
              //**/ matProb(ss,1) has width for bin ss
              if (matProb.getEntry(ss2,0) > 0)
                ddata /= matProb.getEntry(ss2,0);
              else
              {
                printf("Shapley ERROR: delta_Y(%d) = %e <= 0 (b)\n",ss2+1,
                       matProb.getEntry(ss2,0));
                exit(1);
              }
              //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
              if (ddata > 0) ddata = ddata * log(ddata);
              //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
              ddata *= matProb.getEntry(ss2,0);
#else
              //**/ compute bin probability P(Y_i) by dividing by bin width
              ddata /= ((Ymax - Ymin) / nLevels);
              //**/ compute entropy P(Y_i) log(P(Y_i)) of the bin if not zero
              if (ddata > 0) ddata = ddata * log(ddata);
              //**/ multiply for width of the bin => P(Y_i) log(P(Y_i)) dY_i
              ddata *= ((Ymax - Ymin) / nLevels);
#endif
              //**/ accumulate entropy
              entropy -= ddata;
            }
            //**/ sum all entropies from all sample points in sample 2
            entropy2 += entropy;
          } /* Ymax > Ymin */
        }
        //**/ compute mean entropy
        entropy2 /= (double) nSam2;
      }
      //**/ accumulate entropy gain
      VecShapleys_[ii] += PABS(entropy1 - entropy2);

      //**/ store entropy gain above certain ss
      //**/ allow some burn-in to ensure randomness
      if (MapLength_ < MaxMapLength_-1)
      {
        for (ii2 = 0; ii2 < nInp1+1; ii2++)
        {
          ind = vecIRand[ii2];
          MatShapleyMap_.setEntry(MapLength_,ind,iOne);
          VecShapleyTable_[MapLength_] = entropy1;
        }
        MapLength_++;
        if (nInp1 > 0)
        {
          for (ii2 = 0; ii2 < nInp1; ii2++)
          {
            ind = vecIRand[ii2];
            MatShapleyMap_.setEntry(MapLength_,ind,iOne);
            VecShapleyTable_[MapLength_] = entropy2;
          }
          MapLength_++;
        }
      }

      //**/ display progress
      if ((ss * 50) % sampleSize_ == 0) 
      {
        printf(".");
        fflush(stdout);
      }
    } /* different subsets */
    printf("\n");
    VecShapleys_[ii] /= (double) sampleSize_;
#ifndef PSUADE_OMP
    if (adata.printLevel_ > 0)
      printOutTS(PL_INFO," ==> Shapley value = %9.3e\n",VecShapleys_[ii]);
#endif
  }
} /* omp parallel */
  printOutTS(PL_INFO, "Shapley Values (entropy-based):\n");
  double totalChk=0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    printOutTS(PL_INFO,"  Input %3d = %10.3e\n",ii+1,VecShapleys_[ii]);
    totalChk += VecShapleys_[ii];
  }
  printOutTS(PL_INFO,"Sum of Shapley values = %11.4e\n",totalChk);
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO, "Normalized Shapley Values (entropy-based):\n");
  for (ii = 0; ii < nInputs_; ii++)
    printOutTS(PL_INFO,"  Input %3d = %10.3e\n",ii+1,
               VecShapleys_[ii]/totalEntropy);
  printAsterisks(PL_INFO, 0);
  return 0;
}

// ************************************************************************
// create 2 random samples 
// ------------------------------------------------------------------------
int ShapleyAnalyzer::create2RandomSamples(aData &adata, psVector &vecXM1,
                                          psVector &vecXM2)
{
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStds;
  int    nInputs = adata.nInputs_;
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  if (inputMeans == NULL || pdfFlags == NULL || inputStdevs == NULL)
  {
    //**/ note: setLength actually sets the vector to all 0's
    vecPdfFlags.setLength(nInputs);
    pdfFlags = vecPdfFlags.getIVector();
    vecInpMeans.setLength(nInputs);
    inputMeans = vecInpMeans.getDVector();
    vecInpStds.setLength(nInputs);
    inputStdevs = vecInpStds.getDVector();
  }
  pData pCorMat;
  PsuadeData *ioPtr = adata.ioPtr_;
  ioPtr->getParameter("input_cor_matrix", pCorMat);
  psMatrix *corMatp = (psMatrix *) pCorMat.psObject_;
  int largeNumSams = 10000;
  PDFManager *pdfman = new PDFManager();
  psVector vecLB, vecUB;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdevs,
                     *corMatp,NULL,NULL);
  vecXM1.setLength(largeNumSams*nInputs);
  pdfman->genSample(largeNumSams, vecXM1, vecLB, vecUB);
  vecXM2.setLength(largeNumSams*nInputs);
  pdfman->genSample(largeNumSams, vecXM2, vecLB, vecUB);
  delete pdfman;
  return 0;
}

// ************************************************************************
// create random integer matrix 
// ------------------------------------------------------------------------
int ShapleyAnalyzer::createRandomIntMatrix(int nRows, int nCols, 
                                           psIMatrix &matIRan)
{
  int ii, ss;
  psVector vecTmp;
  vecTmp.setLength(nCols);
  psIVector vecInt;
  vecInt.setLength(nCols);
  matIRan.setDim(nRows, nCols);

  for (ss = 0; ss < nRows; ss++)
  {
    for (ii = 0; ii < nCols; ii++)
    {
      vecTmp[ii] = drand48();
      vecInt[ii] = ii;
    }
    sortDbleList2a(nCols,vecTmp.getDVector(),vecInt.getIVector());
    for (ii = 0; ii < nCols; ii++) 
      matIRan.setEntry(ss, ii, vecInt[ii]);
  } 
  return 0;
}

// ************************************************************************
// look up entropy table
// ------------------------------------------------------------------------
double ShapleyAnalyzer::ShapleyLookup(psIVector vecIn, int ind)
{
  int    ii, ss, nActive, nInp = vecIn.length();
  double ddata;
  if (nInp != MatShapleyMap_.ncols())
  {
    printf("Shapley ShapleyLookup ERROR: nInputs mismatch.\n"); 
    return -9999;
  }
  //**/ put the subset S+ into vecIT using 0/1
  psIVector vecIT;
  vecIT.setLength(nInp);
  nActive = 0;
  for (ii = 0; ii < nInp; ii++)
  {
    if (vecIn[ii] == ind) break;
    else
    {
      vecIT[vecIn[ii]] = 1;
      nActive++;
    }
  }
  vecIT[ind] = 1;

  //**/ search for a match in the table for the subset S+
  //**/ and look for H(S+)
  ddata = -9999.0;
  for (ss = 0; ss < MapLength_; ss++)
  {
    //**/ if no match, skip
    for (ii = 0; ii < nInp; ii++)
      if (vecIT[ii] != MatShapleyMap_.getEntry(ss,ii)) break;
    if (ii == nInp)
    {
      ddata = VecShapleyTable_[ss];
      break;
    }
  }
  //**/ if not found, return a token
  if (ddata == -9999.0) return ddata;

  //**/ if S is empty, H(S)=0 so just return H(S+)
  if (nActive == 0) return ddata;

  //**/ search for a match in the table for the subset S
  //**/ and look for H(S)
  vecIT[ind] = 0;
  for (ss = 0; ss < MapLength_; ss++)
  {
    //**/ if no match, skip
    for (ii = 0; ii < nInp; ii++)
      if (vecIT[ii] != MatShapleyMap_.getEntry(ss,ii)) break;
    if (ii == nInp)
    {
      ddata -= VecShapleyTable_[ss];
      return ddata;
    }
  }
  return -9999;
}

// ************************************************************************
// create a response surface
// ------------------------------------------------------------------------
FuncApprox *ShapleyAnalyzer::createResponseSurface(aData &adata)
{
  int  ss, rstype=-1;
  int  nInputs  = adata.nInputs_;
  int  nOutputs = adata.nOutputs_;
  int  nSamples = adata.nSamples_;
  int  outputID = adata.outputID_;
  char pString[1000];
  while (rstype < 0 || rstype >= PSUADE_NUM_RS)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0);
    strcpy(pString, "Choose response surface: ");
    rstype = getInt(0, PSUADE_NUM_RS, pString);
  }
  psVector vecY;
  vecY.setLength(nSamples);
  FuncApprox *faPtr = genFA(rstype, nInputs, 0, nSamples);
  faPtr->setBounds(adata.iLowerB_, adata.iUpperB_);
  faPtr->setOutputLevel(0);
  for (ss = 0; ss < nSamples; ss++)
    vecY[ss] = adata.sampleOutputs_[ss*nOutputs+outputID];
  psConfig_.InteractiveSaveAndReset();
  int status = faPtr->initialize(adata.sampleInputs_,
                                 vecY.getDVector());
  psConfig_.InteractiveRestore();
  if (status != 0)
  {
    printf("ShapleyAnalyzer ERROR: in initializing response surface.\n");
    return NULL;
  }
  return faPtr;
}   

// ************************************************************************
// perform analysis similar to MOAT analysis
// Note: This analysis is different from the one in SobolAnalyzer
// ------------------------------------------------------------------------
int ShapleyAnalyzer::MOATAnalyze(int nInputs, int nSamples, double *xIn,
                       double *yIn, double *xLower, double *xUpper,
                       double *means, double *modifiedMeans, double *stds)
{
  int    ss, ii;
  double xtemp1, xtemp2, ytemp1, ytemp2, scale;
  FILE   *fp;
  psIVector vecCounts;
  psVector  vecYT;

  //**/ ---------------------------------------------------------------
  //**/ first compute the approximate gradients
  //**/ ---------------------------------------------------------------
  vecYT.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      ytemp1 = yIn[ss+ii]; 
      ytemp2 = yIn[ss+ii-1]; 
      xtemp1 = xIn[(ss+ii)*nInputs+ii-1]; 
      xtemp2 = xIn[(ss+ii-1)*nInputs+ii-1]; 
      scale  = xUpper[ii-1] - xLower[ii-1];
      if (xtemp1 != xtemp2)
        vecYT[ss+ii] = (ytemp2-ytemp1)/(xtemp2-xtemp1)*scale;
      else
      {
        printOutTS(PL_ERROR, "Shapleynalyzer ERROR: divide by 0.\n");
        printOutTS(PL_ERROR, "     Check sample (Is this Sobol?) \n");
        exit(1);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ next compute the basic statistics
  //**/ ---------------------------------------------------------------
  vecCounts.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecCounts[ii] = 0;
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      if (vecYT[ss+ii] < 0.9*PSUADE_UNDEFINED)
      {
        means[ii-1] += vecYT[ss+ii];
        modifiedMeans[ii-1] += PABS(vecYT[ss+ii]);
        vecCounts[ii-1]++;
      }
    }
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (vecCounts[ii] > 0)
    {
      means[ii] /= (double) (vecCounts[ii]);
      modifiedMeans[ii] /= (double) (vecCounts[ii]);
    }
  }
  for (ss = 0; ss < nSamples; ss+=(nInputs+2))
  {
    for (ii = 1; ii <= nInputs; ii++)
    {
      if (vecYT[ss+ii] < 0.9*PSUADE_UNDEFINED)
        stds[ii-1] += (vecYT[ss+ii] - means[ii-1]) *
                      (vecYT[ss+ii] - means[ii-1]);
    }
  }
  for (ii = 0; ii < nInputs; ii++)
    if (vecCounts[ii] > 0)
      stds[ii] /= (double) (vecCounts[ii]);
  for (ii = 0; ii < nInputs; ii++) stds[ii] = sqrt(stds[ii]);

  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
int ShapleyAnalyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if      (!strcmp(request, "ana_shapley_entropy"))  costFunction_ = 2;
  else if (!strcmp(request, "ana_shapley_variance")) costFunction_ = 1;
  else
  {
    printOutTS(PL_ERROR,"ShapleyAnalyzer ERROR: setParams - not valid.\n");
    exit(1);
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
ShapleyAnalyzer& ShapleyAnalyzer::operator=(const ShapleyAnalyzer &)
{
  printOutTS(PL_ERROR,
           "ShapleyAnalyzer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int ShapleyAnalyzer::get_nInputs()
{
  return nInputs_;
}
double *ShapleyAnalyzer::get_svalues()
{
  psVector vecS;
  vecS = VecShapleys_;
  double *retVal = vecS.takeDVector();
  return retVal;
}
double *ShapleyAnalyzer::get_sstds()
{
  psVector vecS;
  vecS = VecShapleyStds_;
  double *retVal = vecS.takeDVector();
  return retVal;
}

