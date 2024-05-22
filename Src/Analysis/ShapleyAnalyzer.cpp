// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public 
// License.
//
// PSUADE is free software; you can redistribute it and/or modify it under 
// the terms of the GNU Lesser General Public License (as published by the 
// Free Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of 
// the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class ShapleyAnalyzer
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ShapleyAnalyzer.h"
#include "SobolAnalyzer.h"
#include "RSMEntropy1Analyzer.h"
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
#define NSAM 1000

// ------------------------------------------------------------------------
// Both uniform and adaptive seem to be stable for entropy-based method
// although they give slightly different values
// ------------------------------------------------------------------------
#define _ADAPTIVE_
// ------------------------------------------------------------------------

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
ShapleyAnalyzer::ShapleyAnalyzer() : Analyzer(), nInputs_(0)
{
  setName("Shapley");
  sampleSize_ = 100000;
  costFunction_ = 1;   /* 0: VCE-based, 1: TSI-based, 2: entropy-based */
  MaxMapLength_=10000; /* for entropy-based method */
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
    printOutTS(PL_ERROR,"ShapleyAnalyzer ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 

  //**/ ---------------------------------------------------------------
  //**/ check for valid samples
  //**/ ---------------------------------------------------------------
  int ss, errCnt=0, outputID = adata.outputID_;
  for (ss = 0; ss < nSamples; ss++)
    if (adata.sampleOutputs_[ss*nOutputs+outputID]>0.9*PSUADE_UNDEFINED)
      errCnt++;
  if (errCnt > 0)
  {
    printf("ShapleyAnalyzer ERROR: Found %d invalid sample points.\n",
           errCnt);
    exit(1);
  }

  //**/ ---------------------------------------------------------------
  //**/ analyze using one of the 3 algorithms
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"You can select one of the 3 below : \n");
    printOutTS(PL_INFO,"0. Sobol'-based method : main effect\n");
    printOutTS(PL_INFO,"1. Sobol'-based method : total effect\n");
    printOutTS(PL_INFO,"2. Entropy-based method : E[H(Y|X)]\n");
    snprintf(pString,100,"Select 0, 1 or 2 : ");
    costFunction_ = getInt(0, 2, pString);
  }
  if      (costFunction_ == 0) return analyzeVCE(adata);
  else if (costFunction_ <= 1) return analyzeTSI(adata);
  else                         return analyzeEntropy(adata);
}
 
// ************************************************************************
// perform analysis using VCE as cost function
// The VCE-based Algorithm uses
// (1) Sobol' first-order sensitivity as the Shapley cost function, 
// (2) Sobol' algorithm to compute the first-order sensitivity effect, 
// (3) Monte-Carlo algorithm to compute approximate Shapley effect (instead
//     of going through all permutations) 
// phi(i) = sum_{u \in X\i} [1/m C(m-1,|u|)^{-1} (vce(u+i) - vce(u))]
// where m = nInputs, X is the set of all m inputs
//       1/m C(m-1,|u|)^{-1} = (m - 1 - |u|)! |u|! / m!
//
// Hence, if u is a subset of all the inputs X (let D(u,i)=vce(u+i)-vce(u))
// phi(i) = sum_{u \in X\i} [1/m C(m-1,|u|)^{-1} D(u,i)
//        = 1/m sum_{k=0}^{m-1} C(m-1,k)^-1 sum_{u \in X\i, |u|=k} D(u,i)
// where
// vce(u)   = int_x int_y F(x) F(x_u) p(x) p(y) dx dy
// vce(u+i) = int_x int_y F(x) F(x_u+i) p(x) p(y) dx dy
// 
// Let sample size be N, S_n(k~i) is the subset of size k for sample n
//     that does not contain input i
// Let there be 2 random matrices M1 and M2 (N x nInputs)
// Let F_n(M1(S_n(k~i))) = model output Y by taking subset S_n(k~i) from 
//     the n-th sample point in M1 and the rest from the n-th sample
//     point of M2 including input i (S_n(k~i) does not have i)
// Let F_n(M1(S_n(k~i)+i)) = model output Y by taking subset S_n(k~i) 
//     plue input i from the n-th sample point in M1 and the rest from 
//     the n-th sample point of M2 
// Let F_n(M1) = model output Y by taking all inputs from M1 sample n
//
// phi(i) = sum_n sum_i (F_n(M1) F_n(M1(S_n(k~i)+i)) - u^2) -
//                      (F_n(M1) F_n(M1(S_n(k~i))) - u^2)
//        = sum_n sum_i (F_n(M1) F_n(M1(S_n(k~i)+i))) -
//                      (F_n(M1) F_n(M1(S_n(k~i)))) 
//        = sum_n sum_i  F_n(M1) [F_n(M1(S_n(k~i)+i)) - F_n(M1(S_n(k~i)))] 
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
  sampleSize_ = 100000;
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* SobolVCE-based Shapley uses a large sample ");
    printOutTS(PL_INFO,"to compute sensitivity\n");
    printOutTS(PL_INFO,"* indices. The default sample size M is %d.\n",
               sampleSize_);
    printOutTS(PL_INFO,"* You may, however, select a different M.\n");
    printOutTS(PL_INFO,"* NOTE: Large M may take long time, but ");
    printOutTS(PL_INFO,"gives more accurate results.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 10000-1000000) : ");
    sampleSize_ = getInt(10000, 10000000, pString);
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
  psIMatrix matRandInt;
  createRandomIntMatrix(sampleSize_, nInputs, matRandInt);

  //**/ ---------------------------------------------------------------
  //**/ create response surface 
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = createResponseSurface(adata);
  if (faPtr == NULL) return -1;

  //**/ ---------------------------------------------------------------
  //**/ evaluate vecLargeX using response surface ==> vecLargeY 
  //**/ ---------------------------------------------------------------
  psVector vecYM1;
  vecYM1.setLength(sampleSize_);
  faPtr->evaluatePoint(sampleSize_,vecXM1.getDVector(),
                       vecYM1.getDVector());
     
  //**/ ---------------------------------------------------------------
  //**/ compute basic statistics
  //**/ ---------------------------------------------------------------
  double dmean, dVar;
  int status = computeBasicStatistics(vecYM1, dmean, dVar);
  dVar = dVar * dVar;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"Shapley: Sample mean     = %10.3e\n",dmean);
    printOutTS(PL_INFO,"Shapley: Sample variance = %10.3e\n",dVar);
  }
  if (dVar == 0)
  {
    printf("Shapley INFO: Variance = 0 ==> No input sensitivities.\n");
    VecShapleys_.setLength(nInputs_);
    return 0;
  }

  //**/ ---------------------------------------------------------------
  //**/ perform Shapley analysis (vecXM1, vecYM1 has M1)
  //**/ ---------------------------------------------------------------
  int    ii, ss, kk, ind, count;
  double ddata;
  VecShapleys_.setLength(nInputs_);
  VecShapleyStds_.setLength(nInputs_);
  //**/ XMM, YMM have S(k~i) from M1 and the rest from M2
  psVector vecXMM, vecYMM;
  vecXMM.setLength(sampleSize_*nInputs);
  vecYMM.setLength(sampleSize_);
  //**/ XMP, YMP have S(k~i)+{i} from M1 and the rest from M2
  psVector vecXMP, vecYMP;
  vecXMP.setLength(sampleSize_*nInputs);
  vecYMP.setLength(sampleSize_);

  for (ii = 0; ii < nInputs; ii++)
  {
    //**/ create a sample vecXMM which has random selection from M1
    //**/ and the rest as well as the i-th input coming from M2
    //**/ XMM ==> XMM = M2, XMM(S_n(k~i)) = M1
    //**/ XMP ==> XMM = M2, XMM(S_n(k~i)+i) = M1
    for (ss = 0; ss < sampleSize_; ss++)
    {
      //**/ for XMM, first fill the whole row with M2
      for (kk = 0; kk < nInputs; kk++)
        vecXMM[ss*nInputs+kk] = vecXM2[ss*nInputs+kk];
      //**/ then fill with M1 until index ii is encountered
      //**/ (excluding ii)
      for (kk = 0; kk < nInputs; kk++)
      {
        ind = matRandInt.getEntry(ss, kk);
        if (ind == ii) break;
        vecXMM[ss*nInputs+ind] = vecXM1[ss*nInputs+ind];
      }
      //**/ for XMP, first fill the whole row with M2
      for (kk = 0; kk < nInputs; kk++)
        vecXMP[ss*nInputs+kk] = vecXM2[ss*nInputs+kk];
      //**/ then fill with M1 until index ii is encountered
      //**/ (including ii)
      for (kk = 0; kk < nInputs; kk++)
      {
        ind = matRandInt.getEntry(ss, kk);
        vecXMP[ss*nInputs+ind] = vecXM1[ss*nInputs+ind];
        if (ind == ii) break;
      }
    }
    
    //**/ evaluate the modified sample without input ii 
    faPtr->evaluatePoint(sampleSize_,vecXMM.getDVector(),
                         vecYMM.getDVector());

    //**/ evaluate the modified sample with input ii 
    faPtr->evaluatePoint(sampleSize_,vecXMP.getDVector(),
                         vecYMP.getDVector());

    //**/ sum F_n(M1) [F_n(M2(S_n(k~i)+i)) - F_n(M2(S_n(k~i)))] 
    for (ss = 0; ss < sampleSize_; ss++)
    {
      ddata = vecYM1[ss] * (vecYMP[ss] - vecYMM[ss]);
      VecShapleys_[ind] += ddata; 
      VecShapleyStds_[ind] += (ddata * ddata); 
    }
  }

  //**/ take average of all sample points
  for (ii = 0; ii < nInputs; ii++)
  {
    VecShapleys_[ii] /= sampleSize_;
    VecShapleyStds_[ii] /= sampleSize_;
  }

  printOutTS(PL_INFO, "Shapley Values (VCE-based):\n");
  double totalChk=0, lb, ub;
  for (ii = 0; ii < nInputs; ii++)
  {
    VecShapleyStds_[ii] = 
      (VecShapleyStds_[ii]-pow(VecShapleys_[ii],2.0))/(sampleSize_-1); 
    lb = VecShapleys_[ii] - 1.96*VecShapleyStds_[ii];
    ub = VecShapleys_[ii] + 1.96*VecShapleyStds_[ii];
    if (VecShapleys_[ii] < 0) VecShapleys_[ii] = 0;
    if (lb < 0) lb = 0;
    if (ub < 0) ub = - ub;
    printOutTS(PL_INFO,
      "  Input %3d = %9.3e [%9.3e, %9.3e], Normalized = %9.3e\n",
      ii+1,VecShapleys_[ii],lb,ub,VecShapleys_[ii]/dVar);
    totalChk += VecShapleys_[ii];
  }
  printOutTS(PL_INFO,"Sum of Shapley values = %10.4e\n",totalChk);
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"Normalized Shapley Values (VCE-based):\n");
  for (ii = 0; ii < nInputs_; ii++)
    printOutTS(PL_INFO,"  Input %3d = %9.3e\n",ii+1,
               VecShapleys_[ii]/dVar);
  printAsterisks(PL_INFO, 0);
  if (adata.printLevel_ > 2)
  { 
    //**/ perform Shapley based on MOAT modified gradients
    psVector vecShapleyMOAT;
    vecShapleyMOAT.setLength(nInputs);
    psVector vecLargeSample;
    vecLargeSample.setLength(sampleSize_*(nInputs+2)*nInputs);
    count = 0;
    for (ss = 0; ss < sampleSize_; ss++)
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
    //**/ size = sampleSize_*(nInputs+2)
    //**/ evaluate the sample 
    psVector vecLargeY;
    vecLargeY.setLength(sampleSize_*(nInputs+2));
    count = sampleSize_ * (nInputs + 2);
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
// perform analysis using TSI as cost function
// The TSI-based Algorithm based on that described in
// 'A Simple Algorithm for global sensitivity analysis with Shapley Effect'
// by T. Goda. Three of the main ingredients of the algorithms are:
// (1) use Sobol' total sensitivity effect as the Shapley value function, 
// (2) use the Sobol' algorithm to compute the total sensitivity effect, 
// (3) a Monte-Carlo step to compute approximate Shapley effect (instead
//     of going through all permutations) 
// phi(i) = sum_{u \in X\i} [1/m C(m-1,|u|)^{-1} (tsi(u+i) - tsi(u))]
// where m = nInputs, X is the set of all m inputs
//       1/m C(m-1,|u|)^{-1} = (m - 1 - |u|)! |u|! / m!
//
// Hence, if u is a subset of all the inputs X (let D(u,i)=tsi(u+i)-tsi(u))
// phi(i) = sum_{u \in X\i} [1/m C(m-1,|u|)^{-1} D(u,i)
//        = 1/m sum_{k=0}^{m-1} C(m-1,k)^-1 sum_{u \in X\i, |u|=k} D(u,i)
// where
// tsi(u) = 1/2 int_x int_y (F(x) - F(x_u))^2 p(x) p(y) dx dy
// 
// Let sample size be N, S_n(k~i) is the subset of size k for sample n
//     that does not contain input i
// Let there be 2 random matrices M1 and M2 (N x nInputs)
// Let F_n(M2(S_n(k~i))) = model output Y by taking subset S_n(k~i) from 
//     the n-th sample point in M2 and the rest from the n-th sample
//     point of M1 including input i (S_n(k~i) does not have i)
// Let F_n(M2(S_n(k~i)+i)) = model output Y by taking subset S_n(k~i) 
//     plue input i from the n-th sample point in M2 and the rest from 
//     the n-th sample point of M1 
// Let F_n(M1) = model output Y by taking all inputs from M1 sample n
// phi(i) = sum_n sum_i 1/2 (F_n(M1) - F_n(M2(S_n(k~i)+i)))^2 -
//                      1/2 (F_n(M1) - F_n(M2(S_n(k~i))))^2
// Let A_n(i) = F_n(M2(S_n(k~i)+i)) 
//     B_n(i) = F_n(M2(S_n(k~i)))
// phi(i) 
// = 1/2 sum_n sum_i (F_n(M1) - A_n(i))^2 - (F_n(M1) - B_n(i))^2
// = 1/2 sum_n sum_i [F_n(M1)^2 - 2 * F_n(M1) * A_n(i) + A_n(i)^2 -
//                    F_n(M1)^2 + 2 * F_n(M1) * B_n(i) - B_n(i)^2]
// = 1/2 sum_n sum_i [2 F_n(M1) - A_n(i) - B_n(i)] * [B_n(i) - A_n(i)]]
// = sum_n sum_i [F_n(M1) - 1/2 A_n(i) - 1/2 B_n(i)] * [B_n(i) - A_n(i)]]
// = sum_n sum_i C_n(i) * D_n(i)
// where
// C_n(i) = F_n(M1) - 1/2 F_n(M2(S(k~i)+i)) - 1/2 F_n(M2(S_n(k~i)}))
// D_n(i) = F_n(M2(S(k~i))) - F_n(M2(S_n(k~i)+i))
// ------------------------------------------------------------------------
double ShapleyAnalyzer::analyzeTSI(aData &adata)
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
    printOutTS(PL_INFO,
      "* SobolTSI-based Shapley uses two large samples ");
    printOutTS(PL_INFO,"to compute sensitivity\n");
    printOutTS(PL_INFO,
      "* indices. The default sample size M is %d.\n",sampleSize_);
    printOutTS(PL_INFO,"* You may, however, select a different M.\n");
    printOutTS(PL_INFO,"* NOTE: Large M may take long time, but ");
    printOutTS(PL_INFO,"gives more accurate results.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 10000-1000000) : ");
    sampleSize_ = getInt(10000, 10000000, pString);
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
  psIMatrix matRandInt;
  createRandomIntMatrix(sampleSize_, nInputs, matRandInt);

  //**/ ---------------------------------------------------------------
  //**/ create response surface 
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = createResponseSurface(adata);
  if (faPtr == NULL) return -1;

  //**/ ---------------------------------------------------------------
  //**/ evaluate vecXM1 using response surface ==> vecYM1 
  //**/ ---------------------------------------------------------------
  psVector vecYM1;
  vecYM1.setLength(sampleSize_);
  faPtr->evaluatePoint(sampleSize_,vecXM1.getDVector(),
                       vecYM1.getDVector());
     
  //**/ ---------------------------------------------------------------
  //**/ compute basic statistics
  //**/ ---------------------------------------------------------------
  double dmean, dVar;
  int status = computeBasicStatistics(vecYM1, dmean, dVar);
  dVar = dVar * dVar;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"Shapley: Sample mean     = %10.3e\n",dmean);
    printOutTS(PL_INFO,"Shapley: Sample variance = %10.3e\n",dVar);
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
  int    ss, ii, ind;
  double ddata;
  VecShapleys_.setLength(nInputs_);
  VecShapleyStds_.setLength(nInputs_);
  psVector vecXMM, vecYMP, vecYMM;
  vecXMM = vecXM1; /* X0 = X */
  vecYMP = vecYM1; /* YS = F(X) - set of u */
  vecYMM.setLength(sampleSize_);

  for (ii = 0; ii < nInputs; ii++)
  {
    //**/ evolve sample (X = X1, X(ind) = X2(ind))
    //**/ So for ii = 0, XMM will differ from XM1 by 1
    //**/ So for ii = 1, XMM will differ from XM1 by 2
    //**/ So as ii increases, XMM goes from XM1 to XM2 for
    //**/ every ss
    for (ss = 0; ss < sampleSize_; ss++)
    {
      ind = matRandInt.getEntry(ss, ii);
      vecXMM[ss*nInputs+ind] = vecXM2[ss*nInputs+ind];
    }
    
    //**/ evaluate the newly-evolved sample (FS = F(X))
    faPtr->evaluatePoint(sampleSize_,vecXMM.getDVector(),
                         vecYMM.getDVector());

    //**/ TSI-based
    //**/ gamma = tsi(u+i) - tsi(u) = C(i) * D(i)
    //**/ C_n(i) = F_n(M1) - 1/2[F_n(M2(S(k~i)+i))+F_n(M2(S_n(k~i)))]
    //**/ D_n(i) = F_n(M2(S(k~i))) - F_n(M2(S_n(k~i)+i))
    //**/ update Shapley values (phi1(i)  = phi1(i) + gamma/N)
    //**/ update Shapley variance (var(i) = var(i) + gamma*gamma/N)
    for (ss = 0; ss < sampleSize_; ss++)
    {
      ind = matRandInt.getEntry(ss, ii);
      ddata = (vecYM1[ss]-0.5*(vecYMM[ss]+vecYMP[ss]))*
              (vecYMP[ss]-vecYMM[ss]); 
      VecShapleys_[ind] += ddata; 
      VecShapleyStds_[ind] += (ddata * ddata); 
    }
    //**/ advance
    vecYMP = vecYMM;
  }

  //**/ take average of all sample points
  for (ii = 0; ii < nInputs; ii++)
  {
    VecShapleys_[ii] /= sampleSize_;
    VecShapleyStds_[ii] /= sampleSize_;
  }

  //**/ print results
  printOutTS(PL_INFO, "Shapley Values (TSI-based):\n");
  double totalChk=0, lb, ub;
  for (ii = 0; ii < nInputs; ii++)
  {
    VecShapleyStds_[ii] = 
      (VecShapleyStds_[ii]-pow(VecShapleys_[ii],2.0))/(sampleSize_-1); 
    lb = VecShapleys_[ii] - 1.96*VecShapleyStds_[ii];
    ub = VecShapleys_[ii] + 1.96*VecShapleyStds_[ii];
    if (VecShapleys_[ii] < 0) VecShapleys_[ii] = 0; 
    if (lb < 0) lb = 0;
    if (ub < 0) ub = - ub;
    printOutTS(PL_INFO,
      "  Input %3d = %9.3e [%9.3e, %9.3e], Normalized = %9.3e\n",
      ii+1,VecShapleys_[ii], lb, ub, VecShapleys_[ii]/dVar);
    totalChk += VecShapleys_[ii];
  }
  printOutTS(PL_INFO,"Sum of Shapley values = %10.4e\n",totalChk);
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"Normalized Shapley Values (TSI-based):\n");
  for (ii = 0; ii < nInputs_; ii++)
    printOutTS(PL_INFO,"  Input %3d = %9.3e\n",ii+1,
               VecShapleys_[ii]/dVar);
  if (adata.printLevel_ > 2)
  { 
    int kk, count;
    printAsterisks(PL_INFO, 0);
    //**/ perform Shapley based on MOAT modified gradients
    psVector vecShapleyMOAT;
    vecShapleyMOAT.setLength(nInputs);
    psVector vecLargeSample;
    vecLargeSample.setLength(sampleSize_*(nInputs+2)*nInputs);
    count = 0;
    printf("Perform Morris-one-at-a-time analysis:\n");
    for (ss = 0; ss < sampleSize_; ss++)
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
    //**/ size = sampleSize_*(nInputs+2)
    //**/ evaluate the sample 
    psVector vecLargeY;
    vecLargeY.setLength(sampleSize_*(nInputs+2));
    count = sampleSize_ * (nInputs + 2);
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
  int nLevels=50;      /* number of levels in histogramming */
  sampleSize_ = 10000; /* sample size to compute Shapley */
  int maxSamples = 1000000;
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    char pString[1000];
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* This entropy-based Shapley value ");
    printOutTS(PL_INFO,"method uses a large sample to\n");
    printOutTS(PL_INFO,"* approximate the Shapley permutations, ");
    printOutTS(PL_INFO,"and for each sample point\n");
    printOutTS(PL_INFO,"* computes changes in entropy by ");
    printOutTS(PL_INFO,"adding one input.\n");
    printOutTS(PL_INFO,
      "* The default sample size M = %d\n", sampleSize_);
    printEquals(PL_INFO, 0);
    snprintf(pString,100,
      "Enter a new M (suggestion: 1000-100000) : ");
    sampleSize_ = getInt(1000, 1000000, pString);
    printOutTS(PL_INFO,"* In each of the %d iteration, a LARGE ",
               sampleSize_);
    printOutTS(PL_INFO,"sample is to be created to\n");
    printOutTS(PL_INFO,"* compute entropies. The default LARGE ");
    printOutTS(PL_INFO,"sample size N is %d.\n",maxSamples);
    snprintf(pString,100,
      "Enter a new N (suggestion: 1000000-10000000) : ");
    maxSamples = getInt(1000000, 10000000, pString);
  }

  //**/ ---------------------------------------------------------------
  //**/ create response surface 
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = createResponseSurface(adata);
  if (faPtr == NULL) 
  {
    printf("Shapley1 ERROR: Cannot create response surface.\n");
    return -1;
  }

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
  int      ii, ss, nSam=100000, iOne=1, status;
  double   totalEntropy, dOne=1, Ymax, Ymin, Ywidth, ddata;
  psVector vecSam, vecL, vecU, vecY;
  psMatrix matCov;

  //**/ use PDFManager to create sample with desired distributions
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

  //**/ evaluate sample points using response surface
  vecY.setLength(nSam);
  faPtr->evaluatePoint(nSam,vecSam.getDVector(),vecY.getDVector());

  //**/ create histogram using ProbMatrix, which needs lower and
  //**/ upper bounds of the data (Ymax, Ymin) 
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
    printf("Shapley1 INFO: Ymin (%e) = Ymax (%e)\n",Ymin,Ymax);
    printf("         ===> Assume zero total entropy w.r.t. ");
    printf("input variations.\n");
    printf("         ===> Assume zero Shapley values for all inputs.\n");
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
  RSMEntropy1Analyzer entPtr;
#ifdef _ADAPTIVE_
  //**/ bin widths vary to make all bins having same count
  entPtr.computeEntropy(matProb, vecYL, vecYU, totalEntropy, iOne);
#else
  //**/ bin widths constant = 1/nLevels
  entPtr.computeEntropy(matProb, vecYL, vecYU, totalEntropy, iOne);
#endif
  printf("Output entropy = %e\n", totalEntropy);

  //**/ ---------------------------------------------------------------
  //**/ declare all variables needed for processing
  //**/ ---------------------------------------------------------------
  //**/ need 2 sample generator (2 PDFManagers)
  PDFManager *pdfman1=NULL, *pdfman2=NULL;
  //**/ for generating random sets
  psIVector vecIRand;
  psVector  vecRand;
  //**/ for input and output bounds need for histogramming
  psVector vecL1, vecL2, vecU1, vecU2, vecYL1, vecYU1, vecYL2, vecYU2; 
  //**/ for input means and std devs of the 2 subsets of inputs
  psVector  vecIMean1, vecIMean2, vecIStdv1, vecIStdv2;
  psIVector vecIPDF1, vecIPDF2;
  psMatrix matCov1, matCov2;
  //**/ for storing the 2 sub-samples and entire sample
  psVector vecSam1, vecSam2, vecLargeSam, vecLargeY;
  //**/ for lookup to improve efficiency
  MatShapleyMap_.setDim(MaxMapLength_, nInputs_);
  MapLength_ = 0;
  VecShapleyTable_.setLength(MaxMapLength_);
  //**/ this is for allocating sample sizes for each sub-sample
  int samPerInp = pow(1.0*maxSamples, 1.0/nInputs_) + 1;
  //**/ system variable to store the computed Shapley values
  VecShapleys_.setLength(nInputs_);

  //**/ ---------------------------------------------------------------
  //**/ process each input 
  //**/ ---------------------------------------------------------------
  int    ii2, ss1, ss2, nInp1, nInp2, nSam1, nSam2, ind;
  double entropy1, entropy2, entropy, entTemp, entTemp2;
#pragma omp parallel shared(ii,faPtr) \
    private(ss,ii2,vecRand,vecIRand,nInp1,nInp2,vecIPDF1,vecIMean1,\
        vecIStdv1,vecL1,vecU1,matCov1,vecIPDF2,vecIMean2,vecIStdv2,\
        vecL2,vecU2,matCov2,nSam1,nSam2,pdfman1,pdfman2,vecSam1,vecSam2,\
        ind,ss1,ss2,entropy1,entropy2,entropy,vecLargeSam,vecLargeY,\
        status,vecYL1,vecYL2,vecYU1,vecYU2,matProb,ddata,Ymin,Ymax,\
        entTemp, entTemp2)
{
#pragma omp for
  for (ii = 0; ii < nInputs_; ii++)
  {
#ifdef PSUADE_OMP
    printf("Processing input %d (thread=%d)",ii+1,omp_get_thread_num());
#else
    printf("Processing input %d ", ii+1);
#endif
    //**/---------------------------------------------------------------
    //**/ Let I be the set of all inputs. The algorithms goes like this:
    //**/ For each input:
    //**/ A. Compute mean entropy gain by taking the mean of the 
    //**/    following steps sampleSize_ times:  
    //**/ 1. Select a random subset of I without input ii and call it S
    //**/ 2. Form another subset to be union of S and ii (call it S+)
    //**/ 3. Compute entropy gain for S : H(S+) - H(S)
    //**/    a. Compute entropy for H(S+)
    //**/       - Create a sample for S+ (assume independent inputs)
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
    //**/---------------------------------------------------------------
    //**/ These 2 lines need to be here for OMP as they are private
    vecIRand.setLength(nInputs_);
    vecRand.setLength(nInputs_);

    //**/ cycle through the Shapley samples
    for (ss = 0; ss < sampleSize_; ss++)
    {
      //**/ display progress
      if ((ss % (sampleSize_/100)) == 0) 
      {
        printf(".");
        fflush(stdout);
      }

      //**/ generate a random subset S by using random numbers and 
      //**/ sorting (note the subset is what is in vecIRand up to
      //**/ the entry having the value ii)
      for (ii2 = 0; ii2 < nInputs_; ii2++) 
      {
        vecRand[ii2] = drand48(); 
        vecIRand[ii2] = ii2;
      }
      sortDbleList2a(nInputs_,vecRand.getDVector(),
                     vecIRand.getIVector());

      //**/ look up to see if this permutation has been analyzed
      //**/ before. If so, just return the value (including ii)
      //**/ (If the last index is ii, it is certainly not in 
      //**/ lookup table, so skip lookup)
      ddata = -9999;
      if (vecIRand[nInputs_-1] != ii)
      {
        ddata = ShapleyEntropyLookup(vecIRand, ii, 1);
        if (ddata != -9999)
        {
          //VecShapleys_[ii] += PABS(ddata);
          VecShapleys_[ii] += ddata;
          continue;
        }
      }

      //**/ search for the ii index (the position of the ii index 
      //**/ will be used as random subset size (size(S) = ii2)
      for (ii2 = 0; ii2 < nInputs_; ii2++) 
        if (vecIRand[ii2] == ii) break;

      //**/ ========================================================
      //**/ compute entropies for the 2 sets: S+ and I\S+
      //**/ --------------------------------------------------------
      //**/ set nInp1 = size(S+) (i.e. including input ii)
      //**/ set nInp2 = size(I\S+) 
      nInp1 = ii2 + 1;
      nInp2 = nInputs_ - nInp1;
      
      //**/ Construct a sample for S+ (sample size is somewhat
      //**/ arbitrary - just large enough for reasonable statistics)
      //**/ if S = all inputs, the entropy is just the total entropy
      if (nInp1 == nInputs_) entropy1 = totalEntropy;
      else
      {
        if (adata.printLevel_ > 2)
          printf("Shapley INFO: Running entropy (%d of %d)\n",
                 ss+1,sampleSize_);  
        //nSam1 = (int) pow(1.0*samPerInp, nInp1);
        //if (nSam1 < 1000) nSam1 = 1000;
        nSam1 = NSAM;
        pdfman1 = NULL;
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

        //**/ construct a sample for I\S+ (sample size is also somewhat
        //**/ arbitrary - just large enough for reasonable statistics)
        //**/ If size(I\S+)=0, nSam2 = 1 ==> no need to compute average
        if (nInp2 == 0) nSam2 = 1;
        else            nSam2 = NSAM;
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
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Running function evaluation 1\n");
        vecLargeY.setLength(nSam1*nSam2);
        faPtr->evaluatePoint(nSam1*nSam2,vecLargeSam.getDVector(),
                             vecLargeY.getDVector());
           
        //**/ for each of the sample point for I\S+, compute entropy
        //**/ then take the mean ==> entropy1
        //printf("   - compute entropy\n");
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Computing entropy for S+\n");
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
          if (Ymin < Ymax)
          {
            vecYL1.setLength(iOne);
            vecYU1.setLength(iOne);
            vecYL1[0] = Ymin;
            vecYU1[0] = Ymax;
#ifdef _ADAPTIVE_
            entPtr.computeEntropy(matProb,vecYL1,vecYU1,entropy,iOne);
#else
            entPtr.computeEntropy(matProb,vecYL1,vecYU1,entropy,0);
#endif
            //**/ sum all entropies from all sample points in sample 2
            entropy1 += entropy;
          } /* Ymax > Ymin */
        }
        //**/ compute mean entropy
        entropy1 /= (double) nSam2;

        //**/ ========================================================
        //**/ Process I\S+
        //**/ --------------------------------------------------------
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Computing entropy for I\\S+\n");
        entTemp = 0;
        for (ss1 = 0; ss1 < nSam1; ss1++)
        {
          matProb.setDim(nSam2, iOne);
          //**/ search for lower and upper bounds of Y
          //**/ the reason to use tight bounds is that loose bound does
          //**/ not reflect the true delta Y due to discrete intervals
          //**/ because pdf(interval Y_i) ~ count/delta Y_i
          Ymax = -PSUADE_UNDEFINED;
          Ymin = +PSUADE_UNDEFINED;
          ind  = 0;
          for (ss2 = ss1; ss2 < nSam1*nSam2; ss2+=nSam1)
          {
            ddata = vecLargeY[ss2];
            if (ddata > Ymax) Ymax = ddata;
            if (ddata < Ymin) Ymin = ddata;
            matProb.setEntry(ind,0,ddata);
            ind++;
          }
          if (Ymin < Ymax)
          {
            vecYL1.setLength(iOne);
            vecYU1.setLength(iOne);
            vecYL1[0] = Ymin;
            vecYU1[0] = Ymax;
#ifdef _ADAPTIVE_
            entPtr.computeEntropy(matProb,vecYL1,vecYU1,entropy,iOne);
#else
            entPtr.computeEntropy(matProb,vecYL1,vecYU1,entropy,0);
#endif
            //**/ sum all entropies from all sample points in sample 2
            entTemp += entropy;
          } /* Ymax > Ymin */
        }
        //**/ compute mean entropy
        entTemp /= (double) nSam1;
      }
      //**/ store the entropies for S+ and I\S+
      if (nInp1 != nInputs_)
      {
#pragma omp critical
{
        //**/ store the result
        ShapleyEntropySave(vecIRand, 0, nInp1, entropy1);

        //**/ store the result
        //ShapleyEntropySave(vecIRand, nInp1, nInputs_, entTemp);
}
      }

      //**/ ========================================================
      //**/ Process S
      //**/ nInp1 = size(S)    (note nInp1 may be 0)
      //**/ nInp2 = size(I\S)
      //**/ --------------------------------------------------------
      nInp1--;
      nInp2 = nInputs_ - nInp1;
      
      //**/ look up to see if this permutation has been analyzed
      //**/ before. If so, just return the value
      if (nInp1 > 0)
      {
        //**/ only search up to and before ii
        entropy2 = ShapleyEntropyLookup(vecIRand, ii, 0);
        if (entropy2 != -9999)
        {
          //VecShapleys_[ii] += PABS(entropy1 - entropy2);
          VecShapleys_[ii] += entropy1 - entropy2;
          continue;
        }
      }

      //**/ if nInp1 == 0, entropy H(S)=0 (no variation)
      //**/ if nInp1 == nInputs, entropy H(S)=H(Y) (total variation)
      //**/ Construct a sample for S (sample size is somewhat
      //**/ arbitrary - just large enough for reasonable statistics)
      entropy2 = 0;
      if (nInp1 > 0)
      {
        //nSam1 = (int) pow(1.0*samPerInp, nInp1);
        //if (nSam1 < 1000) nSam1 = 1000;
        nSam1 = NSAM;
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

        //**/ nSam2 fixed
        nSam2 = NSAM;
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
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Function evaluation 2\n");
        faPtr->evaluatePoint(nSam1*nSam2,vecLargeSam.getDVector(),
                             vecLargeY.getDVector());
     
        //**/ compute mean of entropy
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Computing entropy for S\n");
        entropy2 = 0;
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
          //**/ binning
          if (Ymin < Ymax)
          {
            vecYL2.setLength(iOne);
            vecYU2.setLength(iOne);
            vecYL2[0] = Ymin;
            vecYU2[0] = Ymax;
#ifdef _ADAPTIVE_
            entPtr.computeEntropy(matProb,vecYL2,vecYU2,entropy,iOne);
#else
            entPtr.computeEntropy(matProb,vecYL2,vecYU2,entropy,0);
#endif
            //**/ sum all entropies from all sample points in sample 2
            entropy2 += entropy;
          } /* Ymax > Ymin */
        }
        //**/ compute mean entropy
        entropy2 /= (double) nSam2;

        //**/ --------------------------------------------------------
        //**/ compute mean of entropy for the I\S
        //**/ --------------------------------------------------------
        if (psConfig_.DiagnosticsIsOn())
          printf("Shapley INFO: Computing entropy for I\\S\n");
        entTemp2 = 0;
        for (ss1 = 0; ss1 < nSam1; ss1++)
        {
          matProb.setDim(nSam2, iOne);
          Ymax = -PSUADE_UNDEFINED;
          Ymin = +PSUADE_UNDEFINED;
          ind  = 0;
          for (ss2 = ss1; ss2 < nSam1*nSam2; ss2+=nSam1)
          {
            ddata = vecLargeY[ss2];
            if (ddata > Ymax) Ymax = ddata;
            if (ddata < Ymin) Ymin = ddata;
            matProb.setEntry(ind,0,ddata);
            ind++;
          }
          //**/ binning
          if (Ymin < Ymax)
          {
            vecYL2.setLength(iOne);
            vecYU2.setLength(iOne);
            vecYL2[0] = Ymin;
            vecYU2[0] = Ymax;
#ifdef _ADAPTIVE_
            if (psConfig_.DiagnosticsIsOn())
              printf("Shapley INFO: Computing entropy for I\\S (a)\n");
            entPtr.computeEntropy(matProb,vecYL2,vecYU2,entropy,iOne);
            if (psConfig_.DiagnosticsIsOn())
              printf("Shapley INFO: Computing entropy for I\\S (b)\n");
#else
            entPtr.computeEntropy(matProb,vecYL2,vecYU2,entropy,0);
#endif
            //**/ sum all entropies from all sample points in sample 2
            entTemp2 += entropy;
          } /* Ymax > Ymin */
        }
        //**/ compute mean entropy
        entTemp2 /= (double) nSam1;
      }
      //**/ store the entropies for S+ and I\S+
      if (nInp1 > 0)
      {
#pragma omp critical
{
        //**/ store the result
        ShapleyEntropySave(vecIRand, 0, nInp1, entropy2);

        //**/ store the result
        //ShapleyEntropySave(vecIRand, nInp1, nInputs_, entTemp2);
}
      }

      //**/ accumulate entropy gain
      //VecShapleys_[ii] += PABS(entropy1 - entropy2);
      VecShapleys_[ii] += entropy1 - entropy2;
    } /* different subset ss */
    printf("\n");
    VecShapleys_[ii] /= (double) sampleSize_;
#ifndef PSUADE_OMP
    if (adata.printLevel_ > 0)
      printOutTS(PL_INFO,
         " ==> Shapley value = %9.3e\n",VecShapleys_[ii]);
#endif
  }
} /* omp parallel */
  printOutTS(PL_INFO, "Shapley Values (entropy-based):\n");
  double totalChk=0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    printOutTS(PL_INFO,
         "  Input %3d = %10.3e\n",ii+1,VecShapleys_[ii]);
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
  PDFManager *pdfman = new PDFManager();
  psVector vecLB, vecUB;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdevs,
                     *corMatp,NULL,NULL);
  vecXM1.setLength(sampleSize_*nInputs);
  pdfman->genSample(sampleSize_, vecXM1, vecLB, vecUB);
  vecXM2.setLength(sampleSize_*nInputs);
  pdfman->genSample(sampleSize_, vecXM2, vecLB, vecUB);
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
//**/ look up entropy for inputs in vecIn up to ind (and including ind)
//**/ look up entropy for inputs in vecIn up to ind (and excluding ind)
//**/ subtract the second from the first and return the value
// ------------------------------------------------------------------------
double ShapleyAnalyzer::ShapleyEntropyLookup(psIVector vecIn, int ind,
                                             int flag)
{
  int    ii, ss, nActive, nInp = vecIn.length();
  double retdata=0;
  if (nInp != MatShapleyMap_.ncols())
  {
    printf("ShapleyEntropyLookup ERROR: nInputs mismatch.\n"); 
    return -9999;
  }

  //**/ put the subset S into vecIT using 0/1
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

  //**/ form the subset S+ by adding the current input
  vecIT[ind] = 1;

  //**/ if flag == 1 ==> look up Entropy(S+) - Entropy(S)
  if (flag == 1)
  {
    //**/ search for a match in the table for the subset S+
    //**/ and look for H(S+)
    retdata = -9999.0;
    for (ss = 0; ss < MapLength_; ss++)
    {
      //**/ if no match, skip
      for (ii = 0; ii < nInp; ii++)
        if (vecIT[ii] != MatShapleyMap_.getEntry(ss,ii)) break;
      if (ii == nInp)
      {
        retdata = VecShapleyTable_[ss];
        break;
      }
    }
    //**/ if not found, return a token
    if (retdata == -9999.0) return retdata;
    //**/ if S is empty, H(S)=0 so just return H(S+)
    if (nActive == 0) return retdata;
  }

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
      //**/ depending on the value of flag, return differently
      if (flag == 0) retdata = VecShapleyTable_[ss];
      else           retdata -= VecShapleyTable_[ss];
//if (flag == 0) printf("(0) Lookup returns %e\n",retdata);
//if (flag == 1) printf("(1) Lookup returns %e\n",retdata);
      return retdata;
    }
  }
  return -9999;
}

// ************************************************************************
// store data to the entropy table
// ------------------------------------------------------------------------
double ShapleyAnalyzer::ShapleyEntropySave(psIVector vecIn, int ind1, 
                           int ind2, double entropy)
{
  int ii, ind, iOne=1;
  if (MapLength_ < MaxMapLength_-1)
  {
    for (ii = ind1; ii < ind2; ii++)
    {
      ind = vecIn[ii];
      MatShapleyMap_.setEntry(MapLength_,ind,iOne);
      VecShapleyTable_[MapLength_] = entropy;
    }
//if (ind2 == ind1) printf("ERROR\n");
//if (ind2 > ind1)
//{
//printf("Save : ");
//for (ii = 0; ii < MatShapleyMap_.ncols(); ii++)
//{
//ind = MatShapleyMap_.getEntry(MapLength_,ii);
//if (ind == 1) printf("%d ",ii+1);
//}
//printf("\n");
//}
    MapLength_++;
  }
  return 0;
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
// set internal parameter 
// ------------------------------------------------------------------------
int ShapleyAnalyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if      (!strcmp(request, "ana_shapley_entropy")) costFunction_ = 2;
  else if (!strcmp(request, "ana_shapley_tsi"))     costFunction_ = 1;
  else if (!strcmp(request, "ana_shapley_vce"))     costFunction_ = 0;
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

