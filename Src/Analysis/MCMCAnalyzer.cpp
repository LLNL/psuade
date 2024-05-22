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
// Functions for the class MCMCAnalyzer
//**/ This class supports several MCMC methods:
//**/ 1. Metropolis-Hastings (with RS): bfmode = 0, scheme = 1
//**/ 2. Gibbs (with RS): bfmode = 0, scheme = 0
//**/ 3. Brute force (with RS): bfmode = 1
//**/    - can handle any type of input distributions (correlated..)
//**/    - analyzeDirect does the same but uses different input format
//**/       (so that it can be called by ODOE and externally with ease)
//**/    - analyzeDirect_nors performs brute force inference given an
//**/      already evaluated sample
//**/ 4. Metropolis-Hastings (with simulator): mode = 1
//**/ 5. Gibbs (with RS and omp): when PSUADE is compiled in OMP mode
// ------------------------------------------------------------------------
//**/ A little theory of MCMC:
//**/ * A Markov chain is a stochastic process that each transition in the 
//**/   process is dependent only on the current state. 
//**/ * Let pi_t be current state and T be transition matrix, then
//**/   pi_{t+1} = pi_t T, and pi_{t+n} = pi_t T^n.
//**/   (pi is a probability vector, can be infinite-dimensional)
//**/ * Homogeneous Markov Chain - the probability of moving from state A 
//**/   to state B is time-invariant (i.e. T is time-invariant)
//**/ * Limit distribution = lim_{n->infty} pi_i T^n
//**/ * The row sum of T must all be 1 and T_{i,j} >= 0 for all i, j.
// ------------------------------------------------------------------------
//**/ Issues:
//**/ * Q: Does a unique limit distribution exist given T and initial pi? 
//**/ * Q: Does the limit distribution the same as target distribution? 
//**/ * Q: How to construct T to speed up convergence?
// ------------------------------------------------------------------------
//**/ First, need a few definitions about Markov chain (MC)
//**/ * Definition: An irreducible MC has the property that each state is
//**/           reachable from any other state, in finite number of steps 
//**/           (Irreducible = transition graph is strongly connected)
//**/           Reducible: no guarantee to return to a state after leaving
//**/ * Implications:
//**/   - T^m(i,j) > 0 for all i and j and all m >= M for some finite M
//**/   - There is no permutation matrix P such that P T P^T is block
//**/     (upper or lower) triangular
//**/   - Any submatrix of T must have row sums < 1 for all rows
//**/   - by Gershgrin theorem, |lambda - T_{i,i}| < sum_{j!=i} |T_{i,j}|,
//**/     all submatrices of T must have eigenvalues < 1
//**/   - Perron-Frobenius theorem of linear algebra:
//**/      For every irreducible non-negative matrix, its largest
//**/      eigenvalue is positive (and in the case of transition matrices
//**/      lambda=1) and the components of the corresponding
//**/      eigenvector is also positive.
//**/   - Although irreducible T has a single eigenvalue lambda=1, it
//**/     may still have multiple lambda with magnitude=1. Aperiodicity
//**/     eliminates all other |lambda|=1. 
//**/ * Definition: The period d(S) of a state S of a homogeneous MC
//**/           with transition matrix T is given by 
//**/                      d(S) = gcd{n: Pr(S at t=n|S at t=0)>0}
//**/           if d(S)=d=1 for all states of a MC, the MC is aperiodic
//**/           (Aperiodic: chain does not get trapped in cycles)
//**/           Aperiodic: T^m(i,i) > 0 for all i and all m>=N
//**/    - weakly periodic MC: probability of returning to state i after
//**/      d steps is positive but less than 1 for some states.
//**/      (e.g. a circular chain with each node as groups of states)
//**/      Weakly periodic MC has some eigenvalues (maybe complex) with 
//**/      magnitude 1 and the rest less than 1.
//**/    - example of strongly periodic MC: permutation matrix. Strongly
//**/      periodic MC has all its eigenvalues either |lambda| = 1 
//**/      (i.e. all eigenvalues lie on a unit circle in complex space),
//**/ * Note: Periodic MC can have multiple complex eigenvalues with 
//**/         magnitudes=1.   
//**/ * Note: Surely pi = 1/N (N = vector length) is the right eigenvector
//**/         of T corresponding to eigenvalue=1, but we are computing 
//**/         left eigenvector.
//**/ * Definition: Stationary distribution pi is such that pi = pi T
//**/               and sum of pi = 1.
//**/ * Proposition: any aperiodic and irreducible MC has one stationary
//**/                distribution
//**/ ** Theorem: If a Markov chain is irreducible and aperiodic, then:
//**/   a. There exists a stationary distribution pi such that pi T = pi
//**/   b. The limit distribution is pi (convergence to unique distributon)
//**/   # This theorem guarantees that this process will converge to the 
//**/     desired distribution. But how can we make sure that the target 
//**/     distribution is indeed stationary with regards to this MC?
//**/ * Reversibility property: A Markov chain is said to be reversible 
//**/   if there is a probability distribution pi over its states such that 
//**/   pi(i) T_ij = p(j) T_ji for all i and j (detailed balanced equation)
//**/ * Theorem 2: A sufficient, but not necessary condition for pi to be a 
//**/   stationary distribution of a Markov chain with transition matrix 
//**/   T is the reversibility or detailed balance condition.
//**/ * Proof: Need to show pi = pi T, or pi_j = sum_{k=1}^N pi_k T_{k,j}
//**/          Since pi_j = pi_j \sum_{k=1}^N T_{j,k}  - row sum = 1
//**/           pi_j = \sum_{k=1}^N pi_j T_{j,k}
//**/          if pi_j T_{j,k} = pi_k T_{k,j} (reversibility), then
//**/           pi_j = \sum_{k=1}^N pi_k T_{k,j} (proved)
//**/ * Speed of convergence: depends on the 2nd eigenvalue
//**/ * MH: rejection implies aperiodicity, support of Q large enough
//**/       to cover the target implies irreducibility. Choice of
//**/       proposal distribution is critical (if it is too narrow, it
//**/       may not detect multimodal distributions; if it is too
//**/       large, rejection is high and thus high correlations).
//**/       Irreducibility is due to coverage. 
// ------------------------------------------------------------------------
//**/ Practical issues:
//**/ - standard deviation for the proposal distribution (50% acceptance)
//**/ - number of burn-in steps
//**/ - how to decide convergence (multiple chains)
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONG
// DATE   : 2007
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
//using namespace std;
#ifdef PSUADE_OMP
#include <omp.h>
#endif

#include "MCMCAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FunctionInterface.h"
#include "pData.h"
#include "mData.h"
#include "PDFManager.h"
#include "PDFBase.h"
#include "PDFNormal.h"
#include "PDFLogNormal.h"
#include "PDFTriangle.h"
#include "PDFBeta.h"
#include "PDFWeibull.h"
#include "PDFGamma.h"
#include "Psuade.h"
#include "Sampling.h"
#include "PrintingTS.h"
#include "TwoSampleAnalyzer.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
MCMCAnalyzer::MCMCAnalyzer(): Analyzer(), nInputs_(0), nOutputs_(0)
{
  setName("MCMC");
  mode_   = 0; // RS-based mode (mode = 1 : simulator-based mode)
  bfmode_ = 0; // brute force mode (default = 1, brute force)
               // settable via configure to brute-force or Gibbs
  scheme_ = 0; // 0 - Gibbs or brute force, 1 - MH (not available yet)
               // if bfmode_=0, this determines Gibbs/MH/OMP
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MCMCAnalyzer::~MCMCAnalyzer()
{
}

// ************************************************************************
// perform MCMC analysis (entry point to all different versions)
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze(aData &adata)
{
  //**/ ------------------------------------------------------------------
  // error checking
  //**/ ------------------------------------------------------------------
  if (adata.nInputs_ <= 0 || adata.nOutputs_ <= 0)
  {
    printOutTS(PL_ERROR,"MCMC ERROR: in nInputs/nOutputs.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", adata.nInputs_);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", adata.nOutputs_);
    return PSUADE_UNDEFINED;
  }

  //**/ ------------------------------------------------------------------
  //**/ remove psuade_stop if it exists
  //**/ ------------------------------------------------------------------
  char  pString[1001];
  strcpy(pString, "psuade_stop");
  FILE *fp = fopen(pString, "r");
  if (fp != NULL)
  {
    fclose(fp);
    unlink(pString);
  }

  //**/ ------------------------------------------------------------------
  //**/ check to see if user requests simulation and no RS
  //**/ This takes precedence over other MCMC schemes
  //**/ Note: this mode will call analyze_sim later
  //**/ ------------------------------------------------------------------
  char *cString = psConfig_.getParameter("MCMC_simulation_only");
  if (cString != NULL) mode_ = 1;
  if (mode_ == 1) return analyze_sim(adata);

  //**/ ------------------------------------------------------------------
  // more error checking (training sample size cannot be <=1 
  //**/ ------------------------------------------------------------------
  if (adata.nSamples_ <= 1)
  {
    printOutTS(PL_ERROR,"MCMC ERROR: RS training sample too small.\n");
    printOutTS(PL_ERROR,"    nSamples = %d\n", adata.nSamples_);
    return PSUADE_UNDEFINED;
  }

  //**/ ------------------------------------------------------------------
  //**/ check to make sure none of the RS training sample outputs is 
  //**/ UNDEFINED and all sample statuses are valid
  //**/ ------------------------------------------------------------------
  int ii, status = 0;
  for (ii = 0; ii < adata.nSamples_*adata.nOutputs_; ii++)
    if (adata.sampleOutputs_[ii] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"MCMC ERROR: Some outputs are undefined\n");
    printOutTS(PL_ERROR,"     (>= 1e35). Prune the undefined\n");
    printOutTS(PL_ERROR,"     sample points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ------------------------------------------------------------------
  //**/ check to make sure all sample points are valid (status = 1)
  //**/ ------------------------------------------------------------------
  if (adata.sampleStates_ != NULL)
  {
    status = 0;
    for (ii = 0; ii < adata.nSamples_; ii++) 
      status += adata.sampleStates_[ii];
    if (status != adata.nSamples_)
    {
      printOutTS(PL_WARN,
           "MCMC WARNING: Some outputs may not be ready.\n");
      printOutTS(PL_WARN,
           "     (Since their states are not 1's. Will\n");
      printOutTS(PL_WARN,
           "      continue anyway because outputs are not\n");
      printOutTS(PL_WARN,"      UNDEFINED.\n");
    }
  }

  //**/ ------------------------------------------------------------------
  //**/ for certain non-uniform distributions, use brute force method 
  //**/ ------------------------------------------------------------------
  if (adata.inputPDFs_ != NULL)
  {
    for (ii = 0; ii < adata.nInputs_; ii++) 
    {
      if (adata.inputPDFs_[ii] == PSUADE_PDF_SAMPLE)
      {
        printOutTS(PL_INFO,"MCMC INFO: User distribution detected.\n");
        printOutTS(PL_INFO,"           Switch to brute force MCMC.\n");
        return analyze_bf(adata);
      }
      else if (adata.inputPDFs_[ii] == 1000+PSUADE_PDF_NORMAL)
      {
        printOutTS(PL_INFO,"MCMC INFO: Parameter joint PDF detected.\n");
        printOutTS(PL_INFO,"           Switch to brute force MCMC.\n");
        return analyze_bf(adata);
      }
      else if (adata.inputPDFs_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
      {
        printOutTS(PL_INFO,"MCMC INFO: Parameter joint PDF detected.\n");
        printOutTS(PL_INFO,"           Switch to brute force MCMC.\n");
        return analyze_bf(adata);
      }
    }
  }

  //**/ ------------------------------------------------------------------
  //**/ read rs_index_file if there is one available
  //**/ (This option allows one to disable a certain input in the MCMC
  //**/ optimization ==> faPtrs by setting it to some fixed values).
  //**/ Also see if there are uncertain parameters that are not to be
  //**/ calibrated (when rsIndices[ii] = 999). If so, use analyze_bf2.
  //**/ Also, get RS type
  //**/ ------------------------------------------------------------------
  int faType = PSUADE_RS_MARS;
  PsuadeData *dataPtr = adata.ioPtr_;
  psVector  vecRSValues;
  psIVector vecRSIndices;
  psMatrix  dummySample;
  if (dataPtr != NULL)
  {
    //**/ read response surface index file, if available
    vecRSIndices.setLength(adata.nInputs_);
    for (ii = 0; ii < adata.nInputs_; ii++) vecRSIndices[ii] = ii;
    vecRSValues.setLength(adata.nInputs_);
    status = readIndexFile(dataPtr,vecRSIndices,vecRSValues,
                           dummySample);
    if (status == -1) 
    {
      printf("MCMC ERROR: response surface index file error\n");
      return PSUADE_UNDEFINED;
    }
    //**/ sample variables has indices 999 or higher
    //**/ meaning that at least one of the input is a sample
    for (ii = 0; ii < adata.nInputs_; ii++) 
    {
      if (vecRSIndices[ii] >= 999)
      {
        printf("MCMC INFO: >= 1 inputs have 'S' distribution ==> BF2\n");
        return(analyze_bf2(adata));
      }
    }
    pData pPtr;
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
  }

  //**/ ------------------------------------------------------------------
  //**/ now the special cases have been dealt with, the method choice can
  //**/ be one of OMP, Gibbs, BF, or MH. First consider OMP.
  //**/ ------------------------------------------------------------------
  int haveOMP=0;
  cString = psConfig_.getParameter("MCMC_OMP");
  if (cString != NULL) 
  {
#ifdef PSUADE_OMP
    bfmode_ = 0;
    haveOMP = 1;
    printf("MCMC INFO: Will use OMP version of MCMC.\n");
    if (adata.nOutputs_ == 1)
    {
      printf("MCMC INFO: PSUADE has detected that you are using OpenMP.\n");
      return analyze_omp(adata);
    }
    else
    {
      printf("MCMC INFO: OMP cannot handle nOutputs > 1.\n");
      printf("     INFO: OMP will not be used.\n");
      haveOMP = 0;
    }
#else
    bfmode_ = 0;
    haveOMP = 0;
    printf("MCMC INFO: OMP requested but PSUADE build is not OMP.\n");
    printf("     INFO: To build OMP version, define PSUADE_OMP and\n");
    printf("           use -fopenmp option in CXXFLAGS.\n");
    printf("     INFO: OMP will not be used.\n");
#endif
  }

  //**/ ------------------------------------------------------------------
  //**/ The only options left are Gibbs, BF, or MH
  //**/ Look for options (BF, Gibbs, ..) from the configure object
  //**/ If BF is selected, call analyze_bf
  //**/ ------------------------------------------------------------------
  int m1 = 0, m2 = 0, m3 = 0;
  cString = psConfig_.getParameter("MCMC_gibbs");
  if (cString != NULL) m1 = 1;
  cString = psConfig_.getParameter("MCMC_brute_force");
  if (cString != NULL) m2 = 1;
  cString = psConfig_.getParameter("MCMC_mh");
  if (cString != NULL) m3 = 1;
  if (m1 == 1)
  {
    printf("MCMC INFO: Will use Gibbs.\n");
    bfmode_ = 0;
    scheme_ = 0;
  }
  else if (m2 == 1)
  {
    printf("MCMC INFO: Will use brute-force MCMC.\n");
    bfmode_ = 1;
    scheme_ = 0;
  }
  else if (m3 == 1)
  {
    printf("MCMC INFO: Will use Metropolis-Hastings.\n");
    bfmode_ = 0;
    scheme_ = 1;
  }

  //**/ ------------------------------------------------------------------
  //**/ if brute force mode is set, use brute force
  //**/ if Metropolis Hasting is requested, use it 
  //**/ ------------------------------------------------------------------
  if (bfmode_ == 1) return analyze_bf(adata);
  if (scheme_ == 1) return analyze_mh(adata);

  //**/ ------------------------------------------------------------------
  //**/ the following have been taken care of above:
  //**/ analyze_sim (simulation only)
  //**/ analyze_bf (brute force)
  //**/ analyze_bf2 (with uncalibrated uncertain parameters)
  //**/ analyze_mh
  //**/ analyze_omp
  //**/ What is left is: Gibbs sequential
  //**/ ------------------------------------------------------------------
  return analyze_gibbs(adata);
}

// ************************************************************************
// perform Gibbs-MCMC analysis 
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_gibbs(aData &adata)
{
  int   ii, jj, kk, iOne=1, iZero=0;
  char  lineIn[1001], pString[1001];
  FILE  *fp=NULL;
  pData pPtr, qData, pOutputs;

  //**/ ------------------------------------------------------------------
  //**/ display banner
  //**/ ------------------------------------------------------------------
  int printLevel  = adata.printLevel_;
  if (psConfig_.InteractiveIsOn()) displayBanner(printLevel);

  //**/ ------------------------------------------------------------------
  //**/ extract data from the aData object (passed in from outside)
  //**/ ------------------------------------------------------------------
  int    nInputs   = adata.nInputs_;
  int    nOutputs  = adata.nOutputs_;
  int    nSamples  = adata.nSamples_;
  double *sampIns  = adata.sampleInputs_;
  double *samOuts  = adata.sampleOutputs_;
  int    *samStats = adata.sampleStates_;
  double *xLower   = adata.iLowerB_;
  double *xUpper   = adata.iUpperB_;
  int    *pdfTypes = adata.inputPDFs_;
  double *pdfMeans = adata.inputMeans_;
  double *pdfStdvs = adata.inputStdevs_;
  nInputs_  = nInputs;
  nOutputs_ = nOutputs;
  PsuadeData *dataPtr = adata.ioPtr_;
  if (dataPtr != NULL) dataPtr->getParameter("input_names", qData);

  //**/ ------------------------------------------------------------------
  //**/ ask user to see if they would like more diagnostics (this will be
  //**/ asked only when the grandmaster mode is on.
  //**/ ------------------------------------------------------------------
  int masterOption=0;
  if (psConfig_.GMModeIsOn())
  {
    printf("Please choose from the following option (or 0 if none): \n");
    printf("  1. track proposal distribution at each MCMC iteration\n");
    printf("  2. track posterior distributions after each MCMC cycle\n");
    scanf("%s", pString);
    fgets(lineIn,1000,stdin);
    if (pString[0] == '1') masterOption = 1;
    if (pString[0] == '2') masterOption = 2;
  }

  //**/ ------------------------------------------------------------------
  //**/ clean up important psuade signal files
  //**/ ------------------------------------------------------------------
  cleanUp();

  //**/ ------------------------------------------------------------------
  //**/ read rs_index_file if there is one available
  //**/ (This option allows one to disable a certain input in the MCMC
  //**/ optimization ==> faPtrs by setting it to some fixed values).
  //**/ ------------------------------------------------------------------
  int faType = PSUADE_RS_MARS, status;
  psVector  vecRSValues;
  psIVector vecRSIndices;
  psMatrix  dummySample;
  if (dataPtr != NULL)
  {
    //**/ read response surface index file, if available
    vecRSIndices.setLength(adata.nInputs_);
    for (ii = 0; ii < adata.nInputs_; ii++) vecRSIndices[ii] = ii;
    vecRSValues.setLength(adata.nInputs_);
    status = readIndexFile(dataPtr,vecRSIndices,vecRSValues,
                           dummySample);
    if (status == -1) 
    {
      printf("MCMC_Gibbs ERROR: Response surface index file error\n");
      printf("                  Check your RS index file.\n");
      return PSUADE_UNDEFINED;
    }
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
  }

  //**/ ------------------------------------------------------------------
  // get experimental data information from the spec file
  // (vecDesignP, matExpInps, matExpMeans, matExpStdvs, dnReps)
  //**/ ------------------------------------------------------------------
  int    dnSamples=0, dnInputs=0, dnReps=1;
  psIVector vecDesignP; 
  psMatrix  matExpInps, matExpMeans, matExpStdvs; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  if (dstatus != 0.0) return PSUADE_UNDEFINED;
  dnSamples = matExpMeans.nrows();
  dnInputs  = matExpInps.ncols();
  //**/ check to make sure there is no mismatch
  for (ii = 0; ii < nInputs; ii++)
  {
    if ((vecRSIndices.length() > 0 && vecRSIndices[ii] == -1) && 
        (vecDesignP.length() > 0 && vecDesignP[ii] == 1))
    {
      printOutTS(PL_ERROR,
        "MCMC_Gibbs ERROR: Inactive input %d cannot be design input\n",
        ii+1);
      for (jj = 0; jj < nInputs; jj++)
        printf("  Parameter type %d = %d (-1 = inactive), RS index = %d\n",
               jj+1, vecDesignP[jj], vecRSIndices[jj]);
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ------------------------------------------------------------------
  //**/ option to add response surface uncertainties to data std dev
  //**/ mode = 1 still needs response surface, but rs error not allowed
  //**/ (recall: mode = 1 - direct simulation mode) 
  //**/ ==> rsErrFlag
  //**/ ------------------------------------------------------------------
  int rsErrFlag=0;
  if (psConfig_.AnaExpertModeIsOn() && mode_ == 0)
  {
    printOutTS(PL_INFO,
       "*** OPTION TO INCLUDE RESPONSE SURFACE UNCERTAINTIES:\n");
    printOutTS(PL_INFO,
       "\nTo incorporate the response surface errors into the ");
    printOutTS(PL_INFO, "likelihood\n");
    printOutTS(PL_INFO,
      "function, make sure to select Gaussian process/Kriging ");
    printOutTS(PL_INFO, "polynomial\n");
    printOutTS(PL_INFO,
       "regression, or bootstrapped MARS as response surface in ");
    printOutTS(PL_INFO, "the simulation\n");
    printOutTS(PL_INFO,
       "data file. Otherwise, no RS uncertainties will be included.\n");
    printOutTS(PL_INFO,
        "NOTE: if you don't know what this is, just say no.\n");
    printf( "===> Include response surface uncertainties? (y or n) ");
    scanf("%s", pString);
    fgets(lineIn,1000,stdin);
    if (pString[0] == 'y') rsErrFlag = 1;
    printEquals(PL_INFO, 0);
  }

  //**/ ------------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ ------------------------------------------------------------------
  printOutTS(PL_INFO,
       "MCMC_Gibbs INFO: CREATING RESPONSE SURFACES FOR ALL OUTPUTS.\n");
  FuncApprox **faPtrs = new FuncApprox*[nOutputs];
  psVector vecSamOut1;
  vecSamOut1.setLength(nSamples);
  for (ii = 0; ii < nOutputs; ii++)
  {
    faType = -1;
    printOutTS(PL_INFO,
      "MCMC_Gibbs INFO: CREATING RESPONSE SURFACE FOR OUTPUT %d.\n",ii+1);
    faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
    faPtrs[ii]->setNPtsPerDim(16);
    faPtrs[ii]->setBounds(xLower, xUpper);
    faPtrs[ii]->setOutputLevel(0);
    for (kk = 0; kk < nSamples; kk++) 
      vecSamOut1[kk] = samOuts[kk*nOutputs+ii];

    status = faPtrs[ii]->initialize(sampIns, vecSamOut1.getDVector());
    if (status != 0)
    {
      printOutTS(PL_ERROR,
           "MCMC_Gibbs ERROR: Unable to create response surface.\n");
      printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
      for (kk = 0; kk < ii; kk++) delete faPtrs[kk];
      delete [] faPtrs;
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ------------------------------------------------------------------
  //**/ special set up (sample size) using interactive mode
  //    ==> burnInSamples, maxSamples, nbins
  //        (let's keep burnInSamples at 1/2 of maxSamples)
  //**/ ------------------------------------------------------------------
  int maxGlobalIts=20;
  int maxSamples = 10000;
  int burnInSamples = 2500;
  int nbins = 20;
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"*** CURRENT SETTINGS OF MCMC_Gibbs PARAMETERS: \n\n");
  printOutTS(PL_INFO,"MCMC Burn-in sample size       (default) = %d\n", 
             burnInSamples);
  printOutTS(PL_INFO,"MCMC maximum number of samples (default) = %d\n", 
             maxSamples*maxGlobalIts);
  printOutTS(PL_INFO,"MCMC no. of bins in histogram  (default) = %d\n",
             nbins);
  printOutTS(PL_INFO,
     "NOTE: Sample increment - check convergence every sample increment\n");
  //printOutTS(PL_INFO,
  //   "NOTE: Histogram nBins  - granularity of histogram bar graph\n");
  printOutTS(PL_INFO, 
     "Turn on ana_expert mode to change some of these default settings.\n\n");
  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    snprintf(pString,1000,"Enter maximum sample size (100000-1000000): ");
    kk = getInt(100000, 10000000, pString);
    maxGlobalIts = kk / maxSamples;
    //snprintf(pString,1000,"Enter the number of histogram bins (5 - 25) : ");
    //nbins = getInt(5, 50, pString);
  }

  //**/ ------------------------------------------------------------------
  //**/ special set up using interactive mode ==> vecPlotInds, nPlots
  //**/ ------------------------------------------------------------------
  int nPlots = 0;
  psIVector vecPlotIndices;
  if (psConfig_.MasterModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*** OPTION TO CREATE POSTERIOR PLOTS FOR SELECTED INPUTS:\n\n");
    printOutTS(PL_INFO,
       "MCMC creates Matlab/Scilab files for posterior distributions.\n");
    printOutTS(PL_INFO,
       "You can choose to generate posterior plots for all inputs, or\n");
    printOutTS(PL_INFO,
       "just a selected few (in case there are too many inputs).\n");
    printf("Select inputs for which posterior plots are to be created:\n");
    snprintf(pString,1000,
            "Enter input number (-1 for all, 0 to terminate) : ");
    kk = 1;
    vecPlotIndices.setLength(nInputs);
    nPlots = 0;
    while (kk != 0 || nPlots < 1)
    {
      kk = getInt(-1, nInputs, pString);
      if (kk == -1)
      {
        nPlots = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0)
            if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
              vecPlotIndices[nPlots++] = ii;
        }
        break;
      }
      if (kk != 0 && nPlots < nInputs)
      {
        if ((vecRSIndices.length() > 0) && (vecRSIndices[kk-1] < 0)) 
          printOutTS(PL_ERROR,
            "Input %d has been fixed by the RS index file (no plot).\n",
            kk+1);
        else if (vecDesignP.length() > 0 && vecDesignP[kk-1] == 1)
          printOutTS(PL_ERROR,
               "Input %d is a design parameter (no plot)\n",kk);
        else 
          vecPlotIndices[nPlots++] = kk - 1;
      }
      if (kk == 0 && nPlots == 0)
        printOutTS(PL_ERROR,
          "Please select at least 1 input for plotting posteriors.\n");
    }
    if (nPlots > 1) sortIntList(nPlots, vecPlotIndices.getIVector());
  }
  else
  {
    vecPlotIndices.setLength(nInputs);
    nPlots = 0;
    for (ii = 0; ii < nInputs; ii++) 
      if (vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0)
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
          vecPlotIndices[nPlots++] = ii;
  }
  printOutTS(PL_INFO, 
       "MCMC Plot summary: Input number to be plotted are (%d):\n",
       nPlots);
  for (ii = 0; ii < nPlots; ii++)
    printOutTS(PL_INFO, "   Input %4d\n", vecPlotIndices[ii]+1);

  //**/ ------------------------------------------------------------------
  // option to add discrepancy function and a posterior sample
  // ==> modelFormFlag, vecModelForms, modelFormConst, genPosteriors
  //**/ ------------------------------------------------------------------
  int modelFormFlag=0, modelFormConst=0, genPosteriors=1;
  psIVector vecModelForms;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** OPTION TO ADD A DISCREPANCY FUNCTION:\n\n");
    printOutTS(PL_INFO,
       "To use this feature, first make sure that the observation ");
    printOutTS(PL_INFO, "data file\n");
    printOutTS(PL_INFO,
       "specified earlier has design parameters specified since ");
    printOutTS(PL_INFO,
       "the discrepancy\n");
    printOutTS(PL_INFO,
       "function is to be a function of these design parameters.\n");
    printOutTS(PL_INFO,
       "(If not, a constant discrepancy function is to be created.)\n");
    printOutTS(PL_INFO,
         "NOTE: if you don't know what this is, just say NO.\n");
    snprintf(pString,1000,"===> Add discrepancy function ? (y or n) ");
    getString(pString, lineIn);
    if (lineIn[0] == 'y') modelFormFlag = 1;
    if (modelFormFlag == 1)
    {
      snprintf(pString,1000,
        "===> Add discrepancy to (1) all or (2) selected outputs ? ");
      kk = getInt(1,2,pString);
      if (kk == 1)
      {
        vecModelForms.setLength(nOutputs);
        vecModelForms.setConstant(iOne);
      }
      else
      {
        vecModelForms.setLength(nOutputs);
        for (ii = 0; ii < nOutputs; ii++)
        {
          snprintf(pString,1000,"Add discrepancy to output %d ? (y or n) ",
                  ii+1);
          getString(pString, lineIn);
          if (lineIn[0] == 'y') vecModelForms[ii] = 1;
        }
      }
    }
    if (modelFormFlag == 1 && dnInputs == 0)
    {
      modelFormConst = 1;
      printOutTS(PL_INFO,
        "NOTE: No design inputs ==> discrepancy = constant function\n");
    }
    else if (modelFormFlag == 1 && dnSamples == 1)
    {
      modelFormConst = 1;
      printOutTS(PL_INFO,
        "NOTE: 1 experiment ==> discrepancy = a constant function\n");
    }
  } 
  printEquals(PL_INFO, 0);

  //**/ ------------------------------------------------------------------
  //**/ setup input PDF, if there is any (multivariate normal or sample 
  //**/ types have been dealt with before)
  //**/ ------------------------------------------------------------------
  if (printLevel > 2) 
    printOutTS(PL_INFO,
        "*** INFORMATION ON PARAMETER PRIOR DISTRIBUTIONS\n");
  PDFBase **inputPDFs = new PDFBase*[nInputs];
  for (ii = 0; ii < nInputs; ii++)
  {
    inputPDFs[ii] = NULL;
    if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0) &&
        (vecDesignP.length() == 0 || vecDesignP[ii] == 0)) 
    {
      if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_NORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFNormal(pdfMeans[ii], 
                                                  pdfStdvs[ii]);
        if (printLevel > 2) 
           printOutTS(PL_INFO,
                "Parameter %3d has normal prior distribution (%e,%e)\n",
                ii+1, pdfMeans[ii], pdfStdvs[ii]);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_LOGNORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFLogNormal(pdfMeans[ii],
                                                     pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has lognormal prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_TRIANGLE)
      {
        inputPDFs[ii] = (PDFBase *) new PDFTriangle(pdfMeans[ii],
                                                    pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has triangle prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_BETA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFBeta(pdfMeans[ii],
                                                pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has beta prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_WEIBULL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFWeibull(pdfMeans[ii],
                                                   pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has Weibull prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_GAMMA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFGamma(pdfMeans[ii],
                                                 pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
             "Parameter %3d has gamma prior distribution.\n",ii+1);
      }
      else if (pdfTypes == NULL || pdfTypes[ii] == PSUADE_PDF_UNIFORM)
      {
        inputPDFs[ii] = NULL;
        if (printLevel > 2)
          printOutTS(PL_INFO,
             "Parameter %3d has uniform prior distribution.\n",ii+1);
      }
    }
  }
  if (printLevel > 2) printEquals(PL_INFO, 0);

  //**/ ------------------------------------------------------------------
  //**/ create function interface for direct simulation, if needed
  //    ==> funcIO, freq (how often to do simulation and emulation)
  //**/ ------------------------------------------------------------------
  int maxPts = 100;
  if (psConfig_.MasterModeIsOn())
  {
    printOutTS(PL_INFO,"*** SETTING PROPOSAL DISTRIBUTION RESOLUTION\n");
    printOutTS(PL_INFO,
       "Since MCMC uses many function evaluations to construct ");
    printOutTS(PL_INFO, "the proposal\n");
    printOutTS(PL_INFO,
       "distributions, you have the option to set how many points ");
    printOutTS(PL_INFO, "are used to\n");
    printOutTS(PL_INFO,
         "construct it in order to keep the inference cost reasonable.\n");
    printOutTS(PL_INFO,
      "Sample size to construct proposal distribution. Default = %d\n",
      maxPts);
    printOutTS(PL_INFO,"NOTE: Larger sample size gives more ");
    printOutTS(PL_INFO,"accurate posteriors, but it is\n");
    printOutTS(PL_INFO,"      more expensive computationally.\n");
    snprintf(pString,1000,"Enter new sample size (20 - 1000): ");
    maxPts = getInt(20, 1000, pString);
  }

  //**/ ------------------------------------------------------------------
  //**/ create discrepancy sample, if desired ==> discSamOuts, discFunc
  //**/ ------------------------------------------------------------------
  psVector vecDiscSamOuts, vecDiscConstMeans, vecDiscConstStds;
  double *discSamOuts=NULL, *discFuncConstantMeans=NULL;
  double *discFuncConstantStds = NULL;
  if (modelFormFlag == 1)
  {
    dstatus = createDiscrepancyFunctions(nInputs,nOutputs,xLower,xUpper, 
                  vecRSIndices,vecRSValues,vecDesignP,dnInputs,dnSamples, 
                  matExpInps,matExpMeans,dataPtr,vecDiscConstMeans,
                  vecDiscConstStds,vecDiscSamOuts,faPtrs,printLevel,
                  modelFormConst);
    if (dstatus < 0) return PSUADE_UNDEFINED;
    discSamOuts = vecDiscSamOuts.getDVector();
    discFuncConstantMeans = vecDiscConstMeans.getDVector();
    discFuncConstantStds = vecDiscConstStds.getDVector();
  }

  //**/ ------------------------------------------------------------------
  // Gibbs sequential: get information on how many chains to use and 
  // threshold for convergence check
  //**/ ------------------------------------------------------------------
  int    numChains=3;
  double psrfThreshold=1.05;
  if (psConfig_.AnaExpertModeIsOn())
  {
    snprintf(pString,1000,"How many MCMC chains? (2-20, default=3) : ");
    numChains = getInt(2,20,pString);
    snprintf(pString,1000,"PSRF threshold? (1.0 - 1.2, default = 1.05) : ");
    psrfThreshold = getDouble(pString);
    if (psrfThreshold < 1.0 || psrfThreshold > 1.2)
    {
      printOutTS(PL_INFO,
           "MCMC : invalid PSRF threshold ==> reset to 1.05.\n");
      psrfThreshold = 1.05;
    }
  }

  //**/ ------------------------------------------------------------------
  //    set up for MCMC iterations
  // (Note: vecXDist in long double because probability can be very small)
  //**/ ------------------------------------------------------------------
  int    *Ivec, **bins, ****bins2, globalIts, countTrack, cnt, dcnt;
  int    mcmcFail=0, sumBins, index2, count;
  int    ii2, ii3, jj2, kk2, index, length, iChain, chainCnt, mcmcIts;
  int    chainCntSave, nChainGood=0, masterCount=1;
  double *XGuess=NULL, *YDiscS;
  double *YDiscStds=NULL, *XGuessS=NULL, *YGuessS=NULL, *YGuessStds=NULL;
  double Xtemp, Ytemp, Ytemp2, Ymax, dtemp;
  double ***XChains=NULL, stdev, stdev2, ddata, ddata2, WStat, BStat;
  TwoSampleAnalyzer *s2Analyzer=NULL;
  psLDVector vecXDist;
  psVector vecXmax, vecRange, vecSDist, vecS2;
  psVector vecXGuess, vecXGuessS, vecYGuessS, vecYGuessStds;
  psVector vecYDiscS, vecYDiscStds;
  //**/ ----- for storing the input ranges
  vecRange.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    vecRange[ii] = xUpper[ii] - xLower[ii]; 
  //**/ ----- vecXDist is for storing the proposal sample
  //**/ it requires double double precision
  vecXDist.setLength(maxPts+1);
  vecSDist.setLength(maxPts+1);
  //**/ ----- for model evaluation
  vecXGuess.setLength(nInputs);
  XGuess = vecXGuess.getDVector();
  vecXGuessS.setLength(dnSamples*nInputs*(maxPts+1));
  XGuessS = vecXGuessS.getDVector();
  vecYGuessS.setLength(dnSamples*nOutputs*(maxPts+1));
  YGuessS = vecYGuessS.getDVector();
  vecYGuessStds.setLength(dnSamples*nOutputs*(maxPts+1));
  YGuessStds = vecYGuessStds.getDVector();
  //**/ ----- for discrepancy evaluation
  vecYDiscS.setLength(dnSamples*nOutputs*(maxPts+1));
  YDiscS = vecYDiscS.getDVector();
  vecYDiscStds.setLength(dnSamples*nOutputs*(maxPts+1));
  YDiscStds = vecYDiscStds.getDVector();
  //**/ ----- for keeping track of posterior statistics
  vecMeans_.setLength(nInputs_);
  vecSigmas_.setLength(nInputs_);
  //**/ ----- for keeping the point of maximum likelihood
  vecXmax.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecXmax[ii] = 0;
  vecMostLikelyInputs_.setLength(nInputs_);
  vecMostLikelyOutputs_.setLength(nOutputs_);
  Ymax = PSUADE_UNDEFINED;
  //**/ ----- for randomization of order of inputs to be processed
  psIVector vecIT;
  vecIT.setLength(nInputs);
  vecIT[nInputs-1] = -1;
  //**/ ----- for saving the input values at each iteration
  XChains = new double**[numChains];
  for (ii = 0; ii < numChains; ii++)
  {
    XChains[ii] = new double*[maxGlobalIts*maxSamples];
    for (jj = 0; jj < maxGlobalIts*maxSamples; jj++)
      XChains[ii][jj] = new double[nInputs+1];
  }
  psVector  vecChainMeans, vecChainStds, vecPsrfs;
  psIVector vecChainStas;
  vecChainMeans.setLength(numChains);
  vecChainStds.setLength(numChains);
  vecChainStas.setLength(numChains);
  vecPsrfs.setLength(nInputs);

  //**/ ------------------------------------------------------------------
  //**/ ----- initialization for binning 
  //**/ ------------------------------------------------------------------
  bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  //**/ ----- initialization for convergence checking 
  vecS2.setLength(maxGlobalIts*maxSamples);
  double *s2Vec = vecS2.getDVector();
  if (printLevel > 3) s2Analyzer = new TwoSampleAnalyzer();

  //**/ ------------------------------------------------------------------
  //**/ generate LHS or LPTAU MCMC seed points for different chains
  //**/ ------------------------------------------------------------------
  Sampling *sampler;
  if (nInputs > 50) sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  else              sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
  sampler->setInputBounds(nInputs, xLower, xUpper);
  sampler->setOutputParams(1);
  sampler->setSamplingParams(numChains, 1, 1);
  sampler->initialize(0);
  psVector vecRanSeeds;
  vecRanSeeds.setLength(numChains*nInputs);
  psVector  vecOT;
  psIVector vecST;
  vecOT.setLength(numChains);
  vecST.setLength(numChains);
  sampler->getSamples(numChains,nInputs,1,vecRanSeeds.getDVector(),
                      vecOT.getDVector(),vecST.getIVector());
  delete sampler;
  int igFlag=0;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printf("Set your own initial point for Chain 1 ? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y')
    {
      printf("Enter initial parameter values below ");
      printf("(only for the first chain). If\n");
      printf("you don't know what values to set, ");
      printf("enter the suggested midpoints.\n");
      igFlag = 1;
    }
    fgets(pString,1000,stdin);
  }

  //**/ normalize the sample inputs
  for (iChain = 0; iChain < numChains; iChain++)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      ddata = vecRanSeeds[iChain*nInputs+ii];
      if (igFlag == 1 && iChain == 0)
      {
        snprintf(pString,1000,
             "Enter initial value for input %d (midpoint=%e) : ",
             ii+1,0.5*(xLower[ii]+xUpper[ii]));
        ddata = getDouble(pString);
      }
      ddata = (ddata - xLower[ii]) / vecRange[ii];
      vecRanSeeds[iChain*nInputs+ii] = ddata;
    }
  }

  //**/ ------------------------------------------------------------------
  // run the Gibbs algorithm
  //**/ ------------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "MCMC_Gibbs begins ... \n");
  fflush(stdout);
  globalIts = chainCnt = 0;

  //**/ ------------------------------------------------------------------
  //**/ iterate until termination
  //**/ ------------------------------------------------------------------
  while (globalIts < maxGlobalIts)
  {
    //**/ ----------------------------------------------------------------
    //**/ run all chains once
    //**/ ----------------------------------------------------------------
    for (iChain = 0; iChain < numChains; iChain++)
    {
      //**/ initially, set LPTAU/LHS initial guess for the calibration
      //**/ parameters. Otherwise, retrieve the last input values
      printOutTS(PL_INFO,"MCMC_Gibbs : Chain %d, iteration = %d\n",
                 iChain+1,globalIts+1);
      if (iChain == 0) chainCntSave = chainCnt;
      else             chainCnt     = chainCntSave;
      //**/ insert seed points initially
      if (chainCnt == 0)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          //**/ if design points, default = mid point
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
               XGuess[ii] = vecRanSeeds[iChain*nInputs+ii];
          else XGuess[ii] = 0.5;
          XGuess[ii] = XGuess[ii]*(xUpper[ii]-xLower[ii])+xLower[ii];
          //**/ Store the guess in the chain
          XChains[iChain][0][ii] = XGuess[ii];
        }
        //**/ if fixed input, insert the fixed values
        if (vecRSIndices.length() > 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            if (vecRSIndices[ii] < 0) XGuess[ii] = vecRSValues[ii];
        }
        //**/ this is the normalization factor
        XChains[iChain][0][nInputs] = -1;
      }
      //**/ otherwise, fetch from the chain
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          XGuess[ii] = XChains[iChain][chainCnt-1][ii];
      }
      if (printLevel >= 0)
      {
        printOutTS(PL_INFO,"       Chain %d current initial guess : \n",
                   iChain+1);
        for (ii = 0; ii < nInputs; ii++)
          if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >=0) && 
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
            printOutTS(PL_INFO,"          Input %4d = %e\n",
                       ii+1,XGuess[ii]);
      }
       
      //**/ --------------------------------------------------------------
      //**/ run maxSamples for the chain
      //**/ --------------------------------------------------------------
      mcmcIts = countTrack = 0;
      while (mcmcIts < maxSamples)
      {
        //**/ output dots to indicate progress 
        count = (mcmcIts+1) / (maxSamples/10);
        if (count != countTrack)
        { 
          countTrack++;
          printOutTS(PL_INFO, "%3.0f%% ",10.0*countTrack );
          fflush(stdout);
        }
        //**/ cycle through all selected inputs in random order
        //**/ by generate an integer vector in vecIT, then
        //**/ prevent sampling same input in consecutive steps
        //**/ by switching with the last in vecIT
        jj = vecIT[nInputs-1];
        generateRandomIvector(nInputs, vecIT.getIVector());
        if (vecIT[0] == jj && nInputs > 1)
        {
          vecIT[0] = vecIT[nInputs-1];
          vecIT[nInputs-1] = jj;
        }
        for (kk = 0; kk < nInputs; kk++)
        {
          ii = vecIT[kk];
          //**/ only selected inputs will be walking
          //**/ Selected: not fixed inputs, not design inputs
          if ((vecRSIndices.length() == 0 ||
              (vecRSIndices.length() > 0 && vecRSIndices[ii] >=0)) && 
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
          {
            //**/ store away current value of parameter ii
            Xtemp = XGuess[ii];
            //**/ for diagnostics purposes
            fp = NULL;
            if (masterOption == 1 && masterCount == 1)
            {
              fp = fopen("MCMCDistTrack.m", "w");
              if (fp == NULL)
              {
                printOutTS(PL_WARN,
                  "MCMC_Gibbs ERROR: Cannot write to MCMCDistTrack.m\n");
              }
              else
              {
                fprintf(fp,
                  "%% This file contains individual terms in the\n");
                fprintf(fp,
                  "%% exponent of the likelihood function, i.e.\n");
                fprintf(fp,"%% in each row of first column : \n");
                fprintf(fp,"%% S = 1/(p*n) sum_{k=1}^p sum_{i=1)^n ");
                fprintf(fp,"(Y_ki - m_ki)^2/sd_ki^2\n");
                fprintf(fp,"%% next columns: log-likelihood terms\n");
                fprintf(fp,"nOuts = %d;\n", nOutputs);
                fprintf(fp,"nObs  = %d;\n", dnSamples);
                fprintf(fp,"nPts  = %d;\n", maxPts+1);
                fprintf(fp,"A = [\n");
              }
            }
            //**/ creating a CDF for the current input given the others
            //**/ (fix all inputs, scan input ii to build distribution)
            //**/ CDF will be put into vecXDist[0:maxPts]
            //**/ Case 1 : 1 output
            if (nOutputs == 1)
            {
              for (jj = 0; jj <= maxPts; jj++)
              {
                //**/ marching ii-th input upward from lower to upper
                //**/ modify XGuess[ii] and leave others alone
                XGuess[ii] = xLower[ii]+jj*vecRange[ii]/maxPts;
                   
                //**/ create XGuessS (which is to be evaluated with
                //**/        with RS for all experimental points
                index = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  dcnt = 0;
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                  {
                    XGuessS[(index+kk2)*nInputs+ii2] = XGuess[ii2];
                    if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
                    {
                      XGuessS[(index+kk2)*nInputs+ii2] = 
                                      matExpInps.getEntry(kk2,dcnt);
                      dcnt++;
                    }
                  }
                }
                //**/ set design outputs and stds (discrepancy) to zero 
                //**/ since they will be used later but not be set
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  YDiscS[index+kk2] = YDiscStds[index+kk2] = 0.0;
                  YGuessS[index+kk2] = YGuessStds[index+kk2] = 0.0;
                }
              }

              //**/ run XGuessS through response surfaces
              //**/ case 1 : if response surface error information 
              //**/          should be used
              if (rsErrFlag == 1)
              {
                faPtrs[0]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                                          XGuessS,YGuessS,YGuessStds);
                //**/ prepare discrepancy function, if any
                if (modelFormFlag == 1)
                {
                  //**/ if model form is not a constant function
                  if (modelFormConst == 0)
                  {
                    for (ii3 = 0; ii3 <= maxPts; ii3++)
                    {
                      for (kk2 = 0; kk2 < dnSamples; kk2++)
                      {
                        index = ii3 * dnSamples + kk2;
                        YDiscS[index] = discSamOuts[kk2];
                      } 
                    }
                  }
                  //**/ if model form is a constant function
                  else if (discFuncConstantMeans != NULL &&
                           discFuncConstantMeans[0] != PSUADE_UNDEFINED)
                  {
                    for (kk2 = 0; kk2 < (maxPts+1)*dnSamples; kk2++) 
                      YDiscS[kk2] = discFuncConstantMeans[0];
                  }
                }
              }
              //**/ case 2 : if response surface error information not 
              //**/          to be used
              else
              {
                faPtrs[0]->evaluatePoint((maxPts+1)*dnSamples,
                                         XGuessS,YGuessS);
                //**/ prepare discrepancy function, if any
                if (modelFormFlag == 1)
                {
                  //**/ if model form is not a constant function
                  if (modelFormConst == 0)
                  {
                    for (ii3 = 0; ii3 <= maxPts; ii3++)
                    {
                      for (kk2 = 0; kk2 < dnSamples; kk2++)
                      {
                        index = ii3 * dnSamples + kk2;
                        YDiscS[index] = discSamOuts[kk2];
                      } 
                    }
                  }
                  //**/ if model form is a constant function
                  else if (discFuncConstantMeans != NULL &&
                           discFuncConstantMeans[0] != PSUADE_UNDEFINED)
                  {
                    for (kk2 = 0; kk2 < (maxPts+1)*dnSamples; kk2++) 
                      YDiscS[kk2] = discFuncConstantMeans[0];
                  }
                }
              }

              //**/ compute distributions vecXDist
              for (jj = 0; jj <= maxPts; jj++)
              {
                index = jj * dnSamples;
                //**/ add discrepancy if it is requested
                if (vecModelForms.length() > 0 && vecModelForms[0] == 1)
                  for (kk2 = 0; kk2 < dnSamples; kk2++) 
                    YGuessS[index+kk2] += YDiscS[index+kk2];

                //**/ compute negative log likelihood
                vecXDist[jj] = 0.0;
                for (kk2 = 0; kk2 < dnSamples; kk2++)
                {
                  Ytemp = YGuessS[index+kk2];
                  stdev = YGuessStds[index+kk2];
                  stdev2 = YDiscStds[index+kk2];
                  Ytemp2 = pow((Ytemp-matExpMeans.getEntry(kk2,0)),2.0) /
                          (pow(matExpStdvs.getEntry(kk2,0),2.0)+
                           stdev*stdev+stdev2*stdev2);
                  vecXDist[jj] += Ytemp2;
                }
                vecXDist[jj] = vecXDist[jj] / (double) dnReps;
                if (masterOption == 1 && fp != NULL) 
                {
                  fprintf(fp, "%e ", (double) vecXDist[jj]);
                  for (kk2 = 0; kk2 < dnSamples; kk2++)
                  {
                    Ytemp = YGuessS[index+kk2];
                    stdev = YGuessStds[index+kk2];
                    stdev2 = YDiscStds[index+kk2];
                    Ytemp2 = pow(Ytemp-matExpMeans.getEntry(kk2,0),2.0) /
                            (pow(matExpStdvs.getEntry(kk2,0),2.0)+
                             stdev*stdev+stdev2*stdev2);
                    fprintf(fp, "%e ", Ytemp2);
                  }
                  fprintf(fp, "\n");
                }
              }
            }
            //**/ Case 2 : multiple outputs
            else
            {
              //**/ prepare XGuessS and initialize YGuessS 
              //**/ and YDiscS
              for (jj = 0; jj <= maxPts; jj++)
              {
                //**/ marching ii-th input upward from lower to upper
                XGuess[ii] = xLower[ii]+jj*vecRange[ii]/maxPts;

                //**/ set up XGuessS for RS evaluation
                index = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  dcnt = 0;
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                  {
                    XGuessS[(index+kk2)*nInputs+ii2] = XGuess[ii2];
                    if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
                    {
                      XGuessS[(index+kk2)*nInputs+ii2] = 
                            matExpInps.getEntry(kk2,dcnt);
                      dcnt++;
                    }
                  }
                }
                //**/ set design outputs and stds (discrepancy) to zero 
                //**/ since they will be used but may not be set later
                for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++) 
                {
                  YDiscS[index*nOutputs+ii2] = 
                                 YDiscStds[index*nOutputs+ii2] = 0.0;
                  YGuessS[index*nOutputs+ii2] = 
                                 YGuessStds[index*nOutputs+ii2] = 0.0;
                }
              }
              //**/ run XGuessS through response surfaces
              for (ii2 = 0; ii2 < nOutputs; ii2++) 
              {
                //**/ case 1: if RS error is requested
                if (rsErrFlag == 1)
                {
                  faPtrs[ii2]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                                 XGuessS,&YGuessS[ii2*dnSamples*(maxPts+1)],
                                 &YGuessStds[ii2*dnSamples*(maxPts+1)]);
                  //**/ prepare discrepancy function, if available
                  if (modelFormFlag == 1)
                  {
                    //**/ if model form is not a constant function
                    if (modelFormConst == 0)
                    {
                      for (ii3 = 0; ii3 <= maxPts; ii3++)
                      {
                        for (kk2 = 0; kk2 < dnSamples; kk2++)
                        {
                          index = ii2*dnSamples*(maxPts+1)+
                                         ii3*dnSamples+kk2;
                          YDiscS[index] = 
                                      discSamOuts[ii2*dnSamples+kk2];
                          YDiscStds[index] = 0.0;
                        } 
                      }
                    }
                    else if (discFuncConstantMeans != NULL &&
                             discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
                    {
                      for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                      {
                        YDiscS[ii2*dnSamples*(maxPts+1)+kk2] = 
                                           discFuncConstantMeans[ii2];
                        YDiscStds[ii2*dnSamples*(maxPts+1)+kk2]=0.0;
                      }
                    }
                  }
                }
                //**/ case 2: if RS error is to be turned off
                else
                {
                  faPtrs[ii2]->evaluatePoint((maxPts+1)*dnSamples,
                                XGuessS,&YGuessS[ii2*dnSamples*(maxPts+1)]);
                  //**/ prepare discrepancy function, if available
                  if (modelFormFlag == 1)
                  {
                    //**/ if model form is not a constant function
                    if (modelFormConst == 0)
                    {
                      for (ii3 = 0; ii3 <= maxPts; ii3++)
                      {
                        for (kk2 = 0; kk2 < dnSamples; kk2++)
                        {
                          index = ii2*dnSamples*(maxPts+1)+ii3*dnSamples+kk2;
                          YDiscS[index] = discSamOuts[ii2*dnSamples+kk2];
                        } 
                      }
                    }
                    else if (discFuncConstantMeans != NULL &&
                             discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
                    {
                      for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                        YDiscS[ii2*dnSamples*(maxPts+1)+kk2] = 
                                           discFuncConstantMeans[ii2];
                    }
                    for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                      YGuessStds[ii2*dnSamples*(maxPts+1)+kk2] = 
                              YDiscStds[ii2*dnSamples*(maxPts+1)+kk2]=0;
                  }
                }
              }
              //**/ compute vecXDist
              for (jj = 0; jj <= maxPts; jj++)
              {
                vecXDist[jj] = 0.0;
                index = jj * dnSamples;
                for (ii2 = 0; ii2 < nOutputs; ii2++) 
                {
                  for (kk2 = 0; kk2 < dnSamples; kk2++)
                  {
                    if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
                    {
                      Ytemp=YGuessS[ii2*dnSamples*(maxPts+1)+index+kk2]+
                            YDiscS[ii2*dnSamples*(maxPts+1)+index+kk2];
                      stdev2=YDiscStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                    }
                    else
                    {
                      Ytemp=YGuessS[ii2*dnSamples*(maxPts+1)+index+kk2];
                      stdev2 = 0;
                    }
                    stdev = YGuessStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                    Ytemp2=pow(Ytemp-matExpMeans.getEntry(kk2,ii2),2.0)/
                          (pow(matExpStdvs.getEntry(kk2,ii2),2.0) + 
                           stdev*stdev + stdev2*stdev2);
                    vecXDist[jj] += Ytemp2;
                  }
                }
                //**/ more correct? (repeated experiment no accumulative
                vecXDist[jj] = vecXDist[jj] / (double) dnReps;
                if (masterOption == 1 && fp != NULL) 
                {
                  fprintf(fp, "%e ", (double) vecXDist[jj]);
                  for (ii2 = 0; ii2 < nOutputs; ii2++) 
                  {
                    for (kk2 = 0; kk2 < dnSamples; kk2++)
                    {
                      if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
                      {
                        Ytemp=YGuessS[ii2*dnSamples*(maxPts+1)+index+kk2]+
                              YDiscS[ii2*dnSamples*(maxPts+1)+index+kk2];
                        stdev2=YDiscStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                      }
                      else
                      {
                        Ytemp=YGuessS[ii2*dnSamples*(maxPts+1)+index+kk2];
                        stdev2=0.0;
                      }
                      stdev = YGuessStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                      Ytemp2=pow(Ytemp-matExpMeans.getEntry(kk2,ii2),2.0)/
                            (pow(matExpStdvs.getEntry(kk2,ii2),2.0)+
                             stdev*stdev + stdev2*stdev2);
                      fprintf(fp, "%e ", Ytemp2);
                    }
                  }
                  fprintf(fp, "\n");
                }
              }
            }

            //**/ at this point vecXDist[jj] has been computed
            //**/ now compute prior distribution probability in 
            //**/ vecSDist[jj]
            for (jj = 0; jj <= maxPts; jj++)
            {
              //**/ compute prior vecSDist
              ddata = 1.0;
              if (inputPDFs != NULL)
              {
                for (ii2 = 0; ii2 < nInputs; ii2++)
                {
                  if ((vecDesignP.length() == 0 || vecDesignP[ii2] == 0) && 
                       inputPDFs[ii2] != NULL &&
                      (vecRSIndices.length() == 0 || vecRSIndices[ii2] >= 0))
                  {
                    inputPDFs[ii2]->getPDF(iOne,&XGuess[ii2],&ddata2);
                    ddata *= ddata2;
                  }
                }
              }
              vecSDist[jj] = ddata; 
            }

            //**/ store the best negloglikelihood in the current scan
            //**/ best = smallest scaled rms
            ddata = vecXDist[0];
            for (jj = 1; jj <= maxPts; jj++) 
            {
              ddata2 = vecXDist[jj];
              if (ddata2 < ddata) ddata = ddata2;
            }
            XChains[iChain][chainCnt][nInputs] = ddata;

            //**/ store the best results and update vecXDist 
            for (jj = 0; jj <= maxPts; jj++)
            {
              //**/ store the best result so far (Ymax,vecXmax)
              if (vecXDist[jj] < Ymax)
              {
                Ymax = vecXDist[jj];
                for (ii2 = 0; ii2 < nInputs; ii2++) 
                  vecXmax[ii2] = XGuess[ii2];
                vecXmax[ii] = xLower[ii]+(xUpper[ii]-xLower[ii])*jj/maxPts;
                if (printLevel > 5)
                {
                  printf("Found new point at X = \n");
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                    printf("%3d : %e\n", ii2+1, XGuess[ii2]);
                }
              }
              //**/ set vecXDist to be prior * likelihood
              vecXDist[jj] = vecSDist[jj]*exp(-0.5*(vecXDist[jj]-ddata));
            }

            //**/ create CDF using trapezoidal rule
            for (jj = maxPts; jj > 0; jj--)
              vecXDist[jj] = 0.5 * (vecXDist[jj] + vecXDist[jj-1]);
            for (jj = 1; jj <= maxPts; jj++)
              vecXDist[jj] = vecXDist[jj] + vecXDist[jj-1];
               
            //**/ now that the proposal distribution vecXDist has been 
            //**/ computed
            if (masterOption == 1 && fp != NULL)
            {
              fprintf(fp,"];\n");
              fprintf(fp,"S = [\n");
              for (jj = 0; jj <= maxPts; jj++) 
                fprintf(fp, "%e\n", vecSDist[jj]);
              fprintf(fp,"];\n");
              fprintf(fp,"P = A(:,1) .* S;\n");
              fprintf(fp,"figure(1)\n");
              if (plotScilab())
                   fprintf(fp,"        set(gca(),\"auto_clear\",\"on\")\n");
              else fprintf(fp,"        hold off\n");
              fprintf(fp,"subplot(1,2,1)\n");
              fprintf(fp,"X = %e : %e : %e;\n",xLower[ii],
                      (xUpper[ii]-xLower[ii])/maxPts-1e-8,xUpper[ii]);
              fprintf(fp,"if length(X) > nPts\n");
              fprintf(fp,"   X = X(1:nPts);\n");
              fprintf(fp,"end;\n");
              fprintf(fp,"plot(X,P,'lineWidth',2.0)\n");
              fprintf(fp,"set(gca,'linewidth',2)\n");
              fprintf(fp,"set(gca,'fontweight','bold')\n");
              fprintf(fp,"set(gca,'fontsize',12)\n");
              fprintf(fp,"ylabel('-2 log likelihood','FontWeight','bold'");
              fprintf(fp,",'FontSize',12)\n");
              fprintf(fp,"title('Input %d','FontWeight','bold'",ii+1);
              fprintf(fp,",'FontSize',12)\n");
              fprintf(fp,"grid on\n");
              fprintf(fp,"box on\n");
              fprintf(fp,"subplot(1,2,2)\n");
              fprintf(fp,"P2 = exp(-0.5*P);\n");
              fprintf(fp,"for ii = 2 : %d\n",maxPts+1);
              fprintf(fp,"   P2(ii) = P2(ii) + P2(ii-1);\n");
              fprintf(fp,"end;\n");
              fprintf(fp,"P2 = P2 / P2(nPts);\n");
              fprintf(fp,"plot(X,P2,'lineWidth',2.0)\n");
              fprintf(fp,"set(gca,'linewidth',2)\n");
              fprintf(fp,"set(gca,'fontweight','bold')\n");
              fprintf(fp,"set(gca,'fontsize',12)\n");
              fprintf(fp,"title('Input %d','FontWeight','bold'",ii+1);
              fprintf(fp,",'FontSize',12)\n");
              fprintf(fp,"ylabel('likelihood CDF','FontWeight','bold'");
              fprintf(fp,",'FontSize',12)\n");
              fprintf(fp,"grid on\n");
              fprintf(fp,"box on\n");
              fprintf(fp,"ii = %d\n",ii+1);
              fprintf(fp,"disp(['Likelihood CDF : input ' int2str(ii)])\n");
            }

            //**/ if the distribution is nonzero, select a point for X[ii]
            if (printLevel > 4)
              printOutTS(PL_INFO,"Proposal distribution max = %e\n", 
                         (double) vecXDist[maxPts]);
            if (vecXDist[maxPts] - vecXDist[0] > 1.0e-16)
            {
              //**/ normalize vecXDist to [0,1] to make it a CDF
              for (jj = 1; jj <= maxPts; jj++)
                vecXDist[jj] = (vecXDist[jj] - vecXDist[0]) / 
                               (vecXDist[maxPts]-vecXDist[0]);
              vecXDist[0] = 0;
              if (masterOption == 1 && fp != NULL)
              {
                fprintf(fp,"XCDF = [\n");
                for (jj = 0; jj <= maxPts; jj++) 
                  fprintf(fp, "%e\n", (double) vecXDist[jj]);
                fprintf(fp,"];\n");
                fprintf(fp,"X = %e : %e : %e;\n",xLower[ii],
                          (xUpper[ii]-xLower[ii])/maxPts-1e-8,xUpper[ii]);
                fprintf(fp,"figure(2)\n");
                fprintf(fp,"plot(X,XCDF,'lineWidth',2.0)\n");
                fprintf(fp,"set(gca,'linewidth',2)\n");
                fprintf(fp,"set(gca,'fontweight','bold')\n");
                fprintf(fp,"set(gca,'fontsize',12)\n");
                fprintf(fp,"ylabel('likelihood CDF (check)','FontWeight',");
                fprintf(fp,"'bold','FontSize',12)\n");
                fprintf(fp,"title('Input %d','FontWeight','bold'",ii+1);
                fprintf(fp,",'FontSize',12)\n");
                fprintf(fp,"grid on\n");
                fprintf(fp,"box on\n");
              }

              //**/ randomly select a point in (0,1) and look up
              //**/ the CDF to find the actual X value
              XGuess[ii] = PSUADE_drand();

              //**/ CDF lookup
              index = 0;
              for (ii2 = 0; ii2 <= maxPts; ii2++) 
                if (XGuess[ii] <= vecXDist[ii2]) break;
              index = ii2;
              if (index == 0)
              {
                ddata = 0;
                printf("ERROR: This should not happen.\n");
                exit(1);
              }
              else
              {
                if (PABS(vecXDist[index]-vecXDist[index-1]) > 1.0e-16)
                  ddata=index-1+(XGuess[ii]-vecXDist[index-1])/
                              (vecXDist[index]-vecXDist[index-1]);
                else ddata = (double) index - 1.0;
              }
              XGuess[ii] = xLower[ii]+ddata*vecRange[ii]/maxPts;
            }
            //**/ if the distribution is zero, do not move
            else
            {
              XGuess[ii] = Xtemp;
              if (printLevel > 2)
                printOutTS(PL_INFO,
                     "MCMC_Gibbs iteration %7d : No change in input %d\n",
                     mcmcIts+1,ii+1);
            }
            if (masterOption == 1 && fp != NULL)
            {
              if (plotScilab())
                   fprintf(fp,"        set(gca(),\"auto_clear\",\"off\")\n");
              else fprintf(fp,"        hold on\n");
              fprintf(fp,"XX = %e * ones(2,1);\n", XGuess[ii]);
              fprintf(fp,"YY = [0 1]';\n");
              fprintf(fp,"plot(XX,YY,'r-','linewidth',2)\n");
              if (plotScilab())
                   fprintf(fp,"        set(gca(),\"auto_clear\",\"on\")\n");
              else fprintf(fp,"        hold off\n");
              fclose(fp);
              fp = NULL; 
              for (ii2 = 0; ii2 < nInputs; ii2++)
              {
                printOutTS(PL_INFO,
                   "Next guess point %d = %e ",ii2+1,XGuess[ii2]);
                if (ii2 == ii) printOutTS(PL_INFO," ***\n");
                else           printOutTS(PL_INFO,"\n");
              }
              printf("Now examine MCMCDistTrack.m for diagnostics.\n");
              printf("Next, enter 0 to continue without any more\n");
              printf("stops, enter n (an integer > 1) to skip <n>\n");
              printf("iterations, or enter anything else to go to\n");
              printf("the next iteration.\n");
              scanf("%d", &ii2);
              if (ii2 == 0) masterOption = 0;
              if (ii2 > 1) masterCount = ii2 + 1;
            }
            if (masterOption == 1 && masterCount > 1) masterCount--;

            //**/ update statistics/histograms, store all points in XSave
            //**/ for computing overall statistics (leave the first 
            //**/ maxSamples/2 points as burn-in
            if (mcmcIts >= burnInSamples || globalIts > 0)
            {
              //**/ update the 1D histogram
              for (ii2 = 0; ii2 < nInputs; ii2++) 
              {
                XChains[iChain][chainCnt][ii2] = XGuess[ii2];
                ddata = (XGuess[ii2] - xLower[ii2]) / vecRange[ii2];
                index = (int) (ddata * nbins);
                if (index >= nbins) index = nbins - 1;
                bins[index][ii2]++;
              }
              //**/ update the 2D histogram
              for (ii2 = 0; ii2 < nInputs; ii2++) 
              {
                ddata = (XGuess[ii2] - xLower[ii2]) / vecRange[ii2];
                index = (int) (ddata * nbins);
                if (index >= nbins) index = nbins - 1;
                for (ii3 = 0; ii3 < nInputs; ii3++) 
                {
                  ddata2 = (XGuess[ii3] - xLower[ii3]) / vecRange[ii3];
                  index2 = (int) (ddata2 * nbins);
                  if (index2 >= nbins) index2 = nbins - 1;
                  bins2[index][index2][ii2][ii3]++;
                }
              }
              chainCnt++;
            }
            mcmcIts++;
          }
          if (mcmcIts >= maxSamples) break;
        }
        //**/ scanning through all inputs completed
      }
      if (countTrack <= 10) printOutTS(PL_INFO,"100%%\n");
      else                  printOutTS(PL_INFO,"\n");
      //**/ scanning through all maxSamples for the current chain 
      if (printLevel >= 0)
      {
        printOutTS(PL_INFO,
             "       Chain %d current final guess : \n",iChain+1);
        for (ii = 0; ii < nInputs; ii++)
          if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >=0) && 
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
            printOutTS(PL_INFO,"          Input %4d = %e\n",ii+1,
                       XGuess[ii]);
        printOutTS(PL_INFO,
             "       Current best guess (from all chains) : \n");
        for (ii = 0; ii < nInputs; ii++)
          if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >=0) && 
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
            printOutTS(PL_INFO,"          Input %4d = %e\n",ii+1,
                       vecXmax[ii]);
         printOutTS(PL_INFO,"          Negative log likelihood = %e\n",
                    Ymax);
      }
    } // ichain loop
 
    //**/ ----------------------------------------------------------------
    //**/ check chains and compute convergence statistics (PSRF)
    //**/ ----------------------------------------------------------------
    globalIts++;
    mcmcFail = nInputs - dnInputs;
    if (vecRSIndices.length() > 0)
    {
      for (ii = 0; ii < nInputs; ii++)
        if (vecRSIndices[ii] < 0) mcmcFail--; 
    }
    if (chainCnt > 0)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, "Iteration %d summary: (total = %d)\n", 
                 globalIts, globalIts*maxSamples);
      printDashes(PL_INFO, 0);
      for (ii = 0; ii < nInputs; ii++)
      {
        if ((vecRSIndices.length() == 0 || 
            (vecRSIndices.length() > 0 && vecRSIndices[ii] >=0)) && 
            (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
        {
          if (printLevel > 2) printOutTS(PL_INFO, "Input = %d\n", ii+1);
       
          //**/ check that if a chain does not walk at all and turn it off
          for (iChain = 0; iChain < numChains; iChain++)
          {
            ddata = 0.0;
            for (jj = 0; jj < chainCnt; jj++) 
              ddata += XChains[iChain][jj][ii];
            ddata /= (double) chainCnt;
            ddata2 = 0.0;
            for (jj = 0; jj < chainCnt; jj++) 
               ddata2 += pow(XChains[iChain][jj][ii]-ddata,2.0);
            ddata2 /= (double) (chainCnt - 1);
            vecChainMeans[iChain] = ddata;
            vecChainStds[iChain] = ddata2;
            if (globalIts > 2 && vecChainStds[iChain] < 1.0e-20) 
            {
              printOutTS(PL_INFO,
                   "MCMC INFO: chain %d disabled.\n",iChain+1);
              vecChainStas[iChain] = 1;
            }
          }
          nChainGood = 0;
          for (iChain = 0; iChain < numChains; iChain++)
            if (vecChainStas[iChain] == 0) nChainGood++;
          if (nChainGood <= 1)
          {
            printOutTS(PL_ERROR,"MCMC_Gibbs ERROR: too few chains <= 1.\n");
            printOutTS(PL_ERROR,
             "Suggestion: You may want to relax the experimental data\n");
            printOutTS(PL_ERROR,"     uncertainties (make them larger).\n");
            printOutTS(PL_ERROR,
             "     To see if this is the problem, turn on printlevel\n");
            printOutTS(PL_ERROR,
             "     to 3 and run again. If the variance of the chains\n");
            printOutTS(PL_ERROR,
             "     are small, small data uncertainties is probably the\n");
            printOutTS(PL_ERROR,"     problem.\n");
            exit(1);
          }
          //**/ compute PSRF
          WStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            if (vecChainStas[iChain] == 0) 
              WStat += vecChainStds[iChain];
          WStat /= (double) nChainGood;
          if (WStat < 0) WStat = PSUADE_UNDEFINED;
          if (printLevel > 2) 
            printOutTS(PL_INFO,"  Within  chain variance W = %e\n",WStat);
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            if (vecChainStas[iChain] == 0) 
              ddata += vecChainMeans[iChain];
          ddata /= (double) nChainGood;
          BStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            if (vecChainStas[iChain] == 0)
              BStat += pow(vecChainMeans[iChain]-ddata,2.0);
          BStat = BStat / (nChainGood - 1.0) * chainCnt;
          if (printLevel > 2) 
            printOutTS(PL_INFO,
              "  Between chain variance B = %e\n",BStat/chainCnt);
          ddata = (1 - 1.0/chainCnt) * WStat + BStat / chainCnt;
          ddata = ddata / WStat * (nChainGood + 1) / nChainGood - 
                  (chainCnt - 1.0) / (double) (chainCnt * nChainGood); 
          if (ddata < 0) ddata2 = PSUADE_UNDEFINED;
          else           ddata2 = sqrt(ddata);
          if (printLevel > 2)
          {
            for (iChain = 0; iChain < numChains; iChain++)
            {
              if (vecChainStas[iChain] == 0)
                printOutTS(PL_INFO,"  Chain %d mean, var = %e %e\n",
                    iChain+1,
                    vecChainMeans[iChain]*vecRange[ii]+xLower[ii],
                    vecChainStds[iChain]*vecRange[ii]*vecRange[ii]);
              else printf("  Chain %d : disabled.\n",iChain+1);
            }
            printOutTS(PL_INFO,"  Chain length             = %d\n",
                       chainCnt);
            printOutTS(PL_INFO,"  Weighted average of B, W = %e\n", 
                       ddata);
          }
          printOutTS(PL_INFO,"  Input %d PSRF = %e\n", ii+1, ddata2);
          vecPsrfs[ii] = ddata2;
          if (ddata2 < psrfThreshold)
          {
            printOutTS(PL_INFO,
              "MCMC_Gibbs INFO : PSRF < %e ==> converged.\n",
              psrfThreshold);
            mcmcFail--;
          }
          //**/ update vecMeans_ and vecSigmas_
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (vecChainStas[iChain] == 0)
              for (jj = 0; jj < chainCnt; jj++)
                 ddata += XChains[iChain][jj][ii];
          }
          ddata /= (double) (nChainGood * chainCnt);
          vecMeans_[ii] = ddata;
          ddata2 = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (vecChainStas[iChain] == 0)
              for (jj = 0; jj < chainCnt; jj++)
                ddata2 += pow(XChains[iChain][jj][ii]-ddata,2.0);
          }
          ddata2 /= (double) (chainCnt*nChainGood-1);
          vecSigmas_[ii] = sqrt(ddata2);
        }
      }

      //**/ --------------------------------------------------------------
      //**/ if printlevel > 3, do Geweke statistics (but do not use for 
      //**/ convergence termination), since Geweke statistics not good 
      //**/ for multi-modal 
      //**/ --------------------------------------------------------------
      if (mcmcFail == 0 && printLevel > 3 && s2Analyzer != NULL)
      {
        if (vecRSIndices.length() > 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            if (vecRSIndices[ii] < 0) mcmcFail--; 
        }
        for (ii = 0; ii < nInputs; ii++)
        {
          if ((vecRSIndices.length() == 0 || 
              (vecRSIndices.length() > 0 && vecRSIndices[ii] >=0)) && 
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
          {
            printOutTS(PL_INFO, "Geweke Input = %d\n", ii+1);
     
            cnt = chainCnt / 2;
            ddata2 = (double) numChains;
            for (iChain = 0; iChain < numChains; iChain++)
            {
              for (jj = 0; jj < 2*cnt; jj++) 
                vecS2[jj] = XChains[iChain][chainCnt-2*cnt+jj][ii];
              ddata = s2Analyzer->TAnalyze(cnt,vecS2.getDVector(),
                                    cnt,&s2Vec[cnt],1);
            }
          }
        }
      }

      //**/ --------------------------------------------------------------
      //**/ print posterior input statistics
      //**/ --------------------------------------------------------------
      for (ii = 0; ii < nInputs; ii++) 
      {
        if ((vecRSIndices.length() == 0 || 
            (vecRSIndices.length() > 0 && vecRSIndices[ii] >=0)) && 
            (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
        {
          printOutTS(PL_INFO,
             "MCMC_Gibbs: input %3d value at peak of likelihood = %e\n",
                   ii+1, vecXmax[ii]);
          ddata = vecMeans_[ii];
          printOutTS(PL_INFO,
                   "MCMC_Gibbs: input %3d mean    = %e\n", ii+1, ddata);
          ddata = vecSigmas_[ii];
          printOutTS(PL_INFO,
                   "MCMC_Gibbs: input %3d std dev = %e\n", ii+1, ddata);
          vecMostLikelyInputs_[ii] = vecXmax[ii];
        }
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ find MLE negative log-likelihood
    //**/ ----------------------------------------------------------------
    Ymax = PSUADE_UNDEFINED;
    for (iChain = 0; iChain < numChains; iChain++) 
    { 
      if (vecChainStas[iChain] == 0)
      {
        for (jj = 0; jj < chainCnt; jj++) 
        {
          ddata = XChains[iChain][jj][nInputs];
          if (ddata < Ymax) Ymax = ddata;
        }
      }
    }
    printOutTS(PL_INFO,
      "MCMC_Gibbs: MLE Negative log likelihood = %e\n",Ymax);
    printAsterisks(PL_INFO, 0);

    //**/ ----------------------------------------------------------------
    //**/ generate all chains information for user inspection, if GM mode
    //**/ is on
    //**/ ----------------------------------------------------------------
    if (masterOption == 2)
    {
      fp = fopen("MCMCChainHistogram.m", "w");
      if (fp != NULL)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0) &&
              (vecDesignP.length() == 0 || vecDesignP[ii] == 0)) 
          {
            fprintf(fp,"X%d = [\n",ii+1);
            for (jj = 0; jj < chainCnt; jj++)
            {
              for (iChain = 0; iChain < numChains; iChain++)
                 fprintf(fp,"%e ", XChains[iChain][jj][ii]);
              fprintf(fp,"\n");
            }
            fprintf(fp,"];\n");
            fprintf(fp,"[nk, nx] = hist(X%d);\n", ii+1);
            fprintf(fp,"nk = nk / %d;\n", chainCnt);
            fprintf(fp,"bar(nx, nk)\n");
            fprintf(fp,"%%plot(X%d)\n", ii+1);
            snprintf(pString,100,"Input %d", ii+1);
            fwritePlotTitle(fp, pString);
            fwritePlotXLabel(fp, "Input Value");
            fwritePlotYLabel(fp, "Input Count");
            fwritePlotAxes(fp);
            fprintf(fp,
               "text(0.05,0.9,'colors are for different chains','sc')\n");
            fprintf(fp,
               "disp('Colors in histograms are for different chains.\n");
            fprintf(fp,"disp('Press enter to continue to next input')\n");
            fprintf(fp,"pause\n");
          }
        }
        fclose(fp);
        fp = NULL;
        printOutTS(PL_INFO,
           "MCMC_Gibbs INFO: MCMCChainHistogram.m has been created.\n");
        printOutTS(PL_INFO,
           "      Use Matlab to view the histograms of all chains for\n");
        printOutTS(PL_INFO,"      all inputs to assess convergence.\n");
        snprintf(pString,100,"Enter 1 to continue or 0 to terminate : ");
        ii = getInt(0, 10, pString);
        if (ii == 0) break;
      }
      else
      {
        printOutTS(PL_INFO,
             "MCMC_Gibbs INFO: cannot create MCMCChainHistogram.m\n");
      } 
    }

    //**/ ----------------------------------------------------------------
    //**/ if all converge and certain number of samples have been reached
    //**/ ----------------------------------------------------------------
    if (mcmcFail == 0 && globalIts > 3) break;

    //**/ -------------------------------------------------------------
    //**/ create matlabmcmc2.m at every major iteration
    //**/ -------------------------------------------------------------
    genMatlabFile(nInputs,xLower,xUpper,vecRange.getDVector(),nPlots,
             vecPlotIndices.getIVector(),nbins,
             NULL,NULL,bins,bins2,qData,numChains,chainCnt,XChains,
             vecChainStas.getIVector(),vecXmax.getDVector(),0);

    //**/ -------------------------------------------------------------
    //**/ option to stop MCMC
    //**/ -------------------------------------------------------------
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
           "MCMC_Gibbs INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
      fclose(fp);
      fp = NULL;
      strcpy(pString, "psuade_stop");
      unlink(pString);
      break;
    }

    //**/ -------------------------------------------------------------
    //**/ features for better run time diagnostics
    //**/ -------------------------------------------------------------
    fp = fopen("psuade_nogm", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
           "MCMC_Gibbs INFO: psuade_nogm FILE FOUND. GM mode is now off.\n");
      fclose(fp);
      fp = NULL;
      psConfig_.GMModeOff();
      masterOption = 0;
      strcpy(pString, "psuade_nogm");
      unlink(pString);
    }
    fp = fopen("psuade_gm", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
           "MCMC_Gibbs INFO: psuade_gm FILE FOUND. GM mode is now on.\n");
      fclose(fp);
      fp = NULL;
      psConfig_.GMModeOn();
      printf("Please choose from the following option (or 0 if none): \n");
      printf("  1. track proposal distribution at each MCMC iteration\n");
      printf("  2. track posterior distributions after each MCMC cycle\n");
      scanf("%s", pString);
      fgets(lineIn,1000,stdin);
      if (pString[0] == '1') masterOption = 1;
      if (pString[0] == '2') masterOption = 2;
      strcpy(pString, "psuade_gm");
      unlink(pString);
    }
    fp = fopen("psuade_print", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
        "MCMC_Gibbs INFO: psuade_print FILE FOUND. Set print level = 3.\n");
      fclose(fp);
      fp = NULL;
      printLevel = 3;
      strcpy(pString, "psuade_print");
      unlink(pString);
    }
  } // while loop

  //**/ ---------------------------------------------------------------
  //**/ final check for convergence
  //**/ ---------------------------------------------------------------
  if (globalIts >= maxGlobalIts)
  {
    mcmcFail = 0;
    for (ii = 0; ii < nInputs; ii++) 
    {
      if ((vecRSIndices.length() == 0 || 
          (vecRSIndices.length() > 0 && vecRSIndices[ii] >=0)) && 
          (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
         if (vecPsrfs[ii] > psrfThreshold) mcmcFail = 1;
    }
    if (mcmcFail == 1) 
      printOutTS(PL_INFO,
        "MCMC_Gibbs: maximum iterations exceeded but no convergence.\n");
  }
  else printOutTS(PL_INFO, "MCMC_Gibbs iterations completed\n");
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ create final binning for posterior distribution
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++) 
    for (jj = 0; jj < nbins; jj++) bins[jj][ii] = 0;
  for (ii = 0; ii < nInputs; ii++) 
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      for (jj = 0; jj < nbins; jj++)
        for (jj2 = 0; jj2 < nbins; jj2++)
          bins2[jj][jj2][ii][ii2] = 0;
  for (iChain = 0; iChain < numChains; iChain++) 
  { 
    if (vecChainStas[iChain] == 0)
    {
      for (jj = 0; jj < chainCnt; jj++) 
      {
        for (ii2 = 0; ii2 < nInputs; ii2++) 
        {
          ddata = XChains[iChain][jj][ii2];
          ddata = (ddata - xLower[ii2]) / (xUpper[ii2] - xLower[ii2]);
          index = (int) (ddata * nbins);
          if (index >= nbins) index = nbins - 1;
          bins[index][ii2]++;
        }
        for (ii2 = 0; ii2 < nInputs; ii2++) 
        {
          ddata = XChains[iChain][jj][ii2];
          ddata = (ddata - xLower[ii2]) / (xUpper[ii2] - xLower[ii2]);
          index = (int) (ddata * nbins);
          if (index >= nbins) index = nbins - 1;
          for (ii3 = 0; ii3 < nInputs; ii3++) 
          {
            ddata2 = XChains[iChain][jj][ii3];
            ddata2 = (ddata2 - xLower[ii3]) / (xUpper[ii3] - xLower[ii3]);
            index2 = (int) (ddata2 * nbins);
            if (index2 >= nbins) index2 = nbins - 1;
            bins2[index][index2][ii2][ii3]++;
          }
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate matlab plot files 
  //**/ ---------------------------------------------------------------
  dataPtr->getParameter("method_sampling", pPtr);
  int methodSave = pPtr.intData_;
  dataPtr->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
  PDFManager *pdfman = new PDFManager();
  pdfman->initialize(dataPtr);
  psVector vecLB, vecUB, vecOut;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  int nSamps = 100000;
  vecOut.setLength(nSamps*nInputs);
  pdfman->genSample(nSamps, vecOut, vecLB, vecUB);
  dataPtr->updateMethodSection(methodSave,-1,-1,-1,-1);
  delete pdfman;

  //**/ ---------------------------------
  //**/ bins for prior
  //**/ ---------------------------------
  int **pbins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    pbins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) pbins[ii][jj] = 0;
  }
  int ****pbins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    pbins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      pbins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        pbins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          pbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  double ddata1;
  for (ii = 0; ii < nInputs; ii++)
  {
    ddata1 = nbins / (xUpper[ii] - xLower[ii]);
    for (jj = 0; jj < nSamps; jj++)
    {
      ddata = vecOut[jj*nInputs+ii];
      ddata -= xLower[ii];
      ddata *= ddata1;
      kk = (int) ddata;
      if (kk >= nbins) kk = nbins - 1; 
      pbins[kk][ii]++;
    }
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    ddata1 = nbins / (xUpper[ii] - xLower[ii]);
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      ddata2 = nbins / (xUpper[ii2] - xLower[ii2]);
      for (jj = 0; jj < nSamps; jj++)
      {
        ddata = vecOut[jj*nInputs+ii];
        ddata -= xLower[ii];
        ddata *= ddata1;
        kk = (int) ddata;
        if (kk >= nbins) kk = nbins - 1; 
        ddata = vecOut[jj*nInputs+ii2];
        ddata -= xLower[ii2];
        ddata *= ddata2;
        kk2 = (int) ddata;
        if (kk2 >= nbins) kk2 = nbins - 1; 
        pbins2[kk][kk2][ii][ii2]++;
      }
    }
  }
  //**/ ---------------------------------
  //**/ call create matlab with prior and
  //**/ posterior bins. Then clean up
  //**/ ---------------------------------
  genMatlabFile(nInputs,xLower,xUpper,vecRange.getDVector(),nPlots,
        vecPlotIndices.getIVector(),nbins,
        pbins,pbins2,bins,bins2,qData,numChains,chainCnt,XChains,
        vecChainStas.getIVector(), vecXmax.getDVector(),0);
  for (ii = 0; ii < nbins; ii++) delete [] pbins[ii];
  delete [] pbins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] pbins2[jj][jj2][ii];
      delete [] pbins2[jj][jj2];
    }
    delete [] pbins2[jj];
  }
  delete [] pbins2;

  //**/ ---------------------------------------------------------------
  //**/  generate the log likelihood distribution 
  //**/ ---------------------------------------------------------------
  if (genPosteriors == 1 && printLevel > 3)
  {
    cnt = nChainGood * chainCnt;
    if (cnt > 200000) cnt = 200000;
    cnt /= nChainGood;
    genPostLikelihood(nInputs,xLower,xUpper,vecRange.getDVector(),
       numChains, chainCnt,XChains, vecChainStas.getIVector(), cnt, 
       vecRSIndices.getIVector(), vecRSValues.getDVector(),
       vecDesignP,dnInputs,dnSamples,matExpInps,faPtrs,nOutputs,
       discSamOuts,discFuncConstantMeans,matExpMeans,matExpStdvs, 
       modelFormFlag,modelFormConst, vecModelForms);
  }

  //**/ ---------------------------------------------------------------
  //**/  generate the posterior sample file
  //**/ ---------------------------------------------------------------
  if (genPosteriors == 1)
  {
    fp = fopen("MCMCPostSample", "w");
    if (fp != NULL)
    {
      printf("MCMC_Gibbs: posterior sample generation: \n");
      fprintf(fp, "PSUADE_BEGIN\n");
      cnt = nChainGood * chainCnt;
      if (cnt > 50000) cnt = 50000;
      psMatrix matXStore;
      matXStore.setFormat(PS_MAT2D);
      matXStore.setDim(nInputs, cnt);
      double **XStore = matXStore.getMatrix2D();
      cnt /= nChainGood;
      //**/ count the number of uncertain parameters
      kk = 0;
      for (ii = 0; ii < nInputs; ii++) 
        if (vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0)
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) kk++;
      fprintf(fp, "%d %d\n", cnt*nChainGood,kk);
      if (qData.strArray_ != NULL)
      {
        fprintf(fp, "# ");
        for (jj = 0; jj < nInputs; jj++)
          if (vecRSIndices.length() == 0 || vecRSIndices[jj] >= 0)
            if (vecDesignP.length() == 0 || vecDesignP[jj] == 0) 
              fprintf(fp,"%s ", qData.strArray_[jj]);
        fprintf(fp, "\n");
      }
      ii2 = 0;
      for (iChain = 0; iChain < numChains; iChain++)
      { 
        if (vecChainStas[iChain] == 0)
        {
          //**/ select the last batch (stabilized Markov chain)
          for (ii = chainCnt-cnt; ii < chainCnt; ii++)
          {
            fprintf(fp, "%d ", ii2+1);
            for (jj = 0; jj < nInputs; jj++)
            {
              if ((vecRSIndices.length() == 0 || vecRSIndices[jj] >= 0) &&
                  (vecDesignP.length() == 0 || vecDesignP[jj] == 0)) 
              {
                ddata = XChains[iChain][ii][jj];
                fprintf(fp, "%e ", ddata);
                XStore[jj][ii2] = ddata;
              }
              // May 2021: do not include design parameters
              //else if (vecRSIndices.length() > 0 && vecRSIndices[jj] < 0)
              //  fprintf(fp, "%e ", vecRSValues[jj]);
              //else if (vecDesignP.length() > 0 && vecDesignP[jj] != 0) 
              //  fprintf(fp, "%e ", 0.5 * (xUpper[jj] + xLower[jj]));
            }
            fprintf(fp, "\n");
            ii2++;
          }
        }
      }
      fprintf(fp, "PSUADE_END\n");
      fprintf(fp, "#N=%d;\n",nChainGood*cnt);
      fprintf(fp, "#m=%d;\n",cnt);
      for (iChain = 0; iChain < numChains; iChain++)
      {
        if (vecChainStas[iChain] == 0)
          fprintf(fp, "#A%d = A(%d*m+1:%d*m,:);\n",iChain+1,iChain,iChain+1);
      }
      fprintf(fp, "#for ii = 2 : %d\n", nInputs+1);
      for (iChain = 0; iChain < numChains; iChain++)
      {
        if (vecChainStas[iChain] == 0)
        {
          fprintf(fp, "#subplot(*,*,%d)\n",iChain+1);
          fprintf(fp, "#hist(A%d(:,ii))\n",iChain+1);
        }
        fprintf(fp, "#ii-1\n");
        fprintf(fp, "#pause;\n");
      }
      fprintf(fp, "#end;\n");
      fprintf(fp, "Best negLogLikelihood = %e (Ideal=0)\n",Ymax);

      //**/ for the MLE solution, dissect the component of likelihood
      writeMLEInfo(fp, nInputs, nOutputs, faPtrs,  
                   vecDesignP.getIVector(), vecRSIndices.getIVector(), 
                   vecRSValues.getDVector(), vecXmax.getDVector(), 
                   dnSamples, dnInputs, matExpInps.getMatrix1D(), 
                   matExpMeans.getMatrix1D(), matExpStdvs.getMatrix1D(), 
                   modelFormFlag, modelFormConst, discSamOuts, 
                   discFuncConstantMeans, vecModelForms);
      fclose(fp);
      printOutTS(PL_INFO,
         "  * 'MCMCPostSample' file has a posterior sample.\n");
      printf("  * Posterior sample statistics : \n");
      int    ss, nSams=ii2;
      double ddmean, ddstds;
      for (ii = 0; ii < nInputs; ii++)
      {
        if ((vecRSIndices.length() == 0 || vecRSIndices[ii] >= 0) &&
            (vecDesignP.length() == 0 || vecDesignP[ii] == 0)) 
        {
          ddmean = 0;
          for (ss = 0; ss < nSams; ss++)
            ddmean += XStore[ii][ss];
          ddmean /= (double) nSams;
          ddstds = 0;
          for (ss = 0; ss < nSams; ss++)
            ddstds += pow(XStore[ii][ss]-ddmean,2.0);
          ddstds /= (double) nSams;
          ddstds = sqrt(ddstds); 
          printf("      Input %d mean, stds = %12.5e %12.5e\n",  
                 ii+1, ddmean, ddstds);
        }
      }
      //printf("  * Check these statistics against MCMC statistics ");
      //printf("above. If there is\n");
      //printf("    significant discrepancy, some parameters may ");
      //printf("need to be tweaked.\n");
      //printf("    (Consult PSUADE developers.)\n");
      printAsterisks(PL_INFO, 0);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy sample, if desired 
  //**/ ---------------------------------------------------------------
  int nInps, nOuts, nSams, *states;
  psVector  vecAllOuts;
  psStrings strNames;
  PsuadeData *filePtr1, *filePtr2;

  if (modelFormFlag == 1)
  {
    snprintf(pString,100,"psDiscrepancyModel1");
    filePtr1 = new PsuadeData();
    status = filePtr1->readPsuadeFile(pString);
    if (status != 0)
    {
      printOutTS(PL_ERROR,
        "MCMC_Gibbs ERROR: cannot read file %s in PSUADE format.\n",
        pString);
      delete filePtr1;
      exit(1);
    } 
  }
  if (modelFormFlag == 1 && status == 0)
  {
    filePtr1->getParameter("input_ninputs", pPtr);
    nInps = pPtr.intData_;
    filePtr1->getParameter("output_noutputs", pPtr);
    nOuts = pPtr.intData_;
    filePtr1->getParameter("method_nsamples", pPtr);
    nSams = pPtr.intData_;
    filePtr1->getParameter("output_sample", pOutputs);
    unlink(pString);
    vecAllOuts.setLength(nOutputs * nSams);
    if (vecModelForms.length() > 0 && vecModelForms[0] == 1)
    {
      for (jj = 0; jj < nSams; jj++) 
        vecAllOuts[jj*nOutputs] = pOutputs.dbleArray_[jj];
    }
    pOutputs.clean();
    strNames.setNumStrings(nOutputs);
    strcpy(pString, "Y1");
    strNames.loadOneString(0, pString);
    for (ii = 1; ii < nOutputs; ii++)
    {
      filePtr2 = new PsuadeData();
      snprintf(pString,100,"psDiscrepancyModel%d", ii+1);
      status = filePtr2->readPsuadeFile(pString);
      if (status != 0) break;
      filePtr2->getParameter("input_ninputs", pPtr);
      if (pPtr.intData_ != nInps) break;
      filePtr2->getParameter("output_noutputs", pPtr);
      if (pPtr.intData_ != nOuts) break;
      filePtr2->getParameter("method_nsamples", pPtr);
      if (pPtr.intData_ != nSams) break;
      filePtr2->getParameter("output_sample", pOutputs);
      delete filePtr2;
      unlink(pString);
      if (vecModelForms.length() > 0 && vecModelForms[ii] == 1)
      {
        for (jj = 0; jj < nSams; jj++) 
          vecAllOuts[jj*nOutputs+ii] = pOutputs.dbleArray_[jj];
      }
      pOutputs.clean();
      snprintf(pString,100,"Y%d", ii+1);
      strNames.loadOneString(ii, pString);
    }
    if (nOutputs == 1)
    { 
      snprintf(pString,100,"psDiscrepancyModel");
      filePtr1->writePsuadeFile(pString, 0);
    }
    else if (ii == nOutputs)
    {
      psIVector vecStates;
      vecStates.setLength(nSams);
      for (jj = 0; jj < nSams; jj++) vecStates[jj] = 1;
      filePtr1->updateOutputSection(nSams,nOutputs,vecAllOuts.getDVector(),
                       vecStates.getIVector(),strNames.getStrings());
      snprintf(pString,100,"psDiscrepancyModel");
      filePtr1->writePsuadeFile(pString, 0);
      printOutTS(PL_INFO,"MCMC_Gibbs INFO: A sample (inputs/outputs) ");
      printOutTS(PL_INFO,"for the discrepancy model\n");
      printOutTS(PL_INFO,"           is now in psDiscrepancyModel.\n");
    }
    else
    {
      printOutTS(PL_INFO,
         "MCMC_Gibbs INFO: Unsuccessful creation of discrepancy sample file\n");
    }
    delete filePtr1;
  }
  
  //**/ ---------------------------------------------------------------
  // final clean up
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < numChains; ii++)
  {
    for (jj = 0; jj < maxGlobalIts*maxSamples; jj++)
      delete [] XChains[ii][jj];
    delete [] XChains[ii];
  }
  delete [] XChains;
  if (inputPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (inputPDFs[ii] != NULL) delete inputPDFs[ii];
    delete [] inputPDFs;
  }
  if (s2Analyzer != NULL) delete s2Analyzer;
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++) 
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  return 0.0;
}

// ************************************************************************
// perform MCMC-like analysis (brute force)
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_bf(aData &adata)
{
  int    ii, ii2, jj, jj2, kk, kk2, status, cnt, iOne=1, iZero=0;
  double ddata;
  char   lineIn[1001], pString[1000], *rsFile=NULL;
  FILE   *fp=NULL;
  pData  pPtr, qData, pOutputs;

  //**/ ---------------------------------------------------------------
  // display header 
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*              Brute Force Inference\n");
  printEquals(PL_INFO, 0);
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
    printOutTS(PL_INFO," * ana_expert to finetune MCMC parameters, \n");
    printOutTS(PL_INFO,"   (e.g. sample size for burn-in can be adjusted).\n");
    printOutTS(PL_INFO,
         " * rs_expert to customize response surface for MCMC,\n");
    printOutTS(PL_INFO," * printlevel 3 to display more diagnostics info.\n");
    printOutTS(PL_INFO," * printlevel 4 to display even more diagnostics.\n");
    printOutTS(PL_INFO," * printlevel >=5 reserved only for expert only.\n");
    printDashes(PL_INFO,0);
    printOutTS(PL_INFO,"FEATURES AVAILABLE IN THE CURRENT VERSION OF MCMC:\n");
    printOutTS(PL_INFO,
         " * Support likelihood functions from multiple outputs\n");
    printOutTS(PL_INFO," * Option to include response surface errors for ");
    printOutTS(PL_INFO,"polynomial\n");
    printOutTS(PL_INFO,"   regressions, bootstrapped MARS, and ");
    printOutTS(PL_INFO,"Gaussian process (GP).\n");
    printOutTS(PL_INFO,"   - can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO," * Option to include model form errors in the ");
    printOutTS(PL_INFO,"form of discrepancy\n");
    printOutTS(PL_INFO,"   models\n");
    printOutTS(PL_INFO,"   - can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO," * Option to set some inputs as design parameters\n");
    printOutTS(PL_INFO,
         "   - to be specified in the observation data spec file\n");
    printOutTS(PL_INFO,
         " * Option to disable some parameters (set to default)\n");
    printOutTS(PL_INFO,
         "   - in case these parameters are not to be calibrated\n");
    printOutTS(PL_INFO,
         "   - use rs_index_file in PSUADE's ANALYSIS section\n");
    printOutTS(PL_INFO,"   - not available with discrepancy modeling\n");
    printOutTS(PL_INFO,
         " * MCMC can be terminated gracefully by creating a file ");
    printOutTS(PL_INFO, "named\n");
    printOutTS(PL_INFO,
         "   'psuade_stop' in the same directory while it is running\n");
    printOutTS(PL_INFO,"   (if it takes too long).\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  // extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  nInputs_     = nInputs;
  nOutputs_    = nOutputs;
  double *XIn  = adata.sampleInputs_;
  double *YIn  = adata.sampleOutputs_;
  double *lower = adata.iLowerB_;
  double *upper = adata.iUpperB_;
  int    *pdfFlags = adata.inputPDFs_;
  int noPDF = 1;
  if (pdfFlags != NULL)
    for (ii = 0; ii < nInputs; ii++) if (pdfFlags[ii] != 0) noPDF = 0;

  //**/ ---------------------------------------------------------------
  //**/ get input names for plotting later
  //**/ ---------------------------------------------------------------
  PsuadeData *dataPtr = adata.ioPtr_;
  if (dataPtr != NULL) dataPtr->getParameter("input_names", qData);

  //**/ ---------------------------------------------------------------
  //**/ This is an option to specify a rs index file in the data file.
  //**/ This option allows one to disable certain inputs in the MCMC
  //**/ optimization by fixing some inputs 
  //**/ ==> rsIndices, rsValues, faType, faPtrs
  //**/ ---------------------------------------------------------------
  int faType;
  psIVector vecRSIndices;
  psVector  vecRSValues;
  if (dataPtr != NULL)
  {
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
    dataPtr->getParameter("ana_rsindexfile", pPtr);
    rsFile = pPtr.strArray_[0];
    if (strcmp(rsFile, "NONE"))
    {
      printOutTS(PL_INFO,
           "A response surface index file has been specified.\n");
      fp = fopen(rsFile, "r");
      if (fp == NULL)
      {
        printOutTS(PL_ERROR,
             "MCMC_BF ERROR: rs_index_file %s not found.\n",rsFile);
        return PSUADE_UNDEFINED;
      }
      else
      {
        printOutTS(PL_INFO,"INFO: rs_index_file %s found.\n",rsFile);
        fscanf(fp,"%d", &kk);
        if (kk != nInputs)
        {
          printOutTS(PL_ERROR,
            "MCMC_BF ERROR: invalid nInputs in rs_index_file (%d != %d).\n",
            kk, nInputs);
          printOutTS(PL_ERROR,"  Data format should be: \n");
          printOutTS(PL_ERROR,
              "  line 1: nInputs in rs data (driver) file\n");
          printOutTS(PL_ERROR,
              "  line 2: 1 <1 or 0> <default value if 0, -1 if sample>\n");
          printOutTS(PL_ERROR,
              "  line 3: 2 <2 or 0> <0 if it is == 2>\n");
          printOutTS(PL_ERROR,
              "  line 4: 3 <3 or 0> <default value if 0, -1 if sample>\n");
          printOutTS(PL_ERROR,
              "  line 5: 4 <4 or 0> <0 if it is == 4>\n");
          printOutTS(PL_ERROR,"  ...\n");
          fclose(fp);
          return PSUADE_UNDEFINED;
        }
        vecRSIndices.setLength(nInputs);
        vecRSValues.setLength(nInputs);
        for (ii = 0; ii < nInputs; ii++) vecRSIndices[ii] = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          fscanf(fp, "%d", &kk);
          if (kk != ii+1)
          {
            printOutTS(PL_ERROR,
             "MCMC_BF ERROR: 1st index in indexFile = %d (must be %d]).\n",
              kk, ii+1);
            printOutTS(PL_ERROR,"  Data format should be: \n");
            printOutTS(PL_ERROR,
             "  line 1: nInputs in rs data (driver) file\n");
            printOutTS(PL_ERROR,
             "  line 2: 1 <1 or 0> <default value if 0, -1 if sample>\n");
            printOutTS(PL_ERROR,
             "  line 3: 2 <2 or 0> <0 if it is == 2>\n");
            printOutTS(PL_ERROR,
             "  line 4: 3 <3 or 0> <default value if 0, -1 if sample>\n");
            printOutTS(PL_ERROR,
             "  line 5: 4 <4 or 0> <0 if it is == 4>\n");
            printOutTS(PL_ERROR,"  ...\n");
            fclose(fp);
            return PSUADE_UNDEFINED;
          }
          fscanf(fp, "%d", &kk);
          vecRSIndices[ii] = kk;

          if (vecRSIndices[ii] == 999)
          {
            printOutTS(PL_ERROR,
                "MCMC_BF INFO: input %3d has a sample ==> switch to bf2.\n",
                ii+1);
            fclose(fp);
            return (analyze_bf2(adata));
          }
  
          if (vecRSIndices[ii] < 0 || vecRSIndices[ii] > nInputs)
          {
            printOutTS(PL_ERROR,
                "MCMC_BF ERROR: input %3d = %d invalid\n",ii+1,
                vecRSIndices[ii]);
            fclose(fp);
            return PSUADE_UNDEFINED;
          }
          vecRSIndices[ii]--;
          fscanf(fp, "%lg", &ddata);
          vecRSValues[ii] = ddata;
        }
        fclose(fp);
      }
    }
  }
  else if (mode_ == 0)
  {
    printOutTS(PL_INFO,
      "MCMC_BF INFO: since ioPtr=NULL, assume MARS as reponse surface.\n");
    faType = PSUADE_RS_MARS;
  }
  int    *rsIndices = vecRSIndices.getIVector();
  double *rsValues  = vecRSValues.getDVector();

  //**/ ---------------------------------------------------------------
  // get experimental data information from spec file ==> vecDesignP
  // .. matExpInps,matExpMeans,matExpStdvs
  //**/ ---------------------------------------------------------------
  int       dnReps=1;
  psIVector vecDesignP;
  psMatrix  matExpInps, matExpMeans, matExpStdvs;
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  int dnSamples = matExpMeans.nrows();
  int dnInputs  = matExpInps.ncols();
  if (dstatus != 0.0)
  {
    printf("MCMC_BF ERROR: fail to read experimental data file.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ------------------------------------------------------------
  //**/ option to add response surface uncertainties to data std dev
  //**/ ==> rsErrFlag
  //**/ ------------------------------------------------------------
  int rsErrFlag=0;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,
        "*** OPTION TO INCLUDE RESPONSE SURFACE UNCERTAINTIES:\n");
    printOutTS(PL_INFO,
        "\nTo incorporate response surface uncertainties into the\n");
    printOutTS(PL_INFO,
        "likelihood function, make sure stochastic response surfaces\n");
    printOutTS(PL_INFO,
        "are used (GP/Kriging, polynomial regression, or bootstrapped\n");
    printOutTS(PL_INFO,
        "methods). Otherwise, no RS uncertainties will be included.\n");
    printOutTS(PL_INFO,
        "NOTE: if you don't know what this is, just say no below.\n");
    printf( "===> Include response surface uncertainties? (y or n) ");
    scanf("%s", pString);
    fgets(lineIn,1000,stdin);
    if (pString[0] == 'y') rsErrFlag = 1;
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking for rsIndices and vecDesignP
  //**/ (e.g. fixed inputs should not be design parameters)
  //**/ ---------------------------------------------------------------
  if (rsIndices != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      if (rsIndices[ii] == -1)
        printOutTS(PL_INFO,"MCMC_BF INFO: input %3d inactive\n",ii+1);

      if (rsIndices[ii] == -1 && (vecDesignP.length() > 0 && 
          vecDesignP[ii] == 1))
      {
        printOutTS(PL_ERROR,
          "MCMC_BF ERROR: fixed input %d cannot be design parameter\n",
          ii+1);
        return PSUADE_UNDEFINED;
      }

      if (rsIndices[ii] == 999 && 
          (vecDesignP.length() > 0 && vecDesignP[ii] == 1))
      {
        printOutTS(PL_ERROR,
          "MCMC_BF ERROR: fixed input %d cannot be uncertain parameter\n",
          ii+1);
        return PSUADE_UNDEFINED;
      }

      if (rsIndices[ii] < -1 || rsIndices[ii] > nInputs)
      {
        printOutTS(PL_ERROR,
             "MCMC_BF INFO: input %3d = %d invalid\n",ii+1,rsIndices[ii]);
        return PSUADE_UNDEFINED;
      }
    }

    printOutTS(PL_INFO, "Response surface index information: \n");
    for (ii = 0; ii < nInputs; ii++)
    {
      if (rsIndices[ii] == -1)
        printOutTS(PL_INFO, "Input %4d: fixed at default value  = %e\n",
                   ii+1, rsValues[ii]);
      else if (rsIndices[ii] >= 1000)
        printOutTS(PL_INFO, "Input %4d: uncertain, sample index = %4d\n",
                   ii+1, rsIndices[ii]-999);
      else if (vecDesignP.length() > 0 && vecDesignP[ii] == 1)
        printOutTS(PL_INFO, "Input %4d: design parameter\n", ii+1);
      else
        printOutTS(PL_INFO, "Input %4d: calibration parameter\n",ii+1);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ ==> faPtrs
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO,
       "MCMC_BF INFO: CREATING RESPONSE SURFACES FOR ALL OUTPUTS.\n");
  psVector vecYY;
  FuncApprox **faPtrs = new FuncApprox*[nOutputs];
  //**/ if sample has already been loaded
  if (nSamples > 0)
  {
    vecYY.setLength(nSamples);
    for (ii = 0; ii < nOutputs; ii++)
    {
      faType = -1;
      printOutTS(PL_INFO,
           "MCMC_BF INFO: CREATING RESPONSE SURFACE FOR OUTPUT %d.\n",ii+1);
      faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
      faPtrs[ii]->setNPtsPerDim(16);
      faPtrs[ii]->setBounds(lower, upper);
      faPtrs[ii]->setOutputLevel(0);
      for (kk = 0; kk < nSamples; kk++) vecYY[kk] = YIn[kk*nOutputs+ii];

      status = faPtrs[ii]->initialize(XIn, vecYY.getDVector());
      if (status != 0)
      {
        printOutTS(PL_ERROR,
             "MCMC_BF ERROR: Unable to create response surface.\n");
        printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
    }
  }
  //**/ if sample has not been loaded, ask for sample file
  else
  {
    int nInps, nSamps, nOuts;
    double *XX, *YY, *tlower, *tupper;
    PsuadeData *ioOutFile;
    printOutTS(PL_INFO,
      "MCMC_BF INFO: A sample has not be passed in for building RS.\n");
    printOutTS(PL_INFO,"              Please enter sample files below,");
    printOutTS(PL_INFO," one for each output.\n");
    for (ii = 0; ii < nOutputs; ii++)
    {
      faType = -1;
      snprintf(pString,100,
        "Enter file name (PSUADE format) for output %d : ", ii+1);
      getString(pString, lineIn);
      kk = strlen(lineIn);
      lineIn[kk-1] = '\0';
      ioOutFile = new PsuadeData;
      status = ioOutFile->readPsuadeFile(lineIn);
      ioOutFile->getParameter("input_ninputs", pPtr);
      nInps = pPtr.intData_;
      ioOutFile->getParameter("output_noutputs", pPtr);
      nOuts = pPtr.intData_;
      if (nInps != nInputs)
      {
        printOutTS(PL_ERROR,
         "MCMC_BF ERROR: Unable to create RS for output %d\n",
         ii+1);
        printOutTS(PL_ERROR,"            due to nInputs mismatch.\n");
        printOutTS(PL_ERROR,
             "            nInputs expected = %d\n",nInputs);
        printOutTS(PL_ERROR,
             "            nInputs provided = %d\n",nInps);
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
      if (nOuts != 1)
      {
        printOutTS(PL_ERROR,
         "MCMC_BF ERROR: Unable to create response surface for output %d\n",
         ii+1);
        printOutTS(PL_ERROR,"            due to nOutputs != 1.\n");
        printOutTS(PL_ERROR,
             "NOTE: Each sample file should have 1 output.\n");
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
      ioOutFile->getParameter("method_nsamples", pPtr);
      nSamps = pPtr.intData_;
      ioOutFile->getParameter("input_sample", pPtr);
      XX = pPtr.dbleArray_;
      pPtr.dbleArray_ = NULL;
      ioOutFile->getParameter("output_sample", pPtr);
      YY = pPtr.dbleArray_;
      pPtr.dbleArray_ = NULL;
      ioOutFile->getParameter("input_lbounds", pPtr);
      tlower = pPtr.dbleArray_;
      pPtr.dbleArray_ = NULL;
      ioOutFile->getParameter("input_ubounds", pPtr);
      tupper = pPtr.dbleArray_;
      pPtr.dbleArray_ = NULL;
      faPtrs[ii] = genFA(faType, nInps, iZero, nSamps);
      faPtrs[ii]->setNPtsPerDim(16);
      faPtrs[ii]->setBounds(tlower, tupper);
      faPtrs[ii]->setOutputLevel(0);
      status = faPtrs[ii]->initialize(XX, YY);
      delete [] YY;
      delete [] XX;
      delete [] tlower;
      delete [] tupper;
      delete ioOutFile;
      if (status != 0)
      {
        printOutTS(PL_ERROR,
             "MCMC_BF ERROR: Unable to create response surface.\n");
        printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ special set up using expert mode
  //**/ ==> nbins, maxSamples, nPlots, vecPlotInds 
  //**/ ---------------------------------------------------------------
  psIVector vecPlotInds;
  int nbins = 20, nPlots, maxSamples = 500000;
  if (nInputs >= 10) maxSamples = 1000000;
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"*** CURRENT DEFAULT PARAMETER SETTINGS : \n\n");
  printOutTS(PL_INFO,"Inference max sample size = %d\n",maxSamples);
  printOutTS(PL_INFO,"Posterior histogram nbins = %d\n",nbins);
  //printOutTS(PL_INFO,
   //    "NOTE: histogram nBins  - resolution of histogram bar graph\n");
  printOutTS(PL_INFO,
   "Turn on ana_expert mode to change some of these default settings.\n\n");

  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    snprintf(pString,100,
            "Enter maximum inference sample size (500000 - 5000000): ");
    maxSamples = getInt(1000, 50000000, pString);
    if (maxSamples < 500000) maxSamples = 500000;
    if (nInputs >= 10 && maxSamples < 1000000) maxSamples = 1000000;
    //snprintf(pString,100,"Enter the number of histogram bins (10 - 25) : ");
    //nbins = getInt(10, 50, pString);
  }
  if (psConfig_.MasterModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
        "*** OPTION TO CREATE POSTERIOR PLOTS FOR SELECTED INPUTS:\n\n");
    printOutTS(PL_INFO,
        "MCMC will create MATLAB files for the posterior distributions.\n");
    printOutTS(PL_INFO,
        "You can choose to generate posterior plots for all inputs, or \n");
    printOutTS(PL_INFO,
        "just a selected few (in case there are too many inputs).\n");
    printf("Select inputs for which posterior plots are to be generated.\n");
    snprintf(pString,100,"Enter input number (-1 for all, 0 to terminate) : ");
    kk = 1;
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    while (kk != 0 || nPlots < 1)
    {
      kk = getInt(-1, nInputs, pString);
      if (kk == -1)
      {
        nPlots = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          if (rsIndices == NULL || rsIndices[ii] >= 0)
            if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
              vecPlotInds[nPlots++] = ii;
        }
        break;
      }
      if (kk != 0 && nPlots < nInputs)
      {
        if (rsIndices != NULL && rsIndices[kk-1] < 0)
          printOutTS(PL_ERROR,
              "Input %d has been fixed by the rs index file (no plot).\n",
              kk+1);
        else if (vecDesignP.length() > 0 && vecDesignP[kk-1] == 1)
          printOutTS(PL_ERROR,
              "Input %d is a design parameter (no plot)\n",kk);
        else
          vecPlotInds[nPlots++] = kk - 1;
      }
      if (kk == 0 && nPlots == 0)
        printOutTS(PL_ERROR,
            "You need to set at least 1 input for plotting posteriors.\n");
    }
    if (nPlots > 1) sortIntList(nPlots, vecPlotInds.getIVector());
  }
  else
  {
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    for (ii = 0; ii < nInputs; ii++)
      if (rsIndices == NULL || rsIndices[ii] >= 0)
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
          vecPlotInds[nPlots++] = ii;
  }
  printOutTS(PL_INFO,
    "MCMC_BF Plot summary: input number to be plotted are (%d):\n",nPlots);
  for (ii = 0; ii < nPlots; ii++)
    printOutTS(PL_INFO, "   Input %4d\n", vecPlotInds[ii]+1);

  //**/ ---------------------------------------------------------------
  // option to add discrepancy function and a posterior sample
  // ==> modelFormFlag, vecModelForms, modelFormConst, genPosteriors
  //**/ ---------------------------------------------------------------
  int modelFormConst = 0, genPosteriors=1, modelFormFlag=0;
  psIVector vecModelForms;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** OPTION TO ADD A DISCREPANCY FUNCTION:\n\n");
    printOutTS(PL_INFO,
         "To use this feature, first make sure that the observation\n");
    printOutTS(PL_INFO,
         "data file specified earlier has design parameters specified\n");
    printOutTS(PL_INFO,
         "since the discrepancy function is to be a function of these\n");
    printOutTS(PL_INFO,
         "design parameters (if not, a constant discrepancy function\n");
    printOutTS(PL_INFO,"is to be created).\n");
    printOutTS(PL_INFO,
         "NOTE: if you don't know what this is, just say NO.\n");
    snprintf(pString,100,"===> Add discrepancy function ? (y or n) ");
    getString(pString, lineIn);
    if (lineIn[0] == 'y') modelFormFlag = 1;
    if (modelFormFlag == 1)
    {
      snprintf(pString,100,
        "===> Add discrepancy to (1) all or (2) selected outputs ? ");
      kk = getInt(1,2,pString);
      if (kk == 1)
      {
        vecModelForms.setLength(nOutputs);
        vecModelForms.setConstant(iOne);
      }
      else
      {
        vecModelForms.setLength(nOutputs);
        for (ii = 0; ii < nOutputs; ii++)
        {
          snprintf(pString,100,"Add discrepancy to output %d ? (y or n) ",
                  ii+1);
          getString(pString, lineIn);
          if (lineIn[0] == 'y') vecModelForms[ii] = 1;
        }
      }
    }
    if (modelFormFlag == 1 && dnInputs == 0)
    {
      printOutTS(PL_INFO,
       "NOTE: No design inputs ==> discrepancy will be a constant function.\n");
      modelFormConst = 1;
    }
    if (modelFormFlag == 1 && dnSamples == 1)
    {
      printOutTS(PL_INFO,
       "NOTE: 1 experiment ==> discrepancy will be a constant function.\n");
      modelFormConst = 1;
    }

    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
       "*** OPTION TO CREATE A SAMPLE FROM THE POSTERIOR DISTRIBUTIONS:\n\n");
    printOutTS(PL_INFO,
       "In addition to generating the posterior distributions, you can\n");
    printOutTS(PL_INFO,
       "also draw a sample from these posteriors. The posterior sample\n");
    printOutTS(PL_INFO,
       "can be used as prior sample for another simulator/emulator.\n");
    printOutTS(PL_INFO,
       "NOTE: if you don't know what this is, just say no.\n");
    //**/ June 2022 : just turn this on
    //snprintf(pString,100,
    //   "==> Create posterior sample for the input parameters? (y/n) ");
    //getString(pString, lineIn);
    //if (lineIn[0] == 'y') genPosteriors = 1;
  }
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy sample, if desired 
  //**/ ==> ExpSamOutputs, discFuncConstants
  //**/ ---------------------------------------------------------------
  psVector vecDiscFuncConstMeans, vecDiscFuncConstStds, vecDiscSamOuts;
  double *discSamOuts=NULL, *discFuncConstantMeans=NULL;
  double *discFuncConstantStds = NULL;
  if (modelFormFlag == 1)
  {
    psIVector vecDParams;
    psMatrix matSamInps, matSamMeans;

    if (vecDesignP.length() == 0) 
         vecDParams.setLength(nInputs);
    else vecDParams.load(nInputs, vecDesignP.getIVector());
    if (dnInputs > 0)
    {
      matSamInps.setDim(dnSamples,dnInputs);
      for (ii = 0; ii < dnSamples; ii++)
        for (kk = 0; kk < dnInputs; kk++)
          matSamInps.setEntry(ii,kk, matExpInps.getEntry(ii,kk));
    }
    matSamMeans.setDim(dnSamples,nOutputs);
    for (ii = 0; ii < dnSamples; ii++)
      for (kk = 0; kk < nOutputs; kk++)
        matSamMeans.setEntry(ii,kk, matExpMeans.getEntry(ii,kk));
    dstatus = createDiscrepancyFunctions(nInputs,nOutputs,lower,upper,
                vecRSIndices,vecRSValues,vecDParams,dnInputs,dnSamples,
                matSamInps,matSamMeans,dataPtr,vecDiscFuncConstMeans,
                vecDiscFuncConstStds,vecDiscSamOuts,faPtrs,
                printLevel,modelFormConst);
    if (dstatus < 0) return PSUADE_UNDEFINED;
    
    discSamOuts = vecDiscSamOuts.getDVector();
    discFuncConstantMeans = vecDiscFuncConstMeans.getDVector();
    discFuncConstantStds = vecDiscFuncConstStds.getDVector();
  }

  //**/ ---------------------------------------------------------------
  //  set up for inference
  //**/ ---------------------------------------------------------------
  //**/ storage allocation
  psVector vecXmax, vecRange;
  vecRange.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    vecRange[ii] = 1.0 / (upper[ii]-lower[ii]);
  vecXmax.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecXmax[ii] = 0;
  vecMeans_.setLength(nInputs_);
  vecSigmas_.setLength(nInputs_);
  vecMostLikelyInputs_.setLength(nInputs_);
  vecMostLikelyOutputs_.setLength(nOutputs_);

  //**/ bins for prior
  psIMatrix matPBins;
  matPBins.setFormat(PS_MAT2D);
  matPBins.setDim(nbins, nInputs);
  int **pbins = matPBins.getIMatrix2D();
  int ****pbins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    pbins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      pbins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        pbins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          pbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ bins for posterior
  psIMatrix matBins;
  matBins.setFormat(PS_MAT2D);
  matBins.setDim(nbins, nInputs);
  int **bins = matBins.getIMatrix2D();
  int ****bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  //**/ posterior bins need to be double which is turned into integer
  psMatrix matDBins;
  matDBins.setFormat(PS_MAT2D);
  matDBins.setDim(nbins, nInputs);
  double **dbins = matDBins.getMatrix2D();
  double ****dbins2 = new double***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    dbins2[jj] = new double**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      dbins2[jj][jj2] = new double*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        dbins2[jj][jj2][ii] = new double[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          dbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  int samInc = 10000;
  if (samInc * dnSamples > 100000) samInc = 100000 / dnSamples; 
  maxSamples = maxSamples / samInc;
  maxSamples = maxSamples * samInc;

  //**/ generate a large sample ==> inferenceSample
  int      methodSave;
  Sampling *sampler;
  psVector vecLB, vecUB, vecXXX, vecYYY; 
  psIVector vecSSS;
  if (noPDF == 1)
  {
    printf("MCMC_BF INFO: no PDF, use uniform for priors.\n");
    sampler = SamplingCreateFromID(PSUADE_SAMP_MC);
    sampler->setInputBounds(nInputs, lower, upper);
    sampler->setOutputParams(1);
    sampler->setSamplingParams(maxSamples, 1, 1);
    sampler->initialize(0);
    vecXXX.setLength(maxSamples*nInputs);
    vecSSS.setLength(maxSamples);
    vecYYY.setLength(maxSamples);
    sampler->getSamples(maxSamples, nInputs, 1, vecXXX.getDVector(), 
                        vecYYY.getDVector(), vecSSS.getIVector());
    delete sampler;
  }
  else
  {
    printf("MCMC_BF INFO: has PDF, draw sample from distribution.\n");
    dataPtr->getParameter("method_sampling", pPtr);
    methodSave = pPtr.intData_;
    if (nInputs > 50)
         dataPtr->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
    else dataPtr->updateMethodSection(PSUADE_SAMP_LPTAU,-1,-1,-1,-1);
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(dataPtr);
    vecLB.load(nInputs, lower);
    vecUB.load(nInputs, upper);
    vecXXX.setLength(maxSamples*nInputs);
    pdfman->genSample(maxSamples, vecXXX, vecLB, vecUB);
    dataPtr->updateMethodSection(methodSave,-1,-1,-1,-1);
    delete pdfman;
  }

  //**/ ---------------------------------------------------------------
  //  perform inference
  //**/ ---------------------------------------------------------------
  //**/ setting things up
  psVector vecXSam,vecYSam,vecXDes,vecYDes,vecYSamStd,vecYDesStd;
  psVector vecInferOut, vecTempOut, vecTmpMeans, vecTmpStds;
  vecXSam.setLength(samInc*dnSamples*nInputs);
  vecYSam.setLength(samInc*dnSamples*nOutputs);
  vecYSamStd.setLength(samInc*dnSamples*nOutputs);
  vecXDes.setLength(samInc*dnSamples*nInputs);
  vecYDes.setLength(samInc*dnSamples*nOutputs);
  vecYDesStd.setLength(samInc*dnSamples*nOutputs);
  vecInferOut.setLength(maxSamples);
  vecTempOut.setLength(maxSamples);
  double *XSample = vecXSam.getDVector();
  double *YSample = vecYSam.getDVector();
  double *YSamStd = vecYSamStd.getDVector();
  double *XDesign = vecXDes.getDVector();
  double *YDesign = vecYDes.getDVector();
  double *YDesStd = vecYDesStd.getDVector();
  double *inferenceSamOut = vecInferOut.getDVector();
  double *inferenceTmpOut = vecTempOut.getDVector();
  double *inferenceSamIns = vecXXX.getDVector();
  double stdev, stdv2, YT, YT1, YT2;
  int    dcnt, ss = 0, index, printStep=0, twiceFlag=0;
  int    nInpsActive = 0, passCnt;
  double Ymin = PSUADE_UNDEFINED, dd;
  double Ymax = -PSUADE_UNDEFINED;
  vecTmpMeans.setLength(100*nInputs);
  vecTmpStds.setLength(100*nInputs);
  double *tmpMeans = vecTmpMeans.getDVector();
  double *tmpStds  = vecTmpStds.getDVector();

  //**/ find number of active inputs
  fp = NULL;
  for (ii2 = 0; ii2 < nInputs; ii2++)
  {
    if ((rsIndices == NULL || (rsIndices != NULL && rsIndices[ii2] >=0)) &&
        (vecDesignP.length() == 0 || vecDesignP[ii2] == 0)) nInpsActive++;
  }

  //**/ run the samples
  printOutTS(PL_INFO, "MCMC_BF Inference begins ... \n");
  fflush(stdout);
  while (ss < maxSamples)
  {
    cnt = (ss+1) / (maxSamples/20);
    if (cnt != printStep)
    {
      printOutTS(PL_INFO, "%3.0f%% ",5.0*(printStep+1) );
      fflush(stdout);
    }

    //**/ prepare the current sample 
    //**/ ==> XSample, XDesign of size (samInc*dnSamples)
    for (jj = 0; jj < samInc; jj++)
    {
      index = jj * dnSamples;
      for (kk2 = 0; kk2 < dnSamples; kk2++)
      {
        dcnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++)
        {
          XSample[(index+kk2)*nInputs+ii2] = 
                    inferenceSamIns[(ss+jj)*nInputs+ii2]; 
          if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
          {
             XSample[(index+kk2)*nInputs+ii2] =
                                   matExpInps.getEntry(kk2,dcnt);
             XDesign[(index+kk2)*dnInputs+dcnt] =
                                   matExpInps.getEntry(kk2,dcnt);
             dcnt++;
          }
          if (rsIndices != NULL && rsIndices[ii2] < 0) 
            XSample[(index+kk2)*nInputs+ii2] = rsValues[ii2];
        }
      }
      //**/ set design outputs and stds (discrepancy) to zero 
      //**/ since they will be used but may not be set later
      for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++)
      {
        YDesign[index*nOutputs+ii2] =
                YDesStd[index*nOutputs+ii2] = 0.0;
        YSample[index*nOutputs+ii2] =
                YSamStd[index*nOutputs+ii2] = 0.0;
      }
    }
    //**/ run XGuessS and XDesign through response surfaces
    //**/ ==> YDesign, YDesStd, YSample, YSamStd
    for (ii2 = 0; ii2 < nOutputs; ii2++)
    {
      //**/ case 1: if RS error is requested
      if (rsErrFlag == 1)
      {
        faPtrs[ii2]->evaluatePointFuzzy(samInc*dnSamples, 
                        XSample,&YSample[ii2*dnSamples*samInc],
                        &YSamStd[ii2*dnSamples*samInc]);
        //**/ add discrepancy, if available
        if (modelFormFlag == 1)
        {
          if (modelFormConst == 0)
          {
            for (jj = 0; jj < samInc; jj++)
            {
              index = jj * dnSamples;
              for (kk2 = 0; kk2 < dnSamples; kk2++)
              {
                YDesign[ii2*dnSamples*samInc+index+kk2] =
                             discSamOuts[ii2*dnSamples+kk2];
                YDesStd[ii2*dnSamples*samInc+index+kk2] = 0.0;
              }
            }
          }
          else if (discFuncConstantMeans != NULL &&
                   discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
          {
            for (kk2 = 0; kk2 < dnSamples*samInc; kk2++)
            {
              YDesign[ii2*dnSamples*samInc+kk2] =
                             discFuncConstantMeans[ii2];
              YDesStd[ii2*dnSamples*samInc+kk2] = 0.0;
            }
          }
          else
          {
            for (kk2 = 0; kk2 < dnSamples*samInc; kk2++)
            {
              YDesign[ii2*dnSamples*samInc+kk2] = 0;
              YDesStd[ii2*dnSamples*samInc+kk2] = 0;
            }
          }
        }
      }
      //**/ case 2: if RS error is to be turned off
      else
      {
        faPtrs[ii2]->evaluatePoint(samInc*dnSamples,XSample,
                                   &YSample[ii2*dnSamples*samInc]);
        //**/ add discrepancy function, if available
        if (modelFormFlag == 1)
        {
          if (modelFormConst == 0)
          {
            for (jj = 0; jj < samInc; jj++)
            {
              index = jj * dnSamples;
              for (kk2 = 0; kk2 < dnSamples; kk2++)
                YDesign[ii2*dnSamples*samInc+index+kk2] =
                             discSamOuts[ii2*dnSamples+kk2];
            }
          }
          else if (discFuncConstantMeans != NULL &&
                   discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
          {
            for (kk2 = 0; kk2 < dnSamples*samInc; kk2++)
              YDesign[ii2*dnSamples*samInc+kk2] =
                                     discFuncConstantMeans[ii2];
          }
          for (kk2 = 0; kk2 < dnSamples*samInc; kk2++)
            YSamStd[ii2*dnSamples*samInc+kk2] =
                       YDesStd[ii2*dnSamples*samInc+kk2] = 0.0;
        }
      }
    }
    //**/ compute vecXDist
    for (jj = 0; jj < samInc; jj++)
    {
      inferenceSamOut[ss+jj] = 0.0;
      index = jj * dnSamples;
      for (ii2 = 0; ii2 < nOutputs; ii2++)
      {
        for (kk2 = 0; kk2 < dnSamples; kk2++)
        {
          if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
          {
            YT1 = YSample[ii2*dnSamples*samInc+index+kk2] +
                  YDesign[ii2*dnSamples*samInc+index+kk2];
            stdv2 = YDesStd[ii2*dnSamples*samInc+index+kk2];
          }
          else
          {
            YT1 = YSample[ii2*dnSamples*samInc+index+kk2];
            stdv2 = 0;
          }
          stdev = YSamStd[ii2*dnSamples*samInc+index+kk2];
          YT2 = pow((YT1-matExpMeans.getEntry(kk2,ii2)),2.0) /
               (pow(matExpStdvs.getEntry(kk2,ii2),2.0) +
                stdev*stdev + stdv2*stdv2);
          inferenceSamOut[ss+jj] += YT2;
        }
      }
      //**/ more correct?
      //*/ inferenceSamOut[ss+jj] /= (dnSamples*nOutputs);
      inferenceSamOut[ss+jj] /= (double) dnReps;
    }
    //**/ at this point InferenceSamOut[jj] has been computed
    ss += samInc;

    //**/ print interim information
    if (cnt != printStep)
    {
      printStep++;
      for (jj = 0; jj < ss; jj++) 
        inferenceTmpOut[jj] = inferenceSamOut[jj];
      Ymin = PSUADE_UNDEFINED;
      for (jj = 0; jj < ss; jj++) 
      {
        dd = inferenceTmpOut[jj];
        if (dd < Ymin) Ymin = dd;
        if (dd > Ymax) Ymax = dd;
      }
      for (jj = 0; jj < ss; jj++) 
        inferenceTmpOut[jj] = exp(-0.5*(inferenceTmpOut[jj]-Ymin));

      printEquals(PL_INFO, 0);
      passCnt = 0;
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        if ((rsIndices == NULL || 
            (rsIndices != NULL && rsIndices[ii2] >=0)) &&
            (vecDesignP.length() == 0 || vecDesignP[ii2] == 0))
        {
          YT = 0.0;
          for (jj = 0; jj < ss; jj++) YT += inferenceTmpOut[jj];
          YT1 = 0.0;
          //**/ input times likelihood
          for (jj = 0; jj < ss; jj++)
            YT1 += inferenceSamIns[jj*nInputs+ii2] * 
                   inferenceTmpOut[jj] / YT; 
          printOutTS(PL_INFO,"MCMC_BF: input %3d mean    = %12.5e\n", 
                     ii2+1, YT1);
          YT2 = 0.0;
          for (jj = 0; jj < ss; jj++)
            YT2 += pow(inferenceSamIns[jj*nInputs+ii2]-YT1,2.0)*
                       inferenceTmpOut[jj]/YT; 
          printOutTS(PL_INFO,"MCMC_BF: input %3d std dev = %12.5e\n",
                     ii2+1,sqrt(YT2));
          tmpMeans[(printStep-1)+ii2*100] = YT1; 
          tmpStds[(printStep-1)+ii2*100] = sqrt(YT2); 
          if (printStep > 2) 
          {
            YT1 = 1.05;
            YT2 = checkConvergence(3,&tmpMeans[ii2*100+printStep-3],
                         &tmpStds[ii2*100+printStep-3],ss-samInc,YT1);
            if (YT2 < 1.05)
            {
              passCnt++;
              printf("MCMC_BF input %3d converged.\n",ii2+1);
            }
          }
        }
      }
      printEquals(PL_INFO, 0);
      if (passCnt == nInpsActive)
      {
        if      (twiceFlag <= 0) twiceFlag++;
        else if (twiceFlag == 1) maxSamples = ss;
      }
    }

    //**/ stop if requested
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      printOutTS(PL_ERROR,
           "MCMC_BF INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
      fclose(fp);
      fp = NULL;
      strcpy(pString, "psuade_stop");
      unlink(pString);
      maxSamples = ss;
    }
  }

  //**/ normalize inferenceSamOut so that they are not too small
  //**/ (by using Ymin)
  Ymin = PSUADE_UNDEFINED;
  Ymax = -PSUADE_UNDEFINED;
  for (ss = 0; ss < maxSamples; ss++)
  {
    dd = inferenceSamOut[ss];
    if (dd < Ymin) Ymin = dd;
    if (dd > Ymax) Ymax = dd;
  }
  for (ss = 0; ss < maxSamples; ss++)
    inferenceSamOut[ss] = exp(-0.5*(inferenceSamOut[ss]-Ymin));

  //**/ search for the maximum likelihood solution
  //**/ since normalizing factor of the likelihood function is
  //**/ constant, only the exponential term is computed ==> Ymax
  index = -1;
  Ymax = -PSUADE_UNDEFINED;
  for (ss = 0; ss < maxSamples; ss++)
  {
    dd = inferenceSamOut[ss];
    if (dd > Ymax)
    {
      Ymax = dd;
      index = ss;
    }
  } 
  if (index >= 0)
  {
    for (ii = 0; ii < nInputs; ii++) 
      vecXmax[ii] = inferenceSamIns[index*nInputs+ii];
  }
  printAsterisks(PL_INFO, 0);
  printf("Maximum likelihood estimated solution:\n");
  printEquals(PL_INFO, 0);
  for (ii = 0; ii < nInputs; ii++) 
    printf("Input %3d = %16.8e\n",ii+1,vecXmax[ii]);
  printf("MCMC_BF: MLE Negative log likelihood = %11.4e\n", 
         2*log(Ymax)+Ymin);

  //**/ now re-scale inferenceSamOut to control its magnitude
  //**/ in subsequent processing (binning)
  if (Ymax > 0)
  {
    //Ymax = Ymax / exp(-0.5*Ymin);
    for (ss = 0; ss < maxSamples; ss++) inferenceSamOut[ss] /= Ymax;
  }

  //**/ update the 1D and 2D histogram
  int    ii3, index2;
  for (ss = 0; ss < maxSamples; ss++)
  {
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      YT1 = (inferenceSamIns[ss*nInputs+ii2]-lower[ii2])*vecRange[ii2];
      index = (int) (YT1 * nbins);
      if (index > nbins)
        printOutTS(PL_ERROR,
             "MCMC_BF binning error 1 in file %s, line %d.\n",
             __FILE__, __LINE__);
      if (index < 0)
        printOutTS(PL_ERROR,
             "MCMC_BF binning error 2 in file %s, line %d.\n",
             __FILE__, __LINE__);
      if (index >= nbins) index = nbins - 1;
      if (index <  0)     index = 0;
      dbins[index][ii2] += inferenceSamOut[ss];
      pbins[index][ii2]++;
    }
    //**/ update the 2D histogram
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      YT1 = (inferenceSamIns[ss*nInputs+ii2] - lower[ii2])*vecRange[ii2];
      index = (int) (YT1 * nbins);
      if (index >= nbins) index = nbins - 1;
      if (index < 0)      index = 0;
      for (ii3 = 0; ii3 < nInputs; ii3++)
      {
        YT2 = (inferenceSamIns[ss*nInputs+ii3]-lower[ii3])*vecRange[ii3];
        index2 = (int) (YT2 * nbins);
        if (index2 >= nbins) index2 = nbins - 1;
        if (index2 < 0)      index2 = 0;
        dbins2[index][index2][ii2][ii3] += inferenceSamOut[ss];
        pbins2[index][index2][ii2][ii3]++;
      }
    }
  }

  //**/ continue with binning
  Ymax = 0;
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      if (dbins[kk][ii2] > Ymax) Ymax = dbins[kk][ii2];
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      dbins[kk][ii2] = dbins[kk][ii2] / Ymax * maxSamples;
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) bins[kk][ii2] = (int) dbins[kk][ii2];
  Ymax = 0;
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          if (dbins2[jj][kk][ii2][ii3] > Ymax) 
            Ymax = dbins2[jj][kk][ii2][ii3];
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          dbins2[jj][kk][ii2][ii3] = dbins2[jj][kk][ii2][ii3] / Ymax *
                                     maxSamples;
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          bins2[jj][kk][ii2][ii3] = (int) dbins2[jj][kk][ii2][ii3];
     
  //**/ output statistics
  double psum;
  for (ii = 0; ii < nInputs; ii++)
  {
    if ((rsIndices == NULL || (rsIndices != NULL && rsIndices[ii] >=0)) &&
        (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
    {
      printOutTS(PL_INFO,
                 "MCMC_BF: input %3d value at maximum likelihood = %e\n",
                 ii+1, vecXmax[ii]);
      psum = 0.0;
      for (ss = 0; ss < maxSamples; ss++) psum += inferenceSamOut[ss];
      YT = 0.0;
      //**/ input times likelihood
      for (ss = 0; ss < maxSamples; ss++)
        YT += inferenceSamIns[ss*nInputs+ii] * inferenceSamOut[ss] / psum; 
      vecMeans_[ii] = YT;
      printOutTS(PL_INFO,"MCMC_BF: input %3d mean    = %e\n", ii+1, YT);
      YT = 0.0;
      for (ss = 0; ss < maxSamples; ss++)
        YT += pow(inferenceSamIns[ss*nInputs+ii]-vecMeans_[ii],2.0)*
                  inferenceSamOut[ss]/psum; 
      vecSigmas_[ii] = sqrt(YT);
      printOutTS(PL_INFO,"MCMC_BF: input %3d std dev = %e\n",
                 ii+1,vecSigmas_[ii]);
      vecMostLikelyInputs_[ii] = vecXmax[ii];
    }
  }
  printAsterisks(PL_INFO, 0);
 
  //**/ generate matlabmcmc2.m file at every major iteratin
  for (ii = 0; ii < nInputs; ii++) vecRange[ii] = 1.0 / vecRange[ii];
  genMatlabFile(nInputs,lower,upper,vecRange.getDVector(),nPlots,
        vecPlotInds.getIVector(),nbins,pbins,pbins2,bins,bins2,qData,
        0,0,NULL,NULL, vecXmax.getDVector(),Ymin);

  //**/ ---------------------------------------------------------------
  //**/  generate the posterior sample file
  //**/ ---------------------------------------------------------------
  if (genPosteriors == 1)
  {
    int    maxPostSam=50000;
    double dmax = 0.0;
    for (ss = 0; ss < maxSamples; ss++) 
      if (inferenceSamOut[ss] > dmax) dmax = inferenceSamOut[ss];
    if (dmax == 0)
    {
      printf("MCMC_BF: ERROR encountered when generating posteriors.\n");
      fp = NULL;
    }
    else 
    {
      fp = fopen("MCMCPostSample", "w");
      for (ss = 0; ss < maxSamples; ss++) 
        inferenceSamOut[ss] = inferenceSamOut[ss] / dmax;
    }
    if (fp != NULL)
    {
#if 1
      //**/ Sept 29, 2022: mathematical justification of the 
      //**/          following procedure to extract a sub-sample
      //**/          from the inference sample
      //**/ mean = sum_{i=1}^M p_i X_i 
      //**/ * divide the sample into N quantiles by probabilities
      //**/      = sum_{i=1}^K [sum_{j=1}^{M_i} p_{ij} X_{ij}] 
      //**/      = sum_{i=1}^K [p_{i1} X_{i1} + ...+ p_{iM_i} X_{iM_i}]
      //**/ * approximate all p_{ij} with averge mu(p_{i})
      //**/      ~ sum_{i=1}^K mu(p_{i}) (X_{i1} + ...+ X_{iM_i})
      //**/ * expand and sum re-arrangement
      //**/      = sum_{i=1}^K [sum_{j=1}^{M_i} p_{ij}] mu_{X_{i}}
      //**/      = sum_{i=1}^K P_i mu_{X_{i}}
      //**/   where P_i = sum_{j=1}^{M_i} p_{ij}
      //**/ * select a sub-sample from the quantile
      //**/      ~ sum_{i=1}^K P_i X{i*}
      //**/   where X_{i*} is a randomy sample from quantile i
      //**/ * since sum of P_i = 1, P_i gives the likelihood of 
      //**/   the selected sample
      //**/ * finally scale the total posterior sample size to
      //**/   maxPostSam 

      //**/ create quantile partitions (into K partitions)
      int nQuantiles=100;
      psVector vecQuanPart;
      vecQuanPart.setLength(nQuantiles+1);
#if 1
      //============================================
      //**/ geometric
      //**/ Sept 2022: geometric is good
      //--------------------------------------------
      double factor=0.9;
      vecQuanPart[nQuantiles] = 1.0;
      for (ii = nQuantiles-1; ii >= 0; ii--)
        vecQuanPart[ii] = vecQuanPart[ii+1] * factor;
#else
      //--------------------------------------------
      //**/ fixed interval
      //--------------------------------------------
      vecQuanPart[0] = 0e-8;
      for (ii = 1; ii <= nQuantiles; ii++)
        vecQuanPart[ii] = 1.0 * ii / nQuantiles;
      //============================================
#endif

      //**/ find out how many samples (M_i) are in each quantile
      //**/ in quantile ii if [vecQuanPart[ii], vecQuanPart[ii+1]]
      //**/ vecQuanSizes - sample sizes for each quantile
      //**/ vecQuanWts - total likelihood for samples in each quantile
      psVector  vecQuanWts;
      psIVector vecQuanSizes;
      vecQuanSizes.setLength(nQuantiles);
      vecQuanWts.setLength(nQuantiles);
      for (ss = 0; ss < maxSamples; ss++)
      {
        ddata = inferenceSamOut[ss];
        for (ii = 0; ii <= nQuantiles; ii++)
          if (ddata <= vecQuanPart[ii]) break;
        if (ii > 0) 
        {
          vecQuanSizes[ii-1]++;
          vecQuanWts[ii-1] += ddata;
        }
      }
      printAsterisks(PL_INFO, 0);
      printf("MCMC_BF: posterior sample generation: \n");
      printf("  * Total no. of sample points   = %d\n",maxSamples);
      printf("  * No. of posterior candidates  = %d (those with Prob>0).\n",
             vecQuanSizes.sum());
      printf("  * No. of points to be selected ~ %d\n", maxPostSam);

      //**/ check quantile using all valid samples (prob>0)
      psVector  vecChkMean, vecChkStds;
      psIVector vecCnts;
      vecChkMean.setLength(nQuantiles);
      vecChkStds.setLength(nQuantiles);
      vecCnts.setLength(nQuantiles);
      for (ss = 0; ss < maxSamples; ss++)
      {
        ddata = inferenceSamOut[ss];
        for (ii = 0; ii <= nQuantiles; ii++)
          if (ddata <= vecQuanPart[ii]) break;
        if (ii > 0) 
        { 
          vecCnts[ii-1]++;
          vecChkMean[ii-1] += inferenceSamIns[ss*nInputs];
        }
      }
      for (ii = 0; ii < nQuantiles; ii++)
        if (vecCnts[ii] > 0) vecChkMean[ii] /= vecCnts[ii];  
      double ddmean = 0;
      for (ii = 0; ii < nQuantiles; ii++)
        ddmean += (vecChkMean[ii] * vecQuanWts[ii]);
      ddmean /= vecQuanWts.sum();
      for (ss = 0; ss < maxSamples; ss++)
      {
        ddata = inferenceSamOut[ss];
        for (ii = 0; ii <= nQuantiles; ii++)
          if (ddata <= vecQuanPart[ii]) break;
        if (ii > 0) 
          vecChkStds[ii-1] += pow(inferenceSamIns[ss*nInputs]-ddmean,2.0);
      }
      for (ii = 0; ii < nQuantiles; ii++)
        if (vecCnts[ii] > 0) vecChkStds[ii] /= vecCnts[ii];  
      double ddstds = 0;
      for (ii = 0; ii < nQuantiles; ii++)
        ddstds += (vecChkStds[ii] * vecQuanWts[ii]);
      ddstds /= vecQuanWts.sum();
      ddstds = sqrt(ddstds);
      //printf(" * If all sample probabilities are quantized and ");
      //printf("all sub-samples in\n");
      //printf("   the same bin has the SAME WEIGHT, the computed\n");
      //printf("          mean and stdv = %12.5e %12.5e\n",
      //       ddmean,ddstds);
      //printDashes(PL_INFO, 0);

      //**/ create an index matrix for all quantiles
      //**/ quanIndices[ii][jj] - sample indices jj for quantile ii
      //**/ This is done because later on random sub-samples are to
      //**/ be drawn from each quantile, and this index matrix will
      //**/ make this process more efficient
      int nMax = vecQuanSizes.max();
      psIMatrix matQuanInds; 
      matQuanInds.setFormat(PS_MAT2D);
      matQuanInds.setDim(nQuantiles, nMax);
      int **quanIndices = matQuanInds.getIMatrix2D(); 
      vecQuanSizes.setLength(nQuantiles);
      for (ss = 0; ss < maxSamples; ss++)
      {
        ddata = inferenceSamOut[ss];
        for (ii = 0; ii <= nQuantiles; ii++)
          if (ddata <= vecQuanPart[ii]) break;
        if (ii > 0) 
        {
          quanIndices[ii-1][vecQuanSizes[ii-1]] = ss;
          vecQuanSizes[ii-1]++;
        }
      }
     
      //**/ figure out how many samples to select from each 
      //**/ quantile (from total likelihood for each quantile)
      psIVector vecQuanMax;
      vecQuanMax.setLength(nQuantiles);
      for (ii = 0; ii < nQuantiles; ii++)
      {
        //**/ sum_pi multiplied by any large integer 
        ddata = vecQuanWts[ii] * 1000; 
        vecQuanMax[ii] = (int) ddata;
      }

      //**/ vecQuanMax.sum may be large, need to rescale the sum
      //**/ to be <= maxPostSam
      ddata = 1.0 * maxPostSam / vecQuanMax.sum();
      for (ii = 0; ii < nQuantiles; ii++)
        vecQuanMax[ii] = (int) (ddata * vecQuanMax[ii]);
      if (vecQuanMax.sum() > maxPostSam)
      {
        printf("MCMC_BF ERROR: Something is wrong. Consult developers.\n");
        printf("        ERROR in file %s, line %d.\n",__FILE__,__LINE__);
        printf("     vecQuanMax.sum, maxPostSam = %d %d\n",vecQuanMax.sum(),
               maxPostSam);
        exit(1);
      } 

      //**/ now select a posterior sample and put it in vecPostInps
      int nPostSam=0,count=0,samInd;
      psVector vecPostInps;
      vecPostInps.setLength(maxPostSam*nInputs);
      psIVector vecIRan;
      vecIRan.setLength(maxSamples);
      count = 0;
      if (printLevel > 2)
      {
        printf(" * Samples are assigned to %d bins\n",nQuantiles);
        printf(" * Sub-samples are selected from each bin : \n");
      }
      for (ss = 0; ss < nQuantiles; ss++)
      {
        //**/ randomize the sample indices for the quantile ss
        if (vecQuanSizes[ss] > 0)
        {
          vecIRan.setLength(vecQuanSizes[ss]);
          generateRandomIvector(vecQuanSizes[ss],vecIRan.getIVector());
        }
        //**/ now fill in the posterior sample array vecPostInps
        int iSave = vecQuanMax[ss]; 
        samInd = 0;
        if (printLevel > 2)
          printf("      Bin %d has %d points (selected from %d)\n",
                 ss+1, vecQuanMax[ss],vecQuanSizes[ss]);
        while (vecQuanMax[ss] > 0 && vecQuanSizes[ss] > 0)
        {
          //**/ randomly pick an element and fetch index kk
          jj = vecIRan[samInd];
          kk = quanIndices[ss][jj];
          //**/ if probability of sample kk is > 0, add to post sample
          ddata = inferenceSamOut[kk];
          if (ddata > 0)
          {
            for (ii = 0; ii < nInputs; ii++)
            {
              if ((rsIndices == NULL || rsIndices[ii] >= 0) &&
                  (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
                vecPostInps[count++] =
                    inferenceSamIns[kk*nInputs+ii];
            }
            vecQuanMax[ss] = vecQuanMax[ss] - 1; 
            nPostSam++;
            vecCnts[ss]++;
            vecChkMean[ss] += inferenceSamIns[kk*nInputs];
          }
          samInd++;
          //**/ if all exhausted, start all over from beginning
          if (samInd >= vecQuanSizes[ss]) samInd = 0;
        }
      }
      //**/ checking the selected posterior sample for mean and std dev
      //**/ first count number of uncertain parameters
      int nUInps = 0;
      for (jj = 0; jj < nInputs; jj++)
        if ((rsIndices == NULL || rsIndices[jj] >= 0) &&
            (vecDesignP.length() == 0 || vecDesignP[jj] == 0)) nUInps++;
      kk = 0;
      printf("  * Posterior sample statistics : \n");
      for (ii = 0; ii < nInputs; ii++)
      {
        if ((rsIndices == NULL || rsIndices[ii] >= 0) &&
            (vecDesignP.length() == 0 || vecDesignP[ii] == 0))
        {
          ddmean = 0;
          for (ss = 0; ss < nPostSam; ss++)
            ddmean += vecPostInps[ss*nUInps+kk];
          ddmean /= (double) nPostSam;
          ddstds = 0;
          for (ss = 0; ss < nPostSam; ss++)
            ddstds += pow(vecPostInps[ss*nUInps+kk]-ddmean,2.0);
          ddstds /= (double) nPostSam;
          ddstds = sqrt(ddstds); 
          printf("      Input %d mean, stds = %12.5e %12.5e\n",  
                 ii+1, ddmean, ddstds);
          kk++;
        }
      }
      printf("  * Check these statistics against MCMC statistics ");
      printf("above. If there is\n");
      printf("    significant discrepancy, some parameters may ");
      printf("need to be tweaked.\n");
      printf("    (Such as number of quantiles and quantile ");
      printf("multiplier - consult\n");
      printf("    PSUADE developers.)\n");
      printDashes(PL_INFO, 0);
#endif

      fprintf(fp, "PSUADE_BEGIN\n");
      //**/ fetch parameter names
      fprintf(fp, "%d %d\n", nPostSam, nUInps);
      if (qData.strArray_ != NULL)
      {
        fprintf(fp, "# ");
        for (jj = 0; jj < nInputs; jj++)
          if ((rsIndices == NULL || rsIndices[jj] >= 0) &&
              (vecDesignP.length() == 0 || vecDesignP[jj] == 0))
            fprintf(fp,"%s ", qData.strArray_[jj]);
        fprintf(fp, "\n");
      }
      //**/ store the subsample
      for (ss = 0; ss < nPostSam; ss++)
      {
        fprintf(fp, "%d ", ss+1);
        for (jj = 0; jj < nUInps; jj++)
          fprintf(fp, "%e ", vecPostInps[ss*nUInps+jj]);
        fprintf(fp, "\n");
      }
      fprintf(fp, "PSUADE_END\n");

      //**/ for the MLE solution, dissect the component of likelihood
      vecXSam.setLength(nInputs*dnSamples);
      vecYSam.setLength(nOutputs*dnSamples);
      vecXDes.setLength(dnInputs*dnSamples);
      vecYDes.setLength(nOutputs*dnSamples);
      YSample = vecYSam.getDVector();
      YDesign = vecYDes.getDVector();
      fprintf(fp, "Optimal parameter values: \n");
      dcnt = 0;
      for (ii2 = 0; ii2 < nInputs; ii2++) 
      {
        if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
        {
          fprintf(fp, "%16.8e\n", matExpInps.getEntry(0,dcnt));
          dcnt++;
        }
        else if (rsIndices != NULL && rsIndices[ii2] < 0)
          fprintf(fp, "%16.8e\n", rsValues[ii2]);
        else
          fprintf(fp, "%16.8e\n", vecXmax[ii2]);
      }
      fprintf(fp,"Best negLogLikelihood = %e (Ideal=0)\n",Ymin);
      fprintf(fp,
       "MLE Statistics (Ypred, Yexp, sd, -loglikelihood, (Ypred-Yexp)/sd)\n");
      for (kk2 = 0; kk2 < dnSamples; kk2++)
      {
        dcnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++) 
        {
          vecXSam[kk2*nInputs+ii2] = vecXmax[ii2];
          if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
          {
            vecXSam[kk2*nInputs+ii2] = matExpInps.getEntry(kk2,dcnt);
            vecXDes[kk2*dnInputs+dcnt] = matExpInps.getEntry(kk2,dcnt);
            dcnt++;
          }
          if (rsIndices != NULL && rsIndices[ii2] < 0)
            vecXSam[kk2*nInputs+ii2] = rsValues[ii2];
        }
      }
      for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++)
        YDesign[ii2] = YSample[ii2] = 0.0;
      for (ii2 = 0; ii2 < nOutputs; ii2++) 
      {
        faPtrs[ii2]->evaluatePoint(dnSamples,vecXSam.getDVector(),
                                   &YSample[ii2*dnSamples]);
        if (modelFormFlag == 1)
        {
          if (modelFormConst == 0)
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesign[ii2*dnSamples+kk2] = discSamOuts[ii2*dnSamples+kk2];
          }
          else if (discFuncConstantMeans != NULL &&
                   discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesign[ii2*dnSamples+kk2] = discFuncConstantMeans[ii2];
          }
          else
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesign[ii2*dnSamples+kk2] = 0;
          }
        }
      }
      for (kk2 = 0; kk2 < dnSamples; kk2++)
      {
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
            YT1 = YSample[ii2*dnSamples+kk2] + YDesign[ii2*dnSamples+kk2];
          else
            YT1 = YSample[ii2*dnSamples+kk2];
          YT2 = matExpMeans.getEntry(kk2,ii2);
          stdev = matExpStdvs.getEntry(kk2,ii2);
          stdv2 = 0.5 * pow(YT1 - YT2, 2.0) / (stdev * stdev);
          fprintf(fp,"%4d %12.4e %12.4e %12.4e %12.4e (%12.4e)\n",
                  kk2+1,YT1,YT2,stdev,stdv2,sqrt(2*stdv2));
        }
      }
      fclose(fp);
      printf("MCMC_BF: Posterior sample is now in 'MCMCPostSample'.\n");
    }
    printAsterisks(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy sample, if desired 
  //**/ ---------------------------------------------------------------
  int  nInps, nOuts, nSams, *states;
  psVector  vecAllOuts;
  psStrings strNames;
  PsuadeData *filePtr1, *filePtr2;

  if (modelFormFlag == 1)
  {
    //**/ this file contains all discrepancy data
    snprintf(pString,100,"psDiscrepancyModel1");
    filePtr1 = new PsuadeData();
    status = filePtr1->readPsuadeFile(pString);
    if (status != 0)
    {
       printOutTS(PL_ERROR,
            "MCMC ERROR: cannot read file %s in PSUADE format.\n",
            pString);
       exit(1);
    }
  }
  if (modelFormFlag == 1 && status == 0)
  {
    //**/ read the first discrepancy file to get nInputs/nSamples
    filePtr1->getParameter("input_ninputs", pPtr);
    nInps = pPtr.intData_;
    filePtr1->getParameter("output_noutputs", pPtr);
    nOuts = pPtr.intData_;
    filePtr1->getParameter("method_nsamples", pPtr);
    nSams = pPtr.intData_;
    filePtr1->getParameter("output_sample", pOutputs);
    unlink(pString);

    //**/ read the first discrepancy output into vecAllOuts
    vecAllOuts.setLength(nOutputs * nSams);
    if (vecModelForms.length() > 0 && vecModelForms[0] == 1)
    {
      for (jj = 0; jj < nSams; jj++)
        vecAllOuts[jj*nOutputs] = pOutputs.dbleArray_[jj];
    }
    pOutputs.clean();

    //**/ read the rest
    strNames.setNumStrings(nOutputs);
    strcpy(pString, "Y1");
    strNames.loadOneString(0, pString);
    for (ii = 1; ii < nOutputs; ii++)
    {
      filePtr2 = new PsuadeData();
      snprintf(pString,100,"psDiscrepancyModel%d", ii+1);
      status = filePtr2->readPsuadeFile(pString);
      if (status != 0) break;
      filePtr2->getParameter("input_ninputs", pPtr);
      if (pPtr.intData_ != nInps) break;
      filePtr2->getParameter("output_noutputs", pPtr);
      if (pPtr.intData_ != nOuts) break;
      filePtr2->getParameter("method_nsamples", pPtr);
      if (pPtr.intData_ != nSams) break;
      filePtr2->getParameter("output_sample", pOutputs);
      delete filePtr2;
      unlink(pString);
      if (vecModelForms.length() > 0 && vecModelForms[ii] == 1)
      {
        for (jj = 0; jj < nSams; jj++)
          vecAllOuts[jj*nOutputs+ii] = pOutputs.dbleArray_[jj];
      }
      pOutputs.clean();
      snprintf(pString,100,"Y%d", ii+1);
      strNames.loadOneString(ii, pString);
    }
    if (nOutputs == 1)
    {
      snprintf(pString,100,"psDiscrepancyModel");
      filePtr1->writePsuadeFile(pString, 0);
    }
    else if (ii == nOutputs)
    {
      //**/ put outputs from all discrepancy files into a single place
      psIVector vecST;
      vecST.setLength(nSams);
      for (jj = 0; jj < nSams; jj++) vecST[jj] = 1;
      filePtr1->updateOutputSection(nSams,nOutputs,vecAllOuts.getDVector(),
                      vecST.getIVector(),strNames.getStrings());
      snprintf(pString,100,"psDiscrepancyModel");
      filePtr1->writePsuadeFile(pString, 0);
      printOutTS(PL_INFO,"MCMC_BF INFO: A sample (inputs/outputs) ");
      printOutTS(PL_INFO,"for the discrepancy model\n");
      printOutTS(PL_INFO,"              is now in psDiscrepancyModel.\n");
    }
    else
    {
      printOutTS(PL_INFO,
        "MCMC_BF INFO: Discrepancy sample file successfully created.\n");
    }
    delete filePtr1;
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++)
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      for (ii = 0; ii < nInputs; ii++) delete [] dbins2[jj][jj2][ii];
      for (ii = 0; ii < nInputs; ii++) delete [] pbins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
      delete [] pbins2[jj][jj2];
      delete [] dbins2[jj][jj2];
    }
    delete [] bins2[jj];
    delete [] dbins2[jj];
    delete [] pbins2[jj];
  }
  delete [] bins2;
  delete [] dbins2;
  delete [] pbins2;
  return 0.0;
}

// ************************************************************************
// perform MCMC-like analysis (brute force with sample uncertain variables)
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_bf2(aData &adata)
{
  int    ii, ii2, jj, jj2, kk, kk2, status, cnt, iOne=1, iZero=0;
  double ddata;
  char   lineIn[1001], pString[1001], *rsFile=NULL;
  FILE   *fp=NULL;
  pData  pPtr, pOutputs;

  //**/ ---------------------------------------------------------------
  // display header 
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*              Brute Force Inference (2)\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"* To stop inference, create a file psuade_stop\n");
  printEquals(PL_INFO, 0);
  displayBanner_bf2(printLevel);

  //**/ ---------------------------------------------------------------
  // extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  nInputs_     = nInputs;
  int nOutputs = adata.nOutputs_;
  nOutputs_    = nOutputs;
  int nSamples = adata.nSamples_;
  double *XIn   = adata.sampleInputs_;
  double *YIn   = adata.sampleOutputs_;
  double *lower = adata.iLowerB_;
  double *upper = adata.iUpperB_;
  PsuadeData *dataPtr = adata.ioPtr_;
  int *pdfFlags = adata.inputPDFs_;
  int noPDF       = 1;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++) 
      if (pdfFlags[ii] != 0) noPDF = 0;
  }

  //**/ ---------------------------------------------------------------
  //**/ This is an option to specify a rs index file in the data file.
  //**/ This option allows one to disable a certain input in the MCMC
  //**/ or set it to a uncertain but not calibrated input
  //**/ optimization ==> vecRSIndices, vecRSValues, faType, UParamsNumToUse
  //**/                  UParamsSample
  //**/ ---------------------------------------------------------------
  int UParamsNumToUse=1, imax;
  psIVector vecRSIndices;
  psVector  vecRSValues;
  psMatrix  UParamsSample;
  int faType = PSUADE_RS_MARS;
  if (dataPtr != NULL)
  {
    //**/ fetch response surface type
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;

    //**/ read rs_index_file, if there is any
    status = readIndexFile(dataPtr,vecRSIndices,vecRSValues,UParamsSample);
    if (status < 0) return PSUADE_UNDEFINED;

    //**/ if rs_index_file specifies there are uncertain parameters
    //**/ that are not to be calibrated
    if (UParamsSample.nrows() > 0)
    {
      printf("A sample for uncertain parameters has been provided.\n");
      printf("The sample size is %d\n", UParamsSample.nrows());
      imax = UParamsSample.nrows();
      if (imax > 1000) 
      {
        imax = 1000;
        printf("NOTE: To reduce cost, only 1000 points will be used.\n");
      }
      //snprintf(pString,100,
      //"Enter the sub-sample size to use for inference (1 - %d): ",imax);
      //UParamsNumToUse = getInt(1, imax, pString);

      //**/ check and update bounds for the uncertain parameters
      status = 0;
      cnt = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (vecRSIndices[ii] >= 1000)
        {
          for (jj = 0; jj < UParamsSample.nrows(); jj++)
          {
            ddata = UParamsSample.getEntry(jj, cnt);
            if ((ddata < lower[ii] || ddata > upper[ii]) && status == 0)
            {
              printf("MCMC_BF2 ERROR: fail uncertain sample bound check.\n");
              printf("   Sample number = %d\n", jj+1);
              printf("   Input         = %d\n", ii+1);
              printf("   UParams Input = %d\n", cnt+1);
              printf("   Sample data   = %e\n", ddata);
              printf("   Input Bounds  = %e %e\n", lower[ii], upper[ii]);
              status = 1;
            }
            if (ddata < lower[ii]) lower[ii] = ddata;
            if (ddata > upper[ii]) upper[ii] = ddata;
          }
          cnt++;
        }
      }
    }
  }
  else if (mode_ == 0)
  {
    printOutTS(PL_INFO,
      "MCMC_BF2 INFO: Since ioPtr=NULL, assume MARS as reponse surface.\n");
  }

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  //**/ => dnSamples,dnInputs,dParams,matSamInputs,matSamMeans,matSamStdevs
  //**/ => dnReps - degree of experimental replications
  //**/ ---------------------------------------------------------------
  int dnReps;
  psMatrix  matSamInputs, matSamMeans, matSamStdvs; 
  psIVector vecDParams; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDParams,matSamInputs,
                          matSamMeans, matSamStdvs, dnReps, printLevel);
  int dnSamples = matSamMeans.nrows();
  int dnInputs  = matSamInputs.ncols();
  if (psConfig_.AnaExpertModeIsOn())
  {
    printf("Size of experimental data has been detected to be %d.\n",
           dnSamples); 
    printf("In some cases the effective experimental data size ");
    printf("may be smaller due\n");
    printf("to repeated - but slightly different experiments. ");
    printf("To perform inference\n");
    printf("more correctly, please provide information on how ");
    printf("many replications are\n");
    printf("present in the experiemental data (1 means no replication).\n");
    printf("E.g. Experimental data size = 10, number of unique ");
    printf("experiments = 5\n");
    printf("     ==> number of replications = 2.\n");
    snprintf(pString,100, 
       "Enter the number of replications for each unique experiment: ");
    dnReps = getInt(1, dnSamples, pString);
  }

  //**/ compatibility error checking
  if (dstatus != 0.0)
  {
    printf("MCMC_BF2 ERROR: fail to read experimental data file.\n");
    return PSUADE_UNDEFINED;
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (vecRSIndices[ii] < 0 && vecDParams.length() > 0 && 
        vecDParams[ii] == 1)
    {
      printOutTS(PL_ERROR,
        "MCMC_BF2 ERROR: inactive input %d cannot be design parameter\n",
        ii+1);
      return PSUADE_UNDEFINED;
    }

    //**/ uncertain parameter has 999
    if (vecRSIndices[ii] == 999 && 
        (vecDParams.length() > 0 && vecDParams[ii] == 1))
    {
      printOutTS(PL_ERROR,
        "MCMC_BF2 ERROR: inactive input %d cannot be uncertain parameter\n",
        ii+1);
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ------------------------------------------------------------
  //**/ option to add response surface uncertainties to data std dev
  //**/ ==> rsErrFlag
  //**/ ------------------------------------------------------------
  int rsErrFlag=0;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,
        "*** OPTION TO INCLUDE RESPONSE SURFACE UNCERTAINTIES:\n");
    printOutTS(PL_INFO,
        "\nTo incorporate response surface uncertainties into the\n");
    printOutTS(PL_INFO,
        "likelihood function, make sure stochastic response surfaces\n");
    printOutTS(PL_INFO,
        "are used (GP/Kriging, polynomial regression, or bootstrapped\n");
    printOutTS(PL_INFO,
        "methods). Otherwise, no RS uncertainties will be included.\n");
    printOutTS(PL_INFO,
        "NOTE: if you don't know what this is, just say no below.\n");
    snprintf(pString,100,
       "===> Include response surface uncertainties? (y or n) ");
    getString(pString, lineIn);
    if (lineIn[0] == 'y') rsErrFlag = 1;
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ ==> faPtrs
  //**/ ---------------------------------------------------------------
  FuncApprox **faPtrs=NULL;
  printOutTS(PL_INFO,
       "MCMC_BF2 INFO: CREATING RESPONSE SURFACES FOR ALL OUTPUTS.\n");
  //**/ if sample has already been loaded
  if (nSamples <= 0)
  {
    printOutTS(PL_ERROR, "MCMC_BF2 ERROR: No sample loaded yet.\n");
    return PSUADE_UNDEFINED;
  }
  else
  {
    faPtrs = new FuncApprox*[nOutputs];
    psVector vecYY;
    vecYY.setLength(nSamples);
    for (ii = 0; ii < nOutputs; ii++)
    {
      faType = -1;
      printOutTS(PL_INFO,
       "MCMC_BF2 INFO: CREATING RESPONSE SURFACE FOR OUTPUT %d.\n",ii+1);
      faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
      faPtrs[ii]->setNPtsPerDim(16);
      faPtrs[ii]->setBounds(lower, upper);
      faPtrs[ii]->setOutputLevel(0);
      for (kk = 0; kk < nSamples; kk++) vecYY[kk] = YIn[kk*nOutputs+ii];

      status = faPtrs[ii]->initialize(XIn, vecYY.getDVector());
      if (status != 0)
      {
        printOutTS(PL_ERROR,
             "MCMC_BF2 ERROR: Unable to create response surface.\n");
        printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ special set up using expert mode
  //**/ ==> nbins, maxSamples, nPlots, vecPlotInds 
  //**/ ---------------------------------------------------------------
  int nPlots, nbins = 20, maxSamples = 500000;
  psIVector vecPlotInds;
  if (nInputs >= 10) maxSamples = 1000000;
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"*** CURRENT DEFAULT PARAMETER SETTINGS : \n\n");
  printOutTS(PL_INFO,"Inference max sample size = %d\n",maxSamples);
  printOutTS(PL_INFO,"Posterior histogram nbins = %d\n",nbins);
  printOutTS(PL_INFO,
    "NOTE: Histogram nBins  - resolution of histogram bar graph\n");
  printOutTS(PL_INFO,
    "Turn on ana_expert mode to change some of these default settings.\n\n");

  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    snprintf(pString,100,
            "Enter maximum inference sample size (500000 - 5000000): ");
    maxSamples = getInt(1000, 50000000, pString);
    if (maxSamples < 500000) maxSamples = 500000;
    if (nInputs >= 10 && maxSamples < 1000000)
      maxSamples = 1000000;
    //snprintf(pString,100,"Enter the number of histogram bins (10 - 25) : ");
    //nbins = getInt(10, 50, pString);
  }
  if (psConfig_.MasterModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
        "*** OPTION TO CREATE POSTERIOR PLOTS FOR SELECTED INPUTS:\n\n");
    printOutTS(PL_INFO,
        "MCMC will create MATLAB files for the posterior distributions.\n");
    printOutTS(PL_INFO,
        "You can choose to generate posterior plots for all inputs, or \n");
    printOutTS(PL_INFO,
        "just a selected few (in case there are too many inputs).\n");
    printf("Select inputs for which posterior plots are to be generated.\n");
    snprintf(pString,100,"Enter input number (-1 for all, 0 to terminate) : ");
    kk = 1;
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    while (kk != 0 || nPlots < 1)
    {
      kk = getInt(-1, nInputs, pString);
      if (kk == -1)
      {
        nPlots = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecRSIndices.length() == 0 || 
              (vecRSIndices[ii] >= 0 && vecRSIndices[ii] < 1000))
            if (vecDParams.length() == 0 || vecDParams[ii] == 0)
              vecPlotInds[nPlots++] = ii;
        }
        break;
      }
      if (kk != 0 && nPlots < nInputs)
      {
        if (vecRSIndices.length() > 0 && vecRSIndices[kk-1] < 0)
          printOutTS(PL_ERROR,
              "Input %d has been fixed by the rs index file (no plot).\n",
              kk+1);
        else if (vecDParams.length() > 0 && vecDParams[kk-1] == 1)
          printOutTS(PL_ERROR,
              "Input %d is a design parameter (no plot)\n",kk);
        else
          vecPlotInds[nPlots++] = kk - 1;
      }
      if (kk == 0 && nPlots == 0)
        printOutTS(PL_ERROR,
            "You need to set at least 1 input for plotting posteriors.\n");
    }
    if (nPlots > 1) sortIntList(nPlots, vecPlotInds.getIVector());
  }
  else
  {
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    for (ii = 0; ii < nInputs; ii++)
      if (vecRSIndices.length() == 0 || 
          (vecRSIndices[ii] >= 0 && vecRSIndices[ii] < 1000))
        if (vecDParams.length() == 0 || vecDParams[ii] == 0)
          vecPlotInds[nPlots++] = ii;
  }
  printOutTS(PL_INFO,
       "MCMC_BF2 Plot summary: Inputs to be plotted are (%d):\n",
       nPlots);
  for (ii = 0; ii < nPlots; ii++)
    printOutTS(PL_INFO, "   Input %4d\n", vecPlotInds[ii]+1);

  //**/ ---------------------------------------------------------------
  // option to add discrepancy function and a posterior sample
  // ==> modelFormFlag, genPosteriors
  //**/ ---------------------------------------------------------------
  int modelFormFlag=0, modelFormConst=0, genPosteriors=1;
  psIVector vecModelForms;
  if (psConfig_.AnaExpertModeIsOn() && UParamsSample.nrows() == 0)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** OPTION TO ADD A DISCREPANCY FUNCTION:\n\n");
    printOutTS(PL_INFO,
         "To use this feature, first make sure that the observation\n");
    printOutTS(PL_INFO,
         "data file specified earlier has design parameters specified\n");
    printOutTS(PL_INFO,
         "since the discrepancy function is to be a function of these\n");
    printOutTS(PL_INFO,
         "design parameters (if not, a constant discrepancy function\n");
    printOutTS(PL_INFO,"is to be created).\n");
    printOutTS(PL_INFO,
         "NOTE: if you don't know what this is, just say NO.\n");
    snprintf(pString,100,
            "===> Add discrepancy function ? (y or n) ");
    getString(pString, lineIn);
    if (lineIn[0] == 'y') modelFormFlag = 1;
    if (modelFormFlag == 1)
    {
      snprintf(pString,100,
        "===> Add discrepancy to (1) all or (2) selected outputs ? (1 or 2) ");
      kk = getInt(1,2,pString);
      if (kk == 1)
      {
        vecModelForms.setLength(nOutputs);
        vecModelForms.setConstant(iOne);
      }
      else
      {
        vecModelForms.setLength(nOutputs);
        for (ii = 0; ii < nOutputs; ii++)
        {
          snprintf(pString,100,"Add discrepancy to output %d ? (y or n) ",
                  ii+1);
          getString(pString, lineIn);
          if (lineIn[0] == 'y') vecModelForms[ii] = 1;
        }
      }
    }
    if (modelFormFlag == 1 && dnInputs == 0)
    {
      printOutTS(PL_INFO,
       "NOTE: No design inputs ==> discrepancy will be a constant function.\n");
    }
    else if (modelFormFlag == 1 && dnSamples == 1)
    {
      printOutTS(PL_INFO,
       "NOTE: 1 experiment ==> discrepancy will be a constant function.\n");
    }
    if (modelFormFlag == 1 && dnSamples > 1 && dnInputs > 0)
    {
      snprintf(pString,100,"===> Model form a constant function ? (y or n) ");
      getString(pString, lineIn);
      if (lineIn[0] == 'y') modelFormConst = 1;
    }
    printEquals(PL_INFO, 0);
  }
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,
       "*** OPTION TO CREATE A SAMPLE FROM THE POSTERIOR DISTRIBUTIONS:\n\n");
    printOutTS(PL_INFO,
       "In addition to generating the posterior distributions, you can\n");
    printOutTS(PL_INFO,
       "also draw a sample from these posteriors. The posterior sample\n");
    printOutTS(PL_INFO,
       "can be used as prior sample for another simulator/emulator.\n");
    printOutTS(PL_INFO,
       "NOTE: if you don't know what this is, just say no.\n");
    //**/ June 2022 : just turn this on
    //snprintf(pString,100,
    //        "==> Create posterior sample for the input parameters? (y/n) ");
    //getString(pString, lineIn);
    //if (lineIn[0] == 'y') genPosteriors = 1;
  }
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy function, if desired 
  //**/ ==> vecDiscSamOuts, vecDiscFuncConstMeans, vecDiscFuncConstStds
  //**/ ---------------------------------------------------------------
  pData qData; 
  if (dataPtr != NULL) dataPtr->getParameter("input_names", qData);
  psVector vecDiscFuncConstMeans, vecDiscFuncConstStds, vecDiscSamOuts;
  if (modelFormFlag == 1)
  {
    dstatus = createDiscrepancyFunctions(nInputs,nOutputs,lower,upper, 
                  vecRSIndices,vecRSValues,vecDParams,dnInputs,dnSamples, 
                  matSamInputs,matSamMeans,dataPtr,vecDiscFuncConstMeans,
                  vecDiscFuncConstStds,vecDiscSamOuts,faPtrs, 
                  printLevel,modelFormConst);
    if (dstatus < 0) return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //  set up for inference
  //**/ ---------------------------------------------------------------
  //**/ storage allocation
  psVector vecXmax, vecRanges;
  vecRanges.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecRanges[ii]=1.0/(upper[ii]-lower[ii]);
  vecXmax.setLength(nInputs);
  double *Xmax = vecXmax.getDVector();
  vecMeans_.setLength(nInputs_);
  vecSigmas_.setLength(nInputs_);
  vecMostLikelyInputs_.setLength(nInputs_);
  vecMostLikelyOutputs_.setLength(nOutputs_);

  //**/ ---------------------------------------------------------------
  //**/ set up bins for prior
  //**/ ---------------------------------------------------------------
  int **pbins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    pbins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) pbins[ii][jj] = 0;
  }
  int ****pbins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    pbins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      pbins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        pbins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          pbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ set up bins for posterior
  //**/ ---------------------------------------------------------------
  int **bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  int ****bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ posterior bins need to be double which is turned into integer
  //**/ ---------------------------------------------------------------
  double **dbins = new double*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    dbins[ii] = new double[nInputs];
    for (jj = 0; jj < nInputs; jj++) dbins[ii][jj] = 0;
  }
  double ****dbins2 = new double***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    dbins2[jj] = new double**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      dbins2[jj][jj2] = new double*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        dbins2[jj][jj2][ii] = new double[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          dbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ MCMC parameters (samInc, maxSamples)
  //**/ ---------------------------------------------------------------
  int  samInc = 10000;
  long itmp;
  if (UParamsNumToUse == 1)
  {
    if (samInc * dnSamples > 100000) samInc = 100000 / dnSamples; 
    maxSamples = maxSamples / samInc;
    maxSamples = maxSamples * samInc;
  }
  else
  {
    if (maxSamples < 5000000) maxSamples = 5000000;
    itmp = samInc * dnSamples * UParamsNumToUse;
    if (itmp > 100000) samInc = 100000 / (dnSamples * UParamsNumToUse); 
    if (samInc < 500) samInc = 500;
    maxSamples = maxSamples / (samInc * UParamsNumToUse);
    if (maxSamples < 4) maxSamples = 4;
    maxSamples = maxSamples * samInc * UParamsNumToUse;
    while (maxSamples > 50000000)
    {
      maxSamples /= 2;
      samInc /= 2;
    }
    printOutTS(PL_INFO,"MCMC maxSamples         = %d\n",maxSamples);
    printOutTS(PL_INFO,"MCMC sample increment   = %d\n",samInc);
    printOutTS(PL_INFO,"MCMC UParam sample size = %d\n",UParamsNumToUse);
  }

  //**/ ---------------------------------------------------------------
  //**/ generate a large sample ==> inferenceSamIns
  //**/ ---------------------------------------------------------------
  int      methodSave;
  Sampling *sampler;
  psVector  vecLB, vecUB, vecLargeSample, vecYT; 
  psIVector vecST;
  if (noPDF == 1)
  {
    printOutTS(PL_INFO,
               "MCMC_BF2 INFO: No PDF, use uniform for priors.\n");
    sampler = SamplingCreateFromID(PSUADE_SAMP_MC);
    sampler->setInputBounds(nInputs, lower, upper);
    sampler->setOutputParams(1);
    cnt = maxSamples / UParamsNumToUse;
    sampler->setSamplingParams(cnt, 1, 1);
    sampler->initialize(0);
    vecLargeSample.setLength(cnt*nInputs);
    vecYT.setLength(cnt);
    vecST.setLength(cnt);
    sampler->getSamples(cnt,nInputs,1,vecLargeSample.getDVector(),
                        vecYT.getDVector(),vecST.getIVector());
    delete sampler;
  }
  else
  {
    printOutTS(PL_INFO,
               "MCMC_BF2 INFO: Has PDF, draw sample from distribution.\n");
    dataPtr->getParameter("method_sampling", pPtr);
    methodSave = pPtr.intData_;
    dataPtr->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(dataPtr);
    vecLB.load(nInputs, lower);
    vecUB.load(nInputs, upper);
    cnt = maxSamples / UParamsNumToUse;
    vecLargeSample.setLength(cnt*nInputs);
    pdfman->genSample(cnt, vecLargeSample, vecLB, vecUB);
    dataPtr->updateMethodSection(methodSave,-1,-1,-1,-1);
    delete pdfman;
  }

  //**/ ---------------------------------------------------------------
  //  allocate storage for inference
  //**/ ---------------------------------------------------------------
  psVector vecXS, vecYS, vecSamStd, vecXDesign, vecYDesign;
  psVector vecYDesStd, vecInfSamInp, vecInfSamOut, vecInfTmpOut;

  if (samInc*dnSamples*UParamsNumToUse*nInputs > 2000000000 ||
      samInc*dnSamples*UParamsNumToUse*nInputs < 0)
  {
    printf("MCMC_BF2 ERROR: Number of uncertain sample too large.\n");
    printf("         Try something less than %d\n",
           2000000000/(dnSamples*samInc*nInputs));
    exit(1);
  }
  vecXS.setLength(samInc*dnSamples*UParamsNumToUse*nInputs);
  double *XSample = vecXS.getDVector();
  vecYS.setLength(samInc*dnSamples*UParamsNumToUse*nOutputs);
  double *YSample = vecYS.getDVector();
  vecSamStd.setLength(samInc*dnSamples*UParamsNumToUse*nOutputs);
  double *YSamStd = vecSamStd.getDVector();
  vecXDesign.setLength(samInc*dnSamples*UParamsNumToUse*nInputs);
  double *XDesign = vecXDesign.getDVector();
  vecYDesign.setLength(samInc*dnSamples*UParamsNumToUse*nOutputs);
  double *YDesign = vecYDesign.getDVector();
  vecYDesStd.setLength(samInc*dnSamples*UParamsNumToUse*nOutputs);
  double *YDesStd = vecYDesStd.getDVector();
  vecInfSamInp.setLength(maxSamples*nInputs);
  double *inferenceSamIns = vecInfSamInp.getDVector();
  vecInfSamOut.setLength(maxSamples);
  double *inferenceSamOut = vecInfSamOut.getDVector();
  vecInfTmpOut.setLength(maxSamples);
  double *inferenceTmpOut = vecInfTmpOut.getDVector();
  psIVector randIVec;
  double stdev, stdv2, YT, YT1, YT2;
  int    dcnt, irand, nInpsActive = 0, passCnt;

  //**/ ---------------------------------------------------------------
  //  duplicate the generated sample UParamsNumToUse-1 times
  //**/ so inferenceSamIns has a sample of size max/UParamsNumToUse
  //**/ replicated UParamsNumToUse times
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < maxSamples/UParamsNumToUse; ii++)
  {
    for (jj = 0; jj < UParamsNumToUse; jj++)
    {
      cnt = (ii * UParamsNumToUse + jj) * nInputs;
      for (kk = 0; kk < nInputs; kk++)
        inferenceSamIns[cnt+kk] = vecLargeSample[ii*nInputs+kk]; 
    }
  }
  vecLargeSample.clean();

  //**/ ---------------------------------------------------------------
  //**/ if there are uncertain parameters, create a random vector
  //**/ that selects a fixed sub-sample for the uncertain parameters
  //**/ then fill in the slots for uncertain parameters
  //**/ ---------------------------------------------------------------
  if (UParamsSample.ncols() > 0)
  {
    randIVec.setLength(UParamsNumToUse);
    for (int mm = 0; mm < UParamsNumToUse; mm++)
    {
      if (UParamsNumToUse >= UParamsSample.nrows())
           ii = mm;
      else ii = PSUADE_rand() % UParamsSample.nrows();
      randIVec[mm] = ii;
      //printf("Uncertain sample index %d = %d\n", mm+1, randIVec[mm]);
    }
    for (ii = 0; ii < maxSamples; ii+=UParamsNumToUse)
    {
      cnt = 0;
      for (kk = 0; kk < nInputs; kk++)
      {
        if ((vecRSIndices.length() == 0 || 
            (vecRSIndices.length() > 0 && vecRSIndices[kk] >=1000)))
        {
          for (jj = 0; jj < UParamsNumToUse; jj++)
          {
            ii2 = randIVec[jj];     
            ddata = UParamsSample.getEntry(ii2,cnt);
            inferenceSamIns[(ii+jj)*nInputs+kk] = ddata;
          }
          cnt++;
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ find out how many active (calibration) parameters
  //**/ ---------------------------------------------------------------
  for (ii2 = 0; ii2 < nInputs; ii2++)
  {
    if ((vecRSIndices.length() == 0 || 
        (vecRSIndices.length() > 0 && vecRSIndices[ii2] >=0 &&
         vecRSIndices[ii2] < 1000)) &&
        (vecDParams.length() == 0 || vecDParams[ii2] == 0)) 
      nInpsActive++;
  }

  //**/ ---------------------------------------------------------------
  //  perform inference
  //**/ ---------------------------------------------------------------
  int    mm,ss,sTotal=0,runSize,index,printStep=0,twiceFlag=0;
  double Ymin = PSUADE_UNDEFINED, Ymax = -PSUADE_UNDEFINED;
  psVector vecTmpMeans, vecTmpStds;
  vecTmpMeans.setLength(100*nInputs);
  double *tmpMeans = vecTmpMeans.getDVector();
  vecTmpStds.setLength(100*nInputs);
  double *tmpStds = vecTmpStds.getDVector();
  FILE   *fp2=NULL;

  printOutTS(PL_INFO, "MCMC_BF2 Inference begins ... \n");
  fflush(stdout);
  fp = NULL;

  while (sTotal < maxSamples)
  {
    //**/ outputing dots periodically
    cnt = (sTotal+1) / (maxSamples/20);
    if (cnt != printStep)
    {
      printOutTS(PL_INFO, "%3.0f%% ",5.0*cnt);
      fflush(stdout);
    }

    //**/ have one more loop for the uncertain parameters
    for (mm = 0; mm < samInc; mm++)
    {
      dcnt = 0;
      index = mm * dnSamples * UParamsNumToUse;
      //**/ first fill in all XSample and XDesign slots
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        for (kk2 = 0; kk2 < dnSamples; kk2++)
        {
          for (jj = 0; jj < UParamsNumToUse; jj++)
          {
            XSample[(index+kk2*UParamsNumToUse+jj)*nInputs+ii2] = 
             inferenceSamIns[(sTotal+mm*UParamsNumToUse+jj)*nInputs+ii2]; 
          }
        }
        //**/ load design parameters
        if (vecDParams.length() > 0 && vecDParams[ii2] == 1)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
          {
            for (jj = 0; jj < UParamsNumToUse; jj++)
            {
              XSample[(index+kk2*UParamsNumToUse+jj)*nInputs+ii2] = 
                                     matSamInputs.getEntry(kk2, dcnt);
              XDesign[(index+kk2*UParamsNumToUse+jj)*dnInputs+dcnt] = 
                                     matSamInputs.getEntry(kk2, dcnt);
            }
          }
          dcnt++;
        }
        //**/ load fixed parameters
        else if (vecRSIndices.length() > 0 && vecRSIndices[ii2] == -1) 
        {
          for (jj = 0; jj < samInc*UParamsNumToUse*dnSamples; jj++)
          {
            XSample[jj*nInputs+ii2] = vecRSValues[ii2];
          }
        }
      }
    }
    //**/ set design outputs and stds (discrepancy) to zero 
    //**/ since they will be used but may not be set later
    for (jj = 0; jj < samInc*UParamsNumToUse*dnSamples*nOutputs; jj++)
    {
      YDesign[jj] = YDesStd[jj] = 0.0;
      YSample[jj] = YSamStd[jj] = 0.0;
    }
    //**/ run XGuessS and XDesign through response surfaces
    //**/ ==> YDesign, YDesStd, YSample, YSamStd
    runSize = samInc * dnSamples * UParamsNumToUse;
    for (ii2 = 0; ii2 < nOutputs; ii2++)
    {
      //**/ case 1: if RS error is requested
      if (rsErrFlag == 1)
      {
        faPtrs[ii2]->evaluatePointFuzzy(runSize, 
                             XSample,&YSample[ii2*runSize],
                             &YSamStd[ii2*runSize]);
        //**/ add discrepancy function, if available
        if (modelFormFlag == 1 && modelFormConst == 0)
        {
          for (mm = 0; mm < samInc; mm++)
          {
            index = mm * dnSamples * UParamsNumToUse;
            for (kk2 = 0; kk2 < dnSamples; kk2++)
            {
              for (jj = 0; jj < UParamsNumToUse; jj++)
              {
                YDesign[ii2*runSize+index+kk2*UParamsNumToUse+jj] = 
                   vecDiscSamOuts[ii2*dnSamples+kk2];
                YDesStd[ii2*runSize+index+kk2*UParamsNumToUse+jj] = 0;
              }
            }
          }
        }
        else if (modelFormFlag == 1 && modelFormConst == 1 &&
                 vecDiscFuncConstMeans.length() > 0 &&
                 vecDiscFuncConstMeans[ii2] != PSUADE_UNDEFINED)
        {
          for (kk2 = 0; kk2 < runSize; kk2++)
          {
            YDesign[ii2*runSize+kk2] = vecDiscFuncConstMeans[ii2];
            YDesStd[ii2*runSize+kk2] = 0.0;
          }
        }
      }
      //**/ case 2: if RS error is to be turned off
      else
      {
        faPtrs[ii2]->evaluatePoint(runSize,XSample,
                                   &YSample[ii2*runSize]);
        //**/ add discrepancy function, if available
        if (modelFormFlag == 1 && modelFormConst == 0)
        {
          for (mm = 0; mm < samInc; mm++)
          {
            index = mm * dnSamples * UParamsNumToUse;
            for (kk2 = 0; kk2 < dnSamples; kk2++)
            {
              for (jj = 0; jj < UParamsNumToUse; jj++)
              {
                YDesign[ii2*runSize+index+kk2*UParamsNumToUse+jj] = 
                   vecDiscSamOuts[ii2*dnSamples+kk2];
                YDesStd[ii2*runSize+index+kk2*UParamsNumToUse+jj] = 0;
              }
            }
          }
        }
        else if (modelFormFlag == 1 && modelFormConst == 1 &&
                 vecDiscFuncConstMeans.length() > 0 &&
                 vecDiscFuncConstMeans[ii2] != PSUADE_UNDEFINED)
        {
          for (kk2 = 0; kk2 < runSize; kk2++)
            YDesign[ii2*runSize+kk2] = vecDiscFuncConstMeans[ii2];
        }
        for (kk2 = 0; kk2 < runSize; kk2++)
          YSamStd[ii2*runSize+kk2] = YDesStd[ii2*runSize+kk2] = 0.0;
      }
    }

    //**/ compute vecXDist
    for (ii = 0; ii < samInc; ii++)
    {
      index = ii * UParamsNumToUse;
      for (jj = 0; jj < UParamsNumToUse; jj++)
      {
        inferenceSamOut[sTotal+index+jj] = 0.0;
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
          {
            if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
            {
              YT1 = YSample[ii2*runSize+index*dnSamples+kk2*UParamsNumToUse+jj] +
                    YDesign[ii2*runSize+index*dnSamples+kk2*UParamsNumToUse+jj];
              stdv2 = YDesStd[ii2*runSize+index*dnSamples+kk2*UParamsNumToUse+jj];
            }
            else
            {
              YT1 = YSample[ii2*runSize+index*dnSamples+kk2*UParamsNumToUse+jj];
              stdv2 = 0;
            }
            stdev = YSamStd[ii2*runSize+index*dnSamples+kk2*UParamsNumToUse+jj];
            YT2 = pow((YT1-matSamMeans.getEntry(kk2,ii2)),2.0) /
                 (pow(matSamStdvs.getEntry(kk2,ii2),2.0) +
                  stdev*stdev + stdv2*stdv2);
            inferenceSamOut[sTotal+index+jj] += YT2;
          }
          inferenceSamOut[sTotal+index+jj] /= (double) (dnReps);
        }
        //**/ more correct?
        //*/ inferenceSamOut[sTotal+jj] /= (dnSamples*nOutputs);
      }
    }
    sTotal += (samInc * UParamsNumToUse);

    //**/ print interim information
    if (cnt != printStep)
    {
      printStep++;
      for (jj = 0; jj < sTotal; jj++) 
        inferenceTmpOut[jj] = inferenceSamOut[jj];
      Ymin = PSUADE_UNDEFINED;
      for (jj = 0; jj < sTotal; jj++) 
      {
        ddata = inferenceTmpOut[jj];
        if (ddata < Ymin) Ymin = ddata;
        if (ddata > Ymax) Ymax = ddata;
      }
      for (jj = 0; jj < sTotal; jj++) 
        inferenceTmpOut[jj] = exp(-0.5*(inferenceTmpOut[jj]-Ymin));

      printOutTS(PL_INFO,"\n");
      passCnt = 0;
      printOutTS(PL_INFO,"Convergence Checking (%d) =========>\n",sTotal);
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        if ((vecRSIndices.length() == 0 || 
             (vecRSIndices.length() > 0 && vecRSIndices[ii2] >=0 &&
              vecRSIndices[ii2] < 1000)) &&
            (vecDParams.length() == 0 || vecDParams[ii2] == 0))
        {
          YT = 0.0;
          for (jj = 0; jj < sTotal; jj++) YT += inferenceTmpOut[jj];
          YT1 = 0.0;
          //**/ input times likelihood
          for (jj = 0; jj < sTotal; jj++)
          {
            YT1 += inferenceSamIns[jj*nInputs+ii2] * 
                   inferenceTmpOut[jj] / YT; 
          }
          printOutTS(PL_INFO,"MCMC_BF2: input %3d mean    = %e\n", 
                     ii2+1, YT1);
          YT2 = 0.0;
          for (jj = 0; jj < sTotal; jj++)
          {
            YT2 += pow(inferenceSamIns[jj*nInputs+ii2]-YT1,2.0)*
                       inferenceTmpOut[jj]/YT; 
          }
          printOutTS(PL_INFO,"MCMC_BF2: input %3d std dev = %e\n",
                     ii2+1,sqrt(YT2));
          tmpMeans[(printStep-1)+ii2*100] = YT1; 
          tmpStds[(printStep-1)+ii2*100] = sqrt(YT2); 
          if (printStep > 2) 
          {
            YT1 = 1.05;
            YT2 = checkConvergence(3,&tmpMeans[ii2*100+printStep-3],
                     &tmpStds[ii2*100+printStep-3],
                     sTotal-samInc*UParamsNumToUse,YT1);
            if (YT2 < 1.05)
            {
              passCnt++;
              printf("MCMC_BF2: input %3d converged.\n",ii2+1);
            }
          }
        }
      }
      printOutTS(PL_INFO,"<========= Convergence Checking\n");
      if (passCnt == nInpsActive)
      {
        if (twiceFlag <= 0) twiceFlag++;
        else if (twiceFlag >= 1 && sTotal > 0.4*maxSamples) 
          maxSamples = sTotal;
      }
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        fclose(fp);
        printf("MCMC_BF2: file psuade_stop found => terminate\n");
        maxSamples = sTotal;
        strcpy(pString, "psuade_stop");
        unlink(pString);
      }
    }

    if (printLevel < 2)
    {
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        printOutTS(PL_INFO, "MCMC_BF2 INFO: print level set to 2\n");
        printLevel = 2;
        fclose(fp);
      }
    }

    if (printLevel > 1)
    {
      fp2 = fopen("psuadeMCMC.store", "a");
      if (fp2 != NULL)
      {
        int sTotal2 = sTotal + (samInc * UParamsNumToUse);
        for (jj = sTotal2-(samInc*UParamsNumToUse); jj < sTotal2; jj++)
        {
          for (ii2 = 0; ii2 < nInputs; ii2++)
            fprintf(fp2, "%e ", inferenceSamIns[jj*nInputs+ii2]);
          fprintf(fp2, "%e\n", inferenceSamOut[jj]);
        }
        printf("MCMC_BF2 INFO: psuadeMCMC.store has interim MCMC samples\n");
        fclose(fp2);
      }
    }
    printf("MCMC_BF2 INFO: To terminate MCMC gracefully, create ");
    printf("a 'psuade_stop'\n");
    printf("               file in working directory.\n");
  }

  //**/ normalize inferenceSamOut so that they are not too small
  //**/ (by using Ymin)
  Ymin = PSUADE_UNDEFINED;
  Ymax = -PSUADE_UNDEFINED;
  for (ss = 0; ss < maxSamples; ss++)
  {
    ddata = inferenceSamOut[ss];
    if (ddata < Ymin) Ymin = ddata;
    if (ddata > Ymax) Ymax = ddata;
  }
  for (ss = 0; ss < maxSamples; ss++)
    inferenceSamOut[ss] = exp(-0.5*(inferenceSamOut[ss]-Ymin));

  //**/ search for the maximum likelihood solution
  //**/ since normalizing factor of the likelihood function is
  //**/ constant, only the exponential term is computed ==> Ymax
  index = -1;
  Ymax = -PSUADE_UNDEFINED;
  for (ss = 0; ss < maxSamples; ss++)
  {
    ddata = inferenceSamOut[ss] * exp(-0.5*Ymin);
    if (ddata > Ymax)
    {
      Ymax = ddata;
      index = ss;
    }
  } 
  if (index >= 0)
  {
    for (ii = 0; ii < nInputs; ii++) 
      Xmax[ii] = inferenceSamIns[index*nInputs+ii];
  }
  printf("Maximum likelihood estimated solution:\n");
  for (ii = 0; ii < nInputs; ii++) 
  {
    if ((vecRSIndices.length() == 0 || 
        (vecRSIndices.length() > 0 && vecRSIndices[ii] >= 0 &&
         vecRSIndices[ii] < 999)) &&
        (vecDParams.length() == 0 || vecDParams[ii] == 0))
      printf("Input %3d = %16.8e\n",ii+1,Xmax[ii]);
  }
  printf("Negative log likelihood (unnormalized) = %e\n", -log(Ymax));

  //**/ now re-scale inferenceSamOut to control its magnitude
  //**/ in subsequent processing (binning)
  if (Ymax > 0)
  {
    Ymax = Ymax / exp(-0.5*Ymin);
    for (ss = 0; ss < maxSamples; ss++) inferenceSamOut[ss] /= Ymax;
  }

  //**/ update the 1D and 2D histogram
  int    ii3, index2;
  for (ss = 0; ss < maxSamples; ss++)
  {
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      YT1 = (inferenceSamIns[ss*nInputs+ii2] - lower[ii2])*vecRanges[ii2];
      index = (int) (YT1 * nbins);
      if (index > nbins)
      {
        printOutTS(PL_ERROR,"MCMC_BF2 binning error 1 in file %s, line %d.\n",
                   __FILE__, __LINE__);
        printOutTS(PL_ERROR,"Sample input %d = %e\n", ii2+1, 
                   inferenceSamIns[ss*nInputs+ii2]);
        printOutTS(PL_ERROR,"Sample input %d lower bound = %e\n", ii2+1, 
                   lower[ii2]);
        printOutTS(PL_ERROR,"Sample input %d scaled val  = %e\n",ii2+1,YT1);
        printOutTS(PL_ERROR,"Sample input %d bin number  = %d (<%d?)\n",
                   ii2+1,index,nbins);
      }
      if (index < 0)
      {
        printOutTS(PL_ERROR,"MCMC_BF2 binning error 2 in file %s, line %d.\n",
                   __FILE__, __LINE__);
        printOutTS(PL_ERROR,"Sample input %d = %e\n", ii2+1, 
                   inferenceSamIns[ss*nInputs+ii2]);
        printOutTS(PL_ERROR,"Sample input %d lower bound = %e\n", ii2+1, 
                   lower[ii2]);
        printOutTS(PL_ERROR,"Sample input %d scaled val  = %e\n",ii2+1,YT1);
        printOutTS(PL_ERROR,"Sample input %d bin number  = %d (>0?)\n",
                   ii2+1,index);
      }
      if (index >= nbins) index = nbins - 1;
      if (index <  0)     index = 0;
      dbins[index][ii2] += inferenceSamOut[ss];
      pbins[index][ii2]++;
    }
    //**/ update the 2D histogram
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      YT1 = (inferenceSamIns[ss*nInputs+ii2]-lower[ii2])*vecRanges[ii2];
      index = (int) (YT1 * nbins);
      if (index >= nbins) index = nbins - 1;
      if (index < 0)      index = 0;
      for (ii3 = 0; ii3 < nInputs; ii3++)
      {
        YT2 = (inferenceSamIns[ss*nInputs+ii3]-lower[ii3])*vecRanges[ii3];
        index2 = (int) (YT2 * nbins);
        if (index2 >= nbins) index2 = nbins - 1;
        if (index2 < 0)      index2 = 0;
        dbins2[index][index2][ii2][ii3] += inferenceSamOut[ss];
        pbins2[index][index2][ii2][ii3]++;
      }
    }
  }

  //**/ continue with binning
  Ymax = 0;
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      if (dbins[kk][ii2] > Ymax) Ymax = dbins[kk][ii2];
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      dbins[kk][ii2] = dbins[kk][ii2] / Ymax * maxSamples;
  for (kk = 0; kk < nbins; kk++)
    for (ii2 = 0; ii2 < nInputs; ii2++) bins[kk][ii2] = (int) dbins[kk][ii2];
  Ymax = 0;
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          if (dbins2[jj][kk][ii2][ii3] > Ymax) 
            Ymax = dbins2[jj][kk][ii2][ii3];
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          dbins2[jj][kk][ii2][ii3] = dbins2[jj][kk][ii2][ii3] / Ymax *
                                     maxSamples;
  for (jj = 0; jj < nbins; jj++)
    for (kk = 0; kk < nbins; kk++)
      for (ii2 = 0; ii2 < nInputs; ii2++)
        for (ii3 = 0; ii3 < nInputs; ii3++)
          bins2[jj][kk][ii2][ii3] = (int) dbins2[jj][kk][ii2][ii3];
     
  //**/ output statistics
  double psum;
  for (ii = 0; ii < nInputs; ii++)
  {
    if ((vecRSIndices.length() == 0 || (vecRSIndices.length() > 0 && 
         vecRSIndices[ii] >=0 && vecRSIndices[ii] < 1000)) &&
        (vecDParams.length() == 0 || vecDParams[ii] == 0))
    {
      printOutTS(PL_INFO,
                 "MCMC_BF2: input %3d value at peak of likelihood = %e\n",
                 ii+1, Xmax[ii]);
      psum = 0.0;
      for (ss = 0; ss < maxSamples; ss++) psum += inferenceSamOut[ss];
      YT = 0.0;
      //**/ input times likelihood
      for (ss = 0; ss < maxSamples; ss++)
        YT += inferenceSamIns[ss*nInputs+ii]*inferenceSamOut[ss]/psum; 
      vecMeans_[ii] = YT;
      printOutTS(PL_INFO,"MCMC_BF2: input %3d mean    = %e\n", ii+1, YT);
      YT = 0.0;
      for (ss = 0; ss < maxSamples; ss++)
        YT += pow(inferenceSamIns[ss*nInputs+ii]-vecMeans_[ii],2.0)*
              inferenceSamOut[ss]/psum; 
      vecSigmas_[ii] = sqrt(YT);
      printOutTS(PL_INFO,"MCMC_BF2: input %3d std dev = %e\n",
                 ii+1,vecSigmas_[ii]);
      vecMostLikelyInputs_[ii] = Xmax[ii];
    }
  }
 
  //**/ generate matlabmcmc2.m file at every major iteratin
  for (ii = 0; ii < nInputs; ii++) vecRanges[ii] = 1.0 / vecRanges[ii];
  genMatlabFile(nInputs,lower,upper,vecRanges.getDVector(),nPlots,
        vecPlotInds.getIVector(),nbins,pbins,pbins2,bins,bins2,qData,
        0,0,NULL,NULL, Xmax,Ymin);

  //**/ ---------------------------------------------------------------
  //**/  generate the posterior sample file
  //**/ ---------------------------------------------------------------
  if (genPosteriors == 1)
  {
    int    maxPostSam=50000;
    double dmax = 0.0;
    for (ss = 0; ss < maxSamples; ss++) 
      if (inferenceSamOut[ss] > dmax) dmax = inferenceSamOut[ss];
    if (dmax == 0)
    {
      printOutTS(PL_ERROR,
           "MCMC_BF2: ERROR encountered in posterior sample generation.\n");
      fp = NULL;
    }
    else 
    {
      fp = fopen("MCMCPostSample", "w");
      for (ss = 0; ss < maxSamples; ss++) 
        inferenceSamOut[ss] = inferenceSamOut[ss] / dmax;
      //**/ do not remember why I considered this
      //**/for (ss = 1; ss < maxSamples; ss++) 
      //**/  inferenceSamOut[ss] += inferenceSamOut[ss-1];
    }
    if (fp != NULL)
    {
      fprintf(fp, "PSUADE_BEGIN\n");
      //**/ count number of uncertain parameters
      kk = 0;
      for (jj = 0; jj < nInputs; jj++)
        if ((vecRSIndices.length() == 0 || (vecRSIndices[jj] >= 0 &&
             vecRSIndices[jj] < 1000)) &&
            (vecDParams.length() == 0 || vecDParams[jj] == 0)) kk++;
      fprintf(fp, "%d %d\n", maxPostSam, kk);
      if (qData.strArray_ != NULL)
      {
        fprintf(fp, "# ");
        for (jj = 0; jj < nInputs; jj++)
          if ((vecRSIndices.length() == 0 || (vecRSIndices[jj] >= 0 &&
               vecRSIndices[jj] < 1000)) &&
              (vecDParams.length() == 0 || vecDParams[jj] == 0))
            fprintf(fp,"%s ", qData.strArray_[jj]);
        fprintf(fp, "\n");
      }
#if 1
      //**/ Mar 2022: Is this better than the 'else' section?
      //**/ compute the weight of each sample * 5 * maxPostSam
      //**/ then collect ~5 * maxPostSam, and finally select
      //**/ maxSamples out of this 5*maxPostSam (for making sure
      //**/ there will be maxPostSam in view of roundoff)
      ddata = 0;
      for (ss = 0; ss < maxSamples; ss++) 
        ddata += inferenceSamOut[ss];
      psVector vecPostInps;
      vecPostInps.setLength(6*maxPostSam*nInputs);
      int count = 0;
      int nPostSam=0;
      for (ss = 0; ss < maxSamples; ss++) 
      {
        kk = (int) (inferenceSamOut[ss]/ddata*5*maxPostSam);
        nPostSam += kk;
        for (jj = 0; jj < kk; jj++)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            if ((vecRSIndices.length() == 0 || (vecRSIndices[ii] >= 0 &&
                 vecRSIndices[ii] < 1000)) &&
                (vecDParams.length() == 0 || vecDParams[ii] == 0))
              vecPostInps[count++] = inferenceSamIns[ss*nInputs+ii];
          }
        }
      }
      int nUInps = count / nPostSam;
      count = 0;
      while (count < maxPostSam)
      {
        ss = PSUADE_rand() % nPostSam;
        fprintf(fp, "%d ", count+1);
        for (jj = 0; jj < nUInps; jj++)
          fprintf(fp, "%e ", vecPostInps[ss*nUInps+jj]);
        fprintf(fp, "\n");
        count++;
      }
#else
      int    count=0, ichoose, nValid=0;
      double dchoose;
      psIVector vecValid;
      vecValid.setLength(maxSamples);
      for (ss = 0; ss < maxSamples; ss++) 
        if (inferenceSamOut[ss] > 1e-5) vecValid[nValid++] = ss;
      while (count < maxPostSam)
      {
        ichoose = PSUADE_rand() % nValid;
        ichoose = vecValid[ichoose];
        dchoose = PSUADE_drand();
        if (inferenceSamOut[ichoose] > dchoose)
        {
          fprintf(fp, "%d ", count+1);
          for (jj = 0; jj < nInputs; jj++)
          {
             if ((vecRSIndices.length() == 0 || (vecRSIndices[jj] >= 0 &&
                  vecRSIndices[jj] < 1000)) &&
                 (vecDParams.length() == 0 || vecDParams[jj] == 0))
             {
                fprintf(fp, "%e ", 
                    inferenceSamIns[ichoose*nInputs+jj]);
             }
             //**/ only include uncertain parameters
             //else if (vecRSIndices.length() > 0 && vecRSIndices[jj] == -1)
             //   fprintf(fp, "%e ", vecRSValues[jj]);
             //else if (vecDParams.length() > 0 && vecDParams[jj] != 0)
             //   fprintf(fp, "%e ", 0.5 * (upper[jj] + lower[jj]));
             //else if (vecRSIndices.length() > 0 && vecRSIndices[jj] == -2)
             //   fprintf(fp, "0 ");
          }
          //**/ only include uncertain parameters
          //fprintf(fp, " %e\n",inferenceSamOut[ichoose]);
          fprintf(fp, "\n");
          count++;
        }
      }
#endif
      fprintf(fp, "PSUADE_END\n");
      //**/ for the MLE solution, dissect the component of likelihood
      psVector vecXSam, vecYSam, vecXDes, vecYDes;
      vecXSam.setLength(nInputs*dnSamples);
      vecYSam.setLength(nOutputs*dnSamples);
      vecXDes.setLength(dnInputs*dnSamples);
      vecYDes.setLength(nOutputs*dnSamples);
      YSample = vecYSam.getDVector();
      YDesign = vecYDes.getDVector();
      fprintf(fp, "Optimal parameter values: \n");
      dcnt = 0;
      for (ii2 = 0; ii2 < nInputs; ii2++) 
      {
        if (vecDParams.length() > 0 && vecDParams[ii2] == 1) 
        {
          fprintf(fp, "%16.8e\n", matSamInputs.getEntry(0,dcnt));
          dcnt++;
        }
        else if (vecRSIndices.length() > 0 && vecRSIndices[ii2] == -1)
          fprintf(fp, "%16.8e\n", vecRSValues[ii2]);
        else if (vecRSIndices.length() > 0 && vecRSIndices[ii2] == -2)
          fprintf(fp, "0.0\n");
        else
          fprintf(fp, "%16.8e\n", Xmax[ii2]);
      }
      fprintf(fp,"Best negLogLikelihood = %e (Ideal=0)\n",Ymin);
      fprintf(fp,
        "MLE Statistics (prediction, exp data, sd, -loglikelihood)\n");
      for (kk2 = 0; kk2 < dnSamples; kk2++)
      {
        dcnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++) 
        {
          vecXSam[kk2*nInputs+ii2] = Xmax[ii2];
          if (vecDParams.length() > 0 && vecDParams[ii2] == 1)
          {
            vecXSam[kk2*nInputs+ii2] = matSamInputs.getEntry(kk2,dcnt);
            vecXDes[kk2*dnInputs+dcnt] = matSamInputs.getEntry(kk2,dcnt);
            dcnt++;
          }
          if (vecRSIndices.length() > 0 && vecRSIndices[ii2] == -1)
            vecXSam[kk2*nInputs+ii2] = vecRSValues[ii2];
        }
      }
      for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++)
        vecYDes[ii2] = YSample[ii2] = 0.0;
      for (ii2 = 0; ii2 < nOutputs; ii2++) 
      {
        faPtrs[ii2]->evaluatePoint(dnSamples,vecXSam.getDVector(),
                                   &YSample[ii2*dnSamples]);
        if (modelFormFlag == 1 && modelFormConst == 0)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
          {
            YDesign[ii2*dnSamples+kk2] = 
                 vecDiscSamOuts[ii2*dnSamples+kk2];
          }
        }
        else if (modelFormFlag == 1 && modelFormConst == 0 &&
                 vecDiscFuncConstMeans.length() > 0 &&
                 vecDiscFuncConstMeans[ii2] != PSUADE_UNDEFINED)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
            vecYDes[ii2*dnSamples+kk2] = vecDiscFuncConstMeans[ii2];
        }
      }
      for (kk2 = 0; kk2 < dnSamples; kk2++)
      {
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
            YT1 = YSample[ii2*dnSamples+kk2] + vecYDes[ii2*dnSamples+kk2];
          else
            YT1 = YSample[ii2*dnSamples+kk2];
          YT2 = matSamMeans.getEntry(kk2, ii2);
          stdev = matSamStdvs.getEntry(kk2, ii2);
          stdv2 = 0.5 * pow(YT1 - YT2, 2.0) / (stdev * stdev);
          fprintf(fp,"%4d %12.4e %12.4e %12.4e %12.4e (%12.4e)\n",
                  kk2+1,YT1,YT2,stdev,stdv2,sqrt(2*stdv2));
        }
      }
      fclose(fp);
    }
    printOutTS(PL_INFO,
         "MCMC_BF2: 'MCMCPostSample' file has a posterior sample.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy function files
  //**/ ---------------------------------------------------------------
  int  nInps, nOuts, nSams, *states;
  psVector vecAllOuts;
  psStrings strNames;
  PsuadeData *filePtr1, *filePtr2;

  if (modelFormFlag == 1)
  {
    snprintf(pString,100,"psDiscrepancyModel1");
    filePtr1 = new PsuadeData();
    status = filePtr1->readPsuadeFile(pString);
    if (status != 0)
    {
       printOutTS(PL_ERROR,
            "MCMC_BF2 ERROR: cannot read file %s in PSUADE format.\n",
            pString);
       exit(1);
    }
  }
  if (modelFormFlag == 1 && status == 0)
  {
    //**/ read all discrepancy files and collect all outputs
    filePtr1->getParameter("input_ninputs", pPtr);
    nInps = pPtr.intData_;
    filePtr1->getParameter("output_noutputs", pPtr);
    nOuts = pPtr.intData_;
    filePtr1->getParameter("method_nsamples", pPtr);
    nSams = pPtr.intData_;
    filePtr1->getParameter("output_sample", pOutputs);
    unlink(pString);
    vecAllOuts.setLength(nOutputs * nSams);
    if (vecModelForms.length() > 0 && vecModelForms[0] == 1)
    {
      for (jj = 0; jj < nSams; jj++)
        vecAllOuts[jj*nOutputs] = pOutputs.dbleArray_[jj];
    }
    pOutputs.clean();

    strNames.setNumStrings(nOutputs);
    strcpy(pString, "Y1");
    strNames.loadOneString(0, pString);
    for (ii = 1; ii < nOutputs; ii++)
    {
      filePtr2 = new PsuadeData();
      snprintf(pString,100,"psDiscrepancyModel%d", ii+1);
      status = filePtr2->readPsuadeFile(pString);
      if (status != 0) break;
      filePtr2->getParameter("input_ninputs", pPtr);
      if (pPtr.intData_ != nInps) break;
      filePtr2->getParameter("output_noutputs", pPtr);
      if (pPtr.intData_ != nOuts) break;
      filePtr2->getParameter("method_nsamples", pPtr);
      if (pPtr.intData_ != nSams) break;
      filePtr2->getParameter("output_sample", pOutputs);
      delete filePtr2;
      unlink(pString);
      if (vecModelForms.length() > 0 && vecModelForms[ii] == 1)
      {
        for (jj = 0; jj < nSams; jj++)
          vecAllOuts[jj*nOutputs+ii] = pOutputs.dbleArray_[jj];
      }
      pOutputs.clean();
      snprintf(pString,100,"Y%d", ii+1);
      strNames.loadOneString(ii, pString);
    }
    if (nOutputs == 1)
    {
       snprintf(pString,100,"psDiscrepancyModel");
       filePtr1->writePsuadeFile(pString, 0);
    }
    else if (ii == nOutputs)
    {
      //**/ put outputs from all discrepancy files into a single place
      states = new int[nSams];
      for (jj = 0; jj < nSams; jj++) states[jj] = 1;
      filePtr1->updateOutputSection(nSams,nOutputs,
                 vecAllOuts.getDVector(),states,
                 strNames.getStrings());
      snprintf(pString,100,"psDiscrepancyModel");
      filePtr1->writePsuadeFile(pString, 0);
      printOutTS(PL_INFO,"MCMC_BF2 INFO: A sample (inputs/outputs) \n");
      printOutTS(PL_INFO,"for the discrepancy model is\n");
      printOutTS(PL_INFO,"               is now in psDiscrepancyModel.\n");
      delete [] states;
    }
    else
    {
      printOutTS(PL_INFO,
        "MCMC_BF2 INFO: Unsuccessful creation of discrepancy sample file\n");
    }
    delete filePtr1;
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++)
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  for (ii = 0; ii < nbins; ii++) 
  {
    delete [] bins[ii];
    delete [] dbins[ii];
    delete [] pbins[ii];
  }
  delete [] bins;
  delete [] pbins;
  delete [] dbins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      for (ii = 0; ii < nInputs; ii++) delete [] dbins2[jj][jj2][ii];
      for (ii = 0; ii < nInputs; ii++) delete [] pbins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
      delete [] pbins2[jj][jj2];
      delete [] dbins2[jj][jj2];
    }
    delete [] bins2[jj];
    delete [] dbins2[jj];
    delete [] pbins2[jj];
  }
  delete [] bins2;
  delete [] dbins2;
  delete [] pbins2;
  return 0.0;
}

// ************************************************************************
// perform MCMC analysis with simulator
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_sim(aData &adata)
{
  int    ii, ii2, jj, jj2, iOne=1, iZero=0, status;
  double ddata;
  char   pString[1000];
  pData  pdata, pdata2;
  PDFBase **inputPDFs;
  FILE   *fp=NULL;

  //**/ ---------------------------------------------------------------
  //**/ extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int    printLevel  = adata.printLevel_;
  int    nInputs     = adata.nInputs_;
  int    nOutputs    = adata.nOutputs_;
  double *xLower     = adata.iLowerB_;
  double *xUpper     = adata.iUpperB_;
  int    *pdfTypes   = adata.inputPDFs_;
  double *pdfMeans   = adata.inputMeans_;
  double *pdfStdvs   = adata.inputStdevs_;
  nInputs_ = nInputs;
  nOutputs_ = nOutputs;

  //**/ ---------------------------------------------------------------
  //**/ get pointer to PsuadeData object for accessing pertinent
  //**/ information. Also, names for input (for use in plotting)
  //**/ ---------------------------------------------------------------
  PsuadeData *dataPtr = adata.ioPtr_;
  if (dataPtr == NULL)
  {
    printOutTS(PL_ERROR,
         "MCMC_sim ERROR: Direct simulation mode is requested but no\n");
    printOutTS(PL_ERROR,
         "     information on the simulator is given (missing\n");
    printOutTS(PL_ERROR,
         "     PSUADE data file that points to the simulator).\n");
    return PSUADE_UNDEFINED;
  }
  dataPtr->getParameter("input_names", pdata);

  //**/ ---------------------------------------------------------------
  //**/ display header 
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*      Simulation-based MCMC (MH)\n");
  printEquals(PL_INFO, 0);
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
    printOutTS(PL_INFO," * ana_expert to finetune MCMC parameters, \n");
    printOutTS(PL_INFO,
         "   (e.g. sample size for burn-in can be adjusted).\n");
    printDashes(PL_INFO,0);
    printOutTS(PL_INFO,
         "FEATURES AVAILABLE IN THE CURRENT VERSION OF MCMC:\n");
    printOutTS(PL_INFO,
         " * Support other than uniform prior distributions\n");
    printOutTS(PL_INFO,
         " * However, joint prior distribution is not supported.\n");
    printOutTS(PL_INFO,
         " * Support likelihood functions from multiple outputs\n");
    printOutTS(PL_INFO,
         " * Option to set some inputs as design parameters\n");
    printOutTS(PL_INFO,
         "   - to be specified in the observation data spec file\n");
    printOutTS(PL_INFO,
         " * MCMC can be terminated gracefully by creating a file ");
    printOutTS(PL_INFO, "named\n");
    printOutTS(PL_INFO,
         "   'psuade_stop' in the same directory while it is running\n");
    printOutTS(PL_INFO,"   (if it takes too long).\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ option to continue from previous run
  //**/ ---------------------------------------------------------------
  int  useHist = 0;
  FILE *fpSave = NULL, *fpHist = NULL;
  fpHist = fopen("mcmc_sim.sav", "r");
  if (fpHist != NULL)
  {
    printf("NOTE: MCMC_sim detects a MCMC history file mcmc_sim.sav.\n");
    printf("If previous simulation-based inference has ");
    printf("been terminated abruptly,\n");
    printf("mcmc_sim.sav will have stored the history ");
    printf("and can be restarted from\n");
    printf("where it was.\n");
    printf("Do you want to restart using mcmc_sim.sav ? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y') 
    {
      useHist = 1;
      fclose(fpHist);
      status = rename("mcmc_sim.sav", "mcmc_sim.sav1");
      printf("MCMC_sim INFO: Your mcmc_sim.sav has been ");
      printf("saved in mcmc_sim.sav1\n"); 
    }
    fgets(pString,1000,stdin);
    fpHist = NULL;
  }

  //**/ ---------------------------------------------------------------
  //**/ clean up important psuade signal files (e.g. psuade_stop)
  //**/ allocate space for keeping track of posterior statistics
  //**/ ---------------------------------------------------------------
  cleanUp();
  vecMeans_.setLength(nInputs_);
  vecSigmas_.setLength(nInputs_);
  vecMostLikelyInputs_.setLength(nInputs_);
  vecMostLikelyOutputs_.setLength(nOutputs_);

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  // ==> dnSamples, dnInputs, matExpInps, matExpMeans, matExpStdvs
  //**/ ---------------------------------------------------------------
  int    dnSamples=0, dnInputs=0, dnReps;
  psIVector vecDesignP; 
  psMatrix  matExpInps, matExpMeans, matExpStdvs; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  if (dstatus != 0.0) return PSUADE_UNDEFINED;
  dnSamples = matExpMeans.nrows();
  dnInputs  = matExpInps.ncols();

  //**/ ---------------------------------------------------------------
  //**/ special set up (sample size) using interactive mode
  //    ==> burnInSamples, maxSamples, nbins, vecPlotIndices
  //**/ ---------------------------------------------------------------
  int    maxSamples = 2000, burnInSamples = 100, nbins = 20;
  double propScale=0.25;
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"*** CURRENT SETTINGS OF MCMC_sim PARAMETERS: \n\n");
  printOutTS(PL_INFO,"Burn-in sample size         = %d\n",burnInSamples);
  printOutTS(PL_INFO,"Maximum MCMC samples        = %d\n",maxSamples);
  printOutTS(PL_INFO,"Proposal distribution scale = %e\n",propScale);
  printOutTS(PL_INFO,"No. of bins in histogram    = %d\n",nbins);
  printOutTS(PL_INFO,
     "NOTE: Proposal distribution is Normal(X_i,sig^2) and sig = scale.\n");
  printOutTS(PL_INFO,
     "NOTE: Histogram nBins = granularity of histogram bar graph\n");
  printOutTS(PL_INFO, 
     "Turn on ana_expert mode to change these default settings.\n\n");

  //**/ ---------------------------------------------------------------
  //**/ open a file to save history
  //**/ ---------------------------------------------------------------
  fpSave = fopen("mcmc_sim.sav", "w");
  if (fpSave == NULL)
  {
    printf("MCMC_sim INFO: Cannot open mcmc_sim.sav. ");
    printf("History will not be saved.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ if user wishes to restart, open the history file
  //**/ ---------------------------------------------------------------
  if (useHist == 1)
  {
    fpHist = fopen("mcmc_sim.sav1", "r");
    if (fpHist == NULL)
    {
      useHist = 0;
      printf("MCMC_sim ERROR: Failed to open history file.\n");
      printf("INFO: Will start from beginning (no restart).\n");
    }
    else
    {
      fscanf(fpHist, "%d %d", &burnInSamples, &maxSamples);
      if (burnInSamples < 0)
      {
        printf("MCMC_sim ERROR: burn-in nSamples < 0\n");
        printf("NOTE: Check the first line of your mcmc_sim.sav1 file.\n");
        fclose(fpHist);
        exit(1);
      }
      for (ii = 0; ii < vecDesignP.length(); ii++)
      {
        fscanf(fpHist, "%d", &jj);
        if (jj != vecDesignP[ii])
        {
          printf("MCMC_sim ERROR: Input %d mismatch (design parameter).\n", 
                 jj+1);
          printf("NOTE: Check the second line of your mcmc_sim.sav1 file.\n");
          fclose(fpHist);
          exit(1);
        }
      } 
    } 
  } 

  //**/ ---------------------------------------------------------------
  //**/ if not restart, users may select sample size options
  //**/ as well as design parameter selection 
  //**/ (second line,
  //**/ ---------------------------------------------------------------
  int adaptive=0;
  if (useHist == 0 && psConfig_.AnaExpertModeIsOn())
  {
    snprintf(pString,100,"Enter burn in sample size (0 - 1000): ");
    burnInSamples = getInt(0, 1000, pString);
    snprintf(pString,100,"Enter maximum MCMC sample (100 - 10000) : ");
    maxSamples = getInt(100, 10000, pString);
    snprintf(pString,100,
            "Enter proposal distribution scale (>0, <1, 0 - adaptive) : ");
    propScale = getDouble(pString);
    if (propScale == 0) 
    {
      adaptive = 1;
      propScale = 0.25;
    }
    else if (propScale < 0 || propScale > 1)
    {
      printf("MCMC_sim ERROR: Invalid proposal distribution scale.\n");
      printf("                Set to default = 0.25.\n");
      propScale = 0.25;
    }
  }
  if (fpSave != NULL)
  {
    fprintf(fpSave, "%d %d\n", burnInSamples, maxSamples);
    for (ii = 0; ii < vecDesignP.length(); ii++)
    {
      jj = vecDesignP[ii];
      fprintf(fpSave, "%d ", jj);
    }
    if (vecDesignP.length() > 0) fprintf(fpSave,"\n");
  } 

  //**/ ---------------------------------------------------------------
  //**/ set up posterior plotting (not including design parameters)
  //**/ ---------------------------------------------------------------
  psIVector vecPlotIndices;
  vecPlotIndices.setLength(nInputs);
  int nPlots = 0;  
  for (ii = 0; ii < nInputs; ii++) 
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
      vecPlotIndices[nPlots++] = ii;

  //**/ ---------------------------------------------------------------
  //**/ create PDF generators: inputPDFs
  //**/ ---------------------------------------------------------------
  inputPDFs = new PDFBase*[nInputs];
  for (ii = 0; ii < nInputs; ii++)
  {
    inputPDFs[ii] = NULL;
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
    {
      if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_NORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFNormal(pdfMeans[ii], 
                                                  pdfStdvs[ii]);
        if (printLevel > 2) 
           printOutTS(PL_INFO,
                "Parameter %3d has normal prior distribution (%e,%e)\n",
                ii+1, pdfMeans[ii], pdfStdvs[ii]);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_LOGNORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFLogNormal(pdfMeans[ii],
                                                     pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has lognormal prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_TRIANGLE)
      {
        inputPDFs[ii] = (PDFBase *) new PDFTriangle(pdfMeans[ii],
                                                    pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has triangle prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_BETA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFBeta(pdfMeans[ii],
                                                pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has beta prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_WEIBULL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFWeibull(pdfMeans[ii],
                                                   pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has Weibull prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_GAMMA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFGamma(pdfMeans[ii],
                                                 pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
             "Parameter %3d has gamma prior distribution.\n",ii+1);
      }
      else if (pdfTypes == NULL || pdfTypes[ii] == PSUADE_PDF_UNIFORM)
      {
        inputPDFs[ii] = NULL;
        if (printLevel > 2)
          printOutTS(PL_INFO,
             "Parameter %3d has uniform prior distribution.\n",ii+1);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ create function IO
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO, "MCMC_sim: setting up function interface ... \n");
  FunctionInterface *funcIO = createFunctionInterface(dataPtr);

  //**/ ---------------------------------------------------------------
  //    set up for MCMC iterations
  //**/ ---------------------------------------------------------------
  //**/ ----- for model evaluation
  printOutTS(PL_INFO, "MCMC_sim: initialization ... \n");
  psVector vecXBest, vecXGuess, vecYGuess;
  vecXGuess.setLength(nInputs);
  double *XGuess = vecXGuess.getDVector();
  vecYGuess.setLength(dnSamples*nOutputs);
  double *YGuess = vecYGuess.getDVector();
  //**/ ----- for storing the point of maximum likelihood
  vecXBest.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecXBest[ii] = 0;
  //**/ ---------------------------------------------------------------
  //**/ set initial seed to be mid point or distribution means
  //**/ ---------------------------------------------------------------
  int    igFlag = 0;
  double YT;
  psVector vecFuncY;
  vecFuncY.setLength(nOutputs);
  psMatrix matXChains;
  matXChains.setFormat(PS_MAT2D);
  matXChains.setDim(maxSamples, nInputs+1);
  double **mcmcChain = matXChains.getMatrix2D();

  //**/ set initial guess
  if (useHist == 0 && psConfig_.AnaExpertModeIsOn())
  {
    printf("Do you want to set your own initial guess ? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y')
    {
      printf("Enter initial parameter values below. If ");
      printf("you don't know what values to\n");
      printf("set, enter the suggested point.\n");
      igFlag = 1;
    }
    fgets(pString,1000,stdin);
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (inputPDFs[ii] != NULL)
         vecXGuess[ii] = pdfMeans[ii];
    else vecXGuess[ii] = 0.5 * (xUpper[ii] + xLower[ii]);
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
    {
      if (igFlag == 1) 
      {
        snprintf(pString,100,
           "Enter initial value for input %d (suggested=%e) : ",
           ii+1,vecXGuess[ii]);
        vecXGuess[ii] = getDouble(pString);
      }
      if (printLevel > 2) 
        printf("Initial guess %d = %e\n",ii+1,vecXGuess[ii]);
    }
    mcmcChain[0][ii] = vecXGuess[ii];
    if (useHist == 1 && fpHist != NULL)
    {
      fscanf(fpHist, "%lg", &ddata);
      mcmcChain[0][ii] = ddata;
    }
    if (fpSave != NULL)
      fprintf(fpSave, "%e ", mcmcChain[0][ii]);
  }

  //**/ evaluate at the seed point
  int    offset;
  double minLoglikelihood = PSUADE_UNDEFINED;
  double mcmcMean = 0, mcmcStdvs = 0.0, loglikelihood;
  if (dnInputs > 0)
  {
    //**/ if use history, get loglikelihood from history file
    if (useHist == 1 && fpHist != NULL)
      fscanf(fpHist, "%lg", &loglikelihood); 
    else
    {
      loglikelihood = 0;
      for (ii = 0; ii < dnSamples; ii++)
      {
        offset = 0;
        for (jj = 0; jj < nInputs; jj++)
        {
          if (vecDesignP.length() > 0 && vecDesignP[jj] == 1) 
          {
            vecXGuess[jj] = matExpInps.getEntry(ii,offset);
            offset++;
          }
        }
        funcIO->evaluate(ii+1,nInputs,vecXGuess.getDVector(),nOutputs,
                         vecFuncY.getDVector(),0);
        YT = 0;
        for (jj = 0; jj < nOutputs; jj++)
        {
          vecYGuess[ii*nOutputs+jj] = vecFuncY[jj];
          YT += (pow(vecFuncY[jj]-matExpMeans.getEntry(ii,jj),2.0)/
                 pow(matExpStdvs.getEntry(ii,jj),2.0));
        }
        loglikelihood += YT;
      }
    }
  }
  else
  //**/ no design inputs ==> hence number of experiements expected=1
  {
    if (useHist == 1 && fpHist != NULL)
      fscanf(fpHist, "%lg", &loglikelihood); 
    else
    {
      funcIO->evaluate(iOne,nInputs,vecXGuess.getDVector(),nOutputs,
                     vecFuncY.getDVector(),0);
      loglikelihood = 0.0;
      for (jj = 0; jj < nOutputs; jj++)
        loglikelihood += (pow(vecFuncY[jj]-matExpMeans.getEntry(0,jj),2.0)/
                          pow(matExpStdvs.getEntry(0,jj),2.0));
      if (printLevel > 3)
      {
        printf("MCMC_sim Iteration = 0 : \n"); 
        for (jj = 0; jj < nInputs; jj++)
          printf("   Input  %d = %e\n", jj+1, vecXGuess[jj]);
        for (jj = 0; jj < nOutputs; jj++)
          printf("   Output %d = %e\n", jj+1, vecFuncY[jj]);
      }
    }
  }
  if (fpSave != NULL) fprintf(fpSave, "%e\n", loglikelihood);
  loglikelihood /= (double) dnReps;
  if (printLevel > 1) 
    printf("Negative logLikelihood for initial guess = %e\n",
           loglikelihood);
  mcmcChain[0][nInputs] = loglikelihood;
  if (loglikelihood < minLoglikelihood) 
  {
    minLoglikelihood = loglikelihood;
    vecXBest = vecXGuess;
  }
  PDFBase *normalPdf = (PDFBase *) new PDFNormal(0.0, 1.0);

  //**/ ---------------------------------------------------------------
  //**/ run MCMC 
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "MCMC_sim (MH, simulator) begins ... \n");
  fflush(stdout);

  int    mcmcIts = 0, nAccept=1, nAcceptAll=1,errFlag, rejectOrAccept;
  double c12=PSUADE_UNDEFINED, c11, c1, prior, ddata2;
  double alpha, dOne=1, d5=5, dNeg5=-5;
  psVector vecLast;
  vecLast.setLength(nInputs);
  while (mcmcIts < maxSamples)
  {
    //**/ update iteration index
    mcmcIts++;

    //**/ save iteration index
    if (fpSave != NULL) fprintf(fpSave, "%d ", mcmcIts);

    //**/ if use history, make sure iteration indices match
    //**/ if not, turn off history and continue
    if (useHist == 1 && fpHist != NULL)
    {
      fscanf(fpHist, "%d", &ii);
      if (ii != mcmcIts)
      {
        printf("MCMC_sim INFO: History ends at iteration %d.\n",
               mcmcIts);
        fclose(fpHist);
        fpHist = NULL;
        useHist = 0;
      }
    }
    printOutTS(PL_INFO,
      "MCMC_sim : Iteration = %5d ('touch' psuade_stop to terminate)\n", 
      mcmcIts);
    //printOutTS(PL_INFO,
    //  "                             (psuade_print: print=2)\n");

    //**/ save the current values of the calibration parameters
    vecLast = vecXGuess;
    errFlag = 0;
    prior   = 1;

    //**/ generate proposal sample point
    if (useHist == 1 && fpHist != NULL) 
      printf("History %d inputs = ", mcmcIts);
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ only for uncertain parameters (length=0 means all
      //**/ are uncertain, or vecDesignP[ii] = 0 means uncertain
      if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
      {
        if (useHist == 1 && fpHist != NULL)
          fscanf(fpHist, "%lg", &ddata);
        else
        {
          normalPdf->genSample(iOne, &ddata, &dNeg5, &d5); 

          //**/ save the value of the current calibration parameters

          ddata = vecXGuess[ii] + ddata*propScale*(xUpper[ii]-xLower[ii]);
          if (ddata < xLower[ii]) 
          {
            ddata = xLower[ii];
            errFlag = 1;
          }
          if (ddata > xUpper[ii]) 
          {
            ddata = xUpper[ii];
            errFlag = 1;
          }
        }
        if (fpSave != NULL) fprintf(fpSave, "%24.16e ", ddata);
        vecXGuess[ii] = ddata;
        if (inputPDFs[ii] != NULL)
        {
          inputPDFs[ii]->getPDF(iOne, &ddata, &ddata2);
          prior *= ddata2;
        }
      }
    }
    prior = log(prior);

    //**/ evaluate and compute likelihood
    if (printLevel > 3) printf("MCMC_sim Iteration = %d : \n",mcmcIts); 
    if (useHist == 1 && fpHist != NULL)
    {
      //**/ if restart, just read the loglikelihood
      fscanf(fpHist, "%lg", &loglikelihood);
      printf(" - negative log(likelihood) = %e\n", loglikelihood);
    }
    else
    {
      //**/ otherwise, compute the loglikelihood
      loglikelihood = 0;
      for (ii = 0; ii < dnSamples; ii++)
      {
        if (printLevel > 3) 
          printf("    Experimental sample %d : \n",ii+1); 
        offset = 0;
        for (jj = 0; jj < nInputs; jj++)
        {
          if (vecDesignP.length() > 0 && vecDesignP[jj] == 1) 
          {
            vecXGuess[jj] = matExpInps.getEntry(ii,offset);
            offset++;
          }
        }
        ii2 = mcmcIts*dnSamples+ii+1;
        funcIO->evaluate(ii2,nInputs,vecXGuess.getDVector(),nOutputs,
                         vecFuncY.getDVector(),0);
        YT = 0;
        for (jj = 0; jj < nOutputs; jj++)
        {
          vecYGuess[ii*nOutputs+jj] = vecFuncY[jj];
          YT += pow(vecFuncY[jj]-matExpMeans.getEntry(ii,jj),2.0)/
                pow(matExpStdvs.getEntry(ii,jj),2.0);
        }
        if (printLevel > 3)
        {
          for (jj = 0; jj < nInputs; jj++)
            printf("        Input  %d = %e\n", jj+1, vecXGuess[jj]);
          for (jj = 0; jj < nOutputs; jj++)
            printf("        Output %d = %e\n", jj+1, vecFuncY[jj]);
        }
        loglikelihood += YT;
      }
    }
    if (fpSave != NULL) fprintf(fpSave, "%24.16e ", loglikelihood);
    loglikelihood /= (double) dnReps;
    if (printLevel > 3)
      printf("    Negative loglikelihood = %e\n",loglikelihood); 

    //**/ store negative best loglikelihood
    if (loglikelihood < minLoglikelihood)
    {
      vecXBest = vecXGuess;
      minLoglikelihood = loglikelihood;
      if (printLevel > 2)
      {
        printf("    Best solution so far (iteration = %d)\n",mcmcIts);
        for (jj = 0; jj < nInputs; jj++)
          printf("        Input  %d = %e\n", jj+1, vecXGuess[jj]);
        printf("    Best negative loglikelihood so far = %e\n",
               loglikelihood); 
      }
    }

    //**/ compute c11 = log(posterior)
    c11 = prior - 0.5 * loglikelihood;
    
    //**/ if restart, read accept (1) or reject (0)
    rejectOrAccept = 0;
    if (useHist == 1 && fpHist != NULL)
      fscanf(fpHist, "%d", &rejectOrAccept);

    //**/ first iteration
    if (c12 == PSUADE_UNDEFINED) c12 = c11;

    //**/ XGuess outside domain boundaries ==> should be reject
    //**/ errFlag can only be > 0 if history is not used (see above)
    if (errFlag > 0)
    {
      vecXGuess = vecLast;
      if (rejectOrAccept != 0)
      {
        printf("MCMC_sim ERROR: Proposal sample outside domain ");
        printf("but history says\n");
        printf("                otherwise (%d).\n",rejectOrAccept);
        exit(1);
      }
    }

    //**/ if XGuess is not outside domain,
    if (errFlag == 0)
    {
      //**/ if history says reject, then reject
      if (useHist == 1 && fpHist != NULL && rejectOrAccept == 0)
      {
        vecXGuess = vecLast;
      }
      //**/ if history says accept, then accept
      if (useHist == 1 && fpHist != NULL && rejectOrAccept == 1)
      {
        c12 = c11;
        if (mcmcIts > burnInSamples)
        {
          for (ii = 0; ii < nInputs; ii++)
            matXChains.setEntry(nAccept,ii,vecXGuess[ii]);
          matXChains.setEntry(nAccept,nInputs,loglikelihood);
          nAccept++;
          printf("     MCMC_sim its = %d: Accept (%e < %e?) - %% = %e\n",
                 mcmcIts,ddata,alpha,100.0*nAccept/(mcmcIts-burnInSamples+1));
        }
        nAcceptAll++;
      }
      //**/ if history is not used, do MH step
      if (useHist == 0)
      {
        c1 = c11 - c12;
        alpha = c1;
        if (alpha > 0) alpha = 0;

        ddata = log(PSUADE_drand());
        if (ddata < alpha)
        {
          c12 = c11;
          if (mcmcIts > burnInSamples)
          {
            for (ii = 0; ii < nInputs; ii++)
              matXChains.setEntry(nAccept,ii,vecXGuess[ii]);
            matXChains.setEntry(nAccept,nInputs,loglikelihood);
            nAccept++;
            printf("     MCMC_sim its = %d: Accept (%10.3e < %10.3e?) ",
                   mcmcIts,ddata,alpha);
            printf("-> = %4.2f %%\n",
                   100.0*nAccept/(mcmcIts-burnInSamples+1));
            rejectOrAccept = 1;
          }
          nAcceptAll++;
        }
        else
        {
          vecXGuess = vecLast;
        }
      }
    }
    if (fpSave != NULL) fprintf(fpSave, "%d ",rejectOrAccept);

    if (useHist == 0 && mcmcIts < burnInSamples)
    {
      ddata = 1.0 * nAcceptAll / mcmcIts;
      //**/ adjust to make sure acceptance rate is at least 50%
      //if (ddata < 0.25 && propScale > 0.01) propScale *= 0.5;
      //if (ddata > 0.75 && propScale < 1.0)  propScale *= 2.0;
      if (adaptive == 1)
      {
        if (ddata < 0.25) propScale *= 0.5;
        if (ddata > 0.75) propScale *= 2.0;
        if (propScale < 0.01) propScale = 0.01;
        if (propScale > 1.00) propScale = 1.00;
      }
      printf(" Acceptance rate so far (its,nc,ps=%d,%d,%5.3f) = %e %%\n",
             mcmcIts,nAccept,propScale,ddata*100);
    }
    if (useHist == 1 && fpHist != NULL)
      fscanf(fpHist, "%lg", &propScale);
    if (fpSave != NULL) fprintf(fpSave, "%e\n", propScale);

     //**/ compute convergence statistics
    if (mcmcIts % 100 == 0 && nAccept > 10)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
        {
          printOutTS(PL_INFO,
               "MCMC_sim: input %3d value at peak of likelihood = %e\n",
                     ii+1, vecXBest[ii]);
          vecMeans_[ii] = 0;
          for (jj = 0; jj < nAccept; jj++)
            vecMeans_[ii] += matXChains.getEntry(jj,ii);
          vecMeans_[ii] /= (double) nAccept;
          printOutTS(PL_INFO,"MCMC_sim: input %3d mean    = %e\n", ii+1,
                     vecMeans_[ii]);
          vecSigmas_[ii] = 0;
          for (jj = 0; jj < nAccept; jj++)
          {
            ddata = matXChains.getEntry(jj,ii);
            vecSigmas_[ii] += pow(ddata-vecMeans_[ii],2.0);
          }
          vecSigmas_[ii] /= (double) nAccept;
          vecSigmas_[ii] = sqrt(vecSigmas_[ii]);
          printOutTS(PL_INFO,"MCMC_sim: input %3d std dev = %e\n", ii+1,
                     vecSigmas_[ii]);
          vecMostLikelyInputs_[ii] = vecXBest[ii];
        }
      }
    }

    //**/ graceful termination
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      printOutTS(PL_ERROR,
        "MCMC_sim INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
      fclose(fp);
      fp = NULL;
      strcpy(pString, "psuade_stop");
      unlink(pString);
      break;
    }
    if (printLevel < 2)
    {
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        printOutTS(PL_INFO, "MCMC_sim INFO: print level set to 2\n");
        printLevel = 2;
        fclose(fp);
      }
    }
  }
  fclose(fpSave);
  if (fpHist != NULL) fclose(fpHist);
  printf("MCMC_sim Optimal Solution: \n");
  for (jj = 0; jj < vecXBest.length(); jj++)
  {
    if (vecDesignP.length() == 0 || vecDesignP[jj] == 0) 
      printf("   Input %d = %e\n",jj+1,vecXBest[jj]);
  }
  printf("Optimal negative log likelihood = %e\n",minLoglikelihood);

  //**/ ---------------------------------------------------------------
  //**/  MCMC completed, next step: generate the posterior sample file
  //**/ ---------------------------------------------------------------
  int genPosteriors=1;
  if (genPosteriors == 1)
  {
    fp = fopen("MCMCPostSample", "w");
    if (fp != NULL)
    {
      fprintf(fp, "PSUADE_BEGIN\n");
      //**/ count number of uncertain parameters
      int kk = 0;
      for (jj = 0; jj < nInputs; jj++)
         if (vecDesignP.length() == 0 || vecDesignP[jj] == 0) kk++;
      fprintf(fp, "%d %d\n", nAccept,kk);
      if (pdata.strArray_ != NULL)
      {
        fprintf(fp, "# ");
        for (jj = 0; jj < nInputs; jj++)
          if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
            fprintf(fp,"%s ", pdata.strArray_[jj]);
        fprintf(fp, "\n");
      }
      for (ii = 0; ii < nAccept; ii++)
      {
        fprintf(fp, "%d ", ii+1);
        for (jj = 0; jj < nInputs; jj++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
          {
            ddata = mcmcChain[ii][jj];
            fprintf(fp, "%e ", ddata);
          }
          //**/ May 2021: only include uncertain parameters
          //else if (vecDesignP.length() > 0 && vecDesignP[jj] != 0) 
          //  fprintf(fp, "%e ", 0.5 * (xUpper[jj] + xLower[jj]));
        }
        //**/ May 2021: only include uncertain parameters
        //fprintf(fp, "%e\n", mcmcChain[ii][nInputs_]);
        fprintf(fp, "\n");
      }
      fprintf(fp, "PSUADE_END\n");
      fprintf(fp, "Best negLogLikelihood = %e (Ideal=0)\n",
              minLoglikelihood);
      //**/ for the MLE solution, dissect the component of likelihood
      //**/ doesn't work yet. need RS
      //psIVector vecModelForms;
      //writeMLEInfo(fp, nInputs, nOutputs, NULL, vecDesignP.getIVector(), 
      //             NULL, NULL, vecXBest.getDVector(), dnSamples, dnInputs,
      //             matExpInps.getMatrix1D(), matExpMeans.getMatrix1D(), 
      //             matExpStdvs.getMatrix1D(), iZero, iZero,
      //             NULL, NULL, vecModelForms);
      fclose(fp);
    }
    printOutTS(PL_INFO,
         "MCMC_sim: 'MCMCPostSample' file has a posterior sample.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ create bins for prior and posterior and then create matlab file
  //**/ ---------------------------------------------------------------
  //**/ ---------------------------------
  //**/ create bins for posterior
  //**/ ---------------------------------
  int **bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  int ****bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  //**/ ---------------------------------
  //**/ binning for posterior
  //**/ ---------------------------------
  int index, index2;
  for (ii = 0; ii < nAccept; ii++) 
  {
    for (ii2 = 0; ii2 < nInputs; ii2++) 
    {
      ddata = (mcmcChain[ii][ii2] - xLower[ii2]) / 
              (xUpper[ii2] - xLower[ii2]);
      index = (int) (ddata * nbins);
      if (index >= nbins) index = nbins - 1;
      bins[index][ii2]++;
    }
    //**/ update the 2D histogram
    for (ii2 = 0; ii2 < nInputs; ii2++) 
    {
      ddata = (mcmcChain[ii][ii2] - xLower[ii2]) /
              (xUpper[ii2] - xLower[ii2]);
      index = (int) (ddata * nbins);
      if (index >= nbins) index = nbins - 1;
      for (jj = 0; jj < nInputs; jj++) 
      {
        YT = (mcmcChain[ii][jj] - xLower[jj]) /
             (xUpper[jj] - xLower[jj]);
        index2 = (int) (YT * nbins);
        if (index2 >= nbins) index2 = nbins - 1;
        bins2[index][index2][ii2][jj]++;
      }
    }
  }
  printf("MCMC_sim MLE: \n");
  for (ii = 0; ii < nInputs; ii++)
  {
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
      printOutTS(PL_INFO,"          Input %4d = %e\n",ii+1,
                 vecXBest[ii]);
  }
  printOutTS(PL_INFO,"          LogLikelihood = %e\n",minLoglikelihood);
 
  //**/ ---------------------------------
  //**/ create bins for prior
  //**/ ---------------------------------
  int **pbins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    pbins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) pbins[ii][jj] = 0;
  }
  int ****pbins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    pbins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      pbins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        pbins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          pbins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ ---------------------------------
  //**/ generate sample to mimic the prior
  //**/ ---------------------------------
  dataPtr->getParameter("method_sampling", pdata2);
  int methodSave = pdata2.intData_;
  dataPtr->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
  PDFManager *pdfman = new PDFManager();
  pdfman->initialize(dataPtr);
  psVector vecLB, vecUB, vecOut;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  int nSamps = 100000;
  vecOut.setLength(nSamps*nInputs);
  pdfman->genSample(nSamps, vecOut, vecLB, vecUB);
  dataPtr->updateMethodSection(methodSave,-1,-1,-1,-1);
  delete pdfman;

  //**/ ---------------------------------
  //**/ binning for prior
  //**/ ---------------------------------
  int kk, kk2;
  for (ii = 0; ii < nInputs; ii++)
  {
    YT = nbins / (xUpper[ii] - xLower[ii]);
    for (jj = 0; jj < nSamps; jj++)
    {
      ddata = vecOut[jj*nInputs+ii];
      ddata -= xLower[ii];
      ddata *= YT;
      kk = (int) ddata;
      if (kk >= nbins) kk = nbins - 1; 
      pbins[kk][ii]++;
    }
  }
  double Ytmp;
  for (ii = 0; ii < nInputs; ii++)
  {
    YT = nbins / (xUpper[ii] - xLower[ii]);
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      Ytmp = nbins / (xUpper[ii2] - xLower[ii2]);
      for (jj = 0; jj < nSamps; jj++)
      {
        ddata = vecOut[jj*nInputs+ii];
        ddata -= xLower[ii];
        ddata *= YT;
        kk = (int) ddata;
        if (kk >= nbins) kk = nbins - 1; 
        ddata = vecOut[jj*nInputs+ii2];
        ddata -= xLower[ii2];
        ddata *= Ytmp;
        kk2 = (int) ddata;
        if (kk2 >= nbins) kk2 = nbins - 1; 
        pbins2[kk][kk2][ii][ii2]++;
      }
    }
  }

  //**/ ---------------------------------
  //**/ create matlabmcmc2.m 
  //**/ ---------------------------------
  psVector vecRange;
  vecRange.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    vecRange[ii] = xUpper[ii] - xLower[ii];
  genMatlabFile(nInputs,xLower,xUpper,vecRange.getDVector(),nPlots,
        vecPlotIndices.getIVector(),nbins,pbins,pbins2,bins,bins2,
        pdata,iOne,nAccept,&mcmcChain,&iOne,vecXBest.getDVector(),0);

  //**/ ---------------------------------
  //**/ clean up
  //**/ ---------------------------------
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;
  for (ii = 0; ii < nbins; ii++) delete [] pbins[ii];
  delete [] pbins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] pbins2[jj][jj2][ii];
      delete [] pbins2[jj][jj2];
    }
    delete [] pbins2[jj];
  }
  delete [] pbins2;

  //**/ ---------------------------------------------------------------
  // final clean up
  //**/ ---------------------------------------------------------------
  if (inputPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (inputPDFs[ii] != NULL) delete inputPDFs[ii];
    delete [] inputPDFs;
  }
  return 0.0;
}

// ************************************************************************
// perform MCMC analysis using Metropolis Hasting
//**/ * Uses rejection algorithm
//**/ * Proposal distribution is symmetric == Metropolis
//**/ * If proposal distribution = target distribution ==> 100% accept
//**/ * To guarantee it converges to target distribution, must have:
//**/ * 1. target distribution must be invariant, which is satisfied by
//**/ *    the detailed balance (reversibility) property:
//**/ *       T(x'|x) p(x) = T(x|x')p(x')
//**/ *    That is, the probability from x to x' equal to from x' to x
//**/ *    Q: Why does reversibility imply invariance?
//**/ *          int T(x'|x) p(x) dx' = int T(x|x')p(x') dx'
//**/ *       Since int T(x'|x) dx' = 1
//**/ *       So p(x) = int T(x|x')p(x') dx' (this is invariance math)
//**/ *    Proof that MH rejection algorithm satisfies reversibility
//**/ *      T(x'|x) p(x) = p(x) q (x'|x) min(1,p(x')q(x|x')/p(x)q(x'|x))
//**/ *               = min(p(x) q(x'|x), p(x') q(x|x'))
//**/ *               = p(x') q(x|x') min(p(x) q(x'|x)/p(x') q(x|x'), 1)
//**/ *               = T(x|x') p(x')
//**/ * 
//**/ * 2. the MC must be ergodic: it converges to target distribution
//**/ *    irrespective of initial distribution
//**/ *    Note: if the chain is periodic, it is not ergodic
//**/ *    Note: if the chain is reducible, it is not ergodic
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_mh(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  nInputs_       = nInputs;
  nOutputs_      = nOutputs;
  int nSamples   = adata.nSamples_;
  double *sampIns = adata.sampleInputs_;
  double *samOuts = adata.sampleOutputs_;
  double *xLower  = adata.iLowerB_;
  double *xUpper  = adata.iUpperB_;
  int    *pdfTypes = adata.inputPDFs_;
  double *pdfMeans = adata.inputMeans_;
  double *pdfStdvs = adata.inputStdevs_;
  PsuadeData *dataPtr = adata.ioPtr_;
  //**/ get names for input (for use in plotting)
  pData qData, pPtr;
  if (dataPtr != NULL)
  {
    dataPtr->getParameter("input_names", qData);
    dataPtr->getParameter("ana_rsindexfile", pPtr);
    if (strcmp(pPtr.strArray_[0], "NONE"))
    {
      printf("MCMC_MH ERROR: Response surface index file not supported.\n");
      printf("               Remove rs_index_file from your ");
      printf("input file and re-run.\n");
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ display header 
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*                     MCMC (MH)\n");
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"* NOTE: Metropolis-Hastings does not support ");
  printOutTS(PL_INFO,"discrepancy modeling.\n");
  printEquals(PL_INFO, 0);
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
    printOutTS(PL_INFO," * ana_expert to finetune MCMC parameters, \n");
    printOutTS(PL_INFO,
         "   (e.g. sample size for burn-in can be adjusted).\n");
    printOutTS(PL_INFO,
         " * rs_expert to customize response surface for MCMC,\n");
    printDashes(PL_INFO,0);
    printOutTS(PL_INFO,
         "FEATURES AVAILABLE IN THE CURRENT VERSION OF MCMC:\n");
    printOutTS(PL_INFO," * Support other than uniform prior distributions\n");
    printOutTS(PL_INFO,
         " * Support likelihood functions from multiple outputs\n");
    printOutTS(PL_INFO,
         " * Option to set some inputs as design parameters\n");
    printOutTS(PL_INFO,
         "   - to be specified in the observation data spec file\n");
    printOutTS(PL_INFO,
         " * MCMC can be terminated gracefully by creating a file ");
    printOutTS(PL_INFO, "named\n");
    printOutTS(PL_INFO,
         "   'psuade_stop' in the same directory while it is running\n");
    printOutTS(PL_INFO,"   (if it takes too long).\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ clean up local memory allocations
  //**/ ---------------------------------------------------------------
  cleanUp();

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  //**/ ==> vecDesignP, matExpInps, matExpMeans, matExpStdvs
  //**/ ---------------------------------------------------------------
  int    dnReps=1;
  psIVector vecDesignP; 
  psMatrix  matExpInps, matExpMeans, matExpStdvs; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  if (dstatus != 0.0) return PSUADE_UNDEFINED;
  int dnSamples = matExpMeans.nrows();
  int dnInputs  = matExpInps.ncols();

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ This is an option to specify a rs index file in the data file.
  //**/ This option allows one to disable a certain input in the MCMC
  //**/ optimization ==> faPtrs, vecRSInds, vecRSVals.
  //**/ ---------------------------------------------------------------
  int faType;
  if (dataPtr != NULL)
  {
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
  }
  else
  {
    printOutTS(PL_INFO,
       "MCMC_MH INFO: ioPtr=NULL ==> assume MARS as reponse surface.\n");
    faType = PSUADE_RS_MARS;
  }
  printOutTS(PL_INFO,
       "MCMC_MH INFO: CREATING RESPONSE SURFACES FOR ALL OUTPUTS.\n");
  FuncApprox **faPtrs = new FuncApprox*[nOutputs];

  int ii, kk, iZero=0, status;
  psVector vecSamOut1;
  vecSamOut1.setLength(nSamples);
  for (ii = 0; ii < nOutputs; ii++)
  {
    faType = -1;
    printOutTS(PL_INFO,
       "MCMC_MH INFO: CREATING RESPONSE SURFACE FOR OUTPUT %d.\n",ii+1);
    faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
    faPtrs[ii]->setNPtsPerDim(16);
    faPtrs[ii]->setBounds(xLower, xUpper);
    faPtrs[ii]->setOutputLevel(0);
    for (kk = 0; kk < nSamples; kk++) 
      vecSamOut1[kk] = samOuts[kk*nOutputs+ii];

    status = faPtrs[ii]->initialize(sampIns, vecSamOut1.getDVector());
    if (status != 0)
    {
      printOutTS(PL_ERROR,
        "MCMC_MH ERROR: Unable to create response surface.\n");
      printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
      for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
      delete [] faPtrs;
      return PSUADE_UNDEFINED;
    }
  }
  int  rsErrFlag=0;
  char pString[1000], lineIn[1000];
  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    printf("Include RS uncertainty in inference ? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y') rsErrFlag = 1;
    fgets(lineIn,1000,stdin);
  }

  //**/ ---------------------------------------------------------------
  //**/ special set up (sample size) using interactive mode
  //**/ ==> burnInSamples, maxSamples, nbins, vecPlotInds, nPlots
  //**/     adaptiveMode - variable proposal scale
  //**/ ---------------------------------------------------------------
  int    nPlots, maxSamples=100000, burnInSamples=5000, nbins=20;
  int    adaptiveMode=1;
  double propScale=0.25;
  psIVector vecPlotInds;
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,
             "*** CURRENT SETTINGS OF MCMC_MH PARAMETERS: \n\n");
  printOutTS(PL_INFO,"Burn-in sample size         = %d\n", 
             burnInSamples);
  printOutTS(PL_INFO,"Maximum MCMC samples        = %d\n", 
             maxSamples);
  printOutTS(PL_INFO,"Proposal distribution scale = %e\n",
             propScale);
  printOutTS(PL_INFO,"No. of bins in histogram    = %d\n",nbins);
  printOutTS(PL_INFO,
   "NOTE: Proposal distribution is N(X_i,sig^2) and sig = scale.\n");
  printOutTS(PL_INFO,
   "NOTE: Histogram nBins - granularity of histogram bar graph\n");
  printOutTS(PL_INFO, 
   "Turn on ana_expert mode to change some of these default settings.\n\n");
  if (psConfig_.MasterModeIsOn())
  {
    snprintf(pString,100,
        "Enter the number of histogram bins (5 - 25) : ");
    nbins = getInt(5, 50, pString);
  }
  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    snprintf(pString,100,
        "Enter maximum iterations (10000 - 1000000): ");
    maxSamples = getInt(10000, 10000000, pString);
    snprintf(pString,100,
        "Proposal distribution scale (>0, <1, 0 = adaptive) : ");
    propScale = getDouble(pString);
    if (propScale == 0)
    {
      adaptiveMode = 1;
      propScale = 0.25;
    }
    else if (propScale < 0 || propScale > 1)
    {
      printf("MCMC_MH ERROR: Invalid proposal distribution scale.\n");
      printf("               Set to default = 0.25.\n");
      propScale = 0.25;
    }
  }
  if (psConfig_.MasterModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
      "*** OPTION TO CREATE POSTERIOR PLOTS FOR SELECTED INPUTS:\n\n");
    printOutTS(PL_INFO,
      "* MCMC creates MATLAB files for the posterior distributions.\n");
    printOutTS(PL_INFO, "* You can choose to generate posterior ");
    printOutTS(PL_INFO, "plots for all inputs, or just\n");
    printOutTS(PL_INFO,
      "  a selected few (in case there are too many inputs).\n");
    printf("Select which inputs posterior plots are to be created.\n");
    snprintf(pString,100,
      "Enter input number (-1 for all, 0 to terminate) : ");
    kk = 1;
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    while (kk != 0 || nPlots < 1)
    {
      kk = getInt(-1, nInputs, pString);
      if (kk == -1)
      {
        nPlots = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
            vecPlotInds[nPlots++] = ii;
        }
        break;
      }
      if (kk != 0 && nPlots < nInputs)
      {
        if (vecDesignP.length() > 0 && vecDesignP[kk-1] == 1) 
          printOutTS(PL_ERROR,
               "Input %d is a design parameter (no plot)\n",kk);
        else 
          vecPlotInds[nPlots++] = kk - 1;
      }
      if (kk == 0 && nPlots == 0)
        printOutTS(PL_ERROR,
          "Please select at least 1 input for plotting posteriors.\n");
    }
    if (nPlots > 1) sortIntList(nPlots, vecPlotInds.getIVector());
  }
  else
  {
    vecPlotInds.setLength(nInputs);
    nPlots = 0;
    for (ii = 0; ii < nInputs; ii++) 
      if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
        vecPlotInds[nPlots++] = ii;
  }
  printOutTS(PL_INFO, 
    "MCMC_MH Plot summary: inputs to be plotted are (%d):\n",nPlots);
  for (ii = 0; ii < nPlots; ii++)
    printOutTS(PL_INFO, "   Input %4d\n", vecPlotInds[ii]+1);

  //**/ ---------------------------------------------------------------
  // setup input PDF, if there is any
  //**/ inputPDFs will be needed later for prior
  //**/ ---------------------------------------------------------------
  if (printLevel > 2) 
    printf("*** INFORMATION ON PARAMETER PRIOR DISTRIBUTIONS\n");
  PDFBase **inputPDFs = new PDFBase*[nInputs];
  for (ii = 0; ii < nInputs; ii++)
  {
    inputPDFs[ii] = NULL;
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
    {
      if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_NORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFNormal(pdfMeans[ii], 
                                                  pdfStdvs[ii]);
        if (printLevel > 2) 
           printOutTS(PL_INFO,
                "Parameter %3d has normal prior distribution (%e,%e)\n",
                ii+1, pdfMeans[ii], pdfStdvs[ii]);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_LOGNORMAL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFLogNormal(pdfMeans[ii],
                                                     pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has lognormal prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_TRIANGLE)
      {
        inputPDFs[ii] = (PDFBase *) new PDFTriangle(pdfMeans[ii],
                                                    pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has triangle prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_BETA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFBeta(pdfMeans[ii],pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has beta prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_WEIBULL)
      {
        inputPDFs[ii] = (PDFBase *) new PDFWeibull(pdfMeans[ii],
                                                   pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has Weibull prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_GAMMA)
      {
        inputPDFs[ii] = (PDFBase *) new PDFGamma(pdfMeans[ii],
                                                 pdfStdvs[ii]);
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has gamma prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == 1000+PSUADE_PDF_NORMAL)
      {
        inputPDFs[ii] = NULL;
        printOutTS(PL_INFO,
             "Parameter %3d: multi-parameter normal distribution.\n",ii+1);
        printOutTS(PL_INFO,"               curently not supported.\n");
        return -1.0;
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == 1000+PSUADE_PDF_LOGNORMAL)
      {
        inputPDFs[ii] = NULL;
        printOutTS(PL_INFO,
           "Parameter %3d: multi-parameter lognormal distribution.\n",ii+1);
        printOutTS(PL_INFO,"               curently not supported.\n");
        return -1.0;
      }
      else if (pdfTypes == NULL || pdfTypes[ii] == PSUADE_PDF_UNIFORM)
      {
        inputPDFs[ii] = NULL;
        if (printLevel > 2)
          printOutTS(PL_INFO,
               "Parameter %3d has uniform prior distribution.\n",ii+1);
      }
      else if (pdfTypes != NULL && pdfTypes[ii] == PSUADE_PDF_SAMPLE)
      {
        inputPDFs[ii] = NULL;
        printOutTS(PL_INFO,
           "Parameter %3d: user-provided distribution currently not\n",ii+1);
        printOutTS(PL_INFO,"               supported.\n");
        return -1.0;
      }
    }
  }
  if (printLevel > 2) printEquals(PL_INFO, 0);

  //**/ ------------------------------------------------------------------
  //**/ get information on how many chains to use and threshold for
  //**/ convergence check
  //**/ ------------------------------------------------------------------
  int    numChains=3;
  double psrfThreshold=1.05;
  if (psConfig_.AnaExpertModeIsOn())
  {
    snprintf(pString,100,"How many MCMC chains? (1-20, default=1) : ");
    numChains = getInt(1,20,pString);
    if (numChains > 1)
    {
      snprintf(pString,100,"PSRF threshold? (1.0 - 1.2, default = 1.05) : ");
      psrfThreshold = getDouble(pString);
      if (psrfThreshold < 1.0 || psrfThreshold > 1.2)
      {
        printOutTS(PL_INFO,
             "MCMC : invalid PSRF threshold ==> reset to 1.05.\n");
        psrfThreshold = 1.05;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //    initialize object variables
  //**/ ---------------------------------------------------------------
  //**/ ----- for storing the input ranges
  psVector vecRange;
  vecRange.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    vecRange[ii] = xUpper[ii] - xLower[ii]; 
  //**/ ----- for keeping track of posterior statistics
  vecMeans_.setLength(nInputs_);
  vecSigmas_.setLength(nInputs_);
  vecMostLikelyInputs_.setLength(nInputs_);
  vecMostLikelyOutputs_.setLength(nOutputs_);
  //**/ ----- for keeping the point of maximum likelihood
  psVector vecXmax;
  vecXmax.setLength(nInputs);

  //**/ ---------------------------------------------------------------
  //**/ initialization for binning
  //**/ ---------------------------------------------------------------
  int ii2, jj;
  int **bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  int ****bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (kk = 0; kk < nbins; kk++)
    {
      bins2[jj][kk] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][kk][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][kk][ii][ii2] = 0;
      }
    }
  }

  //**/ ------------------------------------------------------------------
  //**/ generate LHS or LPTAU MCMC seed points for different chains
  //**/ ------------------------------------------------------------------
  Sampling *sampler;
  if (nInputs > 50) sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  else              sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
  sampler->setInputBounds(nInputs, xLower, xUpper);
  sampler->setOutputParams(1);
  sampler->setSamplingParams(numChains, 1, 0);
  sampler->initialize(0);
  psVector vecRanSeeds;
  vecRanSeeds.setLength(numChains*nInputs);
  psVector  vecOT;
  psIVector vecST;
  vecOT.setLength(numChains);
  vecST.setLength(numChains);
  sampler->getSamples(numChains,nInputs,1,vecRanSeeds.getDVector(),
                      vecOT.getDVector(),vecST.getIVector());
  delete sampler;

  //**/ ---------------------------------------------------------------
  //**/ for the accepted input values at each iteration and chain
  //**/ then store the system initial guesses in matXChains
  //**/ ---------------------------------------------------------------
  psMatrix **matXChains = new psMatrix*[numChains];
  for (ii = 0; ii < numChains; ii++)
  {
    matXChains[ii] = new psMatrix();
    matXChains[ii]->setDim(maxSamples*2,nInputs+1);
  }
  double ddata = -1;
  for (kk = 0; kk < numChains; kk++)
  {
    for (ii = 0; ii < nInputs; ii++)
      matXChains[kk]->setEntry(0,ii,vecRanSeeds[kk*nInputs+ii]);
    matXChains[kk]->setEntry(0,nInputs,ddata);
  }

  //**/ ---------------------------------------------------------------
  //**/ user may modify initial guess, if requested
  //**/ ---------------------------------------------------------------
  int igFlag=0;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printf("Set your own MCMC input initial values for Chain 1? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y')
    {
      printf("Enter initial parameter values below. If ");
      printf("you don't know what values to\n");
      printf("set, enter the suggested midpoints.\n");
      printf("NOTE: For multiple chain, this initial ");
      printf("guess is for Chain 1 only.\n");
      igFlag = 1;
    }
    fgets(pString,1000,stdin);
  }
  if (igFlag == 1)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
      {
        snprintf(pString,100,
             "Enter initial value for input %d (range=[%e,%e] : ",
             ii+1,xLower[ii],xUpper[ii]);
        ddata = getDouble(pString);
        matXChains[0]->setEntry(0,ii,ddata);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  // allocate memory for running the Metropolis Hasting algorithm
  //**/ vecXGuess is to be used for all dnSamples but vecXGuessLasst
  //**/ is used to store only one sample point but for all chains
  //**/ ---------------------------------------------------------------
  int    iOne=1;
  //**/ memory for proposal points and response surface evaluation
  psVector vecXGuess, vecYGuess, vecYStdev, vecXGuessLast;
  vecXGuess.setLength(dnSamples*nInputs);
  vecYGuess.setLength(dnSamples*nOutputs);
  vecYStdev.setLength(dnSamples*nOutputs);
  vecXGuessLast.setLength(nInputs*numChains);
  double *XGuess = vecXGuess.getDVector();
  double *YGuess = vecYGuess.getDVector();
  double *YStdev = vecYStdev.getDVector();
  for (kk = 0; kk < numChains; kk++)
  {
    for (ii = 0; ii < nInputs; ii++)
      vecXGuessLast[kk*nInputs+ii] = matXChains[kk]->getEntry(0,ii);
  }
  //**/ for generating random proposal point 
  double dZero=0, dOne=1.0;
  PDFBase *normalPtr = (PDFBase *) new PDFNormal(dZero,dOne);
  //**/ for MH acceptance accounting
  //**/ vecChainNAccepts - # of accepts for each chain except burnIn
  //**/ vecChainNAcceptsAll - number of accepts for each chain

  psIVector vecChainNAccepts, vecChainNAcceptsAll, vecChainCnts;
  vecChainNAccepts.setLength(numChains);
  vecChainNAcceptsAll.setLength(numChains);
  vecChainCnts.setLength(numChains);
  //**/ for convergence testing
  psVector vecChainMeans,vecChainStds,vecPsrfs;
  psVector vecTmpMeans,vecTmpStdvs;
  vecChainMeans.setLength(numChains);
  vecChainStds.setLength(numChains);
  vecPsrfs.setLength(nInputs);
  vecTmpMeans.setLength(3*nInputs);
  vecTmpStdvs.setLength(3*nInputs);
  double *tmpMeans = vecTmpMeans.getDVector();
  double *tmpStdvs = vecTmpStdvs.getDVector();
  //**/ MH parameters 
  psVector vecPropScale, vecC12;
  vecPropScale.setLength(numChains);
  vecC12.setLength(numChains);
  for (ii = 0; ii < numChains; ii++) vecPropScale[ii] = propScale;
  for (ii = 0; ii < numChains; ii++) vecC12[ii] = PSUADE_UNDEFINED;

  //**/ ---------------------------------------------------------------
  //**/ iterate
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printf("MCMC_MH begins ... \n");
  printf("NOTE: Proposal distribution scale has been set to %e.\n",
         propScale);
  if (adaptiveMode == 1)
    printf("      This scale will be adaptively modified during burn-in.\n"); 
  fflush(stdout);
  int mcmcIts=0, tmpIts, iChain, ss, samInc=maxSamples/100, convChkCnt=0;
  int count, errFlag, passCnt, mcmcFail=nInputs, postCnt, stopFlag;
  double prior, c11, c1, chiSq, dmean, dstdv, alpha, estdv, ddata2;
  double dFour=4.0, dFourM=-4, chiSqMin=PSUADE_UNDEFINED, WStat, BStat, Ymax; 
  double ddata3;
  FILE *fp=NULL;

  while ((vecChainNAccepts.sum() < maxSamples) && (mcmcIts < maxSamples*2))
  {
    stopFlag = 0;
    count = mcmcIts % samInc;
    //**/ if there is only one chain, no convergence check is done
    if (count == 0 && numChains == 1)
    {
      printAsterisks(PL_INFO, 0);
      printf("NOTE: MH will run %d iterations non-stop ",maxSamples);
      printf("even though convergence\n");
      printf("      will be performed. To terminate gracefully, ");
      printf("create an empty file\n");
      printf("      called psuade_stop in the work directory.\n");
      printAsterisks(PL_INFO, 0);
      fflush(stdout);
    }
    //**/ inner iteration
    for (iChain = 0; iChain < numChains; iChain++)
    {
      tmpIts = mcmcIts;
      //**/ obtain the last guess
      for (ii = 0; ii < nInputs; ii++) 
        vecXGuess[ii] = vecXGuessLast[iChain*nInputs+ii];

      //**/ repeat iteration for samInc times
      for (ss = 0; ss < samInc; ss++)
      {
        //**/ Save current values
        for (ii = 0; ii < nInputs; ii++) 
          vecXGuessLast[iChain*nInputs+ii] = vecXGuess[ii];

        //**/ draw a random point from normal distribution
        //**/ and put it into vecXGuess
        errFlag = 0;
        prior = 1.0;
        for (ii = 0; ii < nInputs; ii++) 
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
          {
            normalPtr->genSample(iOne, &ddata,&dFourM,&dFour);
            ddata = vecXGuess[ii]+ddata*vecPropScale[iChain]*vecRange[ii];
            if (ddata < xLower[ii] || ddata > xUpper[ii])
            {
              errFlag = 1;
              break;
            }
            vecXGuess[ii] = ddata;
            if (inputPDFs[ii] != NULL)
            {
              inputPDFs[ii]->getPDF(iOne, &ddata, &ddata2);
              prior *= ddata2;
            }
          }
        }
        prior = log(prior);

        //**/ if proposal is within range, move ahead 
        if (errFlag == 0)
        {
          //**/ fill up vecXGuess sample point
          for (kk = 0; kk < dnSamples; kk++)
          {
            if (kk > 0)
            {
              for (ii = 0; ii < nInputs; ii++)
                vecXGuess[kk*nInputs+ii] = vecXGuess[ii];
            }
            count = 0;
            for (ii = 0; ii < nInputs; ii++)
            {
              //**/ design inputs
              if (vecDesignP.length() > 0 && vecDesignP[ii] == 1) 
              {
                vecXGuess[kk*nInputs+ii] = matExpInps.getEntry(kk,count);
                count++;
              }
            }
          }

          //**/ evaluate proposal sample if it is within range 
          for (ii = 0; ii < nOutputs; ii++)
          {
            if (rsErrFlag == 0)
              faPtrs[ii]->evaluatePoint(dnSamples,XGuess,
                                        &(YGuess[ii*dnSamples]));
            else
              faPtrs[ii]->evaluatePointFuzzy(dnSamples,XGuess,
                     &(YGuess[ii*dnSamples]),&(YStdev[ii*dnSamples]));
          }

          //**/ compute chi-squared
          chiSq = 0.0;
          for (ii = 0; ii < nOutputs; ii++)
          {
            for (kk = 0; kk < dnSamples; kk++)
            {
              dmean = matExpMeans.getEntry(kk,ii);
              dstdv = matExpStdvs.getEntry(kk,ii);
              estdv = YStdev[ii*dnSamples+kk];
              if (rsErrFlag == 0)
                chiSq += pow((YGuess[ii*dnSamples+kk]-dmean)/dstdv,2.0);
              else
                chiSq += pow((YGuess[ii*dnSamples+kk]-dmean),2.0)/
                         (dstdv*dstdv+estdv*estdv);
            }
          }
          chiSq /= (double) dnReps;
          //**/ chiSqMin is a global variable for all chains
          if (chiSq < chiSqMin)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXmax[ii] = vecXGuess[ii];
            chiSqMin = chiSq;
          }

          /* compute c11 = log(posterior)  */
          c11 = prior - 0.5 * chiSq;
          if (vecC12[iChain] == PSUADE_UNDEFINED) vecC12[iChain] = c11;
          else 
          {
            //**/ NOTE: Since the proposal distribution is normal and
            //**/       thus a symmetric distribution, the rejection
            //**/       algorithm is reduced to the Metropolis algorithm
            //**/ compute log(p(x(t))/p(x(t-1)))  (vecC12 = p(x(t-1)))
            c1 = c11 - vecC12[iChain];
            alpha = c1;
            //**/ log(min(1,p(x(t))/p(x(t+1))))
            if (alpha > 0) alpha = 0;
            //**/ accept with probability alpha
            ddata = log(PSUADE_drand());
            if (ddata < alpha)
            {
              if (printLevel > 4)
                printf("     MCMC_MH chain %d, its = %d: Accept (%e < %e?)\n",
                       iChain+1,tmpIts,ddata,alpha);
              vecC12[iChain] = c11;
              //**/ keep track of how many are accepted after burn-in
              if (tmpIts > burnInSamples) vecChainNAccepts[iChain]++;
              //**/ keep track of how many are accepted including burn-in
              vecChainNAcceptsAll[iChain]++;
            }
            else
            {
              if (printLevel > 4)
                printf("     MCMC_MH chain %d, its = %d: Reject (%e > %e?)\n",
                       iChain+1, tmpIts,ddata,alpha);
              //**/ restore the last one
              for (ii = 0; ii < nInputs; ii++)
                vecXGuess[ii] = vecXGuessLast[iChain*nInputs+ii];
            }
          }
        }
        else
        {
          //**/ if proposal is out of range, restore the last one
          for (ii = 0; ii < nInputs; ii++)
            vecXGuess[ii] = vecXGuessLast[iChain*nInputs+ii];
        }

        //**/ put points in the posterior sample
        if (tmpIts >= burnInSamples)
        {
          postCnt = vecChainCnts[iChain];
          for (ii = 0; ii < nInputs; ii++)
            matXChains[iChain]->setEntry(postCnt,ii,XGuess[ii]);
          matXChains[iChain]->setEntry(postCnt,nInputs,chiSq);
          vecChainCnts[iChain]++;
        }

        //**/ adaptive step 
        tmpIts++;
        if (tmpIts < burnInSamples && adaptiveMode == 1)
        {
          ddata = 1.0 * vecChainNAcceptsAll[iChain] / tmpIts;
          //**/ adjust to make sure acceptance rate is at least 50%
          if (ddata < 0.25 && vecPropScale[iChain] > 0.01) 
          {
            vecPropScale[iChain] *= 0.5;
            printf("INFO: Chain %d, MCMC iter = %d, new prob. scale = %e\n",
                   iChain+1,tmpIts,vecPropScale[iChain]);
          }
          if (ddata > 0.75 && vecPropScale[iChain] < 1.0)
          {
            vecPropScale[iChain] *= 2.0;
            printf("INFO: Chain %d, MCMC iter = %d, new prob. scale = %e\n",
                   iChain+1,tmpIts,vecPropScale[iChain]);
          }
        }
      }

      //**/ store the current guess for this chain
      for (ii = 0; ii < nInputs; ii++) 
        vecXGuessLast[iChain*nInputs+ii] = vecXGuess[ii];

      if (tmpIts > burnInSamples)
      {
        ddata = 1.0 * vecChainNAccepts[iChain] / vecChainCnts[iChain];
        if (numChains > 1)
          printf("Chain %d: Current acceptance rate ", iChain+1);
        else
          printf("Current acceptance rate ");
        printf("(its,na,ps=%d,%d,%5.3f) = %4.2f %%\n",
               tmpIts,vecChainNAccepts[iChain],vecPropScale[iChain],
               ddata*100);
      }
    }
    mcmcIts += samInc;

    //**/ now compute convergence statistics (for numChains=1 only)
    if (mcmcIts > burnInSamples && numChains == 1 && 
        vecChainNAccepts[0] > 10)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, "Iteration %d summary: \n", mcmcIts);
      printDashes(PL_INFO, 0);
      //**/ compute individual input statistics
      for (ii = 0; ii < nInputs; ii++) 
      {
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
        {
          printOutTS(PL_INFO,
               "MCMC_MH: Input %3d value at peak of likelihood = %e\n",
               ii+1, vecXmax[ii]);
          vecMeans_[ii] = 0;
          for (jj = 0; jj < vecChainCnts[0]; jj++)
            vecMeans_[ii] += matXChains[0]->getEntry(jj,ii);
          vecMeans_[ii] /= (double) vecChainCnts[0];
          printOutTS(PL_INFO,"MCMC_MH: Input %3d mean    = %e\n", ii+1, 
                   vecMeans_[ii]);
          vecSigmas_[ii] = 0;
          for (jj = 0; jj < vecChainCnts[0]; jj++)
          {
            ddata = matXChains[0]->getEntry(jj,ii);
            vecSigmas_[ii] += pow(ddata-vecMeans_[ii],2.0);
          }
          vecSigmas_[ii] /= (double) vecChainCnts[0];
          vecSigmas_[ii] = sqrt(vecSigmas_[ii]);
          printOutTS(PL_INFO,"MCMC_MH: Input %3d std dev = %e\n", ii+1,
                     vecSigmas_[ii]);
          vecMostLikelyInputs_[ii] = vecXmax[ii];
          if (convChkCnt < 3)
          {
            vecTmpMeans[ii*3+convChkCnt] = vecMeans_[ii];
            vecTmpStdvs[ii*3+convChkCnt] = vecSigmas_[ii];
          }
          else
          {
            for (jj = 1; jj < convChkCnt; jj++)
            {
              vecTmpMeans[ii*3+jj-1] = vecTmpMeans[ii*3+jj];
              vecTmpStdvs[ii*3+jj-1] = vecTmpStdvs[ii*3+jj];
            }
            vecTmpMeans[ii*3+convChkCnt-1] = vecMeans_[ii];
            vecTmpStdvs[ii*3+convChkCnt-1] = vecSigmas_[ii];
          }
        }
      }

      //**/ compute best negative loglikelihood
      Ymax = PSUADE_UNDEFINED;
      for (iChain = 0; iChain < numChains; iChain++)
      {
        for (jj = 0; jj < vecChainCnts[iChain]; jj++)
        {
          ddata = matXChains[iChain]->getEntry(jj,nInputs);
          if (ddata < Ymax) Ymax = ddata;
        }
      }
      printOutTS(PL_INFO,
         "MCMC_MH: MLE Negative log likelihood = %e\n",Ymax);

      //**/ convergence check
      if (convChkCnt < 3) convChkCnt++;
      passCnt = 0;
      if (convChkCnt >= 3)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
          {
            postCnt = vecChainCnts.sum() / numChains;
            ddata = checkConvergence(3,&tmpMeans[ii*3],&tmpStdvs[ii*3],
                                     postCnt,psrfThreshold);
            if (ddata < psrfThreshold)
            {
              passCnt++;
              printf("MCMC_MH: Input %3d converged.\n",ii+1);
            }
          }
        }
      }
      if (vecDesignP.length() == 0) kk = nInputs;
      else                          kk = nInputs - vecDesignP.sum();
      if (passCnt == kk)
      {
        printf("NOTE: Convergence achieved, but MCMC ");
        printf("may continue if you want.\n");
        printf("Do you want to continue ? (y or n) ");
        scanf("%s", pString);
        if (pString[0] != 'y') stopFlag = 1; 
      }
      printDashes(PL_INFO, 0);
    }

    //**/ ----------------------------------------------------------------
    //**/ for numChains>1, compute convergence statistics (PSRF)
    //**/ This is done if the number of accepts is large enough (>=100)
    //**/ ----------------------------------------------------------------
    status = 1;
    for (iChain = 0; iChain < numChains; iChain++)
      if (vecChainNAccepts[iChain] < 100) status = 0; 
    if (status == 1 && numChains > 1)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, "Iteration %d summary: \n", mcmcIts);
      printDashes(PL_INFO, 0);
      //**/ compute basic statistics for each input
      for (ii = 0; ii < nInputs; ii++)
      {
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
        {
          //**/ update vecMeans_ and vecSigmas_
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            for (jj = 0; jj < vecChainCnts[iChain]; jj++)
               ddata += matXChains[iChain]->getEntry(jj,ii);
          }
          ddata /= (double) vecChainCnts.sum();
          vecMeans_[ii] = ddata;
          ddata2 = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            for (jj = 0; jj < vecChainCnts[iChain]; jj++)
              ddata2 += 
                pow(matXChains[iChain]->getEntry(jj,ii)-ddata,2.0);
          }
          ddata2 /= (double) (vecChainCnts.sum()-1);
          vecSigmas_[ii] = sqrt(ddata2);
          printOutTS(PL_INFO,
             "MCMC_MH: Input %3d value at peak of likelihood = %e\n",
                   ii+1, vecXmax[ii]);
          ddata = vecMeans_[ii];
          printOutTS(PL_INFO,
                     "MCMC_MH: Input %3d mean    = %e\n", ii+1, ddata);
          ddata = vecSigmas_[ii];
          printOutTS(PL_INFO,
                     "MCMC_MH: Input %3d std dev = %e\n", ii+1, ddata);
          vecMostLikelyInputs_[ii] = vecXmax[ii];
        }
      }

      //**/ find best negative log likelihood
      Ymax = PSUADE_UNDEFINED;
      for (iChain = 0; iChain < numChains; iChain++)
      {
        for (jj = 0; jj < vecChainCnts[iChain]; jj++)
        {
          ddata = matXChains[iChain]->getEntry(jj,nInputs);
          if (ddata < Ymax) Ymax = ddata;
        }
      }
      printOutTS(PL_INFO,
                 "MCMC_MH: MLE Negative log likelihood = %e\n",Ymax);

      //**/ convergence check
      mcmcFail = nInputs;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (vecDesignP.length() == 0 || vecDesignP[ii] == 0)
        {
          //**/ check that if a chain does not walk at all - turn off
          for (iChain = 0; iChain < numChains; iChain++)
          {
            ddata = 0.0;
            for (jj = 0; jj < vecChainCnts[iChain]; jj++)
              ddata += matXChains[iChain]->getEntry(jj,ii);
            ddata /= (double) vecChainCnts[iChain];
            ddata2 = 0.0;
            for (jj = 0; jj < vecChainCnts[iChain]; jj++)
               ddata2 += 
                 pow(matXChains[iChain]->getEntry(jj,ii)-ddata,2.0);
            ddata2 /= (double) (vecChainCnts[iChain] - 1);
            vecChainMeans[iChain] = ddata;
            vecChainStds[iChain] = ddata2;
          }
          //**/ compute PSRF
          postCnt = vecChainCnts.sum() / numChains;
          ddata = checkConvergence(numChains, vecChainMeans.getDVector(),
                        vecChainStds.getDVector(),postCnt,psrfThreshold); 
          if (ddata < psrfThreshold)
          {
            printOutTS(PL_INFO,
              "MCMC_MH: Input %d - PSRF %e < %e ==> converged.\n",
                       ii+1, ddata, psrfThreshold);
            mcmcFail--;
          }
        }
      }

      //**/ if all inputs convergence (twice consecutively), exit
      if (mcmcFail == 0 && mcmcIts > 1) 
      {
        printOutTS(PL_INFO,
                   "MCMC_MH INFO: All inputs converged.\n");
        convChkCnt++;
        if (convChkCnt > 2) break;
      }
      else convChkCnt = 0;
    }

    //**/ these are features for better run time diagnostics
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
           "MCMC_MH INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
      fclose(fp);
      fp = NULL;
      strcpy(pString, "psuade_stop");
      unlink(pString);
      break;
    }

    fp = fopen("psuade_print", "r");
    if (fp != NULL)
    {
      printOutTS(PL_INFO,
        "MCMC_MH INFO: psuade_print FILE FOUND. Set print level to 5.\n");
      fclose(fp);
      fp = NULL;
      printLevel = 5;
      strcpy(pString, "psuade_print");
      unlink(pString);
    }
    if (stopFlag == 1) break;
  }
  printf("MCMC_MH INFO: Posterior sample size = %d\n",vecChainCnts.sum());
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ create binning
  //**/ ---------------------------------------------------------------
  int index, index2;
  for (ii = 0; ii < nInputs; ii++) 
    for (jj = 0; jj < nbins; jj++) bins[jj][ii] = 0;
  for (ii = 0; ii < nInputs; ii++) 
    for (ii2 = 0; ii2 < nInputs; ii2++) 
      for (jj = 0; jj < nbins; jj++)
        for (kk = 0; kk < nbins; kk++)
          bins2[jj][kk][ii][ii2] = 0;
  for (iChain = 0; iChain < numChains; iChain++) 
  {
    for (ss = 0; ss < vecChainCnts[iChain]; ss++) 
    { 
      for (ii = 0; ii < nInputs; ii++) 
      {
        ddata = matXChains[iChain]->getEntry(ss,ii);
        ddata = (ddata - xLower[ii]) / (xUpper[ii] - xLower[ii]);
        index = (int) (ddata * nbins);
        if (index > nbins)
          printOutTS(PL_ERROR,
                     "MCMC_MH binning error 1 in file %s, line %d.\n",
                     __FILE__, __LINE__);
        if (index < 0)
          printOutTS(PL_ERROR,
                     "MCMC_MH binning error 2 in file %s, line %d.\n",
                     __FILE__, __LINE__);
        if (index >= nbins) index = nbins - 1;
        if (index <  0)     index = 0;
        bins[index][ii]++;
      }
      for (ii = 0; ii < nInputs; ii++) 
      {
        ddata = matXChains[iChain]->getEntry(ss,ii);
        ddata = (ddata - xLower[ii]) / (xUpper[ii] - xLower[ii]);
        index = (int) (ddata * nbins);
        if (index >= nbins) index = nbins - 1;
        if (index <  0)     index = 0;
        for (jj = 0; jj < nInputs; jj++) 
        {
          ddata2 = matXChains[iChain]->getEntry(ss,jj);
          ddata2 = (ddata2 - xLower[jj]) / (xUpper[jj] - xLower[jj]);
          index2 = (int) (ddata2 * nbins);
          if (index2 >= nbins) index2 = nbins - 1;
          if (index2 <  0)     index2 = 0;
          bins2[index][index2][ii][jj]++;
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate prior for matlab 
  //**/ ---------------------------------------------------------------
  dataPtr->getParameter("method_sampling", pPtr);
  int methodSave = pPtr.intData_;
  dataPtr->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
  PDFManager *pdfman = new PDFManager();
  pdfman->initialize(dataPtr);
  psVector vecLB, vecUB, vecOut;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  int nSamps = 100000;
  vecOut.setLength(nSamps*nInputs);
  pdfman->genSample(nSamps, vecOut, vecLB, vecUB);
  dataPtr->updateMethodSection(methodSave,-1,-1,-1,-1);
  delete pdfman;
  //**/ bins for prior
  int **pbins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    pbins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) pbins[ii][jj] = 0;
  }
  int ****pbins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    pbins2[jj] = new int**[nbins];
    for (kk = 0; kk < nbins; kk++)
    {
      pbins2[jj][kk] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        pbins2[jj][kk][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          pbins2[jj][kk][ii][ii2] = 0;
      }
    }
  }
  int    kk2;
  double ddata1;
  for (ii = 0; ii < nInputs; ii++)
  {
    ddata1 = nbins / (xUpper[ii] - xLower[ii]);
    for (jj = 0; jj < nSamps; jj++)
    {
      ddata = vecOut[jj*nInputs+ii];
      ddata -= xLower[ii];
      ddata *= ddata1;
      kk = (int) ddata;
      if (kk >= nbins) kk = nbins - 1; 
      pbins[kk][ii]++;
    }
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    ddata1 = nbins / (xUpper[ii] - xLower[ii]);
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      ddata2 = nbins / (xUpper[ii2] - xLower[ii2]);
      for (jj = 0; jj < nSamps; jj++)
      {
        ddata = vecOut[jj*nInputs+ii];
        ddata -= xLower[ii];
        ddata *= ddata1;
        kk = (int) ddata;
        if (kk >= nbins) kk = nbins - 1; 
        ddata = vecOut[jj*nInputs+ii2];
        ddata -= xLower[ii2];
        ddata *= ddata2;
        kk2 = (int) ddata;
        if (kk2 >= nbins) kk2 = nbins - 1; 
        pbins2[kk][kk2][ii][ii2]++;
      }
    }
  }
  genMatlabFile(nInputs,xLower,xUpper,vecRange.getDVector(),nPlots,
        vecPlotInds.getIVector(),nbins,pbins,pbins2,bins,bins2,qData,
        0,0,NULL,NULL, vecXmax.getDVector(),chiSqMin);
  for (ii = 0; ii < nbins; ii++) delete [] pbins[ii];
  delete [] pbins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (kk = 0; kk < nbins; kk++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] pbins2[jj][kk][ii];
      delete [] pbins2[jj][kk];
    }
    delete [] pbins2[jj];
  }
  delete [] pbins2;

  //**/ ---------------------------------------------------------------
  //**/  generate the posterior sample file
  //**/ ---------------------------------------------------------------
  int genPosteriors=1;
  if (genPosteriors == 1)
  {
    fp = fopen("MCMCPostSample", "w");
    if (fp != NULL)
    {
      fprintf(fp, "PSUADE_BEGIN\n");
      //**/ count number of uncertain parameters
      kk = 0;
      for (jj = 0; jj < nInputs; jj++)
        if (vecDesignP.length() == 0 || vecDesignP[jj] == 0) kk++; 
      fprintf(fp, "%d %d\n", vecChainCnts.sum(),kk);
      if (qData.strArray_ != NULL)
      {
        fprintf(fp, "# ");
        for (jj = 0; jj < nInputs; jj++)
          if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
            fprintf(fp,"%s ", qData.strArray_[jj]);
        fprintf(fp, "\n");
      }
      count = 0;
      for (iChain = 0; iChain < numChains; iChain++)
      {
        for (ss = 0; ss < vecChainCnts[iChain]; ss++)
        { 
          fprintf(fp, "%d ", count+ss+1);
          for (jj = 0; jj < nInputs; jj++)
          {
            if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
            {
              ddata = matXChains[iChain]->getEntry(ss,jj);
              fprintf(fp, "%e ", ddata);
            }
          }
          fprintf(fp, "\n");
        }
        count += vecChainCnts[iChain];
      }
      fprintf(fp, "PSUADE_END\n");
      fprintf(fp, "Best negLogLikelihood = %e (Ideal=0)\n",chiSqMin);

      //**/ for the MLE solution, dissect the component of likelihood
      psIVector vecModelForms;
      writeMLEInfo(fp,nInputs,nOutputs,faPtrs,vecDesignP.getIVector(), 
         NULL,NULL,vecXmax.getDVector(),dnSamples, dnInputs,
         matExpInps.getMatrix1D(),matExpMeans.getMatrix1D(),
         matExpStdvs.getMatrix1D(),iZero,iZero,NULL,NULL,vecModelForms);
      fclose(fp);
    }
    printOutTS(PL_INFO,
         "MCMC_MH: 'MCMCPostSample' file has a posterior sample.\n");
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  if (inputPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (inputPDFs[ii] != NULL) delete inputPDFs[ii];
    delete [] inputPDFs;
  }
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (kk = 0; kk < nbins; kk++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][kk][ii];
      delete [] bins2[jj][kk];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;
  if (matXChains != NULL)
  {
    for (iChain = 0; iChain < numChains; iChain++) 
      delete matXChains[iChain];
    delete [] matXChains;
  }
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++) 
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  return 0.0;
}

// ************************************************************************
// perform MCMC analysis with OMP
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyze_omp(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ display header 
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*          MCMC_OMP Optimizer\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  if (nOutputs != 1)
  {
    printOutTS(PL_ERROR,"MCMC_OMP ERROR: nOutputs > 1 not supported.\n");
    return PSUADE_UNDEFINED;
  } 
  int nSamples = adata.nSamples_;
  double *XIn = adata.sampleInputs_;
  double *YIn = adata.sampleOutputs_;
  double *XLowerB = adata.iLowerB_;
  double *XUpperB = adata.iUpperB_;
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
  PsuadeData *dataPtr = adata.ioPtr_;
  pData qData;
  if (dataPtr != NULL) dataPtr->getParameter("input_names", qData);
  nInputs_  = nInputs;
  nOutputs_ = nOutputs;

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  //**/ ---------------------------------------------------------------
  int dnReps=1;
  psIVector vecDesignP; 
  psMatrix  matExpInps, matExpMeans, matExpStdvs; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  int dnSamples = matExpMeans.nrows();
  int dnInputs  = matExpInps.ncols();
  if (dstatus != 0.0)
  {
    printOutTS(PL_ERROR,
         "MCMC_OMP ERROR: cannot read experimental data\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ------------------------------------------------------------
  //**/ option to add response surface uncertainties to data std dev
  //**/ ------------------------------------------------------------
  int  rsErrFlag=0;
  char lineIn[1001], pString[1001];
  if (psConfig_.AnaExpertModeIsOn())
  {
    snprintf(pString,100,
            "Include response surface uncertainties? (y or n) ");
    getString(pString, lineIn);
    if (lineIn[0] == 'y') rsErrFlag = 1;
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ special set up (sample size) using interactive mode
  //    ==> burnInSamples, maxSamples, nbins, vecPlotInds, nPlots
  //**/ ---------------------------------------------------------------
  int maxSamples = 10000;
  int burnInSamples = maxSamples / 2;
  int nbins = 20;
  if (psConfig_.InteractiveIsOn())
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** CURRENT SETTINGS OF MCMC PARAMETERS: \n\n");
    printOutTS(PL_INFO,"MCMC Burn-in sample size      (default) = %d\n",
               burnInSamples);
    printOutTS(PL_INFO,"MCMC sample increment         (default) = %d\n", 
               maxSamples);
    printOutTS(PL_INFO,"MCMC no. of bins in histogram (default) = %d\n", 
               nbins);
    printOutTS(PL_INFO,"NOTE: sample increment - sample size to run ");
    printOutTS(PL_INFO,"before convergence check\n");
    printOutTS(PL_INFO,"NOTE: histogram nBins  - define granularity of ");
    printOutTS(PL_INFO,"histogram bar graph\n");
  }
  psIVector vecPlotInds;
  vecPlotInds.setLength(nInputs);
  int nPlots=0, ii;
  for (ii = 0; ii < nInputs; ii++) 
    if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
      vecPlotInds[nPlots++] = ii;

  //**/ ---------------------------------------------------------------
  // setup input PDF, if there is any
  //**/ ---------------------------------------------------------------
  if (printLevel > 2) 
    printOutTS(PL_INFO,
         "*** INFORMATION ON PARAMETER PRIOR DISTRIBUTIONS\n");
  PDFBase **inputPDFs = new PDFBase*[nInputs];
  for (ii = 0; ii < nInputs; ii++)
  {
    inputPDFs[ii] = NULL;
    if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_NORMAL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFNormal(inputMeans[ii],
                                                inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has normal prior distribution (%e,%e).\n",
           ii+1, inputMeans[ii], inputStdevs[ii]);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_LOGNORMAL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFLogNormal(inputMeans[ii],
                                                   inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has lognormal prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_TRIANGLE)
    {
      inputPDFs[ii] = (PDFBase *) new PDFTriangle(inputMeans[ii],
                                                  inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has triangle prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_BETA)
    {
      inputPDFs[ii] = (PDFBase *) new PDFBeta(inputMeans[ii],
                                              inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has beta prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_WEIBULL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFWeibull(inputMeans[ii],
                                                 inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has Weibull prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_GAMMA)
    {
      inputPDFs[ii] = (PDFBase *) new PDFGamma(inputMeans[ii],
                                               inputStdevs[ii]);
      if (printLevel > 2) 
        printOutTS(PL_INFO,
           "Parameter %3d has gamma prior distribution.\n",ii+1);
    }
    else if (pdfFlags == NULL || pdfFlags[ii] == PSUADE_PDF_UNIFORM)
    {
      inputPDFs[ii] = NULL;
      printOutTS(PL_INFO,
           "Parameter %3d has uniform prior distribution.\n",ii+1);
    }
  }
  if (printLevel > 2) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ set max number of points in proposal distributions ==> maxPts
  //**/ set number of chains and threshold ==> numChains, psrfThreshold
  //**/ ---------------------------------------------------------------
  int    maxPts = nbins * 5;
  int    numChains = 3;
  double psrfThreshold=1.05;
  if (psConfig_.MasterModeIsOn())
  {
    printOutTS(PL_INFO,
         "*** SETTING PROPOSAL DISTRIBUTION RESOLUTION\n");
    printOutTS(PL_INFO,
         "Since MCMC uses many function evaluations to construct\n");
    printOutTS(PL_INFO,
         "the proposal distributions, you have the option to set\n");
    printOutTS(PL_INFO,
         "how many points are used to construct it in order to\n");
    printOutTS(PL_INFO,
         "keep the inference cost reasonable.\n");
    printf("Sample size to construct proposal distributions.\n");
    printf("Default is %d.\n",maxPts);
    snprintf(pString,100,
            "Enter new sample size (%d - %d): ",nbins*3,nbins*10);
    maxPts = getInt(nbins*3, nbins*10, pString);
    maxPts = maxPts / nbins * nbins;
    printOutTS(PL_INFO,
         "Proposal distribution sample size = %d.\n",maxPts);
  }
  if (psConfig_.AnaExpertModeIsOn())
  {

    snprintf(pString,100,"How many MCMC chains? (2-20, default=3) : ");
    numChains = getInt(2,20,pString);
    snprintf(pString,100,"PSRF threshold? (1.0 - 1.2, default = 1.05) : ");
    psrfThreshold = getDouble(pString);
    if (psrfThreshold < 1.0 || psrfThreshold > 1.2)
    {
      printf("MCMC_OMP : invalid PSRF threshold ==> reset to 1.05.\n");
      psrfThreshold = 1.05;
    }
  }

  //**/ ---------------------------------------------------------------
  //    set up for MCMC iterations
  //**/ ---------------------------------------------------------------
  int    iChain, maxGlobalIts=20, jj;
  double Ymax, ddata;
  //**/ ----- for storing the input ranges
  double *XRange  = new double[nInputs];
  for (ii = 0; ii < nInputs; ii++) 
    XRange[ii] = XUpperB[ii] - XLowerB[ii]; 
  //**/ ----- XDist is for proposal distribution, SDist for priors
  double *XDist = new double[maxPts+1];
  double *SDist = new double[maxPts+1];
  //**/ ----- for model evaluation
  double *XGuess  = new double[nInputs];
  double *XGuessS = new double[dnSamples*nInputs*(maxPts+1)];
  double *YGuessS = new double[dnSamples*nOutputs*(maxPts+1)];
  double *YGuessStds = new double[dnSamples*nOutputs*(maxPts+1)];
  //**/ ----- for discrepancy evaluation
  double *XDesignS = new double[dnSamples*nInputs*(maxPts+1)];
  double *YDesignS = new double[dnSamples*nOutputs*(maxPts+1)];
  double *YDesignStds = new double[dnSamples*nOutputs*(maxPts+1)];
  //**/ ----- for keeping the point of maximum likelihood
  double *Xmax = new double[nInputs];
  checkAllocate(Xmax, "Xmax in MCMC::analyze");
  vecMeans_.setLength(nInputs);
  vecSigmas_.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    Xmax[ii] = vecMeans_[ii] = vecSigmas_[ii] = 0;
  Ymax = -PSUADE_UNDEFINED;
  //**/ ----- for randomization of order of inputs to be processed
  int *Ivec = new int[nInputs];
  Ivec[nInputs-1] = -1;
  //**/ ----- for saving the input values at each iteration
  double ***XChains = new double**[numChains];
  for (ii = 0; ii < numChains; ii++)
  {
    XChains[ii] = new double*[maxGlobalIts*maxSamples];
    for (jj = 0; jj < maxGlobalIts*maxSamples; jj++)
      XChains[ii][jj] = new double[nInputs+1];
  }
  double *chainMeans = new double[numChains];
  double *chainStdevs = new double[numChains];
  checkAllocate(chainStdevs, "chainStdevs in MCMC::analyze");
  for (ii = 0; ii < numChains; ii++) 
    chainMeans[ii] = chainStdevs[ii] = 0.0;
  double *psrfs = new double[nInputs];
  checkAllocate(psrfs, "psrfs in MCMC::analyze");
  for (ii = 0; ii < nInputs; ii++) psrfs[ii] = 0.0;

  //**/ ---------------------------------------------------------------
  //**/ generate LHS or LPTAU MCMC seed points for different chains
  //**/ ---------------------------------------------------------------
  Sampling *sampler;
  if (nInputs > 50) sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  else              sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
  sampler->setInputBounds(nInputs, XLowerB, XUpperB);
  sampler->setOutputParams(1);
  sampler->setSamplingParams(numChains, 1, 1);
  sampler->initialize(0);
  psVector VecYT, VecMcmcSeeds;
  psIVector VecST;
  VecYT.setLength(numChains);
  VecST.setLength(numChains);
  VecMcmcSeeds.setLength(numChains*nInputs);
  sampler->getSamples(numChains,nInputs,1,VecMcmcSeeds.getDVector(),
                      VecYT.getDVector(), VecST.getIVector());
  delete sampler;
  //**/ normalize the sample
  for (iChain = 0; iChain < numChains; iChain++)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      ddata = VecMcmcSeeds[iChain*nInputs+ii];
      ddata = (ddata - XLowerB[ii]) / (XUpperB[ii] - XLowerB[ii]);
      VecMcmcSeeds[iChain*nInputs+ii] = ddata;
    }
  }
   
  //**/ ---------------------------------------------------------------
  // obtain response surface type
  //**/ ---------------------------------------------------------------
  int iOne=1, iZero=0, faType=-1, ii2, myPID=0;
  FuncApprox *faPtr = genFA(faType, nInputs, iZero, nSamples);
  faType = faPtr->getID();
  delete faPtr;
  faPtr = NULL;

  //**/ ---------------------------------------------------------------
  // run the Gibbs algorithm
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "MCMC_OMP begins ... \n");
  fflush(stdout);

  //**/ loop
  int    globalIts=0,chainCnt=0,status, kk, kk2,mcmcIts,mcmcFail=0;
  double Xtemp, Ytemp, Ytemp2, stdev, stdev2, ddata2, WStat, BStat;
  while (globalIts < maxGlobalIts && mcmcFail != 0) break;
  {
#pragma omp parallel shared(XChains, globalIts) \
    private(iChain,mcmcIts,Ivec,ii,jj,kk,ii2, \
            kk2,Xtemp,XGuess,XGuessS,XDesignS,YDesignS,YDesignStds, \
            YGuessS,YGuessStds,XDist,Ytemp,Ytemp2,stdev,stdev2,faPtr, \
            Xmax,Ymax,chainCnt,status)
    {
      //**/ construct response surface and set initial guess
      if (globalIts == 0)
      {
        faPtr = genFA(faType, nInputs, iZero, nSamples);
        faPtr->setNPtsPerDim(16);
        faPtr->setBounds(XLowerB, XUpperB);
        faPtr->setOutputLevel(0);
        double *YY = new double[nSamples];
        status = faPtr->initialize(XIn, YY);
        //**/ only supports one output
        ii = 0;
        for (kk = 0; kk < nSamples; kk++) YY[kk] = YIn[kk*nOutputs+ii];
        status = faPtr->initialize(XIn, YY);
        delete [] YY;

        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
               XGuess[ii] = VecMcmcSeeds[iChain*nInputs+ii];
          else XGuess[ii] = 0.5;
          XChains[iChain][0][ii] = XGuess[ii];
          XGuess[ii] = XGuess[ii]*(XUpperB[ii]-XLowerB[ii])+XLowerB[ii];
        }
        XChains[iChain][0][nInputs] = -1;
      }
      
      //**/ process each chain
#pragma omp for
      for (iChain = 0; iChain < numChains; iChain++)
      {
        mcmcIts = 0;
        for (mcmcIts = 0; mcmcIts < maxSamples; mcmcIts+=nInputs)
        {
          //**/ prevent sampling same input in consecutive steps
          jj = Ivec[nInputs-1];
          generateRandomIvector(nInputs, Ivec);
          if (Ivec[0] == jj && nInputs > 1)
          {
            Ivec[0] = Ivec[nInputs-1];
            Ivec[nInputs-1] = jj;
          }
          //**/ Gibbs steps
          int    ompCnt=0, ompInd, ompInd2, ompDCnt;
          double ompDdata, ompDdata2;
          for (kk = 0; kk < nInputs; kk++)
          {
            ii = Ivec[kk];
            if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
            {
              //**/ store away current value of parameter ii
              Xtemp = XGuess[ii];
  
              //**/ creating a CDF for the current input 
              //**/ CDF will be put into XDist[0:maxPts]
              for (jj = 0; jj <= maxPts; jj++)
              {
                XGuess[ii] = XLowerB[ii]+jj*XRange[ii]/maxPts;
                ompCnt++;
                ompInd = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  ompDCnt = 0;
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                  {
                    XGuessS[(ompInd+kk2)*nInputs+ii2] = XGuess[ii2];
                    if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1) 
                    {
                      XGuessS[(ompInd+kk2)*nInputs+ii2] = 
                             matExpInps.getEntry(kk2,ompDCnt);
                      XDesignS[(ompInd+kk2)*dnInputs+ompDCnt] = 
                             matExpInps.getEntry(kk2,ompDCnt);
                      ompDCnt++;
                    }
                  }
                }
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  YDesignS[ompInd+kk2] = YDesignStds[ompInd+kk2] = 0.0;
                  YGuessS[ompInd+kk2] = YGuessStds[ompInd+kk2] = 0.0;
                }
              }

              //**/ run XGuessS and XDesignS through response surfaces
              if (rsErrFlag == 1)
              {
                faPtr->evaluatePointFuzzy((maxPts+1)*dnSamples,
                                        XGuessS,YGuessS,YGuessStds);
              }
              else
              {
                faPtr->evaluatePoint((maxPts+1)*dnSamples,
                                     XGuessS,YGuessS);
              }

              //**/ compute distributions XDist
              for (jj = 0; jj <= maxPts; jj++)
              {
                ompInd = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                  YGuessS[ompInd+kk2] += YDesignS[ompInd+kk2];

                XDist[jj] = 0.0;
                for (kk2 = 0; kk2 < dnSamples; kk2++)
                {
                  Ytemp = YGuessS[ompInd+kk2];
                  stdev = YGuessStds[ompInd+kk2];
                  stdev2 = YDesignStds[ompInd+kk2];
                  Ytemp2 = pow(Ytemp-matExpMeans.getEntry(kk2,0),2.0) /
                          (pow(matExpStdvs.getEntry(kk2,0),2.0)+
                           stdev*stdev+stdev2*stdev2);
                  XDist[jj] += Ytemp2;
                }
                XDist[jj] = XDist[jj] / (double) dnReps;
              }

              //**/ compute prior SDist
              ompDdata = 1.0;
              if (inputPDFs != NULL)
              {
                for (ii2 = 0; ii2 < nInputs; ii2++)
                {
                  if ((vecDesignP.length() == 0 || vecDesignP[ii2] == 0) && 
                       inputPDFs[ii2] != NULL)
                  {
                    inputPDFs[ii2]->getPDF(iOne,&XGuess[ii2],&ompDdata2);
                    ompDdata *= ompDdata2;
                  }
                }
              }
              SDist[jj] = ompDdata; 
            }

            //**/ now XDist and SDist are ready
            //**/ find the max exp(-0.5*XDist*SDist) for normalization
            ompDdata = XDist[0];
            for (jj = 1; jj <= maxPts; jj++) 
            {
              ompDdata2 = XDist[jj];
              if (ompDdata2 < ompDdata) ompDdata = ompDdata2;
            }

            //**/ store away the normalization factor for weighting later
            XChains[iChain][chainCnt][nInputs] = ompDdata;
            for (jj = 0; jj <= maxPts; jj++)
            {
              XDist[jj] = SDist[jj] * exp(-0.5 * (XDist[jj] - ompDdata));
              ompDdata2 = XDist[jj] * exp(-0.5*ompDdata);
              if (ompDdata2 > Ymax)
              {
                Ymax = ompDdata2;
                for (ii2 = 0; ii2 < nInputs; ii2++) 
                  Xmax[ii2] = XGuess[ii2];
                Xmax[ii] = XLowerB[ii] + 
                           (XUpperB[ii]-XLowerB[ii]) * jj / maxPts;
              }
              if (jj > 0) XDist[jj] += XDist[jj-1];
            }

            //**/ if the distribution is nonzero, select a point for X[ii]
            if (XDist[maxPts] - XDist[0] > 0.0e-24)
            {
              //**/ normalize XDist to [0,1] to make it a CDF
              for (jj = 1; jj <= maxPts; jj++)
                XDist[jj] = (XDist[jj]-XDist[0])/(XDist[maxPts]-XDist[0]);
              XDist[0] = 0;
              XGuess[ii] = drand48();
              ompInd = binarySearchDble(XGuess[ii], XDist, maxPts+1);
              //if (ompInd == maxPts) ompInd = maxPts - 1;
              if (ompInd >= 0) ompDdata = (double) ompInd;
              else
              {
                ompInd = - ompInd - 1;
                if (PABS(XDist[ompInd]-XDist[ompInd+1]) > 1.0e-16)
                  ompDdata=ompInd + (XGuess[ii]-XDist[ompInd])/
                                (XDist[ompInd+1]-XDist[ompInd]);
                else ompDdata = (double) ompInd;
              }
              //**/ create the next guess values
              XGuess[ii] = XLowerB[ii]+ompDdata*XRange[ii]/maxPts;
            }

            //**/ leave the first maxSamples/2 points as burn-in
            if (mcmcIts >= maxSamples/2 || globalIts > 0) chainCnt++;
          }
        }
      } // for iChain
 
      //**/ now compute convergence statistics
#pragma omp critical
#ifdef PSUADE_OMP
      if (omp_get_thread_num() == 0)
#else
      if (myPID == 0)
#endif
      {
        globalIts++;
        printEquals(PL_INFO, 0);
        printOutTS(PL_INFO, "Iteration %d summary: \n", globalIts);
        printDashes(PL_INFO, 0);
        mcmcFail = nInputs - dnInputs;
        for (ii = 0; ii < nInputs; ii++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
          {
            if (printLevel > 2) printOutTS(PL_INFO, "Input = %d\n", ii+1);
      
            for (iChain = 0; iChain < numChains; iChain++)
            {
              ddata = 0.0;
              for (jj = 0; jj < chainCnt; jj++) 
                ddata += XChains[iChain][jj][ii];
              ddata /= chainCnt;
              ddata2 = 0.0;
              for (jj = 0; jj < chainCnt; jj++) 
                ddata2 += pow(XChains[iChain][jj][ii]-ddata,2.0);
              ddata2 /= (double) (chainCnt - 1);
              chainMeans[iChain] = ddata;
              chainStdevs[iChain] = ddata2;
            }
          }
          //**/ compute PSRF
          WStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            WStat += chainStdevs[iChain];
          WStat /= (double) numChains;
          if (WStat < 0) WStat = PSUADE_UNDEFINED;
          if (printLevel > 2) 
             printf("  Within  chain variance W = %e\n", WStat);
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            ddata += chainMeans[iChain];
          ddata /= (double) numChains;
          BStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
            BStat += pow(chainMeans[iChain]-ddata,2.0);
          BStat = BStat / (numChains - 1.0) * chainCnt;
          if (printLevel > 2) 
            printf("  Between chain variance B = %e\n", BStat/chainCnt);
          ddata = (1 - 1.0/chainCnt) * WStat + BStat / chainCnt;
          ddata = ddata / WStat * (numChains + 1) / numChains - 
                  (chainCnt - 1.0) / (double) (chainCnt * numChains); 
          if (ddata < 0) ddata2 = PSUADE_UNDEFINED;
          else           ddata2 = sqrt(ddata);
          if (printLevel > 2)
          {
            for (iChain = 0; iChain < numChains; iChain++)
              printOutTS(PL_INFO,"  Chain %4d statistics = %16.8e %16.8e\n",
                 iChain+1, chainMeans[iChain]*XRange[ii]+XLowerB[ii],
                 chainStdevs[iChain]*XRange[ii]*XRange[ii]);
            printf("  Chain length             = %d\n", chainCnt);
            printf("  Weighted average of B, W = %e\n", ddata);
          }
          printf("  Input %d PSRF = %e\n", ii+1, ddata2);
          psrfs[ii] = ddata2;
          if (ddata2 < psrfThreshold)
          {
            printOutTS(PL_INFO,"MCMC_OMP INFO : PSRF < %e ==> converged.\n",
                       psrfThreshold);
            mcmcFail--;
          }
          //**/ update 
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            for (jj = 0; jj < chainCnt; jj++)
              ddata += XChains[iChain][jj][ii];
          }
          ddata /= (double) (numChains * chainCnt);
          vecMeans_[ii] = ddata;
          ddata2 = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            for (jj = 0; jj < chainCnt; jj++)
              ddata2 += pow(XChains[iChain][jj][ii]-ddata,2.0);
          }
          ddata2 /= (double) (chainCnt*numChains-1);
          vecSigmas_[ii] = sqrt(ddata2);
        }

        //**/ output statistics
        for (ii = 0; ii < nInputs; ii++) 
        {
          if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
          {
            printOutTS(PL_INFO,
                 "MCMC_OMP: input %3d value at peak of likelihood = %e\n",
                 ii+1, Xmax[ii]);
            ddata = vecMeans_[ii]*(XUpperB[ii]-XLowerB[ii])+XLowerB[ii];
            printOutTS(PL_INFO,"MCMC_OMP: input %3d mean    = %e\n",ii+1,ddata);
            ddata = vecSigmas_[ii]*(XUpperB[ii]-XLowerB[ii]);
            printOutTS(PL_INFO,"MCMC_OMP: input %3d std dev = %e\n",ii+1,ddata);
          }
        }
        printEquals(PL_INFO, 0);
      } // if myPID == 0

      //**/ otherwise, fetch from the chain
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          ddata = XChains[iChain][chainCnt-1][ii];
          XGuess[ii] = ddata * (XUpperB[ii] - XLowerB[ii]) + XLowerB[ii];
        }
      }
    } // omp parallel
  }
  if (globalIts >= maxGlobalIts)
  {
    mcmcFail = 0;
    for (ii = 0; ii < nInputs; ii++) 
    {
      if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
        if (psrfs[ii] > psrfThreshold) mcmcFail = 1;
    }
    if (mcmcFail == 1 && myPID == 0) 
      printOutTS(PL_INFO,
           "MCMC_OMP maximum iterations exceeded but no convergence.\n");
  }
  else if (myPID == 0) printOutTS(PL_INFO, "MCMC iterations completed\n");

  //**/ ---------------------------------------------------------------
  //**/ some cleaning
  //**/ ---------------------------------------------------------------
  delete [] XDist;
  delete [] SDist;
  delete [] XGuess;
  delete [] XGuessS;
  delete [] YGuessS;
  delete [] YGuessStds;
  delete [] XDesignS;
  delete [] YDesignS;
  delete [] YDesignStds;
  if (inputPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (inputPDFs[ii] != NULL) delete inputPDFs[ii];
    delete [] inputPDFs;
  }
  delete [] psrfs;
  if (faPtr != NULL) delete faPtr;
  delete [] Ivec;
  delete [] Xmax;

  //**/ ---------------------------------------------------------------
  //**/ create binning
  //**/ ---------------------------------------------------------------
  int ii3, jj2, index, index2;
  int **bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  int ****bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }
  for (iChain = 0; iChain < numChains; iChain++) 
  { 
    for (jj = 0; jj < chainCnt; jj++) 
    {
      for (ii2 = 0; ii2 < nInputs; ii2++) 
      {
        ddata = XChains[iChain][jj][ii2];
        index = (int) (ddata * nbins);
        if (index >= nbins) index = nbins - 1;
        bins[index][ii2]++;
      }
      for (ii2 = 0; ii2 < nInputs; ii2++) 
      {
        ddata = XChains[iChain][jj][ii2];
        index = (int) (ddata * nbins);
        if (index >= nbins) index = nbins - 1;
        for (ii3 = 0; ii3 < nInputs; ii3++) 
        {
          ddata2 = XChains[iChain][jj][ii3];
          index2 = (int) (ddata2 * nbins);
          if (index2 >= nbins) index2 = nbins - 1;
          bins2[index][index2][ii2][ii3]++;
        }
      }
    }
  }
  
  //**/ ------------------------------------------------------------
  //**/ generate matlab posterior files 
  //**/ ------------------------------------------------------------
  int *chainStatus = new int[nInputs];
  genMatlabFile(nInputs,XLowerB,XUpperB,XRange,nPlots,
        vecPlotInds.getIVector(),nbins,NULL,NULL,bins,bins2,qData,
        numChains,chainCnt,XChains,chainStatus,Xmax,Ymax);
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;

  //**/ ---------------------------------------------------------------
  //**/  generate the log likelihood distribution 
  //**/ ---------------------------------------------------------------
  int tmpcnt = numChains * chainCnt;
  if (tmpcnt > 200000) tmpcnt = 200000;
  tmpcnt /= numChains;
  psIVector VecChainStat, vecIT;
  vecIT.setLength(iOne);
  VecChainStat.setLength(nInputs);
  genPostLikelihood(nInputs,XLowerB,XUpperB,XRange,numChains,chainCnt,
         XChains, VecChainStat.getIVector(),tmpcnt,NULL,NULL,
         vecDesignP,dnInputs,dnSamples,matExpInps,&faPtr,nOutputs, 
         NULL, NULL, matExpMeans, matExpStdvs, 0, 0, vecIT);
  if (chainStatus != NULL) delete [] chainStatus;

  //**/ ------------------------------------------------------------
  //**/  generate the posterior sample file
  //**/ ------------------------------------------------------------
  FILE *fp = fopen("MCMCPostSample", "w");
  if (fp != NULL)
  {
    fprintf(fp, "PSUADE_BEGIN\n");
    tmpcnt = numChains * chainCnt;
    if (tmpcnt > 200000) tmpcnt = 200000;
    tmpcnt /= numChains;
    //**/ count number of uncertain parameters
    kk = 0;
    for (jj = 0; jj < nInputs; jj++)
      if (vecDesignP.length() == 0 || vecDesignP[jj] == 0) kk++;
    fprintf(fp, "%d %d\n", tmpcnt*numChains,kk);
    if (qData.strArray_ != NULL)
    {
      fprintf(fp, "# ");
      for (jj = 0; jj < nInputs; jj++)
        if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
          fprintf(fp,"%s ", qData.strArray_[jj]);
      fprintf(fp, "\n");
    }
    ii2 = 0;
    for (iChain = 0; iChain < numChains; iChain++)
    { 
      for (ii = chainCnt-tmpcnt; ii < chainCnt; ii++)
      {
        fprintf(fp, "%d ", ii2+1);
        for (jj = 0; jj < nInputs; jj++)
        {
          if (vecDesignP.length() == 0 || vecDesignP[jj] == 0)
          {
            ddata = XChains[iChain][ii][jj] * 
                    XRange[jj] + XLowerB[jj];
            fprintf(fp, "%e ", ddata);
          }
        }
        fprintf(fp, "\n");
        ii2++;
      }
    }
    fprintf(fp, "PSUADE_END\n");
    fprintf(fp, "#N=%d;\n",numChains*tmpcnt);
    fprintf(fp, "#m=%d;\n",tmpcnt);
    for (iChain = 0; iChain < numChains; iChain++)
    {
      fprintf(fp, "#A%d = A(%d*m+1:%d*m,:);\n",iChain,
              iChain,iChain+1);
    }
    fprintf(fp, "#for ii = 2 : %d\n", nInputs+1);
    for (iChain = 0; iChain < numChains; iChain++)
    {
      fprintf(fp, "#subplot(*,*,%d)\n",iChain+1);
      fprintf(fp, "#hist(A%d(:,ii))\n",iChain+1);
      fprintf(fp, "#ii-1\n");
      fprintf(fp, "#pause;\n");
    }
    fprintf(fp, "#end;\n");
    fclose(fp);
  }
  printOutTS(PL_INFO,
        "MCMC_OMP: 'MCMCPostSample' file has a posterior sample.\n");

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < numChains; ii++) 
  {
    for (jj = 0; jj < maxGlobalIts*maxSamples; jj++) 
      delete [] XChains[ii][jj];
    delete [] XChains[ii];
  }
  delete [] XChains;
  delete [] XRange;
  return 0.0;
}

// ************************************************************************
// perform MCMC-like analysis (brute force): for direct call externally
// (e.g. from ODOE module) assuming some input parameters are certain 
// and some may be fixed uncertain parameters.
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyzeDirect(McmcData &mdata)
{
  int    ii, ii2, jj, kk, kk2, status, iZero=0, cnt, dnReps=1;
  double ddata;

  //**/ ---------------------------------------------------------------
  //**/ nFUInputs - number of fixed uncertain parameters
  //**/ VecFUInputs - a list of fixed certain parameters
  //**/ MatFUInpSample - sample for the fixed uncertain parameters
  //**/ nCUInputs - number of calibration parameters
  //**/ VecCUInputs - a list of calibration parameters
  //**/ MatPriorSample - prior sample for the calibration parameters
  //**/ MatExpInputs - experimental sample inputs
  //**/ MatExpMeans - experimental sample output means
  //**/ MatExpStds - experimental sample output standard deviation
  //**/ MatPostSample - posterior sample
  //**/ ---------------------------------------------------------------

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  int nFUInputs = mdata.VecFUInputs_.length();
  if (nFUInputs > 0)
  {
    for (ii = 0; ii < nFUInputs; ii++)
    {
      if (mdata.VecFUInputs_[ii] < 0 || 
          mdata.VecFUInputs_[ii] >= mdata.nInputs_)
      {
        printf("MCMC ERROR: FixedUInput list problem.\n");
        printf("            Input = %d is invalid\n",
             mdata.VecFUInputs_[ii]+1);
        exit(1);
      }
    }
    if (mdata.MatFUInpSample_.ncols() != nFUInputs)
    {
      printf("MCMC ERROR: FixedUInput sample problem.\n");
      printf("            Sample matrix has = %d columns\n",
           mdata.MatFUInpSample_.ncols());
      printf("            Expected  = %d\n", nFUInputs);
      exit(1);
    }
  }

  int nCUInputs = mdata.VecCUInputs_.length();
  for (ii = 0; ii < nCUInputs; ii++)
  {
    if (mdata.VecCUInputs_[ii] < 0 || 
        mdata.VecCUInputs_[ii] >= mdata.nInputs_)
    {
      printf("MCMC ERROR: Calibration input list problem.\n");
      printf("            Input = %d is invalid\n",
             mdata.VecCUInputs_[ii]+1);
      exit(1);
    }
  }
  if (mdata.MatPriorSample_.ncols() != nCUInputs)
  {
    printf("MCMC ERROR: Prior sample problem.\n");
    printf("            Prior sample has = %d columns\n",
           mdata.MatPriorSample_.ncols());
    printf("            Expected  = %d\n", nCUInputs);
    exit(1);
  }

  int nDInputs = mdata.nInputs_ - nFUInputs - nCUInputs;
  int nExperiments = mdata.MatExpMeans_.nrows();
  if (mdata.MatExpInputs_.ncols() != nDInputs)
  {
    printf("MCMC ERROR: Experiment sample problem.\n");
    printf("            Experiment matrix has = %d columns\n",
           mdata.MatExpInputs_.ncols());
    printf("            Expected  = %d\n", nDInputs);
    exit(1);
  }
  if (mdata.MatExpInputs_.ncols() > 0 &&
      mdata.MatExpInputs_.nrows() != mdata.MatExpMeans_.nrows())
  {
    printf("MCMC ERROR: Experiment sample size and means mismatch\n");
    printf("            Experiment input matrix has = %d rows\n",
           mdata.MatExpInputs_.nrows());
    printf("            Experiment means matrix has = %d rows\n",
           mdata.MatExpMeans_.nrows());
    exit(1);
  }
  if (mdata.MatExpMeans_.nrows() != mdata.MatExpStds_.nrows())
  {
    printf("MCMC ERROR: MatExpMeans mismatch with MatExpStds\n");
    printf("            Experiment means matrix has = %d rows\n",
           mdata.MatExpMeans_.nrows());
    printf("            Experiment stdvs matrix has = %d rows\n",
           mdata.MatExpStds_.nrows());
    exit(1);
  }
  if (mdata.MatExpMeans_.ncols() != mdata.MatExpStds_.ncols())
  {
    printf("MCMC ERROR: MatExpMeans mismatch with MatExpStds\n");
    printf("            Experiment means matrix has = %d columns\n",
           mdata.MatExpMeans_.ncols());
    printf("            Experiment stdvs matrix has = %d columns\n",
           mdata.MatExpStds_.ncols());
    exit(1);
  }

  int nOutputs = mdata.nOutputs_;
  if (mdata.MatExpMeans_.ncols() != nOutputs)
  {
    printf("MCMC ERROR: Experiment sample output problem.\n");
    printf("            Experiment means matrix has = %d columns\n",
           mdata.MatExpMeans_.ncols());
    printf("            nOutputs = %d\n", nOutputs_);
    exit(1);
  }
  if (mdata.VecLowerB_.length() != mdata.nInputs_)
  {
    printf("MCMC ERROR: Input lower bound problem.\n");
    printf("            Lower bound vector has length %d\n",
           mdata.VecLowerB_.length());
    printf("            nInputs = %d\n", mdata.nInputs_);
    exit(1);
  }
  if (mdata.VecUpperB_.length() != mdata.nInputs_)
  {
    printf("MCMC ERROR: Input upper bound problem.\n");
    printf("            Upper bound vector has length %d\n",
           mdata.VecUpperB_.length());
    printf("            nInputs = %d\n", mdata.nInputs_);
    exit(1);
  }

  //**/ ---------------------------------------------------------------
  //**/ create vecDesParams and vecInpTypes for later use
  //**/ vecInpTypes[ii]  == 1000 if ii is a fixed uncertain parameter
  //**/ vecInpTypes[ii]  == 2000 if ii is a design parameter
  //**/ vecInpTypes[ii]  == ii    if ii is a design parameter
  //**/ vecDesParams[ii] == 1 if ii is a design parameter
  //**/ ---------------------------------------------------------------
  psIVector vecInpTypes, vecDesParams;
  vecInpTypes.setLength(mdata.nInputs_);
  for (ii = 0; ii < mdata.nInputs_; ii++) vecInpTypes[ii] = ii;
  cnt = 0;
  for (ii = 0; ii < nFUInputs; ii++)
    vecInpTypes[mdata.VecFUInputs_[ii]] = 1000;
  cnt = 0;
  for (ii = 0; ii < nCUInputs; ii++)
  {
    kk = mdata.VecCUInputs_[ii];
    if (vecInpTypes[kk] == 1000)
    {
      printf("MCMC ERROR: Input %d both calibration/uncertain input\n",
             kk+1);
      exit(1);
    }
    vecInpTypes[kk] = 2000;
  }
  vecDesParams.setLength(mdata.nInputs_);
  for (ii = 0; ii < mdata.nInputs_; ii++) 
    if (vecInpTypes[ii] < 1000) vecDesParams[ii] = 1;

  //**/ ---------------------------------------------------------------
  //**/ FUNumSampletoUse - if large FU sample, select a fixed subset
  //**/ ---------------------------------------------------------------
  int FUNumSampleToUse=1;
  FUNumSampleToUse = mdata.MatFUInpSample_.nrows();
  if (FUNumSampleToUse == 0) FUNumSampleToUse = 1;
  if (FUNumSampleToUse > 10000) FUNumSampleToUse = 10000;

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ ==> faPtrs
  //**/ ---------------------------------------------------------------
  int localRS = 1;
  if (mdata.rsPtrs_ != NULL)
  {
    for (ii = 0; ii < mdata.nOutputs_; ii++)
      if (mdata.rsPtrs_[ii] == NULL) break;
    if (ii == mdata.nOutputs_) localRS = 0;
  }

  FuncApprox **faPtrs = new FuncApprox*[mdata.nOutputs_];
  psVector vecYT;
  vecYT.setLength(mdata.nSamples_);
  if (localRS == 1)
  {
    psConfig_.InteractiveSaveAndReset();
    for (ii = 0; ii < mdata.nOutputs_; ii++)
    {
      faPtrs[ii] = genFA(mdata.faType_,mdata.nInputs_,iZero,
                         mdata.nSamples_);
      faPtrs[ii]->setNPtsPerDim(16);
      faPtrs[ii]->setBounds(mdata.VecLowerB_.getDVector(),
                            mdata.VecUpperB_.getDVector());
      faPtrs[ii]->setOutputLevel(0);
      for (kk = 0; kk < mdata.nSamples_; kk++) 
        vecYT[kk] = mdata.VecSamOutputs_[kk*mdata.nOutputs_+ii];

      status = faPtrs[ii]->initialize(mdata.VecSamInputs_.getDVector(), 
                                      vecYT.getDVector());
      if (status != 0)
      {
        printOutTS(PL_ERROR,
             "MCMC ERROR: Unable to create response surface.\n");
        printOutTS(PL_ERROR,"            Consult PSUADE developers.\n");
        for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
        delete [] faPtrs;
        return PSUADE_UNDEFINED;
      }
    }
    psConfig_.InteractiveRestore();
  }
  else
  {
    for (ii = 0; ii < mdata.nOutputs_; ii++)
      faPtrs[ii] = mdata.rsPtrs_[ii];
  }

  //**/ ---------------------------------------------------------------
  //**/ ==> max MCMC Samples
  //**/ ---------------------------------------------------------------
  int maxMCMCSamples = 1000000;
  if (mdata.nInputs_ >= 10) maxMCMCSamples = 2000000;

  //**/ ---------------------------------------------------------------
  //**/ option to add discrepancy function 
  //**/ if no design parameter, the best model form is a constant
  //**/ ---------------------------------------------------------------
  int modelFormConst=0, genPosteriors = 1;
  if (mdata.addDiscrepancy_ == 1 && nDInputs == 0) modelFormConst = 1;
  if (mdata.addDiscrepancy_ == 1 && nExperiments == 1) 
    modelFormConst = 1;

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy function, if desired 
  //**/ ==> vecDiscSamOuts or
  //**/ ==> vecDiscFuncConstMeans, vecDistFuncConstStds
  //**/ ---------------------------------------------------------------
  psVector vecDiscFuncConstMeans, vecDiscFuncConstStds, vecDiscSamOuts;
  if (mdata.addDiscrepancy_ == 1)
  {
    double dstatus;
    psIVector vecINULL;
    psVector  vecDNULL;
    dstatus = createDiscrepancyFunctions(mdata.nInputs_,mdata.nOutputs_,
              mdata.VecLowerB_.getDVector(), mdata.VecUpperB_.getDVector(),
              vecINULL,vecDNULL,vecDesParams,nDInputs,nExperiments,
              mdata.MatExpInputs_, mdata.MatExpMeans_,NULL,
              vecDiscFuncConstMeans,vecDiscFuncConstStds,vecDiscSamOuts,
              faPtrs, iZero, modelFormConst);
    if (dstatus < 0) return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ MCMC parameters (samInc, maxMCMCSamples)
  //**/ ---------------------------------------------------------------
  int nPriorSamples = mdata.MatPriorSample_.nrows();
  int nIncrements = 20;
  nPriorSamples = nPriorSamples / nIncrements * nIncrements;
  maxMCMCSamples = FUNumSampleToUse * nPriorSamples;
  
  //**/ ---------------------------------------------------------------
  //**/ generate a large sample ==> vecPriorSample
  //**/ ---------------------------------------------------------------
  psVector vecPriorSample; 
  mdata.MatPriorSample_.convert2Vector(vecPriorSample);

  //**/ ---------------------------------------------------------------
  //  duplicate the generated sample FUNumSampleToUse-1 times
  //**/ so inferenceSamIns has a sample of size nPriorSample
  //**/ replicated FUNumSampleToUse times
  //**/ ---------------------------------------------------------------
  psVector vecInfSamInp, vecInfSamOut, vecInfTmpOut;
  vecInfSamInp.setLength(maxMCMCSamples * mdata.nInputs_);
  double *inferenceSamIns = vecInfSamInp.getDVector();
  vecInfSamOut.setLength(maxMCMCSamples);
  double *inferenceSamOut = vecInfSamOut.getDVector();
  for (ii = 0; ii < nPriorSamples; ii++)
  {
    for (jj = 0; jj < FUNumSampleToUse; jj++)
    {
      cnt = (ii * FUNumSampleToUse + jj) * mdata.nInputs_;
      for (kk = 0; kk < mdata.VecCUInputs_.length(); kk++)
      {
        kk2 = mdata.VecCUInputs_[kk]; 
        inferenceSamIns[cnt+kk2] = 
          vecPriorSample[ii*mdata.VecCUInputs_.length()+kk]; 
      }
    }
  }
  vecPriorSample.clean();

  //**/ ---------------------------------------------------------------
  //**/ if there are uncertain parameters, create a random vector
  //**/ that selects a fixed sub-sample for the uncertain parameters,
  //**/ and fill in the slots in inferenceSamIns for uncertain parameters
  //**/ ---------------------------------------------------------------
  psIVector randIVec;
  if (mdata.MatFUInpSample_.ncols() > 0)
  {
    randIVec.setLength(FUNumSampleToUse);
    for (int mm = 0; mm < FUNumSampleToUse; mm++)
    {
      if (FUNumSampleToUse >= mdata.MatFUInpSample_.nrows())
           ii = mm;
      else ii = PSUADE_rand() % mdata.MatFUInpSample_.nrows();
      randIVec[mm] = ii;
    }
    for (ii = 0; ii < maxMCMCSamples; ii+=FUNumSampleToUse)
    {
      for (kk = 0; kk < nFUInputs; kk++)
      {
        cnt = mdata.VecFUInputs_[kk];
        for (jj = 0; jj < FUNumSampleToUse; jj++)
        {
          ii2 = randIVec[jj];     
          ddata = mdata.MatFUInpSample_.getEntry(ii2,kk);
          inferenceSamIns[(ii+jj)*mdata.nInputs_+cnt] = ddata;
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //  allocate storage for inference
  //**/ ---------------------------------------------------------------
  psVector vecXS, vecYS, vecSamStd, vecYDesign, vecYDesStd;
  cnt = maxMCMCSamples / nIncrements * nExperiments;
  vecXS.setLength(cnt * mdata.nInputs_);
  double *XSample = vecXS.getDVector();
  vecYS.setLength(cnt * mdata.nOutputs_);
  double *YSample = vecYS.getDVector();
  vecSamStd.setLength(cnt * mdata.nOutputs_);
  double *YSamStd = vecSamStd.getDVector();
  vecYDesign.setLength(cnt * mdata.nOutputs_);
  double *YDesign = vecYDesign.getDVector();
  vecYDesStd.setLength(cnt * mdata.nOutputs_);
  double *YDesStd = vecYDesStd.getDVector();

  //**/ ---------------------------------------------------------------
  //  perform inference
  //**/ ---------------------------------------------------------------
  int    ss, mm, runSize, index, ind1, ind2;
  double stdev, stdv2, YT, YT1, YT2;

  if (mdata.printLevel_ > 0)
    printOutTS(PL_INFO, "MCMC_BFS Inference begins ... \n");
  fflush(stdout);

  int samInc = maxMCMCSamples / (FUNumSampleToUse * nIncrements);
  int curMCMCSample = 0;
  while (curMCMCSample < maxMCMCSamples)
  {
    //**/ have one more loop for the uncertain parameters
    for (mm = 0; mm < samInc; mm++)
    {
      cnt = 0;
      index = mm * nExperiments * FUNumSampleToUse;
      //**/ first fill in all XSample slots
      for (ii2 = 0; ii2 < mdata.nInputs_; ii2++)
      {
        //**/ load prior samples from inferenceSamIns
        if (vecDesParams[ii2] == 0)
        {
          for (kk2 = 0; kk2 < nExperiments; kk2++)
          {
            for (jj = 0; jj < FUNumSampleToUse; jj++)
            {
              ind1 = (index+kk2*FUNumSampleToUse+jj)*mdata.nInputs_+ii2; 
              ind2 = (curMCMCSample+mm*FUNumSampleToUse+jj)*mdata.nInputs_+ii2; 
              XSample[ind1] = inferenceSamIns[ind2];
            }
          }
        }
        //**/ load design parameters
        if (vecDesParams[ii2] == 1)
        {
          for (kk2 = 0; kk2 < nExperiments; kk2++)
          {
            for (jj = 0; jj < FUNumSampleToUse; jj++)
            {
              ind1 = (index+kk2*FUNumSampleToUse+jj)*mdata.nInputs_+ii2; 
              XSample[ind1] = mdata.MatExpInputs_.getEntry(kk2, cnt);
            }
          }
          cnt++;
        }
      }
    }

    //**/ set design outputs and stds (discrepancy) to zero 
    //**/ since they will be used but may not be set later
    int totCnt = samInc * nExperiments * mdata.nOutputs_;
    for (jj = 0; jj < totCnt; jj++)
    {
      YDesign[jj] = YDesStd[jj] = 0.0;
      YSample[jj] = YSamStd[jj] = 0.0;
    }
    //**/ run XGuessS through response surfaces
    //**/ ==> YDesign, YDesStd, YSample, YSamStd
    runSize = samInc * nExperiments;
    for (ii2 = 0; ii2 < mdata.nOutputs_; ii2++)
    {
      //**/ case 1: if RS error is requested
      if (mdata.useRSUncertainties_ == 1)
      {
        faPtrs[ii2]->evaluatePointFuzzy(runSize, 
                             XSample,&YSample[ii2*runSize],
                             &YSamStd[ii2*runSize]);
        //**/ add discrepancy function, if available
        if (mdata.addDiscrepancy_ == 1 && modelFormConst == 0)
        {
          for (mm = 0; mm < samInc; mm++)
          {
            index = mm * nExperiments * FUNumSampleToUse;
            for (kk2 = 0; kk2 < nExperiments; kk2++)
            {
              for (jj = 0; jj < FUNumSampleToUse; jj++)
              {
                ind1 = ii2*runSize+index+kk2*FUNumSampleToUse+jj;
                YDesign[ind1] = vecDiscSamOuts[ii2*nExperiments+kk2];
                YDesStd[ind1] = 0.0;
              }
            }
          }
        }
        else if (mdata.addDiscrepancy_ == 1 && modelFormConst == 1 &&
                 vecDiscFuncConstMeans.length() > 0 &&
                 vecDiscFuncConstMeans[ii2] != PSUADE_UNDEFINED)
        {
          for (kk2 = 0; kk2 < runSize; kk2++)
          {
            YDesign[ii2*runSize+kk2] = vecDiscFuncConstMeans[ii2];
            YDesStd[ii2*runSize+kk2] = 0.0;
          }
        }
      }
      //**/ case 2: if RS error is to be turned off
      else
      {
        faPtrs[ii2]->evaluatePoint(runSize,XSample,
                                   &YSample[ii2*runSize]);
        //**/ add discrepancy function, if available
        if (mdata.addDiscrepancy_ == 1 && modelFormConst == 0)
        {
          for (mm = 0; mm < samInc; mm++)
          {
            index = mm * nExperiments * FUNumSampleToUse;
            for (kk2 = 0; kk2 < nExperiments; kk2++)
            {
              for (jj = 0; jj < FUNumSampleToUse; jj++)
              {
                ind1 = ii2*runSize+index+kk2*FUNumSampleToUse+jj;
                YDesign[ind1] = vecDiscSamOuts[ii2*nExperiments+kk2];
              }
            }
          }
        }
        else if (mdata.addDiscrepancy_ == 1 && modelFormConst == 1 &&
                 vecDiscFuncConstMeans.length() > 0 &&
                 vecDiscFuncConstMeans[ii2] != PSUADE_UNDEFINED)
        {
          for (kk2 = 0; kk2 < runSize; kk2++)
            YDesign[ii2*runSize+kk2] = vecDiscFuncConstMeans[ii2];
        }
        for (kk2 = 0; kk2 < runSize; kk2++)
          YSamStd[ii2*runSize+kk2] = YDesStd[ii2*runSize+kk2] = 0.0;
      }
    }

    //**/ compute vecXDist
    for (ii = 0; ii < samInc; ii++)
    {
      index = ii * FUNumSampleToUse;
      for (jj = 0; jj < FUNumSampleToUse; jj++)
      {
        inferenceSamOut[curMCMCSample+index+jj] = 0.0;
        for (ii2 = 0; ii2 < mdata.nOutputs_; ii2++)
        {
          for (kk2 = 0; kk2 < nExperiments; kk2++)
          {
            ind1 = ii2*runSize+index*nExperiments+kk2*FUNumSampleToUse+jj;
            YT1 = YSample[ind1] + YDesign[ind1];
            stdev = YSamStd[ind1] + YDesign[ind1];
            stdev = YSamStd[ind1];
            stdv2 = YDesStd[ind1];
            YT2 = pow((YT1-mdata.MatExpMeans_.getEntry(kk2,ii2)),2.0) /
                 (pow(mdata.MatExpStds_.getEntry(kk2,ii2),2.0) +
                  stdev*stdev + stdv2*stdv2);
            inferenceSamOut[curMCMCSample+index+jj] += YT2;
          }
        }
        inferenceSamOut[curMCMCSample+index+jj] /= (double) dnReps;
      }
    }
    curMCMCSample += (samInc * FUNumSampleToUse);
  }

  //**/ ---------------------------------------------------------------
  //**/  generate the posterior sample 
  //**/ ---------------------------------------------------------------
  int imax=-1;
  if (genPosteriors == 1)
  {
    int    maxPostSam=nPriorSamples;
    double dmax = 0.0, dsum=0;
    for (ss = 0; ss < maxMCMCSamples; ss++) 
    {
      if (inferenceSamOut[ss] > dmax) 
      {
        dmax = inferenceSamOut[ss];
        imax = ss;
      }
    }
    if (imax != -1)
    {
      if (mdata.printLevel_ > 1)
      {
        for (ii = 0; ii < mdata.VecCUInputs_.length(); ii++) 
        {
          kk = mdata.VecCUInputs_[ii];
          printf("Input %4d: xmax = %e\n", kk+1, 
                 inferenceSamIns[imax*mdata.nInputs_+kk]); 
        }
      }
    }
    if (dmax == 0)
    {
      printOutTS(PL_ERROR,
         "MCMC_BFS: ERROR encountered in posterior sample generation.\n");
      printOutTS(PL_ERROR,
         "          Maybe due to empty posterior. Use larger data errors.\n");
      exit(1);
    }
    else 
    {
      for (ss = 0; ss < maxMCMCSamples; ss++) 
        inferenceSamOut[ss] = exp(-inferenceSamOut[ss]-dmax);
      dsum = 0;
      for (ss = 0; ss < maxMCMCSamples; ss++) 
        dsum += inferenceSamOut[ss];
      for (ss = 0; ss < maxMCMCSamples; ss++) 
        inferenceSamOut[ss] /= dsum;

      //**/ this 2D format is to be compatible with ProbMatrix

      //**/ generate posterior sample
#if 1
      maxPostSam = maxMCMCSamples;
      cnt = 0;
      for (ss = 0; ss < maxMCMCSamples; ss++) 
      {
        kk = (int) (inferenceSamOut[ss] * maxPostSam + 0.5);
        cnt += kk;
      }
      maxPostSam = cnt;

      //**/ this 2D format is to be compatible with ProbMatrix
      mdata.MatPostSample_.setFormat(PS_MAT2D);
      mdata.MatPostSample_.setDim(maxPostSam,
                                  mdata.VecCUInputs_.length());
      mdata.VecPostLikelihoods_.setLength(maxPostSam);
      cnt = 0;
      for (ss = 0; ss < maxMCMCSamples; ss++) 
      {
        kk = (int) (inferenceSamOut[ss]*maxMCMCSamples + 0.5);
        for (jj = 0; jj < kk; jj++)
        {  
          for (ii2 = 0; ii2 < mdata.VecCUInputs_.length(); ii2++)
          {
            if (cnt >= maxPostSam)
              printf("ERROR: matrix index out of range = %d %d\n",
                     ss, jj);
            ind1 = mdata.VecCUInputs_[ii2];
            ind2 = ss * mdata.nInputs_ + ind1;
            mdata.MatPostSample_.setEntry(cnt,ii2,inferenceSamIns[ind2]);
            mdata.VecPostLikelihoods_[cnt] = (double) inferenceSamOut[ss];
          }
          cnt++;
        }
      }
#else
      //**/ Feb 2022
      //**/ this segment doesn't perform well if all
      //**/ posterior probabilities are high
      cnt = 0;
      while (cnt < maxPostSam)
      {
        kk2 = PSUADE_rand() % nValid;
        kk2 = vecValid[kk2];
        ddata = PSUADE_drand();
        if (inferenceSamOut[kk2] > ddata)
        {
          for (jj = 0; jj < mdata.VecCUInputs_.length(); jj++)
          {
            ind1 = mdata.VecCUInputs_[jj];
            ind2 = kk2 * mdata.nInputs_ + ind1;
            mdata.MatPostSample_.setEntry(cnt,jj,inferenceSamIns[ind2]);
            mdata.VecPostLikelihoods_[cnt] = (double) inferenceSamOut[kk2];
          }
          cnt++;
        }
      }
#endif
    }
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  if (faPtrs != NULL)
  {
    if (localRS == 1)
    {
      for (ii = 0; ii < mdata.nOutputs_; ii++)
        if (faPtrs[ii] != NULL) delete faPtrs[ii];
      delete [] faPtrs;
    }
  }
  return 0.0;
}

// ************************************************************************
// Compute likelihood only
// ------------------------------------------------------------------------
double MCMCAnalyzer::computeLikelihood(aData &adata)
{
  //**/ ---------------------------------------------------------------
  //**/ extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *sampIns = adata.sampleInputs_;
  double *samOuts = adata.sampleOutputs_;
  double *xLower  = adata.iLowerB_;
  double *xUpper  = adata.iUpperB_;
  PsuadeData *dataPtr = adata.ioPtr_;

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  //**/ ==> vecDesignP, matExpInps, matExpMeans, matExpStdvs
  //**/ ---------------------------------------------------------------
  int       dnReps=1;
  psIVector vecDesignP; 
  psMatrix  matExpInps, matExpMeans, matExpStdvs; 
  double dstatus = readSpecFile(nInputs,nOutputs,vecDesignP,matExpInps,
                   matExpMeans, matExpStdvs, dnReps, printLevel);
  if (dstatus != 0.0) return PSUADE_UNDEFINED;
  int dnSamples = matExpMeans.nrows();
  int dnInputs  = matExpInps.ncols();

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ ==> faPtrs.
  //**/ ---------------------------------------------------------------
  int   faType=0;
  pData pPtr;
  if (dataPtr != NULL)
  {
    dataPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
  }
  else
  {
    printOutTS(PL_INFO,
       "ComputeLikelihood INFO: ioPtr=NULL ==> assume MARS as RS.\n");
    faType = PSUADE_RS_MARS;
  }
  printOutTS(PL_INFO,
       "ComputeLikelihood INFO: CREATING RESPONSE SURFACES.\n");
  FuncApprox **faPtrs = new FuncApprox*[nOutputs];

  int ii, kk, status, iZero=0;
  psVector vecSamOut1;
  vecSamOut1.setLength(nSamples);
  for (ii = 0; ii < nOutputs; ii++)
  {
    faType = -1;
    printOutTS(PL_INFO,
       "ComputeLikelihood INFO: CREATING RS FOR OUTPUT %d.\n",ii+1);
    faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
    faPtrs[ii]->setNPtsPerDim(16);
    faPtrs[ii]->setBounds(xLower, xUpper);
    faPtrs[ii]->setOutputLevel(0);
    for (kk = 0; kk < nSamples; kk++) 
      vecSamOut1[kk] = samOuts[kk*nOutputs+ii];

    status = faPtrs[ii]->initialize(sampIns, vecSamOut1.getDVector());
    if (status != 0)
    {
      printOutTS(PL_ERROR,
        "ComputeLikelihood ERROR: Unable to create response surface.\n");
      printOutTS(PL_ERROR,"       Consult PSUADE developers.\n");
      for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
      delete [] faPtrs;
      return PSUADE_UNDEFINED;
    }
  }
  int  rsErrFlag=0;
  char pString[1000], lineIn[1000];
  if (psConfig_.AnaExpertModeIsOn())
  {
    //**/ change sampling information
    printf("Include RS uncertainty in likelihood ? (y or n) ");
    scanf("%s", pString);
    if (pString[0] == 'y') rsErrFlag = 1;
    fgets(lineIn,1000,stdin);
  }

  //**/ ---------------------------------------------------------------
  // run the Metropolis Hasting algorithm
  //**/ ---------------------------------------------------------------
  psVector vecXGuess, vecYGuess, vecYStdev;
  vecXGuess.setLength(dnSamples*nInputs);
  vecYGuess.setLength(dnSamples*nOutputs);
  vecYStdev.setLength(dnSamples*nOutputs);
  double *XGuess = vecXGuess.getDVector();
  double *YGuess = vecYGuess.getDVector();
  double *YStdev = vecYStdev.getDVector();
  double chiSq, dmean, dstdv, estdv;
  int count;

  //**/ loop until user wants to terminate
  while (1)
  {
    //**/ input calibration parameter values 
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ default = mid point
      vecXGuess[ii] = 0.5 * (xUpper[ii] + xLower[ii]);
      if (vecDesignP.length() == 0 || vecDesignP[ii] == 0) 
      {
        snprintf(pString,100,
           "Please enter calibration value for input %d : ",ii+1);
        vecXGuess[ii] = getDouble(pString);
      }
    }

    //**/ fill up vecXGuess sample point
    for (kk = 0; kk < dnSamples; kk++)
    {
      if (kk > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          vecXGuess[kk*nInputs+ii] = vecXGuess[ii];
      }
      count = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        //**/ design inputs
        if (vecDesignP.length() > 0 && vecDesignP[ii] == 1) 
        {
          vecXGuess[kk*nInputs+ii] = matExpInps.getEntry(kk,count);
          count++;
        }
      }
    }

    //**/ evaluate proposal sample 
    for (ii = 0; ii < nOutputs; ii++)
    {
      if (rsErrFlag == 0)
        faPtrs[ii]->evaluatePoint(dnSamples,XGuess,
                                  &(YGuess[ii*dnSamples]));
      else
        faPtrs[ii]->evaluatePointFuzzy(dnSamples,XGuess,
               &(YGuess[ii*dnSamples]),&(YStdev[ii*dnSamples]));
    }

    //**/ compute chi-squared
    chiSq = 0.0;
    for (ii = 0; ii < nOutputs; ii++)
    {
      for (kk = 0; kk < dnSamples; kk++)
      {
        dmean = matExpMeans.getEntry(kk,ii);
        dstdv = matExpStdvs.getEntry(kk,ii);
        estdv = YStdev[ii*dnSamples+kk];
        if (rsErrFlag == 0)
          chiSq += pow((YGuess[ii*dnSamples+kk]-dmean)/dstdv,2.0);
        else
          chiSq += pow((YGuess[ii*dnSamples+kk]-dmean),2.0)/
                       (dstdv*dstdv+estdv*estdv);
        if (printLevel >= 5)
        {
          printf("Simulation vs experiment %d %d = %e %e\n",ii+1,kk+1,
                 YGuess[ii*dnSamples+kk],dmean);
        }
      }
    }
    chiSq /= (double) dnReps;
    chiSq *= 0.5;
    printf("Negative log-likelihood = %e\n", chiSq);

    printf("Evaluate another point ? (y or n) ");
    scanf("%s", pString);
    fgets(lineIn,1000,stdin);
    if (pString[0] != 'y') break;
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++) 
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  return 0.0;
}

// ************************************************************************
// write to matlab file 
// ------------------------------------------------------------------------
double MCMCAnalyzer::genMatlabFile(int nInputs,double *lower,double *upper,
                       double *XRange, int nPlots, int *plotIndices,
                       int nbins, int **pbins, int ****pbins2, 
                       int **bins, int ****bins2, 
                       pData &qData, int nChains, int chainCnt,
                       double ***XChains, int *chainStatus,
                       double *Xmax, double Ymin)
{
  int    kk, kk2, ii2, jj, jj2, iChain;
  long   sumBins;
  double ddata, dmean, dstd;
  char   cfname[1001], pString[1001];;
  FILE   *fp;

  //**/ ---------------------------------------------------------------
  //**/ create posterior plots
  //**/ ---------------------------------------------------------------
  if (plotScilab()) strcpy(cfname, "scilabmcmc2.sci");
  else              strcpy(cfname, "matlabmcmc2.m");
  fp = fopen(cfname, "w");
  if (fp == NULL)
  {
    printOutTS(PL_ERROR, "ERROR: cannot open %s file.\n", cfname);
    return 0;
  }
  snprintf(pString,100,"This file shows posteriors plots");
  fwriteComment(fp, pString);
  snprintf(pString,100,"ns  - set to 1 for 1-step smoothing of 2D contours");
  fwriteComment(fp, pString);
  snprintf(pString,100,"ns1 - set to 1 for 1-step smoothing of 1D histgrams");
  fwriteComment(fp, pString);
  fprintf(fp, "ns  = 0;\n");
  fprintf(fp, "ns1 = 0;\n");
  fwritePlotCLF(fp);
  fprintf(fp, "active = [\n");
  for (kk = 0; kk < nInputs; kk++)
  {
    ii2 = binarySearchInt(kk, plotIndices, nPlots);
    if (ii2 < 0) fprintf(fp, "0\n");
    else         fprintf(fp, "1\n");
  }
  fprintf(fp, "];\n");
  fprintf(fp, "L = [\n");
  for (kk = 0; kk < nInputs; kk++) fprintf(fp, "%e ",lower[kk]);
  fprintf(fp, "];\n");
  fprintf(fp, "U = [\n");
  for (kk = 0; kk < nInputs; kk++) fprintf(fp, "%e ",upper[kk]);
  fprintf(fp, "];\n");
  if (plotScilab()) fprintf(fp, "iStr = [\n");
  else              fprintf(fp, "iStr = {\n");
  for (kk = 0; kk < nInputs-1; kk++)
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s',", qData.strArray_[kk]);
    else fprintf(fp, "'Input %d',", kk+1);
  }
  if (plotScilab()) 
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s'];\n", qData.strArray_[nInputs-1]);
    else fprintf(fp, "'Input %d'];\n", nInputs);
  }
  else
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s'};\n", qData.strArray_[nInputs-1]);
    else fprintf(fp, "'Input %d'};\n", nInputs);
  }
  fprintf(fp, "X = zeros(%d,%d);\n", nInputs, nbins);
  fprintf(fp, "D = zeros(%d,%d);\n", nInputs, nbins);
  fprintf(fp, "NC = zeros(%d,%d,%d,%d);\n",nInputs,nInputs,nbins,nbins);
  //**/ D and NC contains the posterior histograms
  for (kk = 0; kk < nInputs; kk++)
  {
    for (kk2 = 0; kk2 < nInputs; kk2++)
    {
      if (kk == kk2)
      {
        fprintf(fp, "X(%d,:) = [\n", kk+1);
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", XRange[kk]/nbins*(jj+0.5)+lower[kk]);
        fprintf(fp, "];\n");
        sumBins = 0;
        for (jj = 0; jj < nbins; jj++) sumBins += bins[jj][kk];
        if (sumBins == 0) sumBins = 1;
        fprintf(fp, "D(%d,:) = [\n", kk+1);
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", (double) bins[jj][kk]/(double) sumBins);
        fprintf(fp, "];\n");
      }
      else
      {
        fprintf(fp, "NC(%d,%d,:,:) = [\n", kk+1, kk2+1);
        for (jj = 0; jj < nbins; jj++)
        {
          for (jj2 = 0; jj2 < nbins; jj2++)
            fprintf(fp, "%d ", bins2[jj][jj2][kk][kk2]);
          fprintf(fp, "\n");
        }
        fprintf(fp, "]';\n");
      }
    }
  }
  //**/ D and NC contains the posterior histograms
  if (pbins != NULL)
    fprintf(fp, "DP = zeros(%d,%d);\n", nInputs, nbins);
  if (pbins2 != NULL)
    fprintf(fp, "NCP = zeros(%d,%d,%d,%d);\n",nInputs,nInputs,nbins,nbins);
  for (kk = 0; kk < nInputs; kk++)
  {
    for (kk2 = 0; kk2 < nInputs; kk2++)
    {
      if (kk == kk2)
      {
        if (pbins != NULL)
        {
          sumBins = 0;
          for (jj = 0; jj < nbins; jj++) sumBins += pbins[jj][kk];
          fprintf(fp, "DP(%d,:) = [\n", kk+1);
          for (jj = 0; jj < nbins; jj++)
            fprintf(fp, "%e ", (double) pbins[jj][kk]/(double) sumBins);
          fprintf(fp, "];\n");
        }
      }
      else
      {
        if (pbins2 != NULL)
        {
          fprintf(fp, "NCP(%d,%d,:,:) = [\n", kk+1, kk2+1);
          for (jj = 0; jj < nbins; jj++)
          {
            for (jj2 = 0; jj2 < nbins; jj2++)
              fprintf(fp, "%d ", pbins2[jj][jj2][kk][kk2]);
            fprintf(fp, "\n");
          }
          fprintf(fp, "]';\n");
        }
      }
    }
  }
  //**/ construct an active list for flexibility in plotting
  fprintf(fp,"nInps  = length(active);\n");
  fprintf(fp,"nPlots = 0;\n");
  fprintf(fp,"for ii = 1 : nInps\n");
  fprintf(fp,"   if (active(ii) == 1)\n");
  fprintf(fp,"      nPlots = nPlots + 1;\n");
  fprintf(fp,"      active(ii) = nPlots;\n");
  fprintf(fp,"   end;\n");
  fprintf(fp,"end;\n");
  fprintf(fp,"dzero = 0;\n");
  //**/ write the optimal inputs for plotting
  if (Xmax != NULL)
  {
    for (kk = 0; kk < nInputs; kk++)
      fprintf(fp, "XOpt(%d) = %e;\n", kk+1, Xmax[kk]);
  }
  fprintf(fp,"for ii = 1 : nInps\n");
  fprintf(fp,"  for jj = ii : nInps\n");
  fprintf(fp,"    if (active(ii) ~= 0 & active(jj) ~= 0)\n");
  fprintf(fp,"      index = (active(ii)-1) * nPlots + active(jj);\n");
  fprintf(fp,"      subplot(nPlots,nPlots,index)\n");
  fprintf(fp,"      if (ii == jj)\n");
  fprintf(fp,"        n = length(D(ii,:));\n");
  fprintf(fp,"        DN = D(ii,:);\n");
  fprintf(fp,"        for kk = 1 : ns1\n");
  fprintf(fp,"          DN1 = DN;\n");
  fprintf(fp,"          for ll = 2 : n-1\n");
  fprintf(fp,"            DN(ll) = DN(ll) + DN1(ll+1);\n");
  fprintf(fp,"            DN(ll) = DN(ll) + DN1(ll-1);\n");
  fprintf(fp,"            DN(ll) = DN(ll) / 3;\n");
  fprintf(fp,"          end;\n");
  fprintf(fp,"        end;\n");
  fprintf(fp,"        bar(X(ii,:), DN, 1.0);\n");
  if (plotScilab())
       fprintf(fp,"        set(gca(),\"auto_clear\",\"off\")\n");
  else fprintf(fp,"        hold on\n");
  if (Xmax != NULL)
  {
    fprintf(fp,
      "        plot([XOpt(ii) XOpt(ii)],[0 max(DN)],'r-','LineWidth',2)\n");
  }
  if (pbins != NULL)
    fprintf(fp,"        plot(X(ii,:),DP(ii,:),'c-','LineWidth',2);\n");
  fprintf(fp,"        xmin = L(ii);\n");
  fprintf(fp,"        xmax = U(ii);\n");
  fprintf(fp,"        ymax = max(DN);\n");
  if (plotScilab())
  {
    fprintf(fp,"        //e = gce();\n");
    fprintf(fp,"        //e.children.thickness = 2;\n");
    fprintf(fp,"        //e.children.foreground = 0;\n");
    fprintf(fp,"        //e.children.background = 2;\n");
    fprintf(fp,"        a = gca();\n");
    fprintf(fp,"        a.data_bounds=[xmin,0;xmax,ymax];\n");
    fprintf(fp,"        atmp = a.x_ticks;\n");
    fprintf(fp,"        atmp.locations=[xmin:(xmax-xmin)/3:xmax];\n");
    fprintf(fp,"        atmp.labels=string(atmp.locations);\n");
    fprintf(fp,"        a.x_ticks=atmp;\n");
    fprintf(fp,"        a.x_label.text = iStr(ii);\n");
    fprintf(fp,"        a.x_label.font_size = 3;\n");
    fprintf(fp,"        a.x_label.font_style = 4;\n");
    fprintf(fp,"        a.grid = [1 1];\n");
    fprintf(fp,"        atmp = a.y_ticks;\n");
    fprintf(fp,"        atmp.locations=[0:ymax/3:ymax];\n");
    fprintf(fp,"        atmp.labels=string(atmp.locations);\n");
    fprintf(fp,"        a.y_ticks=atmp;\n");
    fprintf(fp,"        a.y_label.text = 'Probability';\n");
    fprintf(fp,"        a.y_label.font_size = 3;\n");
    fprintf(fp,"        a.y_label.font_style = 4;\n");
    fprintf(fp,"        a.thickness = 2;\n");
    fprintf(fp,"        a.font_size = 3;\n");
    fprintf(fp,"        a.font_style = 4;\n");
    fprintf(fp,"        a.box = \"on\";\n");
  }
  else
  {
    fprintf(fp,"        axis([xmin xmax 0 ymax])\n");
    fprintf(fp,"        set(gca,'linewidth',2)\n");
    fprintf(fp,"        set(gca,'fontweight','bold')\n");
    fprintf(fp,"        set(gca,'fontsize',12)\n");
    fprintf(fp, 
       "        xlabel(iStr(ii),'FontWeight','bold','FontSize',12)\n");
    fprintf(fp,"        if (ii == 1)\n"); 
    fprintf(fp,
       "        ylabel('Probabilities','FontWeight','bold','FontSize',12)\n");
    fprintf(fp,"        end;\n"); 
    fprintf(fp,"        grid on\n");
    fprintf(fp,"        box on\n");
  }
  fprintf(fp,"      else\n");
  fprintf(fp,"        n = length(X(jj,:));\n");
  fprintf(fp,"        XT = X(jj,:);\n");
  fprintf(fp,"        YT = X(ii,:);\n");
  fprintf(fp,"        HX = (XT(n) - XT(1)) / (n-1);\n");
  fprintf(fp,"        HY = (YT(n) - YT(1)) / (n-1);\n");
  fprintf(fp,"        ZZ = squeeze(NC(ii,jj,:,:));\n");
  fprintf(fp,"        for kk = 1 : ns\n");
  fprintf(fp,"          ZZ1 = ZZ;\n");
  fprintf(fp,"          for ll = 2 : n-1\n");
  fprintf(fp,"            for mm = 2 : n-1\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) / 9;\n");
  fprintf(fp,"            end;\n");
  fprintf(fp,"          end;\n");
  fprintf(fp,"        end;\n");
  fprintf(fp,"        ZZ = ZZ / (sum(sum(ZZ)));\n");
  if (plotScilab())
  {
    fprintf(fp,"        XX = [XT(1):HX:XT(n)];\n");
    fprintf(fp,"        YY = [YT(1):HY:YT(n)];\n");
    fprintf(fp,"        DD = splin2d(XX,YY,ZZ);\n");
    fprintf(fp,"        HX = 0.01 * (XT(n) - XT(1));\n");
    fprintf(fp,"        HY = 0.01 * (YT(n) - YT(1));\n");
    fprintf(fp,"        X2 = [XT(1):HX:XT(n)];\n");
    fprintf(fp,"        Y2 = [YT(1):HY:YT(n)];\n");
    fprintf(fp,"        [XI, YI] = ndgrid(X2, Y2);\n");
    fprintf(fp,"        //disp('interpolation')\n");
    fprintf(fp,"        ZI =interp2d(XI, YI, XX, YY, DD, \"natural\");\n");
    fprintf(fp,"        //disp('interpolation done')\n");
    fprintf(fp,"        ZB = ZI;\n");
    fprintf(fp,"        nX = length(X2);\n");
    fprintf(fp,"        nY = length(Y2);\n");
    fprintf(fp,"        for kk = 1 : nX\n");
    fprintf(fp,"          for ll = 1 : nY\n");
    fprintf(fp,"            ZI(kk,ll) = ZB(kk,nY-ll+1);\n");
    fprintf(fp,"          end;\n");
    fprintf(fp,"        end;\n");
    fprintf(fp,"        zmax = max(max(ZI));\n");
    fprintf(fp,"        zmin = min(min(ZI)) / zmax;\n");
    fprintf(fp,"        ZI   = ZI / zmax;\n");
    fprintf(fp,"        Matplot1((ZI'-zmin)/(1-zmin)*64,[L(jj),L(ii),");
    fprintf(fp,"U(jj),U(ii)]);\n");
    fprintf(fp,"        gcf().color_map = jetcolormap(64);\n");
    fprintf(fp,"        colorbar(0,zmax);\n");
    fprintf(fp,
         "        contour2d(X2,Y2,ZB,5,rect=[L(jj),L(ii),U(jj),U(ii)]);\n");
    fprintf(fp,"        //xset(\"fpf\",\" \");\n");
    fprintf(fp,"        a = gca();\n");
    fprintf(fp,"        a.data_bounds = [L(jj), L(ii); U(jj), U(ii)];\n");
    fprintf(fp,"        a.x_label.text = iStr(jj);\n");
    fprintf(fp,"        a.x_label.font_size = 3;\n");
    fprintf(fp,"        a.x_label.font_style = 4;\n");
    fprintf(fp,"        a.y_label.text = iStr(ii);\n");
    fprintf(fp,"        a.y_label.font_size = 3;\n");
    fprintf(fp,"        a.y_label.font_style = 4;\n");
    fwritePlotAxesNoGrid(fp);
  }
  else
  {
    fprintf(fp,"%%      [YY,XX]=meshgrid(XT(1):HX:XT(n),YT(1):HY:YT(n));\n");
    fprintf(fp,"%%      HX = 0.01 * (XT(n) - XT(1));\n");
    fprintf(fp,"%%      HY = 0.01 * (YT(n) - YT(1));\n");
    fprintf(fp,"%%      [YI,XI]=meshgrid(XT(1):HX:XT(n),YT(1):HY:YT(n));\n");
    fprintf(fp,"%%      ZI=interp2(YY, XX, ZZ, YI, XI, 'spline');\n");
    fprintf(fp,"%%      pcolor(XI,YI,ZI)\n");
    fprintf(fp,"%%      shading interp\n");
    fprintf(fp,"%%      hold on\n");
    fprintf(fp,"%%      contour(XI,YI,ZI,5,'k')\n");
    fprintf(fp,"        imagesc(ZZ')\n");
    fprintf(fp,"        xtick = L(jj):(U(jj)-L(jj))/2:U(jj);\n");
    fprintf(fp,"        set(gca,'XTick',0:n/2:n);\n");
    fprintf(fp,"        set(gca,'XTickLabel', xtick);\n");
    fprintf(fp,"        ytick = L(ii):(U(ii)-L(ii))/2:U(ii);\n");
    fprintf(fp,"        set(gca,'YTick',0:n/2:n);\n");
    fprintf(fp,"        set(gca,'YTickLabel', ytick);\n");
    fprintf(fp,"        set(gca,'YDir', 'normal');\n");
    fprintf(fp,
       "        xlabel(iStr(jj),'FontWeight','bold','FontSize',12)\n");
    fprintf(fp,
       "        ylabel(iStr(ii),'FontWeight','bold','FontSize',12)\n");
    fwritePlotAxesNoGrid(fp);
    fprintf(fp,"%%      colorbar\n");
  }
  if (pbins2 != NULL)
  {
    fprintf(fp,"        index = (active(jj)-1) * nPlots + active(ii);\n");
    fprintf(fp,"        subplot(nPlots,nPlots,index)\n");
    fprintf(fp,"        ZZ = squeeze(NCP(ii,jj,:,:));\n");
    fprintf(fp,"        for kk = 1 : ns\n");
    fprintf(fp,"          ZZ1 = ZZ;\n");
    fprintf(fp,"          for ll = 2 : n-1\n");
    fprintf(fp,"            for mm = 2 : n-1\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm+1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm-1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm+1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm-1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm+1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm-1);\n");
    fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) / 9;\n");
    fprintf(fp,"            end;\n");
    fprintf(fp,"          end;\n");
    fprintf(fp,"        end;\n");
    fprintf(fp,"        ZZ = ZZ / (sum(sum(ZZ)));\n");
    if (plotScilab())
    {
      fprintf(fp,"        HX = (XT(n) - XT(1)) / (n-1);\n");
      fprintf(fp,"        HY = (YT(n) - YT(1)) / (n-1);\n");
      fprintf(fp,"        XX = [XT(1):HX:XT(n)];\n");
      fprintf(fp,"        YY = [YT(1):HY:YT(n)];\n");
      fprintf(fp,"        DD = splin2d(XX,YY,ZZ);\n");
      fprintf(fp,"        HX = 0.01 * (XT(n) - XT(1));\n");
      fprintf(fp,"        HY = 0.01 * (YT(n) - YT(1));\n");
      fprintf(fp,"        X2 = [XT(1):HX:XT(n)];\n");
      fprintf(fp,"        Y2 = [YT(1):HY:YT(n)];\n");
      fprintf(fp,"        [XI, YI] = ndgrid(X2, Y2);\n");
      fprintf(fp,"        //disp('interpolation')\n");
      fprintf(fp,"        ZI =interp2d(XI, YI, XX, YY, DD, \"natural\");\n");
      fprintf(fp,"        //disp('interpolation done')\n");
      fprintf(fp,"        ZB = ZI;\n");
      fprintf(fp,"        nX = length(X2);\n");
      fprintf(fp,"        nY = length(Y2);\n");
      fprintf(fp,"        for kk = 1 : nX\n");
      fprintf(fp,"          for ll = 1 : nY\n");
      fprintf(fp,"            ZI(kk,ll) = ZB(kk,nY-ll+1);\n");
      fprintf(fp,"          end;\n");
      fprintf(fp,"        end;\n");
      fprintf(fp,"        zmax = max(max(ZI));\n");
      fprintf(fp,"        zmin = min(min(ZI)) / zmax;\n");
      fprintf(fp,"        ZI   = ZI / zmax;\n");
      fprintf(fp,"        Matplot1((ZI'-zmin)/(1-zmin)*64,[L(jj),L(ii),");
      fprintf(fp,"U(jj),U(ii)]);\n");
      fprintf(fp,"        gcf().color_map = jetcolormap(64);\n");
      fprintf(fp,"        colorbar(0,zmax);\n");
      fprintf(fp,
           "        contour2d(X2,Y2,ZB,5,rect=[L(jj),L(ii),U(jj),U(ii)]);\n");
      fprintf(fp,"        //xset(\"fpf\",\" \");\n");
      fprintf(fp,"        a = gca();\n");
      fprintf(fp,"        a.data_bounds = [L(jj), L(ii); U(jj), U(ii)];\n");
      fprintf(fp,"        a.x_label.text = iStr(jj);\n");
      fprintf(fp,"        a.x_label.font_size = 3;\n");
      fprintf(fp,"        a.x_label.font_style = 4;\n");
      fprintf(fp,"        a.y_label.text = iStr(ii);\n");
      fprintf(fp,"        a.y_label.font_size = 3;\n");
      fprintf(fp,"        a.y_label.font_style = 4;\n");
      fwritePlotAxesNoGrid(fp);
    }
    else
    {
      fprintf(fp,"        imagesc(ZZ')\n");
      fprintf(fp,"        xtick = L(jj):(U(jj)-L(jj))/2:U(jj);\n");
      fprintf(fp,"        set(gca,'XTick',0:n/2:n);\n");
      fprintf(fp,"        set(gca,'XTickLabel', xtick);\n");
      fprintf(fp,"        ytick = L(ii):(U(ii)-L(ii))/2:U(ii);\n");
      fprintf(fp,"        set(gca,'YTick',0:n/2:n);\n");
      fprintf(fp,"        set(gca,'YTickLabel', ytick);\n");
      fprintf(fp,"        set(gca,'YDir', 'normal');\n");
      fprintf(fp,
         "        xlabel(iStr(jj),'FontWeight','bold','FontSize',12)\n");
      fprintf(fp,
         "        ylabel(iStr(ii),'FontWeight','bold','FontSize',12)\n");
      fprintf(fp,
         "        if (ii == 1 & jj == 2)\n");
      fprintf(fp,
         "          title('Prior','FontWeight','bold','FontSize',12)\n");
      fprintf(fp,"        end;\n");
      fwritePlotAxesNoGrid(fp);
      fprintf(fp,"        colorbar\n");
    }
  }
  fprintf(fp,"      end;\n");
  fprintf(fp,"    end;\n");
  fprintf(fp,"  end;\n");
  fprintf(fp,"end;\n");
  double minData=PSUADE_UNDEFINED;
  if (XChains == NULL) minData = Ymin;
  else
  {
    for (iChain = 0; iChain < nChains; iChain++) 
    { 
      if (chainStatus[iChain] == 0)
      {
        for (jj = 0; jj < chainCnt; jj++) 
        {
          ddata = XChains[iChain][jj][nInputs];
          if (ddata < minData) minData = ddata;
        }
      }
    }
  }
  if (plotMatlab())
  {
    fprintf(fp,"      subplot(nPlots,nPlots,1)\n");
    fprintf(fp,"set(gcf,'NextPlot','add');\n");
    fprintf(fp,"axes;\n");
    if (minData < PSUADE_UNDEFINED)
      fprintf(fp,
          "h=title('MCMC Priors/Posteriors, best -log(likelihood)=%e',",
          minData);
    else fprintf(fp,"h=title('MCMC Prior/Posterior Distributions',");
    fprintf(fp,"'fontSize',12,'fontWeight','bold');\n");
    fprintf(fp,"set(gca,'Visible','off');\n");
    fprintf(fp,"set(h,'Visible','on');\n");
    fprintf(fp,"negll = %e;\n", minData);
  }
  if (chainCnt > 0)
  {
    for (kk = 0; kk < nInputs; kk++)
    {
      dmean = dstd = 0.0; 
      kk2 = 0;
      for (iChain = 0; iChain < nChains; iChain++) 
      { 
        if (chainStatus[iChain] == 0)
        {
          for (jj = 0; jj < chainCnt; jj++) 
          {
            dmean += XChains[iChain][jj][kk];
            kk2++;
          }
        }
      }
      if (kk2 > 0) dmean /= (double) kk2;
      for (iChain = 0; iChain < nChains; iChain++) 
      { 
        if (chainStatus[iChain] == 0)
        {
          for (jj = 0; jj < chainCnt; jj++) 
            dstd += pow(XChains[iChain][jj][kk]-dmean,2.0);
        }
      }
      if (kk2 > 0) dstd = sqrt(dstd/(double) kk2);
      if (plotMatlab())
        fprintf(fp,
         "disp(['Stat for Input %d: mean,std = ',num2str(%e),' ',num2str(%e)])\n",
         kk+1,dmean,dstd);
    }
  }
  if (plotScilab())
    printOutTS(PL_INFO, "MCMC: scilabmcmc2.sci file has been created.\n");
  else
  {
    fprintf(fp,"disp('Lower diagonal plots: priors')\n");
    fprintf(fp,"disp('Upper diagonal plots: posteriors')\n");
    printOutTS(PL_INFO, "MCMC: matlabmcmc2.m file has been created.\n");
  }
  fclose(fp);
  return 0;
}

// ************************************************************************
// write to another matlab file the mse of the posterior sample with the
// experimental data
// ------------------------------------------------------------------------
int MCMCAnalyzer::genPostLikelihood(int nInputs, double *lower, 
                  double *upper, double *XRange, int numChains, 
                  int chainCnt, double ***XChains, int *chainStatus, 
                  int chainLimit, int *rsIndices, double *rsValues, 
                  psIVector vecDesignP, int dnInputs, int dnSamples, 
                  psMatrix matExpInps, FuncApprox **faPtrs, 
                  int nOutputs, double *discOutputs,
                  double *discFuncConstantMeans, psMatrix matExpMeans,
                  psMatrix matExpStdvs, int hasModelForm, 
                  int modelFormConst, psIVector &vecModelForms)
                        
{
  int    iChain, ii, jj, ii2, kk2, dcnt;  
  double ddata, *YGuessS, Ytemp, Ytemp2, ddata2;
  FILE   *fp;
  psVector vecXGuessS, vecXDesignS, vecYGuessS, vecYDesignS;

  fp = fopen("matlabpostlikelihood.m", "w");
  if (fp == NULL) return -1;

  vecXGuessS.setLength(dnSamples * nInputs);
  vecXDesignS.setLength(dnSamples * nInputs);
  vecYGuessS.setLength(dnSamples * nOutputs);
  vecYDesignS.setLength(dnSamples * nOutputs);
  YGuessS = vecYGuessS.getDVector();
  fprintf(fp, "A = [\n");
  for (iChain = 0; iChain < numChains; iChain++)
  {
    if (chainStatus[iChain] == 0)
    {
      for (ii = chainCnt-chainLimit; ii < chainCnt; ii++)
      {
        for (jj = 0; jj < nInputs; jj++)
        {
          if ((rsIndices == NULL || rsIndices[jj] >= 0) &&
              (vecDesignP.length() == 0 || vecDesignP[jj] == 0))
          {
            ddata = XChains[iChain][ii][jj];
            fprintf(fp, "%e ", ddata);
          }
          else if (rsIndices != NULL && rsIndices[jj] < 0)
            fprintf(fp, "%e ", rsValues[jj]);
          else if (vecDesignP.length() > 0 && vecDesignP[jj] != 0)
            fprintf(fp, "%e ", 0.5 * (upper[jj] + lower[jj]));
        }
        //**/ fill in input (including design) values for all experiments
        for (kk2 = 0; kk2 < dnSamples; kk2++)
        {
          dcnt = 0;
          for (ii2 = 0; ii2 < nInputs; ii2++)
          {
            vecXGuessS[kk2*nInputs+ii2] = XChains[iChain][ii][ii2]; 
            if (vecDesignP.length() > 0 && vecDesignP[ii2] == 1)
            {
              vecXGuessS[kk2*nInputs+ii2] = matExpInps.getEntry(kk2,dcnt);
              vecXDesignS[kk2*dnInputs+dcnt]=matExpInps.getEntry(kk2,dcnt);
              dcnt++;
            }
          }
        }
        //**/ evaluate the sample point through the response surface
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          faPtrs[ii2]->evaluatePoint(dnSamples,vecXGuessS.getDVector(),
                                     &YGuessS[ii2*dnSamples]);
          if (hasModelForm)
          {
            if (modelFormConst == 0)
            {
              for (kk2 = 0; kk2 < dnSamples; kk2++)
                vecYDesignS[ii2*dnSamples+kk2] = 
                                     discOutputs[ii2*dnSamples+kk2];
            }
            else if (discFuncConstantMeans != NULL &&
                discFuncConstantMeans[0] != PSUADE_UNDEFINED)
            {
              for (kk2 = 0; kk2 < dnSamples; kk2++)
                vecYDesignS[ii2*dnSamples+kk2] = discFuncConstantMeans[ii2];
            }
            else
            {
              for (kk2 = 0; kk2 < dnSamples; kk2++)
                vecYDesignS[ii2*dnSamples+kk2] = 0.0;
            }
          }
        }
        //**/ compute the rms (weighted and unweighted)
        ddata = ddata2 = 0.0;
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
          {
            if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
              Ytemp = vecYGuessS[ii2*dnSamples+kk2] + 
                      vecYDesignS[ii2*dnSamples+kk2];
            else
              Ytemp = vecYGuessS[ii2*dnSamples+kk2]; 
            Ytemp2 = pow(Ytemp-matExpMeans.getEntry(kk2,ii2),2.0) /
                     pow(matExpStdvs.getEntry(kk2,ii2),2.0);
            ddata += Ytemp2;
            Ytemp2 = pow(Ytemp-matExpMeans.getEntry(kk2,ii2),2.0); 
            ddata2 += Ytemp2;
          }
        }
        ddata /= (dnSamples*nOutputs);
        ddata2 /= (dnSamples*nOutputs);
        fprintf(fp, "%e %e\n", ddata, ddata2);
      }
    }
  }
  fprintf(fp,"];\n");
  fprintf(fp,"figure(1)\n");
  fprintf(fp,"Y = A(:,%d);\n", nInputs+1);
  fprintf(fp,"subplot(1,2,1)\n");
  fprintf(fp,"hist(Y, 20);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Weighted MSE ','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Frequencies','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"subplot(1,2,2)\n");
  fprintf(fp,"plot(Y, 'lineWidth', 2);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Sample number','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Weighted MSE','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"figure(2)\n");
  fprintf(fp,"Y2 = A(:,%d);\n", nInputs+2);
  fprintf(fp,"subplot(1,2,1)\n");
  fprintf(fp,"hist(Y2, 20);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('MSE','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Frequencies','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"subplot(1,2,2)\n");
  fprintf(fp,"plot(Y2, 'lineWidth', 2);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Sample number','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Weighted MSE','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fclose(fp);
  printOutTS(PL_INFO,"MCMC: matlabpostlikelihood.m file has been created.\n");
  return 0;
}

// ************************************************************************
// read spec (experimental data) file - more structured version
// ------------------------------------------------------------------------
double MCMCAnalyzer::readSpecFile(int nInputs, int nOutputs,  
                       psIVector &vecDParams, psMatrix &matExpInps, 
                       psMatrix &matExpMeans, psMatrix &matExpStdvs, 
                       int &dnReps, int printLevel)
{
  int    ii, jj, kk, cnt, dnSamples, dnInputs;
  double ddata, ddata2;
  char   lineIn[1001], cfname[1001], cword[101], cword2[101], cword3[101];
  FILE   *fp=NULL;

  //**/ display header
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"*** MCMC readSpecFile\n");
    printOutTS(PL_INFO,
         "*** NEED DATA TO CREATE LIKELIHOOD FUNCTION: \n\n");
    printOutTS(PL_INFO,"MCMC creates a Gaussian likelihood function. ");
    printOutTS(PL_INFO,"Please provide a data\n");
    printOutTS(PL_INFO,"file containing design parameter values, mean, ");
    printOutTS(PL_INFO,"and std. dev. of the\n");
    printOutTS(PL_INFO,"observation data for each output.\n");
    printOutTS(PL_INFO,"NOTE: Design parameters should be defined in ");
    printOutTS(PL_INFO,"the observation data\n");
    printOutTS(PL_INFO,"   file if the data used in MCMC are collected ");
    printOutTS(PL_INFO,"at different design\n");
    printOutTS(PL_INFO,"   points.\n");
    printOutTS(PL_INFO,
         "IMPORTANT: IF m DESIGN PARAMETERS ARE SPECIFIED, ");
    printOutTS(PL_INFO,"YOU NEED TO SPECIFY\n");
    printOutTS(PL_INFO,
         "   WHICH ONES THEY ARE. THESE DESIGN PARAMETERS ");
    printOutTS(PL_INFO,"WILL BE EXCLUDED FROM\n");
    printOutTS(PL_INFO,"   THE CALIBRATION PARAMETER SET.\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,
         "** OBSERVATION DATA FILE FORMAT : (O1 = Output 1) \n");
    printOutTS(PL_INFO,"        M   - no. of design parameters, \n");
    printOutTS(PL_INFO,"        K   - no. of model outputs, \n");
    printOutTS(PL_INFO,"        P   - no. of experiments \n");
    printOutTS(PL_INFO,"        O1m - Output 1 mean\n");
    printOutTS(PL_INFO,"        O1s - Output 1 std. dev.\n");
    printOutTS(PL_INFO,"        OKs - Output K std. dev.\n");
    printOutTS(PL_INFO,"# num_replications = <x> (Optional command)\n");
    printOutTS(PL_INFO,"PSUADE_BEGIN\n");
    printOutTS(PL_INFO,"<P> <K> <M> <design parameter identifiers>\n");
    printOutTS(PL_INFO,"1 <design values...> <O1m> <O1s> ... <OKs> \n");
    printOutTS(PL_INFO,"2 <design values...> <O1m> <O1s> ... <OKs> \n");
    printOutTS(PL_INFO,"...\n");
    printOutTS(PL_INFO,"P <design values...> <O1m> <O1s> ... <OK> \n");
    printOutTS(PL_INFO,"PSUADE_END\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"The likelihood function is in the form of:\n");
    printOutTS(PL_INFO,"  C exp(-0.5*S) \n");
    printOutTS(PL_INFO,"where C is the normalization constant and\n");
    printOutTS(PL_INFO,
         "  S=sum_{p=1}^P sum_{k=1)^K (Y_pk-m_pk)^2/sd_pk^2\n");
    printOutTS(PL_INFO,
         "where K is the number of outputs and m_pk and ");
    printOutTS(PL_INFO,"sd_pk are the mean and\n");
    printOutTS(PL_INFO,"  std. dev. of output k of experiment k.\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"NOTE: Alternatively, your simulator (or ");
    printOutTS(PL_INFO,"response surface) output may\n");
    printOutTS(PL_INFO,
         "      be some error measure from comparison of ");
    printOutTS(PL_INFO,"all model outputs with\n");
    printOutTS(PL_INFO, "      observation data. In this case, set ");
    printOutTS(PL_INFO, "nOutputs=1, mean=0 and sd=1\n");
    printOutTS(PL_INFO,"      in the specification file (i.e. your ");
    printOutTS(PL_INFO,"simulation output is 'S'\n");
    printOutTS(PL_INFO,
         "      above, and MCMC will compute likelihood as :\n");
    printOutTS(PL_INFO,"            C exp(-0.5 S).\n");
    printOutTS(PL_INFO,
         "      However, if you choose this option and you ");
    printOutTS(PL_INFO,"want to add response\n");
    printOutTS(PL_INFO,
         "      surface uncertainty, you need to be very careful.\n");
  }
 
  //**/ read spec file for likelihood function
  printf("==> Enter the name of the experiment file: ");
  scanf("%s", cfname);
  fgets(lineIn, 1000, stdin);
  kk = strlen(cfname);
  if (kk <= 1000)
  {
    cfname[kk] = '\0';
    fp = fopen(cfname, "r");
    if (fp == NULL)
    {
      printOutTS(PL_ERROR,
        "MCMC ERROR: Cannot open experiment spec file %s.\n",cfname);
      return PSUADE_UNDEFINED;
    }
  }
  else
  {
    printOutTS(PL_ERROR,"MCMC ERROR: Spec file name too long.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ look for information to indicate whether experiments are
  //**/ replicated so that the effective dnSamples is less
  //**/ default  = effective sample size is equal to dnSamples
  //**/ optional = 'num_replications = <xx>' 
  int lineCnt = 0;
  lineIn[0] = '#';
  dnReps = 1;
  while (lineIn[0] == '#')
  {
    fgets(lineIn, 2000, fp);
    cword2[0] = 'N';
    sscanf(lineIn, "%s %s %s %d", cword, cword2, cword3, &kk);
    if (!strcmp(cword2, "num_replications"))
    {
      dnReps = kk;
      printf("MCMC SPECFILE INFO: \n");
      printf("     likelihood to be scaled by kk=%d\n",kk);
      printf("     likelihood = 1/kk sum_i (exp_i-model_i)^2\n");
      printf("     This scaling is for repeated experiments.\n");
    }
    lineCnt++;
  }
  sscanf(lineIn, "%s", cword); 
  if (!strcmp(cword, "PSUADE_BEGIN")) lineCnt++;
  lineCnt--;
  fclose(fp);
  fp = fopen(cfname, "r");
  for (ii = 0; ii < lineCnt; ii++) fgets(lineIn, 2000, fp);
  fscanf(fp, "%d %d %d", &dnSamples, &kk, &dnInputs);
  if (dnSamples <= 0)
  {
    printOutTS(PL_ERROR,"MCMC SPEC FILE ERROR: No. of experiments <= 0.\n");
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printOutTS(PL_INFO,
       "MCMC SPEC FILE: Number of experiments found = %d\n",dnSamples);

  if (kk != nOutputs)
  {
    printOutTS(PL_ERROR,"MCMC ERROR: nOutputs in spec file ");
    printOutTS(PL_ERROR,"does not match PSUADE file nOutputs.\n");
    printOutTS(PL_ERROR,"     %d (in spec file) versus %d\n",kk,nOutputs);
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printOutTS(PL_INFO,
     "MCMC SPEC FILE INFO: Number of outputs = %d\n",nOutputs);

  if (dnInputs < 0)
  {
    printOutTS(PL_ERROR,
      "MCMC SPEC FILE ERROR: Number of design variables found < 0.\n");
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  if (dnInputs > nInputs)
  {
    printOutTS(PL_ERROR,
      "MCMC SPEC FILE ERROR: Number of design variables %d\n",dnInputs);
    printOutTS(PL_ERROR,
      "     This cannot be larger than the total number inputs (%d)\n",
      nInputs);
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printOutTS(PL_INFO,
    "MCMC SPEC FILE INFO: Number of design parameters = %d\n",dnInputs);
  if (dnInputs > 0)
  {
    vecDParams.setLength(nInputs);
    cnt = 0;
    for (ii = 0; ii < dnInputs; ii++)
    {
      fscanf(fp, "%d", &kk);
      if (kk <= 0 || kk > nInputs)
      {
        printOutTS(PL_ERROR,
           "MCMC SPEC FILE ERROR: Invalid design parameter %d.\n",kk);
        printOutTS(PL_ERROR,
           "                      Please check your spec file format.\n");
        fclose(fp);
        return PSUADE_UNDEFINED;
      }
      if (kk <= cnt)
      {
        printOutTS(PL_ERROR,"MCMC SPEC FILE ERROR: Design parameters ");
        printOutTS(PL_ERROR,"should be in ascending order.\n");
        fclose(fp);
        return PSUADE_UNDEFINED;
      }
      vecDParams[kk-1] = 1;
      printOutTS(PL_INFO,
        "MCMC SPEC FILE INFO: Input %d is a design parameter\n",kk);
      cnt = kk;
    }
    matExpInps.setFormat(PS_MAT1D);
    matExpInps.setDim(dnSamples, dnInputs);
  }
  matExpMeans.setFormat(PS_MAT1D);
  matExpStdvs.setFormat(PS_MAT1D);
  matExpMeans.setDim(dnSamples, nOutputs);
  matExpStdvs.setDim(dnSamples, nOutputs);

  int stdAllOnes=1;
  for (ii = 0; ii < dnSamples; ii++)
  {
    fscanf(fp, "%d", &kk);
    if (kk != ii+1)
    {
      printOutTS(PL_ERROR,
        "MCMC SPEC FILE ERROR: Invalid experiment number %d\n",kk);
      printOutTS(PL_ERROR,
        "          at line %d in the spec file. Expecting %d.\n",
        ii+2,ii+1);
      printOutTS(PL_ERROR,
        "          ==> check line %d\n", ii+3);
      fclose(fp);
      return PSUADE_UNDEFINED;
    }
    if (printLevel > 0)
      printOutTS(PL_INFO,"MCMC Calibration Data Set %d\n", kk);
    for (jj = 0; jj < dnInputs; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      matExpInps.setEntry(ii, jj, ddata);
      if (printLevel > 0)
        printOutTS(PL_INFO,"   Design parameter %d = %e\n",jj+1,ddata);
    }
    for (jj = 0; jj < nOutputs; jj++)
    {
      fscanf(fp, "%lg %lg", &ddata, &ddata2);
      matExpMeans.setEntry(ii, jj, ddata);
      matExpStdvs.setEntry(ii, jj, ddata2);
      if (ddata2 != 1) stdAllOnes = 0;
      if (printLevel > 0)
         printOutTS(PL_INFO,"      Data mean/stdev = %16.8e %16.8e\n",
                    ddata, ddata2);
      if (ddata2 < 0.0)
      {
        fclose(fp);
        printf("MCMC ERROR: std dev in spec file <= 0.\n");
        printf("==> check the last entry in line %d\n", ii+3);
        return PSUADE_UNDEFINED;
      }
    }
  }
  fclose(fp);
  printEquals(PL_INFO, 0);
  //**/ In case all standard deviation = 1, print a message
#if 0
  if (stdAllOnes == 1)
  {
    printf("MCMC INFO: std dev in specification file = 1.\n");
    printf("     The standard deviations in the specification ");
    printf("file is expected to\n");
    printf("     reflect experimental uncertainties. Setting ");
    printf("them to all ones - \n");
    printf("     which may be because there is a lack of ");
    printf("information - can alter\n");
    printf("     the posterior distributions substantially. ");
    printf("As such, we recommend\n");
    printf("     that the MCMCPostSample file be examined ");
    printf("after inference to see\n");
    printf("     whether the negative log-likelihoods for all ");
    printf("experimental outputs\n");
    printf("     (in the MLE statistics section) may be either ");
    printf("too small or too\n");
    printf("     large. If they are all small, this may explain ");
    printf("why the posterior\n");
    printf("     distributions may be too flat. If they are all ");
    printf("large, this may\n");
    printf("     explain why the posterior distributions may be ");
    printf("too narrow. Please\n");
    printf("     try your best to prescribe experimental ");
    printf("standard deviations more\n";
    printf("     carefully so that the inference results make ");
    printf("sense and are thus\n");
    printf("     more  defensible.\n");
  }
#endif
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
int MCMCAnalyzer::setParams(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  Analyzer::setParams(argc, argv);
  if      (!strcmp(request, "setsim")) mode_ = 1;
  else if (!strcmp(request, "MCMC_brute_force")) bfmode_ = 1;
  else if (!strcmp(request, "MCMC_gibbs")) bfmode_ = 0;
  else
  {
    printOutTS(PL_ERROR,"MCMCAnalyzer ERROR: setParams - not valid.\n");
    exit(1);
  }
  return 0;
}

// ************************************************************************
// check convergence 
// ------------------------------------------------------------------------
double MCMCAnalyzer::checkConvergence(int num, double *means, double *stds,
                                      int leng, double thresh)
{
  int    ii;
  double WStat, BStat, ddata, ddata2;
  //**/ compute PSRF
  WStat = 0.0;
  for (ii = 0; ii < num; ii++) WStat += stds[ii];
  WStat /= (double) num;
  if (WStat < 0) WStat = PSUADE_UNDEFINED;
  ddata = 0.0;
  for (ii = 0; ii < num; ii++) ddata += means[ii];
  ddata /= (double) num;
  BStat = 0.0;
  for (ii = 0; ii < num; ii++) 
     BStat += pow(means[ii]-ddata,2.0);
  BStat = BStat / (num - 1.0) * leng;
  ddata = (1 - 1.0/leng) * WStat + BStat / leng;
  ddata = ddata / WStat * (num + 1) / num - 
          (leng - 1.0) / (double) (leng * num); 
  if (ddata < 0) ddata2 = PSUADE_UNDEFINED;
  else           ddata2 = sqrt(ddata);
  printf("MCMC convergence check: %e <? %e\n", ddata2, thresh);
  return ddata2;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
MCMCAnalyzer& MCMCAnalyzer::operator=(const MCMCAnalyzer &)
{
   printOutTS(PL_ERROR,
        "MCMCAnalyzer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

// ************************************************************************
// read in response surface index file
// rsIndices[ii] == -1  ==> fixed parameters
// samMatrix ==> if there is any uncertain parameter, it has the sample
// ------------------------------------------------------------------------
int MCMCAnalyzer::readIndexFile(PsuadeData *dataPtr, psIVector &rsIndices, 
                          psVector &rsValues, psMatrix &samMatrix)
{
  int    kk, ii, nInputs, nSamp, numUParams=0, status;
  double ddata;
  char   *rsFile, inStr[1000], samFileName[1000];
  pData  pPtr;
  FILE   *fileIn;

  dataPtr->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  dataPtr->getParameter("ana_rsindexfile", pPtr);
  rsFile = pPtr.strArray_[0];
  if (strcmp(rsFile, "NONE"))
  {
    printOutTS(PL_INFO,
         "MCMC: A response surface index file has been specified.\n");
    fileIn = fopen(rsFile, "r");
    if (fileIn == NULL)
    {
      printOutTS(PL_ERROR,
         "MCMC ERROR: rs_index_file %s not found.\n",rsFile);
      return -1;
    }
    else
    {
      printOutTS(PL_INFO,"INFO: rs_index_file %s found.\n",rsFile);
      fscanf(fileIn, "%d", &kk);
      if (kk != nInputs)
      {
        printOutTS(PL_ERROR,
             "MCMC ERROR: invalid nInputs in rs_index_file (%d != %d).\n",
             kk, nInputs);
        printOutTS(PL_ERROR,"  Data format should be: \n");
        printOutTS(PL_ERROR,
             "  line 1: nInputs in rs data (driver) file\n");
        printOutTS(PL_ERROR,
             "  line 2: 1 1 <Input 1 is a calibration parameter>\n");
        printOutTS(PL_ERROR,
             "  line 3: 2 0 val <Input 2 is a parameter fixed at val>\n");
        printOutTS(PL_ERROR,
             "  line 4: 3 999 1 uSamp <Input 3 is an uncertain parameter>\n");
        printOutTS(PL_ERROR,"  (999   means uncertain parameter)\n");
        printOutTS(PL_ERROR,"  (1     means first column in uSamp)\n");
        printOutTS(PL_ERROR,"  (uSamp is a sample file)\n");
        printOutTS(PL_ERROR,"  ...\n");
        fclose(fileIn);
        return -1;
      }
      rsIndices.setLength(nInputs);
      rsValues.setLength(nInputs);
      strcpy(samFileName, "none");
      for (ii = 0; ii < nInputs; ii++)
      {
        rsIndices[ii] = 0;
        fscanf(fileIn, "%d", &kk);
        //**/ first number must be a sequential input number
        if (kk != ii+1)
        {
          printOutTS(PL_ERROR,
               "MCMC ERROR: 1st index in indexFile = %d (must be %d)).\n",
               kk, ii+1);
          printOutTS(PL_ERROR,"  Data format should be: \n");
          printOutTS(PL_ERROR,
               "  line 1: nInputs in rs data (driver) file\n");
          printOutTS(PL_ERROR,
               "  line 2: 1 1 <Input 1 is a calibration parameter>\n");
          printOutTS(PL_ERROR,
               "  line 3: 2 0 val <Input 2 is a parameter fixed at val>\n");
          printOutTS(PL_ERROR,
               "  line 4: 3 999 1 uSamp <Input 3 is an uncertain parameter>\n");
          printOutTS(PL_ERROR,"  (999   means uncertain parameter)\n");
          printOutTS(PL_ERROR,"  (1     means first column in uSamp)\n");
          printOutTS(PL_ERROR,"  (uSamp is a sample file)\n");
          printOutTS(PL_ERROR,"  ...\n");
          fclose(fileIn);
          return -1;
        }
        //**/ second number is the input number, 0, or 999
        fscanf(fileIn, "%d", &kk);
        rsIndices[ii] = kk;
        if (rsIndices[ii] == 0)
          printOutTS(PL_INFO,"MCMC INFO: input %3d inactive\n",ii+1);

        //**/ if uncertain parameter, get index and sample file 
        if (rsIndices[ii] == 999)
        {
          fscanf(fileIn, "%d", &kk);
          fscanf(fileIn, "%s", inStr);
          if (!strcmp(samFileName, "none")) strcpy(samFileName, inStr);
          else if (strcmp(samFileName, inStr))
          {
            printOutTS(PL_ERROR,
                 "MCMC readIndexFile ERROR: sample file for all\n");
            printOutTS(PL_ERROR,
                 "         uncertain parameters must be the same.\n");
            exit(1);
          }
          numUParams++;
          rsIndices[ii] = kk + 1000;
        }
        else if (rsIndices[ii] < 0 || rsIndices[ii] > nInputs)
        {
          printOutTS(PL_ERROR,
               "MCMC readIndexFile ERROR: input %3d = %d invalid\n",ii+1,
               rsIndices[ii]);
          fclose(fileIn);
          return -1;
        }

        rsIndices[ii] = rsIndices[ii] - 1;
        if (rsIndices[ii] >= -1 && rsIndices[ii] < nInputs)
        {
          fscanf(fileIn, "%lg", &ddata);
          rsValues[ii] = ddata;
        }
      }
      fclose(fileIn);
      printOutTS(PL_INFO, "Response surface index information: \n");
      for (ii = 0; ii < nInputs; ii++)
      {
        if (rsIndices[ii] == -1)
          printOutTS(PL_INFO, "Input %4d: fixed at default value  = %e\n",
                     ii+1, rsValues[ii]);
        else if (rsIndices[ii] >= 1000)
          printOutTS(PL_INFO, "Input %4d: uncertain, sample index = %4d\n",
                     ii+1, rsIndices[ii]-999);
        else
          printOutTS(PL_INFO, "Input %4d: calibration/design parameter\n",
                     ii+1);
      }
      //**/ if there is any uncertain parameter, read in the sample
      if (numUParams > 0)
      {
        status = readSampleInputFile(samFileName,rsIndices,samMatrix);
        if (status < 0) return -1;
      }
    }
  }
  return 0;
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void MCMCAnalyzer::cleanUp()
{
  char pString[200];
  FILE *fp = fopen("psuade_stop", "r");
  if (fp != NULL)
  {
     fclose(fp);
     fp = NULL;
     printOutTS(PL_INFO,
          "MCMC INFO: psuade_stop FILE FOUND. WILL BE REMOVED\n");
     strcpy(pString, "psuade_stop");
     unlink(pString);
  }
  fp = fopen("psuade_master", "r");
  if (fp != NULL)
  {
     fclose(fp);
     fp = NULL;
     printOutTS(PL_INFO,
          "MCMC INFO: psuade_master FILE FOUND. WILL BE REMOVED\n");
     strcpy(pString, "psuade_master");
     unlink(pString);
  }
  fp = fopen("psuade_nomaster", "r");
  if (fp != NULL)
  {
     fclose(fp);
     fp = NULL;
     printOutTS(PL_INFO,
          "MCMC INFO: psuade_nomaster FILE FOUND. WILL BE REMOVED\n");
     strcpy(pString, "psuade_nomaster");
     unlink(pString);
  }
  fp = fopen("psuade_gm", "r");
  if (fp != NULL)
  {
     fclose(fp);
     fp = NULL;
     printOutTS(PL_INFO,
          "MCMC INFO: psuade_gm FILE FOUND. WILL BE REMOVED\n");
     strcpy(pString, "psuade_gm");
     unlink(pString);
  }
  fp = fopen("psuade_nogm", "r");
  if (fp != NULL)
  {
     fclose(fp);
     fp = NULL;
     printOutTS(PL_INFO,
          "MCMC INFO: psuade_nogm FILE FOUND. WILL BE REMOVED\n");
     strcpy(pString, "psuade_nogm");
     unlink(pString);
  }
}

// ************************************************************************
// display banner 
// ************************************************************************
void MCMCAnalyzer::displayBanner(int printLevel)
{
  char lineIn[5000];
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*                     MCMC Optimizer\n");
  printEquals(PL_INFO, 0);
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
    printOutTS(PL_INFO," * ana_expert to finetune MCMC parameters, \n");
    printOutTS(PL_INFO,
         "   (e.g. sample size for burn-in can be adjusted).\n");
    printOutTS(PL_INFO,
         " * rs_expert to customize response surface for MCMC,\n");
    printOutTS(PL_INFO," * printlevel 3 to display more diagnostics info.\n");
    printOutTS(PL_INFO," * printlevel 4 to display even more diagnostics.\n");
    printOutTS(PL_INFO," * printlevel >=5 reserved only for expert only.\n");
    printDashes(PL_INFO,0);
    printOutTS(PL_INFO,
         "FEATURES AVAILABLE IN THE CURRENT VERSION OF MCMC:\n");
    printOutTS(PL_INFO," * Support other than uniform prior distributions\n");
    printOutTS(PL_INFO,
         " * Support likelihood functions from multiple outputs\n");
    printOutTS(PL_INFO," * Option to include response surface errors for ");
    printOutTS(PL_INFO,"polynomial\n");
    printOutTS(PL_INFO,"   regressions, bootstrapped MARS, and ");
    printOutTS(PL_INFO,"Gaussian process (GP).\n");
    printOutTS(PL_INFO,"   - can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO," * Option to include model form errors in the form ");
    printOutTS(PL_INFO,"of discrepancy\n");
    printOutTS(PL_INFO,"   models\n");
    printOutTS(PL_INFO,"   - can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO,"   - not available for Metropolis-Hastings MCMC.\n");
    printOutTS(PL_INFO,
         " * Option to set some inputs as design parameters\n");
    printOutTS(PL_INFO,
         "   - to be specified in the observation data spec file\n");
    printOutTS(PL_INFO,
         " * Option to disable some parameters (set to default)\n");
    printOutTS(PL_INFO,
         "   - not available for Metropolis-Hastings MCMC\n");
    printOutTS(PL_INFO,
         "   - in case these parameters are not to be calibrated\n");
    printOutTS(PL_INFO,
         "   - use rs_index_file in PSUADE's ANALYSIS section\n");
    printOutTS(PL_INFO,"   - not available with discrepancy modeling\n");
    printOutTS(PL_INFO,
         " * MCMC can be terminated gracefully by creating a file ");
    printOutTS(PL_INFO, "named\n");
    printOutTS(PL_INFO,
         "   'psuade_stop' in the same directory while it is running\n");
    printOutTS(PL_INFO,"   (if it takes too long).\n");
    printOutTS(PL_INFO,
         " * For multi-modal posteriors, a large number of chains ");
    printOutTS(PL_INFO,"may be needed.\n");
    printOutTS(PL_INFO,"   The number of chains can be adjusted ");
    printOutTS(PL_INFO,"in ana_expert mode.\n");
#if 0
    printOutTS(PL_INFO,
         " * In GM mode, you have a few options to choose from:\n");
    printOutTS(PL_INFO,
         "   1. track proposal distribution at each MCMC iteration\n");
    printOutTS(PL_INFO,
         "      (MCMCDistTrack.m will be created at each iteration to ");
    printOutTS(PL_INFO,"give a\n");
    printOutTS(PL_INFO,
         "       snapshot of the current proposal distribution)\n");
    printOutTS(PL_INFO,
         "       NOTE: iteration will pause until further instructions\n");
    printOutTS(PL_INFO,
         "   2. track posterior distributions after each MCMC cycle\n");
    printOutTS(PL_INFO,
         "      (MCMCChainHistogram.m will be created after each ");
    printOutTS(PL_INFO,
         "cycle to give\n");
    printOutTS(PL_INFO,
         "       a snapshot of the current parameter posterior ");
    printOutTS(PL_INFO,"distributions.\n");
    printOutTS(PL_INFO,
         "       NOTE: iteration will pause until further instructions\n");
    printEquals(PL_INFO, 0);
    printf("Press ENTER to continue ");
    scanf("%c", lineIn);
#endif
  }
}

// ************************************************************************
// display banner for brute force version 2
// ************************************************************************
void MCMCAnalyzer::displayBanner_bf2(int printLevel)
{
  char lineIn[5000];
  if (printLevel > 0)
  {
    printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
    printOutTS(PL_INFO,
         " * ana_expert to finetune MCMC parameters, \n");
    printOutTS(PL_INFO,
         "   (e.g. sample size for burn-in can be adjusted).\n");
    printOutTS(PL_INFO,
         " * rs_expert to customize response surface for MCMC,\n");
    printDashes(PL_INFO,0);

    printOutTS(PL_INFO,"FEATURES AVAILABLE IN THE CURRENT VERSION OF MCMC:\n");
    printOutTS(PL_INFO,
         " * Support likelihood functions from multiple outputs\n");
    printOutTS(PL_INFO," * Option to include response surface errors \n");
    printOutTS(PL_INFO,"   - Can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO,
       " * Option to include model form errors in the form of discrepancy\n");
    printOutTS(PL_INFO,"   models\n");
    printOutTS(PL_INFO,"   - Can be selected in ana_expert mode.\n");
    printOutTS(PL_INFO,
         " * Option to set some inputs as design parameters\n");
    printOutTS(PL_INFO,
         "   - To be specified in the observation data spec file\n");
    printOutTS(PL_INFO,
         " * Option to disable some parameters (set to default)\n");
    printOutTS(PL_INFO,
         "   - In case these parameters are not to be calibrated\n");
    printOutTS(PL_INFO,
         "   - Use rs_index_file in PSUADE's ANALYSIS section\n");
    printOutTS(PL_INFO,"   - not available with discrepancy modeling\n");
    printOutTS(PL_INFO,
      " * Option to set some parameters as uncertain but not calibrated\n");
    printOutTS(PL_INFO,
      "   - A sample file is to be provided to characterize uncertainty\n");
    printOutTS(PL_INFO,
      "   - Use rs_index_file in PSUADE's ANALYSIS section\n");
    printOutTS(PL_INFO,
      "   - For parameters with fixed uncertainty, the code is 999.\n");
    printOutTS(PL_INFO, " * MCMC can be terminated gracefully by ");
    printOutTS(PL_INFO, "creating in the run directory\n");
    printOutTS(PL_INFO, "   a file called 'psuade_stop' (e.g. ");
    printOutTS(PL_INFO, "use 'touch psuade_stop').\n");
    printOutTS(PL_INFO, "   - If you use control-d, no ");
    printOutTS(PL_INFO, "MCMCPostSample/matlabmcmc2.m file will\n");
    printOutTS(PL_INFO, "be created.\n");
    printEquals(PL_INFO, 0);
    printf("Make sure the setup is correct, then press ENTER to continue ");
    scanf("%c", lineIn);
  }
}

// ************************************************************************
// create discrepancy function (either constant or some RS)
// ************************************************************************
double MCMCAnalyzer::createDiscrepancyFunctions(int nInputs, int nOutputs,
                       double *lower, double *upper, psIVector &rsIndices, 
                       psVector &rsValues, psIVector &dParams,
                       int dnInputs, int dnSamples, psMatrix &dSamInputs,
                       psMatrix &dSamMeans, PsuadeData *dataPtr, 
                       psVector &vecDiscFuncConstMeans,
                       psVector &vecDistFuncConstStds,
                       psVector &vecDiscSamOuts, FuncApprox **faPtrs,
                       int printLevel, int constFlag)
{
  int    ii, kk, ii2, cnt, askFlag=0, iOne=1, pOrder, rstype, status;
  double expdata, simdata, ddata;
  char   pString[1000], lineIn[10000];
  pData  qData;
  psVector   vecSettings, vecXT, vecYT;
  PsuadeData *dPtr;
  FuncApprox *localFaPtr;

  //**/ ---------------------------------------------------------------
  //**/ allocate space for discrepancy data ==> vecDiscSamOuts
  //**/ and initialize the constant discrepancy function
  //**/ qData is for fetching input names for storing discrepancy
  //**/ ---------------------------------------------------------------
  int ExpNSamples = dnSamples;
  vecDiscSamOuts.setLength(ExpNSamples*nOutputs);
  double *discSamOutputs = vecDiscSamOuts.getDVector();
  vecDiscFuncConstMeans.setLength(nOutputs);
  vecDistFuncConstStds.setLength(nOutputs);
  for (ii2 = 0; ii2 < nOutputs; ii2++)
  {
    vecDiscFuncConstMeans[ii2] = PSUADE_UNDEFINED;
    vecDistFuncConstStds[ii2]  = PSUADE_UNDEFINED;
  }
  if (dataPtr != NULL) dataPtr->getParameter("input_names", qData);

  //**/ ---------------------------------------------------------------
  //**/ set discrepancy function calibration parameter default 
  //**/ (to the middle of the range). Also set fixed parameters
  //**/ ==> vecSettings
  //**/ ---------------------------------------------------------------
  vecSettings.setLength(nInputs);
  for (ii2 = 0; ii2 < nInputs; ii2++)
  {
    vecSettings[ii2] = 0.5*(lower[ii2] + upper[ii2]);
    if (rsIndices.length() > 0 && rsIndices[ii2] < 0)
      vecSettings[ii2] = rsValues[ii2];
  }

  //**/ ---------------------------------------------------------------
  //**/ create and store discrepancy sample
  //**/ (same as experimental sample but with adjustment)
  //**/ thus, dSamMeans may be changed (if RS is used for discrepancy)
  //**/ ---------------------------------------------------------------
  psVector vecOneSample, vecLowers, vecUppers, vecSamIns;
  vecOneSample.setLength(nInputs);
  vecSamIns.setLength(ExpNSamples*nInputs);
  vecLowers.setLength(nInputs);
  vecUppers.setLength(nInputs);
  dPtr = new PsuadeData();
  psStrings Xnames, Ynames;
  for (ii = 0; ii < nOutputs; ii++)
  {
    //**/ ---------------------------------------------------------------
    //**/ for each output, compute discrepancies between
    //**/ simulation (evaluated at the experimental points)
    //**/ and experiment for every experiment
    //**/ ---------------------------------------------------------------
    for (kk = 0; kk < ExpNSamples; kk++)
    {
      //**/ ---------------------------------------------------------------
      //**/ inject the design, calibration default and fixed 
      //**/ parameter values (dSamInputs has design parameters
      //**/ for each of the ExpNSamples experiments)
      //**/ ---------------------------------------------------------------
      cnt = 0;
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        if (dParams.length() > 0 && dParams[ii2] == 1)
        {
          vecOneSample[ii2] = dSamInputs.getEntry(kk,cnt);
          cnt++;
        }
        else vecOneSample[ii2] = vecSettings[ii2];
      }

      //**/ ---------------------------------------------------------------
      //**/ users can choose whether to re-set the calibration
      //**/ parameters or use the ones in vecSettings
      //**/ ---------------------------------------------------------------
      simdata = 0.0;
      if (psConfig_.AnaExpertModeIsOn() && askFlag == 0)
      {
        printOutTS(PL_INFO,"To create discrepancy functions, the ");
        printOutTS(PL_INFO,"calibration parameters need to\n");
        printOutTS(PL_INFO,"be set to some nominal values. ");
        printOutTS(PL_INFO,"You can choose the nominal values, or\n");
        printOutTS(PL_INFO,"it will be set to the input means ");
        printOutTS(PL_INFO,"or mid points of the ranges.\n");
        snprintf(pString,100,"Set nomininal values yourself ? (y or n) ");
        getString(pString, lineIn);
        if (lineIn[0] == 'y')
        {
          for (ii2 = 0; ii2 < nInputs; ii2++)
          {
            //**/ if it is not fixed, and it is not a design parameter
            if ((rsIndices.length() == 0 || rsIndices[ii2] >= 0) &&
                (dParams.length() == 0 || dParams[ii2] == 0))
            {
              printOutTS(PL_INFO,
                  "Input %d has lower and upper bounds = %e %e\n",
                  ii2+1, lower[ii2], upper[ii2]);
              snprintf(pString,100, 
                      "Nominal value for input %d : ",ii2+1);
              vecOneSample[ii2] = getDouble(pString);
              vecSettings[ii2]  = vecOneSample[ii2];
            }
          }
        }
        askFlag = 1;
        for (ii2 = 0; ii2 < nInputs; ii2++)
        {
          if ((rsIndices.length() == 0 || rsIndices[ii2] >= 0) &&
              (dParams.length() == 0 || dParams[ii2] == 0))
            printf("Nominal value for input %d = %e\n",ii2+1,
                   vecSettings[ii2]);
        }
      }
      //**/ ---------------------------------------------------------------
      //**/ at this point, vecOneSample has been set to calibration default
      //**/ and design default (at sample kk) and at fixed values
      //**/ now evaluate it (ii-th output and kk-th designSample)
      //**/ ---------------------------------------------------------------
      simdata = faPtrs[ii]->evaluatePoint(vecOneSample.getDVector());
      expdata = dSamMeans.getEntry(kk, ii);

      if (printLevel >= 4)
      {
        printOutTS(PL_INFO, 
             "Experiment %4d (out of %d) : \n",kk+1,ExpNSamples);
        printOutTS(PL_INFO, "Inputs = \n");
        for (ii2 = 0; ii2 < nInputs; ii2++)
           printOutTS(PL_INFO, "%12.4e ",vecOneSample[ii2]);
        printOutTS(PL_INFO, 
             "\nSimuation/experimental data = %12.4e %12.4e\n",
             simdata, expdata);
      }
      //**/ ---------------------------------------------------------------
      //**/ generate the difference between experiment and simulation
      //**/ ==> disSamOutputs
      //**/ ---------------------------------------------------------------
      discSamOutputs[kk+ii*ExpNSamples] = expdata - simdata;
    }

    //**/ ---------------------------------------------------------------
    //**/ store discrepancy file
    //**/ (Note: expdata - simdata(interpolated at experimental points)
    //**/        are stored, BEFORE fitted to RS)
    //**/ ---------------------------------------------------------------
    if (dnInputs > 0)
    {
      for (kk = 0; kk < ExpNSamples; kk++)
        for (ii2 = 0; ii2 < dnInputs; ii2++)
          vecSamIns[kk*dnInputs+ii2] = dSamInputs.getEntry(kk,ii2);
      Xnames.setNumStrings(dnInputs);
      cnt = 0;
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        if (dParams[ii2] == 1)
        {
          vecLowers[cnt] = lower[ii2];
          vecUppers[cnt] = upper[ii2];
          if (qData.strArray_ == NULL)
               snprintf(pString,100,"X%d", ii2+1);
          else strcpy(pString, qData.strArray_[ii2]);
          Xnames.loadOneString(cnt, pString);
          cnt++;
        }
      }
      dPtr->updateInputSection(ExpNSamples,dnInputs,NULL,
                  vecLowers.getDVector(),vecUppers.getDVector(),
                  vecSamIns.getDVector(), Xnames.getStrings(), 
                  NULL,NULL,NULL,NULL);
    }
    else
    {
      Xnames.setNumStrings(1);
      strcpy(pString, "X0");
      Xnames.loadOneString(0, pString);
      //**/ set input = 0.5 (and later output = constant)
      for (ii2 = 0; ii2 < ExpNSamples; ii2++) vecSamIns[ii2] = 0.5;
      vecLowers[0] = 0.0;
      vecUppers[0] = 1.0;
      dPtr->updateInputSection(ExpNSamples,iOne,NULL,
                  vecLowers.getDVector(),vecUppers.getDVector(),
                  vecSamIns.getDVector(), Xnames.getStrings(), 
                  NULL,NULL,NULL,NULL);
    }
    psIVector vecStates;
    vecStates.setLength(ExpNSamples);
    for (kk = 0; kk < ExpNSamples; kk++) vecStates[kk] = 1;
    Ynames.setNumStrings(1);
    snprintf(pString,100,"Y%d", ii+1);
    Ynames.loadOneString(0, pString);
    dPtr->updateOutputSection(ExpNSamples,iOne,
                 &discSamOutputs[ii*ExpNSamples],
                 vecStates.getIVector(),Ynames.getStrings());
    dPtr->updateMethodSection(PSUADE_SAMP_MC, ExpNSamples, 1, -1, -1);
    snprintf(pString,100,"psDiscrepancyModel%d", ii+1);
    dPtr->writePsuadeFile(pString, 0);

    //**/ ---------------------------------------------------------------
    //**/ Next fit the discrepancy with a function, if requested
    //**/ and then modify dSamMeans accordingly
    //**/ (This is needed if dnInputs > 0)
    //**/ ---------------------------------------------------------------
    if (dnInputs > 0 && dnSamples > 1)
    {
      pOrder = 0;
      if (ExpNSamples >= dnInputs+1) pOrder++;
      if (ExpNSamples >= dnInputs*(dnInputs+1)/2+1) pOrder++;
      if (ExpNSamples >= dnInputs*(dnInputs+1)*(dnInputs+2)/6+1) pOrder++;
      printf("PSUADE provides the following options discrepancy modeling:\n"); 
      printf("0. Just use (expdata-simdata) as discrepancies at ");
      printf("experimental points.\n");
      printf("   Users are to create their own RS from the stored ");
      printf("discrepancy data\n");
      printf("   at each experimental points.\n");
      printf("1. MARS\n");
      printf("2. Kriging (The reason the internal GP is not an option ");
      printf("is because it\n");
      printf("   is the same as option 0 - exact interpolation at ");
      printf("experimental pts.)\n");
      if (pOrder > 0)
      {
        printf("3. Use polynomial function of order %d.\n",pOrder);
        snprintf(pString,100,"Response surface for output %d? (0-3) ",ii+1);
        rstype = getInt(0,3,pString);
      }
      else
      {
        snprintf(pString,100,"Response surface for output %d? (0-1) ",ii+1);
        rstype = getInt(0,1,pString);
      }
      if (rstype > 0)
      {
        if      (rstype  == 1) rstype = PSUADE_RS_MARS;
        else if (rstype  == 2) rstype = PSUADE_RS_KR;
        else
        {
          rstype = PSUADE_RS_REGR1;
          if      (pOrder == 2) rstype = PSUADE_RS_REGR1 + 1;
          else if (pOrder == 3) rstype = PSUADE_RS_REGR1 + 2;
        }
        localFaPtr = genFA(rstype, dnInputs, iOne, ExpNSamples);
        if (localFaPtr == NULL)
        {
          printOutTS(PL_ERROR,
             "MCMCAnalyzer ERROR: failed to create RS for discrepancy.\n");
          exit(1);
        }
        localFaPtr->setNPtsPerDim(100);
        cnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++)
        {
          if (dParams[ii2] == 1)
          {
            vecLowers[cnt] = lower[ii2];
            vecUppers[cnt] = upper[ii2];
            cnt++;
          }
        }
        localFaPtr->setBounds(lower, upper);
        localFaPtr->setOutputLevel(-1);
        vecXT.setLength(dnInputs*ExpNSamples);
        for (kk = 0; kk < ExpNSamples; kk++)
          for (ii2 = 0; ii2 < dnInputs; ii2++)
            vecXT[kk*dnInputs+ii2] = dSamInputs.getEntry(kk,ii2);
        vecYT.setLength(ExpNSamples);
        for (kk = 0; kk < ExpNSamples; kk++)
          vecYT[kk] = discSamOutputs[kk+ii*ExpNSamples];
        status = localFaPtr->initialize(vecXT.getDVector(), 
                                        vecYT.getDVector());
        if (status != 0)
        {
          printOutTS(PL_ERROR,
            "MCMC ERROR: Crash in RS initialize for discrepancy.\n");
          exit(1);
        }
        vecYT.setLength(ExpNSamples);
        localFaPtr->evaluatePoint(ExpNSamples,vecXT.getDVector(),
                                  vecYT.getDVector());
        double errNorm = 0;
        for (kk = 0; kk < ExpNSamples; kk++)
        {
          printf("Output %d experiment %d: data = %12.6e, RS eval = %12.6e\n",
                 ii+1,kk+1,discSamOutputs[kk+ii*ExpNSamples],vecYT[kk]);
          ddata = discSamOutputs[kk+ii*ExpNSamples] - vecYT[kk];
          if (discSamOutputs[kk+ii*ExpNSamples] != 0)
               errNorm += pow(ddata/discSamOutputs[kk+ii*ExpNSamples],2.0);
          else errNorm += pow(ddata,2.0);
          discSamOutputs[kk+ii*ExpNSamples] = vecYT[kk];
        }
        printf("RMSE of normalized interpolation errors for discrepancy = %e\n",
               sqrt(errNorm / ExpNSamples));
        printf("Please make sure this MSE is acceptable.\n");
        delete localFaPtr;
      }
    }

    //**/ ---------------------------------------------------------------
    //**/ constant discrepancy function is requested
    //**/ set the constant to be the mean
    //**/ ---------------------------------------------------------------
    if (constFlag == 1)
    {
      vecDiscFuncConstMeans[ii] = 0.0;
      for (kk = 0; kk < ExpNSamples; kk++)
        vecDiscFuncConstMeans[ii] += discSamOutputs[ii*ExpNSamples+kk];
      vecDiscFuncConstMeans[ii] /= (double) ExpNSamples;
      vecDistFuncConstStds[ii] = 0.0;
      for (kk = 0; kk < ExpNSamples; kk++)
        vecDistFuncConstStds[ii] +=
         pow(discSamOutputs[ii*ExpNSamples+kk]-vecDiscFuncConstMeans[ii],2.0);
      vecDistFuncConstStds[ii] = 
         sqrt(vecDistFuncConstStds[ii]/ExpNSamples);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ return the discrepancy sample 
  //**/ (in vecDiscFuncConstMeans, &vecDistFuncConstStds, and
  //**/  vecDiscSamOuts)
  //**/ ---------------------------------------------------------------
  delete dPtr;
  return 0.0;
}

// ************************************************************************
// for the MLE solution, dissect the component of likelihood
// ------------------------------------------------------------------------
int MCMCAnalyzer::writeMLEInfo(FILE *fp, int nInputs, int nOutputs,
                  FuncApprox **faPtrs, int *designPs, int *rsIndices, 
                  double *rsValues, double *vecXmax, int dnSamples, 
                  int dnInputs, double *dExpInps, double *dExpMeans, 
                  double *dExpStdvs, int modelFormFlag, int modelFormConst, 
                  double *discOutputs, double *discFuncConstantMeans,
                  psIVector &vecModelForms) 
{
  int      dcnt, ii2, kk2;
  double   *YSample, *YDesign, YT1, YT2, stdev, stdv2;
  psVector vecXSam, vecYSam, vecXDes, vecYDes;

  if (fp == NULL) return -1;
  vecXSam.setLength(nInputs*dnSamples);
  vecYSam.setLength(nOutputs*dnSamples);
  vecXDes.setLength(dnInputs*dnSamples);
  vecYDes.setLength(nOutputs*dnSamples);
  YSample = vecYSam.getDVector();
  YDesign = vecYDes.getDVector();
  fprintf(fp, "Optimal parameter values: \n");
  dcnt = 0;
  for (ii2 = 0; ii2 < nInputs; ii2++) 
    fprintf(fp, "%16.8e\n", vecXmax[ii2]);
  fprintf(fp,
    "MLE Statistics (prediction, exp data, sd, -loglikelihood)\n");
  for (kk2 = 0; kk2 < dnSamples; kk2++)
  {
    dcnt = 0;
    for (ii2 = 0; ii2 < nInputs; ii2++) 
    {
      vecXSam[kk2*nInputs+ii2] = vecXmax[ii2];
      if (designPs != NULL && designPs[ii2] == 1)
      {
        vecXSam[kk2*nInputs+ii2] = dExpInps[kk2+dnSamples*dcnt];
        vecXDes[kk2*dnInputs+dcnt] = dExpInps[kk2+dnSamples*dcnt];
        dcnt++;
      }
      if (rsIndices != NULL && rsIndices[ii2] < 0)
        vecXSam[kk2*nInputs+ii2] = rsValues[ii2];
    }
  }
  for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++)
    YDesign[ii2] = YSample[ii2] = 0.0;
  for (ii2 = 0; ii2 < nOutputs; ii2++) 
  {
    faPtrs[ii2]->evaluatePoint(dnSamples,vecXSam.getDVector(),
                               &YSample[ii2*dnSamples]);
    if (modelFormFlag == 1)
    {
      if (modelFormConst == 0)
      {
        for (kk2 = 0; kk2 < dnSamples; kk2++)
          YDesign[ii2*dnSamples+kk2] = discOutputs[ii2*dnSamples+kk2];
      }
      else if (discFuncConstantMeans != NULL &&
             discFuncConstantMeans[ii2] != PSUADE_UNDEFINED)
      {
        for (kk2 = 0; kk2 < dnSamples; kk2++)
          YDesign[ii2*dnSamples+kk2] = discFuncConstantMeans[ii2];
      }
      else
      {
        for (kk2 = 0; kk2 < dnSamples; kk2++)
          YDesign[ii2*dnSamples+kk2] = 0;
      }
    }
  }
  for (kk2 = 0; kk2 < dnSamples; kk2++)
  {
    for (ii2 = 0; ii2 < nOutputs; ii2++)
    {
      if (vecModelForms.length() > 0 && vecModelForms[ii2] == 1)
        YT1 = YSample[ii2*dnSamples+kk2] + YDesign[ii2*dnSamples+kk2];
      else
        YT1 = YSample[ii2*dnSamples+kk2];
      YT2 = dExpMeans[kk2+dnSamples*ii2];
      stdev = dExpStdvs[kk2+dnSamples*ii2];
      stdv2 = 0.5 * pow(YT1 - YT2, 2.0) / (stdev * stdev);
      fprintf(fp,"%4d %16.8e %16.8e %16.8e %16.8e\n",kk2+1,
              YT1,YT2,stdev,stdv2);
    }
  }
  return 0;
}

// ************************************************************************
// perform MCMC-like analysis (brute force): for when the sample inputs
// and outputs are actually a prior sample with its corresponding outputs.
// ------------------------------------------------------------------------
double MCMCAnalyzer::analyzeDirect_nors(McmcData &mdata)
{
  int    ii, jj, kk, mm, nn;
  double dmean, dstdv, ddata, ddata2;

  //**/ ---------------------------------------------------------------
  //**/ MatExpInputs - experimental sample inputs
  //**/ MatExpMeans - experimental sample output means
  //**/ MatExpStds - experimental sample output standard deviation
  //**/ ---------------------------------------------------------------

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  int nInputs   = mdata.nInputs_;
  int nOutputs  = mdata.nOutputs_;
  int nSamples  = mdata.nSamples_;
  if (mdata.VecLowerB_.length() != nInputs ||
      mdata.VecUpperB_.length() != nInputs)
  {
    printf("MCMC ERROR: input lower/upper bound problem.\n");
    printf("            Lower bound vector has length %d\n",
           mdata.VecLowerB_.length());
    printf("            Upper bound vector has length %d\n",
           mdata.VecUpperB_.length());
    printf("            nInputs = %d\n", nInputs_);
    exit(1);
  }
  if (mdata.VecSamInputs_.length() != nInputs*nSamples)
  {
    printf("MCMC ERROR: sample inputs has wrong length.\n");
    printf("            %d versus (expected) %d\n",
           mdata.VecSamInputs_.length(), nInputs*nSamples);
    exit(1);
  }
  if (mdata.VecSamOutputs_.length() != nSamples*nOutputs)
  {
    printf("MCMC ERROR: sample outputs has wrong length.\n");
    printf("            %d versus (expected) %d\n",
           mdata.VecSamOutputs_.length(), nOutputs*nSamples);
    exit(1);
  }
  if (mdata.MatExpMeans_.nrows() != 1 ||
      mdata.MatExpStds_.nrows() != 1)
  {
    printf("MCMC ERROR: experimental matrix has more than 1 row.\n");
    exit(1);
  }
  if (mdata.MatExpMeans_.ncols() != nOutputs ||
      mdata.MatExpStds_.ncols() != nOutputs)
  {
    printf("MCMC ERROR: experimental matrix has wrong number of columns\n");
    exit(1);
  }

  //**/ ---------------------------------------------------------------
  //**/ Inference ==> vecLogLikelihoods 
  //**/ ---------------------------------------------------------------
  double *samInps = mdata.VecSamInputs_.getDVector(); 
  double *samOuts = mdata.VecSamOutputs_.getDVector();
  psVector vecLogLikelihoods;
  vecLogLikelihoods.setLength(nSamples);

  nn = 0;
  for (ii = 0; ii < nSamples; ii++)
  {
    vecLogLikelihoods[ii] = 0;
    for (jj = 0; jj < nOutputs; jj++)
    {
      dmean = mdata.MatExpMeans_.getEntry(0,jj); 
      dstdv = mdata.MatExpStds_.getEntry(0,jj); 
      vecLogLikelihoods[ii] += 
           pow((samOuts[ii*nOutputs+nn]-dmean)/dstdv, 2.0); 
    }
  }

  //**/ ---------------------------------------------------------------
  // process vecLogLikelihoods - first normalize
  //**/ ---------------------------------------------------------------
  int nPosteriors = nSamples * 100;
  double Ymin =  PSUADE_UNDEFINED;
  for (ii = 0; ii < nSamples; ii++)
  {
    if (vecLogLikelihoods[ii] < Ymin) 
    {
      Ymin = vecLogLikelihoods[ii];
      nn = ii;
    }
  }
  psVector vecXMax;
  vecXMax.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
    vecXMax[ii] = samInps[nn*nInputs+ii];
  for (ii = 0; ii < nSamples; ii++)
    vecLogLikelihoods[ii] = exp(-0.5*vecLogLikelihoods[ii]);
  ddata = 0;
  for (ii = 0; ii < nSamples; ii++) ddata += vecLogLikelihoods[ii];
  for (ii = 0; ii < nSamples; ii++) vecLogLikelihoods[ii] /= ddata;

  //**/ ---------------------------------------------------------------
  // allocate and find the sample size
  //**/ ---------------------------------------------------------------
  psIVector vecICnts;
  vecICnts.setLength(nSamples);
  nPosteriors = 0;
  for (ii = 0; ii < nSamples; ii++)
  {
    vecICnts[ii] = (int) (vecLogLikelihoods[ii] * nSamples * 100 + 0.5);
    nPosteriors += vecICnts[ii];
  }
  mdata.MatPostSample_.setDim(nPosteriors,nInputs);
  int count = 0;
  for (ii = 0; ii < nSamples; ii++)
  {
    for (jj = 0; jj < vecICnts[ii]; jj++)
    {
      for (kk = 0; kk < nInputs; kk++)
        mdata.MatPostSample_.setEntry(count,kk,samInps[ii*nInputs+kk]);
      count++;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ create matlab
  //**/ ---------------------------------------------------------------
  //**/ set up vecRange, vecXMax
  psVector vecRange;
  vecRange.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
    vecRange[ii] = mdata.VecUpperB_[ii] - mdata.VecLowerB_[ii];
  mm = nn = 0;
  for (ii = 0; ii < nSamples; ii++)
  {
    if (vecICnts[ii] > mm) 
    {
      mm = vecICnts[ii];
      nn = ii;
    }
  }

  //**/ binning
  int nbins = 20, **bins, ****bins2;
  bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
    bins[ii] = new int[nInputs];
    for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (kk = 0; kk < nbins; kk++)
    {
      bins2[jj][kk] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][kk][ii] = new int[nInputs];
        for (mm = 0; mm < nInputs; mm++)
          bins2[jj][kk][ii][mm] = 0;
      }
    }
  }
  for (ii = 0; ii < nPosteriors; ii++)
  {
    //**/ update the 1D histogram
    for (jj = 0; jj < nInputs; jj++)
    {
      ddata = (mdata.MatPostSample_.getEntry(ii,jj) - 
               mdata.VecLowerB_[jj]) / vecRange[jj];
      mm = (int) (ddata * nbins);
      if (mm >= nbins) mm = nbins - 1;
      bins[mm][jj]++;
    }
    for (jj = 0; jj < nInputs; jj++)
    {
      ddata = (mdata.MatPostSample_.getEntry(ii,jj) - 
               mdata.VecLowerB_[jj]) / vecRange[jj];
      mm = (int) (ddata * nbins);
      if (mm >= nbins) mm = nbins - 1;
      for (kk = 0; kk < nInputs; kk++)
      {
        ddata2 = (mdata.MatPostSample_.getEntry(ii,kk) - 
               mdata.VecLowerB_[kk]) / vecRange[kk];
        nn = (int) (ddata2 * nbins);
        if (nn >= nbins) nn = nbins - 1;
        bins2[mm][nn][jj][kk]++;
      }
    }
  }

  //**/ vecPlotInds, qdata
  psIVector vecPlotInds;
  vecPlotInds.setLength(nInputs);
  pData qdata;
  if (mdata.StrInpNames_.numStrings() > 0)
  {
    qdata.strArray_ = mdata.StrInpNames_.getStrings();
    for (ii = 0; ii < nInputs; ii++) vecPlotInds[ii] = ii;
  }
  else
  {
    qdata.strArray_ = new char*[nInputs];
    for (ii = 0; ii < nInputs; ii++)
    {
      vecPlotInds[ii] = ii;
      qdata.strArray_[ii] = new char[100];
      snprintf(qdata.strArray_[ii],100,"X%d", ii+1);
    }
  }
  qdata.nStrings_ = nInputs;
    
  //**/ call genMatlabFile
  genMatlabFile(nInputs,mdata.VecLowerB_.getDVector(),
        mdata.VecUpperB_.getDVector(),vecRange.getDVector(),
        nInputs,vecPlotInds.getIVector(),nbins,NULL,NULL,bins,
        bins2,qdata,0,0,NULL,NULL,vecXMax.getDVector(),Ymin);
  qdata.strArray_ = NULL;

  //**/ clean up
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (mm = 0; mm < nbins; mm++)
    {
      for (ii = 0; ii < nInputs; ii++) 
        delete [] bins2[jj][mm][ii];
      delete [] bins2[jj][mm];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;
  return 0.0;
}

// ************************************************************************
// given a sample and the corresponding likelihoods, create a sample based
// on likelihoods
// ------------------------------------------------------------------------
double MCMCAnalyzer::createPosteriorFromLikelihoods(psVector vecSamIns,
                        psVector vecNegLogLikelihoods, int printLevel)
{
  int nSamples = vecNegLogLikelihoods.length();
  //**/ error checking
  if (nSamples == 0)
  {
    printf("MCMC ERROR: In createPosteriorFromLikelihoods: \n");
    printf("            nSamples = 0.\n");
    printf("     ERROR in file %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }
  int nInps = vecSamIns.length() / nSamples;
  if (nInps == 0 || (nInps*nSamples != vecSamIns.length()))
  {
    printf("MCMC ERROR: In createPosteriorFromLikelihoods: \n");
    printf("            Sample and likelihood lengths do not match.\n");
    printf("     ERROR in file %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }

  //**/ get sample size from user
  char pString[1000];
  snprintf(pString,100, 
     "Size of the sample to be created (>=10k, default=50k) ? ");
  int maxPostSam = getInt(10000, 200000, pString);

  //**/ compute and normalize likelihood values
  int ss, ii;
  psVector vecLikelihoods, vecXmax;
  vecLikelihoods = vecNegLogLikelihoods;
  for (ss = 0; ss < vecNegLogLikelihoods.length(); ss++)
    vecLikelihoods[ss] = exp(-0.5*vecNegLogLikelihoods[ss]); 
  double dmax=-PSUADE_UNDEFINED;
  vecXmax.setLength(nInps);
  for (ss = 0; ss < nSamples; ss++)
  {
    if (vecLikelihoods[ss] > dmax) 
    {
      dmax = vecLikelihoods[ss];
      for (ii = 0; ii < nInps; ii++)
        vecXmax[ii] = vecSamIns[ss*nInps+ii];
    }
  }
  for (ss = 0; ss < nSamples; ss++) vecLikelihoods[ss] /= dmax;

  //**/ create quantile partitions (into K partitions)
  int nQuantiles=maxPostSam/200;
  if (nQuantiles > 100) nQuantiles = 100;
  psVector vecQuanPart;
  vecQuanPart.setLength(nQuantiles+1);
  double factor=0.9;
  vecQuanPart[nQuantiles] = 1.0;
  for (ii = nQuantiles-1; ii >= 0; ii--)
    vecQuanPart[ii] = vecQuanPart[ii+1] * factor;

  //**/ find out how many samples (M_i) are in each quantile
  //**/ in quantile ii if [vecQuanPart[ii], vecQuanPart[ii+1]]
  //**/ vecQuanSizes - sample sizes for each quantile
  //**/ vecQuanWts - total likelihood for samples in each quantile
  psVector  vecQuanWts;
  psIVector vecQuanSizes;
  vecQuanSizes.setLength(nQuantiles);
  vecQuanWts.setLength(nQuantiles);
  double ddata;
  for (ss = 0; ss < nSamples; ss++)
  {
    ddata = vecLikelihoods[ss];
    for (ii = 0; ii <= nQuantiles; ii++)
      if (ddata <= vecQuanPart[ii]) break;
    if (ii > 0) 
    {
      vecQuanSizes[ii-1]++;
      vecQuanWts[ii-1] += ddata;
    }
  }
  printAsterisks(PL_INFO, 0);
  printf("MCMC-like inference: posterior sample generation: \n");
  printf("  * Total no. of sample points   = %d\n",nSamples);
  printf("  * No. of posterior candidates  = %d (those with Prob>0).\n",
         vecQuanSizes.sum());
  printf("  * No. of points to be selected ~ %d\n", maxPostSam);

  //**/ create an index matrix for all quantiles
  //**/ quanIndices[ii][jj] - sample indices jj for quantile ii
  //**/ This is done because later on random sub-samples are to
  //**/ be drawn from each quantile, and this index matrix will
  //**/ make this process more efficient
  int nMax = vecQuanSizes.max();
  psIMatrix matQuanInds; 
  matQuanInds.setFormat(PS_MAT2D);
  matQuanInds.setDim(nQuantiles, nMax);
  int **quanIndices = matQuanInds.getIMatrix2D(); 
  vecQuanSizes.setLength(nQuantiles);
  for (ss = 0; ss < nSamples; ss++)
  {
    ddata = vecLikelihoods[ss];
    for (ii = 0; ii <= nQuantiles; ii++)
      if (ddata <= vecQuanPart[ii]) break;
    if (ii > 0) 
    {
      quanIndices[ii-1][vecQuanSizes[ii-1]] = ss;
      vecQuanSizes[ii-1]++;
    }
  }
     
  //**/ figure out how many samples to select from each 
  //**/ quantile (from total likelihood for each quantile)
  psIVector vecQuanMax;
  vecQuanMax.setLength(nQuantiles);
  for (ii = 0; ii < nQuantiles; ii++)
  {
    //**/ sum_pi multiplied by any large integer 
    ddata = vecQuanWts[ii] * 1000; 
    vecQuanMax[ii] = (int) ddata;
  }

  //**/ vecQuanMax.sum may be large, need to rescale the sum
  //**/ to be <= maxPostSam
  ddata = 1.0 * maxPostSam / vecQuanMax.sum();
  for (ii = 0; ii < nQuantiles; ii++)
    vecQuanMax[ii] = (int) (ddata * vecQuanMax[ii]);
  if (vecQuanMax.sum() > maxPostSam)
  {
    printf("MCMC ERROR: Something is wrong. Consult the developers.\n");
    printf("     ERROR in file %s, line %d.\n", __FILE__, __LINE__);
    printf("     vecQuanMax.sum, maxPostSam = %d %d\n",vecQuanMax.sum(),
           maxPostSam);
    exit(1);
  } 

  //**/ now select a posterior sample and put it in vecPostInps
  int nPostSam=0,count=0,samInd,jj,kk;
  psVector vecPostInps;
  vecPostInps.setLength(maxPostSam*nInps);
  psIVector vecIRan;
  vecIRan.setLength(nSamples);
  count = 0;
  if (printLevel > 1)
  {
    printf("MCMC INFO: Samples are assigned to %d bins\n",nQuantiles);
    printf("           Sub-samples are selected from each bin : \n");
  }
  for (ss = 0; ss < nQuantiles; ss++)
  {
    //**/ randomize the sample indices for the quantile ss
    if (vecQuanSizes[ss] > 0)
    {
      vecIRan.setLength(vecQuanSizes[ss]);
      generateRandomIvector(vecQuanSizes[ss],vecIRan.getIVector());
    }
    //**/ now fill in the posterior sample array vecPostInps
    int iSave = vecQuanMax[ss]; 
    samInd = 0;
    if (printLevel > 1)
      printf("      Bin %d has %d points (selected from %d)\n",
             ss+1, vecQuanMax[ss],vecQuanSizes[ss]);
    while (vecQuanMax[ss] > 0 && vecQuanSizes[ss] > 0)
    {
      //**/ randomly pick an element and fetch index kk
      jj = vecIRan[samInd];
      kk = quanIndices[ss][jj];
      //**/ if probability of sample kk is > 0, add to post sample
      ddata = vecLikelihoods[kk];
      if (ddata > 0)
      {
        for (ii = 0; ii < nInps; ii++)
          vecPostInps[count++] = vecSamIns[kk*nInps+ii];
      }
      vecQuanMax[ss] = vecQuanMax[ss] - 1; 
      nPostSam++;
      samInd++;
      //**/ if all exhausted, start all over from beginning
      if (samInd >= vecQuanSizes[ss]) samInd = 0;
    }
  }

  //**/ checking the selected posterior sample for mean and std dev
  //**/ first count number of uncertain parameters
  kk = 0;
  double ddmean = 0;
  double ddstds = 0;
  printf("  * Posterior sample statistics : \n");
  for (ii = 0; ii < nInps; ii++)
  {
    ddmean = 0;
    for (ss = 0; ss < nPostSam; ss++)
      ddmean += vecPostInps[ss*nInps+kk];
    ddmean /= (double) nPostSam;
    ddstds = 0;
    for (ss = 0; ss < nPostSam; ss++)
      ddstds += pow(vecPostInps[ss*nInps+kk]-ddmean,2.0);
    ddstds /= (double) nPostSam;
    ddstds = sqrt(ddstds); 
    printf("      Input %d mean, stds = %12.5e %12.5e\n",  
           ii+1, ddmean, ddstds);
    kk++;
  }
  printDashes(PL_INFO, 0);

  FILE *fp = fopen("LikelihoodSample", "w");
  fprintf(fp, "PSUADE_BEGIN\n");
  //**/ fetch parameter names
  fprintf(fp, "%d %d\n", nPostSam, nInps);
  fprintf(fp, "# ");
  for (jj = 0; jj < nInps; jj++) fprintf(fp,"X%d ", jj+1);
  fprintf(fp, "\n");

  //**/ store the subsample
  for (ss = 0; ss < nPostSam; ss++)
  {
    fprintf(fp, "%d ", ss+1);
    for (jj = 0; jj < nInps; jj++)
      fprintf(fp, "%e ", vecPostInps[ss*nInps+jj]);
    fprintf(fp, "\n");
  }
  fprintf(fp, "PSUADE_END\n");

  //**/ write MLE solution
  fprintf(fp, "Optimal parameter values: \n");
  for (ii = 0; ii < nInps; ii++) 
    fprintf(fp, "%16.8e\n", vecXmax[ii]);
  fclose(fp);
  printf("INFO: The posterior sample is now in LikelihoodSample.\n");
  return 0;
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int MCMCAnalyzer::get_nInputs()
{
  return nInputs_;
}
int MCMCAnalyzer::get_nOutputs()
{
  return nOutputs_;
}
double *MCMCAnalyzer::get_means()
{
  int    ii;
  double *retVal = NULL;
  if (vecMeans_.length() > 0)
  {
    retVal = new double[vecMeans_.length()];
    for (ii = 0; ii < vecMeans_.length(); ii++) retVal[ii] = vecMeans_[ii];
  }
  return retVal;
}
double *MCMCAnalyzer::get_mostLikelyInput()
{
  int    ii;
  double *retVal = NULL;
  if (vecMostLikelyInputs_.length() > 0)
  {
    retVal = new double[vecMostLikelyInputs_.length()];
    checkAllocate(retVal, "retVal in MCMC::get_mostLikelyInput");
    for (ii = 0; ii < vecMostLikelyInputs_.length(); ii++) 
      retVal[ii] = vecMostLikelyInputs_[ii];
  }
  return retVal;
}
double *MCMCAnalyzer::get_mostLikelyOutput()
{
  int    ii;
  double *retVal = NULL;
  if (vecMostLikelyOutputs_.length() > 0)
  {
    retVal = new double[vecMostLikelyOutputs_.length()];
    checkAllocate(retVal, "retVal in MCMC::get_mostLikelyOutput");
    for (ii = 0; ii < vecMostLikelyOutputs_.length(); ii++) 
      retVal[ii] = vecMostLikelyOutputs_[ii];
  }
  return retVal;
}

