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
// Definition for the class MCMCAnalyzer 
// AUTHOR : CHARLES TONG
// DATE   : 2007
// ************************************************************************
#ifndef __MCMCANALYZERH__
#define __MCMCANALYZERH__

#include "Analyzer.h"
#include "pData.h"
#include "mData.h"
#include "FuncApprox.h"
#include "psMatrix.h"
#include "psVector.h"

// ************************************************************************
// class definition
// ************************************************************************
class MCMCAnalyzer : public Analyzer
{
private:
   int    mode_;
   int    bfmode_;
   int    scheme_;
   int    nInputs_;
   int    nOutputs_;
   psVector vecMeans_;
   psVector vecSigmas_;
   psVector vecMostLikelyInputs_;
   psVector vecMostLikelyOutputs_;

public:

   //**/ Constructor
   MCMCAnalyzer();

   //**/ Destructor
   ~MCMCAnalyzer();

   //**/ Perform MCMC analysis using Gibbs
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ Perform Gibbs MCMC analysis
   //**/ @param adata - all data needed for analysis
   double analyze_gibbs(aData &adata);

   //**/ Perform brute force MCMC analysis
   //**/ @param adata - all data needed for analysis
   double analyze_bf(aData &adata);

   //**/ Perform brute force MCMC analysis - with uncertain paramters
   //**/ @param adata - all data needed for analysis
   double analyze_bf2(aData &adata);

   //**/ Perform MCMC analysis using Metropolis Hasting with no RS
   //**/ @param adata - all data needed for analysis
   double analyze_sim(aData &adata);

   //**/ Perform MCMC analysis using already-evaluated sample
   //**/ @param adata - all data needed for analysis
   double analyzeDirect_nors(McmcData &mdata);

   //**/ Perform MCMC analysis using Metropolis Hasting
   //**/ @param adata - all data needed for analysis
   double analyze_mh(aData &adata);

   //**/ Perform MCMC analysis in shared memory mode
   //**/ @param adata - all data needed for analysis
   double analyze_omp(aData &adata);

   //**/ Perform MCMC analysis in direct mode
   //**/ @param mdata - all data needed for analysis
   double analyzeDirect(McmcData &mdata);

   //**/ Compute negative log likelihood from a given calibration
   //**/ setting
   //**/ @param adata - all data needed for analysis
   double computeLikelihood(aData &adata);

   //**/ assign operator
   //**/ @param analyzer
   MCMCAnalyzer& operator=(const MCMCAnalyzer &analyzer);

   //**/ read index file
   //**/ @param rsIndices - fixed input indices
   //**/ @param rsValues  - fixed input values
   //**/ @param matrix    - read sample 
   int readIndexFile(PsuadeData *dataPtr, psIVector &rsIndices, 
                      psVector &rsValues, psMatrix &);

   //**/ clean up memory allocation
   void cleanUp();

   //**/ Generate Matlab plot file 
   //**/ @param nInputs - number of inputs
   //**/ @param lower  - input lower bounds
   //**/ @param upper  - input upper bounds
   //**/ @param ranges - input ranges
   //**/ @param nPlots - number of inputs to be plotted
   //**/ @param plotIndices - plot input indices
   //**/ @param nbins  - number of bins
   //**/ @param bins   - bins for prior histogram
   //**/ @param bins2  - bins for prior 2D histogram
   //**/ @param bins   - bins for posterior histogram
   //**/ @param bins2  - bins for posterior 2D histogram
   //**/ @param nChains- number of Markov chains
   //**/ @param chainCnt - length of each chain
   //**/ @param XChains  - chain data
   //**/ @param chainStatus  - chain status
   //**/ @param Xmax  - optimal values
   //**/ @param Ymin  - minimum log likelihood 
   double genMatlabFile(int nInputs, double *lower, double *upper,
                        double *ranges, int nPlots, int *plotIndices,
                        int nbins, int **pbins, int ****pbins2, 
                        int **bins, int ****bins2, 
                        pData &pData, int nChains, int chainCnt,
                        double ***XChains, int *chainStatus, 
                        double *Xmax, double Ymin);

   //**/ Generate Matlab plot file for negative log likelihood
   int    genPostLikelihood(int nInputs, double *lower,
                 double *upper, double *XRange, int numChains,
                 int chainCnt, double ***XChains, int *chainStatus,
                 int chainLimit, int *rsIndices, double *rsValues,
                 psIVector vecDesignP, int dnInputs, int dnSample, 
                 psMatrix matInps, FuncApprox **faPtrs, 
                 int nOutputs, double *discOutputs,
                 double *discFuncConstantMeans, psMatrix matMeans,
                 psMatrix matStdvs, int, int, psIVector&);

   double createPosteriorFromLikelihoods(psVector, psVector, int);

   //**/ Read the experimental data file
   //**/ @param nInputs  - number of inputs
   //**/ @param nOutputs - number of outputs
   //**/ @param dParams  - design parameter list
   //**/ @param dSamIns  - design parameter values
   //**/ @param dMeans   - data means
   //**/ @param dStds    - data standard deviations
   //**/ @param combFlag - control likelihood 
   //**/ @param printLevel - diagnostics level
   double readSpecFile(int nInputs, int nOutputs,
                       psIVector &dParams, psMatrix &dSamInputs,
                       psMatrix &dSamMeans, psMatrix &dSamStds,
                       int &combineFlag, int printLevel);

   //**/ set internal paramter
   //**/ @param nParams - number of parameters
   //**/ @param params - parameters
   int setParams(int nParams, char **params);

   //**/ check convergence
   //**/ @param num    - number of elements
   //**/ @param means  - mean vector
   //**/ @param stds   - sd vector
   //**/ @param leng   - length of chain
   //**/ @param thresh - threshold 
   double checkConvergence(int num, double *means, double *stds, 
                           int leng, double thresh);

   //**/ display banner information
   void displayBanner(int printLevel);
   void displayBanner_bf2(int printLevel);

   //**/ create discrepancy functions
   double createDiscrepancyFunctions(int nInputs, int nOutputs,
                double *lower, double *upper, psIVector &vecRSIndices,
                psVector &vecRSValues, psIVector &vecDesignP,
                int dinInputs, int dnSamples, psMatrix &matExpInps,
                psMatrix &matExpMeans, PsuadeData *dataPtr,
                psVector &vecDiscConstMeans, psVector &vecDiscConstStds,
                psVector &vecExpSamOuts, FuncApprox **faPtr, 
                int printLevel, int);

  //**/ write optimization results to file
  int writeMLEInfo(FILE *fp, int nInputs, int nOutputs, FuncApprox **, 
                int *designPs, int *rsIndices, double *rsValues, 
                double *vecXmax, int dnSamples, int dnInputs, 
                double *dExpInps, double *dExpMeans, double *dExpStdvs, 
                int, int, double *, double *, psIVector &);

   /** Getters for analysis results */
   int    get_nInputs();
   int    get_nOutputs();
   double *get_means();
   double *get_sigmas();
   double *get_mostLikelyInput();
   double *get_mostLikelyOutput();
};

#endif // __MCMCANALYZERH__

