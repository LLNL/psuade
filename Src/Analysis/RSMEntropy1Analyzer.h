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
// Definition for the class RSMEntropy1Analyzer
// (Entropy-based sensitivity analysis - with response surface)
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#ifndef __RSMENTROPY1ANALYZERH__
#define __RSMENTROPY1ANALYZERH__

#include "Analyzer.h"
#include "psVector.h"
#include "ProbMatrix.h"

// ************************************************************************
// class definition
// ************************************************************************
class RSMEntropy1Analyzer : public Analyzer
{
private:

   int    nInputs_;
   double outputMean_;
   double outputStd_;
   double outputEntropy_;
   double outputDelta_;
   int    adaptive_;
   int    nLevels_;
   int    M1_;
   int    M2_;
   int    entropyDelta_;
   int    useSimulator_;
   int    useRS_;
   psVector VecEnt1_;   // length is nInputs_
   psVector VecDelta1_; // length is nInputs_

public:

   //**/ Constructor 
   RSMEntropy1Analyzer();

   //**/ Destructor 
   ~RSMEntropy1Analyzer();

   //**/ Perform analysis
   //**/ @param nInps - number of inputs
   //**/ @param nSamp - number of samples
   //**/ @param ilbs  - input lower bounds
   //**/ @param iubs  - input upper bounds
   //**/ @param sInps - sample inputs
   //**/ @param sOuts - sample outputs
   void analyze(int nInps, int nSamp, double *ilbs,
                double *iubs, double *sInps, double *sOuts);

   //**/ Perform analysis
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ Perform entropy or delta analysis
   //**/ @param adata - all data needed for analysis
   double analyzeEntDelta(aData &adata);

   //**/ Compute mean and standard deviation
   //**/ @param vecY - output data
   double computeBasicStat(psVector vecY);

   //**/ Compute entropy
   //**/ @param matProb - probability matrix 
   //**/ @param vecL - lower bound for binning 
   //**/ @param vecU - upper bound for binning
   //**/ @param entropy 
   //**/ @param adaptOrNot - flag 
   int computeEntropy(ProbMatrix &matProb, psVector vecL, 
              psVector vecU, double &entropy, int adaptOrNot);

   //**/ Compute total entropy and delta of Y
   //**/ @param vecY - output array
   int computeTotalEntDelta(psVector vecY);

   //**/ assign operator
   //**/ @param analyzer
   RSMEntropy1Analyzer& operator=(const RSMEntropy1Analyzer &analyzer);

   //**/ set internal parameter 
   //**/ @param argc - argument count
   //**/ @param argv - arguments
   int setParam(int argc, char **argv);

   /** Getters for analysis results */
   int    get_nInputs();
   double get_outputMean();
   double get_outputStd();
   double get_outputEntropy();
   double get_outputDelta();
   double get_entropy1(int);
   double get_delta1(int);
};

#endif // __RSMENTROPY1ANALYZERH__

