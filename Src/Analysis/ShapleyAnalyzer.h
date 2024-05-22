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
// Definition for the class ShapleyAnalyzer
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#ifndef __SHAPLEYANALYZERH__
#define __SHAPLEYANALYZERH__

#include "Analyzer.h"
#include "FuncApprox.h"
#include "psVector.h"
#include "psMatrix.h"

// ************************************************************************
// class definition
// ************************************************************************
class ShapleyAnalyzer : public Analyzer 
{
  int nInputs_;
  int sampleSize_;
  int costFunction_;
  int MapLength_;  
  int MaxMapLength_;
  psVector VecShapleys_;
  psVector VecShapleyStds_;
  psVector VecShapleyTable_;
  psIMatrix MatShapleyMap_;

public:

  //**/ Constructor 
  ShapleyAnalyzer();

  //**/ Destructor 
  ~ShapleyAnalyzer();

  //**/ Perform analysis
  //**/ @param adata - all data needed for analysis
  double analyze(aData &adata);

  //**/ Perform analysis using VCE-based algorithm
  //**/ @param adata - all data needed for analysis
  double analyzeVCE(aData &adata);

  //**/ Perform analysis using TSI-based algorithm
  //**/ @param adata - all data needed for analysis
  double analyzeTSI(aData &adata);

  //**/ Perform analysis using entropy-based algorithm
  //**/ @param adata - all data needed for analysis
  double analyzeEntropy(aData &adata);

  //**/ Create 2 random matrices 
  //**/ @param adata - all data needed for analysis
  //**/ @param vecXSam1(2) - random matrices stored in vector form
  int create2RandomSamples(aData &adata, psVector &vecXSam1,
                           psVector &vecXSam2);

  //**/ Create a random integer matrix
  //**/ @param nRows, nCols - matrix dimensions
  //**/ @param matIRan - matrix itself
  int createRandomIntMatrix(int nRows, int nCols, psIMatrix &);

  //**/ Look up Shapley value for a random subset
  //**/ @param vecIn - input set 
  //**/ @param ind   - current input index
  double ShapleyEntropyLookup(psIVector vecIn, int ind, int flag);

  //**/ Look up Shapley value for a random subset
  //**/ @param vecIn - input set 
  double ShapleyEntropySave(psIVector vecIn, int, int, double);

  //**/ Create a response surface 
  //**/ @param adata - all data needed for analysis
  FuncApprox *createResponseSurface(aData &adata);

  //**/ Morris one at a time analysis
  //**/ @param nInps - number of inputs
  //**/ @param nSams - sample size 
  //**/ @param samInps - sample inputs 
  //**/ @param samOuts - sample outputs 
  //**/ @param xLower  - input lower bounds 
  //**/ @param xUpper  - input upper bounds 
  //**/ @param moatMeans - MOAT means to be returned
  //**/ @param moatModMeans - MOAT modified means to be returned
  //**/ @param moatStds  - MOAT standard deviations means to be returned
  int MOATAnalyze(int nInps, int nSams, double *samInps,
            double *samOuts, double *xLower, double *xUpper,
            double *moatMeans, double *moatModMeans, 
            double *moatStds);

  //**/ select cost function 
  //**/ @param argc - argument count
  //**/ @param argv - arguments
  int setParam(int argc, char **argv);

  //**/ assign operator
  //**/ @param analyzer
  ShapleyAnalyzer& operator=(const ShapleyAnalyzer &analyzer);

  /** Getters for analysis results */
  int get_nInputs();
  double *get_svalues();
  double *get_sstds();

};

#endif // __SHAPLEYANALYZERH__

