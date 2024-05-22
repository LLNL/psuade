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
// Definition for the class TwoParamAnalyzer (interaction analysis)
// AUTHOR : CHARLES TONG
// DATE   : updated in 2006
// ************************************************************************

#ifndef __TWOPARAMANALYZERH__
#define __TWOPARAMANALYZERH__

#include "Analyzer.h"
#include "psVector.h"
#undef max
#include <vector>

// ************************************************************************
// class definition
// ************************************************************************
class TwoParamAnalyzer : public Analyzer
{
  
private:

   int    nInputs_;
   double outputMean_;
   double outputStd_;
   psVector VecMainEffects_;
   psMatrix Mat2ParamEffects_;
   int printLevel_;

   int computeMeanVariance(int,int,int,double*,double *,double *,int);
   int computeVCE2(int,int,int,double*,double*,FILE *,double,
                   double *, double *, double *, double *);
   int computeVCECrude(int, int, double *, double *, double, double *,
                       double *, double *);
public:

   //**/ Constructor
   TwoParamAnalyzer();

   //**/ Destructor
   ~TwoParamAnalyzer();

   //**/ Perform analysis
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ assign operator
   //**/ @param analyzer
   TwoParamAnalyzer& operator=(const TwoParamAnalyzer &analyzer);

   //**/ generate plots 
   int genPlots(aData &adata);

   //**/ Perform 1D analysis
   //**/ @param nInputs - number of input variables
   //**/ @param nOutputs - number of output variables
   //**/ @param nSamples - number of sample points
   //**/ @param sampleInputs - sample points 
   //**/ @param sampleOutputs - sample outputs 
   //**/ @param inputID - which input to examine
   //**/ @param outputID - which output to examine
   //**/ @param nSubSamples - number of samples in each group
   //**/ @param mean - sample mean
   //**/ @param var  - sample variance
   //**/ @param retdata - returned total sensitivity index
   double analyze1D(int nInputs, int nOutputs, int nSamples,
                    double *sampleInputs, double *sampleOutputs,
                    int inputID, int outputID, int nSubSamples, 
                    double mean, double var, double *retdata);
   
   /** Getters for analysis results */
   int    getNumInputs();
   double getOutputMean();
   double getOutputStd();
   double *getMainEffects();
   double **get2ParamEffects();
};

#endif // __TWOPARAMANALYZERH__

