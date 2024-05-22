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
// Definition for the class RSMSobolTSIAnalyzer
// (compute total sensitivity index - response surface-based)
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#ifndef __RSMSOBOLTSIANALYZERH__
#define __RSMSOBOLTSIANALYZERH__

#include "PsuadeData.h"
#include "Analyzer.h"

// ************************************************************************
// class definition
// ************************************************************************
class RSMSobolTSIAnalyzer : public Analyzer
{
private:

  int    nInputs_;
  double outputMean_;
  double outputStd_;
  psVector VecTSIs_;
  int    method_;

public:

  //**/ Constructor 
  RSMSobolTSIAnalyzer();

  //**/ Destructor 
  ~RSMSobolTSIAnalyzer();

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
  double analyze2(aData &adata);

  //**/ set internal parameter 
  //**/ @param argc - argument count
  //**/ @param argv - arguments
  int setParam(int argc, char **argv);

  //**/ assign operator
  //**/ @param analyzer
  RSMSobolTSIAnalyzer& operator=(const RSMSobolTSIAnalyzer &analyzer);

  //**/ print results to matlab/scilab file
  //**/ @param nInps - number of inputs
  //**/ @param var   - variance
  //**/ @param tsi   - total sensitivity indices
  //**/ @param io    - pointer to PsuadeData object
  int printResults(int nInps,double var,double *tsi,PsuadeData *io);

  /** Getters for analysis results */
  int    get_nInputs();
  double get_outputMean();
  double get_outputStd();
  double get_tsi(int ind);
};

#endif // __RSMSOBOLTSIANALYZERH__

