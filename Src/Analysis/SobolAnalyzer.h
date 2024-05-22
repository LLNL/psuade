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
// Definition for the class SobolAnalyzer
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#ifndef __SOBOLANALYZERH__
#define __SOBOLANALYZERH__

#include "Analyzer.h"

// ************************************************************************
// class definition
// ************************************************************************
class SobolAnalyzer : public Analyzer 
{
public:

   //**/ Constructor 
   SobolAnalyzer();

   //**/ Destructor 
   ~SobolAnalyzer();

   //**/ Perform analysis
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ Perform second-order analysis
   //**/ @param adata - all data needed for analysis
   double analyze2(aData &adata);

   //**/ Perform group-order analysis
   //**/ @param adata - all data needed for analysis
   double analyze3(aData &adata);

   //**/ Perform MOAT analysis
   int MOATAnalyze(int, int, double *, double *, double *, 
                   double *, double *, double *, double*);

   //**/ Set first or second order analysis
   int setOrder(int);

   //**/ assign operator
   //**/ @param analyzer
   SobolAnalyzer& operator=(const SobolAnalyzer &analyzer);

   /** Getters for analysis results */
   int get_nInputs();
   int get_nGroups();
   double get_variance();
   double get_modifiedMeans(int ii);
   double get_stds(int ii);
   double get_S(int ii);
   double get_S2(int ii);
   double get_SG(int ii);
   double get_ST(int ii);
   double get_PE(int ii);

private:
   int nInputs_;
   int nGroups_;
   int order_;
   int printLevel_;
   double   Variance_;
   psVector VecModMeans_;
   psVector VecStds_;
   psVector VecS_;
   psVector VecS2_;
   psVector VecSG_;
   psVector VecST_;
   psVector VecPE_;
};

#endif // __SOBOLANALYZERH__

