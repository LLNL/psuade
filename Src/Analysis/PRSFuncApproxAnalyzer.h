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
// Definition for the class PRSFuncApproxAnalyzer
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#ifndef __PRSFUNCAPPROXANALYZERH__
#define __PRSFUNCAPPROXANALYZERH__

#include "CommManager.h"
#include "aData.h"

// ************************************************************************
// class definition
// ************************************************************************
class PRSFuncApproxAnalyzer 
{
   int rsType_;
   int nprocs_;
   int mypid_;
   CommManager *commMgr_;

public:

   //**/ Constructor
   PRSFuncApproxAnalyzer(CommManager *);

   //**/ Destructor
   ~PRSFuncApproxAnalyzer();

   //**/ Perform analysis
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ assign operator
   //**/ @param analyzer
   PRSFuncApproxAnalyzer& operator=(const PRSFuncApproxAnalyzer &analyzer);

   //**/ set internal paramter
   //**/ @param nParams - number of parameters
   //**/ @param params - parameters
   int setParams(int nParams, char **params);
};

#endif // __PRSFUNCAPPROXANALYZERH__

