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
// Definition for the class EtaAnalyzer 
// AUTHOR : CHARLES TONG
// DATE   : 2009
// ************************************************************************
#ifndef __ETAANALYZERH__
#define __ETAANALYZERH__

#include "psVector.h"
#include "Analyzer.h"

// ************************************************************************
// class definition
// ************************************************************************

class EtaAnalyzer : public Analyzer
{
private:

   int    mode_;
   int    nInputs_;
   double totalViolation_;
   psVector  VecOrders_;
   psVector  VecRanks_;;
   psIVector VecInpViolations_; 

public:

   //**/ Constructor
   EtaAnalyzer();

   //**/ Destructor
   ~EtaAnalyzer();

   //**/ Perform analysis
   //**/ @param adata - all data needed for analysis
   double analyze(aData &adata);

   //**/ assign operator
   //**/ @param analyzer
   EtaAnalyzer& operator=(const EtaAnalyzer &analyzer);

   /** Getters for analysis results */
   int    get_nInputs();
   double *get_dOrder();
   double *get_dRanks();
   int    *get_inputViolations();
   double get_totalViolation();

};

#endif // __ETAANALYZERH__

