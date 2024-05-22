// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Definition for the class oData
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#ifndef __ODATAH__
#define __ODATAH__

#include "FunctionInterface.h"
#include "PsuadeData.h"

// ************************************************************************
// class definition
// ************************************************************************
class oData 
{
public:
   int    outputLevel_;
   int    nSamples_;
   int    nInputs_;
   int    nOutputs_;
   int    outputID_;
   int    numFuncEvals_;
   int    intData_;
   int    maxParallelJobs_;
   int    setOptDriver_; /* 0 - do nothing, 1 - set, 2 - reset, 3 - both */
   double *initialX_;
   double *lowerBounds_;
   double *upperBounds_;
   double *optimalX_;
   double optimalY_;
   double tolerance_;
   double deltaX_;
   double targetY_; /* expected fmin */
   int    maxFEval_;
   FunctionInterface *funcIO_;
   char targetFile_[200];
   PsuadeData *psIO_;
   void   (*optFunction_)(int, double *, int, double *);

   /** constructor */
   oData();

   /** destructor */
   ~oData();

   /** assign operator */
   oData& operator=(const oData&);
};

#endif // __ODATAH__

