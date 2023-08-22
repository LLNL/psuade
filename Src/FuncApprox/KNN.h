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
// Definition for the class KNN
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#ifndef __KNNH__
#define __KNNH__

#include <stdio.h>
#include "FuncApprox.h"
#include "psVector.h"

// ************************************************************************
// class definition
// ************************************************************************
class KNN : public FuncApprox 
{
   int k_;
   int mode_;
   int kmax_;
   int kfold_;
   psVector VecNormalX_;
   psVector VecNormalY_;
   psVector VecDistances_;
   psVector VecYStored_;
   psVector VecRanges_;
   psVector VecSigmas_;
 
public:

   KNN(int, int);
   ~KNN();

   int    initialize(double*,double*);
   int    genNDGridData(double*,double*,int*,double**,double**);
   int    gen1DGridData(double*,double *,int,double*, 
                        int *, double **, double **);
   int    gen2DGridData(double*,double *,int,int,double*, 
                        int *, double **, double **);
   int    gen3DGridData(double*,double *,int,int,int,double*, 
                        int *, double **, double **);
   int    gen4DGridData(double*,double *,int,int,int,int,double*, 
                        int *, double **, double **);
   double evaluatePoint(int, double *, double *, double *, int, int);
   double evaluatePoint(double *);
   double evaluatePoint(int, double *, double *);
   double evaluatePointFuzzy(double *, double &);
   double evaluatePointFuzzy(int, double *, double *, double *);

private:

   double computeTestError(int,double *,double *,int,int,double *,double *);
   double trainGaussian(psVector&, psVector&);
   void   genRSCode();
};

#endif // __KNNH__

