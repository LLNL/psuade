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
// Utility functions
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#ifndef __PSUADEUTILH__
#define __PSUADEUTILH__

#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <PsuadeConfig.h>
#include "psMatrix.h"

extern "C" 
{
#include "isaac_standard.h"
#include "isaac_rand.h"
}

extern "C" 
{
   //**/ General psuade functions
   void   PSUADE_randInit(long);
   int    PSUADE_rand();
   double PSUADE_drand();
   void   PSUADE_drandn(int, double *);
   char*  PSUADE_strdup(const char *s);
   char*  PSUADE_strndup(const char *s, size_t n);

   //**/ get time functions
#ifdef CYGWIN
int     cygwin_System(char *s);
#define system cygwin_System
int     gettimeofday(struct timeval *, void *);
#define Gettimeofday gettimeofday
#endif

#if defined(IRIX) || defined(IRIX64)
int     BSDgettimeofday(struct timeval *,struct timezone *);
#define Gettimeofday BSDgettimeofday
#endif

#ifdef LINUX
/* int     gettimeofday(struct timeval *,struct timezone *); */
#define Gettimeofday gettimeofday
#endif

#ifdef WINDOWS
#define Gettimeofday gettimeofday
#endif

#ifdef MACOS
#define Gettimeofday gettimeofday
#endif

#ifdef USER
#define Gettimeofday gettimeofday
#endif

#ifdef HPUX
int     gettimeofday(struct timeval *,struct timezone *);
#define Gettimeofday gettimeofday
#endif

#ifdef AIX
/* int     gettimeofday(struct timeval *,struct timezone *); */
#define Gettimeofday gettimeofday
#endif

#ifdef ALPHA
int     gettimeofday(struct timeval *,struct timezone *);
#define Gettimeofday gettimeofday
#endif

#ifdef SUNOS
int     gettimeofday(struct timeval *,struct timezone *);
#define Gettimeofday gettimeofday
#endif

   double getClock();
   void   startTimer();
   void   stopTimer();
   double getElapsedTime();

   //**/ functions for general probability calculations
   double Gamma_Function(double);
   double Gamma_Function_Max_Arg();
   double Incomplete_Gamma_Function(double,double);
   double Beta_Function(double,double);
   double Incomplete_Beta_Function(double,double,double);

   //**/ sort and search functions
   void   sortIntList(int,int*);
   void   sortIntList2a(int,int*,double*);
   void   sortDbleList(int,double*);
   void   sortDbleListBubble(int,double*);
   void   sortDbleListMerge(int,double*);

   void   sortDbleList2(int,double*,double*);
   void   sortDbleList2a(int,double*,int*);
   void   sortDbleList3(int,double*,double*,double*);
   void   sortDbleList4(int,double*,double*,double*,double*);
   int    sortAndDelete(int, int, double **);
   int    sortnDbleList(int,int,double**, int *);

   int    binarySearchInt(int, int *, int);
   int    binarySearchDble(double, double *, int);

   //**/ functions for screen display
   void   printAsterisks(int, int);
   void   printDashes(int, int);
   void   printEquals(int, int);
   void   checkAllocate(void *, const char *);
   void   checkDbleArray(char *, int, double *);

   //**/ miscellaneous functions
   int    computeBasicStatistics(psVector, double &, double &);
   void   generateRandomIvector(int, int *);
   int    checkPrime( int );
   int    *factorize( int );
   int    compareSamples(int, int, int, double *, int *);
   int    checkRepeatedSamplePts(int, int, double *);
   int    computeNumPCEPermutations(int, int);
   int    checkOUUFileFormat(char *, int, int, int);
   int    checkMCMCFileFormat(char *, int, int);
   int    checkSPDFFileFormat(char *, int);
   int    readIReadDataFile(char *, psMatrix &);
   int    readSampleInputFile(const char *, psIVector &, psMatrix &);
   int    readGrpInfoFile(char *fname, int nInputs, psIMatrix &Amat);

   int    read_csv(char *, int *, int *, double **, int *, 
                   char ***, char ***);
   int    write_csv(char *, int, int, double *, int, double *, 
                    char **, char **);
   int    odoeUsingMMDSlow(psIVector, int, int, int, psMatrix &, 
              psMatrix &,psIVector &,psIVector &,double &,long &,int);
   int    odoeUsingMMD(int, psMatrix &, psMatrix &, psIVector &, 
              psIVector &, double &, long &, int);
   int    optimizeSampleUsingCE(int, int, psVector&);

   int    computeFromSampleCovMatEigen2(psVector, int, int, psIVector, 
                                       psVector &, psMatrix &);
   int    computeFromSampleCovMatEigen(psMatrix, psVector &, psMatrix &);

   bool   checkFileExists(char *);
   int    checkConvHull2D(double,double,psVector,psVector);

   //**/ functions for getting input interactively
   int    getTokens(char *, int *, char ***, int);
   int    getIntFromToken(char *, int *);
   int    getDbleFromToken(char *, double *);
   int    getInt(int, int, char *);
   double getDouble(char *);
   int    getString(char *, char *);

   //**/ functions for generating matlab/scilab files
   int    genMatlabPlotFile(int,double *,double *,int, double *, 
                            char **,char *,int);
   bool   plotMatlab();
   bool   plotScilab();
   void   fwriteRSPythonHeader(FILE *);
   void   fwriteRSPythonCommon(FILE *);
   int    fwritePlotScales2D(FILE *);
   int    fwritePlotAxes(FILE *);
   int    fwritePlotAxesNoGrid(FILE *);
   int    fwritePlotCLF(FILE *);
   int    fwritePlotFigure(FILE *, int);
   int    fwritePlotXLabel(FILE *, const char *);
   int    fwritePlotYLabel(FILE *, const char *);
   int    fwritePlotZLabel(FILE *, const char *);
   int    fwritePlotTitle(FILE *, const char *);
   int    fwriteHold(FILE *, int);
   int    fwriteComment(FILE *, char *);
}

#define Macheps     3.40282347e-38
#endif /* __PSUADEUTILH__ */

