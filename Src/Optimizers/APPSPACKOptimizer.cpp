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
// Functions for the class APPSPACKOptimizer
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#ifdef HAVE_APPSPACK

#include <stdio.h>
#include "PsuadeUtil.h"

#include <vector>
//#include "APPSPACK_Common.H"
//#include "APPSPACK_BasePoint.H"
//#include "APPSPACK_APPS.H"
#include "APPSPACKOptimizer.h"

//using namespace APPSPACK;

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
APPSPACKOptimizer::APPSPACKOptimizer()
{
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
APPSPACKOptimizer::~APPSPACKOptimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void APPSPACKOptimizer::optimize(oData *odata)
{
   FILE   *fp;
   int    i, nFuncEvals, nInputs; 
   double h;
   char   *optDriver, inputFile[101], outputFile[101], systemCommand[501];
   char   lineIn[500], *subString;
   char   *keywords[] = {"FINAL MINIMUM", "f=", "x=[", "Fevals:"};
 
   //**/ ------------------------------------------------------------------
   //**/ prepare appspack input file
   //**/ ------------------------------------------------------------------
   snprintf(inputFile,100,"psuadeAPPSPACK.in");
   snprintf(outputFile,100,"psuadeAPPSPACK.out");
   fp = fopen(inputFile, "w");
   if (fp == NULL)
   {
      printf("PSUADE appspack optimize : cannot create input file.\n");
      exit(1);
   }
   nInputs = odata->nInputs_;
   optDriver = odata->funcIO_->getOptimizationDriver();
   fprintf(fp, "executable = %s\n", optDriver);
   fprintf(fp, "n_parameters = %d\n", nInputs);
   fprintf(fp, "initial_point = ");
   for (i = 0; i < nInputs; i++) fprintf(fp, "%16.8e", odata->initialX_[i]);
   fprintf(fp, "\n");
   fprintf(fp, "lower_bounds = ");
   for (i = 0; i < nInputs; i++) fprintf(fp, "%16.8e", odata->lowerBounds_[i]);
   fprintf(fp, "\n");
   fprintf(fp, "upper_bounds = ");
   for (i = 0; i < nInputs; i++) fprintf(fp, "%16.8e", odata->upperBounds_[i]);
   fprintf(fp, "\n");
   fprintf(fp, "scale = ");
   for (i = 0; i < nInputs; i++) fprintf(fp, "1 ");
   fprintf(fp, "\n");
   fprintf(fp, "params_prefix = appspackPSUADE.params.in\n");
   fprintf(fp, "result_prefix = appspackPSUADE.results.out\n");
   fprintf(fp, "num_workers = 1\n");
   fclose(fp);

   //**/ ------------------------------------------------------------------
   //**/ run appspack 
   //**/ ------------------------------------------------------------------
   h = odata->upperBounds_[0] - odata->lowerBounds_[0];
   for (i = 1; i < nInputs; i++)
      if ((odata->upperBounds_[i]-odata->lowerBounds_[i]) < h) 
         h = odata->upperBounds_[i] - odata->lowerBounds_[i];
   //**/h = h / (double) odata->nSamples_;
   h = h * odata->tolerance_;
   snprintf(systemCommand,500,"%s/bin/appspack --tol=%e --debug=%d %s > %s", 
            PSUADE_INSTALL_DIR,h,odata->outputLevel_,inputFile,outputFile);

   if (odata->outputLevel_ > 1) printf("Calling %s\n", systemCommand);
   system(systemCommand);

   fp = fopen(outputFile, "r");
   if (fp == NULL)
   {
      printf("PSUADE appspack evaluate : output file not found.\n");
      exit(1);
   }
   odata->numFuncEvals_ = 0;
   while ((fgets(lineIn, 500, fp) != NULL) && (feof(fp) == 0))
   {
      subString = strstr(lineIn,keywords[0]);
      if ( subString != NULL )
      {
         subString = strstr(lineIn,keywords[1]);
         if ( subString == NULL )
         {
            printf("PSUADE appspack optimize : fmin not found.\n");
            exit(1);
         }
         else
         {
            subString = &(subString[2]);
            sscanf(subString, "%lg", &(odata->optimalY_));
            printf("PSUADE appspack optimize : fmin = %16.8e.\n",
                   odata->optimalY_);
            subString = strstr(lineIn,keywords[2]);
            if ( subString == NULL )
            {
               printf("PSUADE appspack optimize : Xmin not found.\n");
               exit(1);
            }
            else
            {
               subString = &(subString[3]);
               for (i = 0; i < nInputs; i++)
               {
                  sscanf(subString, "%lg", &(odata->optimalX_[i]));
                  subString = &(subString[10]);
               }
            }
         }
      }
      else
      {
         subString = strstr(lineIn,keywords[3]);
         if ( subString != NULL )
         {
            subString = &(subString[8]);
            sscanf(subString, "%d", &nFuncEvals);
            odata->numFuncEvals_ += nFuncEvals;
         }
      }
   }
   fclose(fp);

   //**/ ------------------------------------------------------------------
   //**/ clean up 
   //**/ ------------------------------------------------------------------
   //**/snprintf(systemCommand,500,"/bin/rm -f psuadeAPPSPACK.in");
   //**/system(systemCommand);
   //**/snprintf(systemCommand,500,"/bin/rm -f psuadeAPPSPACK.out");
   //**/system(systemCommand);
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
APPSPACKOptimizer& APPSPACKOptimizer::operator=(const APPSPACKOptimizer &)
{
   printf("APPSPACKAnalyzer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

#else
   int bogus;
#endif


