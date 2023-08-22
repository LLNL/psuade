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
// Functions for the class OUUMinlpOptimizer
// AUTHOR : CHARLES TONG
// DATE   : 2016
// ************************************************************************
#include <PsuadeCmakeConfig.h>

#ifdef WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "PDFManager.h"
#include "OUUOptimizer.h"
#include "OUUMinlpOptimizer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Sampling.h"
#include "FuncApprox.h"
#include "PrintingTS.h"
#ifdef HAVE_NOMAD
#include "../../External/NOMAD/src/nomad.hpp"
#include "../../External/NOMAD/src/Display.hpp"
using namespace NOMAD;
#endif

// ************************************************************************
// Internal 'global' variables (for passing parameters to Fortran and C
// functions - bobyqa, newuoa and cobyla)
// ------------------------------------------------------------------------
//**/ oData object to be passed to function evaluator
//**/ Contains sample information, function evaluation link, etc.
oData  *psOUUMinlpObj_=NULL;
//**/ variables to hold how many of each type of variables
//**/ Type 1: design parameters
//**/ Type 2: recourse parameters
//**/ Type 3: discrete uncertain parameters
//**/ Type 4: continuous uncertain parameters
int psOUUMinlpM1_=-1;
int psOUUMinlpM2_=-1;
int psOUUMinlpM3_=-1;
int psOUUMinlpM4_=-1;
int psOUUMinlpNOuts_=0;

//**/ indicate response surface to be used for type 4 variables and rstype
int psOUUMinlpUseRS_=0;
int psOUUMinlpZ4RSType_=0;
//**/ indicate fast or slow kriging, if response surface is used
//**/ fast and slow depends on sample size (otherwise takes too long)
int psOUUMinlpZ4RSAux_=0;
//**/ flag to indicate the response surface is to be validated on-the-fly
int psOUUMinlpValidateRS_=0;

//**/ sample size for type 3 and 4 variables, if samples are provided
//**/ also storage for type 3 and 4 variable sample inputs
int psOUUMinlpZ3nSamples_=-1;
int psOUUMinlpZ4nSamples_=-1;
double *psOUUMinlpZ3SamInputs_=NULL;
double *psOUUMinlpZ4SamInputs_=NULL;
//**/ Lower and upper bounds for type 4 variables
double *psOUUMinlpZ4LBounds_=NULL;
double *psOUUMinlpZ4UBounds_=NULL;
//**/ probability information for type 3 sample
double *psOUUMinlpSamProbs_=NULL;

//**/ response surface generator: to be passed to callback function 
FuncApprox *psOUUMinlpfaPtr_=NULL;

//**/ for storing large samples to compute statistics on response surface
int psOUUMinlpLargeSampleSize_=0;
double *psOUUMinlpLargeSamInputs_=NULL;
double *psOUUMinlpLargeSamOutputs_=NULL;

//**/ indicate OUU mode (mean, mean+some std, ...)
int psOUUMinlpMode_=1;
//**/ for use in computing value at risk
double psOUUMinlpPercentile_=0.5;
//**/ for use in computing Mode=2 (obj func = mean + alpha * std)
double psOUUMinlpStdevMultiplier_=0;

//**/ indicate constraint mode (how violation should be computed)
int psOUUMinlpCMode_=1;
double psOUUMinlpTolerance_=0;

//**/ for job monitoring
int psOUUMinlpCounter_=0;
int psOUUMinlpParallel_=0;
//**/ indicate whether ensemble evaluation should be activated
int psOUUMinlpEnsembleEval_=0;
int psOUUMinlpCurrDriver_=-1;
int psOUUMinlpStop_=0;
int psOUUMinlpMasterMode_=0;
int psOUUMinlpPrintLevel_=0;

//**/ store optimization history
double *psOUUMinlpObjFcnStore_=NULL;

//**/ store input types (type 1, 2, 3, or 4)
int *psOUUMinlpInputTypes_=NULL;

//**/ local definition
#define PABS(x)  ((x) > 0 ? x : -(x))
#define psOUUMinlpType1 1
#define psOUUMinlpType2 2
#define psOUUMinlpType3 3
#define psOUUMinlpType4 4

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
OUUMinlpOptimizer::OUUMinlpOptimizer()
{
   cleanUp();
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OUUMinlpOptimizer::~OUUMinlpOptimizer()
{
   cleanUp();
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void OUUMinlpOptimizer::optimize(oData *odata)
{
#ifdef HAVE_NOMAD
   int    nInps, nOuts, printLevel=0, maxfun, ii, Mcheck;
   int    M1, M2, M3, M4, index, *inputPDFs, idata, kk;
   double *lbounds, *ubounds, *XValues, ddata;
   char   pString[1000], lineIn[20001];
   FILE   *fp=NULL;
   pData  pdata;

   //**/ ----------------------------------------------------------
   //**/ header and fetch data
   //**/ ----------------------------------------------------------
   printLevel = odata->outputLevel_;
   nInps = odata->nInputs_;
   nOuts = odata->nOutputs_;
   psOUUMinlpTolerance_ = odata->tolerance_;
   if (nOuts > 1)
      printf("OUUMinlpOptimizer INFO: %d inequality constraints detected.\n",
             nOuts-1);
   lbounds = odata->lowerBounds_;
   ubounds = odata->upperBounds_;
   XValues = new double[nInps];
   for (ii = 0; ii < nInps; ii++) XValues[ii] = odata->initialX_[ii];

   //**/ if this function is called again but now with different input
   //**/ and output parameters, then need to reset
   Mcheck = psOUUMinlpM1_ + psOUUMinlpM2_ + psOUUMinlpM3_ + psOUUMinlpM4_;
   if (repeatFlag_ == 1 && (nOuts != psOUUMinlpNOuts_ || Mcheck != nInps))
   {
      printf("OUUMinlpOptimizer WARNING: reuse but different information.\n");
      printf("        NOTE: Everything will be reset.\n");
      cleanUp();
   }

   //**/ ----------------------------------------------------------
   //**/ if this function has been called before, there is no need
   //**/ to spit out header information agains
   //**/ ----------------------------------------------------------
   if (printLevel >= 0 && repeatFlag_ == 0)
   {
      printAsterisks(PL_INFO, 0);
      printAsterisks(PL_INFO, 0);
      printf("*     MINLP OPTIMIZATION UNDER UNCERTAINTY\n");
      printEquals(PL_INFO, 0);
      printf("This optimization capability solves the following problem:\n");
      printf("\n   minimize_{Z1,Z2} ");
      printf("{ Phi_{Z3,Z4} [ G(Z1,Z2,Z3,Z4) ] } \n\n");
      printf("   subject to either:\n");
      printf("    (a) integer constraints on Z1; and \n");
      printf("    (b) bound constraints on Z1, Z2, and Z4; or \n");
      printf("    (c) inequality constraints on Z1,Z2,Z3,Z4\n\n");
      printf("   Z3 is a set of uncertain parameters for which a sample\n\n");
      printf("   is to be provided by the user. \n\n");
      printf("   (1) How to perform deterministic MINLP optimization? \n");
      printf("       In this case \n"); 
      printf("       - Z1 are integer optimization variables\n");
      printf("       - Z2 are continuous optimization variables\n");
      printf("       - Z3 and Z4 are empty\n");
      printf("       - G(Z1,Z2) is the simulator (opt_driver)\n\n");
      printf("   (2) How to perform MINLP OUU? \n");
      printf("       In this case \n"); 
      printf("       - Z1 are integer optimization variables\n");
      printf("       - Z2 are continuous optimization variables\n");
      printf("       - Either Z3 or Z4 below can be empty but not both:\n");
      printf("         - Z3: parameters that you will provide a sample.\n");
      printf("         - Z4: parameters that you do not provide a sample.\n");
      printf("         (optionally, you can choose sampling scheme and\n");
      printf("          whether you want response surface for Z4).\n");
      printf("       - G(Z1,Z2,Z3,Z4) is the simulator (opt_driver)\n\n");
      printf("   Phi_{Z3,Z4} above is a functional on G(Z1,Z2,Z3,Z4)\n");
      printf("   with respect to Z3 and Z4, e.g. Phi_{Z3,Z4} may be:\n");
      printf("   1. mean of G(Z1,Z2,Z3,Z4) with respect to Z3,Z4 (default)\n"); 
      printf("   2. mean of G(Z1,Z2,Z3,Z4) + alpha * std dev of G(Z1,...)\n"); 
      printf("   3. G(Z1,Z2,Z3*,Z4*) such that \n");
      printf("         Prob(G(Z1,Z2,Z3,Z4)>G(Z1,Z2,Z3*,Z4*)) = epsilon\n");
      printf("   4. min_{Z3,Z4} G(Z1,Z2,Z3,Z4) given Z1,Z2 (robust opt.)\n");
      printf("NOTE: The default is 1. Turn on optimization expert mode\n");
      printf("      mode to select the other options.\n");
      printEquals(PL_INFO, 0);
      printf("In the above formulation, the total no. of parameters M = %d\n", 
             nInps);
      printf("These parameters are to be divided into four groups:\n");
      printf("(1) Discrete optimization parameters Z1 (M1 >= 1) \n");
      printf("(2) Continuous optimization parameters Z2 (M2 >= 0) \n");
      printf("(3) Discrete uncertain parameters Z3 (user provides a sample)\n");
      printf("(4) Continuous uncertain parameters Z4 \n");
      printf("Thus, the first M1+M2 parameters are considered to be\n");
      printf("optimization parameters, and M3+M4 are uncertain parameters\n");
      printf("so that M = M1 + M2 + M3 + M4.\n");
      printEquals(PL_INFO, 0);
      //printf("To reuse simulation results (e.g. from before abrupt\n");
      //printf("termination), turn on save_history and use_history\n");
      //printf("optimization options in the ANALYSIS section (e.g.\n");
      //printf("optimization save_history). You will see a text file\n");
      //printf("called 'psuade_ouu_history' afterward.\n");
      printAsterisks(PL_INFO, 0);
      printf("IF YOU ARE READY TO MOVE ON, ENTER 'y' AND RETURN : ");
      lineIn[0] = '0';
      while (lineIn[0] != 'y' && lineIn[0] != 'n')
      {
         scanf("%s", lineIn);
         if (lineIn[0] == 'n') 
         {
            odata->optimalY_ = 0;
            for (ii = 0; ii < nInps; ii++) odata->optimalX_[ii] = 0.0;
            printf("OUUOptimizer INFO: abort.\n");
            exit(1);
         }
      }
      printEquals(PL_INFO, 0);
   }

   //**/ turning on master mode may cause problems with other part of
   //**/ psuade (e.g. response surfaces), so turn off.
   if (psConfig_.MasterModeIsOn())
   {
      printf("OUUMinlpOptimizer INFO: Master mode to be turned off.\n");
      psConfig_.MasterModeOff();
      psOUUMinlpMasterMode_ = 1;
   }

   //**/ ----------------------------------------------------------
   //**/ gather information on  M1, M2, M3, and M4
   //**/ ----------------------------------------------------------
   if (Mcheck == nInps && psOUUMinlpM1_ > 0 && psOUUMinlpM2_ > 0)
   {
      M1 = psOUUMinlpM1_;
      M2 = psOUUMinlpM2_;
      M3 = psOUUMinlpM3_;
      M4 = psOUUMinlpM4_;
   }
   else
   {
      M2 = M3 = M4 = 0;
      M1 = -1;
      printf("M1 = number of continuous optimization variables\n");
      while (M1 < 0 || M1 > nInps)
      {
         printf("Enter M1 (between 0 and %d) : ", nInps);
         scanf("%d", &M1);
      }
      if (M1 < nInps)
      {
         printf("M2 = number of discrete optimization variables\n");
         M2 = -1;
         while (M2 < 0 || M1+M2 > nInps)
         {
            printf("Enter M2 (between 0 and %d) : ", nInps-M1);
            scanf("%d", &M2);
         }
      }
      if ((M1 + M2) < nInps)
      {
         printf("M3 = number of discrete (scenario) uncertain parameters\n");
         M3 = -1;
         while (M3 < 0 || M1+M2+M3 > nInps)
         {
            printf("Enter M3 (between 0 and %d) : ", nInps-M1-M2);
            scanf("%d", &M3);
         }
         M4 = nInps - M1 - M2 - M3;
      }
      psOUUMinlpM1_ = M1;
      psOUUMinlpM2_ = M2;
      psOUUMinlpM3_ = M3;
      psOUUMinlpM4_ = M4;
   }
   if (printLevel >= 0 && repeatFlag_ == 0)
   {
      printDashes(PL_INFO, 0);
      printf("Number of continuous optimization parameters = %d\n",M1);
      printf("Number of discrete   optimization parameters = %d\n",M2);
      printf("Number of discrete   uncertain parameters    = %d\n",M3);
      printf("Number of continuous uncertain parameters    = %d\n",M4);
      printDashes(PL_INFO, 0);
   }

   //**/ ----------------------------------------------------------
   //**/ request user to identify variable types
   //**/ ----------------------------------------------------------
   if (M1 == nInps && repeatFlag_ == 0)
   {
      //**/ not repeat and all inputs are design parameters, no 
      //**/ need to ask for variable type
      psOUUMinlpInputTypes_ = new int[nInps];
      for (ii = 0; ii < nInps; ii++) psOUUMinlpInputTypes_[ii] = 1;
   }
   else if (psOUUMinlpInputTypes_ == NULL)
   {
      //**/ if input types have not been asked, do the following
      printf("In the following, please select type for each variable:\n");
      printf("  1. continuous optimization parameter\n");
      printf("  2. discrete optimization parameter\n");
      printf("  3. discrete uncertain parameter (a sample will be given)\n");
      printf("  4. continuous uncertain parameter\n");
      printf("NOTE: make sure your specification matches with above.\n");
      fgets(lineIn, 5000, stdin);
      psOUUMinlpInputTypes_ = new int[nInps];
      for (ii = 0; ii < nInps; ii++)
      {
        snprintf(pString,100,"Type for variable %d ? ", ii+1);
        psOUUMinlpInputTypes_[ii] = getInt(1,4,pString); 
      }
   }
   //**/ if not reuse, print out variable types
   if (repeatFlag_ == 0)
   {
      printEquals(PL_INFO, 0);
      for (ii = 0; ii < nInps; ii++)
      {
         if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType1) 
            printf("Input %4d is a discrete optimization parameter.\n",ii+1);
         else if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType2) 
            printf("Input %4d is a continuous optimization parameter.\n",ii+1);
         else if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType3) 
            printf("Input %4d is a discrete uncertain parameter.\n",ii+1);
         else if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4) 
            printf("Input %4d is a continuous uncertain parameter.\n",ii+1);
      }
   }
   //**/ error checking
   int c1=0, c2=0, c3=0, c4=0;
   for (ii = 0; ii < nInps; ii++)
   {
      if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType1) c1++;
      if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType2) c2++;
      if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType3) c3++;
      if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4) c4++;
   }
   if (c1 != M1 || c2 != M2 || c3 != M3 || c4 != M4)
   {
      printf("OUUMinlpOptimizer ERROR: input type counts do not match.\n");
      printf("    Number of type 1 = %d (expected = %d)\n",c1,M1);
      printf("    Number of type 2 = %d (expected = %d)\n",c2,M2);
      printf("    Number of type 3 = %d (expected = %d)\n",c3,M3);
      printf("    Number of type 4 = %d (expected = %d)\n",c4,M4);
      exit(1);
   }
   //**/ ----------------------------------------------------------
   //**/ display the PDF type for each of the M4 parameters, if 
   //**/ this is the first time this function is called 
   //**/ ----------------------------------------------------------
   if (M4 > 0 && repeatFlag_ == 0)
   {
      odata->psIO_->getParameter("input_pdfs", pdata);
      inputPDFs = pdata.intArray_;
      pdata.intArray_ = NULL;
      for (ii = 0; ii < nInps; ii++)
      {
         if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
         {
            if (inputPDFs[ii] == PSUADE_PDF_UNIFORM) 
               strcpy(pString, "Uniform");            
            if (inputPDFs[ii] == PSUADE_PDF_NORMAL) 
               strcpy(pString, "Normal");            
            if (inputPDFs[ii] == PSUADE_PDF_LOGNORMAL) 
               strcpy(pString, "Lognormal");            
            if (inputPDFs[ii] == PSUADE_PDF_TRIANGLE) 
               strcpy(pString, "Triangle");            
            if (inputPDFs[ii] == PSUADE_PDF_BETA) 
               strcpy(pString, "Beta");            
            if (inputPDFs[ii] == PSUADE_PDF_WEIBULL) 
               strcpy(pString, "Weibull");            
            if (inputPDFs[ii] == PSUADE_PDF_GAMMA) 
               strcpy(pString, "Gamma");            
            if (inputPDFs[ii] == PSUADE_PDF_EXPONENTIAL) 
               strcpy(pString, "Exponential");            
            if (inputPDFs[ii] == PSUADE_PDF_SAMPLE) 
               strcpy(pString, "User");            
            if (inputPDFs[ii] == PSUADE_PDF_F) 
               strcpy(pString, "F");            
            printf("PDF type for Input %5d = %s\n",ii+1,pString);
         }
      }
      delete [] inputPDFs;
   }
   if (repeatFlag_ == 0) printEquals(PL_INFO, 0);

   //**/ ----------------------------------------------------------
   //**/ prepare for optimization
   //**/ ----------------------------------------------------------
   for (ii = 0; ii < nInps; ii++) odata->optimalX_[ii] = 0.0;
   odata->optimalY_ = 1.0e50;

   //**/ use the opt_driver code
   maxfun = odata->maxFEval_;
   if ((odata->setOptDriver_ & 1))
   {
      if (repeatFlag_ == 0)
         printf("OUUMinlpOptimizer: set optimization simulation driver.\n");
      psOUUMinlpCurrDriver_ = odata->funcIO_->getDriver();
      odata->funcIO_->setDriver(1);
   }
   //**/ store oData pointer for function evaluation
   psOUUMinlpObj_= odata;
   if (repeatFlag_ == 0)
   {
      printAsterisks(PL_INFO, 0);
      printf("OUUMinlpOptimizer: max fevals = %d\n", odata->maxFEval_);
      printf("OUUMinlpOptimizer: tolerance  = %e\n", odata->tolerance_);
      printEquals(PL_INFO, 0);
   }

   //**/ ----------------------------------------------------------
   //**/ ask for more information is this is the first time 
   //**/ set default: no user opt code, no RS
   //**/ then ask for type of objective function ==> psMinlpOUUMode_ 
   //**/ only if M3+M4 > 0 and expert mode is on (default=1: mean)
   //**/ ----------------------------------------------------------
   if (repeatFlag_ == 0)
   {
      psOUUMinlpUseRS_ = 0;
      if (psConfig_.OptExpertModeIsOn() && (nInps > (M1+M2)))
      {
         printEquals(PL_INFO, 0);
         printf("Select which functional Phi_{Z3,Z4} to use: \n");
         printf("  1. mean of G(Z1,...) with respect to Z3,Z4 (default)\n"); 
         printf("  2. mean of G(Z1,...) + beta * std dev of G(Z1,...)\n"); 
         printf("  3. G(Z1,Z2,Z3*,Z4*) such that \n");
         printf("         Prob(G(Z1,...)>G(Z1,Z2,Z3*,Z4*)) = 1 - alpha\n");
         printf("     This is also called value-at-risk with confidence\n");
         printf("     level alpha.\n");
         snprintf(pString,100,"Enter your choice of functional (1, 2 or 3) : ");
         psOUUMinlpMode_ = getInt(1, 3, pString);
         if (psOUUMinlpMode_ == 2)
         {
            snprintf(pString,100,"Enter beta (>= 0) : ");
            psOUUMinlpStdevMultiplier_ = -1;
            while (psOUUMinlpStdevMultiplier_ < 0)
               psOUUMinlpStdevMultiplier_ = getDouble(pString);
         } 
         if (psOUUMinlpMode_ == 3)
         {
            psOUUMinlpPercentile_ = 0.0;
            snprintf(pString,100,"Enter the confidence interval : [0.5 - 1.0] ");
            while ((psOUUMinlpPercentile_ < 0.5) || 
                   (psOUUMinlpPercentile_ > 1.0)) 
            { 
               psOUUMinlpPercentile_ = getDouble(pString);
            } 
         }
         if (nOuts > 1)
         {
            printf("Please select below the scheme to compute the returned\n");
            printf("constraints from the ensemble constraints.\n");
            printf("The returned constraints are computed from:\n");
            printf(" 1. the means of constraints from all sample runs\n");
            printf(" 2. the simulation with largest 2-norm violations\n");
            printf(" 3. the simulation with largest 1-norm violations\n");
            printf(" 4. the simulation with largest infinity-norm violation\n");
            printf("NOTE: constraints are such that they should be >= 0.\n");
            snprintf(pString,100,"Enter your choice (1 - 4) : ");
            psOUUMinlpCMode_ = getInt(1, 4, pString);
         }
      }
   }

   //**/ ----------------------------------------------------------
   //**/ request users for a sample for Z3 
   //**/ ----------------------------------------------------------
   genZ3Sample();

   //**/ ----------------------------------------------------------
   //**/ generate sample for Z4 
   //**/ ----------------------------------------------------------
   genZ4Sample();
 
   //**/ ----------------------------------------------------------
   //**/ ask user if ensemble simulation is desired
   //**/ ----------------------------------------------------------
   if ((M3+M4) > 0 && repeatFlag_ == 0)
   {
      printf("Each simulation calls the opt_driver with 1 sample point.\n");
      printf("For higher efficiency (less I/O), you have the option to\n");
      printf("provide in 'ensemble_opt_driver' an executable that can\n");
      printf("run multiple sample points. In this case, OUU calls the\n");
      printf("ensemble executable with the following sequence: \n");
      printf("    <ensemble_opt_driver> <sampleFile> <outputFile>\n\n");
      printf("where <sampleFile> is in the following format:\n");
      printf("   line 1: <nSamples>\n");
      printf("   line 2: Sample point 1 input values\n");
      printf("   line 3: Sample point 2 input values\n");
      printf("   line n: ...\n\n");
      printf("and <outputFile> should have all sample output values.\n\n");
      printf("Use ensemble opt driver for ensemble runs ? (y or n) ");
      lineIn[0] = '1';
      while (lineIn[0] != 'n' && lineIn[0] != 'y')
      {
         scanf("%s", lineIn);
         if (lineIn[0] == 'y') psOUUMinlpEnsembleEval_ = 1;
         printEquals(PL_INFO, 0);
      }
   }

   //**/ ----------------------------------------------------------
   //**/ if ensemble_opt_driver is not used, find out what run mode
   //**/ ----------------------------------------------------------
   if ((psOUUMinlpEnsembleEval_ == 0) && ((M3+M4)>0) && repeatFlag_ == 0)
   {
      printf("You can configure OUU to run the ensemble simulations in\n");
      printf("parallel/asynchronous using the Linux fork/join. If 'n'\n");
      printf("is entered below, the opt_driver simulator will be\n");
      printf("evaluated sequentially (one sample point at a time).\n");
      printf("If 'y' is selected instead, be careful, because PSUADE\n");
      printf("PSUADE will launch %d jobs simultaneously, which may\n",
             psOUUMinlpZ3nSamples_*psOUUMinlpZ4nSamples_);
      printf("jam up the job queuing system.\n");
      printf("Turn on asynchronous mode ? (y or n) ");
      lineIn[0] = '1';
      while (lineIn[0] != 'n' && lineIn[0] != 'y')
      {
         scanf("%s", lineIn);
         if (lineIn[0] == 'y') psOUUMinlpParallel_ = 1;
         printEquals(PL_INFO, 0);
      }
      fgets(lineIn, 500, stdin);
   }

   //**/ ----------------------------------------------------------
   //**/ set up response surface constructor 
   //**/ ----------------------------------------------------------
   genResponseSurface();

   //**/ ----------------------------------------------------------
   //**/ ----- run optimizer -----
   //**/ ----------------------------------------------------------
   psOUUMinlpCounter_ = 0;
   if (psOUUMinlpParallel_ == 1) odata->funcIO_->setAsynchronousMode();
   printAsterisks(PL_INFO, 0);
   printOutTS(PL_INFO,"MINLP OPTIMIZATION UNDER UNCERTAINTY BEGINS\n");
   psOUUMinlpPrintLevel_ = printLevel;
   Display dout ( std::cout );
   dout.precision( DISPLAY_PRECISION_STD );
   Parameters nomadp(dout);
   //**/ input set up: dimension and types
   nomadp.set_DIMENSION(M1+M2);
   Point lbnds(M1+M2), ubnds(M1+M2), X0(M1+M2);
   index = 0;
   for (ii = 0; ii < nInps; ii++)
   {
      if (psOUUMinlpInputTypes_ != NULL)
      {
         switch(psOUUMinlpInputTypes_[ii])
         {
            case 1: nomadp.set_BB_INPUT_TYPE (ii, CONTINUOUS);
                    lbnds[index] = lbounds[ii];
                    ubnds[index] = ubounds[ii];
                    X0[index] = XValues[ii];
                    index++;
                    break;
            case 2: nomadp.set_BB_INPUT_TYPE (ii, INTEGER);
                    lbnds[index] = lbounds[ii];
                    ubnds[index] = ubounds[ii];
                    X0[index] = (int) XValues[ii];
                    index++;
                    break;
         }
      }
   }
   nomadp.set_LOWER_BOUND (lbnds);
   nomadp.set_UPPER_BOUND (ubnds);
   nomadp.set_X0(X0);

   //**/ set up nomad output (it can handle constraints)
   psOUUMinlpNOuts_ = nOuts;
   vector<bb_output_type> bbot (nOuts);
   bbot[0] = OBJ;
   for (ii = 1; ii < nOuts; ii++) bbot[ii] = PB;
   nomadp.set_BB_OUTPUT_TYPE ( bbot );

   //**/ set other nomad parameters
   nomadp.set_SPECULATIVE_SEARCH ( true );
   if (psConfig_.OptExpertModeIsOn())
   {
     snprintf(pString,100,
             "Enter value for mesh update basis (default = 8) : ");
     ddata = getDouble(pString);
   }
   else ddata = 8.0;
   nomadp.set_MESH_UPDATE_BASIS ( ddata );

   if (psConfig_.OptExpertModeIsOn())
   {
     snprintf(pString,100,
             "Enter value for mesh coarsening exponent (default = 1) : ");
     idata = getInt(0, 10, pString);
   }
   else idata = 1;
   nomadp.set_MESH_COARSENING_EXPONENT ( idata );
   nomadp.set_MAX_BB_EVAL (maxfun);
   for (ii = 0; ii < M1+M2; ii++)
   {
      ddata = 0.5 * (ubnds[ii].value() - lbnds[ii].value());
      //**/ nomadp.set_INITIAL_MESH_SIZE ( 0.4 );
      nomadp.set_INITIAL_MESH_SIZE ( ii, ddata, false );
   }
   //**/ checking required
   //**/ nomadp.display(dout);
   nomadp.check();

   //**/ set up simulation model pointer
   PsuadeMinlpEvaluator madEvaluator(nomadp);
   Mads mads(nomadp, &madEvaluator);
   psOUUMinlpObj_ = odata;
   psOUUMinlpObj_->numFuncEvals_ = 0;
   psOUUMinlpObjFcnStore_ = new double[maxfun];
   for (ii = 0; ii < maxfun; ii++) psOUUMinlpObjFcnStore_[ii] = -1e35;

   //**/ ----------------------------------------------------------
   //**/ ------ run nomad and process outputs ------
   //**/ ----------------------------------------------------------
   psOUUMinlpStop_ = 0;
   mads.run();
   printf("OUUMinlpOptimizer: total number of evaluations = %d\n",
          odata->numFuncEvals_);

   char name[1000];
   if (psConfig_.OptExpertModeIsOn())
   {
      fp = fopen("matlabouuminlp.m", "w");
      if (fp != NULL)
      {
         fprintf(fp, "A = [\n");
         for (ii = 0; ii < maxfun; ii++)
         {
            if (psOUUMinlpObjFcnStore_[ii] <= -1e35) break;
            else fprintf(fp, "%e\n", psOUUMinlpObjFcnStore_[ii]);
         }
         fprintf(fp, "];\n");
         fprintf(fp, "plot(A,'linewidth',2)\n");
         fwritePlotAxes(fp);
         strcpy(name, "Iteration count");
         fwritePlotXLabel(fp, name);
         strcpy(name, "Objective function value");
         fwritePlotYLabel(fp, name);
         strcpy(name,
            "OUUMinlp Convergence Plot (Some values may be infeasible)");
         fwritePlotTitle(fp, name);
         fclose(fp);
         printf("matlabouuminlp.m is now available.\n");
      }
   }
   delete [] psOUUMinlpObjFcnStore_;
   psOUUMinlpObjFcnStore_ = NULL;

   //**/ ----------------------------------------------------------
   //**/ set return values, clean up, and restore settings 
   //**/ ----------------------------------------------------------
   if ((odata->setOptDriver_ & 2) && psOUUMinlpCurrDriver_ >= 0)
   {
      odata->funcIO_->setDriver(psOUUMinlpCurrDriver_);
   }
   delete [] XValues;
   if (psOUUMinlpMasterMode_ != 0) psConfig_.MasterModeOn();

   //**/ ----------------------------------------------------------
   //**/ set repeat flag so that no questions will be repeated later
   //**/ ----------------------------------------------------------
   repeatFlag_ = 1;
#else
   printf("ERROR: Nomads not available.\n");
   exit(1);
#endif
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void OUUMinlpOptimizer::cleanUp()
{
   if (psOUUMinlpZ3SamInputs_ != NULL) delete [] psOUUMinlpZ3SamInputs_;
   if (psOUUMinlpZ4SamInputs_ != NULL) delete [] psOUUMinlpZ4SamInputs_;
   if (psOUUMinlpZ4LBounds_   != NULL) delete [] psOUUMinlpZ4LBounds_;
   if (psOUUMinlpZ4UBounds_   != NULL) delete [] psOUUMinlpZ4UBounds_;
   if (psOUUMinlpSamProbs_    != NULL) delete [] psOUUMinlpSamProbs_;
   if (psOUUMinlpLargeSamInputs_  != NULL) 
      delete [] psOUUMinlpLargeSamInputs_;
   if (psOUUMinlpLargeSamOutputs_ != NULL) 
      delete [] psOUUMinlpLargeSamOutputs_;
   if (psOUUMinlpfaPtr_ != NULL) delete psOUUMinlpfaPtr_;
   if (psOUUMinlpInputTypes_ != NULL) delete psOUUMinlpInputTypes_;
   psOUUMinlpZ3SamInputs_ = NULL;
   psOUUMinlpZ4SamInputs_ = NULL;
   psOUUMinlpZ4LBounds_ = NULL;
   psOUUMinlpZ4UBounds_ = NULL;
   psOUUMinlpSamProbs_ = NULL;
   psOUUMinlpLargeSamInputs_ = NULL;
   psOUUMinlpLargeSamOutputs_ = NULL;
   psOUUMinlpfaPtr_ = NULL;
   psOUUMinlpInputTypes_ = NULL;
   psOUUMinlpM1_ = -1;
   psOUUMinlpM2_ = -1;
   psOUUMinlpM3_ = -1;
   psOUUMinlpM4_ = -1;
   psOUUMinlpZ3nSamples_ = 0;
   psOUUMinlpZ4nSamples_ = 0;
   psOUUMinlpObj_ = NULL;
   psOUUMinlpUseRS_ = 0;
   repeatFlag_ = 0;
   psOUUMinlpEnsembleEval_ = 0;
   psOUUMinlpParallel_ = 0;
}

// ************************************************************************
// create Z3 sample
// ------------------------------------------------------------------------
void OUUMinlpOptimizer::genZ3Sample()
{
   int    M3, nInps, nOuts, ii, kk, index;
   double ddata;
   char   lineIn[10000], filename[1000];
   FILE   *fp;
   oData  *odata;

   //**/ fetch information
   M3    = psOUUMinlpM3_;
   odata = psOUUMinlpObj_;
   nInps = psOUUMinlpM1_ + psOUUMinlpM2_ + psOUUMinlpM3_ + psOUUMinlpM4_;
   nOuts = psOUUMinlpNOuts_;
   //**/ if M3 = 0, sample size is 1
   if (repeatFlag_ == 0) psOUUMinlpZ3nSamples_ = 1;
   if (M3 > 0 && repeatFlag_ == 0)
   {
      printEquals(PL_INFO, 0);
      printf("A sample for Z3 is needed from you. Data format is:\n");
      printf("line 1: <nSamples> <nInputs> \n");
      printf("line 2: <sample 1 input 1> <input 2> ... <probability>\n");
      printf("line 3: <sample 2 input 1> <input 2> ... <probability>\n");
      printf("...\n");
      printf("Enter user sample file name : ");
      scanf("%s", filename);
      fgets(lineIn, 5000, stdin);
      fp = fopen(filename, "r");
      if (fp == NULL)
      {
         printf("OUUMinlpOptimizer ERROR: user sample file %s not found.\n",
                 filename);
            exit(1);
      }
      fgets(lineIn, 10000, fp);
      sscanf(lineIn, "%d %d", &psOUUMinlpZ3nSamples_, &ii);
      if (ii != M3)
      {
         printf("OUUMinlpOptimizer ERROR: user sample nInputs %d != %d\n",
                ii, M3);
         fclose(fp);
         exit(1);
      } 
      if (psOUUMinlpZ3nSamples_ < M3+1)
      {
         printf("OUUMinlpOptimizer ERROR: user sample size must be >= %d\n",
                M3+1);
         fclose(fp);
         exit(1);
      } 
      psOUUMinlpZ3SamInputs_ = new double[psOUUMinlpZ3nSamples_ * M3];
      psOUUMinlpSamProbs_    = new double[psOUUMinlpZ3nSamples_];
      ddata = 0.0;
      if (odata->outputLevel_ > 4) printf("Reading sample for X3.\n");
      for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++)
      {
         fgets(lineIn, 20000, fp);
         index = kk = 0;
         while (kk < M3)
         {
            while (lineIn[index] == ' ') index++;
            if (lineIn[index] == '\0' || lineIn[index] == '\n')
            {
               printf("ERROR: reading sample file %s line %d.\n",
                      filename,ii+2);
               fclose(fp);
               exit(1);
            }
            sscanf(&lineIn[index],"%lg",&psOUUMinlpZ3SamInputs_[ii*M3+kk]);
            while (lineIn[index] != ' ') index++;
            if (lineIn[index] == '\0' || lineIn[index] == '\n')
            {
               printf("ERROR: reading sample file %s line %d.\n",
                      filename,ii+2);
               fclose(fp);
               exit(1);
            }
            if (odata->outputLevel_ > 4)
               printf("%12.4e ",psOUUMinlpZ3SamInputs_[ii*M3+kk]);
            kk++;
         }
         while (lineIn[index] == ' ') index++;
         if (lineIn[index] == '\0' || lineIn[index] == '\n')
         {
            printf("WARNING: reading probability at line %d of %s.\n",
                   ii+2, filename);
            printf("    Will set probabilities to be equiprobable.\n");
            psOUUMinlpSamProbs_[ii] = PSUADE_UNDEFINED;
            ddata += 1.0;
         }
         else sscanf(&lineIn[index],"%lg",&psOUUMinlpSamProbs_[ii]);
         if (odata->outputLevel_ > 4)
            printf("%12.4e\n",psOUUMinlpSamProbs_[ii]);
      }
      fclose(fp);
      if (ddata != 0)
      {
         for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++) 
            psOUUMinlpSamProbs_[ii] = 1.0 / (double) psOUUMinlpZ3nSamples_;
      }
      else
      {
         ddata = 0.0;
         for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++) 
            ddata += psOUUMinlpSamProbs_[ii];
      }
      printf("User sample for Z3 has %d points\n", psOUUMinlpZ3nSamples_);
      printf("User sample for Z3 CDF = %e (should be ~1)\n", ddata);
   }
   return;
}

// ************************************************************************
// create Z4 sample
// ------------------------------------------------------------------------
void OUUMinlpOptimizer::genZ4Sample()
{
   int    M4, *inputPDFs, kk, index, ii, nInps, samplingOption=0, methdZ4;
   int    nOuts, iOne=1, iZero=0, *samStates, *iPdfs, index2, methodZ4;
   double *lowers, *uppers, *samOuts, *inputMeans=NULL, *inputStdevs=NULL;
   double *iMeans, *iStdvs, ddata;
   char   pString[1000];
   oData  *odata;
   pData  pdata;
   psMatrix   *corMat1, corMat2;
   Sampling   *sampler = NULL;
   PDFManager *pdfman=NULL;
   psVector   vecLB, vecUB, vecOut;

   //**/ fetch information
   M4    = psOUUMinlpM4_;
   odata = psOUUMinlpObj_;
   nInps = psOUUMinlpM1_ + psOUUMinlpM2_ + psOUUMinlpM3_ + psOUUMinlpM4_;
   nOuts = psOUUMinlpNOuts_;
   //**/ if M4 = 0, sample size is 1
   if (repeatFlag_ == 0) psOUUMinlpZ4nSamples_ = 1;
   if (M4 > 0 && repeatFlag_ == 0)
   {
      //**/ see if non-uniform pdf is requested
      samplingOption = 0;
      odata->psIO_->getParameter("input_pdfs", pdata);
      inputPDFs = pdata.intArray_;
      pdata.intArray_ = NULL;
      kk = index = 0;
      if (inputPDFs != NULL)
      {
         for (ii = 0; ii < nInps; ii++) 
         {
            if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
            {
               kk += inputPDFs[ii];
               index++;
            }
         }
      }
      if (kk > 0) samplingOption = 1;

      //**/ ask for Z4 sampling method
      if (samplingOption == 0)
      {
         printf("OUUMinlp uses a Z4 sample to estimate the statistics.\n");
         printf("Available sampling method: \n");
         printf("   (1) MC, \n");
         printf("   (2) LHS, \n");
         printf("   (3) quasi-MC (recommended for M4 < 51).\n");
         snprintf(pString,100,"Select sampling method (1, 2 or 3) : ");
         methodZ4 = getInt(1, 3, pString);
      }
      else methodZ4 = 1;

      //**/ select sample size for continuous uncertain parameters
      snprintf(pString,100,"Enter sample size (>= %d, <= 300) : ", kk);
      psOUUMinlpZ4nSamples_ = getInt(2, 1000, pString);
      printf("Z4 sample has sample size = %d\n",psOUUMinlpZ4nSamples_);  

      //**/ option to build response surface if nOutputs = 1
      psOUUMinlpUseRS_ = 0;
      if (nOuts == 1)
      {
         printf("The user sample for Z4 has %d points\n",
                psOUUMinlpZ4nSamples_);
         printf("You have the option to use the Z4 sample to build a\n");
         printf("response surface for more accurately estimating the\n");
         printf("output statistics.\n");
         printf("Use response surface for Z4? (y or n) ");
         scanf("%s", pString);
         if (pString[0] == 'y') psOUUMinlpUseRS_ = 1;
         if (psOUUMinlpMasterMode_ == 1 && psOUUMinlpUseRS_ == 1)
         {
            printf("For diagnostics purposes, you may choose to validate\n");
            printf("the response surface constructed at each OUU iteration.\n");
            printf("The more accurate the cross validation result is, the\n");
            printf("more accurate is the statistics.\n");
            printf("Validate the response surfaces for Z4? (y or n) ");
            scanf("%s", pString);
            if (pString[0] == 'y') psOUUMinlpValidateRS_ = 1;
         }
         fgets(pString, 1000, stdin);
      }
      else
      {
         printf("OUUMinlpOptimizer INFO: response surface not available\n");
         printf("                        for problems with constraints.\n");
      }

      //**/ generate sample with uniform PDF ==> psOUUMinlpZ4SamInputs)
      if (samplingOption == 0)
      { 
         if (methodZ4 == 1)
              sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_MC);
         else if (methodZ4 == 2)
              sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
         else sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
         sampler->setPrintLevel(0);
         lowers = new double[M4];
         uppers = new double[M4];
         index = 0;
         for (ii = 0; ii < nInps; ii++)
         {
            if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
            {
               lowers[index] = odata->lowerBounds_[ii];
               uppers[index] = odata->upperBounds_[ii];
               index++;
            }
         }
         sampler->setInputBounds(M4, lowers, uppers);
         sampler->setOutputParams(iOne);
         sampler->setSamplingParams(psOUUMinlpZ4nSamples_, iOne, iZero);
         sampler->initialize(0);
         psOUUMinlpZ4nSamples_ = sampler->getNumSamples();
         psOUUMinlpZ4SamInputs_ = new double[psOUUMinlpZ4nSamples_ * M4];
         samStates = new int[psOUUMinlpZ4nSamples_];
         samOuts = new double[psOUUMinlpZ4nSamples_];
         sampler->getSamples(psOUUMinlpZ4nSamples_,M4,iOne,
                        psOUUMinlpZ4SamInputs_,samOuts, samStates);
         delete sampler;
         delete [] lowers;
         delete [] uppers;
         delete [] samStates;
         delete [] samOuts;
      }
      else
      //**/ non-uniform PDF 
      {
         //**/ fetch input means and std dev
         odata->psIO_->getParameter("input_means", pdata);
         inputMeans = pdata.dbleArray_;
         if (inputMeans == NULL)
         {
            inputMeans = new double[nInps];
            for (ii = 0; ii < nInps; ii++) inputMeans[ii] = 0;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_stdevs", pdata);
         inputStdevs = pdata.dbleArray_;
         if (inputStdevs == NULL)
         {
            inputStdevs = new double[nInps];
            for (ii = 0; ii < nInps; ii++) inputStdevs[ii] = 1;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_cor_matrix", pdata);
         corMat1 = (psMatrix *) pdata.psObject_;
         pdata.psObject_ = NULL;

         //**/ compress pdf information
         corMat2.setDim(M4,M4);
         iPdfs  = new int[nInps];
         iMeans = new double[nInps];
         iStdvs = new double[nInps];
         lowers = new double[nInps];
         uppers = new double[nInps];
         index = 0;
         for (ii = 0; ii < nInps; ii++)
         {
            if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
            { 
               iMeans[index] = inputMeans[ii];
               iStdvs[index] = inputStdevs[ii];
               iPdfs[index]  = inputPDFs[ii];
               lowers[index] = odata->lowerBounds_[ii];
               uppers[index] = odata->upperBounds_[ii];
               index2 = 0;
               for (kk = 0; kk < nInps; kk++)
               {
                  if (psOUUMinlpInputTypes_[kk] == psOUUMinlpType4)
                  {
                     ddata = corMat1->getEntry(ii,kk);
                     corMat2.setEntry(index,index2,ddata);
                     index2++;
                  }
               }
               index++;
            }
         }
         //**/ generate sample
         pdfman = new PDFManager();
         pdfman->initialize(M4,iPdfs,iMeans,iStdvs,corMat2,NULL,NULL);
         vecLB.load(M4, lowers);
         vecUB.load(M4, uppers);
         vecOut.setLength(psOUUMinlpZ4nSamples_*M4);
         pdfman->genSample(psOUUMinlpZ4nSamples_, vecOut, vecLB, vecUB);
         psOUUMinlpZ4SamInputs_= new double[psOUUMinlpZ4nSamples_*M4];
         for (ii = 0; ii < psOUUMinlpZ4nSamples_*M4; ii++)
            psOUUMinlpZ4SamInputs_[ii] = vecOut[ii];
         delete [] inputPDFs;
         delete [] inputMeans;
         delete [] inputStdevs;
         delete [] iPdfs;
         delete [] iMeans;
         delete [] iStdvs;
         delete [] lowers;
         delete [] uppers;
         delete pdfman;
      }

      //**/ if RS requested, create a large sample for propagation
      if (psOUUMinlpUseRS_ == 1)
      {
         lowers = new double[M4];
         uppers = new double[M4];
         index = 0;
         for (ii = 0; ii < nInps; ii++)
         {
            if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
            {
               lowers[index] = odata->lowerBounds_[ii];
               uppers[index] = odata->upperBounds_[ii];
               index++;
            }
         }
         //**/ if uniform distributions for all parameters
         if (samplingOption == 0)
         {
            sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
            sampler->setPrintLevel(0);
            sampler->setInputBounds(M4, lowers, uppers);
            sampler->setOutputParams(iOne);
            psOUUMinlpLargeSampleSize_ = 100000;
            sampler->setSamplingParams(psOUUMinlpLargeSampleSize_,iOne,iZero);
            sampler->initialize(0);
            psOUUMinlpLargeSampleSize_ = sampler->getNumSamples();
            psOUUMinlpLargeSamInputs_ = 
                       new double[psOUUMinlpLargeSampleSize_*M4];
            kk = psOUUMinlpLargeSampleSize_*nOuts;
            psOUUMinlpLargeSamOutputs_ = new double[kk];
            samStates = new int[psOUUMinlpLargeSampleSize_];
            sampler->getSamples(psOUUMinlpLargeSampleSize_, M4, iOne,
                     psOUUMinlpLargeSamInputs_,psOUUMinlpLargeSamOutputs_,
                     samStates);
            delete [] samStates;
            delete sampler;
         }
         //**/ if non-uniform distributions for some parameters
         else
         {
            //**/ fetch pdf information
            odata->psIO_->getParameter("input_pdfs", pdata);
            inputPDFs = pdata.intArray_;
            pdata.intArray_ = NULL;
            odata->psIO_->getParameter("input_means", pdata);
            inputMeans = pdata.dbleArray_;
            pdata.dbleArray_ = NULL;
            odata->psIO_->getParameter("input_stdevs", pdata);
            inputStdevs = pdata.dbleArray_;
            pdata.dbleArray_ = NULL;
            odata->psIO_->getParameter("input_cor_matrix", pdata);
            corMat1 = (psMatrix *) pdata.psObject_;
            pdata.psObject_ = NULL;
            //**/ compress all pdf information
            corMat2.setDim(M4,M4);
            iPdfs  = new int[nInps];
            iMeans = new double[nInps];
            iStdvs = new double[nInps];
            index = 0;
            for (ii = 0; ii < nInps; ii++)
            {
               if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
               {
                  iMeans[index] = inputMeans[ii];
                  iStdvs[index] = inputStdevs[ii];
                  iPdfs[index]  = inputPDFs[ii];
                  index2 = 0;
                  for (kk = 0; kk < nInps; kk++)
                  {
                     if (psOUUMinlpInputTypes_[kk] == psOUUMinlpType4)
                     {
                        ddata = corMat1->getEntry(ii,kk);
                        corMat2.setEntry(index,index2,ddata);
                        index2++;
                     }
                  }
                  index++;
               }
            }
            //**/ create large sample
            pdfman = new PDFManager();
            pdfman->initialize(M4,iPdfs,iMeans,iStdvs,corMat2,NULL,NULL);
            vecLB.load(M4, lowers);
            vecUB.load(M4, uppers);
            psOUUMinlpLargeSampleSize_ = 100000;
            vecOut.setLength(psOUUMinlpLargeSampleSize_*M4);
            pdfman->genSample(psOUUMinlpLargeSampleSize_,vecOut,vecLB,vecUB);
            psOUUMinlpLargeSamInputs_ = 
                        new double[psOUUMinlpLargeSampleSize_*M4];
            kk = psOUUMinlpLargeSampleSize_*nOuts;
            psOUUMinlpLargeSamOutputs_ = new double[kk];
            for (ii = 0; ii < psOUUMinlpLargeSampleSize_*M4; ii++)
               psOUUMinlpLargeSamInputs_[ii] = vecOut[ii];
            delete [] inputMeans;
            delete [] inputStdevs;
            delete [] iPdfs;
            delete [] iMeans;
            delete [] iStdvs;
            delete pdfman;
         }
         delete [] inputPDFs;
         delete [] lowers;
         delete [] uppers;
      }
   }
   return;
}

// ************************************************************************
// set up response surface 
// ------------------------------------------------------------------------
void OUUMinlpOptimizer::genResponseSurface()
{
   int    M4, nInps, nOuts, rstype, rsaux, kk, index, ii, printLevel=0;
   double *lowers, *uppers;
   char   **targv, pString[1000];
   oData  *odata = psOUUMinlpObj_;

   //**/ fetch information
   M4    = psOUUMinlpM4_;
   odata = psOUUMinlpObj_;
   nInps = psOUUMinlpM1_ + psOUUMinlpM2_ + psOUUMinlpM3_ + psOUUMinlpM4_;
   nOuts = psOUUMinlpNOuts_;
   if (odata != NULL) printLevel = odata->outputLevel_;
   //**/ check condition 
   if (repeatFlag_ == 0 && psOUUMinlpUseRS_ == 1 && M4 > 0)
   {
      psOUUMinlpfaPtr_ = NULL;
      rstype = rsaux = 0;
      if (psOUUMinlpMasterMode_ == 0)
      {
         if (printLevel > 2) 
            printf("OUUMinlpOptimizer: setting up response surface\n");
         if (psOUUMinlpZ4nSamples_ > 600)
         {
            rstype = PSUADE_RS_MARS;
            printf("OUUMinlpOptimizer: use MARS since nSamples > 400\n");
         }
         else if (psOUUMinlpZ4nSamples_ > 300)
         {
            rstype = PSUADE_RS_RBF;
            printf("OUUMinlpOptimizer: use RBF response surface\n");
         }
         else if (psOUUMinlpZ4nSamples_ > 200)
         {
            rstype = PSUADE_RS_KR;
            printf("OUUMinlpOptimizer: use Kriging (fast) response surface\n");
         }
         else
         {
            rstype = PSUADE_RS_KR;
            rsaux = 1;
            printf("OUUMinlpOptimizer: use Kriging (slow) response surface\n");
         }
      }
      else
      {
         printf("Response surface available: \n");
         printf("1. MARS\n");
         printf("2. Kriging (fast)\n");
         printf("3. Kriging (slow)\n");
         printf("4. Radial basis function\n");
         snprintf(pString,100,"Which response surface? ");
         rstype = getInt(1,4,pString); 
         if (rstype == 1) rstype = PSUADE_RS_MARS;
         if (rstype == 2)
         {
            rstype = PSUADE_RS_KR;
            rsaux = 0;
         }
         if (rstype == 3)
         {
             rstype = PSUADE_RS_KR;
             rsaux = 1;
         }
         if (rstype == 4) rstype = PSUADE_RS_RBF;
      }
      //**/ create response surface
      psConfig_.InteractiveSaveAndReset();
      psOUUMinlpfaPtr_ = genFA(rstype, M4, -1, psOUUMinlpZ4nSamples_);
      lowers = new double[M4];
      uppers = new double[M4];
      psOUUMinlpZ4RSType_ = rstype;
      psOUUMinlpZ4RSAux_ = rsaux;
      psOUUMinlpZ4LBounds_ = new double[M4];
      psOUUMinlpZ4UBounds_ = new double[M4];
      index = 0;
      for (ii = 0; ii < nInps; ii++)
      {
         if (psOUUMinlpInputTypes_[ii] == psOUUMinlpType4)
         {
            lowers[index] = odata->lowerBounds_[ii];
            uppers[index] = odata->upperBounds_[ii];
            psOUUMinlpZ4LBounds_[index] = lowers[index];
            psOUUMinlpZ4UBounds_[index] = uppers[index];
            index++;
         }
      }
      psOUUMinlpfaPtr_->setBounds(lowers,uppers);
      psOUUMinlpfaPtr_->setOutputLevel(0);
      psConfig_.InteractiveRestore();
      if (rstype == PSUADE_RS_KR) 
      {
         targv = new char*[1];
         targv[0] = new char[100];
         if (rsaux == 0) strcpy(targv[0], "setMode2");
         else            strcpy(targv[0], "setMode3");
         psOUUMinlpfaPtr_->setParams(1, targv);
         delete [] targv[0];
         delete [] targv;
      }
      delete [] lowers;
      delete [] uppers;
   }
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
OUUMinlpOptimizer& OUUMinlpOptimizer::operator=(const OUUMinlpOptimizer &)
{
   printf("OUUMinlpOptimizer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

#if HAVE_NOMAD
// ************************************************************************
// PsuadeMinlpEvaluator constructor
// ------------------------------------------------------------------------
PsuadeMinlpEvaluator::PsuadeMinlpEvaluator(const Parameters &pset): 
                      Evaluator(pset)
{
}

// ************************************************************************
// PsuadeMinlpEvaluator destructor
// ------------------------------------------------------------------------
PsuadeMinlpEvaluator::~PsuadeMinlpEvaluator()
{
}

// ************************************************************************
// PsuadeMinlpEvaluator evaluation 
// ------------------------------------------------------------------------
bool PsuadeMinlpEvaluator::eval_x(Eval_Point &X,const Double &h_max,
                                  bool &count_eval)
{
   int    ii, jj, kk, MOpt, M, funcID, goodFlag, nOuts, nSamp, ind12, ind3;
   int    ss, ind4, *readys, index;
   double *XVals, *YVals, ddata;
   bool   retFlag=true;
   FILE   *fp=NULL;
   oData  *odata = psOUUMinlpObj_;

   //**/ error checking
   if (odata == NULL)
   {
      printf("MinlpEvaluator ERROR: odata not available.\n");
      exit(1);
   }
   //**/ If stop has been requested, just perform dummy evaluation
   if (psOUUMinlpStop_ == 1)
   {
      ddata = -1e50;
      X.set_bb_output(0, ddata);
      //**/ set all other outputs to zero (as YVals[0])
      ddata = 0.0;
      for (ii = 1; ii < psOUUMinlpNOuts_; ii++) X.set_bb_output(ii,ddata);
      retFlag = false;
      return retFlag;
   }

   //**/ convert nomad X format to double *
   M     = psOUUMinlpM1_ + psOUUMinlpM2_ + psOUUMinlpM3_ + psOUUMinlpM4_;
   MOpt  = X.get_n();
   if (MOpt != psOUUMinlpM1_ + psOUUMinlpM2_)
   {
      printf("MinlpEvaluator ERROR: invalid number of opt parameter.\n");
      exit(1);
   }
   nOuts = psOUUMinlpNOuts_;

   //**/ evaluate function
   nSamp = psOUUMinlpZ3nSamples_ * psOUUMinlpZ4nSamples_;
   XVals = new double[nSamp*M];
   YVals = new double[nSamp*nOuts];

   //**/ fill in the design and uncertain variables
   for (ss = 0; ss < nSamp; ss++)
   {
      ind12 = ind3 = ind4 = 0;
      for (jj = 0; jj < M; jj++)
      {
         if (psOUUMinlpInputTypes_[jj] == psOUUMinlpType1)
         {
            XVals[ss*M+jj] = X[ind12].value();
            ind12++;
         }
         else if (psOUUMinlpInputTypes_[jj] == psOUUMinlpType2)
         {
            XVals[ss*M+jj] = X[ind12].value();
            ind12++;
         }
         else if (psOUUMinlpInputTypes_[jj] == psOUUMinlpType3)
         {
            kk = ss / psOUUMinlpZ4nSamples_;
            XVals[ss*M+jj] = psOUUMinlpZ3SamInputs_[kk*psOUUMinlpM3_+ind3];
            ind3++;
         }
         else if (psOUUMinlpInputTypes_[jj] == psOUUMinlpType4)
         {
            kk = ss / psOUUMinlpZ4nSamples_;
            XVals[ss*M+jj] = psOUUMinlpZ4SamInputs_[kk*psOUUMinlpM4_+ind4];
            ind4++;
         }
      }
   }

   //**/ if not ensemble evaluation
   if (psOUUMinlpEnsembleEval_ == 0)
   {
      readys = new int[nSamp];
      for (ss = 0; ss < nSamp; ss++)
      {
         readys[ss] = -1;
         funcID = psOUUMinlpCounter_ * nSamp + ss;
         readys[ss] = odata->funcIO_->evaluate(funcID,M,&XVals[ss*M],
                                               nOuts,&YVals[ss*nOuts],0);
         if (psOUUMinlpParallel_ == 0) odata->numFuncEvals_++;
      }
      //**/ if asynchronous mode is on, additional processing is needed
      if (psOUUMinlpParallel_ == 1)
      {
         for (ss = 0; ss < nSamp; ss++)
         {
            funcID = psOUUMinlpCounter_ * nSamp + ss;
            if (readys[ss] != 0)
            {
               while (readys[ss] != 0)
               {
#ifdef WINDOWS
                  Sleep(1000);
#else
                  sleep(1);
#endif
                  readys[ss] = odata->funcIO_->evaluate(funcID,M,&XVals[ss*M],
                                                  nOuts,&YVals[ss*nOuts],2);
                  odata->numFuncEvals_++;
               }
            }
         }
      }
   }
   else
   {
      funcID = odata->numFuncEvals_;
      odata->funcIO_->ensembleEvaluate(nSamp,M,XVals,nOuts,YVals,funcID);
      odata->numFuncEvals_ += nSamp;
   }

   //**/ ------ analyze optimization results ------
   int failCnt=0;
   for (ss = 0; ss < nSamp; ss++)
   {
      for (ii = 0; ii < nOuts; ii++)
      {
         if (YVals[ss*nOuts+ii] >= 0.98*PSUADE_UNDEFINED) 
         {
            failCnt++;
            break;
         }
      }
   }
   if (failCnt != 0)
      printf("WARNING: there are %d failed runs out of %d\n",failCnt,nSamp);
   if (failCnt == nSamp || (failCnt > 0 && psOUUMinlpSamProbs_ != NULL))
   {
      printf("ERROR: Type 3 sample cannot admit failures ==> terminate.\n");
      exit(1);
   }
   if (failCnt > 0 && psOUUMinlpMode_ > 2)
   {
      printf("ERROR: OUU mode %d cannot admit failures ==> terminate.\n",
             psOUUMinlpMode_);
      exit(1);
   }

   //**/ compute stochastic metrics
   double *means = new double[nOuts];
   double *stdev = new double[nOuts];
   double *Ytemp = new double[nSamp];
   int    *Jtemp = new int[nSamp];
   if (psOUUMinlpUseRS_ == 0 || nSamp == 1)
   {
      if (odata->outputLevel_ > 2)
         printf("OUUMinlpOpt: computing objective (no RS), nFuncEval = %d\n",
                odata->numFuncEvals_);
      for (ii = 0; ii < nOuts; ii++) means[ii] = 0.0;
      if (psOUUMinlpMode_ == 1 || psOUUMinlpMode_ == 2)
      {
         for (ii = 0; ii < nOuts; ii++) means[ii] = 0.0;
         if (psOUUMinlpZ3nSamples_ == 1)
         {
            for (ss = 0; ss < nSamp; ss++)
            {
               for (ii = 0; ii < nOuts; ii++)
                  if (YVals[ss*nOuts+ii] > 0.98*PSUADE_UNDEFINED)
                     break;
               if (ii == nOuts)
               {
                  for (ii = 0; ii < nOuts; ii++)
                     means[ii] += YVals[ss*nOuts+ii] / (double) (nSamp-failCnt);
               }
            }
         }
         else
         {
            for (ss = 0; ss < nSamp; ss++)
            {
               index = ss / psOUUMinlpZ4nSamples_;
               for (ii = 0; ii < nOuts; ii++)
                  means[ii] += YVals[ss*nOuts+ii] / psOUUMinlpZ4nSamples_ *
                               psOUUMinlpSamProbs_[index];
            }
         }
         if (psOUUMinlpMode_ == 1 || psOUUMinlpStdevMultiplier_ == 0.0)
         {
            for (ii = 0; ii < nOuts; ii++)
            {
               X.set_bb_output(ii, means[ii]);
               YVals[ii] = means[ii];
            }
         }
      }
      //**/ add sample std dev to sample mean, if it is mode 2
      if (psOUUMinlpMode_ == 2 && psOUUMinlpStdevMultiplier_ != 0.0)
      {
         for (ii = 0; ii < nOuts; ii++) stdev[ii] = 0.0;
         for (ss = 0; ss < nSamp; ss++)
         {
            index = ss / psOUUMinlpZ4nSamples_;
            for (ii = 0; ii < nOuts; ii++)
               if (YVals[ss*nOuts+ii] > 0.98*PSUADE_UNDEFINED)
                  break;
            if (ii == nOuts && psOUUMinlpZ3nSamples_ == 1)
            {
               for (jj = 0; jj < nOuts; jj++)
                  stdev[jj] += pow(YVals[ss*nOuts+jj]-means[jj],2.0)/
                                      (double) (nSamp - failCnt);
            }
            else if (ii == nOuts)
            {
               for (jj = 0; jj < nOuts; jj++)
                  stdev[jj] += pow(YVals[ss*nOuts+jj]-means[jj], 2.0)*
                       psOUUMinlpSamProbs_[index] / (double) psOUUMinlpZ4nSamples_;
            }
         }
         for (ii = 0; ii < nOuts; ii++) 
         {
            ddata = means[ii] + psOUUMinlpStdevMultiplier_ * sqrt(stdev[ii]);
            X.set_bb_output(ii, ddata);
            YVals[ii] = ddata;
         }
      }
      //**/ Mode 3: percentile
      if (psOUUMinlpMode_ == 3)
      {
         for (ii = 0; ii < nSamp; ii++) Ytemp[ii] = YVals[ii*nOuts]; 
         for (ii = 0; ii < nSamp; ii++) Jtemp[ii] = ii;
         sortDbleList2a(nSamp, Ytemp, Jtemp);
         kk = (int) (psOUUMinlpPercentile_ * nSamp);
         if (kk >= nSamp) kk = nSamp - 1;
         index = Jtemp[kk];
         for (ii = 0; ii < nOuts; ii++) 
         {
            ddata = YVals[index*nOuts+ii];
            X.set_bb_output(ii, ddata);
            YVals[ii] = ddata;
         }
      } 
   }
   //**/ use response surface
   else
   {
      if (odata->outputLevel_ > 2)
         printf("OUUMinlpOpt: computing objective with RS, nFuncEval = %d\n",
                   odata->numFuncEvals_);

      int    status;
      double totalMean=0.0, totalStdv=0.0, *resultStore, cverrors[3];
      double mean, stdv;
      resultStore = new double[psOUUMinlpZ3nSamples_];
      //**/ compute mean for each Z3 sample point
      for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++)
      {
          //**/ create response surface
          status = psOUUMinlpfaPtr_->initialize(psOUUMinlpZ4SamInputs_,
                                           &YVals[ii*psOUUMinlpZ4nSamples_]);
          //**/ validate response surface
          if (psOUUMinlpValidateRS_ == 1)
          {
             validate(psOUUMinlpZ4nSamples_,psOUUMinlpZ4SamInputs_,
                      &YVals[ii*psOUUMinlpZ4nSamples_], cverrors);
             printf("OUUMinlpOptimizer: Z3 sample %d (of %d)\n",ii+1,
                      psOUUMinlpZ3nSamples_);
             printf("RS CV avg error = %e (scaled) \n", cverrors[0]);
             printf("RS CV rms error = %e (scaled) \n", cverrors[1]);
             printf("RS CV max error = %e (scaled) \n", cverrors[2]);
          }
          //**/ evaluate large sample using response surface
          psOUUMinlpfaPtr_->evaluatePoint(psOUUMinlpLargeSampleSize_,
                  psOUUMinlpLargeSamInputs_,psOUUMinlpLargeSamOutputs_);
         //**/ compute mean
         if (psOUUMinlpMode_ == 1 || psOUUMinlpMode_ == 2)
         {
            mean = 0.0;
            for (ss = 0; ss < psOUUMinlpLargeSampleSize_; ss++)
               mean += psOUUMinlpLargeSamOutputs_[ss];
            mean /= (double) psOUUMinlpLargeSampleSize_;
            resultStore[ii] = mean;
         }
         //**/ compute std deviation, if it is mode 2
         if (psOUUMinlpMode_ == 2 && psOUUMinlpStdevMultiplier_ != 0.0)
         {
            stdv = 0.0;
            for (ss = 0; ss < psOUUMinlpLargeSampleSize_; ss++)
               stdv += pow(psOUUMinlpLargeSamOutputs_[ss] - mean, 2.0);
            stdv /= (double) psOUUMinlpLargeSampleSize_;
            resultStore[ii] = mean + psOUUMinlpStdevMultiplier_ * stdv;
         }
         //**/ mode 3: percentile
         if (psOUUMinlpMode_ == 3)
         {
            sortDbleList(psOUUMinlpLargeSampleSize_,psOUUMinlpLargeSamOutputs_);
            kk = (int) (psOUUMinlpPercentile_ * psOUUMinlpLargeSampleSize_);
            if (kk >= psOUUMinlpLargeSampleSize_)
                  kk = psOUUMinlpLargeSampleSize_ - 1;
            resultStore[ii] = psOUUMinlpLargeSamOutputs_[kk];
         }
      }
      //**/ compute mean/std dev for all Z3 sample points
      if (psOUUMinlpMode_ == 1 || psOUUMinlpMode_ == 2)
      {
         mean = 0.0;
         for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++)
         {
            if (psOUUMinlpSamProbs_ == NULL)
               mean += resultStore[ii] / (double) psOUUMinlpZ3nSamples_;
            else
               mean += resultStore[ii] * psOUUMinlpSamProbs_[ii];
         }
         X.set_bb_output(0, mean);
         YVals[0] = mean;
      }
      //**/ compute mean percentile for mode 3
      if (psOUUMinlpMode_ == 3)
      {
         mean = 0.0;
         for (ii = 0; ii < psOUUMinlpZ3nSamples_; ii++)
            mean += resultStore[ii] / (double) psOUUMinlpZ3nSamples_;
         X.set_bb_output(0, mean);
         YVals[0] = mean;
      }
      if (odata->outputLevel_ > 2)
      {
         printf("OUUMinlpOptimizer: computed  objective (with RS) = %e.\n",
                YVals[0]);
      }
      delete [] resultStore;
   }
   delete [] means;
   delete [] stdev;
   delete [] Ytemp;
   delete [] Jtemp;

   //**/ If objective function is smaller and constraints are satisfied,
   //**/ store the results
   if (YVals[0] < psOUUMinlpObj_->optimalY_) 
   {
      goodFlag = 1;
      for (ii = 1; ii < nOuts; ii++) 
         if (YVals[ii] > 0) goodFlag = 0;
      if (goodFlag == 1)
      {
         for (ii = 0; ii < MOpt; ii++) 
            psOUUMinlpObj_->optimalX_[ii] = XVals[ii];
         psOUUMinlpObj_->optimalY_ = YVals[0];
      }
      else
      {
         if (psOUUMinlpPrintLevel_ > 1)
         {
            printf("OUUMinlpOptimizer INFO: ignore infeasible solution:\n");
            for (ii = 0; ii < MOpt; ii++) 
               printf("   X[%5d] = %e\n", ii+1, XVals[ii]);
            for (ii = 0; ii < nOuts; ii++) 
               printf("   Y[%5d] = %e\n", ii+1, YVals[ii]);
         }
      }
   }

   //**/ analyze convergence
   funcID = psOUUMinlpCounter_;
   psOUUMinlpObjFcnStore_[funcID] = YVals[0];
   int    converged;
   double diff;
   if (funcID >= 5)
   {
      converged = 1;
      for (ii = funcID; ii > funcID-2; ii--) 
      {
         diff = psOUUMinlpObjFcnStore_[ii] - psOUUMinlpObjFcnStore_[ii-1]; 
         if (diff < 0) diff = -diff;
         if (diff > psOUUMinlpTolerance_) 
         {
            converged = 0;
            break;
         }
      }
      if (converged == 1)
      {
         printf("INFO: convergence detected at iteration %d ==> stop.\n",
                funcID);
         psOUUMinlpStop_ = 1;
      }
   }

   //**/ clean up
   delete [] XVals;
   delete [] YVals;
   fp = fopen("psuade_stop", "r");
   if (fp != NULL)
   {
      printf("**************** ==> \n");
      printf("INFO: PSUADE found psuade_stop (REQUEST TO TERMINATE).\n");
      printf("      Remove this file if this is not what you desire.\n");
      printf("<== **************** \n");
      fclose(fp);
      psOUUMinlpStop_ = 1;
   }
   psOUUMinlpCounter_++;
   return retFlag;
}
#endif

