// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public 
// License.
//
// PSUADE is free software; you can redistribute it and/or modify it under 
// the terms of the GNU General Public License (as published by the Free 
// Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or 
// FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of 
// the GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public 
// License along with this program; if not, write to the Free Software 
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class PsuadeBase
// AUTHOR : CHARLES TONG
// DATE   : 2019
// ************************************************************************
//
// ------------------------------------------------------------------------
// system includes
// ------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <PsuadeCmakeConfig.h>

#ifdef WINDOWS
#include <windows.h>
#endif

// ------------------------------------------------------------------------
// local includes : class definition and utilities
// ------------------------------------------------------------------------
#include "PsuadeBase.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "pData.h"
#include "ProbMatrix.h"
#include "psMatrix.h"
#include "psVector.h"
#include "psStrings.h"
#include "GPBasedSampling.h"
#include "MCMCAnalyzer.h"
#include "FuncApprox.h"
#include "FunctionInterface.h"
#include "Optimizer.h"
#include "Globals.h"

// ************************************************************************
// interpret ODOE command from interactive session
// ------------------------------------------------------------------------
int PsuadeBase::ODOEAnalysis(char *lineIn)
{
  int    ii, jj, kk, ll, ind, status, count, faFlag, iOne=1;
  double ddata;
  char   command[1001], winput[1001], pString[1001], lineIn2[10000];
  FILE   *fp;

  //**/ -------------------------------------------------------------
  // read in command and data
  //**/ -------------------------------------------------------------
  sscanf(lineIn,"%s", command);
  strcpy(winput,"NONE");
  
  //**/ -------------------------------------------------------------
  // +++ odoe_mmd 
  //**/ design of experiment based on max-min distance
  //**/ -------------------------------------------------------------
  if (!strcmp(command, "odoe_mmd"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_mmd: create a space-filling sample by maximizing ");
      printf("the minimum\n");
      printf("          distance (in the input space but can be ");
      printf("tailored for input/\n");
      printf("          output space) between the selected points.\n");
      printf("Syntax: odoe_mmd (no argument needed)\n\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command creates a space-filling sample by maximizing ");
    printf("the minimum\n");
    printf("          distance (in the input space but can be ");
    printf("tailored for input/\n");
    printf("          output space) between the selected points. ");
    printf("The candidate\n");
    printf("          set should be loaded before this command is ");
    printf("called. Several\n");
    printf("          variants are available:\n");
    printf("   a. The simplest variant is the unweighted option, ");
    printf("i.e. all points\n");
    printf("      in the candidate set are given equal weight.\n");
    printf("   b. The weighted option uses one of the outputs to ");
    printf("assign weights\n");
    printf("      to candidate points. If the weights are prediction ");
    printf("uncertainties\n");
    printf("      from evaluating the candidate sample using a ");
    printf("response surface\n");
    printf("      (NOTE: prediction uncertainty may be RS prediction ");
    printf("error or\n");
    printf("      induced by some uncertain inputs) and if the selected ");
    printf("set has\n");
    printf("      size 1, this is equivalent to the W-optimal design which ");
    printf("selects\n");
    printf("      the point with the largest prediction uncertainty). If ");
    printf("the size\n");
    printf("      of the selected set is larger than 1, this command ");
    printf("balances\n");
    printf("      W-optimality and space-fillingness.\n");
    printf("   c. To perform input-output space-filling designs, first ");
    printf("move the\n");
    printf("      desired sample outputs to be sample inputs and then ");
    printf("call this\n");
    printf("      command.\n");
    printf("NOTE: Since this method uses Euclidean distances in the ");
    printf("input space,\n");
    printf("      input scales (input ranges) matter. For example, ");
    printf("if an input has\n");
    printf("      range [0,10], it will be more important than another ");
    printf("input with\n");
    printf("      range [0,1].\n");
    printf("      If it is scaled to [0,infinity], other inputs become ");
    printf("irrelevant.\n");
    printDashes(PL_INFO, 0);
    printf("How this command should be used:\n");
    printf(" - Load a candidate design set with 'load', read_std or ");
    printf("iread (sample\n");
    printf("   outputs are needed for the weighted version only.\n");
    printf(" - Run odoe_mmd and follow instructions.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || nSamples_ <= 0)
    {
      printf("ERROR: The candidate set needs to be loaded first using ");
      printf("the load\n");
      printf("       (or iread or read_std) functions. If the candidate ");
      printf("sample is\n");
      printf("       in PSUADE format, one of the outputs can be used as ");
      printf("weights\n");
      printf("       in the weighted option of this command (the weights ");
      printf("tell the\n");
      printf("       optimizer which candidate points are more important ");
      printf("than the\n");
      printf("       the others.)\n");
      return 1;
    }
    if (nSamples_ == 1)
    {
      printf("INFO: nSamples == 1 so no need to search.\n");
      return 1;
    }

    //**/ ------------------------------------------------------------
    //**/ checking input ranges
    //**/ (first find max and min for each input)
    //**/ ------------------------------------------------------------
    psVector vecXMins, vecXMaxs;
    vecXMins.setLength(nInputs_);
    vecXMaxs.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      vecXMins[ii] = VecSamInputs_[ii];
      vecXMaxs[ii] = VecSamInputs_[ii];
    }
    for (jj = 1; jj < nSamples_; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        if (VecSamInputs_[jj*nInputs_+ii] < vecXMins[ii])
          vecXMins[ii] = VecSamInputs_[jj*nInputs_+ii];
        if (VecSamInputs_[jj*nInputs_+ii] > vecXMaxs[ii])
          vecXMaxs[ii] = VecSamInputs_[jj*nInputs_+ii];
      }
    }
    count = 0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (vecXMins[ii] < 0 || vecXMaxs[ii] > 1 || 
          (vecXMaxs[ii] - vecXMins[ii] < 0.5)) count++;
    }
    psVector vecSamXT, vecILBs, vecIUBs; 
    vecSamXT = VecSamInputs_;
    vecILBs = VecILowerBs_;
    vecIUBs = VecIUpperBs_;
    if (count > 0)
    {
      printf("The input ranges for this sample are:\n");
      for (ii = 0; ii < nInputs_; ii++)
        printf("Input %4d: min, max = %16.8e %16.8e\n",ii+1,
               vecXMins[ii],vecXMaxs[ii]);
      printf("INFO: the inputs have different ranges, which will ");
      printf("skew the selection\n");
      printf("      process (see explanation above).\n\n");
      printf("Here are your options:\n");
      printf("1. Do not proceed\n");
      printf("2. Proceed without changing the input ranges\n");
      printf("3. Re-scale the inputs and then proceed\n");
      snprintf(pString,100,"Your choice (1, 2, or 3): ");
      kk = getInt(1,3,pString);
      if (kk == 1) return 1;
      if (kk == 3)
      {
        //**/ get the scaling factors
        printf("Please enter scaling factor for each input below.\n");
        printf("NOTE: Larger scaling factor means more important.\n");
        printf("E.g. If input 1 has scale=2 and input 2 has scale=1, ");
        printf("the selected\n");
        printf("     set will be more space-filling with respect to ");
        printf("input 2\n");
        printf("      relative to input 1.\n");
        for (ii = 0; ii < nInputs_; ii++)
        {
          snprintf(pString,100,"Enter scaling factor for input %d: ", ii+1);
          vecIUBs[ii] = getDouble(pString);
        }

        //**/ scale
        for (jj = 0; jj < nSamples_; jj++)
        {
          for (ii = 0; ii < nInputs_; ii++)
          {
            vecSamXT[jj*nInputs_+ii] = 
              (VecSamInputs_[jj*nInputs_+ii] - vecXMins[ii]) /
              (vecXMaxs[ii] - vecXMins[ii]) * vecIUBs[ii];
          }
        }
      }
    }

    //**/ ------------------------------------------------------------
    //**/ prompt users for additional information
    //**/ ------------------------------------------------------------
    int hasWeights = 0, outputID = 0;
    printAsterisks(PL_INFO, 0);
    printf("Select one of the following options: \n");
    printf("(1) Unweighted approach (no weight given to each sample)\n");
    printf("(2) Weighted approach (use one of the outputs as weights)\n");
    printf("NOTE: This weight is different from the input scaling ");
    printf("factors which\n");
    printf("      you may have been asked to select. While input ");
    printf("scaling factors\n");
    printf("      are for individual inputs, this weight applies to ");
    printf("all inputs\n");
    printf("      (each candidate point has its own weight rather than ");
    printf("each input\n");
    printf("      has its own weight). Generally, this weight corresponds ");
    printf("to the\n");
    printf("      prediction uncertainty of the candidate point.\n");
    snprintf(pString,100,"Enter your choice (1 or 2): ");
    hasWeights = getInt(1,2,pString);
    if (hasWeights == 2)
    {
      printf("Select which output for the weights.\n");
      printf("NOTE: the selected output should have values in [0,1].\n");
      snprintf(pString,100, 
              "Which output to use as weight (1 - %d): ",nOutputs_);
      outputID = getInt(1, nOutputs_, pString);
      outputID--;
      for (ii = 0; ii < nSamples_; ii++)
        if (VecSamOutputs_[ii*nOutputs_+outputID] <= 0.0)
          break;
      if (ii != nSamples_)
      {
        printf("ERROR: some weights are equal to or less than 0.\n");
        return 1;
      }
    }
    printEquals(PL_INFO, 0);
    printf("Select one of the following search methods (recommended: 3): \n");
    printf(" 1. Brute force method (optimal, but expensive)\n");
    printf(" 2. Global optimization search (maybe suboptimal)\n");
    printf(" 3. Hybrid: global search followed by brute force (optimal)\n");
    printf("    (Use global search to get a suboptimal solution ");
    printf("which is then used\n");
    printf("     to improve brute force search efficiency)\n");
    snprintf(pString,100,"Enter your choice (1, 2, or 3): ");
    int mmdOption = getInt(1,3,pString);

    printEquals(PL_INFO, 0);
    snprintf(pString,100,"How many to select from the candidate set? ");
    int nFinalDesigns = getInt(1, nSamples_-2, pString);

    printEquals(PL_INFO, 0);
    printf("For sequential design, you may have already ");
    printf("selected some points in\n");
    printf("the candidate set, and you may want ");
    printf("the current selection to be also\n");
    printf("space-filling with respect to previous selections. ");
    printf("If this is the\n");
    printf("case, you can append the previously-selected points ");
    printf("at the end of the\n");
    printf("candidate file (e.g. if you have 100 candidate points");
    printf(" and you have 5\n");
    printf("previously-selected points that are not part of the ");
    printf("100-candidate set,\n");
    printf("add these 5 points to the end of the candidate file ");
    printf("so that the\n");
    printf("candidate file now has 105 points.)\n\n");
    printf("The loaded candidate set has %d points.\n",nSamples_);
    snprintf(pString,100,
       "How many of them have been selected previously? (0 if none) ");
    int nPreSelected = getInt(0, nSamples_-nFinalDesigns-1, pString);

    //**/ -----------------------------------------------------
    //**/ clean up before processing (psuade_stop is for 
    //**/ graceful termination)
    //**/ -----------------------------------------------------
    fp = fopen("psuade_stop","r");
    if (fp != NULL)
    {
      fclose(fp);
      unlink("psuade_stop");
    }

    //**/ ------------------------------------------------------------
    //**/ Case 2: optimization search
    //**/ This setup is relevant for both option 2 and 3
    //**/ ------------------------------------------------------------
    int        nStarts, maxIter;
    psMatrix   matOptData;
    PsuadeData *psIO=NULL;
    FunctionInterface *funcIO=NULL;
    if (mmdOption == 2 || mmdOption == 3)
    {
      printEquals(PL_INFO, 0);
      printf("Select global optimization parameters.\n");
      if (mmdOption == 3)
      {
        printf("For hybrid search (option 3), >50 multi-starts and\n");
        printf("5000 maximum iterations are recommended.\n");
      }
      strcpy(pString, 
         "How many starts in a multi-start optimization? (10 - 10000) ");
      nStarts = getInt(10,20000,pString);
      strcpy(pString,
         "Maximum number of optimization iterations per start: (>=1000) ");
      maxIter = getInt(1000,1000000000,pString);

      //**/ initialize the data structure for storing the final design
      //**/ first row: initial guess, second row: initial optimal value
      //**/ third row: final design,  fourth row: final optimal value
      matOptData.setFormat(PS_MAT2D);
      matOptData.setDim(4,nFinalDesigns);
      matOptData.setEntry(1, 0, PSUADE_UNDEFINED);

      //**/ ------------------------------------------------------
      //**/ if single-start, ask for initial guess
      //**/ ------------------------------------------------------
      if (nStarts == 1)
      {
        printEquals(PL_INFO, 0);
        printf("Please provide a good selection set as the initial\n");
        printf("guess to the global optimizer, if you have any.\n"); 
        printf("An initial guess is to be provided? (y or n) ");
        scanf("%s",pString); 
        fgets(winput, 500, stdin);
        if (pString[0] == 'y')
        {
          for (ii = 0; ii < nFinalDesigns; ii++)
          {
            snprintf(pString,100,
              "Enter initial guess for input %d (1 - %d) : ",
              ii+1, nSamples_);
            ddata = getDouble(pString);
            if (ddata < 1 || ddata > nSamples_)
            {
              printf("Wrong input: default to 0\n");
              ddata = 1;
            }
            matOptData.setEntry(0, ii, ddata);
          }
        }
        else
        {
          for (jj = 0; jj < nFinalDesigns; jj++)
          {
            ddata = (PSUADE_rand() % (nSamples_-nPreSelected)) + 1;
            if (ddata < 1) ddata = 1;
            if (ddata > (nSamples_-nPreSelected))
              ddata = nSamples_ - nPreSelected;
            matOptData.setEntry(0, jj, ddata);
          }
        }
      }

      //**/ -------------------------------------------------------
      //**/ set up PsuadeData object for passing to OptimizerSearch
      //**/ each independent variable in the optimization problem
      //**/ corresponds to the index of a candidate point so the
      //**/ range should be from 1 to the number of candidate points
      //**/ -------------------------------------------------------
      psVector vLBs, vUBs;
      vLBs.setLength(nFinalDesigns);
      vUBs.setLength(nFinalDesigns);
      for (ii = 0; ii < nFinalDesigns; ii++)
      {
        vLBs[ii] = 1;
        vUBs[ii] = nSamples_ - nPreSelected;
      }
      psStrings tmpStrs;
      tmpStrs.setNumStrings(nFinalDesigns);
      for (ii = 0; ii < nFinalDesigns; ii++) 
      {
        snprintf(pString,100,"X%d", ii+1);
        tmpStrs.loadOneString(ii, pString);
      }
      psIO = new PsuadeData();
      psIO->updateInputSection(0,nFinalDesigns,NULL,vLBs.getDVector(),
                vUBs.getDVector(),NULL,tmpStrs.getStrings(),NULL,
                NULL,NULL,NULL);
      strcpy(tmpStrs[0], "Y");
      psIO->updateOutputSection(0, iOne, NULL, NULL, tmpStrs.getStrings());
      strcpy(pString, "NONE");
      tmpStrs.loadOneString(0, pString);
      psIO->updateApplicationSection(NULL,tmpStrs[0],NULL,NULL,NULL,-1);
      int optMethod = 9;
      psIO->updateOptimizationSection(optMethod,iOne,1.0e-4,maxIter,-1);

      //**/ -----------------------------------------------------
      //**/ set up function interface for passing to OptimizerSearch
      //**/ -----------------------------------------------------
      funcIO = new FunctionInterface();
      funcIO->loadInputData(nFinalDesigns, NULL);
      funcIO->loadOutputData(iOne, NULL);
      strcpy(pString, "NONE");
      tmpStrs.setNumStrings(5);
      tmpStrs.loadOneString(1, pString);
      tmpStrs.loadOneString(2, pString);
      tmpStrs.loadOneString(4, pString);
      strcpy(pString, "PSUADE_LOCAL");
      tmpStrs.loadOneString(0, pString);
      tmpStrs.loadOneString(3, pString);
      funcIO->loadFunctionData(5, tmpStrs.getStrings());
      funcIO->setLocalFunction(30);
      if (mmdOption == 2) funcIO->setOutputLevel(outputLevel_);
  
      //**/ -------------------------------------------------------
      //**/ initialize other optimization parameters via psConfig
      //**/ -------------------------------------------------------
      for (ii = 0; ii < nFinalDesigns; ii++)
      {
        snprintf(winput,100,"iDiscrete%d", ii+1);
        psConfig_.putParameter(winput);
      }
    }

    //**/ ---------------------------------------------------------
    //**/ This display is relevant for option 2 only
    //**/ Option 3 is automated so this question is not asked
    //**/ ---------------------------------------------------------
    if (mmdOption == 2)
    {
      printf("This search may take a very long time. To terminate\n");
      printf("the search, simply create a file called psuade_stop\n");
      printf("in this working directory.\n");
      printf("When you are ready to move on, enter RETURN\n");
      scanf("%c", winput);
    }

    //**/ -------------------------------------------------------
    //**/ copy the matrix and weights to a global place so the 
    //**/ local function in FunctionInterface can access it. Also 
    //**/ initialize nPreSelected in psConfig. Afterward, perform
    //**/ search
    //**/ Note: This is needed for both option 2 and 3
    //**/ -------------------------------------------------------
    double bestY=PSUADE_UNDEFINED;
    psIVector vecBestX;
    if (mmdOption == 2 || mmdOption == 3)
    {
      psConfig_.MatCommonUse_.setDim(nSamples_, nInputs_);
      for (ii = 0; ii < nSamples_; ii++)
        for (jj = 0; jj < nInputs_; jj++)
          psConfig_.MatCommonUse_.setEntry(ii,jj,
                        vecSamXT[ii*nInputs_+jj]);
      psConfig_.intCommonUse_ = nPreSelected;
      psConfig_.VecCommonUse_.clean();
      if (hasWeights == 2)
      {
        psConfig_.VecCommonUse_.setLength(nSamples_);
        for (ii = 0; ii < nSamples_; ii++)
          psConfig_.VecCommonUse_[ii] = 
            VecSamOutputs_[ii*nOutputs_+outputID];
      }   

      //**/ -----------------------------------------------------
      //**/ search
      //**/ -----------------------------------------------------
      int nInitX=1, loopCnt=0;
      vecBestX.setLength(nFinalDesigns);
      for (ii = 0; ii < nStarts; ii++)
      {
        //**/ if multi-start, random initial guess
        loopCnt++;
        printAsterisks(PL_INFO, 0);
        if (nStarts > 1)
        {
          for (jj = 0; jj < nFinalDesigns; jj++) 
          {
            ddata = (PSUADE_rand() % (nSamples_-nPreSelected)) + 1;
            if (ddata < 1) ddata = 1;
            if (ddata > (nSamples_-nPreSelected)) 
              ddata = nSamples_ - nPreSelected;
            matOptData.setEntry(0, jj, ddata);
          }
        }
        printf("Optimization multi-start %d initial guess: \n",ii+1);
        for (jj = 0; jj < nFinalDesigns; jj++) 
        {
          printf("%5d ", int(matOptData.getEntry(0,jj)));
          if (jj > 0 && (jj % 10) == 0) printf("\n");
        }
        printf("\n");
        psConfig_.InteractiveSaveAndReset();
        status = OptimizerSearch(psIO,funcIO,matOptData,nInitX);
        psConfig_.InteractiveRestore();
        ddata = matOptData.getEntry(3,0);
        if (ddata < bestY)
        {
          bestY = ddata;
          for (jj = 0; jj < nFinalDesigns; jj++) 
            vecBestX[jj] = (int) matOptData.getEntry(2,jj);
          loopCnt = 0;
        }
        printf("odoe_mmd Current best X: \n");
        for (jj = 0; jj < nFinalDesigns; jj++) 
        {
          if (jj > 0 && (jj % 10) == 0) printf("\n");
          printf("%5d ",(int)vecBestX[jj]);
        }
        printf("\n");
        printf(" ===> max-min distance = %e\n",-bestY);
        if (loopCnt > 0 && (loopCnt % 20) == 0) 
          printf("INFO: %d restarts without progress so far.\n",
                 loopCnt);
        printAsterisks(PL_INFO, 0);
        if ((fp = fopen("psuade_stop","r")) != NULL)
        {
          fclose(fp);
          printf("psuade_stop detected ==> terminate\n");
          break;
        }
      }

      //**/ -----------------------------------------------------
      //**/ clean up
      //**/ -----------------------------------------------------
      for (ii = 0; ii < nFinalDesigns; ii++)
      {
        snprintf(winput,100,"iDiscrete%d", ii+1);
        psConfig_.removeParameter(winput);
      }
      psConfig_.MatCommonUse_.clean();
      delete psIO;
      delete funcIO;
      printAsterisks(PL_INFO, 0);
    }

    //**/ -------------------------------------------------------
    //**/ display result for option 2 and clean up
    //**/ -------------------------------------------------------
    if (mmdOption == 2)
    {
      printf("* odoe_mmd Final best X: \n");
      for (jj = 0; jj < nFinalDesigns; jj++) 
      {
        printf("%5d ",(int) vecBestX[jj]);
        if (jj > 0 && (jj % 10) == 0) printf("\n");
      }
      printf("\n");
      printf("* ===> max-min (max among all min) distance = %e\n",
             -bestY);
      printAsterisks(PL_INFO, 0);
      fp = fopen("matlabodoe_mmd.m", "w");
      fprintf(fp, "A = [\n");
      for (ii = 0; ii < nFinalDesigns; ii++)
      {
        jj = (int) vecBestX[ii];
        jj--;
        for (kk = 0; kk < nInputs_; kk++)
          fprintf(fp,"%e ",VecSamInputs_[jj*nInputs_+kk]);
        fprintf(fp,"\n");
      }
      fprintf(fp, "];\n");
      fclose(fp);
      printf("odoe_mmd: final selection in matlabodoe_mmd.m\n");
      return 0;
    }

    //**/ ------------------------------------------------------------
    //**/ Case 1 and 3: use brute force search
    //**/ ------------------------------------------------------------
    long   numIter=0;
    double maxMinVal=0;
    psIVector vecIndMap, vecBestInds;
    psMatrix  distMatrix, candMatrix;

    if (mmdOption == 1)
    {
      //**/ -----------------------------------------------------
      //**/ get best so far
      //**/ -----------------------------------------------------
      psIVector vecBinaries;
      vecBinaries.setLength(nFinalDesigns);
      vecBestInds.setLength(nSamples_-nPreSelected);
      printf("Do you have a good design as initial guess? (y or n) ");
      scanf("%s", pString);
      fgets(winput, 500, stdin);
      
      if (pString[0] == 'y')
      {
        for (ii = 1; ii <= nFinalDesigns; ii++)
        {
          snprintf(pString,100,
           "Enter the design (ascending sample index) %d: ",ii);
          kk = getInt(1, nSamples_-nPreSelected, pString);
          while (ii > 1 && (kk <= vecBinaries[ii-1]))
          {
            printf("ERROR: indices should be ascending.\n");
            kk = getInt(1, nSamples_, pString);
          }
          vecBinaries[ii-1] = kk;
        }
        for (ii = 0; ii < nFinalDesigns; ii++) 
          vecBestInds[vecBinaries[ii]-1] = 1;
      }
    }
    else if (mmdOption == 3)
    {
      vecBestInds.setLength(nSamples_-nPreSelected);
      for (jj = 0; jj < nFinalDesigns; jj++) 
      {
        kk = ((int) vecBestX[jj]) - 1;
        vecBestInds[kk] = 1;
      }
    }

    //**/ --------------------------------------------------------
    //**/ vecIndMap[kk] = -1 means kk-th point previously selected
    //**/ This setup is needed for both option 1 and 3
    //**/ --------------------------------------------------------
    int    icnt, jcnt;
    double dtmp;
    vecIndMap.setLength(nSamples_);
    for (ii = 0; ii < nPreSelected; ii++) 
      vecIndMap[nSamples_-ii-1] = 1;
    icnt = 0;
    for (ii = 0; ii < nSamples_; ii++)
    {
      if (vecIndMap[ii] > 0) vecIndMap[ii] = - vecIndMap[ii];
      else                   vecIndMap[ii] = icnt++;
    } 

    //**/ compute the distance matrix for all candidate points
    distMatrix.setDim(nSamples_, nSamples_);
    for (ii = 0; ii < nSamples_; ii++)
    {
      for (jj = 0; jj < nSamples_; jj++)
      {
        ddata = 0.0;
        for (kk = 0; kk < nInputs_; kk++)
        {
          dtmp = (vecSamXT[ii*nInputs_+kk] -
                  vecSamXT[jj*nInputs_+kk]);
          ddata += dtmp * dtmp;
        }
        if (hasWeights == 2)
          ddata *= (VecSamOutputs_[ii*nOutputs_+outputID] *
                    VecSamOutputs_[jj*nOutputs_+outputID]);
        ddata = sqrt(ddata);
        distMatrix.setEntry(ii, jj, ddata);
      }
    }

    //**/ -----------------------------------------------------
    //**/ compute distance matrix for the unselected points
    //**/ -----------------------------------------------------
    if (nPreSelected == 0) candMatrix = distMatrix;
    else
    {
      candMatrix.setDim(nSamples_-nPreSelected,
                        nSamples_-nPreSelected);
      icnt = 0;
      for (ii = 0; ii < nSamples_; ii++)
      {
        if (vecIndMap[ii] == 0)
        {
          jcnt = 0;
          for (jj = 0; jj < nSamples_; jj++)
          {
            if (vecIndMap[jj] == 0)
            {
              ddata = 0.0;
              for (kk = 0; kk < nInputs_; kk++)
              {
                dtmp = (VecSamInputs_[ii*nInputs_+kk] -
                        VecSamInputs_[jj*nInputs_+kk]);
                ddata += dtmp * dtmp;
              }
              if (hasWeights == 2)
                ddata *= (VecSamOutputs_[ii*nOutputs_+outputID] *
                          VecSamOutputs_[jj*nOutputs_+outputID]);
              ddata = sqrt(ddata);
              candMatrix.setEntry(icnt, jcnt, ddata);
              jcnt++;
            }
          }
          icnt++;
        }
      }
    }

    //**/ -------------------------------------------------------
    //**/ This actual search is relevant to option 1 only
    //**/ -------------------------------------------------------
    if (mmdOption == 1)
    {
      printf("This search may take a very long time. To terminate\n");
      printf("the search, simply create a file called psuade_stop\n");
      printf("in this working directory. To see how far the search\n");
      printf("has advanced, create a file called psuade_print.\n");
      printf("When you are ready to move on, enter RETURN\n");
      scanf("%c", winput);
    }

    //**/ -------------------------------------------------------
    //**/ vecIndTest is just a vector for passing information for
    //**/ recursion in odoeUsingMMDSlow
    //**/ row ii of distMatrix maps to vecIndMap[ii] of candMatrix
    //**/ -------------------------------------------------------
    psIVector vecIndTest;
#if 0
    //**/ Jan 2020 - obsolete: slow search version
    vecIndTest.setLength(nSamples_-nPreSelected);
    odoeUsingMMDSlow(vecIndTest,0,nFinalDesigns,nFinalDesigns,
           candMatrix,distMatrix,vecIndMap,vecBestInds,maxMinVal, 
           numIter,outputLevel_);
#else
    odoeUsingMMD(nFinalDesigns, candMatrix, distMatrix, vecIndMap, 
                 vecBestInds, maxMinVal, numIter, outputLevel_);

#endif
    vecIndTest = vecBestInds;
    vecBestInds.setLength(nSamples_);
    for (ii = 0; ii < nSamples_; ii++)
    {
      if (vecIndMap[ii] >= 0)
        vecBestInds[ii] = vecIndTest[vecIndMap[ii]];
    }

    //**/ -----------------------------------------------------
    //**/ display results
    //**/ -----------------------------------------------------
    printEquals(PL_INFO, 0);
    printf("Final design selection (%d): \n",nFinalDesigns);
    jj = 0;
    for (ii = 0; ii < nSamples_; ii++)
    {
      if (vecBestInds[ii] == 1)
      {
        printf("%4d ", ii+1);
        jj++;
        if (jj >= 10)
        {
          jj = 0;
          printf("\n");
        }
      }
    }
    printf("\n");
    printf("Total number of evaluations = %ld\n", numIter);
    printf("max min distance = %e\n", maxMinVal);
    printEquals(PL_INFO, 0);

    //**/ -----------------------------------------------------
    //**/ also write the results to a Matlab file
    //**/ -----------------------------------------------------
    fp = fopen("matlabodoe_mmd.m", "w");
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nSamples_; ii++)
    {
      if (vecBestInds[ii] == 1)
      {
        for (kk = 0; kk < nInputs_; kk++)
          fprintf(fp,"%e ",VecSamInputs_[ii*nInputs_+kk]);
        fprintf(fp,"\n");
      }
    }
    fprintf(fp, "];\n");
    fclose(fp);
    printf("odoe_mmd: final selection in matlabodoe_mmd.m\n");
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_mmv and odoe_mav - optimal design of experiment 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_mmv") ||
           !strcmp(command, "odoe_mav"))
  {
    int whichScheme=0;
    if (!strcmp(command, "odoe_mav")) whichScheme = 1;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (whichScheme == 0)
      {
        printf("odoe_mmv: generate GP-based min-max variance designs.\n");
        printf("          (This is analogous to the G-optimal design ");
      }
      else
      {
        printf("odoe_mmv: generate GP-based min-avg variance designs.\n");
        printf("          (This is analogous to the I-optimal design ");
      }
      printf("but for UNCERTAINTY\n");
      printf("           THAT COMES FROM RS ONLY - NOT FROM ");
      printf("uncertainty inputs.)\n");
      if (whichScheme == 0)
        printf("Syntax: odoe_mmv (no argument needed)\n\n");
      else
        printf("Syntax: odoe_mav (no argument needed)\n\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command finds an optimal candidate design subset ");
    printf("based on the\n");
    printf("given GP hyperparameter values, or, the GP ");
    printf("hyperparameters are to be\n");
    printf("computed from the loaded sample.\n");
    printf("This command ASSUMES UNCERTAINTY FROM RS ONLY so ");
    printf("all parameters in\n");
    printf("the candidate set (and in the GP training sample, ");
    printf("if it is used), are\n");
    printf("assumed to be design parameters. This method searches ");
    printf("through subsets\n");
    if (whichScheme == 0)
      printf("of designs using GP to estimate the MAX prediction ");
    else
      printf("of designs using GP to estimate the AVG prediction ");
    printf("variances at the\n");
    printf("unselected candidate set. The one subset that minimizes ");
    if (whichScheme == 0) printf("the MAXIMUM\n");
    else                  printf("the AVERAGE\n");
    printf("variances is the final selected set.\n");
    printf("NOTE: This method assesses prediction variances rather ");
    printf("than maximizes\n");
    printf("      minimum distances between the selected points.\n");
    if (!strcmp(winput, "-h")) return 0;

    //**/ -----------------------------------------------------
    //**/ Check to make sure the candidate set has been loaded
    //**/ -----------------------------------------------------
    if (nSamples_ <= 0)
    {
      printf("ERROR: The candidate set has not been loaded yet.\n");
      printf("       You must first load a candidate set using load.\n");
      return 1;
    }
    else
    {
      printf("INFO: The candidate set has been loaded.\n");
      printf("      There are %d candidates each with %d inputs.\n",
             nSamples_,nInputs_);
    }

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ -----------------------------------------------------
    //*(*/ ask for how many to select
    //**/ -----------------------------------------------------
    printEquals(PL_INFO, 0);
    int nCandidates = nSamples_;
    snprintf(pString,100,"How many to select from the candidate set? ");
    int nToBeSelected = getInt(1, nCandidates-2, pString);

    //**/ -----------------------------------------------------
    //**/ offer two options for search
    //**/ -----------------------------------------------------
    printEquals(PL_INFO, 0);
    printf("Select one of the following search methods: \n");
    printf(" 1. Brute force method (optimal, but expensive)\n");
    printf(" 2. Global optimization search (maybe suboptimal)\n");
    printf("Recommendation: If candidate set is large or number ");
    printf("to be selected\n");
    printf("                is large, use global search.\n");
    snprintf(pString,100,"Enter your choice (1 or 2): ");
    int mmvOption = getInt(1,3,pString);

    //**/ -----------------------------------------------------
    //**/ This is being passed into GPBasedSampling initialize
    //**/ function to be used as candidate set
    //**/ -----------------------------------------------------
    psConfig_.intCommonUse_ = nToBeSelected;
    psConfig_.MatCommonUse_.setDim(nSamples_, nInputs_);
    for (ii = 0; ii < nSamples_; ii++)
      for (jj = 0; jj < nInputs_; jj++)
        psConfig_.MatCommonUse_.setEntry(ii,jj,
                      VecSamInputs_[ii*nInputs_+jj]);

    //**/ -----------------------------------------------------
    //**/ clean up before processing (psuade_stop is for
    //**/ graceful termination)
    //**/ -----------------------------------------------------
    fp = fopen("psuade_stop","r");
    if (fp != NULL)
    {
      fclose(fp);
      unlink("psuade_stop");
    }

    //**/ -----------------------------------------------------
    //**/ brute force option: use GPBasedSampling's internal
    //**/ exhaustive search
    //**/ -----------------------------------------------------
    if (mmvOption == 1)
    {
      //**/ -----------------------------------------------------
      //**/ create GPBasedSampling, load the training sample and
      //**/ call initialize to perform optimization
      //**/ (If no sample has been loaded, GPBasedSampling will
      //**/ ask for hyperparameters). Nevertheless, the input
      //**/ information still have to be loaded first here
      //**/ (such as number of inputs and input bounds)
      //**/ -----------------------------------------------------
      GPBasedSampling *sampPtr = new GPBasedSampling();
      if (whichScheme == 0) sampPtr->setMMV();
      else                  sampPtr->setMAV();
      if (whichScheme == 0) 
        printf("To run MMV, PSUADE needs to build a Gaussian ");
      else
        printf("To run MAV, PSUADE needs to build a Gaussian ");
      printf("Process either from a\n");
      printf("training sample or from a vector of user-provided ");
      printf("hyperparameters.\n");
      snprintf(pString,100,"Use a training set ? (y or n) ");
      getString(pString, lineIn);
      if (lineIn[0] == 'y')
      {
        printf("This training sample must be in PSUADE format.\n");
        snprintf(pString,100,"Name of the training sample ? ");
        char fname[1000];
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        PsuadeData *psIO = new PsuadeData();
        pData pdata;
        int status = psIO->readPsuadeFile(fname);
        if (status != 0)
        {
          printf("ODOE ERROR when reading training sample\n");
          printf("     Maybe this file is in wrong format?\n");
          exit(1);
        }
        psIO->getParameter("input_ninputs", pdata);
        int nInps = pdata.intData_;
        psIO->getParameter("output_noutputs", pdata);
        int nOuts = pdata.intData_;
        if (nOuts > 1)
        {
          printf("ODOE ERROR: this method only works with nOutputs=1\n");
          printf("Suggestion: delete all but one output and re-run.\n");
          exit(1);
        }
        pData pInps, pOuts, pLBs, pUBs, pStas;
        psIO->getParameter("method_nsamples", pdata);
        int nSamp = pdata.intData_;
        psIO->getParameter("input_lbounds", pLBs);
        psIO->getParameter("input_ubounds", pUBs);
        psIO->getParameter("input_sample", pInps);
        psIO->getParameter("output_sample", pOuts);
        psIO->getParameter("output_states", pStas);
        sampPtr->setInputBounds(nInps, pLBs.dbleArray_,
                                pUBs.dbleArray_);
        sampPtr->setOutputParams(1);
        sampPtr->setSamplingParams(nSamp, -1, -1);
        sampPtr->loadSamples(nSamp,nInps,nOuts,pInps.dbleArray_,
                             pOuts.dbleArray_, pStas.intArray_);
        sampPtr->setPrintLevel(outputLevel_);
        sampPtr->initialize(0);
        delete sampPtr;
        delete psIO;
      }
      else sampPtr->initialize(0);
    }
    else
    {
      //**/ -----------------------------------------------------
      //**/ This section uses SCE for optimization
      //**/ -----------------------------------------------------

      //**/ ------------------------------------------------------------
      //**/ This setup is relevant for both option 2 and 3
      //**/ (set up matOptData and funcIO and psIO)
      //**/ psIO - to pass optimiation variables to the optimizer
      //**/ matOptData - to be passed to optimizer
      //**/ funcIO - to be passed to optimizer 
      //**/ ------------------------------------------------------------
      int        nStarts, maxIter;
      psMatrix   matOptData;
      PsuadeData *psIO=NULL;
      FunctionInterface *funcIO=NULL;
      printEquals(PL_INFO, 0);
      strcpy(pString,
         "How many starts in a multi-start optimization? (10 - 10000) ");
      nStarts = getInt(10,20000,pString);
      strcpy(pString,
         "Maximum number of optimization iterations per start: (>=1000) ");
      maxIter = getInt(1000,1000000000,pString);

      //**/ initialize the data structure for storing the final design
      //**/ first row: initial guess, second row: initial optimal value
      //**/ third row: final design,  fourth row: final optimal value
      matOptData.setFormat(PS_MAT2D);
      matOptData.setDim(4,nToBeSelected);
      matOptData.setEntry(1, 0, PSUADE_UNDEFINED);

      //**/ ------------------------------------------------------
      //**/ if single-start, ask for initial guess (matOptData)
      //**/ ------------------------------------------------------
      if (nStarts == 1)
      {
        printEquals(PL_INFO, 0);
        printf("Please provide a good selection set as the initial\n");
        printf("guess to the global optimizer, if you have any.\n");
        printf("An initial guess is to be provided? (y or n) ");
        scanf("%s",pString);
        fgets(winput, 500, stdin);
        if (pString[0] == 'y')
        {
          for (ii = 0; ii < nToBeSelected; ii++)
          {
            snprintf(pString,100,
              "Enter initial guess for input %d (1 - %d) : ",
              ii+1, nCandidates);
            ddata = getDouble(pString);
            if (ddata < 1 || ddata > nCandidates)
            {
              printf("Wrong input: default to 0\n");
              ddata = 1;
            }
            matOptData.setEntry(0, ii, ddata);
          }
        }
        else
        {
          for (jj = 0; jj < nToBeSelected; jj++)
          {
            ddata = (PSUADE_rand() % nCandidates) + 1;
            if (ddata < 1) ddata = 1;
            if (ddata > nCandidates) ddata = nCandidates;
            matOptData.setEntry(0, jj, ddata);
          }
        }
      }

      //**/ -------------------------------------------------------
      //**/ set up PsuadeData object for passing to OptimizerSearch
      //**/ each independent variable in the optimization problem
      //**/ corresponds to the index of a candidate point so the
      //**/ range should be from 1 to the number of candidate points
      //**/ -------------------------------------------------------
      psVector vLBs, vUBs;
      vLBs.setLength(nToBeSelected);
      vUBs.setLength(nToBeSelected);
      for (ii = 0; ii < nToBeSelected; ii++)
      {
        vLBs[ii] = 1;
        vUBs[ii] = nCandidates;
      }
      psStrings tmpStrs;
      tmpStrs.setNumStrings(nToBeSelected);
      for (ii = 0; ii < nToBeSelected; ii++)
      {
        snprintf(pString,100,"X%d", ii+1);
        tmpStrs.loadOneString(ii, pString);
      }
      psIO = new PsuadeData();
      psIO->updateInputSection(0,nToBeSelected,NULL,vLBs.getDVector(),
                vUBs.getDVector(),NULL,tmpStrs.getStrings(),NULL,
                NULL,NULL,NULL);
      strcpy(tmpStrs[0], "Y");
      psIO->updateOutputSection(0, iOne, NULL, NULL, tmpStrs.getStrings());
      strcpy(pString, "NONE");
      tmpStrs.loadOneString(0, pString);
      psIO->updateApplicationSection(NULL,tmpStrs[0],NULL,NULL,NULL,-1);
      int optMethod = 9; /* SCE */
      psIO->updateOptimizationSection(optMethod,iOne,1.0e-4,maxIter,-1);

      //**/ -----------------------------------------------------
      //**/ set up function interface for passing to OptimizerSearch
      //**/ -----------------------------------------------------
      funcIO = new FunctionInterface();
      funcIO->loadInputData(nToBeSelected, NULL);
      funcIO->loadOutputData(iOne, NULL);
      strcpy(pString, "NONE");
      tmpStrs.setNumStrings(5);
      tmpStrs.loadOneString(1, pString);
      tmpStrs.loadOneString(2, pString);
      tmpStrs.loadOneString(4, pString);
      strcpy(pString, "PSUADE_LOCAL");
      tmpStrs.loadOneString(0, pString);
      tmpStrs.loadOneString(3, pString);
      funcIO->loadFunctionData(5, tmpStrs.getStrings());
      if (whichScheme == 0) funcIO->setLocalFunction(31);
      else                  funcIO->setLocalFunction(32);
      funcIO->setOutputLevel(outputLevel_);

      //**/ -------------------------------------------------------
      //**/ initialize other optimization parameters via psConfig
      //**/ -------------------------------------------------------
      for (ii = 0; ii < nToBeSelected; ii++)
      {
        snprintf(winput,100,"iDiscrete%d", ii+1);
        psConfig_.putParameter(winput);
      }

      //**/ ---------------------------------------------------------
      //**/ This display is relevant for option 2 only
      //**/ Option 3 is automated so this question is not asked
      //**/ ---------------------------------------------------------
      printf("This search may take a very long time. To terminate\n");
      printf("the search, simply create a file called psuade_stop\n");
      printf("in this working directory.\n");
      printf("When you are ready to move on, enter RETURN\n");
      scanf("%c", winput);

      //**/ -----------------------------------------------------
      //**/ search
      //**/ -----------------------------------------------------
      int    nInitX=1, loopCnt=0;
      double bestY=PSUADE_UNDEFINED;
      psIVector vecBestX;
      vecBestX.setLength(nToBeSelected);
      for (ii = 0; ii < nStarts; ii++)
      {
        //**/ if multi-start, random initial guess
        loopCnt++;
        printAsterisks(PL_INFO, 0);
        if (nStarts > 1)
        {
          for (jj = 0; jj < nToBeSelected; jj++)
          {
            ddata = (PSUADE_rand() % nCandidates) + 1;
            if (ddata < 1) ddata = 1;
            if (ddata > nCandidates)
              ddata = nCandidates;
            matOptData.setEntry(0, jj, ddata);
          }
        }
        printf("Optimization multi-start %d initial guess: \n",ii+1);
        for (jj = 0; jj < nToBeSelected; jj++)
        {
          printf("%5d ", int(matOptData.getEntry(0,jj)));
          if (jj > 0 && (jj % 10) == 0) printf("\n");
        }
        printf("\n");
        psConfig_.InteractiveSaveAndReset();
        status = OptimizerSearch(psIO,funcIO,matOptData,nInitX);
        psConfig_.InteractiveRestore();
        ddata = matOptData.getEntry(3,0);
        if (ddata < bestY)
        {
          bestY = ddata;
          for (jj = 0; jj < nToBeSelected; jj++)
            vecBestX[jj] = (int) matOptData.getEntry(2,jj);
          loopCnt = 0;
        }
        if (whichScheme == 0) printf("odoe_mmv current best X: \n");
        else                  printf("odoe_mav current best X: \n");
        for (jj = 0; jj < nToBeSelected; jj++)
        {
          if (jj > 0 && (jj % 10) == 0) printf("\n");
          printf("%5d ",(int)vecBestX[jj]);
        }
        printf("\n");
        if (whichScheme == 0) 
          printf(" ===> max reduction in max variance = %e\n",
                 -bestY);
        else
          printf(" ===> max reduction in avg variance = %e\n",
                 -bestY);
        if (loopCnt > 0 && (loopCnt % 20) == 0)
          printf("INFO: %d restarts without progress so far.\n",
                 loopCnt);
        printAsterisks(PL_INFO, 0);
        if ((fp = fopen("psuade_stop","r")) != NULL)
        {
          fclose(fp);
          printf("psuade_stop detected ==> terminate\n");
          break;
        }
      }

      psVector vecSelected, vecQuality;
      vecSelected.setLength(nToBeSelected*nInputs_);
      vecQuality.setLength(3);
      for (ii = 0; ii < nToBeSelected; ii++)
      {
        kk = (int) vecBestX[ii];
        kk--;
        for (jj = 0; jj < nInputs_; jj++)
          vecSelected[ii*nInputs_+jj] = VecSamInputs_[kk*nInputs_+jj];
      }
      
#if 0
      //**/ 3/2022: no need to do sample quality check for MMV
      printDashes(PL_INFO, 0);
      printf("Sample quality after initial design\n");
      SamplingQuality(nToBeSelected,nInputs_,vecSelected.getDVector(),
                      VecILowerBs_.getDVector(),VecIUpperBs_.getDVector(),
                      vecQuality.getDVector(),0);
      printf("Min-min distance (sample-sample ) = %10.4e (large is good)\n",
             vecQuality[0]);
      printf("Avg-avg distance (sample pairs  ) = %10.4e (large is good)\n",
             vecQuality[2]);
      printf("Avg-min distance (corner-samples) = %10.4e (small is good)\n",
             vecQuality[1]);
#endif

      //**/ -----------------------------------------------------
      //**/ clean up
      //**/ -----------------------------------------------------
      for (ii = 0; ii < nToBeSelected; ii++)
      {
        snprintf(winput,100,"iDiscrete%d", ii+1);
        psConfig_.removeParameter(winput);
      }
      psConfig_.MatCommonUse_.clean();
      delete psIO;
      delete funcIO;
      printAsterisks(PL_INFO, 0);

      //**/ -------------------------------------------------------
      //**/ display result and clean up
      //**/ -------------------------------------------------------
      if (whichScheme == 0) printf("* odoe_mmv final best X: \n");
      else                  printf("* odoe_mav final best X: \n");
      for (jj = 0; jj < nToBeSelected; jj++)
      {
        printf("%5d ",(int) vecBestX[jj]);
        if (jj > 0 && (jj % 10) == 0) printf("\n");
      }
      printf("\n");
      if (whichScheme == 0) 
        printf("* ===> Final max reduction in max variance = %e\n",
               -bestY);
      else
        printf("* ===> Final max reduction in avg variance = %e\n",
               -bestY);
      printAsterisks(PL_INFO, 0);
      if (whichScheme == 0) fp = fopen("mmv_result", "w");
      else                  fp = fopen("mav_result", "w");
      fprintf(fp,"# <numSamples> <numInputs>\n");
      fprintf(fp,"# <Candidate number> <input values> ...\n");
      fprintf(fp,"%d %d 0\n",nToBeSelected,nInputs_);
      for (ii = 0; ii < nToBeSelected; ii++)
      { 
        fprintf(fp,"%d ", (int) vecBestX[ii]);
        for (jj = 0; jj < nInputs_; jj++)
          fprintf(fp,"%12.4e ",vecSelected[ii*nInputs_+jj]);
        fprintf(fp,"\n");
      }
      fclose(fp);
      if (whichScheme == 0)
        printf("INFO: best designs are in mmv_result\n");
      else
        printf("INFO: best designs are in mav_result\n");
      printf("NOTE: You can use 'read_std' to read in the final designs.\n");
    }
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_mav - optimal design of experiment 
  //**/ March 2022 - this has been merged to odoe_mmv above
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_mav_old"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_mmv: generate GP-based min-average variance designs.\n");
      printf("          (This is analogous to the I-optimal design ");
      printf("for UNCERTAINTY\n");
      printf("           THAT COMES FROM RS ONLY - not from ");
      printf("uncertainty inputs.)\n");
      printf("Syntax: odoe_mav (no argument needed)\n\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command finds an optimal candidate design subset ");
    printf("based on the\n");
    printf("given GP hyperparameter values, or, the GP ");
    printf("hyperparameters are to be\n");
    printf("computed from the loaded sample.\n");
    printf("This command ASSUMES UNCERTAINTY FROM RS ONLY so there ");
    printf("is no need to\n");
    printf("distinguish design and uncertain parameters (as both ");
    printf("are used to\n");
    printf("construct RS). This method searches through subsets ");
    printf("of designs and\n");
    printf("uses GP to estimate the AVERAGE prediction variances ");
    printf("for each\n");
    printf("subset, and the one subset that minimizes the AVERAGE ");
    printf("variances is\n");
    printf("the final selected set. There are several search ");
    printf("options each with\n");
    printf("different computational cost for you to choose.\n");
    printf("NOTE: This method assesses prediction variances rather ");
    printf("than maximizes\n");
    printf("      minimum distances between the selected points.\n");
    printf("How it should be used:\n");
    printf(" - Load a PSUADE input file using 'load'\n");
    printf("   * This file must have the INPUT section defined. If ");
    printf("it also has\n");
    printf("     a PSUADE_IO section, the sample can be used to ");
    printf("train a GP for\n");
    printf("     use in computing prediction variances. Alternatively, ");
    printf("you can\n");
    printf("     provide the GP hyperparameters (i.e. when the ");
    printf("PSUADE_IO section\n");
    printf("     is absent).\n");
    printf(" - Run odoe_mav and follow instructions.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: input information has not been loaded yet.\n");
      return 1;
    }
    for (ii = 0; ii < nSamples_*nOutputs_; ii++)
    {
      if (VecSamOutputs_[ii] > 0.9*PSUADE_UNDEFINED)
      {
        printf("odoe_mav ERROR: some sample outputs are undefined.\n");
        return -1;
      }
    }

    //**/ call specific sampler to generate samples
    GPBasedSampling *sampPtr = new GPBasedSampling();
    sampPtr->setMAV();
    sampPtr->setPrintLevel(outputLevel_);
    sampPtr->setInputBounds(nInputs_, VecILowerBs_.getDVector(), 
                            VecIUpperBs_.getDVector());
    sampPtr->setOutputParams(iOne);
    if (nSamples_ > 0)
    {
      sampPtr->setSamplingParams(nSamples_, -1, -1);
      sampPtr->loadSamples(nSamples_,nInputs_,nOutputs_,
                           VecSamInputs_.getDVector(),
                           VecSamOutputs_.getDVector(), 
                           VecSamStates_.getIVector());
    }
    sampPtr->initialize(0);
    delete sampPtr;
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_dmetric 
  //**/ compute product of eigenvalues of input covariance matrix 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_dmetric"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_dmetric: compute the product of input variances\n");
      printf("syntax: odoe_dmetric (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command computes the eigenvalues of the ");
    printf("covariance matrix for\n");
    printf("the uncertain inputs constructed from the loaded ");
    printf("sample and computes\n");
    printf("the product of eigenvalues.\n");
    printf("How to use this command : \n");
    printf(" - Create a sample (e.g. MCMCPostSample from rsmcmc)\n");
    printf(" - load the sample (load or iread depending on sample format)\n");
    printf(" - Use odoe_dmetric to compute the D-metric\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: an input sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    printf("This command uses the loaded sample as a sample ");
    printf("that represents some\n");
    printf("probability distribution of the inputs and ");
    printf("compute D-metric based on\n");
    printf("the selected inputs.\n");
    printf("(NOTE: Outputs are not relevant for this measure).\n");
    printf("First, select which inputs are uncertain inputs.\n");
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    if (ii < 1) 
    {
      printf("ERROR: need at least one uncertain input.\n");
      return 1;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ compute dmetric
    //**/ -----------------------------------------------------
    psVector vecEigs;
    psMatrix matCov;
    computeFromSampleCovMatEigen2(VecSamInputs_,nSamples_,nInputs_,
                                  vecUInputs, vecEigs, matCov);
    double dmetric=1;
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < vecEigs.length(); ii++)
    {
      ind = vecUInputs[ii];
      printf("Parameter %3d: eigenvalue = %e\n",ind+1,vecEigs[ii]);
      dmetric *= vecEigs[ii];
    }
    printf("D-metric = %e\n", dmetric); 
    printEquals(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_ametric 
  //**/ compute sum of eigenvalues of input covariance matrix 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_ametric"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_ametric: compute the sum of input variances\n");
      printf("syntax: odoe_ametric (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command computes the eigenvalues of the ");
    printf("covariance matrix for\n");
    printf("the uncertain inputs constructed from the loaded ");
    printf("sample and computes\n");
    printf("the sum of eigenvalues.\n");
    printf("How to use this command:\n");
    printf(" - Create a sample (e.g. MCMCPostSample from rsmcmc)\n");
    printf(" - load the sample (load or iread depending on sample format)\n");
    printf(" - Use odoe_ametric to compute the A-metric\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: an input sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    printf("This command uses the loaded sample as a sample ");
    printf("that represents some\n");
    printf("probability distribution of the inputs and ");
    printf("compute A-metric based on\n");
    printf("the selected inputs.\n");
    printf("(NOTE: Outputs are not relevant for this measure).\n");
    printf("First select which inputs are uncertain inputs.\n");
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    if (ii < 1) 
    {
      printf("ERROR: need at least one uncertain input.\n");
      return 1;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ compute ametric
    //**/ -----------------------------------------------------
    psVector vecEigs;
    psMatrix matCov;
    computeFromSampleCovMatEigen2(VecSamInputs_,nSamples_,nInputs_,
                                  vecUInputs, vecEigs, matCov);
    double ametric=0;
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < vecEigs.length(); ii++)
    {
      ind = vecUInputs[ii];
      printf("Parameter %3d: eigenvalue = %e\n",ind+1,
             matCov.getEntry(ii,ii));
      ametric += matCov.getEntry(ii,ii);
    }
    printf("A-metric = %e\n", ametric); 
    printEquals(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_emetric 
  //**/ compute sum of eigenvalues of input covariance matrix 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_emetric"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_emetric: compute the sum of input variances\n");
      printf("syntax: odoe_emetric (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command computes the eigenvalues of the ");
    printf("covariance matrix for\n");
    printf("the uncertain inputs constructed from the loaded ");
    printf("sample and finds\n");
    printf("the maximum.\n");
    printf("How to use this command:\n");
    printf(" - Create a sample (e.g. MCMCPostSample from rsmcmc)\n");
    printf(" - load the sample (load or iread depending on sample format)\n");
    printf(" - Use odoe_emetric to compute the E-metric\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: an input sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    printf("This command uses the loaded sample as a sample ");
    printf("that represents some\n");
    printf("probability distribution of the inputs and ");
    printf("compute E-metric based on\n");
    printf("the selected inputs.\n");
    printf("(NOTE: Outputs are not relevant for this measure).\n");
    printf("First select which inputs are uncertain inputs.\n");
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    if (ii < 1) 
    {
      printf("ERROR: need at least one uncertain input.\n");
      return 1;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ compute ametric
    //**/ -----------------------------------------------------
    psVector vecEigs;
    psMatrix matCov;
    computeFromSampleCovMatEigen2(VecSamInputs_,nSamples_,nInputs_,
                                  vecUInputs, vecEigs, matCov);
    double emetric=-PSUADE_UNDEFINED;
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < vecEigs.length(); ii++)
    {
      ind = vecUInputs[ii];
      printf("Parameter %3d: eigenvalue = %e\n",ind+1,
             matCov.getEntry(ii,ii));
      if (matCov.getEntry(ii,ii) > emetric)
        emetric = matCov.getEntry(ii,ii);
    }
    printf("E-metric = %e\n", emetric); 
    printEquals(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_entropy 
  //**/ compute entropy of input parameter distribution
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_entropy"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_entropy: compute entropy of input distribution\n");
      printf("syntax: odoe_entropy (no argument needed)\n");
    }
    if (!strcmp(winput, "-h")) return 0;

    printEquals(PL_INFO, 0);
    printf("This command uses the loaded sample as a sample ");
    printf("that represents some\n");
    printf("probability distribution of the inputs and ");
    printf("compute entropy based on\n");
    printf("the selected inputs (i.e. multi-dimensional ");
    printf("histogram, then -p log(p))\n");
    printf("(NOTE: Outputs are not relevant for this measure).\n");
    printf("How to use this command:\n");
    printf(" - Load a sample (e.g. MCMCPostSample from rsmcmc)\n");
    printf(" - Use odoe_entropy to compute the entropy metric\n");
    printf("NOTE: The loaded sample should be at least 100.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: an input sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    printf("First select which inputs are uncertain inputs.\n");
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    if (ii < 1) 
    {
      printf("ERROR: need at least one uncertain input.\n");
      return 1;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ compute metric
    //**/ -----------------------------------------------------
    //**/ extract sample values for the uncertain inputs only
    //**/ ==> vecReduced
    psVector vecReduced;
    vecReduced.setLength(vecUInputs.length()*nSamples_);
    for (ii = 0; ii < nSamples_; ii++)
    {
      for (jj = 0; jj < vecUInputs.length(); jj++)
      {
        vecReduced[ii*vecUInputs.length()+jj] = 
                    VecSamInputs_[ii*nInputs_+vecUInputs[jj]];
      }
    }
    //**/ load to a probability matrix
    ProbMatrix matProb;
    matProb.load(nSamples_,vecUInputs.length(),vecReduced.getDVector());

    psVector vecLT, vecUT;
    vecLT.setLength(vecUInputs.length());
    vecUT.setLength(vecUInputs.length());
    for (ii = 0; ii < vecUInputs.length(); ii++)
    {
      vecLT[ii] = VecILowerBs_[vecUInputs[ii]];
      vecUT[ii] = VecIUpperBs_[vecUInputs[ii]];
    }

    int nlevels = (int) (pow(1.0*matProb.nrows(),1.0/vecUInputs.length()));
    if (nlevels > 50) nlevels = 50;
    double entropy=0;
    if (nlevels <= 1) 
    {
      printf("INFO: Either the sample size is too small or ");
      printf("there are too many inputs\n");
      printf("      that a multi-dimensional histogram cannot ");
      printf("be created.\n");
      printf("===> skipp overall entropy for all inputs\n"); 
    }
    else
    {
      status = matProb.binUniform(nlevels,vecLT, vecUT);
      if (status != 0)
      {
        printf("ERROR: problem computing entropy for all inputs.\n");
      }
      else
      {
        count = 0;
        for (kk = 0; kk < matProb.nrows(); kk++)
          count += matProb.getCount(kk);
        for (kk = 0; kk < matProb.nrows(); kk++)
        {
          ddata = (double) matProb.getCount(kk)/(double) count;
          entropy -= ddata * log(ddata);
        }
        printEquals(PL_INFO, 0);
        printf("Overall entropy for all inputs  = %e\n", entropy); 
        printEquals(PL_INFO, 0);
      }
    }
    printf("Entropy for individual input distribution: \n");
    printDashes(PL_INFO, 0);
    nlevels = 10;
    for (jj = 0; jj < vecUInputs.length(); jj++)
    {
      for (ii = 0; ii < nSamples_; ii++)
        vecReduced[ii] = VecSamInputs_[ii*nInputs_+vecUInputs[jj]];
      
      matProb.load(nSamples_,iOne,vecReduced.getDVector());
      vecLT[0] = VecILowerBs_[vecUInputs[jj]];
      vecUT[0] = VecIUpperBs_[vecUInputs[jj]];
      status = matProb.binUniform(nlevels, vecLT, vecUT);
      if (status != 0)
      {
        printf("ERROR: problem computing entropy for input %d\n",
               vecUInputs[jj]+1);
      }
      else
      {
        entropy = 0;
        count = 0;
        for (kk = 0; kk < matProb.nrows(); kk++)
          count += matProb.getCount(kk);
        for (kk = 0; kk < matProb.nrows(); kk++)
        {
          ddata = (double) matProb.getCount(kk)/(double) count;
          entropy -= ddata * log(ddata);
        }
        printf("Entropy for input %d = %e\n",vecUInputs[jj]+1,entropy); 
      }
    }
    printEquals(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoe_pv 
  //**/ Compute W-metric (prediction variance) at each candidate
  //**/ point using a stochastic response surface
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoe_pv"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoe_pv: compute prediction variances for every ");
      printf("point of a given\n");
      printf("         candidate set (NOTE: this command does not ");
      printf("select a subset\n");
      printf("         but just display the prediction variances for ");
      printf("all candidates\n");
      printf("         so that users for select for themselves.\n");
      printf("syntax: odoe_pv (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command computes for each point of a candidate ");
    printf("set its prediction\n");
    printf("variance using a RS constructed from the loaded sample.\n");
    printf("(NOTE: this command assumes uncertainty from RS only)\n");
    printf("Steps to use this command (assume no uncertain input): \n");
    printf(" - Prepare a training sample\n");
    printf(" - Start PSUADE and load the training sample\n");
    printf(" - Run odoe_pv, select a stochastic RS (e.g. GP) and ");
    printf("a candidate set\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("ERROR: the training sample has not been loaded yet.\n");
      return 1;
    }
    //**/ build response surface
    printf("Step 1: Select a stochastic response surface ");
    printf("(e.g. regression, GP)\n");
    faFlag = 1;
    FuncApprox *rsPtrs = genFAInteractive(psuadeIO_, faFlag);
    rsPtrs->setBounds(VecILowerBs_.getDVector(), 
                     VecIUpperBs_.getDVector());
    rsPtrs->setOutputLevel(outputLevel_);
    status = rsPtrs->initialize(VecSamInputs_.getDVector(),
                               VecSamOutputs_.getDVector());

    //**/ get candidate design matrix ==> matCandidates
    psMatrix  matCandidates;
    int  nCandidates;
    char fname[1000];
    printf("Step 2: Provide a candidate design set (a set ");
    printf("of sample points for\n");
    printf("        which W-values will be computed). The ");
    printf("candidate set should be\n");
    printf("        stored in a text file having the following ");
    printf("format: \n\n");
    printf("        Line 1: <number of points> <number of inputs>\n");
    printf("        1 input values \n");
    printf("        2 input values \n");
    printf("        .... \n");

    snprintf(pString,100,"Enter the file name of your candidate set : ");
    getString(pString, fname);

    psVector  vecSX, vecSY, vecSD;
    psIVector vecSS;
    kk = strlen(fname);
    fname[kk-1] = '\0';
    status = readIReadDataFile(fname, matCandidates);
    if (status != 0)
    {
      printf("odoe_pv ERROR in reading candidate data\n");
      return 1;
    }
    if (matCandidates.ncols() != nInputs_)
    {
      printf("odoe_pv ERROR: candidate set has incompatible nInputs.\n");
      return 1;
    }
    nCandidates = matCandidates.nrows();
    vecSX.setLength(nCandidates*nInputs_);
    for (ii = 0; ii < nCandidates; ii++)
      for (jj = 0; jj < nInputs_; jj++)
        vecSX[ii*nInputs_+jj] = matCandidates.getEntry(ii,jj);        
    vecSY.setLength(nCandidates);
    vecSD.setLength(nCandidates);
    rsPtrs->evaluatePointFuzzy(nCandidates, vecSX.getDVector(),
                   vecSY.getDVector(),vecSD.getDVector());
    fp = fopen("matlabodoe_pv.m", "w");
    fprintf(fp, "GPW = [\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nCandidates; ii++) 
    {
      printf("Candidate %6d : prediction variance = %e\n", ii+1,
             vecSD[ii]*vecSD[ii]);
      fprintf(fp, "%e\n", vecSD[ii]*vecSD[ii]);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "plot(GPW)\n");
    fclose(fp);
    printEquals(PL_INFO, 0);
    printf("Prediction variances are stored in matlabodoe_pv.m\n");
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_boptnbf (G, I, A, E, and D metrics) 
  //**/ optimal design of experiment for batch size of n
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_boptnbf") ||
           !strcmp(command, "odoeu_foptnbf"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    int fisherMethod = 0;
    if (!strcmp(command, "odoeu_foptnbf")) fisherMethod = 1;
    if (!strcmp(winput, "-h"))
    {
      if (fisherMethod == 0)
      {
        printf("odoeu_boptnbf: compute G,I,D,A,E-metrics for all ");
        printf("n-tuples in a\n");
        printf("      candidate set and find the best one (Bayesian).\n");
        printf("syntax: odoeu_boptnbf (no argument needed)\n");
      }
      else
      {
        printf("odoeu_foptnbf: compute G,I,D,A,E-metrics for all ");
        printf("n-tuples in a\n");
        printf("      candidate set and find the best one (Fisher).\n");
        printf("syntax: odoeu_foptnbf (no argument needed)\n");
      }
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design parameters ");
    printf("X and some are\n");
    printf("uncertain parameters U. It computes for the candidate ");
    printf("set S the\n");
    printf("G,I,D,A,E values for all n-TUPLES of designs where n is ");
    printf("user-defined.\n");
    printf("It goes through all candidate combinations and records ");
    printf("the optimal\n");
    printf("subset for each metric (maybe time-consuming).\n");
    printf("G - maximmum prediction uncertainty among S designs if ");
    printf("X is selected\n");
    printf("I - average prediction uncertainty over S if this X ");
    printf("is selected\n");
    printf("D - product of covariance eigenvalues for U if X ");
    printf("is selected\n");
    printf("A - sum of variances for U if X is selected\n");
    printf("E - max of covariance eigenvalues for U if X is selected\n");
    printf("The steps to use this command are:\n");
    printf(" 1. Load a training sample (any PSUADE data format)\n");
    printf("    + This sample is for building response surfaces to ");
    printf("estimate\n");
    printf("      prediction means and variances of the evaluation ");
    printf("set. These\n");
    printf("      response surfaces are also used to estimate ");
    printf("experimental means\n");
    printf("      and std dev if they are not given in the ");
    printf("candidate set file.\n");
    if (fisherMethod == 0)
      printf(" 2. Run odoeu_boptnbf and do the following: \n");
    else
      printf(" 2. Run odoeu_foptnbf and do the following: \n");
    printf("    a. Specify which inputs are design parameters and ");
    printf("which are\n");
    printf("       uncertain parameters (U)\n");
    printf("    b. Load a prior sample for U (iread data format)\n");
    printf("    d. Load a candidate design set (iread data format)\n");
    printf("    e. Load an evaluation set (iread data format)\n");
    printf("       + this can be the same as the candidate set.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ ----------------------------------------------------------
    //**/ ask candidate information so that combinations can be
    //**/ generated later
    //**/ ----------------------------------------------------------
    printEquals(PL_INFO, 0);
    printf("Enter the number of points in the candidate set.\n");
    printf("Please Make sure the candidate set size you enter here");
    printf(" is consistent\n");
    printf("with the candidate set file you will enter later.\n");
    strcpy(pString, "How many points are in the candidate set? ");
    int nCandidates = getInt(1,1000000,pString);
    strcpy(pString, "How many to select? ");
    int nToBeSelected = getInt(1, nCandidates-1, pString);

    //**/ ----------------------------------------------------------
    //**/ set up function interface to be used for evaluation
    //**/ Note that information to be requested are initiated by
    //**/ by the function interface psLocalFunction.
    //**/ ----------------------------------------------------------
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->loadInputData(nToBeSelected, NULL);
    funcIO->loadOutputData(iOne, NULL);
    char **strArray;
    strArray = new char*[5];
    for (ii = 0; ii < 5; ii++) strArray[ii] = new char[100];
    strcpy(strArray[0], "PSUADE_LOCAL");
    strcpy(strArray[1], "NONE");
    strcpy(strArray[2], "NONE");
    strcpy(strArray[3], "PSUADE_LOCAL");
    strcpy(strArray[4], "NONE");
    funcIO->loadFunctionData(5, strArray);
    for (ii = 0; ii < 5; ii++) delete [] strArray[ii];
    delete [] strArray;
    if (fisherMethod == 0) funcIO->setLocalFunction(20);
    else                   funcIO->setLocalFunction(25);
    funcIO->setOutputLevel(-1);

    //**/ ----------------------------------------------------------
    //**/ declare variables for use inside the search loop
    //**/ ----------------------------------------------------------
    double GMetric, IMetric, DMetric, AMetric, EMetric;
    double gGMetric=PSUADE_UNDEFINED, gIMetric=PSUADE_UNDEFINED;
    double gDMetric=PSUADE_UNDEFINED, gAMetric=PSUADE_UNDEFINED;
    double gEMetric=PSUADE_UNDEFINED;
    psIVector vecTracker, vecItCnts;
    psIVector vecGMinInds, vecIMinInds, vecAMinInds, vecDMinInds;
    psIVector vecEMinInds;
    vecGMinInds.setLength(nToBeSelected);
    vecIMinInds.setLength(nToBeSelected);
    vecDMinInds.setLength(nToBeSelected);
    vecAMinInds.setLength(nToBeSelected);
    vecEMinInds.setLength(nToBeSelected);
    psVector vecInps, vecOuts;
    vecInps.setLength(nToBeSelected);
    vecOuts.setLength(5);
    if (fisherMethod == 0) fp = fopen("odoeu_boptnbf.results","w");
    else                   fp = fopen("odoeu_foptnbf.results","w");
    for (ii = 0; ii < nCandidates-nToBeSelected+1; ii++)
    {
      //**/ set vecTracker for ii candidate: ii+1, ii+2, ...
      //**/ because the ones before have been searched
      vecTracker.setLength(nToBeSelected-1);
      for (jj = 0; jj < nToBeSelected-1; jj++)
        vecTracker[jj] = ii + jj + 2;
      while (1)
      {
        vecInps[0] = ii + 1;
        for (jj = 0; jj < nToBeSelected-1; jj++)
          vecInps[jj+1] = vecTracker[jj];
        funcIO->psLocalFunction(nToBeSelected, vecInps.getDVector(),
                                5, vecOuts.getDVector());

        GMetric = vecOuts[0];
        IMetric = vecOuts[1];
        DMetric = vecOuts[2];
        AMetric = vecOuts[3];
        EMetric = vecOuts[4];
        if (GMetric != PSUADE_UNDEFINED)
        {
          printf("Candidates %5d ", ii+1);
          fprintf(fp,"Candidates %5d ", ii+1);
          for (jj = 0; jj < nToBeSelected-1; jj++)
          {
            printf("%5d ", vecTracker[jj]);
            fprintf(fp,"%5d ", vecTracker[jj]);
          }
          printf("\n");
          fprintf(fp,"\n");
          printf("     G-metric = %12.4e",GMetric);
          fprintf(fp,"     G-metric = %12.4e",GMetric);
          if (GMetric <= gGMetric)
               printf(" (* current optimal)\n");
          else printf("\n");
          fprintf(fp,"\n");
          printf("     I-metric = %12.4e",IMetric);
          fprintf(fp,"     I-metric = %12.4e",IMetric);
          if (IMetric <= gIMetric)
               printf(" (* current optimal)\n");
          else printf("\n");
          fprintf(fp,"\n");
          printf("     D-metric = %12.4e",DMetric);
          fprintf(fp,"     D-metric = %12.4e",DMetric);
          if (DMetric <= gDMetric)
               printf(" (* current optimal)\n");
          else printf("\n");
          fprintf(fp,"\n");
          printf("     A-metric = %12.4e",AMetric);
          fprintf(fp,"     A-metric = %12.4e",AMetric);
          if (AMetric <= gAMetric)
               printf(" (* current optimal)\n");
          else printf("\n");
          fprintf(fp,"\n");
          printf("     E-metric = %12.4e",EMetric);
          fprintf(fp,"     E-metric = %12.4e",EMetric);
          if (EMetric <= gEMetric)
               printf(" (* current optimal)\n");
          else printf("\n");
          fprintf(fp,"\n");
        }
        if (GMetric < gGMetric)
        {
          gGMetric = GMetric;
          vecGMinInds[0] = ii + 1;
          for (jj = 0; jj < nToBeSelected-1; jj++)
            vecGMinInds[jj+1] = vecTracker[jj];
        }
        if (IMetric < gIMetric)
        {
          gIMetric = IMetric;
          vecIMinInds[0] = ii + 1;
          for (jj = 0; jj < nToBeSelected-1; jj++)
            vecIMinInds[jj+1] = vecTracker[jj];
        }
        if (DMetric < gDMetric)
        {
          gDMetric = DMetric;
          vecDMinInds[0] = ii + 1;
          for (jj = 0; jj < nToBeSelected-1; jj++)
            vecDMinInds[jj+1] = vecTracker[jj];
        }
        if (AMetric < gAMetric)
        {
          gAMetric = AMetric;
          vecAMinInds[0] = ii;
          for (jj = 0; jj < nToBeSelected-1; jj++)
            vecAMinInds[jj+1] = vecTracker[jj];
        }
        if (EMetric < gEMetric)
        {
          gEMetric = EMetric;
          vecEMinInds[0] = ii + 1;
          for (jj = 0; jj < nToBeSelected-1; jj++)
            vecEMinInds[jj+1] = vecTracker[jj];
        }
        if (nToBeSelected == 1) break;
        kk = nToBeSelected - 2;
        vecTracker[kk]++;
        while (kk > 0 && vecTracker[kk] >= nCandidates-nToBeSelected+kk+2)
        {
          kk--;
          vecTracker[kk]++;
        }
        for (jj = kk+1; jj < nToBeSelected-1; jj++)
          vecTracker[jj] = vecTracker[jj-1] + 1;
        if (vecTracker[nToBeSelected-2] >= nCandidates)
          break;
      }
    }
    fclose(fp);
    printEquals(PL_INFO, 0);
    if (fisherMethod == 1)
      printf("(Fisher) G-minimum (%11.4e) with candidates ",gGMetric);
    else
      printf("(Bayes) G-minimum (%11.4e) with candidates ",gGMetric);
    for (ii = 0; ii < nToBeSelected; ii++)
      printf("%5d ",vecGMinInds[ii]);
    printf(" (1-based)\n");
    if (fisherMethod == 1)
      printf("(Fisher) I-minimum (%11.4e) with candidates ",gIMetric);
    else
      printf("(Bayes) I-minimum (%11.4e) with candidates ",gIMetric);
    for (ii = 0; ii < nToBeSelected; ii++)
      printf("%5d ",vecIMinInds[ii]);
    printf(" (1-based)\n");
    if (fisherMethod == 1)
      printf("(Fisher) D-minimum (%11.4e) with candidates ",gDMetric);
    else
      printf("(Bayes) D-minimum (%11.4e) with candidates ",gDMetric);
    for (ii = 0; ii < nToBeSelected; ii++)
      printf("%5d ",vecDMinInds[ii]);
    printf(" (1-based)\n");
    if (fisherMethod == 1)
      printf("(Fisher) A-minimum (%11.4e) with candidates ",gAMetric);
    else
      printf("(Bayes) A-minimum (%11.4e) with candidates ",gAMetric);
    for (ii = 0; ii < nToBeSelected; ii++)
      printf("%5d ",vecAMinInds[ii]);
    printf(" (1-based)\n");
    if (fisherMethod == 1)
      printf("(Fisher) E-minimum (%11.4e) with candidates ",gEMetric);
    else
      printf("(Bayes) E-minimum (%11.4e) with candidates ",gEMetric);
    for (ii = 0; ii < nToBeSelected; ii++)
      printf("%5d ",vecEMinInds[ii]);
    printf(" (1-based)\n");
    printEquals(PL_INFO, 0);
    if (fisherMethod == 1)
      printf("All results are in the file: odoeu_foptnbf.results.\n");
    else
      printf("All results are in the file: odoeu_boptnbf.results.\n");
  }

  //**/ -------------------------------------------------------------
  // +++ W-metric 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_wmetric")) 
  {
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_wmetric: compute W-metric for individual ");
      printf("members of a candidate\n");
      printf("set.\n");
      printf("syntax: odoeu_wmetric (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. It evaluates all members ");
    printf("of the candidate set\n");
    printf("individually (W-metric = prediction uncertainty ");
    printf("induced by uncertain\n");
    printf("inputs). Candidates with large W-metric are ");
    printf("good experimental designs.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ read training sample in PSUADE data format
    //**/ nInps = number of inputs in training sample
    //**/ nOuts = number of outputs in training sample
    char pString[1000], fname[1000];
    snprintf(pString,100,
       "Enter name of the training sample (for creating RS): ");
    getString(pString, fname);
    fname[strlen(fname)-1] = '\0';
    PsuadeData *psuadeIO = new PsuadeData();
    int status = psuadeIO->readPsuadeFile(fname);
    if (status != 0)
    {
      printf("ERROR encountered when reading training sample\n");
      printf("Maybe this file is in wrong format?\n");
      return 1;
    }
    pData pdata;
    psuadeIO->getParameter("input_ninputs", pdata);
    int nInps = pdata.intData_;
    psuadeIO->getParameter("output_noutputs", pdata);
    int nOuts = pdata.intData_;
    if (nOuts != 1)
    {
      printf("INFO: Your training sample has more than 1 ");
      printf("outputs. Please use odelete\n");
      printf("      to remove but one output and re-do.\n");
      return 1;
    }  

    //**/ set uncertain inputs (vecUInputs will have it after done)
    psIVector vecUInputs;
    vecUInputs.setLength(nInps);
    snprintf(pString,100,
      "Enter uncertain input number (1 - %d, 0 to end) : ",nInps);
    int ii = 0, kk;
    while (1)
    {
      kk = getInt(0, nInps, pString);
      if (kk == 0 || kk > nInps) break;
      vecUInputs[ii] = kk - 1;
      ii++;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ set uncertain parameter input to 1 (e.g. vecIT[kk] = 1)
    //**  if kk is an uncertain parameter
    psIVector vecIT;
    vecIT.setLength(nInps);
    kk = 1;
    for (ii = 0; ii < vecUInputs.length(); ii++)
    {
      vecIT[vecUInputs[ii]] = kk;
      kk++; 
    }

    //**/ read prior sample (in iread format)
    psMatrix matPriorSample;
    printf("Uncertain parameters need a prior sample for ");
    printf("inference. The prior\n");
    printf("sample format should be the `iread' format.\n");
    snprintf(pString,100,"Enter the file name of your prior sample : ");
    getString(pString, fname);
    fname[strlen(fname)-1] = '\0';
    status = readIReadDataFile(fname, matPriorSample);
    if (status != 0)
    {
      printf("ERROR encountered when reading prior sample\n");
      printf("Maybe this file is in wrong format?\n");
      return 1;
    }
    if (matPriorSample.ncols() != vecUInputs.length())
    {
      printf("ERROR: Prior sample nInputs is incorrect.\n");
      printf("       Should be equal to %d (as indicated before).\n",
             vecUInputs.length());
      return 1;
    }

    //**/ read candidate set
    psMatrix matCandidates;
    printf("Next, please enter the name of file containing ");
    printf("your candidate set. The\n");
    printf("format of this file should be:\n");
    printf("Line 1: <number of candidates> <number of columns>\n");
    printf("Line 2: 1 <design inputs> \n");
    printf("Line 3: 2 <design inputs> \n");
    printf("...\n");
    snprintf(pString,100,"Enter the file name of your candidate set : ");
    getString(pString, fname);
    fname[strlen(fname)-1] = '\0';
    status = readIReadDataFile(fname, matCandidates);
    if (status != 0)
    {
      printf("ERROR encountered when reading candidate set.\n");
      printf("Maybe this file is in wrong format?\n");
      return 1;
    }
    int nCandidates = matCandidates.nrows();
    printf("Size of Candidate set = %d\n", nCandidates);
    if (matCandidates.ncols() != nInps-vecUInputs.length())
    {
      printf("WARNING: Candidate file should have %d columns.\n",
             nInps-vecUInputs.length());
      printf("INFO: only the first %d columns will be used.\n",
             nInps-vecUInputs.length());
    }

    //**/ construct response surface
    pData pInps, pOuts, pLBs, pUBs;
    psuadeIO->getParameter("method_nsamples", pdata);
    int nSamp = pdata.intData_;
    psuadeIO->getParameter("input_lbounds", pLBs);
    psuadeIO->getParameter("input_ubounds", pUBs);
    psuadeIO->getParameter("input_sample", pInps);
    psuadeIO->getParameter("output_sample", pOuts);
 
    faFlag = 1;
    FuncApprox *rsPtr = genFAInteractive(psuadeIO, faFlag);
    psVector vecYT;
    vecYT.setLength(nSamp);
    printf("odoeu_wmetric: Constructing response surface\n");
    rsPtr->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
    rsPtr->setOutputLevel(0);
    for (jj = 0; jj < nSamp; jj++) vecYT[jj] = pOuts.dbleArray_[jj];
    status = rsPtr->initialize(pInps.dbleArray_,vecYT.getDVector());

    //**/ evaluate the prior sample on the response surface
    int    ll, ind, lcnt, USamSize = matPriorSample.nrows();
    double dmean, dstd;
    psVector vecXT, vecStds;
    vecXT.setLength(nInps*USamSize);
    vecYT.setLength(USamSize);
    vecStds.setLength(nCandidates);
    for (ii = 0; ii < nCandidates; ii++)
    {
      lcnt = 0;
      for (jj = 0; jj < nInps; jj++)
      {
        if (vecIT[jj] == 0)
        {
          vecXT[jj] = matCandidates.getEntry(ii,lcnt);
          for (kk = 1; kk < USamSize; kk++)
            vecXT[kk*nInps+jj] = vecXT[jj];
          lcnt++;
        }
      }
      for (ll = 0; ll < vecUInputs.length(); ll++)
      {
        ind = vecUInputs[ll];
        for (kk = 0; kk < USamSize; kk++)
          vecXT[kk*nInps+ind] = matPriorSample.getEntry(kk,ll);
      }
      rsPtr->evaluatePoint(USamSize,vecXT.getDVector(),
                           vecYT.getDVector());
      dmean = 0;
      for (ll = 0; ll < USamSize; ll++) dmean += vecYT[ll];
        dmean /= (double) USamSize;
      dstd = 0;
      for (ll = 0; ll < USamSize; ll++) 
        dstd += pow(vecYT[ll] - dmean, 2);
      dstd = sqrt(dstd / (double) USamSize);
      vecStds[ii] = dstd;
    }
    delete rsPtr;
    delete psuadeIO;
    //for (ii = 0; ii < nCandidates; ii++)
    //  printf("W metric for candidate %d = %e\n",ii+1,vecStds[ii]);
    fp = fopen("odoeu_wmetric.std","w");
    if (fp != NULL)
    {
      fprintf(fp,"%d %d 1\n",nCandidates,nInps-vecUInputs.length());
      for (ii = 0; ii < nCandidates; ii++) 
      {
        for (jj = 0; jj < nInps-vecUInputs.length(); jj++) 
          fprintf(fp,"%e ",matCandidates.getEntry(ii,jj));
        fprintf(fp,"%e\n", vecStds[ii]);
      } 
      fclose(fp);
    }
    //printEquals(PL_INFO, 0);
    printf("W-metric Ranking : \n");
    for (kk = 0; kk < nCandidates; kk++)
    {
      ddata = vecStds[0];
      ind = 0;
      for (ii = 1; ii < nCandidates; ii++)
      {
        if (vecStds[ii] > ddata)
        {
          ddata = vecStds[ii];
          ind = ii; 
        }
      }
      printf("Candidate %5d : W-metric = %10.4e\n",ind+1,vecStds[ind]);
      vecStds[ind] = 0;
    }
    printEquals(PL_INFO, 0);
    printf("odoeu_wmetric: results have been stored in odoeu_wmetric.out\n"); 
    printf("NOTE: You can use this set of W-metric as weights for ");
    printf("use in weighted\n");
    printf("      MMD design (read_std this file and run odoe_mmd).\n");
  }

  //**/ -------------------------------------------------------------
  // +++ optimal design for batch size of n (use global optimzer)
  //**/ odoeu_boptn (use Bayes methods)
  //**/ odoeu_foptn (use Fisher methods)
  //**/ odoeu_boptn_pf (use particle filter)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_foptn") || 
           !strcmp(command, "odoeu_boptn") ||
           !strcmp(command, "odoeu_boptn_pf"))
  {
    int bayesMode=0, pfMode=0;
    if (!strcmp(command, "odoeu_boptn")) bayesMode = 1;
    if (!strcmp(command, "odoeu_boptn_bf")) pfMode = 1;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (bayesMode == 0)
        printf("odoeu_foptn: select optimal design of batch size n ");
      else
        printf("odoeu_boptn: select optimal design of batch size n ");
      printf("using a global\n");
      printf("            optimization method based on one of the ");
      printf("G, I, D, A, or\n");
      if (bayesMode == 0)
      {
        printf("            E metric (use RS and Fisher matrix).\n");
        printf("syntax: odoeu_foptn (no argument needed)\n");
      }
      else
      {
        printf("            E metric (use RS and MCMC).\n");
        printf("syntax: odoeu_boptn (no argument needed)\n");
      }
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. Using global searches, it ");
    printf("seeks to find for\n");
    printf("the candidate set S the subset of size n that ");
    printf("optimizes the G, I, D,\n");
    printf("A, E metrics. Since it uses an optimizer (SCE ");
    printf("in this case) but the\n");
    printf("optimization problem may have many local minima, ");
    printf("the solution may be\n");
    printf("suboptimal given limited search time.\n");
    printf("NOTE: For W-optimal, use odoeu_wmetric because it does ");
    printf("not require\n");
    printf("      numerical optimization (just pick n points ");
    printf("with maximum W-value)\n");
    printf("There are two commands for performing optimal ");
    printf("design searches:\n");
    printf("1. odoeu_boptn, which uses statistical inference, ");
    printf("is recommended when\n");
    printf("   users have better estimates of the design ");
    printf("points in the candidate\n");
    printf("   set than what would have been estimated ");
    printf("from propagating the prior\n");
    printf("   sample through the response surface.\n");
    printf("2. odoeu_foptn, which uses the Fisher ");
    printf("information matrix method to\n");
    printf("   estimate the covariance matrix, is ");
    printf("recommended otherwise.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    int nChoices;
    if (bayesMode == 1)
    {
      printf("A. Specify which metric to use for ");
      printf("MCMC-based optimal design:\n");
      printf("1.  G-optimal (Bayesian)\n");
      printf("2.  I-optimal (Bayesian)\n");
      printf("3.  D-optimal (Bayesian)\n");
      printf("4.  A-optimal (Bayesian)\n");
      printf("5.  E-optimal (Bayesian)\n");
      nChoices = 5;
      snprintf(pString,100,"Make you choice : (1 - 5) ");
    }
    else
    {
      printf("A. Specify which metric to use for ");
      printf("Fisher-based optimal design:\n");
      printf("1.  G-optimal (Fisher approximation)\n");
      printf("2.  I-optimal (Fisher approximation)\n");
      printf("3.  D-optimal (Fisher approximation)\n");
      printf("4.  A-optimal (Fisher approximation)\n");
      printf("5.  E-optimal (Fisher approximation)\n");
      nChoices = 5;
      snprintf(pString,100,"Make you choice : (1 - 5) ");
    }
    int optOption = 0;
    optOption = getInt(1,nChoices,pString);

    printf("B. Enter information about the candidate ");
    printf("set and optimization\n");
    printf("Enter the number of points in the candidate set.\n");
    printf("(Make sure it is consistent with the size of ");
    printf("candidate set you will\n");
    printf("enter later)\n");
    strcpy(pString,"How many points are in the candidate set? ");
    int nCand = getInt(1,1000000,pString);
    strcpy(pString, "How many to select? ");
    int numSelect = getInt(1,nCand-1,pString);
    strcpy(pString,
           "Maximum number of optimization iterations: (>=100) ");
    int maxIter = getInt(100,1000000000,pString);
    psMatrix matOptData;
    matOptData.setFormat(PS_MAT2D);
    matOptData.setDim(4, numSelect);

    //**/ fixed initial guess (to be modified by users later)
    for (ii = 0; ii < numSelect; ii++) 
    {
      ddata = PSUADE_rand() % nCand + 1;
      matOptData.setEntry(0, ii, ddata);
    }
    matOptData.setEntry(1, 0, PSUADE_UNDEFINED);

    int numStarts=1;
    printf("Use multi-start optimization?  (y or n) ");
    scanf("%s",pString); 
    if (pString[0] == 'y')
    {
      fgets(winput, 500, stdin);
      snprintf(pString,100,"How many starts (2 - 100) ? ");
      numStarts = getInt(1, 100, pString);
    }  
    else
    {
      printf("An initial guess is to be provided? (y or n) ");
      scanf("%s",pString); 
      fgets(winput, 500, stdin);
      if (pString[0] == 'y')
      {
        for (ii = 0; ii < numSelect; ii++)
        {
          snprintf(pString,100,
             "Enter initial guess for input %d (1 - %d) : ",ii+1, 
             nCand);
          ddata = getDouble(pString);
          if (ddata < 0 || ddata > nCand)
          {
            printf("Wrong input: default to 0\n");
            ddata = 1;
          }
          matOptData.setEntry(0,ii,ddata);
        }
      }
    }

    //**/ -----------------------------------------------------
    //**/ set up the PsuadeData object
    //**/ -----------------------------------------------------
    psVector vLBs, vUBs;
    vLBs.setLength(numSelect);
    vUBs.setLength(numSelect);
    for (ii = 0; ii < numSelect; ii++)
    {
      vLBs[ii] = 1;
      vUBs[ii] = nCand;
    }
    char **strArray=NULL;
    strArray = new char*[numSelect+5];
    for (ii = 0; ii < numSelect+5; ii++) 
    {
      strArray[ii] = new char[1000];
      snprintf(strArray[ii],100,"X%d", ii+1);
    }
    PsuadeData *psIO = new PsuadeData();
    psIO->updateInputSection(0, numSelect, NULL, vLBs.getDVector(),
                vUBs.getDVector(),NULL,strArray,NULL,NULL,NULL,NULL);
    strcpy(strArray[0], "Y");
    psIO->updateOutputSection(0, iOne, NULL, NULL, strArray);
    psIO->updateApplicationSection(NULL,strArray[0],NULL,NULL,NULL,-1);
    int optMethod = 9; /* SCE */
    psIO->updateOptimizationSection(optMethod,iOne,1.0e-4,maxIter,-1);

    //**/ -----------------------------------------------------
    //**/ set up function interface
    //**/ -----------------------------------------------------
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->loadInputData(numSelect, NULL);
    funcIO->loadOutputData(iOne, NULL);
    strcpy(strArray[0], "PSUADE_LOCAL");
    strcpy(strArray[1], "NONE");
    strcpy(strArray[2], "NONE");
    strcpy(strArray[3], "PSUADE_LOCAL");
    strcpy(strArray[4], "NONE");
    funcIO->loadFunctionData(5, strArray);
    for (ii = 0; ii < numSelect+5; ii++) delete [] strArray[ii];
    delete [] strArray;

    int funcCode;
    if (bayesMode == 1)
    {
      funcCode = 20 + optOption - 1;
      if (pfMode == 1) funcCode -= 10;
    }
    else funcCode = 25 + optOption - 1;
    funcIO->setLocalFunction(funcCode);
    funcIO->setOutputLevel(outputLevel_);

    //**/ -----------------------------------------------------
    //**/ initialize other optimization parameters and run
    //**/ -----------------------------------------------------
    int nInitX = 1;
    for (ii = 0; ii < numSelect; ii++)
    {
      snprintf(winput,100,"iDiscrete%d", ii+1);
      psConfig_.putParameter(winput);
    }
    psConfig_.InteractiveSaveAndReset();
    psIVector vecBestSet;
    vecBestSet.setLength(numSelect);
    double bestVal = PSUADE_UNDEFINED;
    for (kk = 0; kk < numStarts; kk++)
    {
      if (numStarts > 1)
      {
        PSUADE_randInit(-1);
        printf("START %d (out of %d)\n", kk+1, numStarts);
        printf("     Initial selection: ");
        for (ii = 0; ii < numSelect; ii++) 
        {
          ddata = PSUADE_rand() % nCand + 1;
          matOptData.setEntry(0, ii, ddata);
          printf("%d ", (int) ddata);
        }
        printf("\n");
      }
      status = OptimizerSearch(psIO,funcIO,matOptData,nInitX);
      if (matOptData.getEntry(3,0) < bestVal)
      {
        for (ii = 0; ii < numSelect; ii++)
          vecBestSet[ii] = (int) matOptData.getEntry(2,ii);
        bestVal = matOptData.getEntry(3,0);
        printf("START %d (out of %d)\n", kk+1, numStarts);
        if (bayesMode == 0) printf("odoeu_foptn current best = ");
        else                printf("odoeu_boptn current best = ");
        for (ii = 0; ii < numSelect; ii++) 
          printf("%d ", vecBestSet[ii]);
        printf(" (optimal value = %e)\n", bestVal);
      }
    }
    psConfig_.InteractiveRestore();
    printEquals(PL_INFO, 0);
    if (bayesMode == 0) printf("odoeu_foptn best selection = ");
    else                printf("odoeu_boptn best selection = ");
    for (ii = 0; ii < numSelect; ii++) printf("%d ", vecBestSet[ii]);
    printf(" (optimal value = %e)\n", bestVal);
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < numSelect; ii++)
    {
      snprintf(winput,100,"iDiscrete%d", ii+1);
      psConfig_.removeParameter(winput);
    }

    //**/ -----------------------------------------------------
    //**/ clean up
    //**/ -----------------------------------------------------
    funcIO->setLocalFunction(999);
    delete psIO;
    delete funcIO;
  }

  //**/ -------------------------------------------------------------
  // +++ optimal design for batch size of n (use global optimzer)
  //**/ Based on Christen and Sanso's paper (cheap ALC)
  //**/ but here uncertainty is not induced by RS 
  //**/ March 2022:
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_alc2"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_alc2: search optimal design of batch size n ");
      printf("using a global\n");
      printf("            optimizer based on a cheaper version");
      printf("of the I-metric.\n");
      printf("syntax: odoeu_alc2 (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. Using global searches, it ");
    printf("seeks to find for\n");
    printf("the candidate set S the subset of size n ");
    printf("that optimizes the D, A, or E\n");
    printf("metrics. Since it uses an optimizer (SCE in this ");
    printf("case) but the\n");
    printf("optimization problem may have many local minima, ");
    printf("the solution may be\n");
    printf("suboptimal given limited search time.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    printf("Not implemented yet.\n");
    return 1;

    if (nInputs_ <= 0)
    {
      printf("ERROR: Input information has not been loaded yet.\n");
      printf("       You must first load an PSUADE input file that ");
      printf("defines the\n");
      printf("       inputs or a training sample (in PSUADE data ");
      printf("format).\n");
      return 1;
    }
    for (ii = 0; ii < nSamples_*nOutputs_; ii++)
    {
      if (VecSamOutputs_[ii] > 0.9*PSUADE_UNDEFINED)
      {
        printf("odoeu_alc2 ERROR: some sample outputs are undefined.\n");
        return -1;
      }
    }
  }

  //**/ -------------------------------------------------------------
  // +++ optimal design for batch size of n (use global optimzer)
  //**/ odoeu_fdoptn (use Fisher methods with derivative)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_fdoptn"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_fdoptn: select optimal design of batch ");
      printf("size n using a global\n");
      printf("      optimization method based on one of ");
      printf("the D, A or E metric. This\n");
      printf("      command is different from odoeu_foptn ");
      printf("in that it directly calls\n");
      printf("      simulators that returns also derivatives.\n");
      printf("syntax: odoeu_fdoptn (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. Using global searches, it ");
    printf("seeks to find for\n");
    printf("the candidate set S the subset of size n ");
    printf("that optimizes the D, A, or E\n");
    printf("metrics. Since it uses an optimizer (SCE in this ");
    printf("case) but the\n");
    printf("optimization problem may have many local minima, ");
    printf("the solution may be\n");
    printf("suboptimal given limited search time.\n");
    printf("This method differs from odoeu_foptn in that it ");
    printf("performs evaluations\n");
    printf("using a simulator equipped with derivatives ");
    printf("instead of response\n");
    printf("surface with finite difference. As such, ");
    printf("this command may not be\n");
    printf("applicable to expensive simulators or simulators ");
    printf("with no derivatives.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    printf("Methods available are: \n");
    printf("1.  Derivative-based Fisher G-optimal ");
    printf("(may be time-consuming)\n");
    printf("2.  Derivative-based Fisher I-optimal ");
    printf("(may be time-consuming)\n");
    printf("3.  Derivative-based Fisher D-optimal\n");
    printf("4.  Derivative-based Fisher A-optimal\n");
    printf("5.  Derivative-based Fisher E-optimal\n");
    int optOption = 0;
    strcpy(pString, "Select metric? (1 - 5) ");
    optOption = getInt(1,5,pString);

    printf("Next enter the number of points in the candidate set.\n");
    printf("Please Make sure the candidate set size you enter here");
    printf(" is consistent\n");
    printf("with the candidate set file you enter later.\n");
    strcpy(pString, "How many are in the candidate set? ");
    int nCand = getInt(1,1000000,pString);
    strcpy(pString, "How many to select? ");
    int numSelect = getInt(1,nCand-1,pString);
    strcpy(pString,
           "Maximum number of optimization iterations: (>=100) ");
    int maxIter = getInt(100,1000000000,pString);
    psMatrix matOptData;
    matOptData.setFormat(PS_MAT2D);
    matOptData.setDim(4, numSelect);

    //**/ fixed initial guess (to be modified by users later)
    for (ii = 0; ii < numSelect; ii++) 
    {
      ddata = PSUADE_rand() % nCand + 1;
      matOptData.setEntry(0, ii, ddata);
    }
    matOptData.setEntry(1, 0, PSUADE_UNDEFINED);

    int numStarts=1;
    printf("Use multi-start optimization?  (y or n) ");
    scanf("%s",pString); 
    if (pString[0] == 'y')
    {
      fgets(winput, 500, stdin);
      snprintf(pString,100,"How many starts (2 - 100) ? ");
      numStarts = getInt(1, 100, pString);
    }  
    else
    {
      printf("An initial guess is to be provided? (y or n) ");
      scanf("%s",pString); 
      fgets(winput, 500, stdin);
      if (pString[0] == 'y')
      {
        for (ii = 0; ii < numSelect; ii++)
        {
          snprintf(pString,100,"Enter initial guess for input %d (1 - %d) : ",
                   ii+1, nCand);
          ddata = getDouble(pString);
          if (ddata < 0 || ddata > nCand)
          {
            printf("Wrong input: default to 0\n");
            ddata = 1;
          }
          matOptData.setEntry(0,ii,ddata);
        }
      }
    }

    //**/ -----------------------------------------------------
    //**/ set up the PsuadeData object
    //**/ -----------------------------------------------------
    psVector vLBs, vUBs;
    vLBs.setLength(numSelect);
    vUBs.setLength(numSelect);
    for (ii = 0; ii < numSelect; ii++)
    {
      vLBs[ii] = 1;
      vUBs[ii] = nCand;
    }
    char **strArray=NULL;
    strArray = new char*[numSelect+5];
    for (ii = 0; ii < numSelect+5; ii++) 
    {
      strArray[ii] = new char[1000];
      snprintf(strArray[ii],100,"X%d", ii+1);
    }
    PsuadeData *psIO = new PsuadeData();
    psIO->updateInputSection(0, numSelect, NULL, vLBs.getDVector(),
                vUBs.getDVector(),NULL,strArray,NULL,NULL,NULL,NULL);
    strcpy(strArray[0], "Y");
    psIO->updateOutputSection(0, iOne, NULL, NULL, strArray);
    psIO->updateApplicationSection(NULL,strArray[0],NULL,NULL,NULL,-1);
    int optMethod = 9; /* SCE */
    psIO->updateOptimizationSection(optMethod,iOne,1.0e-4,maxIter,-1);

    //**/ -----------------------------------------------------
    //**/ set up function interface to call psLocalFunction
    //**/ during optimizatio
    //**/ -----------------------------------------------------
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->loadInputData(numSelect, NULL);
    funcIO->loadOutputData(iOne, NULL);
    strcpy(strArray[0], "PSUADE_LOCAL");
    strcpy(strArray[1], "NONE");
    strcpy(strArray[2], "NONE");
    strcpy(strArray[3], "PSUADE_LOCAL");
    strcpy(strArray[4], "NONE");
    funcIO->loadFunctionData(5, strArray);
    for (ii = 0; ii < numSelect+5; ii++) delete [] strArray[ii];
    delete [] strArray;

    funcIO->setLocalFunction(45+optOption-1);
    funcIO->setOutputLevel(outputLevel_);

    //**/ -----------------------------------------------------
    //**/ initialize other optimization parameters and run
    //**/ -----------------------------------------------------
    int nInitX = 1;
    for (ii = 0; ii < numSelect; ii++)
    {
      snprintf(winput,100,"iDiscrete%d", ii+1);
      psConfig_.putParameter(winput);
    }
    psConfig_.InteractiveSaveAndReset();
    psIVector vecBestSet;
    vecBestSet.setLength(numSelect);
    double bestVal = PSUADE_UNDEFINED;
    for (kk = 0; kk < numStarts; kk++)
    {
      if (numStarts > 1)
      {
        printf("START %d (out of %d)\n", kk+1, numStarts);
        for (ii = 0; ii < numSelect; ii++) 
        {
          ddata = PSUADE_rand() % nCand + 1;
          matOptData.setEntry(0, ii, ddata);
        }
      }
      status = OptimizerSearch(psIO,funcIO,matOptData,nInitX);
      if (matOptData.getEntry(3,0) < bestVal)
      {
        for (ii = 0; ii < numSelect; ii++)
          vecBestSet[ii] = (int) matOptData.getEntry(2,ii);
        bestVal = matOptData.getEntry(3,0);
      }
    }
    psConfig_.InteractiveRestore();
    printEquals(PL_INFO, 0);
    printf("odoeu_fdoptn best selection = ");
    for (ii = 0; ii < numSelect; ii++) printf("%d ", vecBestSet[ii]);
    printf(" (optimal value = %e)\n", bestVal);
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < numSelect; ii++)
    {
      snprintf(winput,100,"iDiscrete%d", ii+1);
      psConfig_.removeParameter(winput);
    }

    //**/ -----------------------------------------------------
    //**/ clean up
    //**/ -----------------------------------------------------
    funcIO->setLocalFunction(999);
    delete psIO;
    delete funcIO;
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_beval (compute G, I, A, D, E metrics for each member
  //**/  of a given candidate set individually using MCMC) 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_beval"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_beval: compute MCMC-based G,I,D,A,E-metrics ");
      printf("of all individual\n");
      printf("              members of a given candidate set.\n");
      printf("syntax: odoeu_beval (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some\n");
    printf("are uncertain parameters U. It computes for each ");
    printf("member of the given\n");
    printf("candidate set S the G,I,D,A,E values (MCMC-based).\n");
    printf("G - maximum prediction uncertainty (S is used in MCMC)\n");
    printf("I - average prediction uncertainty (S is used in MCMC)\n");
    printf("D - product of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    printf("A - sum of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    printf("E - max of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ instantiate function interface
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->setOutputLevel(0);

    //**/ initialiation
    funcIO->setLocalFunction(999);

    //**/ evaluate all metrics (use code=20 with nInputs=nOutputs=0)
    funcIO->setLocalFunction(20);
    funcIO->psLocalFunction(0, NULL, 0, NULL);

    //**/ final clean up
    funcIO->setLocalFunction(999);
    delete funcIO;
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_feval (compute A, D, E metrics for each member of a
  //**/  given candidate set individually using Fisher method) 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_feval"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_feval: compute Fisher-based D,A,E-metrics ");
      printf("of all individual\n");
      printf("              members of a given candidate set.\n");
      printf("syntax: odoeu_feval (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. It computes for each ");
    printf("member of the given\n");
    printf("candidate set S the D,A,E values based ");
    printf("on Fisher information.\n");
    printf("D - product of eigenvalues of the Fisher matrix\n");
    printf("A - sum of eigenvalues of the Fisher matrix\n");
    printf("E - max of eigenvalues of the Fisher matrix\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ instantiate function interface
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->setOutputLevel(outputLevel_);

    //**/ initialiation
    funcIO->setLocalFunction(999);

    //**/ evaluate all metrics (use code=25 with nInputs=nOutputs=0)
    funcIO->setLocalFunction(25);
    funcIO->psLocalFunction(0, NULL, 0, NULL);

    //**/ final clean up
    funcIO->setLocalFunction(999);
    delete funcIO;
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_bevaln (G, I, A, D, E metrics) 
  //**/ given a selected set, evaluate the metrics on the whole set
  //**/ together (use MCMC) 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_bevaln"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_bevaln: compute G,I,D,A,E-metric ");
      printf("for a SELECTED set of any size\n");
      printf("syntax: odoeu_bevaln (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. It computes for ");
    printf("the SELECTED set S the\n");
    printf("G,I,D,A,E values (MCMC-based).\n");
    printf("G - maximum prediction uncertainty (S is used in MCMC)\n");
    printf("I - average prediction uncertainty (S is used in MCMC)\n");
    printf("D - product of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    printf("A - sum of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    printf("E - max of covariance eigenvalues for ");
    printf("U (S is used in MCMC)\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ instantiate function interface
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->setOutputLevel(0);

    //**/ initialiation
    funcIO->setLocalFunction(999);

    //**/ evaluate all metrics (use code=20 with nInputs=0,nOutputs=5)
    psVector vecMetrics;
    vecMetrics.setLength(5);
    funcIO->setLocalFunction(20);
    funcIO->psLocalFunction(0, NULL, 5, vecMetrics.getDVector());

    //**/ final clean up
    funcIO->setLocalFunction(999);
    delete funcIO;

    //**/ display information 
    printAsterisks(PL_INFO, 0);
    printf("Posterior metrics for the selected designs : \n");
    printEquals(PL_INFO, 0);
    printf("G-metric = %e\n", vecMetrics[0]);
    printf("I-metric = %e\n", vecMetrics[1]);
    printf("D-metric = %e\n", vecMetrics[2]);
    printf("A-metric = %e\n", vecMetrics[3]);
    printf("E-metric = %e\n", vecMetrics[4]);
    printAsterisks(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_fevaln (A, D, E metrics) 
  //**/ given a selected set, evaluate the metrics on the whole set
  //**/ together (use Fisher) 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_fevaln"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_fevaln: compute Fisher-based D,A,E-metric ");
      printf("for a SELECTED set of\n");
      printf("              any size\n");
      printf("syntax: odoeu_fevaln (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design ");
    printf("parameters X and some are\n");
    printf("uncertain parameters U. It computes for ");
    printf("the SELECTED set S the D,A,E\n");
    printf("values based on Fisher information.\n");
    printf("D - product of eigenvalues of the Fisher matrix\n");
    printf("A - sum of eigenvalues of the Fisher matrix\n");
    printf("E - max of eigenvalues of the Fisher matrix\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ instantiate function interface
    FunctionInterface *funcIO = new FunctionInterface();
    funcIO->setOutputLevel(outputLevel_);

    //**/ initialiation
    funcIO->setLocalFunction(999);

    //**/ evaluate all metrics (use code=25 with nInputs=0,nOutputs=3)
    psVector vecMetrics;
    vecMetrics.setLength(3);
    funcIO->setLocalFunction(25);
    funcIO->psLocalFunction(0,NULL,3,vecMetrics.getDVector());

    //**/ final clean up
    funcIO->setLocalFunction(999);
    delete funcIO;

    //**/ display information
    printAsterisks(PL_INFO, 0);
    printf("Posterior metrics for the selected designs : \n");
    printEquals(PL_INFO, 0);
    printf("D-metric = %e\n", vecMetrics[0]);
    printf("A-metric = %e\n", vecMetrics[1]);
    printf("E-metric = %e\n", vecMetrics[2]);
    printAsterisks(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_rsmcmc 
  //**/ generate posterior sample given a set of experiments  
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_rsmcmc")) 
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_rsmcmc: create posterior sample given a ");
      printf("SELECTED set of design\n");
      printf("              points.\n");
      printf("syntax: odoeu_rsmcmc (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design parameters ");
    printf("X and some are\n");
    printf("uncertain parameters U. It computes for a given set of ");
    printf("design D the\n");
    printf("posterior distribution for the uncertain parameters ");
    printf("using Bayesian\n");
    printf("inference, i.e. posterior(U|D) ~ prior(U) L(Y(D)|D,U)\n\n");
    printf("where Y(D) is computed from the response surface to be "); 
    printf("constructed\n");
    printf("from a training sample.\n");
    printf("This command should be used in association with other ");
    printf("odoe commands\n");
    printf("such as odoeu_boptn. For example, in sequential designs, ");
    printf("one can (1)\n");
    printf("use odoeu_boptn to find the best subset of design ");
    printf("points, (2) perform\n");
    printf("experiments on this subset to update the posterior ");
    printf("distributions, and\n");
    printf("(3) use the updated posterior to find the next best ");
    printf("subset of designs.\n\n");
    printf("How this command should be used:\n");
    printf(" - Load a training sample (any PSUADE data format)\n");
    printf(" - Specify which inputs are design parameters and\n");
    printf("   which are uncertain parameters (U)\n");
    printf(" - Load a prior sample for U (iread data format)\n");
    printf(" - Load a selected candidate set (iread data format)\n");
    printf(" - Wait for the posterior file odoeu_posterior.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("odoeu_rsmcmc ERROR: sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    printf("Step 1: Indicate which inputs are uncertain inputs.\n");
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ vecIT is used to indicate which inputs are uncertain 
    //**/ vecIT[ii] >= 1 ==> uncertain parameter
    //**/ -----------------------------------------------------
    psIVector vecIT;
    vecIT.setLength(nInputs_);
    kk = 1;
    for (ii = 0; ii < vecUInputs.length(); ii++)
    {
      vecIT[vecUInputs[ii]] = kk;
      kk++;
    }

    //**/ -----------------------------------------------------
    //**/ get a prior sample ==> matPriorSample
    //**/ -----------------------------------------------------
    char fname[1000];
    psMatrix matPriorSample;
    printf("Step 2: Provide a prior sample for the uncertain ");
    printf("inputs having the\n");
    printf("        following format: \n");
    printf("        Line 1: <# of points> <# of uncertain inputs>\n");
    printf("        1 input values \n");
    printf("        2 input values \n");
    printf("        .... \n");
    snprintf(pString,100,"Enter the file name of your prior sample : ");
    getString(pString, fname);
    kk = strlen(fname);
    fname[kk-1] = '\0';
    status = readIReadDataFile(fname, matPriorSample);
    if (status != 0)
    {
      printf("odoeu_rsmcmc ERROR: in reading prior sample\n");
      return 1;
    }
    if (matPriorSample.ncols() != vecUInputs.length())
    {
      printf("odoeu_rsmcmc ERROR: prior nInputs is not correct.\n"); 
      printf("   Number of uncertain inputs = %d\n",vecUInputs.length());
      printf("   Number of inputs in prior  = %d\n",
             matPriorSample.ncols());
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get selected design matrix ==> matExpInps, Means, Stds
    //**/ -----------------------------------------------------
    psMatrix matExpInps, matExpMeans, matExpStds;

    printf("Step 3: Provide a selected set (a set of ");
    printf("sample points) which is to be\n");
    printf("        treated as experiments.\n");
    printf("        This set should be stored in a text ");
    printf("file having the following");
    printf("        format: \n\n");
    printf("        Line 1: <number of points> <number of inputs>\n");
    printf("        1 <input values> <estimated mean> <estimated std>\n");
    printf("        2 <input values> <estimated mean> <estimated std>\n");
    printf("        .... \n");
    snprintf(pString,100,"Enter the file name of your selected set : ");
    getString(pString, fname);
    kk = strlen(fname);
    fname[kk-1] = '\0';

    status = readIReadDataFile(fname, matExpInps);
    if (status != 0)
    {
      printf("odoeu_rsmcmc ERROR: in reading the selected set\n");
      return 1;
    }
    psVector vecXT;
    int nExps = matExpInps.nrows();
    printf("Selected set has size = %d\n", nExps);
    int nDesignInps = nInputs_ - vecUInputs.length();

    //**/ -----------------------------------------------------
    //**/ build response surface ==> rsPtrs
    //**/ -----------------------------------------------------
    psVector vecYT;
    printf("Constructing response surfaces ...\n");
    faFlag = 1;
    FuncApprox **rsPtrs = new FuncApprox*[nOutputs_];
    vecYT.setLength(nSamples_);
    for (ii = 0; ii < nOutputs_; ii++)
    {
      rsPtrs[ii] = genFAInteractive(psuadeIO_, faFlag);
      rsPtrs[ii]->setBounds(VecILowerBs_.getDVector(), 
                           VecIUpperBs_.getDVector());
      rsPtrs[ii]->setOutputLevel(outputLevel_);
      for (jj = 0; jj < nSamples_; jj++)
        vecYT[jj] = VecSamOutputs_[jj*nOutputs_+ii];
      status = rsPtrs[ii]->initialize(VecSamInputs_.getDVector(),
                                      vecYT.getDVector());
    }

    //**/ if prediction means and std are already available from user
    //**/ load them to matExpMeans and matExpStds
    if (matExpInps.ncols() == 2*nOutputs_+nDesignInps)
    {
      printf("odoeu_rsmcmc: experimental means/stds from selected set.\n");
      vecXT.setLength(nExps);
      matExpMeans.setDim(nExps, nOutputs_);
      matExpStds.setDim(nExps, nOutputs_);
      for (ii = 0; ii < nOutputs_; ii++)
      {
        matExpInps.getCol(nDesignInps+2*ii, vecXT);
        matExpMeans.loadCol(ii,vecXT.length(),vecXT.getDVector());
        matExpInps.getCol(nDesignInps+2*ii+1, vecXT);
        matExpStds.loadCol(ii,vecXT.length(),vecXT.getDVector());
      }
      psMatrix matTmp;
      matTmp = matExpInps;
      matExpInps.submatrix(matTmp, nExps, nDesignInps);
    }
    else
    //**/ if prediction means and std are not available from user
    //**/ have to construct it from response surface
    {
      //**/ -----------------------------------------------------
      //**/ first set up matExpInps and allocate matExpMeans,Stds
      //**/ -----------------------------------------------------
      if (matExpInps.ncols() != nDesignInps)
      {
        printf("odoeu_rsmcmc ERROR: invalid number of design inputs.\n");
        printf("        Expected = %d\n", nDesignInps);
        printf("        Actual   = %d\n", matExpInps.ncols());
        return 1;
      }
      matExpMeans.setDim(nExps, nOutputs_);
      matExpStds.setDim(nExps, nOutputs_);

      //**/ -----------------------------------------------------
      //**/ create experimental outputs
      //**/ -----------------------------------------------------
      double dmean;
      vecXT.setLength(nInputs_);
      for (ii = 0; ii < nExps; ii++)
      {
        //**/ set the experimental input matrix and also vecXT
        //**/ vecXT is used if no experimental mean/std are 
        //**/ available from the candidate set
        count = 0; 
        for (jj = 0; jj < nInputs_; jj++) 
        {
          //**/ if uncertain input, use the prior mean as input
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dmean = 0; 
            for (kk = 0; kk < matPriorSample.nrows(); kk++) 
              dmean += matPriorSample.getEntry(kk,ind);
            vecXT[jj] = dmean / (double) matPriorSample.nrows();
          }
          //**/ if design input, use the actual candidate input
          else
          {
            vecXT[jj] = matExpInps.getEntry(ii,count);
            count++;
          }
        }
        //**/ extract/compute the experimental mean/std
        for (jj = 0; jj < nOutputs_; jj++)
        {
          dmean = rsPtrs[jj]->evaluatePointFuzzy(vecXT.getDVector(),ddata);
          matExpMeans.setEntry(ii,jj,dmean);
          matExpStds.setEntry(ii,jj,ddata);
        }
      }
    }
    if (outputLevel_ > 3)
    {
      printf("Inference on selected designs: \n");
      printf("    Experimental inputs:\n");
      matExpInps.print();
      printf("    Experimental output means\n");
      matExpMeans.print();
      printf("    Experimental output std\n");
      matExpStds.print();
    }

    //**/ -----------------------------------------------------
    //**/ now matExpInps, matExpMeans, matExpStds are ready
    //**/ prepare MCMC data object
    //**/ -----------------------------------------------------
    McmcData mobj;
    mobj.printLevel_ = 0;
    mobj.nSamples_ = nSamples_;
    mobj.nInputs_ = nInputs_;
    mobj.nOutputs_ = nOutputs_;
    mobj.VecLowerB_ = VecILowerBs_;
    mobj.VecUpperB_ = VecIUpperBs_;
    mobj.VecSamInputs_ = VecSamInputs_; 
    mobj.VecSamOutputs_ = VecSamOutputs_; 
    mobj.VecCUInputs_ = vecUInputs;
    //mobj.faType_ = PSUADE_RS_GP3;
    mobj.MatPriorSample_ = matPriorSample;
    mobj.MatExpInputs_ = matExpInps;
    mobj.MatExpMeans_ = matExpMeans;
    mobj.MatExpStds_ = matExpStds;
    mobj.rsPtrs_ = rsPtrs; 
    MCMCAnalyzer *mcmcAnalyzer = new MCMCAnalyzer();
    mcmcAnalyzer->analyzeDirect(mobj);
    delete mcmcAnalyzer;
    
    //**/ store posterior sample
    fp = fopen("odoeu_posterior", "w");
    if (fp == NULL) 
    {
      printf("odoeu_rsmcmc ERROR: cannot create posterior file.\n");
      return 1;
    }
    fprintf(fp, "%d %d\n", mobj.MatPostSample_.nrows(),
            mobj.MatPostSample_.ncols());
    for (ii = 0; ii < mobj.MatPostSample_.nrows(); ii++) 
    {
      fprintf(fp, "%7d ", ii+1);
      for (jj = 0; jj < mobj.MatPostSample_.ncols(); jj++) 
      {
        ddata = mobj.MatPostSample_.getEntry(ii,jj);
        fprintf(fp, "%24.16e ", ddata);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
    for (ii = 0; ii < nOutputs_; ii++) delete rsPtrs[ii]; 
    delete [] rsPtrs; 
    printf("odoeu_rsmcmc: posterior sample is in odoeu_posterior.\n");
  }

  //**/ -------------------------------------------------------------
  // +++ odoeu_rseval 
  //**/ given an evaluation set, compute its RS mean and std dev.
  //**/ whereby std dev is induced by certain uncertain inputs
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "odoeu_rseval")) 
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("odoeu_rseval: given a list of evaluation ");
      printf("points, evaluate their means\n");
      printf("              on a given response surface ");
      printf("and standard deviations\n");
      printf("              (induced by some uncertain ");
      printf("parameters), and write the\n");
      printf("              results into a file.\n");
      printf("syntax: odoeu_rseval (no argument needed)\n");
    }
    printEquals(PL_INFO, 0);
    printf("This command assumes some inputs are design parameters ");
    printf("X and some are\n");
    printf("uncertain parameters U. It computes for a given set of ");
    printf("evaluation\n");
    printf("points the output means/std dev using the response ");
    printf("surfaces\n");
    printf("constructed from the loaded sample.\n");
    if (!strcmp(winput, "-h")) return 0;

    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0)
    {
      printf("odoeu_rseval ERROR: sample has not been loaded yet.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get which inputs are uncertain inputs ==> vecUInputs
    //**/ -----------------------------------------------------
    printf("Step 1: Indicate which inputs are uncertain inputs.\n");
    psIVector vecUInputs;
    vecUInputs.setLength(nInputs_);
    snprintf(pString,100,
       "Enter uncertain input number (1 - %d, 0 to end) : ",nInputs_);
    ii = 0;
    while (1)
    {
      kk = getInt(0, nInputs_, pString);
      if (kk == 0 || kk > nInputs_) break;
      vecUInputs[ii] = kk - 1; 
      ii++;
    }
    if (ii == 0)
    {
      printf("odoeu_rseval ERROR: there must be some uncertain inputs.\n");
      return 1;
    }
    vecUInputs.subvector(0, ii-1);

    //**/ -----------------------------------------------------
    //**/ vecIT is used to indicate which inputs are uncertain 
    //**/ vecIT[ii] >= 1 ==> uncertain parameter
    //**/ -----------------------------------------------------
    psIVector vecIT;
    vecIT.setLength(nInputs_);
    kk = 1;
    for (ii = 0; ii < vecUInputs.length(); ii++)
    {
      vecIT[vecUInputs[ii]] = kk;
      kk++;
    }

    //**/ -----------------------------------------------------
    //**/ get a sample ==> matPriorSample
    //**/ -----------------------------------------------------
    char fname[1000];
    psMatrix matPriorSample;
    printf("Step 2: Provide a sample for the uncertain ");
    printf("inputs in the following\n");
    printf("        format: \n");
    printf("Line 1: <number of points> <number of uncertain inputs>\n");
    printf("1 input values \n");
    printf("2 input values \n");
    printf(".... \n");
    snprintf(pString,100,"Enter the file name of your uncertain sample : ");
    getString(pString, fname);
    kk = strlen(fname);
    fname[kk-1] = '\0';
    status = readIReadDataFile(fname, matPriorSample);
    if (status != 0)
    {
      printf("odoeu_rseval ERROR: in reading uncertain sample\n");
      return 1;
    }
    if (matPriorSample.ncols() != vecUInputs.length())
    {
      printf("odoeu_rseval ERROR: sample nInputs is not correct.\n"); 
      printf("   Number of uncertain inputs = %d\n",vecUInputs.length());
      printf("   Number of inputs in prior  = %d\n",
             matPriorSample.ncols());
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ get selected design matrix ==> matExpInps, Means, Stds
    //**/ -----------------------------------------------------
    psMatrix matEvalSet;
    snprintf(pString,100,"Enter the file name of your evaluation set : ");
    getString(pString, fname);

    kk = strlen(fname);
    fname[kk-1] = '\0';
    status = readIReadDataFile(fname, matEvalSet);
    if (status != 0)
    {
      printf("odoe_rseval ERROR: in reading the evaluation set\n");
      return 1;
    }
    int nEvalPts = matEvalSet.nrows();

    //**/ -----------------------------------------------------
    //**/ build response surface
    //**/ -----------------------------------------------------
    printf("Constructing response surfaces ...\n");
    faFlag = 1;
    psVector vecYT;
    FuncApprox **rsPtrs = new FuncApprox*[nOutputs_];
    vecYT.setLength(nSamples_);
    for (ii = 0; ii < nOutputs_; ii++)
    {
      rsPtrs[ii] = genFAInteractive(psuadeIO_, faFlag);
      rsPtrs[ii]->setBounds(VecILowerBs_.getDVector(), 
                            VecIUpperBs_.getDVector());
      rsPtrs[ii]->setOutputLevel(0);
      for (jj = 0; jj < nSamples_; jj++)
        vecYT[jj] = VecSamOutputs_[jj*nOutputs_+ii];
      status = rsPtrs[ii]->initialize(VecSamInputs_.getDVector(),
                                      vecYT.getDVector());
    }

    //**/ -----------------------------------------------------
    //**/ evaluate response surface
    //**/ -----------------------------------------------------
    psVector  vecSX, vecSY;
    vecSX.setLength(matPriorSample.nrows()*nInputs_);
    vecSY.setLength(matPriorSample.nrows());
    int lcnt, lcnt2;
    double dmean, dstd;
    
    fp = fopen("odoeu_rseval.out","w");
    if (fp == NULL)
    {
      printf("ERROR: cannot write to file odoeu_rseval.out\n");
      return 1;
    }
    fprintf(fp, "%d %d\n", nEvalPts, 
            nInputs_-vecUInputs.length()+nOutputs_*2);
    double smax2=0, smean2=0;
    for (ii = 0; ii < nEvalPts; ii++)
    {
      fprintf(fp, "%d ", ii+1);
      for (kk = 0; kk < matPriorSample.nrows(); kk++)
      {
        lcnt = lcnt2 = 0;
        for (jj = 0; jj < nInputs_; jj++)
        {
          if (vecIT[jj] == 0)
          {
            vecSX[kk*nInputs_+jj] = matEvalSet.getEntry(ii,lcnt);
            if (kk == 0) fprintf(fp, "%16.8e ", vecSX[kk*nInputs_+jj]);
            lcnt++;
          }
          else
          {
            vecSX[kk*nInputs_+jj] = matPriorSample.getEntry(kk,lcnt2);
            lcnt2++;
          }
        }
      }
      for (kk = 0; kk < nOutputs_; kk++)
      {
        rsPtrs[kk]->evaluatePoint(matPriorSample.nrows(),vecSX.getDVector(),
                                  vecSY.getDVector());
        dmean = dstd = 0;
        for (jj = 0; jj < matPriorSample.nrows(); jj++)
          dmean += vecSY[jj];
        dmean /= (double) matPriorSample.nrows();
        for (jj = 0; jj < matPriorSample.nrows(); jj++)
          dstd += pow(vecSY[jj] - dmean, 2.0);
        dstd /= (double) matPriorSample.nrows();
        dstd = sqrt(dstd);
        fprintf(fp, "%16.8e %16.8e ", dmean, dstd);
        printf("%d: mean = %16.8e, s.d. = %16.8e\n",ii+1,dmean,dstd);
        smean2 = smean2 + dstd; 
        if (dstd > smax2) smax2 = dstd;
      }
      fprintf(fp, "\n");
    }
    smean2 /= (nOutputs_ * nEvalPts);
    printf("Average of all %d std. devs. = %e\n", nEvalPts, smean2);
    printf("Maximum of all %d std. devs. = %e\n", nEvalPts, smax2);
    fclose(fp);
    printf("odoeu_rseval: evaluated set is now in odoeu_rseval.out.\n");
  }

  //**/ -------------------------------------------------------------
  // +++ otherwise 
  //**/ -------------------------------------------------------------
  else
  {
    printf("command %s not recognized\n", command);
    fflush(stdout);
    return 1;
  }
  return 0;
}

// ************************************************************************
// display menu for ODOE 
// ------------------------------------------------------------------------
void PsuadeBase::displayHelpMenuODOE(int option)
{
  printAsterisks(PL_INFO, 0);
  printf("TWO CLASSES OF OPTIMAL DoE (ODOE) METHODS:\n");
  printf("(1) Methods for models with design parameters ONLY\n");
  printf("    (uncertainty comes from Response Surfaces: RS)\n");
  printf("(2) Methods for models with design AND uncertain ");
  printf("parameters\n");
  printf("    (RS uncertainty will not be accounted for)\n");
  printEquals(PL_INFO, 0);
  printf("COMMANDS for (1) are:\n");
  printf("   odoe_mmd      (Find max-min INPUT-only ");
  printf("distance designs)\n");
  if (option == 1)
  {
    printf("                 (a) Space-filling design ");
    printf("based on max-min distance\n");
    printf("                        (Use either global ");
    printf("optimization or brute-force\n");
    printf("                  search).\n");
    printf("                 (b) For INPUT-OUTPUT space-filling ");
    printf("MMD design, move\n");
    printf("                     the selected outputs to the ");
    printf("input section first\n");
    printf("                     (use edit commands) and then ");
    printf("run this method.\n");
    printf("                 (c) For space-filling W-optimal ");
    printf("design, populate an\n");
    printf("              OUTPUT with prediction variances for ");
    printf("all points in the\n");
    printf("              candidate set, and use this ");
    printf("command with weights.\n");
    printDashes(PL_INFO, 0);
  }
  printf("   odoe_mmv      (Find GP-based min-max ");
  printf("OUTPUT variance designs)\n");
  if (option == 1)
  {
    printf("                 => Uses GP to guide ");
    printf("space-filling design:\n");
    printf("                 (a) Provide a training sample ");
    printf("(INPUTS/OUTPUTS) to\n");
    printf("                     compute GP hyperparameter ");
    printf("values, or\n");
    printf("                 (b) GP hyperparameters are to ");
    printf("be entered by users.\n");
    printDashes(PL_INFO, 0);
  }
  printf("   odoe_mav      (Find GP-based min-average ");
  printf("OUTPUT variance designs)\n");
  if (option == 1)
  {
    printf("                 Uses GP to guide space-filling ");
    printf("design:\n");
    printf("                 ==> Similar to odoe_mmv except ");
    printf("using min-average\n");
    printf("                     objective.\n");
    printDashes(PL_INFO, 0);
  }
  printf("   odoe_dmetric  (Compute D metric from ");
  printf("the loaded sample INPUTS)\n");
  printf("   odoe_ametric  (Compute A metric from ");
  printf("the loaded sample INPUTS)\n");
  printf("   odoe_emetric  (Compute E metric from ");
  printf("the loaded sample INPUTS)\n");
  printf("   odoe_entropy  (Compute entropy  from ");
  printf("the loaded sample INPUTS)\n");
  if (option == 1) printDashes(PL_INFO, 0);
  printf("   odoe_pv       (Compute prediction variances ");
  printf("for a candidate set)\n");
  if (option == 1)
  {
    printf("                 This command computes prediction ");
    printf("uncertainty for each\n");
    printf("                 point in the candidate set from ");
    printf("some STOCHASTIC RS\n");
    printf("                 (e.g. GP).\n");
    printf("                 NOTE: prediction uncertainties are ");
    printf("induced by RS\n");
    printf("                       uncertainties.\n");
    printf("                 NOTE: This command only computes ");
    printf("prediction variances\n");
    printf("                       for each member of the ");
    printf("candidate set. Users are\n");
    printf("                       to make their own ");
    printf("selection afterward.\n");
  }
  printEquals(PL_INFO, 0);
  printf("COMMANDS for (2) are:\n");
  printf("   odoeu_wmetric (Compute W-metric for ");
  printf("each member of a candidate set)\n");

  printf("   odoeu_boptn   (Search GIDAE-optimal ");
  printf("design of size n: RS/MCMC)\n");

  printf("   odoeu_foptn   (Search GIDAE-optimal ");
  printf("design of size n: RS/Fisher)\n");

  //printf("   odoeu_alc2    (Search I-optimal ");
  //printf("design of size n: RS)\n");

  printf("   odoeu_fdoptn  (Same as odoeu_foptn ");
  printf("but use simulator/derivatives)\n");

  printf("   odoeu_beval   (Compute GIDAE for ");
  printf("each candidate in a set: RS/MCMC)\n");

  printf("   odoeu_feval   (Same as odoeu_beval ");
  printf("but use the Fisher method)\n");

  printf("   odoeu_bevaln  (Compute GIDAE for a ");
  printf("design set as a whole: RS/MCMC)\n");

  printf("   odoeu_rsmcmc  (Run MCMC to create a ");
  printf("posterior sample given a set)\n");

  printf("   odoeu_rseval  (Compute RS mean/std ");
  printf("dev given a set)\n");

  if (option == 1)
  {
    printf("   odoeu_boptnbf (Brute-force search of ");
    printf("GIDAE-optimal design of size\n");
    printf("                  n: RS/MCMC)\n");

    printf("   odoeu_foptnbf (Brute-force search of ");
    printf("GIDAE-optimal design of size\n");
    printf("                  n: RS/Fisher)\n");

    printf("   odoeu_boptn_pf(Search GIDAE-optimal ");
    printf("design of size n: RS/PF)\n");
    printf("         (PF - particle filtering)\n");
  }
  printAsterisks(PL_INFO, 0);
}

