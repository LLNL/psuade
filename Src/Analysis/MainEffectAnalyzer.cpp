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
// functions for the class MainEffectAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include "MainEffectAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "pData.h"
#include "PsuadeData.h"
#include "RSConstraints.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
MainEffectAnalyzer::MainEffectAnalyzer() : Analyzer(), nInputs_(0), 
                        outputID_(0), totalInputVCE_(0), 
                        mainEffectMean_(0), mainEffectStd_(0)
{
  setName("MainEffect");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MainEffectAnalyzer::~MainEffectAnalyzer()
{
}

// ************************************************************************
// perform VCE (McKay) analysis (nSubSamples : number of LHS points)
// ------------------------------------------------------------------------
double MainEffectAnalyzer::analyze(aData &adata)
{
  int    ii, jj, ss, ncount, index, status, iZero=0;
  char   winput1[500], winput2[500], meFileName[500];
  char   pString[500], **inputNames;
  FILE   *fp=NULL, *fp1=NULL;
  pData  pdata;
  PsuadeData    *ioPtr;
  psVector      VecY, VecXT, VecYT, VecTX, VecTY, VecVCE, VecMeanVCEVar;
  psVector      VecVarVCEVar, VecVarVCEMean;
  RSConstraints *constrPtr=NULL;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  int outputID = adata.outputID_;
  double *sampleInputs  = adata.sampleInputs_;
  double *sampleOutputs = adata.sampleOutputs_;
  nInputs_  = nInputs;
  outputID_ = outputID;
  int nSubSamples = adata.nSubSamples_;
  double *iLowerB = adata.iLowerB_;
  double *iUpperB = adata.iUpperB_;
  int printLevel  = adata.printLevel_;
  int whichOutput = outputID;
  ioPtr = adata.ioPtr_;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    exit(1);
  } 
  if (nSamples/nSubSamples*nSubSamples != nSamples)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: nSamples != k*nLevels.\n");
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    printOutTS(PL_ERROR,"    nLevels  = %d\n", nSubSamples);
    exit(1);
  } 
  if (whichOutput >= nOutputs || whichOutput < 0)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: invalid outputID.\n");
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    outputID = %d\n", whichOutput+1);
    exit(1);
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  }
  if (adata.inputPDFs_ != NULL)
  {
    ncount = 0;
    for (ii = 0; ii < nInputs; ii++) ncount += adata.inputPDFs_[ii];
    if (ncount > 0)
    {
      printOutTS(PL_INFO, 
           "MainEffect INFO: Some inputs have non-uniform PDFs.\n");
      printOutTS(PL_INFO, 
           "           However, they are relevant in this analysis\n");
      printOutTS(PL_INFO, 
           "           (the sample should have been generated\n");
      printOutTS(PL_INFO,"            with the desired distributions.)\n");
    }
  }
  VecInputVCE_.clean();

  //**/ ---------------------------------------------------------------
  //**/ get response surface filter information, if any
  //**/ ---------------------------------------------------------------
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    if(constrPtr != NULL) constrPtr->genConstraints(ioPtr);
    else 
    {
      printOutTS(PL_ERROR,"out of memory in file %s line %d, exiting.\n",
                 __FILE__, __LINE__);
      exit(1);
    }
  }
   
  //**/ ---------------------------------------------------------------
  //**/ check for failed samples
  //**/ ---------------------------------------------------------------
  double *X = sampleInputs, ddata;
  VecY.setLength(nSamples);

  ncount = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    VecY[ss] = sampleOutputs[nOutputs*ss+whichOutput];
    ddata = constrPtr->evaluate(&(X[ss*nInputs]), VecY[ss], status);
    if (status == 0) VecY[ss] = PSUADE_UNDEFINED; 
    else             ncount++;
  }
  if (ncount == 0)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: no valid sample point.\n");
    printOutTS(PL_ERROR,"    nSamples before filtering = %d\n", nSamples);
    printOutTS(PL_ERROR,"    nSamples after  filtering = %d\n", ncount);
    printOutTS(PL_ERROR, 
         "    INFO: check your data file for undefined's (1e35)\n");
    if (constrPtr != NULL) delete constrPtr;
    return 1.0;
  } 
  if (ncount != nSamples)
  {
    printOutTS(PL_INFO,
         "MainEffectAnalyzer INFO: CONSTRAINTS HAVE BEEN APPLIED.\n");
    printOutTS(PL_INFO, 
         "    nSamples before filtering = %d\n", nSamples);
    printOutTS(PL_INFO, 
         "    nSamples after  filtering  = %d\n", ncount);
  }
  if (nSamples < 1000)
  {
    printOutTS(PL_INFO,
       "MainEffectAnalyzer INFO: nSamples may be too small to give results\n");
    printOutTS(PL_INFO, 
       "                         with acceptable accuracy (nSamples = %d).\n",
       nSamples);
  }

  //**/ ---------------------------------------------------------------
  //**/ first compute agglomerated mean and variance
  //**/ ---------------------------------------------------------------
  double aMean, aVariance;
  computeMeanVariance(nInputs,1,nSamples,VecY.getDVector(),&aMean,
                      &aVariance,0);
  if (printLevel >= 0)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*              Main Effect Analysis\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO, 
         "* Turn on higher printlevel to display more information\n");
    printOutTS(PL_INFO,"* Turn on ana_expert mode for more plots\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"* Number of sample points = %10d\n",nSamples);
    printOutTS(PL_INFO,"* Number of Inputs        = %10d\n",nInputs);
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"Output %d\n", whichOutput+1);
    printOutTS(PL_INFO,"====> MainEffect: mean               = %12.4e\n",
               aMean);
    printOutTS(PL_INFO,"====> MainEffect: standard deviation = %12.4e\n",
                 sqrt(aVariance));
  }
  mainEffectMean_ = aMean;
  mainEffectStd_ = sqrt(aVariance);
  //**/ ---------------------------------------------------------------
  //**/ allocate spaces for local variables
  //**/ ---------------------------------------------------------------
  int nSubs = nSubSamples;
  int nReplications = nSamples / nSubs;
#if 0
  //**/ ---------------------------------------------------------------
  //**/ This part taken out to accommodate factorial sampling, which 
  //**/ has nReplications = 1
  //**/ ---------------------------------------------------------------
  if (nReplications <= 1)
  {
    printOutTS(PL_INFO, 
         "MainEffectAnalyzer INFO: go no further since nReps = 1.\n");
    return 1.0;
  }
#endif
  VecVCE.setLength(nInputs);
  VecMeanVCEVar.setLength(nInputs);
  VecVarVCEMean.setLength(nInputs);
  VecVarVCEVar.setLength(nInputs);
  VecTX.setLength(nSamples);
  VecTY.setLength(nSamples);

  //**/ ---------------------------------------------------------------
  //**/ first revise nReplications and nSubs (using X1 only to test)
  //**/ ---------------------------------------------------------------
  for (ss = 0; ss < nSamples; ss++) VecTY[ss] = VecY[ss];
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ss = 0; ss < nSamples; ss++) VecTX[ss] = X[nInputs*ss+ii];
    sortDbleList2(nSamples,VecTX.getDVector(),VecTY.getDVector());
    nReplications = 1;
    for (ss = 1; ss < nSamples; ss++)
    {
      if (VecTX[ss] == VecTX[0]) nReplications++;
      else                       break;
    }
    printf("* Number of replications for Input %d = %d\n",ii+1,
           nReplications);
    if (nReplications <= 5)
    {
      printOutTS(PL_INFO,
        "* MainEffect INFO: nReps <= 5 for input %d.\n",ii+1);
      printOutTS(PL_INFO,
        "*     ==> probably not replicated Latin hypercube\n");
      printOutTS(PL_INFO,"*     ==> crude main effect analysis.\n");
      computeVCECrude(nInputs, nSamples, X, VecY.getDVector(), 
                iLowerB, iUpperB, aVariance, VecVCE.getDVector());
      if (ioPtr != NULL)
      {
        pData *pPtr = ioPtr->getAuxData();
        pPtr->nDbles_ = nInputs;
        pPtr->dbleArray_ = new double[nInputs * nInputs];
        for (ii = 0; ii < nInputs; ii++)
          pPtr->dbleArray_[ii] = VecVCE[ii];
        pPtr->dbleData_ = aVariance;
      }
      //**/ ---------------------------------------------------------
      //**/ generate matlab (printLevel < 0 when rsmeb is called)
      //**/ ---------------------------------------------------------
      if (printLevel >= 0)
        printResults(nInputs, aVariance, VecVCE.getDVector(), ioPtr);

      //**/ ---------------------------------------------------------
      //**/ clean up
      //**/ ---------------------------------------------------------
      delete constrPtr;
      return 1.0;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/  fetch matlab file name (interactively or from config file) 
  //**/ ---------------------------------------------------------------
  fp = NULL;
  if (psConfig_.AnaExpertModeIsOn())
  {
    snprintf(pString,100,"Create main effect scatter plot ? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      if (plotScilab())
        snprintf(pString,100,"Enter scatter plot file name (ends with .sci): ");
      else
        snprintf(pString,100,"Enter scatter plot file name (ends with .m): ");
      getString(pString, meFileName);
      meFileName[strlen(meFileName)-1] = '\0';
      fp = fopen(meFileName, "w");
      if (fp != NULL)
        printOutTS(PL_INFO,
             "MainEffect: main effect file = %s\n",meFileName);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ display problem information
  //**/ ---------------------------------------------------------------
  if (fp != NULL)
  {
    strcpy(pString," **********************************************");
    fwriteComment(fp, pString);
    strcpy(pString," *  Main Effect Analysis                     **");
    fwriteComment(fp, pString);
    strcpy(pString," *-------------------------------------------**");
    fwriteComment(fp, pString);
    strcpy(pString," file for main effect plots \n");
    fwriteComment(fp, pString);
  }

  //**/ ---------------------------------------------------------------
  //**/ write, to a matlab file, the settings for x_i's
  //**/ ---------------------------------------------------------------
  if (fp != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      for (ss = 0; ss < nSamples; ss++)
        VecTX[ss] = X[nInputs*ss+ii];
      sortDbleList2(nSamples, VecTX.getDVector(), VecTY.getDVector());
      fprintf(fp, "X%d = [ \n", ii);
      fprintf(fp,"%24.16e\n",VecTX[0]);
      for (ss = 1; ss < nSamples; ss++)
        if (VecTX[ss] != VecTX[ss-1])
          fprintf(fp,"%24.16e\n",VecTX[ss]);
      fprintf(fp, "];\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ calculate statistics for each input
  //**/ ---------------------------------------------------------------
  status = computeVCE(nInputs,nSamples,nSubSamples,X,VecY.getDVector(), 
                  whichOutput,fp,VecVarVCEMean.getDVector(),
                  VecMeanVCEVar.getDVector(),VecVarVCEVar.getDVector(),
                  VecVCE.getDVector());
  if (status == -1)
  {
    computeVCECrude(nInputs,nSamples,X,VecY.getDVector(),iLowerB,
                    iUpperB,aVariance,VecVCE.getDVector());
    //**/ ---------------------------------------------------------------
    //**/ generate matlab (printLevel < 0 when rsmeb is called)
    //**/ ---------------------------------------------------------------
    if (printLevel >= 0)
      printResults(nInputs, aVariance, VecVCE.getDVector(), ioPtr);
    if (ioPtr != NULL)
    {
      pData *pPtr = ioPtr->getAuxData();
      pPtr->nDbles_ = nInputs;
      pPtr->dbleArray_ = new double[nInputs * nInputs];
      for (ii = 0; ii < nInputs; ii++) 
        pPtr->dbleArray_[ii] = VecVCE[ii];
      pPtr->dbleData_ = aVariance;
    }

    //**/ ---------------------------------------------------------------
    //**/ clean up
    //**/ ---------------------------------------------------------------
    if (constrPtr != NULL) delete constrPtr;
    if (fp != NULL) fclose(fp);
    return 0.0;
  }
  if (ioPtr != NULL)
  {
    pData *pPtr = ioPtr->getAuxData();
    pPtr->nDbles_ = nInputs;
    pPtr->dbleArray_ = new double[nInputs * nInputs];
    for (ii = 0; ii < nInputs; ii++) 
      pPtr->dbleArray_[ii] = VecVCE[ii];
    pPtr->dbleData_ = aVariance;
  }

  //**/ --------------- not used anymore -----------------------
  //**/ check to see if inter group variance is greater than 
  //**/ intra-group variance
  //**/flag = 0;
  //**/for (subID = 0; subID < nSubs; subID++)
  //**/   if (vceVariance[subID] > VecVarVCEMean[ii]) flag = 1;
  //**/if (flag == 1)
  //**/{
  //**/   printf("Input %3d : intra-group variance > inter-group ",ii);
  //**/   printf("variance\n");
  //**/}
  //**/ --------------------------------------------------------

  //**/ ---------------------------------------------------------------
  //**/ display VCE for each input
  //**/ ---------------------------------------------------------------
  double totalVCE;
  if (printLevel >= 0)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"            McKay's correlation ratio\n");
    printEquals(PL_INFO,0);
    totalVCE = 0.0;
    if (aVariance == 0) 
      printOutTS(PL_INFO, "Total VCE = %9.2e\n",totalVCE);
    else
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        totalVCE += VecVCE[ii] / aVariance;
        printOutTS(PL_INFO, 
          "Input %4d, normalized 1st-order effect = %9.2e (raw = %9.2e)\n",
          ii+1, VecVCE[ii]/aVariance, VecVCE[ii]);
      }
      printOutTS(PL_INFO, "Total VCE = %9.2e\n", totalVCE);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ display VecVarVCEMean for each input
  //**/ ---------------------------------------------------------------
  if (printLevel > 2)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "     McKay's biased correlation ratio (stdVCEMean)\n");
    printDashes(PL_INFO, 0);
    totalVCE = 0.0;
    for (ii = 0; ii < nInputs; ii++)
    {
      printOutTS(PL_INFO, "Input %2d = %9.2e >>? %9.2e\n", ii+1,
             VecVarVCEMean[ii]/aVariance,1.0/(double)(nSubs*nSubs));
      totalVCE += VecVarVCEMean[ii] / aVariance;
    }
    printOutTS(PL_INFO, "Total VCE = %9.2e\n", totalVCE);
  }

  //**/ ---------------------------------------------------------------
  //**/ display variance of vce variance
  //**/ ---------------------------------------------------------------
  if (printLevel > 2)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, "           Strength of interaction (varVCEVar)\n");
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO,"    Input %2d = %9.2e (meanVCEVar = %9.2e)\n",
                 ii+1, VecVarVCEVar[ii], VecMeanVCEVar[ii]);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"            Hora and Iman sensitivity index\n");
    printOutTS(PL_INFO,
         "   (may not be valid in the presence of constraints)\n");
    printDashes(PL_INFO, 0);
    totalVCE = 0.0;
    for (ii = 0; ii < nInputs; ii++) 
      totalVCE += sqrt(aVariance-VecMeanVCEVar[ii]);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO, "    Input %2d = %9.2e\n", ii+1,
             sqrt(aVariance-VecMeanVCEVar[ii])/totalVCE);
    printAsterisks(PL_INFO, 0);
  }

#ifdef HAVE_PYTHON
  //**/ ---------------------------------------------------------------
  //**/ store data for python graphics
  //**/ ---------------------------------------------------------------
  //**/ Store scalars as python ints/floats
  PyObject *temp, *Xlist, *XIlist, *Ylist, *VCElist, *vVCEMlist;

  PyDict_SetItemString( AnalysisDataDict, "nInputs",
                 temp=PyInt_FromLong(nInputs) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "outputID",
                 temp=PyInt_FromLong(whichOutput) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "nSamples",
                 temp=PyInt_FromLong(nSamples) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "nSubs",
                 temp=PyInt_FromLong(nSubs) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "aVariance",
                 temp=PyFloat_FromDouble(aVariance) ); Py_DECREF(temp);

  //**/ For each sample, store X values and VCE'sfor each input, 
  //**/ in python lists
  //**/ Undefined samples have already been removed.
  Xlist     = PyList_New(0);
  VCElist   = PyList_New(0);
  vVCEMlist = PyList_New(0);
  for (ii = 0; ii < nInputs; ii++)
  {
    PyList_Append(VCElist, temp=PyFloat_FromDouble(VecVCE[ii]));
    Py_DECREF(temp);
    PyList_Append(vVCEMlist,temp=PyFloat_FromDouble(VecVarVCEMean[ii]));
    Py_DECREF(temp);

    XIlist = PyList_New(0);
    for (ss = 0; ss < nSamples; ss++)
    {
      PyList_Append(XIlist, temp=PyFloat_FromDouble(X[nInputs*ss+ii]));
      Py_DECREF(temp);
    }
    PyList_Append(Xlist, XIlist); Py_DECREF(XIlist);
  }
  PyDict_SetItemString(AnalysisDataDict,"X", Xlist ); Py_DECREF(Xlist);
  PyDict_SetItemString(AnalysisDataDict,"VecVCE", VCElist); 
  Py_DECREF(VCElist);
  PyDict_SetItemString(AnalysisDataDict,"VecVarVCEMean", vVCEMlist );
  Py_DECREF(vVCEMlist);

  //**/ For each sample, store Y values for each input, in python lists
  Ylist = PyList_New(0);
  for (ss = 0; ss < nSamples; ss++)
  {
    PyList_Append(Ylist,temp=PyFloat_FromDouble(VecY[ss]) ); 
    Py_DECREF(temp);
  }
  PyDict_SetItemString(AnalysisDataDict, "Y", Ylist ); Py_DECREF(Ylist);
#endif

  //**/ ---------------------------------------------------------------
  //**/ create matlab code for displaying the main effects
  //**/ ---------------------------------------------------------------
  if (fp != NULL)
  {
    fwriteHold(fp, 0);
    fprintf(fp,"nx = ceil(sqrt(%d));", nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      fprintf(fp,"subplot(nx, nx, %d)\n", ii+1);
      if (plotScilab())
        fprintf(fp,"Y = matrix(Y%d_%d,nReps%d,nSymbols%d);\n",
                whichOutput,ii,ii,ii);
      else
        fprintf(fp,"Y = reshape(Y%d_%d,nReps%d,nSymbols%d);\n",
                whichOutput,ii,ii,ii);
      fprintf(fp,"ymax = max(Y%d_%d);\n", whichOutput, ii);
      fprintf(fp,"ymin = min(Y%d_%d);\n", whichOutput, ii);
      fprintf(fp,"xmax = max(X%d);\n", ii);
      fprintf(fp,"xmin = min(X%d);\n", ii);
      if (plotScilab())
      {
        fprintf(fp,"a = get(\"current_axes\");\n");
        fprintf(fp,"a.data_bounds=[xmin,ymin;xmax,ymax];\n");
      }
      else
      {
        fprintf(fp,"axis([xmin xmax ymin ymax])\n");
      }
      fprintf(fp,"for k = 1 : nSymbols%d \n",ii);
      fprintf(fp,"   yt = Y(:, k);\n");
      fprintf(fp,"   mean = sum(yt) / nReps%d;\n",ii);
      fprintf(fp,"   x = X%d(k) * ones(nReps%d, 1);\n",ii,ii);
      fprintf(fp,"   plot(x, yt, '.', 'MarkerSize', 1)\n");
      fprintf(fp,"   if ( k == 1 )\n");
      fwriteHold(fp, 1);
      fprintf(fp,"   end;\n");
      //**/fprintf(fp, "   plot(x(1), mean, 'r*', 'MarkerSize', 12)\n");
      fprintf(fp,"   meanArray(k) = mean;\n");
      fprintf(fp,"   tmaxArray(k) = max(yt);\n");
      fprintf(fp,"   tminArray(k) = min(yt);\n");
      fprintf(fp,"end;\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, "Input");
      fwritePlotYLabel(fp, "Output");
      fprintf(fp, "meanMean = sum(meanArray)/nSymbols%d;\n",ii);
      fprintf(fp, "adjMean  = (meanArray - meanMean);\n");
      fprintf(fp,"varMean = sum(adjMean.^2)/nSymbols%d\n",ii);
      fprintf(fp,"[xx,I] = sort(X%d);\n", ii);
      fprintf(fp,"yy1 = meanArray(I);\n");
      fprintf(fp,"yy2 = tmaxArray(I);\n");
      fprintf(fp,"yy3 = tminArray(I);\n");
      fprintf(fp,"plot(xx, yy1, 'r-', 'lineWidth', 1)\n");
      fprintf(fp,"plot(xx, yy2, 'r-', 'lineWidth', 1)\n");
      fprintf(fp,"plot(xx, yy3, 'r-', 'lineWidth', 1)\n");
      fprintf(fp,"title('Output %d vs Input %d')\n",whichOutput+1,ii+1);
      fprintf(fp,"disp('The center red lines are the conditional means.')\n");
      fprintf(fp,"disp('Conditional means show trends wrt each input.')\n");
      fprintf(fp,
        "disp('The upper/lower red lines are upper/lower envelopes.')\n");
      fwriteHold(fp, 0);
    }
    fclose(fp);
    printOutTS(PL_INFO, 
         "Main effect matlab plot %s has been generated.\n", meFileName);
  }

  //**/ ---------------------------------------------------------------
  //**/ bootstrapping runs
  //**/ ---------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO, 
       "Bootstrap analysis takes the sample, replicates it n times,\n");
    printOutTS(PL_INFO, 
       "and assess whether the sensitivity indices have converged.\n");
    printOutTS(PL_INFO, 
       "If you are performed iterative analysis with refinements,\n");
    printOutTS(PL_INFO, 
       "you will need to enter 'no index reuse' below at the first\n");
    printOutTS(PL_INFO, 
       "iteration and 'yes' afterward until the final refinement.\n");
    snprintf(pString,100,"Perform bootstrap main effect analysis? (y or n) ");
    getString(pString, winput1);
    ncount = 0;
    if (winput1[0] == 'y')
    {
      snprintf(pString,100,"Number of bootstrap samples to use (>=100): ");
      ncount = getInt(100, 2000, pString);
      psMatrix MatBS;
      MatBS.setFormat(PS_MAT2D);
      MatBS.setDim(ncount, nInputs);
      double **bsVCEs = MatBS.getMatrix2D();
      VecXT.setLength(nInputs*nSamples);
      VecYT.setLength(nSamples);
      nReplications = nSamples / nSubSamples;
      winput1[0] = 'n';
      fp1 = fopen(".ME_bootstrap_indset", "r");
      if (fp1 != NULL)
      {
        printOutTS(PL_INFO, ".ME_bootstrap_indset file found.\n");
        snprintf(pString,100,"Re-use file? (y or n) ");
        getString(pString, winput1);
        if (winput1[0] == 'y')
        {
          fscanf(fp1, "%d", &ii);
          if (ii != nReplications*ncount)
          {
            printOutTS(PL_ERROR,"ERROR: expect the first line to be %d.\n",
                   nReplications*ncount);
            printOutTS(PL_ERROR, 
                 "       Instead found the first line to be %d.n",ii);
            exit(1);
          }
        }
        else
        {
          fclose(fp1);
          fp1 = fopen(".ME_bootstrap_indset", "w");
          if (fp1 == NULL)
          {
            printOutTS(PL_ERROR, 
                 "ERROR: cannot open ME_bootstrap_indset file.\n");
            exit(1);
          }
          fprintf(fp1, "%d\n", nReplications*ncount);
        }
      }
      else
      {
        fp1 = fopen(".ME_bootstrap_indset", "w");
        if (fp1 == NULL)
        {
          printOutTS(PL_ERROR, 
               "ERROR: cannot open .ME_bootstrap_indset file.\n");
          exit(1);
        }
        fprintf(fp1, "%d\n", nReplications*ncount);
      }
      fp = NULL;
      for (ii = 0; ii < ncount; ii++)
      {
        //**/ randomly select bootstrap sample
        for (int ir = 0; ir < nReplications; ir++)
        {
          if (fp1 != NULL && winput1[0] == 'y')
          {
            fscanf(fp1, "%d\n", &index);
            if (index < 0 || index >= nReplications)
            {
              printOutTS(PL_ERROR, 
                 "ERROR: reading index from file .ME_bootstrap_indset\n");
              printOutTS(PL_ERROR,"       index read = %d\n", index);
              printOutTS(PL_ERROR,"       expected   = [0,%d]\n", 
                         nReplications-1);
            }
          }
          else index = PSUADE_rand() % nReplications;

          if (fp1 != NULL && winput1[0] != 'y')
            fprintf(fp1, "%d\n", index);
	  // Bill Oliver range check
	  if((index*nSubSamples + nSubSamples - 1) >= nSamples)
          {
	    printOutTS(PL_ERROR, "Buffer overflow in file %s line %d\n",
                 __FILE__, __LINE__);
	    exit(1);
	  }
          for (ss = 0; ss < nSubSamples; ss++)
          { 
            for (jj = 0; jj < nInputs; jj++)
              VecXT[(ir*nSubSamples+ss)*nInputs+jj] =
                      X[(index*nSubSamples+ss)*nInputs+jj];
            VecYT[ir*nSubSamples+ss] = VecY[index*nSubSamples+ss];
          }
        }
        computeMeanVariance(nInputs,1,nSamples,VecYT.getDVector(),
                            &aMean,&aVariance,0);
        computeVCE(nInputs,nSamples,nSubSamples,VecXT.getDVector(),
             VecYT.getDVector(),iZero,fp,VecVarVCEMean.getDVector(),
             VecMeanVCEVar.getDVector(), VecVarVCEVar.getDVector(), 
             VecVCE.getDVector());
        for (jj = 0; jj < nInputs; jj++) 
          bsVCEs[ii][jj] = VecVCE[jj]/aVariance;
      }
      if (fp1 != NULL) fclose(fp1);
      printf("Enter name of matlab file to store bootstrap info: ");
      scanf("%s", winput1);
      fgets(winput2,500,stdin);
      fp = fopen(winput1, "w");
      if (fp != NULL)
      {
        strcpy(winput2,"bootstrap sample of main effects");
        fwriteComment(fp, winput2);
        strcpy(winput2,
               "VCE(i,j) = VCE for input i, bootstrapped sample j");
        fwriteComment(fp, winput2);
        fprintf(fp, "VCE = zeros(%d,%d);\n",nInputs,ncount);
        for (ii = 0; ii < ncount; ii++)
           for (jj = 0; jj < nInputs; jj++)
              fprintf(fp, "VCE(%d,%d) = %e;\n",jj+1,ii+1,
                      bsVCEs[ii][jj]);
        fprintf(fp, "nn = %d;\n", nInputs);
        if (ioPtr != NULL) ioPtr->getParameter("input_names",pdata);
        if (pdata.strArray_ != NULL) inputNames = pdata.strArray_;
        else                         inputNames = NULL;
        if (inputNames == NULL)
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++) fprintf(fp,"'X%d',",ii+1);
          if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
          else              fprintf(fp,"'X%d'};\n",nInputs);
        }
        else
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++)
          {
            if (inputNames[ii] != NULL) fprintf(fp,"'%s',",inputNames[ii]);
            else                        fprintf(fp,"'X%d',",ii+1);
          }
          if (plotScilab())  
          {
            if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'];\n",nInputs);
          }
          else
          {
            if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'};\n",nInputs);
          }
        }
        fwriteHold(fp,0);
        fprintf(fp,"ymin = 0.0;\n");
        fprintf(fp,"ymax = max(max(VCE));\n");
        fprintf(fp,"hh = 0.05 * (ymax - ymin);\n");
        fprintf(fp,"ymax = ymax + hh;\n");
        fprintf(fp,"VMM = sum(VCE')/%d;\n",ncount);
        fprintf(fp,"VMA = max(VCE');\n");
        fprintf(fp,"VMI = min(VCE');\n");
        fprintf(fp,"bar(VMM,0.8);\n");
        fwriteHold(fp,1);
        fprintf(fp,"for ii = 1 : nn\n");
        fprintf(fp,"  XX = [ii ii];\n");
        fprintf(fp,"  YY = [VMI(ii)  VMA(ii)];\n");
        fprintf(fp,"  plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
        fprintf(fp,"'k','MarkerFaceColor','g','MarkerSize',13)\n");
        fprintf(fp,"end;\n");
        fwritePlotAxes(fp);
        if (plotScilab())
        {
          fprintf(fp,"a=gca();\n");
          fprintf(fp,"a.data_bounds=[0, ymin; nn+1, ymax];\n");
          fprintf(fp,"newtick = a.x_ticks;\n");
          fprintf(fp,"newtick(2) = [1:nn]';\n");
          fprintf(fp,"newtick(3) = Str';\n");
          fprintf(fp,"a.x_ticks = newtick;\n");
          fprintf(fp,"a.x_label.font_size = 3;\n");
          fprintf(fp,"a.x_label.font_style = 4;\n");
        }
        else
        {
          fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
          fprintf(fp,"set(gca,'XTickLabel',[]);\n");
          fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),");
          fprintf(fp,"Str,'HorizontalAlignment','left','rotation',90);\n");
          fprintf(fp,"set(th, 'fontsize', 12)\n");
          fprintf(fp,"set(th, 'fontweight', 'bold')\n");
        }
        fwritePlotTitle(fp,"Bootstrapped Sobol first  Order Indices");
        fwritePlotYLabel(fp,"Sobol Indices");
        fclose(fp);
      }
      else
      {
        printOutTS(PL_ERROR, "ERROR: cannot open file %s\n", winput1);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate matlab (printLevel < 0 when rsmeb is called)
  //**/ ---------------------------------------------------------------
  if (printLevel >= 0)
    printResults(nInputs, aVariance, VecVCE.getDVector(), ioPtr);

  //**/ ---------------------------------------------------------------
  //**/ cleaning up
  //**/ ---------------------------------------------------------------
  if (constrPtr != NULL) delete constrPtr;

  // return 1.0 to facilitate continuous refinement
  return 1.0;
}

// *************************************************************************
// Compute agglomerated mean and variance
// -------------------------------------------------------------------------
int MainEffectAnalyzer::computeMeanVariance(int nInputs, int nOutputs, 
                        int nSamples, double *Y, double *aMean, 
                        double *aVariance, int outputID)
{
  int    ss, count;
  double mean, variance;

  count = 0;
  mean = 0.0;
  for (ss = 0; ss < nSamples; ss++)
  {
    if (Y[nOutputs*ss+outputID] < 0.9*PSUADE_UNDEFINED)
    {
      mean += Y[nOutputs*ss+outputID];
      count++;
    }
  }
  if (count <= 0)
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: no valid data.\n");
    exit(1);
  }
  mean /= (double) count;
  variance = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    if (Y[nOutputs*ss+outputID] < 0.9*PSUADE_UNDEFINED)
    {
      variance += ((Y[nOutputs*ss+outputID] - mean) *
                   (Y[nOutputs*ss+outputID] - mean));
    }
  }
  variance /= (double) count;
  (*aMean) = mean;
  (*aVariance) = variance;
  return 0;
}

// *************************************************************************
// Compute VCE 
// -------------------------------------------------------------------------
int MainEffectAnalyzer::computeVCE(int nInputs,int nSamples,int nSubSamples, 
        double *X, double *Y, int whichOutput, FILE *fp, double *varVCEMean, 
        double *meanVCEVar, double *varVCEVar, double *vce)
{
  int    ii, ss, nReplications, nSubs, ncount, symIndex, repID, subID;
  int    validnReps, validnSubs, totalCnt;
  double fixedInput, aMean;
  psVector  VecTX, VecTY, VecVceMean, VecVceVar;;
  psIVector VecBins;

  VecBins.setLength(nSubSamples);
  VecTX.setLength(nSamples);
  VecTY.setLength(nSamples);
  VecVceMean.setLength(nSubSamples);
  VecVceVar.setLength(nSubSamples);

  //**/ ---------------------------------------------------------------
  //**/ calculate statistics for each input
  //**/ ---------------------------------------------------------------
  int nFilled;
  for (ii = 0; ii < nInputs; ii++)
  {
    //**/ copy the input and output into temporary arrays and sort x
    if (fp != NULL) fprintf(fp, "Y%d_%d = [\n",whichOutput,ii);

    for (ss = 0; ss < nSamples; ss++)
    {
      VecTX[ss] = X[nInputs*ss+ii];
      VecTY[ss] = Y[ss];
    }
    sortDbleList2(nSamples, VecTX.getDVector(), VecTY.getDVector());
    nReplications = 1;
    for (ss = 1; ss < nSamples; ss++)
    {
      if (VecTX[ss] == VecTX[0]) nReplications++;
      else                       break;
    }
    nSubs = nSamples / nReplications;

    //**/ compute quantities to obtain vce information
    ncount = 0;
    for (ss = 0; ss < nSamples; ss+=nReplications)
    {
      symIndex = ss / nReplications;
      fixedInput = VecTX[ss];
      VecVceMean[symIndex] = 0.0;
      validnReps = 0;
      for (repID = 0; repID < nReplications; repID++)
      {
        if (PABS(VecTX[ss+repID] - fixedInput) < 1.0E-12)
        {
          ncount++;
          if (VecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
          {
            validnReps++;
            VecVceMean[symIndex] += VecTY[ss+repID];
            if (fp != NULL)
              fprintf(fp,"%24.16e\n",VecTY[ss+repID]);
          }
        }
      }
      VecBins[symIndex] = validnReps;
      if (validnReps > 0) VecVceMean[symIndex] /= (double) validnReps;
      else                VecVceMean[symIndex] = PSUADE_UNDEFINED;
      VecVceVar[symIndex] = 0.0;
      for (repID = 0; repID < nReplications; repID++)
      {
        if (VecVceMean[symIndex] != PSUADE_UNDEFINED &&
            VecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
        {
          VecVceVar[symIndex] += 
                 ((VecTY[ss+repID]-VecVceMean[symIndex])*
                  (VecTY[ss+repID]-VecVceMean[symIndex]));
        }
      }
      if (validnReps > 0)
           VecVceVar[symIndex] /= (double) validnReps;
      else VecVceVar[symIndex] = PSUADE_UNDEFINED;
    }
    if (fp != NULL)
    { 
      fprintf(fp,"];\n");
      fprintf(fp, "nSymbols%d = %d;\n", ii, nSubs);
      fprintf(fp, "nReps%d    = %d;\n", ii, nReplications);
    }

    //**/ error checking
    if (ncount != nSamples)
    {
      printf("MainEffect ERROR: not rLHS, input %d\n", ii+1);
      printf("       error data = %d (%d)\n",ncount, nSamples);
      printf("       Did you use rLHS ?\n");
      return -1;
    }

    nFilled = 0;
    totalCnt = 0;
    for (subID = 0; subID < nSubs; subID++) 
    {
      if (VecBins[subID] > 0) 
      {
        totalCnt += VecBins[subID];
        nFilled++;
      }
    }
    //printf("(INFO) Input %4d: %d out of %d bins populated.\n",
    //       ii+1,nFilled,nSubs);
    varVCEMean[ii] = 0.0;
    meanVCEVar[ii] = 0.0;
    aMean = 0.0;
    validnSubs = 0;
    for (subID = 0; subID < nSubs; subID++)
    {
      if (VecVceVar[subID] != PSUADE_UNDEFINED)
      {
        aMean += (VecVceMean[subID] / totalCnt * VecBins[subID]);
        validnSubs++;
      }
    }
    for (subID = 0; subID < nSubs; subID++)
    {
      if (VecVceVar[subID] != PSUADE_UNDEFINED)
      {
        varVCEMean[ii] += ((VecVceMean[subID]-aMean)*
                  (VecVceMean[subID]-aMean)*VecBins[subID]/totalCnt);
        meanVCEVar[ii] += VecVceVar[subID] * VecBins[subID] / totalCnt;
      }
    }
    varVCEVar[ii] = 0.0;
    for (subID = 0; subID < nSubs; subID++)
    {
      if (VecVceVar[subID] != PSUADE_UNDEFINED)
        varVCEVar[ii] += ((VecVceVar[subID] - meanVCEVar[ii]) *
                          (VecVceVar[subID] - meanVCEVar[ii]) * 
                           VecBins[subID] / totalCnt);
    }

    //**/ Sept 2009: somehow the non-adjusted form works better
    //**/            since validnSubs can be very small after pruning
    //**/vce[ii] = varVCEMean[ii] - meanVCEVar[ii]/((double) validnSubs);
    //**/ Sept 2010: revert to adjusted form (rs_qsa confirms this)
    if (validnSubs > 0) vce[ii] = varVCEMean[ii];
    else                vce[ii] = 0.0;
  }
  return 0;
}

// *************************************************************************
// Compute VCE (by subdividing into bins)
// -------------------------------------------------------------------------
int MainEffectAnalyzer::computeVCECrude(int nInputs, int nSamples, 
                          double *X, double *Y, double *iLowerB, 
                          double *iUpperB, double aVariance, double *vce)
{
  int    ii, ss, nSize, index, totalCnt, nIntervals;
  double aMean, ddata, hstep;
  char   pString[500];
  psIVector VecBins, VecTags;
  psVector  VecVceMean, VecVceVar;

  //**/ ---------------------------------------------------------------
  //**/ set up
  //**/ ---------------------------------------------------------------
  nInputs_ = nInputs;
  if (nSize < 10) nSize = 10;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*                Crude Main Effect\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
      "* For small to moderate sample sizes, this method gives ");
    printOutTS(PL_INFO, 
      "rough estimate\n");
    printOutTS(PL_INFO, 
      "* of main effect (first order sensitivity).  These estimates ");
    printOutTS(PL_INFO, 
      "can vary\n");
    printOutTS(PL_INFO, 
      "* with different choices of internal settings.  For example, you ");
    printOutTS(PL_INFO, 
      "can\n");
    printOutTS(PL_INFO, 
      "* try different number of levels to assess the computed main effect\n");
    printOutTS(PL_INFO, 
      "* measures with respect to the it.\n");
    printOutTS(PL_INFO, 
      "* Turn on analysis expert mode to change the settings.\n");
  }
  nIntervals = (int) sqrt(1.0 * nSamples);
  nSize = nSamples / nIntervals;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"* MainEffect: number of levels   = %d\n", nIntervals);
    printOutTS(PL_INFO,"* MainEffect: sample size/levels = %d\n", nSize);
  }
  if (psConfig_.AnaExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100,"number of levels (>5, default = %d): ", nIntervals);
    nIntervals = getInt(5, nSamples/2, pString);
    nSize = nSamples / nIntervals;
  }
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO, 0);
  VecBins.setLength(nIntervals);
  VecTags.setLength(nSamples);
  VecVceMean.setLength(nIntervals);
  VecVceVar.setLength(nIntervals);

  //**/ ---------------------------------------------------------------
  //**/ calculate statistics for each input
  //**/ ---------------------------------------------------------------
  int nFilled;
  VecInputVCE_.setLength(nInputs_);
  for (ii = 0; ii < nInputs; ii++)
  {
    hstep = (iUpperB[ii] - iLowerB[ii]) / nIntervals;
    for (ss = 0; ss < nIntervals; ss++)
    {
      VecVceMean[ss] = 0.0;
      VecVceVar[ss] = 0.0;
      VecBins[ss] = 0;
    }
    //**/ binning
    for (ss = 0; ss < nSamples; ss++)
    {
      ddata = (X[nInputs*ss+ii] - iLowerB[ii]) / hstep;
      index = (int) ddata;
      if (index <  0) index = 0;
      if (index >= nIntervals) index = nIntervals - 1;
      VecTags[ss] = -1;
      if (Y[ss] < 0.9*PSUADE_UNDEFINED)
      {
        VecVceMean[index] += Y[ss];
        VecBins[index]++;
        VecTags[ss] = index;
      }
    }
    //**/ compute mean for each bin
    for (ss = 0; ss < nIntervals; ss++)
      if (VecBins[ss] > 0) VecVceMean[ss] /= (double) VecBins[ss];
    //**/ compute variance for each bin
    for (ss = 0; ss < nSamples; ss++)
    {
      index = VecTags[ss];
      if (index >= 0 && Y[ss] < 0.9*PSUADE_UNDEFINED)
      {
        VecVceVar[index] += ((Y[ss]-VecVceMean[index])*
                             (Y[ss]-VecVceMean[index]));
      }
    }
    for (ss = 0; ss < nIntervals; ss++)
    {
      if (VecBins[ss] > 0)
           VecVceVar[ss] /= (double) VecBins[ss];
      else VecVceVar[ss] = PSUADE_UNDEFINED;
    }
         
    //**/ compute VCE
    nFilled = 0;
    totalCnt = 0;
    for (ss = 0; ss < nIntervals; ss++) 
    {
      if (VecBins[ss] > 0) 
      {
        totalCnt += VecBins[ss];
        nFilled++;
      }
    }
    if (psConfig_.InteractiveIsOn())
      printf("(ME INFO) Input %4d: %d out of %d bins populated.\n",
             ii+1, nFilled,nIntervals);
    vce[ii] = 0.0;
    aMean = 0.0;
    for (ss = 0; ss < nIntervals; ss++)
    {
      if (VecVceVar[ss] != PSUADE_UNDEFINED)
        aMean += (VecVceMean[ss] / totalCnt * VecBins[ss]);
    }
    for (ss = 0; ss < nIntervals; ss++)
    {
      if (VecVceVar[ss] != PSUADE_UNDEFINED)
      {
        vce[ii] += ((VecVceMean[ss] - aMean) *
                    (VecVceMean[ss] - aMean) * VecBins[ss]/totalCnt);
      }
    }
    VecInputVCE_[ii] = vce[ii];
  }
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO,0);

  //**/ ---------------------------------------------------------------
  //**/ display VCE for each input
  //**/ ---------------------------------------------------------------
  ddata = 0.0;
  if (psConfig_.InteractiveIsOn()) 
  {
    if (aVariance == 0) printOutTS(PL_INFO, "Total VCE = %9.2e\n", ddata);
    else
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        ddata += vce[ii] / aVariance;
        printOutTS(PL_INFO, 
           "Input %4d, normalized 1st-order effect = %9.2e (raw = %9.2e)\n",
           ii+1, vce[ii]/aVariance, vce[ii]);
      }
      printOutTS(PL_INFO, "Total VCE = %9.2e\n", ddata);
    }
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    if (aVariance != 0) 
      for (ii = 0; ii < nInputs; ii++) ddata += vce[ii] / aVariance;
  }
  totalInputVCE_ = ddata;
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
MainEffectAnalyzer& MainEffectAnalyzer::operator=(const MainEffectAnalyzer &)
{
  printOutTS(PL_ERROR,"MainEffect operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// print result
// ------------------------------------------------------------------------
int MainEffectAnalyzer::printResults(int nInputs, double variance,
                                     double *mEffect, PsuadeData *ioPtr)
{
  int   ii;
  FILE  *fp;
  char  **iNames, pString[500];
  pData qData;

  if (ioPtr != NULL) ioPtr->getParameter("input_names", qData);
  if (qData.strArray_ != NULL) iNames = qData.strArray_;
  else                         iNames = NULL;
  if (variance == 0.0)
  {
    printOutTS(PL_WARN, "Total variance = 0. Hence, no main effect plot.\n");
    return 0;
  }
  //**/for (ii = 0; ii < nInputs; ii++)
  //**/{
  //**/   printOutTS(PL_ERROR, 
  //**/          "Input %4d: Sobol' first order sensitivity = %12.4e\n",
  //**/          ii+1,mEffect[ii]/variance);
  //**/}
  if (plotScilab()) fp = fopen("scilabme.sci", "w");
  else              fp = fopen("matlabme.m", "w");
  if (fp != NULL)
  {
    strcpy(pString," This file contains Sobol' first order indices");
    fwriteComment(fp, pString);
    strcpy(pString," set sortFlag = 1 and set nn to be the number");
    fwriteComment(fp, pString);
    strcpy(pString," of inputs to display.");
    fwriteComment(fp, pString);
    fprintf(fp, "sortFlag = 0;\n");
    fprintf(fp, "nn = %d;\n", nInputs);
    fprintf(fp, "Mids = [\n");
    for (ii = 0; ii < nInputs; ii++)
      fprintf(fp,"%24.16e\n", mEffect[ii]/variance);
    fprintf(fp, "];\n");
    if (iNames == NULL)
    {
      if (plotScilab()) fprintf(fp, "Str = [");
      else              fprintf(fp, "Str = {");
      for (ii = 0; ii < nInputs-1; ii++) fprintf(fp,"'X%d',",ii+1);
      if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
      else              fprintf(fp,"'X%d'};\n",nInputs);
    }
    else
    {
      if (plotScilab()) fprintf(fp, "Str = [");
      else              fprintf(fp, "Str = {");
      for (ii = 0; ii < nInputs-1; ii++)
      {
        if (iNames[ii] != NULL) fprintf(fp,"'%s',",iNames[ii]);
        else                    fprintf(fp,"'X%d',",ii+1);
      }
      if (plotScilab()) 
      {
        if (iNames[nInputs-1] != NULL)
             fprintf(fp,"'%s'];\n",iNames[nInputs-1]);
        else fprintf(fp,"'X%d'];\n",nInputs);
      }
      else
      {
        if (iNames[nInputs-1] != NULL)
             fprintf(fp,"'%s'};\n",iNames[nInputs-1]);
        else fprintf(fp,"'X%d'};\n",nInputs);
      }
    }
    fwritePlotCLF(fp);
    fprintf(fp, "if (sortFlag == 1)\n");
    if (plotScilab()) 
         fprintf(fp, "  [Mids, I2] = gsort(Mids);\n");
    else fprintf(fp, "  [Mids, I2] = sort(Mids,'descend');\n");
    fprintf(fp, "  Str  = Str(I2);\n");
    fprintf(fp, "  I2 = I2(1:nn);\n");
    fprintf(fp, "  Mids = Mids(1:nn);\n");
    fprintf(fp, "  Str  = Str(1:nn);\n");
    fprintf(fp, "end\n");
    fprintf(fp, "ymin = min(Mids);\n");
    fprintf(fp, "ymin = 0.0;\n");
    fprintf(fp, "ymax = max(Mids);\n");
    fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
    fprintf(fp, "bar(Mids,0.8);\n");
    fwritePlotAxes(fp);
    if (plotScilab())
    {
      fprintf(fp,"a=gca();\n");
      fprintf(fp,"a.data_bounds=[0, ymin; nn+1, ymax];\n");
      fprintf(fp,"newtick = a.x_ticks;\n");
      fprintf(fp,"newtick(2) = [1:nn]';\n");
      fprintf(fp,"newtick(3) = Str';\n");
      fprintf(fp,"a.x_ticks = newtick;\n");
      fprintf(fp,"a.x_label.font_size = 3;\n");
      fprintf(fp,"a.x_label.font_style = 4;\n");
    }
    else
    {
      fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
      fprintf(fp,"set(gca,'XTickLabel',[]);\n");
      fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),Str,");
      fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp,"set(th, 'fontsize', 12)\n");
      fprintf(fp,"set(th, 'fontweight', 'bold')\n");
    }
    fwritePlotTitle(fp,"Sobol First Order Indices");
    fwritePlotYLabel(fp,"Sobol Indices");
    fprintf(fp,"disp('Switch sortFlag to display ranked Sobol indices')\n");
    fwriteHold(fp, 0);
    fclose(fp);
    if (plotScilab()) 
         printOutTS(PL_INFO, "MainEffect plot matlab file = scilabme.sci\n");
    else printOutTS(PL_INFO, "MainEffect plot matlab file = matlabme.m\n");
    return 0;
  }
  else
  {
    printOutTS(PL_ERROR,"MainEffect ERROR: cannot create matlabme.m file.\n");
    return 0;
  }
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int MainEffectAnalyzer::get_nInputs()
{
  return nInputs_;
}
int MainEffectAnalyzer::get_outputID()
{
  return outputID_;
}
double *MainEffectAnalyzer::get_inputVCE()
{
  double* retVal = NULL;
  if (VecInputVCE_.length() > 0)
  {
    psVector VecVals;
    VecVals = VecInputVCE_;
    retVal = VecVals.takeDVector();
  }
  return retVal;
}
double MainEffectAnalyzer::get_totalInputVCE()
{
  return totalInputVCE_;
}
double MainEffectAnalyzer::get_mainEffectMean()
{
  return mainEffectMean_;
}
double MainEffectAnalyzer::get_mainEffectStd()
{
  return mainEffectStd_;
}

