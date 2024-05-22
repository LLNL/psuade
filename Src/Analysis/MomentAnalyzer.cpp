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
// Functions for the class MomentAnalyzer  
// Reference: from any introductory statistics book
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "MomentAnalyzer.h"
#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "pData.h"
#include "RSConstraints.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
MomentAnalyzer::MomentAnalyzer() : Analyzer(), nSamples_(0), nGroups_(0),
                                nInputs_(0), nOutputs_(0)
{
  setName("MOMENT");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MomentAnalyzer::~MomentAnalyzer()
{
  VecStats_.clean();
}

// ************************************************************************
// perform analysis (library call mode)
// ------------------------------------------------------------------------
void MomentAnalyzer::analyze(int nInps, int nSamp, double *lbs,
                             double *ubs, double *X, double *Y)
{
  aData adata;
  adata.nInputs_ = nInps;
  adata.nOutputs_ = 1;
  adata.nSamples_ = nSamp;
  adata.iLowerB_ = lbs;
  adata.iUpperB_ = ubs;
  adata.sampleInputs_ = X;
  adata.sampleOutputs_ = Y;
  adata.outputID_ = 0;
  adata.printLevel_ = 0;
  analyze(adata);
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double MomentAnalyzer::analyze(aData &adata)
{
  int    nInputs, nOutputs, nSamples, outputID, nSubSamples, status, nValid;
  int    nGroups, groupID, whichOutput, ss, printLevel, cnt, outID;
  int    errFlag, ii;
  double mean, variance, skewness, kurtosis, *XX;
  double *YY, *YT=NULL, variance2;
  char   winput[500], filename[500], pString[500];
  FILE   *fp=NULL;
  PsuadeData    *ioPtr;
  pData         pPtr;
  psVector vecGMeans, vecGVars, vecYT;

  // Providing initialization by Bill Oliver
  outID = 0;
  nValid = 1;  // nValid is set to 1 to avoid a divide by zero later

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  printLevel = adata.printLevel_;
  nInputs  = adata.nInputs_;
  nOutputs = adata.nOutputs_;
  nSamples = adata.nSamples_;
  XX       = adata.sampleInputs_;
  YY       = adata.sampleOutputs_;
  outputID = adata.outputID_;
  nSubSamples = adata.nSubSamples_;
  ioPtr       = adata.ioPtr_;
  if (adata.inputPDFs_ != NULL)
  {
    cnt = 0;
    for (ii = 0; ii < nInputs; ii++) cnt += adata.inputPDFs_[ii];
    if (psConfig_.InteractiveIsOn() && cnt > 0)
    {
      printOutTS(PL_INFO,"MomentAnalyzer INFO: non-uniform probability ");
      printOutTS(PL_INFO,"distributions have been\n");
      printOutTS(PL_INFO,"               defined in the data file, but ");
      printOutTS(PL_INFO,"they will not be used in\n");
      printOutTS(PL_INFO,"               this analysis.\n");
    }
  }
  if (ioPtr != NULL) ioPtr->getParameter("output_names", pPtr);

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR,
         "MomentAnalyzer ERROR: invalid nInputs or nOutputs.\n");
    printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
    return PSUADE_UNDEFINED;
  } 
  if (nSamples <= 1)
  {
    printOutTS(PL_ERROR,"MomentAnalyzer ERROR: nSamples should be > 1.\n");
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (printLevel > 0 && nSubSamples <= 0)
  {
    printOutTS(PL_INFO,"MomentAnalyzer INFO: nSubSamples = 0.\n");
    printOutTS(PL_INFO,"    This affects additional group analysis only.\n");
    printOutTS(PL_INFO,"    Set nSubSamples to nSamples.\n");
    nSubSamples = nSamples;
  } 
  if (psConfig_.InteractiveIsOn() && 
      (nSamples/nSubSamples*nSubSamples != nSamples))
  {
    printOutTS(PL_INFO,"MomentAnalyzer INFO: nSamples != k*nSubSamples.\n");
    printOutTS(PL_ERROR,"   nSamples    = %d\n", nSamples);
    printOutTS(PL_ERROR,"   nSubSamples = %d\n", nSubSamples);
    printOutTS(PL_INFO,"    Set nSubSamples to nSamples.\n");
    nSubSamples = nSamples;
  } 
  whichOutput = outputID;
  if (whichOutput < -1 || whichOutput >= nOutputs)
  {
    printOutTS(PL_ERROR,"MomentAnalyzer ERROR: invalid outputID (%d).\n",
               whichOutput+1);
    printOutTS(PL_ERROR, "   outputID = %d\n", outputID+1);
    return PSUADE_UNDEFINED;
  } 
  VecStats_.clean();
 
  //**/ set this to output 1 for python
  if (whichOutput < 0) whichOutput = 0;
  errFlag = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ii = 0; ii < nOutputs; ii++)
    {
      if (YY[ss*nOutputs+ii] == PSUADE_UNDEFINED)
      {
        errFlag++;
        break;
      }
    }
  }
  if (errFlag == 0) YT = YY;
  else
  {
    printOutTS(PL_INFO,"MomentAnalyzer INFO: Some outputs ");
    printOutTS(PL_INFO,"are UNDEFINED, which are to be purged\n");
    printOutTS(PL_INFO,"               before processing.\n");
    vecYT.setLength(nSamples*nOutputs);
    YT = vecYT.getDVector();

    cnt = 0;
    for (ss = 0; ss < nSamples; ss++)
    {
      for (ii = 0; ii < nOutputs; ii++)
        if (YY[ss*nOutputs+ii] == PSUADE_UNDEFINED) break;
      if (ii == nOutputs)
      {
        for (ii = 0; ii < nOutputs; ii++)
          YT[cnt*nOutputs+ii] = YY[ss*nOutputs+ii];
        cnt++;
      } 
    }
    if (nSubSamples == nSamples) nSubSamples = cnt;
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,
        "MomentAnalyzer INFO: Original sample size = %d\n",nSamples);
    nSamples = cnt;
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,
        "MomentAnalyzer INFO: Purged   sample size = %d\n",nSamples);
  }
  if (nSamples <= 0) 
  {
    printOutTS(PL_INFO,"INFO: Sample size = 0 ==> no analysis\n");
    return -1;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ get response surface filter information, if any
  //**/ process differently if there are filters
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr = NULL;
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
    int nConstr = constrPtr->genConstraints(ioPtr);
    if (nConstr == 0)
    {
      delete constrPtr;
      constrPtr = NULL;
    }
  }
  if (constrPtr != NULL)
  {
    if (outputID < 0 || outputID >= nOutputs)
    {
      printOutTS(PL_INFO,
           "MomentAnalyzer: Generate output distribution plot\n");
      snprintf(pString,100,
               "Enter output number (1 - %d) : ", nOutputs);
      outID = getInt(1, nOutputs, pString);
      outID--;
    }
    else outID = outputID;
    if (plotScilab()) strcpy(filename, "scilabua.sci");
    else              strcpy(filename, "matlabua.m");
    fp = fopen(filename, "w");
    if (fp == NULL)
      printOutTS(PL_ERROR,"ERROR: Cannot open file %s.\n",filename);
    psVector vecY2;
    vecY2.setLength(nSamples);
    nValid = 0;
    if (fp != NULL) fprintf(fp, "Y = [\n");
    for (ss = 0; ss < nSamples; ss++)
    {
      constrPtr->evaluate(&XX[ss*nInputs],YT[ss*nOutputs+outputID],
                            status);
      if (status == 1)
      {
        vecY2[nValid++] = YT[ss*nOutputs+outputID];
        if (fp != NULL) fprintf(fp, "  %24.16e \n",vecY2[nValid-1]);
      }
    }
    if (nValid == 0)
    {
      printOutTS(PL_INFO,
                 "MomentAnalyzer INFO: No valid data points.\n");
      delete constrPtr;
      if (fp != NULL) fclose(fp);
      return 0.0;
    }
    if (fp != NULL)
    {
      fprintf(fp, "];\n");
      if (plotScilab())
      {
        fwritePlotCLF(fp);
        fprintf(fp, "ymin = min(Y);\n");
        fprintf(fp, "ymax = max(Y);\n");
        fprintf(fp, "ywid = 0.1 * (ymax - ymin);\n");
        fprintf(fp, "if (ywid < 1.0e-12)\n");
        fprintf(fp, "   disp('range too small.')\n");
        fprintf(fp, "   halt\n");
        fprintf(fp, "end;\n");
        //fprintf(fp, "histplot(10, Y/ywid, style=2);\n");
        fprintf(fp, "histplot(10, Y, style=2);\n");
        fprintf(fp, "a = gce();\n");
        fprintf(fp, "a.children.fill_mode = \"on\";\n");
        fprintf(fp, "a.children.thickness = 2;\n");
        fprintf(fp, "a.children.foreground = 0;\n");
        fprintf(fp, "a.children.background = 2;\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Output Distribution");
        if (ioPtr != NULL && pPtr.strArray_ != NULL)
             fwritePlotXLabel(fp, pPtr.strArray_[outID]);
        else fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
      }
      else
      {
        fwritePlotCLF(fp);
        fprintf(fp, "[nk,xk]=hist(Y(:,1),10);\n");
        fprintf(fp, "NB = nk;\n");
        fprintf(fp, "N  = nk;\n");
        fprintf(fp, "X  = xk;\n");
        fprintf(fp, "bar(xk,nk/%d,1.0)\n",nValid);
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Output Distribution");
        if (ioPtr != NULL && pPtr.strArray_ != NULL)
             fwritePlotXLabel(fp, pPtr.strArray_[outID]);
        else fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
      }
      printOutTS(PL_INFO,"Output distribution plot is now in %s.\n",
                 filename);
      fclose(fp);
    }
    computeMeanVariance(nInputs,1,nValid,vecY2.getDVector(),&mean,
                        &variance,0);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*       Sample mean       = %12.4e\n",mean);
    printOutTS(PL_INFO,"*       Sample std dev    = %12.4e\n",
               sqrt(variance));
    printOutTS(PL_INFO,"*       Based on nSamples = %d\n",nValid);
    printOutTS(PL_INFO,"*       Output number     = %d\n", outID+1);
    delete constrPtr;
    return sqrt(variance/nValid);
  }

  //**/ ---------------------------------------------------------------
  //**/ allocate spaces for local variables
  //**/ ---------------------------------------------------------------
  if (nSubSamples <= 0) nSubSamples = nSamples;
  nGroups = nSamples / nSubSamples;
  vecGMeans.setLength(nGroups);
  vecGVars.setLength(nGroups);

  //**/ ---------------------------------------------------------------
  //**/ then compute statistics
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO, "\n");
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, "*             Basic Output Statistics\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, "* nSamples = %10d\n",nSamples);
    printOutTS(PL_INFO, "* nGroups  = %10d\n",nGroups);
    printOutTS(PL_INFO, "* nInputs  = %10d\n",nInputs);
    printOutTS(PL_INFO, "* nOutputs = %10d\n",nOutputs);
    printDashes(PL_INFO, 0);
  }
  if (outputID == -1)
  {
    for (ss = 0; ss < nOutputs; ss++)
    {
      if (psConfig_.InteractiveIsOn())
         printOutTS(PL_INFO, "* outputID = %10d\n", ss+1);
      computeMeanVariance(nInputs,nOutputs,nSubSamples, YT,
             vecGMeans.getDVector(),vecGVars.getDVector(),ss);
      if (psConfig_.InteractiveIsOn())
      {
        printOutTS(PL_INFO,"*       Sample mean          = %12.4e\n",
                   vecGMeans[0]);
        printOutTS(PL_INFO,"*       Sample std dev       = %12.4e\n",
                   sqrt(vecGVars[0]));
      }
      if (vecGVars[0] > 0)
      {
        computeSkewnessKurtosis(nInputs,nOutputs,nSubSamples, YT, 
                     sqrt(vecGVars[0]), &skewness,&kurtosis,ss);
        if (psConfig_.InteractiveIsOn())
        {
          printOutTS(PL_INFO,
             "*       Sample skewness      = %12.4e\n",skewness);
          printOutTS(PL_INFO,
             "*       Sample kurtosis      = %12.4e\n",kurtosis);
        }
      }
      else
      {
        if (psConfig_.InteractiveIsOn())
          printOutTS(PL_INFO,
             "*       Std dev=0, skeweness/kurtosis set to 0.\n");
        skewness = kurtosis = 0.0;
      }
      if (psConfig_.InteractiveIsOn() && ss < nOutputs-1 && 
        printLevel >= 0) printEquals(PL_INFO, 0);
    }
    variance = vecGVars[0];
  }
  else
  {
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,"* outputID = %10d\n", outputID+1);
    computeMeanVariance(nInputs,nOutputs,nSubSamples,YT,
           vecGMeans.getDVector(),vecGVars.getDVector(),outputID);
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,"*       Sample mean          = %12.4e\n",
                 vecGMeans[0]);
      printOutTS(PL_INFO,"*       Sample std dev       = %12.4e\n",
                 sqrt(vecGVars[0]));
    }
    if (vecGVars[0] > 0)
    {
      computeSkewnessKurtosis(nInputs,nOutputs,nSubSamples, YT, 
                   sqrt(vecGVars[0]), &skewness,&kurtosis,outputID);
      if (psConfig_.InteractiveIsOn())
      {
        printOutTS(PL_INFO,"*       Sample skewness      = %12.4e\n", 
                   skewness);
        printOutTS(PL_INFO,"*       Sample kurtosis      = %12.4e\n", 
                   kurtosis);
      }
    }
    else
    {
      if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,
           "*       Std dev=0, skeweness/kurtosis set to 0.\n");
      skewness = kurtosis = 0.0;
    }
    variance = vecGVars[0];

    pData *pPtr=NULL;
    if (ioPtr != NULL)
    {
      pPtr = ioPtr->getAuxData();
      if (pPtr != NULL)
      {
        pPtr->nDbles_ = 4;
        pPtr->dbleArray_ = new double[4];
        pPtr->dbleArray_[0] = vecGMeans[0];
        pPtr->dbleArray_[1] = sqrt(vecGVars[0]);
        pPtr->dbleArray_[2] = skewness;
        pPtr->dbleArray_[3] = kurtosis;
      }
    }
  }
  //save moments
  VecStats_.setLength(4);
  VecStats_[0] = vecGMeans[0];
  VecStats_[1] = sqrt(vecGVars[0]);
  VecStats_[2] = skewness;
  VecStats_[3] = kurtosis;
  if (outputID == -1) 
    printf("INFO: calls to get moments return only the last output info.\n");
  if (psConfig_.InteractiveIsOn() && printLevel >= 0)
  {
    printDashes(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }

#ifdef HAVE_PYTHON
  //**/ ---------------------------------------------------------------
  //**/ store data for python graphics
  //**/ ---------------------------------------------------------------
  PyObject *temp, *Ylist;

  //**/ Store nSamples,nGroups,nInputs,nOutputs, and output#, as python ints
  PyDict_SetItemString(AnalysisDataDict, "nSamples",
                       temp=PyInt_FromLong(nSamples) ); Py_DECREF(temp);
  PyDict_SetItemString(AnalysisDataDict, "nGroups",
                       temp=PyInt_FromLong(nGroups) ); Py_DECREF(temp);
  PyDict_SetItemString(AnalysisDataDict, "nInputs",
                       temp=PyInt_FromLong(nInputs) ); Py_DECREF(temp);
  PyDict_SetItemString(AnalysisDataDict, "nOutputs",
                       temp=PyInt_FromLong(nOutputs) ); Py_DECREF(temp);
  PyDict_SetItemString(AnalysisDataDict, "outputID",
                       temp=PyInt_FromLong(whichOutput) ); Py_DECREF(temp);

  //**/ Store mean, std dev, skewness, and kurtosis, as python floats
  PyDict_SetItemString(AnalysisDataDict, "mean",
                 temp=PyFloat_FromDouble(vecGMeans[0]) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "std dev",
                 temp=PyFloat_FromDouble(sqrt(vecGVars[0])) );
  Py_DECREF(temp);
  PyDict_SetItemString(AnalysisDataDict, "skewness",
                temp=PyFloat_FromDouble(skewness) ); Py_DECREF(temp);
  PyDict_SetItemString( AnalysisDataDict, "kurtosis",
                temp=PyFloat_FromDouble(kurtosis) ); Py_DECREF(temp);

  //**/ For each sample, store Y values for this outputID, in python lists
  //**/ Don't throw away undefined samples??
  Ylist = PyList_New(0);
  for (ss = 0; ss < nSamples; ss++)
  //if (YG[ii] != PSUADE_UNDEFINED) {
    PyList_Append(Ylist,
           temp=PyFloat_FromDouble(Y[nOutputs*ss+outputID]) );
    Py_DECREF(temp);
  //}
  PyDict_SetItemString(AnalysisDataDict, "Y",Ylist); Py_DECREF(Ylist);
#endif

  //**/ ---------------------------------------------------------------
  //**/ create matlab file 
  //**/ ---------------------------------------------------------------
  if (outputID < 0 || outputID >= nOutputs)
  {
    if (psConfig_.InteractiveIsOn())
    {
      printf("MomentAnalyzer: Generating output distribution plot\n");
      snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
      outID = getInt(1, nOutputs, pString);
      outID--;
    }
    else
    {
      outID = 0;
      printOutTS(PL_INFO,"Invalid output ID. Set to default output 1.\n");
    }
  }
  else outID = outputID;
  if (plotScilab()) strcpy(filename, "scilabua.sci");
  else              strcpy(filename, "matlabua.m");
  genPlot(filename,nInputs,nOutputs,nSamples,outID,YT,pPtr.strArray_); 

  //**/ ---------------------------------------------------------------
  //**/ process next group
  //**/ ---------------------------------------------------------------
  if (outputID >= 0)
  {
    if (nGroups == 1)
    {
      if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
      variance2 = vecGVars[0];
      if (psConfig_.InteractiveIsOn() && printLevel >= 0)
      {
        printOutTS(PL_INFO,"*       std error of mean       = %16.8e\n",
                sqrt(variance2/(double) nSamples));
        printAsterisks(PL_INFO, 0);
      }
      return sqrt(variance2/(double) nSamples);
    }
    double *dptr1 = vecGMeans.getDVector();
    double *dptr2 = vecGVars.getDVector();
    for (groupID = 0; groupID < nGroups; groupID++)
    {
      computeMeanVariance(nInputs,nOutputs,nSubSamples,
                &YT[nSubSamples*groupID*nOutputs],
                &dptr1[groupID],&dptr2[groupID],whichOutput);
    }
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO, "Output %d\n", whichOutput+1);

    mean = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      mean += vecGVars[groupID];
    mean = mean / (double) nGroups;
    variance2 = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      variance2 += ((vecGVars[groupID] - mean) *
                    (vecGVars[groupID] - mean));
    variance2 = variance2 / (double) nGroups;

    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,
           "*       Mean of group variances = %16.8e\n",mean);
      printOutTS(PL_INFO, 
           "*       Std error of variance   = %16.8e\n",variance2);
    }
    mean = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      mean += vecGMeans[groupID];
    mean = mean / (double) nGroups;
    variance2 = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      variance2 += ((vecGMeans[groupID] - mean) * 
                    (vecGMeans[groupID] - mean));
    variance2 = variance2 / (double) nGroups;
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,
          "*       Mean of group means     = %16.8e\n",mean);
      printOutTS(PL_INFO,
          "*       Std error of mean       = %16.8e\n",variance2);
      printAsterisks(PL_INFO, 0);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ collective analysis
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && printLevel > 1 && outputID >= 0)
     analyzeMore(nInputs,nOutputs,nSamples,nSubSamples,YT,outputID);

  //**/ ---------------------------------------------------------------
  //**/ cleaning up
  //**/ ---------------------------------------------------------------
  return sqrt(variance/nValid);
}

// *************************************************************************
// Compute agglomerated mean and variance
// -------------------------------------------------------------------------
int MomentAnalyzer::computeMeanVariance(int nInputs, int nOutputs, 
                        int nSamples, double *Y, double *mean, 
                        double *variance, int outputID)
{
  int    ss;
  double meanTmp, varTmp;

  meanTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++) meanTmp += Y[nOutputs*ss+outputID];
  meanTmp /= (double) nSamples;
  varTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++)
  {
    varTmp += ((Y[nOutputs*ss+outputID] - meanTmp) *
               (Y[nOutputs*ss+outputID] - meanTmp));
  }
  varTmp /= (double) (nSamples - 1);
  (*mean)     = meanTmp;
  (*variance) = varTmp;
  return 0;
}

// *************************************************************************
// Compute skewness and kurtosis 
// kurtois of a standard normal distribution is 3.
// kurtois-3 measures excess kurtosis
// -------------------------------------------------------------------------
int MomentAnalyzer::computeSkewnessKurtosis(int nInputs, int nOutputs, 
                        int nSamples, double *Y, double stdev,
                        double *skewness, double *kurtosis, int outputID)
{
  int    ss;
  double meanTmp, skewTmp, kurTmp;

  meanTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++) meanTmp += Y[nOutputs*ss+outputID];
  meanTmp /= (double) nSamples;
  skewTmp = kurTmp = 0.0;
  for (ss = 0; ss < nSamples; ss++)
  {
    skewTmp += ((Y[nOutputs*ss+outputID] - meanTmp) *
                (Y[nOutputs*ss+outputID] - meanTmp) *
                (Y[nOutputs*ss+outputID] - meanTmp));
    kurTmp += ((Y[nOutputs*ss+outputID] - meanTmp) *
               (Y[nOutputs*ss+outputID] - meanTmp) *
               (Y[nOutputs*ss+outputID] - meanTmp) *
               (Y[nOutputs*ss+outputID] - meanTmp));
  }
  skewTmp /= (double) (nSamples - 1);
  kurTmp  /= (double) (nSamples - 1);
  if (stdev >= 0.0) (*skewness) = skewTmp / (stdev * stdev * stdev);
  else              (*skewness) = 0.0;
  if (stdev >= 0.0) (*kurtosis) = kurTmp / (stdev * stdev * stdev * stdev);
  else              (*kurtosis) = 0.0;
  return 0;
}

// ************************************************************************
// perform more analysis
// ------------------------------------------------------------------------
int MomentAnalyzer::analyzeMore(int nInputs, int nOutputs, int nSamples,
                                int nSubSamples, double *Y, int outputID)
{
  int    nGroups, groupID, myNSubSamples, nLevels=10, level, multiple;
  double mean, variance;
  psVector vecGMeans, vecGVars;

  //**/ ---------------------------------------------------------------
  //**/ allocate spaces for local variables
  //**/ ---------------------------------------------------------------
  for (level = 1; level <= nLevels; level++)
  {
    multiple = 1 << level;
    nGroups = nSamples / nSubSamples;
    if ((nGroups % multiple) != 0) break;
    myNSubSamples = nSubSamples * multiple;
    nGroups    = nSamples / myNSubSamples;
    vecGMeans.setLength(nGroups);
    vecGVars.setLength(nGroups);

    //**/ ------------------------------------------------------------
    //**/ then compute statistics
    //**/ ------------------------------------------------------------

    double *dptr1 = vecGMeans.getDVector();
    double *dptr2 = vecGVars.getDVector();
    for (groupID = 0; groupID < nGroups; groupID++)
    {
      computeMeanVariance(nInputs,nOutputs,myNSubSamples,
              &Y[myNSubSamples*groupID*nOutputs],
              &(dptr1[groupID]),&(dptr2[groupID]),outputID);
    }
    mean = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      mean += vecGMeans[groupID];
    mean = mean / (double) nGroups;
    variance = 0.0;
    for (groupID = 0; groupID < nGroups; groupID++)
      variance += ((vecGMeans[groupID] - mean) * 
                   (vecGMeans[groupID] - mean));
    variance = variance / (double) nGroups;
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*       Agglomerate %d groups into 1 \n",multiple);
    printOutTS(PL_INFO,
         "*       Mean of group means     = %16.8e\n",mean);
    printOutTS(PL_INFO,
         "*       Std error of mean       = %16.8e\n",variance);
    printEquals(PL_INFO, 0);
  }
  return 0;
}

// ************************************************************************
// perform more analysis
// ------------------------------------------------------------------------
int MomentAnalyzer::genPlot(char *fname, int nInputs, int nOutputs, 
                    int nSamples, int outID, double *Y, char **outNames) 
{
  int ss;
  FILE *fp = fopen(fname, "w");
  if (fp != NULL)
  {
    fprintf(fp, "Y = [\n");
    for (ss = 0; ss < nSamples; ss++)
      fprintf(fp, "  %24.16e\n", Y[nOutputs*ss+outID]);  
    fprintf(fp, "];\n");
    if (plotScilab())
    {
      fwritePlotCLF(fp);
      fprintf(fp, "ymin = min(Y);\n");
      fprintf(fp, "ymax = max(Y);\n");
      fprintf(fp, "ywid = 0.1 * (ymax - ymin);\n");
      fprintf(fp, "if (ywid < 1.0e-12)\n");
      fprintf(fp, "   disp('range too small.')\n");
      fprintf(fp, "   halt\n");
      fprintf(fp, "end;\n");
      //fprintf(fp, "histplot(10, Y/ywid, style=2);\n");
      fprintf(fp, "histplot(10, Y, style=2);\n");
      fprintf(fp, "a = gce();\n");
      fprintf(fp, "a.children.fill_mode = \"on\";\n");
      fprintf(fp, "a.children.thickness = 2;\n");
      fprintf(fp, "a.children.foreground = 0;\n");
      fprintf(fp, "a.children.background = 2;\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Output Distribution");
      if (outNames != NULL)
           fwritePlotXLabel(fp, outNames[outID]);
      else fwritePlotXLabel(fp, "Output Values");
      fwritePlotYLabel(fp, "Probabilities");
    }
    else
    {
      fwritePlotCLF(fp);
      fprintf(fp, "twoPlots = 0;\n");
      fprintf(fp, "if (twoPlots == 1)\n");
      fprintf(fp, "subplot(1,2,1)\n");
      fprintf(fp, "end;\n");
      if (nSamples > 500) fprintf(fp,"[nk,xk]=hist(Y(:,1),20);\n");
      else                fprintf(fp,"[nk,xk]=hist(Y(:,1),10);\n");
      fprintf(fp, "bar(xk,nk/%d,1.0)\n",nSamples);
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Probability Distribution");
      if (outNames != NULL)
           fwritePlotXLabel(fp, outNames[outID]);
      else fwritePlotXLabel(fp, "Output Value");
      fwritePlotYLabel(fp, "Probabilities");
      fprintf(fp, "if (twoPlots == 1)\n");
      fprintf(fp, "Yk = sort(Y(:,1));\n");
      fprintf(fp, "Xk = 1 : %d;\n", nSamples);
      fprintf(fp, "Xk = Xk / %d;\n", nSamples);
      fprintf(fp, "subplot(1,2,2)\n");
      fprintf(fp, "plot(Yk, Xk, 'lineWidth',3)\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Cumulative Distribution");
      if (outNames != NULL)
           fwritePlotXLabel(fp, outNames[outID]);
      else fwritePlotXLabel(fp, "Output Values");
      fwritePlotYLabel(fp, "Probabilities");
      fprintf(fp, "end;\n");
    }
    printOutTS(PL_INFO,
         "Output distribution plot is now in %s.\n",fname);
    fclose(fp);
  }
  else 
  {
    printOutTS(PL_ERROR,"ERROR: Cannot open file %s.\n", fname);
    return -1;
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
MomentAnalyzer& MomentAnalyzer::operator=(const MomentAnalyzer &)
{
  printOutTS(PL_ERROR, 
       "MomentAnalyzer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int MomentAnalyzer::get_nSamples()
{
  return nSamples_;
}
int MomentAnalyzer::get_nGroups()
{
  return nGroups_;
}
int MomentAnalyzer::get_nInputs()
{
  return nInputs_;
}
int MomentAnalyzer::get_nOutputs()
{
  return nOutputs_;
}
double MomentAnalyzer::get_mean()
{
  if (VecStats_.length() > 0) return VecStats_[0];
  else                        return 0.0;
}
double MomentAnalyzer::get_stdev()
{
  if (VecStats_.length() > 1) return VecStats_[1];
  else                        return 0.0;
}
double MomentAnalyzer::get_skewness()
{
  if (VecStats_.length() > 2) return VecStats_[2];
  else                        return 0.0;
}
double MomentAnalyzer::get_kurtosis()
{
  if (VecStats_.length() > 3) return VecStats_[3];
  else                        return 0.0;
}

