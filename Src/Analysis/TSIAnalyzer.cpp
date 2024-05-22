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
// Functions for the TSIAnalyzer class 
// AUTHOR : CHARLES TONG
// DATE   : 2020
// ************************************************************************
#include <stdio.h>
#include <assert.h>

#include "Psuade.h"
#include "TSIAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"

#ifdef HAVE_METIS
extern "C" 
{
void METIS_PartGraphRecursive(int *, int *, int *, int *, int *,
                              int *, int *, int *, int *, int *, int *);
}
#endif

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
TSIAnalyzer::TSIAnalyzer() : Analyzer()
{
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
TSIAnalyzer::~TSIAnalyzer()
{
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
double TSIAnalyzer::analyze(aData &adata)
{
  int    ss, ii, jj, inputID, nnz, itmp, jtmp, n1d, nAggrs, nFilled;
  int    count, graphN, index, totalCnt;
  double dvar, dtmp;
  char   pString[500];

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int printLevel_ = adata.printLevel_;
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *XIn  = adata.sampleInputs_;
  double *YIn  = adata.sampleOutputs_;
  int outputID = adata.outputID_;
  double *lbounds = adata.iLowerB_;
  double *ubounds = adata.iUpperB_;
  PsuadeData *ioPtr = adata.ioPtr_;
  if (ioPtr == NULL)
  {
    printOutTS(PL_ERROR, "TSIAnalyzer ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR,
         "Total Effect ERROR: invalid nInputs or nOutputs.\n");
    printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
    return -1;
  }  
  if (nSamples <= 1)
  {
    printOutTS(PL_ERROR,"Total Effect ERROR: nSamples should be > 1.\n");
    printOutTS(PL_ERROR,"   nSamples = %d\n", nSamples);
    return -1;
  }  
  if (nSamples < 10000)
  {
    printOutTS(PL_WARN,
         "Total Effect WARNING: nSamples may be too small to\n");
    printOutTS(PL_WARN,
         "             give results with acceptable accuracy.\n");
  }  
  int status = 0;
  for (ss = 0; ss < nSamples; ss++)
    if (YIn[nOutputs*ss+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"Total Effect ERROR: Some outputs are ");
    printOutTS(PL_ERROR,"UNDEFINED. Prune the UNDEFINED\n");
    printOutTS(PL_ERROR,"                    sample points first.\n");
    return PSUADE_UNDEFINED;
  }
  psVector vecYT;
  vecYT.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) 
    vecYT[ss] = YIn[ss*nOutputs+outputID];

  //**/ ---------------------------------------------------------------
  //**/ more checking
  //**/ ---------------------------------------------------------------
  psVector vecRanges;
  vecRanges.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    vecRanges[ii] = ubounds[ii] - lbounds[ii];
    if (vecRanges[ii] <= 0.0)
    {
      printOutTS(PL_ERROR,
            "Total Effect ERROR: lbound/ubound mismatch.\n");
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ compute mean and variance
  //**/ ---------------------------------------------------------------
  double dmean = 0.0;
  for (ss = 0; ss < nSamples; ss++) dmean += vecYT[ss];
  dmean /= (double) nSamples;
  double variance = 0.0;
  for (ss = 0; ss < nSamples; ss++)
    variance += ((vecYT[ss] - dmean) * (vecYT[ss] - dmean));
  variance /= (double) (nSamples - 1);
  printOutTS(PL_INFO,"Total Effect: Output mean     = %e\n",dmean);
  printOutTS(PL_INFO,"Total Effect: Output variance = %e\n",variance);

  //**/ ---------------------------------------------------------------
  //**/ set up mesh
  //**/ ---------------------------------------------------------------
  if (nInputs > 12)
  {
    printOutTS(PL_ERROR,
       "Total Effect ERROR: nInputs > 12 currently not supported.\n");
    exit(1);
  }
  if (nInputs == 1 ) n1d = 1000000;
  if (nInputs == 2 ) n1d = 2048;
  if (nInputs == 3 ) n1d = 200;
  if (nInputs == 4 ) n1d = 100;
  if (nInputs == 5 ) n1d = 20;
  if (nInputs == 6 ) n1d = 12;
  if (nInputs == 7 ) n1d = 8;
  if (nInputs == 8 ) n1d = 6;
  if (nInputs == 9 ) n1d = 5;
  if (nInputs == 10) n1d = 4;
  if (nInputs == 11) n1d = 3;
  if (nInputs == 12) n1d = 3;

  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*          Crude Total Sensitivity Indices\n");
  printEquals(PL_INFO,0);
  printOutTS(PL_INFO,
    "* Total Effect: number of subdomains          = %d\n",nSamples/50);
  printOutTS(PL_INFO,
    "* Total Effect: number of point per subdomain = 50\n");
  printDashes(PL_INFO,0);
  printOutTS(PL_INFO,
    "* NOTE: For small to moderate sample size, this method in general\n");
  printOutTS(PL_INFO,
    "*       gives rough estimates of total sensitivity.\n");
  printOutTS(PL_INFO,"* Recommendation: Try different numbers of ");
  printOutTS(PL_INFO,"subdomains to assess the\n");
  printOutTS(PL_INFO,"*   goodness of the measures. ");
  printOutTS(PL_INFO,"A rule of thumb for sample size per\n");
  printOutTS(PL_INFO,"*   subdomain is > 50.\n");
  printOutTS(PL_INFO,
     "* Turn on analysis expert mode to change internal settings.\n");
  printEquals(PL_INFO,0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    strcpy(pString,"Enter the number of subdomains (> 5): ");
    nAggrs = getInt(5,nSamples, pString);
  }
  else nAggrs = nSamples / 50;

  //**/ ---------------------------------------------------------------
  //**/ generate a graph from a (nInputs-1)-dimensional mesh 
  //**/ ---------------------------------------------------------------
  psIVector vecIncrs;
  vecIncrs.setLength(nInputs);
  graphN = 1;
  vecIncrs[0] = graphN;
  for (jj = 1; jj < nInputs; jj++)
  {
    graphN *= n1d;
    vecIncrs[jj] = graphN;
  }
  if (nAggrs > graphN)
  {
    nAggrs = graphN / 2;
    printOutTS(PL_INFO, 
       "* Total Effect: revised number of subdomains = %d\n", nAggrs);
  }

  psIVector vecIA, vecJA;
  vecIA.setLength(graphN+1);
  vecJA.setLength(graphN*(nInputs-1)*2+1);
  nnz = 0;
  vecIA[0] = nnz;
  for (ii = 0; ii < graphN; ii++)
  {
    itmp = ii;
    for (jj = 0; jj < nInputs-1; jj++)
    {
      jtmp = itmp % n1d;
      itmp = itmp / n1d;
      if (jtmp > 0    ) vecJA[nnz++] = ii - vecIncrs[jj];
      if (jtmp < n1d-1) vecJA[nnz++] = ii + vecIncrs[jj];
    }
    vecIA[ii+1] = nnz;
  }

  //**/ ----------------------------------------------------------------
  //**/ call metis to perform partitioning
  //**/ ----------------------------------------------------------------
  int wgtflag=0, numflag=0, edgeCut=0, options[10];
  psIVector vecCellsOccupied;
  vecCellsOccupied.setLength(graphN);
  options[0] = 0;
#ifdef HAVE_METIS
  METIS_PartGraphRecursive(&graphN,vecIA.getIVector(),vecJA.getIVector(), 
        NULL,NULL,&wgtflag,&numflag,&nAggrs,options,&edgeCut,
        vecCellsOccupied.getIVector());
#else
  printOutTS(PL_ERROR, "Total Effect ERROR : METIS not installed.\n");
  nInputs = 0;
#endif

  //**/ ----------------------------------------------------------------
  //**/ allocate temporary storage
  //**/ ----------------------------------------------------------------
  psIVector vecSam2Aggr, vecAggrCnts;
  psVector  vecAggrMeans, vecTSI;
  vecSam2Aggr.setLength(nSamples);
  vecAggrCnts.setLength(nAggrs);
  vecAggrMeans.setLength(nAggrs);
  vecTSI.setLength(nInputs);

  //**/ ----------------------------------------------------------------
  //**/ For each input i, compute 1 - V(E(X_i|X_{~i}))
  //**/ ----------------------------------------------------------------
  for (inputID = 0; inputID < nInputs; inputID++)
  {
    //**/ ----------------------------------------------------------------
    //**/ locate which aggregate each sample point belongs to
    //**/ ----------------------------------------------------------------
    for (ss = 0; ss < nSamples; ss++)
    {
      itmp = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (ii != inputID)
        {
          itmp = itmp * n1d;
          dtmp = XIn[ss*nInputs+ii];
          dtmp = (dtmp - lbounds[ii]) / vecRanges[ii];
          jtmp = (int) (dtmp * n1d);
          if (jtmp < 0) jtmp = 0;
          if (jtmp >= n1d) jtmp = n1d - 1;
          itmp += jtmp;
        }
      }
      if (vecCellsOccupied[itmp] < 0 || vecCellsOccupied[itmp] >= nAggrs)
      {
        printf("FATAL ERROR: mesh cell %d - assigned aggregate %d\n",
               itmp,vecCellsOccupied[itmp]);
        printf("      number number of mesh cells = %d\n", graphN);
        printf("      offending sample = %d\n", ss+1);
        printf("==> Consult PSUADE developers.\n");
        exit(1);
      }
      vecSam2Aggr[ss] = vecCellsOccupied[itmp];
    }

    //**/ ----------------------------------------------------------------
    //**/ processing
    //**/ ----------------------------------------------------------------
    for (ii = 0; ii < nAggrs; ii++)
    {
      vecAggrMeans[ii] = 0.0;
      vecAggrCnts[ii] = 0;
    }
    for (ss = 0; ss < nSamples; ss++)
    {
      index = vecSam2Aggr[ss];
      if (index < 0 || index >= nAggrs)
      {
        printf("FATAL ERROR: index = %d ([0,%d])\n",index,nAggrs);
        exit(1);
      }
      vecAggrMeans[index] += vecYT[ss];
      vecAggrCnts[index]++;
    }
    totalCnt = 0;
    int emptyCnt = 0;
    for (ii = 0; ii < nAggrs; ii++)
    {
      totalCnt += vecAggrCnts[ii];
      if (vecAggrCnts[ii] == 0) emptyCnt++;
    }
    if (totalCnt != nSamples)
    {
      printf("TSI ERROR: totalCnt %d != nSamples %d\n",totalCnt,nSamples);
      exit(1);
    }
    if (emptyCnt > 0)
    {
      printOutTS(PL_WARN,
       "TSIAnalyzer INFO: When computing TSI for input %d, %d bins out of\n",
       inputID+1, emptyCnt);
      printOutTS(PL_WARN, 
       "            %d are empty. This may be due to unconventional input\n",
       nAggrs);
      printOutTS(PL_WARN,
       "            distributions so that sample points are not ");
      printOutTS(PL_WARN,"distributed\n");
      printOutTS(PL_WARN,
       "            evenly over the parameter space.\n");
    }
    nFilled = 0;
    for (ii = 0; ii < nAggrs; ii++)
    {
      if (vecAggrCnts[ii] > 0) 
      {
        vecAggrMeans[ii] /= (double) vecAggrCnts[ii];
        nFilled++;
      }
    }
    if (printLevel_ > 1)
      printf("(TSI INFO) Input %4d : %d out of %d subdomains populated.\n",
             inputID+1, nFilled, nAggrs);
    dmean = 0.0;
    for (ii = 0; ii < nAggrs; ii++) 
      dmean += vecAggrMeans[ii] * vecAggrCnts[ii];
    dmean /= (double) nSamples;
    dvar = 0.0;
    for (ii = 0; ii < nAggrs; ii++)
      if (vecAggrCnts[ii] > 0) 
        dvar += pow(vecAggrMeans[ii]-dmean,2.0)*vecAggrCnts[ii];
    dvar /= (double) (nSamples - 1.0);

    if (dvar > variance)
    {
      printOutTS(PL_INFO,
       "Input %4d: Approximate total sensitivity index %e > variance %e?\n",
       inputID+1, dvar, variance);
      printOutTS(PL_INFO,
       "            Is your sample evenly distributed?\n");
      printOutTS(PL_INFO,
       "            Do you have too many subdomains (too few in each)?\n");
      for (ii = 0; ii < nAggrs; ii++)
        printf("Aggregate mean %d = %e (dmean=%e, count=%d)\n",ii+1,
               vecAggrMeans[ii],dmean,vecAggrCnts[ii]);
    }
    vecTSI[inputID] = variance - dvar;
  }
  if (printLevel_ > 1) printEquals(PL_INFO, 0);
  printResults(nInputs, variance, vecTSI.getDVector(), ioPtr);
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
TSIAnalyzer& TSIAnalyzer::operator=(const TSIAnalyzer &)
{
  printOutTS(PL_ERROR,
       "Total Effect operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// print result
// ------------------------------------------------------------------------
int TSIAnalyzer::printResults(int nInputs, double variance,
                              double *tsi, PsuadeData *ioPtr)
{
  int   ii;
  char  pString[1000];
  FILE  *fp;
  pData qData;

  if (variance == 0.0)
  {
    printOutTS(PL_INFO,
       "Total variance = 0. Hence, no total effect plot.\n");
    return 0;
  }
  printOutTS(PL_INFO, "Approximate Total Effect Statistics: \n");
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO,
      "Input %2d: Sobol' total sensitivity = %8.2e (normalized = %8.2e)\n",
      ii+1,tsi[ii],tsi[ii]/variance);
  if (plotScilab()) fp = fopen("scilabtsi.sci", "w");
  else              fp = fopen("matlabtsi.m", "w");

  if (ioPtr != NULL) ioPtr->getParameter("input_names", qData);
  if (fp != NULL)
  {
    strcpy(pString, "This file contains Sobol' total indices");
    fwriteComment(fp, pString);
    strcpy(pString, "set sortFlag = 1 and set nn to be the number");
    fwriteComment(fp, pString);
    strcpy(pString, "of inputs to display.");
    fwriteComment(fp, pString);
    fprintf(fp, "sortFlag = 0;\n");
    fprintf(fp, "nn = %d;\n", nInputs);
    fprintf(fp, "Mids = [\n");
    for (ii = 0; ii < nInputs; ii++) 
      fprintf(fp,"%24.16e\n",tsi[ii]/variance);
    fprintf(fp, "];\n");
    if (qData.strArray_ == NULL) 
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      for (ii = 0; ii < nInputs-1; ii++) fprintf(fp,"'X%d',",ii+1);
      if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
      else              fprintf(fp,"'X%d'};\n",nInputs);
    }
    else
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      fprintf(fp, "  Str = [");
      for (ii = 0; ii < nInputs-1; ii++)
      {
        if (qData.strArray_[ii] != NULL) 
             fprintf(fp,"'%s',",qData.strArray_[ii]);
        else fprintf(fp,"'X%d',",ii+1);
      }
      if (plotScilab()) 
      {
        if (qData.strArray_[nInputs-1] != NULL) 
             fprintf(fp,"'%s']",qData.strArray_[nInputs-1]);
        else fprintf(fp,"'X%d'];\n",nInputs);
      }
      else
      {
        if (qData.strArray_[nInputs-1] != NULL) 
             fprintf(fp,"'%s'}",qData.strArray_[nInputs-1]);
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
      fprintf(fp, "a=gca();\n");
      fprintf(fp, "a.data_bounds=[0, ymin; nn+1, ymax];\n");
      fprintf(fp, "newtick = a.x_ticks;\n");
      fprintf(fp, "newtick(2) = [1:nn]';\n");
      fprintf(fp, "newtick(3) = Str';\n");
      fprintf(fp, "a.x_ticks = newtick;\n");
      fprintf(fp, "a.x_label.font_size = 3;\n");
      fprintf(fp, "a.x_label.font_style = 4;\n");
    }
    else
    {
      fprintf(fp, "axis([0 nn+1 ymin ymax])\n");
      fprintf(fp, "set(gca,'XTickLabel',[]);\n");
      fprintf(fp, "th=text(1:nn, repmat(ymin-0.07*(ymax-ymin),nn,1),Str,");
      fprintf(fp, "'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp, "set(th, 'fontsize', 12)\n");
      fprintf(fp, "set(th, 'fontweight', 'bold')\n");
    }
    fwritePlotTitle(fp,"Sobol Total Order Indices");
    fwritePlotYLabel(fp, "Sobol Indices");
    fclose(fp);
    if (plotScilab())
         printOutTS(PL_INFO, "tsi plot file = scilabtsi.sci\n");
    else printOutTS(PL_INFO, "tsi plot file = matlabtsi.m\n");
  }
  else
  {
    printOutTS(PL_ERROR,
        "TSIAnalyser ERROR: cannot create tsi plot file.\n");
  }
  printAsterisks(PL_INFO, 0);
  return 0;
}

