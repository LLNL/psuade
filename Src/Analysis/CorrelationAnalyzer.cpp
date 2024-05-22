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
// Functions for the class CorrelationAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "CorrelationAnalyzer.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "FuncApprox.h"
#include "PrintingTS.h"
#include "psVector.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
CorrelationAnalyzer::CorrelationAnalyzer() : Analyzer(), nInputs_(0), 
                                             nOutputs_(0) 
{
  setName("CORRELATION");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
CorrelationAnalyzer::~CorrelationAnalyzer()
{
  cleanUp();
}

// ************************************************************************
// perform mean/variance analysis
// ------------------------------------------------------------------------
double CorrelationAnalyzer::analyze(aData &adata)
{
  int     ii, jj, ss, idata, count, iOne=1;
  double  xmean, xvar, ymean, yvar, yval, ddata, dmean, dvar;
  FILE    *fp;
  FuncApprox *faPtr=NULL;
  PsuadeData *ioPtr=NULL;
  pData   pData;
  psVector vecXMeans, vecYMeans, vecXVars, vecYVars; 
  psVector vecXT, vecYT, vecWT, vecXX, vecYY;

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  cleanUp();

  //**/ ---------------------------------------------------------------
  // extract data
  //**/ ---------------------------------------------------------------
  nInputs_ = adata.nInputs_;
  nOutputs_ = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *X = adata.sampleInputs_;
  double *Y = adata.sampleOutputs_;
  int outputID = adata.outputID_;
  int printLevel = adata.printLevel_;
  if (adata.inputPDFs_ != NULL)
  {
    count = 0;
    for (ii = 0; ii < nInputs_; ii++) count += adata.inputPDFs_[ii];
    if (count > 0)
    {
      printOutTS(PL_INFO,"Correlation INFO: Non-uniform probability ");
      printOutTS(PL_INFO,"distributions have been\n");
      printOutTS(PL_INFO,"            defined in the data file, but ");
      printOutTS(PL_INFO,"they will not be used in\n");
      printOutTS(PL_INFO,"            this analysis.\n");
    }
  }
  ioPtr = adata.ioPtr_;
  if (ioPtr != NULL) ioPtr->getParameter("input_names", pData);

  //**/ ---------------------------------------------------------------
  // error checking
  //**/ ---------------------------------------------------------------
  if (nInputs_ <= 0 || nOutputs_ <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR,"Correlation ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs_);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs_);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (outputID < 0 || outputID >= nOutputs_)
  {
    printOutTS(PL_ERROR,"Correlation ERROR: Invalid outputID.\n");
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs_);
    printOutTS(PL_ERROR,"    outputID = %d\n", outputID+1);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs_ > 1)
  {
    printOutTS(PL_WARN,
      "Correlation INFO: This analysis is primarily for nInputs=1.\n");
  }
  if (nSamples == 1)
  {
    printOutTS(PL_ERROR, 
       "Correlation INFO: analysis not meaningful for nSamples=1\n");
    return PSUADE_UNDEFINED;
  } 
  int status = 0; 
  for (ss = 0; ss < nSamples; ss++)
    if (Y[nOutputs_*ss+outputID] == PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"Correlation ERROR: Some outputs are ");
    printOutTS(PL_ERROR,"undefined. Prune the undefined\n");
    printOutTS(PL_ERROR,"                   sample point first.\n");
    return PSUADE_UNDEFINED;
  } 
   
  //**/ ---------------------------------------------------------------
  // first find the mean of the current set of sample points
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, "*                   Correlation Analysis\n");
  printOutTS(PL_INFO, "* Output of interest = %d\n", outputID+1);
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO, "* Basic Statistics\n");
  printDashes(PL_INFO, 0);
  computeMeanVariance(nSamples,nOutputs_,Y,&ymean,&yvar,outputID,1);
  outputMean_ = ymean;
  outputVar_ = yvar;

  //**/ ---------------------------------------------------------------
  //**/ compute the partial correlation coefficient but it is scaled
  //**/ since the unscaled ranges is [-1,1]
  //**/ ---------------------------------------------------------------
  printEquals(PL_INFO, 0);
  if (nInputs_ == 1)
    printOutTS(PL_INFO,"*  Scaled Spearman Correlation Coefficient: ");
  else
    printOutTS(PL_INFO,"*  Scaled Partial Correlation Coefficients: ");
  printOutTS(PL_INFO,"These coefficients give a\n");
  printOutTS(PL_INFO,
       "*  measure of linear relationship between X_i's & Y.\n");
  printOutTS(PL_INFO,
       "*  The signs show directions of relationship.\n");
  printOutTS(PL_INFO,
       "*  The magnitudes show strength of relationship.\n");
  printDashes(PL_INFO, 0);
  vecXMeans.setLength(nInputs_);
  vecXVars.setLength(nInputs_);
  VecInpMeans_.setLength(nInputs_);
  VecInpVars_.setLength(nInputs_);
  VecInpPearsonCoef_.setLength(nInputs_);

  for (ii = 0; ii < nInputs_; ii++)
  {
    computeMeanVariance(nSamples, nInputs_, X, &dmean, &dvar, ii, 0);
    VecInpMeans_[ii] = vecXMeans[ii] = dmean;
    VecInpVars_[ii] = vecXVars[ii] = dvar;
  }
  computeCovariance(nSamples,nInputs_,X,nOutputs_,Y,
         vecXMeans.getDVector(),vecXVars.getDVector(),ymean,yvar,
         outputID,VecInpPearsonCoef_.getDVector());
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (nInputs_ == 1)
      printOutTS(PL_INFO,
      "* Scaled Correlation Coeffient (Input %3d) = %11.4e\n", 
      ii+1, VecInpPearsonCoef_[ii]);
    else
      printOutTS(PL_INFO,
      "* Scaled Partial Correlation Coeffient (Input %3d) = %11.4e\n", 
      ii+1, VecInpPearsonCoef_[ii]);
  }

  //**/ ---------------------------------------------------------------
  // now write these information to a plot file
  //**/ ---------------------------------------------------------------
  if (plotScilab())
  {
    fp = fopen("scilabca.sci","w");
    if (fp == NULL)
      printOutTS(PL_INFO,
            "CorrelationAnalysis: cannot write to scilab file.\n");
    else
      fprintf(fp,"// This file contains correlation coefficients.\n");
  }
  else
  {
    fp = fopen("matlabca.m","w");
    if (fp == NULL)
      printOutTS(PL_WARN, 
           "CorrelationAnalysis: cannot write to matlab file.\n");
    else
      fprintf(fp,"%% This file contains correlation coefficients.\n");
  }
  if (fp != NULL)
  {
    fprintf(fp, "sortFlag = 0;\n");
    fprintf(fp, "nn = %d;\n", nInputs_);
    fprintf(fp, "PCC = [\n");
    for (ii = 0; ii < nInputs_; ii++) 
      fprintf(fp,"%24.16e\n", VecInpPearsonCoef_[ii]);
    fprintf(fp, "];\n");
    if (pData.strArray_ != NULL)
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      for (ii = 0; ii < nInputs_-1; ii++) fprintf(fp,"'X%d',",ii+1);
      fprintf(fp,"'X%d'];\n",nInputs_);
    }
    else
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      for (ii = 0; ii < nInputs_-1; ii++)
      {
        if (pData.strArray_[ii] != NULL) 
             fprintf(fp,"'%s',",pData.strArray_[ii]);
        else fprintf(fp,"'X%d',",ii+1);
      }
      if (plotScilab()) 
      {
        if (pData.strArray_[nInputs_-1] != NULL) 
             fprintf(fp,"'%s'];\n",pData.strArray_[nInputs_-1]);
        else fprintf(fp,"'X%d'];\n",nInputs_);
      }
      else
      {
        if (pData.strArray_[nInputs_-1] != NULL) 
             fprintf(fp,"'%s'};\n",pData.strArray_[nInputs_-1]);
        else fprintf(fp,"'X%d'};\n",nInputs_);
      }
    }
    fwritePlotCLF(fp);
    fprintf(fp, "if (sortFlag == 1)\n");
    if (plotScilab())
         fprintf(fp, "  [PCC, II] = gsort(PCC,'g','d');\n");
    else fprintf(fp, "  [PCC, II] = sort(PCC,'descend');\n");
    fprintf(fp, "  II   = II(1:nn);\n");
    fprintf(fp, "  PCC  = PCC(1:nn);\n");
    fprintf(fp, "  Str1 = Str(II);\n");
    fprintf(fp, "else\n");
    fprintf(fp, "  Str1 = Str;\n");
    fprintf(fp, "end\n");
    fprintf(fp, "ymin = min(PCC);\n");
    fprintf(fp, "ymax = max(PCC);\n");
    fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
    fprintf(fp, "subplot(1,2,1)\n");
    fprintf(fp, "bar(PCC,0.8);\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp,"Scaled Correlation Coefficients");
    fwritePlotYLabel(fp, "Correlation Coefficient");
    if (plotScilab())
    {
      fprintf(fp,"a=gca();\n");
      fprintf(fp,"a.data_bounds=[0, ymin; nn+1, ymax];\n");
      fprintf(fp,"a.x_ticks(2) = [1:nn]';\n");
      fprintf(fp,"a.x_ticks(3) = Str1';\n");
      fprintf(fp,"a.x_label.font_size = 3;\n");
      fprintf(fp,"a.x_label.font_style = 4;\n");
    }
    else
    {
      fprintf(fp,"axis([0 nn+1 ymin ymax])\n");
      fprintf(fp,"set(gca,'XTickLabel',[]);\n");
      fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),");
      fprintf(fp,"Str1,'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp,"set(th, 'fontsize', 12)\n");
      fprintf(fp,"set(th, 'fontweight', 'bold')\n");
    }
  }

  //**/ ---------------------------------------------------------------
  // compute correlation for outputs
  //**/ ---------------------------------------------------------------
  vecYMeans.setLength(nOutputs_);
  vecYVars.setLength(nOutputs_);

  VecOutMeans_.setLength(nOutputs_);
  VecOutVars_.setLength(nOutputs_);
  VecOutPearsonCoef_.setLength(nOutputs_);
  vecYT.setLength(nSamples);

  if (psConfig_.AnaExpertModeIsOn() && nOutputs_ > 1)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
      "* Scaled Correlation Coefficients for Y_k versus all Ys\n");
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nOutputs_; ii++)
    {
      computeMeanVariance(nSamples,nOutputs_,Y,&dmean,&dvar,ii,0);
      VecOutMeans_[ii] = vecYMeans[ii] = dmean;
      VecOutVars_[ii] = vecYVars[ii] = dvar;
    }
    for (ii = 0; ii < nOutputs_; ii++)
    {
      for (ss = 0; ss < nSamples; ss++) 
        vecYT[ss] = Y[ss*nOutputs_+ii];
      dmean = vecYMeans[ii];
      dvar  = vecYVars[ii];
      computeCovariance(nSamples,iOne,vecYT.getDVector(),nOutputs_,
           Y,&dmean,&dvar,ymean,yvar,outputID,&ddata);
      VecOutPearsonCoef_[ii] = ddata;
      if (ii != outputID)
        printOutTS(PL_INFO, 
         "* Scaled Correlation Coefficient (Output %2d vs %2d) = %e\n", 
         outputID+1, ii+1, VecOutPearsonCoef_[ii]);
    }
  }
  printEquals(PL_INFO, 0);
  //**/ these are allocated here to avoid crash when 
  //**/ these are retrieved
  VecInpSpearmanCoef_.setLength(nInputs_);
  VecOutSpearmanCoef_.setLength(nOutputs_);
  VecInpKendallCoef_.setLength(nInputs_);
  //**/ Do not do Spearman yet since Spearman replaces vectors with
  //**/ ranks so the scales are lost (how to rank then?)
  return 0.0;

  //**/ ---------------------------------------------------------------
  //**/ compute the Spearman coefficient (SPEA)
  //**/ NOTE: The following are turned off because partial correlation
  //**/       after rank transformation doesn't sound good
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO,"* Partial Rank Correlation Coefficients ");
  printOutTS(PL_INFO,"gives a measure of\n");
  printOutTS(PL_INFO,"* monotonic relationship between X_i's & Y\n");
  printOutTS(PL_INFO,"* (Idea: Use input and output ranks instead)\n");
  printDashes(PL_INFO, 0);
  vecXT.setLength(nSamples);
  vecWT.setLength(nSamples);
  vecYT.setLength(nSamples);
  vecXX.setLength(nSamples*nInputs_);
  vecYY.setLength(nSamples);

  //**/ convert X to ranks ==> vecXX
  for (ii = 0; ii < nInputs_; ii++)
  {
    //**/ first sort X 
    for (ss = 0; ss < nSamples; ss++)
    {
      vecXT[ss] = X[ss*nInputs_+ii];
      vecWT[ss] = (double) ss;
    }
    sortDbleList2(nSamples, vecXT.getDVector(), vecWT.getDVector());
    //**/ store the rank of X to vecXX 
    for (ss = 0; ss < nSamples; ss++)
    {
      jj = (int) vecWT[ss];
      vecXX[jj*nInputs_+ii] = 1.0 + ss;
    }
  }
  //**/ convert Y to ranks ==> vecYY
  for (ss = 0; ss < nSamples; ss++)
  {
    vecYT[ss] = Y[ss*nOutputs_+outputID];
    vecWT[ss] = (double) ss;
  }
  sortDbleList2(nSamples, vecYT.getDVector(), vecWT.getDVector());
  for (ss = 0; ss < nSamples; ss++)
  {
    jj = (int) vecWT[ss];
    vecYY[jj] = 1.0 + ss;
  }
  //**/ compute mean and variance of each input for the ranked X
  vecXT.setLength(nInputs_);
  vecWT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    computeMeanVariance(nSamples,nInputs_,vecXX.getDVector(), 
                        &dmean, &dvar, ii, 0);
    vecXT[ii] = dmean;
    vecWT[ii] = dvar;
  }
  //**/ compute mean and variance of the ranked Y 
  computeMeanVariance(nSamples,iOne,vecYY.getDVector(), &dmean, 
                      &dvar, 0, 0);

  //**/ compute partial rank correlations 
  computeCovariance(nSamples,nInputs_,vecXX.getDVector(),iOne,
         vecYY.getDVector(),vecXT.getDVector(),vecWT.getDVector(),
         dmean,dvar,0,VecInpSpearmanCoef_.getDVector());

  for (ii = 0; ii < nInputs_; ii++)
  {
    printOutTS(PL_INFO, 
      "* Partial Rank Correlation Coefficient (Input %3d) = %11.4e\n", 
      ii+1,VecInpSpearmanCoef_[ii]);
  }
  if (fp != NULL)
  {
    fprintf(fp, "SPEA = [\n");
    for (ii = 0; ii < nInputs_; ii++) 
      fprintf(fp,"%24.16e\n", VecInpSpearmanCoef_[ii]);
    fprintf(fp, "];\n");
    fprintf(fp, "if (sortFlag == 1)\n");
    if (plotScilab())
         fprintf(fp, "  [SPEA, II] = gsort(SPEA,'g','d');\n");
    else fprintf(fp, "  [SPEA, II] = sort(SPEA,'descend');\n");
    fprintf(fp, "  II   = II(1:nn);\n");
    fprintf(fp, "  SPEA = SPEA(1:nn);\n");
    fprintf(fp, "  Str2 = Str(II);\n");
    fprintf(fp, "else\n");
    fprintf(fp, "  Str2 = Str;\n");
    fprintf(fp, "end\n");
    fprintf(fp, "ymin = min(SPEA);\n");
    fprintf(fp, "ymax = max(SPEA);\n");
    fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
    fprintf(fp, "subplot(1,2,2)\n");
    fprintf(fp, "bar(SPEA,0.8);\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp,"Spearman Correlation Coefficients");
    fwritePlotYLabel(fp, "Correlation Coefficient");
    if (plotScilab())
    {
      fprintf(fp, "a=gca();\n");
      fprintf(fp, "a.data_bounds=[0, ymin; nn+1, ymax];\n");
      fprintf(fp, "a.x_ticks(2) = [1:nn]';\n");
      fprintf(fp, "a.x_ticks(3) = Str2';\n");
      fprintf(fp, "a.x_label.font_size = 3;\n");
      fprintf(fp, "a.x_label.font_style = 4;\n");
      printEquals(PL_INFO, 0);
      printOutTS(PL_INFO, 
           " Correlation analysis plot file = scilabca.sci.\n");
    }
    else
    {
      fprintf(fp,"axis([0 nn+1 ymin ymax])\n");
      fprintf(fp,"set(gca,'XTickLabel',[]);\n");
      fprintf(fp,"th=text(1:nn,repmat(ymin-0.05*(ymax-ymin),nn,1),");
      fprintf(fp,"Str2,'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp,"set(th, 'fontsize', 12)\n");
      fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      printEquals(PL_INFO, 0);
      printOutTS(PL_INFO,"Correlation analysis plot file = matlabca.m\n");
    }
    fclose(fp);
    fp = NULL;
  }
  printEquals(PL_INFO, 0);

#if 0
  //**/ ---------------------------------------------------------------
  //**/ run Kendall tau rank correlation test
  //**/ (have to figure out how it works with multivariate models)
  //**/ ---------------------------------------------------------------
  if (printLevel > 1)
  {
    int nc=0, nd=0;
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"* Kendall Coefficients of Concordance\n");
    printOutTS(PL_INFO, 
       "* (Idea: Use the ranks for both inputs and outputs)\n");
    printOutTS(PL_INFO,"* Kendall coefficients (in the scale of -1 ");
    printOutTS(PL_INFO,"to 1) measure the strength\n");
    printOutTS(PL_INFO,"* of relationship between an input and ");
    printOutTS(PL_INFO,"the output. The direction of\n");
    printOutTS(PL_INFO,"* the relationship is indicated by the ");
    printOutTS(PL_INFO,"sign of the coefficient.\n");
    printOutTS(PL_INFO,"* This nonparametric test is an alternative ");
    printOutTS(PL_INFO,"to Pearson's correlation\n");
    printOutTS(PL_INFO,"* when the model input-output has a ");
    printOutTS(PL_INFO,"nonlinear relationship, and this\n");
    printOutTS(PL_INFO,"* method is an alternative to the Spearman's ");
    printOutTS(PL_INFO,"correlation for small\n");
    printOutTS(PL_INFO,"* sample size and there are many tied ranks.\n");
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nInputs_; ii++)
    {
      nc = nd = 0;
      for (ss = 0; ss < nSamples; ss++) 
        vecXT[ss] = vecXX[ss*nInputs_+ii];
      for (ss = 0; ss < nSamples; ss++)
      {
        for (jj = ss+1; jj < nSamples; jj++)
        {
          ddata = vecXT[ss] - vecXT[jj];
          if (ddata != 0.0)
          {
            ddata = (vecYY[ss] - vecYY[jj]) / ddata;
            if (ddata >= 0.0) nc++;
            else              nd++;
          }
          else
          {
            ddata = vecYY[ss] - vecYY[jj];
            if (ddata >= 0.0) nc++;
            else              nd++;
          }
        }
      }
      VecInpKendallCoef_[ii] = 2.0*(nc-nd)/(nSamples * (nSamples - 1));
      printOutTS(PL_INFO, 
        "* Kendall Correlation Coefficient (Input %3d) = %11.3e \n",
        ii+1, VecInpKendallCoef_[ii]);
    }
    printDashes(PL_INFO, 0);
  }
#endif
  printAsterisks(PL_INFO, 0);
  return 0.0;
}

// *************************************************************************
// Compute mean and variance
// -------------------------------------------------------------------------
int CorrelationAnalyzer::computeMeanVariance(int nSamples, int xDim, 
              double *X, double *xmean, double *xvar, int xID, int flag)
{
  int    ss;
  double mean, variance;

  mean = 0.0;
  for (ss = 0; ss < nSamples; ss++) mean += X[xDim*ss+xID];
  mean /= (double) nSamples;
  variance = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
    variance += ((X[xDim*ss+xID] - mean) * (X[xDim*ss+xID] - mean));
  variance /= (double) (nSamples - 1);
  (*xmean) = mean;
  (*xvar)  = variance;
  if (flag == 1)
  {
    printOutTS(PL_INFO, "Output mean     = %e\n", mean);
    printOutTS(PL_INFO, "Output variance = %e\n", variance);
  }
  return 0;
}

// *************************************************************************
// Compute agglomerated covariances
// -------------------------------------------------------------------------
int CorrelationAnalyzer::computeCovariance(int nSamples,int nInps,double *X,
          int nOuts, double *Y, double *xmeans, double *xvars, double ymean,
          double yvar, int yID, double *Rvalues)
{
  int    ii, ss;
  double denom, numer;

  //**/ Case: nInputs = 1
  if (nInps == 1)
  {
    numer = 0.0;
    for (ss = 0; ss < nSamples; ss++)
      numer += ((X[ss*nInps] - xmeans[0]) * (Y[ss*nOuts+yID] - ymean));
    numer /= (double) (nSamples - 1);
    denom = sqrt(xvars[0] * yvar);
    if (denom == 0.0)
    {
      printOutTS(PL_INFO,"Correlation ERROR: denom=0 for input 1\n");
      printOutTS(PL_INFO, 
           "denom = xvar * yvar : xvar = %e, yvar = %e\n",xvars[0],yvar);
      Rvalues[0] = 0.0;
    }
    //else Rvalues[0] = numer / denom * yvar / xvars[0];
    else Rvalues[0] = numer / xvars[0];
  }
  else
  //**/ Case: nInputs > 1
  {
    int    jj, iOne=1, iZero=0;
    double Xmean, Xvar, Ymean, Yvar;
    FuncApprox *faPtr=NULL;
    psVector vecXX, vecYY, vecXY;
    vecXX.setLength(nSamples*nInps);
    vecXY.setLength(nSamples);
    vecYY.setLength(nSamples);
    double *XX = vecXX.getDVector();
    double *XY = vecXY.getDVector();
    for (ii = 0; ii < nInps; ii++)
    {
      //**/ build a RS with X \ X_i and Y
      for (ss = 0; ss < nSamples; ss++)
      {
        for (jj = 0; jj < nInps; jj++)
        {
          if (jj < ii) XX[ss*(nInps-1)+jj] = X[ss*nInps+jj];
          if (jj > ii) XX[ss*(nInps-1)+jj-1] = X[ss*nInps+jj];
        }
        vecYY[ss] = Y[ss*nOuts+yID];
      }
      psConfig_.InteractiveSaveAndReset();
      faPtr = genFA(PSUADE_RS_REGR1, nInps-1, iZero, nSamples);
      faPtr->setOutputLevel(0);
      faPtr->initialize(XX, vecYY.getDVector());
      psConfig_.InteractiveRestore();
      //**/ form new(Y) = Y - predicted Y
      for (ss = 0; ss < nSamples; ss++)
      {
        vecYY[ss] = faPtr->evaluatePoint(&XX[ss*(nInps-1)]);
        vecYY[ss] = Y[ss] - vecYY[ss];
      }
      delete faPtr;
      //**/ now build a RS with X\X_i and X_i
      for (ss = 0; ss < nSamples; ss++)
      {
        for (jj = 0; jj < nInps; jj++)
        {
          if (jj < ii) XX[ss*(nInps-1)+jj] = X[ss*nInps+jj];
          if (jj > ii) XX[ss*(nInps-1)+jj-1] = X[ss*nInps+jj];
        }
        vecXY[ss] = X[ss*nInps+ii];
      }
      psConfig_.InteractiveSaveAndReset();
      faPtr = genFA(PSUADE_RS_REGR1, nInps-1, iZero, nSamples);
      faPtr->setOutputLevel(0);
      faPtr->initialize(XX, vecXY.getDVector());
      psConfig_.InteractiveRestore();
      //**/ form new(X_i) = X_i - predicted X_i
      for (ss = 0; ss < nSamples; ss++)
      {
        vecXY[ss] = faPtr->evaluatePoint(&XX[ss*(nInps-1)]);
        vecXY[ss] = X[ss*nInps+ii] - vecXY[ss];
      }
      delete faPtr;
      //**/ compute mean(new(X_i)) and variance(new(X_i))
      computeMeanVariance(nSamples,iOne,vecXY.getDVector(),&Xmean,
                          &Xvar,iZero,iZero);
      computeMeanVariance(nSamples,iOne,vecYY.getDVector(),&Ymean,
                          &Yvar,iZero,iZero);
      computeCovariance(nSamples,iOne,vecXY.getDVector(),iOne,
             vecYY.getDVector(),&Xmean,&Xvar,Ymean,Yvar,iZero,&Rvalues[ii]);
    }
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
CorrelationAnalyzer& CorrelationAnalyzer::operator=(const CorrelationAnalyzer &)
{
  printOutTS(PL_ERROR,"Correlation operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int CorrelationAnalyzer::get_nInputs()
{
  return nInputs_;
}
int CorrelationAnalyzer::get_nOutputs()
{
  return nOutputs_;
}
double CorrelationAnalyzer::get_outputMean()
{
  return outputMean_;
}
double CorrelationAnalyzer::get_outputVar()
{
  return outputVar_;
}
double *CorrelationAnalyzer::get_inputMeans()
{
  psVector vecT;
  vecT = VecInpMeans_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_inputVars()
{
  psVector vecT;
  vecT = VecInpVars_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_outputMeans()
{
  psVector vecT;
  vecT = VecOutMeans_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_outputVars()
{
  psVector vecT;
  vecT = VecOutVars_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_inputPearsonCoef()
{
  psVector vecT;
  vecT = VecInpPearsonCoef_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_outputPearsonCoef()
{
  psVector vecT;
  vecT = VecOutPearsonCoef_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_inputSpearmanCoef()
{
  psVector vecT;
  vecT = VecInpSpearmanCoef_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_outputSpearmanCoef()
{
  psVector vecT;
  vecT = VecOutSpearmanCoef_;
  return vecT.takeDVector();
}
double *CorrelationAnalyzer::get_inputKendallCoef()
{
  psVector vecT;
  vecT = VecInpKendallCoef_;
  return vecT.takeDVector();
}
int CorrelationAnalyzer::cleanUp()
{
  nInputs_ = 0;
  nOutputs_ = 0;
  outputMean_ = 0.0;
  outputVar_ = 0.0;
  VecInpMeans_.clean();
  VecInpVars_.clean();
  VecOutMeans_.clean();
  VecOutVars_.clean();
  VecInpPearsonCoef_.clean();
  VecOutPearsonCoef_.clean();
  VecInpSpearmanCoef_.clean();
  VecOutSpearmanCoef_.clean();
  VecInpKendallCoef_.clean();
  return 0;
}

