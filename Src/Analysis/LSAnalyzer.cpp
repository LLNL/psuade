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
// functions for the class LSAnalyzer (Local sensitivity analysis)  
// AUTHOR : CHARLES TONG
// DATE   : 2012
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "LSAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "pData.h"
#include "PrintingTS.h"
//**/#include <algorithm>

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
LSAnalyzer::LSAnalyzer() : Analyzer(), nInputs_(0)
{
  setName("LSA");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
LSAnalyzer::~LSAnalyzer()
{
  VecLSMeas_.clean();
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double LSAnalyzer::analyze(aData &adata)
{
  int    ii, ss, whichOutput, ncount, index;
  char   pString[500], **iNames=NULL;
  FILE   *fp=NULL;
  pData  qData;

   //**/ ---------------------------------------------------------------
   //**/ extract data
   //**/ ---------------------------------------------------------------
  int  nInputs  = adata.nInputs_;
  int  nOutputs = adata.nOutputs_;
  int  nSamples = adata.nSamples_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  int  outputID = adata.outputID_;
  double *X = adata.sampleInputs_;
  double *Y = adata.sampleOutputs_;
  int printLevel = adata.printLevel_;
  if (adata.inputPDFs_ != NULL)
  {
    ncount = 0;
    for (ii = 0; ii < nInputs; ii++) ncount += adata.inputPDFs_[ii];
    if (ncount > 0)
    {
      printOutTS(PL_INFO, 
           "LocalSA INFO: some inputs have non-uniform PDFs, but\n");
      printOutTS(PL_INFO, 
           "              they are not relevant in this analysis.\n");
    }
  }
  whichOutput = outputID;
  if (whichOutput >= nOutputs || whichOutput < 0) whichOutput = 0;
  PsuadeData *ioPtr = adata.ioPtr_;
  if (ioPtr != NULL)
  {
    ioPtr->getParameter("input_names", qData);
    iNames = qData.strArray_;
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR, "LocalSA ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  //**/if (nSamples != nInputs + 1)
  //**/{
  //**/  printOutTS(PL_ERROR, 
  //**/  "LSAnalyzer ERROR: nSamples should be equal to nInputs+1.\n");
  //**/  printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
  //**/  return PSUADE_UNDEFINED;
  //**/} 

  //**/ ---------------------------------------------------------------
  //**/ display problem information
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO, "\n");
  printAsterisks(PL_INFO, 0);
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"* Local Sensitivity Analysis\n");
  printOutTS(PL_INFO,"* This analysis works if the model output can be\n");
  printOutTS(PL_INFO,"* approximated by linear combination of the inputs.\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO, "* total number of samples = %d\n",nSamples);
  printOutTS(PL_INFO, "* number of Inputs        = %d\n",nInputs);
  printOutTS(PL_INFO, "* Output number           = %d\n", whichOutput+1);
  printDashes(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ allocate temporary storage
  //**/ ---------------------------------------------------------------
  nInputs_ = nInputs;
  VecLSMeas_.setLength(nInputs);

  //**/ ---------------------------------------------------------------
  //**/ compute metrics
  //**/ ---------------------------------------------------------------
  psVector  vecGrads;
  psIVector vecInpInds;
  vecGrads.setLength(nSamples);
  vecInpInds.setLength(nSamples);
  for (ss = 1; ss < nSamples; ss++)
  {
    ncount = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      if (PABS(X[nInputs*ss+ii]-X[ii]) > 1.0e-15) 
      {
        ncount++;
        index = ii;
      }
    }
    if (ncount == 1)
    {
      vecGrads[ss] = Y[nOutputs*ss+whichOutput] - Y[whichOutput];
      vecGrads[ss] /= (X[nInputs*ss+index] - X[index]);
      vecGrads[ss] *= (xUpper[index] - xLower[index]); 
      if (vecGrads[ss] < 0.0) vecGrads[ss] = - vecGrads[ss];
      vecInpInds[ss] = index;
      if (printLevel > 3)
        printf("LSA sample %4d: input = %4d, grad = %12.4e\n",ss+1,
               index+1,vecGrads[ss]);
    }
    else
    {
      printOutTS(PL_INFO,"LSA ERROR: sample not suitable for LSA.\n");
      return 1;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ check validity of the sample and compute statistics
  //**/ ---------------------------------------------------------------
  psIVector vecCnts;
  vecCnts.setLength(nInputs);
  for (ss = 1; ss < nSamples; ss++)
  {
    index = vecInpInds[ss];
    vecCnts[index]++;
    VecLSMeas_[index] += vecGrads[ss];
  }  
  for (ii = 0; ii < nInputs; ii++)
  {
    if (vecCnts[ii] > 0) VecLSMeas_[ii] /= (double) vecCnts[ii];
    printOutTS(PL_INFO,"* Input %3d avg importance measure = %11.4e ", 
               ii+1, VecLSMeas_[ii]);
    printOutTS(PL_INFO,"(based on %d points)\n",vecCnts[ii]); 
  }

  //**/ ---------------------------------------------------------------
  //**/ generate matlab file
  //**/ ---------------------------------------------------------------
  if (plotScilab())
  {
    fp = fopen("scilablsa.sci", "w");
    if (fp != NULL)
    {
      fprintf(fp,"// ********************************************** \n");
      fprintf(fp,"// ********************************************** \n");
      fprintf(fp,"// * Local Sensitivity Analysis                ** \n");
      fprintf(fp,"// *-------------------------------------------** \n");
      fprintf(fp,"// Output %d\n", whichOutput+1);
      fprintf(fp,"// *-------------------------------------------** \n");
    }
  }
  else
  {
    fp = fopen("matlablsa.m", "w");
    if (fp != NULL)
    {
      fprintf(fp,"%% ********************************************** \n");
      fprintf(fp,"%% ********************************************** \n");
      fprintf(fp,"%% * Local Sensitivity Analysis                ** \n");
      fprintf(fp,"%% *-------------------------------------------** \n");
      fprintf(fp,"%% Output %d\n", whichOutput+1);
      fprintf(fp,"%% *-------------------------------------------** \n");
    }
  }
  if (fp != NULL)
  {
    fprintf(fp, "sortFlag = 0;\n");
    fprintf(fp, "morePlots = 0;\n");
    fprintf(fp, "nn = %d;\n", nInputs);
    fprintf(fp, "Y = [\n");
    for (ii = 0; ii < nInputs; ii++)
       fprintf(fp, "%24.16e \n", PABS(VecLSMeas_[ii]));
    fprintf(fp, "]; \n");
    if (iNames == NULL)
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
    fprintf(fp, "if (sortFlag == 1);\n");
    if (plotScilab()) fprintf(fp,"  [AA,II] = gsort(Y);\n");
    else              fprintf(fp,"  [AA,II] = sort(Y, 'descend');\n");
    fprintf(fp, "  II  = II(1:nn);\n");
    fprintf(fp, "  Str = Str(II);\n");
    fprintf(fp, "  Y   = Y(II);\n");
    fprintf(fp, "end\n");
    fprintf(fp, "ymax = max(Y);\n");
    fprintf(fp, "ymin = 0.0;\n");
    fprintf(fp, "if (ymax == ymin)\n");
    fprintf(fp, "   ymax = ymax * 1.01;\n");
    fprintf(fp, "   ymin = ymin * 0.99;\n");
    fprintf(fp, "end;\n");
    fprintf(fp, "if (ymax == ymin)\n");
    fprintf(fp, "   ymax = ymax + 0.01;\n");
    fprintf(fp, "   ymin = ymin - 0.01;\n");
    fprintf(fp,"end;\n");
    fwritePlotCLF(fp);
    fprintf(fp, "bar(Y,0.8);\n");
    fwritePlotAxes(fp);
    if (plotScilab())
    {
      fprintf(fp,"a=gca();\n");
      fprintf(fp,"a.data_bounds=[0, ymin-0.01*(ymax-ymin); nn+1, ");
      fprintf(fp,"ymax+0.01*(ymax-ymin)];\n");
      fprintf(fp,"newtick = a.x_ticks;\n");
      fprintf(fp,"newtick(2) = [1:nn]';\n");
      fprintf(fp,"newtick(3) = Str';\n");
      fprintf(fp,"a.x_ticks = newtick;\n");
      fprintf(fp,"a.x_label.font_size = 3;\n");
      fprintf(fp,"a.x_label.font_style = 4;\n");
    }
    else
    {
      fprintf(fp,
         "axis([0 nn+1 ymin-0.01*(ymax-ymin) ymax+0.01*(ymax-ymin)])\n");
      fprintf(fp,"set(gca,'XTickLabel',[]);\n");
      fprintf(fp,"th=text(1:nn, repmat(ymin-0.07*(ymax-ymin),nn,1),Str,");
      fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp,"set(th, 'fontsize', 12)\n");
      fprintf(fp,"set(th, 'fontweight', 'bold')\n");
    }
    snprintf(pString,100,"Linear Sensitivity Measures");
    fwritePlotTitle(fp, pString);
    //**/snprintf(pString,100,"Input Numbers");
    //**/fwritePlotXLabel(fp, pString);
    snprintf(pString,100,"Sensitivity Measure");
    fwritePlotYLabel(fp, pString);
    if (nSamples > nInputs+1)
    { 
      fprintf(fp,"if morePlots > 0\n");
      fprintf(fp,"  Grads = [\n");
      for (ss = 1; ss < nSamples; ss++)
        fprintf(fp,"  %6d %16.8e\n", vecInpInds[ss]+1, vecGrads[ss]);
      fprintf(fp,"  ];\n");
      if (plotMatlab()) fprintf(fp,"figure(2)\n");
      else              fprintf(fp,"scf(2)\n");
      fprintf(fp,"plot(G(:,1),G(:,2),'*','MarkerSize',12)\n");
      snprintf(pString,100,"Input Numbers");
      fwritePlotXLabel(fp, pString);
      snprintf(pString,100,"Individual Gradients (normalized)");
      fwritePlotYLabel(fp, pString);
      snprintf(pString,100,"Gradients for individual Inputs");
      fwritePlotTitle(fp, pString);
      fwritePlotAxes(fp);
      fprintf(fp, "xmin = 0;\n");
      fprintf(fp, "xmax = %d;\n", nInputs+1);
      fprintf(fp, "ymin = min(G(:,2));\n");
      fprintf(fp, "ymax = max(G(:,2));\n");
      fprintf(fp, "ytmp = ymin;\n");
      fprintf(fp, "ymin = ymin - 0.1 * (ymax - ymin);\n");
      fprintf(fp, "ymax = ymax + 0.1 * (ymax - ytmp);\n");
      fwritePlotScales2D(fp); 
      fprintf(fp,"end;\n");
      fprintf(fp,"disp(\"Note: change morePlots to 1 to display more.\")\n");
    }
    fclose(fp);
    if (plotScilab())
         printOutTS(PL_INFO, "LSA ranking result in scilablsa.sci\n");
    else printOutTS(PL_INFO, "LSA ranking result in matlablsa.m\n");
  }
  printAsterisks(PL_INFO, 0);
  return 0.0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
LSAnalyzer& LSAnalyzer::operator=(const LSAnalyzer &)
{
  printOutTS(PL_ERROR,"LocalSA operator= ERROR: operation not supported\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int LSAnalyzer::get_nInputs()
{
   return nInputs_;
}
double *LSAnalyzer::get_lsMeasures()
{
  psVector vecDT;
  vecDT = VecLSMeas_;
  return vecDT.takeDVector();
}

