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
// Functions for the class PCAnalyzer (principal component analysis)  
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "PCAnalyzer.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PrintingTS.h"
#include "psVector.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

extern "C" {
  void dgels_(char *, int *, int *, int *, double *A, int *LDA,
              double *B, int *LDB, double *WORK, int *LWORK, int *INFO);
  void dgesvd_(char *, char *, int *, int *, double *, int *, double *,
               double *, int *, double *, int *, double *, int *, int *);
}
                                                                                
// ************************************************************************
// constructor
// ------------------------------------------------------------------------
PCAnalyzer::PCAnalyzer() : Analyzer()
{
  setName("PCA");
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"Principal component analysis (PCA) \n");
  printOutTS(PL_INFO,
       "This function performs a PCA of all sample outputs\n"); 
  printOutTS(PL_INFO,"such that YY = U * S * V\n\n");
  printOutTS(PL_INFO,"where YY is the normalized sample outputs,\n");
  printOutTS(PL_INFO,"      U are the left singular vectors,\n");
  printOutTS(PL_INFO,"      S are the singular values, and\n");
  printOutTS(PL_INFO,"      V are the right singular vectors.\n");
  printOutTS(PL_INFO,"At the end, the sample outputs will be replaced\n");
  printOutTS(PL_INFO,"by Y = Y * V^t \n");
  printOutTS(PL_INFO,"If the analysis expert mode is on, you have the\n");
  printOutTS(PL_INFO,"option to store these matrices into matlab file.\n");
  printAsterisks(PL_INFO, 0);
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PCAnalyzer::~PCAnalyzer()
{
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double PCAnalyzer::analyze(aData &adata)
{
  int    jj, ss, ii, info, pcCnt, pcaFlag=0, count;
  double dtemp;
  char   winput1[500], winput2[500], pcaFile[500], *cString;
  char   pString[500];
  FILE   *fp = NULL;

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *Y    = adata.sampleOutputs_;
  if (adata.inputPDFs_ != NULL)
  {
    count = 0;
    for (ii = 0; ii < nInputs; ii++) count += adata.inputPDFs_[ii];
    if (count > 0)
    {
      printOutTS(PL_INFO,
           "PCA INFO: some inputs have non-uniform PDFs, but\n");
      printOutTS(PL_INFO,
           "          they are not relevant in this analysis.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR, "PCA ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nOutputs == 1)
  {
    printOutTS(PL_ERROR, 
         "PCA ERROR: analysis not done since nOutputs=1.\n");
    return PSUADE_UNDEFINED;
  }
   
  //**/ ---------------------------------------------------------------
  //**/ check to see if detailed analysis is needed
  //**/ ---------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn() && plotMatlab())
  {
    snprintf(pString,100,"Write PCA information to a file? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      snprintf(pString,100,"Enter the matlab file name : ");
      getString(pString, pcaFile);
      pcaFile[strlen(pcaFile)-1] = '\0';
      fp = fopen(pcaFile, "w");
      if (fp != NULL)
      {
        fclose(fp);
        pcaFlag = 1;
      }
      else printOutTS(PL_INFO,"PCA: cannot open matlab file %s\n", 
                      pcaFile);
    }
  }
  else
  {
    cString = psConfig_.getParameter("PCA_matlab_file");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %s", winput1, winput2, pcaFile);
      fp = fopen(pcaFile, "w");
      if (fp != NULL)
      {
        fclose(fp);
        pcaFlag = 1;
      }
      else 
        printOutTS(PL_INFO,
              "PCA ERROR: cannot open matlab file %s\n",pcaFile);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ load up the output data matrix
  //**/ ---------------------------------------------------------------
  psVector vecYY;
  vecYY.setLength(nSamples*nOutputs);
  for (jj = 0; jj < nOutputs; jj++)
    for (ss = 0; ss < nSamples; ss++)
      vecYY[nSamples*jj+ss] = Y[ss*nOutputs+jj];

  //**/ ---------------------------------------------------------------
  //**/ adjust for the bias and normalize
  //**/ ---------------------------------------------------------------
  psVector vecMeans, vecStds;
  vecMeans.setLength(nOutputs);
  vecStds.setLength(nOutputs);
  for (jj = 0; jj < nOutputs; jj++)
  {
    vecMeans[jj] = 0.0;
    for (ss = 0; ss < nSamples; ss++) 
      vecMeans[jj] += vecYY[nSamples*jj+ss];
    vecMeans[jj] /= (double) nSamples;
    for (ss = 0; ss < nSamples; ss++) 
      vecYY[nSamples*jj+ss] -= vecMeans[jj];
    vecStds[jj] = 0.0;
    for (ss = 0; ss < nSamples; ss++)
      vecStds[jj] += vecYY[nSamples*jj+ss] * vecYY[nSamples*jj+ss];
    vecStds[jj] /= (double) (nSamples - 1);
    vecStds[jj] = sqrt(vecStds[jj]);
    if (PABS(vecStds[jj]) > 1.0e-14)
      for (ss = 0; ss < nSamples; ss++) 
        vecYY[nSamples*jj+ss] /= vecStds[jj];
    printOutTS(PL_INFO,"PCA: output %5d has mean = %e, std. dev = %e\n", 
               jj+1, vecMeans[jj], vecStds[jj]);
  }

  //**/ ---------------------------------------------------------------
  //**/ perform SVD (instead of eigendecomposition)
  //**/ ---------------------------------------------------------------
  int M = nSamples;
  int N = nOutputs;
  char jobu, jobvt;
  jobu = 'S';
  jobvt = 'A';
  int wlen = 5 * M;
  psVector vecSS, vecUU, vecVV, vecWW;
  vecSS.setLength(N);
  vecUU.setLength(M*N);
  vecVV.setLength(N*N);
  vecWW.setLength(wlen);
  dgesvd_(&jobu, &jobvt, &M, &N,vecYY.getDVector(),&M,vecSS.getDVector(), 
      vecUU.getDVector(), &M, vecVV.getDVector(), &N, vecWW.getDVector(),
      &wlen, &info);
  if (info != 0)
    printOutTS(PL_INFO, 
         "* PCA INFO: dgesvd returns a nonzero (%d).\n",info);
  for (ii = 0; ii < N; ii++) vecSS[ii] = vecSS[ii] * vecSS[ii];
  for (ii = 0; ii < N; ii++)
    printOutTS(PL_INFO, 
       "principal component %3d has variance = %16.8e\n",ii+1,vecSS[ii]);
  snprintf(pString,100,"Enter how many principal components to keep : ");
  pcCnt = getInt(1, N, pString);

  //**/ ---------------------------------------------------------------
  //**/ compute projection of Y onto the PC space
  //**/ ---------------------------------------------------------------
  for (jj = 0; jj < nOutputs; jj++)
    for (ss = 0; ss < nSamples; ss++)
      vecYY[nSamples*jj+ss] = Y[ss*nOutputs+jj];
  for (jj = 0; jj < nOutputs; jj++)
  {
    vecMeans[jj] = 0.0;
    for (ss = 0; ss < nSamples; ss++) 
      vecMeans[jj] += vecYY[nSamples*jj+ss];
    vecMeans[jj] /= (double) nSamples;
    for (ss = 0; ss < nSamples; ss++) 
      vecYY[nSamples*jj+ss] -= vecMeans[jj];
    vecStds[jj] = 0.0;
    for (ss = 0; ss < nSamples; ss++)
      vecStds[jj] += vecYY[nSamples*jj+ss] * vecYY[nSamples*jj+ss];
    vecStds[jj] /= (double) (nSamples - 1);
    vecStds[jj] = sqrt(vecStds[jj]);
    if (PABS(vecStds[jj]) > 1.0e-14)
      for (ss = 0; ss < nSamples; ss++) 
        vecYY[nSamples*jj+ss] /= vecStds[jj];
  }

  //**/ ---------------------------------------------------------------
  //**/ create PCA matlab file
  //**/ ---------------------------------------------------------------
  if (pcaFlag == 1)
  {
    fp = fopen(pcaFile, "w");
    if (fp != NULL)
    {
      fprintf(fp,"%% Principal component analysis results\n");
      fprintf(fp,"%% First display the Scree diagram\n");
      fprintf(fp,"%% A  - original sample outputs (normalized)\n");
      fprintf(fp,"%% Am - mean of each column of A\n");
      fprintf(fp,"%% As - std dev of each column of A\n");
      fprintf(fp,"%% V  - eigenvector matrix\n");
      fprintf(fp,"%% U  - left singular vector matrix\n");
      fprintf(fp,"%% S  - singular values (column vector)\n");

      fprintf(fp,"A = [\n");
      for (ss = 0; ss < M; ss++) 
      {
        for (jj = 0; jj < N; jj++) fprintf(fp,"%e ",vecYY[M*jj+ss]);
        fprintf(fp, "\n");
      }
      fprintf(fp,"];\n");
      fprintf(fp,"Am = [\n");
      for (ii = 0; ii < N; ii++) fprintf(fp," %e\n", vecMeans[ii]);
      fprintf(fp,"];\n");
      fprintf(fp,"As = [\n");
      for (ii = 0; ii < N; ii++) fprintf(fp," %e\n", vecStds[ii]);
      fprintf(fp,"];\n");
      fprintf(fp, "S = [\n");
      for (ii = 0; ii < pcCnt; ii++) fprintf(fp," %e\n",vecSS[ii]);
      fprintf(fp,"];\n");
      fprintf(fp,"hold off\n");
      fprintf(fp,"plot(S)\n");
      fprintf(fp,"grid\n");
      fprintf(fp,"xlabel('Principal component number')\n");
      fprintf(fp,"ylabel('Variances')\n");
      fprintf(fp,"title('Scree Diagram')\n");
      fprintf(fp,"disp('Press enter to continue')\n");
      fprintf(fp,"pause\n");

      fprintf(fp,"V = [\n");
      for (ii = 0; ii < N; ii++)
      {
        for (jj = 0; jj < pcCnt; jj++) fprintf(fp," %e",vecVV[jj*N+ii]);
        fprintf(fp, "\n");
      }
      fprintf(fp,"];\n");
      fprintf(fp,"U = [\n");
      for (ii = 0; ii < nSamples; ii++)
      {
        for (jj = 0; jj < pcCnt; jj++)
           fprintf(fp," %e", vecUU[jj*M+ii]);
        fprintf(fp,"\n");
      }
      fprintf(fp,"];\n");
      fprintf(fp,"disp('Display each principal component')\n");
      fprintf(fp,"US = U * diag(S);\n");
      fprintf(fp,"XU = 1:%d;\n",M);
      fprintf(fp,"for ii = 1 : %d\n", pcCnt);
      fprintf(fp,"   plot(XU,US(:,ii),XU,US(:,ii),'x')\n");
      fprintf(fp,"   disp(['principal component = ' int2str(ii)])\n");
      fprintf(fp,"   disp('Press enter to continue')\n");
      fprintf(fp,"   pause\n");
      fprintf(fp,"end;\n");
      fprintf(fp,"hold off\n");
      fprintf(fp,"%%barh(abs(A)','stacked')\n");
      fprintf(fp,"%%area(XA,abs(A))\n");
      fprintf(fp,"hold off\n");
      fclose(fp);
    }
    else printOutTS(PL_WARN,"WARNING: cannot open file for plotting.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ modify outputs to use principal component score instead
  //**/ ---------------------------------------------------------------
  for (ss = 0; ss < nSamples; ss++)
  {
    for (jj = 0; jj < pcCnt; jj++)
    {
      dtemp = 0.0;
      for (ii = 0; ii < N; ii++)
        dtemp += vecVV[jj*N+ii] * vecYY[ii*nSamples+ss];
      Y[pcCnt*ss+jj] = dtemp;
    }
  }

  fp = fopen("psPCA.out", "w");
  if (fp == NULL)
  {
    printOutTS(PL_ERROR, 
         "PCA ERROR: failed to store principal components.\n");
  }
  else
  {
    fprintf(fp,"# Let Y = normalized sample outputs\n");
    fprintf(fp,"# Let Y = U S V^t\n");
    fprintf(fp,"# This file contains Y * V\n");
    fprintf(fp,"%d %d\n", nSamples, pcCnt);
    for (ss = 0; ss < nSamples; ss++)
    {
      for (jj = 0; jj < pcCnt; jj++)
         fprintf(fp, "%24.16e ", Y[pcCnt*ss+jj]);
      fprintf(fp, "\n");
    }
    fclose(fp);
    printOutTS(PL_INFO, 
         "PCA: principal components (Y*V) stored in psPCA.out file.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ store results and clean up
  //**/ ---------------------------------------------------------------
  adata.nOutputs_ = pcCnt;
  return 0.0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
PCAnalyzer& PCAnalyzer::operator=(const PCAnalyzer &)
{
  printOutTS(PL_ERROR, "PCA operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

