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
// Functions for the class MarsBagg
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "MarsBagg.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class MarsBagg
// ------------------------------------------------------------------------
MarsBagg::MarsBagg(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
#ifdef HAVE_MARS
  int  ii, itmp;
  char pString[500], *cString, *targv[3], winput1[100], winput2[100];

  //**/ ==========================================================
  //**/ set internal parameters
  //**/ ==========================================================
  //**/ set identifier
  faID_ = PSUADE_RS_MARSB;

  //**/ default number of MARS instantiation
  numMars_ = 100;

  //**/ 0: mean, 1: median 
  mode_ = 0;

  //**/ change mode, if interactive
  usageIndex_ = 1;

  //**/ default number of basis functions
  maxBasis_ = 100;
  if (maxBasis_ > nSamples) maxBasis_ = nSamples - 1;

  //**/ default interaction = maximum
  if (nInputs >= 8) varPerBasis_ = 8;
  else              varPerBasis_ = nInputs;

  //**/ check config file for other than defaults
  if (!psConfig_.RSExpertModeIsOn())
  {
    cString = psConfig_.getParameter("RS_MARS_num");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &itmp);
      if (itmp < 2)
      {
        printf("MarsBag INFO: numMars from config file not valid.\n");
        printf("              numMars kept at %d.\n", numMars_);
      }
      else
      {
        numMars_ = itmp;
        printf("MarsBag INFO: number of instantiations set to %d.\n",
               numMars_);
      }
    }
    cString = psConfig_.getParameter("RS_MARS_num_bases");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &itmp);
      if (itmp < 10 || itmp > nSamples)
      {
        printf("MarsBag INFO: nbasis from config file not valid.\n");
        printf("              nbasis kept at %d.\n", maxBasis_);
      }
      else
      {
        maxBasis_ = itmp;
        printf("MarsBag INFO: number of basis set to %d (config).\n",
               maxBasis_);
      }
    }
    cString = psConfig_.getParameter("RS_MARS_interaction");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &itmp);
      if (itmp > nInputs || itmp < 1)
      {
        printf("MarsBag INFO: interaction from config file not valid.\n");
        printf("              interaction kept at %d.\n", varPerBasis_);
      }
      else
      {
        varPerBasis_ = itmp;
        printf("MarsBag INFO: interaction set to %d (config).\n",varPerBasis_);
      }
    }
  }

  //**/ display configuration
  if (outputLevel_ > 1)
  {
    //**/ =======================================================
    //**/ set up mode and thetas
    //**/ =======================================================
    printAsterisks(PL_INFO, 0);
    printf("*                MarsBag Analysis\n");
    printDashes(PL_INFO, 0);
    printf("Default mode = mean (options: mean, median).\n");
    printf("Number of instantiations   = %d\n",numMars_);
    printf("No. of basis functions     = %d\n",maxBasis_);
    printf("No. of variables per basis = %d\n",varPerBasis_);
    printf("* Turn on rs_expert mode to select internal parameters.\n");
    printf("* Set print level to 5 to print out ranking information.\n");
    printEquals(PL_INFO, 0);
  }
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100,"MARS with bagging: mean (0) or median (1) mode ? ");
    mode_ = getInt(0, 1, pString);
    snprintf(pString,100,
             "How many instantiation of MARS (10-5000, default=100) ? ");
    numMars_ = getInt(10, 5000, pString);
    snprintf(pString,100,
             "How many basis functions in MARS (< %d, default = %d) ? ",
             nSamples, maxBasis_);
    maxBasis_ = getInt(1, nSamples, pString);
    snprintf(pString,100,"How many variables per basis (<= %d, default = %d) ? ",
             nInputs, varPerBasis_);
    varPerBasis_ = getInt(1, nInputs, pString);
    if (psConfig_.MasterModeIsOn())
    {
      printf("You can control the probability of using more sample\n");
      printf("points in any instantiation by setting a 'frequency' knob.\n");
      printf("The default is 4, which gives 80-90 percent usage.\n");
      printf("Set this number to a larger value to increase usage.\n");
      printf("If you do not know what this knob does, enter 3.\n");
      printf("To see the actual usage percentage, turn on printlevel 2.\n");
      snprintf(pString,100,
               "What value should be assigned to this knob? (2 - 8) ");
      usageIndex_ = getInt(2, 8, pString);
    }
  }

  //**/ set number of instantiation 
  strcpy(pString, "mars_params");
  targv[0] = (char *) pString;
  targv[1] = (char *) &maxBasis_;
  targv[2] = (char *) &varPerBasis_;
  psConfig_.RSExpertModeSaveAndReset();
  marsObjs_ = new Mars*[numMars_];
  PsuadeConfig tmpConfig;
  tmpConfig = psConfig_;
  psConfig_.reset();
  psConfig_.InteractiveOff();
  for (ii = 0; ii < numMars_; ii++) 
  {
    marsObjs_[ii] = new Mars(nInputs_, nSamples_);
    marsObjs_[ii]->setParams(3, targv);
  }
  psConfig_ = tmpConfig;
  psConfig_.RSExpertModeRestore();

  //**/ Mars info
  marsFms_  = NULL;
  marsIms_  = NULL;
#else
  printf("PSUADE ERROR : MARSBAGG not installed.\n");
  exit(1);
#endif
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
MarsBagg::~MarsBagg()
{
  int ii;

  if (marsObjs_ != NULL) 
  {
    for (ii = 0; ii < numMars_; ii++) delete marsObjs_[ii];
    delete [] marsObjs_;
  }
  if (marsFms_ != NULL)
  {
    for (ii = 0; ii < numMars_; ii++) delete marsFms_[ii];
    delete [] marsFms_;
  }
  if (marsIms_ != NULL)
  {
    for (ii = 0; ii < numMars_; ii++) delete marsIms_[ii];
    delete [] marsIms_;
  }
}

// ************************************************************************
// Set bounds for object class FuncApprox
// ------------------------------------------------------------------------
int MarsBagg::setBounds( double *lower, double *upper )
{
  for (int ii=0 ; ii<numMars_; ii++) marsObjs_[ii]->setBounds(lower,upper);
  return 0;
}

// ************************************************************************
// load output weights
// ------------------------------------------------------------------------
int MarsBagg::loadWeights(int n, double *wgts)
{
  for (int ii=0 ; ii<numMars_; ii++) marsObjs_[ii]->loadWeights(n, wgts);
  return 0;
}

// ************************************************************************
// set number of points to generate in each dimension
// ------------------------------------------------------------------------
void MarsBagg::setNPtsPerDim(int npoints)
{
  nPtsPerDim_ = npoints;
  for (int ii=0 ; ii<numMars_; ii++) marsObjs_[ii]->setNPtsPerDim(npoints);
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int MarsBagg::initialize(double *XX, double *Y)
{
#ifdef HAVE_MARS
  int  ii, ss, jj, index, SAFlag;
  FILE *fp;
  psVector  vecXB, vecYB;
  psIVector ivecT, ivecCnts;
  psMatrix  matSAIndices;

  //**/ these two vectors store boostrapped samples temporarily
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);

  //**/ compute sensitivity indices
  if (outputLevel_ >= 4)
  {
    matSAIndices.setFormat(PS_MAT2D);
    matSAIndices.setDim(numMars_, nInputs_);
    ivecT.setLength(nInputs_);
  }

  //**/ SAFlag tells whether sensitivity information is available
  //**/ next create bootstrapped samples and build MARS
  SAFlag = 0;
  ivecCnts.setLength(nSamples_);
  psConfig_.RSExpertModeSaveAndReset();
  for (ii = 0; ii < numMars_; ii++)
  {
    if (outputLevel_ > 4)
      printf("MarsBagg::initialize : creating Mars #%d (of %d)\n",
             ii+1, numMars_);
    if (MatDataSetX_.nrows() <= 0)
    {
      for (ss = 0; ss < nSamples_; ss++) ivecCnts[ss] = usageIndex_ * 2;
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = -1;
        while (index == -1)
        {
          index = PSUADE_rand() % nSamples_;
          if (ivecCnts[index] > 0 && (ivecCnts[index] % usageIndex_) == 0)
          {
            ivecCnts[index] = ivecCnts[index] - 1;
          }
          else if ((ivecCnts[index] > 0) && 
                   (ivecCnts[index] % usageIndex_) != 0)
          {
            ivecCnts[index] = ivecCnts[index] - 1;
            index = -1;
          }
          else if (ivecCnts[index] <= 0) index = -1;
        }
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XX[index*nInputs_+jj]; 
        vecYB[ss] = Y[index]; 
      }
      index = 0;
      for (ss = 0; ss < nSamples_; ss++) 
        if (ivecCnts[ss] < usageIndex_*2) index++;
      if (outputLevel_ >= 2)
        printf("     Number of unique sample points used = %d (out of %d)\n",
               index,nSamples_);
    }
    else
    {
      double **matX2D = MatDataSetX_.getMatrix2D();
      double **matY2D = MatDataSetY_.getMatrix2D();
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
        vecYB[ss] = matY2D[ii][ss]; 
      }
    }
    marsObjs_[ii]->setOutputLevel(0);
    jj = 0;
    for (ss = 0; ss < nSamples_; ss++)
      if (vecYB[ss] >= PSUADE_UNDEFINED) jj++;
    if (jj > 0)
    {
      printf("MarsBag ERROR: some of the sample outputs are undefined.\n");
      exit(1);
    }
    marsObjs_[ii]->initialize(vecXB.getDVector(), vecYB.getDVector());
    if (outputLevel_ >= 4) 
    {
      double **matI2D = matSAIndices.getMatrix2D();
      SAFlag += getImportance(nInputs_, matI2D[ii]);
    }
    fp = fopen("psuade_print", "r");
    if (fp != NULL)
    {
      printf("MarsBagg: set print level to 2\n");
      outputLevel_ = 2;
      fclose(fp);
    }
    if (psConfig_.RSCodeGenIsOn()) readRSCode(ii);
  }
  if (psConfig_.RSCodeGenIsOn()) genRSCode();

  //**/ if it is requested to compute sensitivity indices
  if (SAFlag == 0 && matSAIndices.nrows() > 0)
  {
    //**/ generate statistics
    psVector vecMeans, vecStds;
    vecMeans.setLength(nInputs_);
    vecStds.setLength(nInputs_);
    double **matI2D = matSAIndices.getMatrix2D();
    for (jj = 0; jj < nInputs_; jj++)
    {
      vecMeans[jj] = 0.0;
      vecStds[jj] = 0.0;
      for (ii = 0; ii < numMars_; ii++) vecMeans[jj] += matI2D[ii][jj];
      vecMeans[jj] /= (double) numMars_;
      for (ii = 0; ii < numMars_; ii++)
        vecStds[jj] += pow(matI2D[ii][jj] - vecMeans[jj], 2.0e0);
      vecStds[jj] /= (double) numMars_;
      vecStds[jj] = sqrt(vecStds[jj]);
    }     
    //**/ output to a matlab file
    if (plotScilab())
    {
      fp = fopen("scilabmarsbsa.sci", "w");
      if (fp == NULL)
      {
        printf("MarsBag ERROR: cannot open scilab file.\n");
        return 0;
      } 
      fprintf(fp,"// This file contains MarsBag ranking measures\n");
      fprintf(fp,"// and also their spreads based on bootstraping.\n");
      fprintf(fp,"// To select the most important ones to display,\n");
      fprintf(fp,"// set sortFlag = 1 and set nn to be the number\n");
      fprintf(fp,"// of inputs to display.\n");
    }
    else
    {
      fp = fopen("matlabmarsbsa.m", "w");
      if (fp == NULL)
      {
        printf("MarsBag ERROR: cannot open matlab file.\n");
        return 0;
      } 
      fprintf(fp,"%% This file contains MarsBag ranking measures\n");
      fprintf(fp,"%% and also their spreads based on bootstraping.\n");
      fprintf(fp,"%% To select the most important ones to display,\n");
      fprintf(fp,"%% set sortFlag = 1 and set nn to be the number\n");
      fprintf(fp,"%% of inputs to display.\n");
    }
    fwritePlotCLF(fp);
    fprintf(fp, "nn = %d;\n", nInputs_);
    fprintf(fp, "Means = [\n");
    for (ii = 0; ii < nInputs_; ii++)
    {
      index = (int) (100 * vecMeans[ii]);
      fprintf(fp, "%d\n", index);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "Stds = [\n");
    for (ii = 0; ii < nInputs_; ii++)
    {
      index = (int) (100 * vecStds[ii]);
      fprintf(fp, "%d\n", index);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "ymax = max(Means);\n");
    fprintf(fp, "ymin = min(Means);\n");
    fprintf(fp, "if (ymax == ymin)\n");
    fprintf(fp, "   ymax = ymax * 1.01;\n");
    fprintf(fp, "   ymin = ymin * 0.99;\n");
    fprintf(fp, "end;\n");
    fprintf(fp, "if (ymax == ymin)\n");
    fprintf(fp, "   ymax = ymax + 0.01;\n");
    fprintf(fp, "   ymin = ymin - 0.01;\n");
    fprintf(fp,"end;\n");
    fprintf(fp, "bar(Means,0.8);\n");
    fprintf(fp, "for ii = 1:nn\n");
    fprintf(fp, "   if (ii == 1)\n");
    if (plotScilab())
         fprintf(fp, "   set(gca(),\"auto_clear\",\"off\")\n");
    else fprintf(fp, "   hold on\n");
    fprintf(fp, "   end;\n");
    fprintf(fp, "   XX = [ii ii];\n");
    fprintf(fp, "   YY = [Means(ii)-Stds(ii) Means(ii)+Stds(ii)];\n");
    fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
    fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',12)\n");
    fprintf(fp, "end;\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "MARSB Rankings");
    fwritePlotXLabel(fp, "Input parameters");
    fwritePlotYLabel(fp, "MARSB ranks");
    if (plotScilab())
    {
      fprintf(fp, "a=gca();\n");
      fprintf(fp, "a.data_bounds=[0, ymin-0.01*(ymax-ymin); ");
      fprintf(fp, "nn+1, ymax+0.01*(ymax-ymin)];\n");
    }
    else
    {
      fprintf(fp,"axis([0 nn+1 ymin-0.01*(ymax-ymin) ");
      fprintf(fp,"ymax+0.01*(ymax-ymin)])\n");
    }
    fclose(fp);

    //**/ output ranking results on screen
    for (ii = 0; ii < nInputs_; ii++) ivecT[ii] = ii;
    sortDbleList2a(nInputs_, vecMeans.getDVector(), ivecT.getIVector());
    printf("* ========== MARS screening rankings =========== *\n");
    for (ii = nInputs_-1; ii >= 0; ii--)
    {
      index = (int) (100 * vecMeans[ii]);
      printf("*  Rank %3d : Input = %3d (measure = %3d, stdev = %9.3e)\n",
             nInputs_-ii, ivecT[ii]+1, index, 100.0*vecStds[ii]);
    }
    printf("* ============================================== *\n");
    //**/ let users know 
    if (plotScilab())
         printf("MarsBag ranking is now in scilabmarsbsa.sci.\n");
    else printf("MarsBag ranking is now in matlabmarsbsa.m.\n");
    //**/ clean up
  }
  psConfig_.RSExpertModeRestore();
  return 0;
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MarsBagg::genNDGridData(double *XIn,double *YIn,int *NOut,
                            double **XOut, double **YOut)
{
#ifdef HAVE_MARS
  int    totPts, ii, ss, jj, index, SAFlag;
  double *XXt, *Yt;
  FILE   *fp;
  psIVector ivecT, ivecCnts;
  psVector  vecXB, vecYB;
  psMatrix  matSAIndices, matYM;

  //**/ these two vectors store boostrapped samples temporarily
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);

  //**/ compute sensitivity indices
  if (outputLevel_ >= 4)
  {
    matSAIndices.setFormat(PS_MAT2D);
    matSAIndices.setDim(numMars_, nInputs_);
    ivecT.setLength(nInputs_);
  }

  SAFlag = 0;
  if ((*NOut) == -999)
  {
    //**/ create MARS but no interpolation needed
    ivecCnts.setLength(nSamples_);
    psConfig_.RSExpertModeSaveAndReset();
    for (ii = 0; ii < numMars_; ii++)
    {
      if (outputLevel_ >= 2)
         printf("MarsBagg::genNDGridData : creating Mars #%d (of %d)\n",
                ii+1, numMars_);
      if (MatDataSetX_.nrows() <= 0)
      {
        for (ss = 0; ss < nSamples_; ss++) ivecCnts[ss] = usageIndex_ * 2;
        for (ss = 0; ss < nSamples_; ss++)
        {
          index = -1;
          while (index == -1)
          {
            index = PSUADE_rand() % nSamples_;
            if (ivecCnts[index] > 0 && (ivecCnts[index] % usageIndex_) == 0)
            {
              ivecCnts[index]--;
            }
            else if (ivecCnts[index] > 0 && 
                     (ivecCnts[index] % usageIndex_) != 0)
            {
              ivecCnts[index]--;
              index = -1;
            }
            else if (ivecCnts[index] <= 0) index = -1;
          }
          for (jj = 0; jj < nInputs_; jj++)
            vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
          vecYB[ss] = YIn[index]; 
        }
        index = 0;
        for (ss = 0; ss < nSamples_; ss++) 
          if (ivecCnts[ss] < usageIndex_*2) index++;
        if (outputLevel_ >= 2)
          printf("     Number of sample points used = %d (out of %d)\n",
                 index,nSamples_);
      }
      else
      {
        double **matX2D = MatDataSetX_.getMatrix2D();
        double **matY2D = MatDataSetY_.getMatrix2D();
        for (ss = 0; ss < nSamples_; ss++)
        {
          for (jj = 0; jj < nInputs_; jj++)
            vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
          vecYB[ss] = matY2D[ii][ss]; 
        }
      }
      marsObjs_[ii]->setOutputLevel(0);
      marsObjs_[ii]->genNDGridData(vecXB.getDVector(),vecYB.getDVector(), 
                                   NOut, NULL, NULL);
      if (outputLevel_ >= 4) 
      {
        double **matS2D = matSAIndices.getMatrix2D();
        SAFlag += getImportance(nInputs_, matS2D[ii]);
      }
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        printf("MarsBagg: set print level to 2\n");
        outputLevel_ = 2;
        fclose(fp);
      }
    }
    if (SAFlag == 0 && matSAIndices.nrows() > 0)
    {
      //**/ generate statistics
      double **matS2D = matSAIndices.getMatrix2D();
      psVector vecMeans, vecStds;
      vecMeans.setLength(nInputs_);
      vecStds.setLength(nInputs_);
      for (jj = 0; jj < nInputs_; jj++)
      {
        vecMeans[jj] = vecStds[jj] = 0.0;
        for (ii = 0; ii < numMars_; ii++) vecMeans[jj] += matS2D[ii][jj];
        vecMeans[jj] /= (double) numMars_;
        for (ii = 0; ii < numMars_; ii++)
          vecStds[jj] += pow(matS2D[ii][jj] - vecMeans[jj], 2.0e0);
        vecStds[jj] /= (double) numMars_;
        vecStds[jj] = sqrt(vecStds[jj]);
      }     
      //**/ output to a matlab file
      if (plotScilab())
      {
        fp = fopen("scilabmarsbsa.sci", "w");
        if (fp == NULL)
        {
          printf("MarsBag ERROR: cannot open scilab file.\n");
          return 0;
        } 
        fprintf(fp,"// This file contains MarsBag ranking measures\n");
        fprintf(fp,"// and also their spreads based on bootstraping.\n");
        fprintf(fp,"// To select the most important ones to display,\n");
        fprintf(fp,"// set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"// of inputs to display.\n");
      }
      else
      {
        fp = fopen("matlabmarsbsa.m", "w");
        if (fp == NULL)
        {
          printf("MarsBag ERROR: cannot open matlab file.\n");
          return 0;
        } 
        fprintf(fp,"%% This file contains MarsBag ranking measures\n");
        fprintf(fp,"%% and also their spreads based on bootstraping.\n");
        fprintf(fp,"%% To select the most important ones to display,\n");
        fprintf(fp,"%% set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"%% of inputs to display.\n");
      }
      fwritePlotCLF(fp);
      fprintf(fp, "nn = %d;\n", nInputs_);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs_; ii++)
      {
        index = (int) (100 * vecMeans[ii]);
        fprintf(fp, "%d\n", index);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs_; ii++)
      {
        index = (int) (100 * vecStds[ii]);
        fprintf(fp, "%d\n", index);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "ymax = max(Means);\n");
      fprintf(fp, "ymin = min(Means);\n");
      fprintf(fp, "if (ymax == ymin)\n");
      fprintf(fp, "   ymax = ymax * 1.01;\n");
      fprintf(fp, "   ymin = ymin * 0.99;\n");
      fprintf(fp, "end;\n");
      fprintf(fp, "if (ymax == ymin)\n");
      fprintf(fp, "   ymax = ymax + 0.01;\n");
      fprintf(fp, "   ymin = ymin - 0.01;\n");
      fprintf(fp,"end;\n");
      fprintf(fp, "bar(Means,0.8);\n");
      fprintf(fp, "for ii = 1:nn\n");
      fprintf(fp, "   if (ii == 1)\n");
      if (plotScilab())
           fprintf(fp, "   set(gca(),\"auto_clear\",\"off\")\n");
      else fprintf(fp, "   hold on\n");
      fprintf(fp, "   end;\n");
      fprintf(fp, "   XX = [ii ii];\n");
      fprintf(fp, "   YY = [Means(ii)-Stds(ii) Means(ii)+Stds(ii)];\n");
      fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
      fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',12)\n");
      fprintf(fp, "end;\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "MARSB Rankings");
      fwritePlotXLabel(fp, "Input parameters");
      fwritePlotYLabel(fp, "MARSB ranks");
      if (plotScilab())
      {
        fprintf(fp, "a=gca();\n");
        fprintf(fp, "a.data_bounds=[0, ymin-0.01*(ymax-ymin); ");
        fprintf(fp, "nn+1, ymax+0.01*(ymax-ymin)];\n");
      }
      else
      {
        fprintf(fp,"axis([0 nn+1 ymin-0.01*(ymax-ymin) ");
        fprintf(fp,"ymax+0.01*(ymax-ymin)])\n");
      }
      fclose(fp);

      //**/ output ranking results on screen
      for (ii = 0; ii < nInputs_; ii++) ivecT[ii] = ii;
      sortDbleList2a(nInputs_,vecMeans.getDVector(),ivecT.getIVector());
      printf("* ========== MARS screening rankings =========== *\n");
      for (ii = nInputs_-1; ii >= 0; ii--)
      {
        index = (int) (100 * vecMeans[ii]);
        printf("*  Rank %3d : Input = %3d (measure = %3d, stdev = %9.3e)\n",
               nInputs_-ii, ivecT[ii]+1, index, 100.0*vecStds[ii]);
      }
      printf("* ============================================== *\n");
      //**/ let users know 
      if (plotScilab())
           printf("MarsBag ranking is now in scilabmarsbsa.sci.\n");
      else printf("MarsBag ranking is now in matlabmarsbsa.m.\n");
      //**/ clean up
    }
    psConfig_.RSExpertModeRestore();
  }
  else
  {
    psConfig_.RSExpertModeSaveAndReset();
    //**/ set up for generating regular grid data
    totPts = nPtsPerDim_;
    for (ii = 1; ii < nInputs_; ii++) totPts = totPts * nPtsPerDim_;
    psVector vecXOut, vecYOut;
    vecXOut.setLength(nInputs_ * totPts);
    vecYOut.setLength(totPts);
    (*XOut) = vecXOut.takeDVector();
    (*YOut) = vecYOut.takeDVector();
    for (ii = 0; ii < totPts; ii++) (*YOut)[ii] = 0.0;

    //**/ if median mode, set up an array for storing data
    if (mode_ != 0)
    {
      matYM.setFormat(PS_MAT2D);
      matYM.setDim(totPts, numMars_);
    }

    //**/ generate response surface and interpolate
    for (ii = 0; ii < numMars_; ii++)
    {
      //**/ generate bootstrap sample 
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
        vecYB[ss] = YIn[index]; 
      }
      marsObjs_[ii]->genNDGridData(vecXB.getDVector(),vecYB.getDVector(), 
                                   NOut, &XXt, &Yt);

      if (outputLevel_ >= 4) 
      {
        double **matS2D = matSAIndices.getMatrix2D();
        SAFlag += getImportance(nInputs_, matS2D[ii]);
      }

      //**/ collect outputs
      if (mode_ != 0)
      {
        //**/ median mode
        for (ss = 0; ss < totPts; ss++) 
          matYM.setEntry(ss, ii, Yt[ss]);
      }
      else
      {
        //**/ mean mode
        for (ss = 0; ss < totPts; ss++) (*YOut)[ss] += Yt[ss];
      }

      //**/ store inputs
      if (ii == numMars_-1)
        for (ss = 0; ss < totPts*nInputs_; ss++) 
          (*XOut)[ss] = XXt[ss];
      delete [] XXt;
      delete [] Yt;
    }

    if (SAFlag == 0 && matSAIndices.nrows() > 0)
    {
      double **matS2D = matSAIndices.getMatrix2D();
      psVector vecMeans, vecStds;
      vecMeans.setLength(nInputs_);
      vecStds.setLength(nInputs_);
      for (jj = 0; jj < nInputs_; jj++)
      {
        vecMeans[jj] = vecStds[jj] = 0.0;
        for (ii = 0; ii < numMars_; ii++) vecMeans[jj] += matS2D[ii][jj];
        vecMeans[jj]/= (double) numMars_;
        for (ii = 0; ii < numMars_; ii++)
          vecStds[jj] += pow(matS2D[ii][jj] - vecMeans[jj], 2.0e0);
        vecStds[jj] /= (double) numMars_;
        vecStds[jj] = sqrt(vecStds[jj]);
      }     
      for (ii = 0; ii < nInputs_; ii++) ivecT[ii] = ii;
      sortDbleList2a(nInputs_, vecMeans.getDVector(), ivecT.getIVector());
      printf("* ========= MARSBAG screening rankings ========= *\n");
      for (ii = nInputs_-1; ii >= 0; ii--)
      {
        index = (int) (100 * vecMeans[ii]);
        printf("*  Rank %3d : Input = %3d (measure = %3d, stdev = %9.3e)\n",
               nInputs_-ii, ivecT[ii]+1, index, 100.0*vecStds[ii]);
      }
      printf("* ============================================== *\n");
    }

    (*NOut) = totPts;

    if (mode_ != 0)
    {
      double **MYM = matYM.getMatrix2D();
      for (ss = 0; ss < totPts; ss++)
      {
        sortDbleList(numMars_, MYM[ss]); 
        (*YOut)[ss] = MYM[ss][numMars_/2];
      }
    }
    else
    { 
      for (ss = 0; ss < totPts; ss++) 
        (*YOut)[ss] /= (double) numMars_;
    }
    psConfig_.RSExpertModeRestore();
  }
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MarsBagg::gen1DGridData(double *XIn, double *YIn, int ind1, 
                            double *settings, int *NOut, double **XOut, 
                            double **YOut)
{
#ifdef HAVE_MARS
  int    totPts, ii, ss, jj, index;
  double *XXt, *YYt;
  psVector vecXB, vecYB;
  psMatrix matYM;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_;
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YOut)[ii] = 0.0;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0)
  {
    matYM.setFormat(PS_MAT2D);
    matYM.setDim(totPts, numMars_);
  }

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numMars_; ii++)
  {
    if (MatDataSetX_.nrows() <= 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
        vecYB[ss] = YIn[index]; 
      }
    }
    else
    {
      double **matX2D = MatDataSetX_.getMatrix2D();
      double **matY2D = MatDataSetY_.getMatrix2D();
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
        vecYB[ss] = matY2D[ii][ss]; 
      }
    }
    marsObjs_[ii]->gen1DGridData(vecXB.getDVector(),vecYB.getDVector(),
                                 ind1,settings,NOut,&XXt,&YYt);

    //**/ collect outputs
    if (mode_ != 0)
    {
      //**/ median mode
      for (ss = 0; ss < totPts; ss++) 
        matYM.setEntry(ss, ii, YYt[ss]);
    }
    else
    {
      //**/ mean mode
      for (ss = 0; ss < totPts; ss++) (*YOut)[ss] += YYt[ss];
    }
    if (ii == numMars_-1)
      for (ss = 0; ss < totPts; ss++) (*XOut)[ss] = XXt[ss];
    delete [] XXt;
    delete [] YYt;
  }

  //**/ process outputs based on the mode
  if (mode_ != 0)
  {
    double **MYM = matYM.getMatrix2D();
    for (ss = 0; ss < totPts; ss++)
    {
      sortDbleList(numMars_, MYM[ss]); 
      (*YOut)[ss] = MYM[ss][numMars_/2];
    }
  }
  else
  { 
    for (ss = 0; ss < totPts; ss++) (*YOut)[ss] /= (double) numMars_;
  }
  psConfig_.RSExpertModeRestore();
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int MarsBagg::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            double *settings, int *NOut, double **XOut, 
                            double **YOut)
{
#ifdef HAVE_MARS
  int    totPts, ii, ss, jj, index;
  double *XXt, *YYt;
  psVector vecXB, vecYB;
  psMatrix matYM;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_;
  (*NOut)   = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*2);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YOut)[ii] = 0.0;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0)
  {
    matYM.setFormat(PS_MAT2D);
    matYM.setDim(totPts, numMars_);
  }

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numMars_; ii++)
  {
    if (outputLevel_ >= 1)
      printf("MarsBagg::gen2DGridData : creating Mars #%d (of %d)\n",
             ii+1, numMars_);
    if (MatDataSetX_.nrows() <= 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
        vecYB[ss] = YIn[index]; 
      }
    }
    else
    {
      double **matX2D = MatDataSetX_.getMatrix2D();
      double **matY2D = MatDataSetY_.getMatrix2D();
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
        vecYB[ss] = matY2D[ii][ss]; 
      }
    }
    marsObjs_[ii]->gen2DGridData(vecXB.getDVector(),vecYB.getDVector(),
                                 ind1,ind2,settings,NOut,&XXt,&YYt);

    //**/ collect outputs
    if (mode_ != 0)
    {
      //**/ median mode
      for (ss = 0; ss < totPts; ss++) 
        matYM.setEntry(ss, ii, YYt[ss]);
    }
    else
    {
      //**/ mean mode
      for (ss = 0; ss < totPts; ss++) (*YOut)[ss] += YYt[ss];
    }

    //**/ collect inputs
    if (ii == numMars_-1)
      for (ss = 0; ss < 2*totPts; ss++) (*XOut)[ss] = XXt[ss];
    delete [] XXt;
    delete [] YYt;
  }

  //**/ process outputs based on the mode
  if (mode_ != 0)
  {
    double **MYM = matYM.getMatrix2D();
    for (ss = 0; ss < totPts; ss++)
    {
      sortDbleList(numMars_, MYM[ss]); 
      (*YOut)[ss] = MYM[ss][numMars_/2];
    }
  }
  else
  { 
    for (ss = 0; ss < totPts; ss++) (*YOut)[ss] /= (double) numMars_;
  }
  psConfig_.RSExpertModeRestore();
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int MarsBagg::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            int ind3, double *settings, int *NOut, 
                            double **XOut, double **YOut)
{
#ifdef HAVE_MARS
  int    totPts, ii, ss, jj, index;
  double *XXt, *YYt;
  psVector vecXB, vecYB;
  psMatrix matYM;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*3);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YOut)[ii] = 0.0;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0)
  {
    matYM.setFormat(PS_MAT2D);
    matYM.setDim(totPts, numMars_);
  }

  //**/ generate response surface and interpolate
  for (ii = 0; ii < numMars_; ii++)
  {
    if (MatDataSetX_.nrows() <= 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
        vecYB[ss] = YIn[index]; 
      }
    }
    else
    {
      double **matX2D = MatDataSetX_.getMatrix2D();
      double **matY2D = MatDataSetY_.getMatrix2D();
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
        vecYB[ss] = matY2D[ii][ss]; 
      }
    }
    marsObjs_[ii]->gen3DGridData(vecXB.getDVector(),vecYB.getDVector(),
                      ind1,ind2,ind3,settings,NOut,&XXt,&YYt);

    //**/ collect outputs
    if (mode_ != 0)
    {
      //**/ median mode
      for (ss = 0; ss < totPts; ss++) 
        matYM.setEntry(ss, ii, YYt[ss]);
    }
    else
    {
      //**/ mean mode
      for (ss = 0; ss < totPts; ss++) (*YOut)[ss] += YYt[ss];
    }

    //**/ collect inputs
    if (ii == numMars_-1)
      for (ss = 0; ss < 3*totPts; ss++) (*XOut)[ss] = XXt[ss];
    delete [] XXt;
    delete [] YYt;
  }

  //**/ process outputs based on the mode
  if (mode_ != 0)
  {
    double **MYM = matYM.getMatrix2D();
    for (ss = 0; ss < totPts; ss++)
    {
      sortDbleList(numMars_, MYM[ss]); 
      (*YOut)[ss] = MYM[ss][numMars_/2];
    }
  }
  else
  { 
    for (ss = 0; ss < totPts; ss++) (*YOut)[ss] /= (double) numMars_;
  }
  psConfig_.RSExpertModeRestore();
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int MarsBagg::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            int ind3, int ind4, double *settings, 
                            int *NOut, double **XOut, double **YOut)
{
#ifdef HAVE_MARS
  int    totPts, ii, ss, jj, index;
  double *XXt, *YYt;
  psVector vecXB, vecYB;
  psMatrix matYM;

  //**/ set up storage
  psConfig_.RSExpertModeSaveAndReset();
  vecXB.setLength(nInputs_ * nSamples_);
  vecYB.setLength(nSamples_);
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;
  psVector vecXOut, vecYOut;
  vecXOut.setLength(totPts*4);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  for (ii = 0; ii < totPts; ii++) (*YOut)[ii] = 0.0;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0)
  {
    matYM.setFormat(PS_MAT2D);
    matYM.setDim(totPts, numMars_);
  }

  for (ii = 0; ii < numMars_; ii++)
  {
    if (MatDataSetX_.nrows() <= 0)
    {
      for (ss = 0; ss < nSamples_; ss++)
      {
        index = PSUADE_rand() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = XIn[index*nInputs_+jj]; 
        vecYB[ss] = YIn[index]; 
      }
    }
    else
    {
      double **matX2D = MatDataSetX_.getMatrix2D();
      double **matY2D = MatDataSetY_.getMatrix2D();
      for (ss = 0; ss < nSamples_; ss++)
      {
        for (jj = 0; jj < nInputs_; jj++)
          vecXB[ss*nInputs_+jj] = matX2D[ii][ss*nInputs_+jj]; 
        vecYB[ss] = matY2D[ii][ss]; 
      }
    }
    marsObjs_[ii]->gen4DGridData(vecXB.getDVector(),vecYB.getDVector(), 
                ind1, ind2, ind3, ind4, settings, NOut, &XXt, &YYt);

    //**/ collect outputs
    if (mode_ != 0)
    {
      //**/ median mode
      for (ss = 0; ss < totPts; ss++) 
        matYM.setEntry(ss, ii, YYt[ss]);
    }
    else
    {
       //**/ mean mode
       for (ss = 0; ss < totPts; ss++) (*YOut)[ss] += YYt[ss];
    }

    //**/ collect inputs
    if (ii == numMars_-1)
       for (ss = 0; ss < 4*totPts; ss++) (*XOut)[ss] = XXt[ss];
    delete [] XXt;
    delete [] YYt;
  }

  //**/ process outputs based on the mode
  if (mode_ != 0)
  {
    double **MYM = matYM.getMatrix2D();
    for (ss = 0; ss < totPts; ss++)
    {
      sortDbleList(numMars_, MYM[ss]); 
      (*YOut)[ss] = MYM[ss][numMars_/2];
    }
  }
  else
  { 
    for (ss = 0; ss < totPts; ss++) (*YOut)[ss] /= (double) numMars_;
  }
  psConfig_.RSExpertModeRestore();
#else
  printf("PSUADE ERROR : MARS not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double MarsBagg::evaluatePoint(double *X)
{
  double Y=0.0;
#ifdef HAVE_MARS
  int    ii;
  double Yt;
  psVector vecYM;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0) vecYM.setLength(numMars_);

  //**/ interpolate
  for (ii = 0; ii < numMars_; ii++) 
  {
    Yt = marsObjs_[ii]->evaluatePoint(X);
    if (mode_ != 0) vecYM[ii] = Yt;
    else            Y += Yt;
  }
  if (mode_ != 0)
  {
    sortDbleList(numMars_, vecYM.getDVector());
    Y = vecYM[numMars_/2];
  }
  else Y /= (double) numMars_;
#else
  printf("PSUADE ERROR : MARS not installed.\n");
#endif
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double MarsBagg::evaluatePoint(int npts, double *X, double *Y)
{
#ifdef HAVE_MARS
  int    in, ii;
  double YY, Yt;
  psVector vecYM;

  //**/ if median mode, set up an array for storing data
  if (mode_ != 0) vecYM.setLength(numMars_);

  //**/ interpolate
  for (in = 0; in < npts; in++) 
  {
    YY = 0.0;
    for (ii = 0; ii < numMars_; ii++) 
    {
      Yt = marsObjs_[ii]->evaluatePoint(&(X[in*nInputs_]));
      if (mode_ != 0) vecYM[ii] = Yt;
      else            YY += Yt;
    }
    if (mode_ != 0)
    {
      sortDbleList(numMars_, vecYM.getDVector());
      Y[in] = vecYM[numMars_/2];
    }
    else Y[in] = YY / (double) numMars_;
  }
#else
  printf("PSUADE ERROR : MARS not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double MarsBagg::evaluatePointFuzzy(double *X, double &std)
{
  double Ymean=0.0;
#ifdef HAVE_MARS
  int    ii;
  psVector vecYY;

  vecYY.setLength(numMars_);
  for (ii = 0; ii < numMars_; ii++) 
  {
    vecYY[ii] = marsObjs_[ii]->evaluatePoint(X);
    Ymean += vecYY[ii];
  }
  Ymean /= (double) numMars_;
  std = 0.0;
  for (ii = 0; ii < numMars_; ii++) 
    std += (vecYY[ii] - Ymean) * (vecYY[ii] - Ymean);
  std = sqrt(std / (double) numMars_);
#else
  printf("PSUADE ERROR : MARS not installed.\n");
#endif
  return Ymean;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double MarsBagg::evaluatePointFuzzy(int npts, double *X, double *Y,
                                    double *Ystd)
{
#ifdef HAVE_MARS
  int in, ii;
  psVector vecYY;

  vecYY.setLength(numMars_);
  for (in = 0; in < npts; in++) 
  {
    for (ii = 0; ii < numMars_; ii++) 
      vecYY[ii] = marsObjs_[ii]->evaluatePoint(&(X[in*nInputs_]));
    Y[in] = 0.0;
    for (ii = 0; ii < numMars_; ii++) Y[in] += vecYY[ii];
    Y[in] /= (double) numMars_;
    Ystd[in] = 0.0;
    for (ii = 0; ii < numMars_; ii++) 
      Ystd[in] += (vecYY[ii] - Y[in]) * (vecYY[ii] - Y[in]);
    Ystd[in] = sqrt(Ystd[in] / (double) numMars_);
  }
#else
  printf("PSUADE ERROR : MARS not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// get the importance indicators
// ------------------------------------------------------------------------
int MarsBagg::getImportance(int nInputs, double *indicators)
{
  int    lineLeng=500, ii, jj, nCount;
  double dmax;
  char   line[500], word1[500], word2[500], word3[500];
  FILE   *fp;

  fp = fopen(".psuade_mars", "r");
  if (fp == NULL) return -1;
  else
  {
    nCount = 0;
    strcpy(word1, "none");
    while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
    {
      sscanf(line,"%s %s %s", word1, word2, word3);
      if (!strcmp(word1,"relative") && !strcmp(word2,"variable") &&
          !strcmp(word3,"importance:"))
      {
        fgets(line, lineLeng, fp);
        fgets(line, lineLeng, fp);
        if (feof(fp) == 0)
        {
          for (ii = 0; ii < nInputs_; ii+=6)
          {
            for (jj = 0; jj < 6; jj++)
            {
              if (ii+jj < nInputs_)
              {
                fscanf(fp,"%lg", &(indicators[ii+jj]));
                nCount++;
              }
            }
            fgets(line, lineLeng, fp);
            fgets(line, lineLeng, fp);
            fgets(line, lineLeng, fp);
          }
        }
      }
    }
    if (nCount != nInputs_)
    {
      fclose(fp);
      return -1;
    }
    dmax = indicators[0];
    for (ii = 1; ii < nInputs_; ii++)
      if (indicators[ii] > dmax) dmax = indicators[ii];
    for (ii = 0; ii < nInputs_; ii++) indicators[ii] /= dmax;
    fclose(fp);
  }
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double MarsBagg::setParams(int targc, char **targv)
{
  int    ii, itmp, leng;
  double *Xdata, *Ydata;
  char   cString[500], *argv[3];

  if (targc == 1 && !strcmp(targv[0], "median"))
  {
    mode_ = 1;
  }
  else if (targc >= 2 && !strcmp(targv[0], "num_mars"))
  {
    if (marsObjs_ != NULL) 
    {
      for (ii = 0; ii < numMars_; ii++) delete marsObjs_[ii];
      delete [] marsObjs_;
    }
    numMars_ = *(int *) targv[1];
    if (numMars_ < 2) numMars_ = 2;
    printf("MARS with bagging: no. of MARS set to = %d.\n", numMars_);
    if (targc > 2) printf("INFO: num_mars setParam has > 2 arguments\n");
    strcpy(cString, "mars_params");
    argv[0] = (char *) cString;
    argv[1] = (char *) &maxBasis_;
    argv[2] = (char *) &varPerBasis_;
    psConfig_.RSExpertModeSaveAndReset();
    psConfig_.InteractiveSaveAndReset();
    marsObjs_ = new Mars*[numMars_];
    for (ii = 0; ii < numMars_; ii++) 
    {
      marsObjs_[ii] = new Mars(nInputs_, nSamples_);
      marsObjs_[ii]->setParams(3, argv);
    }
    //**/strcpy(cString, "no_gen");
    //**/argv[0] = (char *) cString;
    //**/for (ii = 0; ii < numMars_; ii++) marsObjs_[ii]->setParams(1, argv);
    psConfig_.RSExpertModeRestore();
    psConfig_.InteractiveRestore();
  }
  else if (targc == 5 && !strcmp(targv[0], "mars_sample"))
  {
    itmp = *(int *) targv[1];
    if (itmp < 0 || itmp >= numMars_)
    {
      printf("MarsBag ERROR: in loading sample - invalid index.\n");
      exit(1);
    }
    leng = *(int *) targv[2];
    if (leng != nSamples_)
    {
      printf("MarsBag ERROR: in loading sample - nSamples mismatch.\n");
      exit(1);
    }
    if (MatDataSetX_.nrows() <= 0)
    {
      MatDataSetX_.setFormat(PS_MAT2D);
      MatDataSetX_.setDim(numMars_, nSamples_*nInputs_);
      MatDataSetY_.setFormat(PS_MAT2D);
      MatDataSetY_.setDim(numMars_, nSamples_);
    }
    Xdata = (double *) targv[3];
    Ydata = (double *) targv[4];
    if (leng*nInputs_ > MatDataSetX_.ncols())
    {
      printf("MarsBag ERROR: invalid sample size of incoming sample %d.\n",
             leng);
      printf("        Incoming  sample size = %d\n", leng);
      printf("        Allowable sample size = %d\n", MatDataSetX_.ncols());
      printf("TERMINATE\n");
      exit(1);
    }
    for (ii = 0; ii < leng*nInputs_; ii++)
      MatDataSetX_.setEntry(itmp, ii, Xdata[ii]);
    for (ii = 0; ii < leng; ii++) 
      MatDataSetY_.setEntry(itmp, ii, Ydata[ii]);
  }
  else if (targc == 3 && !strcmp(targv[0], "mars_params"))
  {
    maxBasis_ = *(int *) targv[1];
    varPerBasis_ = *(int *) targv[2];
    printf("MARS with bagging: numBasis    set to = %d.\n", maxBasis_);
    printf("MARS with bagging: varPerBasis set to = %d.\n", varPerBasis_);
  }
  else
  {
    printf("MarsBagg setParams ERROR: invalid command %s.\n", targv[0]);
    printf("TERMINATE\n");
    exit(1);
  }
  return 0.0;
}

// ************************************************************************
// read mars information
// ------------------------------------------------------------------------
int MarsBagg::readRSCode(int index)
{
  int    ii, jj, idata, count;
  double ddata;
  char   line[1001], word[1001];
  FILE   *fp;

  fp = fopen("psuade_rs.info", "r");
  if (fp == NULL || index >= numMars_ || index < 0) return 0;
  if (index == 0)
  {
    if (marsFms_ != NULL)
    {
      for (ii = 0; ii < numMars_; ii++)
      {
        if (marsFms_[ii] != NULL) delete marsFms_[ii];
        if (marsIms_[ii] != NULL) delete marsIms_[ii];
      }
      delete [] marsFms_;
      delete [] marsIms_;
    }
    marsFms_  = new psVector*[numMars_];
    marsIms_  = new psIVector*[numMars_];
    marsNfms_.setLength(numMars_);
    marsNims_.setLength(numMars_);
    marsXMeans_.setFormat(PS_MAT2D);
    marsXMeans_.setDim(numMars_, nInputs_);
    marsXStds_.setFormat(PS_MAT2D);
    marsXStds_.setDim(numMars_, nInputs_);
    for (ii = 0; ii < numMars_; ii++)
    {
      marsNfms_[ii] = marsNims_[ii] = 0;
      marsFms_[ii] = NULL;
      marsIms_[ii] = NULL;
    }
  }
  while (1) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%s",word);
    if (!strcmp(word, "FM")) break;
  }
  sscanf(line, "%s %d",word,&count);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  if (count > 0)
  {
    marsNfms_[index] = count;
    marsFms_[index] = new psVector();
    marsFms_[index]->setLength(count);
    for (ii = 0; ii < count; ii++) 
    {
      fgets(line, 500, fp);
      sscanf(line, "%lg", &ddata);
      marsFms_[index]->setEntry(ii, ddata);
    }
  }
  while (1) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%s",word);
    if (!strcmp(word, "IM")) break;
  }
  sscanf(line, "%s %d",word,&count);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  if (count > 0)
  {
    marsNims_[index] = count;
    marsIms_[index] = new psIVector();
    marsIms_[index]->setLength(count);
    for (ii = 0; ii < count; ii++) 
    {
      fgets(line, 500, fp);
      sscanf(line, "%d", &idata);
      marsIms_[index]->setEntry(ii,idata);
    }
  }
  while (1) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%s",word);
    if (!strcmp(word, "XM")) break;
  }
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  for (ii = 0; ii < nInputs_; ii++) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%lg", &ddata);
    marsXMeans_.setEntry(index, ii, ddata);
  }
  while (1) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%s",word);
    if (!strcmp(word, "XS")) break;
  }
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  fgets(line, 500, fp);
  for (ii = 0; ii < nInputs_; ii++) 
  {
    fgets(line, 500, fp);
    sscanf(line, "%lg", &ddata);
    marsXStds_.setEntry(index, ii, ddata);
  }
  fclose(fp);
  return 0;
}

// ************************************************************************
// generate marsbag information
// ------------------------------------------------------------------------
void MarsBagg::genRSCode()
{
  int    mm, ii, countFm, countIm, idata;
  double ddata;
  FILE   *fp;

  fp = fopen("psuade_rs.info", "w");
  if (fp == NULL)
  {
    printf("ERROR: Cannot open file psuade_rs.info.\n");
    return;
  }
  fprintf(fp,"/* This file contains information to re-construct MARSBag\n");
  fprintf(fp,"   response surface offline. Follow the steps below:\n");
  fprintf(fp,"   1. Rename this file to, say, main.c\n");
  fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) and run \n");
  fprintf(fp,"      (optionally, to output also the std. dev., uncomment\n");
  fprintf(fp,"      the corresponding lines below and compile with lapack.)\n");
  fprintf(fp,"   3. Run: main input output (input file has number of \n");
  fprintf(fp,"      inputs followed by the input values. Upon termination,\n");
  fprintf(fp,"      the result will be stored in 'output' */\n");
  fprintf(fp,"/* *************************************/\n");
  fprintf(fp,"/* MARSBag interpolator from PSUADE.   */\n");
  fprintf(fp,"/* ====================================*/\n");
  fprintf(fp,"#include <math.h>\n");
  fprintf(fp,"#include <stdlib.h>\n");
  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
  fprintf(fp,"main(int argc, char **argv) {\n");
  fprintf(fp,"  int    i, iOne=1, nInps;\n");
  fprintf(fp,"  double X[%d], Y, S;\n",nInputs_);
  fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
  fprintf(fp,"  if (argc < 3) {\n");
  fprintf(fp,"     printf(\"ERROR: not enough argument.\\n\");\n");
  fprintf(fp,"     exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
  fprintf(fp,"  if (fIn == NULL) {\n");
  fprintf(fp,"     printf(\"ERROR: cannot open input file.\\n\");\n");
  fprintf(fp,"     exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fIn, \"%%d\", &nInps);\n");
  fprintf(fp,"  if (nInps != %d) {\n", nInputs_);
  fprintf(fp,"    printf(\"ERROR - wrong nInputs.\\n\");\n");
  fprintf(fp,"    exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  for (i=0; i<%d; i++) fscanf(fIn, \"%%lg\", &X[i]);\n",
          nInputs_);
  fprintf(fp,"  fclose(fIn);\n");
  fprintf(fp,"  interpolate(iOne,X,&Y,&S);\n");
  fprintf(fp,"  printf(\"Y = %%e\\n\", Y);\n");
  fprintf(fp,"  fOut = fopen(argv[2], \"w\");\n");
  fprintf(fp,"  if (fOut == NULL) {\n");
  fprintf(fp,"     printf(\"ERROR: cannot open output file.\\n\");\n");
  fprintf(fp,"     exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fprintf(fOut,\" %%e\\n\", Y);\n");
  fprintf(fp,"  fclose(fOut);\n");
  fprintf(fp,"}\n\n");
  fprintf(fp,"/* *************************************/\n");
  fprintf(fp,"/* **** MARS interpolation function ****/\n");
  fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
  fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
  fprintf(fp," * ... */\n");
  fprintf(fp,"/* ====================================*/\n");
  countFm = 0;
  for (mm = 0; mm < numMars_; mm++)
    countFm = (marsNfms_[mm] > countFm) ? marsNfms_[mm] : countFm;
  countIm = 0;
  for (mm = 0; mm < numMars_; mm++)
    countIm = (marsNims_[mm] > countIm) ? marsNims_[mm] : countIm;
  fprintf(fp,"int getCoefs(int, double *, int *,double *, double *);\n");
  fprintf(fp,"int icat(double, int, double *);\n");
  fprintf(fp,"int interpolate(int npts,double *X,double *Y,double *S){\n");
  fprintf(fp,"  int    k, nk, ss, nn, ip, ind, nmars=%d, tt, *im;\n",
          numMars_); 
  fprintf(fp,"  int    nInps=%d;\n", nInputs_);
  fprintf(fp,"  double *YY, Yt, az, *tb, *cm, phi, t, u, v, *XX, *fm;\n");
  fprintf(fp,"  double mean, std, *xm, *xs;\n");
  fprintf(fp,"  XX = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  YY = (double *) malloc(nmars*sizeof(double));\n");
  fprintf(fp,"  fm = (double *) malloc(sizeof(double)* %d);\n",countFm);
  fprintf(fp,"  im = (int *) malloc(sizeof(int)* %d);\n",countIm);
  fprintf(fp,"  xm = (double *) malloc(sizeof(double)* %d);\n",nInputs_);
  fprintf(fp,"  xs = (double *) malloc(sizeof(double)* %d);\n",nInputs_);
  fprintf(fp,"  for (ss = 0; ss < npts; ss++) {\n");
  fprintf(fp,"    Y[ss] = 0.0;\n");
  fprintf(fp,"    for (tt = 0; tt < nmars; tt++) {\n");
  fprintf(fp,"      getCoefs(tt, fm, im, xm, xs);\n");
  fprintf(fp,"      for (nn = 0; nn < nInps; nn++)\n");
  fprintf(fp,"        XX[nn]=(X[nn+nInps*ss]-xm[nn])/xs[nn];\n");
  fprintf(fp,"      nk  = im[4];\n");
  fprintf(fp,"      ind = im[10] - 1; az  = fm[ind];\n");
  fprintf(fp,"      ind = im[11] - 1; tb  = &fm[ind];\n"); 
  fprintf(fp,"      ind = im[14] - 1; cm  = &fm[ind];\n");
  fprintf(fp,"      Yt = az;\n"); 
  fprintf(fp,"      for (nn = 0; nn < nk; nn++) {\n");
  fprintf(fp,"        if (tb[nn*5] != 0.0) {\n");
  fprintf(fp,"          phi = 1.0;\n");
  fprintf(fp,"          ip = nn;\n");
  fprintf(fp,"          while (ip > -1) {\n");
  fprintf(fp,"            t = tb[ip*5+1];\n");
  fprintf(fp,"            v = t;\n");
  fprintf(fp,"            if (v < 0) v = - v;\n");
  fprintf(fp,"            ind = floor(v+0.1) - 1;\n"); 
  fprintf(fp,"            if (cm[2*ind] <= 0.0) {\n");
  fprintf(fp,"              v = 1.0;\n");
  fprintf(fp,"              if (t < 0.0) v = -1.0;\n");
  fprintf(fp,"              u = v * (XX[ind]-tb[ip*5+2]); \n");
  fprintf(fp,"              if (u < 0.0) u = 0.0;\n");
  fprintf(fp,"            }\n");
  fprintf(fp,"            else {\n");
  fprintf(fp,"              k = icat(XX[ind], ind, cm);\n");
  fprintf(fp,"              if (k != 0) {\n");
  fprintf(fp,"                ind = floor(tb[ip*5+2]+0.1) - 1;\n");
  fprintf(fp,"                u = cm[k+ind];\n");
  fprintf(fp,"              }\n");
  fprintf(fp,"              else u = 0.0;\n");
  fprintf(fp,"              if (t < 0.0) {\n");
  fprintf(fp,"                if (u == 0.0) u = 1.0;\n");
  fprintf(fp,"                else          u = 0.0;\n");
  fprintf(fp,"              }\n");
  fprintf(fp,"            } \n");
  fprintf(fp,"            if (u == 0.0) {\n");
  fprintf(fp,"              phi = 0.0;\n");
  fprintf(fp,"              break;\n");
  fprintf(fp,"            }\n");
  fprintf(fp,"            else {\n");
  fprintf(fp,"              phi *= u;\n");
  fprintf(fp,"              ip = floor(tb[ip*5+3] + 0.1) - 1;\n");
  fprintf(fp,"            }\n");
  fprintf(fp,"          }\n");
  fprintf(fp,"          Yt += tb[nn*5] * phi;\n");
  fprintf(fp,"        }\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      YY[tt] = Yt;\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    mean = 0.0;\n");
  fprintf(fp,"    for (tt = 0; tt < nmars; tt++) mean += YY[tt];\n");
  fprintf(fp,"    mean /= nmars;\n");
  fprintf(fp,"    std = 0.0;\n");
  fprintf(fp,"    for (tt=0; tt<nmars; tt++) std += pow(YY[tt]-mean,2.0);\n");
  fprintf(fp,"    std = sqrt(std/(nmars-1));\n");
  fprintf(fp,"    Y[ss] = mean;\n");
  fprintf(fp,"    S[ss] = std;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  free(XX);\n");
  fprintf(fp,"  free(YY);\n");
  fprintf(fp,"  free(fm);\n");
  fprintf(fp,"  free(im);\n");
  fprintf(fp,"  free(xm);\n");
  fprintf(fp,"  free(xs);\n");
  fprintf(fp,"  return 0;\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ====================================*/\n");
  fprintf(fp,"int icat(double X, int input, double *cm) {\n");
  fprintf(fp,"  int j0, j1, j2, k, rdata;\n");
  fprintf(fp,"  j0 = floor(cm[2*input] + 0.1);\n");
  fprintf(fp,"  j1 = j0; j2 = floor(cm[2*input+1] + 0.1);\n");
  fprintf(fp,"  while (j2 != (j1+1)) {\n");
  fprintf(fp,"    k = floor(0.5*(j1+j2)) - 1;\n");
  fprintf(fp,"    if (cm[k] == X) {\n");
  fprintf(fp,"      rdata = k - j0; break;\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    else if (cm[k] < X) j1 = k;\n");
  fprintf(fp,"    else                j2 = k;\n");
  fprintf(fp,"    if (X == cm[j1-1]) rdata = j2 - j0;\n");
  fprintf(fp,"    else {\n");
  fprintf(fp,"      if (X == cm[j2-1]) rdata = j2 - j0;\n");
  fprintf(fp,"      else               rdata = 0;\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  return rdata;\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ====================================*/\n");
  for (mm = countFm-1; mm >= 0; mm--)
  {
    for (ii = 0; ii < numMars_; ii++)
    {
      if (marsNfms_[ii] > mm)
      {
         ddata = marsFms_[ii]->getEntry(mm);
         if (ddata < PSUADE_UNDEFINED) break;
      }
    }
    if (ii != numMars_) break;
    countFm--;
  }
  if (countFm > 0)
  {
    fprintf(fp,"static double\n");
    fprintf(fp,"FMS[%d][%d] = \n", countFm, numMars_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < countFm; mm++)
    {
      fprintf(fp,"  {");
      for (ii = 0; ii < numMars_-1; ii++)
      {
        if (marsNfms_[ii] > mm)
        {
          ddata = marsFms_[ii]->getEntry(mm);
          fprintf(fp," %24.16e,", ddata);
        }
        else fprintf(fp," 0,");
      }
      if (marsNfms_[numMars_-1] > mm)
      {
        ddata = marsFms_[numMars_-1]->getEntry(mm);
        fprintf(fp," %24.16e },\n", ddata);
      }
      else fprintf(fp," 0 },\n");
    }
    fprintf(fp,"};\n");
  }
  for (mm = countIm-1; mm >= 0; mm--)
  {
    for (ii = 0; ii < numMars_; ii++)
    {
      if (marsNims_[ii] > mm)
      {
        idata = marsIms_[ii]->getEntry(mm);
        if (idata != -9999) break;
      }
    }
    if (ii != numMars_) break;
    countIm--;
  }
  if (countIm > 0)
  {
    fprintf(fp,"static int\n");
    fprintf(fp,"IMS[%d][%d] = \n", countIm, numMars_);
    fprintf(fp,"{\n");
    for (mm = 0; mm < countIm; mm++)
    {
      fprintf(fp,"  {");
      for (ii = 0; ii < numMars_-1; ii++)
      {
        if (marsNims_[ii] > mm)
        {
          idata = marsIms_[ii]->getEntry(mm);
          fprintf(fp," %d,", idata);
        }
        else fprintf(fp," 0,");
      }
      if (marsNims_[numMars_-1] > mm)
      {
        idata = marsIms_[numMars_-1]->getEntry(mm);
        fprintf(fp," %d },\n", idata);
      }
      else fprintf(fp," 0 },\n");
    }
  }
  fprintf(fp,"};\n");
  fprintf(fp,"static double\n");
  fprintf(fp,"XMeans[%d][%d] = \n", numMars_, nInputs_);
  fprintf(fp,"{\n");
  double **mat2DM = marsXMeans_.getMatrix2D();
  double **mat2DS = marsXStds_.getMatrix2D();
  for (mm = 0; mm < numMars_; mm++)
  {
    fprintf(fp,"  {");
    for (ii = 0; ii < nInputs_-1; ii++)
      fprintf(fp," %24.16e,", mat2DM[mm][ii]);
    fprintf(fp," %24.16e },\n", mat2DM[mm][nInputs_-1]);
  }
  fprintf(fp,"};\n");
  fprintf(fp,"static double\n");
  fprintf(fp,"XStds[%d][%d] = \n", numMars_, nInputs_);
  fprintf(fp,"{\n");
  for (mm = 0; mm < numMars_; mm++)
  {
    fprintf(fp,"  {");
    for (ii = 0; ii < nInputs_-1; ii++)
      fprintf(fp," %24.16e,", mat2DS[mm][ii]);
    fprintf(fp," %24.16e },\n", mat2DS[mm][nInputs_-1]);
  }
  fprintf(fp,"};\n");
  fprintf(fp,"/* ====================================*/\n");
  fprintf(fp,
      "int getCoefs(int ind,double *fm,int *im,double *XM,double *XS)\n");
  fprintf(fp,"{\n");
  fprintf(fp,"  int mm;\n");
  fprintf(fp,"  for (mm = 0; mm < %d; mm++)\n",countFm);
  fprintf(fp,"    fm[mm] = FMS[mm][ind];\n");
  fprintf(fp,"  for (mm = 0; mm < %d; mm++)\n",countIm);
  fprintf(fp,"    im[mm] = IMS[mm][ind];\n");
  fprintf(fp,"  for (mm = 0; mm < %d; mm++)\n",nInputs_);
  fprintf(fp,"    XM[mm] = XMeans[ind][mm];\n");
  fprintf(fp,"  for (mm = 0; mm < %d; mm++)\n",nInputs_);
  fprintf(fp,"    XS[mm] = XStds[ind][mm];\n");
  fprintf(fp,"  return 0;\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ====================================*/\n");
  fclose(fp);
  printf("MarsBagg parameters are now stored in psuade_rs.info\n");

  //**/ python code
  fp = fopen("psuade_rs.py", "w");
  if (fp == NULL)
  {
    printf("ERROR: Cannot open file psuade_rs.py.\n");
    return;
  }
  fwriteRSPythonHeader(fp);
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"# MARS with bagging interpolation\n");
  fprintf(fp,"#==================================================\n");
  fwriteRSPythonCommon(fp);
  fprintf(fp,"numMars=%d\n",numMars_);
  fprintf(fp,"nInputs=%d\n",nInputs_);
  for (mm = countFm-1; mm >= 0; mm--)
  {
    for (ii = 0; ii < numMars_; ii++)
    {
      if (marsNfms_[ii] > mm)
      {
        ddata = marsFms_[ii]->getEntry(mm);
        if (ddata < PSUADE_UNDEFINED) break;
      }
    }
    if (ii != numMars_) break;
    countFm--;
  }
  if (countFm > 0)
  {
    fprintf(fp,"FMS = [\n");
    for (mm = 0; mm < countFm; mm++)
    {
      fprintf(fp,"  [");
      for (ii = 0; ii < numMars_-1; ii++)
      {
        if (marsNfms_[ii] > mm)
        {
          ddata = marsFms_[ii]->getEntry(mm);
          fprintf(fp," %24.16e,", ddata);
        }
        else fprintf(fp," 0,");
      }
      if (marsNfms_[numMars_-1] > mm)
      {
        ddata = marsFms_[numMars_-1]->getEntry(mm);
        fprintf(fp," %24.16e ],\n", ddata);
      }
      else fprintf(fp," 0 ],\n");
    }
     fprintf(fp,"]\n");
  }
  for (mm = countIm-1; mm >= 0; mm--)
  {
    for (ii = 0; ii < numMars_; ii++)
    {
      if (marsNims_[ii] > mm)
      {
        idata = marsIms_[ii]->getEntry(mm);
        if (idata != -9999) break;
      }
    }
    if (ii != numMars_) break;
    countIm--;
  }
  if (countIm > 0)
  {
    fprintf(fp,"IMS = [\n");
    for (mm = 0; mm < countIm; mm++)
    {
      fprintf(fp,"  [");
      for (ii = 0; ii < numMars_-1; ii++)
      {
        if (marsNims_[ii] > mm)
        {
          idata = marsIms_[ii]->getEntry(mm);
          fprintf(fp," %d,", idata);
        }
        else fprintf(fp," 0,");
      }
      if (marsNims_[numMars_-1] > mm)
      {
        idata = marsIms_[numMars_-1]->getEntry(mm);
        fprintf(fp," %d ],\n", idata);
      }
      else fprintf(fp," 0 ],\n");
    }
  }
  fprintf(fp,"]\n");
  fprintf(fp,"XMeans = [\n");
  for (ii = 0; ii < nInputs_; ii++)
  {
    fprintf(fp,"  [");
    for (mm = 0; mm < numMars_-1; mm++)
      fprintf(fp," %24.16e,", mat2DM[mm][ii]);
    fprintf(fp," %24.16e ],\n", mat2DM[numMars_-1][ii]);
  }
  fprintf(fp,"]\n");
  fprintf(fp,"XStds = [\n");
  for (ii = 0; ii < nInputs_; ii++)
  {
    fprintf(fp,"  [");
    for (mm = 0; mm < numMars_-1; mm++)
      fprintf(fp," %24.16e,", mat2DS[mm][ii]);
    fprintf(fp," %24.16e ],\n", mat2DS[numMars_-1][ii]);
  }
  fprintf(fp,"]\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"def icat(X, input, fm, indcm,tt) :\n");
  fprintf(fp,"  j0 = int(fm[indcm+2*input][tt] + 0.1)\n");
  fprintf(fp,"  j1 = j0 \n");
  fprintf(fp,"  j2 = int(fm[indcm+2*input+1][tt] + 0.1)\n");
  fprintf(fp,"  while (j2 != (j1+1)) :\n");
  fprintf(fp,"    k = int(0.5*(j1+j2)+1.0e-8) - 1\n");
  fprintf(fp,"    if (fm[indcm+k][tt] == X) :\n");
  fprintf(fp,"      rdata = k - j0\n");
  fprintf(fp,"      break\n");
  fprintf(fp,"    elif (fm[indcm+k][tt] < X) :\n");
  fprintf(fp,"      j1 = k\n");
  fprintf(fp,"    else :\n");
  fprintf(fp,"      j2 = k;\n");
  fprintf(fp,"    if (X == fm[indcm+j1-1][tt]) : \n");
  fprintf(fp,"      rdata = j2 - j0\n");
  fprintf(fp,"    else :\n");
  fprintf(fp,"      if (X == fm[indcm+j2-1][tt]) : \n");
  fprintf(fp,"        rdata = j2 - j0\n");
  fprintf(fp,"      else :\n");
  fprintf(fp,"        rdata = 0\n");
  fprintf(fp,"  return rdata\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"# Interpolation function  \n");
  fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
  fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
  fprintf(fp,"# ... \n");
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"def interpolate(XX): \n");
  fprintf(fp,"  nSamp = int(len(XX) / nInputs + 1.0e-8)\n");
  fprintf(fp,"  X = nInputs * [0.0]\n");
  fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
  fprintf(fp,"  Yt = numMars * [0.0]\n");
  fprintf(fp,"  for ss in range(nSamp) : \n");
  fprintf(fp,"    for ii in range(nInputs) : \n");
  fprintf(fp,"      X[ii] = XX[ss*nInputs+ii]\n");
  fprintf(fp,"    for tt in range(numMars) : \n");
  fprintf(fp,"       nk  = IMS[4][tt]\n");
  fprintf(fp,"       ind = IMS[10][tt] - 1; az  = FMS[ind][tt]\n");
  fprintf(fp,"       indtb = IMS[11][tt] - 1\n");
  fprintf(fp,"       indcm = IMS[14][tt] - 1\n");
  fprintf(fp,"       yt = az\n");
  fprintf(fp,"       for nn in range(nk) : \n");
  fprintf(fp,"         if (FMS[indtb+nn*5][tt] != 0.0) :\n");
  fprintf(fp,"           phi = 1.0\n");
  fprintf(fp,"           ip = nn\n");
  fprintf(fp,"           while (ip > -1) :\n");
  fprintf(fp,"             ind = int(indtb+ip*5)+1\n");
  fprintf(fp,"             t = FMS[ind][tt]\n");
  fprintf(fp,"             v = t\n");
  fprintf(fp,"             if (v < 0): \n");
  fprintf(fp,"               v = - v\n");
  fprintf(fp,"             ind = int(v+0.1) - 1\n");
  fprintf(fp,"             ind1 = indcm+2*ind\n");
  fprintf(fp,"             if (FMS[ind1][tt] <= 0.0) :\n");
  fprintf(fp,"               v = 1.0\n");
  fprintf(fp,"               if (t < 0.0) : \n");
  fprintf(fp,"                 v = -1.0\n");
  fprintf(fp,"               dt = (X[ind]-XMeans[ind][tt])/XStds[ind][tt]\n");
  fprintf(fp,"               ind2 = int(indtb+ip*5+2)\n");
  fprintf(fp,"               u = v * (dt-FMS[ind2][tt]) \n");
  fprintf(fp,"               if (u < 0.0) : \n");
  fprintf(fp,"                 u = 0.0\n");
  fprintf(fp,"             else :\n");
  fprintf(fp,"               dt = (X[ind]-XMeans[ind][tt])/XStds[ind][tt]\n");
  fprintf(fp,"               k = icat(dt, ind, FMS, indcm, tt)\n");
  fprintf(fp,"               if (k != 0) :\n");
  fprintf(fp,"                 ind = int(FMS[indtb+ip*5+2][tt]+0.1) - 1\n");
  fprintf(fp,"                 u = FMS[indcm+k+ind][tt]\n");
  fprintf(fp,"               else : \n");
  fprintf(fp,"                 u = 0.0\n");
  fprintf(fp,"               if (t < 0.0) :\n");
  fprintf(fp,"                 if (u == 0.0) : \n");
  fprintf(fp,"                   u = 1.0\n");
  fprintf(fp,"                 else : \n");
  fprintf(fp,"                   u = 0.0\n");
  fprintf(fp,"             if (u == 0.0) :\n");
  fprintf(fp,"               phi = 0.0\n");
  fprintf(fp,"               break\n");
  fprintf(fp,"             else :\n");
  fprintf(fp,"               phi *= u\n");
  fprintf(fp,"               ind3 = int(indtb+ip*5+3)\n");
  fprintf(fp,"               ip = int(FMS[ind3][tt] + 0.1) - 1\n");
  fprintf(fp,"           yt += FMS[indtb+nn*5][tt] * phi\n");
  fprintf(fp,"       Yt[tt] = yt\n");
  fprintf(fp,"    Ymean = 0.0\n");
  fprintf(fp,"    for tt in range(numMars) : \n");
  fprintf(fp,"       Ymean = Ymean + Yt[tt]\n");
  fprintf(fp,"    Ymean = Ymean / numMars\n");
  fprintf(fp,"    Ystd = 0.0\n");
  fprintf(fp,"    for tt in range(numMars) : \n");
  fprintf(fp,"       Ystd = Ystd + (Yt[tt] - Ymean) * (Yt[tt] - Ymean)\n");
  fprintf(fp,"    Ystd = math.sqrt(Ystd / (numMars - 1.0))\n");
  //**/ assume no output scaling, which is guaranteed when rs expert 
  //**/ mode is off
  fprintf(fp,"    Ys[ss*2]   = Ymean\n");
  fprintf(fp,"    Ys[ss*2+1] = Ystd\n");
  fprintf(fp,"  return Ys\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"# main program\n");
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"infileName  = sys.argv[1]\n");
  fprintf(fp,"outfileName = sys.argv[2]\n");
  fprintf(fp,"inputs = getInputData(infileName)\n");
  fprintf(fp,"outputs = interpolate(inputs)\n");
  fprintf(fp,"genOutputFile(outfileName, outputs)\n");
  fclose(fp);
  printf("FILE psuade_rs.py contains the final MarsBag interpolator.\n");
  return;
}

