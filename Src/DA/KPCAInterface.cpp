// ************************************************************************
// Copyright (c) 2015   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// DASSI is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// DASSI is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// KPCA Interface functions 
// DATE   : 2019
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "psVector.h"
#include "PsuadeUtil.h"
#include "KPCA.h"
#include "MetropolisHastingsMCMC.h"

// ************************************************************************
// Forward operation
// ------------------------------------------------------------------------
int KPCAForward(char *inModelFile, char *inDataFile, char *outDataFile)
{
  int    ii;
  double ddata;
  char   pString[1001], dataFile[1000], outFile[1000], modelFileName[1000];
  FILE   *fp=NULL;
  psVector vecXT, vecWT;

  //**/ get model file information
  if (inModelFile != NULL) strcpy(modelFileName,inModelFile);
  else
  {
    strcpy(pString,"Enter the model file name created by createKPCA: ");
    getString(pString, modelFileName);
    modelFileName[strlen(modelFileName)-1] = '\0';
  }

  //**/ instantiate KPCA object and extract parameters
  KPCA *kpcaObj = new KPCA();
  kpcaObj->setOutputLevel(1);
  int status = kpcaObj->retrieveKPCA(modelFileName);
  int fieldDim = kpcaObj->getSnapshotDim();
  int rdim = kpcaObj->getReducedDimension();

  //**/ get snapshot 
  if (inDataFile != NULL) strcpy(dataFile,inDataFile);
  else
  {
    snprintf(pString,100,
      "Enter the name of the snapshot file (length=%d): ",fieldDim);
    getString(pString, dataFile);
    dataFile[strlen(dataFile)-1] = '\0';
  }
  fp = fopen(dataFile, "r");
  if (fp == NULL)
  {
    printf("ERROR: snapshot file %s cannot be read.\n",dataFile);
    delete kpcaObj;
    return 1;
  }
  vecXT.setLength(fieldDim);
  fp = fopen(dataFile, "r");
  for (ii = 0; ii < fieldDim; ii++)
  {
    fscanf(fp, "%lg", &ddata);
    vecXT[ii] = ddata;
  }
  fclose(fp);

  //**/ get feature file name 
  if (outDataFile != NULL) strcpy(outFile,outDataFile);
  else
  {
    snprintf(pString,100,
       "Enter the file name for storing the feature vector: ");
    getString(pString, outFile);
    outFile[strlen(outFile)-1] = '\0';
  }

  //**/ projection 
  vecWT.setLength(rdim);
  kpcaObj->projectFeatureToStdRV(vecXT, vecWT);

  //**/ writing to file 
  fp = fopen(outFile, "w");
  if (fp == NULL) 
  {
    printf("ERROR: cannot write to feature file %s.\n",outFile);
    delete kpcaObj;
    return 1;
  }
  for (ii = 0; ii < rdim; ii++) fprintf(fp,"%16.8e ",vecWT[ii]);
  fprintf(fp, "\n");
  fclose(fp);
  delete kpcaObj;
  printf("Feature file %s has been created.\n", outFile);
  return 0;
}

// ************************************************************************
// Inverse reconstructioin
// ------------------------------------------------------------------------
int KPCAInverse(char *inModelFile, char *inDataFile, char *outDataFile,
                KPCA **kpcaIn)
{
  int    ii, status;
  double ddata;
  char   pString[1000], dataFile[1000], outFile[1000], modelFileName[1000];
  FILE   *fp=NULL;
  psVector vecXT, vecWT;
  KPCA   *kpcaObj;

  //**/ get and retrieve KPCA model information
  if (inModelFile != NULL) strcpy(modelFileName,inModelFile);
  else
  {  
    strcpy(pString,"Enter the model file name created by kpca_create: ");
    getString(pString, modelFileName);
    modelFileName[strlen(modelFileName)-1] = '\0';
  }
  fp = fopen(modelFileName, "r");
  if (fp == NULL)
  {
    printf("KPCA ERROR: model file not found.\n");
    return 0;
  }
  else fclose(fp);

  //*/ instantiate KPCA and extract parameters
  if (kpcaIn == NULL || (*kpcaIn) == NULL)
  {
    kpcaObj = new KPCA();
    kpcaObj->setOutputLevel(0);
    status = kpcaObj->retrieveKPCA(modelFileName);
    if (kpcaIn != NULL) (*kpcaIn) = kpcaObj;
  }
  else kpcaObj = *kpcaIn;

  int fieldDim = kpcaObj->getSnapshotDim();
  int rdim = kpcaObj->getReducedDimension();

  //**/ get and read feature file 
  if (inDataFile != NULL) strcpy(dataFile, inDataFile);
  else
  {
    snprintf(pString,100,
        "Enter the name of the feature file (length=%d): ", rdim);
    getString(pString, dataFile);
    dataFile[strlen(dataFile)-1] = '\0';
  }
  fp = fopen(dataFile, "r");
  if (fp == NULL)
  {
    printf("ERROR: feature file %s cannot be read.\n",dataFile);
    delete kpcaObj;
    return 1;
  }
  else fclose(fp);

  vecXT.setLength(rdim);
  fp = fopen(dataFile, "r");
  for (ii = 0; ii < rdim; ii++)
  {
    fscanf(fp, "%lg", &ddata);
    vecXT[ii] = ddata;
    printf("Feature %5d = %12.5e\n", ii+1, ddata);
  }
  fclose(fp);

  //**/ get output file name
  if (outDataFile != NULL) strcpy(outFile, outDataFile);
  else
  {
    snprintf(pString,100,
       "Enter the file name for storing the reconstructed vector: ");
    getString(pString, outFile);
    outFile[strlen(outFile)-1] = '\0';
  }

  //**/ reconstruction 
  vecWT.setLength(fieldDim);
  kpcaObj->genSnapshotFromStdRV(vecXT, vecWT);

  //**/ writing to file 
  fp = fopen(outFile, "w");
  if (fp == NULL) 
  {
    printf("ERROR: cannot write to snapshot file %s.\n",outFile);
    delete kpcaObj;
    return 1;
  }
  for (ii = 0; ii < fieldDim; ii++) fprintf(fp,"%16.8e\n",vecWT[ii]);
  printf("Reconstructed output 0 = %12.5e\n", vecWT[0]);
  fprintf(fp, "\n");
  fclose(fp);
  printf("Snapshot file %s has been created.\n", outFile);
  return 0;
}

// ************************************************************************
// Inverse reconstructioin
// ------------------------------------------------------------------------
int MCMCWithKPCA()
{
  int    ii;
  double ddata;
  char   pString[1000], winput[1000]; 
  FILE   *fp=NULL;

  //**/ get the likelihood file
  char likelihoodFunc[1000];
  printf("The likelihood operator takes the simulator output\n");
  printf("and the measurement data file, and compute a cost\n");
  printf("such as the mean square of the errors between the\n");
  printf("simulation outputs and the measurements. Afterward,\n");
  printf("the likelihood should be computed by: \n");
  printf("    C * exp(-0.5*cost)\n");
  printf("where C is some constant.\n");
  snprintf(pString,100,"Name of the likelihood function: ");
  getString(pString, likelihoodFunc);
  likelihoodFunc[strlen(likelihoodFunc)-1] = '\0';
  fp = fopen(likelihoodFunc, "r");
  if (fp == NULL)
  {
    printf("ERROR: likelihood operator file %s does not exist.\n",
           likelihoodFunc);
    return 1;
  }

  //**/ get initial guess file 
  char igfile[1000];
  printf("You have the option to enter an initial guess for X\n");
  printf("where X is a low-dimensional feature vector. This file\n");
  printf("should have m floats where m is the dimension of the\n");
  printf("feature vector. If you enter 'none', a random initial\n");
  printf("guess will be generated for you.\n");
  snprintf(pString,100,
          "Name of the initial guess file for X (or none): ");
  getString(pString, igfile);
  igfile[strlen(igfile)-1] = '\0';
  snprintf(pString,100,"Enter the dimension of X? ");
  int rdim = getInt(1,10000,pString);
  psVector vecIG;
  vecIG.setLength(rdim);
  if (!strcmp(igfile,"none"))
    for (ii = 0; ii < rdim; ii++) vecIG[ii] = 0;
  else
  {
    fp = fopen(igfile, "r");
    if (fp == NULL)
    {
      printf("ERROR: initial guess file %s does not exist.\n",igfile);
      printf("INFO: use random initial guess.\n");
      for (ii = 0; ii < rdim; ii++) vecIG[ii] = PSUADE_drand();
    }
    else
    {
      for (ii = 0; ii < rdim; ii++) 
      {
        fscanf(fp, "%lg", &ddata);
        vecIG[ii] = ddata;
      }
      fclose(fp);
    }
  }

  //**/ get MCMC sample size
  snprintf(pString,100,"Enter MCMC sample size : ");
  int mcmcMaxIts = getInt(1,10000, pString);

  //**/ get posterior file 
  char postfile[1000];
  snprintf(pString,100,"Enter the name of the posterior file: ");
  getString(pString, postfile);
  postfile[strlen(postfile)-1] = '\0';

  //**/ get posterior file 
  snprintf(pString,100,"Want to turn on break point for debug? (y or n) ");
  getString(pString, winput);

  //**/ run MCMC
  MetropolisHastingsMCMC *mcmcPtr = new MetropolisHastingsMCMC();
  int randSeed = PSUADE_rand();
  mcmcPtr->setInitialGuess(vecIG);
  mcmcPtr->setPosteriorFile(postfile);
  mcmcPtr->setLikelihoodFunction(likelihoodFunc);
  if (winput[0] == 'y') mcmcPtr->setBreakPoint();
  mcmcPtr->initialize(rdim, randSeed, mcmcMaxIts);
  mcmcPtr->runMCMC();
  return 0;
}

