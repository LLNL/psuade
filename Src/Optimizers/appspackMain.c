/* ************************************************************************
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
// appspack to PSUADE interface program
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "PsuadeUtil.h"

int getPatternData(char *inputString, char *pattern, double *value);
int createInputFiles(int sampleID, int, double *inputs, char **, char *);
int removePattern(char *inputString, char *outputString, char *pattern);
int substitutePattern(char *, char *, char *pattern, double value);

int main(int argc, char **argv)
{
   int    sampleID, nInputs, nOutputs, i, outputCount, status, length;
   double *inputs, *outputs, value;
   char   appDriver[200], appOutputFile[200], appInputFile[200];
   char   appInputTemplate[200], **inputNames, **outputNames;
   char   lineIn[100], command[501], filename[200];
   char   outfile[200], outfileTmp[200], appsPsuadeFile[200];
   FILE   *fIn, *fOut;

   //----------------------------------------------------------------- 
   // check errors and input from files
   //----------------------------------------------------------------- 

   if (argc < 2)
   {
      printf("appspackMain ERROR : argc < 2.\n");
      exit(1);
   }
   snprintf(appsPsuadeFile,100,"appspackMain.in");
   if ((fIn = fopen(appsPsuadeFile,"r")) == NULL) 
   {
      printf("appspackMain ERROR : cannot read input file %s\n",
             appsPsuadeFile);
      exit(1);
   }
   fscanf(fIn, "%d", &sampleID);
   fscanf(fIn, "%d", &nInputs);
   fscanf(fIn, "%d", &nOutputs);
   if (nInputs <= 0 || nOutputs <= 0)
   {
      printf("appspackMain ERROR : nInputs or nOutputs = 0\n");
      exit(1);
   }
   inputNames = (char **) malloc(sizeof(char*)*nInputs);
   outputNames = (char **) malloc(sizeof(char*)*nOutputs);
   for (i = 0; i < nInputs; i++)
   {
      inputNames[i] = (char *) malloc(sizeof(char)*200);
      fscanf(fIn, "%s", inputNames[i]);
   }
   outputs = (double *) malloc(sizeof(double)*nOutputs);
   for (i = 0; i < nOutputs; i++)
   {
      outputNames[i] = (char *) malloc(sizeof(char)*200);
      fscanf(fIn, "%s", outputNames[i]);
   }
   fscanf(fIn, "%s", appDriver);
   fscanf(fIn, "%s", appInputFile);
   fscanf(fIn, "%s", appInputTemplate);
   fscanf(fIn, "%s", appOutputFile);
   fclose(fIn);

   //----------------------------------------------------------------- 
   // create input file 
   //----------------------------------------------------------------- 

   if ((fIn = fopen(argv[1],"r")) == NULL) 
   {
      printf("appspackMain ERROR : cannot read input file %s\n",argv[1]);
      exit(1);
   }
   fscanf(fIn, "%d", &length);
   if (length != nInputs)
   {
      printf("appspackMain ERROR : nInputs mismatch\n");
      exit(1);
   }
   inputs = (double *) malloc(nInputs * sizeof(double));
   for (i = 0; i < nInputs; i++) fscanf(fIn, "%lg", &inputs[i]);

   if (createInputFiles(sampleID,nInputs,inputs,inputNames,
                        appInputTemplate) != 0)
   {
      snprintf(filename,100,"%s", appInputFile);
      fOut = fopen(filename, "w");
      if ( fOut == NULL )
      {
         printf("appspackMain::evaluate ERROR : ");
         printf("cannot open %s file\n", appInputFile);
         exit(1);
      }
      fprintf(fOut, "# SAMPLE_ID = %d\n", sampleID);
      for (i = 0; i < nInputs; i++)
         fprintf(fOut, "%s = %20.12e\n", inputNames[i], inputs[i]);
      fclose(fOut);
      snprintf(outfile,100,"%s", appInputFile);
   }
   else
   {
      removePattern(appInputTemplate, outfileTmp, ".Tmplt");
      length = strlen(outfileTmp);
      for (i = length-1; i >= 0; i--) if (outfileTmp[i] == '/') break; 
      snprintf(outfile,100,"%s", &outfileTmp[i+1]);
   }

   //----------------------------------------------------------------- 
   // execute user application 
   //----------------------------------------------------------------- 

   fIn = fopen(appDriver, "r");
   if (fIn == NULL)
   {
      printf("appspackMain ERROR : executable not found.\n");
      exit(1);
   }
   fclose(fIn);
   snprintf(command,500,"%s %s %s",appDriver,outfile,appOutputFile);
   if (strstr((const char*) appDriver, "rm ") != NULL &&
       strstr((const char*) appDriver, "mv ") != NULL ||
       strstr((const char*) appDriver, " -f ") != NULL ||
       strstr((const char*) appDriver, "/bin/") != NULL) 
   {
      printf("appspackMain::evaluate ERROR : \n");
      printf("\t\t for security reason do not use rm in executable.\n");
      exit(1);
   }
   system(command);   
   outputCount = 0;
   snprintf(filename,100,"%s", appOutputFile);
   if ((fIn=fopen(filename, "r")) == NULL) 
   {
      printf("appspackMain::evaluate ERROR : \n");
      printf("\t\t cannot find output file %s.\n",filename);
      exit(1);
   }
   while ((fgets(lineIn, 100, fIn) != NULL) && (feof(fIn) == 0))
   {
      for (i = 0; i < nOutputs; i++)
      {
         if (strstr((const char*)lineIn,(const char*)outputNames[i])!=NULL) 
         {
            getPatternData(lineIn, outputNames[i], &value);
            outputs[i] = value;
            outputCount++;
         } 
      } 
   }
   fclose(fIn);
   if (outputCount != nOutputs)
   {
      printf("appspackMain ERROR : nOutputs mismatch.\n"); 
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   if (fOut == NULL)
   {
      printf("appspackMain ERROR : cannot open output file.\n"); 
      exit(1);
   }
   for (i = 0; i < nOutputs; i++)
      fprintf(fOut, "%16.8e\n", outputs[i]);
   fclose(fOut);
   for (i = 0; i < nInputs; i++) free(inputNames[i]);
   for (i = 0; i < nOutputs; i++) free(outputNames[i]);
   free(inputNames);
   free(outputNames);
   free(inputs);
   free(outputs);
}

// ************************************************************************
// Given an input string inputString and a pattern, find the pattern
// and get the value
// ------------------------------------------------------------------------

int getPatternData(char *inputString, char *pattern, double *value)
{
   char *stringPtr, *stringPtr2;
   stringPtr  = strstr((const char*) inputString, (const char*) pattern);
   if (stringPtr == NULL) return -1;
   stringPtr2 = strstr((const char*) stringPtr, "=");
   if (stringPtr2 == NULL) return -1;
   sscanf(&stringPtr2[1], "%lg", value);
   return 0;
}

// ************************************************************************
// Given an input string inputString and a pattern, take out the
// pattern from the string
// ------------------------------------------------------------------------

int removePattern(char *inputString, char *outputString, char *pattern)
{
   int  i, pLength, cLength;
   char *stringPtr;

   for (i = 0; i < 80; i++) outputString[i] = '\0';
   strcpy(outputString, "");
   stringPtr = strstr((const char*) inputString, (const char*) pattern);
   if (stringPtr == NULL) return -1;
   cLength = strlen((const char *)inputString) -
             strlen((const char *)stringPtr);
   strncpy(outputString, (const char*) inputString, cLength);
   pLength = (int) strlen(pattern);
   strcpy(&outputString[cLength+1], (const char*) &stringPtr[pLength]);
   return 0;
}

// ************************************************************************
// create application input file
// ------------------------------------------------------------------------

int createInputFiles(int sampleID, int nInputs, double *inputs, 
                     char **inputNames, char *appInputTemplate)
{
   int  i, stat, length;
   char outfile[200], lineIn[100], lineOut[100], lineTemp[100];
   char outfileTmp[200];
   FILE *fIn, *fOut;

   // -------------------------------------------------------------
   // first check to see if input template exists
   // -------------------------------------------------------------

   if (!strcmp(appInputTemplate, "INVALID")) return 1;

   if ((fIn=fopen(appInputTemplate, "r")) == NULL)
   {
      printf("appspackMain::createInputFile ERROR : \n");
      printf("\t\t input template %s does not exist.\n",
             appInputTemplate);
      exit(1);
   }
   removePattern(appInputTemplate, outfileTmp, ".Tmplt");

   // -------------------------------------------------------------
   // open the output file
   // -------------------------------------------------------------

   length = strlen(outfileTmp);
   for (i = length-1; i >= 0; i--) if (outfileTmp[i] == '/') break;
   snprintf(outfile,100,"%s", &(outfileTmp[i+1]));
   if ((fOut=fopen(outfile, "w")) == NULL)
   {
      printf("appspackMain::createInputFile ERROR : \n");
      printf("\t\t cannot open output file %s.\n", outfile);
      fclose(fIn);
      exit(1);
   }
   while ((fgets(lineIn, 100, fIn) != NULL) && (feof(fIn) == 0))
   {
      strcpy(lineTemp, lineIn);
      for (i = 0; i < nInputs; i++)
      {
         if (strstr(lineTemp, inputNames[i]) != NULL)
         {
            stat = substitutePattern(lineTemp,lineOut,inputNames[i],
                                     inputs[i]);
            if ( stat != 0 ) strcpy(lineOut, lineTemp);
            else             strcpy(lineTemp, lineOut);
         }
      }
      fprintf(fOut, "%s", lineOut);
   }
   fclose(fIn);
   fclose(fOut);
   return 0;
}

// ************************************************************************
// Given an input string inputString and a pattern, replace the
// pattern with the given value.
// ------------------------------------------------------------------------
int substitutePattern(char *inputString, char *outputString, 
                      char *pattern, double value)
{
   char *stringPtr, valueStr[101];
   int  i, pLength, cLength;

   for ( i = 0; i < 80; i++ ) outputString[i] = '\0';
   strcpy(outputString, "");
   stringPtr  = strstr(inputString, (const char*) pattern);
   if (stringPtr == NULL) return 1;
   cLength = strlen((const char*) inputString) -
             strlen((const char*) stringPtr);
   strncpy(outputString, (const char*) inputString, cLength);
   snprintf(valueStr,100,"%24.16e ", value);
   strcat(outputString, valueStr);
   pLength = strlen((const char *) pattern);
   strcat(outputString, (const char*) &stringPtr[pLength]);
   return 0;
}

