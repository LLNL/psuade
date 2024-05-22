#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    i, kk, nInputs, nSamples;
   double X[2], Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in files.\n");
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   if (fOut == NULL) 
   {
     printf("Simulator ERROR - cannot open out file %s\n",argv[2]);
     exit(1);
   }
   fscanf(fIn, "%d %d", &nSamples, &nInputs);
   for (kk = 0; kk < nSamples; kk++)
   {
     for (i = 0; i < nInputs; i++) fscanf(fIn, "%lg", &X[i]);
     Y = 0;
     if (X[0] >= -0.5 && X[0] <= 0.5) Y += X[0];
     if (X[1] >= -0.25 && X[1] <= 0.25) Y += 5 * X[1];
     fprintf(fOut, " %24.16e\n", Y);
     fflush(fOut);   
   }
   fclose(fOut);   
   fclose(fIn);   
}

