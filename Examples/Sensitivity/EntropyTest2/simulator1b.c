#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    ii, kk, nSamples, nInputs;
   double X[1], Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   fscanf(fIn, "%d %d", &nSamples, &nInputs);
   printf("nSamples = %d\n", nSamples);
   for (kk = 0; kk < nSamples; kk++)
   {
     for (ii = 0; ii < nInputs; ii++) fscanf(fIn, "%lg", &X[ii]);
     Y = 0;
     if (X[0] >= -0.25 && X[0] <= 0.25) Y = X[0];
     fprintf(fOut, " %24.16e\n", Y);
   }
   fclose(fOut);
}

