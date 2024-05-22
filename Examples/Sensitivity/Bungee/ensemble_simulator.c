#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    ii, kk, nInputs=3, count, nSamples;
   double X[4], Y, kel=1.5, g=9.8;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   fscanf(fIn, "%d %d", &nSamples, &nInputs);
   for (kk = 0; kk < nSamples; kk++)
   {
      for (ii = 0; ii < nInputs; ii++) fscanf(fIn, "%lg", &X[ii]);
      Y = X[0] - 2.0 * X[1] * g / (kel * X[2]);
      fprintf(fOut, " %24.16e\n", Y);
   }
   fclose(fOut);   
}

