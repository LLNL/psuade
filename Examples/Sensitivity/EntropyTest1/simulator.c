#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    i, s, nInputs, count, nSamples;
   double X[20], Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   fscanf(fIn, "%d %d", &nSamples, &nInputs);
   for (s = 0; s < nSamples; s++)
   {
     for (i = 0; i < nInputs; i++) fscanf(fIn, "%lg", &X[i]);
     Y = X[0] * X[0] + 10 * drand48();
     fprintf(fOut, " %24.16e\n", Y);
   }
   fclose(fIn);   
   fclose(fOut);   
}

