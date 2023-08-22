#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    i, nInputs, count;
   double X[4], Y, kel=1.5, g=9.8;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d", &nInputs);
   for (i = 0; i < nInputs; i++) fscanf(fIn, "%lg", &X[i]);
   Y = X[0] - 2.0 * X[1] * g / (kel * X[2]);
   fOut = fopen(argv[2], "w");
   fprintf(fOut, " %24.16e\n", Y);
   fprintf(fOut, " 1.0\n");
   Y = - 2.0 * g / (kel * X[2]);
   fprintf(fOut, " %24.16e\n", Y);
   Y = 2.0 * g * X[1] / (kel * X[2] * X[2]);
   fprintf(fOut, " %24.16e\n", Y);
   fclose(fOut);   
}

