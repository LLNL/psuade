#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PABS(X) (((X) > 0) ? X : -(X))

int main(int argc, char **argv)
{
   int    i, kk, nInputs, nSamples;
   double X[8], Y, AV[8], ddata;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut = NULL;

   if (fIn == NULL)
   {
      printf("SobolG ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fOut = fopen(argv[2], "w");
   fscanf(fIn, "%d %d", &nSamples, &nInputs);
   AV[0] = 0;
   AV[1] = 1;
   AV[2] = 4.5;
   AV[3] = 9.0;
   AV[4] = 99.0;
   AV[5] = 99.0;
   AV[6] = 99.0;
   AV[7] = 99.0;
   for (kk = 0; kk < nSamples; kk++)
   {
     for (i = 0; i < nInputs; i++) fscanf(fIn, "%lg", &X[i]);
     Y = 1.0;
     for (i = 0; i < nInputs; i++)
     {
        ddata = (PABS(4.0 * X[i] - 2.0) + AV[i]) / (1.0 + AV[i]);
        Y *= ddata;
     }
     fprintf(fOut, "%24.16e\n", Y);
   }
   fclose(fIn);   
   fclose(fOut);   
}

