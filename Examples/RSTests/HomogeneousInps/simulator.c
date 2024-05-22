#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    ii, jj, kk, nInputs;
   double X[500], Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;

   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d", &nInputs);
   for (ii = 0; ii < nInputs; ii++) fscanf(fIn, "%lg", &X[ii]);
   Y = 1;
   for (ii = 0; ii < nInputs; ii++) Y += X[ii];
   for (ii = 0; ii < nInputs; ii++) 
     for (jj = ii; jj < nInputs; jj++) Y += 2.0 * X[ii] * X[jj];
   /*Y = 1 + exp(-Y/nInputs);*/

/*
   for (ii = 0; ii < nInputs; ii++) 
     for (jj = ii; jj < nInputs; jj++)
       for (kk = jj; kk < nInputs; kk++) 
         Y += X[ii] * X[jj] * X[kk];
*/
   fOut = fopen(argv[2], "w");
   fprintf(fOut, " %24.16e\n", Y);
   fclose(fOut);   
}

