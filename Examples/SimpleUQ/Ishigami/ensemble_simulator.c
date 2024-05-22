#include <math.h>
#include <stdio.h>
#include <stdlib.h>

main(int argc, char **argv)
{
   int    nSams, nInps, ss, ii;
   double X[3], Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut = fopen(argv[2], "w");

   if (fIn == NULL || fOut == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d %d", &nSams, &nInps);
   if (nInps != 3)
   {
      printf("Simulator ERROR - wrong nInputs.\n");
      exit(1);
   }
   for (ss = 0; ss < nSams; ss++)
   {
     for (ii = 0; ii < 3; ii++) fscanf(fIn, "%lg", &X[ii]);
     Y = sin(X[0]) + 7.0 * sin(X[1]) * sin(X[1]) + 
         0.1 * X[2] * X[2] * X[2] * X[2] * sin(X[0]);
     fprintf(fOut, "%24.16e\n", Y);
   }
   fclose(fIn);   
   fclose(fOut);   
}

