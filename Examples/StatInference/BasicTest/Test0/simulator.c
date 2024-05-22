#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   int    nInputs, ii;
   double X, Y;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut;
                                                                                
   if (fIn == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d", &nInputs);
   Y = 0;
   for (ii = 0; ii < nInputs; ii++)
   {
     fscanf(fIn, "%lg", &X);
     Y += (X - 0.5);
     //Y += 0.5 / (0.1 * 0.1) * (X - 0.5) * (X - 0.5);
   }
   fclose(fIn);
   fOut = fopen(argv[2], "w");
   fprintf(fOut, " %24.16e\n", Y);
   fclose(fOut);
}

