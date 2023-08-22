#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PABS(X) (((X) > 0) ? X : -(X))

int main(int argc, char **argv)
{
   int    count, ii, nn=20;
   double X[20], Y, W[20];
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut = fopen(argv[2], "w");

   if (fIn == NULL || fOut == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d", &count);
   if (count != 20)
   {
      printf("Simulator ERROR - wrong nInputs.\n");
      exit(1);
   }
   for (ii = 0; ii < nn; ii++) fscanf(fIn, "%lg", &X[ii]);

   Y = 5.0 * X[11] / (1.0 + X[0]);
   Y += 5.0 * (X[3] - X[19]) * (X[3] - X[19]);
   Y += X[4];
   Y += 40 * X[18] * X[18] * X[18];
   Y -= 5.0 * X[0];
   Y += 0.05 * X[1];
   Y += 0.08 * X[2];
   Y += 0.03 * X[5];
   Y += 0.03 * X[6];
   Y -= 0.09 * X[8];
   Y -= 0.01 * X[9];
   Y -= 0.07 * X[10];
   Y += 0.25 * X[12] * X[12];
   Y -= 0.04 * X[13];
   Y += 0.06 * X[14];
   Y -= 0.01 * X[16];
   Y -= 0.03 * X[17];
   fprintf(fOut, "%24.16e\n", Y);
   fclose(fIn);   
   fclose(fOut);   
}

