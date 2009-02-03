/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bsid2d.h"
#include "randgen.h"

#include <iostream>
#include <stdlib.h>

using std::cout;
using std::cerr;
using std::flush;
using libbase::matrix;
using libbase::randgen;
using libcomm::bsid2d;

void visualtest(int seed, int type, double p)
   {
   // define an alternating input sequence
   const int M=5, N=5;
   matrix<bool> tx(M,N);
   switch(type)
      {
      case 0:
      case 1:
         tx = type;
         break;
      case 2:
         for(int i=0; i<M; i++)
            for(int j=0; j<N; j++)
               tx(i,j) = (i&1) ^ (j&1);
         break;
      default:
         assertalways("Invalid type");
      }
   cout << "Tx: " << tx << "\n";
   // pass that through the channel
   matrix<bool> rx1, rx2;
   // seed generator
   randgen prng;
   prng.seed(seed);
   // channel1 is a substitution-only channel
   bsid2d channel1(true,false,false);
   channel1.seedfrom(prng);
   channel1.set_parameter(p);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << "\n";
   // channel1 is an insdel-only channel
   bsid2d channel2(false,true,true);
   channel2.seedfrom(prng);
   channel2.set_parameter(p);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << "\n";
   }

int main(int argc, char *argv[])
   {
   int seed=0;
   if(argc > 1)
      seed = atoi(argv[1]);

   // create a test sequence and test 2D BSID transmission
   const double p = 0.1;
   visualtest(seed, 0, p); // all-zero sequence
   visualtest(seed, 1, p); // all-one sequenceee
   visualtest(seed, 2, p); // random sequence

   return 0;
   }
