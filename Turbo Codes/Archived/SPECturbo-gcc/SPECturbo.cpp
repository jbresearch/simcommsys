#include "turbo.h"

#include "logreal.h"
#include "rscc.h"

#include "helical.h"
#include "puncture_stipple.h"
#include "bpsk.h"
#include "awgn.h"
#include "randgen.h"
#include "commsys.h"
#include "timer.h"

#include <iostream.h>
#include <iomanip.h>
#include <math.h>


int main(int argc, char *argv[])
   {
   timer main_timer;
   
   // Simulation parameters
   const double SNR = 0.5;
   const double simtime = argc > 1 ? atoi(argv[1]) : 60;

   // Encoder (from generator matrix)
   const int k=1, n=2, m=2;
   matrix<bitfield> gen(k, n);
   gen(0, 0) = "111";
   gen(0, 1) = "101";
   rscc encoder(k, n, gen);
   // Block interleaver parameters
   const int rows = 13, cols = 12;
   const int tau = rows*cols + m;
   // Helical interleaver (from matrix size, hence block size)
   helical inter(tau, rows, cols);
   // Stipple puncturing
   const int sets=2;
   puncture_stipple punc(tau, k+sets*(n-k));

   // Modulation scheme
   bpsk modem;
   // Channel Model
   awgn chan;
   // Channel Codec (punctured, iterations, simile, endatzero)
   turbo<logreal> codec(encoder, modem, punc, chan, tau, sets, &inter, 10, true, true);
   // Source Generator
   randgen src;
   // The complete communication system
   commsys system(&src, &chan, &codec);

   // Work out at the SNR value required
   chan.set_snr(SNR);

   // Prepare for simulation run
   const int count = system.count();
   vector<double> est(count), sum(count);
   system.seed(0);

   // Time the simulation
   int frames = 0, passes = 0;
   main_timer.start();
   while(main_timer.elapsed() < simtime)
      {
      system.sample(est, frames);
      for(int j=0; j<count; j++)
         sum(j) += est(j);
      passes++;
      cerr << "\rWorking: " << int(100*main_timer.elapsed()/simtime) << "%";
      }
   cerr << "\n";
   main_timer.stop();

   // Work out averages
   for(int j=0; j<count; j++)
      sum(j) /= double(passes);

   // Tabulate standard results
   double std[] = {0.0955762, 0.999273,  0.0776172, 0.916424,  0.0713933, 0.821221,  0.0685143, 0.766715,  0.0669257, 0.728924,  0.066171, 0.704942,  0.0660732, 0.694041,  0.0660219, 0.68532,  0.0658356, 0.680233,  0.06557, 0.679506};

   // Print results (for confirming accuracy
   cout << "Results: (BER, FER)\n";
   cout << "~~~~~~~~~~~~~~~~~~~\n";
   for(int j=0; j<count; j+=2)
      cout << setprecision(6) << sum(j) << " (" << setprecision(3) << 100*(sum(j)-std[j])/std[j] << "%)\t" << setprecision(6) << sum(j+1) << " (" << setprecision(3) << 100*(sum(j+1)-std[j+1])/std[j+1] << "%)\n";
   cout << "\n";

   // Output timing statistics
   main_timer.divide(frames);
   cout << "Statistics: " << passes << " passes, " << frames << " frames.\n";
   cout << "SPECturbo: " << int(3600/main_timer.elapsed()) << " frames/hour (CPU " << int(main_timer.usage()) << "%)\n";
   cout << "SPECturbo: " << setprecision(4) << (1/main_timer.elapsed()) << " frames/sec\n";
   cout << "SPECturbo: " << setprecision(4) << (100/(main_timer.elapsed()*main_timer.usage())) << " frames/CPUsec\n";
   
   return 0;
   }

