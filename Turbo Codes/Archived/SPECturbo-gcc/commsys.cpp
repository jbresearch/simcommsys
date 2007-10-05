#include "commsys.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream.h>

const vcs commsys_version("Communication System module (commsys)", 1.30);

commsys::commsys(randgen *src, channel *chan, codec *cdc, bool fast)
   {
   // bind sub-components
   commsys::src = src;
   commsys::chan = chan;
   commsys::cdc = cdc;
   // get other internal values
   commsys::fast = fast;
   tau = cdc->block_size();
   n = cdc->num_symbols();
   m = cdc->tail_length();
   K = cdc->num_inputs();
   k = int(floor(log(K)/log(2) + 0.5));
   if(K != 1<<k)
      {
      cerr << "FATAL ERROR (commsys): can only estimate BER for a q-ary source (" << k << ", " << K << ").\n";
      exit(1);
      }
   iter = cdc->num_iter();
   // initialise data heap
   source.init(tau);
   encoded.init(tau);
   decoded.init(tau);
   last.init(tau);
   received.init(n);
   }

commsys::~commsys()
   {
   }
   
void commsys::seed(int s)
   {
   src->seed(s);
   chan->seed(s);
   cdc->seed(s);
   }

void commsys::set(double x)
   {
   chan->set_snr(x);
   }

double commsys::get()
   {
   return chan->get_snr();
   }

void commsys::cycleonce(vector<double>& result)
   {
   // Create source stream
   for(int t=0; t<tau-m; t++)
      source(t) = src->ival(K);
   for(int t=tau-m; t<tau; t++)
      source(t) = fsm::tail;
         
   // Encode->Transmit->Demodulate
   cdc->encode(source, encoded);
   cdc->modulate(encoded, received);
   cdc->transmit(received, received);
   cdc->demodulate(received);

   // For every iteration possible
   int delta = 0;
   bool skipping = false;
   for(int i=0; i<iter; i++)
      {
      if(!skipping)
         {
         // Decode
         cdc->decode(decoded);
         // Check if this decoded stream is different from the last one
         if(fast && i>0)
            {
            skipping = true;
            for(int t=0; t<tau-m; t++)
               if(last(t) ^ decoded(t))
                  {
                  skipping = false;
                  break;
                  }
            }
         // Count the number of errors if necessary
         if(!skipping)
            {
            delta = 0;
            for(int t=0; t<tau-m; t++)
               {
               delta += weight(source(t) ^ decoded(t));
               last(t) = decoded(t);
               }
            }
         }
      
      // Estimate the BER
      result(2*i + 0) += delta / double((tau-m)*k);

      // Estimate the FER (Frame Error Rate)
      result(2*i + 1) += delta ? 1 : 0;
      }
   }
 
void commsys::sample(vector<double>& result, int& samplecount)
   {
   // initialise result vector
   result.init(count());
   for(int i=0; i<count(); i++)
      result(i) = 0;

   // iterate until a certain time has elapsed
   // granularity is fixed to 500ms, which is a good compromise between efficiency and usability
   int passes=0, length=0;
   while(length < 1000)
      {
      cycleonce(result);   // will update result
      length += n;     // length is in modulation symbols
      passes++;
      samplecount++;
      }

   // update result
   for(int i=0; i<count(); i++)
      result(i) /= double(passes);
   }
