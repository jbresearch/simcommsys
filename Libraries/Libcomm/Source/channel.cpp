/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "channel.h"

namespace libcomm {

// constructors / destructors

channel<sigspace>::channel()
   {
   channel::Eb = 1;
   channel::set_parameter(0);
   channel::seed(0);
   }

// setting and getting overall channel SNR

void channel<sigspace>::compute_noise()
   {
   No = 0.5*pow(10.0, -snr_db/10.0);
   // call derived class handle
   compute_parameters(Eb,No);
   }

void channel<sigspace>::set_eb(const double Eb)
   {
   channel::Eb = Eb;
   compute_noise();
   }

void channel<sigspace>::set_no(const double No)
   {
   snr_db = -10.0*log10(2*No);
   compute_noise();
   }

void channel<sigspace>::set_parameter(const double snr_db)
   {
   channel::snr_db = snr_db;
   compute_noise();
   }

// channel functions

void channel<sigspace>::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
   {
   // Initialize results vector
   rx.init(tx);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<tx.size(); i++)
      rx(i) = corrupt(tx(i));
   }

void channel<sigspace>::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int tau = rx.size();
   const int M = tx.size();
   // Initialize results vector
   ptable.init(tau, M);
   // Work out the probabilities of each possible signal
   for(int t=0; t<tau; t++)
      for(int x=0; x<M; x++)
         ptable(t,x) = pdf(tx(x), rx(t));
   }

double channel<sigspace>::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const
   {
   // Compute sizes
   const int tau = rx.size();
   // This implementation only works for substitution channels
   assert(tx.size() == tau);
   // Work out the combined probability of the sequence
   double p = 1;
   for(int t=0; t<tau; t++)
      p *= pdf(tx(t), rx(t));
   return p;
   }

double channel<sigspace>::receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const
   {
   // This implementation only works for substitution channels
   assert(rx.size() == 1);
   // Work out the probability of receiving the particular symbol
   return pdf(tx, rx(0));
   }

}; // end namespace
