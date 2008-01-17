/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "channel.h"
#include "serializer.h"

namespace libcomm {

// constructors / destructors

channel::channel()
   {
   channel::Eb = 1;
   channel::set_parameter(0);
   channel::seed(0);
   }

// setting and getting overall channel SNR

void channel::compute_noise()
   {
   No = 0.5*pow(10.0, -snr_db/10.0);
   // call derived class handle
   compute_parameters(Eb,No);
   }

void channel::set_eb(const double Eb)
   {
   channel::Eb = Eb;
   compute_noise();
   }

void channel::set_no(const double No)
   {
   snr_db = -10.0*log10(2*No);
   compute_noise();
   }

void channel::set_parameter(const double snr_db)
   {
   channel::snr_db = snr_db;
   compute_noise();
   }

// channel functions

/*!
   \brief Pass a sequence of modulation symbols through the channel
   \param[in]  tx  Transmitted sequence of modulation symbols
   \param[out] rx  Received sequence of modulation symbols

   \note It is possible that the \c tx and \c rx parameters actually point to the same
         vector.

   \callergraph
*/
void channel::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
   {
   // Initialize results vector
   rx.init(tx);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<tx.size(); i++)
      rx(i) = corrupt(tx(i));
   }

/*!
   \brief Determine the per-symbol likelihoods of a sequence of received modulation symbols
          corresponding to one transmission step
   \param[in]  tx       Set of possible transmitted symbols
   \param[in]  rx       Received sequence of modulation symbols
   \param[out] ptable   Likelihoods corresponding to each possible transmitted symbol

   \callergraph
*/
void channel::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
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

/*!
   \brief Determine the likelihood of a sequence of received modulation symbols, given
          a particular transmitted sequence
   \param[in]  tx       Transmitted sequence being considered
   \param[in]  rx       Received sequence of modulation symbols
   \return              Likelihood \f$ P(rx|tx) \f$

   \callergraph
*/
double channel::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const
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

/*!
   \brief Determine the likelihood of a sequence of received modulation symbols, given
          a particular transmitted symbol
   \param[in]  tx       Transmitted symbol being considered
   \param[in]  rx       Received sequence of modulation symbols
   \return              Likelihood \f$ P(rx|tx) \f$

   \callergraph
*/
double channel::receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const
   {
   // This implementation only works for substitution channels
   assert(rx.size() == 1);
   // Work out the probability of receiving the particular symbol
   return pdf(tx, rx(0));
   }

// serialization functions

std::ostream& operator<<(std::ostream& sout, const channel* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, channel*& x)
   {
   std::string name;
   sin >> name;
   x = (channel*) libbase::serializer::call("channel", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (channel): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
