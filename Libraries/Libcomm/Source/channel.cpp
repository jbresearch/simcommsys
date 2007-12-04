#include "channel.h"
#include "serializer.h"

namespace libcomm {

const libbase::vcs channel::version("Channel Base module (channel)", 1.60);

// constructors / destructors

channel::channel()
   {
   channel::Eb = 1;
   channel::set_snr(0);
   channel::seed(0);
   }

// reset function for random generator
   
void channel::seed(const libbase::int32u s)
   {
   r.seed(s);
   }

// setting and getting overall channel SNR

void channel::compute_noise()
   {
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*pow(10.0, -snr_db/10.0);
   // call derived class handle
   compute_parameters(Eb,No);
   }
   
void channel::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration, obtained from modulator
   channel::Eb = Eb;
   compute_noise();
   }

void channel::set_no(const double No)
   {
   // No is half the noise energy/modulation symbol for a normalised signal
   snr_db = -10.0*log10(2*No);
   compute_noise();
   }

void channel::set_snr(const double snr_db)
   {
   // snr_db is equal to 10 log_10 (Eb/No), obtained from user
   channel::snr_db = snr_db;
   compute_noise();
   }

// channel functions

void channel::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
   {
   // Initialize results vector
   rx.init(tx);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<tx.size(); i++)
      rx(i) = corrupt(tx(i));
   }

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

// object serialization

std::ostream& channel::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& channel::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
