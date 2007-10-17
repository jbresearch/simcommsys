#include "channel.h"
#include "serializer.h"

namespace libcomm {

const libbase::vcs channel::version("Channel Base module (channel)", 1.52);

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
   // TODO: Eb should get into this equation!!!
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   // call derived class handle
   compute_parameters(Eb,No);
   }
   
void channel::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration, obtained from modulator
   channel::Eb = Eb;
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
