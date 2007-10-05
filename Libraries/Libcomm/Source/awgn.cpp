#include "awgn.h"

namespace libcomm {

const libbase::vcs awgn::version("Additive White Gaussian Noise Channel module (awgn)", 1.40);

const libbase::serializer awgn::shelper("channel", "awgn", awgn::create);


// constructors / destructors

awgn::awgn()
   {
   awgn::Eb = 1;
   awgn::set_snr(0);
   awgn::seed(0);
   }

// channel functions
   
void awgn::seed(const libbase::int32u s)
   {
   r.seed(s);
   }

void awgn::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration
   awgn::Eb = Eb;
   sigma = sqrt(Eb*No);
   }

void awgn::set_snr(const double snr_db)
   {
   awgn::snr_db = snr_db;
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   sigma = sqrt(Eb*No);
   }
   
sigspace awgn::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

double awgn::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   sigspace n = rx - tx;
   using libbase::gauss;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

// description output

std::string awgn::description() const
   {
   return "AWGN channel";
   }

// object serialization - saving

std::ostream& awgn::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& awgn::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
