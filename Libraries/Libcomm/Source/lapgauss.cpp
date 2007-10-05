#include "lapgauss.h"

namespace libcomm {

const libbase::vcs lapgauss::version("Additive Laplacian-Gaussian Channel module (lapgauss)", 1.10);

const libbase::serializer lapgauss::shelper("channel", "lapgauss", lapgauss::create);


// constructors / destructors

lapgauss::lapgauss()
   {
   lapgauss::Eb = 1;
   lapgauss::set_snr(0);
   lapgauss::seed(0);
   }

// channel functions
   
void lapgauss::seed(const libbase::int32u s)
   {
   r.seed(s);
   }

void lapgauss::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration
   lapgauss::Eb = Eb;
   sigma = sqrt(Eb*No);
   }

void lapgauss::set_snr(const double snr_db)
   {
   lapgauss::snr_db = snr_db;
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   sigma = sqrt(Eb*No);
   }
   
sigspace lapgauss::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

double lapgauss::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   using libbase::gauss;
   sigspace n = rx - tx;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

// description output

std::string lapgauss::description() const
   {
   return "AWGN channel";
   }

// object serialization - saving

std::ostream& lapgauss::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& lapgauss::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
