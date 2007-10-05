#include "laplacian.h"

namespace libcomm {

const libbase::vcs laplacian::version("Additive Laplacian Noise Channel module (laplacian)", 1.30);

const libbase::serializer laplacian::shelper("channel", "laplacian", laplacian::create);


// constructors / destructors

laplacian::laplacian()
   {
   laplacian::Eb = 1;
   laplacian::set_snr(0);
   laplacian::seed(0);
   }
   
// channel functions
   
void laplacian::seed(const libbase::int32u s)
   {
   r.seed(s);
   }

void laplacian::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration
   laplacian::Eb = Eb;
   const double sigma = sqrt(Eb*No);
   lambda = sigma/sqrt(double(2));
   }

void laplacian::set_snr(const double snr_db)
   {
   laplacian::snr_db = snr_db;
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   const double sigma = sqrt(Eb*No);
   lambda = sigma/sqrt(double(2));
   }
   
sigspace laplacian::corrupt(const sigspace& s)
   {
   const double x = Finv(r.fval());
   const double y = Finv(r.fval());
   return s + sigspace(x, y);
   }

double laplacian::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   sigspace n = rx - tx;
   return f(n.i()) * f(n.q());
   }

// description output

std::string laplacian::description() const
   {
   return "Laplacian channel";
   }

// object serialization - saving

std::ostream& laplacian::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& laplacian::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
