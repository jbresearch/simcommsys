#include "bsid.h"

namespace libcomm {

const libbase::vcs bsid::version("Binary Substitution, Insertion, and Deletion Channel module (bsid)", 1.00);

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);


// constructors / destructors

bsid::bsid()
   {
   bsid::Eb = 1;
   bsid::set_snr(0);
   bsid::seed(0);
   }

// internal helper functions

void bsid::compute_parameters()
   {
   //sigma = sqrt(Eb*No);
   Ps = 0.5*erfc(1/sqrt(Eb*No*2));
   }
   
// channel functions
   
void bsid::seed(const libbase::int32u s)
   {
   r.seed(s);
   }

void bsid::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration
   bsid::Eb = Eb;
   compute_parameters();
   }

void bsid::set_snr(const double snr_db)
   {
   bsid::snr_db = snr_db;
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   compute_parameters();
   }
   
sigspace bsid::corrupt(const sigspace& s)
   {
   const double p = r.fval();
   if(p < Ps)
      return -s;
   return s;
   }

double bsid::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   if(tx != rx)
      return Ps;
   return 1-Ps;
   }

// description output

std::string bsid::description() const
   {
   return "BSID channel";
   }

// object serialization - saving

std::ostream& bsid::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& bsid::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
