#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

/*
  Version 1.00 (12-16 Oct 2007)
  Initial version; implementation of a binary substitution, insertion, and deletion channel.
  * TODO: this class is still unfinished, and only implements the BSC channel right now

  Version 1.01 (17 Oct 2007)
  changed class to conform with channel 1.52.
  
  Version 1.10 (18 Oct 2007)
  * added transmit() and receive() functions to actually handle insertions and deletions
  * kept corrupt() and pdf() to be used internally for dealing with substitution errors
  * added specification of Pd and Pi during creation, defaulting to zero (effectively gives a BSC)
  * added serialization of Pd and Pi
  
  Version 1.20 (23 Oct 2007)
  * added functions to set channel parameters directly (as Ps, Pd, Pi)
  * implemented the receive() function to return Q_m(s) as defined by Davey; one will need to
    first update Ps, depending on whether the receive() is operating wrt the actual channel (ie the
    actual substitution error) or wrt the sparse vector (ie the vector average density).

  Version 1.21 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

class bsid : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new bsid; };
   // channel paremeters
   double   Ps, Pd, Pi;       // specific parameters
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   bsid(const double Pd=0, const double Pi=0);
   bsid *clone() const { return new bsid(*this); };
   const char* name() const { return shelper.name(); };

   // channel parameter updates
   void set_ps(const double Ps);
   void set_pd(const double Pd);
   void set_pi(const double Pi);
   
   // channel functions
   void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);
   void receive(const libbase::matrix<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const;

   // description output
   std::string description() const;
   // object serialization
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

