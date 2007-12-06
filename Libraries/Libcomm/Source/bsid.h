#ifndef __bsid_h
#define __bsid_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
   \brief   Binary substitution/insertion/deletion channel.
   \author  Johann Briffa

   \version 1.00 (12-16 Oct 2007)
   - Initial version; implementation of a binary substitution, insertion, and deletion channel.
   - \b Note: this class is still unfinished, and only implements the BSC channel right now

   \version 1.01 (17 Oct 2007)
   - changed class to conform with channel 1.52.

   \version 1.10 (18 Oct 2007)
   - added transmit() and receive() functions to actually handle insertions and deletions
   - kept corrupt() and pdf() to be used internally for dealing with substitution errors
   - added specification of Pd and Pi during creation, defaulting to zero (effectively gives a BSC)
   - added serialization of Pd and Pi

   \version 1.20 (23 Oct 2007)
   - added functions to set channel parameters directly (as Ps, Pd, Pi)
   - implemented the receive() function to return Q_m(s) as defined by Davey; one will need to
     first update Ps, depending on whether the receive() is operating wrt the actual channel (ie the
     actual substitution error) or wrt the sparse vector (ie the vector average density).

   \version 1.21 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

   \version 1.22 (1 Nov 2007)
   - implemented receive() for a sequence of transmitted symbols
      - uses the forward-backward algorithm by including a derived fba object.
      - to simplify matters, double-precision math is used in fba.
      - this required the creation of a specific implementation, including P() and Q().
      - added I and xmax parameters to this class (variables and constructor).
   - added protected default constructor for use by create()
   - changed constructor to use internal functions for setting parameters.
   - added getters for channel parameters
   - updated serialization to include all parameters
   - fixed include-once definition

   \version 1.22 (2 Nov 2007)
   - removed Pd and Pi from serialization and from construction
   - now setting defaults for Ps,Pd,Pi to zero in all constructors, through a new
     function init()
   - added boolean construction parameters varyPs, varyPd, varyPi, to indicate
     what should be changed when the SNR is updated; all default to true; these
     are held by protected variables, so that they can be accessed by derived classes.

   \version 1.23 (5-6 Nov 2007)
   - updated transmit() to cater for the usual case where the tx and rx vectors
     are actually the same.
   - fixed error in receive(), when tau=1, where the special case of m=-1 was
     not handled.
   - changed varyPx variables from protected to private; these are only changed
     on initialization or serialization - derived classes should delegate to this
     class's serialization routines as needed.
   - added getters for I and xmax (watermarkcode needs them to set up fba)
     @b TODO: this should probably change, separating or integrating bsid & fba
   - fixed ptable and getF indexing errors in receive() for M=1.

   \version 1.24 (14 Nov 2007)
   - optimized receive() for the case when tau=1

   \version 1.30 (15 Nov 2007)
   - implemented refactoring changes in channel 1.60
   - inlined the single-symbol receive() and pdf()
   - inlined myfba::P and Q
   - reduced pdf() using ternary operator
   - added pre-computed parameters, to reduce work in single-symbol receive();
     also updated single-timestep receive() accordingly.
   - updated compute_parameters() to use set_ps/i/d instead of direct-access
   - added pre-computed parameter to reduce work in myfba::P()

   \version 1.31 (28 Nov 2007)
   - moved call to init() from default constructor to end of serialization input

   \version 1.40 (6 Dec 2007)
   - removed I and xmax from user-defined parameters, instead determining the value
     from the current channel parameters; this allows much smaller values (and
     therefore faster simulations) at low error rates.
   - added N as a user-defined parameter, since this is required to determine
     I and xmax
*/

class bsid : public channel {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new bsid; };
   // user-defined parameters
   double   Ps, Pd, Pi; // channel parameters
   int      N;          // fba decoder parameter
   bool     varyPs, varyPd, varyPi; // channel update flags
   // pre-computed parameters
   int      I, xmax;    // fba decoder parameters
   double   a1, a2;     // receiver coefficients
   libbase::vector<double> a3;   // receiver coefficients
   // internal functions
   void init();
   void precompute();
protected:
   // default constructor
   bsid() {};
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   bsid(const int N, const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
   bsid *clone() const { return new bsid(*this); };
   const char* name() const { return shelper.name(); };

   // channel parameter updates
   void set_ps(const double Ps);
   void set_pd(const double Pd);
   void set_pi(const double Pi);
   // channel parameters getters
   double get_ps() const { return Ps; };
   double get_pd() const { return Pd; };
   double get_pi() const { return Pi; };
   // fba decoder parameters
   int get_I() const { return I; };
   int get_xmax() const { return xmax; };

   // channel functions
   void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);
   void receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const;
   double receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const;
   double receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const;

   // description output
   std::string description() const;
   // object serialization
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

inline double bsid::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   return (tx != rx) ? Ps : 1-Ps;
   }

inline double bsid::receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const
   {
   // Compute sizes
   const int m = rx.size()-1;
   // set of possible transmitted symbols for one transmission step
   if(m == -1) // just a deletion, no symbols received
      return Pd;
   // Work out the probabilities of each possible signal
   return (a1 * pdf(tx,rx(m)) + a2) * a3(m);
   }

}; // end namespace

#endif

