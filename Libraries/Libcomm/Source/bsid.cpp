/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bsid.h"
#include "fba.h"
#include "secant.h"
#include <sstream>

namespace libcomm {

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);

// Internal functions

/*!
   \brief Initialization

   Sets the channel with \f$ P_s = P_d = P_i = 0 \f$. This way, any
   of the parameters not flagged to change with channel SNR will remain zero.
*/
void bsid::init()
   {
   // channel parameters
   Ps = 0;
   Pd = 0;
   Pi = 0;
   precompute();
   }

/*!
   \brief Sets up pre-computed values

   This function computes all cached quantities used within actual channel operations.
   Since these values depend on the channel conditions, this function should be called
   any time a channel parameter is changed.
*/
void bsid::precompute()
   {
   // fba decoder parameters
   I = max(int(ceil((log(1e-12) - log(double(N))) / log(Pd))) - 1, 1);
   xmax = max(int(ceil(5 * sqrt(N*Pd*(1-Pd)))), I);
   libbase::trace << "DEBUG (bsid): suggested I = " << I << ", xmax = " << xmax << ".\n";
   I = min(I,2);
   //xmax = min(xmax,25);
   libbase::trace << "DEBUG (bsid): using I = " << I << ", xmax = " << xmax << ".\n";
   // receiver coefficients
   a1 = (1-Pi-Pd);
   a2 = 0.5*Pi*Pd;
   a3.init(xmax+1);
   for(int m=0; m<=xmax; m++)
      a3(m) = 1.0 / ( (1<<m)*(1-Pi)*(1-Pd) );
   }

// Constructors / Destructors

/*!
   \brief Principal constructor

   \sa init()
*/
bsid::bsid(const int N, const bool varyPs, const bool varyPd, const bool varyPi)
   {
   // fba decoder parameter
   assert(N > 0);
   bsid::N = N;
   // channel update flags
   assert(varyPs || varyPd || varyPi);
   bsid::varyPs = varyPs;
   bsid::varyPd = varyPd;
   bsid::varyPi = varyPi;
   // other initialization
   init();
   }

// Channel parameter handling

void bsid::set_parameter(const double p)
   {
   if(varyPs)
      set_ps(p);
   if(varyPd)
      set_pd(p);
   if(varyPi)
      set_pi(p);
   libbase::trace << "DEBUG (bsid): Ps = " << Ps << ", Pd = " << Pd << ", Pi = " << Pi << "\n";
   }

double bsid::get_parameter() const
   {
   if(varyPs)
      return Ps;
   if(varyPd)
      return Pd;
   if(varyPi)
      return Pi;
   std::cerr << "ERROR: BSID channel has no parameters\n";
   exit(1);
   }

// Channel parameter setters

void bsid::set_ps(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsid::Ps = Ps;
   }

void bsid::set_pd(const double Pd)
   {
   assert(Pd >=0 && Pd <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid::Pd = Pd;
   precompute();
   }

void bsid::set_pi(const double Pi)
   {
   assert(Pi >=0 && Pi <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid::Pi = Pi;
   precompute();
   }

// Channel function overrides

/*!
   \copydoc channel::corrupt()

   \note Due to limitations of the interface, which was designed for substitution channels,
         only the substitution part of the channel model is handled here.

   For the purposes of this channel, a \e substitution corresponds to a symbol inversion.
   This corresponds to the \f$ 0 \Leftrightarrow 1 \f$ binary substitution when used with BPSK
   modulation. For MPSK modulation, this causes the output to be the symbol farthest away
   from the input.
*/
bool bsid::corrupt(const bool& s)
   {
   const double p = r.fval();
   if(p < Ps)
      return !s;
   return s;
   }

// Channel functions

/*!
   \copydoc channel::transmit()

   The channel model implemented is described by the following state diagram:
   \dot
   digraph bsidstates {
      // Make figure left-to-right
      rankdir = LR;
      // state definitions
      this [ shape=circle, color=gray, style=filled, label="t(i)" ];
      next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
      // path definitions
      this -> Insert [ label="Pi" ];
      Insert -> this;
      this -> Delete [ label="Pd" ];
      Delete -> next;
      this -> Transmit [ label="1-Pi-Pd" ];
      Transmit -> next [ label="1-Ps" ];
      Transmit -> Substitute [ label="Ps" ];
      Substitute -> next;
   }
   \enddot

   \note We have initially no idea how long the received sequence will be, so we first determine
         the state sequence at every timestep keeping track of
            - the number of insertions \e before given position, and
            - whether the given position is transmitted or deleted.

   \note We have to make sure that we don't corrupt the vector we're reading from (in
         the case where tx and rx are the same vector); therefore, the result is first created
         as a new vector and only copied over at the end.

   \sa corrupt()
*/
void bsid::transmit(const libbase::vector<bool>& tx, libbase::vector<bool>& rx)
   {
   const int tau = tx.size();
   libbase::vector<int> insertions(tau);
   insertions = 0;
   libbase::vector<int> transmit(tau);
   transmit = 1;
   // determine state sequence
   for(int i=0; i<tau; i++)
      {
      double p;
      while((p = r.fval()) < Pi)
         insertions(i)++;
      if(p < (Pi+Pd))
         transmit(i) = 0;
      }
   // Initialize results vector
#ifndef NDEBUG
   if(tau < 10)
      {
      libbase::trace << "DEBUG (bsid): transmit = " << transmit << "\n";
      libbase::trace << "DEBUG (bsid): insertions = " << insertions << "\n";
      }
#endif
   libbase::vector<bool> newrx;
   newrx.init(transmit.sum() + insertions.sum());
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0, j=0; i<tau; i++)
      {
      while(insertions(i)--)
         newrx(j++) = !(r.fval() < 0.5);
      if(transmit(i))
         newrx(j++) = corrupt(tx(i));
      }
   // copy results back
   rx = newrx;
   }

/********************************* FBA sub-class object *********************************/

class myfba : public fba<double,bool> {
   // user-defined parameters
   libbase::vector<bool>  tx;   // presumed transmitted sequence
   const bsid* channel;
   // pre-computed parameters
   libbase::vector<double> Ptable;
   // implementations of channel-specific metrics for fba
   double P(const int a, const int b);
   double Q(const int a, const int b, const int i, const libbase::vector<bool>& s);
public:
   // constructor & destructor
   myfba() {};
   ~myfba() {};
   // set transmitted sequence
   void settx(const libbase::vector<bool>& tx);
   // attach channel
   void attach(const bsid* channel);
};

// implementations of channel-specific metrics for fba

inline double myfba::P(const int a, const int b)
   {
   const int m = b-a;
   return Ptable(m+1);
   }

inline double myfba::Q(const int a, const int b, const int i, const libbase::vector<bool>& s)
   {
   // 'a' and 'b' are redundant because 's' already contains the difference
   assert(s.size() == b-a+1);
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   // compute the conditional probability
   return channel->receive(myfba::tx(i), s);
   }

// set transmitted sequence

inline void myfba::settx(const libbase::vector<bool>& tx)
   {
   myfba::tx.init(tx.size()+1);
   myfba::tx.copyfrom(tx);
   }

// attach channel

inline void myfba::attach(const bsid* channel)
   {
   myfba::channel = channel;
   // pre-compute table
   const double Pd = channel->get_pd();
   const double Pi = channel->get_pi();
   const int xmax = get_xmax();
   Ptable.init(xmax+2);
   Ptable(0) = Pd;   // for m = -1
   for(int m=0; m<=xmax; m++)
      Ptable(m+1) = pow(Pi,m)*(1-Pi)*(1-Pd);
   };

/********************************* END FBA *********************************/

void bsid::receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int M = tx.size();
   const int m = rx.size()-1;
   // Initialize results vector
   ptable.init(1, M);
   // Compute results
   if(m == -1) // just a deletion, no symbols received
      ptable = Pd;
   else
      {
      // Work out the probabilities of each possible signal
      for(int x=0; x<M; x++)
         ptable(0,x) = (a1 * pdf(tx(x),rx(m)) + a2) * a3(m);
      }
   }

double bsid::receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx) const
   {
   // Compute sizes
   const int tau = tx.size();
   const int m = rx.size()-tau;
   // One possible sequence of transmitted symbols
   myfba f;
   f.init(tau+1, I, xmax);
   f.attach(this);
   f.settx(tx);
   f.work_forward(rx);
   return f.getF(tau,m);
   }

// description output

std::string bsid::description() const
   {
   std::ostringstream sout;
   sout << "BSID channel (" << N << "," << varyPs << varyPd << varyPi << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& bsid::serialize(std::ostream& sout) const
   {
   sout << N << "\n";
   sout << varyPs << "\n";
   sout << varyPd << "\n";
   sout << varyPi << "\n";
   return sout;
   }

// object serialization - loading

std::istream& bsid::serialize(std::istream& sin)
   {
   sin >> N;
   sin >> varyPs;
   sin >> varyPd;
   sin >> varyPi;
   init();
   return sin;
   }

}; // end namespace
