/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bsid.h"
#include "secant.h"
#include <sstream>

namespace libcomm {

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);

/// FBA decoder parameter computation

/*!
   \brief Determine limit for insertions between two time-steps

   \f[ I = \left\lceil \frac{ \log{P_r} - \log N }{ \log p } \right\rceil - 1 \f]
   where \f$ P_r \f$ is an arbitrary probability of having a block of size \f$ N \f$
   with at least one event of more than \f$ I \f$ insertions between successive
   time-steps. In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.

   \note The smallest allowed value is \f$ I = 1 \f$
*/
int bsid::compute_I(int N, double p)
   {
   int I = max(int(ceil((log(1e-12) - log(double(N))) / log(p))) - 1, 1);
   libbase::trace << "DEBUG (bsid): suggested I = " << I << ".\n";
   I = min(I,2);
   libbase::trace << "DEBUG (bsid): using I = " << I << ".\n";
   return I;
   }

/*!
   \brief Determine maximum drift over a whole N-bit block

   \f[ x_{max} = 8 \sqrt{\frac{N p}{1-p}} \f]
   where \f$ p = P_i = P_d \f$. This is based directly on Davey's suggestion that
   \f$ x_{max} \f$ should be "several times larger" than the standard deviation of
   the synchronization drift over one block, given by \f$ \sigma = \sqrt{\frac{N p}{1-p}} \f$

   \note The smallest allowed value is \f$ x_{max} = I \f$

   \note While Davey advocates \f$ x_{max} = 5 \sigma \f$, we increase this to
         \f$ 8 \sigma \f$, as we observed that Davey's estimates are off.
*/
int bsid::compute_xmax(int N, double p, int I)
   {
   int xmax = max(int(ceil(8 * sqrt(N*p/(1-p)))), I);
   libbase::trace << "DEBUG (bsid): suggested xmax = " << xmax << ".\n";
   //xmax = min(xmax,25);
   libbase::trace << "DEBUG (bsid): using xmax = " << xmax << ".\n";
   return xmax;
   }

/*!
   \brief Compute receiver coefficient set

   First row has elements where the last bit rx(m) == tx
   \f[ Rtable(0,m) = \frac{(1-P_i-P_d) * (1-Ps) + \frac{1}{2} P_i P_d}
                          {2^m (1-P_i) (1-P_d)}, m \in (0, \ldots x_{max}) \f]
   Second row has elements where the last bit rx(m) != tx
   \f[ Rtable(1,m) = \frac{(1-P_i-P_d) * Ps + \frac{1}{2} P_i P_d}
                          {2^m (1-P_i) (1-P_d)}, m \in (0, \ldots x_{max}) \f]
*/
void bsid::compute_Rtable(libbase::matrix<double>& Rtable, int xmax, double Ps, double Pd, double Pi)
   {
   Rtable.init(2,xmax+1);
   const double a1 = (1-Pi-Pd);
   const double a2 = 0.5*Pi*Pd;
   for(int m=0; m<=xmax; m++)
      {
      const double a3 = (1<<m)*(1-Pi)*(1-Pd);
      Rtable(0,m) = (a1 * (1-Ps) + a2) / a3;
      Rtable(1,m) = (a1 * Ps + a2) / a3;
      }
   }

/*!
   \brief Compute forward recursion 'P' function
*/
void bsid::compute_Ptable(libbase::vector<double>& Ptable, int xmax, double Pd, double Pi)
   {
   Ptable.init(xmax+2);
   Ptable(0) = Pd;   // for m = -1
   for(int m=0; m<=xmax; m++)
      Ptable(m+1) = pow(Pi,m)*(1-Pi)*(1-Pd);
   }

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
   assertalways(Pi == Pd);
   I = compute_I(N, Pd);
   xmax = compute_xmax(N, Pd, I);
   // receiver coefficients
   compute_Rtable(Rtable, xmax, Ps, Pd, Pi);
   // pre-compute 'P' table
   compute_Ptable(Ptable, xmax, Pd, Pi);
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

/*!
   \brief Set channel parameter

   This function sets any of Ps, Pd, or Pi that are flagged to change. Any of these
   parameters that are not flagged to change will instead be set to zero. This ensures
   that there is no leakage between successive uses of this class. (i.e. once this
   function is called, the class will be in a known determined state).
*/
void bsid::set_parameter(const double p)
   {
   set_ps(varyPs ? p : 0);
   set_pd(varyPd ? p : 0);
   set_pi(varyPi ? p : 0);
   libbase::trace << "DEBUG (bsid): Ps = " << Ps << ", Pd = " << Pd << ", Pi = " << Pi << "\n";
   }

/*!
   \brief Get channel parameter

   This returns the value of the first of Ps, Pd, or Pi that are flagged to change.
   If none of these are flagged to change, this constitutes an error condition.
*/
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
   if(tau < 100)
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
         newrx(j++) = (r.fval() < 0.5);
      if(transmit(i))
         newrx(j++) = corrupt(tx(i));
      }
   // copy results back
   rx = newrx;
   }

void bsid::receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int M = tx.size();
   // Initialize results vector
   ptable.init(1, M);
   // Compute results for each possible signal
   for(int x=0; x<M; x++)
      ptable(0,x) = receive(tx(x),rx);
   }

double bsid::receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx) const
   {
   // Compute sizes
   const int tau = tx.size();
   const int m = rx.size()-tau;
   // Set up forward matrix
   libbase::matrix<double> Ftable;
   Ftable.init(tau+1, 2*xmax+1);
   // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   Ftable = 0;
   Ftable(0,0+xmax) = 1;
   // compute remaining matrix values
   for(int j=1; j<tau; j++)
      {
      // determine the strongest path at this point
      double threshold = 0;
      for(int a=-xmax; a<=xmax; a++)
         if(Ftable(j-1,a+xmax) > threshold)
            threshold = Ftable(j-1,a+xmax);
      threshold *= 1e-15;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < rx.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      const int amin = max(-xmax,1-j);
      const int amax = xmax;
      for(int a=amin; a<=amax; a++)
         {
         // ignore paths below a certain threshold
         if(Ftable(j-1,a+xmax) < threshold)
            continue;
         const int ymin = max(-xmax,a-1);
         const int ymax = min(min(xmax,a+I),rx.size()-j);
         for(int y=ymin; y<=ymax; y++)
            Ftable(j,y+xmax) += Ftable(j-1,a+xmax) * Ptable(y-a+1) * receive(tx(j-1),rx.extract(j-1+a,y-a+1));
         }
      }
   // Compute forward metric for known drift, and return
   // determine the strongest path at this point
   double threshold = 0;
   for(int a=-xmax; a<=xmax; a++)
      if(Ftable(tau-1,a+xmax) > threshold)
         threshold = Ftable(tau-1,a+xmax);
   threshold *= 1e-15;
   // limits on insertions and deletions must be respected:
   // 3. m-a <= I
   // 4. m-a >= -1
   const int amin = max(-xmax,max(1-tau,m-I));
   const int amax = min(xmax,m+1);
   for(int a=amin; a<=amax; a++)
      {
      // ignore paths below a certain threshold
      if(Ftable(tau-1,a+xmax) < threshold)
         continue;
      Ftable(tau,m+xmax) += Ftable(tau-1,a+xmax) * Ptable(m-a+1) * receive(tx(tau-1),rx.extract(tau-1+a,m-a+1));
      }
   return Ftable(tau,m+xmax);
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
