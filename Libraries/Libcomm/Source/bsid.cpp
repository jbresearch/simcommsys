#include "bsid.h"
#include "fba.h"
#include "secant.h"
#include <sstream>

namespace libcomm {

const libbase::vcs bsid::version("Binary Substitution, Insertion, and Deletion Channel module (bsid)", 1.40);

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
   \param   N        Block size in bits over which we need to synchronize; typically this is
                     the size of the outer codeword.
   \param   varyPs   Flag to indicate that \f$ P_s \f$ should change with SNR
   \param   varyPd   Flag to indicate that \f$ P_d \f$ should change with SNR
   \param   varyPi   Flag to indicate that \f$ P_i \f$ should change with SNR

   \sa init()
*/
bsid::bsid(const int N, const bool varyPs, const bool varyPd, const bool varyPi)
   {
   // fba decoder parameter
   assert(N > 0);
   bsid::N = N;
   // channel update flags
   bsid::varyPs = varyPs;
   bsid::varyPd = varyPd;
   bsid::varyPi = varyPi;
   // other initialization
   init();
   }

// Channel parameter setters

void bsid::set_ps(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsid::Ps = Ps;
   //libbase::secant Qinv(libbase::Q);
   //const double x = Qinv(Ps);
   //const double No = 1/(get_eb()*x*x);
   //set_no(No);
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
   \brief Determine channel-specific parameters based on given SNR
   \param   Eb    Average signal energy per information bit \f$ E_b \f$. Depends on modulation
                  symbol energy, modulation rate, and overall coding rate.
   \param   No    Half the noise energy/modulation symbol for a normalised signal \f$ N_0 \f$.

   \note \f$ E_b \f$ is fixed by the overall modulation and coding system. The simulator
         determines \f$ N_0 \f$ according to the given SNR (assuming unit signal energy), so
         that the actual band-limited noise energy is given by \f$ E_b N_0 \f$.

   There is no real relationship between SNR and the insertion/deletion probabilities.
   However, the simulator uses SNR as its common channel-quality measure, so that a functional
   relationship has to be chosen.

   For the purposes of this channel, SNR and the error probabilities are related by:
      \f[ p = Q(1/\sigma) \f]
   where \f$ \sigma^2 = E_b N_0 \f$ would be the variance if the channel represented additive
   Gaussian noise. The probabilities \f$ P_s, P_d, P_i \f$ are set to \f$ p \f$ if the corresponding
   flag is set, or left at zero otherwise.

   \note Effectively, if only \f$ P_s \f$ is set to be varied, this channel becomes equivalent to
   a BSC model for a hard-decision AWGN channel.
*/
void bsid::compute_parameters(const double Eb, const double No)
   {
   // computes substitution probability assuming Eb/No describes an AWGN channel with hard-decision demodulation
   const double p = libbase::Q(1/sqrt(Eb*No));
   if(varyPs)
      set_ps(p);
   if(varyPd)
      set_pd(p);
   if(varyPi)
      set_pi(p);
   libbase::trace << "DEBUG (bsid): Eb = " << Eb << ", No = " << No << " -> Ps = " << Ps << ", Pd = " << Pd << ", Pi = " << Pi << "\n";
   }
   
/*!
   \brief Pass a single modulation symbol through the substitution channel
   \param   s  Input (Tx) modulation symbol
   \return  Output (Rx) modulation symbol

   \note Due to limitations of the interface, which was designed for substitution channels,
         only the substitution part of the channel model is handled here.

   For the purposes of this channel, a \e substitution corresponds to a symbol inversion.
   This corresponds to the \f$ 0 \Leftrightarrow 1 \f$ binary substitution when used with BPSK
   modulation. For MPSK modulation, this causes the output to be the symbol farthest away
   from the input.
*/
sigspace bsid::corrupt(const sigspace& s)
   {
   const double p = r.fval();
   //libbase::trace << "DEBUG (bsid): p(s) = " << p << "\n";
   if(p < Ps)
      return -s;
   return s;
   }

// Channel functions

/*!
   \brief Pass a sequence of modulation symbols through the channel
   \param[in]  tx  Transmitted sequence of modulation symbols
   \param[out] rx  Received sequence of modulation symbols

   The channel model implemented is described by the following state diagram:
   \dot
   digraph channel {
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

   \note It is possible that the \c tx and \c rx parameters actually point to the same
         vector. Unlike substitution channels, where this does not cause any problems, we
         here have to make sure that we don't corrupt the vector we're reading from;
         therefore, the result is first created as a new vector and only copied over at
         the end.

   \sa corrupt()
*/
void bsid::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
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
   libbase::vector<sigspace> newrx;
   newrx.init(transmit.sum() + insertions.sum());
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0, j=0; i<tau; i++)
      {
      while(insertions(i)--)
         newrx(j++) = (r.fval() < 0.5) ? sigspace(1,0) : sigspace(-1,0);
      if(transmit(i))
         newrx(j++) = corrupt(tx(i));
      }
   // copy results back
   rx = newrx;
   }

/********************************* FBA sub-class object *********************************/

class myfba : public fba<double> {
   // user-defined parameters
   libbase::vector<sigspace>  tx;   // presumed transmitted sequence
   const bsid* channel;
   // pre-computed parameters
   libbase::vector<double> Ptable;
   // implementations of channel-specific metrics for fba
   double P(const int a, const int b);
   double Q(const int a, const int b, const int i, const libbase::vector<sigspace>& s);
public:
   // constructor & destructor
   myfba() {};
   ~myfba() {};
   // set transmitted sequence
   void settx(const libbase::vector<sigspace>& tx);
   // attach channel
   void attach(const bsid* channel);
};

// implementations of channel-specific metrics for fba

inline double myfba::P(const int a, const int b)
   {
   const int m = b-a;
   return Ptable(m+1);
   }
   
inline double myfba::Q(const int a, const int b, const int i, const libbase::vector<sigspace>& s)
   {
   // 'a' and 'b' are redundant because 's' already contains the difference
   assert(s.size() == b-a+1);
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   // compute the conditional probability
   return channel->receive(myfba::tx(i), s);
   }
   
// set transmitted sequence

inline void myfba::settx(const libbase::vector<sigspace>& tx)
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

/*!
   \brief Determine the per-symbol likelihoods of a sequence of received modulation symbols
          corresponding to one transmission step
   \param[in]  tx       Set of possible transmitted symbols
   \param[in]  rx       Received sequence of modulation symbols
   \param[out] ptable   Likelihoods corresponding to each possible transmitted symbol
*/
void bsid::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
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

double bsid::receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const
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
