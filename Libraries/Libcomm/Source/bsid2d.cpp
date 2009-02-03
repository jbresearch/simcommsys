/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bsid2d.h"
#include "secant.h"
#include <sstream>
#include <limits>

namespace libcomm {

const libbase::serializer bsid2d::shelper("channel", "bsid2d", bsid2d::create);

// FBA decoder parameter computation

/*!
   \brief Determine limit for insertions between two time-steps

   \f[ I = \left\lceil \frac{ \log{P_r} - \log \tau }{ \log p } \right\rceil - 1 \f]
   where \f$ P_r \f$ is an arbitrary probability of having a block of size
   \f$ \tau \f$ with at least one event of more than \f$ I \f$ insertions
   between successive time-steps.
   In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.

   \note The smallest allowed value is \f$ I = 1 \f$
*/
int bsid2d::compute_I(int tau, double p)
   {
   int I = int(ceil((log(1e-12) - log(double(tau))) / log(p))) - 1;
   I = std::max(I,1);
   libbase::trace << "DEBUG (bsid2d): for N = " << tau << ", I = " << I << "/";
   I = std::min(I,2);
   libbase::trace << I << ".\n";
   return I;
   }

/*!
   \brief Determine maximum drift over a whole N-bit block

   \f[ x_{max} = Q^{-1}(\frac{P_r}{2}) \sqrt{\frac{\tau p}{1-p}} \f]
   where \f$ p = P_i = P_d \f$ and \f$ P_r \f$ is an arbitrary probability of
   having a block of size \f$ \tau \f$ where the drift at the end is greater
   than \f$ \pm x_{max} \f$.
   In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.

   The calculation is based on the assumption that the end-of-frame drift has
   a Gaussian distribution with zero mean and standard deviation given by
   \f$ \sigma = \sqrt{\frac{\tau p}{1-p}} \f$.

   \note The smallest allowed value is \f$ x_{max} = I \f$
*/
int bsid2d::compute_xmax(int tau, double p, int I)
   {
   // rather than computing the factor using a root-finding method,
   // we fix factor = 7.1305, corresponding to Qinv(1e-12/2.0)
   const double factor = 7.1305;
   int xmax = int(ceil(factor * sqrt(tau*p/(1-p))));
   xmax = std::max(xmax,I);
   libbase::trace << "DEBUG (bsid2d): for N = " << tau << ", xmax = " << xmax << "/";
   //xmax = min(xmax,25);
   libbase::trace << xmax << ".\n";
   return xmax;
   }

/*!
   \copydoc bsid2d::compute_xmax()

   \note Provided for convenience; will determine I itself, then use that to
         determine xmax.
*/
int bsid2d::compute_xmax(int tau, double p)
   {
   int I = compute_I(tau,p);
   return compute_xmax(tau,p,I);
   }

/*!
   \brief Compute receiver coefficient set

   First row has elements where the last bit \f[ r_\mu = t \f]
   \f[ Rtable(0,\mu) =
      \left(\frac{P_i}{2}\right)^\mu
      \left( (1-P_i-P_d) (1-P_s) + \frac{1}{2} P_i P_d \right)
      , \mu \in (0, \ldots x_{max}) \f]

   Second row has elements where the last bit \f[ r_\mu \neq t \f]
   \f[ Rtable(1,\mu) =
      \left(\frac{P_i}{2}\right)^\mu
      \left( (1-P_i-P_d) P_s + \frac{1}{2} P_i P_d \right)
      , \mu \in (0, \ldots x_{max}) \f]
*/
void bsid2d::compute_Rtable(array2d_t& Rtable, int xmax, double Ps, double Pd, double Pi)
   {
   // Allocate required size
   Rtable.resize(boost::extents[2][xmax+1]);
   // Set values for insertions
   const double a1 = (1-Pi-Pd);
   const double a2 = 0.5*Pi*Pd;
   for(int mu=0; mu<=xmax; mu++)
      {
      const double a3 = pow(0.5*Pi, mu);
      Rtable[0][mu] = a3 * (a1 * (1-Ps) + a2);
      Rtable[1][mu] = a3 * (a1 * Ps + a2);
      }
   }

// Internal functions

/*!
   \brief Sets up pre-computed values

   This function computes all cached quantities used within actual channel
   operations. Since these values depend on the channel conditions, this
   function should be called any time a channel parameter is changed.
*/
void bsid2d::precompute()
   {
   if(N == 0)
      {
      I = 0;
      xmax = 0;
      // reset array
      Rtable.resize(boost::extents[0][0]);
      return;
      }
   assert(N>0);
   // fba decoder parameters
   I = compute_I(N, Pd);
   xmax = compute_xmax(N, Pd, I);
   // receiver coefficients
   compute_Rtable(Rtable, xmax, Ps, Pd, Pi);
   Rval = Pd;
   }

/*!
   \brief Initialization

   Sets the channel with \f$ P_s = P_d = P_i = 0 \f$. This way, any
   of the parameters not flagged to change with channel SNR will remain zero.
*/
void bsid2d::init()
   {
   // channel parameters
   Ps = 0;
   Pd = 0;
   Pi = 0;
   // set block size to unusable value
   N = 0;
   precompute();
   }

/*!
   \brief Determine state-machine values for a single timestep

   Returns the number of insertions \e before given position, and whether the
   given position is transmitted or deleted.
*/
void bsid2d::computestate(int& insertions, bool& transmit)
   {
   // initialize accumulators
   insertions = 0;
   transmit = true;
   // determine state sequence for this timestep
   double p;
   while((p = r.fval()) < Pi)
      insertions++;
   if(p < (Pi+Pd))
      transmit = false;
   }

/*!
   \brief Determine state-machine values for a whole block

   \cf computestate();
*/
void bsid2d::computestate(array2i_t& insertions, array2b_t& transmit)
   {
   // determine matrix sizes
   const int M = insertions.xsize();
   const int N = insertions.ysize();
   assertalways(transmit.xsize() == M);
   assertalways(transmit.ysize() == N);
   // iterate over all timesteps
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         computestate(insertions(i,j), transmit(i,j));
   }

/*!
   \brief Accumulate matrix elements over given dimension
   \param m Matrix containing data to be accumulated
   \param dim Dimension over which to accumulate (0 or 1)
*/
void bsid2d::cumsum(array2i_t& m, int dim)
   {
   assert(dim==0 || dim==1);
   // determine matrix sizes
   const int M = m.xsize();
   const int N = m.ysize();
   // iterate over all matrix elements
   if(dim==0)
      {
      for(int j=0; j<N; j++)
         for(int i=1; i<M; i++)
            m(i,j) += m(i-1,j);
      }
   else
      {
      for(int i=0; i<M; i++)
         for(int j=1; j<N; j++)
            m(i,j) += m(i,j-1);
      }
   }

// Constructors / Destructors

/*!
   \brief Principal constructor

   \sa init()
*/
bsid2d::bsid2d(const bool varyPs, const bool varyPd, const bool varyPi) :
   varyPs(varyPs), varyPd(varyPd), varyPi(varyPi)
   {
   // channel update flags
   assert(varyPs || varyPd || varyPi);
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
void bsid2d::set_parameter(const double p)
   {
   set_ps(varyPs ? p : 0);
   set_pd(varyPd ? p : 0);
   set_pi(varyPi ? p : 0);
   libbase::trace << "DEBUG (bsid2d): Ps = " << Ps << ", Pd = " << Pd << ", Pi = " << Pi << "\n";
   }

/*!
   \brief Get channel parameter

   This returns the value of the first of Ps, Pd, or Pi that are flagged to change.
   If none of these are flagged to change, this constitutes an error condition.
*/
double bsid2d::get_parameter() const
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

void bsid2d::set_ps(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsid2d::Ps = Ps;
   }

void bsid2d::set_pd(const double Pd)
   {
   assert(Pd >=0 && Pd <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid2d::Pd = Pd;
   precompute();
   }

void bsid2d::set_pi(const double Pi)
   {
   assert(Pi >=0 && Pi <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid2d::Pi = Pi;
   precompute();
   }

void bsid2d::set_blocksize(int M, int N)
   {
   if(M != bsid2d::M || N != bsid2d::N)
      {
      assert(M > 0);
      bsid2d::M = M;
      assert(N > 0);
      bsid2d::N = N;
      precompute();
      }
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
bool bsid2d::corrupt(const bool& s)
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
   digraph bsid2dstates {
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

   \note We have initially no idea how long the received sequence will be, so
         we first determine the state sequence at every timestep, keeping
         track of:
            - the number of insertions \e before given position, and
            - whether the given position is transmitted or deleted.

   \note We have to make sure that we don't corrupt the array we're reading
         from (in the case where tx and rx are the same array); therefore,
         the result is first created as a new array and only copied over at
         the end.

   \sa corrupt()
*/
void bsid2d::transmit(const array2b_t& tx, array2b_t& rx)
   {
   // determine matrix sizes
   const int M = tx.xsize();
   const int N = tx.ysize();
   // compute state tables
   array2i_t insertions_h(M,N), insertions_v(M,N);
   array2b_t transmit_h(M,N), transmit_v(M,N);
   computestate(insertions_h, transmit_h);
   computestate(insertions_v, transmit_v);
   // initialize coordinate transformations
   array2i_t ii(M,N), jj(M,N);
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         {
         ii(i,j) = insertions_v(i,j)+(transmit_v(i,j) ? 0 : -1);
         jj(i,j) = insertions_h(i,j)+(transmit_h(i,j) ? 0 : -1);
         }
   // accumulate differential
   cumsum(ii, 0);
   cumsum(jj, 1);
   // compute final coordinates
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         {
         ii(i,j) += i;
         jj(i,j) += j;
         }
   // invert coordinates for transmitted bits
   const int MM = ii.max()+1;
   const int NN = jj.max()+1;
   array2i_t iii(MM,NN), jjj(MM,NN);
   iii = -1;
   jjj = -1;
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         {
         if(transmit_h(i,j) & transmit_v(i,j))
            {
            iii(ii(i,j),jj(i,j)) = i;
            jjj(ii(i,j),jj(i,j)) = j;
            }
         }
   // Initialize results vector
   array2b_t newrx(MM,NN);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<MM; i++)
      for(int j=0; j<NN; j++)
         {
         if(iii(i,j) == -1) // insertion or padding
            {
            assert(jjj(i,j) == -1);
            newrx(i,j) = (r.fval() < 0.5);
            }
         else // transmission
            {
            assert(jjj(i,j) >= 0);
            newrx(i,j) = corrupt(tx(iii(i,j),jjj(i,j)));
            }
         }
   // copy results back
   rx = newrx;
   }

// description output

std::string bsid2d::description() const
   {
   std::ostringstream sout;
   sout << "2D BSID channel (" << varyPs << varyPd << varyPi << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& bsid2d::serialize(std::ostream& sout) const
   {
   sout << 1 << "\n";
   sout << varyPs << "\n";
   sout << varyPd << "\n";
   sout << varyPi << "\n";
   return sout;
   }

// object serialization - loading

std::istream& bsid2d::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> version;
   // read parameters
   sin >> varyPs;
   sin >> varyPd;
   sin >> varyPi;
   init();
   return sin;
   }

}; // end namespace
