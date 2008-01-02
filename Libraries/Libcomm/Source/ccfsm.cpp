/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "ccfsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

// Internal functions

/*! \brief Initialization
    \param  generator   Generator matrix of size \f$ k \times n \f$

    Each generator matrix element is a vector over G, laid out in the same format
    as the internal registers - lower index positions are considered to lie on the
    right, and correspond with register positions farther away from the input
    junction.
*/
template <class G> void ccfsm<G>::init(const matrix< vector<G> >& generator)
   {
   // copy automatically what we can
   gen = generator;
   k = gen.xsize();
   n = gen.ysize();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and create shift registers
   reg.init(k);
   nu = 0;
   for(int i=0; i<k; i++)
      {
      // assume with of register of input 'i' from its generator sequence for first output
      int m = gen(i,0).size() - 1;
      reg(i).init(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for(int j=1; j<n; j++)
         if(gen(i,j).size() != m+1)
            {
            std::cerr << "FATAL ERROR (ccfsm): Generator sequence must have constant width for each input.\n";
            exit(1);
            }
      // update memory order
      if(m > ccfsm<G>::m)
         ccfsm<G>::m = m;
      }
   }


// Helper functions

/*! \brief Conversion from vector spaces to integer
    \param[in]  x  Input in vector representation
    \param[in]  y  Initial integer value (set to zero if this is the first vector)
    \return Value of \c x in integer representation; any prior value \c y is
            shifted to the left before adding the conversion of \c x

    \note Left-most register positions (ie. those closest to the input junction) are
          represented by higher index positions, and get higher-order positions within
          the integer representation.
*/
template <class G> int ccfsm<G>::convert(const vector<G>& x, int y) const
   {
   for(int i=x.size()-1; i>=0; i--)
      {
      y *= G::elements();
      y += x(i);
      }
   return y;
   }

/*! \brief Conversion from integer to vector space
    \param[in]  x  Input in integer representation
    \param[out] y  Pre-allocated vector for storing result - must be of correct size
    \return Any remaining (shifted) higher-order value from \c x

    \note Left-most register positions (ie. those closest to the input junction) are
          represented by higher index positions, and get higher-order positions within
          the integer representation.
*/
template <class G> int ccfsm<G>::convert(int x, vector<G>& y) const
   {
   for(int i=0; i<y.size(); i++)
      {
      y(i) = x % G::elements();
      x /= G::elements();
      }
   return x;
   }

/*! \brief Convolves the shift-in value and register with a generator polynomial
    \param  s  The value at the left shift-in of the register
    \param  r  The register
    \param  g  The corresponding generator polynomial
    \return The output

    \todo Document this function with a diagram.
*/
template <class G> G ccfsm<G>::convolve(const G& s, const vector<G>& r, const vector<G>& g) const
   {
   // Convolve the shift-in value with corresponding generator polynomial
   int m = r.size();
   G thisop = s * g(m);
   // Convolve register with corresponding generator polynomial
   for(m--; m>=0; m--)
      thisop += r(m) * g(m);
   return thisop;
   }


// Constructors / Destructors

/*! \brief Principal constructor
*/
template <class G> ccfsm<G>::ccfsm(const matrix< vector<G> >& generator)
   {
   init(generator);
   }

/*! \brief Copy constructor
*/
template <class G> ccfsm<G>::ccfsm(const ccfsm<G>& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   nu = x.nu;
   m = x.m;
   gen = x.gen;
   reg = x.reg;
   }


// FSM state operations (getting and resetting)

/*! \brief The current state
    \return A unique integer representation of the current state

    \note Lower-order inputs get lower-order positions within the state representation.

    \note Left-most register positions (ie. those closest to the input junction) are
          represented by higher index positions, and get higher-order positions within
          the state representation.

    \invariant The state value should always be between 0 and num_states()-1
*/
template <class G> int ccfsm<G>::state() const
   {
   int state = 0;
   for(int i=k-1; i>=0; i--)
      state = convert(reg(i), state);
   assert(state >= 0 && state < num_states());
   return state;
   }

/*! \brief Reset to a specified state
    \param state  A unique integer representation of the state we want to set to; this
                  can be any value between 0 and num_states()-1

    \cf state()
*/
template <class G> void ccfsm<G>::reset(int state)
   {
   assert(state >= 0 && state < num_states());
   for(int i=k-1; i>=0; i--)
      state = convert(state, reg(i));
   assert(state == 0);
   }


// FSM operations (advance/output/step)

/*! \brief Feeds the specified input and advances the state
    \param[in,out]   input    Integer representation of current input; if this is the
                              'tail' value, it will be updated
*/
template <class G> void ccfsm<G>::advance(int& input)
   {
   input = determineinput(input);
   vector<G> sin = determinefeedin(input);
   // Compute next state for each input register
   for(int i=0; i<k; i++)
      {
      const int m = reg(i).size();
      // Shift entries to the right (ie. down)
      for(int j=1; j<m; j++)
         reg(i)(j-1) = reg(i)(j);
      // Left-most entry gets the shift-in value
      reg(i)(m-1) = sin(i);
      }
   }

/*! \brief Computes the output for the given input and the present state
    \param  input    Integer representation of current input; may be the 'tail' value
    \return Integer representation of the output
*/
template <class G> int ccfsm<G>::output(int input) const
   {
   input = determineinput(input);
   vector<G> sin = determinefeedin(input);
   // Compute output
   vector<G> op(n);
   for(int j=0; j<n; j++)
      {
      G thisop;
      for(int i=0; i<k; i++)
         thisop += convolve(sin(i), reg(i), gen(i,j));
      op(j) = thisop;
      }
   return convert(op);
   }


// Description & Serialization

/*! \brief Description output - common part only, must be preceded by specific name
*/
template <class G> std::string ccfsm<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << G::elements() << "): (nu=" << nu << ", rate " << k << "/" << n << ", G=[";
   // Loop over all generator matrix elements
   for(int i=0; i<k; i++)
      for(int j=0; j<n; j++)
         {
         // Loop over polynomial
         for(int x=gen(i,j).size()-1; x>=0; x--)
            sout << "{" << gen(i,j)(x) << "}";
         sout << (j==n-1 ? (i==k-1 ? "])" : "; ") : ", ");
         }
   return sout.str();
   }

/*! \brief Serialization output
*/
template <class G> std::ostream& ccfsm<G>::serialize(std::ostream& sout) const
   {
   sout << gen;
   return sout;
   }

/*! \brief Serialization input
*/
template <class G> std::istream& ccfsm<G>::serialize(std::istream& sin)
   {
   sin >> gen;
   init(gen);
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "gf.h"

namespace libcomm {

using libbase::gf;

// Degenerate case GF(2)

template class ccfsm< gf<1,0x3> >;

// cf. Lin & Costello, 2004, App. A

template class ccfsm< gf<2,0x7> >;
template class ccfsm< gf<3,0xB> >;
template class ccfsm< gf<4,0x13> >;

}; // end namespace
