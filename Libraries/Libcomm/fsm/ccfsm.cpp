/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "ccfsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

// Internal functions

/*!
 * \brief Initialization
 * \param  generator   Generator matrix of size \f$ k \times n \f$
 * 
 * Each generator matrix element is a vector over G, laid out in the same format
 * as the internal registers - lower index positions are considered to lie on
 * the left, and correspond with register positions closest to the input
 * junction. This follows the usual convention in the coding community.
 */
template <class G>
void ccfsm<G>::init(const matrix<vector<G> >& generator)
   {
   // copy automatically what we can
   gen = generator;
   k = gen.size().rows();
   n = gen.size().cols();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and create
   // shift registers
   reg.init(k);
   nu = 0;
   for (int i = 0; i < k; i++)
      {
      // assume with of register of input 'i' from its generator sequence for
      // first output
      int m = gen(i, 0).size() - 1;
      reg(i).init(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for (int j = 1; j < n; j++)
         assertalways(gen(i, j).size() == m + 1);
      // update memory order
      if (m > ccfsm<G>::m)
         ccfsm<G>::m = m;
      }
   }

// Helper functions

/*!
 * \copydoc fsm::convert()
 *
 * Interface adaptation to make use of GF class concept of alphabet size
 */
template <class G>
int ccfsm<G>::convert(const vector<G>& vec)
   {
   return fsm::convert(vec, G::elements());
   }

/*!
 * \copydoc fsm::convert()
 *
 * Interface adaptation to make use of GF class concept of alphabet size
 */
template <class G>
vector<G> ccfsm<G>::convert(int val, int nu)
   {
   return fsm::convert(val, nu, G::elements());
   }

/*!
 * \brief Convolves the shift-in value and register with a generator polynomial
 * \param  s  The value at the left shift-in of the register
 * \param  r  The register
 * \param  g  The corresponding generator polynomial
 * \return The output
 * 
 * \todo Document this function with a diagram.
 */
template <class G>
G ccfsm<G>::convolve(const G& s, const vector<G>& r, const vector<G>& g) const
   {
   // Convolve the shift-in value with corresponding generator polynomial
   int m = r.size();
   G thisop = s * g(m);
   // Convolve register with corresponding generator polynomial
   for (m--; m >= 0; m--)
      thisop += r(m) * g(m);
   return thisop;
   }

// Constructors / Destructors

/*!
 * \brief Principal constructor
 */
template <class G>
ccfsm<G>::ccfsm(const matrix<vector<G> >& generator)
   {
   init(generator);
   }

/*!
 * \brief Copy constructor
 */
template <class G>
ccfsm<G>::ccfsm(const ccfsm<G>& x)
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

template <class G>
libbase::vector<int> ccfsm<G>::state() const
   {
   libbase::vector<int> state(nu);
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            state(j++) = reg(i)(t);
   assert(j == nu);
   return state;
   }

template <class G>
void ccfsm<G>::reset()
   {
   fsm::reset(state);
   reg = 0;
   }

template <class G>
void ccfsm<G>::reset(libbase::vector<int> state)
   {
   fsm::reset(state);
   assert(state.size() == nu);
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            reg(i)(t) = state(j++);
   assert(j == nu);
   }

// FSM operations (advance/output/step)

template <class G>
void ccfsm<G>::advance(libbase::vector<int>& input)
   {
   fsm::advance(input);
   input = determineinput(input);
   vector<G> sin = determinefeedin(input);
   // Compute next state for each input register
   for (int i = 0; i < k; i++)
      {
      const int m = reg(i).size();
      if (m == 0)
         continue;
      // Shift entries to the right (ie. up)
      for (int j = m - 1; j > 0; j--)
         reg(i)(j) = reg(i)(j - 1);
      // Left-most entry gets the shift-in value
      reg(i)(0) = sin(i);
      }
   }

template <class G>
libbase::vector<int> ccfsm<G>::output(libbase::vector<int> input) const
   {
   input = determineinput(input);
   vector<G> sin = determinefeedin(input);
   // Compute output
   vector<G> op(n);
   for (int j = 0; j < n; j++)
      {
      G thisop;
      for (int i = 0; i < k; i++)
         thisop += convolve(sin(i), reg(i), gen(i, j));
      op(j) = thisop;
      }
   return op;
   }

// Description & Serialization

//! Description output - common part only, must be preceded by specific name
template <class G>
std::string ccfsm<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << G::elements() << "): (nu=" << nu << ", rate " << k << "/"
         << n << ", G=[";
   // Loop over all generator matrix elements
   for (int i = 0; i < k; i++)
      for (int j = 0; j < n; j++)
         {
         // Loop over polynomial
         for (int x = 0; x < gen(i, j).size(); x++)
            sout << "{" << gen(i, j)(x) << "}";
         sout << (j == n - 1 ? (i == k - 1 ? "])" : "; ") : ", ");
         }
   return sout.str();
   }

template <class G>
std::ostream& ccfsm<G>::serialize(std::ostream& sout) const
   {
   sout << gen;
   return sout;
   }

template <class G>
std::istream& ccfsm<G>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> gen;
   init(gen);
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "gf.h"

namespace libcomm {

using libbase::gf;

// Degenerate case GF(2)

template class ccfsm<gf<1, 0x3> > ;

// cf. Lin & Costello, 2004, App. A

template class ccfsm<gf<2, 0x7> > ;
template class ccfsm<gf<3, 0xB> > ;
template class ccfsm<gf<4, 0x13> > ;

} // end namespace
