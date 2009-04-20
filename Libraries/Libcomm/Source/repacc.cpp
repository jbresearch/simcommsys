/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "repacc.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void repacc<real,dbl>::init()
   {
   assertalways(encoder);
   bcjr<real,dbl>::init(*encoder, output_block_size().x);
   // check that encoder is rate-1
   assertalways(num_inputs() == num_outputs());
   assertalways(inter);
   // TODO: check interleaver sizes
   assertalways(iter > 0);

   initialised = false;
   }

template <class real, class dbl>
void repacc<real,dbl>::free()
   {
   if(encoder != NULL)
      delete encoder;
   if(inter != NULL)
      delete inter;
   }

template <class real, class dbl>
void repacc<real,dbl>::reset()
   {
   if(endatzero)
      {
      bcjr<real,dbl>::setstart(0);
      bcjr<real,dbl>::setend(0);
      }
   else
      {
      bcjr<real,dbl>::setstart(0);
      bcjr<real,dbl>::setend();
      }
   }

// constructor / destructor

template <class real, class dbl>
repacc<real,dbl>::repacc()
   {
   encoder = NULL;
   inter = NULL;
   }

// description output

template <class real, class dbl>
std::string repacc<real,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Repeat-Accumulate Code (" << N << "," << q << ") - ";
   sout << encoder->description() << ", ";
   sout << inter->description() << ", ";
   sout << iter << " iterations, ";
   sout << (endatzero ? "Terminated" : "Unterminated");
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& repacc<real,dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << 1 << '\n';
   sout << encoder;
   sout << inter;
   sout << N << '\n';
   sout << q << '\n';
   sout << iter << '\n';
   sout << int(endatzero) << '\n';
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& repacc<real,dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> version;
   // get first-version items
   sin >> encoder;
   sin >> inter;
   sin >> N;
   sin >> q;
   sin >> iter;
   sin >> endatzero;
   init();
   assertalways(sin.good());
   return sin;
   }

}; // end namespace
