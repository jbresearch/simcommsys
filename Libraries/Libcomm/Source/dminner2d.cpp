/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner2d.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

/*!
   \brief Confirm that LUT is valid
   Checks that all LUT entries are of the correct size and that there are no
   duplicate entries.
*/

template <class real, bool normalize>
void dminner2d<real,normalize>::validatelut() const
   {
   assertalways(lut.size() == num_symbols());
   for(int i=0; i<lut.size(); i++)
      {
      // all entries should be of the correct size
      assertalways(lut(i).xsize() == m);
      assertalways(lut(i).ysize() == n);
      // all entries should be distinct
      for(int j=0; j<i; j++)
         assertalways(lut(i).isnotequalto(lut(j)));
      }
   }

/*!
   \brief Object initialization
   Determines code parameters from LUT and sets up object for use.
   This includes validating the LUT, setting up the pilot generator,
   and clearing the pilot sequence.
*/

template <class real, bool normalize>
void dminner2d<real,normalize>::init()
   {
   // Determine code parameters from LUT
   q = lut.size();
   assertalways(q > 0);
   m = lut(0).xsize();
   n = lut(0).ysize();
   // Validate LUT
   validatelut();
   // Seed the watermark generator and clear the sequence
   //r.seed(0);
   //ws.init(0);
   }

// description output

template <class real, bool normalize>
std::string dminner2d<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative 2D DM Inner Code (";
   sout << m << "x" << n << "/" << q << ", ";
   sout << lutname << " codebook)";
   return sout.str();
   }

// object serialization - saving

template <class real, bool normalize>
std::ostream& dminner2d<real,normalize>::serialize(std::ostream& sout) const
   {
   sout << lutname;
   sout << lut;
   return sout;
   }

// object serialization - loading

template <class real, bool normalize>
std::istream& dminner2d<real,normalize>::serialize(std::istream& sin)
   {
   sin >> lutname;
   sin >> lut;
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2d<logrealfast,false>;
template <>
const serializer dminner2d<logrealfast,false>::shelper
   = serializer("blockmodem", "dminner2d<logrealfast>", dminner2d<logrealfast,false>::create);

template class dminner2d<double,true>;
template <>
const serializer dminner2d<double,true>::shelper
   = serializer("blockmodem", "dminner2d<double>", dminner2d<double,true>::create);

}; // end namespace
