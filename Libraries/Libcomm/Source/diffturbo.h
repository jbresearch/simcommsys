#ifndef __diffturbo_h
#define __diffturbo_h

#include "config.h"
#include "serializer.h"

#include "turbo.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Diffused-Input Turbo Decoder.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \note The set of interleavers is passed as a vector of pointers. These must
         be allocated on the heap by the user, and will then be deleted by this
         class.
*/

template <class real>
class diffturbo : public turbo<real> {
private:
   std::string filename;
   libbase::vector<int> lut;
   libbase::vector<int> source2, source3;
   libbase::matrix<double> decoded2, decoded3;
   void load_lut(const char *filename, const int tau);
   void add(libbase::matrix<double>& z, libbase::matrix<double>& x, libbase::matrix<double>& y, int zp, int xp, int yp);
protected:
   void init();
   diffturbo() {};
public:
   diffturbo(const char *filename, fsm& encoder, const int tau, libbase::vector<interleaver *>& inter, \
      const int iter, const bool simile, const bool endatzero, const bool parallel=false);
   ~diffturbo() {};

   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void decode(libbase::vector<int>& decoded);

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(diffturbo)
};

}; // end namespace

#endif
