#ifndef __mapcc_h
#define __mapcc_h

#include "config.h"

#include "codec.h"
#include "fsm.h"
#include "bcjr.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

namespace libcomm {

/*!
   \brief   Maximum A-Posteriori Decoder.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Check for both flags (for terminated and circular trellises) being set.
*/

template <class real>
class mapcc : public codec, private bcjr<real> {
private:
   fsm      *encoder;
   double   rate;
   int      tau;           //!< Block length (including tail, if any)
   bool     endatzero;     //!< True for terminated trellis
   bool     circular;      //!< True for circular trellis
   int      m;             //!< encoder memory order
   int      M;             //!< Number of states
   int      K;             //!< Number of input combinations
   int      N;             //!< Number of output combinations
   libbase::matrix<double> R, ri, ro;   // BCJR statistics
protected:
   void init();
   void free();
   void reset();
   mapcc();
public:
   mapcc(const fsm& encoder, const int tau, const bool endatzero, const bool circular=false);
   ~mapcc() { free(); };

   void encode(const libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);

   int block_size() const { return tau; };
   int num_inputs() const { return K; };
   int num_outputs() const { return N; };
   int tail_length() const { return m; };
   int num_iter() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(mapcc)
};

}; // end namespace

#endif

