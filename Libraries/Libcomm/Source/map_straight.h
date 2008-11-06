#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Straight Mapper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines a straight symbol mapper with:
   * forward transform from blockmodem
   * inverse transform from the various codecs.
*/

class map_straight : public mapper {
private:
   /*! \name Internal object representation */
   int s1;        //!< Number of modulation symbols per encoder output
   int s2;        //!< Number of modulation symbols per translation symbol
   int upsilon;   //!< Block size in symbols at codec translation
   // @}

protected:
   // Interface with mapper
   void setup();
   void dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const;

public:
   // Informative functions
   double rate() const { return 1; };
   int output_block_size() const { return tau*s1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight)
};

}; // end namespace

#endif
