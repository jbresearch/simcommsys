/*!
 * \file
 *
 * Copyright (c) 2010 Stephan Wesemeyer
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ReedSolomonCodec.h
 *
 *  Created on: 26-Jun-2009
 *      Author: swesemeyer
 */

#ifndef REEDSOLOMON_H_
#define REEDSOLOMON_H_

#include "config.h"
#include "codec_softout.h"
#include "gf.h"
#include "matrix.h"
#include "hard_decision.h"
#include <string>

namespace libcomm {

/*!
 * \brief   Reed-Solomon codec
 * \author S Wesemeyer
 * This class will construct a Reed-Solomon code over F_{q} of length n and dimension k
 * Note that n is either q or q-1 and 1<k<n-1
 * The Berlekamp algorithm is used to decode any received word.
 *
 */
template <class GF_q>
class reedsolomon : public codec_softout<libbase::vector, double> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;

protected:
   // Interface with derived classes
   /*!
    * \brief Encoding process
    * \param[in] source Sequence of source symbols, one per timestep
    * \param[out] encoded Sequence of output (encoded) symbols, one per timestep
    *
    * \note If the input or output symbols at every timestep represent the
    * aggregation of a set of symbols, the combination/division has to
    * be done externally.
    */
   void do_encode(const array1i_t& source, array1i_t& encoded);
   /*!
    * \brief Receiver translation process
    * \param[in] ptable Likelihoods of each possible modulation symbol at every
    * (modulation) timestep
    *
    * This function computes the necessary probability tables for the codec
    * from the probabilities of each modulation symbol as received from the
    * channel. This function should be called before the first decode iteration
    * for each block.
    *
    * \note The number of possible modulation symbols does not necessarily
    * correspond to the number of encoder output symbols, and therefore
    * the number of modulation timesteps may be different from tau.
    */
   void do_init_decoder(const array1vd_t& ptable);
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app);

public:
   //!default constructor needed for serialization
   reedsolomon()
      {
      //nothing to do
      }

   /*! \name Codec operations */
   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      // Seed hard-decision box
      hd_functor.seedfrom(r);
      }

   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    *
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   void softdecode(array1vd_t& ri)
      {
      // set up space for output-referred statistics (to be discarded)
      array1vd_t ro;
      softdecode(ri, ro);
      }
   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    * \param[out] ro Likelihood table for output symbols at every timestep
    *
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   void softdecode(array1vd_t& ri, array1vd_t& ro);
   // @}

   /*! \name Codec information functions - fundamental */
   //! Input block size in symbols, ie the dimension of the code
   libbase::size_type<libbase::vector> input_block_size() const
      {
      return libbase::size_type<libbase::vector>(this->dim_k);
      }

   //! Output block size in symbols, ie the length of the code
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(this->length_n);
      }

   //! Number of valid input combinations, ie the number of bits that make up a symbol in the information set.
   int num_inputs() const
      {
      return GF_q::elements();
      }

   //! Number of valid output combinations, ie the number of bits that make up a symbol used by the encoded version.
   int num_outputs() const
      {
      return GF_q::elements();
      }

   //! Length of tail in timesteps, not required for linear block codes
   int tail_length() const
      {
      //no tail
      return 0;
      }
   //! Number of iterations per decoding cycle, will need to be at least 1.
   int num_iter() const
      {
      return 1;
      }
   // @}

   /* \name Codec information functions - use default implementation*/
   // Equivalent length of information sequence in bits
   //double input_bits() const { return log2(num_inputs())*input_block_size(); };
   // Equivalent length of output sequence in bits
   //double output_bits() const { return log2(num_outputs())*output_block_size(); };
   // Overall code rate
   //double rate() const { return input_bits()/output_bits(); };
   //

   /*! \name Description */
   //! Description output
   std::string description() const;
   // @}

   //! ReedSolomonodec specific methods

   /*!
    * \brief check the parameters
    */
   void checkParams(int length, int dim);

   /*! \name Init
    * \brief initialise the RS codec
    * This method will construct the parity check and generator matrix for the RS code
    * under investigation.
    */
   void init();

   // Serialization Support
DECLARE_SERIALIZER(reedsolomon)

private:
   //! the length of the code
   int length_n;
   //! the dimension of the code
   int dim_k;
   //the dimension of the parity_check matrix
   int dim_pchk;

   //The parity check matrix of the code
   libbase::matrix<GF_q> pchk_matrix;

   //!The generator matrix of the code
   libbase::matrix<GF_q> pchk_ref_matrix;

   //!The generator matrix of the code in REF
   libbase::matrix<GF_q> gen_ref_matrix;
   libbase::vector<GF_q> received_word_hd;
   array1vd_t received_likelihoods;
   array1d_t received_word_sd;

   //! Hard-decision box
   hard_decision<libbase::vector, double, GF_q> hd_functor;
};

}

#endif //REEDSOLOMON_H_
