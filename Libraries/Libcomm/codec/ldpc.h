/*
 * LDPC_GFq.h
 *
 *  Created on: 9 Jul 2009
 *      Author: swesemeyer
 */

#ifndef LDPC_H_
#define LDPC_H_

#include <string>
#include <iostream>
#include "codec_softout.h"
#include "vector.h"
#include "matrix.h"
#include "gf.h"

using std::string;
using libcomm::codec_softout;
using libbase::gf;
using libbase::matrix;

namespace libcomm {
/*!
 * \brief   LDPC codec
 * \author S Wesemeyer
 * This class will decode an LDPC code over F_{q} of length n and dimension m
 * The LDPC code is defined by a sparse parity check matrix which needs to be
 * provided in the alist format (see MacKay:http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html)
 *
 * \section svn Version Control
 * - $Revision: 2668 $
 * - $Date: 2009-07-16 15:41:52 +0100 (Thu, 16 Jul 2009) $
 * - $Author: swesemeyer $
 */

template <class GF_q>
class ldpc : public codec_softout<libbase::vector, double> {

private:
   /* see MacKay's Information Theory, Inference and Learning Algs (2003, ch 47.3,pp 559-561)
    * for a proper definition of the following variables.
    */

   //!this struct holds the probabilities that check m is satisfied if symbol n of the received word is
   //fixed at symbols and the other symbols(<>n) have separable distributions given by q_mxn
   struct marginals {
      array1d_t q_mxn;
      array1d_t r_mxn;
   };

public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}


   /*
    * The following methods are needed so that the abstract class codec_softoutput
    * can translate the codec methods translate and decode into the corresponding
    * soft decision methods.
    */

protected:
   /*! \name Internal codec operations */
   /*!
    * \brief A-priori probability initialization
    * 
    * This function resets the a-priori prabability tables for the codec to
    * equally-likely. This function (or setpriors) should be called before the
    * first decode iteration for each block.
    */
   void resetpriors();
   /*!
    * \brief A-priori probability setup
    * \param[in] ptable Likelihoods of each possible input symbol at every
    * (input) timestep
    * 
    * This function updates the a-priori prabability tables for the codec.
    * This function (or resetpriors) should be called before the first decode
    * iteration for each block.
    */
   void setpriors(const array1vd_t& ptable);

   /*!
    * \note Sets up receiver likelihood tables only.
    */
   void setreceiver(const array1vd_t& ptable);
   // @}


public:
   ldpc()
      {
      this->decodingSuccess = false;
      }
   ~ldpc()
      {
      //nothing to do
      }

   /*! \name Codec operations */
   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      //not needed
      }

   /*!
    * \brief Encoding process
    * \param[in] source Sequence of source symbols, one per timestep
    * \param[out] encoded Sequence of output (encoded) symbols, one per timestep
    * 
    * \note If the input or output symbols at every timestep represent the
    * aggregation of a set of symbols, the combination/division has to
    * be done externally.
    */
   void
   encode(const array1i_t & source, array1i_t& encoded);

   /*! \name Softout codec operations */

   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    * 
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   void softdecode(array1vd_t& ri);

   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    * \param[out] ro Likelihood table for output symbols at every timestep
    * 
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   /*
    * some more necessary functions for the codec interface
    */

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
   //! Number of valid input combinations
   int num_inputs() const
      {
      return GF_q::elements();
      }

   //! Number of valid output combinations
   int num_outputs() const
      {
      return GF_q::elements();
      }

   // Channel symbol alphabet size required for translation - use provided implementation.
   // int num_symbols() const

   //! Length of tail in timesteps
   int tail_length() const
      {
      //we don't have a tail.
      return 0;
      }
   //! Number of iterations per decoding cycle
   int num_iter() const
      {
      return this->max_iter;
      }
   // @}

   // \name Codec information functions - derived */
   // Equivalent length of information sequence in bits
   // double input_bits() const; use default implementation

   // Equivalent length of output sequence in bits
   //double output_bits() const; use default implementation

   // Overall code rate
   //double rate() const; - use default implementation
   //

   /*! \name Description */
   //! Description output - describe the LDPC code in detail
   std::string description() const;
   // @}

   /*! \name read_alist
    * reads the parity check matrix given in the "alist" format
    */
   std::istream& read_alist(std::istream& sin);

   /*! \name write_alist
    * writes the parity check matrix of the current code in the "alist" format
    */
   std::ostream& write_alist(std::ostream& sout) const;

   // Serialization Support
DECLARE_SERIALIZER(ldpc);

   //private methods
private:
   //! \brief initialises the LDPC codec
   // simply initialises the LDPC code and checks that the parity check matrix
   // has the right dimensions
   void init();

   /*!
    * \brief compute the r_mn(0) and r_mn(1) values for each iteration
    * This is the horizontal step in the Sum-Product Algorithm
    */
   void compute_r_mn(int m, int n, const array1i_t & N_m);

   /*!
    * \brief compute the q_mn(0) and q_mn(1) values for each iteration
    * * This is the vertical step in the Sum-Product Algorithm
    */
   void compute_q_mn(int m, int n, const array1i_t & M_n);

   /*!
    * \brief compute the probabilities for each symbol given the
    * derived probabilities in this iteration.
    * This is the tentative decoding step
    */
   void compute_probs(array1vd_t& ro);

   /*!
    * \brief output the marginal matrix
    */
   void print_marginal_probs(std::ostream& s);

   /*
    * internal variables needed by LDPC_GFq
    */
private:

   //!the maximum number of iterations
   int max_iter;

   //!Counter indicating the current iteration
   int current_iteration;

   //! length of the binary LDPC code
   int length_n;

   //!the dimension of the parity check matrix
   //Note that this is not necessarily the dimension
   //of the code described by the parity check matrix
   //as the rows of the parity check matrix do not
   //need to be linearly independent.
   int dim_pchk;

   //!the dimension of the code
   //Note that dim_k is the true dimension of the code
   int dim_k;

   //!the max row weight of the parity check matrix
   int max_row_weight;

   //!the max column weight of the parity check matrix
   int max_col_weight;

   //!the weight of each row
   array1i_t row_weight;

   //!the weight of each col
   array1i_t col_weight;

   //the positions of the non-zero entries per row
   array1vi_t N_m;

   //the positions of the non-zero entries per col
   array1vi_t M_n;

   //The parity check matrix of the code
   matrix<GF_q> pchk_matrix;

   //!The generator matrix of the code in REF
   matrix<GF_q> gen_matrix;

   //! the permutation that swaps the columns so that
   //the parity check matrix is in standard form, eg (I|P)
   array1i_t perm_to_systematic;

   //!The positions of the info symbols in a code word
   array1i_t info_symb_pos;

   //!the normalised received probabilities per symbol of the received word
   array1vd_t received_probs;

   //!the probabilities per symbol of the computed solution
   array1vd_t computed_solution;

   //! The syndrome computed from the received word
   libbase::vector<GF_q> syndrome;

   //! flag indicating whether or not the current iteration
   //has resulted in a codeword
   bool decodingSuccess;

   //!this is the most likely received word (eg the maximum value of the
   //received likelihoods are used to form this word)
   array1d_t received_word_sd;

   //!this is the hard decision received word. This is used to compute
   //the syndrome
   libbase::vector<GF_q> received_word_hd;

   //! this matrix holds the r_mxn probabilities
   matrix<marginals> marginal_probs;

};

}

#endif /* LDPC_H_ */
