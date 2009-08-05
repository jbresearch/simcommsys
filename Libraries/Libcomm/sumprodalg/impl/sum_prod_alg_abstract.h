/*
 * sum_product_alg.h
 *
 *  Created on: 21 Jul 2009
 *      Author: swesemeyer
 */

#ifndef SUM_PROD_ALG_ABSTRACT_H_
#define SUM_PROD_ALG_ABSTRACT_H_

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "gf.h"
#include "sumprodalg/sum_prod_alg_inf.h"

namespace libcomm {
using libbase::gf;
using libbase::matrix;
/*! \vrief Sum Product Algorithm(SPA) implementation
 *
 * Currently 2 types of the SPA: trad and gdl
 * The trad version computes the probabilities for the r__mxn's by computing
 * all the possible combinations of info symbols that satisfy the check node.
 * This can be very expensive computationally (especially when GF(q>2)
 * but it easy to code and understand.
 * The gdl version uses the fact that these probabilities can be grouped differently
 * using the distributive law and hence be computed much faster. The version
 * that is implemented here is based on Declercqs and Fossorier's 2006 paper:
 * Decoding Algorithms for Nonbinary LDPC Codes over GF(q)
 */
template <class GF_q, class real = double> class sum_prod_alg_abstract : public sum_prod_alg_inf<
      GF_q, real> {
public:

   /*! \name Type definitions */
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}

   /*! \brief constructor
    * initialise the main variables
    */
   sum_prod_alg_abstract(int n, int m, const array1vi_t& non_zero_col_pos,
         const array1vi_t& non_zero_row_pos, const matrix<GF_q>& pchk_matrix) :
      length_n(n), dim_m(m), M_n(non_zero_col_pos), N_m(non_zero_row_pos)
      {
      marginal_probs.init(m, n);

      int non_zeros = 0;
      int pos = 0;

      for (int loop_m = 0; loop_m < this->dim_m; loop_m++)
         {
         non_zeros = this->N_m(loop_m).size();
         for (int loop_n = 0; loop_n < non_zeros; loop_n++)
            {
            pos = this->N_m(loop_m)(loop_n) - 1;//we count from zero;

            this->marginal_probs(loop_m, pos).val = pchk_matrix(loop_m, pos);
            }
         }
      }
   /*! \brief default destructor
    *
    */
   virtual ~sum_prod_alg_abstract()
      {
      //nothing to do
      }

   /*! \brief initialise the Sum Product algorithm with the relevant probabilities
    *
    */
   virtual void spa_init(const array1vd_t& ptable)=0;
   /*! \brief this returns the type of the Sum Product algorithm
    *
    */
   virtual std::string spa_type()=0;

   /*!\brief carry out one iteration of the SPA
    * This method will carry out the horizontal and vertical step
    * of the SPA and store the result in the ro vector
    */
   void spa_iteration(array1vd_t& ro);

protected:
   /*! \brief carries out the horizontal step of SPA
    * The r_mxn probabilities are computed
    */
   virtual void compute_r_mn(int m, int n, const array1i_t & tmpN_m)=0;
   /*! \brief carried out the horizontal step of the SPA
    * the q_mxn probabilities are computed
    */
   virtual void compute_q_mn(int m, int n, const array1i_t & M_n)=0;

private:
   void compute_probs(array1vd_t& ro);
   void print_marginal_probs(std::ostream& sout);
   void print_marginal_probs(int col, std::ostream& sout);

protected:

   /*! \name Data structures
    * LDPC specific datastructure used by the Sum-Product Algorithm
    */

   /* see MacKay's Information Theory, Inference and Learning Algs (2003, ch 47.3,pp 559-561)
    * for a proper definition of the following variables.
    */

   //!this struct holds the probabilities that check m is satisfied if symbol n of the received word is
   //fixed at symbols and the other symbols(<>n) have separable distributions given by q_mxn
   struct marginals {
      array1d_t q_mxn;
      array1d_t qmn_conv;//! this holds the fast FFT transforms of the q_mxns
      array1d_t r_mxn;
      GF_q val; //this holds the non_zero entry at position (m,n)
   };
   // @}

   //the number of cols
   int length_n;
   //the number of rows
   int dim_m;

   array1vd_t received_probs;

   //the positions of the non-zero entries per col
   array1vi_t M_n;

   //the positions of the non-zero entries per row
   array1vi_t N_m;

   //! this matrix holds the r_mxn probabilities
   matrix<marginals> marginal_probs;
};
}

#endif /* SUM_PROD_ALG_ABSTRACT_H_ */
