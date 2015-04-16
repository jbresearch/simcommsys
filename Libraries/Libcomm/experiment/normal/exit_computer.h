/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __exit_computer_h
#define __exit_computer_h

#include "config.h"
#include "experiment/experiment_normal.h"
#include "commsys.h"
#include "randgen.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   EXIT Chart Computer.
 * \author  Johann Briffa
 *
 * A simulator object for computing the EXIT curves for the inner (modem) and
 * outer (codec) codes in a serially concatenated iterative communication
 * system. The system's channel parameter to be used is serialized as part
 * of this object.
 *
 * The parameter passed by the simulator determines the mutual
 * information at the input; this is used when generating the multi-binary
 * priors (with Gaussian assumption). These are passed through the first stage
 * decoding; the extrinsic information generated is passed to the second stage
 * decoder. The mutual information at the input and output of the second stage
 * decoder are computed and returned as results. The process is repeated with
 * the decoders swapped.
 *
 * \todo Add serialization setting to indicate what type of results to compute
 * (e.g. straight mutual information, sigma/mu of binary representation, etc)
 */

template <class S>
class exit_computer : public experiment_normal {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<S> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   enum exit_t {
      exit_parallel_codec = 0, //!< parallel concatenated code, codec object
      exit_serial_codec, //!< serial concatenated code, codec object
      exit_serial_modem, //!< serial concatenated code, modem object
      exit_undefined
   };
   // @}

protected:
   /*! \name User-defined parameters */
   commsys<S> *sys; //!< Communication systems
   exit_t exit_type; //!< enum indicating storage mode for gamma metric
   bool compute_llr_statistics; //!< switch for computing binary LLR statistics
   double sigma; //!< Sigma value to use when generating binary priors
   // @}
   /*! \name Internally-used objects */
   libbase::randgen src; //!< Random generator for source data sequence and prior probabilities
   // @}
protected:
   /*! \name Setup functions */
   /*!
    * \brief Removes association with bound objects
    *
    * This function performs two things:
    * - Deletes any internally-allocated bound objects
    * - Sets up the system with no bound objects
    *
    * \note This function is only responsible for deleting bound
    * objects that are specific to this object/derivation.
    * Anything else should get done automatically when the base
    * serializer or constructor is called.
    */
   void free()
      {
      delete sys;
      sys = NULL;
      }
   // @}
   /*! \name Internal functions */
   array1i_t createsource();
   array1vd_t createpriors(const array1i_t& tx, const int N, const int q);
   static double compute_mutual_information(const array1i_t& x,
         const array1vd_t& y);
   static void compute_statistics(const array1i_t& x, const array1vd_t& p,
         const int value, double& sigma, double& mu);
   void compute_results(const array1i_t& x, const array1vd_t& pin,
         const array1vd_t& pout, array1d_t& result) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   /*!
    * \brief Copy constructor
    *
    * Initializes system with bound objects cloned from supplied system.
    */
   exit_computer(const exit_computer<S>& c) :
         sys(dynamic_cast<commsys<S> *>(c.sys->clone())), src(c.src)
      {
      }
   exit_computer() :
         sys(NULL)
      {
      }
   virtual ~exit_computer()
      {
      free();
      }
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r)
      {
      src.seed(r.ival());
      sys->seedfrom(r);
      }
   void set_parameter(const double x)
      {
      assertalways(x >= 0);
      sigma = x;
      }
   double get_parameter() const
      {
      return sigma;
      }

   // Experiment handling
   void sample(array1d_t& result);
   int count() const
      {
      int result = 2; // default: mutual information at input+output
      if (compute_llr_statistics)
         result += 8; // sigma+mu for each of 0+1 at input+output
      return result;
      }
   int get_multiplicity(int i) const
      {
      return 1;
      }
   std::string result_description(int i) const
      {
      assert(i >= 0 && i < count());
      switch (i)
         {
         case 0:
            return "I(input)";
         case 1:
            return "I(output)";
         case 2:
            return "𝜎(0,input)";
         case 3:
            return "𝜇(0,input)";
         case 4:
            return "𝜎(1,input)";
         case 5:
            return "𝜇(1,input)";
         case 6:
            return "𝜎(0,output)";
         case 7:
            return "𝜇(0,output)";
         case 8:
            return "𝜎(1,output)";
         case 9:
            return "𝜇(1,output)";
         }
      return ""; // This should never happen
      }
   array1i_t get_event() const
      {
      return array1i_t();
      }

   /*! \name Component object handles */
   //! Get communication system
   const commsys<S> *getsystem() const
      {
      return sys;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(exit_computer)
};

} // end namespace

#endif
