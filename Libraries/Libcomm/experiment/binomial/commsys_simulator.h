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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __commsys_simulator_h
#define __commsys_simulator_h

#include "config.h"
#include "experiment/binomial/result_collector/commsys_errorrates.h"
#include "experiment/experiment_binomial.h"
#include "randgen.h"
#include "commsys.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Communication Systems Simulator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \todo Clean up interface with commsys object, particularly in cycleonce()
 */

template <class S, class R = commsys_errorrates>
class commsys_simulator : public experiment_binomial, public R {
protected:
   /*! \name Bound objects */
   libbase::randgen src; //!< Source data sequence generator
   commsys<S> *sys; //!< Communication systems
   // @}
   /*! \name Internal state */
   libbase::vector<int> last_event;
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
      // note: delete can be safely called with null pointers
      delete sys;
      sys = NULL;
      }
   // @}
   /*! \name Internal functions */
   libbase::vector<int> createsource();
   // @}
   // System Interface for Results
   int get_iter() const
      {
      return sys->num_iter();
      }
   int get_symbolsperblock() const
      {
      return sys->input_block_size();
      }
   int get_alphabetsize() const
      {
      return sys->num_inputs();
      }
public:
   /*! \name Constructors / Destructors */
   /*!
    * \brief Copy constructor
    *
    * Initializes system with bound objects cloned from supplied system.
    */
   commsys_simulator(const commsys_simulator<S, R>& c) :
      src(c.src)
      {
      this->sys = dynamic_cast<commsys<S> *> (c.sys->clone());
      }
   commsys_simulator() :
      sys(NULL)
      {
      }
   virtual ~commsys_simulator()
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
      sys->getchan()->set_parameter(x);
      }
   double get_parameter() const
      {
      return sys->getchan()->get_parameter();
      }

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const
      {
      return R::count();
      }
   int get_multiplicity(int i) const
      {
      return R::get_multiplicity(i);
      }
   std::string result_description(int i) const
      {
      return R::result_description(i);
      }
   libbase::vector<int> get_event() const
      {
      return last_event;
      }

   /*! \name Component object handles */
   //! Get communication system
   const commsys<S> *getsystem() const
      {
      return sys;
      }
   //! Clear list of timers
   void reset_timers()
      {
      sys->reset_timers();
      }
   //! Get the list of timings taken
   std::vector<double> get_timings() const
      {
      return sys->get_timings();
      }
   //! Get the list of friendly names for timings taken
   std::vector<std::string> get_names() const
      {
      return sys->get_names();
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(commsys_simulator)
};

} // end namespace

#endif
