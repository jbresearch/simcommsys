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
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \warning Currently this is just a placeholder; functionality still needs
 * to be written.
 */

template <class S>
class exit_computer : public experiment_normal {
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool internallyallocated;
   libbase::randgen *src; //!< Source data sequence generator
   commsys<S> *sys; //!< Communication systems
   // @}
   /*! \name Internal state */
   libbase::vector<int> last_event;
   // @}
protected:
   /*! \name Setup functions */
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   libbase::vector<int> createsource();
   void cycleonce(libbase::vector<double>& result);
   // @}
public:
   /*! \name Constructors / Destructors */
   exit_computer(libbase::randgen *src, commsys<S> *sys);
   exit_computer(const exit_computer<S>& c);
   exit_computer()
      {
      clear();
      }
   virtual ~exit_computer()
      {
      free();
      }
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r);
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
      return 1;
      }
   int get_multiplicity(int i) const
      {
      return 1;
      }
   std::string result_description(int i) const
      {
      return "";
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
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(exit_computer)
};

} // end namespace

#endif
