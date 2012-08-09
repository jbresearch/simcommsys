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

#ifndef __montecarlo_h
#define __montecarlo_h

#include "config.h"

#include "walltimer.h"
#include "sha.h"
#include "experiment.h"
#include "masterslave.h"
#include "resultsfile.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief   Monte Carlo Estimator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class montecarlo : public libbase::masterslave, private resultsfile {
private:
   /*! \name Object-wide constants */
   static const libbase::int64u min_samples; //!< minimum number of samples
   // @}
   /*! \name Bound objects */
   /*! \note If 'init' is false, and 'system' is not NULL, then there is a dynamically allocated
    * object at this address. This should be deleted when no longer necessary.
    */
   bool bound; //!< Flag to indicate that a system has been bound (only in master)
   experiment *system; //!< System being sampled
   // @}
   /*! \name Internal variables */
   double confidence; //!< confidence level required
   double accuracy; //!< accuracy level required (margin of error must be less than this)
   bool absolute; //!< flag indicating that accuracy is an absolute value
   libbase::walltimer t; //!< timer to keep track of running estimate
   mutable libbase::walltimer tupdate; //!< timer to keep track of display rate
   sha sysdigest; //!< digest of the currently-simulated system
   // @}
private:
   /*! \name Slave process functions & their functors */
   void slave_getcode(void);
   void slave_getparameter(void);
   void slave_work(void);
   libbase::specificfunctor<montecarlo> *fgetcode;
   libbase::specificfunctor<montecarlo> *fgetparameter;
   libbase::specificfunctor<montecarlo> *fwork;
   // @}
private:
   /*! \name Helper functions */
   std::string get_systemstring();
   void seed_experiment();
   void createfunctors(void);
   void destroyfunctors(void);
   // @}
   /*! \name Main estimator helper functions */
   /*!
    * \brief Compute a single sample and accumulate results
    */
   void sampleandaccumulate()
      {
      libbase::vector<double> result;
      system->sample(result);
      system->accumulate(result);
      }
   void updateresults(libbase::vector<double>& result,
         libbase::vector<double>& errormargin) const;
   void initslave(slave *s, std::string systemstring);
   void initnewslaves(std::string systemstring);
   void workidleslaves(bool converged);
   bool readpendingslaves();
   // @}
protected:
   // System-specific file-handler functions
   void writeheader(std::ostream& sout) const;
   void writeresults(std::ostream& sout, libbase::vector<double>& result,
         libbase::vector<double>& errormargin) const;
   void writestate(std::ostream& sout) const;
   void lookforstate(std::istream& sin);
   /*! \name Overrideable user-interface functions */
   /*! \brief User-interrupt check
    * This function should return true if the user has requested an interrupt.
    * Once it returns true, all subsequent evaluations should keep returning
    * true again. Default action is to check for user pressing 'q' or Ctrl-C.
    */
   virtual bool interrupt()
      {
      static bool interrupted = false;
      if (interrupted)
         return true;
      if (libbase::interrupted())
         interrupted = true;
      else if (libbase::keypressed() > 0)
         interrupted = (libbase::readkey() == 'q');
      return interrupted;
      }
   virtual void display(const libbase::vector<double>& result,
         const libbase::vector<double>& errormargin) const;
   // @}
public:
   /*! \name Constructor/destructor */
   montecarlo() :
      bound(false), system(NULL), confidence(0.95), accuracy(0.10), absolute(
            false), t("montecarlo"), tupdate("montecarlo_update")
      {
      createfunctors();
      }
   virtual ~montecarlo()
      {
      release();
      delete system;
      destroyfunctors();
      tupdate.stop();
      }
   // @}
   /*! \name Simulation binding/releasing */
   void bind(experiment *system)
      {
      release();
      assert(montecarlo::system == NULL);
      bound = true;
      montecarlo::system = system;
      }
   void release()
      {
      if (!bound)
         return;
      assert(system != NULL);
      bound = false;
      system = NULL;
      }
   // @}
   /*! \name Simulation parameters */
   //! Set confidence limit, say, 0.95 => 95% probability
   void set_confidence(double confidence)
      {
      assertalways(confidence > 0.5 && confidence < 1.0);
      libbase::trace << "DEBUG (montecarlo): setting confidence level of "
            << confidence << std::endl;
      montecarlo::confidence = confidence;
      }
   //! Set target accuracy, say, 0.10 => 10% of mean
   void set_accuracy(double accuracy)
      {
      assertalways(accuracy > 0 && accuracy < 1.0);
      libbase::trace << "DEBUG (montecarlo): setting accuracy level of "
            << accuracy << std::endl;
      montecarlo::accuracy = accuracy;
      montecarlo::absolute = false;
      }
   //! Set target margin of error (as an absolute value)
   void set_errormargin(double errormargin)
      {
      assertalways(errormargin > 0);
      libbase::trace << "DEBUG (montecarlo): setting margin of error of "
            << errormargin << std::endl;
      montecarlo::accuracy = errormargin;
      montecarlo::absolute = true;
      }
   //! Associates with given results file
   void set_resultsfile(const std::string& fname)
      {
      resultsfile::init(fname);
      }
   //! Get confidence interval as a string
   std::string get_confidence_interval() const
      {
      std::ostringstream sout;
      if (absolute)
         sout << "±" << accuracy;
      else
         sout << "±" << 100 * accuracy << "%";
      sout << " @ " << 100 * confidence << "%";
      return sout.str();
      }
   // @}
   /*! \name Simulation results */
   //! Number of samples taken to produce the result
   libbase::int64u get_samplecount() const
      {
      return system->get_samplecount();
      }
   //! Time taken to produce the result
   const libbase::timer& get_timer() const
      {
      return t;
      }
   // @}
   /*! \name Main process */
   void estimate(libbase::vector<double>& result,
         libbase::vector<double>& errormargin);
   // @}
};

} // end namespace

#endif
