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

#ifndef __montecarlo_h
#define __montecarlo_h

#include "config.h"

#include "walltimer.h"
#include "sha.h"
#include "experiment.h"
#include "masterslave.h"
#include "resultsfile.h"
#include "truerand.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief   Monte Carlo Estimator.
 * \author  Johann Briffa
 */

class montecarlo : public libbase::masterslave, private resultsfile {
private:
   /*! \name Bound objects */
   /*! \note If 'init' is false, and 'system' is not NULL, then there is a dynamically allocated
    * object at this address. This should be deleted when no longer necessary.
    */
   bool bound; //!< Flag to indicate that a system has been bound (only in master)
   experiment *system; //!< System being sampled
   // @}
   /*! \name Internal variables / settings */
   libbase::int32u seed; //! system initialization seed
   int min_samples; //!< minimum number of samples
   double confidence; //!< confidence level for computing margin of error
   double threshold; //!< threshold for convergence (interpretation depends on mode)
   enum mode_t {
      mode_relative_error = 0, //!< converge when error margin as a fraction of result mean is less than threshold
      mode_absolute_error, //!< converge when error margin (ie its absolute value) is less than threshold
      mode_accumulated_result, //!< converge when absolute accumulated result (ie result mean x sample count) is more than threshold
      mode_undefined
   } mode; //! mode for interpreting convergence threshold
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
         bound(false), system(NULL), min_samples(128), confidence(0.95), threshold(
               0.10), mode(mode_relative_error), t("montecarlo"), tupdate(
               "montecarlo_update")
      {
      createfunctors();
      // Use a true RNG to determine the initial seed value
      libbase::truerand trng;
      seed = trng.ival();
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
   //! Set system initialization seed
   void set_seed(libbase::int32u seed)
      {
      if(isinitialized())
         std::cerr << "WARNING (montecarlo): seed value unused in master-slave system" << std::endl;
      montecarlo::seed = seed;
      }
   //! Set minimum number of samples
   void set_min_samples(int min_samples)
      {
      assertalways(min_samples > 0);
      libbase::trace
            << "DEBUG (montecarlo): setting minimum number of samples to "
            << min_samples << std::endl;
      montecarlo::min_samples = min_samples;
      }
   //! Set confidence limit, say, 0.95 => 95% probability
   void set_confidence(double confidence)
      {
      assertalways(confidence > 0.5 && confidence < 1.0);
      libbase::trace << "DEBUG (montecarlo): setting confidence level of "
            << confidence << std::endl;
      montecarlo::confidence = confidence;
      }
   //! Set target error margin as a fraction of result mean (eg 0.10 => 10% of mean)
   void set_relative_error(double threshold)
      {
      assertalways(threshold > 0 && threshold < 1.0);
      libbase::trace
            << "DEBUG (montecarlo): setting threshold for relative error to "
            << threshold << std::endl;
      montecarlo::threshold = threshold;
      montecarlo::mode = mode_relative_error;
      }
   //! Set target error margin (as an absolute value)
   void set_absolute_error(double threshold)
      {
      assertalways(threshold > 0);
      libbase::trace
            << "DEBUG (montecarlo): setting threshold for absolute error to "
            << threshold << std::endl;
      montecarlo::threshold = threshold;
      montecarlo::mode = mode_absolute_error;
      }
   //! Set target accumulated result (ie result mean x sample count)
   void set_accumulated_result(double threshold)
      {
      assertalways(threshold > 0);
      libbase::trace
            << "DEBUG (montecarlo): setting threshold for accumulated result to "
            << threshold << std::endl;
      montecarlo::threshold = threshold;
      montecarlo::mode = mode_accumulated_result;
      }
   //! Associates with given results file
   void set_resultsfile(const std::string& fname)
      {
      resultsfile::init(fname);
      }
   //! Get confidence level as a string
   std::string get_confidence_level() const
      {
      std::ostringstream sout;
      sout << 100 * confidence << "%";
      return sout.str();
      }
   //! Get convergence mode as a string
   std::string get_convergence_mode() const
      {
      std::ostringstream sout;
      switch (mode)
         {
         case mode_relative_error:
            sout << "Margin of error within";
            sout << " ±" << 100 * threshold << "% of result";
            break;
         case mode_absolute_error:
            sout << "Margin of Error within ±" << threshold;
            break;
         case mode_accumulated_result:
            sout << "Accumulated result ≥" << threshold;
            break;
         default:
            failwith("Convergence mode not supported.");
            break;
         }
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
