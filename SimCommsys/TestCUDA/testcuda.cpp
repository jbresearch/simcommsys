#include "testcuda.h"
#include "event_timer.h"

namespace testcuda {

void time_timer()
   {
   // definitions
   libbase::timer t("CPU");
   libbase::event_timer<libbase::timer> tinner("inner timer");
   const int N = int(1e5);
   // timed loop
   t.start();
   for (int i = 0; i < N; i++)
      {
      tinner.start();
      tinner.stop();
      }
   t.stop();
   // compute and show
   const double time = t.elapsed() / N;
   std::cout << "CPU timer overhead: " << libbase::timer::format(time)
         << std::endl;
   }

/*!
 * \brief   Test program for CUDA interface
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   time_timer();
#ifdef USE_CUDA
   cuda::cudaInitialize(std::cout);
   cuda::cudaQueryDevices(std::cout);
   time_kernelcalls();
   test_useofclasses();
   test_sizes();
#else
   failwith("CUDA support not enabled on this system");
#endif
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testcuda::main(argc, argv);
   }
