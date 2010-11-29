#include "serializer_libcomm.h"
#include "fsm.h"

#include <iostream>

namespace testfsm {

/*!
 * \brief   Test program for FSM objects
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * Serializes a FSM object from standard input; computes and displays
 * state table.
 */

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   const libcomm::serializer_libcomm my_serializer_libcomm;

   // get a new fsm from stdin
   cerr << "Enter FSM details:" << std::endl;
   libcomm::fsm *encoder;
   cin >> encoder;
   cout << encoder->description() << std::endl;

   // compute and display state table
   cout << "PS\tIn\tOut\tNS" << std::endl;
   for (int ps = 0; ps < encoder->num_states(); ps++)
      for (int in = 0; in < encoder->num_input_combinations(); in++)
         {
         cout << ps << '\t';
         cout << in << '\t';
         encoder->reset(encoder->convert_state(ps));
         libbase::vector<int> ip = encoder->convert_input(in);
         cout << encoder->convert_output(encoder->step(ip)) << '\t';
         cout << encoder->convert_state(encoder->state()) << std::endl;
         }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testfsm::main(argc, argv);
   }
