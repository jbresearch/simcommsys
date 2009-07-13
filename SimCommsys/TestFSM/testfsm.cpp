#include "serializer_libcomm.h"
#include "fsm.h"

#include <iostream>

namespace testfsm {

/*!
 \brief   Test program for FSM objects
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 Serializes a FSM object from standard input; computes and displays
 state table.
 */

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   const libcomm::serializer_libcomm my_serializer_libcomm;

   // get a new fsm from stdin
   cerr << "Enter FSM details:\n";
   libcomm::fsm *encoder;
   cin >> encoder;
   cout << encoder->description() << "\n";

   // compute and display state table
   cout << "PS\tIn\tOut\tNS\n";
   for (int ps = 0; ps < encoder->num_states(); ps++)
      for (int in = 0; in < encoder->num_inputs(); in++)
         {
         cout << ps << '\t';
         cout << in << '\t';
         encoder->reset(ps);
         cout << encoder->step(in) << '\t';
         cout << encoder->state() << '\n';
         }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testfsm::main(argc, argv);
   }
