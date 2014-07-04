# SimCommSys

The SimCommSys project consists of a set of C++ libraries for simulating
communication systems using a distributed Monte Carlo simulator.
Of principal interest is the error-control coding component, where various
kinds of binary and non-binary codes are implemented, including turbo, LDPC,
repeat-accumulate, and Reed-Solomon.
This project also contains a number of executables implementing various
stages of the communication system (such as the encoder, decoder, and a
complete simulator) and a system benchmark.
Finally, a number of shell and python scripts are provided to encapsulate
routine use cases.

Documentation is available as follows:
- User documentation can be found in the [Wiki](https://github.com/jbresearch/simcommsys/wiki).
   This includes instructions for:
   - Building and installing the executables
   - Setting up and running simulations
   - Collecting and plotting results
- Technical documentation is divided as follows:
   - High level technical documentation can be found in the [Wiki](https://github.com/jbresearch/simcommsys/wiki).
      This includes:
      - An overview of the SimCommSys framework
      - An example extension of the framework (adding a new codec)
   - An introduction to SimCommSys can be found in our paper:
      [SimCommSys: Taking the errors out of error-correcting code simulations](http://jabriffa.wordpress.com/publications/#simcommsys)
   - Detailed documentation for the API can be built from the code:
      1. Build the doxygen documentation using `make doc`
      2. Start with the main page: `doc/html/index.html`

To contact us:
- For bug reports, use the [issue tracker](https://github.com/jbresearch/simcommsys/issues)
- [User and developer project forums](https://groups.google.com/d/forum/simcommsys)
   - Discussions about the use of simcommsys should be tagged with the 'User' category.
   - Longer discussions about simcommsys development should be tagged with the 'Developer' category.


# Copyright and license

Copyright (c) 2010-2013 Johann A. Briffa

This file is part of SimCommSys.

SimCommSys is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SimCommSys is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
