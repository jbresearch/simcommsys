# SimCommSys: A Modular C++ Framework for Communication System Simulation

SimCommSys is a high-performance, object-oriented C++ framework designed for the
end-to-end simulation of classical and quantum communication links. The project
was built to address the inherent risks of implementation errors in complex
simulations by enforcing a strict modular boundary between individual components
and the system-level configuration. This architecture allows researchers to
develop and swap out codecs, modems, and channel models while maintaining a
standardized environment for performance measurement.

The framework provides a mature library of error-control codes, including binary
and non-binary LDPC, turbo, Reed-Solomon, and repeat-accumulate schemes. These
are supported by a versatile set of modulators and channel models, such as AWGN,
binary and non-binary symmetric channels, and synchronization-error channels.
Beyond standard classical modeling, SimCommSys now includes GPU-accelerated
decoding for LDPC codes and synchronisation-error correcting codes, and
end-to-end simulation of CV (GG02) and DV (BB84) QKD protocols.

Users can interact with the framework as a C++ library for custom development,
via standalone binaries for direct simulation, or through a Python CLI designed
to automate common research workflows and large-scale Monte Carlo experiments
across workstation clusters.

# Using SimCommSys

SimCommSys is designed to be used at several different layers of abstraction,
depending on whether you are developing new algorithms or performing large-scale
system benchmarks:

- As a C++ Library: At its core, SimCommSys functions as an extensible library
  of classes. Developers can instantiate individual components—such as specific
  codecs, modems, or channel models—directly within their own C++ projects. This
  allows for fine-grained control and the ability to integrate SimCommSys
  components into larger, custom-built applications.
- Standalone Binaries: The project includes a suite of compiled binaries that
  implement complete end-to-end simulators or specific sub-systems. These tools
  allow for immediate simulation of standard communication links by passing
  configuration parameters directly to the executables, bypassing the need for
  additional coding when evaluating established protocols.
- Python Command-Line Interface: For improved workflow automation, the
  simcommsys-utils package
  ([public](https://github.com/jbresearch/simcommsys-utils) |
  [development](https://dsrg-ict.research.um.edu.mt/simcommsys/simcommsys-utils))
  provides a Python-based CLI that encapsulates common use cases. This interface
  streamlines the process of running batches of simulations, managing data
  output, and orchestrating complex experiments, making the power of the
  underlying C++ engine more accessible for rapid testing and analysis.

# Documentation

- User documentation can be found in the Wiki ([public](https://github.com/jbresearch/simcommsys/wiki) | [development](https://dsrg-ict.research.um.edu.mt/simcommsys/simcommsys/-/wikis)).
   This includes instructions for:
   - Building and installing the executables
   - Setting up and running simulations
   - Collecting and plotting results
- Technical documentation is divided as follows:
   - High level technical documentation can be found in the Wiki ([public](https://github.com/jbresearch/simcommsys/wiki) | [development](https://dsrg-ict.research.um.edu.mt/simcommsys/simcommsys/-/wikis)).
      This includes:
      - An overview of the SimCommSys framework
      - An example extension of the framework (adding a new codec)
   - An introduction to SimCommSys can be found in our paper:
      [SimCommSys: Taking the errors out of error-correcting code simulations](https://jabriffa.wordpress.com/2014/06/30/journal-iet-journal-of-engineering-2/)
   - Detailed documentation for the API can be built from the code:
      1. Build the doxygen documentation using `make doc`
      2. Start with the main page: `doc/html/index.html`

# Contact us

- For bug reports, use the issue tracker ([public](https://github.com/jbresearch/simcommsys/issues) | [development](https://dsrg-ict.research.um.edu.mt/simcommsys/simcommsys/-/issues)).
- [User and developer project forums](https://groups.google.com/d/forum/simcommsys)
   - Discussions about the use of simcommsys should be tagged with the 'User' category.
   - Longer discussions about simcommsys development should be tagged with the 'Developer' category.


# Copyright and license

Copyright (c) 2010-2026 Johann A. Briffa, Stephan Wesemeyer, Noel Farrugia, Aaron Abela, Mark Mizzi

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
