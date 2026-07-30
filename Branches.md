# Branches

This document contains a summary of significant branches in the development
version of simcommsys. It is meant to assist with the process of code reviews
and the integration of work into the main branch.

## Main development timeline

This is the main timeline, and represents what will in due course be published.

- master
  - tip of development repo
  - based on v1.2.3
- blessed/master
  - tip of public repo
  - v2.0.0-rc1

## Current development branches

These are development branches that implement some new feature. They will in due
course be merged into the main timeline.

- feat/add-sparc
  - Started by Mark, to implement SPARC codes
  - Rebased on master-candidate

## Temporary / abandoned development work

These branches were created to test/implement some new feature, but were either
abandoned or have not seen activity in a long time. To decide whether to keep
any material, and merge this in the main timeline, or delete permanently. The
branches are generally listed as most-recent first, and are organised by the
main contributor for the set of branches.

- Trevor
  - personal/ts/qkd-fedora
    - Created by Trevor
    - used by the PRISM Rust code - if this branch is integrated, the Rust
      interface will need to be updated accordingly
    - based on v1.2.3
- Mark
  - hacks/mmiz/remove-ldpc-encoder
    - hack replacing the LDPC encoder with a random sequence generator, in order
      to allow timing simulations of the decoder with larger block sizes
    - to confirm there is nothing of value, and delete
    - based on v1.2.3, but shares history with other branches from Mark
  - feat/more-efficient-gf-storage
    - reduces the integer size to the smallest required (rather than always
      int32)
    - based on v1.2.3, but shares history with other branches from Mark
  - personal/mmiz/gdl-cuda-normalize-probs-kern-opt
    - shared mem optimization on normalisation kernel
    - to check with Mark, likely this is defunct and can be removed
    - based on v1.2.3, but shares history with other branches from Mark
- Noel - postdoc work
  - personal/aab/qkd_commsys_simulator
    - from Noel's attempt to create a QKD setup
    - rebased on blessed/master
  - personal/nf/qkd_commsys_simulator
    - original attempt to create a QKD setup
    - based on v1.2.2
  - personal/nf/commsys_refactor_old
    - aborted work on refactoring commsys
    - to check with Noel, likely to delete
    - based on v1.2.2
- Noel - MSc work
  - personal/md/dissertation
    - matthias's work
    - based on personal/vb/convolutional
  - personal/vb/convolutional
    - edits by victor to make branch work on current systems
    - based on personal/jab/nf-msc
  - personal/jab/nf-msc
    - main derivative of noel's work
    - added some fixes
    - based on nf-msc
  - personal/jab/nf
    - slight divergence from nf-msc
    - based on nf-msc
  - [nf-msc]
    - tagged version of noel's MSc work
    - introduces decoding of convolutional codes on syncrhonisation error
      channels
    - based on v1.0.0
