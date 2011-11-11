#!/bin/bash
#$Id$

# top-level folders for libraries and targets
for folder in Libraries SimCommsys Steganography Windows; do
   echo "*** $folder"
   propval=debug$'\n'release$'\n'profile$'\n'*.suo$'\n'*.ncb$'\n'*cache.dat
   svn ps svn:ignore "$propval" $folder
   #svn pg svn:ignore $folder
   # individual folders for libraries and targets
   for target in $folder/*; do
      echo "*** $target"
      propval=debug$'\n'release$'\n'profile$'\n'*.vcproj.*.user
      svn ps svn:ignore "$propval" $target
      #svn pg svn:ignore $target
   done
done
