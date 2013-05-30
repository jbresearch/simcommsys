#!/bin/bash
#$Id$

# top-level folders for libraries and targets
for folder in Libraries SimCommsys Steganography Test Windows; do
   echo "*** $folder"
   propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.suo$'\n'*.ncb$'\n'*cache.dat
   svn ps svn:ignore "$propval" $folder
   #svn pg svn:ignore $folder
   # individual folders for libraries and targets
   for target in $folder/*; do
      echo "*** $target"
      propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.s$'\n'*.ii$'\n'*.vcxproj.user
      svn ps svn:ignore "$propval" $target
      #svn pg svn:ignore $target
   done
done
