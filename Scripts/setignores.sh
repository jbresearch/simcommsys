#!/bin/bash
#$Id$

# root-level folder
folder="."
echo "*** $folder"
propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'Win32$'\n'x64$'\n'*.suo$'\n'*.ncb
svn ps svn:ignore "$propval" $folder
#svn pg svn:ignore $folder
# top-level folders for libraries and targets
for folder in Libraries SimCommsys Steganography Test Windows; do
   echo "*** $folder"
   propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.suo$'\n'*.ncb$'\n'*cache.dat$'\n'*.aps
   svn ps svn:ignore "$propval" $folder
   #svn pg svn:ignore $folder
   # individual folders for libraries and targets
   for target in $folder/*; do
      echo "*** $target"
      propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.s$'\n'*.ii$'\n'Win32$'\n'x64$'\n'*.vcxproj.user
      svn ps svn:ignore "$propval" $target
      #svn pg svn:ignore $target
   done
done
