#!/bin/bash

# root-level folder
folder="."
echo "*** $folder"
propval=doc$'\n'[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'Win32$'\n'x64$'\n'*.suo$'\n'*.ncb
echo "$propval" > $folder/.gitignore
# top-level folders for libraries and targets
for folder in Libraries SimCommsys Steganography Test Windows; do
   echo "*** $folder"
   propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.suo$'\n'*.ncb$'\n'*cache.dat$'\n'*.aps
   echo "$propval" > $folder/.gitignore
   # individual folders for libraries and targets
   for target in $folder/*; do
      echo "*** $target"
      propval=[Dd]ebug$'\n'[Rr]elease$'\n'[Pp]rofile$'\n'*.s$'\n'*.ii$'\n'Win32$'\n'x64$'\n'*.vcxproj.user
      echo "$propval" > $target/.gitignore
   done
done
