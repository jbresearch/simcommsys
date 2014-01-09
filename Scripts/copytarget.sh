#!/bin/bash

if [[ $# < 2 ]]; then
   echo "Usage: $0 <source/template> <target>"
   exit
fi

# copy user parameters
srcpath=$1
dstpath=$2
# determine source/target folder names
src=`basename $srcpath`
dst=`basename $dstpath`
# determine source/target main source file names
srcfile=`echo $src.cpp |tr 'A-Z' 'a-z'`
dstfile=`echo $dst.cpp |tr 'A-Z' 'a-z'`

# create folder & set properties
mkdir -p "$dstpath"
svn add "$dstpath"
props=`svn pg svn:ignore "$srcpath"`
svn ps svn:ignore "$props" "$dstpath"

# copy target file
svn cp "$srcpath/$srcfile" "$dstpath/$dstfile"

# copy command-line makefile
svn cp "$srcpath/Makefile" "$dstpath"

# copy and update eclipse project files
svn cp "$srcpath/.project" "$dstpath"
svn cp "$srcpath/.cproject" "$dstpath"
sed -i "s/$src/$dst/" "$dstpath/.project"
sed -i "s/$src/$dst/" "$dstpath/.cproject"

# copy and update visual studio project file
svn cp "$srcpath/$src.vcproj" "$dstpath/$dst.vcproj"
sed -i "/ProjectGUID/d" "$dstpath/$dst.vcproj"
sed -i "s/$src/$dst/" "$dstpath/$dst.vcproj"
sed -i "s/$srcfile/$dstfile/" "$dstpath/$dst.vcproj"
