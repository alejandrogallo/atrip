#!/usr/bin/env bash
# Copyright (C) 2022 by Alejandro Gallo <aamsgallo@gmail.com>

set -eu

flags=("${@}")
PROJECTS=()

#
## Check root directory
#
root_project=$(git rev-parse --show-toplevel)
configure=$root_project/configure
if [[ $(basename $PWD) == $(basename $root_project) ]]; then
    cat <<EOF

 You are trying to build in the root directory, create a build folder
 and then configure.

                             mkdir build
                             cd build
                             $(readlink -f $0)

EOF
    exit 1
fi

[[ -f $configure ]] || {
    cat <<EOF
  No configure script at $configure create it with bootstrap.sh or

    autoreconf -vif

EOF
    exit 1
}

#
## Create configuration function
#

create_config () {
    file=$1
    name=$2
    PROJECTS=(${PROJECTS[@]} "$name")
    mkdir -p $name
    cd $name
    echo "> creating: $name"
    cat <<SH > configure
#!/usr/bin/env bash
# created by $0 on $(date)

$root_project/configure $(cat $file | paste -s) \\
$(for word in "${flags[@]}"; do
    printf "  \"%s\"" "$word";
  done)


exit 0
SH
    chmod +x configure
    cd - > /dev/null
}

#
## default configuration
#

tmp=`mktemp`
cat <<EOF > $tmp
--disable-slice
EOF

create_config $tmp default
rm $tmp

#
## only-dgemm configuration
#

tmp=`mktemp`
cat <<EOF > $tmp
--disable-slice
--enable-only-dgemm
EOF

create_config $tmp only-dgemm
rm $tmp

#
## Create makefile
#

cat <<MAKE > Makefile

all: configure do
do: configure

configure: ${PROJECTS[@]/%/\/Makefile}

%/Makefile: %/configure
	cd \$* && ./configure

do: ${PROJECTS[@]/%/\/src\/libatrip.a}


%/src/libatrip.a:
	cd \$* && \$(MAKE)


.PHONY: configure do all
MAKE

cat <<EOF

Now you can do

  make all

or go into one of the directories
   ${PROJECTS[@]}
and do
  ./configure
  make

EOF

## Emacs stuff
# Local Variables:
# eval: (outline-minor-mode)
# outline-regexp: "## "
# End:
