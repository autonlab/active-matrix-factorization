#!/usr/bin/env zsh

mode=${1:-interactive}  # options: "interactive", "force", "abort"

cd $(dirname $0)

REV=$(git rev-parse HEAD)
if [[ $(git ls-files -md) != "" ]]; then
    case $mode in
    "interactive" )
        git status >&2
        read -q "choice?WARNING: files modified from git HEAD. Proceed? [yN] "
        echo >&2
        if [[ $choice != "y" ]]; then; exit 1; fi
        ;;
    "force" )
        ;;
    "abort" )
        echo "ERROR: files modified from git HEAD" >&2
        exit 1
        ;;
    * )
        echo "invalid mode '$mode'; giving up." >&2
        exit -1
        ;;
    esac

    REV="$REV (dirty)"
fi
echo $REV
