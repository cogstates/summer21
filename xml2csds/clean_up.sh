# Commands to clean up a corpus.

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1# Commands to clean up a corpus.

# sed -i 's/\x0c//g' CMU/*.xml
# sed -E -i 's/&lt;\/?LU_ANNOTATE.*>/<\!-- & -->/' CMU/*.xml

LC_ALL=C sed -i '' $'s/\x0c//g' $DIR/*.xml
LC_ALL=C sed -E -i '' 's/&lt;\/?LU_ANNOTATE.*>/<\!-- & -->/' $DIR/*.xml

