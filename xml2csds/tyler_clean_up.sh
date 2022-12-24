sed -i 's/\x0c//g' CMU/*.xml
sed -E -i 's/&lt;\/?LU_ANNOTATE.*>/<\!-- & -->/' CMU/*.xml