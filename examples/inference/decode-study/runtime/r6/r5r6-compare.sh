# r5r6-compare.sh: compare every r6 library and tool with r5's, raw and after replacing the build directory
# name (r6-src -> r5-src, same length) and removing the .note.gnu.build-id section from both copies.
set -eu
R=$HOME/.local/share/SpiralPixcel/runtimes; R5=$R/llama-b11960-pvr-decode-r5; R6=$R/llama-b11960-pvr-decode-r6
T=$(mktemp -d); trap 'rm -rf $T' EXIT
first=1; printf '{"normalization": "LC_ALL=C sed s/r6-src/r5-src/g on the r6 file; objcopy --remove-section .note.gnu.build-id on both", "files": ['
for f in $(cd $R6 && ls *.so* llama-bench llama-completion check-readback check-mmv check-large check-model 2>/dev/null | sort -u); do
  [ -L $R6/$f ] && continue
  LC_ALL=C sed "s/r6-src/r5-src/g" $R6/$f > $T/6; cp $R5/$f $T/5
  objcopy --remove-section .note.gnu.build-id $T/6 2>/dev/null || true; objcopy --remove-section .note.gnu.build-id $T/5 2>/dev/null || true
  cmp -s $T/6 $T/5 && same=true || same=false
  [ $first = 1 ] || printf ', '; first=0
  printf '{"file": "%s", "r5_sha256": "%s", "r6_sha256": "%s", "r5_bytes": %s, "r6_bytes": %s, "identical_after_normalization": %s}' \
    $f $(sha256sum < $R5/$f | cut -c1-64) $(sha256sum < $R6/$f | cut -c1-64) $(stat -c %s $R5/$f) $(stat -c %s $R6/$f) $same
done
printf ']}\n'
