#!/bin/bash
# Copyright 2012-2013 Karel Vesely, Daniel Povey
# Apache 2.0.

# Create denominator lattices for MMI/MPE/sMBR training.
# Creates its output in $dir/lat.*.ark,$dir/lat.scp
# The lattices are uncompressed, we need random access for DNN training.

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
nnet=
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.
use_gpu=no # yes|no|optional
parallel_opts="-pe smp 2"

splice_opts=
norm_vars=
add_deltas=

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/$0 [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo "  e.g.: steps/$0 data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for plain features (or CMN, delta), forwarded through feature-transform."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

[ -z "$splice_opts" ] && splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`

sdata=$data/split$nj
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir

cp -r $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.
new_lang="$dir/"$(basename "$lang")
echo "Making unigram grammar FST in $new_lang"
cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile > $new_lang/G.fst \
   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

echo "Compiling decoding graph in $dir/dengraph"
if [ -s $dir/dengraph/HCLG.fst ] && [ $dir/dengraph/HCLG.fst -nt $srcdir/final.mdl ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  utils/mkgraph.sh $new_lang $srcdir $dir/dengraph || exit 1;
fi


cp $srcdir/{tree,final.mdl} $dir

# Select default locations to model files
[ -z "$nnet" ] && nnet=$srcdir/final.nnet;
class_frame_counts=$srcdir/train_class.counts
model=$dir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $nnet $model $class_frame_counts; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# PREPARE FEATURE EXTRACTION PIPELINE
# Create the feature stream:
## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
feats="$feats nnet-forward --no-softmax=true --class-frame-counts=$class_frame_counts $nnet ark:- ark:- |"


echo "$0: generating denlats from data '$data', putting lattices in '$dir'"
# Generate the lattices
$cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
  latgen-faster-mapped --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
    $dir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
echo "$0: done generating denominator lattices."
