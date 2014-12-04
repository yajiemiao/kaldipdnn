#!/bin/bash
# Copyright 2014     Yajie Miao     Carnegie Mellon University
# Apache 2.0

# Generate DNN input features and also create txt-formatted alignment files
# It's used when PDNN directly reads Kaldi .ark and .ali files for training.

## Begin configuration section.  
stage=1
nj=4
cmd=run.pl

splice_opts="--left-context=4 --right-context=4" # frame-splicing options for neural net input
add_deltas=false
norm_vars=false  # when doing cmvn, whether to normalize variance

## End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "usage: make_nnet_data.sh <data-dir> <srcdata-dir> <feat-dir> <log-dir> <dir>"
   echo "e.g.:  make_nnet_data.sh data/nnet_input data/train exp/feat exp/_log exp/"
   echo "main options (for others, see top of script file)"
   echo "  --stage <stage>                                  # starts from which stage"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdata=$2
feadir=$3
alidir=$4
dir=$5

# get the absolute pathname
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

name=`basename $data`
sdata=$srcdata/split$nj
echo $splice_opts > $dir/splice_opts; echo $add_deltas > $dir/add_deltas; echo $norm_vars > $dir/norm_vars 

mkdir -p $dir/log $data $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# prepare the dir
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;

## First dump the network input into local files
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
if $add_deltas; then
  $cmd JOB=1:$nj $dir/log/nnet_input.$name.JOB.log \
    apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk \
      scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- \| \
      splice-feats $splice_opts ark:- ark:- \|  \
      add-deltas ark:- ark,scp:$feadir/nnet_input.$name.JOB.ark,$feadir/nnet_input.$name.JOB.scp || exit 1;

else
  $cmd JOB=1:$nj $dir/log/nnet_input.$name.JOB.log \
    apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk \
      scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- \| \
      splice-feats $splice_opts ark:- ark,scp:$feadir/nnet_input.$name.JOB.ark,$feadir/nnet_input.$name.JOB.scp || exit 1;
fi

#$cmd JOB=1:$nj $dir/log/gzip.$name.JOB.log \
#  gzip $feadir/nnet_input.$name.JOB.ark || exit 1;
 
N0=$(cat $srcdata/feats.scp | wc -l)
N1=$(cat $feadir/nnet_input.$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: error happens when generating features for $name (Original:$N0  New:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in `seq 1 $nj`; do
  cat $feadir/nnet_input.$name.$n.scp >> $data/feats.scp
done

$cmd $dir/log/ali2post.$name.log \
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz|" "ark,t:|gzip -c >$dir/$name.ali.gz" || exit 1;

echo "$0: done making features"

exit 0;
