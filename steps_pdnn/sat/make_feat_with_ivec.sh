#!/bin/bash

# Copyright 2014    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the SAT-DNN model. You should already have the canonical DNN model
# and the iVecNN network in srcdir.

## Begin configuration section
stage=0
nj=16
cmd=run.pl
num_threads=1

splice_opts=
norm_vars=
add_deltas=
ivec_type=speaker         # the type of the i-vectors: speaker, utterance, frame

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "Usage: steps/decode_dnn.sh [options] <graph-dir> <data-dir> <ali-dir> <iv-dir> <decode-dir>"
   echo " e.g.: steps/decode_dnn.sh exp/tri4/graph data/test exp/tri4_ali exp_ivec/ivector_eval2000 exp/tri4_dnn/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
srcdata=$2
netdir=$3
ivdir=$4
logdir=$5
feadir=$6

name=`basename $data`
sdata=$srcdata/split$nj;
[ -z "$splice_opts" ] && splice_opts=`cat $netdir/splice_opts 2>/dev/null` # frame-splicing options.
[ -z "$add_deltas" ] && add_deltas=`cat $netdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $netdir/norm_vars 2>/dev/null`

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

for f in $netdir/dnn.nnet.adapt; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"

# Setup the additional ivector features
case $ivec_type in
  speaker) ivfeats="ark,s,cs:get-spkvec-feat --utt2spk=ark:$sdata/JOB/utt2spk scp:$ivdir/ivector.scp scp:$sdata/JOB/feats.scp ark:- |";;
  utterance) ivfeats="ark,s,cs:get-spkvec-feat scp:$ivdir/ivector.scp scp:$sdata/JOB/feats.scp ark:- |";;
  frame)  ivfeats="ark:copy-feats $ivdir/ivector.scp ark:- |";;
  *) echo "$0: invalid ivector type $ivec_type" && exit 1;
esac
ivfeats="$ivfeats nnet-forward $netdir/dnn.nnet.adapt ark:- ark:- |"

# prepare the dir
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;

# get the absolute pathname
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

# Add the linear feature shifts to the original DNN inputs
echo "$0: making feature + [ivector shift] scp and ark."
$cmd JOB=1:$nj $logdir/add_feat_$name.JOB.log \
     add-feats "$feats" "$ivfeats" ark,scp:$feadir/add_feat.$name.JOB.ark,$feadir/add_feat.$name.JOB.scp || exit 1;

N0=$(cat $srcdata/feats.scp | wc -l)
N1=$(cat $feadir/add_feat.$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: error happens when generating feature + [ivector shift] for $name (Original:$N0  Now:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in `seq 1 $nj`; do
  cat $feadir/add_feat.$name.$n.scp >> $data/feats.scp
done

echo "$0: done making feature + [ivector shift]"

exit 0;
