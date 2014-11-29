#!/bin/bash
# Copyright 2014     Yajie Miao     Carnegie Mellon University
# Apache 2.0

# Generate activation from convolution layers in CNNs and save the activation into
# Kaldi format.

## Begin configuration section.  
stage=1
nj=4
cmd=run.pl

splice_opts=
norm_vars=
add_deltas=

layer_index=1

## End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
   echo "Wrong #arguments ($#, expected 7)"
   echo "usage: steps_pdnn/make_conv_feat.sh <data-dir> <srcdata-dir> <net-dir> "
   echo "<cnn-param-file> <cnn-cfg-file> <log-dir> <feat-dir>"
   echo "e.g.:  steps_pdnn/make_conv_feat.sh data_conv/train data/train exp/cnn "
   echo "exp/cnn/nnet.param exp/cnn/nnet.cfg exp/cnn/_log exp/cnn/_conv"
   echo "main options (for others, see top of script file)"
   echo "  --stage <stage>                                  # starts from which stage"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdata=$2
netdir=$3
cnnparam=$4
cnncfg=$5
logdir=$6
feadir=$7

# get the absolute pathname
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

name=`basename $data`
sdata=$srcdata/split$nj
[ -z "$splice_opts" ] && splice_opts=`cat $netdir/splice_opts 2>/dev/null` # frame-splicing options.
[ -z "$add_deltas" ] && add_deltas=`cat $netdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $netdir/norm_vars 2>/dev/null`

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

for f in $cnnparam $cnncfg; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# prepare the dir
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;

## First dump the network input into local files
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
if $add_deltas; then
  $cmd JOB=1:$nj $logdir/nnet_input.$name.JOB.log \
    apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk \
      scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- \| \
      splice-feats $splice_opts ark:- ark:- \|  \
      add-deltas ark:- ark,scp:$feadir/nnet_input.$name.JOB.ark,$feadir/nnet_input.$name.JOB.scp || exit 1;

else
  $cmd JOB=1:$nj $logdir/nnet_input.$name.JOB.log \
    apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk \
      scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- \| \
      splice-feats $splice_opts ark:- ark,scp:$feadir/nnet_input.$name.JOB.ark,$feadir/nnet_input.$name.JOB.scp || exit 1;
fi
 
# Generate conv-layer activation by calling PDNN
$cmd JOB=1:$nj $logdir/conv_feat.$name.JOB.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 \; \
    python pdnn/cmds2/run_FeatExt_Kaldi.py --in-scp-file $feadir/nnet_input.$name.JOB.scp --out-ark-file $feadir/conv_feats.$name.JOB.ark  --nnet-param $cnnparam --nnet-cfg $cnncfg --layer-index $layer_index
   
rm $feadir/nnet_input.*

# Generate the final scp and ark files
$cmd JOB=1:$nj $logdir/copy_feat.$name.JOB.log \
    copy-feats ark:$feadir/conv_feats.$name.JOB.ark ark,scp:$feadir/feats.$name.JOB.ark,$feadir/feats.$name.JOB.scp
rm $feadir/conv_feats.*

N0=$(cat $srcdata/feats.scp | wc -l)
N1=$(cat $feadir/feats.$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: error happens when generating features for $name (Original:$N0  New:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in `seq 1 $nj`; do
  cat $feadir/feats.$name.$n.scp >> $data/feats.scp
done

echo "$0: done making features"

exit 0;

