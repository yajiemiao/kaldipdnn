#!/bin/bash
# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Copyright 2014    Xiaohui Zhang    Johns Hopkins University
# Apache 2.0

# Make BNF front-end with the trained neural network

# Begin configuration section.  
stage=1
nj=8
cmd=run.pl

norm_vars=false  # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: steps_pdnn/make_bnf_feat.sh <data-dir> <srcdata-dir> <nnet-dir> <log-dir> <feat-dir>"
   echo "e.g.:  steps_pdnn/make_bnf_feat.sh data_bnf/train data/train exp/bnf_net exp/make_bnf/log exp/bnf"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdata=$2
netdir=$3
logdir=$4
feadir=$5

sdata=$srcdata/split$nj;
splice_opts=`cat $netdir/splice_opts 2>/dev/null` # frame-splicing options.
name=`basename $data`

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

for f in $netdir/bnf.nnet; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
##

# prepare the dir
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;

# get the absolute pathname
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

echo "$0: making BNF scp and ark."
$cmd JOB=1:$nj $logdir/make_bnf_$name.JOB.log \
  nnet-forward --apply-log=false $netdir/bnf.nnet "$feats" \
  ark,scp:$feadir/feats_bnf_$name.JOB.ark,$feadir/feats_bnf_$name.JOB.scp || exit 1;
  

N0=$(cat $srcdata/feats.scp | wc -l) 
N1=$(cat $feadir/feats_bnf_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: error happens when generating BNF for $name (Original:$N0  BNF:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in `seq 1 $nj`; do
  cat $feadir/feats_bnf_$name.$n.scp >> $data/feats.scp
done

echo "$0: done making BNF"

exit 0;
