#!/bin/bash
# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Generate the txt formatted fbank features. This is a workaround for Kaldi lacking  
# the implementation of CNN activation.

# Begin configuration section.  
stage=1
nj=4
cmd=run.pl

input_splice_opts="--left-context=5 --right-context=5" # frame-splicing options for nnet input
norm_vars=true
add_deltas=false
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: steps/generate_txt_fbank.sh <data-dir> <log-dir> <output-dir>"
   echo "e.g.:  steps/generate_txt_fbank.sh data_fbank/train exp_pdnn/_log exp_pdnn/"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --input-splice-opts                               # how frames are spliced for DNN input"
   echo "  --norm-vars                                      # whether with variance normalization"
   echo "  --add-deltas                                     # whether deltas are added"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
dir=$3

mkdir -p $logdir
name=`basename $data`

## Setup features
echo "$0: feature type is fbank: norm_vars(${norm_vars}) add_deltas(${add_deltas})"

if [ $stage -le 1 ]; then
  if $add_deltas; then
    $cmd $logdir/txt_fbank.$name.log \
      apply-cmvn --norm-vars=${norm_vars} --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- \| \
      add-deltas ark:- ark:- \| \
      splice-feats $input_splice_opts ark:- ark,t:$dir/fbank_txt_$name.ark || exit 1;
  else
    $cmd $logdir/txt_fbank.$name.log \
      apply-cmvn --norm-vars=${norm_vars} --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- \| \
      splice-feats $input_splice_opts ark:- ark,t:$dir/fbank_txt_$name.ark || exit 1;
  fi
fi
# 

exit 0;
