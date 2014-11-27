#!/bin/bash
# Copyright 2013     Yajie Miao     Carnegie Mellon University
# Apache 2.0

# Create pfiles for deep neural network training. We assume that the training  alignment
# is ready and features (either fbanks and fMLLRs) have been generated.
# Refer to the following comments for configurations.

## Begin configuration section.  
stage=1
every_nth_frame=1 # for subsampling.
nj=4
cmd=run.pl

splice_opts="--left-context=4 --right-context=4" # frame-splicing options for neural net input
add_deltas=false
norm_vars=false  # when doing cmvn, whether to normalize variance

do_concat=true # whether to concatenate the individual pfiles into a single one

# Config for splitting pfile into training and valid set; not used for SWBD
do_split=false  # whether to do pfile splitting
pfile_unit_size=40 # the number of utterances of each small unit into which the whole pfile is chopped 
cv_ratio=0.05 # the ratio of CV data

shuffle_scp=false  # whether the feature scp is shuffled

## End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "usage: steps/build_nnet_pfile.sh <data-dir> <ali-dir> <exp-dir>"
   echo "e.g.:  steps/build_nnet_pfile.sh data/train exp/tri4_ali exp/tri4_pfile"
   echo "main options (for others, see top of script file)"
   echo "  --stage <stage>                                  # starts from which stage"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
alidir=$2
dir=$3

name=`basename $data`
nj=`cat $alidir/num_jobs` || exit 1;
sdata=$data/split$nj

if ! which pfile_create >/dev/null; then # pfile_create not on our path.
  [ -z "$KALDI_ROOT" ] && KALDI_ROOT=`pwd`/../../.. # normal case.
  try_path=$KALDI_ROOT/tools/pfile_utils-v0_51/bin/
  if [ -f $try_path/pfile_create ]; then
    PPATH=$try_path
  else
    echo "You do not have pfile_create (part of pfile-utils) on your path,"
    echo "and it is not accessible in the normal place e.g. $try_path/pfile_create"
    echo "Try going to KALDI_ROOT/tools/ and running ./install_pfile_utils.sh"
    exit 1
  fi
else
  PPATH=$(dirname `which pfile_create`)
fi
export PATH=$PATH:$PPATH

mkdir -p $dir/log
echo $splice_opts > $dir/splice_opts
echo $norm_vars > $dir/norm_vars
echo $add_deltas > $dir/add_deltas
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Setup features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
if $shuffle_scp; then 
  feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp \"scp:cat $sdata/JOB/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} |\" ark:- | splice-feats $splice_opts ark:- ark:- |"
fi
# Add first and second-order deltas if needed
$add_deltas && feats="$feats add-deltas ark:- ark:- |"

## Get the dimension of the features
$cmd JOB=1:1 $dir/log/get_feat_dim.log \
    feat-to-dim "$feats subset-feats --n=1 ark:- ark:- |" ark,t:$dir/feat_dim || exit 1;
feat_dim=`cat $dir/feat_dim | awk '{print $NF}'`
echo "$0: network inputs have the dimension of $feat_dim"

if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $dir/log/build_pfile.$name.JOB.log \
    build-pfile-from-ali --every-nth-frame=$every_nth_frame $alidir/final.mdl "ark:gunzip -c $alidir/ali.JOB.gz|" \
      "$feats" "|$PPATH/pfile_create -i - -o $dir/$name.pfile.JOB -f $feat_dim -l 1" || exit 1;
  # Concatenate the pfiles into one
  all_pfiles=""
  for n in `seq 1 $nj`; do
    all_pfiles="$all_pfiles $dir/$name.pfile.$n"
  done
  if $do_concat; then
    $cmd $dir/log/pfile_cat.log \
      $PPATH/pfile_concat -q $all_pfiles -o $dir/concat.pfile || exit 1;
    rm -rf $dir/$name.pfile.*
  else
    $cmd JOB=1:$nj $dir/log/gzip.$name.JOB.log gzip $dir/$name.pfile.JOB || exit 1;
  fi
fi

if [ $stage -le 3 ] && $do_split; then
  echo "Split data into training and cross-validation"
  mkdir -p $dir/concat
  # Chop the whole pfile into small units
  $cmd $dir/log/pfile_burst.log \
    perl steps_pdnn/pfile_burst.pl -i $dir/concat.pfile -o $dir/concat -s $pfile_unit_size || exit 1;
fi

if [ $stage -le 4 ] && $do_split; then
  # Split the units accoring to cv_ratio
  $cmd $dir/log/pfile_rconcat.log \
    perl steps_pdnn/pfile_rconcat.pl -t "$dir" -o $dir/valid.pfile,${cv_ratio} -o $dir/train.pfile $dir/concat/*.pfile || exit 1;
  rm -r $dir/concat
  echo "## Info of the training pfile: ##"
  $PPATH/pfile_info $dir/train.pfile
  echo "## Info of the cross-validation pfile: ##"
  $PPATH/pfile_info $dir/valid.pfile
fi

echo "$0: done creating pfiles."

exit 0;
