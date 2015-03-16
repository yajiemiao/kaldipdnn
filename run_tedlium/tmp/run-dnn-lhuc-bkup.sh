#!/bin/bash

# Copyright 2014     Yajie Miao   Carnegie Mellon University      Apache 2.0
# This is the script that trains DNN system over the filterbank features. It
# is to  be  run after run.sh. Before running this, you should already build
# the initial GMM model. This script requires a GPU card, and also the "pdnn"
# toolkit to train the DNN. The input filterbank features are with mean  and
# variance normalization.

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/dnn_fbank
gmmdir=exp/tri3

# Specify the gpu device to be used
gpu=gpu
stage=1

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "             Prepare Adaptation Data & Alignment                   "
echo =====================================================================

firstpass=$working_dir/decode_dev_bd_tgpr
dir=$working_dir/decode_dev_bd_tgpr_lhuc_V3
mkdir -p $dir/log

nj=8
echo $nj > $dir/num_jobs

if [ $stage -eq 1 ]; then

$cmd JOB=1:$nj $dir/log/best_path.JOB.log \
  lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c $firstpass/lat.JOB.gz|" ark:- \| \
  lattice-best-path ark:- ark,t:$dir/tra.JOB "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
rm -rf $dir/tra.*

cp $gmmdir/final.mdl $dir
for set in dev; do
  if [ ! -f $dir/${set}.pfile.done ]; then
    steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
      --norm-vars true --splice-opts "--left-context=5 --right-context=5" \
      $working_dir/data/$set $dir $dir || exit 1
    touch $dir/${set}.pfile.done
  fi
done

fi

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
#feat_dim=$(gunzip -c $dir/dev.pfile.1.gz |head |grep num_features| awk '{print $2}') || exit 1;

#if [ ! -f $decode_dir/dnn.fine.done ]; then
#  $train_cmd JOB=1:$nj $decode_dir/log/dnn.fine.JOB.log \
#    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn_lhuc/ \; \
#    export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 \; \
#    python pdnn_lhuc/cmds/run_DNN.py --train-data "$decode_dir/data.pfile.JOB.gz,partition=2000m,random=true,stream=true" \
#                                    --valid-data "$decode_dir/data.pfile.JOB.gz,partition=600m,random=true,stream=true" \
#                                    --nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
#                                    --ptr-file $working_dir/nnet.param --ptr-layer-number 7 \
#                                    --lrate "C:0.8:3" \
#                                    --wdir $decode_dir --kaldi-output-file $decode_dir/dnn.nnet.JOB || exit 1;
#  touch $working_dir/dnn.fine.done
#fi

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
if [ $stage -eq 2 ]; then
  graph_dir=$gmmdir/graph_bd_tgpr
  steps_pdnn/tmp/decode_dnn_lhuc.sh --nj 8 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    $graph_dir $working_dir/data/dev ${gmmdir}_ali_tr95 $dir || exit 1;
fi

echo "Finish !!"
