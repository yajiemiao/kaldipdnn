#!/bin/bash

# Copyright 2014     Yajie Miao   Carnegie Mellon University      Apache 2.0
# This is the script that trains DNN system over the filterbank features. It
# is to  be  run after run.sh. Before running this, you should already build
# the initial GMM model. This script requires a GPU card, and also the "pdnn"
# toolkit to train the DNN. The input filterbank features are with mean  and
# variance normalization.

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/dnn_fbank_sat
gmmdir=exp/tri3
dnndir=exp_pdnn/dnn_fbank

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

#ivec_dir="/data/ASR5/babel/ymiao/Install/kaldi-latest/egs/sre08/v1/exp_tedlium_V2/ivectors_devtest"
ivec_dir="/data/ASR5/babel/ymiao/Install/kaldi-latest/egs/sre08/v1/exp_tedlium_bnf/ivectors_devtest"

if [ $stage -eq 1 ]; then

  for set in dev test; do
    dir=$working_dir/decode_${set}_bd_tgpr_lhuc
    mkdir -p $dir/log

    case $set in
      dev) nj=8;;
      test) nj=11;;
      *) echo "$0: invalid set name $set" && exit 1;
    esac

    echo $nj > $dir/num_jobs

    steps_pdnn/sat/make_feat_with_ivec.sh --nj $nj --cmd "$train_cmd" --ivec-type speaker \
      $working_dir/data_ivec/$set $working_dir/data/$set $working_dir $ivec_dir $working_dir/_log $working_dir/_ivec || exit 1;
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_ivec/$set $working_dir/_log $working_dir/_ivec || exit 1;

#    $cmd JOB=1:$nj $dir/log/best_path.JOB.log \
#      lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c $working_dir/decode_${set}_bd_tgpr/lat.JOB.gz|" ark:- \| \
#      lattice-best-path ark:- ark,t:$dir/tra.JOB "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

#    $cmd JOB=1:$nj $dir/log/best_path.JOB.log \
#      lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c exp_pdnn/dnn_fbank/decode_${set}_bd_tgpr/lat.JOB.gz|" ark:- \| \
#      lattice-best-path ark:- ark,t:$dir/tra.JOB "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

    rm -rf $dir/tra.*

    cp $gmmdir/final.mdl $dir
    if [ ! -f $dir/${set}.pfile.done ]; then
      steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
        --norm-vars false --splice-opts "--left-context=0 --right-context=0" \
        $working_dir/data_ivec/$set $dir $dir || exit 1
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
  steps_pdnn/tmp/decode_dnn_lhuc.sh --nj 8 --scoring-opts "--min-lmwt 8 --max-lmwt 12" --cmd "$decode_cmd" \
    --norm-vars false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_ivec/dev ${dnndir}_ali_tr95 $working_dir/decode_dev_bd_tgpr_lhuc || exit 1;
  steps_pdnn/tmp/decode_dnn_lhuc.sh --nj 11 --scoring-opts "--min-lmwt 8 --max-lmwt 12" --cmd "$decode_cmd" \
    --norm-vars false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_ivec/test ${dnndir}_ali_tr95 $working_dir/decode_test_bd_tgpr_lhuc || exit 1;

#  steps_pdnn/tmp/decode_dnn_lhuc.sh --nj 8 --scoring-opts "--min-lmwt 8 --max-lmwt 12" --cmd "$decode_cmd" \
#    --norm-vars false --splice-opts "--left-context=0 --right-context=0" \
#    $graph_dir $working_dir/data_ivec/dev ${gmmdir}_ali_tr95 $working_dir/decode_dev_bd_tgpr_lhuc || exit 1;
#  steps_pdnn/tmp/decode_dnn_lhuc.sh --nj 11 --scoring-opts "--min-lmwt 8 --max-lmwt 12" --cmd "$decode_cmd" \
#    --norm-vars false --splice-opts "--left-context=0 --right-context=0" \
#    $graph_dir $working_dir/data_ivec/test ${gmmdir}_ali_tr95 $working_dir/decode_test_bd_tgpr_lhuc || exit 1;

fi

echo "Finish !!"
