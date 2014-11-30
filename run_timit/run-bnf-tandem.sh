#!/bin/bash

# Copyright 2014    Yajie Miao  Carnegie Mellon University      Apache 2.0
# This script  trains tandem systems using bottleneck features (BNFs). The 
# BNF network is trained over fMLLR features. It is to be run after run.sh.
# Before running this, you should already build the initial GMM model. This
# script requires a  GPU, and also the "pdnn" toolkit to train the BNF net.

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/bnf_tandem
gmmdir=exp/tri3

# Specify the gpu device to be used
gpu=gpu

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

# At this point you may want to make sure the directory $working_dir is
# somewhere with a lot of space, preferably on the local GPU-containing machine.
if [ ! -d pdnn ]; then
  echo "Checking out PDNN code."
  svn co https://github.com/yajiemiao/pdnn/trunk pdnn
fi

if [ ! -d steps_pdnn ]; then
  echo "Checking out steps_pdnn scripts."
  svn co https://github.com/yajiemiao/kaldipdnn/trunk/steps_pdnn steps_pdnn
fi

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  echo "(Note: this script might still work, it would just be slower.)"
fi

# The hope here is that Theano has been installed either to python or to python2.6
pythonCMD=python
if ! python -c 'import theano;'; then
  if ! python2.6 -c 'import theano;'; then
    echo "Theano does not seem to be installed on your machine.  Not continuing."
    echo "(Note: this script might still work, it would just be slower.)"
    exit 1;
  else
    pythonCMD=python2.6
  fi
fi

mkdir -p $working_dir/log

! gmm-info $gmmdir/final.mdl >&/dev/null && \
   echo "Error getting GMM info from $gmmdir/final.mdl" && exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "           Data Split & Alignment & Feature Preparation            "
echo =====================================================================
# Split training data into traing and cross-validation sets for DNN
if [ ! -d data/train_tr95 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train data/train_tr95 data/train_cv05 || exit 1
fi
# Alignment on the training and validation data
for set in tr95 cv05; do
  if [ ! -d ${gmmdir}_ali_$set ]; then
    steps/align_fmllr.sh --nj 16 --cmd "$train_cmd" \
      data/train_$set data/lang $gmmdir ${gmmdir}_ali_$set || exit 1
  fi
done

# Dump fMLLR features. "Fake" cmvn states (0 means and 1 variance) are applied. 
for set in tr95 cv05; do
  if [ ! -d $working_dir/data/train_$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 16 --cmd "$train_cmd" \
      --transform-dir ${gmmdir}_ali_$set \
      $working_dir/data/train_$set data/train_$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/train_$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done
for set in dev test; do
  if [ ! -d $working_dir/data/$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
      --transform-dir $gmmdir/decode_$set \
      $working_dir/data/$set data/$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, DNN inputs include 11 frames of fMLLR
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --norm-vars false \
      --splice-opts "--left-context=5 --right-context=5" \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir || exit 1
    ( cd $working_dir; mv concat.pfile ${set}.pfile; gzip ${set}.pfile; )
    touch $working_dir/${set}.pfile.done
  fi
done
# Rename pfiles to keep consistency
( cd $working_dir;
  ln -s tr95.pfile.gz train.pfile.gz; ln -s cv05.pfile.gz valid.pfile.gz
)

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
feat_dim=$(gunzip -c $working_dir/train.pfile.gz |head |grep num_features| awk '{print $2}') || exit 1;

if [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "RBM Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_RBM.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
                               --nnet-spec "$feat_dim:1024:1024:1024:42:1024:$num_pdfs" --wdir $working_dir \
                               --ptr-layer-number 5 --param-output-file $working_dir/dnn.ptr || exit 1;
  touch $working_dir/dnn.ptr.done
fi

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_DNN.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
                          --valid-data "$working_dir/valid.pfile.gz,partition=200m,random=true,stream=false" \
                          --nnet-spec "$feat_dim:1024:1024:1024:42:1024:$num_pdfs" \
                          --ptr-file $working_dir/dnn.ptr --ptr-layer-number 5 \
                          --lrate "D:0.08:0.5:0.2,0.2:8" \
                          --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
fi

( cd $working_dir; ln -s dnn.nnet bnf.nnet )

echo =====================================================================
echo "                    BNF Feature Generation                         "
echo =====================================================================
# Combine fMLLRs of train_* sets into train
if [ ! -d $working_dir/data/train ]; then
  utils/combine_data.sh $working_dir/data/train $working_dir/data/train_tr95 $working_dir/data/train_cv05
fi
# Dump BNF features
for set in train dev test; do
  if [ ! -d $working_dir/data_bnf/${set} ]; then
    steps_pdnn/make_bnf_feat.sh --nj 16 --cmd "$train_cmd"  \
      $working_dir/data_bnf/${set} $working_dir/data/${set} $working_dir $working_dir/_log $working_dir/_bnf || exit 1
    # We will normalize BNF features, thus are not providing --fake here. Intuitively, apply CMN over BNF features 
    # might be redundant. But our experiments on WSJ show gains by doing this.
    steps/compute_cmvn_stats.sh \
      $working_dir/data_bnf/${set} $working_dir/_log $working_dir/_bnf || exit 1;
  fi
done

# Redirect datadir pointing to the BNF dir
datadir=$working_dir/data_bnf

echo =====================================================================
echo "                    LDA+MLLT Systems over BNFs                     "
echo =====================================================================
decode_param="--beam 15.0 --lattice-beam 7.0 --acwt 0.125" # decoding parameters differ from MFCC systems
scoring_opts="--min-lmwt 7 --max-lmwt 14"
denlats_param="--acwt 0.1"                            # Parameters for lattice generation

# LDA+MLLT systems building and decoding
if [ ! -f $working_dir/lda.mllt.done ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    2500 15000 $datadir/train data/lang ${gmmdir}_ali $working_dir/tri4 || exit 1;

  graph_dir=$working_dir/tri4/graph
  $mkgraph_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_test_bg ${working_dir}/tri4 $graph_dir || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" \
      $graph_dir $datadir/dev ${working_dir}/tri4/decode_dev || exit 1;
  steps/decode.sh --nj 12 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" \
      $graph_dir $datadir/test ${working_dir}/tri4/decode_test || exit 1;
  touch $working_dir/lda.mllt.done
fi

echo =====================================================================
echo "                      MMI Systems over BNFs                        "
echo =====================================================================
# MMI systems building and decoding
if [ ! -f $working_dir/mmi.done ]; then
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
    $datadir/train data/lang ${working_dir}/tri4 ${working_dir}/tri4_ali || exit 1;

  steps/make_denlats.sh --nj 30 --cmd "$decode_cmd" $denlats_param \
    $datadir/train data/lang ${working_dir}/tri4 ${working_dir}/tri4_denlats || exit 1;

  # 4 iterations of MMI
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
    $datadir/train data/lang $working_dir/tri4_{ali,denlats} $working_dir/tri4_mmi_b0.1 || exit 1;

  for iter in 1 2 3 4; do
    graph_dir=$working_dir/tri4/graph
    steps/decode.sh --nj 12 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" --iter $iter \
      $graph_dir $datadir/dev ${working_dir}/tri4_mmi_b0.1/decode_dev_it$iter || exit 1;
    steps/decode.sh --nj 12 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" --iter $iter \
      $graph_dir $datadir/test ${working_dir}/tri4_mmi_b0.1/decode_test_it$iter || exit 1;
  done
  touch $working_dir/mmi.done
fi

echo =====================================================================
echo "                      SGMM Systems over BNFs                       "
echo =====================================================================
# SGMM system building and decoding
if [ ! -f $working_dir/sgmm.done ]; then
  steps/train_ubm.sh --cmd "$train_cmd" \
    400 $datadir/train data/lang ${working_dir}/tri4_ali ${working_dir}/ubm5 || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" 7000 9000 \
    $datadir/train data/lang ${working_dir}/tri4_ali ${working_dir}/ubm5/final.ubm ${working_dir}/sgmm5a || exit 1;

  graph_dir=$working_dir/sgmm5a/graph
  $decode_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_test_bg ${working_dir}/sgmm5a $graph_dir || exit 1;

  steps/decode_sgmm2.sh --nj 12 --cmd "$decode_cmd" --acwt 0.125 --scoring-opts "$scoring_opts"  \
    $graph_dir $datadir/dev ${working_dir}/sgmm5a/decode_dev || exit 1;
  steps/decode_sgmm2.sh --nj 12 --cmd "$decode_cmd" --acwt 0.125 --scoring-opts "$scoring_opts"  \
    $graph_dir $datadir/test ${working_dir}/sgmm5a/decode_test || exit 1;
  touch $working_dir/sgmm.done
fi

echo =====================================================================
echo "                        MMI-SGMM over BNFs                         "
echo =====================================================================
# Now discriminatively train the SGMM system
if [ ! -f $working_dir/mmi.sgmm.done ]; then
  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" \
    $datadir/train data/lang ${working_dir}/sgmm5a ${working_dir}/sgmm5a_ali || exit 1;

  # Reduce the beam down to 10 to get acceptable decoding speed.
  steps/make_denlats_sgmm2.sh --nj 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" $denlats_param \
    $datadir/train data/lang ${working_dir}/sgmm5a ${working_dir}/sgmm5a_denlats || exit 1;

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --boost 0.1 \
    $datadir/train data/lang $working_dir/sgmm5a_{ali,denlats} ${working_dir}/sgmm5a_mmi_b0.1 || exit 1;

  for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      data/lang_test_bg $datadir/dev ${working_dir}/sgmm5a/decode_dev ${working_dir}/sgmm5a_mmi_b0.1/decode_dev_it$iter || exit 1;
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      data/lang_test_bg $datadir/test ${working_dir}/sgmm5a/decode_test ${working_dir}/sgmm5a_mmi_b0.1/decode_test_it$iter || exit 1;
  done
  touch $working_dir/mmi.sgmm.done
fi

echo "Finish !!"
