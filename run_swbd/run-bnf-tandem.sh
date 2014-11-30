#!/bin/bash

# Apache 2.0
# This script  trains tandem systems using bottleneck features (BNFs). The 
# BNF network is trained over fMLLR features. It is to be run after run.sh.
# Before running this, you should already build the initial GMM model. This
# script requires a  GPU, and also the "pdnn" toolkit to train the BNF net.

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/bnf_tandem
gmmdir=exp/tri4b

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
echo "                   Alignment & Feature Preparation                 "
echo =====================================================================
# Alignment on the training and validation data
if [ ! -d ${gmmdir}_ali_nodup ]; then
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_nodup data/lang $gmmdir ${gmmdir}_ali_nodup || exit 1
fi
if [ ! -d ${gmmdir}_ali_dev ]; then
  steps/align_fmllr.sh --nj 12 --cmd "$train_cmd" \
    data/train_dev data/lang $gmmdir ${gmmdir}_ali_dev || exit 1
fi

# Dump fMLLR features. "fake" cmvn states (0 means and 1 variance) which apply no normalization
if [ ! -d $working_dir/data/train ]; then
  steps/make_fmllr_feats.sh --nj 24 --cmd "$train_cmd" \
    --transform-dir ${gmmdir}_ali_nodup \
    $working_dir/data/train data/train_nodup $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/train $working_dir/_log $working_dir/_fmllr || exit 1;
fi
if [ ! -d $working_dir/data/valid ]; then
  steps/make_fmllr_feats.sh --nj 12 --cmd "$train_cmd" \
    --transform-dir ${gmmdir}_ali_dev \
    $working_dir/data/valid data/train_dev $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/valid $working_dir/_log $working_dir/_fmllr || exit 1;
fi
if [ ! -d $working_dir/data/eval2000 ]; then
  steps/make_fmllr_feats.sh --nj 12 --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_eval2000_sw1_tg \
    $working_dir/data/eval2000 data/eval2000 $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/eval2000 $working_dir/_log $working_dir/_fmllr || exit 1;
fi

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, DNN inputs include 11 frames of fMLLR
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
    --norm-vars false --splice-opts "--left-context=5 --right-context=5" \
    $working_dir/data/train ${gmmdir}_ali_nodup $working_dir || exit 1
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
    --norm-vars false --splice-opts "--left-context=5 --right-context=5" \
    $working_dir/data/valid ${gmmdir}_ali_dev $working_dir || exit 1
  touch $working_dir/valid.pfile.done
fi

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
feat_dim=$(gunzip -c $working_dir/train.pfile.1.gz |head |grep num_features| awk '{print $2}') || exit 1;

# We use SDA because it's faster than RBM
if [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "SDA Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_SdA.py --train-data "$working_dir/train.pfile.*.gz,partition=2000m,random=true,stream=true" \
                                    --nnet-spec "$feat_dim:2048:2048:2048:2048:42:2048:$num_pdfs" \
                                    --1stlayer-reconstruct-activation "tanh" \
                                    --wdir $working_dir --param-output-file $working_dir/dnn.ptr \
                                    --ptr-layer-number 4 --epoch-number 5 || exit 1;
  touch $working_dir/dnn.ptr.done
fi

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_DNN.py --train-data "$working_dir/train.pfile.*.gz,partition=2000m,random=true,stream=true" \
                                    --valid-data "$working_dir/valid.pfile.*.gz,partition=600m,random=true,stream=true" \
                                    --nnet-spec "$feat_dim:2048:2048:2048:2048:42:2048:$num_pdfs" \
                                    --ptr-file $working_dir/dnn.ptr --ptr-layer-number 4 \
                                    --lrate "D:0.08:0.5:0.2,0.2:8" \
                                    --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
fi

( cd $working_dir; ln -s dnn.nnet bnf.nnet )

echo =====================================================================
echo "                    BNF Feature Generation                         "
echo =====================================================================
# Dump BNF features
for set in train eval2000; do
  if [ ! -d $working_dir/data_bnf/${set} ]; then
    steps_pdnn/make_bnf_feat.sh --nj 24 --cmd "$train_cmd" \
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
decode_param="--beam 15.0 --latbeam 7.0 --acwt 0.04" # decoding parameters differ from MFCC systems
scoring_opts="--min-lmwt 26 --max-lmwt 34"
denlats_param="--acwt 0.05"                        # Parameters for lattice generation

# LDA+MLLT systems building and decoding
if [ ! -f $working_dir/lda.mllt.done ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    11500 200000 $datadir/train data/lang ${gmmdir}_ali_nodup $working_dir/tri4b || exit 1;

  graph_dir=$working_dir/tri4b/graph_sw1_tg
  $mkgraph_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_sw1_tg $working_dir/tri4b $graph_dir || exit 1;
  steps/decode.sh --nj 24 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" \
      $graph_dir $datadir/eval2000 $working_dir/tri4b/decode_eval2000_sw1_tg || exit 1;
  touch $working_dir/lda.mllt.done
fi

echo =====================================================================
echo "                      MMI Systems over BNFs                        "
echo =====================================================================
# MMI systems building and decoding
if [ ! -f $working_dir/mmi.done ]; then
  steps/align_si.sh --nj 64 --cmd "$train_cmd" \
    $datadir/train data/lang ${working_dir}/tri4b ${working_dir}/tri4b_ali || exit 1;

  steps/make_denlats.sh --nj 64 --cmd "$decode_cmd" $denlats_param \
    $datadir/train data/lang ${working_dir}/tri4b ${working_dir}/tri4b_denlats || exit 1;

  # 4 iterations of MMI
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
    $datadir/train data/lang $working_dir/tri4b_{ali,denlats} $working_dir/tri4b_mmi_b0.1 || exit 1;

  for iter in 1 2 3 4; do
    graph_dir=$working_dir/tri4b/graph_sw1_tg
    steps/decode.sh --nj 24 --cmd "$decode_cmd" $decode_param --scoring-opts "$scoring_opts" --iter $iter \
      $graph_dir $datadir/eval2000 ${working_dir}/tri4b_mmi_b0.1/decode_eval2000_sw1_tg_it$iter || exit 1;
  done
  touch $working_dir/mmi.done
fi

echo "Finish !! "
