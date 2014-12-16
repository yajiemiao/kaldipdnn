#!/bin/bash

# Copyright 2014     Yajie Miao   Carnegie Mellon University       Apache 2.0
# This script trains CNN model over the filterbank features. It  is to be run
# after run.sh. Before running this, you should already build the initial GMM
# GMM model. This script requires a GPU, and also the "pdnn" toolkit to train
# CNN. The input filterbank features are with mean and variance normalization. 

# The input features and CNN architecture follow the IBM configuration: 
# Hagen Soltau, George Saon, and Tara N. Sainath. Joint Training of Convolu-
# tional and non-Convolutional Neural Networks

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/cnn
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
    steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
      data/train_$set data/lang $gmmdir ${gmmdir}_ali_$set || exit 1
  fi
done

# Generate the fbank features: 40-dimensional fbanks on each frame
echo "--num-mel-bins=40" > conf/fbank.conf
mkdir -p $working_dir/data
for set in train_tr95 train_cv05; do
  if [ ! -d $working_dir/data/$set ]; then
    cp -r data/$set $working_dir/data/$set
    ( cd $working_dir/data/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_fbank.sh --cmd "$train_cmd" --nj 24 $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
    steps/compute_cmvn_stats.sh $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
  fi
done

for set in dev test; do
  if [ ! -d $working_dir/data/$set ]; then
    cp -r data/$set $working_dir/data/$set
    ( cd $working_dir/data/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
    steps/compute_cmvn_stats.sh $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
  fi
done

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, CNN inputs include 11 frames of filterbanks, and with delta
# and double-deltas.
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
      --norm-vars true --add-deltas true --splice-opts "--left-context=5 --right-context=5" \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir || exit 1
    touch $working_dir/${set}.pfile.done
  fi
done

echo =====================================================================
echo "                        CNN  Fine-tuning                           "
echo =====================================================================
# CNN is configed in the way that it has (approximately) the same number of trainable parameters as DNN
# (e.g., the DNN in run-dnn-fbank.sh). Also, we adopt "--momentum 0.9" becuase CNN over filterbanks seems
# to converge slowly. So we increase momentum to speed up convergence.
if [ ! -f $working_dir/cnn.fine.done ]; then
  echo "Fine-tuning CNN"
  $cmd $working_dir/log/cnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_CNN.py --train-data "$working_dir/train_tr95.pfile.*.gz,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/train_cv05.pfile.*.gz,partition=600m,random=true,stream=true" \
                          --conv-nnet-spec "3x11x40:256,9x9,p1x3:256,3x4,p1x1,f" \
                          --nnet-spec "1024:1024:1024:1024:$num_pdfs" \
                          --lrate "D:0.08:0.5:0.2,0.2:4" --momentum 0.9 \
                          --wdir $working_dir --param-output-file $working_dir/nnet.param \
                          --cfg-output-file $working_dir/nnet.cfg --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/cnn.fine.done
fi

echo =====================================================================
echo "                Dump Convolution-Layer Activation                  "
echo =====================================================================
mkdir -p $working_dir/data_conv
for set in dev; do
  if [ ! -d $working_dir/data_conv/$set ]; then
    steps_pdnn/make_conv_feat.sh --nj 8 --cmd "$decode_cmd" \
      $working_dir/data_conv/$set $working_dir/data/$set $working_dir $working_dir/nnet.param \
      $working_dir/nnet.cfg $working_dir/_log $working_dir/_conv || exit 1;
    # Generate *fake* CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv/$set $working_dir/_log $working_dir/_conv || exit 1;
  fi
done
for set in test; do
  if [ ! -d $working_dir/data_conv/$set ]; then
    steps_pdnn/make_conv_feat.sh --nj 11 --cmd "$decode_cmd" \
      $working_dir/data_conv/$set $working_dir/data/$set $working_dir $working_dir/nnet.param \
      $working_dir/nnet.cfg $working_dir/_log $working_dir/_conv || exit 1;
    # Generate *fake* CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv/$set $working_dir/_log $working_dir/_conv || exit 1;
  fi
done

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
# In decoding, we take the convolution-layer activation as inputs and the 
# fully-connected layers as the DNN model. So we set --norm-vars, --add-deltas
# and --splice-opts accordingly.
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph
  steps_pdnn/decode_dnn.sh --nj 8 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_conv/dev ${gmmdir}_ali_tr95 $working_dir/decode_dev || exit 1;
  steps_pdnn/decode_dnn.sh --nj 11 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_conv/test ${gmmdir}_ali_tr95 $working_dir/decode_test || exit 1;
  touch $working_dir/decode.done
fi
# Decoding with our own LM. This trigram LM is trained over TED talk transcripts and is pruned
if [ ! -f  $working_dir/decode.bd.done ] && [ -d $gmmdir/graph_bd_tgpr ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_bd_tgpr
  steps_pdnn/decode_dnn.sh --nj 8 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    $graph_dir $working_dir/data/dev ${gmmdir}_ali_tr95 $working_dir/decode_dev_bd_tgpr || exit 1;
  steps_pdnn/decode_dnn.sh --nj 11 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    $graph_dir $working_dir/data/test ${gmmdir}_ali_tr95 $working_dir/decode_test_bd_tgpr || exit 1;
  touch $working_dir/decode.bd.done
fi

echo "Finish !!"
