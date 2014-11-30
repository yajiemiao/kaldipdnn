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

working_dir=exp_pdnn_110h/cnn
gmmdir=exp/tri4a 

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
if [ ! -d ${gmmdir}_ali_100k_nodup ]; then
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_100k_nodup data/lang $gmmdir ${gmmdir}_ali_100k_nodup || exit 1
fi
if [ ! -d ${gmmdir}_ali_dev ]; then
  steps/align_fmllr.sh --nj 12 --cmd "$train_cmd" \
    data/train_dev data/lang $gmmdir ${gmmdir}_ali_dev || exit 1
fi

# Generate the fbank features. We generate the 40-dimensional fbanks on each frame
echo "--num-mel-bins=40" > conf/fbank.conf
echo "--sample-frequency=8000" >> conf/fbank.conf
mkdir -p $working_dir/data
if [ ! -d $working_dir/data/train ]; then
  cp -r data/train_100k_nodup $working_dir/data/train
  ( cd $working_dir/data/train; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 24 $working_dir/data/train $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/train || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/train $working_dir/_log $working_dir/_fbank || exit 1;
fi
if [ ! -d $working_dir/data/valid ]; then
  cp -r data/train_dev $working_dir/data/valid
  ( cd $working_dir/data/valid; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 12 $working_dir/data/valid $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/valid || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/valid $working_dir/_log $working_dir/_fbank || exit 1;
fi
if [ ! -d $working_dir/data/eval2000 ]; then
  cp -r data/eval2000 $working_dir/data/eval2000
  ( cd $working_dir/data/eval2000; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 12 $working_dir/data/eval2000 $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/eval2000 || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/eval2000 $working_dir/_log $working_dir/_fbank || exit 1;
fi

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, CNN inputs include 11 frames of filterbanks, and with delta
# and double-deltas.
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
    --norm-vars true --add-deltas true --splice-opts "--left-context=5 --right-context=5" \
    $working_dir/data/train ${gmmdir}_ali_100k_nodup $working_dir || exit 1
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --do-concat false \
    --norm-vars true --add-deltas true --splice-opts "--left-context=5 --right-context=5" \
    $working_dir/data/valid ${gmmdir}_ali_dev $working_dir || exit 1
  touch $working_dir/valid.pfile.done
fi

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
    $pythonCMD pdnn/cmds/run_CNN.py --train-data "$working_dir/train.pfile.*.gz,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/valid.pfile.*.gz,partition=600m,random=true,stream=true" \
                          --conv-nnet-spec "3x11x40:256,9x9,p1x3:256,3x4,p1x1,f" \
                          --nnet-spec "1024:1024:1024:1024:$num_pdfs" \
                          --lrate "D:0.08:0.5:0.2,0.2:8" --momentum 0.9 \
                          --wdir $working_dir --param-output-file $working_dir/nnet.param \
                          --param-output-file $working_dir/nnet.cfg --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/cnn.fine.done
fi

echo =====================================================================
echo "                Dump Convolution-Layer Activation                  "
echo =====================================================================
mkdir -p $working_dir/data_conv
for set in eval2000; do
  if [ ! -d $working_dir/data_conv/$set ]; then
    steps_pdnn/make_conv_feat.sh --nj 24 --cmd "$decode_cmd" \
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
  graph_dir=$gmmdir/graph_sw1_tg
  steps_pdnn/decode_dnn.sh --nj 24 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" \
    --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_conv/eval2000 ${gmmdir}_ali_100k_nodup $working_dir/decode_eval2000_sw1_tg || exit 1;

  touch $working_dir/decode.done
fi

echo "Finish !!"
