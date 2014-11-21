#!/bin/bash

# Apache 2.0
# This is the script that trains DNN system over the fbank+pitch features.It
# is to  be  run after run.sh. Before running this, you should already build
# the initial GMM model. This script requires a GPU card, and also the "pdnn"
# toolkit to train the DNN. The input filterbank features are with mean  and
# variance normalization. 

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/dnn_fbank_pitch
do_ptr=true    # whether to do pre-training
delete_pfile=true # whether to delete pfiles after DNN training

gmmdir=exp/tri5a

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
  svn co svn://svn.code.sf.net/p/kaldipdnn/code-0/trunk/pdnn pdnn
fi

if [ ! -d steps_pdnn ]; then
  echo "Checking out steps_pdnn scripts."
  svn co svn://svn.code.sf.net/p/kaldipdnn/code-0/trunk/steps_pdnn steps_pdnn
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

echo ---------------------------------------------------------------------
echo "Creating DNN training and validation data (pfiles)"
echo ---------------------------------------------------------------------
# Alignment on the training data
if [ ! -d ${gmmdir}_ali ]; then
  echo "Generate alignment on train"
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train data/lang $gmmdir ${gmmdir}_ali || exit 1
fi

# Generate the fbank features. The fbanks are 40-dimensional on each frame
echo "--num-mel-bins=40" > conf/fbank.conf
echo "--sample-frequency=8000" >> conf/fbank.conf
mkdir -p $working_dir/data_fbank
for set in train dev; do
  if [ ! -d $working_dir/data_fbank/$set ]; then
    cp -r data/$set $working_dir/data_fbank/$set
    ( cd $working_dir/data_fbank/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_fbank.sh --cmd "$train_cmd" --nj 16 \
      $working_dir/data_fbank/$set $working_dir/_log $working_dir/_fbank || exit 1;
  fi
done

# Generate the pitch features. The pitches are 3-dimensional on each frame
echo "--sample-frequency=8000" >> conf/pitch.conf
mkdir -p $working_dir/data_pitch
for set in train dev; do
  if [ ! -d $working_dir/data_pitch/$set ]; then
    cp -r data/$set $working_dir/data_pitch/$set
    ( cd $working_dir/data_pitch/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_pitch_kaldi.sh --cmd "$train_cmd" --nj 16 \
      $working_dir/data_pitch/$set $working_dir/_log $working_dir/_pitch || exit 1;
  fi
done

# Combine fbank and pitch together
mkdir -p $working_dir/data
for set in train dev; do
  if [ ! -d $working_dir/data/$set ]; then
    steps/append_feats.sh --cmd "$train_cmd" --nj 16 \
      $working_dir/data_fbank/$set $working_dir/data_pitch/$set \
      $working_dir/data/$set $working_dir/_log $working_dir/_append || exit 1;
    # We need to compute CMVN stats on the appended features
    steps/compute_cmvn_stats.sh $working_dir/data/$set $working_dir/_log $working_dir/_append || exit 1;
  fi
done

# By default, inputs include 11 frames (+/-5) of 43-dimensional appended features, with 473 dimensions.
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --every-nth-frame 1 --norm-vars true \
    --do-split true --pfile-unit-size 30 --cv-ratio 0.05 \
    --splice-opts "--left-context=5 --right-context=5" --input-dim 473 \
    $working_dir/data/train ${gmmdir}_ali $working_dir || exit 1
  ( cd $working_dir; rm concat.pfile; )
  touch $working_dir/train.pfile.done
fi

echo ---------------------------------------------------------------------
echo "Starting DNN training"
echo ---------------------------------------------------------------------
feat_dim=$(cat $working_dir/train.pfile |head |grep num_features| awk '{print $2}') || exit 1;

if $do_ptr && [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "SDA Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_SdA.py --train-data "$working_dir/train.pfile,partition=2000m,random=true,stream=true" \
                          --nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
                          --first-reconstruct-activation "tanh" \
                          --wdir $working_dir --output-file $working_dir/dnn.ptr \
                          --ptr-layer-number 6 --epoch-number 5 || exit 1;
  touch $working_dir/dnn.ptr.done
fi

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_DNN.py --train-data "$working_dir/train.pfile,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/valid.pfile,partition=600m,random=true,stream=true" \
                          --nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
                          --ptr-file $working_dir/dnn.ptr --ptr-layer-number 6 \
                          --output-format kaldi --lrate "D:0.08:0.5:0.2,0.2:8" \
                          --wdir $working_dir --output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
  $delete_pfile && rm -rf $working_dir/*.pfile
fi

echo ---------------------------------------------------------------------
echo "Decode the final system"
echo ---------------------------------------------------------------------
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph
  # Here norm-vars has to be the same as steps_pdnn/build_nnet_pfile.sh 
  steps_pdnn/decode_dnn.sh --nj 24 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" --norm-vars true \
    $graph_dir $working_dir/data/dev ${gmmdir}_ali $working_dir/decode || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
