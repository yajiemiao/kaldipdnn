#!/bin/bash

# Apache 2.0
# This  is the script that trains DNN system. It is to be run after run.sh. 
# Before running this, you should already build the initial GMM model. This
# script requires a GPU, and also the "pdnn" toolkit to train the DNN.

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn_110h/dnn
do_ptr=true      # whether to do pre-training
delete_pfile=false # whether to delete pfiles after DNN training

gmmdir=exp/tri4a # tri4a is the SAT model trained with 110 hours
            # data/train_100k_nodup

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
echo "Generate alignment and prepare fMLLR features"
echo ---------------------------------------------------------------------
# Alignment on the training and validation data
if [ ! -d ${gmmdir}_ali_100k_nodup ]; then
  echo "Generate alignment on train data"
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_100k_nodup data/lang $gmmdir ${gmmdir}_ali_100k_nodup || exit 1
fi
if [ ! -d ${gmmdir}_ali_dev ]; then
  echo "Generate alignment on valid data"
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_dev data/lang $gmmdir ${gmmdir}_ali_dev || exit 1
fi

# Dump fMLLR features. We generate "fake" cmvn states (0 means and 1 variance) which apply no normalization
if [ ! -d $working_dir/data/train ]; then
  echo "Save fmllr features of train data"
  steps/nnet/make_fmllr_feats.sh --nj 24 --cmd "$train_cmd" \
    --transform-dir ${gmmdir}_ali_100k_nodup \
    $working_dir/data/train data/train_100k_nodup $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/train $working_dir/_log $working_dir/_fmllr || exit 1;
fi
if [ ! -d $working_dir/data/valid ]; then
  echo "Save fmllr features of valid data"
  steps/nnet/make_fmllr_feats.sh --nj 24 --cmd "$train_cmd" \
    --transform-dir ${gmmdir}_ali_dev \
    $working_dir/data/valid data/train_dev $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/valid $working_dir/_log $working_dir/_fmllr || exit 1;
fi
if [ ! -d $working_dir/data/eval2000 ]; then
  echo "Save fmllr features of eval2000"
  steps/nnet/make_fmllr_feats.sh --nj 24 --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_eval2000_sw1_tg \
    $working_dir/data/eval2000 data/eval2000 $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
  steps/compute_cmvn_stats.sh --fake \
    $working_dir/data/eval2000 $working_dir/_log $working_dir/_fmllr || exit 1;
fi

echo ---------------------------------------------------------------------
echo "Create DNN training and validation pfiles"
echo ---------------------------------------------------------------------

# By default, DNN inputs include: spliced 11 frames (+/-5) of fMLLR with 440 dimensions
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars false --splice-opts "--left-context=5 --right-context=5" --input-dim 440 \
    $working_dir/data/train ${gmmdir}_ali_100k_nodup $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile train.pfile; )
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars false --splice-opts "--left-context=5 --right-context=5" --input-dim 440 \
    $working_dir/data/valid ${gmmdir}_ali_dev $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile valid.pfile; )
  touch $working_dir/valid.pfile.done
fi

echo ---------------------------------------------------------------------
echo "Start DNN training"
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
  graph_dir=$gmmdir/graph_sw1_tg
  steps_pdnn/decode_dnn.sh --nj 24 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" --norm-vars false \
     $graph_dir $working_dir/data/eval2000 ${gmmdir}_ali_100k_nodup $working_dir/decode_eval2000_sw1_tg || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
