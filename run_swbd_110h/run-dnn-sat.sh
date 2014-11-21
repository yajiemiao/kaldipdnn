#!/bin/bash

# Apache 2.0
# This is the script that performs speaker adaptive training (SAT) of the
# DNN model which has been trained on the fMLLR features. It is to be run
# after run-dnn.sh.

# Yajie Miao, Hao Zhang, Florian Metze. "Towards Speaker Adaptive Training
# of Deep Neural Network Acoustic Models". Interspeech 2014.

# You need two additional commands to execute this recipe: get-spkvec-feat
# and add-feats.Download the following two source files and put them under
# src/featbin. Then compiling them will give you the required commands.

# http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/get-spkvec-feat.cc
# http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/add-feats.cc

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn_110h/dnn_sat
initdnn_dir=exp_pdnn_110h/dnn # the directory of the initial DNN model

gmmdir=exp/tri4a # GMM model directory

# I-vectors for the training and decoding speakers. There should be an ivector.scp
# file in each of both directories.
train_ivec=exp_ivec/ivectors_swbd1
decode_ivec=exp_ivec/ivectors_eval2000

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

# Check whether i-vectors have been generated
for f in $train_ivec/ivector.scp $decode_ivec/ivector.scp; do
  [ ! -f $f ] && echo "Error i-vectors for $f have NOT been extracted. Check/Run run_swbd_110h/run-ivec-extract.sh." && exit 1;
done
# Check whether the initial DNN has been trained 
[ ! -f $initdnn_dir/nnet.finetune.tmp ] && echo "Error the initial DNN $initdnn_dir/nnet.finetune.tmp has NOT been trained" && exit 1;

# Prepare dataset; copy related files from the initial DNN directory
ln -s $PWD/$initdnn_dir/data $working_dir/data
cp $initdnn_dir/splice_opts $working_dir
splice_opts=`cat $working_dir/splice_opts 2>/dev/null` # frame-splicing options.

echo ---------------------------------------------------------------------
echo "Create SAT-DNN training and validation pfiles"
echo ---------------------------------------------------------------------
# By default, DNN inputs include: spliced 11 frames (+/-5) of fMLLR with 440 dimensions.
# The i-vectors have the dimension of 100. Thus, the pfile has the dimension of 540.
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile_ivec.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars false --splice-opts "$splice_opts" --input-dim 540 --is-spk-mode true \
    $working_dir/data/train ${gmmdir}_ali_100k_nodup $train_ivec $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile train.pfile; )
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile_ivec.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars false --splice-opts "$splice_opts" --input-dim 540 --is-spk-mode true \
    $working_dir/data/valid ${gmmdir}_ali_dev $train_ivec $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile valid.pfile; )
  touch $working_dir/valid.pfile.done
fi

echo ---------------------------------------------------------------------
echo "Start SAT-DNN training"
echo ---------------------------------------------------------------------
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;
ivec_dim=`feat-to-dim scp:ivector.scp ark,t:- | head -1 | awk '{print $2}'` || exit 1;
feat_dim=$(cat $working_dir/train.pfile |head |grep num_features| awk '{print $2}') || exit 1;
feat_dim=$[$feat_dim-$ivec_dim]

if [ ! -f $working_dir/sat.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/sat.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_DNN_SAT.py --train-data "$working_dir/train.pfile,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/valid.pfile,partition=600m,random=true,stream=true" \
                          --nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
                          --ivec-nnet-spec "$ivec_dim:512:512:512:$feat_dim" \
                          --si-model $initdnn_dir/nnet.finetune.tmp \
                          --output-format kaldi --lrate "D:0.08:0.5:0.05,0.05:1" \
                          --wdir $working_dir --output-file $working_dir/dnn.nnet \
                          --ivec-output-file $working_dir/ivec.nnet || exit 1;
  touch $working_dir/sat.fine.done
fi

# Remove the last line "<sigmoid> *** ***" of ivec.nnet, because the output layer of iVecNN uses the linear
# activation function 
( cd $working_dir; head -n -1 ivec.nnet > ivec.nnet.tmp; mv ivec.nnet.tmp ivec.nnet; )

echo ---------------------------------------------------------------------
echo "Decode the final system"
echo ---------------------------------------------------------------------
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_sw1_tg
  steps_pdnn/decode_dnn_ivec.sh --nj 24 --scoring-opts "--min-lmwt 8 --max-lmwt 16" --cmd "$decode_cmd" \
     --norm-vars false --is-spk-mode true \
     $graph_dir $working_dir/data/eval2000 ${gmmdir}_ali_100k_nodup $decode_ivec $working_dir/decode_eval2000_sw1_tg || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
