#!/bin/bash

# Apache 2.0
# This is the script that performs speaker adaptive training (SAT) of the
# DNN model which has been trained on the filterbank features.It is to be
# run after run-dnn-fbank.sh.

# Yajie Miao, Hao Zhang, Florian Metze. "Towards Speaker Adaptive Training
# of Deep Neural Network Acoustic Models". Interspeech 2014.

# You need two additional commands to execute this recipe: get-spkvec-feat
# and add-feats.Download the following two source files and put them under
# src/featbin. Then compiling them will give you the required commands.

# http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/get-spkvec-feat.cc
# http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/add-feats.cc

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn_110h/dnn_fbank_sat
initdnn_dir=exp_pdnn_110h/dnn_fbank # the directory of the initial DNN model

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
  [ ! -f $f ] && echo "Error i-vectors for $f have NOT been extracted. Check/Run run_swbd_110h/sat/run-ivec-extract.sh." && exit 1;
done
# Check whether the initial DNN has been trained 
if [ ! -f $initdnn_dir/nnet.param ]; then
  echo "Error the initial DNN $initdnn_dir/nnet.param has NOT been trained" && exit 1;
fi

# Prepare dataset; copy related files from the initial DNN directory
ln -s $PWD/$initdnn_dir/data $working_dir/data || exit 1;
cp $initdnn_dir/{splice_opts,norm_vars,add_deltas} $working_dir || exit 1;
splice_opts=`cat $working_dir/splice_opts 2>/dev/null` # frame-splicing options.
norm_vars=`cat $working_dir/norm_vars 2>/dev/null`     # variance normalization?
add_deltas=`cat $working_dir/add_deltas 2>/dev/null`   # add deltas?

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================

if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/sat/build_nnet_pfile_ivec.sh --cmd "$train_cmd" --every-nth-frame 1 --do-concat false \
    --norm-vars $norm_vars --splice-opts "$splice_opts" --add-deltas $add_deltas \
    --ivec-type speaker \
    $working_dir/data/train ${gmmdir}_ali_100k_nodup $train_ivec $working_dir || exit 1
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile_ivec.sh --cmd "$train_cmd" --every-nth-frame 1 --do-concat false \
    --norm-vars $norm_vars --splice-opts "$splice_opts" --add-deltas $add_deltas \ 
    --ivec-type speaker \
    $working_dir/data/valid ${gmmdir}_ali_dev $train_ivec $working_dir || exit 1
  touch $working_dir/valid.pfile.done
fi

echo =====================================================================
echo "                        SAT-DNN Fine-tuning                        "
echo =====================================================================
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;
ivec_dim=`feat-to-dim scp:$train_ivec/ivector.scp ark,t:- | head -1 | awk '{print $2}'` || exit 1;
feat_dim=$(gunzip -c $working_dir/train.pfile.1.gz |head |grep num_features| awk '{print $2}') || exit 1;
feat_dim=$[$feat_dim-$ivec_dim]

# NOTE: the definition of "--si-nnet-spec" here has to be the same as "--nnet-spec" in run-dnn-fbank.sh
if [ ! -f $working_dir/sat.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/sat.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_DNN_SAT.py --train-data "$working_dir/train.pfile.*.gz,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/valid.pfile.*.gz,partition=600m,random=true,stream=true" \
                          --si-nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
                          --adapt-nnet-spec "$ivec_dim:512:512:512" --init-model $initdnn_dir/nnet.param \
                          --lrate "D:0.08:0.5:0.05,0.05:0" --param-output-file $working_dir/nnet.param \
                          --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;                          
  touch $working_dir/sat.fine.done
fi

# Remove the last line "<sigmoid> *** ***" of dnn.nnet.adapt, because the output layer of the adaptation network
# uses the linear activation function 
( cd $working_dir; head -n -1 dnn.nnet.adapt > dnn.nnet.adapt.tmp; mv dnn.nnet.adapt.tmp dnn.nnet.adapt; )

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_sw1_tg
  steps_pdnn/sat/decode_dnn_ivec.sh --nj 24 --scoring-opts "--min-lmwt 8 --max-lmwt 16" --cmd "$decode_cmd" --ivec-type speaker \
    $graph_dir $working_dir/data/eval2000 ${gmmdir}_ali_100k_nodup $decode_ivec $working_dir/decode_eval2000_sw1_tg || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
