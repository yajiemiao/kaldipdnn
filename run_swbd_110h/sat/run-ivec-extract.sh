#!/bin/bash

# Apache 2.0
# This is the script that trains an i-vector extractor on the entire set of
# swbd1. The i-vectors will be used in the run-dnn-fbank-sat.sh and run-dnn
# -sat.sh recipes.

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

# Link the scripts from the sre recipe to here
if [ ! -d sid ]; then
  ln -s ../../sre08/v1/sid ./
fi
mkdir -p data_ivec mfcc_ivec exp_ivec

echo ---------------------------------------------------------------------
echo "Train the i-vector extractor with the entire SWBD set (310 hours)"
echo ---------------------------------------------------------------------

# MFCC config; slightly different from GMM/HMM
echo "--sample-frequency=8000" > conf/mfcc.conf.ivec
echo "--frame-length=20" >> conf/mfcc.conf.ivec
echo "--low-freq=20" >> conf/mfcc.conf.ivec
echo "--high-freq=3700" >> conf/mfcc.conf.ivec
echo "--num-ceps=20" >> conf/mfcc.conf.ivec
# Config for VAD (voice activity detection)
echo "--vad-energy-threshold=5.5" > conf/vad.conf
echo "--vad-energy-mean-scale=0.5" >> conf/vad.conf

if [ ! -d data_ivec/swbd1 ]; then
  echo "Save features for swbd1"
  mkdir -p data_ivec/swbd1; cat data/train/wav.scp | awk '{gsub("^sw0","",$1); print $1 " " $2;}' > data_ivec/swbd1/wav.scp
  ( cd data_ivec/swbd1; cat wav.scp | awk '{print $1 " " $1}' > utt2spk; cp utt2spk spk2utt; )

  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf.ivec --nj 24 --cmd "$train_cmd" \
      data_ivec/swbd1 exp_ivec/make_mfcc mfcc_ivec || exit 1;
  sid/compute_vad_decision.sh --nj 24 --cmd "$train_cmd" \
      data_ivec/swbd1 exp_ivec/make_mfcc mfcc_ivec || exit 1;
  utils/fix_data_dir.sh data_ivec/swbd1 || exit 1;
fi

if [ ! -d data_ivec/eval2000 ]; then
  echo "Save features for eval2000"
  mkdir -p data_ivec/eval2000; cp data/eval2000/wav.scp data_ivec/eval2000
  ( cd data_ivec/eval2000; cat wav.scp | awk '{print $1 " " $1}' > utt2spk; cp utt2spk spk2utt; )
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf.ivec --nj 24 --cmd "$train_cmd" \
      data_ivec/eval2000 exp_ivec/make_mfcc mfcc_ivec || exit 1;
  sid/compute_vad_decision.sh --nj 24 --cmd "$train_cmd" \
      data_ivec/eval2000 exp_ivec/make_mfcc mfcc_ivec || exit 1;
  utils/fix_data_dir.sh data_ivec/eval2000 || exit 1;
fi

# Train the diagonal and full UBMs
if [ ! -f exp_ivec/diag_ubm_2048/final.dubm ]; then
  echo "Train the diagonal UBM"
  sid/train_diag_ubm.sh --parallel-opts "" --nj 24 --cmd "$train_cmd" \
    data_ivec/swbd1 2048 exp_ivec/diag_ubm_2048 || exit 1;
fi
if [ ! -f full_ubm_2048/final.ubm ]; then
  echo "Train the full UBM"
  sid/train_full_ubm.sh --nj 24 --cmd "$train_cmd" \
    data_ivec/swbd1 exp_ivec/diag_ubm_2048 exp_ivec/full_ubm_2048 || exit 1;
fi

# Train the i-vector extractor
if [ ! -f exp_ivec/extractor_2048/final.ie ]; then
  echo "Train the i-vector extractor"
  sid/train_ivector_extractor.sh --nj 24 --num-threads 1 --num-processes 1 \
    --cmd "$train_cmd" --ivector-dim 100 --num-iters 10 \
    exp_ivec/full_ubm_2048/final.ubm data_ivec/swbd1 exp_ivec/extractor_2048
fi

# Generate i-vectors on the training and testing (decoding) sets
if [ ! -f exp_ivec/ivectors_swbd1/ivector.scp ]; then
  echo "Extract i-vectors for the swbd1 speakers"
  sid/extract_ivectors.sh --cmd "$train_cmd" --nj 24 \
    exp_ivec/extractor_2048 data_ivec/swbd1 exp_ivec/ivectors_swbd1
fi
if [ ! -f exp_ivec/ivectors_eval2000/ivector.scp ]; then
  echo "Extract i-vectors for the eval2000 speakers"
  sid/extract_ivectors.sh --cmd "$train_cmd" --nj 24 \
    exp_ivec/extractor_2048 data_ivec/eval2000 exp_ivec/ivectors_eval2000
fi

echo "Finish !! Now you can safely delete data_ivec and mfcc_ivec."
