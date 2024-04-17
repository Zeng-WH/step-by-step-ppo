#export HF_HOME="/ssddata/model_hub"
# export CUDA_HOME=/usr/local/cuda-11.7  #指定cuda根目录
# export PATH=$PATH:/usr/local/cuda-11.7/bin  #安装的cuda的路径下的bin文件夹，包含了nvcc等二进制程序
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64  ##安装的cuda的路径下的lib64文件夹，包含很多库文件



rm -rf ~/.cache/torch/kernels/*
export HF_HOME="/ssddata/weihao00/model_zoo"
CUDA_VISIBLE_DEVICES=4 python gsm8k_test_wandb.py --model_dir /ssddata/weihao00/save_dir/full_ppo/ppo_tbsz256_prm_bbsz128_pep4_lr2e7_24share_pen_0.4_l2048_reweight_new2/ --model_id test --start 0 --end 1400 --batch_size 80 --tensor_parallel_size 1