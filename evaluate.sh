eval_mode=$1
ckp_dir=${2:-"./results/coop2"}
phase=${3:-"coop"}

GPUS="0"
ablation_list_dir="ablation_list.mat"

if [[ "${phase}" == "coop" ]]; then
  condition=MultiAgentSync_fullobs
elif [[ "${phase}" == "non_coop" ]]; then
  condition=MultiAgentSing_fullobs
fi

## generate video for the best iteration 
if [[ "${eval_mode}" == "gen_video" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS} python evaluate_gen_video.py --gpu_id ${GPUS} --condition ${condition} \
    --checkpoint_dir ${ckp_dir} 
## roll out and save performance metrics across checkpoints 
elif [[ "${eval_mode}" == "save_metrics" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS} python evaluate_events.py --gpu_id ${GPUS} --condition ${condition} \
    --checkpoint_dir ${ckp_dir} --eval_step 50 
## save activations
elif [[ "${eval_mode}" == "save_activations" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS} python evaluate_save_activation.py --gpu_id ${GPUS} --condition ${condition} \
    --checkpoint_dir ${ckp_dir} 
## ablate certain neuron populations 
elif [[ "${eval_mode}" == "ablation" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS} python eval_ablation.py --gpu_id ${GPUS} --condition ${condition} \
    --checkpoint_dir ${ckp_dir} --ablation_list_dir ${ablation_list_dir}
fi