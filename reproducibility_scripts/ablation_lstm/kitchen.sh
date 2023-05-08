python train.py \
  env=kitchen \
  experiment.num_cv_runs=3 \
  model=lstm \
  experiment.save_subdir=reproduction/ablation_lstm/paper


python run_on_env.py \
  env=kitchen \
  model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/ablation_lstm/paper/0 \
  experiment.save_subdir=reproduction/ablation_lstm/paper/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=True \
  experiment.num_envs=20 \
  experiment.num_eval_eps=50 \
  experiment.device=cpu


python compute_metrics.py \
  load_dir="$(pwd)"/eval_runs/eval_kitchen/reproduction/ablation_lstm/paper/0 \
  tags=\'ablation,lstm,paper,compute_metrics\' \
  save_subdir=kitchen/reproduction/ablation_lstm/paper/0
