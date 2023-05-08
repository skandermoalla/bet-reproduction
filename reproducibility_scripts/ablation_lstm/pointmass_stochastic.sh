# Stochastic environment

# To train.

# Pointmass1.
python train.py \
  env=pointmass1 \
  experiment.num_cv_runs=3 \
  model=lstm \
  experiment.save_subdir=reproduction/ablation_lstm/stochastic

# Pointmass2.
python train.py \
  env=pointmass2 \
  experiment.num_cv_runs=3 \
  model=lstm \
  experiment.save_subdir=reproduction/ablation_lstm/stochastic

# Evaluation.

# Pointmass1.
python run_on_env.py \
  env=pointmass1 \
  model.load_dir="$(pwd)"/train_runs/train_pointmass1/reproduction/ablation_lstm/stochastic/0 \
  experiment.save_subdir=reproduction/ablation_lstm/stochastic/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=False \
  experiment.num_envs=1 \
  experiment.num_eval_eps=10 \
  experiment.device=cpu

# Pointmass2.
python run_on_env.py \
  env=pointmass2 \
  model.load_dir="$(pwd)"/train_runs/train_pointmass2/reproduction/ablation_lstm/stochastic/0 \
  experiment.save_subdir=reproduction/ablation_lstm/stochastic/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=False \
  experiment.num_envs=1 \
  experiment.num_eval_eps=10 \
  experiment.device=cpu
