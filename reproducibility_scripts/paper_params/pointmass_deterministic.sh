# Deterministic environment.

# To train.

# Pointmass1.
python train.py \
  env=pointmass1 \
  experiment.num_cv_runs=3 \
  env.dataset.noise_scale=0 \
  experiment.save_subdir=reproduction/paper_params_deterministic

# Pointmass2.
python train.py \
  env=pointmass2 \
  experiment.num_cv_runs=3 \
  env.dataset.noise_scale=0 \
  experiment.save_subdir=reproduction/paper_params_deterministic

# To evaluate.

python run_on_env.py \
  env=pointmass1 \
  env.gym_name=multipath-fixed-start-v1 \
  model.load_dir="$(pwd)"/train_runs/train_pointmass1/reproduction/paper_params_deterministic/0 \
  experiment.save_subdir=reproduction/paper_params_deterministic/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=False \
  experiment.num_envs=1 \
  experiment.num_eval_eps=10 \
  experiment.device=cpu

python run_on_env.py \
  env=pointmass2 \
  env.gym_name=multipath-fixed-start-v2 \
  model.load_dir="$(pwd)"/train_runs/train_pointmass2/reproduction/paper_params_deterministic/0 \
  experiment.save_subdir=reproduction/paper_params_deterministic/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=False \
  experiment.num_envs=1 \
  experiment.num_eval_eps=10 \
  experiment.device=cpu
