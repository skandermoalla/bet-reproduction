# To train.

python train.py \
  env=kitchen \
  experiment.num_cv_runs=3 \
  experiment.window_size=1,5,10,20 \
  experiment.save_subdir=reproduction/sweep_window_size

# To evaluate.

for job in {0..3}; do
  python run_on_env.py \
    env=kitchen \
    model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/sweep_window_size/"$job" \
    experiment.save_subdir=reproduction/sweep_window_size/"$job" \
    experiment.vectorized_env=True \
    experiment.async_envs=True \
    experiment.num_envs=20 \
    experiment.num_eval_eps=50 \
    experiment.device=cpu
done

# To compute metrics.

for job in {0..3}; do
  python compute_metrics.py \
    load_dir="$(pwd)"/eval_runs/eval_kitchen/reproduction/sweep_window_size/"$job" \
    tags=\'window_size,sweep,compute_metrics\' \
    save_subdir=kitchen/reproduction/sweep_window_size/"$job"
done