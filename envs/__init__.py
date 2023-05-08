import logging
from gym.envs.registration import register
import numpy as np

# Register the environment
register(
    id="multipath-v1",
    entry_point="envs.multi_route.v1:MultiRouteEnvV0",
    max_episode_steps=50,
    reward_threshold=1.0,
)

register(
    id="multipath-fixed-start-v1",
    entry_point="envs.multi_route.v1:MultiRouteEnvV0",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs=dict(
        start_state_noise=0,
    )
)

register(
    id="multipath-v2",
    entry_point="envs.multi_route.v1:MultiRouteEnvV0",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs=dict(
        obs_high_bound=10,
        obs_low_bound=-2,
        starting_point=np.array([0, 0]),
        target=np.array([8, 8]),
    )
)

register(
    id="multipath-fixed-start-v2",
    entry_point="envs.multi_route.v1:MultiRouteEnvV0",
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs=dict(
        obs_high_bound=10,
        obs_low_bound=-2,
        start_state_noise=0,
        starting_point=np.array([0, 0]),
        target=np.array([8, 8]),
    )
)

try:
    import adept_envs

    register(
        id="kitchen-microwave-kettle-light-slider-v0",
        entry_point="envs.kitchen.v0:KitchenMicrowaveKettleLightSliderV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )

    register(
        id="kitchen-microwave-kettle-burner-light-v0",
        entry_point="envs.kitchen.v0:KitchenMicrowaveKettleBottomBurnerLightV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )

    register(
        id="kitchen-kettle-microwave-light-slider-v0",
        entry_point="envs.kitchen.v0:KitchenKettleMicrowaveLightSliderV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )

    register(
        id="kitchen-all-v0",
        entry_point="envs.kitchen.v0:KitchenAllV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )

except ImportError:
    logging.warning("Kitchen not installed, skipping")

try:
    import envs.block_pushing.block_pushing
    import envs.block_pushing.block_pushing_multimodal
    import envs.block_pushing.block_pushing_discontinuous

except:
    logging.error(
        "Block pushing could not be imported. Make sure you have PyBullet installed."
    )
