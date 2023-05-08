import gym
import gym.spaces as spaces
import numpy as np

from envs.multi_route import multi_route


class MultiRouteEnvV0(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    NUM_ENV_STEPS = 50

    def __init__(
            self,
            obs_high_bound=6,
            obs_low_bound=-1,
            start_state_noise=0.1,
            starting_point=np.array([1, 2]),
            target=np.array([5, 2]),
            use_snap: bool = True,
    ) -> None:
        super().__init__()
        self.obs_high_bound = obs_high_bound
        self.obs_low_bound = obs_low_bound
        self._dim = 2
        self._start_state_noise = start_state_noise
        self._starting_point = starting_point
        self._target = target
        self.use_snap = use_snap

        action_limit = 2
        self.action_space = spaces.Box(
            -action_limit,
            action_limit,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            self.obs_low_bound,
            self.obs_high_bound,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )

        self._target_bounds = spaces.Box(
            low=self._target-0.5,
            high=self._target+0.5,
            dtype=np.float64,
        )

    def reset(self):
        self._state = self._starting_point + np.random.normal(
            0, self._start_state_noise, size=self._starting_point.shape
        )
        return np.copy(self._state)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self._state += action
        if self.use_snap:
            self._state = np.round(self._state)
            assert np.all(self._state % 1 == 0)
        reward = 0
        done = False

        if self._target_bounds.contains(self._state):
            reward = 1
            done = True
        if not self.observation_space.contains(self._state):
            reward = -1
            done = True

        return np.copy(self._state), reward, done, {}

    def render(self, *args, **kwargs):
        pass

    def set_state(self, state):
        err_msg = f"{state!r} ({type(state)}) invalid"
        assert self.observation_space.contains(state), err_msg
        self._state = np.copy(state)
        return self._state
