from collections import OrderedDict
import numpy as np
import pickle
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from skimage.transform import rescale, resize, downscale_local_mean, rotate


class SawyerPusherEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(self,
            use_reward='yes',
            sparse='yes',
            task=0,
            radius_reward=0.1,
            file_goals="goals_sawyer_pusher_10tests_tasks.pkl",
            #file_goals=None,
            file_env=None,
            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)

        self.file_goals = file_goals
        if file_env == None:
            self.file_env = 'sawyer_xyz/sawyer_pick_and_place.xml'
        else:
            self.file_env = file_env

        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )

        self.choice = task
        self.sparse = sparse
        self.max_path_length = 150
        self.use_reward = use_reward
        self.radius_reward = radius_reward

        if file_goals == None:
            self.all_goals = [np.zeros(10)]
        else:
            self.all_goals = pickle.load(open(file_goals, "rb"))


        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        #if obj_low is None:
        obj_low = self.hand_low
        #if obj_high is None:
        obj_high = self.hand_high

        #if goal_low is None:
        goal_low = np.hstack((self.hand_low, obj_low))
#    if goal_high is None:
        goal_high = np.hstack((self.hand_high, obj_high))

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, obj_low, obj_low, self.hand_low)),
            np.hstack((self.hand_high, obj_high, obj_high, obj_high, obj_high, self.hand_high)),
        )
        """self.observation_space = Dict([
            ('observation', self.hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),

            ('state_observation', self.hand_and_obj_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
        ])"""

        self.observation_space = self.hand_and_obj_space

        #import IPython
        #IPython.embed()

        self.reset_model()

    @property
    def model_name(self):
        return get_asset_full_path(self.file_env)

    def viewer_setup(self):
        #pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
         self.viewer.cam.distance = 2.1
         self.viewer.cam.elevation = -30
         self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1


    def get_images(self):
        img = self.sim.render(700, 700, mode='offscreen', camera_name="camera_images")
        img = resize(img, (84,84))
        #img = rotate(img, 180)
        return img

    def step(self, action):
        #action = action / np.linalg.norm(action)
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

        #print(action)

        #import IPython
        #IPython.embed()
        # The marker seems to get reset every time you do a simulation
        #self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_rewards(action, ob)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, dict()

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        f = self.get_fingers_com()
        flat_obs = np.concatenate((e, b, f))

        #print(flat_obs)

        return flat_obs
        """return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )"""

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        obj_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obj_distance = np.linalg.norm(obj_goal - self.get_obj_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_obj_pos()
        )
        return dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            hand_and_obj_distance=hand_distance+obj_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        return np.concatenate((self.data.get_body_xpos('obj').copy(),
                               self.data.get_body_xpos('obj2').copy(),
                               self.data.get_body_xpos('obj3').copy(),
                               self.data.get_body_xpos('obj4').copy()))

    def get_fingers_com(self):
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        return  (rightFinger + leftFinger)/2


    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )


    def _set_objs_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):

        ## Add reset with goals (just at the beginning)
        self._reset_hand()

        self.goal_data = self.all_goals[self.choice]
        self.blockchoice = 0
        self._state_goal = goal_pos = self.goal_data[0:2]

        blockpositions = self.goal_data[2:]

        body_pos = self.sim.model.body_pos.copy()
        self.sim.model.body_pos[-1][0] = goal_pos[0]
        self.sim.model.body_pos[-1][1] = goal_pos[1]
        #self.sim.model.body_pos[-5:-1][:, 0] = blockpositions[0::2]
        #self.sim.model.body_pos[-5:-1][:, 1] = blockpositions[1::2]


        for i in range(0,int(len(blockpositions)/2)):
            #print(i)
            pos_3d = np.zeros(3)
            pos_3d[0:2] = blockpositions[2*i:2*i+2]
            pos_3d[2] = 0.02
            self._set_obj_xyz(pos_3d,i)

        #import IPython
        #IPython.embed()

        #print(blockpositions)


        self.curr_path_length = 0

        return self._get_obs()

    def _set_obj_xyz(self, pos, i):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        #import IPython
        #IPython.embed()

        qpos[9+i*7:12+i*7] = pos.copy()
        qvel[9+i*7:15+i*7] = 0
        self.set_state(qpos, qvel)

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))
        self._set_objs_xyz(state_goal[3:])
        self.sim.forward()




    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obs):

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')

        placingGoal = self._state_goal[:3]
        objPos = self.get_body_com("obj")

        placingDist = np.linalg.norm(objPos[0:2] - placingGoal)

        if self.sparse == 'no':
            fingerCOM = (rightFinger + leftFinger)/2

            graspDist = np.linalg.norm(objPos - fingerCOM)
            graspRew = -graspDist

            reward = graspRew - 10 * placingDist # + pickRew
        else:

            if np.linalg.norm(placingDist) < self.radius_reward:
                reward = -placingDist + self.radius_reward
            else:
                reward = 0.0

        return reward

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        # for stat_name in [
        #     'hand_distance',
        #     'obj_distance',
        #     'hand_and_obj_distance',
        #     'touch_distance',
        #     'hand_success',
        #     'obj_success',
        #     'hand_and_obj_success',
        #     'touch_success',
        # ]:
        #     stat_name = stat_name
        #     stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        #     statistics.update(create_stats_ordered_dict(
        #         '%s%s' % (prefix, stat_name),
        #         stat,
        #         always_show_all_stats=True,
        #     ))
        #     statistics.update(create_stats_ordered_dict(
        #         'Final %s%s' % (prefix, stat_name),
        #         [s[-1] for s in stat],
        #         always_show_all_stats=True,
        #     ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }


    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(batch_size, self.hand_and_obj_space.low.size),
            )


if __name__ == "__main__":
    for n in range(10):
        env = SawyerPusherEnv(task=n)
        env.reset()
        t = 0
        while t<1:
            env.step(np.zeros(3))
            #env.render()
            import IPython
            IPython.embed()

            img = env.get_images()
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.savefig('task-'+str(n))
            #print('TEST!!!')
            t += 1
        # env.render()
        #import IPython
        #IPython.embed()
