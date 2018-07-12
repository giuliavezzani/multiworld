from collections import OrderedDict
import numpy as np
import pickle
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from skimage.transform import rescale, resize, downscale_local_mean, rotate


class SawyerPickPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(self,obj_low=None,obj_high=None, use_reward='yes',sparse='yes',
            indicator_threshold=0.06,obj_init_pos=(0, 0.6, 0.02),fix_goal=True,
            fixed_goal= (0, 0.85, 0.02, 0.1) ,
            #3D placing goal, followed by height target for picking
            goal_low=None,goal_high=None,
            hide_goal_markers=False,
            task=0, radius_reward=0.1,
            file_goals=None,
            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        #if obj_low is None:
        #    obj_low = self.hand_low
        #if obj_high is None:
        #    obj_high = self.hand_high

        #if goal_low is None:
        #    goal_low = np.hstack((self.hand_low, obj_low))
        #if goal_high is None:
        #    goal_high = np.hstack((self.hand_high, obj_high))

        self.max_path_length = 150

        self.use_reward = use_reward
        self.sparse = sparse
        self.choice = task
        if file_goals == None:
            self.all_goals = np.zeros(10)
        else:
            self.all_goals = pickle.load(open(file_goals, "rb"))
        self.indicator_threshold = indicator_threshold
        self.radius_reward = radius_reward
        self.obj_init_pos = np.array(obj_init_pos)

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )
        self.observation_space = Dict([
            ('observation', self.hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),

            ('state_observation', self.hand_and_obj_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
        ])

        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
        #pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.3
         self.viewer.cam.elevation = -70
         self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1


    def get_images(self):
        img = self.sim.render(700, 700, mode='offscreen', camera_name="camera_images")
        img = resize(img, (84,84))
        img = rotate(img, 180)
        return img

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

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
        return ob, reward, done

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

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
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('goal'), 2] = (
                -1000
            )


    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):

        ## Add reset with goals (just at the beginning)
        self._reset_hand()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        init_obj = self.obj_init_pos

        heightTarget , placingGoal = self._state_goal[3], self._state_goal[:3]

        self.maxPlacingDist = np.linalg.norm([init_obj[0], init_obj[1], heightTarget] - placingGoal) + heightTarget

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def put_obj_in_hand(self):
        new_obj_pos = self.data.get_site_xpos('endeffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation(-1)
        self.do_simulation(1)
        self._set_obj_xyz(new_obj_pos)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))
        self._set_obj_xyz(state_goal[3:])
        self.sim.forward()

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


        # num_objs_in_hand = int(batch_size * p_obj_in_hand)

        # # Put object in hand
        # goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
        # goals[:num_objs_in_hand, 4] -= 0.01

        # # Put object one the table (not floating)
        # goals[num_objs_in_hand:, 5] = self.obj_init_pos[2]
        return {
            'state_desired_goal': goals,

        }



    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obs):

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')

        heightTarget = self._state_goal[3]
        placingGoal = self._state_goal[:3]

        objPos = self.get_body_com("obj")


        fingerCOM = (rightFinger + leftFinger)/2


        graspDist = np.linalg.norm(objPos - fingerCOM)
        graspRew = -graspDist


        def graspAttained():
            if graspDist <0.1:
                return True

            else:
                return False

        def pickCompletionCriteria():

            tolerance = 0.01

            if objPos[2] >= (heightTarget - tolerance):
                return True
            else:
                return False


        if pickCompletionCriteria():

            self.pickCompleted = True

        def pickReward():

            if self.pickCompleted and graspAttained():
                return 10*heightTarget

            elif (objPos[2]> 0.025) and graspAttained():

                return 10* min(heightTarget, objPos[2])

            else:
                return 0

        def placeReward():


            placingDist = np.linalg.norm(objPos - placingGoal)
            #print(placingDist)


            if self.pickCompleted and graspAttained():

                placeRew = max(100*(self.maxPlacingDist - placingDist),0)


                return placeRew

            else:
                return 0

        #pickRew = pickReward()
        placeRew = placeReward()
        reward = graspRew + placeRew # + pickRew

        return [reward]
        #returned in a list because that's how compute_reward in multiTask.env expects it

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


if __name__ == "__main__":
    env = SawyerPickPlaceEnv()
    env.reset()
    t = 0
    while t<10000:
        env.step(np.random.rand(3))
        env.render()
        #img = env.get_images()
        #print('TEST!!!')
        t += 1
    # env.render()
        #import IPython
        #IPython.embed()
