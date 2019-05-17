import numpy as np
from math import pi, asin, sqrt, cos, fabs
from gym.utils import seeding
import gym

import PIL
from PIL import Image
from PIL import ImageDraw

## MY ENV STARTS HERE ##
def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    #theta = (np.pi / 180.0) * angle
    theta = angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


class PendulumPixelsEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3))
        self.seed()
        self.state = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, test=True):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        if test:
            costs = 1-np.abs(angle_normalize(th))/np.pi
        else:
            costs = - (angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2))

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        #return self._get_obs(), -costs, False, {}
        return self.render("rgb_array"), costs, False, {}

    def reset(self):
        high = np.array([np.pi/2, 2])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self.render("rgb_array")

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        img = Image.new('RGB', (500,500), (255, 255, 255))

        # Draw a rotated rectangle on the image.
        draw = ImageDraw.Draw(img)
        rect = get_rect(x=250-10*np.sin(self.state[0]+np.pi/2), y=250-10*np.cos(self.state[0]+np.pi/2), width=120, height=20, angle=self.state[0]+np.pi/2)
        draw.polygon([tuple(p) for p in rect], fill=(255,0,0))
        draw.ellipse([(250-10, 250-10),(250+10, 250+10)], fill=(0,0,0))
        # Convert the Image data to a numpy array.
        new_data = np.asarray(img)
        #plt.imshow(new_data)
        #plt.show()
        frame = Image.fromarray(new_data[120:-120,120:-120,:])
        frame = np.array(frame.resize([64,64], resample=PIL.Image.BILINEAR))

        return frame
        #return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)