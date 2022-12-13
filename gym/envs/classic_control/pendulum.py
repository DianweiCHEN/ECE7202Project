import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import pi
from os import path
import math
from gym.envs.classic_control import rendering
import logging
import numpy as np
from scipy.stats import multivariate_normal
# import gym
# from gym import spaces
from gym.utils import seeding
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import linalg as LA
import gym
from gym.envs.classic_control import rendering
from matplotlib import animation
import matplotlib.pyplot as plt
import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # <<<<<<<<<<<<<<<<<<<<
seed = 24
# torch.manual_seed(seed)
np.random.seed(seed)

class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        #initialize the physical environment
        self.state = None
        n_towers=5
        self.randtable=np.random.rand(n_towers, 2000, 2000)
        np.save("randtable.npy",self.randtable)
        self.n_towers = n_towers

        omega = 0.1
        vel = 50
        radian = vel/omega

        angle = 1-2*np.random.rand()
        rx = radian*cos(pi*angle)
        ry = radian*sin(pi*angle)
        vx = vel*cos(pi/2+pi*angle)
        vy = vel*sin(pi/2+pi*angle)
        self.x = np.transpose(np.array([rx, ry, vx, vy]))

        n_buildings=20
        self.n_buildings=n_buildings
        self.sigma_bias=50000*np.array([[1, 0.6], [0.6, 2]])
        self.r_building=2000*np.random.rand(n_towers, 2, n_buildings)-1000
        np.save("r_building.npy", self.r_building)

        x_min=-1000
        x_max=1000
        y_min = -1000
        y_max=1000
        vx_min=-15
        vx_max=15
        vy_min=-15
        vy_max=15
        state_low=np.transpose(np.array([x_min,y_min, vx_min, vy_min]))
        state_high = np.transpose(np.array([x_max, y_max, vx_max, vy_max]))

        self.x_min=-1000
        self.x_max=1000

        # dynamics model: 1= turn -1= straight
        self.dynamics=1
        # define two dynamics models
        # turning

        omega=0.1
        s_w = 0.01

        ts=0.2

        self.q_mx_circle=s_w*np.array([[2*(omega * ts - sin(omega * ts)) / omega ** 3, 0, (1 - cos(omega * ts))/omega**2, (omega * ts - sin(omega * ts))/omega ** 2],
                    [0, 2 * (omega * ts - sin(omega * ts)) / omega ** 3, -(omega*ts-sin(omega*ts))/omega ** 2, (1-cos(omega*ts))/omega **2],
                    [(1-cos(omega * ts))/omega ** 2, -(omega*ts-sin(omega*ts))/omega ** 2, ts,0],
                    [(omega*ts-sin(omega*ts))/omega ** 2, (1-cos(omega*ts))/omega ** 2,0, ts]])
        self.f_mx_circle=np.array([[1,0, sin(omega*ts)/omega, (cos(omega*ts)-1)/omega],
           [0,1,(1-cos(omega*ts))/omega,sin(omega*ts)/omega],
           [0,0, cos(omega*ts),        -sin(omega*ts)],
           [0,0,sin(omega*ts),        cos(omega*ts)]])

        #straight line
        q=0.1
        self.q_mx_straight= q*np.array([[1/3*ts**3, 0, 0.5*ts**2, 0],
                                 [0, 1/3*ts**3, 0, 0.5*ts**2],
                                 [0.5*ts**2, 0, ts,  0],
                                 [0,  0.5*ts**2, 0,  ts]])
        self.f_mx_straight=np.array([[1, 0, ts, 0],
                                     [0, 1, 0, ts],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])

        sigma2_sm = 100

        self.r_mx = sigma2_sm*np.eye(n_towers, dtype = np.float)


        #initialize EKF
        self.p_mx = 10*np.diag([10, 10, 1, 1])
        self.xhat=self.x+np.dot(np.sqrt(self.p_mx), np.random.randn(4))

        #initialize benchmark ekf

        # self.p_mx2=self.p_mx
        # self.xhat2=self.xhat

        # Define action range
        self.max_bias = 30
        self.min_bias = 0

        self.dt = ts
        self.viewer = None

        self.action_space = spaces.Box(
            low=-self.max_bias*np.ones((n_towers,1 )), high=self.max_bias*np.ones((n_towers,1)), shape=(n_towers,1), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        self.done=False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        n_states=4
        x = self.x
        dynamics=self.dynamics
        # dynamics change probability
        p=0

        if np.random.rand()<p:
            dynamics=-dynamics

        if dynamics==1:
            q_mx=self.q_mx_circle
            f_mx=self.f_mx_circle
        else:
            q_mx=self.q_mx_straight
            f_mx=self.f_mx_straight

        r_mx=self.r_mx
        f_mx = np.asarray(f_mx)
        # system state propagation
        x_next=f_mx.dot(x)+0*np.random.multivariate_normal(np.zeros(n_states), q_mx)
        self.x=x_next

        self.state=x_next

        xhat, bias_true,y_tilde=self._get_obs(f_mx, q_mx, u)
        # reward=-abs(sum((y_tilde)))
        # reward=-np.sum(np.sqrt(np.diag(self.p_mx)[0:2]))
        reward=-sum(abs(bias_true-u))

        x_min=self.x_min
        x_max = self.x_max

        if x_min<x[0]<x_max and x_min<x[1]<x_max and x_min<xhat[0]<x_max and x_min<xhat[1]<x_max :
            self.done=False
        else:
            self.done=True

        # print(self.xhat - self.x)
        return xhat, reward, self.done, {}, bias_true, x


    def reset(self):

        # self.state = self.np_random.uniform(low=-high, high=high)
        omega = 0.1
        vel = 50
        radian = vel/omega

        angle = 1-2*np.random.rand()
        rx = radian*cos(pi*angle)
        ry = radian*sin(pi*angle)
        vx = vel*cos(pi/2+pi*angle)
        vy = vel*sin(pi/2+pi*angle)
        self.x = np.transpose(np.array([rx, ry, vx, vy]))
        self.dynamics = 1
        self.p_mx = np.diag([10, 10, 1, 1])
        self.xhat = self.x + np.dot(np.sqrt(self.p_mx), np.random.randn(4))

        # reset benchmark ekf
        # self.p_mx2=self.p_mx
        # self.xhat2=self.xhat
        # print(self.xhat-self.x)

        return self.xhat, self.x

    def _get_obs(self, f_mx,q_mx, action):
        n_towers=self.n_towers


        #tower postions
        rs1_0 = np.array([-810, 240]).transpose()
        rs2_0 = np.array([-550, 640]).transpose()
        rs3_0 = np.array([615, -800]).transpose()
        rs4_0 = np.array([-250, -105]).transpose()
        rs5_0 = np.array([330, 900]).transpose()
        rs = np.column_stack((rs1_0, rs2_0, rs3_0, rs4_0, rs5_0))

        p_mx=self.p_mx

        r_mx=self.r_mx

        # observation is the state estimates from EKF
        x=self.x
        n_states=x.size

        bias=self.get_bias()
        # mesasurement
        # print(type(meas_func(x,rs)),type(bias))
        # test1 = meas_func(x,rs)
        # test2 = np.random.multivariate_normal(np.zeros(n_towers), r_mx)
        z= meas_func(x,rs)+bias+np.random.multivariate_normal(np.zeros(n_towers), r_mx)
        xhat_current=self.xhat

        x_pred = f_mx.dot(xhat_current)
        p_mx_pred=f_mx.dot(p_mx).dot(f_mx.transpose())+q_mx
        y_hat=meas_func(x_pred,rs)
        y_tilde=z-action-y_hat

        h_mx=meas_jacobian(x_pred, rs)
        s_mx=h_mx.dot(p_mx).dot(h_mx.transpose())+r_mx
        k_mx=p_mx.dot(h_mx.transpose()).dot(LA.inv(s_mx))

        xhat=x_pred+k_mx.dot(y_tilde)
        p_mx_upd=(np.eye(n_states) -k_mx.dot(h_mx)).dot(p_mx_pred).dot(np.transpose(np.eye(n_states) -k_mx.dot(h_mx)))+k_mx.dot(r_mx).dot(k_mx.transpose())


        # ------------------benchmark ---------------------

        # p_mx2 = self.p_mx2
        # xhat_current2 = self.xhat2
        #
        # x_pred2 = f_mx.dot(xhat_current2)
        # p_mx_pred2 = f_mx.dot(p_mx2).dot(f_mx.transpose()) + q_mx
        # y_hat2 = meas_func(x_pred2, rs)
        # y_tilde2 = z - y_hat2
        # h_mx2 = meas_jacobian(x_pred2, rs)
        # s_mx2 = h_mx2.dot(p_mx2).dot(h_mx2.transpose()) + r_mx
        # k_mx2 = p_mx2.dot(h_mx2.transpose()).dot(LA.inv(s_mx2))
        #
        # xhat2 = x_pred2 + k_mx2.dot(y_tilde2)
        # p_mx_upd2 = (np.eye(n_states) - k_mx2.dot(h_mx2)).dot(p_mx_pred2).dot(
        #     np.transpose(np.eye(n_states) - k_mx2.dot(h_mx2))) + k_mx2.dot(r_mx).dot(k_mx2.transpose())

        self.p_mx=p_mx_upd
        self.xhat=xhat

        # self.p_mx2=p_mx_upd2
        # self.xhat2=xhat2
        return xhat, bias, y_tilde

    def get_bias(self):
        x=self.x
        n_buildings=self.n_buildings
        n_towers=self.n_towers
        sigma_bias=self.sigma_bias
        r_building=self.r_building
        randtable=self.randtable
        bias=np.zeros(n_towers)

        for i_tower in range(n_towers):
            for i_building in range(n_buildings):

                # bias[i_tower] += 1-2*np.random.rand()+15*1000000*multivariate_normal.pdf(x[0:2], mean=r_building[i_tower,:,i_building], cov=sigma_bias)


                bias[i_tower] = bias[i_tower]+1000000 * multivariate_normal.pdf(x[0:2],

                                                                                                   mean=r_building[
                                                                                                        i_tower, :,
                                                                                                        i_building],
                                                                                                  cov=sigma_bias)
            # x_index = 1000 + math.ceil(min(999, x[0]))
            # y_index = 1000 + math.ceil(min(999, x[1]))
            # bias[i_tower] += 1 - 2 * randtable[i_tower, x_index, y_index]
            # print(bias)
            if bias[i_tower] >30:
                bias[i_tower]=30

        return bias

    
    def render(self, s,x, mode="human", close=False):
        # close()
        # if self.viewer is None:
        carwidth = 10
        carlength = 10
        rs1_0 = np.array([-810, 240]).transpose()
        rs2_0 = np.array([-550, 640]).transpose()
        rs3_0 = np.array([615, -800]).transpose()
        rs4_0 = np.array([-250, -105]).transpose()
        rs5_0 = np.array([330, 900]).transpose()
        rs = np.column_stack((rs1_0, rs2_0, rs3_0, rs4_0, rs5_0))
        # cary = 300
        screen_width = 2300
        screen_height = 2300
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # self.score_label = pyglet.text.Label(
            #     "0000",
            #     font_size=50,
            #     x=1250,
            #     y=1250,
            #     anchor_x="left",
            #     anchor_y="center",
            #     color=(255, 255, 255, 255),
            # )
            # linelength1 = rendering.line((self.last_xcar
            # l, r, t, b = -carlength / 2, carlength / 2, carwidth / 2, -carwidth / 2
            # car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            ped1 = rendering.make_circle(10)
            ped2 = rendering.make_circle(10)
            # ped3 = rendering.make_circle(10)
            T1 = rendering.make_circle(30)
            T2 = rendering.make_circle(30)
            T3 = rendering.make_circle(30)
            T4 = rendering.make_circle(30)
            T5 = rendering.make_circle(30)
            # self.cartrans = rendering.Transform()/
            self.ped1trans = rendering.Transform()
            self.ped2trans = rendering.Transform()
            # self.ped3trans = rendering.Transform()
            self.t1trans = rendering.Transform()
            self.t2trans = rendering.Transform()
            self.t3trans = rendering.Transform()
            self.t4trans = rendering.Transform()
            self.t5trans = rendering.Transform()

            # car.add_attr(self.cartrans)
            ped1.add_attr(self.ped1trans)
            ped2.add_attr(self.ped2trans)
            # ped3.add_attr(self.ped3trans)
            T1.add_attr(self.t1trans)
            T2.add_attr(self.t2trans)
            T3.add_attr(self.t3trans)
            T4.add_attr(self.t4trans)
            T5.add_attr(self.t5trans)

            # car.set_color(255, 0, 0)
            ped1.set_color(0, 0, 255)
            ped2.set_color(255, 0, 0)
            # ped3.set_color(0, 255, 0)
            # self.viewer.add_geom(car)
            self.viewer.add_geom(ped1)
            self.viewer.add_geom(ped2)
            self.viewer.add_geom(ped3)
            self.viewer.add_geom(T1)
            self.viewer.add_geom(T2)
            self.viewer.add_geom(T3)
            self.viewer.add_geom(T4)
            self.viewer.add_geom(T5)

            # carx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # 设置平移属性
        
        if self.state is None: return None

        s1 = s
        x1 = x
        # s2= xhat2
        # carx = (s[0]-475) * 100 + 50  # MIDDLE OF CART
        # cary = s[1] * 100 + 1050

        ped1x = s1[0]*1.2+1150
        # print(s[0]-x[0])
        ped1y = s1[1]*1.2+1150
        ped2x = x1[0]*1.2+1150
        ped2y = x1[1]*1.2+1150
        # ped3x = s2[0] * 1.2 + 1150
        # ped3y = s2[1] * 1.2 + 1150

        # 设置平移属性
        # self.trackc = rendering.Line((50, 1050), (carx, cary))
        # self.trackc.set_color(0, 0, 0)
        # self.viewer.add_geom(self.trackc)
        # self.trackp = rendering.Line((2050, 550), (pedx, pedy))  # Rl
        # self.trackp.set_color(0, 0, 0)
        # self.viewer.add_geom(self.trackp)
        # self.cartrans.set_translation(carx, cary)
        self.ped1trans.set_translation(ped1x, ped1y)
        self.ped2trans.set_translation(ped2x, ped2y)
        # self.ped3trans.set_translation(ped3x, ped3y)
        self.t1trans.set_translation(rs1_0[0]*1.2+1150, rs1_0[1]*1.21150)
        self.t2trans.set_translation(rs2_0[0]*1.2+1150, rs2_0[1]*1.2+1150)
        self.t3trans.set_translation(rs3_0[0]*1.2+1150, rs3_0[1]*1.2+1150)
        self.t4trans.set_translation(rs4_0[0]*1.2+1150, rs4_0[1]*1.2+1150)
        self.t5trans.set_translation(rs5_0[0]*1.2+1150, rs5_0[1]*1.2+1150)
        # self.poletrans.set_rotation(-x[2])
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        # self.score_label.text = "5555"
        # self.score_label.draw()

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

        # Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)

def meas_func(x,rs):

    n_towers=rs.shape[1]
    h=LA.norm(rs-np.tile(x[0:2], (n_towers,1)).transpose(), axis = 0)
    return h

def meas_jacobian(x,rs):
    n_states=x.size
    n_towers = rs.shape[1]
    h_mx=np.zeros([n_towers, n_states])

    h_mx[:,0]=np.divide(x[0]-rs[0,:],  LA.norm(rs-np.tile(x[0:2], (n_towers,1)).transpose(), axis = 0))
    h_mx[:, 1] = np.divide(x[1] - rs[1, :], LA.norm(rs - np.tile(x[0:2], (n_towers, 1)).transpose(), axis=0))
    return h_mx
