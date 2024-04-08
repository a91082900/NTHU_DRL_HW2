import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import cv2
    
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions=12):
        super(DuelingDQN, self).__init__()
        print(f"using Dueling DQN")
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.calculate_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.fc_adv = nn.Linear(512, n_actions)
        self.fc_val = nn.Linear(512, 1)

    def calculate_conv_output(self, shape):
        return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        q = val + adv - adv.mean(1, keepdim=True)
        # print("val", val)
        # print("adv", adv)
        return q

def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # print(obs.shape)
    return obs

class Agent:
    def __init__(self):
        self._input_shape = (4, 84, 84)
        self._device = "cpu"
        self._target_net = DuelingDQN(self._input_shape, 12).to(self._device)
        self._learning_net = DuelingDQN(self._input_shape, 12).to(self._device)
        self.model_num = 0
        self._frames = deque(maxlen=self._input_shape[0])
        self._eval_epsilon = 0.3
        self._skip_frame = 4
        self._skip_count = 0
        self._prev_action = 0
        self.load_model()

    def act(self, obs):
        if self._skip_count % self._skip_frame == 0:
            obs = preprocess(obs)
            while len(self._frames) < self._input_shape[0] - 1:
                self._frames.append(obs)
            self._frames.append(obs)
            if np.random.rand() < self._eval_epsilon:
                obs = torch.from_numpy(np.array(self._frames) / 255).float().unsqueeze(0).to(self._device)
                self._prev_action = self._learning_net(obs).argmax().item()
            else:
                self._prev_action = np.random.choice(np.arange(5)) # only sample NOOP and right actions
        self._skip_count += 1
        return self._prev_action

    def load_model(self):
        try:
            filename = f"109062212_hw2_data"
            assert os.path.exists(filename)
            self._learning_net.load_state_dict(torch.load(filename))
            print(f"Model loaded: {filename}")
            self.model_num += 1
        except:
            print("No model loaded")
        self._target_net.load_state_dict(self._learning_net.state_dict())