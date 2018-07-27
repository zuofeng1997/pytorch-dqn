from zuofengRL.deepq.replay_buffer import ReplayBuffer,PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
import visdom


class Brain(object):
    def __init__(self,env, model, max_timestep, buffer_size, learning_start,
          target_network_update_freq, batch_size, lr, env_name, double_q, dueling, prioritized, atari=False, use_cuda=True):
        self.prioritized = prioritized
        self.env = env
        self.model = model
        self.max_timestep = max_timestep
        self.buffer_size = buffer_size
        self.learning_start = learning_start
        self.target_network_update_freq = target_network_update_freq
        self.batch_size = batch_size
        self.lr = lr
        if self.prioritized:
            self.buffer = PrioritizedReplayBuffer(size=self.buffer_size, alpha=0.6)
        else:
            self.buffer = ReplayBuffer(size=self.buffer_size)
        if atari:
            self.policy_model = self.model(env.action_space.n)
            self.target_model = self.model(self.env.action_space.n)
        else:
            self.policy_model = self.model(64, self.env.observation_space.shape[0], env.action_space.n, dueling=dueling)
            self.target_model = self.model(64, self.env.observation_space.shape[0], self.env.action_space.n, dueling=dueling)
        if use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.policy_model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(), lr=self.lr)
        self.env_name = env_name
        self.double_q = double_q
        self.atari = atari

    def run_loop(self):
        vis = visdom.Visdom(env="mydqn")
        obs = self.env.reset()
        win = vis.line(
            Y=np.array([0]),
            X=np.array([0]),
            opts=dict(title=self.env_name,xlabel="steps",ylabel="mean_reward"))

        sum_reward = []
        eps_r = 0
        mean_rs = []
        for t in range(1, self.max_timestep):
            beta = 0.4 + (0.6/self.max_timestep)*t
            if t <= 0.1 * self.max_timestep:
                epsilon = 1.0 - (t / (0.1 * self.max_timestep)) * 0.9
            a = self.policy_model.choose_action(obs, epsilon, self.device)
            obs_, r, done, _ = self.env.step(a)

            eps_r += r

            self.buffer.add(obs, a, r, obs_, int(done))
            obs = obs_
            if done:
                sum_reward.append(eps_r)
                if len(sum_reward) == 100:
                    mean_r = np.mean(sum_reward)
                    print("time_step:", t, "mearn_reward:", mean_r)
                    mean_rs.append(mean_r)
                    vis.line(
                        Y=np.array([mean_r]),
                        X=np.array([t]),
                        win=win,
                        update="append",)
                    sum_reward = []
                eps_r = 0
                obs = self.env.reset()
            if t > self.learning_start:
                self.learn(beta)
            if t > self.learning_start and t % self.target_network_update_freq == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())

        self.env.close()
        print("saving model")
        torch.save(self.policy_model.state_dict(), 'cartpole.pkl')
        return mean_rs

    def learn(self, beta):
        if self.prioritized:
            experience = self.buffer.sample(self.batch_size, beta)
            (obs, acs, rs, obs_, ds, weights, batch_idxes) = experience
        else:
            train_data = self.buffer.sample(self.batch_size)
            obs, acs, rs, obs_, ds, = train_data
            weights, batch_idxes = np.ones_like(rs), None
        if self.atari:
            obs = obs.transpose(0, 3, 2, 1)
            obs_ = obs_.transpose(0, 3, 2, 1)
        obs = torch.from_numpy(obs).type(torch.float).to(self.device)
        acs = torch.from_numpy(acs).unsqueeze(1).to(self.device)
        rs = torch.from_numpy(rs).type(torch.float).view(-1, 1).to(self.device)
        obs_ = torch.from_numpy(obs_).type(torch.float).to(self.device)
        ds = torch.from_numpy(ds).to(self.device)
        with torch.no_grad():
            if self.double_q:
                max_a = self.policy_model(obs_).topk(k=1)[1]
                next_max_obs_value = self.target_model(obs_).gather(1, max_a)
            else:
                next_max_obs_value = self.target_model(obs_).topk(k=1, dim=1)[0]
            for i in range(ds.shape[0]):
                if ds[i].item() == 1:
                    next_max_obs_value[i, :] = 0

        target = next_max_obs_value + rs
        output = self.policy_model(obs).gather(1, acs)
        if self.prioritized:
            td_errors = np.abs((output-target).squeeze(1).cpu().detach().numpy())
            new_priorities = np.abs(td_errors) + 1e-6
            self.buffer.update_priorities(batch_idxes, new_priorities)
        # loss = F.smooth_l1_loss(output, target)
        loss = torch.mean(((output-target)**2).squeeze(1) * torch.tensor(weights, dtype=torch.float, device=self.device))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

