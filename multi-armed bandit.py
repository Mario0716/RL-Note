import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """
    bernoulli multi-armed bandit
    """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)     #随机生成K个0~1的数，作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)   #获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] #最大的获奖概率
        self.K = K
    
    def step(self, k):
        """
        当玩家选择了k号拉杆后 根据拉动该老虎机的k号拉杆获得奖励的概率返回
        1   获奖
        0   未获奖
        """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


# 设定随机种子，使实验具有可重复性
# ? 通过设置相同的种子，可以确保每次运行程序时生成的随机数序列都是相同的。
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)

print("随机生成一个%d的伯努利老虎机" % K)
print("获奖概率最大的拉杆使%d号其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))



"""
Strategy of Multi-Armed Bandit
"""
class Solver:
    """
    the framework of multi-armed bandit
    """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)   #每根拉杆尝试的次数
        self.regret = 0     #当前步的累计懊悔
        self.actions = []   #list   save action
        self.regrets = []   #list   save regret
    
    def update_regret(self, k):
        """
        calculate and save cumulative regret, k is the number of action bandit 
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        """
        return the number of action bandit
        """
        raise NotImplementedError

    def run(self, num_steps):
        """
        num_steps: the total number epoch
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


"""
epsilon-greedy algorithm
"""
class EpsilonGreedy(Solver):
    """
    son of solver class :)
    """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # initialize reward estimation
        self.estimates = np.array([init_prob] * self.bandit.K)
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # choose one bandit randomly
        else:
            k = np.argmax(self.estimates)   # choose the best reward estimation bandit
        
        r = self.bandit.step(k)     # get reward of this action
        # ?
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    

# plot results
def plot_results(solvers, solver_names):
    """
    plot image of regret changing with time
    solver: list, each mab strategy
    solver_names:list, name of strategy
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-greedy algorithm cumulative regrets: ', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# hyperparameter test trick
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# epsilon-decaying greedy algorithm
class DecayingEpsilonGreedy(Solver):
    """
    epsilon decays by time
    """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        
        return k

np.random.seed(1)
decaying_epsilon_greedy_slover = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_slover.run(5000)
print('epsilon-decaying greedy algorithm cumulative regrets: ', decaying_epsilon_greedy_slover.regret)
plot_results([decaying_epsilon_greedy_slover], ['DecayingEpsilonGreedy'])


"""
upper confidence bound
"""
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))) # calculate upper confidence bound
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(1)
coef = 1
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('upper confidence bound cumulative regret: ', UCB_solver.regret)
plot_results([UCB_solver], ['UCB'])

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)    # list,  number of reward = 1 for each bandit
        self._b = np.ones(self.bandit.K)    # list,  number of reward = 0 for each bandit
    
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # sampling by beta-distribution  
        k = np.argmax(samples)      # select best reward bandit
        r = self.bandit.step(k)

        self._a[k] += r # update 1st param of beta-distribution
        self._b[k] += (1 - r)   # update 2nd param of beta
        return k

np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('thompson sampling algorithm regret:', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ['ThompsonSampling'])
