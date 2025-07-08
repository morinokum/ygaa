
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import argparse
import os
import csv
from datetime import datetime

# REINFORCEアルゴリズムの実装
class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.policy_network = self._build_policy_network()

    def _build_policy_network(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def choose_action(self, state):
        state = np.array(state).reshape(1, -1) # 状態をモデルの入力形式に変換
        prob_dist = self.policy_network.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=prob_dist)
        return action, prob_dist[action]

    def learn(self, rewards, log_probs):
        # 割引報酬の計算
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.append(cumulative_reward)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)

        # 報酬の正規化 (学習の安定化のため)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        # ポリシーネットワークの更新
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(tf.convert_to_tensor(log_probs) * tf.convert_to_tensor(discounted_rewards, dtype=tf.float32))
        
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

def train_reinforce_cartpole(episodes, learning_rate, gamma, output_path, log_file):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size, learning_rate, gamma)

    episode_rewards = []

    print(f"--- CartPole REINFORCE学習を開始します (episodes: {episodes}, learning_rate: {learning_rate}, gamma: {gamma}) ---")

    for e in range(episodes):
        state, _ = env.reset()
        done = False
        rewards = []
        log_probs = []

        while not done:
            action, log_prob = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(tf.math.log(log_prob)) # log(prob) を保存
            state = next_state
            done = done or truncated

        agent.learn(rewards, log_probs)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if (e + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"エピソード: {e+1}/{episodes}, 平均報酬 (過去10エピソード): {avg_reward:.2f}")
        
        # CartPole-v1 の成功基準は200エピソードの平均報酬が195以上
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195:
            print(f"環境解決！エピソード {e+1} で平均報酬が195に到達しました。")
            break

    env.close()
    print("--- 学習が完了しました ---")

    # モデルの保存
    if output_path:
        print(f"--- 学習済みモデルを保存中: {output_path} ---")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        agent.policy_network.save(output_path)
        print("--- 保存が完了しました ---")

    # 結果のロギング
    if log_file:
        print(f"--- 実験結果を記録中: {log_file} ---")
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'episodes', 'learning_rate', 'gamma', 'avg_last_100_rewards', 'solved', 'output_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            avg_last_100_rewards = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            solved = "Yes" if avg_last_100_rewards >= 195 else "No"

            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'episodes': episodes,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'avg_last_100_rewards': f"{avg_last_100_rewards:.2f}",
                'solved': solved,
                'output_path': output_path
            })
        print("--- 記録が完了しました ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CartPole REINFORCE学習スクリプト')
    parser.add_argument('--episodes', type=int, default=1000, help='学習のエポック数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--gamma', type=float, default=0.99, help='割引率')
    parser.add_argument('--output_path', type=str, default=None, help='学習済みモデルの保存先パス')
    parser.add_argument('--log_file', type=str, default=None, help='実験結果の記録用CSVファイル')
    args = parser.parse_args()

    train_reinforce_cartpole(episodes=args.episodes, learning_rate=args.learning_rate, gamma=args.gamma, output_path=args.output_path, log_file=args.log_file)
