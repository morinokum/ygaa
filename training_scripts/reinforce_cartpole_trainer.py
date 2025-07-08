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
        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32) # Add batch dimension
        action_probs = self.policy_network(state_tensor) # Get probabilities from the policy network
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0].numpy() # Sample action
        return action, action_probs[0, action] # Return action and its probability tensor

    def learn(self, states, actions, rewards):
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
            # 状態をテンソルに変換
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            # ポリシーネットワークを再実行して行動確率を取得
            action_probs = self.policy_network(states_tensor)
            # 選択された行動の対数確率を計算
            action_indices = tf.convert_to_tensor(actions, dtype=tf.int32)
            indices = tf.stack([tf.range(tf.shape(action_indices)[0]), action_indices], axis=1)
            log_probs = tf.math.log(tf.gather_nd(action_probs, indices))
            
            discounted_rewards_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
            
            loss = -tf.reduce_sum(log_probs * discounted_rewards_tensor)
        
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
        states_history = []
        actions_history = []
        rewards = []

        while not done:
            action, _ = agent.choose_action(state) # 確率テンソルはここでは不要
            next_state, reward, done, truncated, _ = env.step(action)
            
            states_history.append(state)
            actions_history.append(action)
            rewards.append(reward)
            
            state = next_state
            done = done or truncated

        agent.learn(states_history, actions_history, rewards) # 履歴を learn に渡す
        
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