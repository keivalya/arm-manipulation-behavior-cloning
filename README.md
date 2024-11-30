# Robotic Arm Manipulation with Human Experience and Hierarchical Reinforcement Learning

**Franka Robotic Arm Manipulation using Demonstrations by Humans**
by [Keivalya Pandya](www.keivalya.com)

---

### **Soft Actor-Critic: The Big Picture**

SAC is a reinforcement learning algorithm that trains an agent to act optimally in continuous action spaces, such as controlling a robot arm or navigating a drone. In the code:
- The environment is **FrankaKitchen-v1**, where the agent completes tasks like opening a cabinet.
- The agent optimizes its policy using the **Soft Actor-Critic (SAC)** algorithm.
- The algorithm prioritizes **reward maximization** while **encouraging exploration** via entropy.

---

### **How SAC Works in This Code**

SAC involves three key networks:
1. **Actor (Policy)**: Learns which actions to take in a given state to maximize reward.
2. **Critics (Q-value estimators)**: Evaluate how good a given action is in a particular state.
3. **Target Critic**: Provides stable Q-value targets for training the critics.

The overall flow can be broken into **three phases**.

---

### **Phase 1: Initialization**

1. **Set Up Environment**:
   - The environment is created (`gym.make`), and a wrapper processes observations for compatibility.

2. **Agent Initialization**:
   - **Actor**:
     - Learns a policy represented as a probability distribution.
     - Outputs:
       - **Mean** and **log standard deviation** of action distributions.
       - Ensures exploration via stochastic sampling.
   - **Critics**:
     - Two independent networks (Q1 and Q2) estimate action values for stability (avoids overestimation bias).
   - **Target Critic**:
     - Initially copies the weights of the Critic and updates slowly to ensure stable targets.

3. **Replay Buffer**:
   - Stores past experiences (`state`, `action`, `reward`, `next_state`, `done`).
   - Enables efficient learning by reusing past experiences.

4. **Loading Expert Data**:
   - In Phase 1, the agent leverages human demonstration data (`human_memory.npz`) to jumpstart training.

---

### **Phase 2: Training Loop**

The **core training happens in three stages** with decreasing reliance on expert data:

#### **Step 1: Interaction with the Environment**
- The agent uses the Actor to:
  - Sample an action based on the current policy.
  - Observe the resulting next state, reward, and whether the episode ends.
- The transition (`state, action, reward, next_state`) is stored in the Replay Buffer.

#### **Step 2: Sampling from the Replay Buffer**
- The agent randomly samples a batch of transitions to train itself, ensuring diverse learning.

#### **Step 3: Critic Updates**
- The Critics learn to predict Q-values, which represent the expected reward for a state-action pair.
- **Target Q-value computation**:
  - Uses the **Target Critic** to estimate future rewards for `next_state`.
  - Incorporates the current reward and a discount factor (`gamma`) to compute the target:
    $`Q_{\text{target}} = r + \gamma \cdot (1 - \text{done}) \cdot \min(Q_1', Q_2') - \alpha \cdot \text{log\_prob}`$
  - The entropy term ($`\alpha \cdot \text{log\_prob}`$) encourages exploration by penalizing deterministic policies.
- **Critic Loss**:
  - Compares the predicted Q-values ($`Q_1, Q_2`$) to the computed target Q-value using Mean Squared Error.

#### **Step 4: Actor Updates**
- The Actor improves its policy to maximize the Q-values predicted by the critics.
- Actor Loss:
  - Encourages actions that:
    - Maximize Q-values ($`\min(Q_1, Q_2)`$).
    - Maintain high entropy (exploration).

#### **Step 5: Target Critic Updates**
- The Target Critic's weights are **soft-updated**:
  ```math
  \theta_{\text{target}} \gets \tau \cdot \theta + (1 - \tau) \cdot \theta_{\text{target}}
  ```
- Ensures smoother, more stable training.

#### **Step 6: Logging and Checkpointing**
- TensorBoard logs:
  - Critic loss, Actor loss, and rewards.
- Saves checkpoints to allow resuming training later.

---

### **Phase 3: Gradual Transition to Full Autonomy**

The agent is trained in **three phases**:
1. **Phase 1**: High reliance on expert data:
   - **Expert data ratio = 50%**.
   - Balances learning from the replay buffer and human-provided data.
2. **Phase 2**: Reduced expert reliance:
   - **Expert data ratio = 25%**.
   - Encourages the agent to learn more from its own exploration.
3. **Phase 3**: Full autonomy:
   - **Expert data ratio = 0%**.
   - The agent learns purely from its own experience.

---

### **Why Each Component Matters**

1. **Actor (Policy)**:
   - Learns how to act optimally by maximizing rewards while maintaining exploration.

2. **Critics (Q-Values)**:
   - Evaluate the quality of actions taken by the policy.
   - Two critics reduce overestimation bias.

3. **Replay Buffer**:
   - Ensures sample efficiency by reusing past experiences.
   - Decorrelation: Helps prevent learning from sequentially correlated data.

4. **Entropy Regularization**:
   - Encourages exploration, preventing premature convergence to suboptimal strategies.

5. **Target Networks**:
   - Provide stable targets for critic training, avoiding instability caused by rapidly changing Q-values.

6. **Expert Data**:
   - Jumpstarts training by introducing good behaviors early on, especially useful in complex tasks like robotics.

---

### **Summary of Training Flow**

1. **Initialize environment, agent, and replay buffer.**
2. **Phase 1 (Exploration with Expert Data):**
   - Train using a mix of expert and self-collected data.
3. **Phase 2 (Reduced Expert Reliance):**
   - Gradually shift focus to agent-collected experiences.
4. **Phase 3 (Full Autonomy):**
   - Train entirely on self-collected experiences.
5. **For Each Episode**:
   - Interact with the environment.
   - Store experiences in the replay buffer.
   - Periodically sample experiences to:
     - Update Critics using target Q-values.
     - Update Actor using learned Q-values and entropy regularization.
6. **Log metrics and save model checkpoints.**

---

### Environment setup:
- MacOS Sequoia 15.1.1
- Python 3.11.9
- required installation is mentioned in `requirements.txt`

