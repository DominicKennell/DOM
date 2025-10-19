import numpy as np
import random
import pygame
import sounddevice as sd

# --- Audio settings ---
SAMPLE_RATE = 22050  # Reduced sample rate for safety

def play_sound_async(sound_array):
    """Play a sound asynchronously without spawning extra threads."""
    sd.play(sound_array, samplerate=SAMPLE_RATE, blocking=False)

# --- Base pidgin signals (now sequence-ready for creole evolution) ---
base_pidgin_language = {
    "gather": [[400, 500]],
    "trade": [[500, 600]],
    "fight": [[700, 800]],
    "social": [[300, 400]],
    "reproduce": [[600, 700]]
}

def play_signal_sequence(signal_sequence, agent):
    """Generate and play a sequence of chords based on agent traits."""
    full_sound = np.zeros(0)
    for freqs in signal_sequence:
        duration = 0.2 + 0.2 * agent.rhythm_complexity
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        chord = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
        chord *= 0.25 + 0.25 * agent.harmony_preference
        full_sound = np.concatenate([full_sound, chord])
    play_sound_async(full_sound)

def decode_signal(freqs):
    """Infer action from average frequency."""
    avg = np.mean(freqs)
    if avg > 700:
        return "fight"
    elif avg > 600:
        return "reproduce"
    elif avg > 500:
        return "trade"
    elif avg > 400:
        return "gather"
    else:
        return "social"

def decode_signal_sequence(freq_sequence):
    """Decode a sequence of signals into actions."""
    return [decode_signal(freqs) for freqs in freq_sequence]

# --- Neural network ---
class NumpyNN:
    def __init__(self, input_size=12, hidden_size=16, output_size=5, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))
        # Meta-learning rates
        self.lr_W1 = np.ones_like(self.W1) * lr
        self.lr_b1 = np.ones_like(self.b1) * lr
        self.lr_W2 = np.ones_like(self.W2) * lr
        self.lr_b2 = np.ones_like(self.b2) * lr
        self.meta_rate = 0.001

    def forward(self, x):
        self.x = x.reshape(-1, 1)
        self.z1 = np.dot(self.W1, self.x) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2))
        self.a2 = exp_scores / np.sum(exp_scores)
        return self.a2.flatten()

    def train(self, action_idx, reward):
        dZ2 = self.a2.copy().reshape(-1, 1)
        dZ2[action_idx] -= 1
        dZ2 *= reward
        dW2 = np.dot(dZ2, self.a1.T)
        db2 = dZ2
        da1 = np.dot(self.W2.T, dZ2)
        dz1 = da1 * (1 - self.a1 ** 2)
        dW1 = np.dot(dz1, self.x.T)
        db1 = dz1

        self.W1 -= self.lr_W1 * dW1
        self.b1 -= self.lr_b1 * db1
        self.W2 -= self.lr_W2 * dW2
        self.b2 -= self.lr_b2 * db2

        # Meta-learning adjustments
        self.lr_W1 += self.meta_rate * dW1 * reward
        self.lr_b1 += self.meta_rate * db1 * reward
        self.lr_W2 += self.meta_rate * dW2 * reward
        self.lr_b2 += self.meta_rate * db2 * reward

        # Clip learning rates
        self.lr_W1 = np.clip(self.lr_W1, 1e-4, 0.1)
        self.lr_b1 = np.clip(self.lr_b1, 1e-4, 0.1)
        self.lr_W2 = np.clip(self.lr_W2, 1e-4, 0.1)
        self.lr_b2 = np.clip(self.lr_b2, 1e-4, 0.1)

# --- Environment ---
class Resource:
    def __init__(self, name, amount, regen_rate):
        self.name = name
        self.amount = amount
        self.regen_rate = regen_rate

    def regenerate(self):
        self.amount = min(100, self.amount + self.regen_rate)

class Environment:
    def __init__(self):
        self.resources = {
            "food": Resource("food", 500, 50),
            "energy": Resource("energy", 500, 30),
            "metal": Resource("metal", 300, 20)
        }

    def regenerate_resources(self):
        for res in self.resources.values():
            res.regenerate()

# --- Agent ---
class Agent:
    ACTIONS = ["gather","trade","fight","social","reproduce"]
    MAX_SEQUENCE_LENGTH = 2  # max signal sequence length for creole

    def __init__(self, name, parent=None):
        self.name = name
        self.food = 20
        self.energy = 20
        self.metal = 10
        self.alive = True
        self.strength = random.randint(1, 10)
        self.cooperation = random.uniform(0, 1)
        self.nn = NumpyNN()
        self.last_state = None
        self.last_action = None
        self.last_action_name = "None"
        self.memory = []
        self.MAX_MEMORY = 5
        self.trust = {}

        # Music traits
        if parent:
            self.melody_tendency = np.clip(parent.melody_tendency + random.uniform(-0.1,0.1),0,1)
            self.rhythm_complexity = np.clip(parent.rhythm_complexity + random.uniform(-0.1,0.1),0,1)
            self.harmony_preference = np.clip(parent.harmony_preference + random.uniform(-0.1,0.1),0,1)
            # Inherit and mutate signal sequences
            self.signal_map = {}
            for act, seqs in parent.signal_map.items():
                new_seqs = []
                for freqs in seqs:
                    mutated = [f + random.uniform(-2,2) for f in freqs]
                    new_seqs.append(mutated)
                self.signal_map[act] = new_seqs
        else:
            self.melody_tendency = random.uniform(0,1)
            self.rhythm_complexity = random.uniform(0,1)
            self.harmony_preference = random.uniform(0,1)
            # Initialize signal sequences
            self.signal_map = {act: seqs[:] for act, seqs in base_pidgin_language.items()}

    def get_state(self, env):
        state = [
            self.food/50, self.energy/50, self.metal/30,
            self.strength/10, self.cooperation,
            self.melody_tendency, self.rhythm_complexity, self.harmony_preference
        ]
        avg_reward = np.mean([m[2] for m in self.memory]) if self.memory else 0
        avg_trust = np.mean(list(self.trust.values())) if self.trust else 0
        state.extend([avg_reward, avg_trust, len(self.memory)/self.MAX_MEMORY, random.random()])
        return np.array(state, dtype=float)

    def choose_action(self, env):
        state = self.get_state(env)
        probs = self.nn.forward(state)
        action_idx = np.random.choice(len(probs), p=probs)
        self.last_state = state
        self.last_action = action_idx
        self.last_action_name = self.ACTIONS[action_idx]
        return self.last_action_name

    def learn(self, reward):
        if self.last_state is not None and self.last_action is not None:
            self.nn.train(self.last_action, reward)
            self.memory.append((self.last_state, self.last_action, reward))
            if len(self.memory) > self.MAX_MEMORY:
                self.memory.pop(0)
            self.last_state = None
            self.last_action = None

    def perceive_music(self, other_agents):
        for other in other_agents:
            if other == self or not other.alive: continue
            signal_seq = other.signal_map.get(other.last_action_name, [[200]])
            play_signal_sequence(signal_seq, other)
            perceived_actions = decode_signal_sequence(signal_seq)
            for perceived_action in perceived_actions:
                if perceived_action in ["gather","social","trade","reproduce"]:
                    self.trust[other.name] = np.clip(self.trust.get(other.name,0.5)+0.05,0,1)
                    self.cooperation = np.clip(self.cooperation+0.02,0,1)
                    # Mutate the signal slightly
                    other.signal_map[other.last_action_name] = [
                        [f + random.uniform(-2,2) for f in freqs] for freqs in signal_seq
                    ]
                elif perceived_action == "fight":
                    self.trust[other.name] = np.clip(self.trust.get(other.name,0.5)-0.05,0,1)
                    self.cooperation = np.clip(self.cooperation-0.02,0,1)
                    other.signal_map[other.last_action_name] = [
                        [f + random.uniform(-5,5) for f in freqs] for freqs in signal_seq
                    ]

    def act(self, env, agents):
        if not self.alive: return
        self.perceive_music(agents)
        choice = self.choose_action(env)
        reward = 0

        # Initialize trust for new agents
        for a in agents:
            self.trust.setdefault(a.name,0.5)
            a.trust.setdefault(self.name,0.5)

        if choice=="gather":
            res = random.choice(list(env.resources.values()))
            gathered = min(res.amount, random.randint(1,5))
            res.amount -= gathered
            if res.name=="food": self.food += gathered
            elif res.name=="energy": self.energy += gathered
            elif res.name=="metal": self.metal += gathered
            reward += gathered/5

        elif choice=="trade":
            partners = [a for a in agents if a!=self and a.alive and self.trust.get(a.name,0)>0.3]
            if partners:
                partner = random.choice(partners)
                trade_amount = min(self.metal, random.randint(1,3))
                if trade_amount>0:
                    self.metal -= trade_amount
                    partner.food += trade_amount
                    reward += trade_amount/3
                    self.trust[partner.name] += 0.05
                    partner.trust[self.name] += 0.05

        elif choice=="fight":
            targets = [a for a in agents if a!=self and a.alive]
            if targets:
                target = random.choice(targets)
                if self.strength > target.strength:
                    stolen = min(target.food, random.randint(1,3))
                    target.food -= stolen
                    self.food += stolen
                    reward += stolen/3
                    self.trust[target.name] -= 0.1
                else:
                    reward -= 0.2
                    self.trust[target.name] += 0.05

        elif choice=="social":
            self.food -= 1
            reward -= 0.1
            for a in agents:
                if a!=self: self.trust[a.name] += 0.01

        elif choice=="reproduce":
            if self.food>10 and self.energy>10:
                self.food -= 10
                self.energy -= 10
                child = Agent(self.name+"_child", parent=self)
                agents.append(child)
                reward += 1

        # Survival cost
        self.food -= 1
        self.energy -= 1
        if self.food <= 0 or self.energy <= 0:
            self.alive = False
            reward -= 1

        self.learn(reward)
        # Play the chosen signal sequence
        play_signal_sequence(self.signal_map[choice], self)

# --- Pygame simulation ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ANNIE")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

env = Environment()
agents = [Agent("ALEX"), Agent("JOE"), Agent("SAM"), Agent("JESSE")]

running = True
step = 0

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((30,30,30))
        alive_agents = [a for a in agents if a.alive]

        if alive_agents:
            for agent in alive_agents:
                agent.act(env, alive_agents)

            if step % 5 == 0:
                env.regenerate_resources()
            step += 1

        # Draw agent stats
        y = 10
        for agent in agents:
            text = f"{agent.name}: {'ALIVE' if agent.alive else 'DEAD'} | Food:{agent.food} Energy:{agent.energy} | Last:{agent.last_action_name}"
            surf = font.render(text, True, (255,255,255))
            screen.blit(surf, (10, y))
            y += 25

        pop_text = font.render(f"Population: {len(alive_agents)}", True, (255,200,0))
        screen.blit(pop_text, (600,10))

        pygame.display.flip()
        clock.tick(1)

finally:
    pygame.quit()
    input("Simulation ended. Press Enter to exit...")
