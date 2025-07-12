import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MinesweeperEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, width=10, height=10, n_mines=10, render_mode='human',
                 reward_win=1.0, reward_lose=-1.0, reward_reveal=0.1, reward_invalid=-0.1,
                 max_reward_per_step=None, render_fps=30):
        super().__init__()

        self.width = width
        self.height = height
        self.n_mines = n_mines
        self.grid_size = (height, width)
        self.action_space_size = height * width
        self.render_mode = render_mode
        self.render_fps = render_fps  # Configurable FPS
        
        # Non-blocking delay system
        self.pause_counter = 0  # Frame counter for pauses
        self.is_paused = False  # Whether currently in pause state

        # 奖励设置
        self.reward_win = reward_win  # 胜利奖励
        self.reward_lose = reward_lose  # 失败惩罚
        self.reward_reveal = reward_reveal  # 每揭开一个安全格子的奖励
        self.reward_invalid = reward_invalid  # 点击已揭开格子的惩罚
        self.max_reward_per_step = max_reward_per_step  # 单步最大奖励限制

        # 修改为 (添加通道维度, 使用 float32):
        # 注意：low 和 high 可以根据你的归一化策略调整，这里暂时保持原逻辑范围
        # 但建议在 _get_obs 中进行归一化，并将 low/high 设置为归一化后的范围，例如 0/1 或 -1/1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, # 假设我们将归一化到 0-1 范围
            shape=(1, self.height, self.width), # (通道数, 高度, 宽度)
            dtype=np.float32
        )

        # 动作空间: Discrete(height * width)
        # 每个动作对应揭开一个格子 (row * width + col)
        self.action_space = spaces.Discrete(self.action_space_size)

        # Pygame 初始化 (仅在需要渲染时)
        self._pygame = None  # Lazy-loaded pygame module
        self.window = None
        self.clock = None
        self.cell_size = 30 # 可调整单元格大小
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.font = None

    def _get_obs(self):
        # 返回给智能体的观察状态
        # 隐藏地雷位置，只显示 -2 (未揭开) 或 0-8 (已揭开)
        obs = self.board.copy()
        obs[self.mine_locations] = -2 # 隐藏地雷
        # mlp 可以直接返回 obs
        # return obs

        # --- 新增：归一化和类型转换 ---
        # 简单的线性归一化: 将 [-2, 8] 映射到 [0, 1]
        # -2 -> 0
        # 8 -> 1
        # 公式: (value - min_val) / (max_val - min_val)
        min_val = -2.0
        max_val = 8.0
        obs_normalized = (obs.astype(np.float32) - min_val) / (max_val - min_val)

        # --- 新增：调整形状 ---
        # 添加通道维度: (H, W) -> (1, H, W)
        obs_final = np.expand_dims(obs_normalized, axis=0)

        return obs_final # 返回 shape=(1, H, W), dtype=float32, 范围 [0, 1] 的数组

    def _get_info(self):
        # 返回辅助信息，例如剩余地雷数、是否胜利等
        return {"remaining_mines": self.n_mines,
                "revealed_cells": np.sum(self.revealed),
                "is_success": self.win}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化游戏状态
        self.board = np.full(self.grid_size, -2, dtype=np.int32) # -2: 未揭开
        self.mines = np.zeros(self.grid_size, dtype=bool)
        self.revealed = np.zeros(self.grid_size, dtype=bool)

        # 随机布雷
        mine_indices = self.np_random.choice(self.action_space_size, self.n_mines, replace=False)
        self.mine_locations = np.unravel_index(mine_indices, self.grid_size)
        self.mines[self.mine_locations] = True

        # 计算每个格子的邻近地雷数 (只在内部计算，不直接暴露给 obs)
        self._calculate_neighbors()

        self.game_over = False
        self.win = False
        self.first_step = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _calculate_neighbors(self):
        self.neighbor_counts = np.zeros(self.grid_size, dtype=np.int32)
        for r in range(self.height):
            for c in range(self.width):
                if not self.mines[r, c]:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.height and 0 <= nc < self.width and self.mines[nr, nc]:
                                count += 1
                    self.neighbor_counts[r, c] = count

    def step(self, action):
        if self.game_over:
            # 如果游戏已结束，可以返回一个零奖励和当前状态
            # 或者根据 Gymnasium 指南处理
            obs = self._get_obs()
            return obs, 0, True, False, self._get_info()

        row, col = np.unravel_index(action, self.grid_size)

        if self.first_step and self.mines[row, col]:
            # 第一次点击踩到了雷，需要移动这个雷
            self.mines[row, col] = False # 移除当前位置的雷

            # 寻找一个新的、安全的位置放置这个雷
            possible_new_locations_indices = np.argwhere(self.mines == False)
            # 确保新位置不是当前点击的位置 (理论上已经是 False 了，但以防万一)
            valid_new_locations = [(r, c) for r, c in possible_new_locations_indices if not (r == row and c == col)]

            if not valid_new_locations:
                 # 如果没有其他位置放雷了（几乎不可能，除非格子数=雷数），这里需要处理
                 # 但我们在 reset 中已经做了检查，所以这里理论上总能找到位置
                 raise Exception("Cannot find a new location for the mine!")

            # 随机选择一个新位置
            new_loc_idx = self.np_random.choice(len(valid_new_locations))
            new_r, new_c = valid_new_locations[new_loc_idx]
            self.mines[new_r, new_c] = True

            # 雷的位置变了，必须重新计算邻居数量
            self._calculate_neighbors()

        # 标记第一次点击已完成
        self.first_step = False

        reward = 0
        terminated = False # 游戏是否因胜利或失败而结束
        truncated = False # 游戏是否因时间限制等外部因素结束 (这里不用)

        if self.revealed[row, col]:
            # 点击已揭开的格子 -> 惩罚
            reward = self.reward_invalid
            # terminated = True # 可以选择是否因此结束游戏
        elif self.mines[row, col]:
            # 点击到地雷 -> 失败，大惩罚
            reward = self.reward_lose
            self.revealed[row, col] = True
            self.board[row, col] = -1 # 标记为地雷
            self.game_over = True
            terminated = True
            self.win = False
        else:
            # 点击到安全格子 -> 揭开，奖励为揭开的格子数量 * reward_reveal
            revealed_before = np.sum(self.revealed)
            self._reveal_cell(row, col)
            revealed_after = np.sum(self.revealed)
            revealed_count = revealed_after - revealed_before
            
            # 计算基础奖励
            reward = revealed_count * self.reward_reveal
            
            # 如果设置了单步最大奖励限制，则对奖励进行裁剪
            if self.max_reward_per_step is not None:
                reward = min(reward, self.max_reward_per_step)

            if self._check_win():
                reward += self.reward_win
                self.game_over = True
                terminated = True
                self.win = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _reveal_cell(self, row, col):
        if not (0 <= row < self.height and 0 <= col < self.width) or self.revealed[row, col] or self.mines[row, col]:
            return

        self.revealed[row, col] = True
        self.board[row, col] = self.neighbor_counts[row, col]

        # 如果揭开的是 0，递归揭开邻居
        if self.neighbor_counts[row, col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_cell(row + dr, col + dc)

    def _check_win(self):
        # 如果所有非地雷格子都被揭开，则胜利
        return np.sum(self.revealed) == (self.height * self.width - self.n_mines)
    
    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask indicating valid actions.
        An action is valid if the cell has not been revealed yet.
        """
        # self.revealed is a (height, width) boolean array (True means revealed)
        # We want the mask to be True for *unrevealed* cells.
        # So, the mask is the logical NOT of self.revealed.
        # Flatten the mask to match the 1D action space.
        return ~self.revealed.flatten()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # human 模式下，_render_frame 已经在 step 和 reset 中调用

    def _get_pygame(self):
        """Lazy load pygame module to avoid import warnings in headless environments."""
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.init()
        return self._pygame
    
    def _render_frame(self):
        pygame = self._get_pygame()  # Lazy load pygame
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Minesweeper RL")
            self.font = pygame.font.SysFont(None, int(self.cell_size * 0.6))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((200, 200, 200)) # 背景色

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(canvas, (100, 100, 100), rect, 1) # 网格线

                cell_value = self.board[r, c]
                text_surface = None
                text_color = (0, 0, 0)

                if self.revealed[r, c]:
                    pygame.draw.rect(canvas, (230, 230, 230), rect) # 已揭开背景
                    if cell_value == -1: # 踩到雷
                         text_surface = self.font.render("X", True, (255, 0, 0))
                    elif cell_value > 0:
                        # 不同数字用不同颜色 (可选)
                        colors = [(0,0,255), (0,128,0), (255,0,0), (0,0,128),
                                  (128,0,0), (0,128,128), (0,0,0), (128,128,128)]
                        text_color = colors[min(cell_value - 1, len(colors) - 1)]
                        text_surface = self.font.render(str(cell_value), True, text_color)
                    # cell_value == 0 时不显示数字
                else:
                    pygame.draw.rect(canvas, (180, 180, 180), rect) # 未揭开背景

                if text_surface:
                    text_rect = text_surface.get_rect(center=rect.center)
                    canvas.blit(text_surface, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Handle pause state (non-blocking delay)
            if self.is_paused:
                self.pause_counter -= 1
                if self.pause_counter <= 0:
                    self.is_paused = False
            
            self.clock.tick(self.render_fps)
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def get_user_action(self):
        """
        Get user action from pygame events.
        Returns action index or None if no action.
        Also returns a quit flag.
        """
        if self.render_mode != "human" or self._pygame is None:
            return None, False
        
        pygame = self._pygame
        action = None
        quit_flag = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // self.cell_size
                row = y // self.cell_size
                
                if (0 <= row < self.height and 
                    0 <= col < self.width and 
                    not self.revealed[row, col]):
                    action = row * self.width + col
        
        return action, quit_flag
    
    def check_quit_key(self):
        """
        Check if user pressed quit key (Q).
        Returns True if quit key was pressed.
        """
        if self.render_mode != "human" or self._pygame is None:
            return False
            
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
        return False
    
    def wait_frames(self, frames):
        """
        Non-blocking wait for specified number of frames.
        
        Args:
            frames (int): Number of frames to wait
        """
        self.pause_counter = frames
        self.is_paused = True
    
    def wait_seconds(self, seconds):
        """
        Non-blocking wait for specified seconds (converted to frames).
        
        Args:
            seconds (float): Number of seconds to wait
        """
        frames = int(seconds * self.render_fps)
        self.wait_frames(frames)
    
    def is_waiting(self):
        """
        Check if environment is currently in a wait state.
        
        Returns:
            bool: True if waiting, False otherwise
        """
        return self.is_paused
    
    def close(self):
        if self.window is not None and self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
            self.window = None
            self.clock = None
            self._pygame = None

# (可选) 注册环境，方便 gym.make() 调用
# from gymnasium.envs.registration import register
# register(
#      id='Minesweeper-v0',
#      entry_point='minesweeper_env:MinesweeperEnv',
#      max_episode_steps=200, # 可选，设置最大步数
# )