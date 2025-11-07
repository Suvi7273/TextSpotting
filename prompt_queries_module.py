import torch
import torch.nn as nn


class PromptQueriesGenerationModule(nn.Module):
    """
    Module that generates and refines prompt queries for different spotting tasks.
    """

    def __init__(self, d_model, num_tasks=3, num_prompt_queries=10):
        """
        Args:
            d_model (int): Feature dimension
            num_tasks (int): Number of tasks (word-level, line-level, video-level)
            num_prompt_queries (int): Number of prompt queries per task
        """
        super().__init__()
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.num_prompt_queries = num_prompt_queries

        # Task embeddings for different spotting tasks
        # 0: word-level, 1: line-level, 2: video-level
        self.task_embeddings = nn.Embedding(num_tasks, d_model)

        # Prompt query embeddings for each task
        self.prompt_queries = nn.Parameter(
            torch.randn(num_tasks, num_prompt_queries, d_model)
        )

        # Cross-task interaction via multi-head attention
        self.cross_task_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, task_id=0):
        """
        Args:
            task_id (int): Task identifier (0=word-level, 1=line-level, 2=video-level)
        Returns:
            torch.Tensor: (1, num_prompt_queries, d_model)
        """
        # Get prompt queries for the specific task
        task_prompts = self.prompt_queries[task_id].unsqueeze(0)  # (1, num_prompt_queries, d_model)

        # Get all prompt queries for cross-task interaction
        all_prompts = self.prompt_queries.reshape(-1, self.d_model).unsqueeze(0)  # (1, num_tasks*num_prompt_queries, d_model)

        # Cross-task attention: current task queries attend to all task queries
        attended_prompts, _ = self.cross_task_attention(
            task_prompts, all_prompts, all_prompts
        )

        # Residual connection and normalization
        prompt_queries = self.layer_norm(task_prompts + attended_prompts)

        return prompt_queries


class PromptAdapter(nn.Module):
    """
    Adapter that integrates prompt queries into the encoder/decoder.
    """

    def __init__(self, d_model):
        super().__init__()
        self.adapter_down = nn.Linear(d_model, d_model // 4)
        self.adapter_up = nn.Linear(d_model // 4, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, features, prompt_queries):
        """
        Args:
            features (torch.Tensor): (B, N, d_model) - encoder/decoder features
            prompt_queries (torch.Tensor): (1, num_prompt_queries, d_model)
        Returns:
            torch.Tensor: (B, N, d_model)
        """
        batch_size = features.shape[0]

        # Expand prompt queries to batch size
        prompt_queries = prompt_queries.expand(batch_size, -1, -1)

        # Concatenate prompt queries with features
        combined = torch.cat([features, prompt_queries], dim=1)  # (B, N+num_prompt, d_model)

        # Pass through adapter
        residual = combined
        x = self.adapter_down(combined)
        x = self.activation(x)
        x = self.adapter_up(x)
        adapted = self.layer_norm(residual + x)

        # Remove prompt queries and return original length
        adapted_features = adapted[:, :features.shape[1], :]

        return adapted_features
