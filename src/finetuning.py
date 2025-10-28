import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model


class LoRA(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: int = 32):
        """
        Low-Rank Adaptation (LoRA) module.

        Args:
            original_layer (nn.Linear): Capa lineal original a la que se aplica LoRA.
            r (int): Rango de la aproximación de baja-rank.
            alpha (int): Factor de escala de LoRA.
        """
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("LoRA solo admite nn.Linear como capa original.")

        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Matrices de baja-rank A y B
        # A: (out_features, r), B: (r, in_features)
        self.A = nn.Parameter(torch.empty(out_features, r))
        self.B = nn.Parameter(torch.empty(r, in_features))

        # Inicialización: A con Kaiming, B a ceros (como en el paper)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Escalado alpha/r
        self.scaling = float(self.alpha) / float(self.r)

        # Congelar parámetros de la capa original
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la capa original con un update de baja-rank fusionado en los pesos.

        Efecto: y = x @ (W + scaling * (A @ B))^T + b
        """
        # ΔW con forma (out, in). Para F.linear necesitamos (in, out) -> transponemos.
        delta_w_t = (self.A @ self.B).t()  # (in, out)

        # Pesos efectivos: W_eff^T = W^T + scaling * ΔW^T
        weight_eff = self.original_layer.weight + self.scaling * delta_w_t

        return F.linear(x, weight_eff, self.original_layer.bias)


def inject_lora_into_model(model, r: int = 4, alpha: int = 32, device: str = "cpu"):
    """
    Sustituye las proyecciones q,k,v,o de los bloques T5Attention por LoRA(nn.Linear).

    Args:
        model (PreTrainedModel): Modelo preentrenado.
        r (int): Rango LoRA.
        alpha (int): Escalado LoRA.
        device (str): 'cuda' o 'cpu'.

    Returns:
        model con LoRA inyectado en las capas de atención.
    """
    # Recorremos módulos y reemplazamos las lineales q,k,v,o dentro de T5Attention
    for _, module in model.named_modules():
        if module.__class__.__name__ == "T5Attention":
            for attr in ["q", "k", "v", "o"]:
                linear = getattr(module, attr, None)
                if isinstance(linear, nn.Linear):
                    setattr(module, attr, LoRA(linear, r=r, alpha=alpha))
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length: int, model_hidden_size: int):
        """
        Soft prompts entrenables para anteponer a embeddings de entrada.

        Args:
            prompt_length (int): Nº de tokens virtuales del soft prompt.
            model_hidden_size (int): Dimensión oculta del modelo (p.ej., 512 en T5-small).
        """
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, model_hidden_size))

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend de los soft prompts a los embeddings de entrada.

        Args:
            input_embeddings (Tensor): (batch, seq_len, hidden)

        Returns:
            Tensor: (batch, prompt_length + seq_len, hidden)
        """
        batch_size = input_embeddings.size(0)
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft_prompt_expanded, input_embeddings], dim=1)
