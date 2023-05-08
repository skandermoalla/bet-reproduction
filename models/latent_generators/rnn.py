import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import models.latent_generators.latent_generator as latent_generator

from models.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple


class LSTM_LG(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        pdrop: float = 0,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.discrete_output = discrete_input
        self.predict_offsets = predict_offsets
        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.predict_offsets:
            output_dim = self.vocab_size * (1 + self.action_dim)
        else:
            output_dim = self.vocab_size

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.model = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=pdrop,
            bidirectional=False,
        )
        self.fc2 = nn.Linear(
                in_features=self.hidden_dim,
                out_features=output_dim,
            )

    def get_latent_and_loss(
            self,
            obs_rep: torch.Tensor,  # N, S, E ordered inputs.
            target_latents: torch.Tensor,
            seq_masks: Optional[torch.Tensor] = None,
            return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.predict_offsets:
            target_latents, target_offsets = target_latents

        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )

        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        if self.predict_offsets:
            # reshaping to transformer output shape
            # TODO: check if it reconstructs correctly from the flatten in MLP
            output = self.fc2(self.model(self.fc1(obs_rep))[0])
            assert output.ndim == 3, output.shape
            output = output.reshape(
                -1, self.block_size, self.vocab_size * (1 + self.action_dim)
            )
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size:]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
            # calculate (optionally soft) cross entropy and offset losses
            class_loss = criterion(logits.reshape(-1, logits.size(-1)), target_latents)
            # offset loss is only calculated on the target class
            # if soft targets, argmax is considered the target class
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.offset_loss_scale * F.mse_loss(
                selected_offsets, target_offsets.view(-1, self.action_dim)
            )
            loss = offset_loss + class_loss
            logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
            offsets = einops.rearrange(
                offsets,
                "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
                N=batch,
                T=seq,
            )
            if return_loss_components:
                return (
                    (logits, offsets),
                    loss,
                    {"offset": offset_loss, "class": class_loss, "total": loss},
                )
            else:
                return (logits, offsets), loss
        else:
            # No loss for the offsets
            logits = self.fc2(self.model(self.fc1(obs_rep))[0])
            assert logits.ndim == 3
            # reshaping to transformer output shape
            # TODO: check if it reconstructs correctly from the flatten in MLP
            logits = logits.reshape(-1, self.block_size, self.vocab_size)
            loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            logits = einops.rearrange(
                logits, "batch seq classes -> seq batch classes"
            )  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            if return_loss_components:
                return logits, loss, {"class": loss, "total": loss}
            else:
                return logits, loss

    def generate_latents(
        self, seq_obses: torch.Tensor, seq_masks: torch.Tensor
    ) -> torch.Tensor:
        seq, batch, embed = seq_obses.size()
        obs_rep = einops.rearrange(seq_obses, "seq batch embed -> batch seq embed")
        # if we have run less episodes than the window size, we pad with zeros
        num_of_lacking_episodes = self.block_size - obs_rep.shape[1]
        if num_of_lacking_episodes:
            obs_rep = F.pad(obs_rep, (0, 0, num_of_lacking_episodes, 0), "constant", 0)

        output = self.fc2(self.model(self.fc1(obs_rep))[0])
        # reshaping to transformer output shape
        # TODO: check if it reconstructs correctly from the flatten in MLP
        output = output.reshape(
            -1, self.block_size, self.vocab_size * (1 + self.action_dim)
        )
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
        else:
            logits = output
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.action_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:

        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
