"""
Adapted from https://github.com/jacarvalho/mpd-public

MIT License

Copyright (c) 2024 Itamar Mishani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# General includes.
from copy import deepcopy
import einops
import torch
from typing import *
# Project includes.
from .diffusion_model_base import *
from mmd.config import MMDParams as params


class DiffusionsEnsemble(nn.Module):
    def __init__(self,
                 models: Dict[int, GaussianDiffusionModel],
                 transforms: Dict[int, torch.Tensor],
                 context_model=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.models = models
        # Make sure all models have the same number of diffusion steps.
        assert len(set([model.n_diffusion_steps for model in models.values()])) == 1
        self.n_diffusion_steps = models[0].n_diffusion_steps
        # Make sure all predict the same values (predict epsilon or not).
        assert len(set([model.predict_epsilon for model in models.values()])) == 1

        self.transforms = transforms

        self.context_model = context_model

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, cross_conds,
                      n_diffusion_steps=None,
                      contexts=None, return_chain=False,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      warm_start_path_b: torch.Tensor = None,
                      **sample_kwargs):
        device = self.models[0].betas.device

        batch_size = shape[0]
        x = {}
        chains = {}
        for m in self.models.keys():
            if warm_start_path_b is not None:
                x[m] = warm_start_path_b[:, m * params.horizon: (m+1) * params.horizon, :].clone()
                # Move to range with transform.
                x[m][:, :, :2] -= self.transforms[m]
                print(CYAN, "Using warm start path in p_sample_loop. Steps (negative attempts to recon. t=0)",
                      [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
            else:
                x[m] = torch.randn(shape, device=device)
                print(CYAN, "Using random noise in p_sample_loop. Steps (negative attempts to recon. t=0)",
                      [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
            try:
                x[m] = apply_hard_conditioning(x[m], hard_conds[m])
            except KeyError:
                hard_conds[m] = {}
        x = apply_cross_conditioning(x, cross_conds, self.transforms)
        chains = {k: [x[k]] for k in self.models.keys()} if return_chain else None

        for i in reversed(range(-n_diffusion_steps_without_noise,
                                n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            for m in self.models.keys():
                ctx = deepcopy(contexts[m]) if contexts is not None else None
                if ctx is not None:
                    for k, v in ctx.items():
                        ctx[k] = einops.repeat(v, 'd -> b d', b=x[m].shape[0])
                x[m], values = sample_fn(self.models[m], x[m], hard_conds[m],
                                         ctx, t, **sample_kwargs['sample_kwargs'][m])
                x[m] = apply_hard_conditioning(x[m], hard_conds[m])
                x = apply_cross_conditioning(x, cross_conds, self.transforms)
            if return_chain:
                for m in self.models.keys():
                    chains[m].append(x[m])

        if return_chain:
            chains = {k: torch.stack(v, dim=1) for k, v in chains.items()}
            return x, chains

        return x

    @torch.no_grad()
    def ddim_sample(
            self, shape, hard_conds,
            n_diffusion_steps=None,
            context=None, return_chain=False,
            t_start_guide=torch.inf,
            guide=None,
            n_guide_steps=1,
            **sample_kwargs,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        device = self.betas.device
        batch_size = shape[0]
        total_timesteps = n_diffusion_steps
        sampling_timesteps = n_diffusion_steps // 5
        eta = 0.

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)
        times = torch.cat((torch.tensor([-1], device=device), times))
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None

        for time, time_next in time_pairs:
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            model_out = self.model(x, t, context)

            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            if time_next < 0:
                x = x_start
                x = apply_hard_conditioning(x, hard_conds)
                if return_chain:
                    chain.append(x)
                break

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                    eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise

            # guide gradient steps before adding noise
            if guide is not None:
                if torch.all(t_next < t_start_guide):
                    x = guide_gradient_steps(
                        x,
                        hard_conds=hard_conds,
                        guide=guide,
                        **sample_kwargs
                    )

            # add noise
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def joint_conditional_sampling(self,
                                   hard_conds: Dict[int, dict],
                                   cross_conds: Dict[Tuple[int, int], Tuple[int, int]],
                                   n_diffusion_steps: int = None,
                                   batch_size: int = 1,
                                   ddim: bool = False,
                                   warm_start_path_b: torch.Tensor = None,
                                   **sample_kwargs):
        # The horizon, in this implementation, is the same for all models.
        horizon = params.horizon
        shape = (batch_size, horizon, self.models[0].state_dim)
        if n_diffusion_steps is None:
            raise ValueError("n_diffusion_steps must be provided.")
        if ddim:
            if warm_start_path_b is not None:
                raise ValueError("warm_start_path_b is not supported for ddim sampling.")
            return self.ddim_sample(shape, hard_conds, cross_conds, n_diffusion_steps=n_diffusion_steps, **sample_kwargs)

        return self.p_sample_loop(shape, hard_conds, cross_conds, n_diffusion_steps=n_diffusion_steps,
                                  warm_start_path_b=warm_start_path_b, **sample_kwargs)

    def forward(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def warmup(self,
               horizon: int = 64,
               device: str = "cuda",
               context: List[dict] = None):
        context = deepcopy(context)
        if context is None:
            context = [None] * len(self.models)
        # Warmup all models
        for model, ctx in zip(self.models, context):
            model.warmup(horizon=horizon, device=device, context=ctx)

    @torch.no_grad()
    def run_inference(self,
                      contexts: List[dict] = None,
                      hard_conds: Dict[int, dict] = None,
                      cross_conds: Dict[Tuple[int, int], Tuple[int, int]] = None,
                      n_samples: int = 1,
                      return_chain: bool = False,
                      # sample_kwargs: List[dict] = None,
                      **diffusion_kwargs):
        """

        :param contexts:
        :param hard_conds:
        :param cross_conds: Keys are tuples of model indices and values are tuples of trajectory timesteps.
                            For example, {(0, 1): {0, 64}} means that the value of the state 0 in the trajectory
                            of model 0 needs to be the same as the value of the state 64 in the trajectory of model 1.
        :param n_samples:
        :param return_chain:
        :param diffusion_kwargs:
        :return:
        """
        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        samples, chains = self.joint_conditional_sampling(hard_conds=hard_conds,
                                                          cross_conds=cross_conds,
                                                          n_diffusion_steps=self.n_diffusion_steps,
                                                          contexts=contexts,
                                                          batch_size=n_samples,
                                                          return_chain=return_chain,
                                                          **diffusion_kwargs)

        for k_chain, v_chain in chains.items():
            chains[k_chain] = einops.rearrange(v_chain, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return chains

        return {k: v[-1] for k, v in chains.items()}

    def run_local_inference(self, seed_trajectory_b: torch.Tensor,
                            n_noising_steps: int,  # If None, then pass that and later on a true noise sample will be used.
                            n_denoising_steps: int,  # Must be a real value that can be used for denoising.
                            contexts: List[dict] = None,
                            hard_conds: Dict[int, dict] = None,
                            cross_conds: Dict[Tuple[int, int], Tuple[int, int]] = None,
                            n_samples: int = 1,
                            return_chain: bool = False,
                            **diffusion_kwargs):

        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Noise the given seed trajectory for n_noising_steps.
        if n_noising_steps is None:
            seed_trajectory_b_noised = None
        else:
            B = seed_trajectory_b.shape[0]
            t = make_timesteps(B, n_noising_steps, seed_trajectory_b.device)  # Shape: (B,).
            seed_trajectory_b_noised = self.models[0].q_sample(x_start=seed_trajectory_b, t=t)

        samples, chains = self.joint_conditional_sampling(hard_conds=hard_conds,
                                                          cross_conds=cross_conds,
                                                          n_diffusion_steps=n_denoising_steps,
                                                          contexts=contexts,
                                                          batch_size=n_samples,
                                                          return_chain=return_chain,
                                                          warm_start_path_b=seed_trajectory_b_noised,
                                                          **diffusion_kwargs)

        for k_chain, v_chain in chains.items():
            chains[k_chain] = einops.rearrange(v_chain, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return chains

        return {k: v[-1] for k, v in chains.items()}
