import torch
from torch.cuda.amp import autocast
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel,
    StableDiffusionAdapterPipeline, T2IAdapter
)
from peft import LoraConfig, get_peft_model

class BaseTrainingStrategy:
    def __init__(self, train_cfg, model_cfg, strategy_cfg):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.strategy_cfg = strategy_cfg
        self.device = train_cfg.device
        self.dtype = train_cfg.dtype
        self.pipe = None
        self.optimizer = None
        self.scheduler = None
        self.trainable_model = None 
        self._load_models()
        self._setup_optimizer()
        self.pipe.to(self.device)

    def _load_models(self):
        raise NotImplementedError("_load_models() need to be defined.")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.trainable_model.parameters(),
            lr=self.strategy_cfg["learning_rate"],
            weight_decay=self.strategy_cfg["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

    def _get_text_embeds(self, batch_size, prompt):
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding=self.train_cfg.padding,
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors=self.train_cfg.return_tensors
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            encoder_hidden_states = self.pipe.text_encoder(text_input_ids)[0]
        
        if encoder_hidden_states.shape[0] != batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)
        return encoder_hidden_states

    def _prepare_batch_latents(self, target_images):
        with torch.no_grad():
            target_images_dtype = target_images.to(self.pipe.vae.dtype)
            latents = self.pipe.vae.encode(target_images_dtype).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        return noisy_latents, timesteps, noise, batch_size

    def train_step(self, batch, scaler):
        raise NotImplementedError("train_step() need to be defined.")

    @torch.no_grad()
    def val_step(self, batch):
        raise NotImplementedError("val_step() need to be defined.")

    def save_model(self, path):
        self.trainable_model.save_pretrained(path)

# Finetune full ControlNet
class FullControlNetStrategy(BaseTrainingStrategy):
    def _load_models(self):
        controlnet = ControlNetModel.from_pretrained(self.model_cfg.base_controlnet_name)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_cfg.stable_diff_name,
            controlnet=controlnet,
            torch_dtype=self.dtype
        )
        
        # Freeze
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        
        controlnet.to(torch.float32) 
        controlnet.requires_grad_(True)
        self.trainable_model = self.pipe.controlnet

    def _common_forward(self, batch):
        condition_images = batch["condition_image"].to(self.device, dtype=torch.float32) # match controlnet
        target_images = batch["target_image"].to(self.device)

        noisy_latents, timesteps, noise, bsz = self._prepare_batch_latents(target_images)
        
        use_null_prompt = torch.rand(1).item() < self.train_cfg.null_prompt_prob
        prompt = "" if use_null_prompt else self.train_cfg.default_prompt
        encoder_hidden_states = self._get_text_embeds(bsz, prompt)

        # Forward ControlNet
        controlnet_output = self.pipe.controlnet(
            sample=noisy_latents.to(torch.float32), 
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states.to(torch.float32),
            controlnet_cond=condition_images,
            return_dict=True
        )
        
        down_block_res_samples = [res.to(self.dtype) for res in controlnet_output.down_block_res_samples]
        mid_block_res_sample = controlnet_output.mid_block_res_sample.to(self.dtype)

        # Forward UNet
        noise_pred = self.pipe.unet(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        ).sample
        
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def train_step(self, batch, scaler):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
            loss_scaled = loss / self.train_cfg.accumulation_steps
        
        scaler.scale(loss_scaled).backward()
        return loss.item()

    @torch.no_grad()
    def val_step(self, batch):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
        return loss.item()

# LoRA on ControlNet
class LoRAControlNetStrategy(BaseTrainingStrategy):
    def _load_models(self):
        controlnet = ControlNetModel.from_pretrained(
            self.model_cfg.base_controlnet_name, torch_dtype=self.dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_cfg.stable_diff_name,
            controlnet=controlnet,
            torch_dtype=self.dtype
        )
        # Freeze
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.controlnet.requires_grad_(False)
        
        # LoRA
        lora_config = LoraConfig(
            r=self.strategy_cfg["lora_r"],
            lora_alpha=self.strategy_cfg["lora_alpha"],
            target_modules=self.strategy_cfg["lora_target_modules"],
            lora_dropout=self.strategy_cfg["lora_dropout"],
            bias="none"
        )
        self.pipe.controlnet = get_peft_model(self.pipe.controlnet, lora_config)
        self.trainable_model = self.pipe.controlnet
        self.trainable_model.print_trainable_parameters()

    def _common_forward(self, batch):
        condition_images = batch["condition_image"].to(self.device, dtype=self.dtype)
        target_images = batch["target_image"].to(self.device)

        noisy_latents, timesteps, noise, bsz = self._prepare_batch_latents(target_images)
        
        use_null_prompt = torch.rand(1).item() < self.train_cfg.null_prompt_prob
        prompt = "" if use_null_prompt else self.train_cfg.default_prompt
        encoder_hidden_states = self._get_text_embeds(bsz, prompt)

        # Forward ControlNet (LoRA)
        controlnet_output = self.pipe.controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_images,
            return_dict=True
        )
        
        # Forward UNet
        noise_pred = self.pipe.unet(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_output.down_block_res_samples,
            mid_block_additional_residual=controlnet_output.mid_block_res_sample
        ).sample
        
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def train_step(self, batch, scaler):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
            loss_scaled = loss / self.train_cfg.accumulation_steps
        
        scaler.scale(loss_scaled).backward()
        return loss.item()

    @torch.no_grad()
    def val_step(self, batch):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
        return loss.item()

# Finetuned ControlNet + LoRA on UNet
class LoRAUNetStrategy(BaseTrainingStrategy):
    def _load_models(self):
        finetuned_controlnet_path = self.strategy_cfg["finetuned_controlnet_path"]
        controlnet = ControlNetModel.from_pretrained(finetuned_controlnet_path, torch_dtype=self.dtype)
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_cfg.stable_diff_name,
            controlnet=controlnet,
            torch_dtype=self.dtype
        )
        
        # Freeze
        self.pipe.controlnet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)

        # LoRA on UNet
        lora_config = LoraConfig(
            r=self.strategy_cfg["lora_r"],
            lora_alpha=self.strategy_cfg["lora_alpha"],
            target_modules=self.strategy_cfg["lora_target_modules"],
            lora_dropout=self.strategy_cfg["lora_dropout"],
            bias="none"
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
        self.trainable_model = self.pipe.unet
        print("Đã tải xong Chiến lược 3: Finetuned ControlNet + LoRA on UNet.")
        self.trainable_model.print_trainable_parameters()

    _common_forward = LoRAControlNetStrategy._common_forward
    train_step = LoRAControlNetStrategy.train_step
    val_step = LoRAControlNetStrategy.val_step

# Finetune T2I Adapter
class T2IAdapterStrategy(BaseTrainingStrategy):
    def _load_models(self):
        adapter = T2IAdapter.from_pretrained(
            self.model_cfg.base_t2i_adapter_name
        )
        self.pipe = StableDiffusionAdapterPipeline.from_pretrained(
            self.model_cfg.stable_diff_name,
            adapter=adapter,
            torch_dtype=self.dtype
        )
        # Freeze
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        
        # Train Adapter
        adapter.to(torch.float32) 
        adapter.requires_grad_(True)
        self.trainable_model = self.pipe.adapter

    def _common_forward(self, batch):
        condition_images = batch["condition_image"].to(self.device, dtype=torch.float32) # match adapter
        target_images = batch["target_image"].to(self.device)

        noisy_latents, timesteps, noise, bsz = self._prepare_batch_latents(target_images)
        
        use_null_prompt = torch.rand(1).item() < self.train_cfg.null_prompt_prob
        prompt = "" if use_null_prompt else self.train_cfg.default_prompt
        encoder_hidden_states = self._get_text_embeds(bsz, prompt)

        # Forward Adapter
        adapter_features = self.pipe.adapter(condition_images)
        
        adapter_features_dtype = [feat.to(self.dtype) for feat in adapter_features]

        # Forward UNet
        noise_pred = self.pipe.unet(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_intrablock_additional_residuals=adapter_features_dtype
        ).sample
        
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def train_step(self, batch, scaler):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
            loss_scaled = loss / self.train_cfg.accumulation_steps
        
        scaler.scale(loss_scaled).backward()
        return loss.item()

    @torch.no_grad()
    def val_step(self, batch):
        with autocast(enabled=self.train_cfg.device == 'cuda'):
            loss = self._common_forward(batch)
        return loss.item()

def get_strategy(strategy_name, train_cfg, model_cfg):
    
    strategy_cfg = getattr(model_cfg, strategy_name.lower(), None)
    if strategy_cfg is None:
        raise ValueError(strategy_name)

    if strategy_name == "strategy_1":
        return FullControlNetStrategy(train_cfg, model_cfg, strategy_cfg)
    elif strategy_name == "strategy_2":
        return LoRAControlNetStrategy(train_cfg, model_cfg, strategy_cfg)
    elif strategy_name == "strategy_3":
        return LoRAUNetStrategy(train_cfg, model_cfg, strategy_cfg)
    elif strategy_name == "strategy_4":
        return T2IAdapterStrategy(train_cfg, model_cfg, strategy_cfg)
    else:
        raise ValueError(f"Cannot define strategy: {strategy_name}")