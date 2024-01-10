import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.dataset_utils import *
from deepfashion.models.embedder.builder import *


class CSANet(nn.Module):
    def __init__(
            self,
            img_backbone: str = 'resnet-18',
            embedding_dim: int = 64,
            num_category: int = 12
            ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_category = num_category
        self.img_embedder = build_img_embedder(img_backbone, embedding_dim)
        self.mask = nn.Parameter(torch.ones((num_category, embedding_dim)))
        self.attn = nn.Sequential(
            nn.Linear(num_category * 2, num_category * 2),
            nn.ReLU(),
            nn.Linear(num_category * 2, num_category)
            )
        self._reset()

    def _reset(self) -> None:
        initrange = 0.1
        self.mask.data.uniform_(-initrange, initrange)

    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs) # [B, S, ...] -> [B * V, ...]
        value = self.img_embedder(inputs['img']) # [B * V, embedding_dim]
        if target_category:
            target_category = stack_tensors(inputs['input_mask'], target_category)
            category_embeds = torch.concat([
                torch.nn.functional.one_hot(inputs['category'], num_classes=self.num_category).squeeze(1), 
                torch.nn.functional.one_hot(target_category, num_classes=self.num_category).squeeze(1)
                ], dim=-1).to(torch.float32) # [B * V, num_category * 2]
            attn = F.softmax(self.attn(category_embeds), dim=-1) # [B * V, num_category]
            embed = value * torch.matmul(self.mask.T.unsqueeze(0), attn.unsqueeze(-1)).squeeze(-1)

            outputs = unstack_dict({
                'input_mask': inputs['input_mask'],
                'embed': embed
                })
            
        else:
            embed_list = []
            for i in range(self.num_category):
                target_category = torch.ones((inputs['img'].shape[0], 1), dtype=torch.long, device=inputs['category'].get_device()) * i
                category_embeds = torch.concat([
                    torch.nn.functional.one_hot(inputs['category'], num_classes=self.num_category).squeeze(1), 
                    torch.nn.functional.one_hot(target_category, num_classes=self.num_category).squeeze(1)
                    ], dim=-1).to(torch.float32) # [B * V, num_category * 2]
                attn = F.softmax(self.attn(category_embeds), dim=-1) # [B * V, num_category]
                embed = value * torch.matmul(self.mask.T.unsqueeze(0), attn.unsqueeze(-1)).squeeze(-1)
                embed_list.append(embed)

            outputs = {
                'input_mask': inputs['input_mask'],
                'embeds': torch.stack([unstack_tensors(inputs['input_mask'], i) for i in embed_list])
                }

        return outputs

    def iteration(
            self,
            dataloader,
            epoch,
            is_train,
            device,
            optimizer=None,
            scheduler=None,
            use_wandb=False
            ):
        type_str = 'Train' if is_train else 'Valid'

        epoch_iterator = tqdm(dataloader)
        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            # Forward
            anchors = {key: value.to(device) for key, value in batch['anchors'].items()}
            positives = {key: value.to(device) for key, value in batch['positives'].items()}
            negatives = {key: value.to(device) for key, value in batch['negatives'].items()}

            # Compute running loss
            B, S = anchors['input_mask'].shape
            B, NEG = negatives['input_mask'].shape
            anchor_outputs = self(anchors) # -> embeds = [C, B, S, E]
            positive_outputs = self(positives) # -> embeds = [C, B, 1, E]
            negative_outputs = self(negatives) # -> embeds = [C, B, N, E]
            running_loss = []
            for batch_i in range(B):
                for anc_i in range(torch.sum(~anchors['input_mask'][batch_i])):
                    for neg_i in range(torch.sum(~negatives['input_mask'][batch_i])):
                        anc_category = anchors['category'][batch_i + anc_i][0]
                        pos_category = positives['category'][batch_i + 0][0]

                        anc_embed = anchor_outputs['embeds'][pos_category][batch_i][anc_i]
                        pos_embed = positive_outputs['embeds'][anc_category][batch_i][0]
                        neg_embed = negative_outputs['embeds'][anc_category][batch_i][neg_i]
                        running_loss.append(nn.TripletMarginLoss(margin=2, reduction='mean')(anc_embed, pos_embed, neg_embed))
            running_loss = torch.mean(torch.stack(running_loss))

            total_loss += running_loss.item()
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            # Log
            epoch_iterator.set_description(
                f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}'
                )
            if use_wandb:
                log = {
                    f'{type_str}_loss': running_loss, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                if is_train == True:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)

        # Final Log
        total_loss = total_loss / iter
        if is_train == False:
            print( f'[E N D] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')
        return total_loss