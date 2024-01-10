import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.dataset_utils import *
from deepfashion.models.embedder.builder import *


class TypeAwareNet(nn.Module):
    def __init__(
            self,
            img_backbone: str = 'resnet-18',
            embedding_dim: int = 64,
            num_category: int = 12
            ):
        super().__init__()
        self.img_embedder = build_img_embedder(img_backbone, embedding_dim)
        self.category_embedder = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        initrange = 0.1
        self.category_embedder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        inputs = stack_dict(inputs)
        
        category_embed = self.category_embedder(inputs['category']).squeeze(-2)
        img_embed = self.img_embedder(inputs['img'])
        masked_embed = img_embed * category_embed

        outputs = unstack_dict({
            'input_mask': inputs['input_mask'],
            'embed': masked_embed
            })
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
            anchors = self(anchors)
            positives = self(positives)
            negatives = self(negatives)
            input_mask = anchors['input_mask']
            mask = input_mask.view(-1)
            B, S, E = anchors['embed'].shape
            B, N, E = negatives['embed'].shape
            # [B, S, E] -> [B, S, 1, E] -> [B, S, N, E] -> [B * S, N, E] -> [B * V, N, E] -> [B * V * N, E]
            anchors = anchors['embed'].unsqueeze(2).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
            # [B, 1, E] -> [B, V * N, E]
            positives = positives['embed'].unsqueeze(2).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
            # [B, N, E] -> [B, 1, N, E] -> [B, S, N, E] -> [B * S, N, E] -> [B * V, N, E] -> [B * V * N, E]
            negatives = negatives['embed'].unsqueeze(1).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
            running_loss = nn.TripletMarginLoss(margin=2, reduction='mean')(anchors, positives, negatives)

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