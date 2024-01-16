from deepfashion.models.baseline import *


def outfit_ranking_loss(
          anc_outs: DeepFashionOutput, 
          pos_outs: DeepFashionOutput, 
          neg_outs: DeepFashionOutput, 
          margin: float = 0.3
          ):
        n_outfit = anc_outs.mask.shape[0]
        ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
        neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

        loss = []
        for b_i in range(n_outfit):
            D_p = []
            for a_i in range(ans_per_batch[b_i]):
                anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                D_p.append(nn.PairwiseDistance(p=2)(anc_embed, pos_embed))
            D_p = torch.mean(torch.stack(D_p))

            D_n = []
            for a_i in range(ans_per_batch[b_i]):
                for n_i in range(neg_per_batch[b_i]):
                    anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    D_n.append(nn.PairwiseDistance(p=2)(anc_embed, neg_embed))
            D_n = torch.mean(torch.stack(D_n))

            hinge_dist = torch.clamp(margin + D_p - D_n, min=0.0)
            loss.append(torch.mean(hinge_dist))

        loss = torch.mean(torch.stack(loss))
        return loss


def triplet_loss(
          anc_outs: DeepFashionOutput, 
          pos_outs: DeepFashionOutput, 
          neg_outs: DeepFashionOutput, 
          margin: float = 0.3
          ):
        n_outfit = anc_outs.mask.shape[0]
        ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
        neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

        loss = []
        for b_i in range(n_outfit):
            for a_i in range(ans_per_batch[b_i]):
                for n_i in range(neg_per_batch[b_i]):
                    anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                    neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss = torch.mean(torch.stack(loss))
        return loss


def vse_loss(
          anc_outs: DeepFashionOutput, 
          pos_outs: DeepFashionOutput, 
          neg_outs: DeepFashionOutput, 
          margin: float = 0.3
          ):
        n_outfit = anc_outs.mask.shape[0]
        ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
        neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

        loss = []
        for b_i in range(n_outfit):
            for a_i in range(ans_per_batch[b_i]):
                for n_i in range(neg_per_batch[b_i]):
                    anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    pos_embed = pos_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                    neg_embed = neg_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))

                    anc_embed = anc_outs.txt_embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                    neg_embed = neg_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))

                    anc_embed = anc_outs.txt_embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    pos_embed = pos_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                    neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss = torch.mean(torch.stack(loss))
        return loss


def sim_loss(
          anc_outs: DeepFashionOutput, 
          pos_outs: DeepFashionOutput, 
          neg_outs: DeepFashionOutput, 
          margin: float = 0.3,
          l_1: float = 5e-4,
          l_2: float = 5e-4,
          ):
        n_outfit = anc_outs.mask.shape[0]
        ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
        neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

        loss_1 = []
        loss_2 = []
        for b_i in range(n_outfit):
            for a_i in range(ans_per_batch[b_i]):
                for n_i in range(neg_per_batch[b_i]):
                    anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                    pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                    neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                    loss_1.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(pos_embed, neg_embed, anc_embed))

                    if hasattr(anc_outs, 'txt_embed_by_category'):
                        anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
                        pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
                        neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
                        loss_2.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(pos_embed, neg_embed, anc_embed))
        loss = l_1 * torch.mean(torch.stack(loss_1))
        if loss_2:
             loss += l_2 * torch.mean(torch.stack(loss_2))
        return loss