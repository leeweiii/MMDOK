import math

import torch
import torch.nn as nn
import numpy as np
from .. import builder
import torch.nn.functional as F
from einops import rearrange
from src.efficient_kan import KAN
import copy


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MGLRL(nn.Module):
    def __init__(self, cfg):
        super(MGLRL, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)

        self.img_encoder_a = builder.build_img_model(cfg)
        self.img_encoder_b = builder.build_img_model(cfg)

        # self.SAF_module = SA(cfg.model.text.embedding_dim, 0.4)

        self.kan_patch = KAN([cfg.model.text.embedding_dim, 64, cfg.model.text.embedding_dim])
        self.patch_local_atten_layer = nn.MultiheadAttention(cfg.model.text.embedding_dim, 1, batch_first=True)

        self.kan_word = KAN([cfg.model.text.embedding_dim, 64, cfg.model.text.embedding_dim])
        self.word_local_atten_layer = nn.MultiheadAttention(cfg.model.text.embedding_dim, 1, batch_first=True)

        self.num_classes = cfg.model.vision.num_targets
        # self.attn_proj = nn.Linear(cfg.model.text.embedding_dim, (2 + self.num_classes) * 256)
        self.attn_proj = KAN([cfg.model.text.embedding_dim, (2 + self.num_classes) * 256])

        self.final_pred_fc = nn.Linear(256, self.num_classes)

        # self.prototype_layer = nn.Linear(cfg.model.text.embedding_dim, 256, bias=False)  # num_prototypes:500
        self.prototype_layer = KAN([cfg.model.text.embedding_dim, 256])  # num_prototypes:500
        self.get_assignments = sinkhorn
        self.prototype_align_layer = nn.Linear(256, cfg.model.text.embedding_dim)

        # self.classifier = nn.Linear(cfg.model.text.embedding_dim, cfg.model.vision.num_targets)

        self.mean_fc = nn.Linear(cfg.model.text.embedding_dim, self.num_classes)

        self.count = 0

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs_a, imgs_b=[]):
        # print('---------imgs_a--------------', np.shape(imgs_a))  # [16, 3, 224, 224]
        img_feat_g_a, img_feat_l_a = self.img_encoder_a(imgs_a, get_local=True)
        # print('------img_feat_g_a-------', np.shape(img_feat_g_a))  # [16, 2048]
        # print('------img_feat_l_a-------', np.shape(img_feat_l_a))  # [16, 1024, 19, 19]
        img_emb_g_a, img_emb_l_a = self.img_encoder_a.generate_embeddings(img_feat_g_a, img_feat_l_a)
        # print('--------img_emb_g_a------', np.shape(img_emb_g_a))  # [16, 768]
        # print('--------img_emb_l_a--------', np.shape(img_emb_l_a))  # [16, 768, 19, 19]

        if self.cfg.model.mglrl.b_model and torch.all(imgs_b != 0):
            img_b_global = torch.zeros(img_emb_g_a.shape, dtype=img_emb_g_a.dtype).cuda()
            tmp_remove = imgs_b[:, 0, 0, 0]
            ind_b = (tmp_remove == 0)
            # feed forward to the feature extractor only cases with b_modal images
            imgs_b_continue = imgs_b[~ind_b]

            img_feat_g_b, img_feat_l_b = self.img_encoder_b(imgs_b_continue, get_local=True)
            img_emb_g_b, img_emb_l_b = self.img_encoder_b.generate_embeddings(img_feat_g_b, img_feat_l_b)

            img_b_global[~ind_b] = img_emb_g_b
            img_emb_g = torch.stack([img_emb_g_a, img_b_global], dim=1)
            # img_emb_g = img_emb_g.mean(axis=0)
        else:
            ind_b = []
            img_emb_l_b = []
            img_b_global = torch.zeros(img_emb_g_a.shape, dtype=img_emb_g_a.dtype).cuda()
            img_emb_g = torch.stack([img_emb_g_a, img_b_global], dim=1)
            # img_emb_g = img_emb_g_a

        # print('----------img_emb_l_a---------', np.shape(img_emb_l_a))  # [16, 768, 19, 19]
        # print('----------img_emb_l_b---------', np.shape(img_emb_l_b))  # [16, 768, 19, 19]
        # print('----------img_emb_g-----------', np.shape(img_emb_g))  # [16, 768]

        return img_emb_l_b, img_emb_l_a, img_emb_g, ind_b

    def forward(self, x):
        # img encoder branch
        img_emb_l_b, img_emb_l_a, img_emb_g, ind_b = self.image_encoder_forward(x["imgs_A"], x["imgs_B"])
        if img_emb_l_b != []:
            img_emb_l_b = F.normalize(img_emb_l_b, dim=-1)  # [32, 768, 19, 19]
        img_emb_l_a = F.normalize(img_emb_l_a, dim=-1)  # [32, 768, 19, 19]
        img_emb_g = F.normalize(img_emb_g, dim=-1)  # [32, 2, 768]

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(x["caption_ids"], x["attention_mask"], x["token_type_ids"])  # sents: # (32, 97)
        text_emb_l = F.normalize(text_emb_l, dim=-1)  # [32, 768, 97]
        text_emb_g = F.normalize(text_emb_g, dim=-1)  # [32, 768]

        labels = x["labels"]  # [32, 3]

        ########### Token-level alignment ################
        # cross attention patch to sentences
        mask = torch.from_numpy(np.array(sents)[:, :] == "[PAD]").type_as(img_emb_l_a).bool()
        bz = img_emb_l_a.size(0)
        ih, iw = img_emb_l_a.size(2), img_emb_l_a.size(3)
        sourceL = ih * iw
        img_emb_l_a = img_emb_l_a.view(img_emb_l_a.size(0), -1, sourceL)  # [32, 768, 361]

        if self.cfg.model.mglrl.b_model and img_emb_l_b != []:
            img_emb_l_b = img_emb_l_b.view(img_emb_l_b.size(0), -1, sourceL)
            img_emb_l = torch.zeros(img_emb_l_a.shape, dtype=img_emb_l_a.dtype).cuda()
            img_emb_l[~ind_b] = img_emb_l_b
            img_emb_l[ind_b] = 1e-8 * torch.ones((768, 361), dtype=img_emb_l_a.dtype).cuda()
            img_emb = torch.cat((img_emb_l_a, img_emb_l), dim=2)  # [32, 768, 722]
        else:
            img_emb = img_emb_l_a

        text_emb_l_per = text_emb_l.permute((0, 2, 1))  # [32, 97, 768]
        img_emb_per = img_emb.permute((0, 2, 1))  # [32, 722, 768]

        bt, tt, dt = text_emb_l_per.shape
        text_emb_l_per = self.kan_word(text_emb_l_per.reshape(-1, text_emb_l_per.shape[-1]))
        text_emb_l_per = text_emb_l_per.reshape(bt, tt, dt)

        bi, ti, di = img_emb_per.shape
        img_emb_per = self.kan_patch(img_emb_per.reshape(-1, img_emb_per.shape[-1]))
        img_emb_per = img_emb_per.reshape(bi, ti, di)

        weiContext, attn = self.word_local_atten_layer(text_emb_l_per, img_emb_per, img_emb_per)
        weiContext = F.normalize(weiContext, dim=-1)  # [32, 97, 768]
        text_emb_l_add = torch.add(text_emb_l_per, weiContext)

        weiContext_img, _ = self.patch_local_atten_layer(img_emb_per, text_emb_l_per, text_emb_l_per, key_padding_mask=mask)
        weiContext_img = F.normalize(weiContext_img, dim=-1)  # [32, 722, 768]
        img_emb_l_add = torch.add(img_emb_per, weiContext_img)

        img_emb_l_add_a = img_emb_l_add[:, 0: 361, :]  # [32, 361, 768]

        final_img_g_a = torch.add(img_emb_l_add_a.mean(dim=1), img_emb_g[:, 0, :])  # [32, 768]
        if self.cfg.model.mglrl.b_model and img_emb_l_b != []:
            img_emb_l_add_b = img_emb_l_add[:, 361:, :]  # [32, 361, 768]
            final_img_g_b = torch.add(img_emb_l_add_b.mean(dim=1), img_emb_g[:, 1, :])  # [32, 768]

        text_emb_l_agg = torch.mean(text_emb_l_add, dim=1)  # [32, 768]
        text_emb_l_agg = F.normalize(text_emb_l_agg, dim=-1)

        final_text_g = torch.add(text_emb_l_agg, text_emb_g)  # -[32, 768]

        ########### Disease-level alignment ################
        # normalize prototype layer
        # with torch.no_grad():
        #     w = self.prototype_layer.weight.data.clone()
        #     w = F.normalize(w, dim=1, p=2)
        #     self.prototype_layer.weight.copy_(w)
        with torch.no_grad():
            w = copy.deepcopy(self.prototype_layer.layers)
            self.prototype_layer.layers = w

        # Compute assign code of images
        img_proto_out_a = self.prototype_layer(final_img_g_a)
        report_proto_out = self.prototype_layer(final_text_g)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code_a = torch.exp(
                img_proto_out_a / 0.05).t()  # [256, 32]
            img_code_a = self.get_assignments(
                img_code_a, 20)  # bz, 500  [32, 256]
            report_code = torch.exp(
                report_proto_out / 0.05).t()  # [256, 32]
            report_code = self.get_assignments(
                report_code, 20)  # bz, 500  [32, 256]

        img_proto_prob_a = F.softmax(img_proto_out_a / 0.2, dim=1)
        if self.cfg.model.mglrl.b_model and img_emb_l_b != []:
            img_proto_out_b = self.prototype_layer(final_img_g_b)
            with torch.no_grad():
                img_code_b = torch.exp(
                    img_proto_out_b / 0.05).t()  # [256, 32]
                img_code_b = self.get_assignments(
                    img_code_b, 20)  # bz, 500  [32, 256]
            img_code = torch.add(img_code_a, img_code_b) / 2
            img_proto_prob_b = F.softmax(img_proto_out_b / 0.2, dim=1)
            img_proto_prob = torch.add(img_proto_prob_a, img_proto_prob_b) / 2
        else:
            img_code = img_code_a
            img_proto_prob = img_proto_prob_a

        report_proto_prob = F.softmax(report_proto_out / 0.2, dim=1)

        # img_code_a, report_code
        # save_path_clu_img = './results/diss_clus/split1_image/e8_s539'
        # save_path_clu_repo = './results/diss_clus/split1_report/e8_s539'
        # torch.save(img_code_b, save_path_clu_img + '/clus_img_b' + str(self.count) + '.pt')
        # torch.save(report_code, save_path_clu_repo + '/clus_repo' + str(self.count) + '.pt')
        # torch.save(labels, save_path_clu_img + '/labels' + str(self.count) + '.pt')
        # torch.save(labels, save_path_clu_repo + '/labels' + str(self.count) + '.pt')
        # self.count = self.count + 1

        loss_i2t_proto = - torch.mean(torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - torch.mean(torch.sum(report_code * torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.

        img_proto_prob_a = self.prototype_align_layer(img_proto_prob_a)
        img_proto_prob_a = F.normalize(img_proto_prob_a, dim=-1)  # [32, 768]

        final_img_g_a = torch.add(img_proto_prob_a, img_emb_g[:, 0, :])

        if self.cfg.model.mglrl.b_model and img_emb_l_b != []:
            img_proto_prob_b = self.prototype_align_layer(img_proto_prob_b)
            img_proto_prob_b = F.normalize(img_proto_prob_b, dim=-1)  # [32, 768]

            final_img_g_b = torch.add(img_proto_prob_b, img_emb_g[:, 1, :])

        report_proto_prob = self.prototype_align_layer(report_proto_prob)
        report_proto_prob = F.normalize(report_proto_prob, dim=-1)  # [32, 768]

        final_text_g = torch.add(report_proto_prob, text_emb_g)

        ########### Global-level attention ################
        # Disease-wise Attention
        # After excluding [CLS] and [SEP], characters less than 3 are set to 0.
        mask_mat = (np.array(sents)[:, :] != "[PAD]").astype(int)
        for i in range(len(final_text_g)):
            if sum(mask_mat[i]) - 2 <= 3:
                final_text_g[i] = 0.

        final_img_g_a = final_img_g_a.unsqueeze(1)
        final_text_g = final_text_g.unsqueeze(1)
        if self.cfg.model.mglrl.b_model and img_emb_l_b != []:
            final_img_g_b = final_img_g_b.unsqueeze(1)
            attn_input = torch.cat([final_img_g_a, final_img_g_b, final_text_g], dim=1)  # [32, 3, 768]
        else:
            attn_input = torch.cat([final_img_g_a, torch.zeros_like(final_img_g_a), final_text_g], dim=1)  # [32, 3, 768]

        atten_input_re = attn_input.reshape(-1, attn_input.size(-1))
        qkvs = self.attn_proj(atten_input_re)  # [[32, 3, 1280]]
        # print('-------------', np.shape(qkvs))
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)  # q:[32, 3, 256] v:[32, 3, 256] len(*k):3
        k = [i.reshape(attn_input.size(0), attn_input.size(1), -1) for i in k]
        q = q.reshape(attn_input.size(0), attn_input.size(1), -1)
        v = v.reshape(attn_input.size(0), attn_input.size(1), -1)

        non_zero_rows = []
        for i in range(q.shape[0]):
            attn_i = attn_input[i]
            non_zero_mask = ~torch.all(attn_i == 0, dim=1)
            non_zero_rows.append(q[i][non_zero_mask])

        mean_values = [torch.mean(rows.float(), dim=0) for rows in non_zero_rows]
        q_mean = torch.stack(mean_values)  # [32, 256]

        # compute attention weighting
        ks = torch.stack(k, dim=1)  # [32, 4, 3, 256]
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)  # [32, 4, 3]
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        indices = torch.where(torch.all(attn_input == 0, dim=2))
        for i in range(len(indices[0])):
            attn_logits[indices[0], :, indices[1]] = float('-inf')

        attn_weights = F.softmax(attn_logits, dim=-1)  # [32, 4, 3]

        # save_path = './results/inter_atten'
        # torch.save(attn_weights, save_path + '/attn_weights' + str(self.count) + '.pt')
        # self.count = self.count + 1

        feat_final = torch.matmul(attn_weights, v)  # [32, 4, 256]

        pred_final = self.final_pred_fc(feat_final)  # [32, 4, 4]
        pred = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()  # [32, 4]

        loss_dwa = nn.CrossEntropyLoss()(pred, labels)

        mean_pred = self.mean_fc(attn_input.mean(dim=1))
        loss_cmc = nn.CrossEntropyLoss()(mean_pred, labels)

        loss_pred = (loss_dwa + loss_cmc) / 2.

        # save_path = './results/train100_split3/testNoReportNoWLE_inter/e6_s433'
        # torch.save(pred, save_path + '/pred' + str(self.count) + '.pt')
        # torch.save(labels, save_path + '/labels' + str(self.count) + '.pt')
        # self.count = self.count + 1

        return pred, labels, loss_pred, loss_proto


def attention_fn(query, context, temp1, context_img, use_mask, mask):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    if context_img:
        sourceL = context.size(2)
        contextT = torch.transpose(context, 1, 2).contiguous()
        query = torch.transpose(query, 1, 2).contiguous()
        batch_size, queryL = query.size(0), query.size(2)
    else:
        batch_size = query.size(0)
        sourceL = context.size(1)
        queryL = query.size(2)
        contextT = context

    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)

    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    if use_mask:
        patch_num = queryL
        attn[mask.repeat(patch_num, 1)] = float("-inf")
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    if context_img:
        weightedContext = torch.bmm(context, attnT)
    else:
        contextT = torch.transpose(context, 1, 2).contiguous()
        weightedContext = torch.bmm(contextT, attnT)

    return weightedContext, attn


class SA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(SA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global, weights


def sinkhorn(Q, nmb_iters):
    '''
        :param Q: (num_prototypes, batch size)

    '''
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        K, B = Q.shape

        if torch.cuda.is_available():
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
        else:
            u = torch.zeros(K)
            r = torch.ones(K) / K
            c = torch.ones(B) / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
