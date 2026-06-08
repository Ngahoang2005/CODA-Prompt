import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy

class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # ── G-prompt init (tầng nông, shared, không cần K/A) ──
        for g in self.g_layers:
            p = tensor_prompt(self.g_pool_size, self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        # ── E-prompt init (tầng sâu, task-specific, có K/A) ──
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length  = int(prompt_param[1])
        self.ortho_mu = prompt_param[2]
        
        # Nhận giá trị 200.0 từ bash script (nếu có truyền)
        if len(prompt_param) > 3:
            self.ortho_mu_ge = float(prompt_param[3])
        else:
            self.ortho_mu_ge = 0.01
        # ── Vị trí: G nông, E sâu (theo DualPrompt + triết lý) ──
        self.g_layers    = [0, 1]
        self.e_layers    = [2, 3, 4]

        # ── G chỉ cần 1 vector tĩnh, length = e_length ──
        self.g_pool_size = 1
        self.g_p_length  = self.e_p_length   # giữ cùng length để ViT nhất quán
    @torch.no_grad()
    def get_raw_losses(self):
        """Hàm gom toàn bộ raw loss từ các layer để gửi ra ngoài cho Learner log"""
        raw_e = 0.0
        raw_ge = 0.0
        
        # Gom G_flat một lần
        G_flat = torch.cat([getattr(self, f'g_p_{g}').reshape(-1, self.emb_d) for g in self.g_layers], dim=0)
        
        for l in self.e_layers:
            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            
            raw_e += ortho_penalty(K) + ortho_penalty(A) + ortho_penalty(p.view(p.shape[0], -1))
            E_flat = p.reshape(-1, self.emb_d)
            raw_ge += cross_ortho_penalty(E_flat, G_flat)
            
        return raw_e.item(), raw_ge.item()
    def process_task_count(self):
        self.task_count += 1
        # Chỉ reinit E (G được update liên tục, không cần reinit)
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            setattr(self, f'e_k_{e}', self.gram_schmidt(K))
            setattr(self, f'e_a_{e}', self.gram_schmidt(A))
            setattr(self, f'e_p_{e}', self.gram_schmidt(P))

    def gram_schmidt(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            return (v * u).sum() / denominator * u

        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        vv = vv.T
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        pt = int(self.e_pool_size / self.n_tasks)
        s  = int(self.task_count * pt)
        f  = int((self.task_count + 1) * pt)

        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj   = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        uu = uu.T
        if is_3d:
            uu = uu.view(shape_2d)
        return nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        B, C = x_querry.shape
        loss = 0

        # ════════════════════════════════════════════════
        # G-LAYERS (tầng nông 0,1): Shared, tĩnh, expand
        # ════════════════════════════════════════════════
        if l in self.g_layers:
            p  = getattr(self, f'g_p_{l}')      # (1, g_p_length, emb_d)
            P_ = p.expand(B, -1, -1)             # (B, g_p_length, emb_d)

            i  = self.g_p_length // 2
            Gk = P_[:, :i, :]
            Gv = P_[:, i:, :]

            # G không có ortho loss (pool_size=1 → vô nghĩa)
            return [Gk, Gv], 0, x_block

        # ════════════════════════════════════════════════
        # E-LAYERS (tầng sâu 2,3,4): Task-specific, CODA
        # ════════════════════════════════════════════════
        if l in self.e_layers:
            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')

            pt = int(self.e_pool_size / self.n_tasks)
            s  = int(self.task_count * pt)
            f  = int((self.task_count + 1) * pt)

            # Freeze past, train current
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # CODA attention weighting (giữ nguyên từ code gốc)
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            n_K      = F.normalize(K, dim=1)
            q        = F.normalize(a_querry, dim=2)
            aq_k     = torch.einsum('bkd,kd->bk', q, n_K)
            P_       = torch.einsum('bk,kld->bld', aq_k, p)

            i  = self.e_p_length // 2
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            if train:
                # 1. Ortho loss cho E (giữ nguyên từ code gốc)
                if self.ortho_mu > 0:
                    loss  = ortho_penalty(K) * self.ortho_mu
                    loss += ortho_penalty(A) * self.ortho_mu
                    loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu

                # 2. Cross-ortho G↔E: ép E né không gian của G
                if self.ortho_mu_ge > 0:
                    with torch.no_grad():          # G đứng yên, không nhận gradient
                        G_flat = torch.cat([
                            getattr(self, f'g_p_{g}').reshape(-1, self.emb_d)
                            for g in self.g_layers
                        ], dim=0)                  # (2 * g_p_length//2, emb_d)
                    E_flat = p.reshape(-1, self.emb_d)
                    loss  += cross_ortho_penalty(E_flat, G_flat) * self.ortho_mu_ge

                  
            return [Ek, Ev], loss, x_block

        # Layer không thuộc G hay E
        return None, 0, x_block


def ortho_penalty(t):
    # Fix: dùng device của t thay vì hardcode .cuda()
    return ((t @ t.T - torch.eye(t.shape[0], device=t.device)) ** 2).mean()


def cross_ortho_penalty(t1, t2):
    """Ép E-prompt phải trực giao với G-prompt"""
    n1 = F.normalize(t1, dim=1)
    n2 = F.normalize(t2, dim=1)
    return (n1 @ n2.T).pow(2).mean()


# ─────────────────────────────────────────────────────────
# Các class còn lại giữ nguyên hoàn toàn từ code gốc
# ─────────────────────────────────────────────────────────
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 1
        self.task_id_bootstrap = True
        self.g_layers    = [0, 1]
        self.e_layers    = [2, 3, 4]
        self.g_p_length  = int(prompt_param[2])
        self.e_p_length  = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f'e_k_{l}')
            p = getattr(self, f'e_p_{l}')
            n_K     = F.normalize(K, dim=1)
            q       = F.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            if train:
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_   = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss  = (1.0 - cos_sim[:, k_idx]).sum()
                    P_    = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_    = p[k_idx]
            i  = self.e_p_length // 2
            if train and self.task_id_bootstrap:
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j  = self.g_p_length // 2
            p  = getattr(self, f'g_p_{l}')
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        if e_valid and g_valid:
            p_return = [torch.cat((Ek, Gk), dim=1), torch.cat((Ev, Gv), dim=1)]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k            = 5
        self.task_id_bootstrap = False
        self.g_layers         = []
        self.e_layers         = [0,1,2,3,4] if prompt_param[2] > 0 else [0]
        self.g_p_length       = -1
        self.e_p_length       = int(prompt_param[1])
        self.e_pool_size      = int(prompt_param[0])


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    nn.init.orthogonal_(p) if ortho else nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()
        self.prompt_flag = prompt_flag
        self.task_id     = None

        if pt:
            zoo_model = VisionTransformer(
                img_size=224, patch_size=16, embed_dim=768,
                depth=12, num_heads=12, ckpt_layer=0, drop_path_rate=0
            )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight'], load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        self.last = nn.Linear(768, num_classes)

        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        self.feat = zoo_model

    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q    = q[:, 0, :]
            out, prompt_loss = self.feat(
                x, prompt=self.prompt, q=q, train=train, task_id=self.task_id
            )
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out    = out[:, 0, :]

        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        return out


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)