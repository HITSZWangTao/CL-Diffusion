#encoding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Hyperparameters"""

#WiFi CSI parameters
NUM_LINKS = 240
NUM_CHANNELS = 14
NUM_SUBCARRIERS = 52  

#Embedding dim parameters
PER_LINK_EMB = 128 #dim D in paper
TIME_EMB_DIM = 128
COND_EMB_DIM = 64 
DECODER_HIDDEN = 128

#Transformer (Attention-Residual mechanism)
TRANSFORMER_DIM = 256
TRANSFORMER_HEAD = 8
TRANSFORMER_LAYERS = 4

#other
USE_SPECTRAL_NORM = False

def spectral_norm(module):
    if USE_SPECTRAL_NORM:
        return nn.utils.spectral_norm(module)
    return module

def default_init(m,std=0.02):
    if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
        nn.init.normal_(m.weight,0.0,std)
    if getattr(m,"bias",None) is not None:
        nn.init.constant_(m.bias,0.0)

#Time Embedding For Diffusion step
class TimePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        """
        t: (B,)  scalar timesteps (int or float)
        returns (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(- math.log(10000) * torch.arange(0, half, device=device).float() / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0,1), value=0)
        return emb  # (B, dim)
    
#Link Encoder, through convlution layers and a adaptive weight scheme to
#obtain link-level feature
class PerLinkEncoder(nn.Module):
    def __init__(self,in_ch=NUM_CHANNELS,subc=NUM_SUBCARRIERS,out_dim=PER_LINK_EMB):
        super().__init__()
        #two one-dimensional convlution layers for subcarriers
        self.spectral = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch,64,kernel_size=(1,5),padding=(0,2),bias=False)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            spectral_norm(nn.Conv2d(64,128,kernel_size=(1,5),padding=(0,2),bias=False)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        #another convolution along the frquency channel 
        self.channel_fuse = nn.Sequential(
            spectral_norm(nn.Conv2d(128,128,kernel_size=(3,1),padding=(1,0),bias=False)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)) #adaptive layer to reduce size
        )

        #adaptive weighting scheme
        self.se_fc = nn.Sequential(
            nn.Linear(128,128//4),
            nn.GELU(),
            nn.Linear(128//4,128),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(128,out_dim)
        self.apply(default_init)
    
    def forward(self,x):
        '''
        x: (B * Link Num * Measurement Num, Frequency Channel, Subcarrier)
        return: B * Link Num * Measurement Num, out_dim (D in papper)
        '''
        x = x.contiguous()
        x = x.unsqueeze(2) # B,C,1,S
        x = self.spectral(x) #Subcarrier Dimension
        x_cf = self.channel_fuse(x) # Frequency Channel Dimension
        x_cf = x_cf.view(x_cf.size(0),-1) # flatten
        adaptive_weight = self.se_fc(x_cf)
        x_adaptive = x_cf * adaptive_weight
        out = self.fc(x_adaptive) #B, D
        return out
    
class PerLinkDecoder(nn.Module):
    def __init__(self,out_ch=NUM_CHANNELS,subc=NUM_SUBCARRIERS,in_dim=PER_LINK_EMB):
        super().__init__()
        self.out_ch  = out_ch
        self.subc = subc
        self.fc = nn.Linear(in_dim,out_ch*subc)
        self.refine = nn.Sequential(
            spectral_norm(nn.Conv1d(out_ch,out_ch,kernel_size=3,padding=1,bias=False)),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            spectral_norm(nn.Conv1d(out_ch,out_ch,kernel_size=3,padding=1,bias=False)),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.apply(default_init)
    
    def forward(self,z):
        """
        z: (B,in_dim)
        returns: (B,frequency channels,subcarriers)
        """
        B = z.size(0)
        x = self.fc(z) # B, out_ch * subc
        x = x.view(B,self.out_ch,self.subc)
        x = self.refine(x)
        return  x
    
class LinkTransformer(nn.Module):
    def __init__(self,d_model=TRANSFORMER_DIM,nhead=TRANSFORMER_HEAD,num_layers=TRANSFORMER_LAYERS,dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(PER_LINK_EMB+TIME_EMB_DIM+COND_EMB_DIM,d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                                                   dim_feedforward= d_model * 4,
                                                   dropout=dropout,activation='gelu',batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.apply(default_init)
    
    def forward(self,per_link_emb,time_emb_broadcast,cond_emb_broadcast,src_mask=None):
        """
        per_link_emb: B,L,PER_LINL_EMB
        time_emb_broadcast: (B,L,TIME_EMB_DIM) 
        cond_emb_broadcast: (B,L,COND_EMB_DIM)
        return B,L,d_model 
        """
        B,L,_ = per_link_emb.shape
        #concatenating link-level repesentation with embedding corresponding to diffusion step t
        # and auxiliary conditions c
        x = torch.cat([per_link_emb,time_emb_broadcast,cond_emb_broadcast],dim=-1) 
        x = self.input_proj(x) #adjust dimension
        x = self.ln(x)
        x = self.transformer(x,src_key_padding_mask=None)
        return x


class DiffusionModel(nn.Module):
    def __init__(self,LinkMeasurement=1):
        super().__init__()
        self.per_link_encoder = PerLinkEncoder(in_ch=NUM_CHANNELS,subc=NUM_SUBCARRIERS,out_dim=PER_LINK_EMB)
        self.time_emb = TimePositionalEmbedding(TIME_EMB_DIM)
        self.time_mlp = nn.Sequential(nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM), nn.GELU(), nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM))
        self.cond_mlp = nn.Sequential(nn.Linear(COND_EMB_DIM if False else 3, COND_EMB_DIM),
                                      nn.GELU(), nn.Linear(COND_EMB_DIM, COND_EMB_DIM))

        self.link_proj = nn.Linear(PER_LINK_EMB,PER_LINK_EMB)
        self.link_transformer = LinkTransformer(d_model=TRANSFORMER_DIM,nhead=TRANSFORMER_HEAD,
                                                num_layers=TRANSFORMER_LAYERS)
        
        self.fuison_mlp = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM,DECODER_HIDDEN),
            nn.GELU(),
            nn.Linear(DECODER_HIDDEN,PER_LINK_EMB),
        )

        self.per_link_decoder = PerLinkDecoder(out_ch=NUM_CHANNELS,
                                               subc=NUM_SUBCARRIERS,
                                               in_dim=PER_LINK_EMB)

        #accelete training
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        if LinkMeasurement > 1:
            self.measurement_fusion = nn.Sequential(
                nn.Linear(PER_LINK_EMB * LinkMeasurement,PER_LINK_EMB),
                nn.GELU(),
                nn.Linear(PER_LINK_EMB,LinkMeasurement * PER_LINK_EMB)
            )


        self.apply(default_init)
        self.LinkMeasurement = LinkMeasurement

    def forward(self,x_t,t,cond):
        """
        x_t: (B,L,N,C,S)
        t: (B,)
        cond: B,N,3 (tuber ID, tubuer_position_x,tuber_position_y)
        """
        B,L,N,C,S = x_t.shape
        assert L == NUM_LINKS and C == NUM_CHANNELS and S == NUM_SUBCARRIERS
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)
        

        link_measurment_results = []

        for ii in range(N):
            cond_emb = self.cond_mlp(cond[:,ii,:])
            x_resh = x_t[:,:,ii,:,:].view(B*L,C,S)
            per_link = self.per_link_encoder(x_resh)
            per_link = per_link.view(B,L,-1) #B,L,D
            per_link = self.link_proj(per_link) #B,L,D
            t_b = t_emb.unsqueeze(1).expand(-1,L,-1) #B,L,TimeEmb
            c_b = cond_emb.unsqueeze(1).expand(-1,L,-1) #B,L,CondEmb
            
            #attention-residual
            link_ctx = self.link_transformer(per_link,t_b,c_b)
            fused = self.fuison_mlp(link_ctx)
            fused = fused + per_link #B,L,PER_LINK_EMB
            link_measurment_results.append(fused)

        #capture multiple link measurement (B,L,PER_LINK_EMB*N)
        link_measurment_results = torch.cat(link_measurment_results,dim=-1)
        if self.LinkMeasurement > 1:
            fused_results = self.measurement_fusion(link_measurment_results)
        else:
            fused_results = link_measurment_results
        
        #B,L,N,PER_LINK_EMB
        fused_results = fused_results.reshape(B,L,N,-1)

        #decoder
        recon_results = []
        for jj in range(N):
            fused_flat = fused_results[:,:,jj,:].reshape(B*L,-1)
            #B*L,C,S
            recon_single = self.per_link_decoder(fused_flat)
            recon_results.append(recon_single.reshape(B,L,C,S))
        
        recon = torch.stack(recon_results,dim=2)
        x0_preed = recon * self.output_scale
        x0_final = x_t + x0_preed
        return x0_final.contiguous()
        

        
        



