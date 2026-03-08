"""
S&P 500 对比实验 1: SAC
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random, os, warnings
warnings.filterwarnings('ignore')

CSV_PATH = "sp500_1000d.csv"
LOOKBACK = 15
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TRANSACTION_COST = 0.001
EPISODES = 200
BATCH_SIZE = 64
GAMMA = 0.99
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path):
    df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(20).std()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    delta = df['close'].diff(); gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    df = df.dropna().reset_index(drop=True)
    for col in ['open','high','low','close','volume','return','log_return','volatility','ma5','ma20','rsi']:
        rm = df[col].rolling(min(252, len(df)//2), min_periods=20).mean()
        rs = df[col].rolling(min(252, len(df)//2), min_periods=20).std()
        df[col+'_norm'] = (df[col] - rm) / (rs + 1e-8)
    return df.dropna().reset_index(drop=True)

class TradingEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True); self.feature_cols = [c for c in df.columns if c.endswith('_norm')]
        self.n_features = len(self.feature_cols); self.reset()
    def reset(self):
        self.step_idx = LOOKBACK; self.position = 0.0; self.nav = 1.0; self.nav_peak = 1.0
        self.nav_history = [1.0]; self.done = False; return self._obs()
    def _obs(self):
        w = self.df[self.feature_cols].iloc[self.step_idx-LOOKBACK:self.step_idx].values.flatten().astype(np.float32)
        dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        return np.concatenate([w, [self.position, dd]])
    @property
    def obs_dim(self): return LOOKBACK * self.n_features + 2
    def step(self, action):
        action = np.clip(action, -1, 1); cost = abs(action - self.position) * TRANSACTION_COST
        ret = self.df['return'].iloc[self.step_idx]; pnl = self.position * ret - cost
        self.nav *= (1 + pnl); self.position = action; self.nav_peak = max(self.nav_peak, self.nav)
        self.nav_history.append(self.nav); self.step_idx += 1
        if self.step_idx >= len(self.df) - 1: self.done = True
        if len(self.nav_history) > 2:
            r = np.diff(self.nav_history[-20:]); reward = np.mean(r)/(np.std(r)+1e-8) if np.std(r)>1e-8 else pnl*10
        else: reward = pnl * 10
        return self._obs(), reward, self.done

class GaussianPolicy(nn.Module):
    def __init__(self, od):
        super().__init__(); self.fc1=nn.Linear(od,128); self.fc2=nn.Linear(128,128)
        self.mu=nn.Linear(128,1); self.ls=nn.Linear(128,1)
    def forward(self, x):
        x=F.relu(self.fc1(x)); x=F.relu(self.fc2(x)); return self.mu(x), self.ls(x).clamp(-20,2)
    def sample(self, x):
        m,ls=self.forward(x); d=torch.distributions.Normal(m,ls.exp())
        z=d.rsample(); a=torch.tanh(z); lp=d.log_prob(z)-torch.log(1-a.pow(2)+1e-6)
        return a, lp.sum(-1,keepdim=True)

class SoftQ(nn.Module):
    def __init__(self, od):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(od+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self, o, a): return self.net(torch.cat([o,a],-1))

def compute_metrics(nav):
    nav=np.array(nav); r=np.diff(nav)/nav[:-1]; cr=(nav[-1]/nav[0]-1)*100
    pk=np.maximum.accumulate(nav); mdd=np.max((pk-nav)/pk)*100
    n=len(r); ar=(nav[-1]/nav[0])**(252/max(n,1))-1; av=np.std(r)*np.sqrt(252)
    sh=ar/(av+1e-8); ca=ar/(mdd/100+1e-8)
    ds=r[r<0]; dv=np.std(ds)*np.sqrt(252) if len(ds)>0 else 1e-8; so=ar/(dv+1e-8)
    return {'CR(%)':round(cr,2),'MDD(%)':round(mdd,2),'Sharpe':round(sh,2),'Calmar':round(ca,2),'Sortino':round(so,2)}

def main():
    print("🚀 SAC on S&P 500")
    df = load_data(CSV_PATH); n=len(df); tr=df.iloc[:int(n*0.7)]; te=df.iloc[int(n*0.8):]
    env=TradingEnv(tr); od=env.obs_dim
    pi=GaussianPolicy(od).to(DEVICE); q1=SoftQ(od).to(DEVICE); q2=SoftQ(od).to(DEVICE)
    q1t=SoftQ(od).to(DEVICE); q2t=SoftQ(od).to(DEVICE)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    op=optim.Adam(pi.parameters(),lr=LR); oq1=optim.Adam(q1.parameters(),lr=LR); oq2=optim.Adam(q2.parameters(),lr=LR)
    buf=deque(maxlen=50000); best=0
    for ep in range(EPISODES):
        s=env.reset()
        while not env.done:
            st=torch.FloatTensor(s).unsqueeze(0).to(DEVICE); a,_=pi.sample(st)
            an=a.cpu().detach().numpy().flatten()[0]; ns,r,d=env.step(an)
            buf.append((s,an,r,ns,float(d)))
            if len(buf)>=BATCH_SIZE:
                batch=random.sample(buf,BATCH_SIZE); ss,aa,rr,nss,dd=[np.array(x) for x in zip(*batch)]
                ss=torch.FloatTensor(ss).to(DEVICE);aa=torch.FloatTensor(aa).unsqueeze(1).to(DEVICE)
                rr=torch.FloatTensor(rr).unsqueeze(1).to(DEVICE);nss=torch.FloatTensor(nss).to(DEVICE)
                dd=torch.FloatTensor(dd).unsqueeze(1).to(DEVICE)
                with torch.no_grad():
                    na,nlp=pi.sample(nss); tgt=rr+GAMMA*(1-dd)*(torch.min(q1t(nss,na),q2t(nss,na))-0.2*nlp)
                for o,q in [(oq1,q1),(oq2,q2)]: o.zero_grad();F.mse_loss(q(ss,aa),tgt).backward();o.step()
                na2,lp2=pi.sample(ss); op.zero_grad();(0.2*lp2-torch.min(q1(ss,na2),q2(ss,na2))).mean().backward();op.step()
                for p,tp in list(zip(q1.parameters(),q1t.parameters()))+list(zip(q2.parameters(),q2t.parameters())):
                    tp.data.copy_(0.005*p.data+0.995*tp.data)
            s=ns
        if env.nav_history[-1]>best: best=env.nav_history[-1]; bs={k:v.clone() for k,v in pi.state_dict().items()}
        if (ep+1)%50==0: print(f"  Episode {ep+1}/{EPISODES} | NAV: {env.nav_history[-1]:.4f}")
    pi.load_state_dict(bs)
    et=TradingEnv(te); s=et.reset()
    while not et.done:
        st=torch.FloatTensor(s).unsqueeze(0).to(DEVICE); m,_=pi(st); s,_,_=et.step(torch.tanh(m).cpu().detach().numpy().flatten()[0])
    m=compute_metrics(et.nav_history)
    print(f"\n{'='*50}\n  SAC S&P 500 OOS Results\n{'='*50}")
    for k,v in m.items(): print(f"  {k}: {v}")
    pd.DataFrame([m],index=["SAC"]).to_csv("sp500_sac_results.csv"); print("✅ 保存到 sp500_sac_results.csv")

if __name__=="__main__": main()
