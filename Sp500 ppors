"""
S&P 500 对比实验 2: PPO-RS (PPO + Risk Shaping)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os, warnings
warnings.filterwarnings('ignore')

CSV_PATH = "sp500_1000d.csv"
LOOKBACK = 15
EPISODES = 200
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
        self.nav_history = [1.0]; self.ret_history = []; self.done = False; return self._obs()
    def _obs(self):
        w = self.df[self.feature_cols].iloc[self.step_idx-LOOKBACK:self.step_idx].values.flatten().astype(np.float32)
        dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        return np.concatenate([w, [self.position, dd]])
    @property
    def obs_dim(self): return LOOKBACK * self.n_features + 2
    def step(self, action):
        action = np.clip(action, -1, 1); cost = abs(action - self.position) * 0.001
        ret = self.df['return'].iloc[self.step_idx]; pnl = self.position * ret - cost
        self.nav *= (1 + pnl); self.position = action; self.nav_peak = max(self.nav_peak, self.nav)
        self.nav_history.append(self.nav); self.ret_history.append(pnl); self.step_idx += 1
        if self.step_idx >= len(self.df) - 1: self.done = True
        # Risk-shaped reward
        if len(self.ret_history) > 2:
            rc = np.array(self.ret_history[-20:]); base = np.mean(rc)/(np.std(rc)+1e-8) if np.std(rc)>1e-8 else pnl*10
        else: base = pnl * 10
        dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        dd_pen = -5.0 * (dd - 0.07) if dd > 0.07 else 0.0
        vol_pen = -2.0 * max(np.std(self.ret_history[-20:]) - 0.02, 0) if len(self.ret_history) > 5 else 0.0
        return self._obs(), base + dd_pen + vol_pen, self.done

class ActorCritic(nn.Module):
    def __init__(self, od):
        super().__init__()
        self.a1=nn.Linear(od,128); self.a2=nn.Linear(128,128); self.am=nn.Linear(128,1)
        self.als=nn.Parameter(torch.zeros(1))
        self.c1=nn.Linear(od,128); self.c2=nn.Linear(128,128); self.co=nn.Linear(128,1)
    def actor(self, x):
        x=F.relu(self.a1(x)); x=F.relu(self.a2(x)); return self.am(x), self.als.exp().expand_as(self.am(x))
    def critic(self, x):
        x=F.relu(self.c1(x)); x=F.relu(self.c2(x)); return self.co(x)

def compute_metrics(nav):
    nav=np.array(nav); r=np.diff(nav)/nav[:-1]; cr=(nav[-1]/nav[0]-1)*100
    pk=np.maximum.accumulate(nav); mdd=np.max((pk-nav)/pk)*100
    n=len(r); ar=(nav[-1]/nav[0])**(252/max(n,1))-1; av=np.std(r)*np.sqrt(252)
    sh=ar/(av+1e-8); ca=ar/(mdd/100+1e-8)
    ds=r[r<0]; dv=np.std(ds)*np.sqrt(252) if len(ds)>0 else 1e-8; so=ar/(dv+1e-8)
    return {'CR(%)':round(cr,2),'MDD(%)':round(mdd,2),'Sharpe':round(sh,2),'Calmar':round(ca,2),'Sortino':round(so,2)}

def main():
    print("🚀 PPO-RS on S&P 500")
    df = load_data(CSV_PATH); n=len(df); tr=df.iloc[:int(n*0.7)]; te=df.iloc[int(n*0.8):]
    env=TradingEnv(tr); model=ActorCritic(env.obs_dim).to(DEVICE)
    optimizer=optim.Adam(model.parameters(),lr=LR); best=0; best_state=None
    for ep in range(EPISODES):
        s=env.reset(); states,actions,rewards,dones,lps,vals=[],[],[],[],[],[]
        while not env.done:
            st=torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
            m,std=model.actor(st); d=Normal(m,std); z=d.sample(); a=torch.tanh(z)
            lp=(d.log_prob(z)-torch.log(1-a.pow(2)+1e-6)).sum().item(); v=model.critic(st).item()
            an=a.cpu().detach().numpy().flatten()[0]; ns,r,dn=env.step(an)
            states.append(s);actions.append(an);rewards.append(r);dones.append(float(dn));lps.append(lp);vals.append(v)
            s=ns
        lv=model.critic(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).item()
        vs=np.array(vals+[lv]); rw=np.array(rewards); dn=np.array(dones)
        adv=np.zeros_like(rw); g=0
        for t in reversed(range(len(rw))): d=rw[t]+GAMMA*vs[t+1]*(1-dn[t])-vs[t]; g=d+GAMMA*0.95*(1-dn[t])*g; adv[t]=g
        ret=adv+vs[:-1]; adv=(adv-adv.mean())/(adv.std()+1e-8)
        st_t=torch.FloatTensor(np.array(states)).to(DEVICE); at_t=torch.FloatTensor(np.array(actions)).unsqueeze(1).to(DEVICE)
        olp=torch.FloatTensor(np.array(lps)).unsqueeze(1).to(DEVICE); rt=torch.FloatTensor(ret).unsqueeze(1).to(DEVICE)
        av_t=torch.FloatTensor(adv).unsqueeze(1).to(DEVICE)
        for _ in range(4):
            m,std=model.actor(st_t); d=Normal(m,std); z=torch.atanh(at_t.clamp(-0.999,0.999))
            nlp=(d.log_prob(z)-torch.log(1-at_t.pow(2)+1e-6)).sum(-1,keepdim=True)
            ratio=torch.exp(nlp-olp); s1=ratio*av_t; s2=torch.clamp(ratio,0.8,1.2)*av_t
            v=model.critic(st_t); loss=-torch.min(s1,s2).mean()+0.5*F.mse_loss(v,rt)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); optimizer.step()
        if env.nav_history[-1]>best: best=env.nav_history[-1]; best_state={k:v.clone() for k,v in model.state_dict().items()}
        if (ep+1)%50==0: print(f"  Episode {ep+1}/{EPISODES} | NAV: {env.nav_history[-1]:.4f}")
    if best_state: model.load_state_dict(best_state)
    et=TradingEnv(te); s=et.reset()
    while not et.done:
        st=torch.FloatTensor(s).unsqueeze(0).to(DEVICE); m,_=model.actor(st); s,_,_=et.step(torch.tanh(m).cpu().detach().numpy().flatten()[0])
    m=compute_metrics(et.nav_history)
    print(f"\n{'='*50}\n  PPO-RS S&P 500 OOS Results\n{'='*50}")
    for k,v in m.items(): print(f"  {k}: {v}")
    pd.DataFrame([m],index=["PPO-RS"]).to_csv("sp500_ppors_results.csv"); print("✅ 保存到 sp500_ppors_results.csv")

if __name__=="__main__": main()
