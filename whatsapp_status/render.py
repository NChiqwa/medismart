import cv2, numpy as np, subprocess, sys
V, IMG, OUT = sys.argv[1:4]; ST=float(sys.argv[4]); LN=float(sys.argv[5]); BASE=float(sys.argv[6])
W,H,FPS=720,1280,30
src=cv2.imread(IMG).astype(np.float32)/255
ih,iw=src.shape[:2]
PW,PH=W,int(ih*W/iw)            # full photo, fit to width (720x960)
PY=0
ph=cv2.resize(src,(PW,PH),interpolation=cv2.INTER_AREA)
# bright, radiant grade
ph=np.clip(ph*1.12+0.02,0,1)
g=ph.mean(2,keepdims=True); ph=g+(ph-g)*1.08
lum=ph.mean(2,keepdims=True)
ph=ph*(np.array([0.93,0.98,1.06])*(1-lum)+np.array([0.97,1.0,1.04])*lum)
bloom=cv2.GaussianBlur(np.clip(ph-0.6,0,1),(0,0),20); ph=np.clip(1-(1-ph)*(1-bloom*0.6),0,1)
# backdrop: blurred, darker cover version of the same photo
cov=cv2.resize(src,(int(iw*H/ih),H),interpolation=cv2.INTER_AREA); ox=(cov.shape[1]-W)//2
back=cv2.GaussianBlur(cov[:,ox:ox+W],(0,0),30)*0.42
yy=np.arange(H)[:,None]
back=back*(1-0.35*np.clip((yy-PH)/(H-PH),0,1))[...,None]
m=np.ones((PH,PW),np.float32); fe=60
m[PH-fe:]=np.linspace(1,0,fe)[:,None]
bg=back.copy(); bg[PY:PY+PH]=ph*m[...,None]+bg[PY:PY+PH]*(1-m[...,None])
bg=bg.astype(np.float32); bw,bh=W,H
def P(X,Y): return (X*PW/iw, PY+Y*PH/ih)
# laptop screen quad (photo coords) -> bg coords
quad=np.float32([P(488,2030),P(1278,2100),P(1295,2640),P(500,2700)])
SW,SH=720,560
Hscr=cv2.getPerspectiveTransform(np.float32([[0,0],[SW,0],[SW,SH],[0,SH]]),quad)
gy=np.linspace(0,1,SH)[:,None,None]
screen_bg=(np.array([0.22,0.10,0.06])*(1-gy)+np.array([0.35,0.16,0.08])*gy).astype(np.float32)*np.ones((SH,SW,3),np.float32)*0.6
dur=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',V]))
N=int(dur*FPS)
dec=subprocess.Popen(['ffmpeg','-v','error','-ss',str(ST),'-t',str(LN),'-i',V,'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
  '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',OUT],stdin=subprocess.PIPE)
rng=np.random.default_rng(7); NP=50
Pt=dict(x=rng.uniform(0,W,NP),y=rng.uniform(0,H,NP),r=rng.uniform(2,8,NP),v=rng.uniform(8,28,NP),
        ph=rng.uniform(0,6.28,NP),sw=rng.uniform(0.3,1.0,NP),a=rng.uniform(0.25,0.7,NP))
pc=np.array([[0.55,0.85,1.0],[0.8,0.95,1.0],[0.6,1.0,0.85]],np.float32); Pt['c']=pc[rng.integers(0,3,NP)]
ry,rx=np.mgrid[0:H,0:W].astype(np.float32)
ang=np.arctan2(ry+150,rx-W*0.85); dist=np.hypot(ry+150,rx-W*0.85)
def rays(t):
    r=0.5+0.5*np.sin(ang*22+t*0.35)*np.sin(ang*9-t*0.22)
    return (np.clip(r,0,1)**3*np.exp(-dist/900)).astype(np.float32)
SHIFT=340
base=None; env=1.0; i=0; prev_mass=0; pulse=0.0
while True:
    buf=dec.stdout.read(W*H*3)
    if len(buf)<W*H*3: break
    f=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)/255
    ts=ST+i/FPS; t=ts*FPS/max(N,1)
    mx=f.max(2); sat=mx-f.min(2)
    a=np.maximum(np.clip((mx-0.80)/0.10,0,1), np.clip((mx-0.62)/0.08,0,1)*np.clip((sat-0.30)/0.08,0,1))
    a[:480]=0; a[1080:]=0
    mass=float(a.mean())
    if i==0: prev_mass=mass
    if mass-prev_mass>0.004: pulse=1.0
    prev_mass=0.7*prev_mass+0.3*mass; pulse*=0.92
    a=cv2.GaussianBlur(a,(3,3),0)
    # camera: gentle push-in, framing laptop + face
    s=1.0+0.03*t+0.004*np.sin(ts*0.9)+0.012*pulse
    tx=W/2-W/2*s; ty=480-480*s
    tx=min(0,max(W-bw*s,tx)); ty=min(0,max(H-bh*s,ty))
    A=np.float64([[s,0,tx],[0,s,ty],[0,0,1]])
    fr=cv2.warpAffine(bg,A[:2],(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    l=float(f[200:1100].mean())
    if base is None: base=BASE
    if i==0: env=np.clip(l/base*1.05,0.25,1)
    env=0.85*env+0.15*np.clip(l/base*1.05,0.25,1)
    fr*=env
    # --- laptop screen shows the message
    cap=(f*a[...,None])[480:1040]            # 560 rows of caption area
    scr=screen_bg*(0.85+0.15*np.sin(ts*1.3))
    scr=np.clip(scr+cap*1.15+cv2.GaussianBlur(cap,(0,0),8)*0.8,0,1)
    Hs=A@Hscr
    warped=cv2.warpPerspective(scr,Hs,(W,H),flags=cv2.INTER_AREA)
    sm=cv2.warpPerspective(np.ones((SH,SW),np.float32),Hs,(W,H))
    glow=cv2.GaussianBlur(warped,(0,0),22)*1.4+cv2.GaussianBlur(sm[...,None]*np.array([0.9,0.55,0.3],np.float32),(0,0),30)*0.25
    fr=fr*(1-sm[...,None])+warped*sm[...,None]
    fr=1-(1-np.clip(fr,0,1))*(1-np.clip(glow,0,1)*0.7)
    # --- effects
    fr=fr+rays(ts)[...,None]*np.array([0.6,0.8,0.95],np.float32)*0.16*env
    pl=np.zeros((H//2,W//2,3),np.float32)
    yy_=(Pt['y']-Pt['v']*ts)%(H+40)-20; xx_=Pt['x']+25*np.sin(ts*Pt['sw']+Pt['ph'])
    tw=0.55+0.45*np.sin(ts*2.2*Pt['sw']+Pt['ph'])
    for k in range(NP):
        cv2.circle(pl,(int(xx_[k]/2),int(yy_[k]/2)),max(1,int(Pt['r'][k]/2)),(Pt['c'][k]*Pt['a'][k]*tw[k]).tolist(),-1,cv2.LINE_AA)
    pl=cv2.resize(cv2.GaussianBlur(pl,(0,0),1.6),(W,H)); pl=pl+cv2.GaussianBlur(pl,(0,0),8)*1.5
    fr=1-(1-np.clip(fr,0,1))*(1-np.clip(pl,0,1)*0.7*env)
    fr=fr+pulse*0.08*env
    if ts<1.2: fr*=ts/1.2
    # --- main captions, moved lower so they don't cover the laptop
    a2=np.zeros_like(a); f2=np.zeros_like(f)
    a2[SHIFT:]=a[:H-SHIFT]; f2[SHIFT:]=f[:H-SHIFT]
    band=np.exp(-((np.arange(H)[:,None]-1100)/180.0)**2)*0.35
    sh=np.maximum(cv2.GaussianBlur(a2,(0,0),9)*0.8, band*(mass>0.001))
    fr=fr*(1-sh[...,None])
    fr=fr+cv2.GaussianBlur(f2*a2[...,None],(0,0),6)*0.6
    fr=fr*(1-a2[...,None])+f2*a2[...,None]
    fr=fr+rng.normal(0,0.01,(H//2,W//2,1)).astype(np.float32).repeat(2,0).repeat(2,1)
    enc.stdin.write((np.clip(fr,0,1)*255).astype(np.uint8).tobytes()); i+=1
enc.stdin.close(); enc.wait(); print('frames',i)
