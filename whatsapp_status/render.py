import cv2, numpy as np, subprocess, sys
V, IMG, OUT = sys.argv[1:4]
W,H,FPS=720,1280,30
# --- background: crop photo to 9:16 around the person, grade it ---
img=cv2.imread(IMG).astype(np.float32)/255
ih,iw=img.shape[:2]; cw=int(ih*9/16); x0=min(iw-cw, int(iw*0.25))
bg=img[:,x0:x0+cw]; bg=cv2.resize(bg,(int(W*1.25),int(H*1.25)),interpolation=cv2.INTER_AREA)
g=bg.mean(2,keepdims=True); bg=g+(bg-g)*0.8                    # slight desaturate
bg=np.clip((bg-0.5)*1.12+0.47,0,1)                               # contrast
lum=bg.mean(2,keepdims=True)
sh=np.array([0.62,0.48,0.40]); hi=np.array([0.92,0.97,1.04])    # BGR: cool/teal shadows, warm highlights
bg=bg*(sh*(1-lum)+hi*lum)*1.18
bg=np.clip(bg*0.82,0,1)**1.45                                    # moody, darker wall
bh,bw=bg.shape[:2]
yy,xx=np.mgrid[0:bh,0:bw]; cx,cy=bw*0.62,bh*0.40
vig=1-0.80*np.clip(((xx-cx)/(bw*0.75))**2+((yy-cy)/(bh*0.6))**2,0,1)
grad=1-0.6*np.clip((yy/bh-0.5)/0.5,0,1)**1.2                 # darker bottom so captions pop
bg=(bg*(vig*grad)[...,None]).astype(np.float32)
dur=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',V]))
N=int(dur*FPS)
dec=subprocess.Popen(['ffmpeg','-v','error','-i',V,'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
  '-i',V,'-map','0:v','-map','1:a','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
  '-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',OUT],stdin=subprocess.PIPE)
fx,fy=bw*0.71,bh*0.42   # focus point (face)
base=None; env=1.0; i=0
yb=np.arange(H)[:,None,None]; band=(1-0.35*np.exp(-((yb-740)/170.0)**2)).astype(np.float32)
while True:
    buf=dec.stdout.read(W*H*3)
    if len(buf)<W*H*3: break
    f=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)/255
    t=i/max(N,1)
    # Ken Burns: slow push-in toward the face + gentle breathing
    s=(W/bw)*(1.30+0.14*t+0.01*np.sin(i/FPS*0.9))
    tx=W*0.56-fx*s; ty=H*0.27-fy*s
    tx=min(0,max(W-bw*s,tx)); ty=min(0,max(H-bh*s,ty))
    M=np.float32([[s,0,tx],[0,s,ty]])
    fr=cv2.warpAffine(bg,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    # follow the original's fades to black
    l=float(f[200:1100].mean())
    if base is None: base=max(l,0.05)
    env=0.85*env+0.15*np.clip(l/base*1.05,0,1)
    fr*=env*band
    # caption matte from the original frame
    mx=f.max(2); sat=mx-f.min(2)
    a=np.maximum(np.clip((mx-0.80)/0.10,0,1), np.clip((mx-0.62)/0.08,0,1)*np.clip((sat-0.30)/0.08,0,1))
    a[:480]=0; a[1080:]=0
    a=cv2.GaussianBlur(a,(3,3),0)
    shadow=cv2.GaussianBlur(a,(0,0),9)*0.75
    fr=fr*(1-shadow[...,None])
    glow=cv2.GaussianBlur(f*a[...,None],(0,0),6)*0.6
    fr=fr+glow
    fr=fr*(1-a[...,None])+f*a[...,None]
    enc.stdin.write((np.clip(fr,0,1)*255).astype(np.uint8).tobytes()); i+=1
enc.stdin.close(); enc.wait(); print('frames',i)
