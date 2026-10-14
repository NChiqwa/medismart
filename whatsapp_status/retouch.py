import cv2, numpy as np, sys
img=cv2.imread(sys.argv[1]); H,W=img.shape[:2]
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32)
L=lab[...,0]
def ell(cx,cy,rx,ry,ang=0):
    m=np.zeros((H,W),np.float32); cv2.ellipse(m,(cx,cy),(rx,ry),ang,0,360,1,-1)
    return cv2.GaussianBlur(m,(0,0),max(rx,ry)*0.35)
# under-eye areas (below each eye)
under=np.maximum(ell(2350,2025,125,52,-10), ell(2725,1948,150,55,-8))
low=cv2.GaussianBlur(L,(0,0),8); ref=np.roll(cv2.GaussianBlur(L,(0,0),20),-95,axis=0)
lift=np.clip(ref-low,0,None)*0.95+3
lab[...,0]=L+lift*under
for c in (1,2):
    ch=lab[...,c]; lab[...,c]=ch+(np.roll(cv2.GaussianBlur(ch,(0,0),20),-95,axis=0)-cv2.GaussianBlur(ch,(0,0),10))*under*0.8
# gentle skin smoothing on face (forehead/cheeks/nose), frequency separation keeps texture
face=ell(2560,1900,430,380)
out=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)
sm=cv2.bilateralFilter(out,0,18,9)
skin=cv2.inRange(cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_BGR2HSV),(0,25,90),(25,200,255)).astype(np.float32)/255
skin=cv2.GaussianBlur(skin,(0,0),6)*face
out=out*(1-0.4*skin[...,None])+sm*0.4*skin[...,None]
# subtle face brighten / glow
out=out*(1+0.06*face[...,None])
cv2.imwrite(sys.argv[2],np.clip(out,0,255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,96])
b=cv2.imread(sys.argv[1])[1650:2250,2150:3000]; a=np.clip(out,0,255).astype(np.uint8)[1650:2250,2150:3000]
cv2.imwrite('frames/retouch_cmp.jpg',np.hstack([b,a]))
