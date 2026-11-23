"""Create a diagnostic contact sheet of the validation videos (no metric changes)."""
import argparse
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--videos',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.videos)
seeds=sorted(root.glob('seed_*'))
w,h,label=256,144,24
sheet=Image.new('RGB',(5*w,len(seeds)*3*(h+label)),(245,245,245))
draw=ImageDraw.Draw(sheet)
for seed_index,folder in enumerate(seeds):
 for level in range(5):
  cap=cv2.VideoCapture(str(folder/f'flowedit_rgb{float(level)}.mp4'))
  n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  for row,index in enumerate([0,n//2,n-1]):
   cap.set(cv2.CAP_PROP_POS_FRAMES,index)
   ok,frame=cap.read()
   if not ok:raise RuntimeError(f'Cannot read {folder}/{level}/{index}')
   image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((w,h))
   x=level*w;y=(seed_index*3+row)*(h+label)
   draw.text((x+5,y+4),f'{folder.name}  scale={level}  frame={index}',fill=(0,0,0))
   sheet.paste(image,(x,y+label))
  cap.release()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
sheet.save(a.output)
