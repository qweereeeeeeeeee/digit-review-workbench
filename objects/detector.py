"""Use Ultralytics' letterbox/NMS/coordinate restoration, not a custom approximation."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'var/matplotlib'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'var/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
from ultralytics import YOLO, settings

# Local photographs must not be sent to experiment-tracking services.
settings.update({'sync':False,'wandb':False,'comet':False,'mlflow':False,'clearml':False})

NAMES_ZH='人 自行车 汽车 摩托车 飞机 公交车 火车 卡车 船 交通灯 消防栓 停车标志 停车计时器 长椅 鸟 猫 狗 马 羊 牛 大象 熊 斑马 长颈鹿 背包 雨伞 手提包 领带 行李箱 飞盘 滑雪板 单板滑雪板 运动球 风筝 棒球棒 棒球手套 滑板 冲浪板 网球拍 瓶子 葡萄酒杯 杯子 叉子 刀 勺子 碗 香蕉 苹果 三明治 橙子 西兰花 胡萝卜 热狗 披萨 甜甜圈 蛋糕 椅子 沙发 盆栽 床 餐桌 马桶 电视 笔记本电脑 鼠标 遥控器 键盘 手机 微波炉 烤箱 烤面包机 水槽 冰箱 书 时钟 花瓶 剪刀 泰迪熊 吹风机 牙刷'.split()

class Detector:
    def __init__(self):
        folder=ROOT/'artifacts/objects'
        self.source=json.loads((folder/'detector-source.json').read_text())
        checkpoint=folder/'yolo11s.pt'
        self.sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.sha!=self.source['sha256']:raise RuntimeError('检测模型校验失败，请重新下载官方权重')
        self.model=YOLO(str(checkpoint),task='detect')
        self.names=self.model.names
        assert len(self.names)==len(NAMES_ZH)==80
        self.lock=threading.Lock()

    def predict(self,image,crop=None):
        from .engine import selected_region
        start=time.perf_counter();region=selected_region(image,crop)
        ox=round(crop[0]*image.width) if crop else 0
        oy=round(crop[1]*image.height) if crop else 0
        with self.lock:
            result=self.model.predict(source=region,imgsz=640,conf=.35,iou=.7,max_det=100,device='cpu',verbose=False,save=False)[0]
        items=[]
        for b in result.boxes:
            cls=int(b.cls.item());xyxy=b.xyxy[0].tolist()
            box=[(xyxy[0]+ox)/image.width,(xyxy[1]+oy)/image.height,(xyxy[2]+ox)/image.width,(xyxy[3]+oy)/image.height]
            items.append(dict(label=self.names[cls],display=NAMES_ZH[cls],class_id=cls,score=float(b.conf.item()),box=[max(0.,min(1.,v)) for v in box]))
        return dict(items=items,model='Ultralytics YOLO11s / COCO',sha256=self.sha,threshold=.35,iou=.7,input_size=640,
                    elapsed_ms=round((time.perf_counter()-start)*1000,1),
                    scope='80 类常见物体；框的位置和类别均可能出错。未检测到不代表图片没有物体。')
