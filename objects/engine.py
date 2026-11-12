"""Official pretrained natural-image model; no runtime model downloads."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'artifacts/objects'
MAX_BYTES = 10 * 1024 * 1024

# English labels remain available so translations never change the model taxonomy.
ZH = {
 'rocket':'火箭','helmet':'头盔','flag':'旗帜','woman':'女性','man':'男性','person':'人','human face':'人脸','coffee cup':'咖啡杯','spoon':'勺子','paper':'纸','white paper':'白纸','notebook':'笔记本',
 'tabby':'虎斑猫', 'tiger cat':'虎斑猫', 'Persian cat':'波斯猫', 'Siamese cat':'暹罗猫', 'Egyptian cat':'埃及猫',
 'golden retriever':'金毛寻回犬', 'Labrador retriever':'拉布拉多犬', 'German shepherd':'德国牧羊犬',
 'Siberian husky':'西伯利亚哈士奇', 'beagle':'比格犬', 'pug':'巴哥犬', 'Chihuahua':'吉娃娃',
 'coffee mug':'马克杯', 'cup':'杯子', 'espresso':'浓缩咖啡', 'coffeepot':'咖啡壶', 'teapot':'茶壶',
 'water bottle':'水瓶', 'beer bottle':'啤酒瓶', 'wine bottle':'葡萄酒瓶', 'pop bottle':'汽水瓶',
 'cellular telephone':'手机', 'laptop':'笔记本电脑', 'notebook':'笔记本电脑', 'desktop computer':'台式电脑',
 'computer keyboard':'电脑键盘', 'mouse':'电脑鼠标', 'monitor':'显示器', 'screen':'屏幕',
 'remote control':'遥控器', 'television':'电视机', 'camera':'照相机', 'reflex camera':'单反相机',
 'analog clock':'指针时钟', 'digital clock':'数字时钟', 'wall clock':'挂钟', 'digital watch':'电子表',
 'backpack':'双肩包', 'purse':'手提包', 'wallet':'钱包', 'umbrella':'雨伞', 'sunglasses':'太阳镜',
 'running shoe':'跑鞋', 'sandal':'凉鞋', 'sock':'袜子', 'jean':'牛仔裤', 'jersey':'运动衫',
 'banana':'香蕉', 'orange':'橙子', 'lemon':'柠檬', 'pineapple':'菠萝', 'strawberry':'草莓',
 'Granny Smith':'青苹果', 'pomegranate':'石榴', 'fig':'无花果', 'cucumber':'黄瓜', 'broccoli':'西兰花',
 'cauliflower':'花椰菜', 'bell pepper':'甜椒', 'carrot':'胡萝卜', 'mushroom':'蘑菇',
 'pizza':'披萨', 'cheeseburger':'芝士汉堡', 'hotdog':'热狗', 'ice cream':'冰淇淋', 'bagel':'贝果',
 'French loaf':'法棍面包', 'pretzel':'椒盐卷饼', 'plate':'盘子', 'soup bowl':'汤碗',
 'sports car':'跑车', 'minivan':'小型面包车', 'pickup':'皮卡', 'cab':'出租车', 'jeep':'吉普车',
 'school bus':'校车', 'minibus':'小巴', 'trolleybus':'无轨电车', 'fire engine':'消防车', 'ambulance':'救护车',
 'motor scooter':'踏板摩托车', 'mountain bike':'山地自行车', 'bicycle-built-for-two':'双人自行车',
 'airliner':'客机', 'warplane':'军用飞机', 'space shuttle':'航天飞机', 'missile':'导弹',
 'lifeboat':'救生艇', 'speedboat':'快艇', 'canoe':'独木舟', 'container ship':'集装箱船',
 'elephant':'大象', 'African elephant':'非洲象', 'Indian elephant':'亚洲象', 'lion':'狮子', 'tiger':'老虎',
 'zebra':'斑马', 'giant panda':'大熊猫', 'brown bear':'棕熊', 'ice bear':'北极熊', 'koala':'考拉',
 'goldfish':'金鱼', 'great white shark':'大白鲨', 'dolphin':'海豚', 'king penguin':'王企鹅',
 'flamingo':'火烈鸟', 'peacock':'孔雀', 'macaw':'金刚鹦鹉', 'hen':'母鸡', 'cock':'公鸡',
 'dragonfly':'蜻蜓', 'bee':'蜜蜂', 'ant':'蚂蚁', 'ladybug':'瓢虫', 'monarch':'帝王蝶',
 'dining table':'餐桌', 'desk':'书桌', 'studio couch':'沙发床', 'rocking chair':'摇椅',
 'refrigerator':'冰箱', 'microwave':'微波炉', 'washing machine':'洗衣机', 'electric fan':'电风扇',
 'vacuum':'吸尘器', 'toaster':'烤面包机', 'iron':'熨斗', 'stove':'炉灶', 'dishwasher':'洗碗机',
 'bookcase':'书柜', 'library':'图书馆', 'bookshop':'书店', 'comic book':'漫画书',
 'ballpoint':'圆珠笔', 'fountain pen':'钢笔', 'pencil box':'文具盒', 'pencil sharpener':'卷笔刀',
 'ruler':'尺子', 'eraser':'橡皮', 'rubber eraser':'橡皮', 'scissors':'剪刀', 'envelope':'信封',
 'tennis ball':'网球', 'soccer ball':'足球', 'basketball':'篮球', 'volleyball':'排球', 'baseball':'棒球',
 'guitar':'吉他', 'acoustic guitar':'原声吉他', 'electric guitar':'电吉他', 'grand piano':'三角钢琴',
 'violin':'小提琴', 'drum':'鼓', 'traffic light':'交通信号灯', 'street sign':'路牌',
 'mountain tent':'帐篷', 'seashore':'海岸', 'volcano':'火山', 'alp':'高山', 'valley':'山谷',
 'sandbar':'沙洲', 'lakeside':'湖岸', 'pier':'码头', 'fountain':'喷泉', 'park bench':'公园长椅',
}


def display_label(label):
    return ZH.get(label, label)


def decode(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('图片大小须在 1 字节到 10 MB 之间')
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {'JPEG','PNG','WEBP'}:
                raise ValueError('请选择 JPEG、PNG 或 WebP 图片')
            if source.width * source.height > 20_000_000:
                raise ValueError('图片超过 2000 万像素，请缩小后上传')
            if getattr(source,'n_frames',1) > 1:
                raise ValueError('请上传单帧静态图片')
            corrected=ImageOps.exif_transpose(source).convert('RGBA')
            image=Image.new('RGBA',corrected.size,'white')
            image.alpha_composite(corrected)
            image=image.convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('无法读取图片，请检查文件是否损坏') from exc
    if min(image.size) < 32:
        raise ValueError('图片太小，请使用宽高至少 32 像素的照片')
    small=np.asarray(image.resize((32,32)),dtype=np.float32)
    if small.std(axis=(0,1)).max() < 2:
        raise ValueError('图片接近纯色，缺少可识别的细节')
    return image


def selected_region(image, crop):
    if crop is None:
        return image
    x,y,w,h=crop
    if min(w,h)<=0 or min(x,y)<0 or x+w>1.000001 or y+h>1.000001:
        raise ValueError('框选区域必须位于图片内')
    width,height=image.size
    box=(round(x*width),round(y*height),round((x+w)*width),round((y+h)*height))
    if box[2]-box[0]<32 or box[3]-box[1]<32:
        raise ValueError('框选区域过小，请保留至少 32×32 像素')
    return image.crop(box)


class ObjectEngine:
    def __init__(self, general=True):
        self.source=json.loads((MODEL_DIR/'model-source.json').read_text())
        path=MODEL_DIR/'resnet50-11ad3fa6.pth'
        self.sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha!=self.source['sha256']:
            raise RuntimeError('模型文件校验失败，请重新下载官方权重')
        torch.set_num_threads(2)
        self.model=resnet50(weights=None)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.model.eval()
        self.weights=ResNet50_Weights.IMAGENET1K_V2
        self.transform=self.weights.transforms()
        self.categories=self.weights.meta['categories']
        self.lock=threading.Lock()
        from .detector import Detector
        self.detector=Detector()
        from .general import GeneralVision
        self.general=GeneralVision() if general else None

    def candidates(self, probs):
        values,indices=probs.topk(5)
        return [dict(class_id=int(i),label=self.categories[int(i)],display=display_label(self.categories[int(i)]),score=float(v)) for v,i in zip(values,indices)]

    def predict(self,image,crop=None):
        start=time.perf_counter()
        vision=self.general.predict(image,crop) if self.general else None
        detection=self.detector.predict(image,crop)
        image=selected_region(image,crop)
        # Standard center crop and a whole-image padded view are shown separately.
        # Agreement is only a diagnostic, not a calibrated correctness guarantee.
        center=self.transform(image)
        whole=ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104))
        whole=F.normalize(F.to_tensor(whole),[.485,.456,.406],[.229,.224,.225])
        with self.lock,torch.inference_mode():
            probs=self.model(torch.stack([center,whole])).softmax(dim=1)
        first=self.candidates(probs[0]);second=self.candidates(probs[1])
        agreement=first[0]['class_id']==second[0]['class_id']
        uncertain=not agreement or first[0]['score']<.35
        return dict(vision=vision,detection=detection,candidates=first,whole_candidates=second,views_agree=agreement,needs_review=uncertain,
                    review_reason='两种取景的首选不同' if not agreement else '首选得分低于提示阈值' if uncertain else '两种取景首选一致，仍需核对原图',
                    elapsed_ms=round((time.perf_counter()-start)*1000,1),model_sha=self.sha,
                    model='ResNet50 / ImageNet-1K V2',crop=crop,region_size=list(image.size),
                    scope=vision['scope'] if vision else '1000 个预定义类别；得分不是正确率，类别外图片也可能得到高分')
