# CELL 1: Kaggle dependencies and imports
!pip install -q optuna albumentations pycocotools scikit-image

import os, glob, gc, json, random, time
from collections import defaultdict
import cv2, numpy as np, optuna, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from skimage.segmentation import watershed
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# CELL 2: Configuration
class Config:
    BASE_DIR = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train/train_images')
    TEST_DIR = os.path.join(BASE_DIR, 'test/test_images')
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    OUT_DIR = '/kaggle/working/ver3_7'
    ORIGINAL_SIZE, IMG_SIZE, SEED, NUM_WORKERS = 2048, 512, 42, 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    BATCH_SIZE = 4 if GPU_COUNT >= 2 else 2
    ACCUMULATION_STEPS = 1 if GPU_COUNT >= 2 else 2
    N_TRIALS, TRIAL_EPOCHS, TRIAL_TIMEOUT_SECONDS = 3, 8, 75 * 60
    EPOCHS, EARLY_STOP_WARMUP, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA = 40, 10, 8, .001
    LR, WEIGHT_DECAY = 2e-4, 1e-4
    SKELETON_WEIGHT, BOUNDARY_WEIGHT, SPINE_WEIGHT, AUX_WARMUP_EPOCHS = .2, .2, .2, 3
    THRESHOLD, BOUNDARY_THRESHOLD, SPINE_THRESHOLD, MIN_AREA, CLOSING_RADIUS = .4, .5, .5, 200, 0
    USE_WATERSHED, PQ_IOU_THRESHOLD, RUN_PREFLIGHT = False, .5, True
    SPLIT_FILE = os.path.join(OUT_DIR, 'grouped_split_seed_42.json')
    SETTINGS_FILE = os.path.join(OUT_DIR, 'postprocessing_settings.json')

os.makedirs(Config.OUT_DIR, exist_ok=True)
print(f'PyTorch={torch.__version__}; device={Config.DEVICE}; GPUs={Config.GPU_COUNT}')

# CELL 3: Shared image preprocessing
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    return image if image is not None else cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def remove_limb_darkening(image):
    h, w = image.shape; yy, xx = np.ogrid[:h, :w]; cy, cx = h // 2, w // 2
    disk = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < min(cx, cy) * .98
    lo, hi = np.percentile(image[disk], (1, 99))
    image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
    image[~disk] = 0
    return image.astype(np.float32), disk

def get_magnetic_proxy(image):
    u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    dx, dy = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jxy, jyy = (cv2.GaussianBlur(v, (15, 15), 5) for v in (dx ** 2, dx * dy, dy ** 2))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return ((coherence - coherence.min()) / (coherence.max() - coherence.min() + 1e-8)).astype(np.float32)

def image_features(path, size=Config.IMG_SIZE):
    image = load_image(path)
    if image is None: raise FileNotFoundError(path)
    image = image.astype(np.float32); image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    limb, _ = remove_limb_darkening(image); proxy = get_magnetic_proxy(limb)
    return cv2.resize(np.stack([limb, limb, proxy], -1), (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

# CELL 4: Grouped COCO views and targets
def physical_cases(coco):
    grouped = defaultdict(list)
    for image_id, info in coco.imgs.items(): grouped[info['file_name']].append(image_id)
    return [dict(file_name=name, image_ids=sorted(ids)) for name, ids in sorted(grouped.items())]

def grouped_split(coco, cases):
    if os.path.exists(Config.SPLIT_FILE):
        with open(Config.SPLIT_FILE) as f: split = json.load(f)
        if set(split['train'] + split['dev'] + split['audit']) == {x['file_name'] for x in cases}: return split
    rng, bins = random.Random(Config.SEED), defaultdict(list)
    values = [(sum(len(coco.imgToAnns.get(i, [])) for i in c['image_ids']) / len(c['image_ids']),
               sum(a.get('area', 0) for i in c['image_ids'] for a in coco.imgToAnns.get(i, [])) / len(c['image_ids'])) for c in cases]
    counts, areas = np.array([x[0] for x in values]), np.array([x[1] for x in values])
    ce, ae = np.unique(np.quantile(counts, [.25, .5, .75])), np.unique(np.quantile(areas, [.25, .5, .75]))
    for case, (count, area) in zip(cases, values): bins[(np.digitize(count, ce), np.digitize(area, ae))].append(case['file_name'])
    split = dict(train=[], dev=[], audit=[])
    for names in bins.values():
        rng.shuffle(names); a, b = round(.1 * len(names)), round(.2 * len(names))
        split['dev'] += names[:a]; split['audit'] += names[a:b]; split['train'] += names[b:]
    for key in ('dev', 'audit'):
        target = round(.1 * len(cases))
        while len(split[key]) < target: split[key].append(split['train'].pop())
        while len(split[key]) > target: split['train'].append(split[key].pop())
    assert not (set(split['train']) & set(split['dev']) or set(split['train']) & set(split['audit']) or set(split['dev']) & set(split['audit']))
    with open(Config.SPLIT_FILE, 'w') as f: json.dump({**split, 'seed': Config.SEED}, f, indent=2)
    return split

def skeletonize(binary):
    binary, skeleton = binary.astype(np.uint8), np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while binary.any():
        eroded = cv2.erode(binary, kernel); skeleton |= binary & ~cv2.dilate(eroded, kernel); binary = eroded
    return skeleton

def rasterize_spine(points, shape):
    target = np.zeros(shape, np.uint8)
    if not isinstance(points, (list, tuple)) or len(points) < 4: return target, False
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if not np.isfinite(points).all(): return target, False
    points[:, 0] *= shape[1] / Config.ORIGINAL_SIZE; points[:, 1] *= shape[0] / Config.ORIGINAL_SIZE
    points = np.round(points).astype(np.int32); points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1); points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
    cv2.polylines(target, [points.reshape(-1, 1, 2)], False, 1, 1)
    return target, True

def build_targets(coco, image_id, size):
    instances, spine, valid = [], np.zeros((size, size), np.uint8), False
    for ann in coco.imgToAnns.get(image_id, []):
        mask = cv2.resize(coco.annToMask(ann).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        if mask.any(): instances.append(mask)
        line, exists = rasterize_spine(ann.get('spine'), (size, size)); spine = np.maximum(spine, line); valid |= exists
    fg = np.maximum.reduce(instances) if instances else np.zeros((size, size), np.uint8)
    boundary, kernel = np.zeros_like(fg), np.ones((3, 3), np.uint8)
    for mask in instances: boundary |= cv2.dilate(mask, kernel) ^ cv2.erode(mask, kernel)
    return dict(fg=fg, boundary=boundary, spine=spine, skeleton=skeletonize(fg), spine_valid=np.uint8(valid))

class SolarDataset(Dataset):
    def __init__(self, coco, records, transforms): self.coco, self.records, self.transforms = coco, records, transforms
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]; image = image_features(os.path.join(Config.TRAIN_DIR, record['file_name']))
        target = build_targets(self.coco, record['image_id'], Config.IMG_SIZE)
        augmented = self.transforms(image=image, **{k: target[k] for k in ('fg', 'boundary', 'spine', 'skeleton')})
        output = {k: torch.as_tensor(augmented[k]).unsqueeze(0).float() for k in ('fg', 'boundary', 'spine', 'skeleton')}
        output['spine_valid'] = torch.tensor(float(target['spine_valid']))
        return augmented['image'].float(), output, record['image_id'], record['file_name']

extras = {'fg': 'mask', 'boundary': 'mask', 'spine': 'mask', 'skeleton': 'mask'}
train_transforms = A.Compose([A.HorizontalFlip(.5), A.VerticalFlip(.5), A.RandomRotate90(.5), A.RandomBrightnessContrast(.12, .12, p=.35), A.RandomGamma(p=.25), A.GaussNoise(std_range=(.01, .03), p=.15), ToTensorV2()], additional_targets=extras)
val_transforms = A.Compose([ToTensorV2()], additional_targets=extras)

# CELL 5: ~10.85M three-head Mag-Agent U2-Net
def norm(c): return nn.GroupNorm(8 if c >= 8 else 1, c)
class AttentionGate(nn.Module):
    def __init__(self, g, x, h):
        super().__init__(); self.g = nn.Sequential(nn.Conv2d(g, h, 1), norm(h)); self.x = nn.Sequential(nn.Conv2d(x, h, 1), norm(h)); self.psi = nn.Sequential(nn.ReLU(True), nn.Conv2d(h, 1, 1), nn.Sigmoid())
    def forward(self, g, x): return x * self.psi(F.interpolate(self.g(g), size=x.shape[-2:], mode='bilinear', align_corners=False) + self.x(x))
class RSU(nn.Module):
    def __init__(self, inp, mid, out):
        super().__init__(); self.inp = nn.Conv2d(inp, out, 1); self.e1 = nn.Sequential(nn.Conv2d(out, mid, 3, padding=1), norm(mid), nn.ReLU(True)); self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(mid, mid, 3, padding=1), norm(mid), nn.ReLU(True)); self.d2 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), norm(mid), nn.ReLU(True)); self.d1 = nn.Sequential(nn.Conv2d(mid * 2, out, 3, padding=1), norm(out), nn.ReLU(True))
    def forward(self, x):
        residual = self.inp(x); e1 = self.e1(residual); d2 = self.d2(self.e2(e1)); return self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], 1)) + residual
class CognitiveRefiner(nn.Module):
    def __init__(self, c): super().__init__(); self.a = nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c); self.b = nn.Conv2d(c, c, 3, padding=4, dilation=4, groups=c); self.gate = nn.Sequential(nn.Conv2d(c + 1, c, 1), nn.Sigmoid())
    def forward(self, x, magnetic):
        x = x + F.relu(self.a(x)) + F.relu(self.b(x)); return x * self.gate(torch.cat([x, magnetic], 1))
class MagAgentU2Net(nn.Module):
    def __init__(self):
        super().__init__(); self.e1, self.p1 = RSU(3, 32, 64), nn.MaxPool2d(2); self.e2, self.p2 = RSU(64, 64, 128), nn.MaxPool2d(2); self.e3, self.p3 = RSU(128, 128, 256), nn.MaxPool2d(2); self.b = RSU(256, 256, 512); self.a3, self.d3 = AttentionGate(512, 256, 256), RSU(768, 256, 256); self.a2, self.d2 = AttentionGate(256, 128, 128), RSU(384, 128, 128); self.a1, self.d1 = AttentionGate(128, 64, 64), RSU(192, 64, 64); self.refine = CognitiveRefiner(64); self.fg, self.boundary, self.spine = nn.Conv2d(64, 1, 1), nn.Conv2d(64, 1, 1), nn.Conv2d(64, 1, 1)
    def forward(self, x):
        mag = x[:, 2:3]; e1 = self.e1(x); e2 = self.e2(self.p1(e1)); e3 = self.e3(self.p2(e2)); b = self.b(self.p3(e3)); d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), self.a3(b, e3)], 1)); d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), self.a2(d3, e2)], 1)); d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), self.a1(d2, e1)], 1)); f = self.refine(d1, mag); return {'fg': self.fg(f), 'boundary': self.boundary(f), 'spine': self.spine(f)}

# CELL 6: Multi-task topology loss
class MultiTaskLoss(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def dice(self, logits, target):
        p, y = torch.sigmoid(logits.float()), target.float(); return 1 - ((2 * (p * y).sum((1,2,3)) + 1) / (p.sum((1,2,3)) + y.sum((1,2,3)) + 1)).mean()
    def forward(self, pred, target, aux=1.):
        fg = self.bce(pred['fg'], target['fg']) + self.dice(pred['fg'], target['fg'])
        skeleton = 1 - ((torch.sigmoid(pred['fg'].float()) * target['skeleton']).sum() + 1) / (target['skeleton'].sum() + 1)
        boundary = self.bce(pred['boundary'], target['boundary']) + self.dice(pred['boundary'], target['boundary'])
        valid = target['spine_valid'].view(-1, 1, 1, 1); spine = (F.binary_cross_entropy_with_logits(pred['spine'], target['spine'], reduction='none') * valid).sum() / (valid.sum() * pred['spine'].shape[-1] * pred['spine'].shape[-2] + 1e-8)
        return fg + aux * (Config.SKELETON_WEIGHT * skeleton + Config.BOUNDARY_WEIGHT * boundary + Config.SPINE_WEIGHT * spine)

# CELL 7: Official-style PQ and conservative instance decoding
def raw_model(model): return model.module if isinstance(model, nn.DataParallel) else model
def annotations_to_masks(coco, image_id): return [coco.annToMask(a).astype(np.uint8) for a in coco.imgToAnns.get(image_id, [])]
def official_stats(gt, pred):
    if not gt: return dict(iou_sum=0., tp=0, fp=len(pred), fn=0, splits=0, merges=0)
    if not pred: return dict(iou_sum=0., tp=0, fp=0, fn=len(gt), splits=0, merges=0)
    ious = np.array([[np.logical_and(a,b).sum() / max(1, np.logical_or(a,b).sum()) for b in pred] for a in gt], np.float32); hits, overlaps = ious > Config.PQ_IOU_THRESHOLD, ious > 0
    return dict(iou_sum=float(ious[hits].sum()), tp=int(hits.sum()), fp=int((hits.sum(0) == 0).sum()), fn=int((hits.sum(1) == 0).sum()), splits=int((overlaps.sum(1) > 1).sum()), merges=int((overlaps.sum(0) > 1).sum()))
def finalize(t):
    d = t['tp'] + .5 * t['fp'] + .5 * t['fn']; return {**t, 'pq':t['iou_sum']/d if d else 0., 'sq':t['iou_sum']/t['tp'] if t['tp'] else 0., 'rq':t['tp']/d if d else 0.}
def decode_instances(fg, boundary=None, spine=None, threshold=None, min_area=None, use_watershed=None):
    threshold, min_area = Config.THRESHOLD if threshold is None else threshold, Config.MIN_AREA if min_area is None else min_area; use_watershed = Config.USE_WATERSHED if use_watershed is None else use_watershed
    binary = cv2.resize((fg > threshold).astype(np.uint8), (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), interpolation=cv2.INTER_NEAREST); labels = None
    if use_watershed and boundary is not None and spine is not None:
        boundary = cv2.resize(boundary, (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), interpolation=cv2.INTER_LINEAR); markers = cv2.resize((spine > Config.SPINE_THRESHOLD).astype(np.uint8), (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), interpolation=cv2.INTER_NEAREST) & binary; n, markers = cv2.connectedComponents(markers)
        if n > 1: labels = watershed(boundary, markers, mask=binary.astype(bool)).astype(np.int32)
    if labels is None: _, labels = cv2.connectedComponents(binary)
    return [(labels == i).astype(np.uint8) for i in range(1, labels.max()+1) if (labels == i).sum() >= min_area]

def validate(model, loader, criterion):
    model.eval(); totals = dict(iou_sum=0.,tp=0,fp=0,fn=0,splits=0,merges=0); losses=[]; dices=[]
    with torch.no_grad():
        for images, target, ids, _ in tqdm(loader, desc='Validating'):
            target = {k:v.to(Config.DEVICE) for k,v in target.items()}
            with autocast(enabled=Config.DEVICE=='cuda'): out=model(images.to(Config.DEVICE)); loss=criterion(out,target)
            losses.append(loss.item()); probs={k:torch.sigmoid(v.float()).cpu().numpy() for k,v in out.items()}; binary=probs['fg'] > Config.THRESHOLD; dices += ((2*(torch.from_numpy(binary).to(Config.DEVICE)*target['fg']).sum((1,2,3))+1e-6)/(torch.from_numpy(binary).to(Config.DEVICE).sum((1,2,3))+target['fg'].sum((1,2,3))+1e-6)).cpu().tolist()
            for i, image_id in enumerate(ids):
                stats=official_stats(annotations_to_masks(loader.dataset.coco,image_id),decode_instances(probs['fg'][i,0],probs['boundary'][i,0],probs['spine'][i,0])); [totals.__setitem__(k,totals[k]+stats[k]) for k in totals]
    metrics=finalize(totals); metrics['foreground_dice']=float(np.mean(dices)); return float(np.mean(losses)),metrics

# CELL 8: Long training, early stopping, and pruning
def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); values=[]; aux=min(1.,(epoch+1)/Config.AUX_WARMUP_EPOCHS)
    for step,(images,target,_,_) in enumerate(tqdm(loader,desc='Training')):
        target={k:v.to(Config.DEVICE,non_blocking=True) for k,v in target.items()}
        with autocast(enabled=Config.DEVICE=='cuda'): loss=criterion(model(images.to(Config.DEVICE,non_blocking=True)),target,aux)/Config.ACCUMULATION_STEPS
        scaler.scale(loss).backward(); values.append(loss.item()*Config.ACCUMULATION_STEPS)
        if (step+1)%Config.ACCUMULATION_STEPS==0 or step+1==len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(values))
def payload(model,opt,scheduler,epoch,best,params): return dict(model=raw_model(model).state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best_pq=best,parameters=params)
def fit_model(train,dev,lr,weight_decay,epochs,checkpoint,trial=None,resume=None):
    random.seed(Config.SEED);np.random.seed(Config.SEED);torch.manual_seed(Config.SEED); model=MagAgentU2Net().to(Config.DEVICE); model=nn.DataParallel(model) if Config.GPU_COUNT>=2 else model; opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3); scaler=GradScaler(enabled=Config.DEVICE=='cuda'); criterion=MultiTaskLoss(); start,best,stale,history=0,-1.,0,[]
    if resume and os.path.exists(resume):
        saved=torch.load(resume,map_location=Config.DEVICE,weights_only=False);raw_model(model).load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start,best=saved['epoch']+1,saved['best_pq']
    print(f'Parameters={sum(p.numel() for p in raw_model(model).parameters()):,}; batch={Config.BATCH_SIZE}; accumulation={Config.ACCUMULATION_STEPS}; lr={lr:.2g}')
    try:
        for epoch in range(start,epochs):
            if Config.DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter(); train_loss=train_epoch(model,train,opt,criterion,scaler,epoch); dev_loss,metrics=validate(model,dev,criterion);scheduler.step(metrics['pq']);row=dict(epoch=epoch+1,train_loss=train_loss,dev_loss=dev_loss,seconds=time.perf_counter()-began,lr=opt.param_groups[0]['lr'],peak_gib=torch.cuda.max_memory_allocated()/2**30 if Config.DEVICE=='cuda' else 0.,**metrics);history.append(row);print(pd.Series(row).round(4).to_dict())
            if metrics['pq']>best+Config.EARLY_STOP_MIN_DELTA: best,stale=metrics['pq'],0;torch.save(payload(model,opt,scheduler,epoch,best,dict(lr=lr,weight_decay=weight_decay)),checkpoint)
            else: stale+=1
            pd.DataFrame(history).to_csv(checkpoint+'.history.csv',index=False)
            if trial is not None:
                trial.report(metrics['pq'],epoch)
                if trial.should_prune():raise optuna.TrialPruned()
            if epoch+1>=Config.EARLY_STOP_WARMUP and stale>=Config.EARLY_STOP_PATIENCE: print(f'Early stopped; best PQ={best:.4f}');break
        return best
    finally:
        del model,opt,scheduler,scaler;gc.collect()
        if Config.DEVICE=='cuda':torch.cuda.empty_cache()

# CELL 9: Cached calibration and Kaggle RLE submission
def cache_predictions(model,loader):
    model.eval();cached={}
    with torch.no_grad():
        for images,_,ids,_ in tqdm(loader,desc='Caching development maps'):
            with autocast(enabled=Config.DEVICE=='cuda'): out=model(images.to(Config.DEVICE))
            probs={k:torch.sigmoid(v.float()).cpu().numpy().astype(np.float16) for k,v in out.items()}
            for i,image_id in enumerate(ids):cached[image_id]={k:probs[k][i,0] for k in probs}
    return cached
def tune_postprocessing(model,dev):
    cached,best=cache_predictions(model,dev),None
    for threshold in (.3,.4,.5):
        for area in (100,200,400):
            for use_ws in (False,True):
                totals=dict(iou_sum=0.,tp=0,fp=0,fn=0,splits=0,merges=0)
                for image_id,maps in cached.items():
                    stats=official_stats(annotations_to_masks(dev.dataset.coco,image_id),decode_instances(maps['fg'],maps['boundary'],maps['spine'],threshold,area,use_ws));[totals.__setitem__(k,totals[k]+stats[k]) for k in totals]
                candidate=(finalize(totals)['pq'],threshold,area,use_ws,finalize(totals))
                if best is None or candidate[0]>best[0]:best=candidate
    _,Config.THRESHOLD,Config.MIN_AREA,Config.USE_WATERSHED,metrics=best;settings=dict(threshold=Config.THRESHOLD,min_area=Config.MIN_AREA,use_watershed=Config.USE_WATERSHED,development_pq=metrics['pq'])
    with open(Config.SETTINGS_FILE,'w') as f:json.dump(settings,f,indent=2)
    print('Calibration:',settings);return settings
def encode(mask):return maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('utf-8')
def generate_submission(model,paths):
    model.eval();rows=[]
    with torch.no_grad():
        for path in tqdm(paths,desc='Generating submission'):
            image=torch.from_numpy(image_features(path)).permute(2,0,1).unsqueeze(0).to(Config.DEVICE)
            with autocast(enabled=Config.DEVICE=='cuda'):out=model(image)
            maps={k:torch.sigmoid(v.float()).cpu().numpy()[0,0] for k,v in out.items()};stem=os.path.splitext(os.path.basename(path))[0]
            for n,mask in enumerate(decode_instances(maps['fg'],maps['boundary'],maps['spine']),1):rows.append(dict(filament_id=f'{stem}_{n}',segmentation_rle=encode(mask)))
    path=os.path.join(Config.OUT_DIR,'submission.csv');pd.DataFrame(rows,columns=['filament_id','segmentation_rle']).to_csv(path,index=False);print(f'Saved {len(rows)} instances: {path}')

# CELL 10: Run full V3-V7 pipeline
def records_for(cases,names):return [dict(file_name=c['file_name'],image_id=i) for c in cases if c['file_name'] in set(names) for i in c['image_ids']]
def preflight():
    model=MagAgentU2Net().to(Config.DEVICE);out=model(torch.randn(1,3,64,64,device=Config.DEVICE));assert set(out)=={'fg','boundary','spine'} and all(v.shape==(1,1,64,64) for v in out.values());assert official_stats([np.ones((4,4),np.uint8)],[np.ones((4,4),np.uint8)])['pq']==1.;assert official_stats([],[])['pq']==0.;del model;print('Preflight passed.')
def main():
    random.seed(Config.SEED);np.random.seed(Config.SEED);torch.manual_seed(Config.SEED)
    if Config.RUN_PREFLIGHT:preflight()
    coco=COCO(Config.ANNOTATIONS_FILE);cases=[c for c in physical_cases(coco) if os.path.exists(os.path.join(Config.TRAIN_DIR,c['file_name']))];split=grouped_split(coco,cases);train_records,dev_records,audit_records=(records_for(cases,split[k]) for k in ('train','dev','audit'));print(f'Physical images train/dev/audit={len(split["train"])}/{len(split["dev"])}/{len(split["audit"])}; views={len(train_records)}/{len(dev_records)}/{len(audit_records)}')
    train=DataLoader(SolarDataset(coco,train_records,train_transforms),batch_size=Config.BATCH_SIZE,shuffle=True,num_workers=Config.NUM_WORKERS,pin_memory=True);dev=DataLoader(SolarDataset(coco,dev_records,val_transforms),batch_size=Config.BATCH_SIZE,shuffle=False,num_workers=Config.NUM_WORKERS,pin_memory=True);audit=DataLoader(SolarDataset(coco,audit_records,val_transforms),batch_size=Config.BATCH_SIZE,shuffle=False,num_workers=Config.NUM_WORKERS,pin_memory=True)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=Config.SEED),pruner=optuna.pruners.MedianPruner(n_startup_trials=2,n_warmup_steps=4,interval_steps=1));study.enqueue_trial({'lr':Config.LR,'weight_decay':Config.WEIGHT_DECAY})
    def objective(trial):
        lr=trial.suggest_float('lr',1e-4,4e-4,log=True);wd=trial.suggest_float('weight_decay',1e-6,1e-3,log=True);return fit_model(train,dev,lr,wd,Config.TRIAL_EPOCHS,os.path.join(Config.OUT_DIR,f'trial_{trial.number}.pth'),trial)
    study.optimize(objective,n_trials=Config.N_TRIALS,timeout=Config.TRIAL_TIMEOUT_SECONDS,n_jobs=1,gc_after_trial=True);study.trials_dataframe().to_csv(os.path.join(Config.OUT_DIR,'optuna_trials.csv'),index=False);winner=study.best_trial;best_path=os.path.join(Config.OUT_DIR,'best_ver3_7.pth');fit_model(train,dev,**winner.params,epochs=Config.EPOCHS,checkpoint=best_path,resume=os.path.join(Config.OUT_DIR,f'trial_{winner.number}.pth'))
    model=MagAgentU2Net().to(Config.DEVICE);model.load_state_dict(torch.load(best_path,map_location=Config.DEVICE,weights_only=False)['model']);tune_postprocessing(model,dev);loss,metrics=validate(model,audit,MultiTaskLoss());
    with open(os.path.join(Config.OUT_DIR,'audit_metrics.json'),'w') as f:json.dump(dict(loss=loss,**metrics),f,indent=2)
    print(f'Frozen audit: PQ={metrics["pq"]:.4f}; SQ={metrics["sq"]:.4f}; RQ={metrics["rq"]:.4f}');generate_submission(model,sorted(glob.glob(os.path.join(Config.TEST_DIR,'*.jpeg'))))
main()
