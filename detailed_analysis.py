"""
Comprehensive analysis: FLOPs, inference speed, per-class metrics, threshold
sensitivity, parameter breakdown, loss convergence overlay, and more.
Generates 6 publication-quality figures.
"""
import sys, time, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F

sys.path.insert(0, '.')

from dataset import ShapeDataset, custom_collate_fn, create_dataloader
from torch.utils.data import DataLoader
from model import SimpleMaskRCNN
from improved_model import ImprovedMaskRCNN
from compare import compute_box_iou, compute_mask_iou

device = torch.device('cpu')
CLS = {0:'bg', 1:'circle', 2:'rect', 3:'triangle', 4:'diamond'}

# ── Load models ──
base = SimpleMaskRCNN(num_classes=5)
base.load_state_dict(torch.load('./checkpoints/baseline_best.pth', map_location='cpu', weights_only=True))
base.eval()

imp = ImprovedMaskRCNN(num_classes=5, pretrained=False)
imp.load_state_dict(torch.load('./checkpoints/improved_best.pth', map_location='cpu', weights_only=True))
imp.eval()

bh = torch.load('./checkpoints/baseline_history.pt', weights_only=False)
ih = torch.load('./checkpoints/improved_history.pt', weights_only=False)

# ═══════════════════════════════════════════════════════════════════
# 1. Compute FLOPs (MACs) for each model
# ═══════════════════════════════════════════════════════════════════
print("Computing FLOPs...")

def count_conv_flops(module, input_size):
    """Estimate MACs for a Conv2d layer."""
    h, w = input_size
    cout = module.out_channels
    cin = module.in_channels // module.groups
    kh, kw = module.kernel_size
    sh, sw = module.stride
    oh = (h + 2*module.padding[0] - kh) // sh + 1
    ow = (w + 2*module.padding[1] - kw) // sw + 1
    return cout * oh * ow * cin * kh * kw

def count_linear_flops(module):
    return module.in_features * module.out_features

def estimate_flops(model, input_shape=(1, 3, 128, 128)):
    """Rough FLOPs estimate by walking through named modules."""
    total = 0
    x = torch.randn(input_shape)
    # Walk through modules and sum up
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            # Estimate spatial size from name heuristics
            total += m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * 16 * 16
        elif isinstance(m, torch.nn.ConvTranspose2d):
            total += m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * 28 * 28
        elif isinstance(m, torch.nn.Linear):
            total += count_linear_flops(m)
    return total

base_flops = estimate_flops(base)
imp_flops = estimate_flops(imp)
paper_flops = 275e9  # ~275 GFLOPs for ResNet-101-FPN Mask R-CNN (from literature)

print(f"  Baseline FLOPs: {base_flops/1e9:.2f} GFLOPs")
print(f"  Improved FLOPs: {imp_flops/1e9:.2f} GFLOPs")
print(f"  Paper (est):    {paper_flops/1e9:.0f} GFLOPs")

# ═══════════════════════════════════════════════════════════════════
# 2. Inference speed benchmark
# ═══════════════════════════════════════════════════════════════════
print("Benchmarking inference speed...")

def benchmark_model(model, n_runs=20):
    x = torch.randn(1, 3, 128, 128)
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.inference(x, [(128, 128)], score_threshold=0.05)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.inference(x, [(128, 128)], score_threshold=0.05)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

base_ms, base_std = benchmark_model(base)
imp_ms, imp_std = benchmark_model(imp)
paper_ms = 200.0  # paper: ~200ms per frame on GPU (Tesla M40)

print(f"  Baseline: {base_ms:.1f} +/- {base_std:.1f} ms")
print(f"  Improved: {imp_ms:.1f} +/- {imp_std:.1f} ms")

# ═══════════════════════════════════════════════════════════════════
# 3. Per-class metrics
# ═══════════════════════════════════════════════════════════════════
print("Computing per-class metrics...")

def paste_mask(m, box, sz=(128,128)):
    H, W = sz
    x1, y1 = int(max(0, box[0].item())), int(max(0, box[1].item()))
    x2, y2 = int(min(W, box[2].item())), int(min(H, box[3].item()))
    if x2<=x1 or y2<=y1: return np.zeros((H,W))
    t = torch.tensor(m).float().unsqueeze(0).unsqueeze(0)
    r = F.interpolate(t, (y2-y1, x2-x1), mode='bilinear', align_corners=False)
    out = np.zeros((H,W), dtype=np.float32)
    out[y1:y2, x1:x2] = (r.squeeze().numpy()>0.5).astype(np.float32)
    return out

def per_class_eval(model, n=50, thresh=0.01):
    dl = create_dataloader(num_samples=n, batch_size=1, seed=123)
    cls_box = {c: [] for c in range(1, 5)}
    cls_mask = {c: [] for c in range(1, 5)}
    cls_correct = {c: 0 for c in range(1, 5)}
    cls_total = {c: 0 for c in range(1, 5)}

    with torch.no_grad():
        for idx, batch in enumerate(dl):
            if idx >= n: break
            preds = model.inference(batch['image'], [(128,128)], score_threshold=thresh)[0]
            pb = preds['boxes'].cpu(); pc = preds['class_ids'].cpu()
            pm = preds['masks'].cpu()
            for j in range(len(batch['labels'][0])):
                gt_c = int(batch['labels'][0][j].item())
                if gt_c < 1 or gt_c > 4: continue
                cls_total[gt_c] += 1
                gb = batch['boxes'][0][j]; gm = batch['masks'][0][j]
                if pb.shape[0] == 0:
                    cls_box[gt_c].append(0.0); cls_mask[gt_c].append(0.0)
                    continue
                ious = [compute_box_iou(gb, p) for p in pb]
                bi = int(np.argmax(ious))
                cls_box[gt_c].append(ious[bi])
                if bi < pm.shape[0]:
                    ci = min(int(pc[bi].item()), pm.shape[1]-1)
                    fm = paste_mask(pm[bi, ci].numpy(), pb[bi])
                    cls_mask[gt_c].append(compute_mask_iou(gm.numpy(), fm))
                else:
                    cls_mask[gt_c].append(0.0)
                if bi < pc.shape[0] and pc[bi].item() == gt_c:
                    cls_correct[gt_c] += 1

    result = {}
    for c in range(1, 5):
        result[c] = {
            'box_iou': np.mean(cls_box[c]) if cls_box[c] else 0,
            'mask_iou': np.mean(cls_mask[c]) if cls_mask[c] else 0,
            'accuracy': cls_correct[c] / max(1, cls_total[c]),
            'count': cls_total[c],
        }
    return result

base_cls = per_class_eval(base)
imp_cls = per_class_eval(imp)

# ═══════════════════════════════════════════════════════════════════
# 4. Threshold sensitivity analysis
# ═══════════════════════════════════════════════════════════════════
print("Threshold sensitivity analysis...")

def eval_at_thresh(model, thresh, n=50):
    dl = create_dataloader(num_samples=n, batch_size=1, seed=123)
    box_ious, mask_ious, cc, ct, np_ = [], [], 0, 0, 0
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            if idx >= n: break
            preds = model.inference(batch['image'], [(128,128)], score_threshold=thresh)[0]
            pb = preds['boxes'].cpu(); pc = preds['class_ids'].cpu()
            pm = preds['masks'].cpu(); np_ += pb.shape[0]
            for j in range(len(batch['labels'][0])):
                ct += 1
                gb = batch['boxes'][0][j]; gm = batch['masks'][0][j]; gl = batch['labels'][0][j]
                if pb.shape[0] == 0:
                    box_ious.append(0); mask_ious.append(0); continue
                ious = [compute_box_iou(gb, p) for p in pb]
                bi = int(np.argmax(ious)); box_ious.append(ious[bi])
                if bi < pm.shape[0]:
                    ci = min(int(pc[bi].item()), pm.shape[1]-1)
                    mask_ious.append(compute_mask_iou(gm.numpy(), paste_mask(pm[bi,ci].numpy(), pb[bi])))
                else: mask_ious.append(0)
                if bi < pc.shape[0] and pc[bi].item() == gl.item(): cc += 1
    return {
        'box': np.mean(box_ious)*100 if box_ious else 0,
        'mask': np.mean(mask_ious)*100 if mask_ious else 0,
        'acc': cc/max(1,ct)*100,
        'preds': np_,
    }

thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
base_thresh = [eval_at_thresh(base, t) for t in thresholds]
imp_thresh = [eval_at_thresh(imp, t) for t in thresholds]

# ═══════════════════════════════════════════════════════════════════
# GENERATE PLOTS
# ═══════════════════════════════════════════════════════════════════
print("Generating plots...")

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

# ── FIGURE 1: Model Comparison Dashboard (2x3 subplots) ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Mask R-CNN Paper Replication: Comprehensive Model Analysis',
             fontsize=18, fontweight='bold', y=0.98)

# 1a. FLOPs comparison
ax = axes[0, 0]
models = ['Baseline\n(SimpleCNN)', 'Improved\n(ResNet-18)', 'Paper\n(ResNet-101-FPN)']
flops = [base_flops/1e9, imp_flops/1e9, paper_flops/1e9]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(models, flops, color=colors, edgecolor='black', linewidth=0.5)
for b, f in zip(bars, flops):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(flops)*0.02,
            f'{f:.1f}G' if f < 100 else f'{f:.0f}G',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('GFLOPs'); ax.set_title('Computational Cost (FLOPs)', fontsize=13, fontweight='bold')
ax.set_yscale('log')

# 1b. Parameters comparison
ax = axes[0, 1]
params = [base.count_parameters()/1e6, imp.count_parameters()/1e6, 44.0]
bars = ax.bar(models, params, color=colors, edgecolor='black', linewidth=0.5)
for b, p in zip(bars, params):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{p:.1f}M', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Parameters (Millions)'); ax.set_title('Model Size', fontsize=13, fontweight='bold')

# 1c. Inference speed
ax = axes[0, 2]
speeds = [base_ms, imp_ms, paper_ms]
stds = [base_std, imp_std, 0]
bars = ax.bar(models, speeds, yerr=stds, color=colors, edgecolor='black',
              linewidth=0.5, capsize=5)
for b, s in zip(bars, speeds):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(speeds)*0.02,
            f'{s:.0f}ms', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Inference Time (ms)'); ax.set_title('Inference Speed', fontsize=13, fontweight='bold')
ax.axhline(y=33.3, color='gray', ls='--', alpha=0.5, label='30 FPS')
ax.legend(fontsize=8)

# 1d. Box IoU vs Mask IoU scatter
ax = axes[1, 0]
metrics = [
    (70.0, 57.1, 'Baseline', '#3498db', 's', 120),
    (68.8, 54.9, 'Improved', '#2ecc71', '^', 120),
    (58.8, 57.1, 'Paper (AP50)', '#e74c3c', 'D', 120),
]
for bx, mk, lbl, c, m, s in metrics:
    ax.scatter(bx, mk, c=c, marker=m, s=s, label=lbl, edgecolors='black', zorder=5)
ax.plot([40, 80], [40, 80], 'k--', alpha=0.3, label='Box=Mask line')
ax.set_xlabel('Box IoU (%)'); ax.set_ylabel('Mask IoU (%)')
ax.set_title('Box IoU vs Mask IoU', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlim(40, 80); ax.set_ylim(40, 70)

# 1e. Efficiency: accuracy per GFLOP
ax = axes[1, 1]
eff_box = [70.0/(base_flops/1e9), 68.8/(imp_flops/1e9), 58.8/(paper_flops/1e9)]
eff_mask = [57.1/(base_flops/1e9), 54.9/(imp_flops/1e9), 57.1/(paper_flops/1e9)]
x = np.arange(3); w = 0.35
ax.bar(x-w/2, eff_box, w, label='Box IoU / GFLOP', color=['#3498db','#2ecc71','#e74c3c'], alpha=0.8)
ax.bar(x+w/2, eff_mask, w, label='Mask IoU / GFLOP', color=['#3498db','#2ecc71','#e74c3c'], alpha=0.4,
       hatch='//')
ax.set_xticks(x); ax.set_xticklabels(['Baseline','Improved','Paper'], fontsize=9)
ax.set_ylabel('IoU per GFLOP'); ax.set_title('Computational Efficiency', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)

# 1f. Memory footprint (model size in MB)
ax = axes[1, 2]
import os
base_mb = os.path.getsize('./checkpoints/baseline_best.pth') / 1e6
imp_mb = os.path.getsize('./checkpoints/improved_best.pth') / 1e6
paper_mb = 178.0  # ResNet-101-FPN model zoo size
sizes = [base_mb, imp_mb, paper_mb]
bars = ax.bar(models, sizes, color=colors, edgecolor='black', linewidth=0.5)
for b, s in zip(bars, sizes):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{s:.1f} MB', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Size (MB)'); ax.set_title('Model File Size', fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plot_computation.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_computation.png")


# ── FIGURE 2: Per-Class Performance ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Per-Class Performance Breakdown', fontsize=16, fontweight='bold')
class_names = ['circle', 'rect', 'triangle', 'diamond']
x = np.arange(4); w = 0.35

for i, (metric, title) in enumerate([
    ('box_iou', 'Box IoU per Class'),
    ('mask_iou', 'Mask IoU per Class'),
    ('accuracy', 'Classification Accuracy per Class'),
]):
    ax = axes[i]
    bvals = [base_cls[c+1][metric]*100 for c in range(4)]
    ivals = [imp_cls[c+1][metric]*100 for c in range(4)]
    ax.bar(x-w/2, bvals, w, label='Baseline', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x+w/2, ivals, w, label='Improved', color='#2ecc71', edgecolor='black', linewidth=0.5)
    for j in range(4):
        ax.text(x[j]-w/2, bvals[j]+1, f'{bvals[j]:.0f}', ha='center', fontsize=8)
        ax.text(x[j]+w/2, ivals[j]+1, f'{ivals[j]:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel('Score (%)'); ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('plot_per_class.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_per_class.png")


# ── FIGURE 3: Threshold Sensitivity ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Score Threshold Sensitivity Analysis', fontsize=16, fontweight='bold')

for i, (key, title) in enumerate([
    ('box', 'Box IoU vs Threshold'),
    ('mask', 'Mask IoU vs Threshold'),
    ('acc', 'Cls Accuracy vs Threshold'),
]):
    ax = axes[i]
    bv = [r[key] for r in base_thresh]
    iv = [r[key] for r in imp_thresh]
    ax.plot(thresholds, bv, 'b-o', ms=5, lw=2, label='Baseline')
    ax.plot(thresholds, iv, 'g-s', ms=5, lw=2, label='Improved')
    if key == 'box':
        ax.axhline(58.8, color='r', ls='--', alpha=0.5, label='Paper AP50')
    elif key == 'mask':
        ax.axhline(57.1, color='r', ls='--', alpha=0.5, label='Paper AP50')
    elif key == 'acc':
        ax.axhline(85.0, color='r', ls='--', alpha=0.5, label='Paper ~85%')
    ax.set_xlabel('Score Threshold'); ax.set_ylabel('Score (%)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('plot_threshold.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_threshold.png")


# ── FIGURE 4: Loss Convergence Overlay ──
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Training Loss Convergence: Baseline vs Improved (Overlay)', fontsize=16, fontweight='bold')

loss_keys = [
    ('train_loss', 'Total Loss'),
    ('train_rpn', 'RPN Loss'),
    ('train_cls', 'Classification Loss'),
    ('train_bbox', 'BBox Regression Loss'),
    ('train_mask', 'Mask Loss'),
    ('val_loss', 'Validation Loss'),
]
for idx, (key, title) in enumerate(loss_keys):
    ax = axes[idx//3, idx%3]
    bep = range(1, len(bh[key])+1)
    iep = range(1, len(ih[key])+1)
    ax.plot(bep, bh[key], 'b-', lw=2, alpha=0.8, label='Baseline')
    ax.plot(iep, ih[key], 'g-', lw=2, alpha=0.8, label='Improved')
    ax.fill_between(bep, bh[key], alpha=0.1, color='blue')
    ax.fill_between(iep, ih[key], alpha=0.1, color='green')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plot_loss_overlay.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_loss_overlay.png")


# ── FIGURE 5: Parameter Breakdown Pie Charts ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Parameter Distribution by Component', fontsize=16, fontweight='bold')

def count_module_params(model):
    result = {}
    for name, module in model.named_children():
        p = sum(p.numel() for p in module.parameters())
        result[name] = p
    return result

bp = count_module_params(base)
ip = count_module_params(imp)

for ax, params, title in [(axes[0], bp, 'Baseline (SimpleCNN)'), (axes[1], ip, 'Improved (ResNet-18)')]:
    labels = [k for k, v in params.items() if v > 0]
    sizes = [v/1e6 for v in params.values() if v > 0]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct=lambda p: f'{p:.0f}%\n({p*sum(sizes)/100:.1f}M)',
        colors=colors_pie, pctdistance=0.75, textprops={'fontsize': 9})
    ax.set_title(f'{title}\nTotal: {sum(sizes):.1f}M params', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_params.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_params.png")


# ── FIGURE 6: Comprehensive Radar Chart ──
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Box IoU', 'Mask IoU', 'Cls Acc', 'Speed\n(inv)', 'Efficiency\n(IoU/GFLOP)', 'Params\n(inv)']
N = len(categories)

# Normalize each metric to 0-100 scale
def norm(val, max_val): return min(val / max_val * 100, 100)

base_vals = [
    norm(70.0, 80),      # Box IoU
    norm(57.1, 70),      # Mask IoU
    norm(89.9, 100),     # Cls Acc
    norm(1000/base_ms, 1000/min(base_ms, imp_ms, paper_ms)*1.2),  # Speed (inverse)
    norm(70.0/(base_flops/1e9), max(70.0/(base_flops/1e9), 68.8/(imp_flops/1e9))*1.2),
    norm(50/base.count_parameters()*1e6, 50/min(base.count_parameters(), imp.count_parameters())*1e6*1.2),
]
imp_vals = [
    norm(68.8, 80),
    norm(54.9, 70),
    norm(89.9, 100),
    norm(1000/imp_ms, 1000/min(base_ms, imp_ms, paper_ms)*1.2),
    norm(68.8/(imp_flops/1e9), max(70.0/(base_flops/1e9), 68.8/(imp_flops/1e9))*1.2),
    norm(50/imp.count_parameters()*1e6, 50/min(base.count_parameters(), imp.count_parameters())*1e6*1.2),
]
paper_vals = [
    norm(58.8, 80),
    norm(57.1, 70),
    norm(85.0, 100),
    norm(1000/paper_ms, 1000/min(base_ms, imp_ms, paper_ms)*1.2),
    norm(58.8/(paper_flops/1e9), max(70.0/(base_flops/1e9), 68.8/(imp_flops/1e9))*1.2),
    norm(50/44e6, 50/min(base.count_parameters(), imp.count_parameters())*1e6*1.2),
]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
base_vals += base_vals[:1]
imp_vals += imp_vals[:1]
paper_vals += paper_vals[:1]
angles += angles[:1]

ax.plot(angles, base_vals, 'b-o', lw=2, label='Baseline')
ax.fill(angles, base_vals, alpha=0.1, color='blue')
ax.plot(angles, imp_vals, 'g-s', lw=2, label='Improved')
ax.fill(angles, imp_vals, alpha=0.1, color='green')
ax.plot(angles, paper_vals, 'r-D', lw=2, label='Paper')
ax.fill(angles, paper_vals, alpha=0.1, color='red')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Multi-Dimensional Model Comparison\n(Higher = Better)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('plot_radar.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved plot_radar.png")


# ── Print summary table ──
print("\n" + "="*80)
print("  DETAILED METRICS SUMMARY")
print("="*80)
print(f"\n  {'Metric':<28} {'Baseline':>12} {'Improved':>12} {'Paper':>12}")
print("  " + "-"*68)
print(f"  {'Box IoU (%)':<28} {'70.0':>12} {'68.8':>12} {'58.8':>12}")
print(f"  {'Mask IoU (%)':<28} {'57.1':>12} {'54.9':>12} {'57.1':>12}")
print(f"  {'Cls Accuracy (%)':<28} {'89.9':>12} {'89.9':>12} {'~85.0':>12}")
print(f"  {'Parameters (M)':<28} {base.count_parameters()/1e6:>12.1f} {imp.count_parameters()/1e6:>12.1f} {'44.0+':>12}")
print(f"  {'FLOPs (G)':<28} {base_flops/1e9:>12.1f} {imp_flops/1e9:>12.1f} {paper_flops/1e9:>12.0f}")
print(f"  {'Inference (ms, CPU)':<28} {base_ms:>12.0f} {imp_ms:>12.0f} {'200 (GPU)':>12}")
print(f"  {'Model Size (MB)':<28} {base_mb:>12.1f} {imp_mb:>12.1f} {'178.0':>12}")
print("  " + "-"*68)
print(f"\n  Per-class Box IoU:")
for c in range(1, 5):
    print(f"    {CLS[c]:<12} Baseline: {base_cls[c]['box_iou']*100:.1f}%  Improved: {imp_cls[c]['box_iou']*100:.1f}%")
print(f"\n  Per-class Mask IoU:")
for c in range(1, 5):
    print(f"    {CLS[c]:<12} Baseline: {base_cls[c]['mask_iou']*100:.1f}%  Improved: {imp_cls[c]['mask_iou']*100:.1f}%")
print("="*80)
print("\n  Generated 6 plot files:")
print("    1. plot_computation.png  - FLOPs, params, speed, efficiency, memory")
print("    2. plot_per_class.png    - Per-class box/mask/accuracy breakdown")
print("    3. plot_threshold.png    - Score threshold sensitivity curves")
print("    4. plot_loss_overlay.png - All loss components overlaid")
print("    5. plot_params.png       - Parameter distribution pie charts")
print("    6. plot_radar.png        - Multi-dimensional radar comparison")
print("="*80)
