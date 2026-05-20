# FrameBeam

## Function

`FrameBeam` provides a minimal implementation of multimodal slope segmentation based on `GatedGridUNet`.
It supports:

- model definition and topology-aware loss (`GatedGridUNet.py`);
- multimodal data loading for `grays`, `depths`, `curves`, and `masks` (`dataset.py`);
- supervised training with best-checkpoint saving (`train.py`);
- single-sample inference with binary mask prediction (`infer.py`).

## Method

### Network

The model uses a dual-encoder gated-fusion U-Net structure:

- a texture encoder for image texture channels;
- a geometric encoder for depth/curve channels;
- multi-scale gated fusion blocks to combine encoder features;
- a U-Net-style decoder to produce the final segmentation logits.

### Loss

Training uses `TopoLoss`, a weighted combination of:

- binary cross-entropy loss;
- Dice loss;
- soft clDice loss based on differentiable skeletonization.

### Input and Output

Training data format:

```text
data_root/
  train/
    grays/*.png
    depths/*.png
    curves/*.png
    masks/*.png
  val/
    grays/*.png
    depths/*.png
    curves/*.png
    masks/*.png
```

Inference sample format:

```text
sample_dir/
  img.png
  depth.png
  curve.png
```

### Execution

Training:

```bash
python train.py --data-root /path/to/data_root --epochs 20 --batch-size 8 --device cuda
```

Inference:

```bash
python infer.py --model-path /path/to/best_model.pth --sample-dir /path/to/sample_dir --out-path /path/to/prediction.png --device cuda
```
