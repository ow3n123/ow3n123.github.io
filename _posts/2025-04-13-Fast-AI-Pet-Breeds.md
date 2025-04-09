# Key Insights from Implementing Pet Breed Classification with fastai

TOC {:toc}

## 🐾 Introduction

I took a keen intreset in implmenting the pet breeds classification notebook since I was personally a big fan of pets and wanted to tinker with how the deep learning worked./ While working through the **Pet Breeds** classification notebook from the fastai course, I picked up several useful strategies and deepened my understanding of key deep learning concepts. 

---

## 🖼️ 1. Smart Data Augmentation with Presizing

fastai introduces a clever image transformation trick called **presizing**:

```python
item_tfms = Resize(460)
batch_tfms = aug_transforms(size=224, min_scale=0.75)
```

Presizing helps retain image quality by resizing *before* cropping and applying augmentations. This improves model accuracy and training efficiency.

---
## 📦 2. The Power of `DataBlock`

Creating flexible and clean pipelines is so much easier with fastai's `DataBlock` API:

```python
pets = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)
```

Regex-based labeling and `DataBlock.summary()` were extremely helpful during setup and debugging.

---

## 🎯 3. Training & Transfer Learning

Training a high-performing model with minimal effort:

```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
```

This uses fastai’s transfer learning pipeline:
- Train the new (randomly initialized) layers first
- Then unfreeze and train the whole model

---

## 📉 4. Cross-Entropy Loss Explained

For multi-class classification, fastai uses:

```python
loss_func = nn.CrossEntropyLoss()
```

Which combines **softmax** and **negative log likelihood** into:

$$
L = -\sum_{i=1}^{C} y_i \log(p_i)
$$

Where:
- \( y_i \) is the true label (one-hot encoded)
- \( p_i \) is the predicted probability for class \( i \)
- \( C \) is the number of classes

This function is the backbone of classification training and important to understand for debugging and improving accuracy.

---

**Next Steps:** I'll be exploring model interpretation and deployment in the next stage of my fastai journey!
