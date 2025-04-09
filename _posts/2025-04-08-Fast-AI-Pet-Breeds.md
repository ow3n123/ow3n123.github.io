# Key Insights from Implementing Pet Breed Classification with fastai

TOC {:toc}

## 🐾 Introduction

I took a keen intreset in implmenting the pet breeds classification notebook since I was personally a big fan of pets and wanted to tinker with how the deep learning worked. While working through the **Pet Breeds** classification notebook from the fastai course, I picked up several useful strategies and deepened my understanding of key deep learning concepts. 

---

## 🖼️ 1. Smart Data Augmentation with Presizing

fastai introduces a clever image transformation trick called **presizing**:

```python
item_tfms = Resize(460)
batch_tfms = aug_transforms(size=224, min_scale=0.75)
```

Presizing helps retain image quality by resizing *before* cropping and applying augmentations. This improves model accuracy and training efficiency.

![](/images/Screenshot 2025-04-09 194707.png "Presizing")

---

## 🎯 2. Training & Transfer Learning

Training a high-performing model with minimal effort:

```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
```

This uses fastai’s transfer learning pipeline:
- Train the new (randomly initialized) layers first
- Then unfreeze and train the whole model

---

## 📉 3. Cross-Entropy Loss Explained

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

