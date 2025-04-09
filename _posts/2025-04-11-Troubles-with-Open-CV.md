## 🧩 Problem: `cv2.ximgproc` Module Not Found

While working on the fingerprint processing task, I tried using `cv2.ximgproc` for edge detection and got this error:

```python
ImportError: module 'cv2.cv2' has no attribute 'ximgproc'
```
Turns out, `ximgproc` is part of OpenCV’s extra modules, which aren't included in the default `opencv-python` package. To access them, I needed the **contrib** version.

---

## 🛠️ Solution: Install `opencv-contrib-python`

Here’s what worked for me:

### 1. **Uninstall the default OpenCV package**

```bash
pip uninstall opencv-python
```

### 2. **Install the contrib version**

```bash
pip install opencv-contrib-python
```

> ✅ This version includes all modules from `opencv-python` **plus** the `ximgproc`, `xfeatures2d`, and other extra goodies.

---

## 🧪 Quick Test

After installation, I verified it by running the fingerprint detection with no errors:

```python
skeleton = cv.ximgproc.thinning(ridge_lines, thinningType=cv.ximgproc.THINNING_GUOHALL)
```

![](/images/image.png "Code Running Succesfully")

---
Hope this helps if you're stuck on the same issue! 🚀
