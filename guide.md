# Melanoma Classifier — Deep Dive Guide

This guide walks through every piece of the pipeline from raw dermoscopy images to a trained classifier, explaining the **why** behind every decision. Written for someone learning C++ and machine learning simultaneously.

---

## Table of Contents

1. [Big Picture — What Does This Pipeline Do?](#1-big-picture)
2. [Repository Layout](#2-repository-layout)
3. [The Neural Network (Python side)](#3-the-neural-network-python-side)
4. [ONNX — The Bridge Between Python and C++](#4-onnx--the-bridge-between-python-and-c)
5. [Building the Project (CMake)](#5-building-the-project-cmake)
6. [C++ Deep Dive — pipeline.cpp](#6-c-deep-dive--pipelinecpp)
   - [Includes and Namespaces](#61-includes-and-namespaces)
   - [Configuration Constants](#62-configuration-constants)
   - [Helper: loadMetadata](#63-helper-loadmetadata)
   - [Helper: findImagePath](#64-helper-findimagepath)
   - [Helper: extractFeatures — the core preprocessing pipeline](#65-helper-extractfeatures--the-core-preprocessing-pipeline)
   - [Helper: standardScale / applyScale](#66-helper-standardscale--applyscale)
   - [main() — Orchestrating Everything](#67-main--orchestrating-everything)
7. [Machine Learning Concepts Explained](#7-machine-learning-concepts-explained)
   - [Why Features, Not Raw Pixels?](#71-why-features-not-raw-pixels)
   - [Transfer Learning](#72-transfer-learning)
   - [StandardScaler — Why Scaling Matters](#73-standardscaler--why-scaling-matters)
   - [PCA — Dimensionality Reduction](#74-pca--dimensionality-reduction)
   - [SVM — Support Vector Machine](#75-svm--support-vector-machine)
   - [The RBF Kernel](#76-the-rbf-kernel)
   - [C and Gamma — The Two SVM Knobs](#77-c-and-gamma--the-two-svm-knobs)
   - [trainAuto — Automatic Hyperparameter Search](#78-trainauto--automatic-hyperparameter-search)
8. [Train/Test Split — Why You Need Held-Out Data](#8-trainttest-split--why-you-need-held-out-data)
9. [Evaluation Metrics — Reading the Results](#9-evaluation-metrics--reading-the-results)
   - [Confusion Matrix](#91-confusion-matrix)
   - [Accuracy](#92-accuracy)
   - [Precision](#93-precision)
   - [Recall (Sensitivity)](#94-recall-sensitivity)
   - [F1 Score](#95-f1-score)
   - [Which Metric Matters Most Here?](#96-which-metric-matters-most-here)
10. [Saved Artefacts — What Gets Written to Disk](#10-saved-artefacts--what-gets-written-to-disk)
11. [Data Flow Summary](#11-data-flow-summary)
12. [Tuning Knobs — What to Change to Improve Results](#12-tuning-knobs--what-to-change-to-improve-results)

---

## 1. Big Picture

### What is the problem?

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7303/syn18143926) is a dataset of 10,015 dermoscopy images (photos of skin lesions taken with a special camera). Each image is labeled with one of 7 skin condition types. Our goal is **binary classification**: given a new lesion image, predict whether it is **melanoma** (label `mel`, class 1) or **not melanoma** (class 0).

Melanoma is the most dangerous form of skin cancer. Early detection is critical — a missed melanoma (a "false negative") is far more dangerous than a false alarm.

### The two-stage approach

The pipeline uses a well-established technique called **feature extraction + classifier**:

```
Raw Image
    │
    ▼
[ ResNet50 Neural Network ]  ← pretrained on ImageNet, frozen
    │
    ▼
2048-dimensional feature vector  ← a "summary" of the image
    │
    ▼  (StandardScaler → PCA)
128-dimensional reduced vector
    │
    ▼
[ SVM Classifier ]  ← trained from scratch on our data
    │
    ▼
Melanoma / Not Melanoma
```

Why not train an end-to-end neural network? With ~1000 melanoma images, there's not enough labeled data to train a big network reliably. Using a pretrained feature extractor (ResNet50 already knows what textures, edges, and shapes look like from 1.2 million ImageNet images) and then fitting a lightweight SVM on top gives strong results with much less data.

---

## 2. Repository Layout

```
classifier/
├── feature_extractor.onnx         ← The exported ResNet50 (89 MB, all weights embedded)
├── CMakeLists.txt                 ← Build system instructions
├── build.sh                       ← Convenience build script
├── model_design/
│   ├── model.py                   ← Python definition of the ResNet50 feature extractor
│   └── export_opencv.py           ← Script to export model.py → ONNX in OpenCV-compatible format
├── src/
│   └── pipeline.cpp               ← The entire C++ training pipeline
└── skin-cancer-mnist-ham10000/
    ├── HAM10000_metadata.csv      ← image_id, dx (diagnosis), age, sex, etc.
    ├── HAM10000_images_part_1/    ← ISIC_XXXXXXX.jpg (first ~5000 images)
    └── HAM10000_images_part_2/    ← ISIC_XXXXXXX.jpg (remaining ~5000 images)
ham10000-lesion-segmentations/
└── HAM10000_segmentations_lesion_tschandl/
    └── ISIC_XXXXXXX_segmentation.png  ← binary mask: white=lesion, black=background
```

### What is HAM10000_metadata.csv?

A CSV (comma-separated values) file. Each row is one image. Key columns:

| Column | Meaning |
|--------|---------|
| `image_id` | e.g. `ISIC_0032797` — matches the filename |
| `dx` | Diagnosis: `mel`=melanoma, `nv`=mole, `bcc`=basal cell carcinoma, etc. |
| `dx_type` | How the diagnosis was confirmed (histopathology, follow-up, etc.) |
| `age`, `sex`, `localization` | Patient metadata (not used by this pipeline) |

### What are the segmentation masks?

Each mask is a grayscale PNG where:
- **White pixels (255)** = the skin lesion
- **Black pixels (0)** = everything else (ruler, hair, healthy skin, background)

Applying the mask before feeding the image into the network forces the neural network to focus on only the lesion itself, ignoring distracting background content.

---

## 3. The Neural Network (Python side)

File: [model_design/model.py](model_design/model.py)

### What is ResNet50?

ResNet50 is a **convolutional neural network** with 50 layers, originally designed to classify images into 1000 categories (ImageNet dataset). "ResNet" stands for Residual Network — it uses "skip connections" that let gradients flow backward more easily during training, solving the "vanishing gradient" problem that prevented very deep networks from training.

```
Input: (batch, 3, 224, 224)   ← 224×224 RGB image
  conv1 (64 filters)
  bn1, relu, maxpool
  layer1 (3 residual blocks, 64 channels)
  layer2 (4 residual blocks, 128 channels)
  layer3 (6 residual blocks, 256 channels)
  layer4 (3 residual blocks, 512 channels)
  avgpool                      ← collapses spatial dimensions
Output: (batch, 2048)         ← feature vector per image
```

### What did we change?

Normally ResNet50 ends with `fc` (fully connected layer): `2048 → 1000` classes.

We replace `fc` with `nn.Identity()` — which just passes through whatever it receives unchanged. So instead of class probabilities, we get the raw 2048-dimensional feature vector from the last pooling layer.

```python
self.resnet.fc = nn.Identity()   # "do nothing" — expose the 2048-d features
```

### Why freeze the weights?

```python
for param in self.resnet.parameters():
    param.requires_grad = False
```

The network was trained on ImageNet. Its weights already encode useful visual knowledge (edges, textures, shapes, colors). We don't want to modify this knowledge — we just want to use it. Freezing also means no backward pass through the network is needed, which is much faster.

### L2 Normalisation

```python
features = F.normalize(features, p=2, dim=1)
```

After the network produces a 2048-d vector, we normalise it to have length 1 (unit L2 norm). This puts all feature vectors on the surface of a high-dimensional sphere, which:
- Removes magnitude differences between images (a brighter image doesn't produce larger numbers)
- Improves SVM performance with RBF kernel (which measures distances between points)

---

## 4. ONNX — The Bridge Between Python and C++

**ONNX** (Open Neural Network Exchange) is a standardised file format for neural networks. Think of it as a PDF for ML models — it stores the network structure and all weights in a single, language-agnostic file.

### Why do we need it?

The model was defined and trained in Python/PyTorch. The inference pipeline is in C++. ONNX lets us:
1. Export the network from Python once
2. Load and run it in C++ using OpenCV's `dnn` module — no Python or PyTorch needed at runtime

### Why does export_opencv.py exist separately from model.py?

PyTorch ≥ 2.5 uses a new "TorchDynamo" exporter by default, which produces graphs that OpenCV can't parse. [export_opencv.py](model_design/export_opencv.py) forces the **legacy exporter** (`dynamo=False`) with opset 13 (older ONNX operator set version) that OpenCV 4.x fully supports.

```python
torch.onnx.export(
    extractor,
    dummy,
    str(output_path),
    opset_version=13,     # OpenCV 4.x supports up to opset 13 well
    dynamo=False,         # ← CRITICAL: forces legacy exporter
    ...
)
```

---

## 5. Building the Project (CMake)

File: [CMakeLists.txt](CMakeLists.txt)

### What is CMake?

CMake is a **build system generator**. Instead of manually typing `g++ -o pipeline src/pipeline.cpp -lopencv_core ...` (which would be very long), you write a `CMakeLists.txt` that describes what to build and what libraries it needs, and CMake figures out all the compiler flags.

### Key parts of CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)   # Minimum CMake version required
project(melanoma_classifier CXX)       # Project name, CXX = C++ language

set(CMAKE_CXX_STANDARD 17)             # Use C++17 features (e.g. structured bindings)
```

```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn ml)
```
This finds the OpenCV installation on your system and makes its headers and libraries available. `REQUIRED` means CMake will error out if OpenCV is not found. The `COMPONENTS` list is the specific OpenCV modules we use:
- `core` — `Mat`, `FileStorage`, basic types
- `imgproc` — `resize`, `medianBlur`, `split`, `merge`
- `imgcodecs` — `imread` (reads JPEG/PNG files)
- `dnn` — loads and runs the ONNX model
- `ml` — SVM, PCA, TrainData

```cmake
set(CMAKE_INSTALL_RPATH "/run/host/usr/lib64:/run/host/usr/lib:/usr/lib64:/usr/lib")
add_link_options(-Wl,--disable-new-dtags)
```
This embeds the OpenCV library search path **inside the compiled binary** (called RPATH). Without this, on Linux you'd get "cannot open shared object file" errors when running the binary, because the runtime linker wouldn't know where OpenCV's `.so` files are. The `--disable-new-dtags` flag produces `DT_RPATH` (which transitive dependencies inherit) instead of `DT_RUNPATH` (which they don't).

---

## 6. C++ Deep Dive — pipeline.cpp

File: [src/pipeline.cpp](src/pipeline.cpp)

### 6.1 Includes and Namespaces

```cpp
#include <opencv2/opencv.hpp>   // Main OpenCV header (pulls in most modules)
#include <opencv2/dnn.hpp>      // Neural network inference (readNetFromONNX, etc.)
#include <opencv2/ml.hpp>       // Machine learning (SVM, PCA, TrainData)
#include <algorithm>            // std::shuffle, std::min, std::remove
#include <fstream>              // std::ifstream (reading files)
#include <iostream>             // std::cout, std::cerr (printing to terminal)
#include <numeric>              // std::iota (fills a range with sequential numbers)
#include <random>               // std::mt19937 (random number generator)
#include <sstream>              // std::istringstream (parse strings as streams)
#include <string>               // std::string
#include <vector>               // std::vector (dynamic array)
```

**What is an `#include`?** In C++, code is split across multiple files. `#include` copy-pastes the contents of another file (called a **header**) into your source file at compile time. Headers declare what functions and classes exist — the actual compiled code is linked in separately.

```cpp
using namespace cv;
using namespace cv::dnn;
using namespace cv::ml;
using namespace std;
```

Without these lines, you'd have to write `cv::Mat`, `cv::dnn::Net`, `std::vector<std::string>` everywhere. `using namespace` lets you drop the prefix and write just `Mat`, `Net`, `vector<string>`. It's a convenience — in large codebases it's often avoided because it can cause name conflicts, but for a single-file program like this it's fine.

### 6.2 Configuration Constants

```cpp
static const int    N_SAMPLES_PER_CLASS = 500;
static const float  TEST_RATIO          = 0.2f;
static const int    PCA_COMPONENTS      = 128;
static const int    CV_FOLDS            = 5;
static const string MODEL_PATH    = "../feature_extractor.onnx";
```

**`static const`** — `const` means the value cannot change after initialization (it's a constant). `static` at file scope means the variable is local to this translation unit (this `.cpp` file) — it won't conflict with a variable of the same name in another file.

**`f` suffix on floats** — `0.2f` is a `float` literal. Without the `f`, `0.2` is a `double` (64-bit) and the compiler would warn about narrowing it to `float` (32-bit).

**Why `../ `paths?** The binary is run from `build/`. The data lives in the parent directory (`../`). These paths are relative to the working directory at runtime.

### 6.3 Helper: loadMetadata

```cpp
void loadMetadata(vector<string>& melanoma_ids, vector<string>& other_ids)
```

**`void`** — this function returns nothing (it fills the two vectors passed to it).

**`vector<string>&`** — `vector<string>` is a dynamic array of strings. The `&` means **pass by reference** — the function operates directly on the caller's variables, not on a copy. Without `&`, modifying the vectors inside the function would have no effect on the caller.

```cpp
ifstream file(METADATA_CSV);     // Open the file for reading
if (!file.is_open()) { ... }     // Check it actually opened
```

`ifstream` = input file stream. Think of it as a cursor pointing into a file — you can call `getline` to read one line at a time, moving the cursor forward.

```cpp
string line, token;
getline(file, line);    // Read the header row and discard it
```

`getline(stream, string)` reads characters until a newline `\n` and stores them in `string`.

```cpp
istringstream hdr(line);              // Wrap the header line in a stream
while (getline(hdr, token, ','))      // Split on commas
```

`istringstream` lets you use stream operations on a plain string. Here we're using it as a CSV parser — `getline(stream, token, ',')` reads up to the next comma delimiter.

```cpp
token.erase(remove(token.begin(), token.end(), '\r'), token.end());
```

This is the **erase-remove idiom** — the standard C++ way to remove all occurrences of a character from a string. It removes carriage returns (`\r`) which appear on Windows line endings (`\r\n`) and would silently break string comparisons like `token == "image_id"`.

### 6.4 Helper: findImagePath

```cpp
string findImagePath(const string& image_id)
{
    string p1 = IMG_DIR_1 + image_id + ".jpg";
    string p2 = IMG_DIR_2 + image_id + ".jpg";
    ifstream f1(p1);
    if (f1.good()) return p1;
    return p2;
}
```

The 10,015 images are split across two directories. This function checks whether a given `image_id` is in part 1 by trying to open the file. `ifstream::good()` returns `true` if the file opened and is ready to read. If it's not in part 1, we assume it's in part 2 (no second check needed, since we verified the data earlier).

**Why not use `imread` to check?** `imread` decodes the entire JPEG just to check if the file exists — wasteful. A file-open check is instantaneous.

### 6.5 Helper: extractFeatures — the core preprocessing pipeline

This is the most important function. It takes one image ID, loads it, preprocesses it, and runs it through the neural network to get a 2048-dimensional feature vector.

```cpp
Mat extractFeatures(Net& net, const string& image_id)
```

`Mat` is OpenCV's universal matrix / image type. Almost everything in OpenCV is a `Mat`.

**Step 1: Load image and mask**

```cpp
Mat img  = imread(img_path,  IMREAD_COLOR);      // Load as 3-channel BGR
Mat mask = imread(mask_path, IMREAD_GRAYSCALE);  // Load as 1-channel grayscale
```

`IMREAD_COLOR` loads as 3-channel BGR (Blue, Green, Red). Note: OpenCV uses **BGR** order by default, not RGB — a common source of bugs.
`IMREAD_GRAYSCALE` loads as a single channel (0–255 brightness).

**Step 2: Apply the segmentation mask**

```cpp
Mat maskedImg;
img.copyTo(maskedImg, mask);
```

`copyTo(destination, mask)` copies pixels from `img` to `maskedImg` only where `mask` is non-zero (white). Black mask pixels → black destination pixels. This removes hair, rulers, and healthy background skin, forcing the network to evaluate only the lesion.

**Step 3: Resize**

```cpp
resize(maskedImg, maskedImg, Size(224, 224), 0, 0, INTER_LINEAR);
```

The ResNet50 model was designed for 224×224 input. The HAM10000 images are ~600×450 pixels. `INTER_LINEAR` = bilinear interpolation (smooth resizing). We write back to the same Mat (in-place), which is fine with OpenCV.

**Step 4: Median filter**

```cpp
medianBlur(maskedImg, maskedImg, 3);
```

The median filter replaces each pixel with the median of its 3×3 neighbourhood. It removes **salt-and-pepper noise** (isolated extreme-value pixels from the camera sensor) while preserving edges better than a mean/Gaussian blur. The `3` is the kernel size (must be odd).

**Step 5: Convert to float32**

```cpp
maskedImg.convertTo(maskedImg, CV_32FC3, 1.0 / 255.0);
```

Neural networks expect floating-point input in [0, 1], but images are stored as uint8 (0–255). `CV_32FC3` = 32-bit float, 3 channels. `1.0/255.0` is the scale factor applied to every pixel during conversion.

**Step 6: ImageNet normalisation**

```cpp
vector<Mat> bgr(3);
split(maskedImg, bgr);           // Separate into 3 single-channel images
bgr[0] = (bgr[0] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]; // B channel
bgr[1] = (bgr[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]; // G channel
bgr[2] = (bgr[2] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]; // R channel
merge(bgr, maskedImg);           // Recombine into one 3-channel image
```

ResNet50 was trained on ImageNet where each channel was normalised to zero mean and unit variance using these specific statistics:
- R mean=0.485, std=0.229
- G mean=0.456, std=0.224
- B mean=0.406, std=0.225

If we don't apply the same normalisation at inference time, the pixel value distribution the network sees is completely different from what it was trained on, and its outputs become meaningless.

The index juggling (`bgr[0]` → MEAN[2]) is because OpenCV stores channels as BGR but the ImageNet stats are in RGB order, so the B channel (index 0 in OpenCV) needs the B stats (index 2 in the RGB-ordered arrays).

**Step 7: Build blob and forward pass**

```cpp
Mat blob = blobFromImage(maskedImg, 1.0, Size(224, 224),
                         Scalar(), /*swapRB=*/true, /*crop=*/false);
net.setInput(blob, "image");
Mat features = net.forward("features");   // Shape: (1, 2048)
return features.reshape(1, 1);            // Flatten to 1×2048 row vector
```

`blobFromImage` converts a Mat into the 4D tensor format that DNN networks expect: `(batch, channels, height, width)`. With `swapRB=true`, it also swaps the B and R channels (BGR→RGB), matching what the model was trained on.

`net.setInput(blob, "image")` feeds the blob into the layer named "image" (the input layer name we specified during ONNX export).

`net.forward("features")` runs the network forward and returns the output of the layer named "features" (our output layer name from ONNX export).

`features.reshape(1, 1)` reshapes the output to a single row — important because `allFeatures.push_back(feat)` stacks rows vertically to build the feature matrix.

### 6.6 Helper: standardScale / applyScale

```cpp
void standardScale(Mat& data, Mat& featureMean, Mat& featureStd)
{
    cv::reduce(data, featureMean, 0, REDUCE_AVG);  // Column-wise mean
    ...
    sqrt(sqMean - meanSq, featureStd);             // Column-wise std
    ...
    for (int r = 0; r < data.rows; ++r)
        data.row(r) = (data.row(r) - featureMean) / featureStd;
}
```

`cv::reduce(data, featureMean, 0, REDUCE_AVG)`:
- `0` means reduce along dimension 0 (rows) → result is 1 row, same number of columns as input
- `REDUCE_AVG` = compute the average
- Result: `featureMean` is a 1×2048 row containing the mean of each feature across all images

The standard deviation is computed via Var(X) = E[X²] − E[X]²:

```cpp
multiply(data, data, sq);             // sq[i,j] = data[i,j]²
cv::reduce(sq, sqMean, 0, REDUCE_AVG); // E[x²] per column
multiply(featureMean, featureMean, meanSq); // (E[x])² per column
sqrt(sqMean - meanSq, featureStd);    // std = sqrt(Var(x))
```

`featureStd.setTo(1.0f, featureStd < 1e-8f)` — if a feature has near-zero variance (constant across all images), we'd divide by ~0. This replaces those std values with 1 so those dimensions are effectively unscaled rather than producing infinities.

`applyScale` just applies the same pre-computed mean/std to new data (the test set), without recomputing — this is critical to avoid data leakage (see section 8).

### 6.7 main() — Orchestrating Everything

```cpp
mt19937 rng(42);
```

`mt19937` is the Mersenne Twister random number generator. The `42` is the **seed** — starting from the same seed always produces the same sequence of random numbers. This makes the experiment **reproducible**: running the program twice always picks the same sample of images and the same train/test split.

```cpp
shuffle(melanoma_ids.begin(), melanoma_ids.end(), rng);
```

`shuffle` randomly reorders a container. `.begin()` and `.end()` are **iterators** — pointers to the first element and one-past-the-last element of the vector. Shuffling before `resize()` ensures we pick a random subset, not just the first N images.

```cpp
melanoma_ids.resize(n_mel);   // Keep only first n_mel elements (after shuffle, these are random)
```

**Train/test split:**

```cpp
vector<int> indices(nTotal);
iota(indices.begin(), indices.end(), 0);   // Fill with 0, 1, 2, ..., nTotal-1
shuffle(indices.begin(), indices.end(), rng);
```

`iota` (from Greek ι, iota) fills a range with incrementing values. We shuffle an index array so we can split into train/test without physically reordering the heavy feature matrix.

```cpp
Ptr<SVM> svm = SVM::create();
```

`Ptr<T>` is OpenCV's reference-counted smart pointer (similar to `std::shared_ptr`). It automatically frees memory when no references remain — no need to call `delete`.

---

## 7. Machine Learning Concepts Explained

### 7.1 Why Features, Not Raw Pixels?

A 224×224 RGB image has 224 × 224 × 3 = **150,528 numbers**. Training an SVM directly on raw pixels would be:
- **Computationally infeasible** (SVMs scale poorly to high dimensions)
- **Statistically hopeless** — with only ~800 training images and 150,528 features, the model would just memorise noise

A ResNet50 feature vector compresses each image down to **2048 numbers** that represent *semantically meaningful* information: "this image has irregular borders", "this has a dark central region", "this has a blue-white veil" — patterns the network learned from millions of ImageNet images.

### 7.2 Transfer Learning

ResNet50 was trained on ImageNet (1.2 million photos of dogs, cats, keyboards, etc.) — not on skin lesions. Why does it still work?

Neural networks learn features in a **hierarchy**:
- Early layers detect edges, gradients, simple color patches
- Middle layers detect textures, shapes, curves
- Deep layers detect complex structures

The early and middle layers are **general** and transfer well to any visual domain. A network that can distinguish a husky from a malamute has learned to see very fine visual details — the same capabilities that distinguish melanoma from moles.

### 7.3 StandardScaler — Why Scaling Matters

Imagine a dataset with two features:
- Feature A: ranges from 0 to 1 (pixel intensity)
- Feature B: ranges from 0 to 10,000 (some large magnitude feature)

The SVM's RBF kernel computes **distances** between feature vectors. Without scaling, Feature B completely dominates — a difference of 1 in Feature A is invisible next to a difference of 10,000 in Feature B. The SVM effectively ignores Feature A.

StandardScaler transforms each feature to have **mean 0 and standard deviation 1**:

$$x' = \frac{x - \mu}{\sigma}$$

After scaling, moving 1 unit along any feature dimension represents the same "statistical distance". All features contribute equally to the kernel computation.

**Key detail:** We compute the mean/std on the **training set only**, then apply those same numbers to the test set. Computing stats on the test set and using them during training would constitute **data leakage** — the model would indirectly see test data, inflating measured performance.

### 7.4 PCA — Dimensionality Reduction

PCA (Principal Component Analysis) finds the directions of maximum variance in your data and projects onto those directions.

After StandardScaling, we have 2048 features. But many of these features are **correlated** with each other — they carry redundant information. PCA finds 128 new axes (linear combinations of the original 2048) that explain the most variance.

Benefits:
- **Speed**: SVM training is O(n²) to O(n³) in the number of features. 128 features trains in seconds; 2048 features takes much longer.
- **Noise reduction**: the discarded 1920 dimensions often capture noise rather than signal, so removing them can actually *improve* accuracy.
- **Addresses the curse of dimensionality**: in very high dimensions, all points tend to be equidistant from each other, making distance-based classifiers (like RBF SVM) struggle.

The `PCA::DATA_AS_ROW` flag tells OpenCV each row is one data point (our feature matrix is N_samples × 2048).

### 7.5 SVM — Support Vector Machine

An SVM finds the **maximum-margin hyperplane** — the decision boundary that separates the two classes with the largest possible gap.

**In 2D** (for intuition): imagine melanoma images as red dots and non-melanoma as blue dots on a 2D scatter plot. The SVM draws the line that:
1. Separates red from blue
2. Is as far from any point as possible (maximising the "margin")

The points closest to the boundary are called **support vectors** — they're the only points that matter for defining the boundary. Points far from the boundary have no influence.

**Why SVM for this problem?**
- Works well in high-dimensional spaces (128-d after PCA)
- Effective with a moderate number of training samples (~800 here)
- Has good theoretical guarantees and doesn't overfit as easily as neural networks when data is limited

### 7.6 The RBF Kernel

Real data is rarely linearly separable. The **kernel trick** implicitly maps data into a higher-dimensional space where linear separation becomes possible, without actually computing the transformation.

The **RBF (Radial Basis Function) kernel** is:

$$K(x, y) = e^{-\gamma \|x - y\|^2}$$

It measures **similarity** between two points: if they're close ($\|x-y\|^2 \approx 0$), $K \approx 1$. If far apart, $K \approx 0$.

The effect is that each training point creates a "bump" of influence — nearby test points get influenced, far ones don't. The network of bumps forms the decision boundary.

This is why StandardScaling is critical: the RBF kernel computes distances, so all features must be on the same scale.

### 7.7 C and Gamma — The Two SVM Knobs

**C (regularisation parameter):**
- Controls the trade-off between a clean margin and correct classification of training points
- **Small C**: allows some misclassification to get a wider, smoother margin → underfitting risk
- **Large C**: tries to classify all training points correctly, allows a narrower margin → overfitting risk
- Typical range: 0.1 to 1000

**Gamma (γ):**
- Controls how far the influence of a single training point reaches
- **Small γ**: large radius of influence → smoother decision boundary, may underfit
- **Large γ**: small radius → wiggly boundary that can fit exactly around each training point → overfitting
- A good starting point: γ ≈ 1/n_features (in this case ~1/128 ≈ 0.008)

### 7.8 trainAuto — Automatic Hyperparameter Search

```cpp
svm->trainAuto(trainData, CV_FOLDS,
               SVM::getDefaultGrid(SVM::C),
               SVM::getDefaultGrid(SVM::GAMMA), ...);
```

`trainAuto` performs **k-fold cross-validation grid search**:

1. Define a grid of candidate (C, γ) pairs on log scales
2. For each pair: split the training data into 5 folds, train on 4 folds, evaluate on the 5th, repeat 5 times, average the accuracy
3. Pick the (C, γ) with the best average cross-validation accuracy
4. Retrain the final SVM on all training data using those best parameters

OpenCV's default grids cover:
- C: roughly [0.1, 0.5, 2.5, 12.5, 62.5, 312.5] (log scale, factor ~5)
- γ: roughly [1e-5, 5e-5, 2.5e-4, 1.25e-3, ...] (log scale, factor ~5)

This is why the run shows "Best C: 62.5, Best γ: 1e-05" — those were the best from the grid search on your data.

---

## 8. Train/Test Split — Why You Need Held-Out Data

After feature extraction we have (approximately) 1000 feature vectors. We split them:
- **Training set (80%)**: used to fit the StandardScaler, PCA, and SVM
- **Test set (20%)**: held out completely — never seen during training

**Why?** If you measured accuracy on the same data you trained on, you'd be measuring memorisation, not generalisation. The model could perfectly "remember" all 800 training images and report 100% accuracy on them while failing completely on new images.

The test set simulates "new images arriving from a clinic" — measuring performance there tells you what to expect in the real world.

**Data leakage pitfall (what we avoid):**
The StandardScaler and PCA are fit **only on the training set**. The test set's transform uses training-set statistics. If we computed stats on the full dataset (train + test) before splitting, the test set would have subtly "leaked" information into the training process, inflating test performance.

---

## 9. Evaluation Metrics — Reading the Results

### 9.1 Confusion Matrix

The confusion matrix is the most complete summary of classifier performance. With 2 classes (melanoma=1, other=0):

```
                     Predicted 0 (other)    Predicted 1 (melanoma)
Actual 0 (other)          TN                      FP
Actual 1 (melanoma)       FN                      TP
```

| Term | Full Name | Meaning |
|------|-----------|---------|
| **TP** | True Positive | Correctly predicted as melanoma |
| **TN** | True Negative | Correctly predicted as not melanoma |
| **FP** | False Positive | Predicted melanoma but actually wasn't (false alarm) |
| **FN** | False Negative | Missed melanoma — predicted healthy but was actually melanoma |

**Example output:**
```
Confusion Matrix:
                 Predicted 0   Predicted 1
  Actual 0 (other)    78             2
  Actual 1 (mel)       3            17
```
→ 78 non-melanoma correctly identified, 17 melanoma correctly identified, 2 false alarms, 3 missed melanomas.

### 9.2 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{all correct}}{\text{all predictions}}$$

**"What fraction of all predictions were correct?"**

In the example above: (78 + 17) / (78 + 2 + 3 + 17) = 95/100 = **95%**

**Limitation:** Accuracy is misleading on **imbalanced datasets**. If 90% of your test images are non-melanoma, a model that always predicts "not melanoma" gets 90% accuracy — while completely failing at its job.

Our dataset is balanced (equal melanoma and non-melanoma samples), so accuracy is more meaningful here.

### 9.3 Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

**"Of all the cases I flagged as melanoma, how many actually were?"**

In the example: 17 / (17 + 2) = **89.5%**

Precision measures how trustworthy the positive predictions are. Low precision → lots of false alarms → patient anxiety, unnecessary biopsies.

### 9.4 Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

**"Of all the actual melanoma cases, how many did I catch?"**

In the example: 17 / (17 + 3) = **85%**

Recall measures how good the model is at finding all the true positives. Low recall → missed cancers → patients go undiagnosed, with potentially fatal consequences.

This is also called **sensitivity** in medical contexts, and is the most clinically important metric here. In screening applications, you want recall as close to 100% as possible, even at the cost of more false alarms.

### 9.5 F1 Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**"A single number that balances precision and recall."**

In the example: 2 × (0.895 × 0.85) / (0.895 + 0.85) = **87.2%**

F1 is the **harmonic mean** of precision and recall. The harmonic mean penalises extreme imbalances — a model with 100% precision but 10% recall gets F1 = 18%, not 55% as the arithmetic mean would give. You need both precision and recall to be high to get a high F1.

The harmonic mean formula is used (instead of arithmetic) because:
- If precision = 0 or recall = 0, F1 = 0 (the model is useless, regardless of the other metric)
- A model that achieves balance between the two is rewarded more than one that excels at one at the expense of the other

### 9.6 Which Metric Matters Most Here?

| Priority | Metric | Why |
|----------|--------|-----|
| **1st** | **Recall** | Missing a melanoma = patient may die. Must minimise FN. |
| **2nd** | **F1 Score** | Balanced measure — prevents gaming recall by flagging everything as melanoma |
| **3rd** | **Precision** | False positives cause unnecessary biopsies but aren't fatal |
| **4th** | **Accuracy** | Useful sanity check but can be misleading on imbalanced data |

A clinically deployable melanoma classifier should aim for **recall ≥ 90%** as a hard requirement.

---

## 10. Saved Artefacts — What Gets Written to Disk

After a successful run, three files are written (inside `build/`):

### isic_svm_model.xml
The trained SVM. Contains:
- The support vectors (the training points that define the boundary)
- Their weights (α coefficients)
- The bias term
- The C and γ parameters used

Load for inference: `Ptr<SVM> svm = SVM::load("isic_svm_model.xml");`

### isic_pca.xml
The PCA projection matrix. Contains:
- `mean`: the mean vector subtracted before projection
- `eigenvalues`: variances explained by each component
- `eigenvectors`: the projection axes (128 × 2048 matrix)

Load for inference:
```cpp
FileStorage fs("isic_pca.xml", FileStorage::READ);
PCA pca;
pca.read(fs.root());
Mat reduced = pca.project(newFeatures);
```

### isic_scaler.xml
The StandardScaler stats. Contains:
- `mean`: per-feature mean of the training set (1 × 2048)
- `std`: per-feature standard deviation of the training set (1 × 2048)

Load for inference:
```cpp
FileStorage fs("isic_scaler.xml", FileStorage::READ);
Mat mean, std;
fs["mean"] >> mean;
fs["std"]  >> std;
// Then apply: newFeature = (newFeature - mean) / std
```

---

## 11. Data Flow Summary

```
HAM10000_metadata.csv
    │
    ▼ loadMetadata()
    ├── melanoma_ids[]     (1113 IDs, shuffled, truncated to N_SAMPLES_PER_CLASS)
    └── other_ids[]        (8902 IDs, shuffled, truncated to N_SAMPLES_PER_CLASS)
         │
         ▼  for each image_id:
         │
         ├── findImagePath()  →  ISIC_XXXXXXX.jpg  (part1 or part2)
         ├── imread mask      →  ISIC_XXXXXXX_segmentation.png
         └── extractFeatures()
               │
               ├── img.copyTo(maskedImg, mask)          apply lesion mask
               ├── resize(224×224)                       match network input
               ├── medianBlur(3×3)                       remove noise
               ├── /255 → float32                        pixel scale to [0,1]
               ├── (x - ImageNet_mean) / ImageNet_std    channel normalisation
               ├── blobFromImage(swapRB=true)             BGR→RGB + 4D tensor
               └── net.forward() → 1×2048 float32        ResNet50 inference
                    │
                    ▼
allFeatures: N×2048 matrix, usedLabels: N×1 vector
    │
    ▼ 80/20 random split
    ├── trainFeatures (N*0.8 × 2048)
    └── testFeatures  (N*0.2 × 2048)
         │
         ▼ standardScale(trainFeatures, mean, std)
         ▼ applyScale(testFeatures, mean, std)
    ├── trainFeatures (normalised)
    └── testFeatures  (normalised, using TRAIN stats)
         │
         ▼ PCA fit on train, project both
    ├── reducedTrain (N*0.8 × 128)
    └── reducedTest  (N*0.2 × 128)
         │
         ▼ svm->trainAuto(reducedTrain, trainLabels)
         │   (5-fold CV grid search over C and γ)
         │
         ▼ svm->predict(reducedTest) → predictions
         │
         ▼ Compute TP, TN, FP, FN
         ▼ Print Accuracy, Precision, Recall, F1
         │
         ▼ Save: isic_svm_model.xml, isic_pca.xml, isic_scaler.xml
```

---

## 12. Tuning Knobs — What to Change to Improve Results

| Parameter | Location | Current | Try |
|-----------|----------|---------|-----|
| `N_SAMPLES_PER_CLASS` | pipeline.cpp:42 | 500 | Use all 1113 melanoma (set to 1113) — more data = better |
| `TEST_RATIO` | pipeline.cpp:43 | 0.2 | 0.2 is standard; 0.25 gives more test data on large sets |
| `PCA_COMPONENTS` | pipeline.cpp:44 | 128 | Try 64 (faster), 256 (more expressive) |
| `CV_FOLDS` | pipeline.cpp:45 | 5 | 10 gives more reliable estimates but takes 2× longer |
| Median filter kernel | pipeline.cpp:151 | 3 | 5 for more aggressive denoising |
| RBF grid C range | SVM::getDefaultGrid | auto | Custom grid: `SVM::ParamGrid(1, 1000, 10)` |
| RBF grid γ range | SVM::getDefaultGrid | auto | Custom grid: `SVM::ParamGrid(1e-6, 0.1, 5)` |

**Most impactful change:** Use all available melanoma data. With `N_SAMPLES_PER_CLASS = 1113`, you use all 1113 melanoma images and 1113 non-melanoma images (balanced). More data almost always improves recall for the minority class (melanoma).
