# Beginner Guide: Rewriting pipeline.cpp Into pipe.cpp

This file is a practical guide for rewriting the working program in `src/pipeline.cpp` into `src/pipe.cpp` by hand.

The goal is not to copy the full file line by line. The goal is to rebuild it in small pieces so you understand what each part does and keep the program compiling as you go.

There is already a long explanation in `guide.md`. Use that for deeper theory. This file is the short implementation checklist.

## Important First Note

Right now the project builds `src/pipeline.cpp`, not `src/pipe.cpp`.

In `CMakeLists.txt` the target is:

```cmake
add_executable(pipeline src/pipeline.cpp)
```

When your rewrite is ready to test, change that line to:

```cmake
add_executable(pipeline src/pipe.cpp)
```

Do not do that too early. First make `pipe.cpp` compile.

## Best Beginner Strategy

Build the program in this order:

1. Make `pipe.cpp` compile with empty helper functions.
2. Implement one helper at a time.
3. Test each step before moving on.
4. Keep the first version simple and close to `pipeline.cpp`.
5. Only refactor into cleaner classes after the first working version exists.

This is much safer than trying to write the entire pipeline in one pass.

## Step 1: Fix the Current Syntax Problems in pipe.cpp

Before you implement any logic, fix the broken skeleton.

Your current `pipe.cpp` has several compile errors:

- Missing `;` after `model_path`
- Missing `;` after `struct DataRecord`
- Missing `;` after `struct LoadedData`
- Constructor parameters use `;` instead of `,`
- Missing headers like `<fstream>`, `<sstream>`, `<string>`, `<algorithm>`, `<numeric>`, and `<random>`
- `readNet(model_path)` should be `readNetFromONNX(model_path)` for this ONNX file
- The vectors in `main()` are missing `;`
- The class closing braces are incomplete

Do not try to implement the whole dataset class first. First make the file syntactically valid.

## Step 2: Start From a Small Working Skeleton

Before writing real logic, create empty helper functions with the same job as the ones in `pipeline.cpp`.

Recommended helper functions:

```cpp
void loadMetadata(std::vector<std::string>& melanomaIds,
                  std::vector<std::string>& otherIds);

std::string findImagePath(const std::string& imageId);

cv::Mat extractFeatures(cv::dnn::Net& net, const std::string& imageId);

void standardScale(cv::Mat& data, cv::Mat& featureMean, cv::Mat& featureStd);

void applyScale(cv::Mat& data,
                const cv::Mat& featureMean,
                const cv::Mat& featureStd);
```

Then give each one a temporary empty implementation so the file compiles.

Example idea:

```cpp
cv::Mat extractFeatures(cv::dnn::Net& net, const std::string& imageId)
{
    return cv::Mat();
}
```

This lets you build the structure first and fill in the logic later.

## Step 3: Prefer the pipeline.cpp Design Over the Current ImageDataset Idea

`pipeline.cpp` uses free helper functions. That is a good choice for a beginner because:

- each function has one clear job
- it is easier to compile and debug
- it is easier to compare with the working reference file

My recommendation: do not force the first working version into an `ImageDataset` class.

If you want to keep `ImageDataset`, keep it small:

- the constructor should only parse metadata and store records
- a separate method should load image and mask files
- training logic should stay in `main()`

For a first rewrite, a function-based design is simpler and better.

## Step 4: Add the Same Configuration Constants

Copy the idea of the constants from `pipeline.cpp`.

You need constants for:

- number of samples per class
- test ratio
- PCA component count
- cross-validation fold count
- ONNX model path
- metadata CSV path
- image directory 1
- image directory 2
- mask directory

Best practice:

- use relative paths, not your absolute home directory path
- keep all config values near the top of the file
- give constants clear names in all caps or `kCamelCase`

## Step 5: Implement loadMetadata()

This is the first real helper to write.

Job of `loadMetadata()`:

1. Open `HAM10000_metadata.csv`
2. Read the header row
3. Find the `image_id` column
4. Find the `dx` column
5. For each row:
   - if `dx == "mel"`, add the image ID to the melanoma list
   - otherwise add it to the other list

Why this function matters:

- it turns the CSV into training labels
- it separates melanoma from non-melanoma
- it gives you the list of images to process later

Best practices for this step:

- print how many melanoma and non-melanoma IDs you loaded
- check that the column names were actually found
- exit early with a clear error if the file cannot be opened

## Step 6: Implement findImagePath()

The images are split across two folders. This helper checks both.

Job of `findImagePath()`:

1. Build `part_1/image_id.jpg`
2. Build `part_2/image_id.jpg`
3. Return the path that exists

Best practice improvement over `pipeline.cpp`:

- if neither path exists, return an empty string instead of silently returning a bad path

That makes debugging much easier.

## Step 7: Implement extractFeatures()

This is the core image-processing function.

Write it in this exact order:

1. Find the image path
2. Build the mask path
3. Load the color image with `imread(..., IMREAD_COLOR)`
4. Load the mask with `imread(..., IMREAD_GRAYSCALE)`
5. If either load fails, print a warning and return an empty `Mat`
6. Apply the lesion mask with `copyTo(maskedImg, mask)`
7. Resize to `224 x 224`
8. Apply `medianBlur(..., 3)`
9. Convert to `CV_32FC3` and scale pixels into `[0, 1]`
10. Apply ImageNet mean and standard deviation normalization
11. Create a blob with `blobFromImage(...)`
12. Call `net.setInput(blob, "image")`
13. Call `net.forward("features")`
14. Reshape the result into one row and return it

Do not skip the empty checks. Many training bugs come from trying to process missing files.

## Step 8: Test extractFeatures() Before the Full Dataset Loop

Before running hundreds of images, test one image ID manually.

You want to confirm:

- the image loads
- the mask loads
- the ONNX model runs
- the result is not empty
- the feature vector has the expected shape

Print the output size. A healthy result should be one row with 2048 columns.

This single test will save you a lot of time.

## Step 9: Build the Labeled Dataset

Once metadata loading works:

1. Shuffle the melanoma IDs
2. Shuffle the non-melanoma IDs
3. Cut both down to the same sample count
4. Build two vectors:
   - `allIds`
   - `allLabels`
5. Use label `1` for melanoma and `0` for other

Then loop over all IDs and call `extractFeatures()`.

For each successful result:

- append the feature row to a `Mat` such as `allFeatures`
- append the matching label to `usedLabels`

Best practice:

- skip samples that fail to load instead of crashing the whole run
- print progress every 50 images

## Step 10: Split Into Train and Test Sets

You now need a random train/test split.

Implementation idea:

1. Create an `indices` vector from `0` to `nTotal - 1`
2. Shuffle it with a fixed random seed like `42`
3. Put the first part into training
4. Put the rest into test

Keep the first version simple. You do not need a complicated class for this.

Best practice:

- keep the seed fixed while debugging so your results are repeatable
- print how many samples ended up in train and test

## Step 11: Implement Standard Scaling

Write `standardScale()` and `applyScale()` exactly as separate steps.

Important rule:

- fit the scaler on the training data only
- use that same mean and standard deviation on the test data

Do not compute scaling statistics on the test set. That leaks information from test into training.

## Step 12: Add PCA

After scaling, reduce the feature dimension.

The order is:

1. fit PCA on the scaled training features
2. project training features
3. project test features

Again, do not fit PCA on the test data.

## Step 13: Train the SVM

Use OpenCV's SVM just like in `pipeline.cpp`.

Key settings:

- type: `SVM::C_SVC`
- kernel: `SVM::RBF`
- training helper: `trainAuto(...)`

You also need:

- `TrainData::create(...)`
- integer labels in a `CV_32S` matrix

Best practice:

- keep the SVM setup in one block so it is easy to read
- print the final selected `C` and `gamma`

## Step 14: Evaluate the Model

After prediction, count:

- true positives
- true negatives
- false positives
- false negatives

Then compute:

- accuracy
- precision
- recall
- F1 score

This is a good beginner checkpoint because it proves the whole training pipeline works end to end.

## Step 15: Save the Trained Artifacts

At the end, save:

- the SVM model
- the PCA object
- the scaler mean and standard deviation

That gives you everything needed for later inference.

## Suggested Rewrite Order

Follow this exact order while coding:

1. Fix syntax and missing includes in `pipe.cpp`
2. Add constants
3. Add empty helper functions
4. Make the file compile
5. Implement `loadMetadata()`
6. Implement `findImagePath()`
7. Implement `extractFeatures()`
8. Test feature extraction on one image
9. Implement the dataset loop
10. Implement train/test split
11. Implement scaling
12. Implement PCA
13. Implement SVM training
14. Implement evaluation
15. Implement saving
16. Switch CMake to build `pipe.cpp`

If you follow this order, you will usually know exactly which step caused a bug.

## Common Beginner Mistakes to Avoid

- Mixing `cv::String` and `std::string` without a reason
- Using an absolute path like `/home/.../feature_extractor.onnx`
- Writing too much code before compiling once
- Not checking `Mat.empty()` after `imread()`
- Fitting the scaler on both train and test
- Fitting PCA on both train and test
- Returning invalid paths from `findImagePath()`
- Trying to debug the full dataset loop before testing a single image
- Changing too many things at once

## What to Compare Against While You Work

Use `src/pipeline.cpp` as your reference for:

- helper function responsibilities
- preprocessing order
- machine-learning pipeline order
- evaluation logic
- saved output files

Do not try to invent a new design during the first rewrite. First match the working behavior. Then refactor.

## If You Still Want to Keep ImageDataset

If you really want your own class-based version, this is the safe design:

- `DataRecord` stores image ID or file paths and the diagnosis label
- `ImageDataset` reads the CSV and stores `vector<DataRecord>`
- one method returns the list of records
- one method loads image and mask data for one record
- feature extraction, scaling, PCA, SVM, and evaluation stay outside the class

That keeps the class focused on data access instead of trying to do everything.

## Final Advice

Your first target is not a pretty program. Your first target is a correct program that compiles, runs on one image, then runs on the whole dataset.

Once that works, you can clean up names, improve structure, and refactor repeated code.