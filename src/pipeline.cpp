// =============================================================================
// Melanoma Binary Classifier — Feature Extraction + SVM Pipeline
//
// Dataset layout (paths relative to the repo root, assumed CWD):
//   dataset/skin-cancer-lesions-segmentation/data/metadata.csv  — image + MEL column
//   dataset/skin-cancer-lesions-segmentation/data/images/       — ISIC_*.jpg
//   dataset/skin-cancer-lesions-segmentation/data/masks/        — ISIC_*.png
//   feature_extractor.onnx   — EfficientNetV2-S backbone, 1280-d L2-normed output
//
// Model input:  (1, 3, 224, 224) float32, ImageNet-normalised, BGR→RGB swap
// Model output: (1, 1280)        float32, L2-normalised feature vector
//
// SVM hyper-parameters:
//   trainAuto() performs 5-fold cross-validated grid search over C and γ.
//   For EfficientNetV2-S 1280-d L2 features after PCA(128) the literature
//   suggests C ≈ 10 and γ ≈ 1/128 ≈ 0.008 as warm-start values; trainAuto()
//   explores a log-scale grid bracketing these ranges automatically.
// =============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace cv::ml;
using namespace std;

//config
static const int    N_SAMPLES_PER_CLASS = 1500;  // use all ~1113 melanoma + equal other
static const float  TEST_RATIO          = 0.2f;
static const int    PCA_COMPONENTS      = 128; // after pca; 1280-d → 128-d
static const int    CV_FOLDS            = 5;   // 5-fold cross-validation (standard)
static const string MODEL_PATH    = "../feature_extractor.onnx";
static const string METADATA_CSV  = "../dataset/skin-cancer-lesions-segmentation/data/metadata.csv";
static const string IMG_DIR      = "../dataset/skin-cancer-lesions-segmentation/data/images/";
static const string MASK_DIR     = "../dataset/skin-cancer-lesions-segmentation/data/masks/";

// ImageNet channel statistics (RGB order)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f}; // R, G, B
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f}; // R, G, B

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


void loadMetadata(vector<string>& melanoma_ids, vector<string>& other_ids)
{
    ifstream file(METADATA_CSV);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << METADATA_CSV << endl;
        exit(1);
    }

    string line, token;
    getline(file, line); // skip header

    // Find column indices
    istringstream hdr(line);
    int col = 0, image_col = -1, mel_col = -1;
    while (getline(hdr, token, ','))
    {
        // Trim CR/LF
        token.erase(remove(token.begin(), token.end(), '\r'), token.end());
        if (token == "image") image_col = col;
        if (token == "MEL")   mel_col   = col;
        ++col;
    }
    if (image_col < 0 || mel_col < 0)
    {
        cerr << "ERROR: Could not find 'image' or 'MEL' column in metadata." << endl;
        exit(1);
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        string image_id, mel_val;
        int c = 0;
        while (getline(ss, token, ','))
        {
            token.erase(remove(token.begin(), token.end(), '\r'), token.end());
            if (c == image_col) image_id = token;
            if (c == mel_col)   mel_val  = token;
            ++c;
        }
        if (mel_val == "1.0" || mel_val == "1")
            melanoma_ids.push_back(image_id);
        else
            other_ids.push_back(image_id);
    }
    cout << "Metadata loaded: " << melanoma_ids.size() << " melanoma, "
         << other_ids.size() << " other." << endl;
}

// Locate an image file in the images directory.
// Returns an empty string if the file is not found.
string findImagePath(const string& image_id)
{
    string p = IMG_DIR + image_id + ".jpg";
    ifstream f(p);
    if (f.good()) return p;
    return "";
}

// Preprocess one image and run it through the ONNX feature extractor.
// Returns a 1×1280 float32 Mat, or an empty Mat on failure.
Mat extractFeatures(Net& net, const string& image_id)
{
    string img_path  = findImagePath(image_id);
    string mask_path = MASK_DIR + image_id + ".png";

    if (img_path.empty())
    {
        cerr << "  WARN: Image not found in either directory: " << image_id << endl;
        return Mat();
    }

    Mat img  = imread(img_path,  IMREAD_COLOR);
    Mat mask = imread(mask_path, IMREAD_GRAYSCALE);

    if (img.empty())
    {
        cerr << "  WARN: Cannot load image: " << img_path << endl;
        return Mat();
    }
    if (mask.empty())
    {
        cerr << "  WARN: Cannot load mask: " << mask_path << endl;
        return Mat();
    }

    // 1. Apply lesion segmentation mask (background → black)
    Mat maskedImg;
    img.copyTo(maskedImg, mask);

    // 2. Resize to network input resolution
    resize(maskedImg, maskedImg, Size(224, 224), 0, 0, INTER_LINEAR);

    // 3. Median filter (3×3) — removes sensor noise before feature extraction
    medianBlur(maskedImg, maskedImg, 3);

    // 4. Convert to float32 in [0, 1]
    maskedImg.convertTo(maskedImg, CV_32FC3, 1.0 / 255.0);

    // 5. Per-channel ImageNet normalisation: (x − mean) / std
    //    OpenCV stores channels as BGR; ImageNet stats are in RGB order,
    //    so the index mapping is: B=ch0→idx2, G=ch1→idx1, R=ch2→idx0.
    vector<Mat> bgr(3);
    split(maskedImg, bgr);
    bgr[0] = (bgr[0] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]; // B
    bgr[1] = (bgr[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]; // G
    bgr[2] = (bgr[2] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]; // R
    merge(bgr, maskedImg);

    // 6. Build blob: swapRB=true converts BGR→RGB to match model expectations
    Mat blob = blobFromImage(maskedImg, 1.0, Size(224, 224),
                             Scalar(), /*swapRB=*/true, /*crop=*/false);

    // 7. Forward pass → 1280-d L2-normalised feature vector
    net.setInput(blob, "image");
    Mat features = net.forward("features"); // shape: (1, 1280)
    return features.reshape(1, 1);          // ensure row vector: 1×1280
}

// Zero-mean / unit-variance scaling (StandardScaler).
// Computes stats from `data` in-place and returns the mean/stddev rows
// so the same transform can be applied to new samples at inference time.
void standardScale(Mat& data, Mat& featureMean, Mat& featureStd)
{
    // Compute column-wise mean and std
    cv::reduce(data, featureMean, 0, REDUCE_AVG);    // 1 × D

    Mat sq;
    multiply(data, data, sq);
    Mat sqMean;
    cv::reduce(sq, sqMean, 0, REDUCE_AVG);           // E[x²]

    Mat meanSq;
    multiply(featureMean, featureMean, meanSq);  // (E[x])²

    sqrt(sqMean - meanSq, featureStd);           // std = sqrt(E[x²] − E[x]²)
    featureStd.setTo(1.0f,
        featureStd < 1e-8f);  // avoid division by zero for constant features

    for (int r = 0; r < data.rows; ++r)
    {
        data.row(r) = (data.row(r) - featureMean) / featureStd;
    }
}

// Apply a previously computed StandardScaler transform to new data.
void applyScale(Mat& data, const Mat& featureMean, const Mat& featureStd)
{
    for (int r = 0; r < data.rows; ++r)
    {
        data.row(r) = (data.row(r) - featureMean) / featureStd;
    }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------
int main()
{
    // ------------------------------------------------------------------
    // 1. Load ONNX feature extractor
    // ------------------------------------------------------------------
    cout << "Loading ONNX model from: " << MODEL_PATH << endl;
    Net net = readNetFromONNX(MODEL_PATH);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // ------------------------------------------------------------------
    // 2. Parse metadata and sample a balanced subset
    // ------------------------------------------------------------------
    vector<string> melanoma_ids, other_ids;
    loadMetadata(melanoma_ids, other_ids);

    mt19937 rng(42);
    shuffle(melanoma_ids.begin(), melanoma_ids.end(), rng);
    shuffle(other_ids.begin(),    other_ids.end(),    rng);

    int n_mel   = min((int)melanoma_ids.size(), N_SAMPLES_PER_CLASS);
    int n_other = min((int)other_ids.size(),    N_SAMPLES_PER_CLASS);
    melanoma_ids.resize(n_mel);
    other_ids.resize(n_other);

    cout << "Sampling " << n_mel << " melanoma and "
         << n_other << " non-melanoma images." << endl;

    // Build combined list with binary labels  (1 = melanoma, 0 = other)
    vector<string> all_ids;
    vector<int>    all_labels;
    for (auto& id : melanoma_ids) { all_ids.push_back(id); all_labels.push_back(1); }
    for (auto& id : other_ids)    { all_ids.push_back(id); all_labels.push_back(0); }

    // ------------------------------------------------------------------
    // 3. Feature extraction loop
    // ------------------------------------------------------------------
    Mat allFeatures;
    vector<int> usedLabels;

    cout << "Extracting features (" << all_ids.size() << " images)..." << endl;
    for (size_t i = 0; i < all_ids.size(); ++i)
    {
        if (i % 50 == 0)
            cout << "  [" << i << "/" << all_ids.size() << "] " << all_ids[i] << endl;

        Mat feat = extractFeatures(net, all_ids[i]);
        if (feat.empty()) continue;  // skip failed loads

        allFeatures.push_back(feat);
        usedLabels.push_back(all_labels[i]);
    }
    cout << "Feature matrix: " << allFeatures.rows << " × "
         << allFeatures.cols << endl;

    // ------------------------------------------------------------------
    // 4. Train / Test split (stratified-ish: same shuffle seed)
    // ------------------------------------------------------------------
    int nTotal = allFeatures.rows;
    int nTest  = max(1, (int)(nTotal * TEST_RATIO));
    int nTrain = nTotal - nTest;

    // Shuffle indices so train/test is random
    vector<int> indices(nTotal);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    Mat trainFeatures, testFeatures;
    vector<int> trainLabels, testLabels;
    for (int i = 0; i < nTotal; ++i)
    {
        int idx = indices[i];
        if (i < nTrain) {
            trainFeatures.push_back(allFeatures.row(idx));
            trainLabels.push_back(usedLabels[idx]);
        } else {
            testFeatures.push_back(allFeatures.row(idx));
            testLabels.push_back(usedLabels[idx]);
        }
    }
    cout << "Split: " << nTrain << " train, " << nTest << " test." << endl;

    // ------------------------------------------------------------------
    // 5. StandardScaler — fit on TRAIN, apply to both
    //    (critical for RBF SVM: all dimensions must be on the same scale)
    // ------------------------------------------------------------------
    cout << "Applying StandardScaler (fit on train)..." << endl;
    Mat featureMean, featureStd;
    standardScale(trainFeatures, featureMean, featureStd);
    applyScale(testFeatures, featureMean, featureStd);

    // ------------------------------------------------------------------
    // 6. PCA dimensionality reduction: 2048-d → PCA_COMPONENTS-d
    //    Fit on train, project both sets.
    // ------------------------------------------------------------------
    cout << "Running PCA (" << trainFeatures.cols << " → "
         << PCA_COMPONENTS << " dims)..." << endl;
    PCA pca(trainFeatures, Mat(), PCA::DATA_AS_ROW, PCA_COMPONENTS);
    Mat reducedTrain = pca.project(trainFeatures);
    Mat reducedTest  = pca.project(testFeatures);

    // ------------------------------------------------------------------
    // 7. Prepare labels
    // ------------------------------------------------------------------
    Mat trainLabelsMat(trainLabels, /*copyData=*/true);
    trainLabelsMat.convertTo(trainLabelsMat, CV_32S);

    Mat testLabelsMat(testLabels, /*copyData=*/true);
    testLabelsMat.convertTo(testLabelsMat, CV_32S);

    // ------------------------------------------------------------------
    // 8. SVM (C-SVC, RBF kernel) with automated hyper-parameter search
    //
    //    trainAuto() runs CV_FOLDS-fold cross-validation over log-scale
    //    grids for C and γ.  For PCA(128) features from EfficientNetV2-S
    //    the sweet-spot typically falls around C ≈ 10, γ ≈ 1/128 ≈ 0.008.
    //    trainAuto() covers this range with its default ParamGrid:
    //      C grid  : [0.1 .. 500]   (log-scale)
    //      γ grid  : [1e-5 .. 0.6]  (log-scale)
    //    Adjust CV_FOLDS or supply custom ParamGrid objects if you want a
    //    finer or coarser search.
    // ------------------------------------------------------------------
    cout << "Training SVM with RBF kernel (trainAuto, "
         << CV_FOLDS << "-fold CV)..." << endl;

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                     10000, 1e-6));

    Ptr<TrainData> trainData = TrainData::create(reducedTrain,
                                                  ROW_SAMPLE, trainLabelsMat);
    // trainAuto performs grid-search over C and Gamma via k-fold CV
    svm->trainAuto(trainData, CV_FOLDS,
                   SVM::getDefaultGrid(SVM::C),
                   SVM::getDefaultGrid(SVM::GAMMA),
                   SVM::getDefaultGrid(SVM::P),
                   SVM::getDefaultGrid(SVM::NU),
                   SVM::getDefaultGrid(SVM::COEF),
                   SVM::getDefaultGrid(SVM::DEGREE));

    cout << "Best C: "     << svm->getC()
         << "  Best \xCE\xB3: "   << svm->getGamma() << endl;

    // ------------------------------------------------------------------
    // 9. Evaluate on the held-out test set
    // ------------------------------------------------------------------
    cout << endl << "=== Evaluation on test set (" << nTest << " samples) ===" << endl;

    Mat predictions;
    svm->predict(reducedTest, predictions);

    // Confusion matrix: rows = actual, cols = predicted
    //   [TN  FP]
    //   [FN  TP]
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < nTest; ++i)
    {
        int actual = testLabels[i];
        int pred   = (int)predictions.at<float>(i, 0);
        if      (actual == 1 && pred == 1) ++TP;
        else if (actual == 0 && pred == 0) ++TN;
        else if (actual == 0 && pred == 1) ++FP;
        else                               ++FN;
    }

    float accuracy  = (float)(TP + TN) / nTest;
    float precision = TP + FP > 0 ? (float)TP / (TP + FP) : 0.0f;
    float recall    = TP + FN > 0 ? (float)TP / (TP + FN) : 0.0f;
    float f1        = precision + recall > 0
                      ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    cout << endl;
    cout << "Confusion Matrix:" << endl;
    cout << "                 Predicted 0   Predicted 1" << endl;
    cout << "  Actual 0 (other)    " << TN << "             " << FP << endl;
    cout << "  Actual 1 (mel)      " << FN << "             " << TP << endl;
    cout << endl;
    cout << "Accuracy  : " << (accuracy  * 100) << "%" << endl;
    cout << "Precision : " << (precision * 100) << "%  (of predicted melanoma, how many are correct)" << endl;
    cout << "Recall    : " << (recall    * 100) << "%  (of actual melanoma, how many were found)" << endl;
    cout << "F1 Score  : " << (f1        * 100) << "%" << endl;
    cout << endl;

    // ------------------------------------------------------------------
    // 10. Save all artefacts needed for inference
    // ------------------------------------------------------------------
    svm->save("isic_svm_model.xml");
    {
        FileStorage pcaFs("isic_pca.xml", FileStorage::WRITE);
        pca.write(pcaFs);   // writes mean, eigenvalues, eigenvectors
        // Load back at inference with: pca.read(pcaFs.root())
    }
    FileStorage scaler("isic_scaler.xml", FileStorage::WRITE);
    scaler << "mean" << featureMean << "std" << featureStd;
    scaler.release();

    cout << "Done.  Artefacts saved:\n"
         << "  isic_svm_model.xml  — trained SVM\n"
         << "  isic_pca.xml        — PCA projection\n"
         << "  isic_scaler.xml     — StandardScaler mean/std\n";

    return 0;
}