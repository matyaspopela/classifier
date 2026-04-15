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

//config
static const int    N_SAMPLES_PER_CLASS = 500;  // balanced subset per class
static const float  TEST_RATIO          = 0.2f; 
static const int    PCA_COMPONENTS      = 80;  //after pca
static const int    CV_FOLDS            = 10;    
static const string MODEL_PATH    = "../feature_extractor.onnx";
static const string METADATA_CSV  = "../skin-cancer-mnist-ham10000/HAM10000_metadata.csv";
static const string IMG_DIR_1     = "../skin-cancer-mnist-ham10000/HAM10000_images_part_1/";
static const string IMG_DIR_2     = "../skin-cancer-mnist-ham10000/HAM10000_images_part_2/";
static const string MASK_DIR      = "../ham10000-lesion-segmentations/"
                                    "HAM10000_segmentations_lesion_tschandl/";

// ImageNet channel statistics (RGB order)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f}; // R, G, B
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f}; // R, G, B


/**
 * Categorizes images from the metadata CSV into Melanoma and non-Melanoma groups.
 * @param melanoma_ids Output list for Melanoma image filenames.
 * @param other_ids    Output list for all other diagnosis filenames.
 */
void loadMetadata(vector<string>& melanoma_ids, vector<string>& other_ids) 
{
    const string TARGET_DIAGNOSIS = "mel";
    ifstream file(METADATA_CSV);

    if (!file.is_open()) {
        throw runtime_error("Critical Error: Metadata file not found at " + METADATA_CSV);
    }

    string line;
    if (!getline(file, line)) return; // Empty file

    // 1. Identify Column Indices
    int img_idx = -1, dx_idx = -1;
    stringstream header_stream(line);
    string column_name;
    int current_col = 0;

    while (getline(header_stream, column_name, ',')) {
        // Remove whitespace/newlines
        column_name.erase(remove_if(column_name.begin(), column_name.end(), ::isspace), column_name.end());
        
        if (column_name == "image_id") img_idx = current_col;
        if (column_name == "dx")       dx_idx  = current_col;
        current_col++;
    }
    if (image_id_col < 0 || dx_col < 0)
    {
        cerr << "ERROR: Could not find image_id or dx column in metadata." << endl;
        exit(1);
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        string image_id, dx;
        int c = 0;
        while (getline(ss, token, ','))
        {
            token.erase(remove(token.begin(), token.end(), '\r'), token.end());
            if (c == image_id_col) image_id = token;
            if (c == dx_col)       dx        = token;
            ++c;
        }
        if (dx == "mel")
            melanoma_ids.push_back(image_id);
        else
            other_ids.push_back(image_id);
    }
    cout << "Metadata loaded: " << melanoma_ids.size() << " melanoma, "
         << other_ids.size() << " other." << endl;
}

string findImagePath(const string& image_id) {
    string path = IMG_DIR + image_id;
    ifstream file(path);
    if (file.good()) return path;
    cerr << "WARN: couldnt load at path: " << path << endl; return "";
}

Mat loadImage(const string& path, bool isMask = false) {
    if (isMask) Mat mask = imread(path, IMREAD_GRAYSCALE); return mask;
    Mat image = imread(path, IMREAD_COLOR); return image;
}

//for now we only copy, but consider refactoring for direct usage
Mat extractFeatures(Net& net,const string& image_path) {

    Mat image = imread(IMG_DIR + image_path, IMREAD_COLOR);
    Mat mask = imread(MASK_DIR + image_path, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "WARN: image at path: " << IMG_DIR + image_path <<" is empty" << endl;
        return Mat();
    }
    if (mask.empty()) {
        cerr << "WARN: mask at path: " << MASK_DIR + image_path <<" is empty" << endl;
        return Mat();
    }
    //apply mask
    Mat maskedImg;
    image.copyTo(maskedImg, mask);

    resize(maskedImg, maskedImg, Size(224, 224), 0, 0, INTER_LINEAR);

    medianBlur(maskedImg, maskedImg, 3);

    maskedImg.convertTo(maskedImg, CV_32FC3, 1.0/255.0);

    vector<Mat> bgr(3);
    split(maskedImg, bgr);

    bgr[0] = (bgr[0]-IMAGENET_MEAN[2]) / IMAGENET_STD[2];//B
    bgr[0] = (bgr[0]-IMAGENET_MEAN[1]) / IMAGENET_STD[1];//G
    bgr[0] = (bgr[0]-IMAGENET_MEAN[0]) / IMAGENET_STD[0];//R

    merge(bgr, maskedImg);

    Mat blob = blobFromImage(maskedImg, 1.0, Size(224, 224),
                            Scalar(), /*swaprb*/true, /*crop*/false);
    net.setInput(blob, "image");    




}