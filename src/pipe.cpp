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

    if (img_idx == -1 || dx_idx == -1) {
        throw runtime_error("Metadata Error: Missing required columns 'image_id' or 'dx'.");
    }

    // 2. Process Records
    while (getline(file, line)) {
        stringstream line_stream(line);
        string cell;
        string current_id, current_dx;
        int col_counter = 0;

        while (getline(line_stream, cell, ',')) {
            cell.erase(remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            
            if (col_counter == img_idx) current_id = cell;
            if (col_counter == dx_idx)  current_dx = cell;
            col_counter++;
        }

        if (!current_id.empty()) {
            if (current_dx == TARGET_DIAGNOSIS) {
                melanoma_ids.push_back(current_id);
            } else {
                other_ids.push_back(current_id);
            }
        }
    }

    cout << "[Dataset Summary]\n"
         << " - Melanoma (Positive): " << melanoma_ids.size() << "\n"
         << " - Others   (Negative): " << other_ids.size() << endl;
}

string findImagePath(const string& image_id) {
    string path = IMG_DIR + image_id;
    ifstream file(path);
    if (file.good()) return path;
    cout << "could not find image at path:" << path << endl; return "";
}

//param 1 -> cv::dnn::Net (loaded from ONNX)
//param 2 -> image to infere Net on
Mat extractFeatures(Net& net, const string& image_id) {
    //returns 1x1280 vec
    string image_path = findImagePath(image_id);
    string mask_path = MASK_DIR + image_id + ".png";
    if (image_path.empty()) {
        cerr << "WARN: couldnt load img: " << image_path << endl;
    }

    Mat img = imread(image_path, IMREAD_COLOR);
    Mat mask = imread(mask_path, IMREAD_GRAYSCALE);

    if (img.empty()) {
        cerr << "WARN: couldnt load img" << image_path << endl;
    }
    if (mask.empty()) {
        cerr << "WARN: couldnt load mask" << mask_path << endl;
    }

    Mat maskedImg;
    img.copyTo(maskedImg, mask);
    //resize for cnn input
    resize(maskedImg, maskedImg, size(224,224),0 ,0, INTER_LINEAR );
    //3x3 kernel blur
    medianBlur(maskedImg, maskedImg, 3);






}
