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
static const string IMG_DIR    = "../skin-cancer-mnist-ham10000/HAM10000_images_part_1/";
static const string MASK_DIR      = "../ham10000-lesion-segmentations/"
                                    "HAM10000_segmentations_lesion_tschandl/";
//global namespaces
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::ml;


// ImageNet channel statistics (RGB order)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f}; // R, G, B
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f}; // R, G, B

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
    int col = 0, image_id_col = -1, dx_col = -1;
    while (getline(hdr, token, ','))
    {
        // Trim CR/LF
        token.erase(remove(token.begin(), token.end(), '\r'), token.end());
        if (token == "image_id") image_id_col = col;
        if (token == "dx")       dx_col       = col;
        ++col;
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