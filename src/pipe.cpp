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

// ImageNet channel statistics (RGB order) -- opencv uses BGR
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
