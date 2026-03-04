//cv
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>
//math & std
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::ml;

const String model_path = "/home/matyaspopela/Repos/classifier/feature_extractor.onnx"

struct DataRecord {
    String image_path;
    String mask_path;
    int diagnosis;
}

struct LoadedData {
    Mat image;
    Mat mask;
    int diagnosis;
}

class ImageDataset {
    private: 
        vector<DataRecord> records;
        string base_dir;
    public:
        ImageDataset(const String& csv_file; const String& base_dir="")
            : base_dir(base_dir) {

            ifstream file(csv_file);
            if (!file.is_open()) {
                cerr << "error, could not open CSV" << endl;
                return;
            }
            String line, img_path, mask_path, diag_str;

            //skip header
            getline(file, line);

            while (getline(file,line)) {
                stringstream ss(line);
                
            }

            }
}




int main() {
    //1. load ONXX model
    
    Net feature_net = readNet(model_path);

    //1.2 load dataset (!CURRENTLY A DUMMY)
    vector<String> image_paths = {}
    vector<String> mask_paths = {}
    vector<int> classificators = {0,1} //benign, malignant

    //2. pass imgs thru ONXX model (resnet50)

    
    //3. use opencvs PCA

    //4. set up SVMRBF

    //5. pass vector into SVM RBF

}

