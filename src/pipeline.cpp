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

static const int    N_SAMPLES_PER_CLASS = 1500;
static const float  TEST_RATIO          = 0.2f;
static const int    PCA_COMPONENTS      = 128;
static const int    CV_FOLDS            = 5;
static const string MODEL_PATH   = "../feature_extractor.onnx";
static const string METADATA_CSV = "../dataset/skin-cancer-lesions-segmentation/data/metadata.csv";
static const string IMG_DIR      = "../dataset/skin-cancer-lesions-segmentation/data/images/";
static const string MASK_DIR     = "../dataset/skin-cancer-lesions-segmentation/data/masks/";

static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};


class StandardScaler
{
public:
    void fit(const Mat& data)
    {
        reduce(data, mean_, 0, REDUCE_AVG);

        Mat sq;
        multiply(data, data, sq);
        Mat sq_mean;
        reduce(sq, sq_mean, 0, REDUCE_AVG);

        Mat mean_sq;
        multiply(mean_, mean_, mean_sq);
        sqrt(sq_mean - mean_sq, std_);
        std_.setTo(1.0f, std_ < 1e-8f);
    }

    void transform(Mat& data) const
    {
        for (int r = 0; r < data.rows; ++r)
            data.row(r) = (data.row(r) - mean_) / std_;
    }

    void fitTransform(Mat& data)
    {
        fit(data);
        transform(data);
    }

    void save(FileStorage& fs) const
    {
        fs << "mean" << mean_ << "std" << std_;
    }

private:
    Mat mean_, std_;
};


void loadMetadata(vector<string>& mel_ids, vector<string>& other_ids)
{
    ifstream file(METADATA_CSV);
    if (!file.is_open())
    {
        cerr << "Cannot open: " << METADATA_CSV << endl;
        exit(1);
    }

    string line, token;
    getline(file, line);

    istringstream hdr(line);
    int col = 0, img_col = -1, mel_col = -1;
    while (getline(hdr, token, ','))
    {
        token.erase(remove(token.begin(), token.end(), '\r'), token.end());
        if (token == "image") img_col = col;
        if (token == "MEL")   mel_col = col;
        ++col;
    }

    if (img_col < 0 || mel_col < 0)
    {
        cerr << "Missing 'image' or 'MEL' column in metadata." << endl;
        exit(1);
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        string img_id, mel_val;
        int c = 0;
        while (getline(ss, token, ','))
        {
            token.erase(remove(token.begin(), token.end(), '\r'), token.end());
            if (c == img_col) img_id  = token;
            if (c == mel_col) mel_val = token;
            ++c;
        }
        if (mel_val == "1.0" || mel_val == "1")
            mel_ids.push_back(img_id);
        else
            other_ids.push_back(img_id);
    }

    cout << "Metadata: " << mel_ids.size() << " melanoma, "
         << other_ids.size() << " other." << endl;
}

string findImagePath(const string& id)
{
    string p = IMG_DIR + id + ".jpg";
    if (ifstream(p).good()) return p;
    return "";
}

Mat extractFeatures(Net& net, const string& id)
{
    string img_path  = findImagePath(id);
    string mask_path = MASK_DIR + id + ".png";

    if (img_path.empty())
    {
        cerr << "  WARN: image not found: " << id << endl;
        return Mat();
    }

    Mat img  = imread(img_path,  IMREAD_COLOR);
    Mat mask = imread(mask_path, IMREAD_GRAYSCALE);

    if (img.empty() || mask.empty())
    {
        cerr << "  WARN: failed to load: " << id << endl;
        return Mat();
    }

    Mat masked;
    img.copyTo(masked, mask);
    resize(masked, masked, Size(224, 224), 0, 0, INTER_LINEAR);
    medianBlur(masked, masked, 3);
    masked.convertTo(masked, CV_32FC3, 1.0 / 255.0);

    // BGR channels normalised against ImageNet stats (RGB order, so B↔R indices flip)
    vector<Mat> bgr(3);
    split(masked, bgr);
    bgr[0] = (bgr[0] - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    bgr[1] = (bgr[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    bgr[2] = (bgr[2] - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    merge(bgr, masked);

    Mat blob = blobFromImage(masked, 1.0, Size(224, 224), Scalar(), true, false);
    net.setInput(blob, "image");
    Mat features = net.forward("features");
    return features.reshape(1, 1);
}


int main()
{
    Net net = readNetFromONNX(MODEL_PATH);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    vector<string> mel_ids, other_ids;
    loadMetadata(mel_ids, other_ids);

    mt19937 rng(42);
    shuffle(mel_ids.begin(),   mel_ids.end(),   rng);
    shuffle(other_ids.begin(), other_ids.end(), rng);

    mel_ids.resize(min((int)mel_ids.size(),   N_SAMPLES_PER_CLASS));
    other_ids.resize(min((int)other_ids.size(), N_SAMPLES_PER_CLASS));

    cout << "Sampling " << mel_ids.size() << " melanoma, "
         << other_ids.size() << " other." << endl;

    vector<string> all_ids;
    vector<int>    all_labels;
    for (auto& id : mel_ids)   { all_ids.push_back(id); all_labels.push_back(1); }
    for (auto& id : other_ids) { all_ids.push_back(id); all_labels.push_back(0); }

    Mat allFeatures;
    vector<int> used_labels;

    cout << "Extracting features (" << all_ids.size() << " images)..." << endl;
    for (size_t i = 0; i < all_ids.size(); ++i)
    {
        if (i % 50 == 0)
            cout << "  [" << i << "/" << all_ids.size() << "] " << all_ids[i] << endl;

        Mat feat = extractFeatures(net, all_ids[i]);
        if (feat.empty()) continue;

        allFeatures.push_back(feat);
        used_labels.push_back(all_labels[i]);
    }
    cout << "Feature matrix: " << allFeatures.rows << " x " << allFeatures.cols << endl;

    int n_total = allFeatures.rows;
    int n_test  = max(1, (int)(n_total * TEST_RATIO));
    int n_train = n_total - n_test;

    vector<int> indices(n_total);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    Mat train_feats, test_feats;
    vector<int> train_labels, test_labels;
    for (int i = 0; i < n_total; ++i)
    {
        int idx = indices[i];
        if (i < n_train)
        {
            train_feats.push_back(allFeatures.row(idx));
            train_labels.push_back(used_labels[idx]);
        }
        else
        {
            test_feats.push_back(allFeatures.row(idx));
            test_labels.push_back(used_labels[idx]);
        }
    }
    cout << "Split: " << n_train << " train, " << n_test << " test." << endl;

    StandardScaler scaler;
    scaler.fitTransform(train_feats);
    scaler.transform(test_feats);

    cout << "PCA: " << train_feats.cols << " -> " << PCA_COMPONENTS << " dims" << endl;
    PCA pca(train_feats, Mat(), PCA::DATA_AS_ROW, PCA_COMPONENTS);
    Mat reduced_train = pca.project(train_feats);
    Mat reduced_test  = pca.project(test_feats);

    Mat train_labels_mat(train_labels, true);
    train_labels_mat.convertTo(train_labels_mat, CV_32S);

    cout << "Training SVM (RBF, " << CV_FOLDS << "-fold CV)..." << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 1e-6));

    Ptr<TrainData> train_data = TrainData::create(reduced_train, ROW_SAMPLE, train_labels_mat);
    svm->trainAuto(train_data, CV_FOLDS,
                   SVM::getDefaultGrid(SVM::C),
                   SVM::getDefaultGrid(SVM::GAMMA),
                   SVM::getDefaultGrid(SVM::P),
                   SVM::getDefaultGrid(SVM::NU),
                   SVM::getDefaultGrid(SVM::COEF),
                   SVM::getDefaultGrid(SVM::DEGREE));

    cout << "Best C: " << svm->getC() << "  gamma: " << svm->getGamma() << endl;

    Mat predictions;
    svm->predict(reduced_test, predictions);

    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < n_test; ++i)
    {
        int actual = test_labels[i];
        int pred   = (int)predictions.at<float>(i, 0);
        if      (actual == 1 && pred == 1) ++TP;
        else if (actual == 0 && pred == 0) ++TN;
        else if (actual == 0 && pred == 1) ++FP;
        else                               ++FN;
    }

    float accuracy  = (float)(TP + TN) / n_test;
    float precision = TP + FP > 0 ? (float)TP / (TP + FP) : 0.0f;
    float recall    = TP + FN > 0 ? (float)TP / (TP + FN) : 0.0f;
    float f1        = precision + recall > 0
                      ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    cout << "\nTest set (" << n_test << " samples):\n";
    cout << "           Pred 0   Pred 1\n";
    cout << "  Act 0     " << TN << "       " << FP << "\n";
    cout << "  Act 1     " << FN << "       " << TP << "\n\n";
    cout << "Accuracy  : " << accuracy  * 100 << "%\n";
    cout << "Precision : " << precision * 100 << "%\n";
    cout << "Recall    : " << recall    * 100 << "%\n";
    cout << "F1        : " << f1        * 100 << "%\n";

    svm->save("isic_svm_model.xml");
    {
        FileStorage pca_fs("isic_pca.xml", FileStorage::WRITE);
        pca.write(pca_fs);
    }
    {
        FileStorage scaler_fs("isic_scaler.xml", FileStorage::WRITE);
        scaler.save(scaler_fs);
    }

    cout << "\nSaved: isic_svm_model.xml, isic_pca.xml, isic_scaler.xml\n";
    return 0;
}
