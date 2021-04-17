#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

double getSimilarity(const Mat face1, const Mat face2);
void showTrainingDebugData(const Ptr<EigenFaceRecognizer> model, const int faceWidth, const int faceHeight);
Mat getImgFro1DFMat(const Mat mat_row, int h);
Mat reconstructFace(const Ptr<EigenFaceRecognizer> model, const Mat preprocessed);