#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void detectObj(const Mat &img, CascadeClassifier &cascade, Rect &obj, int scaledw);
void detectTheEyes(Mat &face, Point &leftEye, Point &rightEye, CascadeClassifier &eyesCascade1, CascadeClassifier &eyesCascade2);