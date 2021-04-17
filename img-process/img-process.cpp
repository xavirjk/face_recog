#include <iostream>
#include <img-process/img-process.hpp>
using namespace std;
void equilizefaceHalves(Mat &faceImg)
{
    int width = faceImg.cols;
    int height = faceImg.rows;

    //equalize the whole face.
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    //Equalize left and the right half of the face separately.
    int middle = width / 2;
    Mat leftSide = faceImg(Rect(0, 0, middle, height));
    Mat rightSide = faceImg(Rect(middle, 0, width - middle, height));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int v;
            if (x < width / 4)
            {
                // Left 25%
                v = leftSide.at<uchar>(y, x);
            }
            else if (x < width * 2 / 4)
            {
                // Mid-left 25%
                int lv = leftSide.at<uchar>(y, x);
                int wv = wholeFace.at<uchar>(y, x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - width * 1 / 4) / (float)(width * 0.25f);
                v = cvRound((1.0f - f) * lv + (f)*wv);
            }
            else if (x < width * 3 / 4)
            {
                // Mid-right 25%
                int rv = rightSide.at<uchar>(y, x - middle);
                int wv = wholeFace.at<uchar>(y, x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - width * 2 / 4) / (float)(width * 0.25f);
                v = cvRound((1.0f - f) * wv + (f)*rv);
            }
            else
            {
                // Right 25%
                v = rightSide.at<uchar>(y, x - middle);
            }
            faceImg.at<uchar>(y, x) = v;
        }
    }
}

Mat processImage(Point leftEye, Point rightEye, Mat faceROI)
{
    Mat dest;
    //Center b2n the 2 eyes
    Point2f eyesCenter;
    eyesCenter.x = (leftEye.x + rightEye.x) * 0.5f;
    eyesCenter.y = (leftEye.y + rightEye.y) * 0.5f;
    //Angle b2n the 2 eyes
    double dy = rightEye.y - leftEye.y;
    double dx = rightEye.x - leftEye.x;
    double len = sqrt(dx * dx + dy * dy);
    //Convert radians to degrees

    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    const double DESIRED_LEFT_EYE_X = 0.16;
    const double DESIRED_LEFT_EYE_Y = 0.14;
    const double DESIRED_RIGHT_EYE_X = (1.0f - 0.16);
    //const double DESIRED_RIGHT_EYE_X = (0.8);

    const int DESIRED_FACE_WIDTH = 70;
    const int DESIRED_FACE_HEIGHT = 70;

    double desired_len = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * DESIRED_FACE_WIDTH;
    double scale = desired_len / len;

    Mat rotated_mat = getRotationMatrix2D(eyesCenter, angle, scale);

    double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
    double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;

    rotated_mat.at<double>(0, 2) += ex;
    rotated_mat.at<double>(1, 2) += ey;

    Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(128));
    warpAffine(faceROI, warped, rotated_mat, warped.size());
    equilizefaceHalves(warped);

    Mat filtered = Mat(warped.size(), CV_8U);
    bilateralFilter(warped, filtered, 0, 20.0, 2.0);

    Mat mask = Mat(warped.size(), CV_8U, Scalar(0));
    Point faceCenter = Point(cvRound(DESIRED_FACE_WIDTH * 0.5), cvRound(DESIRED_FACE_HEIGHT * 0.4));
    Size size = Size(cvRound(DESIRED_FACE_WIDTH * 0.5), cvRound(DESIRED_FACE_HEIGHT * 0.8));
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), FILLED);

    dest = Mat(warped.size(), CV_8U, Scalar(128));
    filtered.copyTo(dest, mask);
    return dest;
}