#include <iostream>
#include <recognition/recognition.hpp>
double getSimilarity(const Mat face1, const Mat face2)
{
    if (face1.rows > 0 && face1.rows == face2.rows && face1.cols > 0 && face1.cols == face2.cols)
    {
        double errL2 = norm(face1, face2, NORM_L2);
        double similarity = errL2 / (double)(face1.rows * face2.cols);
        return similarity;
    }
    else
    {
        return 10000000000.0;
    }
}

void showTrainingDebugData(const Ptr<EigenFaceRecognizer> model, const int faceWidth, const int faceHeight)
{
    try
    {
        Mat averageFaceRow = model->getMean();
        Mat averageFace = getImgFro1DFMat(averageFaceRow, faceHeight);
        imshow("Average face", averageFace);
        Mat eiganVectors = model->getEigenVectors();

        for (int i = 0; i < min(20, eiganVectors.cols); i++)
        {
            Mat eiganVectorCol = eiganVectors.col(i).clone();
            Mat eiganFace = getImgFro1DFMat(eiganVectorCol, faceHeight);
            imshow(format("EiganFace%d", i), eiganFace);
        }
        Mat eiganValues = model->getEigenValues();
        vector<Mat> projections = model->getProjections();
        cout << "Projections..." << projections.size() << endl;
    }
    catch (Exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

Mat getImgFro1DFMat(const Mat mat_row, int h)
{
    Mat rectMat = mat_row.reshape(1, h);
    Mat dest;
    normalize(rectMat, dest, 0, 255, NORM_MINMAX, CV_8UC1);
    return dest;
}

Mat reconstructFace(const Ptr<EigenFaceRecognizer> model, const Mat preprocessed)
{
    try
    {
        Mat eiganVectors = model->getEigenVectors();
        Mat averageFaceRow = model->getMean();

        int faceHeight = preprocessed.rows;

        Mat projection = LDA::subspaceProject(eiganVectors, averageFaceRow, preprocessed.reshape(1, 1));
        Mat reconstructionRow = LDA::subspaceReconstruct(eiganVectors, averageFaceRow, projection);

        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        return reconstructedFace;
    }
    catch (Exception &e)
    {
        //cout << "err occured" << endl;
        //cerr << e.what() << '\n';
        return Mat();
    }
}