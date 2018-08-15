#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void fitPlane(const Mat points,double* plane){
    int row=points.rows;
    int col=3;
    //求点云的质心
    Mat centroid=Mat(1,col,CV_64FC1,Scalar(0));
    for(int i=0;i<col;i++){
        for(int j=0;j<row;j++){
            centroid.at<double>(0,i)+=points.at<double>(j,i);
        }
        centroid.at<double>(0,i)/=row;
    }
    Mat points2=Mat(row,3,CV_64FC1);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            points2.at<double>(i,j)=points.at<double>(i,j)-centroid.at<double>(0,j);
        }
    }
    Mat A=Mat(3,3,CV_64FC1);
    Mat W=Mat(3,3,CV_64FC1);
    Mat V=Mat(3,3,CV_64FC1);
    Mat T=Mat(3,3,CV_64FC1);
    gemm(points2,points,1,Mat(),0,A,CV_GEMM_A_T);
    SVD::compute(A,W,T,V);
    plane[3]=0;
    for(int i=0;i<col;i++){
        plane[i]=V.at<double>(2,i);
        plane[3]+=plane[i]*centroid.at<double>(0,i);
    }
}

int main(){
    Mat points=Mat(3,3,CV_64FC1);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            points.at<double>(i,j)=i;
        }
    }
    double plane[4]={ 0 };
    fitPlane(points,plane);
    cout<<"The equation is :"<<endl;
    cout<<plane[0]<<"x + "<<plane[1]<<"y + "<<plane[2]<<"z = "<<plane[3]<<endl;
    cout<<"Normal vector of a plane is :"<<endl;
    cout<<"( "<<plane[0]<<" , "<<plane[1]<<" , "<<plane[2]<<" )"<<endl;
    return 0;
}