#ifndef _C3_ALGORITHM_H
#define _C3_ALGORITHM_H


#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

enum C3_ANGLE
{
	//�Ƕ�ѡ��
	C3_ALL,
	C3_0,
	C3_45,
	C3_90,
	C3_135
};

enum C3_ALGORITHM
{
	//�㷨ѡ��
	C31,
	C32,
	C33,
	C34,
	C35
};


class C3correlation
{
	//����һ��Э��ؾ���
	//ͨ����������������ֵ
	//ͨ��ѡ����㷨�õ�ê������ֵ
public:
	C3correlation(cv::Mat & correlation, int C3_algorithm);
	C3correlation() = default;
private:
	void getEigenValue();

public:
	cv::Mat correlation;
	int N; //Э��������С
	int c3algorithm;
	double coherence;
private:
	std::vector<double> eigenValue;
	void getCoherence();
};

class C3win
{
	//����һ��ʱ�������ݽǶ�������Э��ؾ���
	//ͨ��������ڵõ���ê���Ӧ�����ֵ
public:
	C3win(const std::vector<cv::Mat> & _win, int C3_angel, int C3_algorithm);
	C3win() = default;
private:
	void getLine();
	void _getLine(cv::Point x, cv::Point y);
	void getcorrelation();

public:
	const std::vector<cv::Mat> win;
	cv::Mat correlation;
	int row;
	int col;
	size_t layers; //ͨ����
	int c3angel;
	int c3algorithm;
	double coherence; //ÿ�����ڶ�Ӧһ�����ֵ
private:
	std::vector<cv::Mat> c3win;
	std::vector<std::vector<cv::Point> > winLine; //�����ĳ����������������ֱ���ϵĵ�
};

class C3coherence
{
public:
	C3coherence
	(
		const std::vector<cv::Mat> & src,
		int N,
		int C3_angel = C3_ALL,
		int C3_algorithm = C31,
		bool normal = true
	);
	C3coherence() = default;
	//û���ֶ�������ڴ棬����Ҫ�������������ͷ�
private:
	void init(); //��ʼ��
	void getCoherence();
public:
	const std::vector<cv::Mat> picData; //����Ķ�ͨ��ͼƬ
	cv::Mat output; //����õ���ɾ���
	int n;
	int row;
	int col;
	int layers;
	int c3angel;
	int c3algorithm;
	bool _normalize; //�Ƿ��׼��

private:
	std::vector<cv::Mat> _picData;
	cv::Mat coherence;
};

C3correlation::C3correlation
(
	cv::Mat & _correlation,
	int C3_algorithm
) : correlation(_correlation), c3algorithm(C3_algorithm)
{
	N = correlation.rows;
	getEigenValue(); //������ֵ
	getCoherence(); //�����
}



void C3correlation::getEigenValue() 
{
	//��Э��ؾ�������ֵ
	cv::Mat eValuesMat;
	cv::Mat eVectorsMat;
	cv::eigen(correlation, eValuesMat, eVectorsMat);
	for (int i = 0; i < N; i++) {
		eigenValue.push_back(eValuesMat.at<double>(i, 0));
	}
}

void C3correlation::getCoherence() 
{
	//���ݲ�ͬ�Ĺ�ʽ�������
	double deno = 0.0; //��ĸ
	double numer = 0.0; // ����
	switch (c3algorithm) {
	case C31:
		for (int i = 0; i < N; i++)
			deno += eigenValue[i];
		numer = eigenValue[0];
		break;
	case C32:
		deno = eigenValue[0] + eigenValue[2];
		numer = 2 * eigenValue[1] - eigenValue[0] - eigenValue[2];
		break;
	case C33:
		deno = (eigenValue[0] + eigenValue[1]) * (eigenValue[1] + eigenValue[2]);
		numer = 2 * eigenValue[1] * (eigenValue[1] - eigenValue[2]);
		break;
	case C34:
		deno = 2 * (eigenValue[0] + eigenValue[1] + eigenValue[2]);
		numer = 2 * eigenValue[0] - eigenValue[1] - eigenValue[2];
		break;
	case C35:
		deno = eigenValue[0];
		numer = eigenValue[0] - eigenValue[1];
		break;
	default:
		assert("algorithm error");
	}

	//��ĸ����0��C33���ֵС��0�����
	if (deno == 0) coherence = 0;
	else coherence = numer / deno;
	if (coherence < 0) coherence = 0 - coherence;
}



C3win::C3win
(
	const std::vector<cv::Mat> & _win,
	int C3_angel,
	int C3_algorithm
) : win(_win), c3angel(C3_angel), c3algorithm(C3_algorithm)
{
	layers = win.size();
	row = win[0].rows;
	col = win[0].cols;
	//���ݽǶ�ѡ��ȷ����N*N����N*1�Ĵ���
	switch (c3angel) {
	case C3_ALL:
		for (size_t k = 0; k < layers; k++)
			c3win.push_back(win[k]);
		break;
	default:
		getLine();
		size_t n = winLine[0].size();
		for (size_t i = 0; i < layers; i++)
		{
			cv::Mat temp(n, 1, CV_64FC1);
			for (int k = 0; k < n; k++)
				temp.at<double>(k, 0) = win[i].at<double>(winLine[i][k]);
			c3win.push_back(temp);
		}
	}

	correlation = cv::Mat::zeros(row, row, CV_64FC1);
	getcorrelation();
	C3correlation c(correlation, c3algorithm);
	coherence = c.coherence;
}

void C3win::_getLine(cv::Point x, cv::Point y)
{
	//�õ�N*1�����ϵ�ÿ����
	for (int k = 0; k < layers; k++)
	{
		cv::LineIterator lit(win[k], x, y, 8);
		std::vector<cv::Point> temp;
		for (int i = 0; i < lit.count; i++)
		{
			temp.push_back(cv::Point(lit.pos()));
			lit++;
		}
		winLine.push_back(temp);
	}
}

void C3win::getLine()
{
	switch (c3angel)
	{
	case C3_0:
		_getLine(cv::Point(0, 0), cv::Point(0, row - 1));
		break;
	case C3_45:
		_getLine(cv::Point(col - 1, 0), cv::Point(0, row - 1));
		break;
	case C3_90:
		_getLine(cv::Point(0, 0), cv::Point(col - 1, 0));
		break;
	case C3_135:
		_getLine(cv::Point(0, 0), cv::Point(col - 1, row - 1));
		break;
	default:
		assert("angel error");
	}
}

void C3win::getcorrelation()
{
	for (int k = 0; k < layers; k++)
	{
		//���ھ������ת�õõ�Э��ؾ���
		cv::Mat temp = cv::Mat::zeros(row, row, CV_64FC1);
		cv::Mat trans;
		transpose(c3win[k], trans);
		temp = c3win[k] * trans; // Mat �� * ����Ϊ����˷�
		correlation = correlation + temp;
	}
}


C3coherence::C3coherence
(
	const std::vector<cv::Mat> & src,
	int N,
	int C3_angel,
	int C3_algorithm,
	bool normal
) : picData(src), n(N), c3angel(C3_angel), c3algorithm(C3_algorithm), _normalize(normal)
{
	//��ʼ��
	layers = picData.size();
	row = picData[0].rows;
	col = picData[0].cols;
	init();
	getCoherence();

	if (_normalize)
		cv::normalize(coherence, coherence, 0, 1, cv::NORM_MINMAX);
	coherence.convertTo(output, CV_8UC1, 255);
}

void C3coherence::init()
{
	//������ʼ�����еĸ�������
	for (int i = 0; i < layers; i++)
	{
		cv::Mat temp;
		int ex = (n - 1) / 2;
		//����߽�
		cv::copyMakeBorder(picData[i], temp, ex, ex, ex, ex, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		temp.convertTo(temp, CV_64FC1, 1 / 255.0);
		_picData.push_back(temp);
	}
	coherence = cv::Mat::zeros(row, row, CV_64FC1);

}

void C3coherence::getCoherence()
{
	//ͨ������step_x step_y���д��ڵ��ƶ�
	//ÿ�����ڶ�Ӧһ��Э��ؾ���
	//ÿ�����ڶ�Ӧһ�����ֵ
	for (int step_x = 0; step_x < col; step_x++)
		for (int step_y = 0; step_y < row; step_y++)
		{
			std::vector<cv::Mat> win;
			for (int k = 0; k < layers; k++)
			{
				cv::Mat temp(_picData[k], cv::Range(step_x, step_x + n), cv::Range(step_y, step_y + n));
				win.push_back(temp);
			}
			C3win cwin(win, c3angel, c3algorithm);
			coherence.at<double>(step_x, step_y) = cwin.coherence;
			if (step_y == 0)
				std::cout << step_x << " / 1000" << std::endl;
		}

}

//������ȡͼƬ
template <class T>
std::string reName(std::string & name, T & s)
{
	std::stringstream temp;
	temp << s;
	return name + temp.str();
}


void readPictureName(std::vector<std::string> &picName, int num = 10, std::string heads = "20120717_", std::string ends = "_Bb.png")
{
	//��ö�ȡͼƬ��·��

	for (int i = 0; i < num; i++)
	{
		picName.push_back(reName(heads, i) + ends);
	}

}

void readPicture(const std::vector<std::string> &picName, std::vector<cv::Mat> & data)
{
	//��ͼƬ·���м���ͼƬ
	data.clear(); //���
	size_t num = picName.size();
	for (size_t i = 0; i < num; i++)
	{
		cv::Mat src = cv::imread(picName[i], 1);
		assert(src.data);
		cv::Mat tmp[3];
		cv::split(src, tmp);
		for (int j = 0; j < 3; j++)
		{
			data.push_back(tmp[j]);
		}
	}

}
//�ӿ�
//����Ψһ�Ľӿ�
//����ѡ��ʱ����С���Ƕȣ��㷨���Ƿ��׼��
void c3edge(
	const std::vector<cv::Mat> &src, //����
	cv::Mat & dst, //���
	int N = 3, //ʱ����С
	int C3_angel = C3_ALL, //�Ƕ�
	int C3_algorithm = C31, //�㷨ѡ��
	bool normal = true //��׼��
)
{
	using namespace cv;
	//û��Ԥ����
	C3coherence c(src, N, C3_angel, C3_algorithm, normal);
	c.output.copyTo(dst);
	if (C3_algorithm == C33) dst = 255 - dst;
	circle(dst, Point(500, 500), 500, Scalar(0, 0, 0), 1);

}

//Ԥ����
void pretreat(std::vector<cv::Mat> &src)
{
	int layers = src.size();
	for (int i = 0; i < layers; ++i)
	{
		cv::Mat temp;
		src[i].copyTo(temp);
		bilateralFilter(temp, src[i], 10, 50, 50);

	}
}
#endif
