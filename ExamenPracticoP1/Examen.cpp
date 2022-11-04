//PRIMERA EVALUACIÓN-PARCIAL 1
//Alumno: Dueñas Jiménez Cristian Alexis
//Grupo: 5BM1
//Profesor: Sánchez García Octavio
//Materia: Visión Artificial

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>

//Declaracion de valores que usaremos para la formula.
#define e 2.7182
#define PI 3.1415

using namespace cv;
using namespace std;

//Declaracion de funciones
vector<vector<double>> kernel(int n, double sigma);
Mat padding(Mat& original, int n);
double getMaskValues(int sigma, int x, int y);
vector<Mat> sobelFilter(Mat imageFilter);
Mat getAngle(Mat imageFilter, Mat gx, Mat gy);
Mat nonMaximumSupression(Mat angle, Mat sobel);
Mat KanyeWest(Mat nms, float htp = 0, float ltv = 0);

//--------------------------------------------------------------------------------------------------------------------------

//Funcion en la que aplicamos el filtro Gaussiano.
Mat gaussianF(Mat pImage, int n, double sigma) {

	//Mandamos a llamar a nuestro Kernel.
	auto filter = kernel(n, sigma);
	int mid = (n - 1) / 2;

	//Creamos una matriz de ceros, quitando los bordes generados en la función de padding.
	Mat filtered = Mat::zeros(pImage.rows - (2 * mid), pImage.cols - (2 * mid), pImage.type());

	//Verificamos que el Kernel es correcto, así que lo imprimimos desde esta función.
	for (int i = 0; i < filter.size(); i++) {
		cout << "\n";
		for (int j = 0; j < filter[i].size(); j++) {
			cout << filter[i][j] << "\t";
		}
	}

	//Anidamos 4 fors, los 2 primeros recorren cada uno de los pixeles de nuestra imagen. Y los 2 últimos recorren el Kernel para poder aplicar el filtro.
	for (int i = mid; i < pImage.rows - mid; i++)
		for (int j = mid; j < pImage.cols - mid; j++)
			for (int k = -mid; k <= mid; k++)
				for (int l = -mid; l <= mid; l++)
					filtered.at<uchar>(i - mid, j - mid) += filter[k + mid][l + mid] * pImage.at<uchar>(i + k, j + l);

	return filtered; //Retornamos la matriz de la imagen con el filtro Gaussiano aplicado

}

//Funcion main, aqui mandamos a llamar al resto de nuestras funciones para realizar el proceso de cada una de ellas.
int main() {

	//Leemos la imagen.
	char imageName[] = "lena.png";
	Mat original;
	int n = 0; //Valor de nuestras filas y columnas del Kernel.
	int sigma = 0; //Valor de sigma.

	do {
		cout << "\nIngrese el valor para n (tamanio del Kernel): "; //Solicitamos la cantidad de filas y columnas de nuestro Kernel
		cin >> n;
		if (n % 2 == 0)
			printf("El valor debe de ser impar. Ingrese el dato nuevamente.");
	} while (n % 2 == 0);

	cout << "Ingrese el valor de sigma: "; //Solicitamos el valor de sigma.
	cin >> sigma;

	original = imread(imageName);
	if (!original.data) {
		cout << "Error al cargar la imagen: " << imageName << endl;
		exit(1);
	}

	int rows = original.rows;
	int cols = original.cols;

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", original);

	//Convertimos nuestra imagen a escala de grises y la mostramos. 
	Mat eg(rows, cols, CV_8UC1);
	cvtColor(original, eg, COLOR_BGR2GRAY);
	namedWindow("Escala de Grises", WINDOW_AUTOSIZE);
	imshow("Escala de Grises", eg);

	//Mostramos la imagen con los bordes cero agregados.
	Mat p = padding(eg, n);
	namedWindow("Bordes 0", WINDOW_AUTOSIZE);
	imshow("Bordes 0", p);

	//Aplicamos el filtro Gaussiano a nuestra imagen y la mostramos.
	Mat gF = gaussianF(p, n, sigma);
	namedWindow("Imagen Suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen Suavizada", gF);

	//Aplicamos Sobel, y mandamos a llamar la imagen generada final, la generada de Gx, Gy y del angulo.
	vector<Mat> sobElements = sobelFilter(gF);
	Mat ang = sobElements[0];
	Mat sF = sobElements[1];
	Mat gx = sobElements[2];
	Mat gy = sobElements[3];

	namedWindow("Filtro Sobel", WINDOW_AUTOSIZE);
	imshow("Filtro Sobel", sF);

	namedWindow("Sobel-ANGULO", WINDOW_AUTOSIZE);
	imshow("Sobel-ANGULO", ang);

	namedWindow("Gx", WINDOW_AUTOSIZE);
	imshow("Gx", gx);

	namedWindow("Gy", WINDOW_AUTOSIZE);
	imshow("Gy", gy);

	//Aplicamos non Maximum Supression a la imagen filtrada con Sobel.
	Mat nms_app = nonMaximumSupression(ang, sF);
	namedWindow("NMS", WINDOW_AUTOSIZE);
	imshow("NMS", nms_app);

	//Aplicamos la detección de bordes e imprimimos la imagen.
	Mat canny = KanyeWest(nms_app);
	imshow("Deteccion de bordes", canny);

	//Imprimimos los tamaños de cada una de nuestra imagen.
	cout << "\n\n\n*********Cantidad de filas para cada una de nuestras imagenes*********" << endl;
	cout << "\nImagen Original                Filas: " << rows << "     Columnas: " << cols;
	cout << "\nImagen Escala de Grises        Filas: " << eg.rows << "     Columnas: " << eg.cols;
	cout << "\nImagen con bordes cero         Filas: " << p.rows << "     Columnas: " << p.cols;
	cout << "\nImagen suavizada               Filas: " << gF.rows << "     Columnas: " << gF.cols;
	cout << "\nImagen con Sobel               Filas: " << sF.rows << "     Columnas: " << sF.cols;
	cout << "\nImagen detección de bordes     Filas: " << canny.rows << "     Columnas: " << canny.cols;
	cout << "\nImagen NMS                     Filas: " << nms_app.rows << "     Columnas: " << nms_app.cols;

	waitKey(0);
	return 0;
}

//Creamos nuestro Kernel, y lo almacenamos en un vector que almacena datos de tipo double.
vector<vector<double>> kernel(int n, double sigma) {

	vector<vector<double>> filter(n, vector<double>(n));
	int add = (n - 1) / 2; //Establecemos la cantidad de filas y columnas que habra hacia los lados y las partes superior e inferior de nuestra imagen.
	double sum = 0;

	for (int i = add; i >= -add; i--) {
		for (int j = -add; j <= add; j++) {
			filter[i + add][j + add] = getMaskValues(sigma, i, j);  //Para cada una de nuestras coordenadas, le mandamos los valores de i,j y el avlor de sigma para obtener los valores de la funcion.
			sum += filter[i + add][j + add];
		}
	}

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			filter[i][j] /= sum;

	return filter; //Mandamos el filtro
}

//Función que contiene la fórmula para el calculo de valores del Kernel
double getMaskValues(int sigma, int x, int y) {
	double division1 = (1 / (2 * PI * pow(sigma, 2)));
	double power = (-(pow(x, 2) + pow(y, 2)) / 2 * sigma);
	double result = division1 * (exp(power));
	return result;
}

//--------------------------------------------------------------------------------------------------------------------------

//Creamos nuestra imagen con bordes ceros, esto para facilitar la aplicacion del filtro.
Mat padding(Mat& original, int n) {

	double size;
	size = (n - 1) / 2;

	Mat pix = Mat::zeros(original.cols + size * 2, original.rows + size * 2, original.type());
	Mat roi;
	roi = pix(Rect(size, size, original.cols, original.rows));
	original.copyTo(roi);
	return pix;

}
//--------------------------------------------------------------------------------------------------------------------------

//Función para generar la imagen filtrada con Sobel
vector<Mat> sobelFilter(Mat imageFilter) {
	Mat GxMask = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //Máscara de Gx
	Mat GyMask = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1); //Máscara de Gy

	Mat temp(3, 3, CV_8UC1);
	Mat Gx = Mat::zeros(imageFilter.rows, imageFilter.cols, CV_32FC1);
	Mat Gy = Mat::zeros(imageFilter.rows, imageFilter.cols, CV_32FC1);
	Mat sobel = Mat::zeros(imageFilter.rows, imageFilter.cols, CV_8UC1);

	for (int i = 2; i < imageFilter.rows - 2; i++) {
		for (int j = 2; j < imageFilter.cols - 2; j++) {
			float t1 = 0;
			float t2 = 0;
			temp = Mat(imageFilter, Rect(j, i, 3, 3));
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					t1 += temp.at<uchar>(i, j) * GxMask.at<float>(i, j);
					t2 += temp.at<uchar>(i, j) * GyMask.at<float>(i, j);
				}
			}
			Gx.at<float>(i, j) = t1;
			Gy.at<float>(i, j) = t2;
		}
	}

	for (int i = 0; i < sobel.cols; i++) {
		for (int j = 0; j < sobel.cols; j++) {
			sobel.at<uchar>(i, j) = sqrt(pow((Gx.at<float>(i, j)), 2) + pow((Gy.at<float>(i, j)), 2));
		}
	}

	Mat ang = getAngle(imageFilter, Gx, Gy); //Le mandamos a la función del ángulo, la imagen de Gx y Gy

	vector<Mat> res = { ang, sobel, Gx, Gy };

	return res; //Retornamos la imagen filtrada
}

//Función para obtener el ángulo 
Mat getAngle(Mat imageFilter, Mat gx, Mat gy) {
	Mat angle = Mat::zeros(imageFilter.rows, imageFilter.cols, CV_32FC1);

	for (int i = 0; i < imageFilter.rows; i++) {
		for (int j = 0; j < imageFilter.cols; j++) {
			float valueY = gy.at<float>(i, j);
			float valueX = gx.at<float>(i, j);
			angle.at<float>(i, j) = atan2(valueY, valueX);
		}
	}

	return angle;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------

//Función que sirve para pasar por cada uno de los puntos de la matriz y encuentra los pixeles con el valor máximo en las direcciones de los bordes.
Mat nonMaximumSupression(Mat angle, Mat sobel) {

	Mat nms = Mat::zeros(sobel.rows, sobel.cols, CV_8UC1);

	for (int i = 1; i < sobel.rows; i++) {
		for (int j = 1; j < sobel.cols; j++) {

			int v1, v2;
			auto a = abs(angle.at<float>(i, j));
			//Angulo 0
			if ((0 <= a < 22.5) || (157.5 <= a <= 180)) {
				v1 = sobel.at<uchar>(i, j + 1);
				v2 = sobel.at<uchar>(i, j - 1);
				//Angulo 45
			}
			else if (22.5 <= a < 67.5) {
				v1 = sobel.at<uchar>(i + 1, j - 1);
				v2 = sobel.at<uchar>(i - 1, j + 1);
				//Angulo 90
			}
			else if (67.5 <= a < 112.5) {
				v1 = sobel.at<uchar>(i + 1, j);
				v2 = sobel.at<uchar>(i - 1, j);
				//Angulo 135
			}
			else if (112.5 <= a < 157.5) {
				v1 = sobel.at<uchar>(i - 1, j - 1);
				v2 = sobel.at<uchar>(i + 1, j + 1);
			}

			if (sobel.at<uchar>(i, j) >= v1 && sobel.at<uchar>(i, j) >= v2)
				nms.at<uchar>(i, j) = sobel.at<uchar>(i, j);
			else
				nms.at<uchar>(i, j) = 0;
		}
	}

	return nms;
}

//Función para obtener los máximos y mínimos, es decir 255 y 0 en nuestra imagen procesada con el non Maximum Supression.
pair<int, int> threshold(Mat nms) {
	int min = 255, max = 0;
	for (int i = 0; i < nms.rows; i++) {
		for (int j = 0; j < nms.cols; j++) {
			if (nms.at<uchar>(i, j) < min) {
				min = nms.at<uchar>(i, j);
			}
			else if (nms.at<uchar>(i, j) > max) {
				max = nms.at<uchar>(i, j);
			}
		}
	}
	return make_pair(max, min);
}

//Detectamos los bordes y los convertimos en 0 o 255 respectivamente, esto considerando los porcentaje brindados en la presentación.
Mat KanyeWest(Mat nms, float htp, float ltv) {

	htp = 0.9; //Asignamos el porcentaje para el valor alto
	ltv = 0.35; //Asignamos el porcentaje para el valor bajo
	Mat kanyeWest = Mat::zeros(nms.rows, nms.cols, CV_8UC1); //Matriz donde se imprimirá la detección de bordes.

	pair<int, int> mm = threshold(nms);
	int max = mm.first;
	int min = mm.second;

	float highThreshold = max * htp; //Obtenemos el valor, a partir del porcentaje de htp
	float lowThreshold = highThreshold * ltv; //Obtenemos el valor a partir del porcentaje de ltv

	int irrelevant = 0;
	int weak = lowThreshold;
	int strong = 255;

	cout << "Max: " << max << endl;
	cout << "Min: " << min << endl;


	for (int i = 1; i < nms.rows - 1; i++) {
		for (int j = 1; j < nms.cols; j++) {
			if (nms.at<uchar>(i, j) > lowThreshold && nms.at<uchar>(i, j) < highThreshold) {
				kanyeWest.at<uchar>(i, j) = weak;
			}
			else if (nms.at<uchar>(i, j) >= highThreshold) {
				kanyeWest.at<uchar>(i, j) = strong;
			}

		}
	}

	return kanyeWest;
}