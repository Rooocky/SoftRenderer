#include <iostream>
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <tchar.h>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <ctime>
#include "stb_image.h"



#define PI 3.1415926535f

//==================================
//数据与运算
//==================================

//向量
typedef struct vector4
{
	float x;
	float y;
	float z;
	float w;
	vector4(float X, float Y, float Z, float W) {
		x = X;
		y = Y;
		z = Z;
		w = 0;
	}
	vector4(){};
}Vector4;
typedef struct vector3
{
	float x;
	float y;
	float z;
	vector3(float X, float Y, float Z ) {
		x = X;
		y = Y;
		z = Z;
	}
	vector3& operator *(float k) {
		Vector3 res;
		res.x = this->x * k;
		res.y = this->y * k;
		res.z = this->z * k;
		return res;
	}
	vector3& operator +(vector3 v) {
		Vector3 res;
		res.x = this->x + v.x;
		res.y = this->y + v.y;
		res.z = this->z + v.z;
		return res;
	}
	vector3& operator -(vector3 v) {
		Vector3 res;
		res.x = this->x - v.x;
		res.y = this->y - v.y;
		res.z = this->z - v.z;
		return res;
	}
	vector3() {};
}Vector3;
typedef struct vector2
{
	float x;
	float y;
	vector2(float X, float Y) {
		x = X;
		y = Y;
	}
}Vector2;


//点
typedef struct point
{
	float x;
	float y;
	float z;
	float w;
	point(float X, float Y, float Z, float W) {
		x = X;
		y = Y;
		z = Z;
		w = W;
	}
	point() {};
    bool operator==( point p) {
		if (x == p.x && y == p.y && z == p.z)
			return true;
		else
			return false;
	}

	Vector3 xyz() const { return Vector3(x, y, z); }
}Point;

//矩阵
typedef struct matrix
{
	float m[4][4];
	void set_unitmatrix(matrix& M) {
		M.m[0][0] = M.m[1][1] = M.m[2][2] = M.m[3][3]=1.0f;
		M.m[0][1] = M.m[0][2] = M.m[0][3] = 0.0f;
		M.m[1][0] = M.m[1][2] = M.m[1][3] = 0.0f;
		M.m[2][0] = M.m[2][1] = M.m[2][3] = 0.0f;
		M.m[3][0] = M.m[3][1] = M.m[3][2] = 0.0f;
	}
	void set_zeromatrix(matrix& M) {
		for(int i=0;i<4;++i)
			for (int j = 0; j < 4; ++j) {
				M.m[i][j] = 0.0f;
			}
	}
}Matrix;
typedef struct matrix3
{
	float m[3][3];
	void set_unitmatrix() {
		m[0][0] = m[1][1] = m[2][2] = 1.0f;
		m[0][1] = m[0][2] =  0.0f;
		m[1][0] = m[1][2] =  0.0f;
		m[2][0] = m[2][1] =  0.0f;
		
	}
	void set_zeromatrix(matrix& M) {
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) {
				M.m[i][j] = 0.0f;
			}
	}
}Matrix3;
//RGB颜色
typedef struct color {
	float r;
	float g;
	float b;
	//float a;
}Color;
typedef struct texcoord {
	float u;
	float v;
}Texcoord;
//顶点结构
typedef struct vertex
{
	Point position;
	Texcoord texcoord;
	Vector3 normal;
	Color color;//值控制在0.0f与1.0f之间
	float rhw;

}Vertex;

typedef struct Edge
{
	Vertex v, v1, v2; 
	Vector3 v_origin, v1_origin, v2_origin;
} edge;//v用于表示介于v1,v2组成的直线上的点，用于暂存插值计算结果
typedef struct trapezoid{ 
	float top, bottom; 
	edge left, right; 
	
} trapezoid_t;
typedef struct scanline{ Vertex v, step; int x, y, w; } scanline_t;


//三角mesh
typedef struct triangle
{
	Vertex p1, p2, p3;

}Triangle;
//线
typedef struct line
{
	Vertex p1, p2;
}Line;
//矢量加
Vector2 add(Vector2 v1, Vector2 v2) {
	
	return vector2(v1.x + v2.x, v1.y + v2.y);
}
Vector3 add(Vector3 v1, Vector3 v2) {
	return vector3(v1.x + v2.x, v1.y + v2.y,v1.z + v2.z);
}
Vector4 add(Vector4 v1, Vector4 v2) {
	return vector4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z,0.0f);
}
//矢量减
Vector2 sub(Vector2 v1, Vector2 v2) {
	return Vector2(v1.x - v2.x, v1.y - v2.y);
}
Vector3 sub(Vector3 v1, Vector3 v2) {
	return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
Vector4 sub(Vector4 v1, Vector4 v2) {
	
	return Vector4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0.0f);
}
//点积
float dot(Vector2 v1, Vector2 v2) {
	return v1.x* v2.x + v1.y * v2.y;
}
float dot(Vector3 v1, Vector3 v2) {
	return v1.x* v2.x + v1.y * v2.y + v1.z * v2.z;
}
float dot(Vector4 v1, Vector4 v2) {
	return v1.x* v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}
//向量取模
float vec_length(Vector2 v1) {
	return (float)sqrt(v1.x * v1.x + v1.y * v1.y);
}
float vec_length(Vector3 v1) {
	return (float)sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
}
float vec_length(Vector4 v1) {
	return (float)sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
}
//插值公式
float interp(float x1, float x2, float t) {
	return x1 * (1 - t) + x2 * t;
}
Vector2 vector_interp(Vector2 v1, Vector2 v2,float t) {
	return	Vector2((1 - t) * v1.x + t * v2.x, (1 - t) * v1.y + t * v2.y);
}
Vector3 vector_interp(Vector3 v1, Vector3 v2, float t) {
	return	Vector3((1 - t) * v1.x + t * v2.x, (1 - t) * v1.y + t * v2.y , (1 - t) * v1.z + t * v2.z);
}
Vector4 vector_interp(Vector4 v1, Vector4 v2, float t) {
	return	Vector4((1 - t) * v1.x + t * v2.x, (1 - t) * v1.y + t * v2.y , (1 - t) * v1.z + t * v2.z, 1.0f);
}
Point vector_interp(Point p1, Point p2, float t) {
	return	Point((1 - t) * p1.x + t * p2.x, (1 - t) * p1.y + t * p2.y, (1 - t) * p1.z + t * p2.z, 1.0f);
}
//向量单位化
Vector2 vec_normalize(Vector2 v1) {
	return Vector2(v1.x / vec_length(v1), v1.y / vec_length(v1));
}
Vector3 vec_normalize(Vector3 &v1) {
	float div = 1.0f / vec_length(v1);
	return Vector3(v1.x *div, v1.y *div, v1.z * div);
}
Vector4 vec_normalize(Vector4 v1) {
	return Vector4(v1.x / vec_length(v1), v1.y / vec_length(v1), v1.z / vec_length(v1),0.0f);
}
//矢量积
vector3 cross(Vector3 v1, Vector3 v2) {
	return Vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);

}
vector4 cross(Vector4 v1, Vector4 v2) {
	
	return Vector4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x,0.0f);
}
//矢量与矩阵的乘积
Vector4 vector_multi_matrix(Matrix M, Vector4 v) {
	Vector4 res(0,0,0,0);
	res.x = M.m[0][0] * v.x + M.m[1][0] * v.y + M.m[2][0] * v.z + M.m[3][0] * v.w;
	res.y = M.m[0][1] * v.x + M.m[1][1] * v.y + M.m[2][1] * v.z + M.m[3][1] * v.w;
	res.z = M.m[0][2] * v.x + M.m[1][2] * v.y + M.m[2][2] * v.z + M.m[3][2] * v.w;
	res.w = M.m[0][3] * v.x + M.m[1][3] * v.y + M.m[2][3] * v.z + M.m[3][3] * v.w;
	return res;
}
Vector3 vector_multi_matrix(Vector3 v, Matrix3 M ) {
	Vector3 res( 0, 0, 0);
	res.x = M.m[0][0] * v.x + M.m[1][0] * v.y + M.m[2][0] * v.z;
	res.y = M.m[0][1] * v.x + M.m[1][1] * v.y + M.m[2][1] * v.z;
	res.z = M.m[0][2] * v.x + M.m[1][2] * v.y + M.m[2][2] * v.z;
	return res;
}
point vector_multi_matrix(Matrix M, Point p) {
	point res;
	res.x = M.m[0][0] * p.x + M.m[1][0] * p.y + M.m[2][0] * p.z + M.m[3][0] * p.w;
	res.y = M.m[0][1] * p.x + M.m[1][1] * p.y + M.m[2][1] * p.z + M.m[3][1] * p.w;
	res.z = M.m[0][2] * p.x + M.m[1][2] * p.y + M.m[2][2] * p.z + M.m[3][2] * p.w;
	res.w = M.m[0][3] * p.x + M.m[1][3] * p.y + M.m[2][3] * p.z + M.m[3][3] * p.w;
	return res;
}
//矩阵与矩阵的乘积
Matrix matrix_multi_matrix(Matrix M1, Matrix M2) {
	Matrix res;
	for(int i=0;i<4;++i)
		for (int j = 0; j < 4; ++j) {
			res.m[i][j] = M1.m[i][0] * M2.m[0][j] + M1.m[i][1] * M2.m[1][j] + M1.m[i][2] * M2.m[2][j] + M1.m[i][3] * M2.m[3][j];
		}
	return res;

}
void swapvec(Vector3& v1, Vector3& v2)
{
	std::swap(v1.x, v2.x);
	std::swap(v1.y, v2.y);
	std::swap(v1.z, v2.z);
}
//==================================================================
//mesh类
//==================================================================
typedef struct mesh {
	Vertex p1, p2, p3;
	Vector3 normal;
	void get_mesh_normal() {
		Vector3 p1p2(p2.position.x - p1.position.x, p2.position.y - p1.position.y, p2.position.z - p1.position.z);
		Vector3 p1p3(p3.position.x - p1.position.x, p3.position.y - p1.position.y, p3.position.z - p1.position.z);
		normal=cross(p1p2, p1p3);
		normal = vec_normalize(normal);
	}
}Mesh;

//==================================================================
//相机类
//==================================================================
typedef struct camera {
	Vector3 pos;
	Vector3 lookat;
	Vector3 right;
	Vector3 up;
}Camera;
void update_camera_vector(Camera *camera, Matrix3 rotatex, Matrix3 rotatey) {
	camera->lookat = vector_multi_matrix( camera->lookat,rotatex);
	camera->up = vector_multi_matrix(camera->up, rotatex);
	camera->right = vector_multi_matrix(camera->right, rotatex);
		  
	camera->lookat = vector_multi_matrix(camera->lookat, rotatey);
	camera->up = vector_multi_matrix(camera->up, rotatey);
	camera->right = vector_multi_matrix(camera->right, rotatey);

}
//Camera camera;//唯一实例化，作为全局传递相机参数使用

//===================================================================
//光照系统
//===================================================================
class Light {
public:
	Vector3 pos;
	Color lightcol;
};
Light light;//光源实例化，控制全局光源



Vector3 negative(Vector3 v) {
	return Vector3(-v.x, -v.y, -v.z);
}//反射
Vector3 reflect(Vector3 N, Vector3 I) {//I的方向为片元到光源
	return N * 2.0f* dot(N, I) - I;
}
//blinn_phong 返回系数
void blinn_phong(Color  &color, Vector3 N, Vector3 I, Vector3 V, float shiness) {
	Vector3 H = vec_normalize(I + V);
	float Rs = pow(dot(N, H), shiness);
	color.r = light.lightcol.r * Rs;
	color.g = light.lightcol.g * Rs;
	color.b = light.lightcol.b * Rs;

}
//lambert 返回系数
void diffuse(Color *color,Vector3 N, Vector3 I) {
	I = vec_normalize(I);
	float k = dot(N, I);
	color->r = k * light.lightcol.r;
	color->g = k * light.lightcol.g;
	color->b = k * light.lightcol.b;
}
//Gouraud着色 双线性颜色插值
void Gouraud(Vector3 light_pos, Vertex v1, Vertex v2, Vertex v3) {

}


//=====================================
//坐标变换矩阵
//=====================================

//模型矩阵
//平移
Matrix get_translate_matrix(float a, float b, float c) {
	Matrix M;
	M.set_unitmatrix(M);
	M.m[3][0] = a;
	M.m[3][1] = b;
	M.m[3][2] = c;
	return M;
}
//缩放
Matrix get_scale_matrix(float s) {
	Matrix M;
	M.set_unitmatrix(M);
	M.m[0][0] = M.m[1][1] = M.m[2][2] = s;
	return M;
}
//旋转
Matrix get_rotate_matrix(Vector3 axis, float theta) {
	Matrix M;
	float S = (float)sin(theta);
	float C = (float)cos(theta);
	float x = axis.x;
	float y = axis.y;
	float z = axis.z;
	M.set_unitmatrix(M);
	M.m[0][0] = C + x * x * (1 - C);
	M.m[0][1] = x * y*(1 - C) + z * S;
	M.m[0][2] = x * z * (1 - C )- y * S;
	M.m[1][0] = x * y * (1 - C) - z * S;
	M.m[1][1] = C + y * y * (1 - C);
	M.m[1][2] = y * z * (1 - C) + x * S;
	M.m[2][0] = x * z * (1 - C) + y * S;
	M.m[2][1] = y * z * (1 - C) - x * S;
	M.m[2][2] = C + z * z * (1 - C);
	return M;
}
//绕right轴旋转
Matrix3 r_rotate_matrix(float theta, Vector3 right) {
	Matrix3 M;
	M.m[0][0] = right.x * right.x * (1 - cosf(theta)) + cosf(theta);
	M.m[0][1] = right.x * right.y * (1 - cosf(theta)) + right.z * sinf(theta);
	M.m[0][2] = right.x * right.z * (1 - cosf(theta)) - right.y * sinf(theta);
	M.m[1][0] = right.x * right.y * (1 - cosf(theta)) - right.z * sinf(theta);
	M.m[1][1] = right.y * right.y * (1 - cosf(theta)) + cos(theta);
	M.m[1][2] = right.y * right.z * (1 - cosf(theta)) + right.x * sinf(theta);
	M.m[2][0] = right.x * right.y * (1 - cosf(theta)) + right.y * sinf(theta);
	M.m[2][1] = right.y * right.z * (1 - cosf(theta)) - right.x * sinf(theta);
	M.m[2][2] = right.z * right.z * (1 - cosf(theta)) + cosf(theta);
	return M;
}
//绕up轴旋转
Matrix3 u_rotate_matrix(float theta,Vector3 up) {
	Matrix3 M;
	M.m[0][0] = up.x*up.x*(1-cosf(theta))+cosf(theta);
	M.m[0][1] = up.x * up.y*(1-cosf(theta))+up.z*sinf(theta);  
	M.m[0][2] = up.x*up.z*(1-cosf(theta))-up.y*sinf(theta);
	M.m[1][0] =up.x*up.y*(1-cosf(theta))-up.z*sinf(theta);       
	M.m[1][1] = up.y*up.y*(1-cosf(theta))+cos(theta); 
	M.m[1][2] = up.y*up.z*(1-cosf(theta))+up.x*sinf(theta);
	M.m[2][0] = up.x*up.y*(1-cosf(theta))+up.y*sinf(theta); 
	M.m[2][1] = up.y*up.z*(1-cosf(theta))-up.x*sinf(theta);  
	M.m[2][2] = up.z*up.z*(1-cosf(theta))+cosf(theta);
	return M;
}
//绕front旋转
Matrix3 l_rotate_matrix(float theta, Vector3 lookat) {
	Matrix3 M;
	M.m[0][0] = lookat.x * lookat.x * (1 - cosf(theta)) + cosf(theta);
	M.m[0][1] = lookat.x * lookat.y * (1 - cosf(theta)) + lookat.z * sinf(theta);
	M.m[0][2] = lookat.x * lookat.z * (1 - cosf(theta)) - lookat.y * sinf(theta);
	M.m[1][0] = lookat.x * lookat.y * (1 - cosf(theta)) - lookat.z * sinf(theta);
	M.m[1][1] = lookat.y * lookat.y * (1 - cosf(theta)) + cos(theta);
	M.m[1][2] = lookat.y * lookat.z * (1 - cosf(theta)) + lookat.x * sinf(theta);
	M.m[2][0] = lookat.x * lookat.y * (1 - cosf(theta)) + lookat.y * sinf(theta);
	M.m[2][1] = lookat.y * lookat.z * (1 - cosf(theta)) - lookat.x * sinf(theta);
	M.m[2][2] = lookat.z * lookat.z * (1 - cosf(theta)) + cosf(theta);
	return M;
}
//观察矩阵
Matrix get_view_matrix(Vector3 CamPosition, Vector3 up, Vector3 right, Vector3 lookat) {
	Matrix M;
	M.set_unitmatrix(M);
	lookat = vec_normalize(lookat);

	
	right = cross (up,lookat);

	right = vec_normalize(right);
	up = cross( lookat,right);
	up = vec_normalize(up);
	Vector3 P(CamPosition.x, CamPosition.y, CamPosition.z);
	float a = -dot(P, right);
	float b = -dot(P, up);
	float c = -dot(P, lookat);
	M.m[0][0] = right.x;
	M.m[1][0] = right.y;
	M.m[2][0] = right.z;
	M.m[3][0] = a;
	M.m[0][1] = up.x;
	M.m[1][1] = up.y;
	M.m[2][1] = up.z;
	M.m[3][1] = b;
	M.m[0][2] = lookat.x;
	M.m[1][2] = lookat.y;
	M.m[2][2] = lookat.z;
	M.m[3][2] = c;
	return M;
}
//投影矩阵
//D3DXMatrixPerspectiveFovLH  fovy:y方向视角，aspect:高/宽
Matrix get_projection_martix(float fovy,float zn,float zf,float aspect) {
	Matrix M;
	M.set_unitmatrix(M);
	float fax = 1.0f / (float)tan(fovy * 0.5f);
	
	M.m[0][0] = (float)(fax / aspect);
	M.m[1][1] = (float)(fax);
	M.m[2][2] = zf / (zf - zn);
	M.m[3][2] = -zn * zf / (zf - zn);
	M.m[2][3] = 1.0f;
	M.m[3][3] = 0.0f;
	return M;
}
//====================================================================
//transform
//====================================================================
typedef struct transform {
	Matrix model;
	Matrix view;
	Matrix projection;
	Matrix MVP;//model*view*ptrject
	float width, height;
}transform;
//计算MVP矩阵
void transform_update(transform* tr) {
	
	tr->MVP= matrix_multi_matrix(tr->model,tr->view);
	tr->MVP = matrix_multi_matrix(tr->MVP, tr->projection);
}
//transform初始化
void transform_init(transform* tr, int width, int height) {
	tr->model.set_unitmatrix(tr->model);
	/*Vector3 pos(0.0f, 0.0f, -5.0f);
	Vector3 up(0.0f, 1.0f, 0.0f);
	Vector3 right(1.0f, 0.0f, 0.0f);
	Vector3 look(0.0f, 0.0f, 1.0f);
	tr->view = get_view_matrix(pos,up,right,look);*/
	tr->view.set_unitmatrix(tr->view);
	float aspect = (float)width/(float)height  ;
	tr->projection = get_projection_martix(3.1415926f * 0.5, 1.0f, 500.0f, aspect);
	tr->width = width;
	tr->height = height;
	transform_update(tr);
}
//检查是否在cvv空间中，为裁剪做准备
int transform_check_cvv(const Point& v) {
	float w = v.w;
	int check = 0;
	if (v.z < 0.0f) check |= 1;
	if (v.z > w) check |= 2;
	if (v.x < -w) check |= 4;
	if (v.x > w) check |= 8;
	if (v.y < -w) check |= 16;
	if (v.y > w) check |= 32;
	return check;
}

// 归一化，得到屏幕坐标
void transform_homogenize(const transform* tr, Point& y, const Point x) {
	float rhw = 1.0f / x.w;
	y.x = (x.x * rhw + 1.0f) * tr->width * 0.5f;
	y.y = (1.0f - x.y * rhw) * tr->height * 0.5f;
	y.z = x.z * rhw;
	y.w = 1.0f;
}




//=====================================================================
// 渲染设备
//=====================================================================
typedef struct {
	transform transform;      // 坐标变换器
	int width;                  // 窗口宽度
	int height;                 // 窗口高度
	unsigned int** framebuffer;      // 像素缓存：framebuffer[y] 代表第 y行
	float** zbuffer;            // 深度缓存：zbuffer[y] 为第 y行指针
	unsigned int** texture;          // 纹理：同样是每行索引
	int tex_width;              // 纹理宽度
	int tex_height;             // 纹理高度
	float max_u;                // 纹理最大宽度：tex_width - 1（即纹理数组的索引最大值）
	float max_v;                // 纹理最大高度：tex_height - 1（同上）
	int render_state;           // 渲染状态
	unsigned int background;         // 背景颜色
	unsigned int foreground;         // 线框颜色
	Camera camera;
}	device_t;

#define RENDER_STATE_WIREFRAME      1		// 渲染线框
#define RENDER_STATE_TEXTURE        2		// 渲染纹理
#define RENDER_STATE_GOURAUD        4		// gouraud顶点光照模式
#define RENDER_STATE_COLOR          8

// 设备初始化，fb为外部帧缓存，非 NULL 将引用外部帧缓存（每行 4字节对齐）
void device_init(device_t* device, int width, int height, void* fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
	char* ptr = (char*)malloc(need + 64);
	char* framebuf, *zbuf;
	int j;
	assert(ptr);
	device->framebuffer = (unsigned int * *)ptr;
	device->zbuffer = (float**)(ptr + sizeof(void*) * height);
	ptr += sizeof(void*) * height * 2;
	device->texture = (unsigned int * *)ptr;
	ptr += sizeof(void*) * 1024;
	framebuf = (char*)ptr;
	zbuf = (char*)ptr + width * height * 4;
	ptr += width * height * 8;
	if (fb != NULL) framebuf = (char*)fb;//使用外部帧缓存
	for (j = 0; j < height; j++) {
		device->framebuffer[j] = (unsigned int*)(framebuf + width * 4 * j);//为帧缓冲及zbuffer中的每一行设置起始位置
		device->zbuffer[j] = (float*)(zbuf + width * 4 * j);
	}
	device->texture[0] = (unsigned int*)ptr;
	device->texture[1] = (unsigned int*)(ptr + 16);//2张纹理？
	memset(device->texture[0], 0, 64);
	device->tex_width = 2;
	device->tex_height = 2;
	device->max_u = 1.0f;
	device->max_v = 1.0f;
	device->width = width;
	device->height = height;
	device->background = 0x808080;
	device->foreground = 0;
	transform_init(&device->transform, width, height);
	device->render_state = RENDER_STATE_WIREFRAME;
}

// 删除设备时，至空各种指针，释放动态内存
void device_destroy(device_t * device) {
	if (device->framebuffer)
		free(device->framebuffer);
	device->framebuffer = NULL;
	device->zbuffer = NULL;
	device->texture = NULL;
}

// 设置当前纹理 pitch:纹理的步长 等于每行纹理的数据大小
void device_set_texture(device_t * device, void* bits, long pitch, int w, int h) {
	char* ptr = (char*)bits;
	int j;
	assert(w <= 1024 && h <= 1024);
	for (j = 0; j < h; ptr += pitch, j++) 	// 重新计算每行纹理的指针
		device->texture[j] = (unsigned int*)ptr;
	device->tex_width = w;
	device->tex_height = h;
	device->max_u = (float)(w - 1);
	device->max_v = (float)(h - 1);
}

// 清空 framebuffer 和 zbuffer ***与删除设备的区别为缓存清空后还可继续写入，删除设备则只能重新初始化***
void device_clear(device_t * device) {
	int y, x, height = device->height;
	//clear颜色缓存
	for (y = 0; y < device->height; y++) {
		unsigned int* dst = device->framebuffer[y];
		
		unsigned int cc = device->background;//设置为背景颜色（灰）
		for (x = device->width; x > 0; dst++, x--) 
			dst[0] = cc;
	}
	//清空zbuffer每个元素置为0
	for (y = 0; y < device->height; y++) {
		float* dst = device->zbuffer[y];
		for (x = device->width; x > 0; dst++, x--) 
			dst[0] = 0.0f;
	}
}

//=====================================================================
// 几何计算：顶点、扫描线、边缘、矩形、步长计算
//=====================================================================



//用于初始化透视修正的纹理插值坐标，及1/z缓冲
void vertex_rhw_init(Vertex* v) {
	float rhw = 1.0f / v->position.w;
	v->rhw = rhw;
	v->texcoord.u *= rhw;
	v->texcoord.v *= rhw;
	v->color.r *= rhw;
	v->color.g *= rhw;
	v->color.b *= rhw;
}

void vertex_interp(Vertex* y, const Vertex* x1, const Vertex* x2, float t) {
	y->position=vector_interp(x1->position, x2->position, t);
	y->texcoord.u = interp(x1->texcoord.u, x2->texcoord.u, t);
	y->texcoord.v = interp(x1->texcoord.v, x2->texcoord.v, t);
	y->color.r = interp(x1->color.r, x2->color.r, t);
	y->color.g = interp(x1->color.g, x2->color.g, t);
	y->color.b = interp(x1->color.b, x2->color.b, t);
	y->rhw = interp(x1->rhw, x2->rhw, t);
}
void color_interp(Vertex* y, const Vertex* x1, const Vertex* x2, float t) {
	
	y->color.r = interp(x1->color.r, x2->color.r, t);
	y->color.g = interp(x1->color.g, x2->color.g, t);
	y->color.b = interp(x1->color.b, x2->color.b, t);
	
}
void vertex_division(Vertex* y, const Vertex* x1, const Vertex* x2, float w) {
	float inv = 1.0f / w;
	y->position.x = (x2->position.x - x1->position.x) * inv;
	y->position.y = (x2->position.y - x1->position.y) * inv;
	y->position.z = (x2->position.z - x1->position.z) * inv;
	y->position.w = (x2->position.w - x1->position.w) * inv;
	y->texcoord.u = (x2->texcoord.u - x1->texcoord.u) * inv;
	y->texcoord.v = (x2->texcoord.v - x1->texcoord.v) * inv;
	y->color.r = (x2->color.r - x1->color.r) * inv;
	y->color.g = (x2->color.g - x1->color.g) * inv;
	y->color.b = (x2->color.b - x1->color.b) * inv;
	y->rhw = (x2->rhw - x1->rhw) * inv;
}

void vertex_add(Vertex * y, const Vertex * x) {
	y->position.x += x->position.x;
	y->position.y += x->position.y;
	y->position.z += x->position.z;
	y->position.w += x->position.w;
	y->rhw += x->rhw;
	y->texcoord.u += x->texcoord.u;
	y->texcoord.v += x->texcoord.v;
	y->color.r += x->color.r;
	y->color.g += x->color.g;
	y->color.b += x->color.b;
}

// 根据三角形生成 0-2 个梯形，并且返回合法梯形的数量
int trapezoid_init_triangle(trapezoid_t * trap, const Vertex * p1,
	const Vertex * p2, const Vertex * p3,bool &left_or_right, Vector3 v1, Vector3 v2, Vector3 v3) {
	const Vertex* p;
	float k, x;

	//检查三角形三个点的高度，确保y值p3>p2>p1
	if (p1->position.y > p2->position.y)
	{
		p = p1, p1 = p2, p2 = p;
		swapvec(v1, v2);
	}
	if (p1->position.y > p3->position.y)
	{
		p = p1, p1 = p3, p3 = p;
		swapvec(v1, v3);
	}
	if (p2->position.y > p3->position.y)
	{
		p = p2, p2 = p3, p3 = p;
		swapvec(v2, v3);
	}
	if (p1->position.y == p2->position.y && p1->position.y == p3->position.y) return 0;//水平共线返回0
	if (p1->position.x == p2->position.x && p1->position.x == p3->position.x) return 0;//垂直共线返回0

	if (p1->position.y == p2->position.y) {	//平底三角形
		if (p1->position.x > p2->position.x)
		{
			p = p1, p1 = p2, p2 = p;//p2.x>p1.x
			swapvec(v1, v2);
		}
		trap[0].top = p1->position.y;//以y值较小的点为top，扫描线自顶向下扫描。
		trap[0].bottom = p3->position.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;//每个边的值v2.y>v1.y
		trap[0].left.v1_origin = v1;
		trap[0].left.v2_origin = v3;
		trap[0].right.v1 = *p2;
		trap[0].right.v2 = *p3;
		trap[0].right.v1_origin = v2;
		trap[0].right.v2_origin = v3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	if (p2->position.y == p3->position.y) {	//平顶三角形
		if (p2->position.x > p3->position.x)
		{
			p = p2, p2 = p3, p3 = p;
			swapvec(v2, v3);
		}
		trap[0].top = p1->position.y;
		trap[0].bottom = p3->position.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].left.v1_origin = v1;
		trap[0].left.v2_origin = v2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		trap[0].right.v1_origin = v1;
		trap[0].right.v2_origin = v3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	trap[0].top = p1->position.y;
	trap[0].bottom = p2->position.y;
	trap[1].top = p2->position.y;
	trap[1].bottom = p3->position.y;

	k = (p3->position.y - p1->position.y) / (p2->position.y - p1->position.y);
	x = p1->position.x + (p2->position.x - p1->position.x) * k;//*k?
    
	left_or_right = (x <= p3->position.x);//判断三角形左偏（true）还是右偏
	if (left_or_right) {		// p1.x>p2.x时
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].left.v1_origin = v1;
		trap[0].left.v2_origin = v2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		trap[0].right.v1_origin = v1;
		trap[0].right.v2_origin = v3;
		trap[1].left.v1 = *p2;
		trap[1].left.v2 = *p3;
		trap[1].left.v1_origin = v2;
		trap[1].left.v2_origin = v3;
		trap[1].right.v1 = *p1;
		trap[1].right.v2 = *p3;
		trap[1].right.v1_origin = v1;
		trap[1].right.v2_origin = v3;
	}
	else {					// p1.x<p2.x时
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].left.v1_origin = v1;
		trap[0].left.v2_origin = v3;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p2;
		trap[0].right.v1_origin = v1;
		trap[0].right.v2_origin = v2;
		trap[1].left.v1 = *p1;
		trap[1].left.v2 = *p3;
		trap[1].left.v1_origin = v1;
		trap[1].left.v2_origin = v3;
		trap[1].right.v1 = *p2;
		trap[1].right.v2 = *p3;
		trap[1].right.v1_origin = v2;
		trap[1].right.v2_origin = v3;
	}

	return 2;
}

// 按照 Y 坐标计算出左右两条边纵坐标等于 Y 的顶点
void trapezoid_edge_interp(trapezoid_t * trap, float y) {
	float s1 = trap->left.v2.position.y - trap->left.v1.position.y;
	float s2 = trap->right.v2.position.y - trap->right.v1.position.y;
	float t1 = (y - trap->left.v1.position.y) / s1;
	float t2 = (y - trap->right.v1.position.y) / s2;
	vertex_interp(&trap->left.v, &trap->left.v1, &trap->left.v2, t1);
	vertex_interp(&trap->right.v, &trap->right.v1, &trap->right.v2, t2);
}

// 根据左右两边的端点，初始化计算出扫描线的起点和步长
void trapezoid_init_scan_line(const trapezoid_t * trap, scanline_t * scanline, int y) {
	float width = trap->right.v.position.x - trap->left.v.position.x;
	scanline->x = (int)(trap->left.v.position.x + 0.5f);
	scanline->w = (int)(trap->right.v.position.x + 0.5f) - scanline->x;
	scanline->y = y;
	scanline->v = trap->left.v;
	if (trap->left.v.position.x >= trap->right.v.position.x) scanline->w = 0;
	vertex_division(&scanline->step, &trap->left.v, &trap->right.v, width);
}

//=====================================================================
//光栅化/区域填充
//=====================================================================
// 绘制扫描线
float clamp(float res, float min, float max) {
	if (res < min)
		return min;
	else if (res > max)
		return max;
	else
		return res;
}
Vector3 showPosColor(Vector3 pos)
{
	return Vector3((pos.x + 1) * 128.f
		, (pos.y + 1) * 128.f
		, (pos.z + 1) * 128.f)
		;
}
int CMID(int x, int min, int max);
unsigned int device_texture_read(const device_t* device, float u, float v);
void device_draw_scanline(device_t* device, scanline_t* scanline,trapezoid_t* trap,Mesh mesh, int trapsID,bool l_or_r) {
	
	unsigned int* framebuffer = device->framebuffer[scanline->y];
	float* zbuffer = device->zbuffer[scanline->y];
	int x = scanline->x;
	int w = scanline->w;
	int width = device->width;
	int render_state = device->render_state;
	//std::cout << light.pos.y << std::endl;

	
	

//===========================================
	for (; w > 0; x++, w--) {
		if (x >= 0 && x < width) {
			float rhw = scanline->v.rhw;
			float z = scanline->v.position.z;
			if (rhw >= zbuffer[x]&&z>=0.0f&&z<0.9f) {
				//std::cout << scanline->v.rhw << std::endl;
				float w = 1.0f / rhw;
				zbuffer[x] = rhw;
				
				
				if (render_state == RENDER_STATE_TEXTURE) {
					float u = scanline->v.texcoord.u * w;
					float v = scanline->v.texcoord.v * w;
					unsigned int cc = device_texture_read(device, u, v);
					//纹理值转换为float
					float uv_r = (float)(cc >> 16) / (float)255;
					float uv_g = (float)((cc & 0x00FF00) >> 8) / (float)255;
					float uv_b = (float)(cc & 0x0000FF) / (float)255;
					//==============================
					//逐像素计算光照
					//==============================
					//mesh.get_mesh_normal();
					Vector3 lightdir;
					//插值计算出像素点坐标，从而计算入射向量
					float s1 = trap->left.v2.position.y - trap->left.v1.position.y;
					float s2 = trap->right.v2.position.y - trap->right.v1.position.y;
					float t1 = (scanline->y - trap->left.v1.position.y) / s1;
					float t2 = (scanline->y - trap->right.v1.position.y) / s2;
					
					//Vector3 world_v1(mesh.p1.position.x, mesh.p1.position.y, mesh.p1.position.z);
					//Vector3 world_v2(mesh.p2.position.x, mesh.p2.position.y, mesh.p2.position.z);
					//Vector3 world_v3(mesh.p3.position.x, mesh.p3.position.y, mesh.p3.position.z);
					
					Vector3 frag1 = vector_interp(trap->left.v1_origin, trap->left.v2_origin, t1);
					Vector3 frag2 = vector_interp(trap->right.v1_origin, trap->right.v2_origin, t2);
					
					/*if (l_or_r) {
						if (trapsID==0) {
							frag1 = vector_interp(world_v1, world_v2, t1);
							frag2 = vector_interp(world_v1, world_v3, t2);
						}
						else {
							frag1 = vector_interp(world_v2, world_v3, t1);
							frag2 = vector_interp(world_v1, world_v3, t2);
						}
					}
					else  {
						
						if (trapsID==0) {
							frag1 = vector_interp(world_v1, world_v3, t1);
							frag2 = vector_interp(world_v1, world_v2, t2);
						}
						else {
							frag1 = vector_interp(world_v1, world_v3, t1);
							frag2 = vector_interp(world_v2, world_v3, t2);
						}
					}*/

					//if (l_or_r) {
					//	if (trapsID == 0) {
					//		frag1 = vector_interp(world_v1, world_v2, t1);
					//		frag2 = vector_interp(world_v1, world_v3, t2);
					//	}
					//	else {
					//		frag1 = vector_interp(world_v2, world_v3, t1);
					//		frag2 = vector_interp(world_v1, world_v3, t2);
					//	}
					//}
					//else {
					//	
					//	if (trapsID == 0) {
					//		frag1 = vector_interp(world_v1, world_v2, t1);
					//		frag2 = vector_interp(world_v1, world_v3, t2);
					//	}
					//	else {
					//		frag1 = vector_interp(world_v1, world_v2, t1);
					//		frag2 = vector_interp(world_v3, world_v2, t2);
					//	}
					//}
					
					
					
					float t3 = (float)(x - scanline->x) / (float)scanline->w;
					Vector3 frag=vector_interp(frag1, frag2, t3);
					
					light.pos = Vector3(3.5f,4.5f, 2.0f);
					lightdir = light.pos - frag;
					//lightdir = Vector3(0.0f, 2.0f, 5.0f);
					lightdir = vec_normalize(lightdir);
			
					//float k = fabs(dot(mesh.normal, lightdir))*0.8f;
					float k = max(0.0f,dot(mesh.normal, lightdir)) * 0.8f;

					k = k * k;
					//std::cout << k << std::endl;
					//float转整数
					light.lightcol = { 0.0f, 0.3f, 0.3f };

					//unsigned int c_r = 255 * light.lightcol.r * k+255* 0.2 *uv_r;
					//unsigned int c_g = 255 * light.lightcol.g * k+255* 0.2 *uv_g;
					//unsigned int c_b = 255 * light.lightcol.b * k+255* 0.2 *uv_b;

					unsigned int c_r = (float)255*uv_r * k+(float)0.25*255*uv_r;
					unsigned int c_g = (float)255*uv_g * k+(float)0.25*255*uv_g;
					unsigned int c_b = (float)255*uv_b * k+(float)0.25*255*uv_b;


					// jj
					//int c_r = clamp(t3, 0, 1) * 255;//frag2.x * 80;
					//int c_g = clamp(t3, 0, 1) * 255;//frag2.y * 80;
					//int c_b = clamp(t3, 0, 1) * 255;//frag2.z * 80;
					
					//Vector3 color = showPosColor(frag);
					//int c_r = color.x;
					//int c_g = color.y;
					//int c_b = color.z;

					//int c_r = clamp((trapsID) * 100, 0, 255);
					//int c_g = clamp((trapsID) * 100, 0, 255);
					//int c_b = clamp((trapsID) * 100, 0, 255);

					c_r = c_r<0? 0 : (c_r > 255 ? 255 : c_r);
					c_g = c_g<0? 0 : (c_g  > 255 ? 255 : c_g);
					c_b = c_b<0? 0 : (c_b  > 255 ? 255 : c_b);


					framebuffer[x] = (c_r << 16) | (c_g << 8) | (c_b);
					//std::cout << mesh.p1.position.x << mesh.p1.position.y << mesh.p1.position.z << std::endl;
					//std::cout << lightdir.x<<lightdir.y<<lightdir.z << std::endl;
				}
			}
		}
 		vertex_add(&scanline->v, &scanline->step);
		if (x >= width) break;
	}
}

// 主渲染函数
void device_render_trap(device_t * device, trapezoid_t * trap,Mesh mesh, int trapsID,bool l_or_r) {
	scanline_t scanline;
	int j, top, bottom;
	top = (int)(trap->top + 0.5f);
	bottom = (int)(trap->bottom + 0.5f);
	for (j = top; j < bottom; j++) {//左闭右开 避免光栅化重绘
		if (j >= 0 && j < device->height) {
			trapezoid_edge_interp(trap, (float)j + 0.5f);//对边插值计算出y=j这条扫描线对应顶点的各种属性存放在edge::v中
			trapezoid_init_scan_line(trap, &scanline, j);//确定扫描线两边端点，步长，宽度
			device_draw_scanline(device, &scanline,trap,mesh,trapsID,l_or_r);//绘制扫描线并在绘制的同时进行zbuffer消隐
		}
		if (j >= device->height) break;
	}
}

//=======================================================
//点、线、三角、平面、正方体绘制及纹理映射算法
//=======================================================
// 画点
void draw_pixel(device_t * device, int x, int y, unsigned int color) {
	if (((unsigned int)x) < (unsigned int)device->width && ((unsigned int)y) < (unsigned int)device->height) {
		device->framebuffer[y][x] = color;
	}
}

// 绘制线段 基于Bresenham算法
void draw_line(device_t * device, int x1, int y1, int x2, int y2, unsigned int color) {
	int x, y, rem = 0;
	if (x1 == x2 && y1 == y2) {//特殊情况
		draw_pixel(device, x1, y1, color);
	}
	else if (x1 == x2) {
		int inc = (y1 <= y2) ? 1 : -1;
		for (y = y1; y != y2; y += inc) 
			draw_pixel(device, x1, y, color);
		draw_pixel(device, x2, y2, color);
	}
	else if (y1 == y2) {
		int inc = (x1 <= x2) ? 1 : -1;
		for (x = x1; x != x2; x += inc) 
			draw_pixel(device, x, y1, color);
		draw_pixel(device, x2, y2, color);
	}
	else {//一般情况
		int dx = (x1 < x2) ? x2 - x1 : x1 - x2;
		int dy = (y1 < y2) ? y2 - y1 : y1 - y2;
		if (dx >= dy) {
			if (x2 < x1) 
				x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y;//交换x1y1与x2y2保证x2>x1
			for (x = x1, y = y1; x <= x2; x++) {
				rem += dy;
				if (rem >= dx) {
					rem -= dx;
					y += (y2 >= y1) ? 1 : -1;
					draw_pixel(device, x, y, color);
					continue;
				}
				draw_pixel(device, x, y, color);
			}
			draw_pixel(device, x2, y2, color);
		}
		else {
			if (y2 < y1) 
				x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y;//交换x1y1与x2y2保证y2>y1
			for (x = x1, y = y1; y <= y2; y++) {
				rem += dx;
				if (rem >= dy) {
					rem -= dy;
					x += (x2 >= x1) ? 1 : -1;
					draw_pixel(device, x, y, color);
					continue;
				}
				draw_pixel(device, x, y, color);
			}
			draw_pixel(device, x2, y2, color);
		}
	}
}
//画三角形

void draw_triangle(device_t* device, const vertex* v1, const vertex* v2, const vertex* v3,Point wp1,Point wp2,Point wp3) {
	Point p1, p2, p3, c1, c2, c3;
	Mesh mesh;
	mesh.p1 = *v1;
	mesh.p2 = *v2;
	mesh.p3 = *v3;
	mesh.get_mesh_normal();
	/*mesh.p1.position = wp1;
	mesh.p2.position = wp2;
	mesh.p3.position = wp3;*/
	Vector3 w_v1 = wp2.xyz() - wp1.xyz();
	Vector3 w_v2 = wp3.xyz() - wp1.xyz();
	Vector3 normal = cross(w_v1,w_v2);
	Vector3 cameradir = device->camera.pos - wp1.xyz();
	float d = dot(normal, cameradir);
	//std::cout << d << std::endl;
	if (d<0) {
		//std::cout << "undrawn" << std::endl;
		return;
	}
		
	//将顶点按规定顺序排布
	//if (m_p1.y > m_p3.y) { temp = mesh.p3; mesh.p3 = mesh.p1; mesh.p1 = temp; }
	//if (m_p1.y > m_p2.y) { temp = mesh.p2; mesh.p2 = mesh.p1; mesh.p1 = temp; }
	//if (m_p2.y > m_p3.y) { temp = mesh.p2; mesh.p2 = mesh.p3; mesh.p3 = temp; }
	//if (m_p2.x > m_p3.x) { temp = mesh.p2; mesh.p2 = mesh.p3; mesh.p3 = temp; }

	int render_state = device->render_state;

	// 按照 Transform 变化
	
	c1 = vector_multi_matrix(device->transform.MVP, v1->position);
	c2 = vector_multi_matrix(device->transform.MVP, v2->position);
	c3 = vector_multi_matrix(device->transform.MVP, v3->position);
	
	// 裁剪，注意此处可以完善为具体判断几个点在 cvv内以及同cvv相交平面的坐标比例
	// 进行进一步精细裁剪，将一个分解为几个完全处在 cvv内的三角形
	// 此处的算法仅进行了最简单的裁剪剔除，即当三角形只要有一个点不在裁剪空间内则放弃该三角形的绘制
	// TODO：多边形裁剪
	std::vector<Vertex> renderlist = {*v1,*v2,*v3};

	
	int check_v1 = transform_check_cvv(c1);
	int check_v2 = transform_check_cvv(c2);
	int check_v3 = transform_check_cvv(c3);
	
	//std::vector<Vertex> newlist;
	if (check_v1 != 0 && check_v2!=0 && check_v3!=0) return;
	//if (check_v1 & 12 != 0 || check_v1 & 12 != 0 || check_v1 & 12 != 0) return;
	/*if (check_v1 == 0 && check_v2 == 0)
		newlist.push_back(*v1);*/
	/*else if (check_v1 == 0 && check_v2 != 0) {
		newlist.push_back(*v1);
		Vertex v11;
		float t;
		if (check_v2 < 4) {
			if(v1->position.z>v2->position.z)
				t = fabs(v1->position.z - w) / fabs(v2->position.z);
		}
		vertex_interp(&v11, v1, v2,t);
	}
*/

	// 归一化，从cvv空间中的坐标直接转换到屏幕
	transform_homogenize(&device->transform, p1, c1);
	transform_homogenize(&device->transform, p2, c2);
	transform_homogenize(&device->transform, p3, c3);
	//std::cout << p1.x << p2.y << p1.z << std::endl;
	// 判断进入纹理或者色彩绘制
	if (render_state == RENDER_STATE_TEXTURE || render_state==RENDER_STATE_GOURAUD|| render_state == RENDER_STATE_COLOR) {
		vertex t1 = *v1, t2 = *v2, t3 = *v3;
		trapezoid_t traps[2];
		int n;
		

		t1.position = p1;
		t2.position = p2;
		t3.position = p3;
		t1.position.w = c1.w;
		t2.position.w = c2.w;
		t3.position.w = c3.w;

		vertex_rhw_init(&t1);	// 初始化 w
		vertex_rhw_init(&t2);	// 初始化 w
		vertex_rhw_init(&t3);	// 初始化 w

		// 拆分三角形为0-2个梯形，并且返回可用梯形数量
		bool l_or_r;
		//n = trapezoid_init_triangle(traps, &t1, &t2, &t3, l_or_r, v1->position.xyz(), v2->position.xyz(), v3->position.xyz());
		n = trapezoid_init_triangle(traps, &t1, &t2, &t3, l_or_r, wp1.xyz(), wp2.xyz(), wp3.xyz());

		if (n >= 1) device_render_trap(device, &traps[0],mesh,0,l_or_r);
		if (n >= 2) device_render_trap(device, &traps[1],mesh,1, l_or_r);
	}
	//判断进入线框绘制
	if (render_state == RENDER_STATE_WIREFRAME) {		
		draw_line(device, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, device->foreground);
		draw_line(device, (int)p1.x, (int)p1.y, (int)p3.x, (int)p3.y, device->foreground);
		draw_line(device, (int)p3.x, (int)p3.y, (int)p2.x, (int)p2.y, device->foreground);
	}
}

void draw_triangle(device_t* device, const vertex* v1, const vertex* v2, const vertex* v3) {
	Point p1, p2, p3, c1, c2, c3;
	Mesh mesh;
	mesh.p1 = *v1;
	mesh.p2 = *v2;
	mesh.p3 = *v3;
	Point m_p1 = mesh.p1.position;
	Point m_p2 = mesh.p2.position;
	Point m_p3 = mesh.p3.position;
	Vertex temp;
	//将顶点按规定顺序排布
	//if (m_p1.y > m_p3.y) { temp = mesh.p3; mesh.p3 = mesh.p1; mesh.p1 = temp; }
	//if (m_p1.y > m_p2.y) { temp = mesh.p2; mesh.p2 = mesh.p1; mesh.p1 = temp; }
	//if (m_p2.y > m_p3.y) { temp = mesh.p2; mesh.p2 = mesh.p3; mesh.p3 = temp; }
	//if (m_p2.x > m_p3.x) { temp = mesh.p2; mesh.p2 = mesh.p3; mesh.p3 = temp; }

	int render_state = device->render_state;

	// 按照 Transform 变化

	c1 = vector_multi_matrix(device->transform.MVP, v1->position);
	c2 = vector_multi_matrix(device->transform.MVP, v2->position);
	c3 = vector_multi_matrix(device->transform.MVP, v3->position);

	// 裁剪，注意此处可以完善为具体判断几个点在 cvv内以及同cvv相交平面的坐标比例
	// 进行进一步精细裁剪，将一个分解为几个完全处在 cvv内的三角形
	// 此处的算法仅进行了最简单的裁剪剔除，即当三角形只要有一个点不在裁剪空间内则放弃该三角形的绘制
	// TODO：多边形裁剪
	std::vector<Vertex> renderlist = { *v1,*v2,*v3 };

	//int bit=7;
	/*if (transform_check_cvv(c1) != 0) bit=bit&6;
	if (transform_check_cvv(c2) != 0) bit=bit&5;
	if (transform_check_cvv(c3) != 0) bit=bit&3;*/
	//if (bit == 0)
	//	return;
	//else if(bit == 1 || bit == 2 || bit == 4) {
	//	int index = 0;
	//	while(bit) {
	//		index++;
	//		bit >> 1;
	//	}
	//	vertex_interp(&renderlist[index], &renderlist[index],)
	//}
	int check_v1 = transform_check_cvv(c1);
	int check_v2 = transform_check_cvv(c2);
	int check_v3 = transform_check_cvv(c3);

	//std::vector<Vertex> newlist;
	if (check_v1 != 0 && check_v2 != 0 && check_v3 != 0) return;
	//if (check_v1 & 12 != 0 || check_v1 & 12 != 0 || check_v1 & 12 != 0) return;
	/*if (check_v1 == 0 && check_v2 == 0)
		newlist.push_back(*v1);*/
		/*else if (check_v1 == 0 && check_v2 != 0) {
			newlist.push_back(*v1);
			Vertex v11;
			float t;
			if (check_v2 < 4) {
				if(v1->position.z>v2->position.z)
					t = fabs(v1->position.z - w) / fabs(v2->position.z);
			}
			vertex_interp(&v11, v1, v2,t);
		}
	*/

	// 归一化，从cvv空间中的坐标直接转换到屏幕
	transform_homogenize(&device->transform, p1, c1);
	transform_homogenize(&device->transform, p2, c2);
	transform_homogenize(&device->transform, p3, c3);
	//std::cout << p1.x << p2.y << p1.z << std::endl;
	// 判断进入纹理或者色彩绘制
	if (render_state == RENDER_STATE_TEXTURE || render_state == RENDER_STATE_GOURAUD || render_state == RENDER_STATE_COLOR) {
		vertex t1 = *v1, t2 = *v2, t3 = *v3;
		trapezoid_t traps[2];
		int n;
		//计算t1,t2,t3的颜色值
		//====================================
		Vector3 lightdirection(0.0f, 5.0f, 0.0f);
		lightdirection = vec_normalize(lightdirection);

		t1.normal = vec_normalize(t1.normal);
		float k1 = dot(t1.normal, lightdirection);
		t2.normal = vec_normalize(t2.normal);
		float k2 = dot(t2.normal, lightdirection);
		t3.normal = vec_normalize(t3.normal);
		float k3 = dot(t3.normal, lightdirection);
		t1.color.r *= k1;
		t1.color.g *= k1;
		t1.color.b *= k1;
		t2.color.r *= k1;
		t2.color.g *= k1;
		t2.color.b *= k1;
		t3.color.r *= k1;
		t3.color.g *= k1;
		t3.color.b *= k1;
		//==================================

		t1.position = p1;
		t2.position = p2;
		t3.position = p3;
		t1.position.w = c1.w;
		t2.position.w = c2.w;
		t3.position.w = c3.w;

		vertex_rhw_init(&t1);	// 初始化 w
		vertex_rhw_init(&t2);	// 初始化 w
		vertex_rhw_init(&t3);	// 初始化 w

		// 拆分三角形为0-2个梯形，并且返回可用梯形数量
		bool l_or_r;
		n = trapezoid_init_triangle(traps, &t1, &t2, &t3, l_or_r, v1->position.xyz(), v2->position.xyz(), v3->position.xyz());
		//n = trapezoid_init_triangle(traps, &t1, &t2, &t3, l_or_r, wp1.xyz(), wp2.xyz(), wp3.xyz());

		if (n >= 1) device_render_trap(device, &traps[0], mesh, 0, l_or_r);
		if (n >= 2) device_render_trap(device, &traps[1], mesh, 1, l_or_r);
	}
	//判断进入线框绘制
	if (render_state == RENDER_STATE_WIREFRAME) {
		draw_line(device, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, device->foreground);
		draw_line(device, (int)p1.x, (int)p1.y, (int)p3.x, (int)p3.y, device->foreground);
		draw_line(device, (int)p3.x, (int)p3.y, (int)p2.x, (int)p2.y, device->foreground);
	}
}

//画面
void draw_plane(device_t* device, vertex mesh[], int a, int b, int c, int d) {
	vertex p1 = mesh[a], p2 = mesh[b], p3 = mesh[c], p4 = mesh[d];
	p1.texcoord.u = 0, p1.texcoord.v = 0, p2.texcoord.u = 0, p2.texcoord.v = 1;
	p3.texcoord.u = 1, p3.texcoord.v = 1, p4.texcoord.u = 1, p4.texcoord.v = 0;
	draw_triangle(device, &p1, &p2, &p3);
	draw_triangle(device, &p3, &p4, &p1);
}
void draw_plane(device_t* device,vertex mesh[] ,Point worldmesh[],int a, int b, int c, int d) {
	vertex p1 = mesh[a], p2 = mesh[b], p3 = mesh[c], p4 = mesh[d];
	Point wp1 = worldmesh[a], wp2 = worldmesh[b], wp3 = worldmesh[c], wp4 = worldmesh[d];
	p1.texcoord.u = 0, p1.texcoord.v = 0, p2.texcoord.u = 0, p2.texcoord.v = 1;
	p3.texcoord.u = 1, p3.texcoord.v = 1, p4.texcoord.u = 1, p4.texcoord.v = 0;
	draw_triangle(device, &p1, &p2, &p3,wp1,wp2,wp3);
	draw_triangle(device, &p3, &p4, &p1,wp3,wp4,wp1);
}
void draw_box(device_t* device,vertex mesh[], float theta) {
	matrix m;
	Vector3 axis(-1.0f, -0.5f, 1.0f);
	m = get_rotate_matrix(axis, theta);
	device->transform.model = m;
	transform_update(&device->transform);
	draw_plane(device, mesh, 0, 1, 2, 3);
	draw_plane(device, mesh, 7, 6, 5, 4);
	draw_plane(device, mesh, 0, 4, 5, 1);
	draw_plane(device, mesh, 1, 5, 6, 2);
	draw_plane(device, mesh, 2, 6, 7, 3);
	draw_plane(device, mesh, 3, 7, 4, 0);
}
void draw_other_box(device_t* device, vertex mesh[], float theta,Matrix model) {
	matrix m;
	Vector3 axis(-1.0f, -0.5f, 1.0f);
	m = get_rotate_matrix(axis, theta);
	model = matrix_multi_matrix(model, m);
	Point worldmesh[8];
	for (int i = 0; i < 8; ++i) {
		worldmesh[i] = vector_multi_matrix(model, mesh[i].position);
	}
	device->transform.model = model;
	transform_update(&device->transform);
	draw_plane(device, mesh,worldmesh, 0, 1, 2, 3);
	draw_plane(device, mesh,worldmesh, 7, 6, 5, 4);
	draw_plane(device, mesh,worldmesh, 0, 4, 5, 1);
	draw_plane(device, mesh,worldmesh, 1, 5, 6, 2);
	draw_plane(device, mesh,worldmesh, 2, 6, 7, 3);
	draw_plane(device, mesh,worldmesh, 3, 7, 4, 0);
}
//控制x的范围在（min,max）之间
int CMID(int x, int min, int max) { 
	if (x < min)
		return min;
	else if (x > max)
		return max;
	else
		return x;
}

// 根据坐标读取纹理 返回纹理buffer中的元素(纹素)
unsigned int device_texture_read(const device_t * device, float u, float v) {
	int x, y;
	u = u * device->max_u;
	v = v * device->max_v;
	x = (int)(u + 0.5f);
	y = (int)(v + 0.5f);
	x = CMID(x, 0, device->tex_width - 1);//x,y不能超过纹理数组边界
	y = CMID(y, 0, device->tex_height - 1);
	return device->texture[y][x];
}









//=====================================================================
// Win32 窗口及图形绘制：为 device 提供一个 DibSection 的 FB
//=====================================================================
int screen_w, screen_h, screen_exit = 0;
int screen_mx = 0, screen_my = 0, screen_mb = 0;
int screen_keys[512];	// 当前键盘按下状态
float MousePos[2] = {400.0f,300.0f};
bool MouseState=0;
static HWND screen_handle = NULL;		// 主窗口 HWND
static HDC screen_dc = NULL;			// 配套的 HDC
static HBITMAP screen_hb = NULL;		// DIB
static HBITMAP screen_ob = NULL;		// 老的 BITMAP
unsigned char* screen_fb = NULL;		// frame buffer
long screen_pitch = 0;

int screen_init(int w, int h, const TCHAR* title);	// 屏幕初始化
int screen_close(void);								// 关闭屏幕
void screen_dispatch(void);							// 处理消息
void screen_update(void);							// 显示 FrameBuffer

// win32 event handler
static LRESULT screen_events(HWND, UINT, WPARAM, LPARAM);

#ifdef _MSC_VER
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#endif

// 初始化窗口并设置标题
int screen_init(int w, int h, const TCHAR* title) {
	WNDCLASS wc = { CS_BYTEALIGNCLIENT, (WNDPROC)screen_events, 0, 0, 0,
		NULL, NULL, NULL, NULL, _T("SCREEN3.1415926") };
	BITMAPINFO bi = { { sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB,
		w * h * 4, 0, 0, 0, 0 } };
	RECT rect = { 0, 0, w, h };
	int wx, wy, sx, sy;
	LPVOID ptr;
	HDC hDC;

	screen_close();

	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.hInstance = GetModuleHandle(NULL);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	if (!RegisterClass(&wc)) return -1;

	screen_handle = CreateWindow(_T("SCREEN3.1415926"), title,
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
		0, 0, 0, 0, NULL, NULL, wc.hInstance, NULL);
	if (screen_handle == NULL) return -2;

	screen_exit = 0;
	hDC = GetDC(screen_handle);
	screen_dc = CreateCompatibleDC(hDC);
	ReleaseDC(screen_handle, hDC);

	screen_hb = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &ptr, 0, 0);
	if (screen_hb == NULL) return -3;

	screen_ob = (HBITMAP)SelectObject(screen_dc, screen_hb);
	screen_fb = (unsigned char*)ptr;
	screen_w = w;
	screen_h = h;
	screen_pitch = w * 4;

	AdjustWindowRect(&rect, GetWindowLong(screen_handle, GWL_STYLE), 0);
	wx = rect.right - rect.left;
	wy = rect.bottom - rect.top;
	sx = (GetSystemMetrics(SM_CXSCREEN) - wx) / 2;
	sy = (GetSystemMetrics(SM_CYSCREEN) - wy) / 2;
	if (sy < 0) sy = 0;
	SetWindowPos(screen_handle, NULL, sx, sy, wx, wy, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));
	SetForegroundWindow(screen_handle);

	ShowWindow(screen_handle, SW_NORMAL);
	screen_dispatch();

	memset(screen_keys, 0, sizeof(int) * 512);
	memset(screen_fb, 0, w * h * 4);

	return 0;
}

int screen_close(void) {
	if (screen_dc) {
		if (screen_ob) {
			SelectObject(screen_dc, screen_ob);
			screen_ob = NULL;
		}
		DeleteDC(screen_dc);
		screen_dc = NULL;
	}
	if (screen_hb) {
		DeleteObject(screen_hb);
		screen_hb = NULL;
	}
	if (screen_handle) {
		CloseWindow(screen_handle);
		screen_handle = NULL;
	}
	return 0;
}

static LRESULT screen_events(HWND hWnd, UINT msg,
	WPARAM wParam, LPARAM lParam) {
	switch (msg) {
	case WM_CLOSE: screen_exit = 1; break;
	case WM_KEYDOWN: screen_keys[wParam & 511] = 1; break;
	case WM_KEYUP: screen_keys[wParam & 511] = 0; break;
	//鼠标按下
	
	case WM_LBUTTONDOWN: {
		MouseState = 1;
		break;
	}
	case WM_LBUTTONUP: {
		MouseState = 0;
		break;
	}
	if (MouseState == 1) {
		case WM_MOUSEMOVE://鼠标移动
		{
			float x = LOWORD(lParam);
			float y = HIWORD(lParam);
			MousePos[0] = x;
			MousePos[1] = y;
			break;
		}
	}
	
	default: return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	
	return 0;
}

void screen_dispatch(void) {
	MSG msg;
	while (1) {
		if (!PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) break;
		if (!GetMessage(&msg, NULL, 0, 0)) break;
		DispatchMessage(&msg);
	}
}

void screen_update(void) {
	HDC hDC = GetDC(screen_handle);
	BitBlt(hDC, 0, 0, screen_w, screen_h, screen_dc, 0, 0, SRCCOPY);
	ReleaseDC(screen_handle, hDC);
	screen_dispatch();
}


Vertex mesh[8] = {
	{ {  -1, -1,  1,1 }, { 0, 0 },{  -1, -1,  1},{1.0f, 0.2f, 0.2f}, 1 },
	{ {  1, -1,  1, 1 }, { 0, 1 }, {  1, -1,  1}, {0.2f, 1.0f, 0.2f}, 1 },
	{ {  1,  1,  1, 1 }, { 1, 1 }, {  1,  1,  1},{0.2f, 0.2f, 1.0f}, 1 },
	{ { -1,  1,  1, 1 }, { 1, 0 }, { -1,  1,  1}, {1.0f, 0.2f, 1.0f},1 },
	{ { -1, -1, -1, 1 }, { 0, 0 }, { -1, -1, -1},{1.0f,1.0f,0.2f}, 1 },
	{ {  1, -1, -1, 1 }, { 0, 1 }, {  1, -1, -1},{0.2f,0.1f,1.0f}, 1 },
	{ {  1,  1, -1, 1 }, { 1, 1 }, {  1,  1, -1}, {1.0f,0.3f,0.3f},1 },
	{ { -1,  1, -1, 1 }, { 1, 0 }, { -1,  1, -1}, {0.2f,1.0f,0.3f},1 },
};
//相机初始化
void camera_init(device_t* device, float x, float y, float z) {
	Vector3 at(0, 0, 0);
	Vector3 pos(x, y, z);
	Vector3 up(0, 0, 1);
	Vector3 lookat = at - pos;
	Vector3 right = cross(up, lookat);
	device->transform.view = get_view_matrix(pos,up,right,lookat);
	transform_update(&device->transform);
}
//初始朝向一致的相机初始化
void camera_init_front(device_t* device, float x, float y, float z,Camera camera) {
	
	
	device->transform.view = get_view_matrix(camera.pos, camera.up, camera.right, camera.lookat);
	transform_update(&device->transform);
}
//用现阶段的相机向量重载观察矩阵
void camera_update(device_t* device ,Camera camera) {
	device->transform.view = get_view_matrix(camera.pos, camera.up, camera.right, camera.lookat);
	transform_update(&device->transform);
}
//棋盘格纹理
void init_texture(device_t* device) {
	static unsigned texture[256][256];
	int i, j;
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;
			texture[j][i] = ((x + y) & 1) ? 0xFF6A6A : 0x1A1A1A;
		}
	}
	device_set_texture(device, texture, 256 * 4, 256, 256);
}
//图片加载纹理
void init_std_texture(device_t* device,const char* path) {
	static unsigned int texture[256][256];
	int width, height, channels;
	unsigned char* data = stbi_load(path, &width, &height, &channels, 0);
	for (int y = 0; y < height; y+=2) {
		unsigned char* pdata = data + y * width*channels;
		unsigned int* Texture = *texture + (y/2) * 256;
		for (int x = 0; x < width; x++) {
			
				/*unsigned int* r = reinterpret_cast<unsigned int*>(pdata[0]);
				unsigned int* g = reinterpret_cast<unsigned int*>(pdata[1]);
				unsigned int* b = reinterpret_cast<unsigned int*>(pdata[2]);*/
				unsigned int r = pdata[0];
				unsigned int g = pdata[1];
				unsigned int b = pdata[2];
				/*int R = *r;
				int G = *g;
				int B = *b;
*/
				Texture[x/2] = (r << 16) | (g << 8) | b;
				pdata += channels;
		}
	}

	device_set_texture(device, texture, 256 * 4, 256, 256);
	stbi_image_free(data);
}
int main()
{
	
	device_t device;
	int states[] = { RENDER_STATE_TEXTURE, RENDER_STATE_WIREFRAME,RENDER_STATE_GOURAUD,RENDER_STATE_COLOR };
	int indicator = 0;
	int kbhit = 0;
	float theta = 1;
	//设置相机起始位置
	Camera& camera = device.camera;
	camera.pos.x = 1.5938;
	camera.pos.y = 6.19593;
	camera.pos.z = 6.80285;
	//设置光源起始位置
	light.pos.x = 4.5f;
	light.pos.y = 1.5f;
	light.pos.z = -1.0f;
	light.lightcol = { 1.0f, 0.0f, 1.0f };

	float deltax = 0.0f;
	float deltay = 0.0f;
	float past_MousePos[2];
	//TCHAR ch[10] = { "RASTER" };
	TCHAR* title = NULL;
		
	if (screen_init(800, 600, title))
		return -1;
	
	device_init(&device, 800, 600, screen_fb);
	
	//camera_init(&device, 3, 0, 0);
	//init_texture(&device);
	
	device.render_state = RENDER_STATE_TEXTURE;

	Vector3 zero,up;
	zero.x = 3.0f; zero.y = 2.5f; zero.z = 0.0f;
	camera.lookat = zero - camera.pos;
	camera.lookat = vec_normalize(camera.lookat);
	up = { 0.275345,-0.275707,0.798942 };
	camera.right = cross(up, camera.lookat);
	camera.right = vec_normalize(camera.right);

	camera.up = cross(camera.lookat, camera.right);
	camera.up = vec_normalize(camera.up);
	float pitch = 0;//0到89度
	float yaw = 0;//0到180度
	
	
	
	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {

		clock_t start, finish;
		start = clock();


		screen_dispatch();
		device_clear(&device);


		//std::cout << camera.pos.x <<"   "<< camera.pos.y <<"   "<< camera.pos.z << std::endl;
		//std::cout << camera.pos.x << std::endl;

		//====================================================
		camera_init_front(&device, camera.pos.x, camera.pos.y, camera.pos.z, camera);

		float velocity = 0.5f;



		//相机控制：沿x,y,z轴移动，俯仰和偏航
		if (MouseState) {
			deltax = MousePos[0] - past_MousePos[0];
			deltay = MousePos[1] - past_MousePos[1];
			//std::cout << deltay << std::endl;
			Matrix3 rotateU;
			rotateU.set_unitmatrix();
			Matrix3 rotateR;
			rotateR.set_unitmatrix();

			/*if (deltax != 0 || deltay != 0) {
				pitch = deltay ;
				if (pitch > 89.0f) pitch = 89.0f;
				if (pitch < -89.0f) pitch = -89.0f;
				yaw = deltax ;

				lookat.x = cos(PI * pitch / 180) * cos((yaw / 180) * PI);
				lookat.y = sin(PI * pitch / 180);
				lookat.z = cos(PI * pitch / 180) * sin((yaw / 180) * PI);

				lookat = vec_normalize(lookat);
			}*/
			if (deltax != 0) {
				rotateU = u_rotate_matrix(-deltax*0.5 / 800 * PI, camera.up);//旋转方向？
			}
			if (deltay != 0) {
				rotateR = r_rotate_matrix(-deltay*0.5 / 600 * PI, camera.right);
			}
			update_camera_vector(&camera, rotateU, rotateR);
			camera_update(&device, camera);

			//lookat = lookat + deltax*0.1f*right;
			//lookat = lookat + Vector3(0.0f, -deltay * 0.1f, 0.0f);

		}

		if (screen_keys[VK_INSERT]) {
			Matrix3 rotateL;
			rotateL = l_rotate_matrix(PI / 30, camera.lookat);
			camera.lookat = vector_multi_matrix(camera.lookat, rotateL);
			camera.up = vector_multi_matrix(camera.up, rotateL);
			camera.right = vector_multi_matrix(camera.right, rotateL);
			camera_update(&device, camera);
		}
		if (screen_keys[VK_DELETE]) {
			Matrix3 rotateL;
			rotateL = l_rotate_matrix(-PI / 30, camera.lookat);
			camera.lookat = vector_multi_matrix(camera.lookat, rotateL);
			camera.up = vector_multi_matrix(camera.up, rotateL);
			camera.right = vector_multi_matrix(camera.right, rotateL);
			camera_update(&device, camera);
		}

		if (screen_keys[VK_UP])//posz += 0.1f;
			camera.pos = camera.pos + camera.lookat * velocity;
		if (screen_keys[VK_DOWN])// posz -= 0.1f;
			camera.pos = camera.pos - camera.lookat * velocity;
		if (screen_keys[VK_LEFT]) //posx -= 0.1f;
			camera.pos = camera.pos - camera.right * velocity;
		if (screen_keys[VK_RIGHT])//posx += 0.1f;
			camera.pos = camera.pos + camera.right * velocity;
		if (screen_keys[VK_HOME])
			camera.pos = camera.pos + camera.up * velocity;
		if (screen_keys[VK_END])
			camera.pos = camera.pos - camera.up * velocity;

		if (screen_keys[VK_NUMPAD8])
			camera.lookat = camera.lookat + Vector3(0.0f, 0.0f, 0.05f);
		if (screen_keys[VK_NUMPAD2])
			camera.lookat = camera.lookat + Vector3(0.0f, 0.0f, -0.05f);
		if (screen_keys[VK_NUMPAD6])
			camera.lookat = camera.lookat + Vector3(0.05f, 0.0f, 0.0f);
		if (screen_keys[VK_NUMPAD4])
			camera.lookat = camera.lookat + Vector3(-0.05f, 0.0f, 0.0f);

		//std::cout << MouseState << std::endl;
		past_MousePos[0] = MousePos[0];
		past_MousePos[1] = MousePos[1];

		if (screen_keys[VK_SPACE]) {

			if (kbhit == 0) {
				kbhit = 1;
				if (++indicator >= 2) indicator = 0;
				device.render_state = states[indicator];
			}
		}
		else {
			kbhit = 0;
		}

		/*draw_line(&device, 600, 400, 400, 100, 9);
		draw_line(&device, 200, 400, 600, 400, 8);
		draw_line(&device, 400, 100, 200, 400, 7);*/

		Vertex* v1 = &mesh[5];
		Vertex* v2 = &mesh[6];
		Vertex* v3 = &mesh[7];
		//draw_triangle(&device, v1, v2, v3);
		//draw_plane(&device, mesh, 7, 6, 5, 4);


		Matrix model;
		model.set_unitmatrix(model);
		//第一个box
		init_texture(&device);
		draw_other_box(&device, mesh, theta, model);
		//其他box

		model = matrix_multi_matrix(get_translate_matrix(5.0f, 2.0f, 3.0f), get_scale_matrix(0.5));
		const char* path1 = "../matrix.jpg";
		init_std_texture(&device, path1);
		draw_other_box(&device, mesh, theta, model);

		model = get_translate_matrix(4.0f, 0.0f, 0.0f);
		const char* path2 = "../container.jpg";
		init_std_texture(&device, path2);
		draw_other_box(&device, mesh, theta, model);

		model = matrix_multi_matrix(get_translate_matrix(4.0f, 3.0f, 0.0f), get_scale_matrix(0.3));
		const char* path4 = "../container2.png";
		init_std_texture(&device, path4);
		draw_other_box(&device, mesh, theta, model);



		screen_update();
		Sleep(1);

		finish = clock();
		std::cout << (double)(finish - start)/CLOCKS_PER_SEC << std::endl;
	}

	
	//system("pause");
	return 0;
}
