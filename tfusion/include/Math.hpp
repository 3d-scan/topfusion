#pragma once

#include "MathUtils.hpp"

#ifndef NULL
#define NULL 0
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#include "Vector.hpp"
#include "Matrix.hpp"

typedef class tfusion::Matrix3<float> Matrix3f;
typedef class tfusion::Matrix4<float> Matrix4f;

typedef class tfusion::Vector2<short> Vector2s;
typedef class tfusion::Vector2<int> Vector2i;
typedef class tfusion::Vector2<float> Vector2f;
typedef class tfusion::Vector2<double> Vector2d;

typedef class tfusion::Vector3<short> Vector3s;
typedef class tfusion::Vector3<double> Vector3d;
typedef class tfusion::Vector3<int> Vector3i;
typedef class tfusion::Vector3<uint> Vector3ui;
typedef class tfusion::Vector3<uchar> Vector3u;
typedef class tfusion::Vector3<float> Vector3f;

typedef class tfusion::Vector4<float> Vector4f;
typedef class tfusion::Vector4<int> Vector4i;
typedef class tfusion::Vector4<short> Vector4s;
typedef class tfusion::Vector4<uchar> Vector4u;

typedef class tfusion::Vector6<float> Vector6f;

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (x).toIntRound()
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (x).toIntRound()
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(inted, coeffs, in) inted = (in).toIntFloor(coeffs)
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (x).toShortFloor()
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (x).toUChar()
#endif

#ifndef TO_UCHAR4
#define TO_UCHAR4(x) (x).toUChar()
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (x).toFloat()
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) (a).toVector3()
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif
