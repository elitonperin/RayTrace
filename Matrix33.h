#ifndef __Matrix33_h
#define __Matrix33_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                        GVSG Foundation Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|                Copyright® 2007, Paulo Aristarco Pagliosa                 |
//|                All Rights Reserved.                                      |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: Matrix33.h
//  ========
//  Class definition for 3x3 matrix.

#ifndef __Quaternion_h
#include "Quaternion.h"
#endif

//
// Auxiliary functions
//
template <typename real>
__host__ __device__ inline void
multiply(const real a[3][3], const real b[3][3], real c[3][3])
{
	real c00 = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
	real c01 = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
	real c02 = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
	real c10 = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
	real c11 = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
	real c12 = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
	real c20 = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
	real c21 = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
	real c22 = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];

	c[0][0] = c00; c[0][1] = c01; c[0][2] = c02;
	c[1][0] = c10; c[1][1] = c11; c[1][2] = c12;
	c[2][0] = c20; c[2][1] = c21; c[2][2] = c22;
}

template <typename real>
__host__ __device__ inline void
multiply(const real a[3][3], real s, real c[3][3])
{
	c[0][0] = a[0][0] * s;
	c[0][1] = a[0][1] * s;
	c[0][2] = a[0][2] * s;
	c[1][0] = a[1][0] * s;
	c[1][1] = a[1][1] * s;
	c[1][2] = a[1][2] * s;
	c[2][0] = a[2][0] * s;
	c[2][1] = a[2][1] * s;
	c[2][2] = a[2][2] * s;
}

template <typename real>
__host__ __device__ inline bool
invert(const real a[3][3], real c[3][3], real eps = zero<real>())
{
	real b00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
	real b01 = a[0][2] * a[2][1] - a[0][1] * a[2][2];
	real b02 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
	real b10 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
	real b11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
	real b12 = a[0][2] * a[1][0] - a[0][0] * a[1][2];
	real b20 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
	real b21 = a[0][1] * a[2][0] - a[0][0] * a[2][1];
	real b22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
	real d = b00 * a[0][0] + b01 * a[1][0] + b02 * a[2][0];

	if (Math::isZero(d, eps))
		return false;
	d = Math::inverse(d);
	c[0][0] = b00 * d; c[0][1] = b01 * d; c[0][2] = b02 * d;
	c[1][0] = b10 * d; c[1][1] = b11 * d; c[1][2] = b12 * d;
	c[2][0] = b20 * d; c[2][1] = b21 * d; c[2][2] = b22 * d;
	return true;
}

template <typename real>
__host__ __device__ inline void
swap(real& a, real& b)
{
	real temp = a;

	a = b;
	b = temp;
}


//////////////////////////////////////////////////////////
//
// Matrix33: 3x3 matrix class
// ========
template <typename real>
class Matrix33
{
public:
	// Constructors
	__host__ __device__
	Matrix33()
	{
		// do nothing
	}

	__host__ __device__
	Matrix33(const Vector3<real>* v)
	{
		set(v);
	}

	__host__ __device__
	Matrix33(const real* a, const real* b, const real* c)
	{
		set(a, b, c);
	}

	__host__ __device__
	Matrix33(const real* m)
	{
		set(m, m + 3, m + 6);
	}

	__host__ __device__
	explicit Matrix33(const Quaternion<real>& q)
	{
		set(q);
	}

	__host__ __device__
	void set(const Vector3<real>* v)
	{
		M[_X][_X] = v[_X].x; M[_X][_Y] = v[_X].y; M[_X][_Z] = v[_X].z;
		M[_Y][_X] = v[_Y].x; M[_Y][_Y] = v[_Y].y; M[_Y][_Z] = v[_Y].z;
		M[_Z][_X] = v[_Z].x; M[_Z][_Y] = v[_Z].y; M[_Z][_Z] = v[_Z].z;
	}

	__host__ __device__
	void set(const real* a, const real* b, const real* c)
	{
		M[_X][_X] = a[_X]; M[_X][_Y] = a[_Y]; M[_X][_Z] = a[_Z];
		M[_Y][_X] = b[_X]; M[_Y][_Y] = b[_Y]; M[_Y][_Z] = b[_Z];
		M[_Z][_X] = c[_X]; M[_Z][_Y] = c[_Y]; M[_Z][_Z] = c[_Z];
	}

	__host__ __device__
	void set(const Quaternion<real>& q)
	{
		const real x = q.v.x;
		const real y = q.v.y;
		const real z = q.v.z;
		const real w = q.w;

		// _X row
		M[_X][_X] = 1 - 2 * (y * y + z * z);
		M[_X][_Y] = 2 * (x * y - w * z);
		M[_X][_Z] = 2 * (x * z + w * y);
		// _Y row
		M[_Y][_X] = 2 * (x * y + w * z);
		M[_Y][_Y] = 1 - 2 * (x * x + z * z);
		M[_Y][_Z] = 2 * (y * z - w * x);
		// _Z row
		M[_Z][_X] = 2 * (x * z - w * y);
		M[_Z][_Y] = 2 * (y * z + w * x);
		M[_Z][_Z] = 1 - 2 * (x * x + y * y);
	}

	__host__ __device__
	Matrix33<real>& operator =(const Quat& q)
	{
		set(q);
		return *this;
	}

	__host__ __device__
	Matrix33<real>& zero()
	{
		M[_X][_X] = 0; M[_X][_Y] = 0; M[_X][_Z] = 0;
		M[_Y][_X] = 0; M[_Y][_Y] = 0; M[_Y][_Z] = 0;
		M[_Z][_X] = 0; M[_Z][_Y] = 0; M[_Z][_Z] = 0;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& identity()
	{
		M[_X][_X] = 1; M[_X][_Y] = 0; M[_X][_Z] = 0;
		M[_Y][_X] = 0; M[_Y][_Y] = 1; M[_Y][_Z] = 0;
		M[_Z][_X] = 0; M[_Z][_Y] = 0; M[_Z][_Z] = 1;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& diagonal(const Vector3<real>& d)
	{
		M[_X][_X] = d.x; M[_X][_Y] = 0;   M[_X][_Z] = 0;
		M[_Y][_X] = 0;   M[_Y][_Y] = d.y; M[_Y][_Z] = 0;
		M[_Z][_X] = 0;   M[_Z][_Y] = 0;   M[_Z][_Z] = d.z;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& rotationX(real angle)
	{
		real cosA = cos(angle);
		real sinA = sin(angle);

		M[_X][_X] = 1; M[_X][_Y] = 0;    M[_X][_Z] =  0;
		M[_Y][_X] = 0; M[_Y][_Y] = cosA; M[_Y][_Z] = -sinA;
		M[_Z][_X] = 0; M[_Z][_Y] = sinA; M[_Z][_Z] =  cosA;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& rotationY(real angle)
	{
		real cosA = cos(angle);
		real sinA = sin(angle);

		M[_X][_X] =  cosA; M[_X][_Y] = 0; M[_X][_Z] = sinA;
		M[_Y][_X] =  0;    M[_Y][_Y] = 1; M[_Y][_Z] = 0;
		M[_Z][_X] = -sinA; M[_Z][_Y] = 0; M[_Z][_Z] = cosA;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& rotationZ(real angle)
	{
		real cosA = cos(angle);
		real sinA = sin(angle);

		M[_X][_X] = cosA; M[_X][_Y] = -sinA; M[_X][_Z] = 0;
		M[_Y][_X] = sinA; M[_Y][_Y] =  cosA; M[_Y][_Z] = 0;
		M[_Z][_X] = 0;    M[_Z][_Y] =  0;    M[_Z][_Z] = 1;
		return *this;
	}

	__host__ __device__
	Matrix33<real>& rotation(real angle, const Vector3<real>& axis)
	{
		set(Quat(angle, axis));
		return *this;
	}

	__host__ __device__
	Matrix33<real>& star(const Vector3<real>& v)
	{
		M[_X][_X] =  0.0; M[_X][_Y] = -v.z; M[_X][_Z] =  v.y;
		M[_Y][_X] =  v.z; M[_Y][_Y] =  0.0; M[_Y][_Z] = -v.x;
		M[_Z][_X] = -v.y; M[_Z][_Y] =  v.x; M[_Z][_Z] =  0.0;
		return *this;
	}

	__host__ __device__
	real trace() const
	{
		return M[_X][_X] + M[_Y][_Y] + M[_Z][_Z];
	}

	__host__ __device__
	Vector3<real> getColumn(int j) const
	{
		return Vector3<real>(M[_X][j], M[_Y][j], M[_Z][j]);
	}

	__host__ __device__
	Vector3<real> getRow(int i) const
	{
		return Vector3<real>(M[i][_X], M[i][_Y], M[i][_Z]);
	}

	__host__ __device__
	real operator ()(int i, int j) const
	{
		return M[i][j];
	}

	__host__ __device__
	Vector3<real> getDiagonal() const
	{
		return Vector3<real>(M[_X][_X], M[_Y][_Y], M[_Z][_Z]);
	}

	__host__ __device__
	void setColumn(int j, const Vector3<real>& v)
	{
		M[_X][j] = v.x;
		M[_Y][j] = v.y;
		M[_Z][j] = v.z;
	}

	__host__ __device__
	void setRow(int i, const Vector3<real>& v)
	{
		M[i][_X] = v.x; M[i][_Y] = v.y; M[i][_Z] = v.z;
	}

	__host__ __device__
	real& operator ()(int i, int j)
	{
		return M[i][j];
	}

	__host__ __device__
	Vector3<real> multiply(const Vector3<real>& v) const
	{
		real x = M[_X][_X] * v.x + M[_X][_Y] * v.y + M[_X][_Z] * v.z;
		real y = M[_Y][_X] * v.x + M[_Y][_Y] * v.y + M[_Y][_Z] * v.z;
		real z = M[_Z][_X] * v.x + M[_Z][_Y] * v.y + M[_Z][_Z] * v.z;

		return Vector3<real>(x, y, z);
	}

	__host__ __device__
	Vector3<real> multiplyTranspose(const Vector3<real>& v) const
	{
		real x = M[_X][_X] * v.x + M[_Y][_X] * v.y + M[_Z][_X] * v.z;
		real y = M[_X][_Y] * v.x + M[_Y][_Y] * v.y + M[_Z][_Y] * v.z;
		real z = M[_X][_Z] * v.x + M[_Y][_Z] * v.y + M[_Z][_Z] * v.z;

		return Vector3<real>(x, y, z);
	}

	__host__ __device__
	Matrix33<real> multiplyDiagonal(const Vector3<real>& d) const
	{
		Matrix33<real> r;

		r.M[_X][_X] = M[_X][_X] * d.x;
		r.M[_X][_Y] = M[_X][_Y] * d.y;
		r.M[_X][_Z] = M[_X][_Z] * d.z;
		r.M[_Y][_X] = M[_Y][_X] * d.x;
		r.M[_Y][_Y] = M[_Y][_Y] * d.y;
		r.M[_Y][_Z] = M[_Y][_Z] * d.z;
		r.M[_Z][_X] = M[_Z][_X] * d.x;
		r.M[_Z][_Y] = M[_Z][_Y] * d.y;
		r.M[_Z][_Z] = M[_Z][_Z] * d.z;
		return r;
	}

	__host__ __device__
	Matrix33<real>& operator *=(real s)
	{
		::multiply(M, s, M);
		return *this;
	}

	__host__ __device__
	Matrix33<real>& operator *=(const Matrix33<real>& m)
	{
		::multiply(M, m.M, M);
		return *this;
	}

	__host__ __device__
	Matrix33<real> operator *(real s) const
	{
		Matrix33<real> r;

		::multiply(M, s, r.M);
		return r;
	}

	__host__ __device__
	Matrix33<real> operator *(const Matrix33<real>& m) const
	{
		Matrix33<real> r;

		::multiply(M, m.M, r.M);
		return r;
	}

	__host__ __device__
	Vector3<real> operator *(const Vector3<real>& v) const
	{
		return multiply(v);
	}

	__host__ __device__
	bool isZero() const
	{
		return getRow(0).isZero() && getRow(1).isZero() && getRow(2).isZero();
	}

	__host__ __device__
	Matrix33<real>& setTranspose()
	{
		::swap(M[_X][_Y], M[_Y][_X]);
		::swap(M[_X][_Z], M[_Z][_X]);
		::swap(M[_Y][_Z], M[_Z][_Y]);
		return *this;
	}

	__host__ __device__
	Matrix33<real> transpose() const
	{
		Matrix33<real> a;

		a.M[_X][_X] = M[_X][_X];
		a.M[_X][_Y] = M[_Y][_X];
		a.M[_X][_Z] = M[_Z][_X];
		a.M[_Y][_X] = M[_X][_Y];
		a.M[_Y][_Y] = M[_Y][_Y];
		a.M[_Y][_Z] = M[_Z][_Y];
		a.M[_Z][_X] = M[_X][_Z];
		a.M[_Z][_Y] = M[_Y][_Z];
		a.M[_Z][_Z] = M[_Z][_Z];
		return a;
	}

	__host__ __device__
	bool invert(real eps = Math::zero<real>())
	{
		return ::invert(M, M, eps);
	}

	__host__ __device__
	bool inverse(Matrix33<real>& m, real eps = Math::zero<real>()) const
	{
		return (m = *this).invert(eps);
	}

	void print(const char* s, FILE* f = stdout) const
	{
		fprintf(f, "%s(1)[%9.4f %9.4f %9.4f]\n", s, M[0][0], M[0][1], M[0][2]);
		fprintf(f, "%s(2)[%9.4f %9.4f %9.4f]\n", s, M[1][0], M[1][1], M[1][2]);
		fprintf(f, "%s(3)[%9.4f %9.4f %9.4f]\n", s, M[2][0], M[2][1], M[2][2]);
	}

private:
	real M[3][3];

}; // Matrix33

//
// Quaternion implementation (see Quaternion.h)
//
template <typename real>
__host__ __device__ inline void
Quaternion<real>::set(const Matrix33<real>& m)
{
	real tr = m(_X, _X) + m(_Y, _Y) + m(_Z, _Z);

	if (tr >= 0)
	{
		real s = sqrt(tr + 1);

		w   = real(0.5) * s;
		s   = real(0.5) / s;
		v.x = (m(_Z, _Y) - m(_Y, _Z)) * s;
		v.y = (m(_X, _Z) - m(_Z, _X)) * s;
		v.z = (m(_Y, _X) - m(_X, _Y)) * s;
	}
	else if (m(_Y, _Y) > m(_X, _X))
	{
		real s = sqrt(m(_Y, _Y) - (m(_Z, _Z) + m(_X, _X)) + 1);

		v.y = real(0.5) * s;
		s   = real(0.5) / s;
		v.z = (m(_Y, _Z) + m(_Z, _Y)) * s;
		v.x = (m(_X, _Y) + m(_Y, _X)) * s;
		w   = (m(_X, _Z) - m(_Z, _X)) * s;
	}
	else if (m(_Z, _Z) > m(_X, _X))
	{
		real s = sqrt(m(_Z, _Z) - (m(_X, _X) + m(_Y, _Y)) + 1);

		v.z = real(0.5) * s;
		s   = real(0.5) / s;
		v.x = (m(_Z, _X) + m(_X, _Z)) * s;
		v.y = (m(_Y, _Z) + m(_Z, _Y)) * s;
		w   = (m(_Y, _X) - m(_X, _Y)) * s;
	}
	else
	{
		real s = sqrt(m(_X, _X) - (m(_Y, _Y) + m(_Z, _Z)) + 1);

		v.x = real(0.5) * s;
		s   = real(0.5) / s;
		v.y = (m(_X, _Y) + m(_Y, _X)) * s;
		v.z = (m(_Z, _X) + m(_X, _Z)) * s;
		w   = (m(_X, _Y) - m(_Y, _Z)) * s;
	}
}

//
// Auxiliary function
//
template <typename real>
__host__ __device__ inline Matrix33<real>
operator *(double s, const Matrix33<real>& m)
{
	return m * (real)s;
}

typedef Matrix33<REAL> Mat33;

#endif // __Matrix33_h
