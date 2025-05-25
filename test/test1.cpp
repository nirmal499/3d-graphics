#include <iostream>

class Vec3Class
{
private:
    float x, y, z;

public:
    Vec3Class(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Friend non-member function to access private members
    friend Vec3Class operator+(const Vec3Class& a, const Vec3Class& b);

    // Optional: for printing
    friend std::ostream& operator<<(std::ostream& os, const Vec3Class& v);
};

// Implementation of friend operator+
Vec3Class operator+(const Vec3Class& a, const Vec3Class& b)
{
    return Vec3Class(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

// Optional: friend for printing
std::ostream& operator<<(std::ostream& os, const Vec3Class& v)
{
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
}

struct Vec3Struct
{
    float x;
    float y;
    float z;
};

// Non-member operator+ — no friend needed because members are public
Vec3Struct operator+(const Vec3Struct& a, const Vec3Struct& b)
{
    return Vec3Struct{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

// Optional: printing operator
std::ostream& operator<<(std::ostream& os, const Vec3Struct& v)
{
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
}

int main()
{
    Vec3Struct vs1{1.0f, 2.0f, 3.0f};
    Vec3Struct vs2{4.0f, 5.0f, 6.0f};
    Vec3Struct sresult = vs1 + vs2;

    std::cout << "vs1 + vs2 = " << sresult << '\n';  // Outputs: (5, 7, 9)

    Vec3Class vc1(1.0f, 2.0f, 3.0f);
    Vec3Class vc2(4.0f, 5.0f, 6.0f);

    Vec3Class cresult = vc1 + vc2;

    std::cout << "vc1 + vc2 = " << cresult << '\n'; // Output: (5, 7, 9)
}

