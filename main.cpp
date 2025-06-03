#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdio>
#include <stdexcept>
#include <memory>
#include <array>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <iostream>

#define FPS 30
#define FRAME_TARGET_TIME (1000 / FPS)

#define CUBE_MESH_VERTICES 8
#define CUBE_MESH_FACES (6 * 2) /* 6 cube face and 2 triangles per face */

struct Vec2
{
    float x;
    float y;
};

struct Vec3
{
    float x;
    float y;
    float z;
};

struct Vec4
{
    float x;
    float y;
    float z;
    float w;
};

struct Mat4
{
    float m[4][4];
};

/* It stores the indices */
struct Face
{
    int a;
    int b;
    int c;
    uint32_t color;
};

struct Triangle
{
    std::array<Vec2, 3> points;
    uint32_t color;
};

struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<Face> faces;
    Vec3 rotation; /* rotation with x, y and z values */
    Vec3 scale;
    Vec3 translation;
};

struct Light
{
    Vec3 direction;
};

uint32_t light_apply_intensity(uint32_t original_color, float percentage_factor)
{
    if (percentage_factor < 0) percentage_factor = 0;
    if (percentage_factor > 1) percentage_factor = 1;

    uint32_t a = (original_color & 0xFF000000);
    uint32_t r = (original_color & 0x00FF0000) * percentage_factor;
    uint32_t g = (original_color & 0x0000FF00) * percentage_factor;
    uint32_t b = (original_color & 0x000000FF) * percentage_factor;

    uint32_t new_color = a | (r & 0x00FF0000) | (g & 0x0000FF00) | (b & 0x000000FF);

    return new_color;
}

float vec3_length(const Vec3& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 vec3_add(const Vec3& a, const Vec3& b)
{
    Vec3 result = {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
    return result;
}

Vec3 vec3_sub(const Vec3& a, const Vec3& b)
{
    Vec3 result = {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
    return result;
}

Vec3 vec3_mul(const Vec3& v, float factor)
{
    Vec3 result = {
        v.x * factor,
        v.y * factor,
        v.z * factor
    };
    return result;
}

Vec3 vec3_div(const Vec3& v, float factor)
{
    if(factor == 0)
    {
        throw std::runtime_error("Division by 0");
    }

    Vec3 result = {
        v.x / factor,
        v.y / factor,
        v.z / factor
    };
    return result;
}

Vec3 vec3_cross(const Vec3& a, const Vec3& b)
{
    Vec3 result = {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
    return result;
}

float vec3_dot(const Vec3& a, const Vec3& b)
{
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

Vec3 vec3_rotate_x(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        v.x,
        v.y * std::cos(angle) - v.z * std::sin(angle),
        v.y * std::sin(angle) + v.z * std::cos(angle)
    };
    return rotated_vector;
}

Vec3 vec3_rotate_y(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        v.x * std::cos(angle) - v.z * std::sin(angle),
        v.y,
        v.x * std::sin(angle) + v.z * std::cos(angle)
    };
    return rotated_vector;
}

Vec3 vec3_rotate_z(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        v.x * std::cos(angle) - v.y * std::sin(angle),
        v.x * std::sin(angle) + v.y * std::cos(angle),
        v.z
    };
    return rotated_vector;
}

void vec3_normalize(Vec3& v)
{
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    v.x /= length;
    v.y /= length;
    v.z /= length;
}

Mat4 mat4_identity(void)
{
    // | 1 0 0 0 |
    // | 0 1 0 0 |
    // | 0 0 1 0 |
    // | 0 0 0 1 |
    Mat4 m = {{
        { 1, 0, 0, 0 },
        { 0, 1, 0, 0 },
        { 0, 0, 1, 0 },
        { 0, 0, 0, 1 }
    }};
    return m;
}

Mat4 mat4_make_scale(float sx, float sy, float sz)
{
    // | sx  0  0  0 |
    // |  0 sy  0  0 |
    // |  0  0 sz  0 |
    // |  0  0  0  1 |
    Mat4 m = mat4_identity();
    m.m[0][0] = sx;
    m.m[1][1] = sy;
    m.m[2][2] = sz;
    return m;
}

Mat4 mat4_make_translation(float tx, float ty, float tz)
{
    // | 1  0  0  tx |
    // | 0  1  0  ty |
    // | 0  0  1  tz |
    // | 0  0  0  1  |
    Mat4 m = mat4_identity();
    m.m[0][3] = tx;
    m.m[1][3] = ty;
    m.m[2][3] = tz;
    return m;
}

Mat4 mat4_make_rotation_x(float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    // | 1  0  0  0 |
    // | 0  c -s  0 |
    // | 0  s  c  0 |
    // | 0  0  0  1 |
    Mat4 m = mat4_identity();
    m.m[1][1] = c;
    m.m[1][2] = -s;
    m.m[2][1] = s;
    m.m[2][2] = c;
    return m;
}

Mat4 mat4_make_rotation_y(float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    // |  c  0  s  0 |
    // |  0  1  0  0 |
    // | -s  0  c  0 |
    // |  0  0  0  1 |
    Mat4 m = mat4_identity();
    m.m[0][0] = c;
    m.m[0][2] = s;
    m.m[2][0] = -s;
    m.m[2][2] = c;
    return m;
}

Mat4 mat4_make_rotation_z(float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);
    // | c -s  0  0 |
    // | s  c  0  0 |
    // | 0  0  1  0 |
    // | 0  0  0  1 |
    Mat4 m = mat4_identity();
    m.m[0][0] = c;
    m.m[0][1] = -s;
    m.m[1][0] = s;
    m.m[1][1] = c;
    return m;
}

Vec4 mat4_mul_vec4(const Mat4& m, const Vec4& v)
{
    Vec4 result;
    result.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
    result.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
    result.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
    result.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;
    return result;
}

Mat4 mat4_mul_mat4(const Mat4& a, const Mat4& b)
{
    Mat4 m;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j] + a.m[i][3] * b.m[3][j];
        }
    }
    return m;
}

Mat4 mat4_make_perspective(float fov, float aspect, float znear, float zfar)
{
    // | (h/w)*1/tan(fov/2)             0              0                 0 |
    // |                  0  1/tan(fov/2)              0                 0 |
    // |                  0             0     zf/(zf-zn)  (-zf*zn)/(zf-zn) |
    // |                  0             0              1                 0 |
    Mat4 m = {{{ 0 }}};
    m.m[0][0] = aspect * (1 / std::tan(fov / 2));
    m.m[1][1] = 1 / std::tan(fov / 2);
    m.m[2][2] = zfar / (zfar - znear);
    m.m[2][3] = (-zfar * znear) / (zfar - znear);
    m.m[3][2] = 1.0;
    return m;
}

Vec4 mat4_mul_vec4_project(const Mat4& mat_proj, const Vec4& v)
{
    // multiply the projection matrix by our original vector
    Vec4 result = mat4_mul_vec4(mat_proj, v);

    // perform perspective divide with original z-value that is now stored in w
    if (result.w != 0.0)
    {
        result.x /= result.w;
        result.y /= result.w;
        result.z /= result.w;
    }
    return result;
}

Vec4 vec4_from_vec3(const Vec3& v)
{
    Vec4 result = { v.x, v.y, v.z, 1.0 };
    return result;
}

Vec3 vec3_from_vec4(const Vec4& v)
{
    Vec3 result = { v.x, v.y, v.z };
    return result;
}

template<uint32_t WIDTH, uint32_t HEIGHT>
class Engine
{
    /* If any of them is zero then do assert */
    static_assert(WIDTH != 0 || HEIGHT != 0, "Either WIDTH or HEIGHT must be non-zero");

    public:
        Engine()
        {}

        void Setup()
        {
            _colorBufferTexture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);


            // Initialize the perspective projection matrix
            float fov = M_PI / 3.0; // the same as 180/3, or 60deg
            float aspect = (float)(WIDTH) / (float)(HEIGHT);
            float znear = 0.1;
            float zfar = 100.0;
            _proj_matrix = mat4_make_perspective(fov, aspect, znear, zfar);

            _mesh.rotation = Vec3{0.0, 0.0, 0.0};
            _mesh.scale = Vec3{1.0, 1.0, 1.0};
            _mesh.translation = Vec3{0.0, 0.0, 0.0};

            _light.direction = Vec3{0.0, 0.0, 1.0};

            // LoadCubeMeshData();
            // LoadObjFileData("C:\\my_code\\win_cpp\\3D_Graphics\\assets\\f22.obj");
            LoadObjFileData("C:\\my_code\\win_cpp\\3D_Graphics\\assets\\cube.obj");
        }

        void ProcessInput()
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            switch(event.type)
            {
                case SDL_QUIT: /* When close button is pressed */
                    _isRunning = false;
                    break;

                case SDL_KEYDOWN:
                    if(event.key.keysym.sym == SDLK_ESCAPE)
                    {
                        /* When ESC button is pressed */
                        _isRunning = false;
                    }
                    break;
            }
        }

        void Update()
        {
            /* Wait some time until it reach the target frame time in milliseconds */
            int time_to_wait = FRAME_TARGET_TIME - (SDL_GetTicks() - _previous_frame_time);

            /* Only delay execution if we are running too fast */
            if(time_to_wait > 0 && time_to_wait <= FRAME_TARGET_TIME)
            {
                SDL_Delay(time_to_wait);
            }

            /* Get a delta time factor converted to seconds to be used to update our game object */
            _delta_time = (SDL_GetTicks() - _previous_frame_time) / 1000.0f;
            _previous_frame_time = SDL_GetTicks();

            _mesh.rotation.x += 0.5 * _delta_time;
            _mesh.rotation.y += 0.5 * _delta_time;
            _mesh.rotation.z += 0.5 * _delta_time;

            // _mesh.scale.x += 0.1 * _delta_time;
            // _mesh.scale.y += 0.1 * _delta_time;

            // _mesh.translation.x += 0.1 * _delta_time;
            _mesh.translation.z = 5.0;

            // Create scale, rotation, and translation matrices that will be used to multiply the mesh vertices
            Mat4 scale_matrix = mat4_make_scale(_mesh.scale.x, _mesh.scale.y, _mesh.scale.z);
            Mat4 translation_matrix = mat4_make_translation(_mesh.translation.x, _mesh.translation.y, _mesh.translation.z);
            Mat4 rotation_matrix_x = mat4_make_rotation_x(_mesh.rotation.x);
            Mat4 rotation_matrix_y = mat4_make_rotation_y(_mesh.rotation.y);
            Mat4 rotation_matrix_z = mat4_make_rotation_z(_mesh.rotation.z);

            /* Loop all triangle faces of our mesh */
            for(int i = 0; i < _mesh.faces.size(); i++)
            {   
                const Face& mesh_face = _mesh.faces[i];

                Vec3 face_vertices[3];
                face_vertices[0] = _mesh.vertices[mesh_face.a - 1];
                face_vertices[1] = _mesh.vertices[mesh_face.b - 1];
                face_vertices[2] = _mesh.vertices[mesh_face.c - 1];

                Vec4 transformed_vertices[3];
                /* Loop all three vertices of this current face and apply transformations */
                for(int j = 0; j < 3; j++)
                {
                    Vec4 transformed_vertex = vec4_from_vec3(face_vertices[j]);

                    /* Create a World Matrix combining scale, rotation, and translation matrices */
                    Mat4 world_matrix = mat4_identity();

                    /* Order matters: First scale, then rotate, then translate. [T]*[R]*[S]*v */
                    world_matrix = mat4_mul_mat4(scale_matrix, world_matrix);
                    world_matrix = mat4_mul_mat4(rotation_matrix_z, world_matrix);
                    world_matrix = mat4_mul_mat4(rotation_matrix_y, world_matrix);
                    world_matrix = mat4_mul_mat4(rotation_matrix_x, world_matrix);
                    world_matrix = mat4_mul_mat4(translation_matrix, world_matrix);

                    /* Multiply the world matrix by the original vector */
                    transformed_vertex = mat4_mul_vec4(world_matrix, transformed_vertex);
                    
                    /* Save transformed vertex in the array of transformed vertices */
                    transformed_vertices[j] = transformed_vertex;
                }

                /* Check backface culling */
                Vec3 vector_a = vec3_from_vec4(transformed_vertices[0]);
                Vec3 vector_b = vec3_from_vec4(transformed_vertices[1]);
                Vec3 vector_c = vec3_from_vec4(transformed_vertices[2]);

                /* Get the vector subtraction of B-A and C-A */
                Vec3 vector_ab = vec3_sub(vector_b, vector_a);
                Vec3 vector_ac = vec3_sub(vector_c, vector_a);
                vec3_normalize(vector_ab);
                vec3_normalize(vector_ac);

                /* Compute the face normal (using cross product to find perpendicular) */
                Vec3 normal = vec3_cross(vector_ab, vector_ac);
                vec3_normalize(normal);

                /* Find the vector between a point in the triangle and the camera origin */
                Vec3 camera_ray = vec3_sub(_cameraPosition, vector_a);

                /* Calculate how aligned the camera ray is with the face normal (using dot product) */
                float dot_normal_camera = vec3_dot(normal, camera_ray);

                /* Bypass the triangles that are looking away from the camera */
                if(dot_normal_camera < 0)
                {
                    /* Skip the current face (aka triangle) */
                    continue;
                }

                // Triangle projected_triangle;
                Vec4 projected_points[3];
                for(int j = 0; j < 3; j++)
                {
                    /* Project the current point */
                    projected_points[j] = mat4_mul_vec4_project(_proj_matrix, transformed_vertices[j]);
                    
                    /* Invert the y values to account for flipped screen y coordinate */
                    projected_points[j].y *= -1;

                    /* Scale the projected points into the view so that we can see properly */
                    projected_points[j].x *= (WIDTH/2.0f);
                    projected_points[j].y *= (HEIGHT/2.0f);

                    /* Translate the projected points to the middle of the screen */
                    projected_points[j].x += (WIDTH/2.0f);
                    projected_points[j].y += (HEIGHT/2.0f);
                }

                /* Calculate the shade intensity based on how aliged is the face normal and the opposite of the light direction */
                float light_intensity_factor = -vec3_dot(normal, _light.direction);

                /* Calculate the triangle color based on the light angle */
                uint32_t triangle_color = light_apply_intensity(mesh_face.color, light_intensity_factor);

                Triangle projected_triangle = Triangle{
                    {
                        Vec2{ projected_points[0].x, projected_points[0].y},
                        Vec2{ projected_points[1].x, projected_points[1].y},
                        Vec2{ projected_points[2].x, projected_points[2].y},
                    }, 
                    triangle_color
                };

                /* Save the projected triangle in the array of triangles to render */
                _triangles_to_render.push_back(std::move(projected_triangle));
            }
        }

        void Render()
        {
            FillColorBuffer(0xFF000000);
            DrawGrid();
            
            // Loop all projected triangles and render them
            for (int i = 0; i < _triangles_to_render.size(); i++)
            {
                const Triangle& triangle = _triangles_to_render[i];

                // Draw vertex points
                // DrawRectangle({static_cast<uint32_t>(triangle.points[0].x), static_cast<uint32_t>(triangle.points[0].y), 3, 3, 0xFFFFFF00});
                // DrawRectangle({static_cast<uint32_t>(triangle.points[1].x), static_cast<uint32_t>(triangle.points[1].y), 3, 3, 0xFFFFFF00});
                // DrawRectangle({static_cast<uint32_t>(triangle.points[2].x), static_cast<uint32_t>(triangle.points[2].y), 3, 3, 0xFFFFFF00});

                // Draw filled triangle
                DrawFilledTriangle(
                    triangle.points[0].x, triangle.points[0].y, // vertex A
                    triangle.points[1].x, triangle.points[1].y, // vertex B
                    triangle.points[2].x, triangle.points[2].y, // vertex C
                    triangle.color
                );

                // Draw unfilled triangle
                // DrawTriangle(
                //     triangle.points[0].x,
                //     triangle.points[0].y,
                //     triangle.points[1].x,
                //     triangle.points[1].y,
                //     triangle.points[2].x,
                //     triangle.points[2].y,
                //     triangle.color
                // );
            }

            /* Clear the array of triangles to render every frame loop */
            _triangles_to_render.clear(); /* Its .size() becomes but .capacity() is not changed */

            RenderColorBuffer();
            SDL_RenderPresent(_renderer);
        }

        void InitializeWindow()
        {
            if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
            {
                throw std::runtime_error("Error initializing SDL");
            }

            _window = SDL_CreateWindow(nullptr, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_BORDERLESS);
            if(!_window)
            {
                throw std::runtime_error("Error creating SDL Window");
            }

            _renderer = SDL_CreateRenderer(_window, -1, 0);
            if(!_renderer)
            {
                throw std::runtime_error("Error creating SDL Renderer");
            }

            _isRunning = true;
        }

        void DestroyWindow()
        {
            SDL_DestroyRenderer(_renderer);
            SDL_DestroyWindow(_window);
            SDL_Quit();
        }

        bool IsRunning() const
        {
            return _isRunning;
        }

    private:
        void DrawGrid()
            {
                for(int i = 0; i < WIDTH; i++)
                {
                    for(int j = 0; j < HEIGHT; j++)
                    {
                        if(i % 10 == 0 && j % 10 == 0)
                        {
                            DrawPixel(i, j, 0xFF333333);
                        }
                    }
                }
                
            }

        void DrawRectangle(const std::array<uint32_t, 5>& rectInfo)
        {
            /*  
                <x:0,y:1,w:2,h:3,color:4> :- 
            
                (x,y) is Position
                (w,h) is Size
                color is SDL_PIXELFORMAT_ARGB8888
            
            */
            for(int i = 0; i < rectInfo[2]; i++)
            {
                for(int j = 0; j < rectInfo[3]; j++)
                {
                    uint32_t current_x = rectInfo[0] + i;
                    uint32_t current_y = rectInfo[1] + j;
                    
                    DrawPixel(current_x, current_y, rectInfo[4]);
                }
            }

        }

        void DrawLine(int x0, int y0, int x1, int y1, uint32_t color)
        {
            int delta_x = (x1 - x0);
            int delta_y = (y1 - y0);

            int longest_side_length = (std::abs(delta_x) >= std::abs(delta_y)) ? std::abs(delta_x) : std::abs(delta_y);

            float x_inc = delta_x / (float)longest_side_length; 
            float y_inc = delta_y / (float)longest_side_length;

            float current_x = x0;
            float current_y = y0;
            for (int i = 0; i <= longest_side_length; i++)
            {
                DrawPixel(std::round(current_x), std::round(current_y), color);
                current_x += x_inc;
                current_y += y_inc;
            }
        }

        void DrawTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
        {
            DrawLine(x0, y0, x1, y1, color);
            DrawLine(x1, y1, x2, y2, color);
            DrawLine(x2, y2, x0, y0, color);
        }

        void FillColorBuffer(uint32_t color)
        {
            for(int i = 0; i < WIDTH; i++)
            {
                for(int j = 0; j < HEIGHT; j++)
                {
                    DrawPixel(i, j, color);
                }
            }
        }

        void DrawPixel(int x, int y, uint32_t color)
        {
            if(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
            {
                _colorBuffer[(WIDTH * y) + x] = color;
            }
        }

        /* Function that receives a 3D vector and returns a projected 2D point */
        Vec2 Project(const Vec3& point)
        {
            Vec2 projected_point = {
                ((_fovFactor * point.x) / point.z),
                ((_fovFactor * point.y) / point.z)
            };

            return projected_point;
        }

        void RenderColorBuffer()
        {
            SDL_UpdateTexture(_colorBufferTexture, nullptr, _colorBuffer.data(), WIDTH * sizeof(uint32_t));
            SDL_RenderCopy(_renderer, _colorBufferTexture, nullptr, nullptr);
        }

        void FillFlatBottomTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
        {
            /* Find the two slopes (two triangle legs) */
            float inv_slope_1 = (float)(x1 - x0) / (y1 - y0);
            float inv_slope_2 = (float)(x2 - x0) / (y2 - y0);

            /* Start x_start and x_end from the top vertex (x0,y0) */
            float x_start = x0;
            float x_end = x0;

            /* Loop all the scanlines from top to bottom */
            for (int y = y0; y <= y2; y++)
            {
                // DrawLine(std::round(x_start), y, std::round(x_end), y, color);
                int xs = std::round(x_start);
                int xe = std::round(x_end);
                if (xs > xe)
                {
                    std::swap(xs, xe);
                }
                for (int x = xs; x <= xe; x++)
                {
                    DrawPixel(x, y, color);
                }
                x_start += inv_slope_1;
                x_end += inv_slope_2;
            }
        }

        void FillFlatTopTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
        {
            /* Find the two slopes (two triangle legs) */
            float inv_slope_1 = (float)(x2 - x0) / (y2 - y0);
            float inv_slope_2 = (float)(x2 - x1) / (y2 - y1);

            /* Start x_start and x_end from the bottom vertex (x2,y2) */
            float x_start = x2;
            float x_end = x2;

            /* Loop all the scanlines from bottom to top */
            for (int y = y2; y >= y0; y--)
            {
                // DrawLine(std::round(x_start), y, std::round(x_end), y, color);
                int xs = std::round(x_start);
                int xe = std::round(x_end);
                if (xs > xe)
                {
                    std::swap(xs, xe);
                }
                for (int x = xs; x <= xe; x++)
                {
                    DrawPixel(x, y, color);
                }
                x_start -= inv_slope_1;
                x_end -= inv_slope_2;
            }
        }

        void DrawFilledTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
        {   
            /* We need to sort the vertices by y-coordinate ascending (y0 < y1 < y2) */
            SortInAscendingOrder(x0, y0, x1, y1, x2, y2);

            if (y1 == y2)
            {
                /* Draw flat-bottom triangle */
                FillFlatBottomTriangle(x0, y0, x1, y1, x2, y2, color);
            }
            else if (y0 == y1)
            {
                /* Draw flat-top triangle */
                FillFlatTopTriangle(x0, y0, x1, y1, x2, y2, color);
            }
            else
            {
                /* Calculate the new vertex (Mx,My) using triangle similarity */
                int My = y1;
                int Mx = (((x2 - x0) * (y1 - y0)) / (y2 - y0)) + x0;

                /* Draw flat-bottom triangle */
                FillFlatBottomTriangle(x0, y0, x1, y1, Mx, My, color);

                /* Draw flat-top triangle */
                FillFlatTopTriangle(x1, y1, Mx, My, x2, y2, color);
            }
        }

        void SortInAscendingOrder(int& x0, int& y0, int& x1, int& y1, int& x2, int& y2)
        {
            std::array<std::pair<int, int>, 3> points = {
                std::make_pair(x0, y0),
                std::make_pair(x1, y1),
                std::make_pair(x2, y2),
            };

            std::sort(points.begin(), points.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b){
                return a.second < b.second; /* compare by y value */
            });

            x0 = points[0].first; y0 = points[0].second;
            x1 = points[1].first; y1 = points[1].second;
            x2 = points[2].first; y2 = points[2].second;
        }

        void LoadCubeMeshData()
        {
            std::array<Vec3, CUBE_MESH_VERTICES> cube_mesh_vertices = {
                Vec3{ -1.0f, -1.0f, -1.0f }, // 1
                Vec3{ -1.0f,  1.0f, -1.0f }, // 2
                Vec3{  1.0f,  1.0f, -1.0f }, // 3
                Vec3{  1.0f, -1.0f, -1.0f }, // 4
                Vec3{  1.0f,  1.0f,  1.0f }, // 5
                Vec3{  1.0f, -1.0f,  1.0f }, // 6
                Vec3{ -1.0f,  1.0f,  1.0f }, // 7
                Vec3{ -1.0f, -1.0f,  1.0f }  // 8
            };

            std::array<Face, CUBE_MESH_FACES> cube_mesh_faces = {
                
                // front
                Face{ 1, 2, 3, 0xFFFFFFFF},
                Face{ 1, 3, 4, 0xFFFFFFFF},
                // right
                Face{ 4, 3, 5, 0xFFFFFFFF},
                Face{ 4, 5, 6, 0xFFFFFFFF},
                // back
                Face{ 6, 5, 7, 0xFFFFFFFF},
                Face{ 6, 7, 8, 0xFFFFFFFF},
                // left
                Face{ 8, 7, 2, 0xFFFFFFFF},
                Face{ 8, 2, 1, 0xFFFFFFFF},
                // top
                Face{ 2, 7, 5, 0xFFFFFFFF},
                Face{ 2, 5, 3, 0xFFFFFFFF},
                // bottom
                Face{ 6, 8, 1, 0xFFFFFFFF},
                Face{ 6, 1, 4, 0xFFFFFFFF}
            };

            for(int i = 0; i < CUBE_MESH_VERTICES; i++)
            {
                _mesh.vertices.push_back(std::move(cube_mesh_vertices[i]));
            }

            for(int i = 0; i < CUBE_MESH_FACES; i++)
            {
                _mesh.faces.push_back(std::move(cube_mesh_faces[i]));
            }
        }

        void LoadObjFileData(const char* filename)
        {
            std::ifstream infile(filename);
            if(!infile.is_open())
            {
                throw std::runtime_error("Unable to open OBJ file");
            }

            std::string line;
            while(std::getline(infile, line, '\n'))
            {
                /* Vertex information */
                if(line.find("v ") == 0)
                {
                    Vec3 vertex;
                    sscanf(line.data(), "v %f %f %f", &vertex.x, &vertex.y, &vertex.z);
                    _mesh.vertices.push_back(std::move(vertex));
                }

                /* Face information */
                if(line.find("f ") == 0)
                {
                    int vertex_indices[3];
                    int texture_indices[3];
                    int normal_indices[3];

                    sscanf(
                        line.data(), "f %d/%d/%d %d/%d/%d %d/%d/%d",
                        &vertex_indices[0], &texture_indices[0], &normal_indices[0], 
                        &vertex_indices[1], &texture_indices[1], &normal_indices[1], 
                        &vertex_indices[2], &texture_indices[2], &normal_indices[2]
                    ); 

                    Face face{vertex_indices[0], vertex_indices[1], vertex_indices[2], 0xFFFFFFFF};
                    _mesh.faces.push_back(std::move(face));
                }
            }

        }

    private:
        std::array<uint32_t, WIDTH * HEIGHT> _colorBuffer;
        Mat4 _proj_matrix;
        Light _light;

        bool _isRunning = false;
        uint32_t _fovFactor = 640;
        Vec3 _cameraPosition = {0, 0, 0};

        SDL_Window* _window;
        SDL_Renderer* _renderer;
        SDL_Texture* _colorBufferTexture;

        int _previous_frame_time = 0;
        float _delta_time = 0;

        std::vector<Triangle> _triangles_to_render;
        Mesh _mesh;
};

int main()
{   
    auto engine = std::make_unique<Engine<800, 600>>();
    try
    {
        engine->InitializeWindow();
        engine->Setup();

        while(engine->IsRunning())
        {
            engine->ProcessInput();
            engine->Update();
            engine->Render();
        }

        engine->DestroyWindow();
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
    }
    catch(...)
    {
        fprintf(stderr, "%s\n", "Unexpected Error\n");
    }
    return 0;
}
