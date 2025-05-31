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

/* It stores the indices */
struct Face
{
    int a;
    int b;
    int c;
};

struct Triangle
{
    std::array<Vec2, 3> points;
};

struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<Face> faces;
    Vec3 rotation; /* rotation with x, y and z values */
};

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

            // LoadCubeMeshData();
            LoadObjFileData("C:\\my_code\\win_cpp\\3D_Graphics\\assets\\f22.obj");
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

            _previous_frame_time = SDL_GetTicks();

            _mesh.rotation.x += 0.00;
            _mesh.rotation.y += 0.00;
            _mesh.rotation.z += 0.00;

            /* Loop all triangle faces of our mesh */
            for(int i = 0; i < _mesh.faces.size(); i++)
            {   
                const Face& mesh_face = _mesh.faces[i];

                Vec3 face_vertices[3];
                face_vertices[0] = _mesh.vertices[mesh_face.a - 1];
                face_vertices[1] = _mesh.vertices[mesh_face.b - 1];
                face_vertices[2] = _mesh.vertices[mesh_face.c - 1];

                Vec3 transformed_vertices[3];
                /* Loop all three vertices of this current face and apply transformations */
                for(int j = 0; j < 3; j++)
                {
                    Vec3 transformed_vertex = face_vertices[j];

                    transformed_vertex = vec3_rotate_x(transformed_vertex, _mesh.rotation.x);
                    transformed_vertex = vec3_rotate_y(transformed_vertex, _mesh.rotation.y);
                    transformed_vertex = vec3_rotate_z(transformed_vertex, _mesh.rotation.z);
                    
                    /* Translate the vertex away from the camera */
                    transformed_vertex.z -= -5;

                    /* Save transformed vertex in the array of transformed vertices */
                    transformed_vertices[j] = std::move(transformed_vertex);
                }

                /* Check backface culling */
                const Vec3& vector_a = transformed_vertices[0];
                const Vec3& vector_b = transformed_vertices[1];
                const Vec3& vector_c = transformed_vertices[2];

                /* Get the vector subtraction of B-A and C-A */
                Vec3 vector_ab = vec3_sub(vector_b, vector_a);
                Vec3 vector_ac = vec3_sub(vector_c, vector_a);

                /* Compute the face normal (using cross product to find perpendicular) */
                Vec3 normal = vec3_cross(vector_ab, vector_ac);

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

                Triangle projected_triangle;
                for(int j = 0; j < 3; j++)
                {
                    /* Project the current point */
                    Vec2 projected_point = Project(transformed_vertices[j]);
                    
                    /* Invert the y values to account for flipped screen y coordinate */
                    projected_point.y *= -1;

                    /* Scale and translate the projected points to the middle of the screen */
                    projected_point.x += (WIDTH/2.0f);
                    projected_point.y += (HEIGHT/2.0f);

                    projected_triangle.points[j] = std::move(projected_point);
                }

                /* Save the projected triangle in the array of triangles to render */
                _triangles_to_render.push_back(std::move(projected_triangle));
            }
        }

        void Render()
        {
            FillColorBuffer(0xFF000000);
            // DrawGrid();
            
            // Loop all projected triangles and render them
            for (int i = 0; i < _triangles_to_render.size(); i++)
            {
                const Triangle& triangle = _triangles_to_render[i];

                // Draw vertex points
                DrawRectangle({static_cast<uint32_t>(triangle.points[0].x), static_cast<uint32_t>(triangle.points[0].y), 3, 3, 0xFFFFFF00});
                DrawRectangle({static_cast<uint32_t>(triangle.points[1].x), static_cast<uint32_t>(triangle.points[1].y), 3, 3, 0xFFFFFF00});
                DrawRectangle({static_cast<uint32_t>(triangle.points[2].x), static_cast<uint32_t>(triangle.points[2].y), 3, 3, 0xFFFFFF00});

                // Draw filled triangle
                DrawFilledTriangle(
                    triangle.points[0].x, triangle.points[0].y, // vertex A
                    triangle.points[1].x, triangle.points[1].y, // vertex B
                    triangle.points[2].x, triangle.points[2].y, // vertex C
                    0xFFFFFFFF
                );

                // Draw unfilled triangle
                DrawTriangle(
                    triangle.points[0].x,
                    triangle.points[0].y,
                    triangle.points[1].x,
                    triangle.points[1].y,
                    triangle.points[2].x,
                    triangle.points[2].y,
                    0xFF000000
                );
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
                        if(i % 10 == 0 || j % 10 == 0)
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
                Face{ 1, 2, 3 },
                Face{ 1, 3, 4 },
                // right
                Face{ 4, 3, 5 },
                Face{ 4, 5, 6 },
                // back
                Face{ 6, 5, 7 },
                Face{ 6, 7, 8 },
                // left
                Face{ 8, 7, 2 },
                Face{ 8, 2, 1 },
                // top
                Face{ 2, 7, 5 },
                Face{ 2, 5, 3 },
                // bottom
                Face{ 6, 8, 1 },
                Face{ 6, 1, 4 }
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

                    Face face{vertex_indices[0], vertex_indices[1], vertex_indices[2]};
                    _mesh.faces.push_back(std::move(face));
                }
            }

        }

    private:
        std::array<uint32_t, WIDTH * HEIGHT> _colorBuffer;

        bool _isRunning = false;
        uint32_t _fovFactor = 640;
        Vec3 _cameraPosition = {0, 0, 0};

        SDL_Window* _window;
        SDL_Renderer* _renderer;
        SDL_Texture* _colorBufferTexture;

        int _previous_frame_time;

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
